import ast
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
from normalizer import normalize
from transformers import AutoTokenizer

from .config import ProjectConfig


NO_ANSWER = "উত্তর নেই"


class BanglaRQADataModel:
    """Loads, preprocesses, tokenizes, validates, and saves BanglaRQA splits."""

    RAW_FILES = {
        "train": "BanglaRQA_targeted_domains.csv",
        "validation": "validation_targeted.csv",
        "test": "test_targeted.csv",
    }

    REQUIRED_COLUMNS = {
        "passage_id",
        "context",
        "question_id",
        "question_text",
        "is_answerable",
        "question_type",
    }

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.config.ensure_directories()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=False,
        )

    @staticmethod
    def _basic_clean_text(text) -> str:
        if pd.isna(text):
            return ""
        text = unicodedata.normalize("NFC", str(text))
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _clean_and_normalize_bangla(cls, text) -> str:
        text = cls._basic_clean_text(text)
        if not text:
            return ""
        text = normalize(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_array_field(value, field_name: str):
        if pd.isna(value):
            return []

        pattern = (
            rf"'{field_name}'\s*:\s*"
            rf"array\((\[.*?\])\s*,\s*dtype=object\)"
        )
        match = re.search(pattern, str(value), flags=re.DOTALL)
        if match is None:
            return []

        try:
            result = ast.literal_eval(match.group(1))
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []

    @staticmethod
    def _split_pipe_field(value):
        if pd.isna(value):
            return []
        return [item.strip() for item in str(value).split("|")]

    @classmethod
    def _parse_answers(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "answers" in df.columns:
            df["answer_texts_raw"] = df["answers"].apply(
                lambda x: cls._extract_array_field(x, "answer_text")
            )
            df["answer_types_raw"] = df["answers"].apply(
                lambda x: cls._extract_array_field(x, "answer_type")
            )
        elif "answer_text" in df.columns and "answer_type" in df.columns:
            df["answer_texts_raw"] = df["answer_text"].apply(cls._split_pipe_field)
            df["answer_types_raw"] = df["answer_type"].apply(cls._split_pipe_field)
        else:
            raise ValueError("No recognized BanglaRQA answer columns found.")

        return df

    @staticmethod
    def _normalize_single_answer(answer) -> str:
        if answer is None:
            return ""
        answer = re.sub(r"\s+", " ", str(answer)).strip()
        if not answer:
            return ""
        answer = normalize(answer)
        return re.sub(r"\s+", " ", answer).strip()

    @classmethod
    def _normalize_reference_answers(cls, answers):
        cleaned = []
        for answer in answers:
            answer = cls._normalize_single_answer(answer)
            if answer and answer not in cleaned:
                cleaned.append(answer)
        return cleaned

    @staticmethod
    def _create_input_text(row) -> str:
        return (
            "প্রসঙ্গ: " + row["context_clean"]
            + "\nপ্রশ্ন: " + row["question_clean"]
            + "\nউত্তর:"
        )

    def load_raw_splits(self) -> Dict[str, pd.DataFrame]:
        splits = {}
        for split, filename in self.RAW_FILES.items():
            path = self.config.raw_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Put the original targeted CSV files in data/raw/."
                )
            df = pd.read_csv(path)
            missing = self.REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
            splits[split] = df
        return splits

    def _process_split(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["context_clean"] = df["context"].apply(self._clean_and_normalize_bangla)
        df["question_clean"] = df["question_text"].apply(self._clean_and_normalize_bangla)
        df = self._parse_answers(df)

        df["reference_answers_raw"] = df.apply(
            lambda row: [NO_ANSWER]
            if int(row["is_answerable"]) == 0
            else row["answer_texts_raw"],
            axis=1,
        )
        df["reference_answers"] = df["reference_answers_raw"].apply(
            self._normalize_reference_answers
        )
        df["target_text"] = df["reference_answers"].apply(
            lambda refs: refs[0] if refs else None
        )
        df["input_text"] = df.apply(self._create_input_text, axis=1)
        return df

    def process(self) -> Dict[str, pd.DataFrame]:
        raw = self.load_raw_splits()
        processed = {name: self._process_split(df) for name, df in raw.items()}
        self._validate_processed(processed)
        return processed

    @staticmethod
    def _validate_processed(processed: Dict[str, pd.DataFrame]) -> None:
        # These checks do not alter data; they only prevent silent preprocessing errors.
        passage_sets = {name: set(df["passage_id"]) for name, df in processed.items()}
        if passage_sets["train"] & passage_sets["validation"]:
            raise ValueError("Passage overlap detected between train and validation.")
        if passage_sets["train"] & passage_sets["test"]:
            raise ValueError("Passage overlap detected between train and test.")
        if passage_sets["validation"] & passage_sets["test"]:
            raise ValueError("Passage overlap detected between validation and test.")

        for name, df in processed.items():
            if df["input_text"].astype(str).str.strip().eq("").any():
                raise ValueError(f"Empty input_text detected in {name}.")
            if df["target_text"].isna().any() or df["target_text"].astype(str).str.strip().eq("").any():
                raise ValueError(f"Missing or empty target_text detected in {name}.")

    def _tokenize_batch(self, examples):
        model_inputs = self.tokenizer(
            examples["input_text"],
            max_length=self.config.max_source_length,
            truncation=True,
            padding=False,
        )
        labels = self.tokenizer(
            text_target=examples["target_text"],
            max_length=self.config.max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize(self, processed: Dict[str, pd.DataFrame]):
        tokenized = {}
        for name, df in processed.items():
            dataset = Dataset.from_pandas(df, preserve_index=False)
            tokenized[name] = dataset.map(
                self._tokenize_batch,
                batched=True,
                desc=f"Tokenizing {name}",
            )
        return tokenized

    def save_artifacts(self, processed, tokenized) -> None:
        prepared = self.config.prepared_dir
        prepared.mkdir(parents=True, exist_ok=True)

        for name, df in processed.items():
            df.to_parquet(prepared / f"{name}_processed.parquet", index=False)

        for name, dataset in tokenized.items():
            path = prepared / f"tokenized_{name}"
            if path.exists():
                shutil.rmtree(path)
            dataset.save_to_disk(str(path))

        tokenizer_dir = prepared / "tokenizer"
        if tokenizer_dir.exists():
            shutil.rmtree(tokenizer_dir)
        self.tokenizer.save_pretrained(tokenizer_dir)

        metadata = {
            "model_name": self.config.model_name,
            "max_source_length": self.config.max_source_length,
            "max_target_length": self.config.max_target_length,
            "seed": self.config.seed,
            "rows": {name: len(df) for name, df in processed.items()},
        }
        (prepared / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class PreparedDataRepository:
    """Loads the one prepared data snapshot shared by every experiment."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.prepared = config.prepared_dir

    @staticmethod
    def _restore_list_columns(df: pd.DataFrame) -> pd.DataFrame:
        for column in (
            "answer_texts_raw",
            "answer_types_raw",
            "reference_answers_raw",
            "reference_answers",
        ):
            if column in df.columns:
                df[column] = df[column].apply(
                    lambda value: value.tolist() if isinstance(value, np.ndarray) else value
                )
        return df

    def load(self):
        required = [
            self.prepared / "tokenized_train",
            self.prepared / "tokenized_validation",
            self.prepared / "tokenized_test",
            self.prepared / "train_processed.parquet",
            self.prepared / "validation_processed.parquet",
            self.prepared / "test_processed.parquet",
            self.prepared / "tokenizer",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Prepared artifacts are missing. Run 00_Data_Preprocessing.ipynb first.\n"
                + "\n".join(missing)
            )

        tokenizer = AutoTokenizer.from_pretrained(self.prepared / "tokenizer", use_fast=False)
        processed = {
            name: self._restore_list_columns(
                pd.read_parquet(self.prepared / f"{name}_processed.parquet")
            )
            for name in ("train", "validation", "test")
        }
        tokenized = {
            name: load_from_disk(str(self.prepared / f"tokenized_{name}"))
            for name in ("train", "validation", "test")
        }
        return tokenizer, processed, tokenized
