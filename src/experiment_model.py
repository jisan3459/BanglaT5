import gc
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from normalizer import normalize
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .config import ExperimentConfig, ProjectConfig


class BanglaT5ExperimentModel:
    """Training, CodeCarbon measurement, generation, and EM/F1 evaluation."""

    def __init__(self, project, experiment, tokenizer, processed, tokenized):
        self.project: ProjectConfig = project
        self.experiment: ExperimentConfig = experiment
        self.tokenizer = tokenizer
        self.processed = processed
        self.tokenized = tokenized
        self.model = None
        self.trainer = None
        self.data_collator = None
        self.total_params = None
        self.trainable_params = None

    def build(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is not available. In Colab select Runtime > Change runtime type > T4 GPU."
            )

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.project.model_name)

        if self.experiment.is_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_config = LoraConfig(
                r=self.experiment.rank,
                lora_alpha=self.experiment.lora_alpha,
                target_modules=["q", "v"],
                lora_dropout=self.experiment.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            self.model = get_peft_model(self.model, lora_config)

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

        output_dir = self.project.models_dir / self.experiment.run_name / "trainer_output"
        args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.experiment.num_train_epochs,
            learning_rate=self.experiment.learning_rate,
            per_device_train_batch_size=self.experiment.train_batch_size,
            per_device_eval_batch_size=self.experiment.eval_batch_size,
            gradient_accumulation_steps=self.experiment.gradient_accumulation_steps,
            fp16=False,
            bf16=False,
            max_grad_norm=self.experiment.max_grad_norm,
            seed=self.project.seed,
            data_seed=self.project.seed,
            logging_strategy="steps",
            logging_steps=50,
            eval_strategy="no",
            save_strategy="no",
            predict_with_generate=False,
            report_to="none",
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=self.tokenized["train"],
            eval_dataset=self.tokenized["validation"],
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
        )

        self.total_params = sum(parameter.numel() for parameter in self.model.parameters())
        self.trainable_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    def sanity_check(self) -> Dict[str, object]:
        features = [
            {
                "input_ids": self.tokenized["train"][i]["input_ids"],
                "attention_mask": self.tokenized["train"][i]["attention_mask"],
                "labels": self.tokenized["train"][i]["labels"],
            }
            for i in range(2)
        ]
        batch = self.data_collator(features)
        device = next(self.model.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}

        self.model.eval()
        with torch.no_grad():
            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

        return {
            "loss": float(output.loss.item()),
            "loss_finite": bool(torch.isfinite(output.loss).item()),
            "nan_logits": bool(torch.isnan(output.logits).any().item()),
            "inf_logits": bool(torch.isinf(output.logits).any().item()),
            "batch": batch,
        }

    def train(self) -> Dict[str, object]:
        # Lazy import: preprocessing never depends on CodeCarbon.
        from codecarbon import EmissionsTracker

        carbon_dir = self.project.carbon_logs_dir / self.experiment.run_name
        carbon_dir.mkdir(parents=True, exist_ok=True)
        emissions_csv = carbon_dir / "emissions.csv"
        if emissions_csv.exists():
            emissions_csv.unlink()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        tracker = EmissionsTracker(
            project_name=f"BanglaT5_{self.experiment.run_name}",
            output_dir=str(carbon_dir),
            output_file="emissions.csv",
            measure_power_secs=5,
            save_to_file=True,
            save_to_api=False,
            log_level="error",
        )

        tracker.start()
        start = time.perf_counter()
        try:
            train_result = self.trainer.train()
        finally:
            end = time.perf_counter()
            tracker.stop()

        if not emissions_csv.exists():
            raise RuntimeError(f"CodeCarbon did not create {emissions_csv}")

        carbon_row = pd.read_csv(emissions_csv).iloc[-1]
        return {
            "metrics": train_result.metrics,
            "training_time_seconds": end - start,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "energy_kwh": float(carbon_row["energy_consumed"]),
            "co2_kg": float(carbon_row["emissions"]),
        }

    def post_training_check(self, batch) -> Dict[str, object]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        return {
            "loss": float(output.loss.item()),
            "loss_finite": bool(torch.isfinite(output.loss).item()),
            "nan_logits": bool(torch.isnan(output.logits).any().item()),
            "inf_logits": bool(torch.isinf(output.logits).any().item()),
        }

    @staticmethod
    def _normalize_for_eval(text) -> str:
        if text is None:
            return ""
        text = normalize(str(text)).lower()
        text = "".join(
            char for char in text
            if not unicodedata.category(char).startswith("P")
        )
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _exact_match(cls, prediction, reference) -> int:
        return int(
            cls._normalize_for_eval(prediction)
            == cls._normalize_for_eval(reference)
        )

    @classmethod
    def _token_f1(cls, prediction, reference) -> float:
        pred_tokens = cls._normalize_for_eval(prediction).split()
        ref_tokens = cls._normalize_for_eval(reference).split()

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
        if overlap == 0:
            return 0.0

        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def _score_references(cls, prediction, references) -> Tuple[int, float]:
        if isinstance(references, np.ndarray):
            references = references.tolist()
        if not references:
            raise ValueError("A sample has no reference answers.")
        em = max(cls._exact_match(prediction, ref) for ref in references)
        f1 = max(cls._token_f1(prediction, ref) for ref in references)
        return em, f1

    def evaluate(self, split: str) -> Dict[str, object]:
        self.trainer.args.predict_with_generate = True
        output = self.trainer.predict(
            self.tokenized[split],
            num_beams=1,
            do_sample=False,
            max_new_tokens=self.project.max_target_length,
        )

        predictions = (
            output.predictions[0]
            if isinstance(output.predictions, tuple)
            else output.predictions
        )
        pred_ids = np.asarray(predictions).copy()
        pred_ids[pred_ids == -100] = self.tokenizer.pad_token_id
        if (pred_ids < 0).any():
            raise ValueError("Unexpected negative token IDs remain after prediction cleanup.")

        decoded = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        results = self.processed[split].copy()
        if len(decoded) != len(results):
            raise RuntimeError("Prediction count does not match dataset row count.")
        results["prediction"] = decoded

        scores = results.apply(
            lambda row: self._score_references(
                row["prediction"],
                row["reference_answers"],
            ),
            axis=1,
        )
        results["EM"] = scores.apply(lambda pair: pair[0])
        results["F1"] = scores.apply(lambda pair: pair[1])

        blanks = results["prediction"].astype(str).str.strip().eq("").sum()
        return {
            "results": results,
            "em": 100 * results["EM"].mean(),
            "f1": 100 * results["F1"].mean(),
            "exact_matches": int(results["EM"].sum()),
            "blank_predictions": int(blanks),
            "blank_percentage": 100 * blanks / len(results),
        }

    def save_model(self) -> Path:
        model_dir = self.project.models_dir / self.experiment.run_name / "final_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(model_dir))
        self.tokenizer.save_pretrained(model_dir)
        return model_dir

    def save_results(self, train_info, post_check, validation_eval, test_eval) -> pd.DataFrame:
        results_dir = self.project.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        summary = pd.DataFrame([{
            "method": self.experiment.method,
            "rank": self.experiment.rank,
            "total_parameters": self.total_params,
            "trainable_parameters": self.trainable_params,
            "trainable_percent": 100 * self.trainable_params / self.total_params,
            "validation_em": validation_eval["em"],
            "validation_f1": validation_eval["f1"],
            "validation_blank_percent": validation_eval["blank_percentage"],
            "test_em": test_eval["em"],
            "test_f1": test_eval["f1"],
            "test_blank_percent": test_eval["blank_percentage"],
            "training_time_seconds": train_info["training_time_seconds"],
            "peak_gpu_memory_gb": train_info["peak_gpu_memory_gb"],
            "energy_kwh": train_info["energy_kwh"],
            "co2_kg": train_info["co2_kg"],
            "post_training_loss": post_check["loss"],
        }])

        run = self.experiment.run_name
        summary.to_csv(results_dir / f"{run}_summary.csv", index=False)
        validation_eval["results"].to_csv(
            results_dir / f"{run}_validation_predictions.csv",
            index=False,
        )
        test_eval["results"].to_csv(
            results_dir / f"{run}_test_predictions.csv",
            index=False,
        )
        return summary
