from pathlib import Path
import pandas as pd


RUNS = ["full_ft", "lora_r4", "lora_r8", "lora_r16", "lora_r32"]


def load_master_results(results_dir: Path) -> pd.DataFrame:
    frames = []
    for run in RUNS:
        path = results_dir / f"{run}_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing result: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def question_type_scores(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    result = (
        df.groupby("question_type")
        .agg(
            Questions=("question_type", "size"),
            EM=("EM", "mean"),
            F1=("F1", "mean"),
        )
        .reset_index()
    )
    result["EM"] *= 100
    result["F1"] *= 100
    result["Model"] = model_name
    return result[["Model", "question_type", "Questions", "EM", "F1"]]
