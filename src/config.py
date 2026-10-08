from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    model_name: str = "csebuetnlp/banglat5_small"
    max_source_length: int = 768
    max_target_length: int = 96
    seed: int = 42

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def prepared_dir(self) -> Path:
        return self.project_root / "data" / "prepared"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def carbon_logs_dir(self) -> Path:
        # Deliberately NOT named "codecarbon". A local folder with that name
        # shadows the installed CodeCarbon package when the project is on sys.path.
        return self.project_root / "carbon_logs"

    def ensure_directories(self) -> None:
        for path in (
            self.raw_dir,
            self.prepared_dir,
            self.results_dir,
            self.models_dir,
            self.carbon_logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    learning_rate: float
    rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    num_train_epochs: int = 3
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    lora_dropout: float = 0.1

    @property
    def is_lora(self) -> bool:
        return self.rank is not None

    @property
    def run_name(self) -> str:
        return "full_ft" if not self.is_lora else f"lora_r{self.rank}"

    @classmethod
    def full_ft(cls) -> "ExperimentConfig":
        return cls(method="Full Fine-Tuning", learning_rate=1e-3)

    @classmethod
    def lora(cls, rank: int) -> "ExperimentConfig":
        if rank not in {4, 8, 16, 32}:
            raise ValueError("This study uses LoRA ranks 4, 8, 16, and 32 only.")
        return cls(
            method=f"LoRA r={rank}",
            rank=rank,
            lora_alpha=2 * rank,
            learning_rate=1e-3,
        )
