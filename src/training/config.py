from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from src.model import ModelConfig


def _known_dataclass_values(cls, values: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


@dataclass
class DataConfig:
    train_manifest: str | None = None
    val_manifest: str | None = None
    tokenizer: str | None = None
    shuffle_shards: bool = True
    shuffle_seed: int = 1337
    eval_batches: int = 20
    synthetic: bool = True


@dataclass
class TrainConfig:
    run_id: str | None = None
    runs_dir: str = "runs"
    seed: int = 1337
    device: str = "auto"
    precision: str = "bf16"
    compile: bool = False
    compile_mode: str = "default"
    tf32: bool = True
    sequence_length: int = 1_024
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    global_batch_tokens: int | None = None
    max_tokens: int | None = None
    max_steps: int | None = None
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_tokens: int = 20_000_000
    total_tokens: int = 1_000_000_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    log_every_steps: int = 10
    eval_every_tokens: int = 10_000_000
    checkpoint_every_steps: int = 0
    checkpoint_every_tokens: int = 0
    checkpoint_every_minutes: float = 15.0
    permanent_checkpoint_every_tokens: int = 50_000_000
    keep_last_n: int = 3
    pin_memory: bool = True
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 or fp32")
        if self.micro_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch sizes must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when set")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when set")
        if self.global_batch_tokens is not None and self.global_batch_tokens != self.tokens_per_step:
            raise ValueError("global_batch_tokens must equal sequence_length * micro_batch_size * accumulation")

    @property
    def tokens_per_step(self) -> int:
        return self.sequence_length * self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def effective_global_batch_tokens(self) -> int:
        return self.global_batch_tokens or self.tokens_per_step


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig, DataConfig, dict[str, Any]]:
    source = Path(path)
    raw = yaml.safe_load(source.read_text()) or {}
    model = ModelConfig.from_dict(raw.get("model", {}))
    training = TrainConfig(**_known_dataclass_values(TrainConfig, raw.get("training", {})))
    data = DataConfig(**_known_dataclass_values(DataConfig, raw.get("data", {})))
    return model, training, data, raw


def config_as_dict(model: ModelConfig, training: TrainConfig, data: DataConfig) -> dict[str, Any]:
    return {"model": model.to_dict(), "training": asdict(training), "data": asdict(data)}
