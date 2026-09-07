from __future__ import annotations

import contextlib
import math
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.data.mmap_dataset import MMapTokenStream, SyntheticTokenStream
from src.data.multiple_choice import (
    MultipleChoiceBatch,
    MultipleChoiceStream,
    collate_multiple_choice,
    combined_manifest_hash,
)
from src.model import DecoderLM, ModelConfig

from .checkpoint import (
    CheckpointManager,
    capture_rng_state,
    default_checkpoint_metadata,
    restore_rng_state,
)
from .config import DataConfig, TrainConfig
from .metrics import MetricsWriter, atomic_json_write
from .scheduler import TokenCosineScheduler
from .telemetry import GPUTelemetry
from .contrastive import batch_ranking_loss


def configure_torch(tf32: bool = True) -> None:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def build_optimizer(model: DecoderLM, config: TrainConfig) -> tuple[torch.optim.Optimizer, str]:
    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and not name.endswith("alpha_p") and not name.endswith("alpha_n"):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {
        "lr": config.learning_rate,
        "betas": (config.beta1, config.beta2),
        "eps": config.adam_eps,
    }
    if model.token_embedding.weight.device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(groups, fused=True, **kwargs)
            return optimizer, "AdamW(fused=True)"
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, **kwargs), "AdamW"


@dataclass
class TrainerState:
    step: int = 0
    tokens_seen: int = 0
    epoch: int = 0
    started_at: str = ""
    last_checkpoint_at: str | None = None
    last_checkpoint_tokens: int = 0
    last_permanent_tokens: int = 0
    last_eval_tokens: int = 0
    validation_loss: float | None = None
    smoothed_loss: float | None = None
    status: str = "created"


class Trainer:
    """Direct PyTorch training loop with observable, restartable state."""

    def __init__(
        self,
        model: DecoderLM,
        model_config: ModelConfig,
        train_config: TrainConfig,
        data_config: DataConfig,
        run_dir: str | Path,
        *,
        train_stream: MMapTokenStream | SyntheticTokenStream | None = None,
        val_stream: MMapTokenStream | SyntheticTokenStream | None = None,
        tokenizer_hash: str | None = None,
        dataset_manifest_hash: str | None = None,
    ) -> None:
        configure_torch(train_config.tf32)
        self.model_config = model_config
        self.config = train_config
        self.data_config = data_config
        self.device = resolve_device(train_config.device)
        self.raw_model = model.to(self.device)
        self.optimizer, self.optimizer_impl = build_optimizer(self.raw_model, train_config)
        self.scheduler = TokenCosineScheduler(
            self.optimizer,
            train_config.learning_rate,
            train_config.min_learning_rate,
            train_config.warmup_tokens,
            train_config.total_tokens,
        )
        self.train_model: torch.nn.Module = self.raw_model
        if train_config.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("this PyTorch build has no torch.compile")
            self.train_model = torch.compile(
                self.raw_model,
                mode=train_config.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = MetricsWriter(self.run_dir)
        self.log_handle = (self.run_dir / "train.log").open("a", encoding="utf-8", buffering=1)
        self.checkpoints = CheckpointManager(self.run_dir, train_config.keep_last_n)
        self.state = TrainerState(started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        self.stop_requested = False
        self.stop_reason = ""
        self.checkpoint_requested = False
        self.train_stream = train_stream or self._make_stream(False)
        self.val_stream = val_stream or self._make_stream(True)
        self.mc_stream: MultipleChoiceStream | None = None
        self.mc_val_stream: MultipleChoiceStream | None = None
        if self.config.objective == "contrastive_mc":
            if not data_config.mc_train_manifest:
                raise ValueError("contrastive_mc requires data.mc_train_manifest")
            if data_config.mc_batch_size <= 0:
                raise ValueError("data.mc_batch_size must be positive")
            if data_config.mc_eval_batches < 0:
                raise ValueError("data.mc_eval_batches must be non-negative")
            if self.config.mc_sequence_length > self.model_config.max_seq_len:
                raise ValueError("mc_sequence_length exceeds model max_seq_len")
            self.mc_stream = MultipleChoiceStream(
                data_config.mc_train_manifest,
                split="train",
                shuffle=True,
                shuffle_seed=data_config.mc_shuffle_seed,
            )
            if data_config.mc_val_manifest:
                self.mc_val_stream = MultipleChoiceStream(
                    data_config.mc_val_manifest,
                    split="val",
                    shuffle=False,
                    shuffle_seed=data_config.mc_shuffle_seed,
                )
        self.tokenizer_hash = tokenizer_hash
        base_manifest_hash = dataset_manifest_hash or getattr(self.train_stream, "manifest_hash", None)
        if self.mc_stream is not None:
            self.dataset_manifest_hash = combined_manifest_hash(
                base_manifest_hash,
                self.mc_stream.manifest_hash,
                self.mc_val_stream.manifest_hash if self.mc_val_stream is not None else None,
            )
        else:
            self.dataset_manifest_hash = base_manifest_hash
        if self.mc_stream is not None and self.tokenizer_hash is not None:
            if self.mc_stream.tokenizer_hash != self.tokenizer_hash:
                raise ValueError("multiple-choice tokenizer hash differs from training tokenizer")
        self._val_initial_state = self.val_stream.state_dict() if self.val_stream is not None else None
        self._mc_val_initial_state = (
            self.mc_val_stream.state_dict() if self.mc_val_stream is not None else None
        )
        self.telemetry = GPUTelemetry(self.device)
        self.control_path = self.run_dir / "control.json"
        self._started_monotonic = time.monotonic()
        self._last_checkpoint_monotonic = self._started_monotonic
        self._next_eval_tokens = self.config.eval_every_tokens if self.config.eval_every_tokens > 0 else math.inf

    def _emit(self, message: str) -> None:
        print(message, flush=True)
        self.log_handle.write(message + "\n")

    def _make_stream(self, validation: bool):
        manifest = self.data_config.val_manifest if validation else self.data_config.train_manifest
        if manifest:
            return MMapTokenStream(
                manifest,
                shuffle_shards=False if validation else self.data_config.shuffle_shards,
                shuffle_seed=self.data_config.shuffle_seed,
                split="val" if validation else "train",
            )
        return SyntheticTokenStream(self.model_config.vocab_size, self.config.seed + (1 if validation else 0))

    @property
    def target_tokens(self) -> int:
        if self.config.max_steps is not None:
            return int(self.config.max_steps * self.effective_global_batch_tokens)
        return int(self.config.max_tokens or self.config.total_tokens)

    @property
    def effective_global_batch_tokens(self) -> int:
        return self.config.effective_global_batch_tokens

    def request_stop(self, reason: str = "signal") -> None:
        self.stop_requested = True
        self.stop_reason = reason

    def _check_control(self) -> None:
        try:
            if self.control_path.is_file():
                import json

                payload = json.loads(self.control_path.read_text())
                if payload.get("action") in {"stop", "request_stop"}:
                    self.request_stop("control.json")
                elif payload.get("action") in {"checkpoint", "save_checkpoint"}:
                    self.checkpoint_requested = True
                self.control_path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
        if (self.run_dir / "STOP_REQUESTED").exists():
            self.request_stop("STOP_REQUESTED")

    def _autocast(self):
        if self.config.precision != "bf16":
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def _batch_to_device(self) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_np, targets_np = self.train_stream.next_batch(self.config.micro_batch_size, self.config.sequence_length)
        inputs = torch.from_numpy(inputs_np.astype("int64", copy=True))
        targets = torch.from_numpy(targets_np.astype("int64", copy=True))
        if self.device.type == "cuda":
            if self.config.pin_memory:
                inputs = inputs.pin_memory()
                targets = targets.pin_memory()
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
        else:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        return inputs, targets

    def _multiple_choice_batch_to_device(self) -> MultipleChoiceBatch:
        if self.mc_stream is None:
            raise RuntimeError("multiple-choice stream is not configured")
        batch = collate_multiple_choice(
            self.mc_stream.next_batch(self.data_config.mc_batch_size),
            pad_id=self.mc_stream.pad_id,
            pad_to_length=self.config.mc_sequence_length,
        )
        return MultipleChoiceBatch(
            input_ids=batch.input_ids.to(self.device),
            continuation_mask=batch.continuation_mask.to(self.device),
            group_offsets=batch.group_offsets,
            positive_indices=batch.positive_indices,
            example_ids=batch.example_ids,
        )

    def _data_position(self) -> dict[str, Any]:
        state = self.train_stream.state_dict()
        return {
            "epoch": state.get("epoch", 0),
            "shard_index": state.get("shard_index"),
            "offset_in_shard": state.get("offset_in_shard", state.get("position")),
        }

    def _mc_data_position(self) -> dict[str, Any] | None:
        if self.mc_stream is None:
            return None
        state = self.mc_stream.state_dict()
        return {
            "epoch": state["epoch"],
            "order_position": state["order_position"],
            "example_count": len(self.mc_stream.examples),
        }

    def _write_status(self, status: str | None = None, **extra: Any) -> None:
        if status is not None:
            self.state.status = status
        elapsed = max(0.0, time.monotonic() - self._started_monotonic)
        payload = {
            "run_id": self.run_dir.name,
            "status": self.state.status,
            "model_name": self.model_config.model_name,
            "parameters": self.raw_model.parameter_count,
            "objective": self.config.objective,
            "device": str(self.device),
            "step": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "tokens_target": self.target_tokens,
            "epoch": self.state.epoch,
            "started_at": self.state.started_at,
            "elapsed_seconds": elapsed,
            "learning_rate": self.scheduler.last_lr,
            "last_checkpoint_at": self.state.last_checkpoint_at,
            "last_checkpoint_tokens": self.state.last_checkpoint_tokens,
            "current_data": self._data_position(),
            "current_multiple_choice_data": self._mc_data_position(),
            "optimizer": self.optimizer_impl,
            "compile": self.config.compile,
            "compile_mode": self.config.compile_mode if self.config.compile else None,
            **extra,
        }
        atomic_json_write(self.run_dir / "status.json", payload)

    def _checkpoint_payload(self) -> dict[str, Any]:
        payload = {
            "format_version": 1,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "trainer": asdict(self.state),
            "data": self.train_stream.state_dict(),
            "rng": capture_rng_state(),
        }
        if self.mc_stream is not None:
            payload["multiple_choice_data"] = self.mc_stream.state_dict()
        return payload

    def _checkpoint(self, reason: str, *, permanent_name: str | None = None) -> Path:
        base = permanent_name or f"step_{self.state.step:09d}"
        name = base
        suffix = 1
        while (self.checkpoints.root / name).exists():
            suffix += 1
            name = f"{base}_{suffix}"
        metadata = default_checkpoint_metadata(
            model_config=self.model_config.to_dict(),
            training_config=asdict(self.config),
            tokenizer_hash=self.tokenizer_hash,
            dataset_manifest_hash=self.dataset_manifest_hash,
            step=self.state.step,
            tokens_seen=self.state.tokens_seen,
            reason=reason,
        )
        previous_checkpoint_at = self.state.last_checkpoint_at
        previous_checkpoint_tokens = self.state.last_checkpoint_tokens
        previous_permanent_tokens = self.state.last_permanent_tokens
        # Store checkpoint markers in the payload so interval and milestone
        # decisions continue identically after an exact resume.
        self.state.last_checkpoint_at = metadata["created_at"]
        self.state.last_checkpoint_tokens = self.state.tokens_seen
        if permanent_name is not None:
            self.state.last_permanent_tokens = self.state.tokens_seen
        try:
            path = self.checkpoints.save(name, self._checkpoint_payload(), metadata)
        except BaseException:
            self.state.last_checkpoint_at = previous_checkpoint_at
            self.state.last_checkpoint_tokens = previous_checkpoint_tokens
            self.state.last_permanent_tokens = previous_permanent_tokens
            raise
        self._last_checkpoint_monotonic = time.monotonic()
        self._write_status(last_checkpoint=str(path))
        self._emit(f"checkpoint saved: {path} ({reason})")
        return path

    def load_checkpoint(self, reference: str | Path | None = "auto") -> Path:
        payload, path = self.checkpoints.load(reference)
        metadata = self.checkpoints.metadata(path)
        checkpoint_model_config = metadata.get("model_config")
        if checkpoint_model_config is not None and checkpoint_model_config != self.model_config.to_dict():
            raise ValueError("model config differs from checkpoint")
        checkpoint_tokenizer_hash = metadata.get("tokenizer_hash")
        if "tokenizer_hash" in metadata and checkpoint_tokenizer_hash != self.tokenizer_hash:
            raise ValueError("tokenizer hash differs from checkpoint")
        checkpoint_manifest_hash = metadata.get("dataset_manifest_hash")
        if "dataset_manifest_hash" in metadata and checkpoint_manifest_hash != self.dataset_manifest_hash:
            raise ValueError("dataset manifest hash differs from checkpoint")
        self.raw_model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        trainer_values = payload.get("trainer", {})
        known = {field for field in asdict(self.state)}
        self.state = TrainerState(**{key: value for key, value in trainer_values.items() if key in known})
        self.train_stream.load_state_dict(payload["data"])
        if self.mc_stream is not None:
            multiple_choice_state = payload.get("multiple_choice_data")
            if not isinstance(multiple_choice_state, dict):
                raise ValueError("checkpoint has no multiple-choice data state")
            self.mc_stream.load_state_dict(multiple_choice_state)
        restore_rng_state(payload["rng"])
        self._next_eval_tokens = (
            ((self.state.tokens_seen // self.config.eval_every_tokens) + 1) * self.config.eval_every_tokens
            if self.config.eval_every_tokens > 0
            else math.inf
        )
        self._started_monotonic = time.monotonic()
        self._last_checkpoint_monotonic = self._started_monotonic
        meta = self.checkpoints.metadata(path)
        self._emit(
            "resumed checkpoint: "
            f"previous step={self.state.step}, tokens={self.state.tokens_seen}, "
            f"shard={self._data_position()['shard_index']}, offset={self._data_position()['offset_in_shard']}, "
            f"lr={self.scheduler.last_lr:.6g}, created={meta.get('created_at', 'unknown')}"
        )
        self._write_status("running", resumed_from=str(path))
        return path

    def initialize_from_checkpoint(self, reference: str | Path) -> Path:
        """Transfer a trained model to a new dataset without resuming its cursor.

        This intentionally preserves model/optimizer/RNG state while resetting
        the new dataset progress to zero. Exact ``load_checkpoint`` remains strict and
        is still the path for resuming an interrupted run on the same manifest.
        """

        payload, path = self.checkpoints.load(reference)
        metadata = self.checkpoints.metadata(path)
        checkpoint_model_config = metadata.get("model_config")
        if checkpoint_model_config is not None and checkpoint_model_config != self.model_config.to_dict():
            raise ValueError("model config differs from checkpoint")
        checkpoint_tokenizer_hash = metadata.get("tokenizer_hash")
        if (
            checkpoint_tokenizer_hash is not None
            and self.tokenizer_hash is not None
            and checkpoint_tokenizer_hash != self.tokenizer_hash
        ):
            raise ValueError("tokenizer hash differs from checkpoint")
        self.raw_model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng"])
        self.state = TrainerState(started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        self.scheduler.step(0)
        self._next_eval_tokens = self.config.eval_every_tokens if self.config.eval_every_tokens > 0 else math.inf
        self._emit(
            "initialized from checkpoint: "
            f"previous step={metadata.get('step', 'unknown')}, "
            f"tokens={metadata.get('tokens_seen', 'unknown')}, source={path}"
        )
        self._write_status("initialized", initialized_from=str(path))
        return path

    def _should_checkpoint(self) -> tuple[bool, str, str | None]:
        if self.config.checkpoint_every_steps and self.state.step % self.config.checkpoint_every_steps == 0:
            return True, "step", None
        permanent = self.config.permanent_checkpoint_every_tokens
        if permanent > 0:
            milestone = (self.state.tokens_seen // permanent) * permanent
            if milestone > self.state.last_permanent_tokens and milestone > 0:
                return True, "permanent_tokens", f"tokens_{milestone // 1_000_000:06d}m"
        if self.config.checkpoint_every_tokens:
            interval = self.config.checkpoint_every_tokens
            if self.state.tokens_seen // interval > self.state.last_checkpoint_tokens // interval:
                return True, "tokens", None
        if self.config.checkpoint_every_minutes > 0 and time.monotonic() - self._last_checkpoint_monotonic >= self.config.checkpoint_every_minutes * 60:
            return True, "time", None
        return False, "", None

    def evaluate(self) -> float | None:
        if self.val_stream is None or self.data_config.eval_batches <= 0:
            return None
        if self._val_initial_state is not None:
            self.val_stream.load_state_dict(self._val_initial_state)
        was_training = self.train_model.training
        self.train_model.eval()
        loss_sum = 0.0
        batches = 0
        with torch.no_grad():
            for _ in range(self.data_config.eval_batches):
                inputs_np, targets_np = self.val_stream.next_batch(
                    self.config.micro_batch_size, self.config.sequence_length
                )
                inputs = torch.from_numpy(inputs_np.astype("int64", copy=True)).to(self.device)
                targets = torch.from_numpy(targets_np.astype("int64", copy=True)).to(self.device)
                with self._autocast():
                    output = self.train_model(inputs, targets)
                if output.loss is not None:
                    loss_sum += float(output.loss.detach().float().cpu())
                    batches += 1
        if was_training:
            self.train_model.train()
        return loss_sum / batches if batches else None

    def evaluate_multiple_choice(self) -> dict[str, float] | None:
        if self.mc_val_stream is None or self.data_config.mc_eval_batches <= 0:
            return None
        if self._mc_val_initial_state is not None:
            self.mc_val_stream.load_state_dict(self._mc_val_initial_state)
        was_training = self.train_model.training
        self.train_model.eval()
        loss_sum = 0.0
        correct = 0.0
        examples = 0.0
        margins: list[float] = []
        batches = 0
        with torch.no_grad():
            for _ in range(self.data_config.mc_eval_batches):
                raw_batch = collate_multiple_choice(
                    self.mc_val_stream.next_batch(self.data_config.mc_batch_size),
                    pad_id=self.mc_val_stream.pad_id,
                    pad_to_length=self.config.mc_sequence_length,
                )
                batch = MultipleChoiceBatch(
                    input_ids=raw_batch.input_ids.to(self.device),
                    continuation_mask=raw_batch.continuation_mask.to(self.device),
                    group_offsets=raw_batch.group_offsets,
                    positive_indices=raw_batch.positive_indices,
                    example_ids=raw_batch.example_ids,
                )
                with self._autocast():
                    rank_loss, diagnostics = batch_ranking_loss(
                        self.train_model,
                        batch,
                        temperature=self.config.ranking_temperature,
                        negative_mode=self.config.ranking_negative_mode,
                    )
                loss_sum += float(rank_loss.detach().float().cpu())
                correct += float(diagnostics["correct"].detach().cpu())
                examples += float(diagnostics["examples"].detach().cpu())
                margins.append(float(diagnostics["mean_positive_margin"].detach().float().cpu()))
                batches += 1
        if was_training:
            self.train_model.train()
        if not batches:
            return None
        return {
            "validation_ranking_loss": loss_sum / batches,
            "validation_ranking_accuracy": correct / max(1.0, examples),
            "validation_positive_margin": sum(margins) / len(margins),
        }

    def _log_interval(
        self,
        interval_loss: torch.Tensor,
        interval_steps: int,
        interval_started: float,
        last_grad_norm: torch.Tensor | None,
        validation_loss: float | None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.monotonic()
        elapsed = max(1e-9, now - interval_started)
        loss = float((interval_loss / max(1, interval_steps)).float().cpu())
        if self.state.smoothed_loss is None:
            self.state.smoothed_loss = loss
        else:
            self.state.smoothed_loss = 0.95 * self.state.smoothed_loss + 0.05 * loss
        tokens = interval_steps * self.effective_global_batch_tokens
        throughput = tokens / elapsed
        total_elapsed = max(0.0, now - self._started_monotonic)
        remaining = max(0, self.target_tokens - self.state.tokens_seen)
        eta = remaining / throughput if throughput > 0 else None
        telemetry = self.telemetry.sample()
        position = self._data_position()
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "step": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "tokens_target": self.target_tokens,
            "train_loss": loss,
            "smoothed_loss": self.state.smoothed_loss,
            "validation_loss": validation_loss if validation_loss is not None else self.state.validation_loss,
            "perplexity": math.exp(min(20.0, validation_loss)) if validation_loss is not None else None,
            "learning_rate": self.scheduler.last_lr,
            "tokens_per_sec": throughput,
            "step_time": elapsed / max(1, interval_steps),
            "elapsed_time": total_elapsed,
            "eta_seconds": eta,
            "grad_norm": float(last_grad_norm.detach().float().cpu()) if last_grad_norm is not None else None,
            "current_shard": position["shard_index"],
            "current_offset": position["offset_in_shard"],
            "epoch": position["epoch"],
            "checkpoint_status": self.state.last_checkpoint_at,
            "global_batch_tokens": self.effective_global_batch_tokens,
            "micro_batch_size": self.config.micro_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "sequence_length": self.config.sequence_length,
            "optimizer": self.optimizer_impl,
            "objective": self.config.objective,
            **telemetry,
        }
        if extra_metrics:
            record.update(extra_metrics)
        self.metrics.write(record)
        self._write_status(
            current_metrics={key: value for key, value in record.items() if key not in {"timestamp"}},
        )
        self._emit(
            f"step={self.state.step} tokens={self.state.tokens_seen}/{self.target_tokens} "
            f"loss={loss:.4f} lr={self.scheduler.last_lr:.3g} "
            f"tok/s={throughput:,.0f} eta={eta if eta is not None else '-'}"
        )
        return now, throughput

    def _run_contrastive(self) -> TrainerState:
        """Train replay LM and multiple-choice ranking losses together."""

        self.train_model.train()
        if self.state.tokens_seen >= self.target_tokens:
            self.state.status = "completed"
            self._write_status()
            self.telemetry.close()
            self.metrics.close()
            self.log_handle.close()
            return self.state
        self.state.status = "running"
        self._write_status()
        interval_started = time.monotonic()
        interval_steps = 0
        interval_loss = torch.zeros((), device=self.device)
        interval_lm_loss = torch.zeros((), device=self.device)
        interval_ranking_loss = torch.zeros((), device=self.device)
        interval_correct = 0.0
        interval_examples = 0.0
        interval_margins: list[float] = []
        last_grad_norm: torch.Tensor | None = None
        try:
            while self.state.tokens_seen < self.target_tokens:
                self.optimizer.zero_grad(set_to_none=True)
                step_lm_loss = torch.zeros((), device=self.device)
                step_ranking_loss = torch.zeros((), device=self.device)
                step_correct = 0.0
                step_examples = 0.0
                step_margins: list[float] = []
                for _ in range(self.config.gradient_accumulation_steps):
                    inputs, targets = self._batch_to_device()
                    with self._autocast():
                        output = self.train_model(inputs, targets)
                        if output.loss is None:
                            raise RuntimeError("model did not return a loss")
                        lm_loss = output.loss
                    (self.config.lm_loss_weight * lm_loss / self.config.gradient_accumulation_steps).backward()

                    multiple_choice_batch = self._multiple_choice_batch_to_device()
                    with self._autocast():
                        ranking_loss, diagnostics = batch_ranking_loss(
                            self.train_model,
                            multiple_choice_batch,
                            temperature=self.config.ranking_temperature,
                            negative_mode=self.config.ranking_negative_mode,
                        )
                    (
                        self.config.ranking_loss_weight
                        * ranking_loss
                        / self.config.gradient_accumulation_steps
                    ).backward()
                    step_lm_loss = step_lm_loss + lm_loss.detach()
                    step_ranking_loss = step_ranking_loss + ranking_loss.detach()
                    step_correct += float(diagnostics["correct"].detach().cpu())
                    step_examples += float(diagnostics["examples"].detach().cpu())
                    step_margins.append(
                        float(diagnostics["mean_positive_margin"].detach().float().cpu())
                    )

                if self.config.grad_clip > 0:
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), self.config.grad_clip, foreach=True
                    )
                self.optimizer.step()
                self.state.step += 1
                self.state.tokens_seen += self.effective_global_batch_tokens
                self.state.epoch = int(self._data_position()["epoch"])
                self.scheduler.step(self.state.tokens_seen)

                mean_lm_loss = step_lm_loss / self.config.gradient_accumulation_steps
                mean_ranking_loss = step_ranking_loss / self.config.gradient_accumulation_steps
                weighted_loss = (
                    self.config.lm_loss_weight * mean_lm_loss
                    + self.config.ranking_loss_weight * mean_ranking_loss
                )
                interval_steps += 1
                interval_loss = interval_loss + weighted_loss
                interval_lm_loss = interval_lm_loss + mean_lm_loss
                interval_ranking_loss = interval_ranking_loss + mean_ranking_loss
                interval_correct += step_correct
                interval_examples += step_examples
                interval_margins.extend(step_margins)

                validation_mc: dict[str, float] | None = None
                if self.state.tokens_seen >= self._next_eval_tokens:
                    self.state.validation_loss = self.evaluate()
                    validation_mc = self.evaluate_multiple_choice()
                    self.state.last_eval_tokens = self.state.tokens_seen
                    self._next_eval_tokens = (
                        ((self.state.tokens_seen // self.config.eval_every_tokens) + 1)
                        * self.config.eval_every_tokens
                        if self.config.eval_every_tokens > 0
                        else math.inf
                    )

                if interval_steps >= max(1, self.config.log_every_steps):
                    extras: dict[str, Any] = {
                        "lm_loss": float(
                            (interval_lm_loss / interval_steps).float().cpu()
                        ),
                        "ranking_loss": float(
                            (interval_ranking_loss / interval_steps).float().cpu()
                        ),
                        "lm_loss_weight": self.config.lm_loss_weight,
                        "ranking_loss_weight": self.config.ranking_loss_weight,
                        "ranking_temperature": self.config.ranking_temperature,
                        "ranking_negative_mode": self.config.ranking_negative_mode,
                        "ranking_accuracy": interval_correct / max(1.0, interval_examples),
                        "positive_margin": (
                            sum(interval_margins) / len(interval_margins)
                            if interval_margins
                            else None
                        ),
                    }
                    if validation_mc is not None:
                        extras.update(validation_mc)
                    interval_started, _ = self._log_interval(
                        interval_loss,
                        interval_steps,
                        interval_started,
                        last_grad_norm,
                        self.state.validation_loss,
                        extras,
                    )
                    interval_steps = 0
                    interval_loss = torch.zeros((), device=self.device)
                    interval_lm_loss = torch.zeros((), device=self.device)
                    interval_ranking_loss = torch.zeros((), device=self.device)
                    interval_correct = 0.0
                    interval_examples = 0.0
                    interval_margins = []
                    self._check_control()

                due, reason, permanent_name = self._should_checkpoint()
                if due:
                    self._checkpoint(reason, permanent_name=permanent_name)
                if self.checkpoint_requested:
                    self._checkpoint("control")
                    self.checkpoint_requested = False
                if self.stop_requested:
                    self._emit(f"Received {self.stop_reason}; Saving emergency checkpoint...")
                    self.state.status = "stopping"
                    self._checkpoint("emergency")
                    self.state.status = "stopped"
                    self._write_status()
                    return self.state

            if interval_steps:
                self._log_interval(
                    interval_loss,
                    interval_steps,
                    interval_started,
                    last_grad_norm,
                    self.state.validation_loss,
                    {
                        "lm_loss": float((interval_lm_loss / interval_steps).float().cpu()),
                        "ranking_loss": float(
                            (interval_ranking_loss / interval_steps).float().cpu()
                        ),
                        "lm_loss_weight": self.config.lm_loss_weight,
                        "ranking_loss_weight": self.config.ranking_loss_weight,
                        "ranking_temperature": self.config.ranking_temperature,
                        "ranking_negative_mode": self.config.ranking_negative_mode,
                        "ranking_accuracy": interval_correct / max(1.0, interval_examples),
                        "positive_margin": (
                            sum(interval_margins) / len(interval_margins)
                            if interval_margins
                            else None
                        ),
                    },
                )
            self._checkpoint("final")
            self.state.status = "completed"
            self._write_status()
            return self.state
        except BaseException as exc:
            self.state.status = "error"
            self._write_status(error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.telemetry.close()
            self.metrics.close()
            self.log_handle.close()

    def run(self) -> TrainerState:
        if self.config.objective == "contrastive_mc":
            return self._run_contrastive()
        self.train_model.train()
        if self.state.tokens_seen >= self.target_tokens:
            self.state.status = "completed"
            self._write_status()
            self.telemetry.close()
            self.metrics.close()
            self.log_handle.close()
            return self.state
        self.state.status = "running"
        self._write_status()
        interval_started = time.monotonic()
        interval_steps = 0
        interval_loss = torch.zeros((), device=self.device)
        last_grad_norm: torch.Tensor | None = None
        try:
            while self.state.tokens_seen < self.target_tokens:
                self.optimizer.zero_grad(set_to_none=True)
                step_loss = torch.zeros((), device=self.device)
                for _ in range(self.config.gradient_accumulation_steps):
                    inputs, targets = self._batch_to_device()
                    with self._autocast():
                        output = self.train_model(inputs, targets)
                        if output.loss is None:
                            raise RuntimeError("model did not return a loss")
                        raw_loss = output.loss
                        (raw_loss / self.config.gradient_accumulation_steps).backward()
                    step_loss = step_loss + raw_loss.detach()
                if self.config.grad_clip > 0:
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), self.config.grad_clip, foreach=True
                    )
                self.optimizer.step()
                self.state.step += 1
                self.state.tokens_seen += self.effective_global_batch_tokens
                self.state.epoch = int(self._data_position()["epoch"])
                self.scheduler.step(self.state.tokens_seen)
                interval_steps += 1
                interval_loss = interval_loss + step_loss / self.config.gradient_accumulation_steps

                if self.state.tokens_seen >= self._next_eval_tokens:
                    self.state.validation_loss = self.evaluate()
                    self.state.last_eval_tokens = self.state.tokens_seen
                    self._next_eval_tokens = (
                        ((self.state.tokens_seen // self.config.eval_every_tokens) + 1) * self.config.eval_every_tokens
                        if self.config.eval_every_tokens > 0
                        else math.inf
                    )

                if interval_steps >= max(1, self.config.log_every_steps):
                    interval_started, _ = self._log_interval(
                        interval_loss, interval_steps, interval_started, last_grad_norm, self.state.validation_loss
                    )
                    interval_steps = 0
                    interval_loss = torch.zeros((), device=self.device)
                    self._check_control()

                due, reason, permanent_name = self._should_checkpoint()
                if due:
                    self._checkpoint(reason, permanent_name=permanent_name)
                if self.checkpoint_requested:
                    self._checkpoint("control")
                    self.checkpoint_requested = False
                if self.stop_requested:
                    self._emit(f"Received {self.stop_reason}; Saving emergency checkpoint...")
                    self.state.status = "stopping"
                    self._checkpoint("emergency")
                    self.state.status = "stopped"
                    self._write_status()
                    return self.state

            if interval_steps:
                self._log_interval(interval_loss, interval_steps, interval_started, last_grad_norm, self.state.validation_loss)
            self._checkpoint("final")
            self.state.status = "completed"
            self._write_status()
            return self.state
        except BaseException as exc:
            self.state.status = "error"
            self._write_status(error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.telemetry.close()
            self.metrics.close()
            self.log_handle.close()


def install_signal_handlers(trainer: Trainer) -> None:
    def handle(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        trainer.request_stop(name)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
