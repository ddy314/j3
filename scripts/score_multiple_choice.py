from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.multiple_choice import (  # noqa: E402
    MultipleChoiceBatch,
    MultipleChoiceStream,
    collate_multiple_choice,
)
from src.model import DecoderLM  # noqa: E402
from src.training.checkpoint import CheckpointManager  # noqa: E402
from src.training.config import load_config  # noqa: E402
from src.training.contrastive import batch_ranking_loss  # noqa: E402
from src.training.trainer import configure_torch, resolve_device  # noqa: E402


def _resolve_checkpoint(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.name == "latest" and path.is_symlink():
        path = path.resolve()
    if path.is_dir() and path.name != "checkpoints" and (path / "checkpoints").is_dir():
        resolved = CheckpointManager(path).resolve("auto")
        if resolved is None:
            raise FileNotFoundError(f"no checkpoint in {path}")
        return resolved
    if path.is_dir() and path.name == "latest" and path.is_symlink():
        return path.resolve()
    return path


def score(
    *,
    config_path: Path,
    checkpoint: Path,
    manifest_path: Path,
    split: str,
    device_spec: str | None,
    batch_size: int,
    max_batches: int | None,
) -> dict[str, Any]:
    model_config, train_config, _, _ = load_config(config_path)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if train_config.mc_sequence_length > model_config.max_seq_len:
        raise ValueError("mc_sequence_length exceeds model max_seq_len")
    configure_torch(train_config.tf32)
    device = resolve_device(device_spec or train_config.device)
    run_dir = checkpoint.parent.parent
    manager = CheckpointManager(run_dir)
    payload, resolved = manager.load(checkpoint)
    metadata = manager.metadata(resolved)
    if metadata.get("model_config") not in (None, model_config.to_dict()):
        raise ValueError("checkpoint model config differs from evaluation config")
    model = DecoderLM(model_config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    stream = MultipleChoiceStream(manifest_path, split=split, shuffle=False)
    if metadata.get("tokenizer_hash") is not None and metadata.get("tokenizer_hash") != stream.tokenizer_hash:
        raise ValueError("checkpoint and multiple-choice artifact tokenizer hashes differ")

    loss_sum = 0.0
    correct = 0.0
    examples = 0.0
    margin_sum = 0.0
    option_counts: Counter[str] = Counter()
    batches = 0
    examples_seen = 0
    target_examples = int(stream.manifest.get(f"{split}_count", 0))
    if target_examples <= 0:
        raise ValueError(f"multiple-choice manifest has no {split}_count")
    with torch.no_grad():
        while examples_seen < target_examples and (max_batches is None or batches < max_batches):
            current_batch_size = min(batch_size, target_examples - examples_seen)
            raw_batch = collate_multiple_choice(
                stream.next_batch(current_batch_size),
                pad_id=stream.pad_id,
                pad_to_length=train_config.mc_sequence_length,
            )
            batch = MultipleChoiceBatch(
                input_ids=raw_batch.input_ids.to(device),
                continuation_mask=raw_batch.continuation_mask.to(device),
                group_offsets=raw_batch.group_offsets,
                positive_indices=raw_batch.positive_indices,
                example_ids=raw_batch.example_ids,
            )
            rank_loss, diagnostics = batch_ranking_loss(
                model,
                batch,
                temperature=train_config.ranking_temperature,
                negative_mode=train_config.ranking_negative_mode,
            )
            loss_sum += float(rank_loss.float().cpu())
            correct += float(diagnostics["correct"].cpu())
            examples += float(diagnostics["examples"].cpu())
            margin_sum += float(diagnostics["mean_positive_margin"].float().cpu())
            for index in range(len(raw_batch.positive_indices)):
                option_counts[str(int(raw_batch.group_offsets[index + 1] - raw_batch.group_offsets[index]))] += 1
            batches += 1
            examples_seen += current_batch_size

    if examples <= 0:
        raise RuntimeError("multiple-choice scorer evaluated zero examples")
    return {
        "checkpoint": str(resolved),
        "manifest": str(manifest_path.resolve()),
        "split": split,
        "device": str(device),
        "batches": batches,
        "examples": int(examples),
        "correct": int(correct),
        "accuracy": correct / examples,
        "mean_ranking_loss": loss_sum / max(1, batches),
        "mean_positive_margin": margin_sum / max(1, batches),
        "option_count_histogram": dict(sorted(option_counts.items(), key=lambda item: int(item[0]))),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score tokenized multiple-choice data with causal-LM likelihood")
    parser.add_argument("--config", default="configs/contrastive_mc_posttrain.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        raise SystemExit("--max-batches must be positive when set")
    config_path = Path(args.config)
    _, _, data_config, _ = load_config(config_path)
    manifest_value = args.manifest or data_config.mc_val_manifest or data_config.mc_train_manifest
    if not manifest_value:
        raise SystemExit("--manifest is required when the config has no MC manifest")
    result = score(
        config_path=config_path,
        checkpoint=_resolve_checkpoint(args.checkpoint),
        manifest_path=Path(manifest_value),
        split=args.split,
        device_spec=args.device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
