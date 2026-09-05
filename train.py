from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data.tokenizer import tokenizer_sha256
from src.model import DecoderLM
from src.training.config import config_as_dict, load_config
from src.training.metrics import atomic_json_write
from src.training.trainer import Trainer, install_signal_handlers
from src.utils.environment import collect_environment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train D32-1216-R12 with resumable direct PyTorch loop")
    parser.add_argument("--config", required=True, help="YAML training configuration")
    parser.add_argument("--resume", default=None, help="auto or a checkpoint directory")
    parser.add_argument(
        "--init-from",
        default=None,
        help="initialize model/optimizer from a prior checkpoint while starting this dataset at offset zero",
    )
    parser.add_argument("--run", default=None, help="run directory, useful with --resume auto")
    parser.add_argument("--device", default=None, help="override config device, e.g. cuda or cpu")
    parser.add_argument("--max-steps", type=int, default=None, help="override the run length for a smoke test")
    return parser.parse_args()


def _actual_run_dir(config, resume: str | None, run: str | None) -> tuple[Path, str | Path | None]:
    runs_root = Path(config.runs_dir)
    if resume is None:
        run_id = config.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = runs_root / run_id
        return run_dir, None
    if run:
        run_dir = Path(run)
        if run_dir.is_symlink():
            run_dir = run_dir.resolve()
        return run_dir, "auto" if resume == "auto" else resume
    if resume == "auto":
        latest_run = runs_root / "latest"
        if not latest_run.exists() and not latest_run.is_symlink():
            raise FileNotFoundError("--resume auto requires runs/latest or --run")
        return latest_run.resolve(), "auto"
    checkpoint = Path(resume)
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    if checkpoint.name == "latest" and checkpoint.is_symlink():
        checkpoint = checkpoint.resolve()
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent, checkpoint
    if (checkpoint / "checkpoints").is_dir():
        return checkpoint, "auto"
    raise FileNotFoundError(f"cannot infer run directory from --resume {resume}")


def _make_latest_link(runs_root: Path, run_dir: Path) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    pointer = runs_root / "latest"
    temporary = runs_root / f".latest.tmp-{run_dir.name}"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(run_dir.resolve())
    temporary.replace(pointer)


def main() -> None:
    args = _parse_args()
    if args.resume is not None and args.init_from is not None:
        raise SystemExit("--resume and --init-from are mutually exclusive")
    model_config, train_config, data_config, raw_config = load_config(args.config)
    if args.device:
        train_config.device = args.device
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
        train_config.max_tokens = None
    run_dir, resume_ref = _actual_run_dir(train_config, args.resume, args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        shutil.copy2(args.config, run_dir / "config.yaml")
        (run_dir / "environment.json").write_text(json.dumps(collect_environment(), ensure_ascii=False, indent=2, sort_keys=True))
        atomic_json_write(
            run_dir / "command.json",
            {"argv": __import__("sys").argv, "config": str(Path(args.config).resolve())},
        )
        _make_latest_link(Path(train_config.runs_dir), run_dir)
    else:
        print(f"resuming run directory: {run_dir}", flush=True)

    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    model = DecoderLM(model_config)
    tokenizer_hash = tokenizer_sha256(data_config.tokenizer) if data_config.tokenizer else None
    trainer = Trainer(
        model,
        model_config,
        train_config,
        data_config,
        run_dir,
        tokenizer_hash=tokenizer_hash,
    )
    if args.init_from is not None:
        trainer.initialize_from_checkpoint(args.init_from)
    elif args.resume is not None:
        trainer.load_checkpoint(resume_ref)
    print(f"model={model_config.model_name} parameters={model.parameter_count:,}", flush=True)
    print(
        f"device={trainer.device} precision={train_config.precision} optimizer={trainer.optimizer_impl} "
        f"tokens/step={trainer.effective_global_batch_tokens:,} target={trainer.target_tokens:,}",
        flush=True,
    )
    install_signal_handlers(trainer)
    trainer.run()


if __name__ == "__main__":
    main()
