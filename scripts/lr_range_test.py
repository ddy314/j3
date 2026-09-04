from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import DecoderLM
from src.training.config import load_config
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short independent LR proxy runs before a long pretrain")
    parser.add_argument("--config", default="configs/pretrain_1b.yaml")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 2e-4, 3e-4, 5e-4, 8e-4])
    parser.add_argument(
        "--tokens",
        type=int,
        default=20_000_000,
        help="proxy prefix length; the formal schedule remains the scheduler time base",
    )
    parser.add_argument("--sequence-length", type=int, default=None, help="override formal sequence length")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="override formal microbatch")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="override accumulation; omitted batch overrides preserve formal global tokens when divisible",
    )
    parser.add_argument("--output-dir", default="artifacts/lr-range")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the formal compile setting (default: use the config)",
    )
    return parser.parse_args()


def build_proxy_config(
    base_train,
    *,
    proxy_tokens: int,
    learning_rate: float,
    sequence_length: int | None = None,
    micro_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    device: str | None = None,
    compile: bool | None = None,
):
    """Build a proxy config without changing the formal optimizer time base.

    The default path keeps the formal sequence length, microbatch,
    accumulation, global batch, warmup, and cosine horizon. The proxy simply
    stops after a short prefix of that schedule, so an LR decision is made at
    the same optimizer batch and on the same token-indexed schedule as the
    actual run. Explicit sequence/microbatch overrides preserve the formal
    global batch by adjusting accumulation when possible.
    """

    if proxy_tokens <= 0:
        raise ValueError("proxy_tokens must be positive")
    sequence = sequence_length if sequence_length is not None else base_train.sequence_length
    micro = micro_batch_size if micro_batch_size is not None else base_train.micro_batch_size
    if sequence <= 0 or micro <= 0:
        raise ValueError("sequence_length and micro_batch_size must be positive")

    if gradient_accumulation_steps is None:
        if sequence_length is None and micro_batch_size is None:
            accumulation = base_train.gradient_accumulation_steps
        else:
            formal_batch = base_train.effective_global_batch_tokens
            per_micro_tokens = sequence * micro
            if formal_batch % per_micro_tokens:
                raise ValueError(
                    "batch override does not divide formal global_batch_tokens; "
                    "pass --gradient-accumulation-steps explicitly"
                )
            accumulation = formal_batch // per_micro_tokens
    else:
        accumulation = gradient_accumulation_steps
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    return replace(
        base_train,
        device=device or base_train.device,
        compile=base_train.compile if compile is None else compile,
        sequence_length=sequence,
        micro_batch_size=micro,
        gradient_accumulation_steps=accumulation,
        global_batch_tokens=sequence * micro * accumulation,
        max_tokens=proxy_tokens,
        max_steps=None,
        # Keep the formal schedule as the time base. max_tokens above is only
        # the stopping prefix for this proxy experiment.
        total_tokens=base_train.total_tokens,
        warmup_tokens=base_train.warmup_tokens,
        learning_rate=learning_rate,
        min_learning_rate=learning_rate / 10,
        eval_every_tokens=0,
        checkpoint_every_steps=0,
        checkpoint_every_tokens=0,
        checkpoint_every_minutes=0,
        permanent_checkpoint_every_tokens=0,
        keep_last_n=1,
    )


def main() -> None:
    args = parse_args()
    model_config, base_train, data_config, _ = load_config(args.config)
    if args.tokens <= 0:
        raise SystemExit("--tokens must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for learning_rate in args.learning_rates:
        train_config = build_proxy_config(
            base_train,
            proxy_tokens=args.tokens,
            learning_rate=learning_rate,
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=args.device,
            compile=args.compile,
        )
        run_dir = output_dir / f"lr_{learning_rate:.0e}"
        trainer = Trainer(DecoderLM(model_config), model_config, train_config, data_config, run_dir)
        final_state = trainer.run()
        metrics = []
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        last = metrics[-1] if metrics else {}
        results.append(
            {
                "learning_rate": learning_rate,
                "tokens": final_state.tokens_seen,
                "effective_global_batch_tokens": train_config.effective_global_batch_tokens,
                "final_loss": last.get("train_loss"),
                "smoothed_loss": last.get("smoothed_loss"),
                "status": final_state.status,
                "run_dir": str(run_dir),
            }
        )
    proxy_config = build_proxy_config(
        base_train,
        proxy_tokens=args.tokens,
        learning_rate=base_train.learning_rate,
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=args.device,
        compile=args.compile,
    )
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": args.config,
        "schedule_mode": "formal_prefix",
        "proxy_tokens": args.tokens,
        "formal_total_tokens": base_train.total_tokens,
        "formal_warmup_tokens": base_train.warmup_tokens,
        "formal_global_batch_tokens": base_train.effective_global_batch_tokens,
        "proxy_global_batch_tokens": proxy_config.effective_global_batch_tokens,
        "sequence_length": proxy_config.sequence_length,
        "micro_batch_size": proxy_config.micro_batch_size,
        "gradient_accumulation_steps": proxy_config.gradient_accumulation_steps,
        "compile": proxy_config.compile,
        "compile_mode": proxy_config.compile_mode if proxy_config.compile else None,
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
