from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import DecoderLM
from src.training.config import load_config
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short independent LR proxy runs before a long pretrain")
    parser.add_argument("--config", default="configs/pretrain_1b.yaml")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 2e-4, 3e-4, 5e-4, 8e-4])
    parser.add_argument("--tokens", type=int, default=5_000_000)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/lr-range")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config, base_train, data_config, _ = load_config(args.config)
    if args.tokens <= 0:
        raise SystemExit("--tokens must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for learning_rate in args.learning_rates:
        train_config = copy.deepcopy(base_train)
        train_config.device = args.device or train_config.device
        train_config.compile = args.compile
        train_config.sequence_length = args.sequence_length
        train_config.micro_batch_size = args.micro_batch_size
        train_config.gradient_accumulation_steps = 1
        train_config.global_batch_tokens = args.sequence_length * args.micro_batch_size
        train_config.max_tokens = args.tokens
        train_config.max_steps = None
        train_config.total_tokens = args.tokens
        train_config.warmup_tokens = max(1, args.tokens // 20)
        train_config.learning_rate = learning_rate
        train_config.min_learning_rate = learning_rate / 10
        train_config.log_every_steps = max(1, train_config.log_every_steps)
        train_config.eval_every_tokens = 0
        train_config.checkpoint_every_minutes = 0
        train_config.permanent_checkpoint_every_tokens = 0
        train_config.keep_last_n = 1
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
                "final_loss": last.get("train_loss"),
                "smoothed_loss": last.get("smoothed_loss"),
                "status": final_state.status,
                "run_dir": str(run_dir),
            }
        )
    payload = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": args.config, "results": results}
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
