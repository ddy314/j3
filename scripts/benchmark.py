from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import DecoderLM
from src.training.config import TrainConfig, load_config
from src.training.trainer import build_optimizer, configure_torch, resolve_device
from src.utils.environment import collect_environment


class NvidiaPoller:
    def __init__(self, device: torch.device, interval: float = 0.5) -> None:
        self.device = device
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.device.index or 0),
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            fields = [item.strip() for item in result.stdout.split(",")]
            self.samples.append(
                {
                    "gpu_utilization_pct": float(fields[0]),
                    "gpu_memory_used_mb": float(fields[1]),
                    "gpu_temperature_c": float(fields[2]),
                    "gpu_power_w": float(fields[3]),
                }
            )
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self.device.type == "cuda":
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if not self.samples:
            return {
                "gpu_utilization_pct_mean": None,
                "gpu_utilization_pct_max": None,
                "gpu_temperature_c_max": None,
                "gpu_power_w_mean": None,
                "gpu_memory_used_mb_max": None,
            }
        def mean(key: str) -> float:
            return sum(sample[key] for sample in self.samples) / len(self.samples)
        return {
            "gpu_utilization_pct_mean": mean("gpu_utilization_pct"),
            "gpu_utilization_pct_max": max(sample["gpu_utilization_pct"] for sample in self.samples),
            "gpu_temperature_c_max": max(sample["gpu_temperature_c"] for sample in self.samples),
            "gpu_power_w_mean": mean("gpu_power_w"),
            "gpu_memory_used_mb_max": max(sample["gpu_memory_used_mb"] for sample in self.samples),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure eager and torch.compile training throughput")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--microbatches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--modes", nargs="+", default=["eager", "default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--device", default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seq-len", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--micro-batch", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _time_step(
    train_step: Callable[[], None],
    device: torch.device,
    steps: int,
) -> tuple[float, float, float, float]:
    if device.type == "cuda":
        forward_times: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        backward_times: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        optimizer_times: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        total_times: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        # train_step records the segment events through the mutable holder.
        holder: dict[str, tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]] = {}
        for index in range(steps):
            train_step(index, holder)
            forward_times.append((holder["f_start"], holder["f_end"]))
            backward_times.append((holder["b_start"], holder["b_end"]))
            optimizer_times.append((holder["o_start"], holder["o_end"]))
            total_times.append((holder["last_total_start"], holder["last_total_end"]))
        torch.cuda.synchronize(device)
        total_ms = sum(start.elapsed_time(end) for start, end in total_times) / steps
        forward_ms = sum(start.elapsed_time(end) for start, end in forward_times) / steps
        backward_ms = sum(start.elapsed_time(end) for start, end in backward_times) / steps
        optimizer_ms = sum(start.elapsed_time(end) for start, end in optimizer_times) / steps
        return total_ms, forward_ms, backward_ms, optimizer_ms
    total_times: list[float] = []
    forward_times_cpu: list[float] = []
    backward_times_cpu: list[float] = []
    optimizer_times_cpu: list[float] = []
    for index in range(steps):
        train_step(index, {"cpu_times": (forward_times_cpu, backward_times_cpu, optimizer_times_cpu), "total_times": total_times})
    return (
        sum(total_times) / steps,
        sum(forward_times_cpu) / steps,
        sum(backward_times_cpu) / steps,
        sum(optimizer_times_cpu) / steps,
    )


def run_trial(
    model_config,
    device: torch.device,
    seq_len: int,
    micro_batch_size: int,
    mode: str,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sequence_length": seq_len,
        "micro_batch_size": micro_batch_size,
        "mode": mode,
        "status": "ok",
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    try:
        torch.manual_seed(1234)
        raw_model = DecoderLM(model_config).to(device)
        raw_model.train()
        model: torch.nn.Module = raw_model
        if mode != "eager":
            model = torch.compile(raw_model, mode=mode, fullgraph=False, dynamic=False)
        train_config = TrainConfig(
            device=str(device),
            precision="bf16",
            learning_rate=3e-4,
            weight_decay=0.1,
            sequence_length=seq_len,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=1,
            global_batch_tokens=micro_batch_size * seq_len,
            total_tokens=micro_batch_size * seq_len,
            max_steps=1,
            log_every_steps=1000,
            eval_every_tokens=0,
            checkpoint_every_minutes=0,
            permanent_checkpoint_every_tokens=0,
            pin_memory=False,
        )
        optimizer, optimizer_impl = build_optimizer(raw_model, train_config)
        inputs = torch.randint(model_config.vocab_size, (micro_batch_size, seq_len), device=device)
        labels = torch.randint(model_config.vocab_size, (micro_batch_size, seq_len), device=device)
        if device.type == "cuda":
            def train_step(index: int, holder: dict[str, Any]) -> None:
                optimizer.zero_grad(set_to_none=True)
                total_start = torch.cuda.Event(enable_timing=True)
                total_end = torch.cuda.Event(enable_timing=True)
                f_start = torch.cuda.Event(enable_timing=True)
                f_end = torch.cuda.Event(enable_timing=True)
                b_start = torch.cuda.Event(enable_timing=True)
                b_end = torch.cuda.Event(enable_timing=True)
                o_start = torch.cuda.Event(enable_timing=True)
                o_end = torch.cuda.Event(enable_timing=True)
                total_start.record()
                f_start.record()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    output = model(inputs, labels)
                    loss = output.loss
                f_end.record()
                b_start.record()
                if loss is None:
                    raise RuntimeError("benchmark model returned no loss")
                loss.backward()
                b_end.record()
                o_start.record()
                optimizer.step()
                o_end.record()
                total_end.record()
                holder.update({
                    "f_start": f_start, "f_end": f_end, "b_start": b_start, "b_end": b_end,
                    "o_start": o_start, "o_end": o_end, "last_total_start": total_start, "last_total_end": total_end,
                })
        else:
            def train_step(index: int, holder: dict[str, Any]) -> None:
                optimizer.zero_grad(set_to_none=True)
                start = time.perf_counter()
                f_start = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    output = model(inputs, labels)
                    loss = output.loss
                f_end = time.perf_counter()
                if loss is None:
                    raise RuntimeError("benchmark model returned no loss")
                loss.backward()
                b_end = time.perf_counter()
                optimizer.step()
                end = time.perf_counter()
                holder["cpu_times"][0].append(f_end - f_start)
                holder["cpu_times"][1].append(b_end - f_end)
                holder["cpu_times"][2].append(end - b_end)
                holder["total_times"].append(end - start)
        for index in range(warmup):
            train_step(index, {"cpu_times": ([], [], []), "total_times": []} if device.type != "cuda" else {})
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        monitor = NvidiaPoller(device)
        monitor.start()
        total_ms, forward_ms, backward_ms, optimizer_ms = _time_step(train_step, device, steps)
        monitor_stats = monitor.stop()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
            peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
        else:
            peak_allocated = None
            peak_reserved = None
        tokens = micro_batch_size * seq_len
        result.update(
            {
                "tokens_per_step": tokens,
                "tokens_per_sec": tokens * 1000.0 / total_ms,
                "step_ms": total_ms,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_ms": optimizer_ms,
                "other_ms": max(0.0, total_ms - forward_ms - backward_ms - optimizer_ms),
                "peak_vram_allocated_mb": peak_allocated,
                "peak_vram_reserved_mb": peak_reserved,
                "optimizer": optimizer_impl,
                **monitor_stats,
            }
        )
        del model, raw_model, optimizer
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "oom", "error": str(exc)[:500]})
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def run_isolated_trial(args: argparse.Namespace, seq_len: int, micro_batch: int, mode: str) -> dict[str, Any]:
    """Run one trial in a fresh process so CUDA Graph/Inductor pools cannot leak."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        args.config,
        "--seq-len",
        str(seq_len),
        "--micro-batch",
        str(micro_batch),
        "--mode",
        mode,
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
    ]
    if args.device:
        command.extend(["--device", args.device])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    for line in reversed(completed.stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and result.get("sequence_length") == seq_len and result.get("mode") == mode:
            return result
    return {
        "sequence_length": seq_len,
        "micro_batch_size": micro_batch,
        "mode": mode,
        "status": "error",
        "error": f"worker exit={completed.returncode}: {(completed.stderr or completed.stdout)[-1000:]}",
    }


def main() -> None:
    args = parse_args()
    model_config, train_config, _, _ = load_config(args.config)
    if args.device:
        train_config.device = args.device
    configure_torch(train_config.tf32)
    device = resolve_device(train_config.device)
    if args.worker:
        if args.seq_len is None or args.micro_batch is None or args.mode is None:
            raise SystemExit("benchmark worker requires --seq-len, --micro-batch and --mode")
        print(json.dumps(run_trial(model_config, device, args.seq_len, args.micro_batch, args.mode, args.warmup, args.steps), sort_keys=True), flush=True)
        return
    trials: list[dict[str, Any]] = []
    for seq_len in args.seq_lens:
        for mode in args.modes:
            previous = None
            stagnant = 0
            for micro_batch in args.microbatches:
                trial = run_isolated_trial(args, seq_len, micro_batch, mode)
                trials.append(trial)
                if trial["status"] != "ok":
                    break
                current = trial["tokens_per_sec"]
                if previous is not None and current <= previous * 1.01:
                    stagnant += 1
                else:
                    stagnant = 0
                previous = current
                if stagnant >= 2:
                    break
    successful = [trial for trial in trials if trial["status"] == "ok"]
    best_by_sequence = {}
    for seq_len in args.seq_lens:
        candidates = [trial for trial in successful if trial["sequence_length"] == seq_len]
        if candidates:
            best_by_sequence[str(seq_len)] = max(candidates, key=lambda trial: trial["tokens_per_sec"])
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "model": model_config.to_dict(),
        "parameter_report": DecoderLM(model_config).parameter_report(),
        "environment": collect_environment(),
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "trials": trials,
        "best_by_sequence": best_by_sequence,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmarks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    fields = sorted({key for trial in trials for key in trial})
    with (output_dir / "benchmarks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(trials)
    print("BEST_BY_SEQUENCE=" + json.dumps(best_by_sequence, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
