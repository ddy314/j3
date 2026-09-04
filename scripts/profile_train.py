from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import DecoderLM
from src.training.config import load_config
from src.training.trainer import build_optimizer, configure_torch, resolve_device
from src.utils.environment import collect_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a steady-state training step with torch.profiler")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--mode", default="reduce-overhead", choices=["eager", "default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--profiler-warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=5)
    parser.add_argument("--output-dir", default="artifacts/profile")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config, train_config, _, _ = load_config(args.config)
    if args.device:
        train_config.device = args.device
    configure_torch(train_config.tf32)
    device = resolve_device(train_config.device)
    model = DecoderLM(model_config).to(device)
    model.train()
    for module in model.modules():
        module.profile_ranges = True
    if args.mode != "eager":
        model_for_train: torch.nn.Module = torch.compile(model, mode=args.mode, fullgraph=False, dynamic=False)
    else:
        model_for_train = model
    optimizer, optimizer_impl = build_optimizer(model, train_config)
    inputs = torch.randint(model_config.vocab_size, (args.micro_batch_size, args.sequence_length), device=device)
    labels = torch.randint(model_config.vocab_size, (args.micro_batch_size, args.sequence_length), device=device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autograd.profiler.record_function("train.forward"):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model_for_train(inputs, labels)
                loss = output.loss
        if loss is None:
            raise RuntimeError("profile model returned no loss")
        with torch.autograd.profiler.record_function("train.backward"):
            loss.backward()
        with torch.autograd.profiler.record_function("train.optimizer"):
            optimizer.step()

    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    schedule = torch.profiler.schedule(wait=args.wait, warmup=args.profiler_warmup, active=args.active, repeat=1)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as profiler:
        for _ in range(args.wait + args.profiler_warmup + args.active):
            step()
            profiler.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    key_averages = profiler.key_averages()
    rows = []
    for event in key_averages:
        rows.append(
            {
                "name": event.key,
                "calls": event.count,
                "self_cpu_ms": event.self_cpu_time_total / 1000.0,
                "cpu_ms": event.cpu_time_total / 1000.0,
                "self_cuda_ms": getattr(event, "self_device_time_total", 0.0) / 1000.0,
                "cuda_ms": getattr(event, "device_time_total", 0.0) / 1000.0,
                "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                "self_cuda_memory_bytes": getattr(event, "self_device_memory_usage", 0),
            }
        )
    top_cuda = sorted(rows, key=lambda row: row["self_cuda_ms"], reverse=True)
    top_cpu = sorted(rows, key=lambda row: row["self_cpu_ms"], reverse=True)
    named_ranges: dict[str, dict[str, float]] = {}
    for row in rows:
        if not (row["name"].startswith("train.") or row["name"].startswith("model.")):
            continue
        aggregate = named_ranges.setdefault(
            row["name"],
            {"name": row["name"], "calls": 0, "self_cpu_ms": 0.0, "cpu_ms": 0.0, "self_cuda_ms": 0.0, "cuda_ms": 0.0},
        )
        for key in ("calls", "self_cpu_ms", "cpu_ms", "self_cuda_ms", "cuda_ms"):
            aggregate[key] += row[key]
    segment_rows = sorted(named_ranges.values(), key=lambda row: row["self_cuda_ms"], reverse=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(device),
        "mode": args.mode,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "optimizer": optimizer_impl,
        "environment": collect_environment(),
        "peak_vram_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "breakdown": segment_rows,
        "segments": segment_rows,
        "top_cuda_kernels": top_cuda[:50],
        "top_cpu_ops": top_cpu[:50],
        "trace_dir": str(trace_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({"summary": str(output_dir / "summary.json"), "segments": segment_rows, "top_cuda": top_cuda[:15]}, indent=2))
    del model_for_train, model, optimizer
    gc.collect()


if __name__ == "__main__":
    main()
