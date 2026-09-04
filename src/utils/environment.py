from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from src.training.checkpoint import git_commit


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "error": "nvidia-smi not found"}
    try:
        result = subprocess.run([executable], capture_output=True, text=True, timeout=5, check=True)
        return {"available": True, "output": result.stdout}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}


def collect_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "tf32_cudnn": getattr(torch.backends.cudnn, "allow_tf32", None),
        "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
        "nvidia_smi": _nvidia_smi(),
    }
    try:
        import triton

        report["triton"] = {"available": True, "version": getattr(triton, "__version__", "unknown")}
    except ImportError as exc:
        report["triton"] = {"available": False, "error": str(exc)}
    try:
        import flash_attn

        report["flash_attn"] = {"available": True, "version": getattr(flash_attn, "__version__", "unknown")}
    except ImportError as exc:
        report["flash_attn"] = {"available": False, "error": str(exc)}
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": f"{properties.major}.{properties.minor}",
                    "total_memory_mb": properties.total_memory / 2**20,
                    "multi_processor_count": properties.multi_processor_count,
                }
            )
        report["cuda_devices"] = devices
    else:
        report["cuda_devices"] = []
    return report


def write_environment(path: str | Path) -> dict[str, Any]:
    report = collect_environment()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report
