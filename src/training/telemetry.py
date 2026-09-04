from __future__ import annotations

import csv
import io
import subprocess
import threading
import time
from typing import Any

import torch


class GPUTelemetry:
    """Background low-frequency nvidia-smi sampler."""

    def __init__(self, device: torch.device, interval_seconds: float = 3.0) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.last_sample_at = 0.0
        self.latest: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.device.type == "cuda":
            self._thread = threading.Thread(target=self._run, name="j3-gpu-telemetry", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def _sample_once(self) -> None:
        now = time.monotonic()
        self.last_sample_at = now
        if self.device.type != "cuda":
            latest = {
                "gpu_utilization_pct": None,
                "gpu_memory_used_mb": None,
                "gpu_memory_allocated_mb": None,
                "gpu_temperature_c": None,
                "gpu_power_w": None,
            }
            with self._lock:
                self.latest = latest
            return
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
                timeout=2.0,
                check=True,
            )
            row = next(csv.reader(io.StringIO(result.stdout.strip()), skipinitialspace=True))
            latest = {
                "gpu_utilization_pct": float(row[0]),
                "gpu_memory_used_mb": float(row[1]),
                "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(self.device) / 2**20),
                "gpu_temperature_c": float(row[2]),
                "gpu_power_w": float(row[3]),
            }
        except (OSError, StopIteration, subprocess.SubprocessError, ValueError, IndexError):
            latest = {
                "gpu_utilization_pct": None,
                "gpu_memory_used_mb": None,
                "gpu_memory_allocated_mb": float(torch.cuda.memory_allocated(self.device) / 2**20),
                "gpu_temperature_c": None,
                "gpu_power_w": None,
            }
        with self._lock:
            self.latest = latest

    def sample(self, force: bool = False) -> dict[str, Any]:
        # Normal metrics calls only copy the most recent background sample.
        # `force` is retained for diagnostics and intentionally synchronous.
        if force:
            self._sample_once()
        with self._lock:
            return dict(self.latest)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
