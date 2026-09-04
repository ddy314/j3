from __future__ import annotations

import json
import os
import random
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


class CheckpointManager:
    """Atomic, portable checkpoint directories with retention policy."""

    def __init__(self, run_dir: str | Path, keep_last_n: int = 3) -> None:
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = max(0, int(keep_last_n))

    @staticmethod
    def _complete(path: Path) -> bool:
        return path.is_dir() and (path / "state.pt").is_file() and (path / "metadata.json").is_file()

    def latest(self) -> Path | None:
        pointer = self.root / "latest"
        if pointer.is_symlink():
            resolved = pointer.resolve()
            if self._complete(resolved):
                return resolved
        candidates = [path for path in self.root.iterdir() if self._complete(path) and not path.name.startswith(".")]
        if not candidates:
            return None
        return max(candidates, key=lambda path: self._metadata(path).get("step", -1))

    def resolve(self, reference: str | Path | None) -> Path | None:
        if reference is None or str(reference) == "auto":
            return self.latest()
        path = Path(reference)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.name == "latest" and path.is_symlink():
            path = path.resolve()
        if path.is_dir() and path.name != "checkpoints" and (path / "checkpoints").is_dir():
            nested = CheckpointManager(path, self.keep_last_n).latest()
            return nested
        if self._complete(path):
            return path
        raise FileNotFoundError(f"incomplete or missing checkpoint: {path}")

    @staticmethod
    def _metadata(path: Path) -> dict[str, Any]:
        try:
            return json.loads((path / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def load(self, reference: str | Path | None = "auto") -> tuple[dict[str, Any], Path]:
        path = self.resolve(reference)
        if path is None:
            raise FileNotFoundError("no complete checkpoint found")
        payload = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        return payload, path

    def save(self, name: str, payload: dict[str, Any], metadata: dict[str, Any]) -> Path:
        if not name or name in {".", ".."} or "/" in name:
            raise ValueError(f"invalid checkpoint name: {name!r}")
        target = self.root / name
        if target.exists():
            raise FileExistsError(f"checkpoint already exists: {target}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=self.root))
        try:
            torch.save(_to_cpu(payload), temporary / "state.pt", _use_new_zipfile_serialization=True)
            with (temporary / "state.pt").open("rb") as handle:
                os.fsync(handle.fileno())
            with (temporary / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(temporary)
            os.replace(temporary, target)
            _fsync_directory(self.root)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        pointer_tmp = self.root / f".latest.tmp-{os.getpid()}"
        try:
            if pointer_tmp.exists() or pointer_tmp.is_symlink():
                pointer_tmp.unlink()
            pointer_tmp.symlink_to(target.name)
            os.replace(pointer_tmp, self.root / "latest")
            _fsync_directory(self.root)
        finally:
            if pointer_tmp.exists() or pointer_tmp.is_symlink():
                pointer_tmp.unlink()
        self._prune()
        return target

    def _prune(self) -> None:
        if self.keep_last_n <= 0:
            return
        step_dirs = []
        for path in self.root.iterdir():
            if path.name.startswith("step_") and self._complete(path):
                step_dirs.append((self._metadata(path).get("step", -1), path))
        step_dirs.sort(reverse=True)
        for _, path in step_dirs[self.keep_last_n :]:
            shutil.rmtree(path)

    def metadata(self, checkpoint: str | Path | None = "auto") -> dict[str, Any]:
        path = self.resolve(checkpoint)
        return {} if path is None else self._metadata(path)


def default_checkpoint_metadata(
    *,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    tokenizer_hash: str | None,
    dataset_manifest_hash: str | None,
    step: int,
    tokens_seen: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "python": os.sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_config": model_config,
        "training_config": training_config,
        "tokenizer_hash": tokenizer_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "step": step,
        "tokens_seen": tokens_seen,
        "reason": reason,
    }
