from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class ShardWriter:
    """Write a stream of token ids into fsynced, fixed-size uint16 shards."""

    def __init__(self, directory: Path, prefix: str, shard_tokens: int, manifest_root: Path) -> None:
        if shard_tokens <= 0:
            raise ValueError("shard_tokens must be positive")
        self.directory = directory
        self.manifest_root = manifest_root
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.index = 0
        self.handle = None
        self.current_tokens = 0
        self.total_tokens = 0
        self.entries: list[dict[str, object]] = []

    def _open(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
        path = self.directory / f"{self.prefix}_{self.index:06d}.bin"
        self.handle = path.open("wb")
        self.current_tokens = 0
        self.entries.append({"path": str(path.relative_to(self.manifest_root)), "token_count": 0})

    def write(self, ids: list[int] | np.ndarray) -> None:
        if not len(ids):
            return
        values = np.asarray(ids, dtype=np.uint16).reshape(-1)
        start = 0
        while start < len(values):
            if self.handle is None or self.current_tokens >= self.shard_tokens:
                self._open()
            take = min(len(values) - start, self.shard_tokens - self.current_tokens)
            self.handle.write(values[start : start + take].tobytes(order="C"))
            self.current_tokens += take
            self.total_tokens += take
            self.entries[-1]["token_count"] = self.current_tokens
            start += take
            if self.current_tokens >= self.shard_tokens:
                self.index += 1

    def close(self) -> None:
        if self.handle is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None
