from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MMapTokenStream:
    """Sequential, exact-resume token stream over uint16 binary shards.

    A training batch consumes B*T tokens and uses the following token as the
    final target. Shard crossings are handled without padding, so every
    position is a real next-token training example.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        shuffle_shards: bool = True,
        shuffle_seed: int = 1337,
        split: str = "train",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.manifest_hash = manifest_sha256(self.manifest_path)
        self.split = split
        self.dtype = np.dtype(self.manifest.get("dtype", "uint16"))
        if self.dtype != np.dtype("uint16"):
            raise ValueError(f"only uint16 shards are supported, got {self.dtype}")
        shard_entries = self.manifest.get(f"{split}_shards")
        if shard_entries is None:
            shard_entries = self.manifest.get("shards", []) if split == "train" else []
        if not shard_entries:
            raise ValueError(f"manifest has no {split} shards: {self.manifest_path}")
        self.shards = [self._resolve_entry(entry) for entry in shard_entries]
        self.shard_lengths = [int(entry.get("token_count", self._count_file(path))) for entry, path in zip(shard_entries, self.shards)]
        self.total_tokens = sum(self.shard_lengths)
        if self.total_tokens < 2:
            raise ValueError("token stream must contain at least two tokens")
        self.shuffle_shards = shuffle_shards
        self.shuffle_seed = int(shuffle_seed)
        self.epoch = 0
        self.order = list(range(len(self.shards)))
        self.order_position = 0
        self.offset_in_shard = 0
        self._maps: dict[int, np.memmap] = {}
        self._reset_order()

    def _resolve_entry(self, entry: str | dict[str, Any]) -> Path:
        raw = entry if isinstance(entry, str) else entry["path"]
        path = Path(raw)
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        return path

    @staticmethod
    def _count_file(path: Path) -> int:
        return path.stat().st_size // np.dtype("uint16").itemsize

    def _reset_order(self) -> None:
        self.order = list(range(len(self.shards)))
        if self.shuffle_shards:
            random.Random(self.shuffle_seed + self.epoch).shuffle(self.order)
        self.order_position = 0
        self.offset_in_shard = 0

    def _map(self, physical_index: int) -> np.memmap:
        mapped = self._maps.get(physical_index)
        if mapped is None:
            mapped = np.memmap(self.shards[physical_index], mode="r", dtype=self.dtype)
            self._maps[physical_index] = mapped
        return mapped

    def _advance_shard(self) -> None:
        self.order_position += 1
        self.offset_in_shard = 0
        if self.order_position >= len(self.order):
            self.epoch += 1
            self._reset_order()

    def _read_tokens(self, count: int) -> np.ndarray:
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            physical_index = self.order[self.order_position]
            mapped = self._map(physical_index)
            available = self.shard_lengths[physical_index] - self.offset_in_shard
            take = min(remaining, available)
            pieces.append(np.asarray(mapped[self.offset_in_shard : self.offset_in_shard + take]))
            self.offset_in_shard += take
            remaining -= take
            if self.offset_in_shard == self.shard_lengths[physical_index]:
                self._advance_shard()
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces)

    def _peek_token(self) -> np.uint16:
        """Read the next token without moving the exact-resume cursor."""

        physical_index = self.order[self.order_position]
        mapped = self._map(physical_index)
        return np.asarray(mapped[self.offset_in_shard], dtype=np.uint16)

    def next_batch(self, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        token_count = batch_size * seq_len
        flat = self._read_tokens(token_count)
        inputs = flat.reshape(batch_size, seq_len)
        targets = np.empty_like(flat)
        targets[:-1] = flat[1:]
        targets[-1] = self._peek_token()
        targets = targets.reshape(batch_size, seq_len)
        return inputs, targets

    def state_dict(self) -> dict[str, Any]:
        physical_index = self.order[self.order_position] if self.order_position < len(self.order) else None
        return {
            "version": 1,
            "manifest_hash": self.manifest_hash,
            "split": self.split,
            "epoch": self.epoch,
            "shard_index": physical_index,
            "order_position": self.order_position,
            "offset_in_shard": self.offset_in_shard,
            "order": list(self.order),
            "shuffle_shards": self.shuffle_shards,
            "shuffle_seed": self.shuffle_seed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("manifest_hash") != self.manifest_hash:
            raise ValueError("dataset manifest hash does not match checkpoint")
        if state.get("split", "train") != self.split:
            raise ValueError("dataset split differs from checkpoint")
        if bool(state.get("shuffle_shards", self.shuffle_shards)) != self.shuffle_shards:
            raise ValueError("shuffle_shards differs from checkpoint")
        order = [int(item) for item in state["order"]]
        if sorted(order) != list(range(len(self.shards))):
            raise ValueError("invalid shard order in checkpoint")
        self.epoch = int(state["epoch"])
        self.order = order
        self.order_position = int(state["order_position"])
        self.offset_in_shard = int(state["offset_in_shard"])
        if not 0 <= self.order_position < len(self.order):
            raise ValueError("invalid order_position in checkpoint")
        physical_index = self.order[self.order_position]
        if state.get("shard_index") != physical_index:
            raise ValueError("checkpoint shard index/order mismatch")
        if not 0 <= self.offset_in_shard <= self.shard_lengths[physical_index]:
            raise ValueError("invalid offset_in_shard in checkpoint")


class SyntheticTokenStream:
    """Deterministic fallback used only for smoke tests and benchmarks."""

    def __init__(self, vocab_size: int, seed: int = 1337) -> None:
        self.vocab_size = vocab_size
        self.seed = int(seed)
        self.position = 0
        self.epoch = 0

    def next_batch(self, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        count = batch_size * seq_len
        start = self.position
        values = (np.arange(start, start + count + 1, dtype=np.int64) * 1103515245 + self.seed) % self.vocab_size
        self.position += count
        return values[:-1].astype(np.uint16).reshape(batch_size, seq_len), values[1:].astype(np.uint16).reshape(batch_size, seq_len)

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "position": self.position, "epoch": self.epoch, "seed": self.seed, "vocab_size": self.vocab_size}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("seed") != self.seed or state.get("vocab_size") != self.vocab_size:
            raise ValueError("synthetic data configuration differs from checkpoint")
        self.position = int(state["position"])
        self.epoch = int(state.get("epoch", 0))
