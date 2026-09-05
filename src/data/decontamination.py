from __future__ import annotations

import re
import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np


WORD_RE = re.compile(r"[a-z0-9]+", re.ASCII)
MASK64 = (1 << 64) - 1
BASE = 1_000_003


def normalized_words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def shingle_hashes(text: str, width: int = 12) -> Iterable[int]:
    """Yield deterministic rolling hashes for normalized word shingles."""

    words = normalized_words(text)
    if len(words) < width:
        return
    word_hashes = [
        int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
        for word in words
    ]
    power = pow(BASE, width - 1, 1 << 64)
    value = 0
    for item in word_hashes[:width]:
        value = (value * BASE + item) & MASK64
    yield value
    for outgoing, incoming in zip(word_hashes, word_hashes[width:], strict=False):
        value = ((value - outgoing * power) * BASE + incoming) & MASK64
        yield value


class BenchmarkDecontaminator:
    """Reject documents sharing at least two benchmark question/context shingles."""

    def __init__(self, path: str | Path, minimum_matches: int = 2) -> None:
        self.path = Path(path)
        artifact = np.load(self.path, allow_pickle=False)
        self.hashes = artifact["hashes"]
        self.masks = artifact["masks"]
        self.width = int(artifact["width"])
        self.minimum_matches = minimum_matches

    def match_mask(self, text: str) -> int:
        counts = [0, 0, 0, 0]
        for value in shingle_hashes(text, self.width):
            index = int(np.searchsorted(self.hashes, np.uint64(value)))
            if index >= len(self.hashes) or int(self.hashes[index]) != value:
                continue
            mask = int(self.masks[index])
            for benchmark in range(4):
                if mask & (1 << benchmark):
                    counts[benchmark] += 1
                    if counts[benchmark] >= self.minimum_matches:
                        return 1 << benchmark
        return 0
