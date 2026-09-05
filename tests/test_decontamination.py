from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.decontamination import BenchmarkDecontaminator, shingle_hashes


def test_decontamination_requires_two_benchmark_shingles(tmp_path: Path) -> None:
    benchmark = "the quick brown fox jumps over the lazy dog near the river every morning"
    hashes = sorted(set(shingle_hashes(benchmark, width=8)))
    path = tmp_path / "index.npz"
    np.savez_compressed(
        path,
        hashes=np.asarray(hashes, dtype=np.uint64),
        masks=np.asarray([1] * len(hashes), dtype=np.uint8),
        width=np.asarray(8, dtype=np.uint16),
    )
    matcher = BenchmarkDecontaminator(path, minimum_matches=2)
    assert matcher.match_mask(benchmark) == 1
    assert matcher.match_mask("the quick brown fox jumps over the lazy") == 0
    assert matcher.match_mask("completely unrelated ordinary web content") == 0
