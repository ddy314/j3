from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.decontamination import shingle_hashes


BENCHMARKS = (
    ("hellaswag", "Rowan/hellaswag", ("default",), ("ctx", "ctx_a", "ctx_b", "activity_label")),
    ("piqa", "baber/piqa", ("default",), ("goal",)),
    ("arc", "allenai/ai2_arc", ("ARC-Easy", "ARC-Challenge"), ("question",)),
    ("winogrande", "allenai/winogrande", ("winogrande_xl",), ("sentence",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an auditable public benchmark decontamination index")
    parser.add_argument("--output", default="data/decontamination/public_benchmarks.npz")
    parser.add_argument("--report", default="data/decontamination/public_benchmarks.json")
    parser.add_argument("--shingle-width", type=int, default=12)
    return parser.parse_args()


def _text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    return " ".join(str(row.get(field, "")) for field in fields if row.get(field))


def main() -> None:
    import requests
    from datasets import load_dataset

    args = parse_args()
    if args.shingle_width < 8:
        raise SystemExit("--shingle-width must be at least 8")
    masks: dict[int, int] = defaultdict(int)
    counts: dict[str, int] = {}
    for benchmark_index, (name, dataset, config_names, fields) in enumerate(BENCHMARKS):
        rows_seen = 0
        for config in config_names:
            response = requests.get(
                "https://datasets-server.huggingface.co/parquet",
                params={"dataset": dataset},
                timeout=60,
            )
            response.raise_for_status()
            files = [
                item["url"]
                for item in response.json()["parquet_files"]
                if item["config"] == config and item["split"] in {"train", "validation", "test"}
            ]
            if not files:
                raise RuntimeError(f"no public parquet files found for {dataset}/{config}")
            for url in files:
                rows = load_dataset("parquet", data_files=url, split="train")
                for row in rows:
                    rows_seen += 1
                    for value in shingle_hashes(_text(dict(row), fields), args.shingle_width):
                        masks[value] |= 1 << benchmark_index
        counts[name] = rows_seen
    ordered = sorted(masks.items())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        hashes=np.asarray([item[0] for item in ordered], dtype=np.uint64),
        masks=np.asarray([item[1] for item in ordered], dtype=np.uint8),
        width=np.asarray(args.shingle_width, dtype=np.uint16),
    )
    report = {
        "version": 1,
        "benchmarks": [item[0] for item in BENCHMARKS],
        "public_rows": counts,
        "shingle_width": args.shingle_width,
        "unique_shingles": len(ordered),
        "rejection_rule": "at least two exact normalized word shingles from the same benchmark",
        "includes_answers": False,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
