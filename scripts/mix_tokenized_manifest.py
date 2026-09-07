from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.prepare_stage3 import _scan_manifest  # noqa: E402
from src.data.hf_mix import source_slug  # noqa: E402
from src.data.shards import ShardWriter  # noqa: E402
from src.data.tokenizer import tokenizer_sha256  # noqa: E402
from src.training.metrics import atomic_json_write  # noqa: E402


@dataclass(frozen=True)
class ChunkRef:
    path: Path
    offset: int
    token_count: int
    source: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_entry(root: Path, entry: dict[str, Any]) -> Path:
    raw = Path(str(entry["path"]))
    path = raw if raw.is_absolute() else root / raw
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _source_entries(
    manifest: dict[str, Any], source_name: str, split: str
) -> list[dict[str, Any]]:
    prefix = f"sources/{source_slug(source_name)}/{split}/"
    entries = [
        entry
        for entry in manifest.get(f"{split}_shards", [])
        if str(entry.get("path", "")).replace("\\", "/").startswith(prefix)
    ]
    if not entries:
        raise ValueError(f"no {split} shards for source {source_name!r}")
    return entries


def _build_chunk_refs(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    split: str,
    chunk_tokens: int,
) -> tuple[list[ChunkRef], dict[str, int]]:
    root = manifest_path.parent
    refs: list[ChunkRef] = []
    source_chunk_counts: dict[str, int] = {}
    listed_paths: set[str] = set()

    for source in manifest.get("sources", []):
        name = str(source["name"])
        expected_source_tokens = int(source[f"actual_{split}_tokens"])
        source_tokens = 0
        source_chunks = 0
        for entry in _source_entries(manifest, name, split):
            path = _resolve_entry(root, entry)
            path_key = str(entry["path"])
            if path_key in listed_paths:
                raise ValueError(f"duplicate shard entry: {path_key}")
            listed_paths.add(path_key)
            token_count = int(entry["token_count"])
            if path.stat().st_size != token_count * 2:
                raise ValueError(
                    f"byte/token mismatch for {path}: {path.stat().st_size} != {token_count * 2}"
                )
            offset = 0
            while offset < token_count:
                take = min(chunk_tokens, token_count - offset)
                refs.append(ChunkRef(path, offset, take, name))
                source_chunks += 1
                source_tokens += take
                offset += take
        if source_tokens != expected_source_tokens:
            raise ValueError(
                f"{name} {split} tokens mismatch: {source_tokens:,} != "
                f"{expected_source_tokens:,}"
            )
        source_chunk_counts[name] = source_chunks

    manifest_paths = {
        str(entry["path"])
        for entry in manifest.get(f"{split}_shards", [])
    }
    if listed_paths != manifest_paths:
        missing = sorted(manifest_paths - listed_paths)
        extra = sorted(listed_paths - manifest_paths)
        raise ValueError(f"{split} shard/source coverage mismatch: missing={missing}, extra={extra}")
    return refs, source_chunk_counts


def _write_shuffled_chunks(
    refs: list[ChunkRef],
    *,
    output_dir: Path,
    split: str,
    shard_tokens: int,
    vocab_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    random.Random(seed).shuffle(refs)
    writer = ShardWriter(output_dir / split, split, shard_tokens, output_dir)
    maps: dict[Path, np.memmap] = {}
    minimum = vocab_size
    maximum = 0
    written = 0
    try:
        for ref in refs:
            mapped = maps.get(ref.path)
            if mapped is None:
                mapped = np.memmap(ref.path, mode="r", dtype="<u2")
                maps[ref.path] = mapped
            end = ref.offset + ref.token_count
            if end > mapped.size:
                raise ValueError(f"chunk exceeds shard: {ref.path}:{ref.offset}+{ref.token_count}")
            values = np.asarray(mapped[ref.offset:end])
            if values.size != ref.token_count:
                raise ValueError(f"chunk count mismatch for {ref.path}: {values.size} != {ref.token_count}")
            if values.size:
                chunk_min = int(values.min())
                chunk_max = int(values.max())
                minimum = min(minimum, chunk_min)
                maximum = max(maximum, chunk_max)
                if chunk_max >= vocab_size:
                    raise ValueError(f"out-of-range token ID in {ref.path}: {chunk_max}")
            writer.write(values)
            written += int(values.size)
    finally:
        writer.close()
        for mapped in maps.values():
            del mapped

    sources = [ref.source for ref in refs]
    source_chunk_counts = Counter(sources)
    first_window = sources[:1024]
    switches = sum(left != right for left, right in zip(first_window, first_window[1:]))
    return writer.entries, {
        "seed": seed,
        "chunk_count": len(refs),
        "written_tokens": written,
        "min_id": minimum,
        "max_id": maximum,
        "source_chunk_counts": dict(sorted(source_chunk_counts.items())),
        "first_32_sources": sources[:32],
        "source_switches_first_1024_chunks": switches,
    }


def _mix(
    input_manifest_path: Path,
    output_dir: Path,
    *,
    chunk_tokens: int,
    seed: int,
    dataset_type: str = "stage3_globally_mixed",
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dtype", "uint16") != "uint16":
        raise ValueError("only uint16 input manifests are supported")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    tokenizer_path = Path(str(manifest["tokenizer"]))
    if not tokenizer_path.is_absolute():
        tokenizer_path = (input_manifest_path.parent / tokenizer_path).resolve()
    if manifest.get("tokenizer_hash") != tokenizer_sha256(tokenizer_path):
        raise ValueError("input manifest tokenizer hash does not match its tokenizer file")
    shard_tokens = int(manifest.get("shard_token_target", 4_194_304))
    if shard_tokens <= 0:
        raise ValueError("input shard_token_target must be positive")

    train_refs, train_source_chunks = _build_chunk_refs(
        input_manifest_path, manifest, split="train", chunk_tokens=chunk_tokens
    )
    val_refs, val_source_chunks = _build_chunk_refs(
        input_manifest_path, manifest, split="val", chunk_tokens=chunk_tokens
    )
    train_entries, train_stats = _write_shuffled_chunks(
        train_refs,
        output_dir=output_dir,
        split="train",
        shard_tokens=shard_tokens,
        vocab_size=int(manifest["vocab_size"]),
        seed=seed,
    )
    val_entries, val_stats = _write_shuffled_chunks(
        val_refs,
        output_dir=output_dir,
        split="val",
        shard_tokens=shard_tokens,
        vocab_size=int(manifest["vocab_size"]),
        seed=seed + 1,
    )
    mixing = {
        "mode": "global_chunk_shuffle",
        "chunk_tokens": chunk_tokens,
        "seed": seed,
        "input_manifest": str(input_manifest_path.resolve()),
        "input_manifest_sha256": _sha256(input_manifest_path),
        "train_source_chunk_counts_before_shuffle": dict(sorted(train_source_chunks.items())),
        "val_source_chunk_counts_before_shuffle": dict(sorted(val_source_chunks.items())),
        "train": train_stats,
        "val": val_stats,
    }
    mixing_hash = hashlib.sha256(
        json.dumps(mixing, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "type": dataset_type,
            "input_manifest": str(input_manifest_path.resolve()),
            "input_manifest_sha256": mixing["input_manifest_sha256"],
        },
        "source": {"type": dataset_type, "input_manifest": str(input_manifest_path.resolve())},
        "mix_config": "global_chunk_shuffle",
        "mix_config_sha256": mixing_hash,
        "mixing": mixing,
        "target_train_tokens": int(manifest["target_train_tokens"]),
        "sources": manifest["sources"],
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_hash": manifest["tokenizer_hash"],
        "vocab_size": int(manifest["vocab_size"]),
        "dtype": "uint16",
        "shard_token_target": shard_tokens,
        "val_ratio": manifest.get("val_ratio"),
        "train_shards": train_entries,
        "val_shards": val_entries,
        "train_token_count": train_stats["written_tokens"],
        "val_token_count": val_stats["written_tokens"],
        "total_token_count": train_stats["written_tokens"] + val_stats["written_tokens"],
    }
    manifest_path = output_dir / "manifest.json"
    atomic_json_write(manifest_path, output_manifest)
    verification = _scan_manifest(manifest_path, tokenizer_path=tokenizer_path)
    verification["mixing"] = {
        "mode": mixing["mode"],
        "chunk_tokens": chunk_tokens,
        "seed": seed,
        "train_source_switches_first_1024_chunks": train_stats[
            "source_switches_first_1024_chunks"
        ],
        "val_source_switches_first_1024_chunks": val_stats[
            "source_switches_first_1024_chunks"
        ],
    }
    atomic_json_write(output_dir / "verification.json", verification)
    if not verification["passed"]:
        raise RuntimeError(f"mixed manifest verification failed: {verification['errors']}")
    return output_manifest


def _verify(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_path = Path(str(manifest["tokenizer"]))
    result = _scan_manifest(manifest_path, tokenizer_path=tokenizer_path)
    result["mixing"] = manifest.get("mixing", {})
    atomic_json_write(output_dir / "verification.json", result)
    if not result["passed"]:
        raise RuntimeError(f"verification failed: {result['errors']}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Globally shuffle token chunks into mixed shards")
    parser.add_argument("--input-manifest", default="data/stage3_tokenized/manifest.json")
    parser.add_argument("--output-dir", default="data/stage3_mixed_tokenized")
    parser.add_argument("--chunk-tokens", type=int, default=8_192)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--dataset-type",
        default="stage3_globally_mixed",
        help="dataset metadata type for the output manifest",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_manifest = Path(args.input_manifest)
    output_dir = Path(args.output_dir)
    if not input_manifest.is_absolute():
        input_manifest = REPOSITORY_ROOT / input_manifest
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    if args.verify_only:
        result = _verify(output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    result = _mix(
        input_manifest,
        output_dir,
        chunk_tokens=args.chunk_tokens,
        seed=args.seed,
        dataset_type=args.dataset_type,
    )
    print(
        json.dumps(
            {
                "manifest": str((output_dir / "manifest.json").resolve()),
                "train_token_count": result["train_token_count"],
                "val_token_count": result["val_token_count"],
                "mixing": result["mixing"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
