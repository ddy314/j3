from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.hf_mix import (
    HFMixSource,
    iter_parquet_rows,
    iter_hf_rows,
    load_mix_config,
    row_matches_filters,
    row_text,
    source_slug,
)
from src.data.shards import ShardWriter
from src.data.tokenizer import clean_text_for_tokenization, load_tokenizer, tokenizer_sha256
from src.training.metrics import atomic_json_write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a weighted Hugging Face mix into exact uint16 token budgets"
    )
    parser.add_argument("--config", default="configs/hf_mix_1b.yaml")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="data/tokenized")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply every source training quota; use 0.01 for an 11M-token pilot",
    )
    parser.add_argument("--shard-tokens", type=int, default=4_194_304)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument(
        "--parquet-root",
        default=None,
        help=(
            "read downloaded parquet shards from <root>/<source-name>/ instead of Hub streaming; "
            "useful with aria2c or another resumable downloader"
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scaled_budget(source: HFMixSource, scale: float) -> int:
    return max(2, int(round(source.token_budget * scale)))


def take_exact(ids: list[int], count: int, eos_id: int) -> list[int]:
    """Take exactly ``count`` ids, putting EOS at a truncated document end."""

    if count <= 0:
        return []
    if len(ids) <= count:
        return ids
    if count == 1:
        return [eos_id]
    return [*ids[: count - 1], eos_id]


def _source_seed(seed: int, source_name: str) -> int:
    digest = hashlib.sha256(source_name.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:8], "little")


def _top_frequency(frequency: np.ndarray, limit: int = 100) -> list[dict[str, int]]:
    return [
        {"token_id": int(token_id), "count": int(frequency[token_id])}
        for token_id in np.argsort(frequency)[-limit:][::-1]
        if frequency[token_id] > 0
    ]


def _entries_are_intact(root: Path, result: dict[str, Any]) -> bool:
    for split in ("train_shards", "val_shards"):
        for entry in result.get(split, []):
            path = root / entry["path"]
            expected_bytes = 2 * int(entry["token_count"])
            if not path.is_file() or path.stat().st_size != expected_bytes:
                return False
    return True


def _prepare_source(
    source: HFMixSource,
    *,
    tokenizer: Any,
    output_dir: Path,
    scale: float,
    val_ratio: float,
    seed: int,
    shard_tokens: int,
    deduplicate: bool,
    max_retries: int,
    retry_backoff_seconds: float,
    parquet_root: Path | None = None,
) -> dict[str, Any]:
    train_target = _scaled_budget(source, scale)
    val_target = int(round(train_target * val_ratio)) if val_ratio else 0
    slug = source_slug(source.name)
    source_dir = output_dir / "sources" / slug
    train_writer = ShardWriter(source_dir / "train", slug, shard_tokens, output_dir)
    val_writer = ShardWriter(source_dir / "val", f"{slug}_val", shard_tokens, output_dir)
    rng = random.Random(_source_seed(seed, source.name))
    seen: set[str] = set()
    frequency = np.zeros(tokenizer.vocab_size, dtype=np.int64)
    stats: dict[str, Any] = {
        "raw_rows_seen": 0,
        "filtered_rows": 0,
        "missing_or_nontext_rows": 0,
        "empty_document_count": 0,
        "duplicate_count": 0,
        "accepted_document_count": 0,
        "train_document_count": 0,
        "val_document_count": 0,
        "unk_token_count": 0,
    }

    try:
        if parquet_root is None:
            rows = iter_hf_rows(
                source,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        else:
            source_dir = parquet_root / slug
            parquet_paths = sorted(source_dir.glob("*.parquet"))
            if not parquet_paths:
                raise FileNotFoundError(
                    f"no parquet shards found for {source.name!r} under {source_dir}"
                )
            print(
                f"[hf] {source.name}: reading {len(parquet_paths)} local parquet shard(s)",
                flush=True,
            )
            rows = iter_parquet_rows(parquet_paths)
        for raw_rows_seen, row in rows:
            stats["raw_rows_seen"] = raw_rows_seen
            if train_writer.total_tokens >= train_target and val_writer.total_tokens >= val_target:
                break
            if not row_matches_filters(row, source.filters):
                stats["filtered_rows"] += 1
                continue
            text = row_text(row, source.text_field)
            if text is None:
                stats["missing_or_nontext_rows"] += 1
                continue
            text = clean_text_for_tokenization(text)
            if not text:
                stats["empty_document_count"] += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if deduplicate and digest in seen:
                stats["duplicate_count"] += 1
                continue
            if deduplicate:
                seen.add(digest)

            ids = tokenizer.encode(text, add_bos=False, add_eos=True)
            if not ids:
                stats["empty_document_count"] += 1
                continue
            if max(ids) >= tokenizer.vocab_size:
                raise RuntimeError(f"tokenizer emitted an out-of-range id for {source.name}")
            stats["accepted_document_count"] += 1

            train_remaining = train_target - train_writer.total_tokens
            val_remaining = val_target - val_writer.total_tokens
            choose_validation = bool(
                val_remaining > 0 and train_remaining > 0 and rng.random() < val_ratio
            )
            if train_remaining <= 0:
                if val_remaining <= 0:
                    break
                choose_validation = True
            if choose_validation:
                selected_ids = take_exact(ids, val_remaining, tokenizer.eos_id)
                val_writer.write(selected_ids)
                stats["val_document_count"] += 1
            elif train_remaining > 0:
                selected_ids = take_exact(ids, train_remaining, tokenizer.eos_id)
                train_writer.write(selected_ids)
                stats["train_document_count"] += 1
            else:
                continue
            frequency += np.bincount(selected_ids, minlength=tokenizer.vocab_size)
            stats["unk_token_count"] += selected_ids.count(tokenizer.unk_id)
    finally:
        train_writer.close()
        val_writer.close()

    if train_writer.total_tokens != train_target:
        raise RuntimeError(
            f"source {source.name!r} ended at {train_writer.total_tokens:,} training tokens; "
            f"needed {train_target:,}. The source or filter may be too small. "
            f"Raw rows seen: {stats['raw_rows_seen']:,}."
        )
    if val_target and val_writer.total_tokens < 2:
        print(
            f"[hf] warning: {source.name} has only {val_writer.total_tokens} validation tokens; "
            "the final manifest may still use validation shards from other sources",
            flush=True,
        )

    stats["average_accepted_document_tokens"] = (
        (train_writer.total_tokens + val_writer.total_tokens)
        / max(1, stats["accepted_document_count"])
    )
    stats["unk_rate"] = stats["unk_token_count"] / max(
        1, train_writer.total_tokens + val_writer.total_tokens
    )
    return {
        "status": "completed",
        "source": source.as_dict(),
        "requested_token_budget": source.token_budget,
        "target_train_tokens": train_target,
        "target_val_tokens": val_target,
        "train_token_count": train_writer.total_tokens,
        "val_token_count": val_writer.total_tokens,
        "train_shards": train_writer.entries,
        "val_shards": val_writer.entries,
        "statistics": stats,
        "token_frequency": [int(value) for value in frequency],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _progress_metadata(
    *,
    config_path: Path,
    tokenizer_path: Path,
    scale: float,
    val_ratio: float,
    seed: int,
    shard_tokens: int,
    deduplicate: bool,
    parquet_root: Path | None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_sha256": tokenizer_sha256(tokenizer_path),
        "scale": scale,
        "val_ratio": val_ratio,
        "seed": seed,
        "shard_tokens": shard_tokens,
        "deduplicate": deduplicate,
        "parquet_root": str(parquet_root.resolve()) if parquet_root is not None else None,
    }


def _load_progress(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {**metadata, "sources": {}}
    progress = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in metadata.items():
        if progress.get(key) != expected:
            raise RuntimeError(
                f"{path} belongs to a different extraction ({key} mismatch); "
                "choose a new --output-dir or remove only that old output"
            )
    progress.setdefault("sources", {})
    return progress


def _finalize_manifest(
    *,
    output_dir: Path,
    config_path: Path,
    tokenizer_path: Path,
    raw_config: dict[str, Any],
    sources: list[HFMixSource],
    results: dict[str, dict[str, Any]],
    scale: float,
    val_ratio: float,
    shard_tokens: int,
    deduplicate: bool,
) -> dict[str, Any]:
    train_shards: list[dict[str, Any]] = []
    val_shards: list[dict[str, Any]] = []
    total_frequency = np.zeros(0, dtype=np.int64)
    source_manifest: list[dict[str, Any]] = []
    source_statistics: dict[str, Any] = {}
    for source in sources:
        result = results[source.name]
        train_shards.extend(result["train_shards"])
        val_shards.extend(result["val_shards"])
        frequency = np.asarray(result["token_frequency"], dtype=np.int64)
        if total_frequency.size == 0:
            total_frequency = np.zeros_like(frequency)
        total_frequency += frequency
        source_manifest.append(
            {
                **source.as_dict(),
                "requested_token_budget": source.token_budget,
                "target_train_tokens": result["target_train_tokens"],
                "actual_train_tokens": result["train_token_count"],
                "actual_val_tokens": result["val_token_count"],
            }
        )
        source_statistics[source.name] = result["statistics"]

    train_token_count = sum(int(entry["token_count"]) for entry in train_shards)
    val_token_count = sum(int(entry["token_count"]) for entry in val_shards)
    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {"type": "hf_mix", "config": str(config_path.resolve())},
        "source": {"type": "hf_mix", "config": str(config_path.resolve())},
        "mix_config": raw_config.get("name", config_path.stem),
        "mix_config_sha256": _sha256(config_path),
        "sources": source_manifest,
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_hash": tokenizer_sha256(tokenizer_path),
        "vocab_size": int(total_frequency.size),
        "dtype": "uint16",
        "shard_token_target": shard_tokens,
        "scale": scale,
        "val_ratio": val_ratio,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "train_token_count": train_token_count,
        "val_token_count": val_token_count,
        "total_token_count": train_token_count + val_token_count,
        "statistics": {
            "source_statistics": source_statistics,
            "unk_rate": sum(
                int(result["statistics"]["unk_token_count"])
                for result in results.values()
            )
            / max(1, train_token_count + val_token_count),
            "token_frequency_top_100": _top_frequency(total_frequency),
            "duplicate_filtering": "enabled" if deduplicate else "disabled",
        },
    }
    atomic_json_write(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    if args.scale <= 0:
        raise SystemExit("--scale must be positive")
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be in [0, 1)")
    if args.shard_tokens <= 0:
        raise SystemExit("--shard-tokens must be positive")

    config_path = Path(args.config)
    tokenizer_path = Path(args.tokenizer)
    output_dir = Path(args.output_dir)
    parquet_root = Path(args.parquet_root).resolve() if args.parquet_root else None
    raw_config, sources = load_mix_config(config_path)
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size > 65_536:
        raise SystemExit("uint16 tokenization requires a vocabulary of at most 65,536 pieces")
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = output_dir / "mix_progress.json"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not progress_path.exists():
        raise SystemExit(
            f"{manifest_path} already exists; choose a clean --output-dir "
            "so an older dataset cannot be overwritten accidentally"
        )
    metadata = _progress_metadata(
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        scale=args.scale,
        val_ratio=args.val_ratio,
        seed=args.seed,
        shard_tokens=args.shard_tokens,
        deduplicate=args.deduplicate,
        parquet_root=parquet_root,
    )
    progress = _load_progress(progress_path, metadata)
    results: dict[str, dict[str, Any]] = progress["sources"]

    for source in sources:
        existing = results.get(source.name)
        if (
            isinstance(existing, dict)
            and existing.get("status") == "completed"
            and _entries_are_intact(output_dir, existing)
        ):
            print(f"[hf] resume: {source.name} already complete", flush=True)
            continue
        source_dir = output_dir / "sources" / source_slug(source.name)
        if source_dir.exists():
            shutil.rmtree(source_dir)
        print(
            f"[hf] {source.name}: target {_scaled_budget(source, args.scale):,} train tokens "
            f"from {source.dataset}/{source.config or 'default'}",
            flush=True,
        )
        result = _prepare_source(
            source,
            tokenizer=tokenizer,
            output_dir=output_dir,
            scale=args.scale,
            val_ratio=args.val_ratio,
            seed=args.seed,
            shard_tokens=args.shard_tokens,
            deduplicate=args.deduplicate,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            parquet_root=parquet_root,
        )
        results[source.name] = result
        atomic_json_write(progress_path, progress)
        print(
            f"[hf] {source.name}: wrote {result['train_token_count']:,} train + "
            f"{result['val_token_count']:,} val tokens",
            flush=True,
        )

    if any(source.name not in results for source in sources):
        raise RuntimeError("HF mix is incomplete; no training manifest was written")
    manifest = _finalize_manifest(
        output_dir=output_dir,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        raw_config=raw_config,
        sources=sources,
        results=results,
        scale=args.scale,
        val_ratio=args.val_ratio,
        shard_tokens=args.shard_tokens,
        deduplicate=args.deduplicate,
    )
    print(
        json.dumps(
            {
                "manifest": str((output_dir / "manifest.json").resolve()),
                "train_token_count": manifest["train_token_count"],
                "val_token_count": manifest["val_token_count"],
                "total_token_count": manifest["total_token_count"],
                "source_count": len(sources),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
