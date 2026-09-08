from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.prepare_hf_mix import (  # noqa: E402
    _entries_are_intact,
    _prepare_source,
)
from src.data.decontamination import BenchmarkDecontaminator  # noqa: E402
from src.data.hf_mix import HFMixSource, source_slug  # noqa: E402
from src.data.shards import ShardWriter  # noqa: E402
from src.data.tokenizer import load_tokenizer, tokenizer_sha256  # noqa: E402
from src.training.metrics import atomic_json_write  # noqa: E402


COPY_CHUNK_TOKENS = 1_048_576


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("sources"), list):
        raise ValueError(f"Stage 3 config must contain a sources list: {path}")
    names: set[str] = set()
    for index, item in enumerate(value["sources"]):
        if not isinstance(item, dict):
            raise ValueError(f"Stage 3 source {index} is not a mapping")
        name = str(item.get("name", "")).strip()
        kind = str(item.get("kind", "")).strip()
        if not name or not kind:
            raise ValueError(f"Stage 3 source {index} needs name and kind")
        if name in names:
            raise ValueError(f"duplicate Stage 3 source name: {name}")
        names.add(name)
        if kind not in {"hf", "existing_source", "tokenized_manifest"}:
            raise ValueError(f"unsupported Stage 3 source kind {kind!r} for {name!r}")
    target = int(value.get("target_train_tokens", 0))
    if target <= 0:
        raise ValueError("Stage 3 config needs a positive target_train_tokens")
    return value


def load_stage3_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a Stage 3 assembly config for tests and callers."""

    return _load_yaml(Path(path))


def _manifest_tokenizer_path(manifest_path: Path, manifest: dict[str, Any]) -> Path | None:
    raw = manifest.get("tokenizer")
    if isinstance(raw, str):
        candidate = Path(raw)
        if candidate.is_absolute() and candidate.is_file():
            return candidate
        local = manifest_path.parent / candidate
        if local.is_file():
            return local
    fallback = manifest_path.parent / "tokenizer.json"
    return fallback if fallback.is_file() else None


def _json_object(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_input_tokenizer(
    manifest_path: Path,
    manifest: dict[str, Any],
    canonical_path: Path,
) -> dict[str, Any]:
    canonical_hash = tokenizer_sha256(canonical_path)
    manifest_hash = manifest.get("tokenizer_hash")
    tokenizer_path = _manifest_tokenizer_path(manifest_path, manifest)
    semantic_equal = False
    if tokenizer_path is not None:
        try:
            semantic_equal = _json_object(tokenizer_path) == _json_object(canonical_path)
        except (OSError, json.JSONDecodeError):
            semantic_equal = False
    if manifest_hash != canonical_hash and not semantic_equal:
        raise ValueError(
            f"tokenizer mismatch for {manifest_path}: manifest={manifest_hash!r}, "
            f"canonical={canonical_hash!r}; the token IDs cannot be merged"
        )
    if manifest.get("dtype", "uint16") != "uint16":
        raise ValueError(f"only uint16 input shards are supported: {manifest_path}")
    return {
        "manifest_tokenizer_hash": manifest_hash,
        "canonical_tokenizer_hash": canonical_hash,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "semantic_json_equal": semantic_equal,
        "hash_equal": manifest_hash == canonical_hash,
    }


def _entry_path(
    manifest_path: Path,
    entry: dict[str, Any],
    *,
    split: str,
    prefer_val_directory: bool = False,
) -> Path:
    raw = Path(str(entry["path"]))
    if not raw.is_absolute() and split == "val" and prefer_val_directory:
        val_candidate = manifest_path.parent / "val" / raw.name
        if val_candidate.is_file():
            return val_candidate
    path = raw if raw.is_absolute() else manifest_path.parent / raw
    if not path.is_file():
        raise FileNotFoundError(f"manifest entry does not exist: {path}")
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


def _write_selected_entries(
    *,
    manifest_path: Path,
    entries: Iterable[dict[str, Any]],
    split: str,
    writer: ShardWriter,
    limit: int | None,
    vocab_size: int,
    prefer_val_directory: bool = False,
) -> dict[str, int]:
    remaining = limit
    written = 0
    minimum = vocab_size
    maximum = 0
    for entry in entries:
        if remaining is not None and remaining <= 0:
            break
        path = _entry_path(
            manifest_path,
            entry,
            split=split,
            prefer_val_directory=prefer_val_directory,
        )
        expected = int(entry["token_count"])
        if path.stat().st_size != expected * 2:
            raise ValueError(
                f"byte/token mismatch for {path}: {path.stat().st_size} bytes, "
                f"manifest says {expected} tokens"
            )
        mapped = np.memmap(path, mode="r", dtype="<u2")
        if mapped.size != expected:
            raise ValueError(f"token count mismatch for {path}: {mapped.size} != {expected}")
        offset = 0
        while offset < mapped.size and (remaining is None or remaining > 0):
            take = min(COPY_CHUNK_TOKENS, mapped.size - offset)
            if remaining is not None:
                take = min(take, remaining)
            chunk = np.asarray(mapped[offset : offset + take])
            if chunk.size:
                chunk_min = int(chunk.min())
                chunk_max = int(chunk.max())
                minimum = min(minimum, chunk_min)
                maximum = max(maximum, chunk_max)
                if chunk_max >= vocab_size:
                    raise ValueError(f"out-of-range token ID in {path}: {chunk_max}")
                writer.write(chunk)
                written += int(chunk.size)
                if remaining is not None:
                    remaining -= int(chunk.size)
            offset += take
        del mapped
    if limit is not None and written != limit:
        raise ValueError(
            f"source {manifest_path} {split} ended at {written:,} tokens; "
            f"needed {limit:,}"
        )
    return {"token_count": written, "min_id": minimum, "max_id": maximum}


def _new_writers(output_dir: Path, name: str, shard_tokens: int) -> tuple[ShardWriter, ShardWriter]:
    slug = source_slug(name)
    source_dir = output_dir / "sources" / slug
    return (
        ShardWriter(source_dir / "train", slug, shard_tokens, output_dir),
        ShardWriter(source_dir / "val", f"{slug}_val", shard_tokens, output_dir),
    )


def _copy_tokenized_manifest_source(
    item: dict[str, Any],
    *,
    output_dir: Path,
    canonical_tokenizer: Path,
    shard_tokens: int,
) -> dict[str, Any]:
    name = str(item["name"])
    manifest_path = _repo_path(str(item["manifest"]))
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_evidence = _validate_input_tokenizer(
        manifest_path, input_manifest, canonical_tokenizer
    )
    train_writer, val_writer = _new_writers(output_dir, name, shard_tokens)
    requested_train = int(item.get("target_train_tokens", input_manifest.get("train_token_count", 0)))
    if requested_train <= 0:
        raise ValueError(f"{name!r} needs a positive target_train_tokens")
    available_train = int(input_manifest.get("train_token_count", 0))
    if requested_train > available_train:
        raise ValueError(
            f"{name!r} requests {requested_train:,} train tokens but input has only "
            f"{available_train:,}"
        )
    try:
        train_stats = _write_selected_entries(
            manifest_path=manifest_path,
            entries=input_manifest.get("train_shards", []),
            split="train",
            writer=train_writer,
            limit=requested_train,
            vocab_size=int(input_manifest["vocab_size"]),
        )
        val_stats = _write_selected_entries(
            manifest_path=manifest_path,
            entries=input_manifest.get("val_shards", []),
            split="val",
            writer=val_writer,
            limit=None,
            vocab_size=int(input_manifest["vocab_size"]),
            prefer_val_directory=True,
        )
    finally:
        train_writer.close()
        val_writer.close()
    return {
        "status": "completed",
        "name": name,
        "kind": "tokenized_manifest",
        "category": str(item["category"]),
        "declared_train_tokens": requested_train,
        "actual_train_tokens": train_stats["token_count"],
        "actual_val_tokens": val_stats["token_count"],
        "train_shards": train_writer.entries,
        "val_shards": val_writer.entries,
        "provenance": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "dataset": input_manifest.get("dataset"),
            "source": input_manifest.get("source"),
            "tokenizer": tokenizer_evidence,
            "val_path_fallback_used": True,
        },
        "statistics": {
            "input_total_token_count": input_manifest.get("total_token_count"),
            "input_target_reached": input_manifest.get("target_reached"),
            "input_final_tokenization_performed": input_manifest.get(
                "final_tokenization_performed"
            ),
            "min_id": min(train_stats["min_id"], val_stats["min_id"]),
            "max_id": max(train_stats["max_id"], val_stats["max_id"]),
        },
    }


def _copy_existing_source(
    item: dict[str, Any],
    *,
    output_dir: Path,
    canonical_tokenizer: Path,
    shard_tokens: int,
    val_ratio: float,
) -> dict[str, Any]:
    name = str(item["name"])
    manifest_path = _repo_path(str(item["manifest"]))
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_evidence = _validate_input_tokenizer(
        manifest_path, input_manifest, canonical_tokenizer
    )
    source_name = str(item["source_name"])
    source_record = next(
        (source for source in input_manifest.get("sources", []) if source.get("name") == source_name),
        None,
    )
    if source_record is None:
        raise ValueError(f"source {source_name!r} is absent from {manifest_path}")
    train_target = int(item["token_budget"])
    val_target = int(round(train_target * val_ratio)) if val_ratio else 0
    train_writer, val_writer = _new_writers(output_dir, name, shard_tokens)
    try:
        train_stats = _write_selected_entries(
            manifest_path=manifest_path,
            entries=_source_entries(input_manifest, source_name, "train"),
            split="train",
            writer=train_writer,
            limit=train_target,
            vocab_size=int(input_manifest["vocab_size"]),
        )
        val_stats = _write_selected_entries(
            manifest_path=manifest_path,
            entries=_source_entries(input_manifest, source_name, "val"),
            split="val",
            writer=val_writer,
            limit=val_target,
            vocab_size=int(input_manifest["vocab_size"]),
        )
    finally:
        train_writer.close()
        val_writer.close()
    return {
        "status": "completed",
        "name": name,
        "kind": "existing_source",
        "category": str(item["category"]),
        "declared_train_tokens": train_target,
        "actual_train_tokens": train_stats["token_count"],
        "actual_val_tokens": val_stats["token_count"],
        "train_shards": train_writer.entries,
        "val_shards": val_writer.entries,
        "provenance": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "source_name": source_name,
            "source_record": source_record,
            "tokenizer": tokenizer_evidence,
        },
        "statistics": {
            "input_source_train_tokens": source_record.get("actual_train_tokens"),
            "input_source_val_tokens": source_record.get("actual_val_tokens"),
            "min_id": min(train_stats["min_id"], val_stats["min_id"]),
            "max_id": max(train_stats["max_id"], val_stats["max_id"]),
        },
    }


def _hf_source(item: dict[str, Any]) -> HFMixSource:
    filters = item.get("filters", {})
    if not isinstance(filters, dict):
        raise ValueError(f"filters for {item['name']!r} must be a mapping")
    text_template = item.get("text_template")
    if text_template is not None and not isinstance(text_template, str):
        raise ValueError(f"text_template for {item['name']!r} must be a string")
    min_text_chars = int(item.get("min_text_chars", 0))
    if min_text_chars < 0:
        raise ValueError(f"min_text_chars for {item['name']!r} must be non-negative")
    return HFMixSource(
        name=str(item["name"]),
        dataset=str(item["dataset"]),
        config=item.get("config"),
        split=str(item.get("split", "train")),
        text_field=str(item.get("text_field", "text")),
        token_budget=int(item["token_budget"]),
        filters=dict(filters),
        local_subdir=item.get("local_subdir"),
        local_pattern=str(item.get("local_pattern", "*.parquet")),
        text_template=text_template,
        min_text_chars=min_text_chars,
    )


def _prepare_hf_source(
    item: dict[str, Any],
    *,
    output_dir: Path,
    tokenizer: Any,
    val_ratio: float,
    seed: int,
    shard_tokens: int,
    deduplicate: bool,
    decontaminator: BenchmarkDecontaminator | None,
    parquet_root: Path | None,
    max_retries: int,
    retry_backoff_seconds: float,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    source = _hf_source(item)
    result = _prepare_source(
        source,
        tokenizer=tokenizer,
        output_dir=output_dir,
        scale=1.0,
        val_ratio=val_ratio,
        seed=seed,
        shard_tokens=shard_tokens,
        deduplicate=deduplicate,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        parquet_root=parquet_root,
        workers=workers,
        batch_size=batch_size,
        decontaminator=decontaminator,
    )
    return {
        **result,
        "name": source.name,
        "kind": "hf",
        "category": str(item["category"]),
        "declared_train_tokens": source.token_budget,
        "actual_train_tokens": result["train_token_count"],
        "actual_val_tokens": result["val_token_count"],
        "provenance": source.as_dict(),
    }


def _result_entries_intact(output_dir: Path, result: dict[str, Any]) -> bool:
    return _entries_are_intact(output_dir, result)


def _top_frequency(frequency: np.ndarray, limit: int = 100) -> list[dict[str, int]]:
    return [
        {"token_id": int(token_id), "count": int(frequency[token_id])}
        for token_id in np.argsort(frequency)[-limit:][::-1]
        if frequency[token_id] > 0
    ]


def _scan_manifest(manifest_path: Path, *, tokenizer_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    vocab_size = int(manifest["vocab_size"])
    frequency = np.zeros(vocab_size, dtype=np.int64)
    actual: dict[str, int] = {}
    listed: list[str] = []
    minimum = vocab_size
    maximum = 0
    special_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    errors: list[str] = []
    for split in ("train", "val"):
        total = 0
        for entry in manifest.get(f"{split}_shards", []):
            raw = str(entry["path"])
            listed.append(raw)
            path = Path(raw)
            path = path if path.is_absolute() else root / path
            expected = int(entry["token_count"])
            if not path.is_file():
                errors.append(f"missing:{path}")
                continue
            if path.stat().st_size != expected * 2:
                errors.append(f"bytes:{path}:{path.stat().st_size}!={expected * 2}")
                continue
            mapped = np.memmap(path, mode="r", dtype="<u2")
            if mapped.size != expected:
                errors.append(f"count:{path}:{mapped.size}!={expected}")
                del mapped
                continue
            for start in range(0, mapped.size, COPY_CHUNK_TOKENS):
                chunk = np.asarray(mapped[start : start + COPY_CHUNK_TOKENS])
                if chunk.size:
                    chunk_min = int(chunk.min())
                    chunk_max = int(chunk.max())
                    minimum = min(minimum, chunk_min)
                    maximum = max(maximum, chunk_max)
                    if chunk_max >= vocab_size:
                        errors.append(f"id_range:{path}:{chunk_max}")
                    frequency += np.bincount(chunk, minlength=vocab_size)
                    for token_id in special_counts:
                        special_counts[token_id] += int(np.count_nonzero(chunk == token_id))
            del mapped
            total += expected
        actual[split] = total
    all_bin = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.bin")
        if path.is_file()
    }
    listed_set = set(listed)
    unlisted = sorted(all_bin - listed_set)
    missing_listed = sorted(listed_set - all_bin)
    if unlisted:
        errors.append(f"unlisted_bin_files:{len(unlisted)}")
    if missing_listed:
        errors.append(f"missing_listed_bin_files:{len(missing_listed)}")
    canonical_hash = tokenizer_sha256(tokenizer_path)
    if manifest.get("tokenizer_hash") != canonical_hash:
        errors.append("tokenizer_hash_mismatch")
    expected_train = int(manifest["train_token_count"])
    expected_val = int(manifest["val_token_count"])
    if actual.get("train") != expected_train:
        errors.append(f"train_tokens:{actual.get('train')}!={expected_train}")
    if actual.get("val") != expected_val:
        errors.append(f"val_tokens:{actual.get('val')}!={expected_val}")
    return {
        "manifest": str(manifest_path.resolve()),
        "passed": not errors,
        "errors": errors,
        "actual_train_tokens": actual.get("train", 0),
        "actual_val_tokens": actual.get("val", 0),
        "actual_total_tokens": sum(actual.values()),
        "manifest_total_tokens": int(manifest["total_token_count"]),
        "train_shard_count": len(manifest.get("train_shards", [])),
        "val_shard_count": len(manifest.get("val_shards", [])),
        "id_range": [minimum, maximum],
        "special_token_counts": special_counts,
        "unlisted_bin_files": unlisted,
        "missing_listed_bin_files": missing_listed,
        "tokenizer_hash": canonical_hash,
        "top_frequency": _top_frequency(frequency),
    }


def _build_manifest(
    *,
    config_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    tokenizer_path: Path,
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    train_shards: list[dict[str, Any]] = []
    val_shards: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    source_statistics: dict[str, Any] = {}
    for item in config["sources"]:
        name = str(item["name"])
        result = results[name]
        train_shards.extend(result["train_shards"])
        val_shards.extend(result["val_shards"])
        source_records.append(
            {
                "name": name,
                "kind": result["kind"],
                "category": result["category"],
                "declared_train_tokens": result["declared_train_tokens"],
                "actual_train_tokens": result["actual_train_tokens"],
                "actual_val_tokens": result["actual_val_tokens"],
                "provenance": result["provenance"],
            }
        )
        source_statistics[name] = result.get("statistics", {})
    train_count = sum(int(entry["token_count"]) for entry in train_shards)
    val_count = sum(int(entry["token_count"]) for entry in val_shards)
    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {"type": "stage3_assembled", "config": str(config_path.resolve())},
        "source": {"type": "stage3_assembled", "config": str(config_path.resolve())},
        "mix_config": config.get("name", config_path.stem),
        "mix_config_sha256": _sha256(config_path),
        "target_train_tokens": int(config["target_train_tokens"]),
        "sources": source_records,
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_hash": tokenizer_sha256(tokenizer_path),
        "vocab_size": int(load_tokenizer(tokenizer_path).vocab_size),
        "dtype": "uint16",
        "shard_token_target": int(config["shard_tokens"]),
        "val_ratio": float(config["val_ratio"]),
        "train_shards": train_shards,
        "val_shards": val_shards,
        "train_token_count": train_count,
        "val_token_count": val_count,
        "total_token_count": train_count + val_count,
        "statistics": {"source_statistics": source_statistics},
    }
    atomic_json_write(output_dir / "manifest.json", manifest)
    return manifest


def _metadata(
    config_path: Path,
    config: dict[str, Any],
    tokenizer_path: Path,
    parquet_root: Path | None,
) -> dict[str, Any]:
    decontamination = config.get("decontamination_index")
    decontamination_path = _repo_path(decontamination) if decontamination else None
    return {
        "version": 1,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_sha256": tokenizer_sha256(tokenizer_path),
        "shard_tokens": int(config["shard_tokens"]),
        "val_ratio": float(config["val_ratio"]),
        "seed": int(config["seed"]),
        "deduplicate": bool(config["deduplicate"]),
        "decontamination_index": (
            str(decontamination_path.resolve()) if decontamination_path else None
        ),
        "decontamination_sha256": (
            _sha256(decontamination_path) if decontamination_path else None
        ),
        "parquet_root": str(parquet_root.resolve()) if parquet_root else None,
    }


def _load_progress(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {**metadata, "sources": {}}
    progress = json.loads(path.read_text(encoding="utf-8"))
    for key, expected in metadata.items():
        if progress.get(key) != expected:
            raise RuntimeError(f"{path} belongs to a different Stage 3 extraction: {key} mismatch")
    progress.setdefault("sources", {})
    return progress


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble Stage 3 from FinePDF, HF sources, and verified local token shards"
    )
    parser.add_argument("--config", default="configs/stage3_1b.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--parquet-root", default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config_path = _repo_path(args.config)
    config = _load_yaml(config_path)
    tokenizer_path = _repo_path(str(config["tokenizer"]))
    output_dir = _repo_path(args.output_dir or str(config["output_dir"]))
    parquet_root = _repo_path(args.parquet_root) if args.parquet_root else None
    manifest_path = output_dir / "manifest.json"

    if args.verify_only:
        if not manifest_path.is_file():
            raise SystemExit(f"Stage 3 manifest does not exist: {manifest_path}")
        summary = _scan_manifest(manifest_path, tokenizer_path=tokenizer_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary["passed"]:
            raise SystemExit(1)
        return

    if not tokenizer_path.is_file():
        raise SystemExit(f"tokenizer does not exist: {tokenizer_path}")
    if args.workers < 0 or args.batch_size <= 0:
        raise SystemExit("workers must be non-negative and batch-size must be positive")
    if not 0 <= float(config["val_ratio"]) < 1:
        raise SystemExit("val_ratio must be in [0, 1)")
    if manifest_path.exists() and not (output_dir / "stage3_progress.json").exists():
        raise SystemExit(
            f"{manifest_path} exists without a matching progress file; choose a clean output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or min(4, os.cpu_count() or 1)
    raw_decontamination = config.get("decontamination_index")
    decontamination_path = _repo_path(raw_decontamination) if raw_decontamination else None
    decontaminator = (
        BenchmarkDecontaminator(decontamination_path)
        if decontamination_path is not None
        else None
    )
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size > 65_536:
        raise SystemExit("uint16 tokenization requires a vocabulary of at most 65,536 pieces")

    progress_path = output_dir / "stage3_progress.json"
    progress = _load_progress(
        progress_path,
        _metadata(config_path, config, tokenizer_path, parquet_root),
    )
    results: dict[str, dict[str, Any]] = progress["sources"]
    for item in config["sources"]:
        name = str(item["name"])
        existing = results.get(name)
        if (
            isinstance(existing, dict)
            and existing.get("status") == "completed"
            and _result_entries_intact(output_dir, existing)
        ):
            print(f"[stage3] resume: {name} already complete", flush=True)
            continue
        source_dir = output_dir / "sources" / source_slug(name)
        if source_dir.exists():
            shutil.rmtree(source_dir)
        kind = str(item["kind"])
        print(f"[stage3] {name}: preparing {kind}", flush=True)
        if kind == "hf":
            result = _prepare_hf_source(
                item,
                output_dir=output_dir,
                tokenizer=tokenizer,
                val_ratio=float(config["val_ratio"]),
                seed=int(config["seed"]),
                shard_tokens=int(config["shard_tokens"]),
                deduplicate=bool(config["deduplicate"]),
                decontaminator=decontaminator,
                parquet_root=parquet_root,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                workers=workers,
                batch_size=args.batch_size,
            )
        elif kind == "existing_source":
            result = _copy_existing_source(
                item,
                output_dir=output_dir,
                canonical_tokenizer=tokenizer_path,
                shard_tokens=int(config["shard_tokens"]),
                val_ratio=float(config["val_ratio"]),
            )
        else:
            result = _copy_tokenized_manifest_source(
                item,
                output_dir=output_dir,
                canonical_tokenizer=tokenizer_path,
                shard_tokens=int(config["shard_tokens"]),
            )
        results[name] = result
        atomic_json_write(progress_path, progress)
        print(
            f"[stage3] {name}: wrote {result['actual_train_tokens']:,} train + "
            f"{result['actual_val_tokens']:,} val tokens",
            flush=True,
        )

    if any(str(item["name"]) not in results for item in config["sources"]):
        raise RuntimeError("Stage 3 source preparation is incomplete")
    manifest = _build_manifest(
        config_path=config_path,
        config=config,
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        results=results,
    )
    if manifest["train_token_count"] != int(config["target_train_tokens"]):
        raise RuntimeError(
            f"Stage 3 train count is {manifest['train_token_count']:,}; "
            f"expected {int(config['target_train_tokens']):,}"
        )
    summary = _scan_manifest(output_dir / "manifest.json", tokenizer_path=tokenizer_path)
    atomic_json_write(output_dir / "verification.json", summary)
    if not summary["passed"]:
        raise RuntimeError(f"Stage 3 verification failed: {summary['errors']}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
