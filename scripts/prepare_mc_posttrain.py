from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.decontamination import BenchmarkDecontaminator  # noqa: E402
from src.data.hf_mix import iter_local_rows, row_value  # noqa: E402
from src.data.multiple_choice import example_from_record, manifest_sha256  # noqa: E402
from src.data.tokenizer import clean_text_for_tokenization, load_tokenizer, tokenizer_sha256  # noqa: E402
from src.training.metrics import atomic_json_write  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TARGET_BENCHMARKS = ("hellaswag", "piqa", "arc", "winogrande")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_mc_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise ValueError(f"multiple-choice config must contain a sources list: {config_path}")
    max_seq_len = int(raw.get("max_seq_len", 0))
    if max_seq_len <= 1:
        raise ValueError("multiple-choice config max_seq_len must be at least 2")
    val_ratio = float(raw.get("val_ratio", 0.02))
    if not 0 <= val_ratio < 1:
        raise ValueError("multiple-choice config val_ratio must be in [0, 1)")
    names: set[str] = set()
    for index, source in enumerate(raw["sources"]):
        if not isinstance(source, dict):
            raise ValueError(f"multiple-choice source {index} is not a mapping")
        name = str(source.get("name", "")).strip()
        category = str(source.get("category", "")).strip()
        path_value = source.get("path")
        if not name or name in names:
            raise ValueError(f"invalid or duplicate multiple-choice source name: {name!r}")
        if not category:
            raise ValueError(f"multiple-choice source {name!r} needs category")
        if not isinstance(path_value, (str, list, tuple)):
            raise ValueError(f"multiple-choice source {name!r} needs path or path list")
        names.add(name)
    return raw


def _expand_source_paths(value: str | list[str] | tuple[str, ...]) -> list[Path]:
    values = [value] if isinstance(value, str) else list(value)
    paths: list[Path] = []
    for item in values:
        pattern = str(_repo_path(item))
        matches = [Path(path) for path in sorted(glob.glob(pattern, recursive=True))]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.extend(path for path in matches if path.is_file())
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError(f"no multiple-choice input files matched {values}")
    return unique


def _string_field(row: Mapping[str, Any], field: str, *, location: str) -> str:
    value = row_value(row, field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} field {field!r} must be non-empty text")
    return value.strip()


def _choices_value(value: Any, *, location: str) -> list[str]:
    if isinstance(value, Mapping):
        for key in ("text", "choices", "values", "options"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{location} choices must be a list with at least two items")
    choices = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(choices) != len(value):
        raise ValueError(f"{location} choices must contain only non-empty text")
    if len(set(choices)) != len(choices):
        raise ValueError(f"{location} choices must be distinct")
    return choices


def _answer_index(value: Any, choice_count: int, *, location: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{location} answer/label must not be boolean")
    if isinstance(value, int):
        answer = value
    elif isinstance(value, str):
        normalized = value.strip()
        if normalized.isdigit():
            answer = int(normalized)
        elif len(normalized) == 1 and normalized.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer = ord(normalized.upper()) - ord("A")
        else:
            raise ValueError(f"{location} answer/label must be a zero-based integer or A-Z")
    else:
        raise ValueError(f"{location} answer/label must be a zero-based integer or A-Z")
    if not 0 <= answer < choice_count:
        raise ValueError(f"{location} answer {answer} is outside 0..{choice_count - 1}")
    return answer


def _normalized_split(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"val", "valid", "validation", "dev", "development"}:
        return "val"
    if normalized in {"train", "training"}:
        return "train"
    return None


def _is_validation(identity: str, *, seed: int, val_ratio: float) -> bool:
    if val_ratio <= 0:
        return False
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    threshold = int(val_ratio * 1_000_000)
    return int.from_bytes(digest[:8], "little") % 1_000_000 < threshold


def _effective_record(
    row: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    source_name: str,
    row_number: int,
    tokenizer: Any,
    max_seq_len: int,
    global_choice_prefix: str,
) -> tuple[dict[str, Any], str | None]:
    location = f"{source_name} row {row_number}"
    prompt_field = str(source.get("prompt_field", "prompt"))
    choices_field = str(source.get("choices_field", "choices"))
    answer_field = str(source.get("answer_field", "answer"))
    id_field = str(source.get("id_field", "id"))
    split_field = str(source.get("split_field", "split"))
    prompt = clean_text_for_tokenization(_string_field(row, prompt_field, location=location))
    raw_choices = row_value(row, choices_field)
    choices = [clean_text_for_tokenization(item) for item in _choices_value(raw_choices, location=location)]
    answer = _answer_index(row_value(row, answer_field), len(choices), location=location)
    source_prefix = str(source.get("choice_prefix", global_choice_prefix))
    prompt_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    choice_ids = [
        tokenizer.encode(f"{source_prefix}{choice}", add_bos=False, add_eos=False)
        for choice in choices
    ]
    if not prompt_ids:
        raise ValueError(f"{location} prompt tokenized to an empty sequence")
    if any(not ids for ids in choice_ids):
        raise ValueError(f"{location} choice tokenized to an empty sequence")
    if any(len(ids) >= max_seq_len for ids in choice_ids):
        return {}, "choice_too_long"

    raw_id = row_value(row, id_field)
    record_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else str(row_number)
    identity = f"{source_name}:{record_id}:{prompt}\n{chr(0).join(choices)}"
    record = {
        "id": f"{source_name}:{record_id}",
        "category": str(source["category"]),
        "source": source_name,
        "prompt_ids": [int(value) for value in prompt_ids],
        "choice_ids": [[int(value) for value in ids] for ids in choice_ids],
        "answer": answer,
    }
    split = _normalized_split(row_value(row, split_field))
    return record, split or identity


def _verify_output(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("type") != "multiple_choice_tokenized_v1":
        raise ValueError(f"unsupported multiple-choice manifest type: {manifest_path}")
    tokenizer_path = Path(str(manifest["tokenizer"]))
    if tokenizer_sha256(tokenizer_path) != manifest.get("tokenizer_hash"):
        raise ValueError("multiple-choice tokenizer hash mismatch")
    result: dict[str, Any] = {"passed": True, "errors": [], "splits": {}}
    for split in ("train", "val"):
        records_path = output_dir / str(manifest[f"{split}_file"])
        expected_hash = manifest.get(f"{split}_file_sha256")
        if expected_hash is not None and manifest_sha256(records_path) != expected_hash:
            raise ValueError(f"multiple-choice records hash mismatch: {records_path}")
        count = 0
        for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            example_from_record(
                record,
                location=f"{records_path}:{line_number}",
                vocab_size=int(manifest["vocab_size"]),
            )
            count += 1
        expected = int(manifest[f"{split}_count"])
        result["splits"][split] = {"count": count, "expected": expected}
        if count != expected:
            result["passed"] = False
            result["errors"].append(f"{split} count {count} != {expected}")
    atomic_json_write(output_dir / "verification.json", result)
    if not result["passed"]:
        raise RuntimeError(f"multiple-choice verification failed: {result['errors']}")
    return result


def prepare(
    config_path: Path,
    *,
    tokenizer_path: Path,
    output_dir: Path,
    decontamination_index: Path | None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    config = load_mc_config(config_path)
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size > 65_536:
        raise ValueError("multiple-choice artifacts require a vocabulary that fits uint16")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a fresh directory so data is not overwritten"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    val_ratio = float(config.get("val_ratio", 0.02))
    seed = int(config.get("seed", 1337))
    max_seq_len = int(config["max_seq_len"])
    global_choice_prefix = str(config.get("choice_prefix", ""))
    decontaminator = BenchmarkDecontaminator(decontamination_index) if decontamination_index else None
    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    seen: set[str] = set()
    source_stats: dict[str, dict[str, Any]] = {}
    source_files: list[dict[str, str]] = []
    total_seen = 0

    for source in config["sources"]:
        source_name = str(source["name"])
        paths = _expand_source_paths(source["path"])
        stats: dict[str, Any] = {
            "raw_rows_seen": 0,
            "accepted": 0,
            "train": 0,
            "val": 0,
            "duplicate": 0,
            "decontaminated": 0,
            "invalid": 0,
            "choice_too_long": 0,
        }
        for path in paths:
            source_files.append({"source": source_name, "path": str(path), "sha256": _sha256(path)})
            for row_number, row in iter_local_rows([path]):
                stats["raw_rows_seen"] = int(stats["raw_rows_seen"]) + 1
                total_seen += 1
                try:
                    record, split_hint = _effective_record(
                        row,
                        source=source,
                        source_name=source_name,
                        row_number=row_number,
                        tokenizer=tokenizer,
                        max_seq_len=max_seq_len,
                        global_choice_prefix=global_choice_prefix,
                    )
                except (TypeError, ValueError, KeyError):
                    stats["invalid"] = int(stats["invalid"]) + 1
                    continue
                if not record:
                    stats["choice_too_long"] = int(stats["choice_too_long"]) + 1
                    continue
                identity = json.dumps(
                    {
                        "prompt_ids": record["prompt_ids"],
                        "choice_ids": record["choice_ids"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                if digest in seen:
                    stats["duplicate"] = int(stats["duplicate"]) + 1
                    continue
                combined_text = " ".join(
                    [
                        _string_field(row, str(source.get("prompt_field", "prompt")), location=source_name),
                        *[
                            item
                            for item in _choices_value(
                                row_value(row, str(source.get("choices_field", "choices"))),
                                location=source_name,
                            )
                        ],
                    ]
                )
                if decontaminator and decontaminator.match_mask(clean_text_for_tokenization(combined_text)):
                    stats["decontaminated"] = int(stats["decontaminated"]) + 1
                    continue
                seen.add(digest)
                if split_hint == "val":
                    split = "val"
                elif split_hint == "train":
                    split = "train"
                else:
                    split = "val" if _is_validation(str(split_hint), seed=seed, val_ratio=val_ratio) else "train"
                records[split].append(record)
                stats[split] = int(stats[split]) + 1
                stats["accepted"] = int(stats["accepted"]) + 1
                if max_examples is not None and sum(len(value) for value in records.values()) >= max_examples:
                    break
            if max_examples is not None and sum(len(value) for value in records.values()) >= max_examples:
                break
        source_stats[source_name] = stats
        if max_examples is not None and sum(len(value) for value in records.values()) >= max_examples:
            break

    if not records["train"]:
        raise RuntimeError("multiple-choice preparation produced no training examples")
    for split in records:
        records[split].sort(key=lambda item: item["id"])
    for split in ("train", "val"):
        (output_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records[split]),
            encoding="utf-8",
        )

    choice_counts = Counter(len(record["choice_ids"]) for values in records.values() for record in values)
    category_counts = Counter(record["category"] for values in records.values() for record in values)
    manifest = {
        "version": 1,
        "type": "multiple_choice_tokenized_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "name": str(config.get("name", config_path.stem)),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_hash": tokenizer_sha256(tokenizer_path),
        "vocab_size": int(tokenizer.vocab_size),
        "pad_id": int(tokenizer.pad_id),
        "max_seq_len": max_seq_len,
        "tokenization": {
            "mode": "separate_prompt_choice_v1",
            "choice_prefix": global_choice_prefix,
            "add_bos": False,
            "add_eos": False,
            "score": "mean log P(choice token | prompt and preceding choice tokens)",
        },
        "train_file": "train.jsonl",
        "val_file": "val.jsonl",
        "train_file_sha256": _sha256(output_dir / "train.jsonl"),
        "val_file_sha256": _sha256(output_dir / "val.jsonl"),
        "train_count": len(records["train"]),
        "val_count": len(records["val"]),
        "total_count": sum(len(value) for value in records.values()),
        "choice_count_histogram": {str(key): value for key, value in sorted(choice_counts.items())},
        "category_counts": dict(sorted(category_counts.items())),
        "sources": source_files,
        "target_benchmark_exclusion": list(TARGET_BENCHMARKS),
        "decontamination_index": str(decontamination_index.resolve()) if decontamination_index else None,
        "decontamination_index_sha256": _sha256(decontamination_index) if decontamination_index else None,
        "statistics": {
            "raw_rows_seen": total_seen,
            "source_statistics": source_stats,
            "split_rule": "respect explicit train/validation/dev split; otherwise deterministic hash split",
            "val_ratio": val_ratio,
            "deduplicate": True,
        },
    }
    atomic_json_write(output_dir / "manifest.json", manifest)
    _verify_output(output_dir)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pre-tokenized multiple-choice ranking data")
    parser.add_argument("--config", default="configs/mc_posttrain.yaml")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--decontamination-index", default=None)
    parser.add_argument("--max-examples", type=int, default=None, help="bounded fixture/pilot limit")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = _repo_path(args.config)
    config = load_mc_config(config_path)
    output_dir = _repo_path(args.output_dir or config.get("output_dir", "data/mc_posttrain_tokenized"))
    if args.verify_only:
        print(json.dumps(_verify_output(output_dir), ensure_ascii=False, indent=2, sort_keys=True))
        return
    tokenizer_path = _repo_path(args.tokenizer or config.get("tokenizer", "data/tokenized/tokenizer.json"))
    index_value = args.decontamination_index or config.get("decontamination_index")
    index_path = _repo_path(index_value) if index_value else None
    if args.max_examples is not None and args.max_examples <= 0:
        raise SystemExit("--max-examples must be positive when set")
    manifest = prepare(
        config_path,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        decontamination_index=index_path,
        max_examples=args.max_examples,
    )
    print(
        json.dumps(
            {
                "manifest": str((output_dir / "manifest.json").resolve()),
                "train_count": manifest["train_count"],
                "val_count": manifest["val_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
