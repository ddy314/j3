from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def iter_file_documents(paths: Iterable[str | Path], text_field: str = "text") -> Iterator[str]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            nested = sorted(item for item in path.rglob("*") if item.is_file())
            yield from iter_file_documents(nested, text_field)
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".text", ".md", ".markdown"}:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    text = line.strip()
                    if text:
                        yield text
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    text = value.get(text_field) if isinstance(value, dict) else value
                    if isinstance(text, str) and text.strip():
                        yield text.strip()
                    elif text is not None:
                        raise ValueError(f"{path}:{line_number} field {text_field!r} is not text")
        elif suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value = value.get(text_field, value)
            if isinstance(value, str):
                yield value.strip()
            elif isinstance(value, list):
                for item in value:
                    text = item.get(text_field) if isinstance(item, dict) else item
                    if isinstance(text, str) and text.strip():
                        yield text.strip()
        else:
            raise ValueError(f"unsupported raw file type: {path}")


def iter_hf_documents(
    dataset: str,
    *,
    config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    max_documents: int | None = None,
) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HF input requires the optional data dependency: uv sync --extra data") from exc
    kwargs = {"path": dataset, "split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    rows = load_dataset(**kwargs)
    for index, row in enumerate(rows):
        if max_documents is not None and index >= max_documents:
            break
        text = row.get(text_field)
        if isinstance(text, str) and text.strip():
            yield text.strip()
