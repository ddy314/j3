from __future__ import annotations

import itertools
import gzip
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


@dataclass(frozen=True)
class HFMixSource:
    name: str
    dataset: str
    config: str | None
    split: str
    text_field: str
    token_budget: int
    filters: dict[str, Any]
    local_subdir: str | None = None
    local_pattern: str = "*.parquet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
            "text_field": self.text_field,
            "token_budget": self.token_budget,
            "filters": self.filters,
            "local_subdir": self.local_subdir,
            "local_pattern": self.local_pattern,
        }


def load_mix_config(path: str | Path) -> tuple[dict[str, Any], list[HFMixSource]]:
    """Load and validate a YAML HF source mix description."""

    import yaml

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise ValueError(f"HF mix config must contain a sources list: {config_path}")
    sources: list[HFMixSource] = []
    names: set[str] = set()
    for index, item in enumerate(raw["sources"]):
        if not isinstance(item, dict):
            raise ValueError(f"source {index} is not a mapping")
        name = str(item.get("name", "")).strip()
        dataset = str(item.get("dataset", "")).strip()
        if not name or not dataset:
            raise ValueError(f"source {index} needs name and dataset")
        if name in names:
            raise ValueError(f"duplicate HF mix source name: {name}")
        names.add(name)
        token_budget = int(item.get("token_budget", 0))
        if token_budget <= 0:
            raise ValueError(f"source {name} needs a positive token_budget")
        filters = item.get("filters", {})
        if not isinstance(filters, dict):
            raise ValueError(f"source {name} filters must be a mapping")
        sources.append(
            HFMixSource(
                name=name,
                dataset=dataset,
                config=item.get("config"),
                split=str(item.get("split", "train")),
                text_field=str(item.get("text_field", "text")),
                token_budget=token_budget,
                filters=dict(filters),
                local_subdir=item.get("local_subdir"),
                local_pattern=str(item.get("local_pattern", "*.parquet")),
            )
        )
    return raw, sources


def source_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")
    if not slug:
        raise ValueError(f"source name has no usable path slug: {name!r}")
    return slug.lower()


def row_value(row: Mapping[str, Any], field: str) -> Any:
    """Read a field from a row or the common metadata containers."""

    if field in row:
        return row[field]
    for container_name in ("metadata", "meta", "attributes"):
        container = row.get(container_name)
        if isinstance(container, str):
            try:
                container = json.loads(container)
            except json.JSONDecodeError:
                container = None
        if isinstance(container, Mapping) and field in container:
            return container[field]
    return None


def row_text(row: Mapping[str, Any], text_field: str) -> str | None:
    value = row_value(row, text_field)
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _language_is_english(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return any(_language_is_english(item) for item in value)
    normalized = str(value).strip().lower().replace("_", "-")
    return normalized in {"en", "eng", "english"} or normalized.startswith("en-")


def row_matches_filters(row: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    language = filters.get("language")
    if language is not None:
        value = row_value(row, "language")
        if value is None or not _language_is_english(value):
            return False
        if language not in {"en", "eng", "english"}:
            raise ValueError(f"unsupported language filter: {language!r}")

    if "edu_int_score_min" in filters:
        value = row_value(row, "edu_int_score")
        try:
            if value is None or float(value) < float(filters["edu_int_score_min"]):
                return False
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid edu_int_score value: {value!r}") from exc

    if "language_score_min" in filters:
        value = row_value(row, "language_score")
        try:
            if value is None or float(value) < float(filters["language_score_min"]):
                return False
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid language_score value: {value!r}") from exc

    for suffix, comparator in (("_min", lambda left, right: left >= right), ("_max", lambda left, right: left < right)):
        for key, threshold in filters.items():
            if not key.endswith(suffix) or key in {"edu_int_score_min", "language_score_min"}:
                continue
            field = key[: -len(suffix)]
            value = row_value(row, field)
            try:
                if value is None or not comparator(float(value), float(threshold)):
                    return False
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric filter value for {field}: {value!r}") from exc

    for key, expected in filters.items():
        if not key.endswith("_in"):
            continue
        field = key[:-3]
        if not isinstance(expected, (list, tuple, set)):
            raise ValueError(f"{key} filter must be a sequence")
        if row_value(row, field) not in expected:
            return False
    return True


def iter_parquet_rows(
    paths: Iterable[str | Path], *, batch_size: int = 8_192
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read local parquet shards without going through the Hub streaming client.

    This is the reliable fallback for large public datasets whose Hub/Xet
    streaming URL is reachable for metadata but stalls on the first range
    request.  The caller supplies an ordered list of already downloaded
    shards, so this function never performs network I/O.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("local parquet input requires the pyarrow data dependency") from exc

    raw_rows_seen = 0
    normalized_paths = [Path(path) for path in paths]
    if not normalized_paths:
        raise ValueError("at least one parquet path is required")
    for path in normalized_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, use_threads=True):
            for row in batch.to_pylist():
                if not isinstance(row, Mapping):
                    raise TypeError(f"parquet row is not a mapping: {type(row).__name__}")
                raw_rows_seen += 1
                yield raw_rows_seen, dict(row)


def iter_local_rows(paths: Iterable[str | Path]) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream supported local source files in deterministic path order."""

    normalized_paths = [Path(path) for path in paths]
    if normalized_paths and all(path.suffix == ".parquet" for path in normalized_paths):
        yield from iter_parquet_rows(normalized_paths)
        return
    raw_rows_seen = 0
    for path in normalized_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name.endswith(".json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise TypeError(f"JSON row is not a mapping: {type(row).__name__}")
                    raw_rows_seen += 1
                    yield raw_rows_seen, dict(row)
        elif path.suffix == ".txt":
            with path.open("rt", encoding="utf-8") as handle:
                pending: list[str] = []
                for line in handle:
                    if "<|endoftext|>" not in line:
                        pending.append(line)
                        continue
                    before, _, after = line.partition("<|endoftext|>")
                    pending.append(before)
                    text = "".join(pending).strip()
                    if text:
                        raw_rows_seen += 1
                        yield raw_rows_seen, {"text": text}
                    pending = [after] if after else []
                text = "".join(pending).strip()
                if text:
                    raw_rows_seen += 1
                    yield raw_rows_seen, {"text": text}
        else:
            raise ValueError(f"unsupported local source file: {path}")


def _is_access_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "gated dataset",
            "access request",
            "accept the terms",
            "data agreement",
            "401",
            "403",
            "repository not found",
        )
    )


def iter_hf_rows(
    source: HFMixSource,
    *,
    start_row: int = 0,
    max_retries: int = 5,
    retry_backoff_seconds: float = 5.0,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream rows with bounded reconnects and row-level replay on retry.

    ``start_row`` is a raw-row offset. It is intentionally independent of
    filtering, so a retry resumes at the same dataset position even when
    rows were rejected by a source filter.
    """

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("HF input requires: uv sync --locked --extra data") from exc
    if start_row < 0:
        raise ValueError("start_row must be non-negative")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds must be non-negative")

    raw_rows_seen = int(start_row)
    consecutive_failures = 0
    hub_token = os.environ.get("HF_TOKEN")
    if hub_token is None:
        try:
            from huggingface_hub import get_token

            hub_token = get_token()
        except ImportError:
            hub_token = None
    while True:
        kwargs: dict[str, Any] = {
            "path": source.dataset,
            "split": source.split,
            "streaming": True,
            # Reuse a configured/cached token without exposing it in logs. Public
            # datasets remain usable on machines that have no HF login.
            "token": hub_token or False,
        }
        if source.config:
            kwargs["name"] = source.config
        try:
            rows = load_dataset(**kwargs)
            if raw_rows_seen:
                skip = getattr(rows, "skip", None)
                rows = skip(raw_rows_seen) if callable(skip) else itertools.islice(rows, raw_rows_seen, None)
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError(f"HF row is not a mapping: {type(row).__name__}")
                raw_rows_seen += 1
                consecutive_failures = 0
                yield raw_rows_seen, dict(row)
            return
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            if _is_access_error(exc):
                raise RuntimeError(
                    f"cannot access HF dataset {source.dataset!r} ({source.config or 'default'}); "
                    "check login and any gated-dataset agreement"
                ) from exc
            if consecutive_failures >= max_retries:
                raise RuntimeError(
                    f"HF stream failed for {source.name!r} after {max_retries} retries; "
                    "rerun the source to resume from the last completed source"
                ) from exc
            consecutive_failures += 1
            delay = min(retry_backoff_seconds * (2 ** (consecutive_failures - 1)), 60.0)
            print(
                f"[hf] {source.name}: stream error ({type(exc).__name__}); "
                f"retry {consecutive_failures}/{max_retries} in {delay:g}s",
                flush=True,
            )
            time.sleep(delay)
