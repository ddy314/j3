"""Export a compact index over the raw training records kept for submission."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _last_metric(path: Path) -> dict[str, Any]:
    last: dict[str, Any] = {}
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    last = value
    except (OSError, json.JSONDecodeError):
        pass
    return last


def export(source_root: Path, output: Path, selected_run: str) -> None:
    records: list[dict[str, Any]] = []
    for run_dir in sorted(
        path for path in source_root.iterdir() if path.is_dir() and not path.is_symlink()
    ):
        status = _read_json(run_dir / "status.json")
        command = _read_json(run_dir / "command.json")
        environment = _read_json(run_dir / "environment.json")
        checkpoint_metadata: list[str] = []
        for metadata in sorted(run_dir.glob("checkpoints/*/metadata.json")):
            checkpoint_metadata.append(str(metadata.relative_to(source_root)))
        records.append(
            {
                "run_id": run_dir.name,
                "status": status,
                "command": command,
                "environment": {
                    "hostname": environment.get("hostname"),
                    "python": environment.get("python"),
                    "torch_version": environment.get("torch_version"),
                    "torch_cuda_version": environment.get("torch_cuda_version"),
                    "cuda_devices": environment.get("cuda_devices"),
                    "git_commit": environment.get("git_commit"),
                },
                "last_metrics": _last_metric(run_dir / "metrics.jsonl"),
                "raw_record_dir": str(Path("docs/training/raw-runs") / run_dir.name),
                "checkpoint_metadata": checkpoint_metadata,
                "selected": run_dir.name == selected_run,
            }
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_run": selected_run,
        "record_scope": "all local runs present at export time; checkpoint state.pt files are intentionally excluded",
        "runs": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index J3 training records")
    parser.add_argument("--source-root", default="runs")
    parser.add_argument("--output", default="docs/training/experiment-register.json")
    parser.add_argument("--selected-run", default="20260906-225321")
    args = parser.parse_args()
    export(Path(args.source_root), Path(args.output), args.selected_run)


if __name__ == "__main__":
    main()
