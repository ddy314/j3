#!/usr/bin/env bash
set -euo pipefail

run_dir="${1:-runs/latest}"
if [[ $# -gt 0 ]]; then
  shift
fi
exec uv run python dashboard.py --run "$run_dir" "$@"
