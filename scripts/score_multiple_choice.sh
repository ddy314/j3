#!/usr/bin/env bash
set -euo pipefail

exec uv run python scripts/score_multiple_choice.py "$@"
