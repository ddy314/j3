#!/usr/bin/env bash
set -euo pipefail

uv sync --locked
uv run python scripts/check_env.py
