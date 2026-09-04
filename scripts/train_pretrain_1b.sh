#!/usr/bin/env bash
set -euo pipefail

exec uv run python train.py --config configs/pretrain_1b.yaml "$@"
