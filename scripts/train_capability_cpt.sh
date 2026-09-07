#!/usr/bin/env bash
set -euo pipefail

exec uv run python train.py --config configs/pretrain_capability_cpt_125m.yaml "$@"
