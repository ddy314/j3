#!/usr/bin/env bash
set -euo pipefail

exec uv run python train.py --config configs/contrastive_mc_posttrain.yaml "$@"
