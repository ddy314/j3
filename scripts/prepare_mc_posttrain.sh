#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

exec uv run python scripts/prepare_mc_posttrain.py \
  --config configs/mc_posttrain.yaml \
  --tokenizer data/tokenized/tokenizer.json \
  --output-dir data/mc_posttrain_tokenized \
  --decontamination-index data/decontamination/public_benchmarks.npz \
  "$@"
