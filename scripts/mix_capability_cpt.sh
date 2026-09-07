#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

exec uv run python scripts/mix_tokenized_manifest.py \
  --input-manifest data/capability_cpt_tokenized/manifest.json \
  --output-dir data/capability_cpt_mixed_tokenized \
  --chunk-tokens 8192 \
  --seed 1337 \
  --dataset-type capability_globally_mixed \
  "$@"
