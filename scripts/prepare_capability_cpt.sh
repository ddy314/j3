#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

exec uv run python scripts/prepare_hf_mix.py \
  --config configs/capability_cpt_mix_125m.yaml \
  --tokenizer data/tokenized/tokenizer.json \
  --output-dir data/capability_cpt_tokenized \
  --parquet-root "$repo_dir" \
  --shard-tokens 4194304 \
  --val-ratio 0.01 \
  --seed 1337 \
  --deduplicate \
  --decontamination-index data/decontamination/public_benchmarks.npz \
  "$@"
