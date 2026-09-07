#!/usr/bin/env bash
set -euo pipefail

# Download only the new HF inputs for configs/stage3_final_500m.yaml. Files are
# kept outside the repository because they are large, resumable source cache
# artifacts; the final token shards and manifest remain under data/.
stage3_hf_root="${1:-/home/xia/.cache/j3-stage3-final-hf}"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required for resumable HF downloads" >&2
  exit 1
fi

download() {
  local url="$1"
  local destination="$2"
  local expected_bytes="$3"
  local directory
  directory="$(dirname "$destination")"
  mkdir -p "$directory"
  if [[ -f "$destination" ]] && [[ "$(stat -c '%s' "$destination")" == "$expected_bytes" ]]; then
    echo "[stage3] already complete: $destination"
    return
  fi
  aria2c \
    --continue=true \
    --file-allocation=none \
    --max-connection-per-server=8 \
    --split=8 \
    --min-split-size=8M \
    --max-tries=8 \
    --retry-wait=5 \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --dir="$directory" \
    --out="$(basename "$destination")" \
    "$url"
  if [[ "$(stat -c '%s' "$destination")" != "$expected_bytes" ]]; then
    echo "size mismatch for $destination" >&2
    echo "expected $expected_bytes bytes, got $(stat -c '%s' "$destination")" >&2
    exit 1
  fi
}

download \
  'https://huggingface.co/datasets/ecreeth/cosmo_v2_filtered/resolve/main/cosmo_v2_text.parquet' \
  "$stage3_hf_root/cosmopedia_filtered/cosmo_v2_text.parquet" \
  1478101043

download \
  'https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/1M-GPT4-Augmented.parquet' \
  "$stage3_hf_root/openorca/1M-GPT4-Augmented.parquet" \
  1008442855

# Five million tokens is intentionally only a small auxiliary math slice. One
# shard has ample headroom; add the later shards only if a future filter change
# makes this exact quota unavailable.
download \
  'https://huggingface.co/datasets/open-r1/OpenThoughts-114k-math/resolve/main/data/train-00000-of-00005.parquet' \
  "$stage3_hf_root/openthoughts/train-00000-of-00005.parquet" \
  199474121

echo "[stage3] new HF source cache is complete: $stage3_hf_root"
