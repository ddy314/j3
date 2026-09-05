#!/usr/bin/env bash
set -euo pipefail

root="${1:-/home/xia/.cache/j3-hf-parquet}"
aria=(aria2c --file-allocation=none --continue=true --max-connection-per-server=16 --split=16 --min-split-size=16M --max-tries=12 --retry-wait=5 --timeout=90 --connect-timeout=30)

download() {
  local directory="$1" output="$2" url="$3"
  mkdir -p "$root/$directory"
  "${aria[@]}" --dir="$root/$directory" --out="$output" "$url"
}

for snapshot in 30 33 38 43 47 51; do
  download "ultra_fineweb_l1_2025_${snapshot}" 0000.parquet \
    "https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L1/resolve/refs%2Fconvert%2Fparquet/CC-MAIN-2025-${snapshot}/train/0000.parquet"
done

download ultra_fineweb_l2 0000.parquet \
  "https://huggingface.co/datasets/openbmb/Ultra-FineWeb/resolve/refs%2Fconvert%2Fparquet/default/en/0000.parquet"

# Cosmopedia shard 0000 was already used by Stage 1. Add three independent
# shards because the requested format strata are much smaller than the total.
for shard in 00001 00002 00003 00004 00005 00006; do
  download cosmopedia_v2 "${shard}.parquet" \
    "https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/main/cosmopedia-v2/train-${shard}-of-00104.parquet"
done

download pes2o_v2 train-00000-of-00020.json.gz \
  "https://huggingface.co/datasets/allenai/peS2o/resolve/main/data/v2/train-00000-of-00020.json.gz"

for part in 1 2 3 4 5 6 7 8 9; do
  download common_corpus "subset_100_${part}.parquet" \
    "https://huggingface.co/datasets/PleIAs/common_corpus/resolve/main/common_corpus_1/subset_100_${part}.parquet"
done

download tinystories_gpt4_v2 TinyStoriesV2-GPT4-train.txt \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
