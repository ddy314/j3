# J3

J3 is a direct PyTorch pretraining system for a 49M-parameter decoder-only language model. It is designed for long single-GPU runs: the data stream is offline-tokenized and mmap-backed, the training loop is small, checkpoints are atomic and portable, and the dashboard is a separate read-only process.

## Frozen model

`D32-1216-R12` has exactly `49,001,408` trainable parameters:

- vocabulary 16,384; model width 384; 32 independent decoder blocks;
- 6 Q heads / 2 KV heads, head dimension 64, native SDPA GQA;
- per-head Q/K RMSNorm and RoPE;
- pre-RMSNorm blocks, bias-free MLP with 1,216 hidden units and constrained trainable scalar xIELU;
- tied token embedding/language-model head plus a rank-12 untied residual output head;
- configurable context length, default 1,024.

No RPR, depth sharing, recurrent block, or depth-delta mechanism is present.

## Environment

Python 3.13 is selected by `.python-version`. Install the locked environment and print the complete hardware/runtime report:

```bash
./scripts/setup.sh
# or
uv sync --locked
uv run python scripts/check_env.py
```

The baseline intentionally does not install `flash-attn`: PyTorch SDPA is benchmarked first and selects the CUDA flash path on supported shapes. Triton is supplied by the locked PyTorch environment and is checked by `check_env.py`.

## Data preparation

Training never tokenizes raw text. For the English pretraining mix, use the
true byte-level BPE path, then create uint16 shards:

```bash
uv run python scripts/train_tokenizer.py \
  --input /path/to/text-or-jsonl \
  --output-dir data/tokenized \
  --tokenizer-type byte-bpe \
  --vocab-size 16384 \
  --min-token-frequency 10 \
  --max-token-length 32

uv run python scripts/prepare_data.py \
  --input /path/to/text-or-jsonl \
  --tokenizer data/tokenized/tokenizer.json \
  --output-dir data/tokenized \
  --shard-tokens 4194304 \
  --val-ratio 0.001 \
  --deduplicate
```

For the requested English mixture, use the checked-in [HF mix config](configs/hf_mix_1b.yaml). It selects DCLM-Edu with `edu_int_score >= 3` (600M tokens), FineWeb-Edu `sample-10BT` (300M), Cosmopedia v2 from `HuggingFaceTB/smollm-corpus` (100M), and WikiText-103 (100M). The extractor tokenizes while streaming and stops at the exact per-source token quota; it does not materialize a 4–5 GB raw-text corpus.

Install the HF extra once, then train a tokenizer from a bounded sample of every source:

```bash
uv sync --locked --extra data

uv run python scripts/train_tokenizer.py \
  --mix-config configs/hf_mix_1b.yaml \
  --parquet-root /cache/j3-hf-parquet \
  --output-dir data/tokenized-pilot \
  --tokenizer-type byte-bpe \
  --vocab-size 16384 \
  --max-documents-per-source 10000 \
  --min-token-frequency 10 \
  --max-token-length 32

uv run python scripts/audit_tokenizer.py \
  --tokenizer data/tokenized-pilot/tokenizer.json \
  --mix-config configs/hf_mix_1b.yaml \
  --parquet-root /cache/j3-hf-parquet \
  --json-output data/tokenized-pilot/tokenizer_audit.json
```

Pull an 11M-token pilot first (1% of every requested quota):

```bash
uv run python scripts/prepare_hf_mix.py \
  --config configs/hf_mix_1b.yaml \
  --tokenizer data/tokenized-pilot/tokenizer.json \
  --output-dir data/tokenized-pilot \
  --scale 0.01 \
  --shard-tokens 4194304 \
  --val-ratio 0.01
```

After checking the manifest and disk/network behavior, repeat into a clean
`data/tokenized` directory with `--scale 1.0`. Copy the pilot tokenizer into
that clean directory first, or train it again from the same bounded sample:

```bash
mkdir -p data/tokenized
cp data/tokenized-pilot/tokenizer.json data/tokenized/tokenizer.json
cp data/tokenized-pilot/tokenizer_meta.json data/tokenized/tokenizer_meta.json

uv run python scripts/prepare_hf_mix.py \
  --config configs/hf_mix_1b.yaml \
  --tokenizer data/tokenized/tokenizer.json \
  --output-dir data/tokenized \
  --scale 1.0 \
  --shard-tokens 4194304 \
  --val-ratio 0.01
```

This produces the requested 1.1B training tokens and is the path used by
`configs/pretrain_1b.yaml`.
The extractor writes `mix_progress.json` after each completed source, resumes
completed sources, and retries transient stream failures a bounded number of
times. For a slow or unstable Xet/CDN route, use bounded HTTP timeouts and the
regular Hub route:

```bash
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60
```

`load_dataset` uses `HF_TOKEN` or the credential saved by `hf auth login` when
available, without printing or copying the token; public datasets fall back to
anonymous access when no credential is present.

If Hub/Xet metadata works but a streaming shard stalls, download only the
needed parquet shards with a resumable downloader such as `aria2c`, placing
them under one directory per source (for example,
`/cache/j3-hf-parquet/dclm_edu_3plus/*.parquet`). Then pass that root to the
same extractor:

```bash
uv run python scripts/prepare_hf_mix.py \
  --config configs/hf_mix_1b.yaml \
  --tokenizer data/tokenized/tokenizer.json \
  --output-dir data/tokenized \
  --parquet-root /cache/j3-hf-parquet \
  --scale 1.0 \
  --shard-tokens 4194304 \
  --val-ratio 0.01
```

Local parquet mode uses the same filters, tokenizer, exact quotas, progress
file, and manifest; it only replaces the network row iterator.

The selected sources are English datasets/configurations; the DCLM and
FineWeb rows are additionally checked for `language: en`. The 100M WikiText
target remains approximate: if the split cannot supply that many tokens, the
extractor stops with an error instead of silently repeating it.

The older `--hf-dataset` mode remains available in both tokenizer/data scripts
for a single source. The trainer itself does not use HF streaming.
`manifest.json` records source, tokenizer hash, dtype, shard counts, token
counts, source filters, duplicate-filter status, and token-frequency statistics.
The legacy SentencePiece trainer remains available with
`--tokenizer-type sentencepiece`; its Unicode vocabulary is not used for this
English 1.1B-token artifact. `audit_tokenizer.py` treats the 256 byte alphabet
as required infrastructure and audits only learned merges for corpus-specific
garbage.

### Stage 3 mixture

Stage 3 is assembled by `configs/stage3_1b.yaml` into an exact 1.0B-token
training stream:

- FinePDF: the supplied nominal 300M-token artifact;
- 350,206,468 Stage 1-style clean Web/Edu tokens from FineWeb-Edu;
- 250M Stage 2-style tokens from natural Web, narrative, and procedural sources;
- 100M WikiText-103 encyclopedic tokens.

The Stage 2-style slices are copied from the verified local Stage 2 shards, while
FineWeb-Edu and WikiText are streamed from the Hub. DCLM-Edu is deliberately
omitted because its Parquet route was previously too slow and memory-heavy in
this environment. The assembler
reuses the canonical byte-level tokenizer, enables duplicate filtering and the
existing benchmark decontamination index for new HF rows, and writes a separate
auditable `manifest.json` plus `verification.json`:

```bash
UV_CACHE_DIR=/tmp/j3-uv-cache uv run python scripts/prepare_stage3.py \
  --config configs/stage3_1b.yaml

UV_CACHE_DIR=/tmp/j3-uv-cache uv run python scripts/prepare_stage3.py \
  --config configs/stage3_1b.yaml --verify-only
```

The supplied FinePDF manifest has a validation-path typo (`shard_000000.bin`
instead of `val/shard_000000.bin`). Stage 3 repairs this only in its assembled
copy and records the correction in the source provenance; the original supplied
directory is left untouched. Use `configs/pretrain_stage3_1b.yaml` only after
the Stage 3 verification passes.

Before training, make the mixture real rather than relying on the runtime's
coarse shard permutation:

```bash
UV_CACHE_DIR=/tmp/j3-uv-cache uv run python scripts/mix_tokenized_manifest.py \
  --input-manifest data/stage3_tokenized/manifest.json \
  --output-dir data/stage3_mixed_tokenized \
  --chunk-tokens 8192 \
  --seed 1337

UV_CACHE_DIR=/tmp/j3-uv-cache uv run python scripts/mix_tokenized_manifest.py \
  --output-dir data/stage3_mixed_tokenized --verify-only
```

This globally shuffles 8,192-token chunks before writing new shards; it does
not permute individual token IDs, so local language continuity is retained.
The training config uses `1e-4` peak LR and `1e-5` minimum LR, and starts from
the completed Stage 2 checkpoint with `--init-from` so the Stage 3 data cursor
starts at zero.

### Final high-information 500M mixture

For the final evaluation-oriented continuation, use
`configs/stage3_final_500m.yaml`. It is an exact 500M-token train stream with
no generic Web, SEO, template-page, or low-information fragment bucket:

- 250M long-form FinePDF textbook tokens;
- 40M curated synthetic textbook tokens from Cosmopedia;
- 75M scientific-explanation tokens from the verified local peS2o slice;
- 30M encyclopedic WikiText-103 tokens;
- 55M OpenOrca question-answer knowledge tokens;
- 30M coherent synthetic narrative tokens;
- 15M synthetic procedural knowledge tokens from WikiHow;
- 5M verified synthetic reasoning tokens, kept as a small auxiliary slice
  because the final evaluation has no math-reasoning item.

The new Hub inputs are downloaded in a resumable, size-checked way:

```bash
./scripts/download_stage3_final_sources.sh
```

Then assemble, globally mix, and verify the final artifact:

```bash
UV_CACHE_DIR=/tmp/j3-uv-cache uv run python scripts/prepare_stage3.py \
  --config configs/stage3_final_500m.yaml \
  --parquet-root /home/xia/.cache/j3-stage3-final-hf \
  --workers 4 --batch-size 2048

UV_CACHE_DIR=/tmp/j3-uv-cache uv run python scripts/mix_tokenized_manifest.py \
  --input-manifest data/stage3_final_tokenized/manifest.json \
  --output-dir data/stage3_final_mixed_tokenized \
  --chunk-tokens 8192 --seed 1337

UV_CACHE_DIR=/tmp/j3-uv-cache uv run python scripts/mix_tokenized_manifest.py \
  --output-dir data/stage3_final_mixed_tokenized --verify-only
```

After verification, `configs/pretrain_stage4_final_500m.yaml` points both train
and validation streams at the mixed manifest and keeps the continuation LR at
`1e-4` peak / `1e-5` minimum. Start it from the completed Stage 3 checkpoint
with `--init-from`; do not use `--resume` across the manifest change. The
earlier `pretrain_stage3_final_500m.yaml` filename is retained only for
compatibility.

## Smoke training

The smoke configuration uses deterministic synthetic tokens when no manifest is supplied and runs about one million tokens:

```bash
./scripts/train_smoke.sh
# short runtime check
./scripts/train_smoke.sh --max-steps 4
```

Each run creates `runs/<run_id>/` with `config.yaml`, `environment.json`, `command.json`, `metrics.jsonl`, `status.json`, `train.log`, and `checkpoints/`. `runs/latest` points to the most recent run.

The local GPU smoke validation completed 128 optimizer steps / 1,048,576 tokens with status `completed`, BF16 + compiled `default` mode, fused AdamW, and a final checkpoint. The observed final logged training loss was 3.3990; use the run artifacts rather than this example value for later comparisons.

## Benchmark and profiling

Benchmark uses CUDA events after warmup and compares eager, `default`, `reduce-overhead`, and `max-autotune` across sequence lengths 512/1,024 and increasing microbatches. It reports tokens/sec, step time, forward/backward/optimizer breakdown, GPU utilization, VRAM, temperature, power, and OOM boundaries:

```bash
./scripts/benchmark.sh
```

Results are written to `artifacts/benchmarks.json` and `artifacts/benchmarks.csv`. Profile traces and summaries are produced with:

```bash
uv run python scripts/profile_train.py \
  --mode reduce-overhead \
  --sequence-length 1024 \
  --micro-batch-size 8
```

The trace is under `artifacts/profile/<mode>/traces/`; the summary includes module ranges and top CUDA/CPU operations. The implemented optimization order is eager → `torch.compile` → profiler. No custom kernel is kept unless a measured stable gain justifies it.

The checked-in benchmark was run on one RTX 4060 Laptop GPU with eight warmup and twenty measured steps per isolated trial. It contains 38 completed/explicit-boundary trials; each compile mode completed successfully. The `seq=1,024, microbatch=16` boundary was OOM. For the first large-scale stage, the selected point is `seq=512, microbatch=16`; the 1,024-token result remains available for later context-length expansion:

| mode | seq 512 best | seq 512 tok/s | seq 512 step | seq 1,024 best | seq 1,024 tok/s | seq 1,024 step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | 4 | 16,611 | 123.29 ms | 2 | 16,105 | 127.17 ms |
| `default` | 16 | **44,342** | 184.74 ms | 8 | **41,306** | 198.33 ms |
| `reduce-overhead` | 16 | 44,053 | 185.96 ms | 8 | 41,082 | 199.41 ms |
| `max-autotune` | 16 | 44,316 | 184.86 ms | 8 | 41,331 | 198.20 ms |

The formal config keeps `default`; `max-autotune` was only 0.06% faster at
sequence 1,024 in this scan, below a meaningful stability margin.

At the selected `default` operating points, peak PyTorch allocated VRAM was 4,867.1 MiB, reserved VRAM was 5,142 MiB, NVIDIA-reported memory peaked at 5,322 MiB, and sampled GPU utilization reached 100% (temperature up to 77°C, power about 88–89 W). Relative to the best eager point, `default` compile improved throughput by 2.67× at sequence 512 and 2.56× at sequence 1,024. These results are synthetic-token model-step measurements; the mmap input path is separately tested and should be rechecked on the target storage system.

The compiled profile shows PyTorch flash-attention forward/backward kernels, BF16 Tensor Core GEMMs, compiler-generated Triton fused xIELU/RMSNorm, and fused AdamW. The eager profile identifies MLP/xIELU elementwise work and launch count as the main small-model overhead. Since the compiled path already fuses these operations and max-autotune did not win, no custom CUDA/Triton kernel was added.

Before a long run, use the LR proxy sweep. By default it reuses the formal
sequence length, microbatch, accumulation, effective global batch, warmup, and
cosine horizon; `--tokens` only limits how far into that formal schedule each
independent run proceeds. This avoids selecting an LR with a different
effective batch from the formal 131,072 tokens per optimizer step:

```bash
uv run python scripts/lr_range_test.py --config configs/pretrain_1b.yaml --tokens 20000000
```

The model stores xIELU parameters in an unconstrained optimization space but
uses `softplus(alpha_p)` and `beta + softplus(alpha_n)` in the forward pass,
so the effective slopes cannot become invalid during training. Q/K RMSNorm is
applied independently to each head vector. The loss uses the autocast-native
logits dtype instead of making an explicit full FP32 logits copy; the
benchmark artifact verifies the resulting memory boundary.

## Pretraining

After `data/stage2_tokenized/manifest.json` exists, inspect and adjust the LR proxy result, then start the formal 1.0B-token Stage 2 configuration:

```bash
./scripts/train_pretrain_1b.sh
```

The default first-stage settings are BF16, `torch.compile(mode="default")`, fused AdamW when supported, weight decay 0.1, gradient clipping 1.0, token-indexed warmup plus cosine decay, sequence length 512, microbatch 16, accumulation 16, and effective global batch 131,072 tokens. The model still supports a 1,024-token context; that length is reserved for a later stage. The script does not run automatically. This mode/batch was selected from the checked-in RTX 4060 benchmark artifact; rerun the benchmark on a different GPU.

`global_batch_tokens` must equal `sequence_length * micro_batch_size * gradient_accumulation_steps`; this is validated at startup. A larger microbatch can be selected from the benchmark and the accumulation adjusted to preserve the desired effective batch.

## Resume and safe pause

Resume the latest complete checkpoint:

```bash
uv run python train.py --config configs/pretrain_1b.yaml --resume auto
```

To continue pretraining from the completed first-stage model while starting
the Stage 2 token stream at its beginning, initialize a new run from the
first-stage checkpoint:

```bash
./scripts/train_pretrain_1b.sh --init-from runs/20260905-021323/checkpoints/latest
```

`--init-from` carries over model weights, Adam state, and RNG state but resets
the Stage 2 data cursor and its additional-token counter. Use `--resume` only
for an interrupted run on the same manifest.

Or resume a specific checkpoint:

```bash
uv run python train.py \
  --config configs/pretrain_1b.yaml \
  --resume runs/<run_id>/checkpoints/tokens_000050m
```

Checkpoint state includes model, optimizer, token scheduler, global step/tokens, epoch/shard/offset/order, Python/NumPy/PyTorch CPU/CUDA RNG, model/training config, tokenizer and manifest hashes, Git commit, software versions, host, timestamp, and reason. Checkpoint directories are written to a temporary directory, fsynced, and atomically renamed; `checkpoints/latest` is an atomic relative symlink. Time checkpoints, token milestones, emergency checkpoints, `keep_last_n`, and permanent milestone retention are supported.

`Ctrl+C`, `SIGTERM`, `runs/<run_id>/STOP_REQUESTED`, and `control.json` request a graceful emergency checkpoint. The process saves after the current optimizer step and exits cleanly. A dashboard checkpoint/stop request uses the same control file and never kills the trainer.

To move a run to another computer, copy the entire run directory and the immutable tokenizer/manifest/shards, install the locked environment, and use the same resume command. The checkpoint validates model and dataset identity before loading.

## Dashboard

The dashboard is decoupled from training and incrementally tails `metrics.jsonl` rather than rereading the file on every refresh:

```bash
./scripts/dashboard.sh runs/<run_id>
# or
uv run python dashboard.py --run runs/latest --port 7860
```

Open `http://127.0.0.1:7860`. It shows overview/progress, interactive Plotly loss curves, throughput, step time, GPU telemetry, training settings, ETA windows, checkpoint metadata, and recent logs. Hover, zoom, pan, and toggle curves directly in the browser. The optional buttons request a checkpoint or graceful stop through `control.json`.

## Repository layout

```text
src/model/       model, attention, MLP/xIELU, RMSNorm, RoPE
src/data/        tokenizer wrapper, raw readers, mmap token stream
src/training/    trainer, atomic checkpoints, scheduler, metrics, telemetry
scripts/         environment, data, benchmark, profile, launch utilities
configs/         smoke, benchmark, and 1.1B-token pretraining configs
tests/           model, data, checkpoint, scheduler, RNG, and exact-resume tests
```

## Common issues

- `nvidia-smi`/CUDA unavailable: run `uv run python scripts/check_env.py`; the code can validate on CPU, but GPU throughput claims require a working driver-visible CUDA process.
- CUDA OOM: select the largest successful benchmark microbatch and increase accumulation; do not simply raise the microbatch in the formal config.
- First `torch.compile` invocation is slow: it compiles Triton/Inductor kernels. Benchmark numbers exclude this one-time compilation after warmup.
- Missing or mismatched tokenizer/manifest: run the offline data steps again and keep the tokenizer, manifest, and shards together; resume rejects hash mismatches. Checkpoints created before a model-config change (including the constrained xIELU parameterization) must be restarted from the matching code/config rather than silently resumed.
- Missing `flash_attn`: expected for this baseline. Native PyTorch SDPA is the supported attention implementation.

## Verification

```bash
uv run pytest
uv run python -m compileall -q src scripts train.py dashboard.py
```
