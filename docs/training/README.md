# Training records

The selected submission is J3 Stage 3, run `20260906-225321`. The complete local run record was exported before cleanup:

- [`experiment-register.json`](experiment-register.json) is the machine-readable index of every local run, command, status, last metric, environment summary, and checkpoint metadata path.
- [`raw-runs/`](raw-runs) contains the copied JSON metadata, YAML configs, JSONL metrics, logs, and checkpoint metadata. `state.pt` tensors are excluded from the repository because they are large binary release artifacts.
- [`abandoned-experiments.md`](abandoned-experiments.md) explains which later or auxiliary experiments were removed from the active tree and why they are not selected.

Historical records can contain the old internal shape label `D32-1216-R12`. That label is preserved for audit fidelity; the public model name is now J3.

## Selected run facts

The selected run used `configs/pretrain_stage3_1b.yaml`, a globally mixed Stage 3 manifest, BF16, sequence length 512, micro-batch 16, gradient accumulation 16, effective batch 131,072 tokens, seed 1337, and a 1B-token target. It completed at step 7,630 with 1,000,079,360 tokens seen in 24,595.263 seconds on the recorded RTX 4060 Laptop GPU environment.

The final checkpoint's SHA-256, byte size, and metadata hash are authoritative in [`../submission/release-manifest.json`](../submission/release-manifest.json). The binary is intentionally not committed to Git.
