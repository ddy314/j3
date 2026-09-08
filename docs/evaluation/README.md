# GIBC V2 evaluation

The official Track 01 protocol calls for `lm-evaluation-harness` evaluation on HellaSwag, ARC-Easy, PIQA, and WinoGrande, plus held-out WikiText-103 perplexity. J3 uses the checked-in local adapter [`scripts/evaluate_gibc.py`](../../scripts/evaluate_gibc.py) so the harness can score the native PyTorch model with the exact checked-in tokenizer and local checkpoint.

Run:

```bash
uv sync --locked --extra eval
uv run python scripts/evaluate_gibc.py \
  --checkpoint runs/j3-stage3-final/checkpoints/latest \
  --config configs/pretrain_stage3_1b.yaml \
  --tasks hellaswag arc_easy piqa winogrande j3_wikitext_103 \
  --output docs/evaluation/results/j3-gibc.json
```

The custom `j3_wikitext_103` task is defined in [`eval_tasks/wikitext_103.yaml`](../../eval_tasks/wikitext_103.yaml) and reports rolling log-likelihood-derived word perplexity, byte perplexity, and bits per byte for the WikiText-103 test split. The task is kept separate from the training validation stream.

The full result file is [`results/j3-gibc.json`](results/j3-gibc.json) when generated. A file named `j3-gibc-smoke.json` is only a harness/model smoke check and must not be presented as the competition score.
