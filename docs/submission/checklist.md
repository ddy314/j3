# GIBC V2 submission checklist

The official rules are the authority; this checklist is the repository-side preparation record.

## Track 01 TECH

- [x] Public model name is J3.
- [x] Trainable parameter count is 49,001,408, below the 50,000,000 cap.
- [x] Training path uses local source code and local checkpoint weights; no hosted inference API.
- [x] Training is from scratch with no pretrained initialization, fine-tuning, or distillation.
- [x] Public scripts/configs expose the model parameter count and training configuration.
- [x] Hardware, elapsed training time, precision, batch size, and training tokens are recorded.
- [x] HellaSwag, ARC-Easy, PIQA, WinoGrande, and WikiText-103 evaluation protocol is implemented.
- [x] Full evaluation result file is generated and recorded: `docs/evaluation/results/j3-gibc.json`.

## Devpost materials

- [x] Project description draft: [`project-description.md`](project-description.md).
- [x] Public source repository README and reproducibility instructions.
- [x] Built With and AI-assistance disclosure: [`built-with.md`](built-with.md).
- [ ] 2–5 minute English demo video with English audio or subtitles: **TO BE RECORDED**.
- [ ] Team members, roles, and contact information: **TO BE COMPLETED**.
- [ ] At least three final screenshots showing the project in use: **TO BE REVIEWED**.
- [ ] Confirm the public repository and release asset are accessible without login.
- [x] Publish the 588 MB checkpoint as a GitHub Release asset and verify its SHA-256 against [`release-manifest.json`](release-manifest.json).
- [ ] Re-check the official deadline and submission form immediately before submitting.

## Evidence boundaries

The Stage 3 training validation metric is not substituted for the required held-out WikiText-103 score. Smoke evaluation is not a competition result. Raw experiment records are retained even when an experiment is not selected, and unknown dataset-license details are called out rather than inferred.
