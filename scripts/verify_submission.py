"""Verify the local J3 submission bundle before publishing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.tokenizer import tokenizer_sha256  # noqa: E402
from src.model import ModelConfig  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the J3 GIBC submission invariants")
    parser.add_argument("--checkpoint", default="runs/j3-stage3-final/checkpoints/latest/state.pt")
    parser.add_argument("--manifest", default="data/stage3_mixed_tokenized/manifest.json")
    parser.add_argument("--tokenizer", default="data/tokenized/tokenizer.json")
    parser.add_argument("--release-manifest", default="docs/submission/release-manifest.json")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest)
    tokenizer = Path(args.tokenizer)
    release_manifest = Path(args.release_manifest)
    failures: list[str] = []

    model = ModelConfig()
    if model.model_name != "J3":
        failures.append(f"model name is {model.model_name!r}, expected 'J3'")
    if model.parameter_count != 49_001_408:
        failures.append(f"parameter count is {model.parameter_count}, expected 49001408")
    if model.parameter_count > 50_000_000:
        failures.append("model exceeds the GIBC 50M trainable-parameter cap")

    try:
        manifest_payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"cannot read training manifest: {error}")
        manifest_payload = {}
    if manifest_payload.get("train_token_count") != 1_000_000_000:
        failures.append("selected manifest does not contain exactly 1B train tokens")
    if manifest_payload.get("tokenizer_hash") != tokenizer_sha256(tokenizer):
        failures.append("manifest and tokenizer hashes differ")
    mixing = manifest_payload.get("mixing", {})
    if mixing.get("mode") != "global_chunk_shuffle" or mixing.get("chunk_tokens") != 8192:
        failures.append("selected manifest is not the verified global 8192-token mix")

    actual_checkpoint_hash = None
    if not checkpoint.is_file():
        failures.append(f"checkpoint does not exist: {checkpoint}")
    else:
        actual_checkpoint_hash = sha256(checkpoint)

    if release_manifest.is_file():
        release = json.loads(release_manifest.read_text())
        expected_hash = release.get("checkpoint", {}).get("state_sha256")
        if expected_hash and actual_checkpoint_hash != expected_hash:
            failures.append("checkpoint hash differs from release-manifest.json")
    else:
        failures.append(f"release manifest does not exist: {release_manifest}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "status": "passed",
                "model": model.model_name,
                "parameters": model.parameter_count,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": actual_checkpoint_hash,
                "manifest": str(manifest.resolve()),
                "tokenizer_sha256": tokenizer_sha256(tokenizer),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
