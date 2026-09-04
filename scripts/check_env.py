from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.environment import collect_environment
from src.training.trainer import configure_torch


def main() -> None:
    # Report the effective settings used by the trainer, rather than only the
    # process defaults before training initialization.
    configure_torch(tf32=True)
    report = collect_environment()
    output = Path("artifacts/environment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["cuda_available"]:
        print("ENVIRONMENT_STATUS=CUDA_READY")
    else:
        print("ENVIRONMENT_STATUS=CPU_ONLY")


if __name__ == "__main__":
    main()
