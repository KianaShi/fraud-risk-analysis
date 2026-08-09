"""Generate Fraud v1.0 production metadata from frozen research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_detection.production import build_production_metadata


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", type=Path, default=root / "docs" / "results" / "best_hyperparameters.json")
    parser.add_argument("--model-comparison", type=Path, default=root / "docs" / "results" / "model_family_comparison.csv")
    parser.add_argument("--output", type=Path, default=root / "docs" / "results" / "production_model_metadata.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_production_metadata(args.frozen_config, args.model_comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
