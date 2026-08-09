"""Run Stage C foundation-model benchmarking on Train -> Validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_detection.foundation_models import run_stage_c


def run(input_path: Path, output_path: Path, metadata_path: Path) -> None:
    results, metadata = run_stage_c(
        pd.read_csv(input_path),
        output_csv=output_path,
        metadata_json=metadata_path,
    )
    print("Exact native representation passed to both foundation models:")
    print(json.dumps(metadata["representation"], indent=2))
    print("\nStage C Validation results:")
    print(results.to_string(index=False))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "outputs" / "p2_dataset_clean.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "docs" / "results" / "foundation_model_validation.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "docs" / "results" / "foundation_model_validation_metadata.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input, arguments.output, arguments.metadata)
