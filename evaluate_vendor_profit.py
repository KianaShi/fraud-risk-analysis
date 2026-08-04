"""Compare model performance and expected profit with and without vendor data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection.profit import compare_vendor_scenarios


def run(input_path: Path, output_dir: Path, decline_threshold: float, approve_threshold: float) -> None:
    """Evaluate both scenarios and save the comparison and scored holdout."""
    comparison, scored = compare_vendor_scenarios(pd.read_csv(input_path), decline_threshold, approve_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.round(4).to_csv(output_dir / "profit_scenario_comparison.csv", index=False)
    scored.to_csv(output_dir / "scored_testset.csv", index=False)
    print(comparison.round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "outputs" / "p2_dataset_clean.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--decline-threshold", type=float, default=0.8)
    parser.add_argument("--approve-threshold", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output_dir, args.decline_threshold, args.approve_threshold)
