"""Normalize the vendor-enriched modeling dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection.vendor import clean_vendor_data


def run(input_path: Path, sheet: str, output_path: Path) -> None:
    """Read, normalize, and save the vendor dataset."""
    cleaned = clean_vendor_data(pd.read_excel(input_path, sheet_name=sheet))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"Saved {len(cleaned):,} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "FraudCaseStudy.xlsx")
    parser.add_argument("--sheet", default="p2-dataset")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "p2_dataset_clean.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.sheet, args.output)
