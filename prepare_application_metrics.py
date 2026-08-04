"""Generate daily fraud metrics from the case-study workbook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection.applications import build_daily_metrics, build_product_metrics, clean_applications


def run(input_path: Path, sheet: str, output_dir: Path) -> None:
    """Read applications and write daily overall and product metrics."""
    applications = clean_applications(pd.read_excel(input_path, sheet_name=sheet))
    output_dir.mkdir(parents=True, exist_ok=True)
    build_daily_metrics(applications).to_csv(output_dir / "p1_daily_metrics.csv", index=False)
    build_product_metrics(applications).to_csv(output_dir / "p1_daily_metrics_by_product.csv", index=False)
    print(f"Saved metrics for {len(applications):,} cleaned applications to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "FraudCaseStudy.xlsx")
    parser.add_argument("--sheet", default="p1-dataset")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.sheet, args.output_dir)
