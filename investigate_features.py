"""Run Train/Validation feature investigation and one final Test confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection.feature_analysis import run_feature_investigation


def run(input_path: Path, results_dir: Path, figures_dir: Path, n_repeats: int) -> None:
    """Run the investigation and print compact empirical summaries."""
    outputs = run_feature_investigation(
        pd.read_csv(input_path), results_dir, figures_dir, n_repeats=n_repeats
    )
    print(f"Selected configuration: {outputs['selected_configuration']}")
    for name in (
        "missingness_analysis", "permutation_importance", "shap_feature_importance",
        "feature_group_ablation", "reduced_feature_comparison", "final_feature_confirmation",
    ):
        print(f"\n{name}:")
        print(outputs[name].round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "outputs" / "p2_dataset_clean.csv")
    parser.add_argument("--results-dir", type=Path, default=root / "docs" / "results")
    parser.add_argument("--figures-dir", type=Path, default=root / "docs" / "figures")
    parser.add_argument("--n-repeats", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.results_dir, args.figures_dir, args.n_repeats)
