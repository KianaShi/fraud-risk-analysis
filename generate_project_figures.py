"""Generate curated figures for the GitHub README from workflow outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"navy": "#17324D", "blue": "#2F6B9A", "teal": "#2A9D8F", "orange": "#E76F51", "gray": "#68737D"}


def _save(figure: plt.Figure, path: Path) -> None:
    """Apply shared layout settings and save a figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def generate_figures(outputs_dir: Path, figures_dir: Path, results_dir: Path) -> None:
    """Create curated charts and compact, publishable result tables."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    daily = pd.read_csv(outputs_dir / "p1_daily_metrics.csv", parse_dates=["app_day"])
    anomaly_start, anomaly_end = pd.Timestamp("2019-06-25"), pd.Timestamp("2019-07-04")

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(daily["app_day"], daily["applications"], color=COLORS["blue"], linewidth=1.8)
    axis.axvspan(anomaly_start, anomaly_end, color=COLORS["orange"], alpha=0.18, label="Specified anomaly window")
    axis.set(title="Daily Payment Applications", xlabel="Application date", ylabel="Applications")
    axis.legend(frameon=False)
    figure.autofmt_xdate()
    _save(figure, figures_dir / "daily_applications.png")

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(daily["app_day"], daily["approved_fraud_rate"] * 100, color=COLORS["orange"], linewidth=1.8)
    axis.axvspan(anomaly_start, anomaly_end, color=COLORS["orange"], alpha=0.14)
    axis.set(title="Fraud Rate Among Approved Applications", xlabel="Application date", ylabel="Approved fraud rate (%)")
    figure.autofmt_xdate()
    _save(figure, figures_dir / "approved_fraud_rate.png")

    models = pd.read_csv(outputs_dir / "model_comparison.csv").sort_values("validation_pr_auc")
    models.sort_values("validation_pr_auc", ascending=False).to_csv(results_dir / "model_comparison.csv", index=False)
    pd.read_csv(outputs_dir / "split_audit.csv").to_csv(results_dir / "split_audit.csv", index=False)
    labels = models["model"].str.replace("_", " ").str.title()
    positions = range(len(models))
    figure, axis = plt.subplots(figsize=(9, 4.8))
    width = 0.36
    axis.bar([position - width / 2 for position in positions], models["validation_pr_auc"], width, label="Validation PR-AUC", color=COLORS["blue"])
    axis.bar([position + width / 2 for position in positions], models["test_pr_auc"], width, label="Test PR-AUC", color=COLORS["teal"])
    axis.set_xticks(list(positions), labels)
    axis.set_ylim(0.75, 1.0)
    axis.set(title="Stratified Task 2 Model Performance", ylabel="PR-AUC")
    axis.legend(frameon=False)
    _save(figure, figures_dir / "model_comparison.png")

    vendor = pd.read_csv(outputs_dir / "profit_scenario_comparison.csv")
    vendor_labels = vendor["scenario"].map({"tree_without_vendor": "Without vendor", "tree_with_vendor": "With vendor"})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(vendor_labels, vendor["roc_auc"], color=[COLORS["gray"], COLORS["teal"]])
    axes[0].set_ylim(0.75, 1.0)
    axes[0].set(title="Predictive Performance", ylabel="Holdout ROC-AUC")
    profit_bars = axes[1].bar(vendor_labels, vendor["profit"], color=[COLORS["gray"], COLORS["orange"]])
    axes[1].set(title="Expected Profit Under Case Assumptions", ylabel="Expected profit ($)")
    axes[1].bar_label(profit_bars, labels=[f"${value:,.0f}" for value in vendor["profit"]], padding=3)
    figure.suptitle("Incremental Value of FraudKiller Data", fontsize=14, fontweight="bold")
    _save(figure, figures_dir / "vendor_evaluation.png")

    print(f"Saved 4 curated figures to {figures_dir} and model results to {results_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=root / "outputs")
    parser.add_argument("--figures-dir", type=Path, default=root / "docs" / "figures")
    parser.add_argument("--results-dir", type=Path, default=root / "docs" / "results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_figures(args.outputs_dir, args.figures_dir, args.results_dir)
