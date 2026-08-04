"""Plot daily metrics and summarize a specified anomaly window."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOT_COLUMNS = ("applications", "approvals", "frauds", "approval_rate", "application_fraud_rate", "approved_fraud_rate", "avg_credit_score", "avg_fraud_score")


def analyze(daily: pd.DataFrame, output_dir: Path, anomaly_start: str, anomaly_end: str) -> pd.DataFrame:
    """Create plots and compare anomaly-window means with baseline means."""
    frame = daily.copy()
    frame["app_day"] = pd.to_datetime(frame["app_day"], errors="raise")
    start, end = pd.Timestamp(anomaly_start), pd.Timestamp(anomaly_end)
    if start > end:
        raise ValueError("anomaly_start must not be after anomaly_end")
    anomaly = frame["app_day"].between(start, end, inclusive="both")
    if not anomaly.any() or anomaly.all():
        raise ValueError("Anomaly and baseline periods must both contain data.")
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [column for column in PLOT_COLUMNS if column in frame]
    for column in columns:
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(frame["app_day"], frame[column], linewidth=1.5)
        axis.axvspan(start, end, alpha=0.2, color="tab:red", label="Anomaly window")
        axis.set(title=column.replace("_", " ").title(), xlabel="Date", ylabel=column.replace("_", " ").title())
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(output_dir / f"ts_{column}.png", dpi=200)
        plt.close(figure)
    summary = pd.DataFrame({"metric": columns, "baseline_mean": [frame.loc[~anomaly, c].mean() for c in columns], "anomaly_mean": [frame.loc[anomaly, c].mean() for c in columns]})
    summary["pct_change"] = (summary["anomaly_mean"] - summary["baseline_mean"]) / summary["baseline_mean"].replace(0, pd.NA)
    summary = summary.sort_values("pct_change", key=lambda values: values.abs(), ascending=False)
    summary.to_csv(output_dir / "anomaly_vs_baseline_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "outputs" / "p1_daily_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "figures")
    parser.add_argument("--anomaly-start", default="2019-06-25")
    parser.add_argument("--anomaly-end", default="2019-07-04")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(analyze(pd.read_csv(args.input), args.output_dir, args.anomaly_start, args.anomaly_end).round(4).to_string(index=False))
