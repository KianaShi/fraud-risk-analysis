"""Score merchant JSON or CSV records with the Fraud v1.0 CatBoost candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_detection.production import ProductionCatBoost, apply_decision_policy


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=root / "artifacts" / "catboost_fraud_model.cbm")
    parser.add_argument("--preprocessor", type=Path, default=root / "artifacts" / "catboost_preprocessor.json")
    parser.add_argument("--manifest", type=Path, default=root / "artifacts" / "catboost_artifact_manifest.json")
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=root / "docs" / "results" / "best_hyperparameters.json",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        help="Optional business-supplied threshold where manual review begins; no default is provided.",
    )
    parser.add_argument(
        "--decline-threshold",
        type=float,
        help="Optional business-supplied decline threshold; no default is provided.",
    )
    return parser.parse_args()


def _read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload if isinstance(payload, list) else [payload])
    raise ValueError("Input must be a .csv or .json file.")


def main() -> None:
    args = parse_args()
    supplied = (args.review_threshold is not None, args.decline_threshold is not None)
    if supplied[0] != supplied[1]:
        raise ValueError("Supply both review and decline thresholds, or neither.")
    scorer = ProductionCatBoost.load(
        args.model, args.preprocessor, args.frozen_config, args.manifest
    )
    result = scorer.score(_read_input(args.input))
    if all(supplied):
        result["decision"] = apply_decision_policy(
            result["fraud_probability"].to_numpy(),
            review_threshold=args.review_threshold,
            decline_threshold=args.decline_threshold,
        )
        result["decision_policy"] = "explicit caller-supplied thresholds"
    result["model_identifier"] = scorer.identifier
    print(result.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
