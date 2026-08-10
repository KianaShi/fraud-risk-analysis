"""Build the Fraud v1.0 CatBoost artifact from an approved development-only dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_detection.common import to_snake_case
from fraud_detection.production import EXPECTED_FEATURES, build_production_artifacts


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "User-prepared Train+Validation CSV that must exclude Test rows. "
            "The program cannot verify an arbitrary CSV's split membership."
        ),
    )
    parser.add_argument("--target", default="is_fraud")
    parser.add_argument("--frozen-config", type=Path, default=root / "docs" / "results" / "best_hyperparameters.json")
    parser.add_argument("--model-output", type=Path, default=root / "artifacts" / "catboost_fraud_model.cbm")
    parser.add_argument("--preprocessor-output", type=Path, default=root / "artifacts" / "catboost_preprocessor.json")
    parser.add_argument("--manifest-output", type=Path, default=root / "artifacts" / "catboost_artifact_manifest.json")
    parser.add_argument(
        "--confirm-development-only",
        action="store_true",
        help=(
            "Required user attestation that --input already excludes Test rows. "
            "This flag records the declaration only; it is not proof and performs no split-membership verification."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_development_only:
        raise SystemExit(
            "Refusing to fit: first verify outside this program that --input contains only the frozen "
            "Train+Validation development rows, then pass --confirm-development-only. The program "
            "cannot determine whether an arbitrary CSV contains Test rows."
        )
    frame = pd.read_csv(args.input)
    normalized_target = to_snake_case(args.target)
    if normalized_target in EXPECTED_FEATURES:
        raise ValueError(
            f"Target {args.target!r} normalizes to predictive feature {normalized_target!r}; "
            "the target must be separate from every predictive feature."
        )
    if args.target not in frame:
        raise ValueError(f"Missing target column: {args.target}")
    unexpected = sorted(set(frame.columns) - set((*EXPECTED_FEATURES, args.target)))
    if unexpected:
        raise ValueError(f"Unexpected development columns: {unexpected}")
    result = build_production_artifacts(
        frame.loc[:, EXPECTED_FEATURES],
        frame[args.target],
        args.frozen_config,
        args.model_output,
        args.preprocessor_output,
        args.manifest_output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
