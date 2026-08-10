"""Build the Fraud v1.0 CatBoost artifact from an approved development-only dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_detection.common import to_snake_case
from fraud_detection.production import (
    EXPECTED_FEATURES,
    build_production_artifacts,
    validate_development_membership,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "User-prepared future-protocol development CSV that must exclude Test rows and "
            "match the externally approved membership manifest."
        ),
    )
    parser.add_argument("--target", default="is_fraud")
    parser.add_argument(
        "--development-membership-manifest",
        type=Path,
        required=True,
        help=(
            "Externally approved stable-ID membership manifest for this development CSV. "
            "This complements, but does not replace, --confirm-development-only."
        ),
    )
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
            "Refusing to fit: first verify outside this program that --input contains only the "
            "future-protocol development rows approved by the supplied membership manifest, then "
            "pass --confirm-development-only. The flag itself cannot prove Test exclusion."
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
    membership = validate_development_membership(
        frame, args.development_membership_manifest
    )
    unexpected = sorted(set(frame.columns) - set((*EXPECTED_FEATURES, "id", args.target)))
    if unexpected:
        raise ValueError(f"Unexpected development columns: {unexpected}")
    result = build_production_artifacts(
        frame.loc[:, EXPECTED_FEATURES],
        frame[args.target],
        args.frozen_config,
        args.model_output,
        args.preprocessor_output,
        args.manifest_output,
        development_membership=membership,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
