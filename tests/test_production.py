"""Tests for the probability-only Fraud v1.0 production boundary."""

from __future__ import annotations

import inspect
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from fraud_detection.production import (
    EXPECTED_FEATURES,
    ProductionCatBoost,
    apply_decision_policy,
    build_production_artifacts,
    build_production_metadata,
    load_frozen_catboost_config,
    validate_feature_frame,
)
import build_production_model as production_cli


ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = ROOT / "docs" / "results" / "best_hyperparameters.json"
MODEL_COMPARISON = ROOT / "docs" / "results" / "model_family_comparison.csv"


def synthetic_features(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "ea_score": rng.normal(500, 40, rows),
            "identity_rank": rng.integers(1, 20, rows),
            "reputation_level": rng.normal(3, 1, rows),
            "volume_score": rng.normal(20, 4, rows),
            "result_number": rng.integers(0, 5, rows),
            "email_days": rng.integers(1, 1000, rows),
            "is_valid": rng.integers(0, 2, rows),
            "is_connected": rng.integers(0, 2, rows),
            "personal_device": rng.integers(0, 2, rows),
            "receiving_mail": rng.integers(0, 2, rows),
            "area_code": rng.choice(["206", "415", "650"], rows),
            "device_browser_type": rng.choice(["Chrome", "Safari"], rows),
            "ip_address_loc_country": rng.choice(["US", "CA"], rows),
            "type": rng.choice(["mobile", "desktop"], rows),
        }
    ).loc[:, EXPECTED_FEATURES]


class ProductionTests(unittest.TestCase):
    def _small_config(self, root: Path, *, depth: int = 3) -> Path:
        base = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
        base["models"]["catboost"]["selected_parameters"] = {
            "iterations": 8,
            "depth": depth,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
            "random_strength": 0.0,
            "bagging_temperature": 0.0,
        }
        path = root / f"frozen-depth-{depth}.json"
        path.write_text(json.dumps(base), encoding="utf-8")
        return path

    def test_production_config_matches_frozen_catboost(self) -> None:
        frozen = load_frozen_catboost_config(FROZEN_CONFIG)
        source = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))["models"]["catboost"]
        self.assertEqual(frozen["selected_parameters"], source["selected_parameters"])
        self.assertEqual(frozen["optimization_seed"], 42)

    def test_schema_validation_orders_features_and_rejects_missing(self) -> None:
        frame = synthetic_features(4)
        reversed_frame = frame.loc[:, list(reversed(frame.columns))]
        self.assertEqual(list(validate_feature_frame(reversed_frame).columns), list(EXPECTED_FEATURES))
        with self.assertRaisesRegex(ValueError, "Missing required feature"):
            validate_feature_frame(frame.drop(columns="email_days"))

    def test_schema_rejects_extra_and_invalid_boolean_values(self) -> None:
        frame = synthetic_features(4)
        with self.assertRaisesRegex(ValueError, "Unexpected feature"):
            validate_feature_frame(frame.assign(merchant_id="secret"))
        invalid = frame.copy()
        invalid.loc[0, "is_valid"] = 2
        with self.assertRaisesRegex(ValueError, "Boolean features"):
            validate_feature_frame(invalid)

    def test_schema_rejects_positive_and_negative_infinity(self) -> None:
        for value in (np.inf, -np.inf):
            with self.subTest(value=value):
                frame = synthetic_features(4)
                frame.loc[0, "ea_score"] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_feature_frame(frame)

    def test_cli_rejects_target_feature_overlap_before_fit(self) -> None:
        frame = synthetic_features(8)
        for target_name in ("is_valid", "ea_score"):
            args = SimpleNamespace(
                confirm_development_only=True,
                input=Path("synthetic.csv"),
                target=target_name,
                frozen_config=FROZEN_CONFIG,
                model_output=Path("model.cbm"),
                preprocessor_output=Path("preprocessor.json"),
                manifest_output=Path("manifest.json"),
            )
            with self.subTest(target=target_name), patch.object(
                production_cli, "parse_args", return_value=args
            ), patch.object(production_cli.pd, "read_csv", return_value=frame), patch.object(
                production_cli, "build_production_artifacts"
            ) as build:
                with self.assertRaisesRegex(ValueError, f"{target_name}.*predictive feature"):
                    production_cli.main()
                build.assert_not_called()

    def test_cli_accepts_separate_fraud_target(self) -> None:
        frame = synthetic_features(8).assign(is_fraud=np.tile([0, 1], 4))
        args = SimpleNamespace(
            confirm_development_only=True,
            input=Path("synthetic.csv"),
            target="is_fraud",
            frozen_config=FROZEN_CONFIG,
            model_output=Path("model.cbm"),
            preprocessor_output=Path("preprocessor.json"),
            manifest_output=Path("manifest.json"),
        )
        with patch.object(production_cli, "parse_args", return_value=args), patch.object(
            production_cli.pd, "read_csv", return_value=frame
        ), patch.object(
            production_cli, "build_production_artifacts", return_value={}
        ) as build:
            production_cli.main()
        build.assert_called_once()

    def test_native_model_round_trip_and_probability_bounds(self) -> None:
        features = synthetic_features()
        target = pd.Series(np.tile([0, 1], len(features) // 2), index=features.index)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._small_config(root)
            model = root / "model.cbm"
            preprocessor = root / "preprocessor.json"
            manifest = root / "manifest.json"
            build_production_artifacts(
                features, target, config, model, preprocessor, manifest
            )
            scorer = ProductionCatBoost.load(model, preprocessor, config, manifest)
            probability = scorer.predict_proba(features)
            self.assertEqual(list(scorer.score(features).columns), ["fraud_probability"])
            tampered = json.loads(config.read_text(encoding="utf-8"))
            tampered["models"]["catboost"]["selected_parameters"]["depth"] = 7
            mismatched = root / "mismatched.json"
            mismatched.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ from the frozen"):
                ProductionCatBoost.load(model, preprocessor, mismatched, manifest)
        self.assertEqual(probability.shape, (len(features),))
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))

    def test_artifact_manifest_rejects_swapped_model_and_modified_preprocessor(self) -> None:
        features = synthetic_features()
        target = pd.Series(np.tile([0, 1], len(features) // 2), index=features.index)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._small_config(root, depth=3)
            other_config = self._small_config(root, depth=5)
            model = root / "model.cbm"
            preprocessor = root / "preprocessor.json"
            manifest = root / "manifest.json"
            other_model = root / "other.cbm"
            other_preprocessor = root / "other-preprocessor.json"
            other_manifest = root / "other-manifest.json"
            build_production_artifacts(
                features, target, config, model, preprocessor, manifest
            )
            build_production_artifacts(
                features, target, other_config, other_model, other_preprocessor, other_manifest
            )
            ProductionCatBoost.load(model, preprocessor, config, manifest)

            original_model = model.read_bytes()
            shutil.copyfile(other_model, model)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                ProductionCatBoost.load(model, preprocessor, config, manifest)
            model.write_bytes(original_model)

            document = json.loads(preprocessor.read_text(encoding="utf-8"))
            document["numeric_fill"]["ea_score"] += 1
            preprocessor.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                ProductionCatBoost.load(model, preprocessor, config, manifest)

    def test_production_build_rejects_misaligned_label_index(self) -> None:
        features = synthetic_features(8)
        labels = pd.Series(np.tile([0, 1], 4), index=features.index)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (
                self._small_config(root),
                root / "model.cbm",
                root / "preprocessor.json",
            )
            with self.assertRaisesRegex(ValueError, "label.*index alignment"):
                build_production_artifacts(features, labels.iloc[::-1], *paths)
            with self.assertRaisesRegex(ValueError, "label.*index alignment"):
                build_production_artifacts(features, labels.iloc[:-1], *paths)

    def test_decision_policy_requires_explicit_valid_thresholds(self) -> None:
        signature = inspect.signature(apply_decision_policy)
        self.assertIs(signature.parameters["review_threshold"].default, inspect.Parameter.empty)
        self.assertIs(signature.parameters["decline_threshold"].default, inspect.Parameter.empty)
        decisions = apply_decision_policy(
            np.array([0.1, 0.5, 0.9]), review_threshold=0.3, decline_threshold=0.8
        )
        self.assertEqual(decisions.tolist(), ["approve", "manual_review", "decline"])
        with self.assertRaises(ValueError):
            apply_decision_policy(
                np.array([0.5]), review_threshold=0.8, decline_threshold=0.3
            )

    def test_production_build_does_not_split_or_evaluate_test(self) -> None:
        source = inspect.getsource(build_production_artifacts)
        self.assertNotIn("stratified_split", source)
        self.assertNotIn("probability_metrics", source)
        self.assertNotIn("predict_proba", source)

    def test_metadata_preserves_exact_frozen_gaps_and_caveats(self) -> None:
        metadata = build_production_metadata(FROZEN_CONFIG, MODEL_COMPARISON)
        comparison = pd.read_csv(MODEL_COMPARISON).set_index("model")
        for model, values in metadata["frozen_benchmark_metrics"].items():
            expected = comparison.loc[model, "validation_pr_auc"] - comparison.loc[model, "test_pr_auc"]
            self.assertAlmostEqual(values["validation_minus_test_pr_auc"], expected)
        self.assertIn("validation_reuse", metadata["evaluation_caveats"])
        self.assertFalse(metadata["threshold_policy"]["model_specific_threshold_optimization_performed"])
        self.assertFalse(metadata["training"]["production_artifact_built"])
        self.assertFalse(metadata["training"]["deployment_ready"])
        self.assertEqual(
            metadata["training"]["manifest_path"],
            "artifacts/catboost_artifact_manifest.json",
        )
        self.assertIn("cannot verify split membership", metadata["training"]["build_input_attestation"])


if __name__ == "__main__":
    unittest.main()
