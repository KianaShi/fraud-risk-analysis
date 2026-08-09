"""Tests for the probability-only Fraud v1.0 production boundary."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

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

    def test_native_model_round_trip_and_probability_bounds(self) -> None:
        features = synthetic_features()
        target = pd.Series(np.tile([0, 1], len(features) // 2), index=features.index)
        base = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
        base["models"]["catboost"]["selected_parameters"] = {
            "iterations": 8,
            "depth": 3,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
            "random_strength": 0.0,
            "bagging_temperature": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "frozen.json"
            model = root / "model.cbm"
            preprocessor = root / "preprocessor.json"
            config.write_text(json.dumps(base), encoding="utf-8")
            build_production_artifacts(features, target, config, model, preprocessor)
            scorer = ProductionCatBoost.load(model, preprocessor, config)
            probability = scorer.predict_proba(features)
            self.assertEqual(list(scorer.score(features).columns), ["fraud_probability"])
            tampered = json.loads(config.read_text(encoding="utf-8"))
            tampered["models"]["catboost"]["selected_parameters"]["depth"] = 7
            mismatched = root / "mismatched.json"
            mismatched.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ from the frozen"):
                ProductionCatBoost.load(model, preprocessor, mismatched)
        self.assertEqual(probability.shape, (len(features),))
        self.assertTrue(np.all((probability >= 0) & (probability <= 1)))

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
        self.assertIn("cannot verify split membership", metadata["training"]["build_input_attestation"])


if __name__ == "__main__":
    unittest.main()
