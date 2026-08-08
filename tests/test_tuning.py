"""Synthetic tests for Train-only Optuna hyperparameter optimization."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fraud_detection.modeling import DEFAULT_FEATURES, FeatureGroups
from fraud_detection.tuning import (
    CV_FOLDS,
    CV_SEED,
    OPTIMIZATION_SEED,
    PRIMARY_FEATURES,
    PRIMARY_OBJECTIVE,
    FrozenConfiguration,
    _final_test_evaluation,
    cross_validated_pr_auc,
    freeze_configuration,
    optimize_model,
    suggest_catboost_parameters,
    suggest_xgboost_parameters,
)
import fraud_detection.tuning as tuning


class MidpointTrial:
    """Small deterministic stand-in for search-space boundary tests."""

    def suggest_int(self, name, low, high, step=1):
        del name
        return low + ((high - low) // (2 * step)) * step

    def suggest_float(self, name, low, high, log=False):
        del name
        return float(np.sqrt(low * high) if log else (low + high) / 2)


class TuningTests(unittest.TestCase):
    """Verify leakage barriers, search spaces, selection, and lightweight studies."""

    def setUp(self) -> None:
        size = 60
        self.groups = FeatureGroups(("score",), ("flag",), ("country",))
        self.X = pd.DataFrame(
            {
                "score": np.linspace(0, 1, size),
                "flag": np.arange(size) % 2,
                "country": np.where(np.arange(size) % 3 == 0, "US", "CA"),
            },
            index=np.arange(1000, 1000 + size),
        )
        self.y = pd.Series(np.arange(size) % 2, index=self.X.index)

    def test_primary_objective_and_frozen_feature_sets_are_unchanged(self) -> None:
        self.assertEqual(PRIMARY_OBJECTIVE, "mean_cv_pr_auc")
        self.assertEqual(PRIMARY_FEATURES, tuple(DEFAULT_FEATURES.all))
        self.assertEqual(len(PRIMARY_FEATURES), 14)
        compact = pd.read_csv(
            Path(__file__).resolve().parents[1] / "docs" / "results" / "final_feature_confirmation.csv"
        )
        top_five = compact.loc[compact["feature_configuration"].eq("top_5"), "feature_names"]
        self.assertEqual(
            set(top_five),
            {"ea_score|identity_rank|email_days|device_browser_type|type"},
        )

    def test_search_spaces_generate_valid_conservative_parameters(self) -> None:
        xgb = suggest_xgboost_parameters(MidpointTrial())
        cat = suggest_catboost_parameters(MidpointTrial())
        self.assertTrue(200 <= xgb["n_estimators"] <= 1000)
        self.assertTrue(0.01 <= xgb["learning_rate"] <= 0.20)
        self.assertTrue(3 <= xgb["max_depth"] <= 8)
        self.assertTrue(1e-4 <= xgb["reg_alpha"] <= 10)
        self.assertTrue(200 <= cat["iterations"] <= 1000)
        self.assertTrue(0.01 <= cat["learning_rate"] <= 0.20)
        self.assertTrue(4 <= cat["depth"] <= 8)
        self.assertTrue(1 <= cat["l2_leaf_reg"] <= 20)

    def test_cv_uses_five_train_only_stratified_folds(self) -> None:
        fitted_indices = []
        validation_indices = []

        class RecordingModel:
            def fit(self, X, y):
                del y
                fitted_indices.append(set(X.index))
                return self

            def predict_proba(self, X):
                validation_indices.append(set(X.index))
                probability = np.where(X["flag"].to_numpy() == 1, 0.6, 0.4)
                return np.column_stack([1 - probability, probability])

        with patch("fraud_detection.tuning.build_model", return_value=RecordingModel()):
            result = cross_validated_pr_auc(
                "xgboost", self.X, self.y, self.groups, "default"
            )
        self.assertEqual(len(fitted_indices), CV_FOLDS)
        self.assertEqual(len(result.fold_scores), CV_FOLDS)
        allowed = set(self.X.index)
        for fitted, validated in zip(fitted_indices, validation_indices):
            self.assertTrue(fitted <= allowed)
            self.assertTrue(validated <= allowed)
            self.assertFalse(fitted & validated)

    @unittest.skipIf(tuning.optuna is None, "Optuna is not installed")
    def test_seeded_tpe_sampler_is_reproducible(self) -> None:
        def sequence():
            sampler = tuning.optuna.samplers.TPESampler(seed=OPTIMIZATION_SEED)
            study = tuning.optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=3)
            return [trial.params for trial in study.trials]

        self.assertEqual(sequence(), sequence())
        self.assertEqual(CV_SEED, 42)

    @unittest.skipIf(tuning.optuna is None, "Optuna is not installed")
    def test_xgboost_tuning_runs_one_small_trial(self) -> None:
        result = optimize_model(
            "xgboost", self.X, self.y, self.groups, n_trials=1, cv_folds=2
        )
        self.assertEqual(len(result.study.trials), 1)
        self.assertGreater(result.cv_result.mean_pr_auc, 0)

    @unittest.skipIf(tuning.optuna is None, "Optuna is not installed")
    def test_catboost_tuning_runs_one_small_trial(self) -> None:
        result = optimize_model(
            "catboost", self.X, self.y, self.groups, n_trials=1, cv_folds=2
        )
        self.assertEqual(len(result.study.trials), 1)
        self.assertGreater(result.cv_result.mean_pr_auc, 0)

    def test_configuration_selection_uses_validation_pr_auc(self) -> None:
        cv = tuning.CVResult(0.8, 0.01, (0.79, 0.81))
        optimization = tuning.OptimizationResult(
            "xgboost", {"max_depth": 4}, cv, 1.0, study=None
        )
        selected_tuned = freeze_configuration(
            "xgboost",
            {"pr_auc": 0.90},
            {"pr_auc": 0.901},
            cv,
            optimization,
            1,
            42,
            42,
            5,
        )
        selected_default = freeze_configuration(
            "xgboost",
            {"pr_auc": 0.90},
            {"pr_auc": 0.89},
            cv,
            optimization,
            1,
            42,
            42,
            5,
        )
        self.assertEqual(selected_tuned.selected_configuration, "tuned")
        self.assertEqual(selected_default.selected_configuration, "default")

    def test_test_evaluation_requires_a_frozen_configuration(self) -> None:
        with self.assertRaisesRegex(TypeError, "frozen"):
            _final_test_evaluation(
                {"model": "xgboost"},
                self.X.iloc[:40],
                self.y.iloc[:40],
                self.X.iloc[40:],
                self.y.iloc[40:],
                self.groups,
            )
        self.assertTrue(FrozenConfiguration.__dataclass_params__.frozen)


if __name__ == "__main__":
    unittest.main()
