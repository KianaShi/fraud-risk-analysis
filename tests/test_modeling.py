"""Tests for stratified Task 2 splitting and the modern model benchmark."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fraud_detection.modeling import (
    DEFAULT_FEATURES,
    CatBoostPreprocessor,
    FeatureGroups,
    build_preprocessor,
    candidate_models,
    compare_models,
    prepare_model_data,
    probability_metrics,
    split_audit,
    stratified_split,
)


class ModelingTests(unittest.TestCase):
    """Verify stratification, candidates, metrics, and leakage-safe preprocessing."""

    def setUp(self) -> None:
        self.groups = FeatureGroups(("score",), ("flag",), ("country",))

    @staticmethod
    def _full_raw(size: int = 100) -> pd.DataFrame:
        values = np.arange(size)
        return pd.DataFrame({
            "id": [f"record-{value:03d}" for value in values],
            "is_fraud": values % 2,
            "ea_score": values.astype(float),
            "identity_rank": values.astype(float),
            "reputation_level": values % 3,
            "volume_score": values % 7,
            "result_number": values % 5,
            "email_days": values * 2,
            "is_valid": values % 2,
            "is_connected": values % 2,
            "personal_device": values % 2,
            "receiving_mail": values % 2,
            "area_code": np.where(values % 2, "212", "415"),
            "device_browser_type": np.where(values % 2, "CHROME", "FIREFOX"),
            "ip_address_loc_country": np.where(values % 2, "US", "CA"),
            "type": np.where(values % 2, "M", "L"),
        })

    def test_active_model_families_are_xgboost_and_catboost(self) -> None:
        """Expose only the two intended gradient-boosting candidates."""
        self.assertEqual(set(candidate_models(self.groups)), {"xgboost", "catboost"})

    def test_stratified_split_is_reproducible_and_balanced(self) -> None:
        """Preserve prevalence across disjoint 70/15/15 partitions."""
        target = pd.Series([0] * 60 + [1] * 40, index=np.arange(100, 200))
        stable_ids = pd.Series([f"id-{value}" for value in target.index], index=target.index)
        first = stratified_split(target, random_state=42, stable_ids=stable_ids)
        second = stratified_split(target, random_state=42, stable_ids=stable_ids)
        self.assertEqual(first.train.tolist(), second.train.tolist())
        self.assertEqual((len(first.train), len(first.validation), len(first.test)), (70, 15, 15))
        self.assertFalse(set(first.train) & set(first.validation))
        self.assertFalse(set(first.train) & set(first.test))
        self.assertFalse(set(first.validation) & set(first.test))
        for index in (first.train, first.validation, first.test):
            self.assertAlmostEqual(target.loc[index].mean(), target.mean(), delta=0.04)

    def test_split_does_not_depend_on_open_date_order(self) -> None:
        """Task 2 partitions are label-stratified rather than chronological."""
        target = pd.Series([0, 1] * 50)
        stable_ids = pd.Series([f"id-{value}" for value in target.index], index=target.index)
        partitions = stratified_split(target, random_state=42, stable_ids=stable_ids)
        self.assertNotEqual(set(partitions.train), set(range(70)))
        self.assertNotEqual(set(partitions.validation), set(range(70, 85)))
        self.assertNotEqual(set(partitions.test), set(range(85, 100)))
        self.assertAlmostEqual(target.loc[partitions.test].mean(), 0.5, delta=0.04)

    def test_split_audit_reports_counts_and_prevalence(self) -> None:
        """Expose the positive-class baseline for every partition."""
        target = pd.Series([0, 1] * 50)
        stable_ids = pd.Series([f"id-{value}" for value in target.index], index=target.index)
        audit = split_audit(target, stratified_split(target, stable_ids=stable_ids))
        self.assertEqual(audit["split"].tolist(), ["train", "validation", "test"])
        self.assertEqual(set(audit.columns), {"split", "n", "fraud", "non_fraud", "fraud_rate"})
        self.assertTrue(np.allclose(audit["fraud_rate"], 0.5, atol=0.04))

    def test_primary_features_exclude_ambiguous_open_date_fields(self) -> None:
        """Do not use raw or derived open-date fields in the primary benchmark."""
        self.assertFalse({"open_date", "open_year", "open_month", "open_day_of_week"} & set(DEFAULT_FEATURES.all))

    def test_requested_feature_schema_is_strict_by_default(self) -> None:
        exact = pd.DataFrame({
            "is_fraud": [0, 1] * 4,
            "score": np.arange(8),
            "flag": [0, 1] * 4,
            "country": ["US", "CA"] * 4,
        })
        features, _, groups = prepare_model_data(exact, self.groups)
        self.assertEqual(groups.all, self.groups.all)
        self.assertEqual(list(features.columns), self.groups.all)
        with self.assertRaisesRegex(ValueError, "Missing required columns.*country"):
            prepare_model_data(exact.drop(columns="country"), self.groups)
        partial, _, partial_groups = prepare_model_data(
            exact.drop(columns="country"), self.groups, allow_partial=True
        )
        self.assertEqual(partial_groups.all, ["score", "flag"])
        self.assertEqual(list(partial.columns), ["score", "flag"])

    def test_missing_one_of_default_fourteen_features_is_rejected(self) -> None:
        raw = self._full_raw()
        prepare_model_data(raw)
        with self.assertRaisesRegex(ValueError, "Missing required columns.*email_days"):
            prepare_model_data(raw.drop(columns="email_days"))

    def test_split_membership_is_invariant_to_input_row_order(self) -> None:
        first = self._full_raw()
        reordered = first.sample(frac=1, random_state=7).reset_index(drop=True)
        memberships = []
        for frame in (first, reordered):
            target = frame["is_fraud"]
            stable_ids = frame["id"]
            partitions = stratified_split(target, random_state=42, stable_ids=stable_ids)
            memberships.append(tuple(
                set(stable_ids.loc[index])
                for index in (partitions.train, partitions.validation, partitions.test)
            ))
        self.assertEqual(memberships[0], memberships[1])

    def test_split_identity_is_required_unique_nonmissing_and_not_predictive(self) -> None:
        raw = self._full_raw()
        features, target, _ = prepare_model_data(raw)
        self.assertNotIn("id", features.columns)
        with self.assertRaisesRegex(ValueError, "stable_ids.*required"):
            stratified_split(target)
        for invalid_ids in (
            raw["id"].mask(raw.index == 0),
            raw["id"].mask(raw.index == 0, "   "),
            raw["id"].mask(raw.index == 0, "NULL"),
            raw["id"].mask(raw.index == 1, raw.loc[0, "id"]),
        ):
            with self.assertRaisesRegex(ValueError, "stable_ids"):
                stratified_split(target, stable_ids=invalid_ids)

    def test_preprocessors_learn_imputation_from_training_only(self) -> None:
        """Validation and test values must not influence either preprocessing path."""
        train = pd.DataFrame(
            {"score": [1.0, np.nan, 3.0], "flag": [1.0, np.nan, 0.0], "country": ["US", "US", None]}
        )
        unseen = pd.DataFrame({"score": [999.0], "flag": [1.0], "country": ["GB"]})
        xgb_preprocessor = build_preprocessor(self.groups).fit(train)
        self.assertEqual(xgb_preprocessor.named_transformers_["numeric"].named_steps["imputer"].statistics_[0], 2.0)
        self.assertEqual(xgb_preprocessor.transform(unseen).shape[0], 1)

        cat_preprocessor = CatBoostPreprocessor(self.groups).fit(train)
        self.assertEqual(cat_preprocessor.numeric_fill_["score"], 2.0)
        self.assertEqual(cat_preprocessor.transform(unseen).loc[0, "country"], "GB")

    def test_validation_and_test_rows_are_never_passed_to_fit(self) -> None:
        """Fit both candidates on Train only, leaving Validation and Test untouched."""
        size = 100
        raw = self._full_raw(size)
        _, target, _ = prepare_model_data(raw)
        partitions = stratified_split(target, random_state=42, stable_ids=raw.loc[target.index, "id"])

        class RecordingModel:
            def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecordingModel":
                self.fit_index = pd.Index(X.index)
                return self

            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
                probability = np.full(len(X), 0.5)
                return np.column_stack([1 - probability, probability])

        models = {"xgboost": RecordingModel(), "catboost": RecordingModel()}
        with patch("fraud_detection.modeling.candidate_models", return_value=models):
            compare_models(raw, random_state=42)
        for model in models.values():
            self.assertEqual(set(model.fit_index), set(partitions.train))
            self.assertFalse(set(model.fit_index) & set(partitions.validation))
            self.assertFalse(set(model.fit_index) & set(partitions.test))

    def test_both_models_return_probabilities(self) -> None:
        """Ensure both benchmark paths implement binary predict_proba."""
        size = 40
        frame = pd.DataFrame(
            {
                "score": np.linspace(0, 1, size),
                "flag": np.arange(size) % 2,
                "country": np.where(np.arange(size) % 3 == 0, "US", "CA"),
            }
        )
        target = pd.Series(np.arange(size) % 2)
        for model in candidate_models(self.groups).values():
            model.fit(frame, target)
            self.assertEqual(model.predict_proba(frame.iloc[:4]).shape, (4, 2))

    def test_probability_metrics_are_complete(self) -> None:
        """Report all required probability and threshold metrics."""
        metrics = probability_metrics(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
        self.assertEqual(
            set(metrics),
            {"pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"},
        )

    def test_full_benchmark_reports_both_models_and_prevalence(self) -> None:
        """Exercise train-only fitting and common validation/test reporting."""
        size = 80
        raw = self._full_raw(size)
        raw["open_date"] = pd.date_range("2024-01-01", periods=size, freq="D")
        comparison, best, audit = compare_models(raw)
        self.assertEqual(set(comparison["model"]), {"xgboost", "catboost"})
        self.assertTrue(
            {
                "validation_pr_auc", "validation_roc_auc", "validation_positive_prevalence",
                "test_pr_auc", "test_roc_auc", "test_f1", "test_positive_prevalence",
            }.issubset(comparison.columns)
        )
        self.assertEqual(audit["n"].sum(), size)
        features, _, _ = prepare_model_data(raw)
        self.assertEqual(len(best.predict_proba(features.tail(2))), 2)


if __name__ == "__main__":
    unittest.main()
