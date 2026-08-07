"""Synthetic tests for feature audit, missingness, importance, and ablation."""

import unittest

import numpy as np
import pandas as pd

from fraud_detection.feature_analysis import (
    FeatureConfiguration,
    augment_missingness_features,
    build_feature_audit,
    build_missingness_analysis,
    evaluate_configurations,
    feature_groups_for,
    permutation_importance_table,
    select_missing_indicators,
)
from fraud_detection.modeling import FeatureGroups, candidate_models, stratified_split


class FeatureAnalysisTests(unittest.TestCase):
    """Verify analysis helpers without requiring the private workbook."""

    def setUp(self) -> None:
        size = 80
        self.groups = FeatureGroups(("score",), ("flag",), ("country",))
        self.X = pd.DataFrame(
            {
                "score": np.where(np.arange(size) % 10 == 0, np.nan, np.linspace(0, 1, size)),
                "flag": np.arange(size) % 2,
                "country": np.where(np.arange(size) % 3 == 0, "US", "CA"),
            }
        )
        self.y = pd.Series(np.arange(size) % 2)
        self.partitions = stratified_split(self.y)

    def test_feature_audit_has_required_columns_and_uses_legitimate_features(self) -> None:
        audit = build_feature_audit(self.X.loc[self.partitions.train], self.y.loc[self.partitions.train], self.groups)
        required = {
            "feature", "feature_type", "source", "missing_count", "missing_rate", "unique_count",
            "constant", "near_constant", "mean", "median", "std", "fraud_mean", "non_fraud_mean",
            "value_counts_json", "fraud_rate_by_value_json",
        }
        self.assertTrue(required.issubset(audit.columns))
        self.assertEqual(set(audit["feature"]), set(self.groups.all))
        self.assertNotIn("is_fraud", audit["feature"].tolist())

    def test_missingness_analysis_and_indicators(self) -> None:
        train = self.X.loc[self.partitions.train]
        analysis = build_missingness_analysis(train, self.y.loc[self.partitions.train])
        self.assertEqual(analysis["feature"].tolist(), ["score"])
        indicators = select_missing_indicators(train)
        self.assertEqual(indicators, ("score",))
        augmented, groups = augment_missingness_features(self.X, self.groups, indicators, False)
        self.assertIn("score_missing", augmented)
        self.assertIn("score_missing", groups.boolean)
        self.assertTrue(augmented.loc[self.X["score"].isna(), "score_missing"].eq(1).all())

    def test_feature_subset_rejects_excluded_or_unknown_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown or excluded"):
            feature_groups_for(("score", "open_date"), self.groups)

    def test_permutation_importance_executes_at_original_feature_level(self) -> None:
        models = candidate_models(self.groups)
        for model in models.values():
            model.fit(self.X.loc[self.partitions.train], self.y.loc[self.partitions.train])
        importance = permutation_importance_table(
            models,
            self.X.loc[self.partitions.validation],
            self.y.loc[self.partitions.validation],
            n_repeats=2,
        )
        self.assertEqual(set(importance["feature"]), set(self.groups.all))
        self.assertEqual(set(importance["model"]), {"xgboost", "catboost"})

    def test_ablation_uses_fixed_train_validation_partitions(self) -> None:
        configurations = [
            FeatureConfiguration("numeric_only", ("score",)),
            FeatureConfiguration("combined", tuple(self.groups.all)),
        ]
        result = evaluate_configurations(
            self.X, self.y, self.groups, self.partitions, configurations
        )
        self.assertEqual(set(result["feature_configuration"]), {"numeric_only", "combined"})
        self.assertEqual(set(result["model"]), {"xgboost", "catboost"})
        self.assertTrue(result.columns.str.startswith("test_").sum() == 0)


if __name__ == "__main__":
    unittest.main()
