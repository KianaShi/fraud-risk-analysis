"""Tests for leakage-safe preprocessing."""

import unittest

import numpy as np
import pandas as pd

from fraud_detection.modeling import FeatureGroups, build_preprocessor, candidate_models


class ModelingTests(unittest.TestCase):
    """Verify model families and training-time imputation."""

    def test_original_model_families_are_present(self) -> None:
        """Retain logistic regression, decision tree, and random forest."""
        self.assertEqual(set(candidate_models()), {"logistic_regression", "decision_tree", "random_forest"})

    def test_preprocessor_handles_missing_and_unknown_categories(self) -> None:
        """Fit imputers on training data and accept unseen categories at transform."""
        groups = FeatureGroups(("score",), ("flag",), ("country",))
        train = pd.DataFrame({"score": [1.0, np.nan], "flag": [1.0, np.nan], "country": ["US", "CA"]})
        test = pd.DataFrame({"score": [np.nan], "flag": [np.nan], "country": ["GB"]})
        preprocessor = build_preprocessor(groups).fit(train)
        self.assertEqual(preprocessor.transform(test).shape[0], 1)


if __name__ == "__main__":
    unittest.main()
