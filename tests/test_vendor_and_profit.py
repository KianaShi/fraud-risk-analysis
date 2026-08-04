"""Tests for vendor cleaning and profit decisions."""

import unittest

import numpy as np
import pandas as pd

from fraud_detection.profit import apply_three_way_decision, compute_profit
from fraud_detection.vendor import VENDOR_FEATURES, clean_vendor_data


class VendorAndProfitTests(unittest.TestCase):
    """Verify missing-value preservation and business calculations."""

    def test_vendor_aliases_and_unknown_boolean(self) -> None:
        """Normalize compact column names without turning missing into false."""
        raw = pd.DataFrame({"is_fraud": [0], "IsConnected": [np.nan], "EAScore": ["4"]})
        cleaned = clean_vendor_data(raw)
        self.assertIn("is_connected", cleaned)
        self.assertIn("ea_score", cleaned)
        self.assertTrue(pd.isna(cleaned.loc[0, "is_connected"]))
        self.assertEqual(cleaned.loc[0, "ea_score"], 4)

    def test_vendor_definition_is_complete(self) -> None:
        """Include all fields listed under FraudKiller in the workbook definition."""
        self.assertTrue({"volume_score", "result_number", "email_days"}.issubset(VENDOR_FEATURES))

    def test_threshold_validation_and_profit(self) -> None:
        """Assign all three decisions and calculate the expected breakdown."""
        decisions = apply_three_way_decision(np.array([0.1, 0.5, 0.9]), 0.8, 0.2)
        self.assertEqual(decisions.tolist(), ["approve", "manual_review", "decline"])
        result = compute_profit(np.array([0, 1, 1]), decisions, vendor_called=False)
        self.assertEqual(result["n_approve"], 1)
        self.assertEqual(result["n_manual_review"], 1)
        self.assertEqual(result["n_decline"], 1)
        with self.assertRaises(ValueError):
            apply_three_way_decision(np.array([0.5]), 0.2, 0.8)


if __name__ == "__main__":
    unittest.main()
