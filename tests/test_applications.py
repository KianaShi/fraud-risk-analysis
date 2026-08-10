"""Tests for application cleaning and aggregation."""

import unittest

import pandas as pd

from fraud_detection.applications import build_daily_metrics, clean_applications


class ApplicationMetricsTests(unittest.TestCase):
    """Verify score validation and fraud-rate definitions."""

    def test_credit_scale_and_approved_fraud_rate(self) -> None:
        """Keep numeric scores and use approved frauds in the approval rate."""
        raw = pd.DataFrame({
            "application_id": [1, 2, 3], "product": ["A"] * 3,
            "industry": ["I"] * 3, "city": ["C"] * 3, "state": ["S"] * 3,
            "application_date": ["2019-01-01"] * 3,
            "final_decision": ["APPROVED", "DECLINED", "APPROVED"],
            "is_fraud": [1, 1, 0], "credit_score": [80, 70, 101],
            "fraud_score": [0.9, 0.8, 0.1],
            "first_transaction_date": ["2019-01-02", None, "2019-01-03"],
        })
        cleaned = clean_applications(raw)
        metrics = build_daily_metrics(cleaned).iloc[0]
        self.assertEqual(cleaned["credit_score"].notna().sum(), 3)
        self.assertAlmostEqual(metrics["application_fraud_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["approved_fraud_rate"], 1 / 2)

    def test_impossible_transaction_is_removed(self) -> None:
        """Remove a first transaction dated before the application."""
        raw = pd.DataFrame({
            "application_id": [1], "product": ["A"], "industry": ["I"],
            "city": ["C"], "state": ["S"], "application_date": ["2019-01-02"],
            "final_decision": ["APPROVED"], "is_fraud": [0], "credit_score": [80],
            "fraud_score": [0.1], "first_transaction_date": ["2019-01-01"],
        })
        self.assertTrue(clean_applications(raw).empty)

    @staticmethod
    def _single_application(final_decision) -> pd.DataFrame:
        return pd.DataFrame({
            "application_id": [1], "product": ["A"], "industry": ["I"],
            "city": ["C"], "state": ["S"], "application_date": ["2019-01-02"],
            "final_decision": [final_decision], "is_fraud": [0], "credit_score": [80],
            "fraud_score": [0.1], "first_transaction_date": ["2019-01-03"],
        })

    def test_missing_final_decision_raises_controlled_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "final_decision.*count=1"):
            clean_applications(self._single_application(None))

    def test_unrecognized_final_decision_raises_controlled_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "final_decision.*count=1"):
            clean_applications(self._single_application("PENDING_UNKNOWN"))


if __name__ == "__main__":
    unittest.main()
