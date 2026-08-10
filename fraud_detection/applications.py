"""Application-level cleaning and daily metric aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import coerce_binary_series, normalize_columns, require_columns

APPLICATION_COLUMNS = (
    "application_id", "product", "industry", "city", "state",
    "application_date", "final_decision", "is_fraud", "credit_score",
    "fraud_score", "first_transaction_date",
)
FINAL_DECISIONS = frozenset({"APPROVED", "DECLINED"})


def clean_applications(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean application records without mutating the input.

    Duplicate application IDs retain their latest record. Rows lacking an
    application date or fraud label and rows with impossible transaction dates
    are removed. Scores are converted to numeric values without imposing an
    undocumented range.
    """
    frame = normalize_columns(raw)
    require_columns(frame, APPLICATION_COLUMNS)
    frame = frame.loc[:, APPLICATION_COLUMNS].copy()
    frame["application_date"] = pd.to_datetime(frame["application_date"], errors="coerce")
    frame["first_transaction_date"] = pd.to_datetime(frame["first_transaction_date"], errors="coerce")
    frame["is_fraud"] = coerce_binary_series(frame["is_fraud"], "is_fraud")
    frame["final_decision"] = frame["final_decision"].astype("string").str.strip().str.upper()
    invalid_decision = frame["final_decision"].isna() | ~frame["final_decision"].isin(FINAL_DECISIONS)
    if invalid_decision.any():
        examples = (
            frame.loc[invalid_decision, "final_decision"]
            .astype("string")
            .fillna("<MISSING>")
            .unique()
            .tolist()[:5]
        )
        raise ValueError(
            "Column 'final_decision' contains missing or unrecognized values: "
            f"count={int(invalid_decision.sum())}, examples={examples}"
        )
    frame["is_approved"] = frame["final_decision"].eq("APPROVED").astype(int)
    frame = frame.dropna(subset=["application_id", "application_date", "is_fraud"])
    frame = frame.sort_values(["application_id", "application_date"]).drop_duplicates("application_id", keep="last")
    valid_transaction = frame["first_transaction_date"].isna() | (frame["first_transaction_date"] >= frame["application_date"])
    frame = frame.loc[valid_transaction].copy()
    frame["credit_score"] = pd.to_numeric(frame["credit_score"], errors="coerce")
    frame["fraud_score"] = pd.to_numeric(frame["fraud_score"], errors="coerce")
    frame["approved_fraud"] = (frame["is_approved"].eq(1) & frame["is_fraud"].eq(1)).astype(int)
    frame["date_gap_days"] = (frame["first_transaction_date"] - frame["application_date"]).dt.days
    frame["app_day"] = frame["application_date"].dt.normalize()
    return frame


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide series while returning NaN for zero denominators."""
    return numerator / denominator.replace(0, np.nan)


def build_daily_metrics(applications: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned records into one row per application date."""
    def flagged_mean(values: pd.Series, flag: str) -> float:
        return values.loc[applications.loc[values.index, flag].eq(1)].mean()

    daily = applications.groupby("app_day").agg(
        applications=("application_id", "count"),
        approvals=("is_approved", "sum"),
        frauds=("is_fraud", "sum"),
        approved_frauds=("approved_fraud", "sum"),
        avg_credit_score=("credit_score", "mean"),
        avg_fraud_score=("fraud_score", "mean"),
        avg_credit_score_on_approvals=("credit_score", lambda s: flagged_mean(s, "is_approved")),
        avg_fraud_score_on_approvals=("fraud_score", lambda s: flagged_mean(s, "is_approved")),
        avg_credit_score_on_frauds=("credit_score", lambda s: flagged_mean(s, "is_fraud")),
        avg_fraud_score_on_frauds=("fraud_score", lambda s: flagged_mean(s, "is_fraud")),
        avg_date_gap_days=("date_gap_days", "mean"),
    ).reset_index()
    daily["approval_rate"] = _safe_divide(daily["approvals"], daily["applications"])
    daily["application_fraud_rate"] = _safe_divide(daily["frauds"], daily["applications"])
    daily["approved_fraud_rate"] = _safe_divide(daily["approved_frauds"], daily["approvals"])
    return daily.sort_values("app_day").reset_index(drop=True)


def build_product_metrics(applications: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned records by application date and product."""
    result = applications.groupby(["app_day", "product"]).agg(
        applications=("application_id", "count"),
        approvals=("is_approved", "sum"),
        frauds=("is_fraud", "sum"),
        approved_frauds=("approved_fraud", "sum"),
    ).reset_index()
    result["approval_rate"] = _safe_divide(result["approvals"], result["applications"])
    result["application_fraud_rate"] = _safe_divide(result["frauds"], result["applications"])
    result["approved_fraud_rate"] = _safe_divide(result["approved_frauds"], result["approvals"])
    return result
