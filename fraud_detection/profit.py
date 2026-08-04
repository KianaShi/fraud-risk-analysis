"""Three-way fraud decisions and business profit calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .modeling import DEFAULT_FEATURES, build_preprocessor, prepare_model_data, probability_metrics
from .vendor import VENDOR_FEATURES


@dataclass(frozen=True)
class ProfitAssumptions:
    """Business inputs used by the expected-value calculation."""

    monthly_revenue: float = 40.0
    months: int = 12
    fraud_loss: float = 500.0
    manual_review_cost: float = 50.0
    vendor_call_cost: float = 0.50
    manual_review_approval_rate: float = 0.30


def apply_three_way_decision(probability: np.ndarray, decline_threshold: float, approve_threshold: float) -> np.ndarray:
    """Assign decline, manual review, or approve from fraud probabilities."""
    if not 0 <= approve_threshold < decline_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= approve < decline <= 1.")
    return np.where(probability >= decline_threshold, "decline", np.where(probability >= approve_threshold, "manual_review", "approve"))


def compute_profit(y_true: np.ndarray, decisions: np.ndarray, vendor_called: bool, assumptions: ProfitAssumptions = ProfitAssumptions()) -> dict[str, float | int]:
    """Calculate expected profit and its revenue/cost breakdown."""
    target, decision = np.asarray(y_true), np.asarray(decisions)
    if len(target) != len(decision):
        raise ValueError("Targets and decisions must have equal length.")
    fraud, clean = target == 1, target == 0
    approve, decline, review = decision == "approve", decision == "decline", decision == "manual_review"
    revenue = assumptions.monthly_revenue * assumptions.months * np.sum(approve & clean)
    revenue += assumptions.monthly_revenue * assumptions.months * assumptions.manual_review_approval_rate * np.sum(review & clean)
    fraud_loss = assumptions.fraud_loss * np.sum(approve & fraud)
    fraud_loss += assumptions.fraud_loss * assumptions.manual_review_approval_rate * np.sum(review & fraud)
    review_cost = assumptions.manual_review_cost * np.sum(review)
    vendor_cost = assumptions.vendor_call_cost * len(target) if vendor_called else 0.0
    return {"profit": float(revenue - fraud_loss - review_cost - vendor_cost), "revenue": float(revenue), "fraud_loss": float(fraud_loss), "manual_review_cost": float(review_cost), "vendor_cost": float(vendor_cost), "n_approve": int(approve.sum()), "n_manual_review": int(review.sum()), "n_decline": int(decline.sum())}


def compare_vendor_scenarios(raw: pd.DataFrame, decline_threshold: float = 0.8, approve_threshold: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare identical decision trees without and with all vendor fields."""
    features, target, groups = prepare_model_data(raw, DEFAULT_FEATURES)
    train_index, test_index = train_test_split(features.index, test_size=0.2, stratify=target, random_state=random_state)
    scenarios = {
        "tree_without_vendor": [column for column in groups.all if column not in VENDOR_FEATURES],
        "tree_with_vendor": groups.all,
    }
    rows, scored = [], None
    for name, columns in scenarios.items():
        scenario_groups = groups.subset(columns)
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(scenario_groups)),
            ("classifier", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=random_state)),
        ])
        pipeline.fit(features.loc[train_index, columns], target.loc[train_index])
        probability = pipeline.predict_proba(features.loc[test_index, columns])[:, 1]
        decisions = apply_three_way_decision(probability, decline_threshold, approve_threshold)
        metrics = probability_metrics(target.loc[test_index], probability)
        profit = compute_profit(target.loc[test_index].to_numpy(), decisions, vendor_called=name == "tree_with_vendor")
        rows.append({"scenario": name, **metrics, **profit, "decline_threshold": decline_threshold, "approve_threshold": approve_threshold})
        if name == "tree_with_vendor":
            scored = features.loc[test_index].copy()
            scored["y_true"] = target.loc[test_index]
            scored["fraud_probability"] = probability
            scored["decision"] = decisions
    if scored is None:
        raise RuntimeError("Vendor scenario was not evaluated.")
    return pd.DataFrame(rows), scored.reset_index(drop=True)
