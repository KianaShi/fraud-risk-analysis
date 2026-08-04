"""Leakage-safe model construction and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .common import require_columns
from .vendor import BOOLEAN_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, clean_vendor_data


@dataclass(frozen=True)
class FeatureGroups:
    """Numeric, boolean, and categorical feature names."""

    numeric: tuple[str, ...]
    boolean: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def all(self) -> list[str]:
        """Return all feature names in transformer order."""
        return [*self.numeric, *self.boolean, *self.categorical]

    def subset(self, selected: list[str]) -> "FeatureGroups":
        """Return groups restricted to selected columns."""
        keep = set(selected)
        return FeatureGroups(tuple(c for c in self.numeric if c in keep), tuple(c for c in self.boolean if c in keep), tuple(c for c in self.categorical if c in keep))


DEFAULT_FEATURES = FeatureGroups(
    numeric=(*NUMERIC_COLUMNS, "open_month", "open_day_of_week"),
    boolean=BOOLEAN_COLUMNS,
    categorical=CATEGORICAL_COLUMNS,
)


def prepare_model_data(raw: pd.DataFrame, groups: FeatureGroups = DEFAULT_FEATURES) -> tuple[pd.DataFrame, pd.Series, FeatureGroups]:
    """Clean a frame and return features, binary target, and available groups."""
    frame = clean_vendor_data(raw)
    available = groups.subset([column for column in groups.all if column in frame])
    require_columns(frame, available.all)
    keep = frame["is_fraud"].notna()
    target = frame.loc[keep, "is_fraud"].astype(int)
    if target.nunique() != 2:
        raise ValueError("Target must contain both fraud and non-fraud records.")
    return frame.loc[keep, available.all].copy(), target, available


def build_preprocessor(groups: FeatureGroups) -> ColumnTransformer:
    """Build preprocessing fitted only on the training data."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    boolean = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, list(groups.numeric)), ("boolean", boolean, list(groups.boolean)), ("categorical", categorical, list(groups.categorical))])


def candidate_models(random_state: int = 42) -> dict[str, object]:
    """Return the original three model families with imbalance handling."""
    return {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=10, random_state=random_state, n_jobs=-1),
    }


def probability_metrics(y_true: pd.Series, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Calculate probability and threshold-based binary metrics."""
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {"roc_auc": roc_auc_score(y_true, probability), "pr_auc": average_precision_score(y_true, probability), "accuracy": accuracy_score(y_true, prediction), "balanced_accuracy": balanced_accuracy_score(y_true, prediction), "precision": precision_score(y_true, prediction, zero_division=0), "recall": recall_score(y_true, prediction, zero_division=0)}


def compare_models(raw: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, Pipeline]:
    """Select a model by training CV PR-AUC, then score one untouched holdout."""
    features, target, groups = prepare_model_data(raw)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=random_state)
    folds = min(5, int(y_train.value_counts().min()))
    if folds < 2:
        raise ValueError("At least two training observations per class are required.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    rows, pipelines = [], {}
    for name, estimator in candidate_models(random_state).items():
        pipeline = Pipeline([("preprocess", build_preprocessor(groups)), ("classifier", estimator)])
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "balanced_accuracy": "balanced_accuracy"}, n_jobs=-1)
        rows.append({"model": name, "cv_roc_auc": scores["test_roc_auc"].mean(), "cv_pr_auc": scores["test_pr_auc"].mean(), "cv_balanced_accuracy": scores["test_balanced_accuracy"].mean()})
        pipelines[name] = pipeline
    selected = max(rows, key=lambda row: row["cv_pr_auc"])["model"]
    best = pipelines[selected].fit(X_train, y_train)
    holdout = probability_metrics(y_test, best.predict_proba(X_test)[:, 1])
    for row in rows:
        row["selected"] = row["model"] == selected
        row.update({f"holdout_{key}": value if row["selected"] else np.nan for key, value in holdout.items()})
    return pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True), best
