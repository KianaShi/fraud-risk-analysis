"""Stratified model construction and evaluation for Task 2 fraud risk."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .common import normalize_columns, require_columns
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
        return FeatureGroups(
            tuple(c for c in self.numeric if c in keep),
            tuple(c for c in self.boolean if c in keep),
            tuple(c for c in self.categorical if c in keep),
        )


@dataclass(frozen=True)
class BenchmarkPartitions:
    """Row indices for stratified train, validation, and test partitions."""

    train: pd.Index
    validation: pd.Index
    test: pd.Index


DEFAULT_FEATURES = FeatureGroups(
    numeric=NUMERIC_COLUMNS,
    boolean=BOOLEAN_COLUMNS,
    categorical=CATEGORICAL_COLUMNS,
)
SPLIT_ID_COLUMN = "id"
SPLIT_PROTOCOL_VERSION = "stable-id-stratified-70-15-15-v1"


def prepare_model_data(
    raw: pd.DataFrame,
    groups: FeatureGroups = DEFAULT_FEATURES,
    *,
    allow_partial: bool = False,
) -> tuple[pd.DataFrame, pd.Series, FeatureGroups]:
    """Clean a frame and return the exact requested feature schema by default."""
    frame = clean_vendor_data(raw)
    if allow_partial:
        available = groups.subset([column for column in groups.all if column in frame])
    else:
        require_columns(frame, groups.all)
        available = groups
    keep = frame["is_fraud"].notna()
    target = frame.loc[keep, "is_fraud"].astype(int)
    if target.nunique() != 2:
        raise ValueError("Target must contain both fraud and non-fraud records.")
    return frame.loc[keep, available.all].copy(), target, available


def extract_stable_ids(raw: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Extract the documented source record ID without adding it to predictors."""
    normalized = normalize_columns(raw)
    require_columns(normalized, [SPLIT_ID_COLUMN])
    return normalized.loc[index, SPLIT_ID_COLUMN].copy()


def _validated_stable_ids(target: pd.Series, stable_ids: pd.Series | None) -> pd.Series:
    if stable_ids is None:
        raise ValueError("stable_ids are required for reproducible split membership.")
    if not isinstance(stable_ids, pd.Series) or not stable_ids.index.equals(target.index):
        raise ValueError("stable_ids must be a pandas Series aligned exactly to target.index.")
    if target.index.has_duplicates:
        raise ValueError("Target row index must be unique before splitting.")
    if stable_ids.isna().any():
        raise ValueError(f"stable_ids contain {int(stable_ids.isna().sum())} missing values.")
    canonical = stable_ids.astype("string").str.strip()
    missing_tokens = canonical.eq("") | canonical.str.lower().eq("null")
    if missing_tokens.any():
        raise ValueError(
            f"stable_ids contain {int(missing_tokens.sum())} blank/null values."
        )
    if canonical.duplicated().any():
        raise ValueError(
            f"stable_ids contain {int(canonical.duplicated(keep=False).sum())} duplicate rows."
        )
    return canonical


def stratified_split(
    target: pd.Series,
    random_state: int = 42,
    *,
    stable_ids: pd.Series | None = None,
) -> BenchmarkPartitions:
    """Create row-order-invariant 70/15/15 partitions using stable record IDs."""
    if len(target) < 7 or target.value_counts().min() < 4:
        raise ValueError("At least four observations per class are required for a 70/15/15 split.")
    canonical_ids = _validated_stable_ids(target, stable_ids)
    ordered_index = canonical_ids.sort_values(kind="mergesort").index
    train_index, holdout_index = train_test_split(
        ordered_index,
        test_size=0.30,
        stratify=target.loc[ordered_index],
        random_state=random_state,
    )
    validation_index, test_index = train_test_split(
        holdout_index,
        test_size=0.50,
        stratify=target.loc[holdout_index],
        random_state=random_state,
    )
    return BenchmarkPartitions(
        train=pd.Index(train_index),
        validation=pd.Index(validation_index),
        test=pd.Index(test_index),
    )


def _membership_sha256(stable_ids: pd.Series) -> str:
    values = sorted(stable_ids.astype("string").str.strip().tolist())
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_id_membership_sha256(stable_ids: pd.Series) -> str:
    """Validate and hash a set of stable record identifiers."""
    placeholder_target = pd.Series(0, index=stable_ids.index)
    canonical = _validated_stable_ids(placeholder_target, stable_ids)
    return _membership_sha256(canonical)


def split_membership_fingerprints(
    stable_ids: pd.Series, partitions: BenchmarkPartitions
) -> dict[str, str]:
    """Hash sorted stable-ID sets for the dataset and every frozen partition."""
    return {
        "split_protocol_version": SPLIT_PROTOCOL_VERSION,
        "identity_column": SPLIT_ID_COLUMN,
        "dataset_membership_sha256": _membership_sha256(stable_ids),
        "train_membership_sha256": _membership_sha256(stable_ids.loc[partitions.train]),
        "validation_membership_sha256": _membership_sha256(
            stable_ids.loc[partitions.validation]
        ),
        "test_membership_sha256": _membership_sha256(stable_ids.loc[partitions.test]),
    }


def split_audit(target: pd.Series, partitions: BenchmarkPartitions) -> pd.DataFrame:
    """Summarize sample size, class counts, and PR-AUC prevalence baseline."""
    rows = []
    for name, index in (
        ("train", partitions.train),
        ("validation", partitions.validation),
        ("test", partitions.test),
    ):
        labels = target.loc[index]
        fraud = int(labels.sum())
        rows.append(
            {
                "split": name,
                "n": len(labels),
                "fraud": fraud,
                "non_fraud": int(len(labels) - fraud),
                "fraud_rate": float(labels.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_preprocessor(groups: FeatureGroups) -> ColumnTransformer:
    """Build the one-hot preprocessing path used by XGBoost."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    boolean = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(groups.numeric)),
            ("boolean", boolean, list(groups.boolean)),
            ("categorical", categorical, list(groups.categorical)),
        ]
    )


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """Impute values while retaining named categorical columns for CatBoost."""

    def __init__(self, groups: FeatureGroups):
        self.groups = groups

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CatBoostPreprocessor":
        """Learn imputation values from the training partition only."""
        del y
        frame = X.loc[:, self.groups.all]
        self.numeric_fill_ = {
            column: frame[column].median() for column in (*self.groups.numeric, *self.groups.boolean)
        }
        self.categorical_fill_ = {}
        for column in self.groups.categorical:
            mode = frame[column].mode(dropna=True)
            self.categorical_fill_[column] = str(mode.iloc[0]) if not mode.empty else "__MISSING__"
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned fills and keep categorical values as strings."""
        frame = X.loc[:, self.groups.all].copy()
        for column, value in self.numeric_fill_.items():
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(value)
        for column, value in self.categorical_fill_.items():
            frame[column] = frame[column].astype("string").fillna(value).astype(str)
        return frame


def candidate_models(groups: FeatureGroups = DEFAULT_FEATURES, random_state: int = 42) -> dict[str, Pipeline]:
    """Return the active, extensible gradient-boosting benchmark models."""
    xgboost = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    categorical_indices = [groups.all.index(column) for column in groups.categorical]
    catboost = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        cat_features=categorical_indices,
    )
    return {
        "xgboost": Pipeline([("preprocess", build_preprocessor(groups)), ("classifier", xgboost)]),
        "catboost": Pipeline(
            [("preprocess", CatBoostPreprocessor(groups)), ("classifier", catboost)]
        ),
    }


def probability_metrics(
    y_true: pd.Series | np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Calculate probability and threshold-based binary metrics."""
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, probability),
        "roc_auc": roc_auc_score(y_true, probability),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
    }


def compare_models(
    raw: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, Pipeline, pd.DataFrame]:
    """Benchmark both models on one stratified validation/test partition."""
    features, target, groups = prepare_model_data(raw)
    stable_ids = extract_stable_ids(raw, target.index)
    partitions = stratified_split(target, random_state, stable_ids=stable_ids)
    audit = split_audit(target, partitions)
    prevalence = dict(zip(audit["split"], audit["fraud_rate"]))
    rows: list[dict[str, float | str | bool]] = []

    fitted_models: dict[str, Pipeline] = {}
    for name, model in candidate_models(groups, random_state).items():
        model.fit(features.loc[partitions.train], target.loc[partitions.train])
        validation_probability = model.predict_proba(features.loc[partitions.validation])[:, 1]
        validation_metrics = probability_metrics(target.loc[partitions.validation], validation_probability)
        test_probability = model.predict_proba(features.loc[partitions.test])[:, 1]
        test_metrics = probability_metrics(target.loc[partitions.test], test_probability)
        fitted_models[name] = model
        rows.append(
            {
                "model": name,
                "validation_positive_prevalence": prevalence["validation"],
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "test_positive_prevalence": prevalence["test"],
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

    selected = max(rows, key=lambda row: float(row["validation_pr_auc"]))["model"]
    for row in rows:
        row["selected"] = row["model"] == selected
    comparison = pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False).reset_index(drop=True)
    return comparison, fitted_models[str(selected)], audit
