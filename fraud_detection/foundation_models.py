"""Stage C Train-to-Validation benchmarking for tabular foundation models."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import to_snake_case
from .modeling import (
    DEFAULT_FEATURES,
    SPLIT_ID_COLUMN,
    FeatureGroups,
    prepare_model_data,
    probability_metrics,
    split_membership_fingerprints,
    stratified_split,
)
from .vendor import TARGET_COLUMN, clean_vendor_data


RANDOM_STATE = 42
TABPFN_CHECKPOINT = "tabpfn-v3-classifier-v3_default.ckpt"
TABICL_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
MODEL_NAMES = ("tabpfn_3", "tabicl_v2")


def _target_and_identity(raw: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Normalize only label and stable ID so Test feature values remain untouched."""
    sources: dict[str, list[object]] = {TARGET_COLUMN: [], SPLIT_ID_COLUMN: []}
    for column in raw:
        normalized = to_snake_case(column)
        if normalized in sources:
            sources[normalized].append(column)
    for name, columns in sources.items():
        if len(columns) != 1:
            raise ValueError(f"Expected exactly one {name!r} column.")
    selected = [sources[TARGET_COLUMN][0], sources[SPLIT_ID_COLUMN][0]]
    cleaned = clean_vendor_data(raw.loc[:, selected])
    target = cleaned[TARGET_COLUMN]
    keep = target.notna()
    target = target.loc[keep].astype(int)
    if target.nunique() != 2:
        raise ValueError("Target must contain both fraud and non-fraud records.")
    return target, cleaned.loc[keep, SPLIT_ID_COLUMN].copy()


def _target_only(raw: pd.DataFrame) -> pd.Series:
    """Return the normalized label while enforcing availability of stable identity."""
    return _target_and_identity(raw)[0]


def prepare_stage_c_data(
    raw: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    *,
    return_membership: bool = False,
) -> tuple[Any, ...]:
    """Return only the frozen Train and Validation feature partitions.

    The full label vector is required to reproduce the existing stratified split.
    Feature cleaning is deliberately restricted to Train and Validation indices;
    no Test feature row is selected, cleaned, returned, fitted, or predicted.
    """
    target, stable_ids = _target_and_identity(raw)
    partitions = stratified_split(
        target, random_state=random_state, stable_ids=stable_ids
    )
    allowed_index = partitions.train.append(partitions.validation)
    features, prepared_target, groups = prepare_model_data(raw.loc[allowed_index])
    if tuple(groups.all) != tuple(DEFAULT_FEATURES.all) or len(groups.all) != 14:
        raise ValueError("Stage C requires the exact frozen 14-feature information set.")
    if not prepared_target.equals(target.loc[allowed_index]):
        raise ValueError("Prepared Train/Validation targets do not match the frozen split labels.")
    prepared = (
        features.loc[partitions.train].copy(),
        prepared_target.loc[partitions.train].copy(),
        features.loc[partitions.validation].copy(),
        prepared_target.loc[partitions.validation].copy(),
        groups,
    )
    if return_membership:
        return (*prepared, split_membership_fingerprints(stable_ids, partitions))
    return prepared


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def representation_audit(
    train: pd.DataFrame, validation: pd.DataFrame, groups: FeatureGroups
) -> dict[str, Any]:
    """Describe the exact unencoded pandas representation passed to both APIs."""
    expected = groups.all
    if expected != DEFAULT_FEATURES.all or list(train.columns) != expected or list(validation.columns) != expected:
        raise ValueError("Both Stage C partitions must use the ordered frozen 14 features.")

    def describe(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(frame)),
            "features": int(frame.shape[1]),
            "column_names": list(frame.columns),
            "pandas_dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            "missing_value_counts": {
                column: int(count) for column, count in frame.isna().sum().items()
            },
            "fingerprint_sha256": _frame_fingerprint(frame),
        }

    return {
        "train": describe(train),
        "validation": describe(validation),
        "categorical_columns": list(groups.categorical),
        "numeric_columns": list(groups.numeric),
        "boolean_columns": list(groups.boolean),
        "information_set_identical_for_both_models": True,
        "external_encoding_or_imputation": False,
    }


def package_versions() -> dict[str, str]:
    """Resolve installed versions without relying on package __version__ fields."""
    return {
        package: importlib.metadata.version(package)
        for package in ("torch", "tabpfn", "tabicl")
    }


def build_foundation_model(model_name: str, groups: FeatureGroups) -> Any:
    """Construct one official classifier while leaving inference defaults unchanged."""
    from tabicl import TabICLClassifier
    from tabpfn import TabPFNClassifier

    categorical_indices = [groups.all.index(column) for column in groups.categorical]
    if model_name == "tabpfn_3":
        return TabPFNClassifier(
            model_path="auto",
            device="cuda",
            categorical_features_indices=categorical_indices,
            random_state=RANDOM_STATE,
        )
    if model_name == "tabicl_v2":
        return TabICLClassifier(
            checkpoint_version=TABICL_CHECKPOINT,
            device="cuda",
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unsupported Stage C model: {model_name}")


def _checkpoint_path(model_name: str, model: Any) -> str:
    if model_name == "tabicl_v2" and getattr(model, "model_path_", None):
        return str(model.model_path_)
    if model_name == "tabpfn_3":
        candidate = Path(os.environ.get("APPDATA", "")) / "tabpfn" / TABPFN_CHECKPOINT
        if candidate.is_file():
            return str(candidate)
    return str(getattr(model, "model_path", ""))


def _native_preprocessor(model_name: str, model: Any) -> str:
    if model_name == "tabpfn_3":
        return "TabPFN native preprocessing with categorical_features_indices"
    transformer = getattr(model, "feature_transformer_", None)
    if transformer is None:
        transformer = getattr(model, "x_encoder_", None)
    suffix = f" ({type(transformer).__name__})" if transformer is not None else ""
    return "TabICL native pandas dtype preprocessing" + suffix


def _cuda_memory(torch_module: Any) -> tuple[float, float]:
    return (
        float(torch_module.cuda.max_memory_allocated() / 1048576),
        float(torch_module.cuda.max_memory_reserved() / 1048576),
    )


def evaluate_foundation_models(
    train: pd.DataFrame,
    y_train: pd.Series,
    validation: pd.DataFrame,
    y_validation: pd.Series,
    groups: FeatureGroups,
    model_builder: Callable[[str, FeatureGroups], Any] = build_foundation_model,
    torch_module: Any | None = None,
    split_membership: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit on Train and evaluate only Validation using each native API."""
    if torch_module is None:
        import torch as torch_module

    audit = representation_audit(train, validation, groups)
    rows: list[dict[str, Any]] = []
    model_metadata: dict[str, Any] = {}

    for model_name in MODEL_NAMES:
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()
        torch_module.cuda.synchronize()
        total_started = time.perf_counter()
        model = None
        stage = "constructor_initialization"
        caught_warnings: list[str] = []
        try:
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                constructor_started = time.perf_counter()
                model = model_builder(model_name, groups)
                torch_module.cuda.synchronize()
                constructor_seconds = time.perf_counter() - constructor_started
                # Constructors are intentionally lazy in both installed APIs. Their
                # cached checkpoint load occurs inside fit and cannot be separated
                # without patching private model-loading internals.
                stage = "fit_context"
                fit_started = time.perf_counter()
                model.fit(train.copy(), y_train.copy())
                torch_module.cuda.synchronize()
                fit_seconds = time.perf_counter() - fit_started
                stage = "validation_predict_proba"
                predict_started = time.perf_counter()
                probability = np.asarray(model.predict_proba(validation.copy()))[:, 1]
                torch_module.cuda.synchronize()
                predict_seconds = time.perf_counter() - predict_started
                caught_warnings = [str(record.message) for record in records]
        except torch_module.cuda.OutOfMemoryError:
            allocated, reserved = _cuda_memory(torch_module)
            rows.append(
                {
                    "model": model_name,
                    "status": "cuda_oom",
                    "oom_stage": stage,
                    "cuda_peak_allocated_mib": allocated,
                    "cuda_peak_reserved_mib": reserved,
                }
            )
            model_metadata[model_name] = {
                "status": "cuda_oom",
                "oom_stage": stage,
                "defaults": model.get_params(deep=False) if model is not None else {},
            }
            if model is not None:
                del model
            gc.collect()
            torch_module.cuda.empty_cache()
            continue

        metrics = probability_metrics(y_validation, probability, threshold=0.5)
        allocated, reserved = _cuda_memory(torch_module)
        total_seconds = time.perf_counter() - total_started
        checkpoint = TABPFN_CHECKPOINT if model_name == "tabpfn_3" else TABICL_CHECKPOINT
        row = {
            "model": model_name,
            "status": "passed",
            "checkpoint": checkpoint,
            **metrics,
            "classification_threshold": 0.5,
            "constructor_initialization_seconds": constructor_seconds,
            "fit_context_including_cached_load_seconds": fit_seconds,
            "validation_predict_proba_seconds": predict_seconds,
            "total_downstream_evaluation_seconds": total_seconds,
            "cuda_peak_allocated_mib": allocated,
            "cuda_peak_reserved_mib": reserved,
            "cuda_device": torch_module.cuda.get_device_name(torch_module.cuda.current_device()),
        }
        rows.append(row)
        model_metadata[model_name] = {
            "status": "passed",
            "checkpoint": checkpoint,
            "resolved_checkpoint_path": _checkpoint_path(model_name, model),
            "native_preprocessing": _native_preprocessor(model_name, model),
            "constructor_is_lazy": True,
            "cached_model_load_timing_scope": "included in fit_context_including_cached_load_seconds",
            "defaults": model.get_params(deep=False),
            "warnings": caught_warnings,
        }
        del model
        gc.collect()
        torch_module.cuda.empty_cache()

    metadata = {
        "stage": "C",
        "scope": "real_data_train_to_validation_only",
        "random_state": RANDOM_STATE,
        "classification_threshold": 0.5,
        "primary_metric": "pr_auc",
        "packages": package_versions(),
        "representation": audit,
        "models": model_metadata,
    }
    if split_membership is not None:
        metadata["split_membership"] = split_membership
    return pd.DataFrame(rows), metadata


def run_stage_c(
    raw: pd.DataFrame,
    output_csv: Path,
    metadata_json: Path,
    model_builder: Callable[[str, FeatureGroups], Any] = build_foundation_model,
    torch_module: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Stage C and write Validation-only artifacts."""
    train, y_train, validation, y_validation, groups, membership = prepare_stage_c_data(
        raw, return_membership=True
    )
    results, metadata = evaluate_foundation_models(
        train,
        y_train,
        validation,
        y_validation,
        groups,
        model_builder=model_builder,
        torch_module=torch_module,
        split_membership=membership,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    metadata_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return results, metadata
