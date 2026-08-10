"""Production-oriented CatBoost scoring with explicit schema and policy boundaries."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .modeling import DEFAULT_FEATURES, CatBoostPreprocessor
from .profit import apply_three_way_decision


MODEL_NAME = "catboost_fraud_v1"
RANDOM_SEED = 42
BENCHMARK_THRESHOLD = 0.5
EXPECTED_FEATURES = tuple(DEFAULT_FEATURES.all)
FROZEN_CATBOOST_PARAMETERS = (
    "iterations",
    "depth",
    "learning_rate",
    "l2_leaf_reg",
    "random_strength",
    "bagging_temperature",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "selected_configuration": config["selected_configuration"],
        "selected_parameters": config["selected_parameters"],
        "random_seed": int(config["optimization_seed"]),
        "feature_names": list(EXPECTED_FEATURES),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_catboost_config(path: Path) -> dict[str, Any]:
    """Load and validate the Phase 3 CatBoost configuration."""
    document = json.loads(path.read_text(encoding="utf-8"))
    design = document.get("design", {})
    config = document.get("models", {}).get("catboost")
    if not isinstance(config, dict):
        raise ValueError("Frozen artifact does not contain models.catboost.")
    if config.get("selected_configuration") != "tuned":
        raise ValueError("The frozen CatBoost configuration is not marked as tuned.")
    if tuple(design.get("features", ())) != EXPECTED_FEATURES:
        raise ValueError("Frozen CatBoost feature schema differs from Fraud v1.0.")
    if int(config.get("optimization_seed", -1)) != RANDOM_SEED:
        raise ValueError("Frozen CatBoost random seed differs from Fraud v1.0.")
    parameters = config.get("selected_parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Frozen CatBoost parameters are missing.")
    return config


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and order the exact Fraud v1.0 source-feature schema."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Prediction input must be a pandas DataFrame.")
    if frame.columns.has_duplicates:
        duplicates = sorted(frame.columns[frame.columns.duplicated()].astype(str).tolist())
        raise ValueError(f"Duplicate input columns: {duplicates}")
    missing = sorted(set(EXPECTED_FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(EXPECTED_FEATURES))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected feature columns: {extra}")

    ordered = frame.loc[:, EXPECTED_FEATURES].copy()
    invalid_numeric: dict[str, list[str]] = {}
    for column in (*DEFAULT_FEATURES.numeric, *DEFAULT_FEATURES.boolean):
        original = ordered[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            invalid_numeric[column] = sorted(original.loc[invalid].astype(str).unique().tolist())[:5]
        ordered[column] = converted
    if invalid_numeric:
        raise ValueError(f"Non-numeric values in numeric/boolean features: {invalid_numeric}")
    nonfinite = {
        column: int(np.isinf(ordered[column].to_numpy(dtype=float, na_value=np.nan)).sum())
        for column in (*DEFAULT_FEATURES.numeric, *DEFAULT_FEATURES.boolean)
        if np.isinf(ordered[column].to_numpy(dtype=float, na_value=np.nan)).any()
    }
    if nonfinite:
        raise ValueError(f"Observed numeric/boolean feature values must be finite: {nonfinite}")

    invalid_boolean = {
        column: sorted(ordered.loc[ordered[column].notna() & ~ordered[column].isin([0, 1]), column].astype(str).unique().tolist())[:5]
        for column in DEFAULT_FEATURES.boolean
        if (ordered[column].notna() & ~ordered[column].isin([0, 1])).any()
    }
    if invalid_boolean:
        raise ValueError(f"Boolean features must contain only 0, 1, or missing: {invalid_boolean}")
    return ordered


def _preprocessor_document(
    preprocessor: CatBoostPreprocessor, frozen_config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_identifier": MODEL_NAME,
        "feature_names": list(EXPECTED_FEATURES),
        "numeric_features": list(DEFAULT_FEATURES.numeric),
        "boolean_features": list(DEFAULT_FEATURES.boolean),
        "categorical_features": list(DEFAULT_FEATURES.categorical),
        "numeric_fill": {key: float(value) for key, value in preprocessor.numeric_fill_.items()},
        "categorical_fill": dict(preprocessor.categorical_fill_),
        "frozen_configuration": {
            "source": "docs/results/best_hyperparameters.json",
            "selected_configuration": frozen_config["selected_configuration"],
            "selected_parameters": frozen_config["selected_parameters"],
            "random_seed": int(frozen_config["optimization_seed"]),
        },
        "missing_value_policy": "Training-development medians for numeric/boolean fields and modes for categorical fields.",
    }


def _load_preprocessor(path: Path) -> CatBoostPreprocessor:
    document = json.loads(path.read_text(encoding="utf-8"))
    if tuple(document.get("feature_names", ())) != EXPECTED_FEATURES:
        raise ValueError("Serialized preprocessor schema differs from Fraud v1.0.")
    preprocessor = CatBoostPreprocessor(DEFAULT_FEATURES)
    preprocessor.numeric_fill_ = {
        str(key): float(value) for key, value in document.get("numeric_fill", {}).items()
    }
    preprocessor.categorical_fill_ = {
        str(key): str(value) for key, value in document.get("categorical_fill", {}).items()
    }
    expected_numeric = set((*DEFAULT_FEATURES.numeric, *DEFAULT_FEATURES.boolean))
    if set(preprocessor.numeric_fill_) != expected_numeric:
        raise ValueError("Serialized numeric imputation state is incomplete.")
    if set(preprocessor.categorical_fill_) != set(DEFAULT_FEATURES.categorical):
        raise ValueError("Serialized categorical imputation state is incomplete.")
    return preprocessor


@dataclass
class ProductionCatBoost:
    """Loaded native CatBoost model plus its frozen preprocessing state."""

    model: CatBoostClassifier
    preprocessor: CatBoostPreprocessor
    configuration_metadata: dict[str, Any]
    identifier: str = MODEL_NAME

    @classmethod
    def load(
        cls,
        model_path: Path,
        preprocessor_path: Path,
        frozen_config_path: Path | None = None,
        manifest_path: Path | None = None,
    ) -> "ProductionCatBoost":
        if not model_path.is_file():
            raise FileNotFoundError(f"CatBoost model artifact not found: {model_path}")
        if not preprocessor_path.is_file():
            raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")
        if manifest_path is None:
            manifest_path = model_path.parent / "catboost_artifact_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_identifier") != MODEL_NAME:
            raise ValueError("Artifact manifest model identifier is invalid.")
        if tuple(manifest.get("feature_names", ())) != EXPECTED_FEATURES:
            raise ValueError("Artifact manifest feature schema differs from Fraud v1.0.")
        if manifest.get("target_classes") != [0, 1]:
            raise ValueError("Artifact manifest does not declare binary target classes [0, 1].")
        expected_hashes = manifest.get("sha256", {})
        for label, path in (("model", model_path), ("preprocessor", preprocessor_path)):
            if expected_hashes.get(label) != _sha256_file(path):
                raise ValueError(f"{label.title()} artifact SHA-256 does not match the manifest.")
        document = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        configuration = document.get("frozen_configuration")
        if not isinstance(configuration, dict):
            raise ValueError("Serialized frozen CatBoost configuration is missing.")
        if frozen_config_path is not None:
            frozen = load_frozen_catboost_config(frozen_config_path)
            if configuration.get("selected_parameters") != frozen["selected_parameters"]:
                raise ValueError("Serialized model parameters differ from the frozen Phase 3 artifact.")
            if int(configuration.get("random_seed", -1)) != int(frozen["optimization_seed"]):
                raise ValueError("Serialized model seed differs from the frozen Phase 3 artifact.")
            if manifest.get("frozen_configuration_sha256") != _configuration_fingerprint(frozen):
                raise ValueError("Artifact manifest differs from the frozen CatBoost configuration.")
        else:
            frozen = {
                "selected_parameters": configuration.get("selected_parameters", {}),
                "optimization_seed": configuration.get("random_seed"),
            }
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        serialized_parameters = model.get_params()
        actual_parameters = model.get_all_params()
        for name in FROZEN_CATBOOST_PARAMETERS:
            expected = frozen["selected_parameters"].get(name)
            actual = serialized_parameters.get(name, actual_parameters.get(name))
            if expected is None or actual is None or not np.isclose(
                float(actual), float(expected), rtol=1e-6, atol=1e-9
            ):
                raise ValueError(
                    f"Loaded CatBoost parameter {name!r} differs from the frozen configuration."
                )
        actual_seed = serialized_parameters.get(
            "random_seed", actual_parameters.get("random_seed")
        )
        if actual_seed is None or int(actual_seed) != int(frozen["optimization_seed"]):
            raise ValueError("Loaded CatBoost random seed differs from the frozen configuration.")
        if list(model.feature_names_) != list(EXPECTED_FEATURES):
            raise ValueError("Loaded CatBoost feature names/order differ from Fraud v1.0.")
        classes = np.asarray(model.classes_).tolist()
        if classes != [0, 1]:
            raise ValueError(f"Loaded CatBoost classes are not binary [0, 1]: {classes}")
        return cls(
            model=model,
            preprocessor=_load_preprocessor(preprocessor_path),
            configuration_metadata=configuration,
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        features = validate_feature_frame(frame)
        transformed = self.preprocessor.transform(features)
        probability = np.asarray(self.model.predict_proba(transformed), dtype=float)[:, 1]
        if probability.shape != (len(frame),) or not np.isfinite(probability).all():
            raise RuntimeError("CatBoost returned invalid fraud probabilities.")
        if ((probability < 0) | (probability > 1)).any():
            raise RuntimeError("CatBoost probabilities are outside [0, 1].")
        return probability

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"fraud_probability": self.predict_proba(frame)}, index=frame.index
        )


def build_production_artifacts(
    development: pd.DataFrame,
    target: pd.Series,
    frozen_config_path: Path,
    model_path: Path,
    preprocessor_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Fit the frozen configuration on an explicitly supplied development set only."""
    if not isinstance(target, pd.Series):
        raise TypeError("Development labels must be a pandas Series with an aligned index.")
    if len(development) != len(target) or not target.index.equals(development.index):
        raise ValueError(
            "Development label/index alignment is invalid; target.index must exactly equal features.index."
        )
    features = validate_feature_frame(development)
    labels = pd.to_numeric(target, errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Development labels must contain only 0/1.")
    features = features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    frozen = load_frozen_catboost_config(frozen_config_path)
    preprocessor = CatBoostPreprocessor(DEFAULT_FEATURES).fit(features, labels)
    transformed = preprocessor.transform(features)
    categorical_indices = [EXPECTED_FEATURES.index(column) for column in DEFAULT_FEATURES.categorical]
    model = CatBoostClassifier(
        **frozen["selected_parameters"],
        loss_function="Logloss",
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        cat_features=categorical_indices,
        thread_count=1,
    )
    model.fit(transformed, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = model_path.parent / "catboost_artifact_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path), format="cbm")
    preprocessor_path.write_text(
        json.dumps(_preprocessor_document(preprocessor, frozen), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "model_identifier": MODEL_NAME,
        "model_format": "cbm",
        "sha256": {
            "model": _sha256_file(model_path),
            "preprocessor": _sha256_file(preprocessor_path),
        },
        "frozen_configuration_source": "docs/results/best_hyperparameters.json",
        "frozen_configuration_sha256": _configuration_fingerprint(frozen),
        "selected_parameters": frozen["selected_parameters"],
        "random_seed": int(frozen["optimization_seed"]),
        "feature_names": list(EXPECTED_FEATURES),
        "target_classes": [0, 1],
        "catboost_version": importlib.metadata.version("catboost"),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "model_identifier": MODEL_NAME,
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "manifest_path": str(manifest_path),
        "training_rows": len(features),
        "training_data_scope": "caller-supplied frozen Train+Validation development partition; Test excluded",
        "random_seed": RANDOM_SEED,
        "feature_count": len(EXPECTED_FEATURES),
        "threshold_optimized": False,
    }


def apply_decision_policy(
    probability: np.ndarray,
    *,
    review_threshold: float,
    decline_threshold: float,
) -> np.ndarray:
    """Apply explicitly supplied business thresholds to model scores."""
    return apply_three_way_decision(probability, decline_threshold, review_threshold)


def build_production_metadata(
    frozen_config_path: Path,
    model_comparison_path: Path,
) -> dict[str, Any]:
    """Build metadata only from frozen artifacts; never fit or score a model."""
    frozen_document = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    frozen = load_frozen_catboost_config(frozen_config_path)
    comparison = pd.read_csv(model_comparison_path)
    expected_models = {"Tuned XGBoost", "Tuned CatBoost", "TabPFN-3", "TabICLv2"}
    if set(comparison["model"]) != expected_models:
        raise ValueError("Frozen model-family comparison has an unexpected model set.")
    metrics: dict[str, dict[str, float]] = {}
    for row in comparison.to_dict(orient="records"):
        validation_pr_auc = float(row["validation_pr_auc"])
        test_pr_auc = float(row["test_pr_auc"])
        metrics[str(row["model"])] = {
            "validation_pr_auc": validation_pr_auc,
            "test_pr_auc": test_pr_auc,
            "test_roc_auc": float(row["test_roc_auc"]),
            "test_f1_at_0_5": float(row["test_f1"]),
            "validation_minus_test_pr_auc": validation_pr_auc - test_pr_auc,
        }
    return {
        "schema_version": 1,
        "model": "CatBoost",
        "model_version": "Fraud v1.0",
        "model_identifier": MODEL_NAME,
        "role": "production-oriented candidate",
        "research_winner": "TabICLv2",
        "selection_statement": (
            "CatBoost was not selected because it had the highest benchmark score. "
            "It is retained as an operationally simpler candidate for further production validation."
        ),
        "configuration": {
            "source": "docs/results/best_hyperparameters.json",
            "selected_configuration": frozen["selected_configuration"],
            "selected_parameters": frozen["selected_parameters"],
            "random_seed": int(frozen["optimization_seed"]),
            "primary_objective": frozen_document["design"]["primary_objective"],
        },
        "features": {
            "count": len(EXPECTED_FEATURES),
            "names": list(EXPECTED_FEATURES),
            "numeric": list(DEFAULT_FEATURES.numeric),
            "boolean": list(DEFAULT_FEATURES.boolean),
            "categorical": list(DEFAULT_FEATURES.categorical),
        },
        "runtime": {
            "catboost_version": importlib.metadata.version("catboost"),
            "native_model_format": "cbm",
        },
        "training": {
            "scope": "frozen Train+Validation development partition; Test excluded",
            "artifact_status": "not built in repository; reproducible development-only build required",
            "production_artifact_built": False,
            "deployment_ready": False,
            "model_path": "artifacts/catboost_fraud_model.cbm",
            "preprocessor_path": "artifacts/catboost_preprocessor.json",
            "manifest_path": "artifacts/catboost_artifact_manifest.json",
            "build_input_attestation": (
                "The build flag records the user's declaration that the supplied CSV excludes Test rows; "
                "the code cannot verify split membership from an arbitrary CSV."
            ),
        },
        "threshold_policy": {
            "status": "unresolved; must be supplied separately from model scoring",
            "benchmark_threshold": 0.5,
            "benchmark_threshold_is_production_optimal": False,
            "model_specific_threshold_optimization_performed": False,
        },
        "calibration": {
            "status": "not performed",
            "production_calibration_available": False,
        },
        "frozen_benchmark_metrics": metrics,
        "evaluation_caveats": {
            "threshold": (
                "Precision, recall, and F1 use a shared 0.5 threshold. No model-specific "
                "threshold optimization or calibration was performed; these are not production-optimal operating points."
            ),
            "production_prevalence": (
                "Task 2 is an intentionally selected approximately balanced sample; its fraud rate is not expected production prevalence."
            ),
            "production_decision": (
                "Final model and threshold selection requires real prevalence, business costs, review capacity, calibration, and infrastructure evidence."
            ),
            "single_benchmark": (
                "TabICLv2 led discrimination on this fixed benchmark; this does not establish universal model-family superiority."
            ),
            "validation_reuse": (
                "The Validation partition was reused across development stages and is not repeatedly independent confirmation."
            ),
            "test_discipline": (
                "Test was used for the frozen Phase 3/4 confirmations and was not reused for the deferred TabPrep experiment."
            ),
        },
        "production_inputs_unavailable": [
            "natural fraud prevalence",
            "false-negative fraud cost",
            "false-positive rejection cost",
            "manual-review cost and capacity",
            "merchant lifetime or revenue assumptions",
            "production probability calibration",
            "deployment latency and infrastructure requirements",
        ],
    }
