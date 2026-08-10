"""Freeze Stage C configurations and perform one final Stage E Test evaluation."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
import warnings
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .foundation_models import (
    MODEL_NAMES,
    FOUNDATION_RUNTIME,
    RANDOM_STATE,
    TABICL_CHECKPOINT,
    TABPFN_CHECKPOINT,
    _cuda_memory,
    _target_and_identity,
    build_foundation_model,
    package_versions,
)
from .modeling import (
    DEFAULT_FEATURES,
    SPLIT_ID_COLUMN,
    SPLIT_PROTOCOL_VERSION,
    FeatureGroups,
    prepare_model_data,
    probability_metrics,
    split_membership_fingerprints,
    stratified_split,
)


FOUNDATION_LABELS = {"tabpfn_3": "TabPFN-3", "tabicl_v2": "TabICLv2"}
EXPECTED_PACKAGES = {"tabpfn_3": ("tabpfn", "8.1.0"), "tabicl_v2": ("tabicl", "2.1.1")}
MEMBERSHIP_FIELDS = (
    "split_protocol_version",
    "identity_column",
    "dataset_membership_sha256",
    "train_membership_sha256",
    "validation_membership_sha256",
    "test_membership_sha256",
)
CHECKPOINT_PROVENANCE_FIELDS = (
    "checkpoint_filename",
    "checkpoint_sha256",
    "source_repository",
    "source_revision_or_snapshot",
    "resolution",
)


def _validation_metrics(row: pd.Series) -> dict[str, float]:
    names = ("pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy", "accuracy")
    return {name: float(row[name]) for name in names}


def build_freeze_document(
    validation_results: pd.DataFrame, stage_c_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the exact two-model freeze document from persisted Stage C evidence."""
    if set(validation_results["model"]) != set(MODEL_NAMES):
        raise ValueError("Stage C results must contain exactly both foundation models.")
    if not validation_results["status"].eq("passed").all():
        raise ValueError("Both Stage C foundation-model runs must have passed before freezing.")
    if stage_c_metadata.get("stage") != "C":
        raise ValueError("Expected persisted Stage C metadata.")
    features = stage_c_metadata["representation"]["train"]["column_names"]
    if features != DEFAULT_FEATURES.all or len(features) != 14:
        raise ValueError("Cannot freeze anything other than the ordered full 14-feature set.")
    membership = stage_c_metadata.get("split_membership")
    if not isinstance(membership, Mapping) or any(
        field not in membership for field in MEMBERSHIP_FIELDS
    ):
        raise ValueError("Stage C metadata lacks required stable split membership fingerprints.")
    if stage_c_metadata.get("packages") != FOUNDATION_RUNTIME:
        raise ValueError("Stage C runtime packages do not match the approved frozen runtime.")

    models: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        row = validation_results.loc[validation_results["model"].eq(model_name)].iloc[0]
        package, version = EXPECTED_PACKAGES[model_name]
        if stage_c_metadata["packages"].get(package) != version:
            raise ValueError(f"Stage C {package} version does not match the approved freeze.")
        defaults = stage_c_metadata["models"][model_name]["defaults"]
        provenance = stage_c_metadata["models"][model_name].get("checkpoint_provenance")
        if not isinstance(provenance, Mapping) or any(
            field not in provenance for field in CHECKPOINT_PROVENANCE_FIELDS
        ):
            raise ValueError(f"Stage C checkpoint provenance is missing for {model_name}.")
        checkpoint = TABPFN_CHECKPOINT if model_name == "tabpfn_3" else TABICL_CHECKPOINT
        if int(defaults["n_estimators"]) != 8 or defaults["device"] != "cuda":
            raise ValueError(f"Stage C defaults for {model_name} do not match the approved freeze.")
        if float(row["classification_threshold"]) != 0.5:
            raise ValueError("Stage C classification threshold must be 0.5.")
        categorical_indices = [10, 11, 12, 13] if model_name == "tabpfn_3" else None
        models[model_name] = {
            "package": package,
            "package_version": version,
            "model_family": FOUNDATION_LABELS[model_name],
            "estimator_class": "TabPFNClassifier" if model_name == "tabpfn_3" else "TabICLClassifier",
            "checkpoint": checkpoint,
            "checkpoint_provenance": dict(provenance),
            "n_estimators": 8,
            "random_state": RANDOM_STATE,
            "device": "cuda",
            "feature_names": list(features),
            "categorical_columns": list(DEFAULT_FEATURES.categorical),
            "categorical_indices": categorical_indices,
            "preprocessing_policy": stage_c_metadata["models"][model_name]["native_preprocessing"],
            "classification_threshold": 0.5,
            "stage_c_validation_metrics": _validation_metrics(row),
            "configuration_status": "frozen",
            "model_specific_hpo_performed": False,
            "hpo_statement": "No model-specific hyperparameter optimization was performed.",
        }
    return {
        "schema_version": 3,
        "stage": "D",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "Configurations frozen from Stage C before Stage E Test feature access.",
        "split_membership": dict(membership),
        "runtime_packages": dict(FOUNDATION_RUNTIME),
        "models": models,
    }


def validate_freeze_document(
    document: Mapping[str, Any], *, require_membership: bool = False,
    require_checkpoint_provenance: bool = False,
) -> None:
    """Reject incomplete or mutated freeze documents."""
    if document.get("stage") != "D" or set(document.get("models", {})) != set(MODEL_NAMES):
        raise ValueError("Freeze document must contain both approved Stage D models.")
    for model_name in MODEL_NAMES:
        config = document["models"][model_name]
        package, version = EXPECTED_PACKAGES[model_name]
        expected_checkpoint = TABPFN_CHECKPOINT if model_name == "tabpfn_3" else TABICL_CHECKPOINT
        if config.get("configuration_status") != "frozen":
            raise ValueError(f"{model_name} is not frozen.")
        if config.get("package") != package or config.get("package_version") != version:
            raise ValueError(f"{model_name} package freeze is invalid.")
        if config.get("checkpoint") != expected_checkpoint:
            raise ValueError(f"{model_name} checkpoint freeze is invalid.")
        if config.get("n_estimators") != 8 or config.get("random_state") != 42:
            raise ValueError(f"{model_name} inference defaults are not frozen correctly.")
        if config.get("device") != "cuda" or config.get("classification_threshold") != 0.5:
            raise ValueError(f"{model_name} device or threshold freeze is invalid.")
        if list(config.get("feature_names", ())) != DEFAULT_FEATURES.all:
            raise ValueError(f"{model_name} feature order is not the frozen 14-feature order.")
        if config.get("model_specific_hpo_performed") is not False:
            raise ValueError(f"{model_name} must remain a non-HPO configuration.")
    if list(document["models"]["tabpfn_3"].get("categorical_indices", ())) != [10, 11, 12, 13]:
        raise ValueError("TabPFN categorical indices do not match Stage C.")
    if document["models"]["tabicl_v2"].get("categorical_indices") is not None:
        raise ValueError("TabICL must retain native pandas dtype categorical detection.")
    membership = document.get("split_membership")
    if membership is None:
        if require_membership or document.get("schema_version") in (2, 3):
            raise ValueError(
                "Legacy Stage D freeze lacks membership fingerprints and cannot authorize Stage E."
            )
        return
    if document.get("schema_version") not in (2, 3) or not isinstance(membership, Mapping):
        raise ValueError("Stable membership fingerprints require Stage D schema_version 2 or 3.")
    if any(field not in membership for field in MEMBERSHIP_FIELDS):
        raise ValueError("Stage D membership fingerprints are incomplete.")
    if membership["split_protocol_version"] != SPLIT_PROTOCOL_VERSION:
        raise ValueError("Stage D split protocol version is invalid.")
    if membership["identity_column"] != SPLIT_ID_COLUMN:
        raise ValueError("Stage D split identity column is invalid.")
    for field in MEMBERSHIP_FIELDS[2:]:
        value = membership[field]
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"Stage D membership fingerprint {field!r} is invalid.")
    if document.get("schema_version") != 3:
        if require_checkpoint_provenance:
            raise ValueError(
                "Legacy Stage D freeze lacks verified checkpoint provenance and cannot authorize Stage E."
            )
        return
    if document.get("runtime_packages") != FOUNDATION_RUNTIME:
        raise ValueError("Stage D foundation runtime package provenance is invalid.")
    for model_name in MODEL_NAMES:
        provenance = document["models"][model_name].get("checkpoint_provenance")
        if not isinstance(provenance, Mapping) or any(
            field not in provenance for field in CHECKPOINT_PROVENANCE_FIELDS
        ):
            raise ValueError(f"Stage D checkpoint provenance is incomplete for {model_name}.")
        digest = provenance["checkpoint_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"Stage D checkpoint SHA-256 is invalid for {model_name}.")
        expected = TABPFN_CHECKPOINT if model_name == "tabpfn_3" else TABICL_CHECKPOINT
        if provenance["checkpoint_filename"] != expected:
            raise ValueError(f"Stage D checkpoint filename is invalid for {model_name}.")


def ensure_stage_e_outputs_available(
    output_paths: tuple[Path, ...], *, allow_test_reproduction: bool
) -> None:
    """Refuse to overwrite any final Stage E output unless explicitly authorized."""
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not allow_test_reproduction:
        raise FileExistsError(
            "Final Stage E artifacts already exist; refusing Test reevaluation/overwrite: "
            f"{existing}. Use --allow-test-reproduction only for deliberate maintenance reproduction."
        )


def verify_foundation_runtime(frozen: Mapping[str, Any]) -> None:
    """Require the exact runtime recorded by the new Stage D protocol."""
    actual = package_versions()
    if dict(frozen["runtime_packages"]) != actual:
        raise RuntimeError(
            f"Foundation runtime differs from the Stage D freeze: expected "
            f"{dict(frozen['runtime_packages'])}, active {actual}."
        )


def verify_checkpoint_files(
    frozen: Mapping[str, Any], checkpoint_paths: Mapping[str, Path] | None
) -> dict[str, Path]:
    """Verify caller-supplied checkpoint files against frozen filename and SHA-256."""
    if checkpoint_paths is None or set(checkpoint_paths) != set(MODEL_NAMES):
        raise ValueError("Stage E requires explicit checkpoint paths for both frozen models.")
    verified: dict[str, Path] = {}
    for model_name in MODEL_NAMES:
        path = Path(checkpoint_paths[model_name])
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found for {model_name}: {path}")
        provenance = frozen["models"][model_name]["checkpoint_provenance"]
        if path.name != provenance["checkpoint_filename"]:
            raise ValueError(f"Checkpoint filename differs from freeze for {model_name}.")
        digest_builder = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest_builder.update(block)
        digest = digest_builder.hexdigest()
        if digest != provenance["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint SHA-256 differs from freeze for {model_name}.")
        verified[model_name] = path
    return verified


def persist_freeze_artifact(
    validation_csv: Path, stage_c_metadata_json: Path, freeze_json: Path
) -> dict[str, Any]:
    """Atomically persist, reload, and validate the Stage D artifact."""
    validation = pd.read_csv(validation_csv)
    metadata = json.loads(stage_c_metadata_json.read_text(encoding="utf-8"))
    document = build_freeze_document(validation, metadata)
    validate_freeze_document(document)
    freeze_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = freeze_json.with_suffix(freeze_json.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    validate_freeze_document(json.loads(temporary.read_text(encoding="utf-8")))
    os.replace(temporary, freeze_json)
    reloaded = json.loads(freeze_json.read_text(encoding="utf-8"))
    validate_freeze_document(reloaded)
    return reloaded


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def load_frozen_configs(path: Path) -> Mapping[str, Any]:
    """Load validated configurations into recursively immutable mappings."""
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_freeze_document(document)
    return _deep_freeze(document)


def prepare_stage_e_data(
    raw: pd.DataFrame, frozen: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, FeatureGroups]:
    """Access Test features only after receiving a validated frozen document."""
    validate_freeze_document(
        frozen, require_membership=True, require_checkpoint_provenance=True
    )
    target, stable_ids = _target_and_identity(raw)
    partitions = stratified_split(
        target, random_state=RANDOM_STATE, stable_ids=stable_ids
    )
    actual_membership = split_membership_fingerprints(stable_ids, partitions)
    if dict(frozen["split_membership"]) != actual_membership:
        raise ValueError(
            "Stage E dataset/split membership fingerprint differs from the Stage D freeze."
        )
    combined_index = partitions.train.append(partitions.validation)
    all_index = combined_index.append(partitions.test)
    features, prepared_target, groups = prepare_model_data(raw.loc[all_index])
    if groups.all != DEFAULT_FEATURES.all or features.shape[1] != 14:
        raise ValueError("Stage E requires the exact ordered 14-feature set.")
    return (
        features.loc[combined_index].copy(),
        prepared_target.loc[combined_index].copy(),
        features.loc[partitions.test].copy(),
        prepared_target.loc[partitions.test].copy(),
        groups,
    )


def _verify_model_against_freeze(model_name: str, model: Any, config: Mapping[str, Any]) -> None:
    params = model.get_params(deep=False)
    for name in ("n_estimators", "random_state", "device"):
        if params.get(name) != config[name]:
            raise ValueError(f"Constructed {model_name} changed frozen parameter {name}.")
    if model_name == "tabpfn_3":
        if list(params.get("categorical_features_indices", [])) != list(config["categorical_indices"]):
            raise ValueError("Constructed TabPFN changed frozen categorical indices.")
    elif params.get("checkpoint_version") != config["checkpoint"]:
        raise ValueError("Constructed TabICL changed the frozen checkpoint.")


def evaluate_frozen_test_models(
    train_validation: pd.DataFrame,
    y_train_validation: pd.Series,
    test: pd.DataFrame,
    y_test: pd.Series,
    groups: FeatureGroups,
    frozen: Mapping[str, Any],
    model_builder: Callable[[str, FeatureGroups], Any] = build_foundation_model,
    torch_module: Any | None = None,
    checkpoint_paths: Mapping[str, Path] | None = None,
) -> pd.DataFrame:
    """Evaluate each immutable frozen model exactly once on Test."""
    validate_freeze_document(frozen)
    if torch_module is None:
        import torch as torch_module
    if list(train_validation.columns) != DEFAULT_FEATURES.all or list(test.columns) != DEFAULT_FEATURES.all:
        raise ValueError("Stage E feature information set or order changed.")
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        config = frozen["models"][model_name]
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
                started = time.perf_counter()
                if model_builder is build_foundation_model:
                    model = model_builder(
                        model_name, groups,
                        checkpoint_path=checkpoint_paths[model_name] if checkpoint_paths else None,
                    )
                else:
                    model = model_builder(model_name, groups)
                _verify_model_against_freeze(model_name, model, config)
                torch_module.cuda.synchronize()
                initialization_seconds = time.perf_counter() - started
                stage = "fit_context"
                started = time.perf_counter()
                model.fit(train_validation.copy(), y_train_validation.copy())
                torch_module.cuda.synchronize()
                fit_seconds = time.perf_counter() - started
                stage = "test_predict_proba"
                started = time.perf_counter()
                probability = np.asarray(model.predict_proba(test.copy()))[:, 1]
                torch_module.cuda.synchronize()
                predict_seconds = time.perf_counter() - started
                caught_warnings = [str(record.message) for record in records]
        except torch_module.cuda.OutOfMemoryError:
            allocated, reserved = _cuda_memory(torch_module)
            rows.append({
                "model": model_name,
                "status": "cuda_oom",
                "oom_stage": stage,
                "cuda_peak_allocated_mib": allocated,
                "cuda_peak_reserved_mib": reserved,
            })
            if model is not None:
                del model
            gc.collect()
            torch_module.cuda.empty_cache()
            continue
        metrics = probability_metrics(y_test, probability, threshold=float(config["classification_threshold"]))
        allocated, reserved = _cuda_memory(torch_module)
        rows.append({
            "model": model_name,
            "model_label": FOUNDATION_LABELS[model_name],
            "status": "passed",
            "checkpoint": config["checkpoint"],
            **metrics,
            "classification_threshold": float(config["classification_threshold"]),
            "constructor_initialization_seconds": initialization_seconds,
            "fit_context_including_cached_load_seconds": fit_seconds,
            "test_predict_proba_seconds": predict_seconds,
            "total_downstream_evaluation_seconds": time.perf_counter() - total_started,
            "cuda_peak_allocated_mib": allocated,
            "cuda_peak_reserved_mib": reserved,
            "cuda_device": torch_module.cuda.get_device_name(torch_module.cuda.current_device()),
            "warnings": json.dumps(caught_warnings),
        })
        del model
        gc.collect()
        torch_module.cuda.empty_cache()
    return pd.DataFrame(rows)


def build_model_family_comparison(
    foundation_validation: pd.DataFrame,
    foundation_test: pd.DataFrame,
    classical_validation: pd.DataFrame,
    classical_test: pd.DataFrame,
) -> pd.DataFrame:
    """Combine saved classical results with the two frozen foundation models."""
    rows: list[dict[str, Any]] = []
    selected = classical_validation.loc[classical_validation["selected"].astype(str).str.lower().eq("true")]
    for model_name, label in (("xgboost", "Tuned XGBoost"), ("catboost", "Tuned CatBoost")):
        validation = selected.loc[selected["model"].eq(model_name)].iloc[0]
        test = classical_test.loc[classical_test["model"].eq(model_name)].iloc[0]
        rows.append({
            "model": label,
            "model_family_type": "task-specific tuned gradient boosting",
            "validation_pr_auc": float(validation["validation_pr_auc"]),
            "validation_roc_auc": float(validation["validation_roc_auc"]),
            "test_pr_auc": float(test["test_pr_auc"]),
            "test_roc_auc": float(test["test_roc_auc"]),
            "test_precision": float(test["test_precision"]),
            "test_recall": float(test["test_recall"]),
            "test_f1": float(test["test_f1"]),
            "test_balanced_accuracy": float(test["test_balanced_accuracy"]),
            "test_accuracy": float(test["test_accuracy"]),
        })
    for model_name in MODEL_NAMES:
        validation = foundation_validation.loc[foundation_validation["model"].eq(model_name)].iloc[0]
        test = foundation_test.loc[foundation_test["model"].eq(model_name)].iloc[0]
        rows.append({
            "model": FOUNDATION_LABELS[model_name],
            "model_family_type": "pretrained tabular foundation model (default, no HPO)",
            "validation_pr_auc": float(validation["pr_auc"]),
            "validation_roc_auc": float(validation["roc_auc"]),
            "test_pr_auc": float(test["pr_auc"]),
            "test_roc_auc": float(test["roc_auc"]),
            "test_precision": float(test["precision"]),
            "test_recall": float(test["recall"]),
            "test_f1": float(test["f1"]),
            "test_balanced_accuracy": float(test["balanced_accuracy"]),
            "test_accuracy": float(test["accuracy"]),
        })
    result = pd.DataFrame(rows)
    result["validation_minus_test_pr_auc"] = result["validation_pr_auc"] - result["test_pr_auc"]
    return result


def generate_model_family_figure(comparison: pd.DataFrame, output_path: Path) -> None:
    """Generate the single requested Validation/Test PR-AUC comparison figure."""
    plt.style.use("seaborn-v0_8-whitegrid")
    positions = np.arange(len(comparison))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.2))
    validation = axis.bar(
        positions - width / 2,
        comparison["validation_pr_auc"],
        width,
        label="Validation PR-AUC",
        color="#2F6B9A",
    )
    test = axis.bar(
        positions + width / 2,
        comparison["test_pr_auc"],
        width,
        label="Test PR-AUC",
        color="#2A9D8F",
    )
    axis.bar_label(validation, fmt="%.3f", padding=3, fontsize=9)
    axis.bar_label(test, fmt="%.3f", padding=3, fontsize=9)
    axis.set_xticks(positions, comparison["model"])
    axis.set_ylim(0.90, 0.99)
    axis.set(title="Frozen Model Family Comparison", ylabel="PR-AUC")
    axis.legend(frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_stage_e(
    raw: pd.DataFrame,
    freeze_json: Path,
    foundation_validation_csv: Path,
    classical_validation_csv: Path,
    classical_test_csv: Path,
    foundation_test_csv: Path,
    comparison_csv: Path,
    figure_path: Path,
    model_builder: Callable[[str, FeatureGroups], Any] = build_foundation_model,
    torch_module: Any | None = None,
    checkpoint_paths: Mapping[str, Path] | None = None,
    allow_test_reproduction: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the gated one-time Stage E confirmation and final comparisons."""
    ensure_stage_e_outputs_available(
        (foundation_test_csv, comparison_csv, figure_path),
        allow_test_reproduction=allow_test_reproduction,
    )
    frozen = load_frozen_configs(freeze_json)
    train_validation, y_train_validation, test, y_test, groups = prepare_stage_e_data(raw, frozen)
    verify_foundation_runtime(frozen)
    verified_checkpoints = verify_checkpoint_files(frozen, checkpoint_paths)
    test_results = evaluate_frozen_test_models(
        train_validation,
        y_train_validation,
        test,
        y_test,
        groups,
        frozen,
        model_builder=model_builder,
        torch_module=torch_module,
        checkpoint_paths=verified_checkpoints,
    )
    foundation_test_csv.parent.mkdir(parents=True, exist_ok=True)
    test_results.to_csv(foundation_test_csv, index=False)
    if not test_results["status"].eq("passed").all():
        return test_results, pd.DataFrame()
    comparison = build_model_family_comparison(
        pd.read_csv(foundation_validation_csv),
        test_results,
        pd.read_csv(classical_validation_csv),
        pd.read_csv(classical_test_csv),
    )
    comparison.to_csv(comparison_csv, index=False)
    generate_model_family_figure(comparison, figure_path)
    return test_results, comparison
