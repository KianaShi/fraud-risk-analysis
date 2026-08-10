"""Train-only Bayesian optimization for the frozen Task 2 benchmark."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .modeling import (
    DEFAULT_FEATURES,
    BenchmarkPartitions,
    CatBoostPreprocessor,
    FeatureGroups,
    build_preprocessor,
    candidate_models,
    extract_stable_ids,
    prepare_model_data,
    probability_metrics,
    stratified_split,
)

try:
    import optuna
except ImportError:  # Allows the rest of the project to explain a missing optional runtime cleanly.
    optuna = None


OPTIMIZATION_SEED = 42
CV_SEED = 42
CV_FOLDS = 5
PRIMARY_OBJECTIVE = "mean_cv_pr_auc"
PRIMARY_FEATURES = tuple(DEFAULT_FEATURES.all)
MODEL_NAMES = ("xgboost", "catboost")


@dataclass(frozen=True)
class CVResult:
    """Cross-validation Average Precision summary."""

    mean_pr_auc: float
    std_pr_auc: float
    fold_scores: tuple[float, ...]


@dataclass(frozen=True)
class FrozenConfiguration:
    """A model configuration selected on Validation before Test is accessed."""

    model: str
    selected_configuration: str
    selected_parameters: dict[str, Any]
    best_optuna_parameters: dict[str, Any]
    train_cv_pr_auc: float
    train_cv_pr_auc_std: float
    validation_pr_auc: float
    validation_pr_auc_delta: float
    improvement_assessment: str
    optimization_seed: int
    cv_seed: int
    cv_folds: int
    n_trials: int


@dataclass(frozen=True)
class OptimizationResult:
    """Best Optuna trial plus diagnostics needed for later reporting."""

    model: str
    best_parameters: dict[str, Any]
    cv_result: CVResult
    optimization_seconds: float
    study: Any


def _require_optuna() -> Any:
    if optuna is None:
        raise ImportError(
            "Optuna is required for tuning. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        )
    return optuna


def suggest_xgboost_parameters(trial: Any) -> dict[str, Any]:
    """Conservative XGBoost search space for the small Task 2 sample."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.60, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    }


def suggest_catboost_parameters(trial: Any) -> dict[str, Any]:
    """Conservative CatBoost search space with native categoricals intact."""
    return {
        "iterations": trial.suggest_int("iterations", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
    }


def suggest_parameters(model_name: str, trial: Any) -> dict[str, Any]:
    """Dispatch one legitimate model search space."""
    if model_name == "xgboost":
        return suggest_xgboost_parameters(trial)
    if model_name == "catboost":
        return suggest_catboost_parameters(trial)
    raise ValueError(f"Unsupported model: {model_name}")


def build_tuned_model(
    model_name: str,
    groups: FeatureGroups,
    parameters: dict[str, Any],
    random_state: int = OPTIMIZATION_SEED,
) -> Pipeline:
    """Build a tuned model while preserving each existing preprocessing path."""
    if model_name == "xgboost":
        classifier = XGBClassifier(
            **parameters,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
        return Pipeline(
            [("preprocess", build_preprocessor(groups)), ("classifier", classifier)]
        )
    if model_name == "catboost":
        categorical_indices = [groups.all.index(column) for column in groups.categorical]
        classifier = CatBoostClassifier(
            **parameters,
            loss_function="Logloss",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            cat_features=categorical_indices,
            thread_count=1,
        )
        return Pipeline(
            [("preprocess", CatBoostPreprocessor(groups)), ("classifier", classifier)]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(
    model_name: str,
    groups: FeatureGroups,
    configuration: str,
    parameters: dict[str, Any] | None = None,
    random_state: int = OPTIMIZATION_SEED,
) -> Pipeline:
    """Build either the frozen repository default or a tuned candidate."""
    if configuration == "default":
        if parameters:
            raise ValueError("Default configuration does not accept tuned parameters.")
        return candidate_models(groups, random_state)[model_name]
    if configuration == "tuned":
        if not parameters:
            raise ValueError("Tuned configuration requires hyperparameters.")
        return build_tuned_model(model_name, groups, parameters, random_state)
    raise ValueError(f"Unsupported configuration: {configuration}")


def cross_validated_pr_auc(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: FeatureGroups,
    configuration: str,
    parameters: dict[str, Any] | None = None,
    cv_folds: int = CV_FOLDS,
    cv_seed: int = CV_SEED,
    model_seed: int = OPTIMIZATION_SEED,
) -> CVResult:
    """Evaluate Average Precision using stratified folds created strictly inside Train."""
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
    scores: list[float] = []
    for fold_train, fold_validation in splitter.split(X_train, y_train):
        model = build_model(model_name, groups, configuration, parameters, model_seed)
        model.fit(X_train.iloc[fold_train], y_train.iloc[fold_train])
        probability = model.predict_proba(X_train.iloc[fold_validation])[:, 1]
        metrics = probability_metrics(y_train.iloc[fold_validation], probability)
        scores.append(float(metrics["pr_auc"]))
    return CVResult(float(np.mean(scores)), float(np.std(scores)), tuple(scores))


def optimize_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: FeatureGroups,
    n_trials: int,
    optimization_seed: int = OPTIMIZATION_SEED,
    cv_seed: int = CV_SEED,
    cv_folds: int = CV_FOLDS,
) -> OptimizationResult:
    """Maximize mean Train-only CV Average Precision with a seeded TPE sampler."""
    optuna_module = _require_optuna()
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    optuna_module.logging.set_verbosity(optuna_module.logging.WARNING)
    sampler = optuna_module.samplers.TPESampler(seed=optimization_seed)
    study = optuna_module.create_study(direction="maximize", sampler=sampler)

    def objective(trial: Any) -> float:
        parameters = suggest_parameters(model_name, trial)
        result = cross_validated_pr_auc(
            model_name,
            X_train,
            y_train,
            groups,
            "tuned",
            parameters,
            cv_folds,
            cv_seed,
            optimization_seed,
        )
        trial.set_user_attr("cv_pr_auc_std", result.std_pr_auc)
        trial.set_user_attr("fold_pr_auc", list(result.fold_scores))
        return result.mean_pr_auc

    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    elapsed = time.perf_counter() - started
    best = study.best_trial
    cv_result = CVResult(
        mean_pr_auc=float(best.value),
        std_pr_auc=float(best.user_attrs["cv_pr_auc_std"]),
        fold_scores=tuple(float(value) for value in best.user_attrs["fold_pr_auc"]),
    )
    return OptimizationResult(model_name, dict(best.params), cv_result, elapsed, study)


def _evaluate_validation(
    model_name: str,
    configuration: str,
    parameters: dict[str, Any] | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    groups: FeatureGroups,
    random_state: int,
) -> tuple[dict[str, float], float, float]:
    model = build_model(model_name, groups, configuration, parameters, random_state)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probability = model.predict_proba(X_validation)[:, 1]
    inference_seconds = time.perf_counter() - started
    return probability_metrics(y_validation, probability), training_seconds, inference_seconds


def assess_improvement(delta: float) -> str:
    """Describe a Validation Average Precision change without overstating small gains."""
    if delta < 0:
        return "negative"
    if delta < 0.001:
        return "negligible"
    if delta < 0.005:
        return "marginal"
    return "meaningful"


def freeze_configuration(
    model_name: str,
    default_metrics: dict[str, float],
    tuned_metrics: dict[str, float],
    default_cv: CVResult,
    optimization: OptimizationResult,
    n_trials: int,
    optimization_seed: int,
    cv_seed: int,
    cv_folds: int,
) -> FrozenConfiguration:
    """Select solely by fixed Validation Average Precision and freeze before Test."""
    delta = float(tuned_metrics["pr_auc"] - default_metrics["pr_auc"])
    use_tuned = delta > 0
    selected_cv = optimization.cv_result if use_tuned else default_cv
    return FrozenConfiguration(
        model=model_name,
        selected_configuration="tuned" if use_tuned else "default",
        selected_parameters=dict(optimization.best_parameters) if use_tuned else {},
        best_optuna_parameters=dict(optimization.best_parameters),
        train_cv_pr_auc=selected_cv.mean_pr_auc,
        train_cv_pr_auc_std=selected_cv.std_pr_auc,
        validation_pr_auc=float(tuned_metrics["pr_auc"] if use_tuned else default_metrics["pr_auc"]),
        validation_pr_auc_delta=delta,
        improvement_assessment=assess_improvement(delta),
        optimization_seed=optimization_seed,
        cv_seed=cv_seed,
        cv_folds=cv_folds,
        n_trials=n_trials,
    )


def _trials_table(optimization: OptimizationResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in optimization.study.trials:
        row: dict[str, Any] = {
            "model": optimization.model,
            "trial_number": int(trial.number),
            "state": str(trial.state.name),
            "cv_mean_pr_auc": float(trial.value) if trial.value is not None else np.nan,
            "cv_pr_auc_std": trial.user_attrs.get("cv_pr_auc_std", np.nan),
            "parameters_json": json.dumps(trial.params, sort_keys=True),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_optimization_history(trials: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, model_name in zip(axes, MODEL_NAMES):
        subset = trials.loc[trials["model"].eq(model_name)].sort_values("trial_number")
        values = pd.to_numeric(subset["cv_mean_pr_auc"], errors="coerce")
        axis.plot(subset["trial_number"], values, marker="o", alpha=0.45, label="Trial")
        axis.plot(subset["trial_number"], values.cummax(), linewidth=2, label="Best so far")
        axis.set(title=model_name.title(), xlabel="Trial", ylabel="Train-CV Average Precision")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Optuna Optimization History (Train-only 5-fold CV)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _validate_primary_benchmark(groups: FeatureGroups) -> None:
    if tuple(groups.all) != PRIMARY_FEATURES or len(groups.all) != 14:
        raise ValueError(
            "Tuning requires the frozen 14-feature primary benchmark; "
            f"received {len(groups.all)} features: {groups.all}"
        )


def _final_test_evaluation(
    frozen: FrozenConfiguration,
    X_train_validation: pd.DataFrame,
    y_train_validation: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups: FeatureGroups,
) -> dict[str, Any]:
    if not isinstance(frozen, FrozenConfiguration):
        raise TypeError("Test evaluation requires a frozen Validation-selected configuration.")
    model = build_model(
        frozen.model,
        groups,
        frozen.selected_configuration,
        frozen.selected_parameters,
        frozen.optimization_seed,
    )
    started = time.perf_counter()
    model.fit(X_train_validation, y_train_validation)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probability = model.predict_proba(X_test)[:, 1]
    inference_seconds = time.perf_counter() - started
    metrics = probability_metrics(y_test, probability)
    return {
        "model": frozen.model,
        "selected_configuration": frozen.selected_configuration,
        "validation_pr_auc": frozen.validation_pr_auc,
        **{f"test_{key}": value for key, value in metrics.items()},
        "final_training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
    }


def run_tuning_experiment(
    raw: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    xgb_trials: int = 40,
    catboost_trials: int = 40,
    random_state: int = OPTIMIZATION_SEED,
    cv_folds: int = CV_FOLDS,
) -> dict[str, Any]:
    """Tune on Train, select on Validation, then confirm once on Test."""
    X, y, groups = prepare_model_data(raw, DEFAULT_FEATURES)
    _validate_primary_benchmark(groups)
    stable_ids = extract_stable_ids(raw, y.index)
    partitions: BenchmarkPartitions = stratified_split(
        y, random_state, stable_ids=stable_ids
    )
    X_train, y_train = X.loc[partitions.train], y.loc[partitions.train]
    X_validation, y_validation = X.loc[partitions.validation], y.loc[partitions.validation]

    trial_counts = {"xgboost": xgb_trials, "catboost": catboost_trials}
    validation_rows: list[dict[str, Any]] = []
    trial_tables: list[pd.DataFrame] = []
    frozen_configurations: list[FrozenConfiguration] = []

    for model_name in MODEL_NAMES:
        optimization = optimize_model(
            model_name,
            X_train,
            y_train,
            groups,
            trial_counts[model_name],
            random_state,
            CV_SEED,
            cv_folds,
        )
        default_cv = cross_validated_pr_auc(
            model_name, X_train, y_train, groups, "default", cv_folds=cv_folds
        )
        default_metrics, default_train_seconds, default_inference_seconds = _evaluate_validation(
            model_name,
            "default",
            None,
            X_train,
            y_train,
            X_validation,
            y_validation,
            groups,
            random_state,
        )
        tuned_metrics, tuned_train_seconds, tuned_inference_seconds = _evaluate_validation(
            model_name,
            "tuned",
            optimization.best_parameters,
            X_train,
            y_train,
            X_validation,
            y_validation,
            groups,
            random_state,
        )
        frozen = freeze_configuration(
            model_name,
            default_metrics,
            tuned_metrics,
            default_cv,
            optimization,
            trial_counts[model_name],
            random_state,
            CV_SEED,
            cv_folds,
        )
        frozen_configurations.append(frozen)
        for configuration, cv_result, metrics, train_seconds, inference_seconds in (
            (
                "default",
                default_cv,
                default_metrics,
                default_train_seconds,
                default_inference_seconds,
            ),
            (
                "tuned",
                optimization.cv_result,
                tuned_metrics,
                tuned_train_seconds,
                tuned_inference_seconds,
            ),
        ):
            validation_rows.append(
                {
                    "model": model_name,
                    "configuration": configuration,
                    "selected": configuration == frozen.selected_configuration,
                    "cv_mean_pr_auc": cv_result.mean_pr_auc,
                    "cv_pr_auc_std": cv_result.std_pr_auc,
                    **{f"validation_{key}": value for key, value in metrics.items()},
                    "validation_pr_auc_delta_tuned_minus_default": frozen.validation_pr_auc_delta,
                    "improvement_assessment": frozen.improvement_assessment,
                    "optimization_seconds": optimization.optimization_seconds
                    if configuration == "tuned"
                    else 0.0,
                    "validation_training_seconds": train_seconds,
                    "validation_inference_seconds": inference_seconds,
                }
            )
        trial_tables.append(_trials_table(optimization))

    # Test remains untouched until both Validation decisions above are frozen.
    train_validation = partitions.train.append(partitions.validation)
    final_rows = [
        _final_test_evaluation(
            frozen,
            X.loc[train_validation],
            y.loc[train_validation],
            X.loc[partitions.test],
            y.loc[partitions.test],
            groups,
        )
        for frozen in frozen_configurations
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    validation_comparison = pd.DataFrame(validation_rows)
    final_comparison = pd.DataFrame(final_rows)
    trials = pd.concat(trial_tables, ignore_index=True)
    validation_comparison.to_csv(results_dir / "tuned_model_comparison.csv", index=False)
    final_comparison.to_csv(results_dir / "final_tuned_test_comparison.csv", index=False)
    trials.to_csv(results_dir / "hyperparameter_trials.csv", index=False)
    payload = {
        "design": {
            "primary_objective": PRIMARY_OBJECTIVE,
            "optimization_seed": random_state,
            "cv_seed": CV_SEED,
            "cv_folds": cv_folds,
            "features": list(PRIMARY_FEATURES),
            "test_access": "after both configurations were frozen on Validation",
        },
        "models": {frozen.model: asdict(frozen) for frozen in frozen_configurations},
    }
    (results_dir / "best_hyperparameters.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot_optimization_history(trials, figures_dir / "optimization_history.png")
    return {
        "validation_comparison": validation_comparison,
        "final_test_comparison": final_comparison,
        "trials": trials,
        "frozen_configurations": frozen_configurations,
        "partitions": partitions,
    }
