"""Train/validation-only feature investigation for the selected Task 2 sample."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.inspection import permutation_importance
from xgboost import DMatrix

from .modeling import (
    DEFAULT_FEATURES,
    BenchmarkPartitions,
    FeatureGroups,
    candidate_models,
    extract_stable_ids,
    prepare_model_data,
    probability_metrics,
    stratified_split,
)
from .vendor import (
    EXCLUDED_AMBIGUOUS_FEATURES,
    IDENTIFIER_COLUMNS,
    IN_HOUSE_FEATURES,
    TARGET_COLUMN,
    VENDOR_FEATURES,
)


@dataclass(frozen=True)
class FeatureConfiguration:
    """A named, auditable collection of original or engineered features."""

    name: str
    features: tuple[str, ...]


def _json(data: dict[object, object]) -> str:
    """Serialize audit dictionaries with stable, readable scalar values."""
    cleaned = {}
    for key, value in data.items():
        if pd.isna(key):
            key = "__MISSING__"
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        cleaned[str(key)] = value
    return json.dumps(cleaned, sort_keys=True)


def feature_groups_for(features: list[str] | tuple[str, ...], groups: FeatureGroups) -> FeatureGroups:
    """Return typed groups for a legitimate feature subset."""
    unknown = set(features) - set(groups.all)
    if unknown:
        raise ValueError(f"Unknown or excluded features: {sorted(unknown)}")
    return groups.subset(list(features))


def build_feature_audit(X_train: pd.DataFrame, y_train: pd.Series, groups: FeatureGroups) -> pd.DataFrame:
    """Build one machine-readable Train-only row per candidate feature."""
    rows: list[dict[str, object]] = []
    numeric = set(groups.numeric)
    boolean = set(groups.boolean)
    vendor = set(VENDOR_FEATURES)
    for feature in groups.all:
        values = X_train[feature]
        observed = values.dropna()
        counts = values.astype("object").where(values.notna(), "__MISSING__").value_counts(dropna=False)
        feature_type = "numeric" if feature in numeric else "boolean" if feature in boolean else "categorical"
        suspicious_id_like = bool(
            feature in IDENTIFIER_COLUMNS
            or feature.lower().endswith("_id")
            or (len(observed) > 0 and observed.nunique() / len(observed) >= 0.98)
        )
        row: dict[str, object] = {
            "feature": feature,
            "feature_type": feature_type,
            "source": "vendor" if feature in vendor else "in_house",
            "allowed_primary_benchmark": True,
            "missing_count": int(values.isna().sum()),
            "missing_rate": float(values.isna().mean()),
            "unique_count": int(values.nunique(dropna=True)),
            "cardinality": int(values.nunique(dropna=True)) if feature_type != "numeric" else np.nan,
            "constant": bool(values.nunique(dropna=False) <= 1),
            "near_constant": bool(not counts.empty and counts.iloc[0] / len(values) >= 0.99),
            "suspicious_id_like": suspicious_id_like,
            "review_note": "Potential identifier or near-unique field; review before production."
            if suspicious_id_like
            else "",
            "value_counts_json": "",
            "fraud_rate_by_value_json": "",
            "rare_category_rows": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "fraud_mean": np.nan,
            "fraud_median": np.nan,
            "non_fraud_mean": np.nan,
            "non_fraud_median": np.nan,
        }
        if feature_type == "numeric":
            numeric_values = pd.to_numeric(values, errors="coerce")
            quantiles = numeric_values.quantile([0.05, 0.25, 0.75, 0.95])
            row.update(
                {
                    "mean": numeric_values.mean(),
                    "median": numeric_values.median(),
                    "std": numeric_values.std(),
                    "q05": quantiles.loc[0.05],
                    "q25": quantiles.loc[0.25],
                    "q75": quantiles.loc[0.75],
                    "q95": quantiles.loc[0.95],
                    "fraud_mean": numeric_values.loc[y_train.eq(1)].mean(),
                    "fraud_median": numeric_values.loc[y_train.eq(1)].median(),
                    "non_fraud_mean": numeric_values.loc[y_train.eq(0)].mean(),
                    "non_fraud_median": numeric_values.loc[y_train.eq(0)].median(),
                }
            )
        else:
            labeled = pd.DataFrame({"value": values.astype("object").where(values.notna(), "__MISSING__"), "target": y_train})
            fraud_rates = labeled.groupby("value", dropna=False)["target"].mean()
            rare_threshold = max(5, int(np.ceil(len(values) * 0.01)))
            row.update(
                {
                    "value_counts_json": _json(counts.to_dict()),
                    "fraud_rate_by_value_json": _json(fraud_rates.to_dict()),
                    "rare_category_rows": int(counts.loc[counts < rare_threshold].sum()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_missingness_analysis(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Compare Train fraud prevalence when each incomplete feature is missing or observed."""
    rows = []
    for feature in X_train.columns:
        missing = X_train[feature].isna()
        if not missing.any() or missing.all():
            continue
        missing_rate = float(missing.mean())
        fraud_missing = float(y_train.loc[missing].mean())
        fraud_present = float(y_train.loc[~missing].mean())
        rows.append(
            {
                "feature": feature,
                "missing_count": int(missing.sum()),
                "missing_rate": missing_rate,
                "fraud_rate_when_missing": fraud_missing,
                "fraud_rate_when_present": fraud_present,
                "fraud_rate_difference": fraud_missing - fraud_present,
            }
        )
    return pd.DataFrame(rows).sort_values("fraud_rate_difference", key=lambda x: x.abs(), ascending=False)


def select_missing_indicators(
    X_train: pd.DataFrame, min_missing_rate: float = 0.01, max_missing_rate: float = 0.95
) -> tuple[str, ...]:
    """Select features whose Train missingness is common enough to test explicitly."""
    rates = X_train.isna().mean()
    return tuple(rates.index[(rates >= min_missing_rate) & (rates <= max_missing_rate)])


def augment_missingness_features(
    frame: pd.DataFrame,
    groups: FeatureGroups,
    indicator_features: tuple[str, ...],
    add_vendor_missing_count: bool,
) -> tuple[pd.DataFrame, FeatureGroups]:
    """Add documented binary indicators and optional vendor missing-count feature."""
    result = frame.copy()
    indicators = []
    for feature in indicator_features:
        if feature not in result:
            raise ValueError(f"Missing-indicator source is unavailable: {feature}")
        indicator = f"{feature}_missing"
        result[indicator] = result[feature].isna().astype(int)
        indicators.append(indicator)
    numeric = list(groups.numeric)
    if add_vendor_missing_count:
        available_vendor = [feature for feature in VENDOR_FEATURES if feature in result]
        result["vendor_missing_count"] = result[available_vendor].isna().sum(axis=1)
        numeric.append("vendor_missing_count")
    augmented = FeatureGroups(tuple(numeric), (*groups.boolean, *indicators), groups.categorical)
    return result, augmented


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: FeatureGroups,
    random_state: int = 42,
) -> dict[str, object]:
    """Fit both frozen benchmark model families on Train only."""
    models = candidate_models(groups, random_state)
    for model in models.values():
        model.fit(X_train.loc[:, groups.all], y_train)
    return models


def permutation_importance_table(
    models: dict[str, object], X_validation: pd.DataFrame, y_validation: pd.Series,
    n_repeats: int = 8, random_state: int = 42,
) -> pd.DataFrame:
    """Measure validation Average Precision permutation importance by source feature."""
    rows = []
    for model_name, model in models.items():
        result = permutation_importance(
            model, X_validation, y_validation, scoring="average_precision",
            n_repeats=n_repeats, random_state=random_state, n_jobs=1,
        )
        for feature, mean, std in zip(X_validation.columns, result.importances_mean, result.importances_std):
            rows.append({"model": model_name, "feature": feature, "mean_importance": mean, "std_importance": std})
    return pd.DataFrame(rows).sort_values(["model", "mean_importance"], ascending=[True, False])


def _original_xgb_feature(transformed: str, groups: FeatureGroups) -> str:
    """Map a transformed XGBoost feature back to its original business field."""
    name = transformed.split("__", 1)[-1]
    for feature in sorted(groups.all, key=len, reverse=True):
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def shap_importance_table(
    models: dict[str, object], X_validation: pd.DataFrame, groups: FeatureGroups,
    max_rows: int = 500, random_state: int = 42,
) -> pd.DataFrame:
    """Compute native TreeSHAP importance on Validation without causal claims."""
    sample = X_validation.sample(n=min(max_rows, len(X_validation)), random_state=random_state)
    rows = []
    for model_name, pipeline in models.items():
        preprocessor = pipeline.named_steps["preprocess"]
        classifier = pipeline.named_steps["classifier"]
        transformed = preprocessor.transform(sample)
        if model_name == "xgboost":
            values = classifier.get_booster().predict(DMatrix(transformed), pred_contribs=True)[:, :-1]
            names = preprocessor.get_feature_names_out()
            original = [_original_xgb_feature(name, groups) for name in names]
        else:
            cat_features = classifier.get_param("cat_features") or []
            values = classifier.get_feature_importance(
                Pool(transformed, cat_features=cat_features), type="ShapValues"
            )[:, :-1]
            original = list(groups.all)
        importance = pd.DataFrame({"feature": original, "value": np.abs(values).mean(axis=0)})
        importance = importance.groupby("feature", as_index=False)["value"].sum()
        importance = importance.sort_values("value", ascending=False).reset_index(drop=True)
        importance["rank"] = np.arange(1, len(importance) + 1)
        importance["model"] = model_name
        rows.append(importance.rename(columns={"value": "mean_abs_shap"}))
    return pd.concat(rows, ignore_index=True).loc[:, ["model", "feature", "mean_abs_shap", "rank"]]


def evaluate_configurations(
    X: pd.DataFrame, y: pd.Series, groups: FeatureGroups, partitions: BenchmarkPartitions,
    configurations: list[FeatureConfiguration], random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate candidate feature configurations on Validation only."""
    rows = []
    for configuration in configurations:
        subset_groups = feature_groups_for(configuration.features, groups)
        for model_name, model in candidate_models(subset_groups, random_state).items():
            model.fit(X.loc[partitions.train, subset_groups.all], y.loc[partitions.train])
            probability = model.predict_proba(X.loc[partitions.validation, subset_groups.all])[:, 1]
            metrics = probability_metrics(y.loc[partitions.validation], probability)
            rows.append(
                {
                    "model": model_name,
                    "feature_configuration": configuration.name,
                    "n_features": len(subset_groups.all),
                    "feature_names": "|".join(subset_groups.all),
                    **{f"validation_{key}": value for key, value in metrics.items()},
                }
            )
    return pd.DataFrame(rows)


def _choose_final_configuration(candidate_rows: pd.DataFrame, tolerance: float = 0.002) -> str:
    """Choose the smallest configuration within tolerance of mean validation Average Precision."""
    summary = candidate_rows.groupby("feature_configuration", as_index=False).agg(
        mean_validation_pr_auc=("validation_pr_auc", "mean"), n_features=("n_features", "first")
    )
    best = summary["mean_validation_pr_auc"].max()
    eligible = summary.loc[summary["mean_validation_pr_auc"] >= best - tolerance]
    return str(eligible.sort_values(["n_features", "mean_validation_pr_auc"], ascending=[True, False]).iloc[0]["feature_configuration"])


def _test_confirmation(
    X: pd.DataFrame, y: pd.Series, groups: FeatureGroups, partitions: BenchmarkPartitions,
    configurations: list[FeatureConfiguration], validation_results: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate only baseline and the already selected final configuration on Test."""
    rows = []
    for configuration in configurations:
        subset_groups = feature_groups_for(configuration.features, groups)
        for model_name, model in candidate_models(subset_groups, random_state).items():
            model.fit(X.loc[partitions.train, subset_groups.all], y.loc[partitions.train])
            probability = model.predict_proba(X.loc[partitions.test, subset_groups.all])[:, 1]
            test = probability_metrics(y.loc[partitions.test], probability)
            validation = validation_results.loc[
                validation_results["model"].eq(model_name)
                & validation_results["feature_configuration"].eq(configuration.name)
            ].iloc[0]
            rows.append(
                {
                    "model": model_name,
                    "feature_configuration": configuration.name,
                    "n_features": len(subset_groups.all),
                    "feature_names": "|".join(subset_groups.all),
                    "validation_pr_auc": validation["validation_pr_auc"],
                    "test_pr_auc": test["pr_auc"],
                    "test_roc_auc": test["roc_auc"],
                    "test_precision": test["precision"],
                    "test_recall": test["recall"],
                    "test_f1": test["f1"],
                }
            )
    return pd.DataFrame(rows)


def run_feature_investigation(
    raw: pd.DataFrame, results_dir: Path, figures_dir: Path,
    random_state: int = 42, n_repeats: int = 8,
) -> dict[str, pd.DataFrame | str]:
    """Run the complete investigation while quarantining Test until final confirmation."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    X, y, groups = prepare_model_data(raw)
    stable_ids = extract_stable_ids(raw, y.index)
    partitions = stratified_split(y, random_state, stable_ids=stable_ids)
    train, validation = partitions.train, partitions.validation

    audit = build_feature_audit(X.loc[train], y.loc[train], groups)
    missingness = build_missingness_analysis(X.loc[train], y.loc[train])
    audit.to_csv(results_dir / "feature_audit.csv", index=False)
    missingness.to_csv(results_dir / "missingness_analysis.csv", index=False)

    baseline_models = fit_models(X.loc[train], y.loc[train], groups, random_state)
    permutation = permutation_importance_table(
        baseline_models, X.loc[validation, groups.all], y.loc[validation], n_repeats, random_state
    )
    shap_table = shap_importance_table(baseline_models, X.loc[validation, groups.all], groups, random_state=random_state)
    permutation.to_csv(results_dir / "permutation_importance.csv", index=False)
    shap_table.to_csv(results_dir / "shap_feature_importance.csv", index=False)

    in_house = tuple(feature for feature in IN_HOUSE_FEATURES if feature in groups.all)
    vendor = tuple(feature for feature in VENDOR_FEATURES if feature in groups.all)
    group_configurations = [
        FeatureConfiguration("in_house_only", in_house),
        FeatureConfiguration("vendor_only", vendor),
        FeatureConfiguration("in_house_plus_vendor", tuple(groups.all)),
    ]
    group_ablation = evaluate_configurations(X, y, groups, partitions, group_configurations, random_state)
    group_ablation.to_csv(results_dir / "feature_group_ablation.csv", index=False)

    vendor_configurations = [FeatureConfiguration("none_removed", tuple(groups.all))]
    vendor_configurations.extend(
        FeatureConfiguration(f"without_{feature}", tuple(item for item in groups.all if item != feature))
        for feature in vendor
    )
    vendor_ablation = evaluate_configurations(X, y, groups, partitions, vendor_configurations, random_state)
    baseline_by_model = vendor_ablation.loc[vendor_ablation["feature_configuration"].eq("none_removed")].set_index("model")["validation_pr_auc"]
    vendor_ablation["validation_pr_auc_delta"] = vendor_ablation.apply(
        lambda row: row["validation_pr_auc"] - baseline_by_model.loc[row["model"]], axis=1
    )
    vendor_ablation.to_csv(results_dir / "vendor_feature_ablation.csv", index=False)

    indicators = select_missing_indicators(X.loc[train])
    X_indicators, indicator_groups = augment_missingness_features(X, groups, indicators, False)
    X_vendor_missing, vendor_missing_groups = augment_missingness_features(X, groups, indicators, True)
    missing_results = []
    for frame, typed_groups, name in (
        (X, groups, "simple_imputation"),
        (X_indicators, indicator_groups, "imputation_plus_missing_indicators"),
        (X_vendor_missing, vendor_missing_groups, "indicators_plus_vendor_missing_count"),
    ):
        configuration = FeatureConfiguration(name, tuple(typed_groups.all))
        missing_results.append(evaluate_configurations(frame, y, typed_groups, partitions, [configuration], random_state))
    missing_comparison = pd.concat(missing_results, ignore_index=True)
    missing_comparison.to_csv(results_dir / "missing_indicator_comparison.csv", index=False)

    ranking = (
        permutation.groupby("feature")["mean_importance"].mean().sort_values(ascending=False).index.tolist()
    )
    sizes = sorted({size for size in (5, 8, 10, 12, len(groups.all)) if size <= len(groups.all)})
    reduced_configurations = [
        FeatureConfiguration("all_legitimate_features" if size == len(groups.all) else f"top_{size}", tuple(ranking[:size]))
        for size in sizes
    ]
    reduced = evaluate_configurations(X, y, groups, partitions, reduced_configurations, random_state)
    reduced.to_csv(results_dir / "reduced_feature_comparison.csv", index=False)

    candidate_results = pd.concat(
        [
            reduced,
            missing_comparison.loc[missing_comparison["feature_configuration"].ne("simple_imputation")],
        ],
        ignore_index=True,
    )
    selected_name = _choose_final_configuration(candidate_results)
    if selected_name == "imputation_plus_missing_indicators":
        final_X, final_groups = X_indicators, indicator_groups
    elif selected_name == "indicators_plus_vendor_missing_count":
        final_X, final_groups = X_vendor_missing, vendor_missing_groups
    else:
        selected_row = candidate_results.loc[candidate_results["feature_configuration"].eq(selected_name)].iloc[0]
        selected_features = tuple(str(selected_row["feature_names"]).split("|"))
        final_X, final_groups = X, feature_groups_for(selected_features, groups)

    baseline_configuration = FeatureConfiguration("original_baseline", tuple(groups.all))
    final_configuration = FeatureConfiguration(selected_name, tuple(final_groups.all))
    baseline_validation = evaluate_configurations(X, y, groups, partitions, [baseline_configuration], random_state)
    selected_validation = candidate_results.loc[candidate_results["feature_configuration"].eq(selected_name)].copy()
    confirmation_validation = pd.concat([baseline_validation, selected_validation], ignore_index=True)
    baseline_confirmation = _test_confirmation(
        X, y, groups, partitions, [baseline_configuration], confirmation_validation, random_state
    )
    final_confirmation = _test_confirmation(
        final_X, y, final_groups, partitions, [final_configuration], confirmation_validation, random_state
    )
    confirmation = pd.concat([baseline_confirmation, final_confirmation], ignore_index=True)
    confirmation.to_csv(results_dir / "final_feature_confirmation.csv", index=False)

    plot_data = shap_table.groupby("feature", as_index=False)["mean_abs_shap"].mean().nlargest(10, "mean_abs_shap").sort_values("mean_abs_shap")
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(plot_data["feature"], plot_data["mean_abs_shap"], color="#2A9D8F")
    axis.set(title="Global SHAP Feature Importance", xlabel="Mean absolute SHAP value", ylabel="Original feature")
    figure.tight_layout()
    figure.savefig(figures_dir / "shap_feature_importance.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    metadata = pd.DataFrame(
        [
            {"role": "target", "features": TARGET_COLUMN},
            {"role": "identifiers", "features": "|".join(IDENTIFIER_COLUMNS)},
            {"role": "allowed_predictive", "features": "|".join(groups.all)},
            {"role": "vendor", "features": "|".join(vendor)},
            {"role": "excluded_ambiguous", "features": "|".join(EXCLUDED_AMBIGUOUS_FEATURES)},
        ]
    )
    metadata.to_csv(results_dir / "feature_availability.csv", index=False)
    return {
        "feature_audit": audit,
        "missingness_analysis": missingness,
        "permutation_importance": permutation,
        "shap_feature_importance": shap_table,
        "feature_group_ablation": group_ablation,
        "vendor_feature_ablation": vendor_ablation,
        "missing_indicator_comparison": missing_comparison,
        "reduced_feature_comparison": reduced,
        "final_feature_confirmation": confirmation,
        "selected_configuration": selected_name,
    }
