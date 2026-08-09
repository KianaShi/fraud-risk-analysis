# Fraud Risk Analysis

A leakage-aware fraud case study covering operational monitoring, vendor-value analysis, classical model tuning, tabular foundation-model benchmarking, and a production-oriented CatBoost probability-scoring path.

Fraud v1.0 freezes the completed Phases 1–4 research results. It does not continue model tuning, reuse Test for new decisions, or claim that a benchmark winner is automatically the best deployment choice.

## Business problem

The repository contains two deliberately separate analyses:

- **Task 1 — operational monitoring:** analyzes payment applications from April through September 2019 using a genuine `application_date` time axis.
- **Task 2 — fraud modeling and vendor evaluation:** uses a selected sample of approximately 3,000 historical merchants, roughly half labeled as fraud, with FraudKiller enrichment fields.

Task 2 asks whether the available signals discriminate fraudulent merchants and whether the vendor fields add predictive value. It is a fixed retrospective benchmark, not a production rollout or natural-prevalence study.

## Data and evaluation design

The original `FraudCaseStudy.xlsx` is excluded from Git. To reproduce the research workflows, place an authorized copy in the repository root or pass its path with `--input`.

Task 2 uses a reproducible label-stratified split with `random_state=42`:

| Split | N | Fraud | Non-fraud | Fraud rate |
|---|---:|---:|---:|---:|
| Train | 1,942 | 939 | 1,003 | 48.35% |
| Validation | 416 | 201 | 215 | 48.32% |
| Test | 417 | 202 | 215 | 48.44% |

This intentionally selected, approximately balanced sample must not be interpreted as expected production fraud prevalence. Validation was used across several development stages and is not repeatedly independent confirmation. Test was used only for the frozen Phase 3/4 confirmations and was not reused for the deferred TabPrep work.

The business meaning of `open_date` is not established by the case-study description. Raw and derived open-date fields are excluded from the primary benchmark.

## Leakage-safe modeling

The frozen primary feature set contains 14 source fields:

```text
ea_score, identity_rank, reputation_level, volume_score, result_number,
email_days, is_valid, is_connected, personal_device, receiving_mail,
area_code, device_browser_type, ip_address_loc_country, type
```

Preprocessing is learned from the fitting partition only. XGBoost receives imputed and one-hot-encoded data. CatBoost receives median/mode-imputed data while retaining named categorical features for native handling. The target never enters the feature matrix.

`result_number` is a numeric vendor count—the number of results returned for a record—not an identifier. Predictive importance is described as association with model predictions, not evidence that a feature causes fraud.

## Feature investigation

Train/Validation-only permutation importance, TreeSHAP, missingness analysis, and ablations identified `email_days`, `ea_score`, `device_browser_type`, `identity_rank`, and `type` as the strongest recurring signals. The five-feature result remains a separate compact ablation; hyperparameter tuning and the foundation-model benchmark use the frozen 14-feature primary representation.

Vendor-only features were stronger than in-house-only features on Validation, while their combination performed best. `email_days` showed the largest incremental association in the leave-one-vendor-feature-out analysis. These findings are sample-specific and do not establish causal vendor value.

Relevant artifacts include:

- [`feature_audit.csv`](docs/results/feature_audit.csv)
- [`permutation_importance.csv`](docs/results/permutation_importance.csv)
- [`shap_feature_importance.csv`](docs/results/shap_feature_importance.csv)
- [`feature_group_ablation.csv`](docs/results/feature_group_ablation.csv)
- [`vendor_feature_ablation.csv`](docs/results/vendor_feature_ablation.csv)
- [`final_feature_confirmation.csv`](docs/results/final_feature_confirmation.csv)

![Global SHAP feature importance](docs/figures/shap_feature_importance.png)

## Vendor value analysis

The FraudKiller comparison uses fixed illustrative case-study assumptions for revenue, fraud loss, manual-review cost, vendor cost, and review approval rate. It is not a production profit forecast.

| Scenario | Holdout ROC-AUC | Holdout PR-AUC | Expected profit under case assumptions |
|---|---:|---:|---:|
| Without FraudKiller | 0.8829 | 0.8520 | $53,520.00 |
| With FraudKiller | 0.9443 | 0.9506 | $95,010.50 |

![Vendor predictive and economic evaluation](docs/figures/vendor_evaluation.png)

## Classical model tuning

Optuna's seeded TPE optimizer evaluated XGBoost and CatBoost using five-fold stratified CV entirely inside Train. Validation was then used to freeze one configuration per model. Only after both configurations were frozen were they refitted on Train+Validation for a one-time Test confirmation.

The exact selected parameters are stored in [`best_hyperparameters.json`](docs/results/best_hyperparameters.json). No further HPO is performed during productionization.

## Tabular foundation-model benchmark

TabPFN-3 and TabICLv2 received the same ordered, unencoded 14-feature information set. Both used frozen/default pretrained eight-estimator configurations with no task-specific HPO. Their configurations were frozen after Train-to-Validation evaluation and reloaded before the one-time Test confirmation.

The foundation-model results are meaningful research findings rather than automatic deployment recommendations.

## Final model-family conclusion

| Model | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1 at 0.5 |
|---|---:|---:|---:|---:|
| Tuned XGBoost | 0.9670 | 0.9483 | 0.9382 | 0.8698 |
| Tuned CatBoost | 0.9623 | 0.9524 | 0.9457 | 0.8756 |
| TabPFN-3 | 0.9695 | 0.9555 | 0.9466 | 0.9063 |
| **TabICLv2** | **0.9719** | **0.9593** | **0.9509** | **0.9115** |

**Best research benchmark:** TabICLv2. On this fixed benchmark, tuning-free TabICLv2 achieved the strongest held-out PR-AUC and ROC-AUC, slightly outperforming the tuned boosting baselines. This does not establish that TabICL is universally better than boosting or that it will generalize identically elsewhere.

**Production-oriented candidate:** tuned CatBoost. Benchmark ranking alone was not treated as a deployment decision. CatBoost retains competitive discrimination, native categorical handling, mature feature-importance/SHAP tooling, low measured inference latency in this experiment, and a simpler operational footprint than the local foundation-model stack. CatBoost was not the highest-scoring model and is not presented as a proven production winner, the cheapest model, or the optimal deployment model.

The foundation models achieved slightly stronger discrimination, but those gains alone cannot determine whether their additional operational requirements justify replacing a simpler boosting candidate.

All four models showed a modest Validation-to-Test PR-AUC decrease, on the order of roughly 0.01–0.02 on this split. Exact differences are preserved in [`production_model_metadata.json`](docs/results/production_model_metadata.json). They are descriptive gaps, not formal statistical evidence of overfitting.

### Threshold caveat

Precision, recall, and F1 use a shared probability threshold of 0.5 for comparability. No model-specific threshold optimization or probability calibration was performed. Observed operating-point differences may partly reflect this shared, non-optimized threshold and must not be interpreted as each model's optimal business policy. PR-AUC and ROC-AUC are the cleaner threshold-independent discrimination comparisons here.

## Production-oriented CatBoost path

The production layer separates three responsibilities:

```text
frozen CatBoost + frozen preprocessing
    -> fraud_probability

optional, explicitly configured DecisionPolicy
    -> approve / manual_review / decline
```

[`fraud_detection/production.py`](fraud_detection/production.py) provides:

- loading and validation of the frozen Phase 3 CatBoost configuration;
- exact 14-feature schema and ordering checks;
- numeric, boolean, and categorical validation;
- native CatBoost model plus persisted preprocessing-state loading;
- bounded `predict_proba` fraud probabilities; and
- an optional decision-policy function requiring explicit thresholds.

No policy thresholds are configured by default. Without both `review_threshold` and `decline_threshold`, the CLI returns probabilities only. Any supplied thresholds are caller-owned business policy inputs, not optimized or recommended values.

### Serialized model status

No reusable `.cbm` existed when this pass began, and the repository does not contain a saved Train+Validation membership artifact that would allow refitting without reconstructing the split from labels. To avoid touching Test labels again, this pass does not rebuild the model from the complete Task 2 file.

[`build_production_model.py`](build_production_model.py) instead requires an explicitly prepared development-only CSV containing the frozen Train+Validation rows and no Test rows. The caller must attest this with `--confirm-development-only`. This flag records the user's declaration only: the code cannot determine or prove that an arbitrary CSV has the correct split membership or excludes Test. Split membership must be verified independently before invoking the build. The script then loads the exact CatBoost parameters from the frozen JSON and writes:

```text
artifacts/catboost_fraud_model.cbm
artifacts/catboost_preprocessor.json
```

Generated model binaries and preprocessing state are ignored by Git. The tracked metadata records that the artifact has not yet been built in-repository and that the current repository state is not deployment-ready.

### Input schema

- Numeric: `ea_score`, `identity_rank`, `reputation_level`, `volume_score`, `result_number`, `email_days`
- Boolean encoded as 0/1/missing: `is_valid`, `is_connected`, `personal_device`, `receiving_mail`
- Categorical: `area_code`, `device_browser_type`, `ip_address_loc_country`, `type`

All 14 columns are required, even when individual values are missing. Unknown, duplicate, missing, or extra columns fail clearly. Non-numeric values in numeric fields and non-binary values in boolean fields are rejected rather than silently reinterpreted. Missing values retain the established Train-fitted median/mode policy.

## Production monitoring considerations

**Design considerations only—not an implemented monitoring system.** A real deployment should eventually observe:

- schema violations and inference errors;
- missingness and numeric-distribution drift;
- categorical-distribution drift;
- prediction-score drift;
- approve/review/decline volume drift after a policy exists;
- observed fraud-rate drift when delayed ground truth arrives;
- precision, recall, and calibration degradation once labels mature;
- latency and model/version metadata; and
- evidence-based triggers for retraining review.

No dashboard, alert threshold, logging pipeline, scheduler, registry integration, or retraining automation is implemented here.

## Limitations

- Task 2's approximately 50% fraud rate is selected benchmark prevalence, not expected production prevalence.
- Production threshold selection requires real prevalence, false-negative loss, false-positive rejection cost, review cost/capacity, merchant economics, and calibrated probabilities. These inputs are unavailable.
- Validation was reused across development stages and should not be represented as repeated independent confirmation.
- The single fixed benchmark does not establish external or prospective generalization.
- Shared-threshold F1 differences do not prove calibration differences, and threshold tuning is not assumed to reverse model ordering.
- Existing SHAP and importance values describe association with predictions, not causality.
- Deployment SLA, throughput, infrastructure cost, probability calibration, and live policy outcomes were not measured.

## What Fraud v1.0 includes

Implemented:

- frozen Phases 1–4 research documentation;
- production-oriented CatBoost probability inference;
- strict schema validation and deterministic config metadata;
- reproducible development-only model build/loading path;
- optional explicitly configured decision-policy interface;
- unit tests and documented monitoring considerations; and
- a deferred TabPrep backlog.

Not implemented:

- business-calibrated approve/review/decline thresholds;
- probability calibration or natural production prevalence estimation;
- external prospective validation;
- live monitoring, retraining, registry, or deployment infrastructure; and
- real-data TabPrep experiments.

## Reproduction

Create and activate an environment, then install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run synthetic-data tests:

```bash
python -m unittest discover -s tests -v
```

Regenerate production metadata from frozen artifacts without fitting or scoring:

```bash
python generate_production_metadata.py
```

Build the native CatBoost artifact only from an independently verified Train+Validation CSV that excludes Test:

```bash
python build_production_model.py \
  --input <train_validation_only.csv> \
  --confirm-development-only
```

Prediction requires both production artifacts to exist first:

```text
artifacts/catboost_fraud_model.cbm
artifacts/catboost_preprocessor.json
```

A fresh clone cannot run `predict_fraud.py` until `build_production_model.py` has been run with an independently verified development-only input, or equivalent approved artifacts have been supplied. Missing artifacts produce an explicit `FileNotFoundError`.

After those artifacts exist, return probability only:

```bash
python predict_fraud.py --input <merchants.json>
```

Optionally apply caller-supplied business thresholds:

```bash
python predict_fraud.py \
  --input <merchants.csv> \
  --review-threshold <business-supplied-value> \
  --decline-threshold <business-supplied-value>
```

These thresholds have no defaults and are not derived from production prevalence, cost optimization, or probability calibration.

Research workflow scripts remain available for reproducibility, but frozen Test experiments should not be rerun for model selection. Compact results are tracked under `docs/results/`; intermediate data remains ignored under `outputs/`.

## Future work

- production probability calibration;
- cost-sensitive policy selection using real prevalence and business inputs;
- external or prospective validation;
- implemented drift and performance monitoring; and
- exploratory TabPrep representation ablation.

TabPrep's implementation was verified and smoke-tested synthetically, expanding 14 synthetic features to 964. No real fraud data was used. The real-data ablation was deferred to avoid extending the benchmark after Test results were known. Details and recoverable local exploration references are archived in [`docs/phase5_tabprep_backlog.md`](docs/phase5_tabprep_backlog.md).
