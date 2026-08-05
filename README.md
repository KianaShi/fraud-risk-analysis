# Fraud Risk Analysis and Vendor Evaluation

A reproducible Python case study for payment-application monitoring, fraud-model comparison, and third-party vendor evaluation.

## Overview

The project analyzes application, identity, device, location, email, and temporal features to:

- monitor application volume and fraud-risk trends;
- compare Logistic Regression, Decision Tree, and Random Forest classifiers;
- convert fraud probabilities into approve, decline, and manual-review decisions; and
- estimate the incremental value of FraudKiller data under explicit business assumptions.

The implementation uses pandas, NumPy, Matplotlib, and scikit-learn—the original project technology stack.

## Repository structure

```text
fraud_detection/
    applications.py             Application cleaning and daily aggregation
    common.py                   Shared normalization and validation helpers
    modeling.py                 Leakage-safe preprocessing and model comparison
    profit.py                   Decision thresholds and expected-profit logic
    vendor.py                   Vendor-data cleaning and feature definitions
tests/                          Synthetic-data unit tests
docs/
    figures/                    Curated README visualizations
    results/                    Compact publishable result tables
prepare_application_metrics.py  Generate daily application metrics
plot_application_anomalies.py   Plot a specified anomaly window
clean_vendor_dataset.py         Normalize the vendor-enriched dataset
compare_fraud_models.py         Compare the three model families
evaluate_vendor_profit.py       Evaluate with/without-vendor scenarios
generate_project_figures.py     Rebuild README figures and result tables
```

## Data

The original `FraudCaseStudy.xlsx` workbook is intentionally excluded from Git because application-level case-study data should not be published without explicit permission.

To reproduce the analysis, place the workbook in the repository root or pass its local path with `--input`. The expected worksheet names are `p1-dataset` and `p2-dataset`.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Run the workflows

```bash
python prepare_application_metrics.py
python plot_application_anomalies.py \
  --anomaly-start 2019-06-25 \
  --anomaly-end 2019-07-04

python clean_vendor_dataset.py
python compare_fraud_models.py
python evaluate_vendor_profit.py \
  --approve-threshold 0.2 \
  --decline-threshold 0.8

python generate_project_figures.py
```

Intermediate files are written to the ignored `outputs/` directory. Curated figures and compact result tables are written to `docs/` and tracked by Git.

## Evaluation design

- Missing-value imputation, scaling, and one-hot encoding are fitted inside scikit-learn pipelines.
- Candidate models are ranked by five-fold cross-validated PR-AUC on the training partition.
- The independent holdout set is used only for the selected model's final evaluation.
- Metrics include ROC-AUC, PR-AUC, accuracy, balanced accuracy, precision, and recall.
- The no-vendor scenario excludes all nine fields identified as FraudKiller data in the workbook definition.

## Results

### Application trends

![Daily payment applications](docs/figures/daily_applications.png)

![Fraud rate among approved applications](docs/figures/approved_fraud_rate.png)

The highlighted period is a specified case-study window, not an automatically detected anomaly or a causal estimate.

### Cross-validated model comparison

| Model | CV ROC-AUC | CV PR-AUC | CV balanced accuracy | Selected |
|---|---:|---:|---:|:---:|
| Decision Tree | 0.9340 | **0.9345** | 0.8839 | Yes |
| Random Forest | 0.9159 | 0.9180 | 0.8652 | No |
| Logistic Regression | 0.8732 | 0.8661 | 0.8012 | No |

![Cross-validated model comparison](docs/figures/model_comparison.png)

The exact machine-readable results are available in [`docs/results/model_comparison.csv`](docs/results/model_comparison.csv).

### Selected-model holdout performance

The Decision Tree was selected using training-set CV PR-AUC and evaluated once on the fixed holdout set.

| ROC-AUC | PR-AUC | Accuracy | Balanced accuracy | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 0.9453 | 0.9446 | 0.9009 | 0.8990 | 0.9456 | 0.8433 |

### Vendor evaluation

![Vendor predictive and economic evaluation](docs/figures/vendor_evaluation.png)

| Scenario | Holdout ROC-AUC | Holdout PR-AUC | Expected profit |
|---|---:|---:|---:|
| Without FraudKiller data | 0.8876 | 0.8526 | $62,744.00 |
| With FraudKiller data | 0.9453 | 0.9446 | $92,962.50 |

Profit values depend on the case-study thresholds and assumptions. They are scenario outputs, not production forecasts.

## Testing

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic data and cover cleaning, preprocessing, model definitions, decision thresholds, and profit calculations. GitHub Actions runs the same command for each push and pull request.
