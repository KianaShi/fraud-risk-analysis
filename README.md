# Fraud Detection Case Study

A reproducible Python project for application-risk monitoring, fraud-model comparison, and vendor profitability analysis. It retains the original technology stack: pandas, NumPy, Matplotlib, and scikit-learn with logistic regression, decision tree, and random forest models.

## Project layout

```text
fraud_detection/                 Reusable cleaning, modeling, and profit logic
tests/                           Synthetic-data unit tests
prepare_application_metrics.py  Build application and product daily metrics
plot_application_anomalies.py   Plot a user-specified anomaly window
clean_vendor_dataset.py         Normalize the vendor-enriched dataset
compare_fraud_models.py         Cross-validate and compare three model families
evaluate_vendor_profit.py       Compare with/without-vendor decision economics
```

## Data

Place `FraudCaseStudy.xlsx` in the repository root. The workbook is intentionally ignored by Git because case-study or customer data should not be published by default. The scripts expect sheets named `p1-dataset` and `p2-dataset`; both names can be overridden from the command line.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the workflows

```bash
python prepare_application_metrics.py
python plot_application_anomalies.py --anomaly-start 2019-06-25 --anomaly-end 2019-07-04

python clean_vendor_dataset.py
python compare_fraud_models.py
python evaluate_vendor_profit.py --approve-threshold 0.2 --decline-threshold 0.8
```

Generated CSVs and figures are written under `outputs/`.

## Evaluation design

- Missing-value imputation and categorical encoding are fitted inside sklearn pipelines, preventing holdout leakage.
- Candidate models are ranked by cross-validated PR-AUC on the training partition.
- Only the selected model receives final holdout metrics.
- Accuracy is reported alongside PR-AUC, ROC-AUC, balanced accuracy, precision, and recall.
- The no-vendor scenario excludes every field identified as FraudKiller data in the workbook definition.

The anomaly dates and profit assumptions remain explicit case-study inputs rather than claimed facts. Change them through CLI thresholds or the `ProfitAssumptions` dataclass as appropriate for the business context.

## Test

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same command on every push and pull request.
