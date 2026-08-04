# Fraud Risk Analysis

This project explores how application data can be used to identify potentially fraudulent activity and support real-world approval decisions.

The goal was not simply to train the model with the highest accuracy. I wanted to understand which application patterns were associated with fraud, compare several interpretable classification models, and translate predicted probabilities into practical actions such as approval, decline, or manual review.

The analysis was completed using 2,775 application records containing identity, device, location, email, product, and time-related information.

## Project Questions

I organized the project around four main questions:

1. Are there unusual application patterns or time periods that may indicate elevated fraud activity?
2. Which applicant and application features are most useful for distinguishing fraud from legitimate activity?
3. Which classification model performs most reliably on unseen data?
4. Does third-party vendor data improve fraud detection enough to justify its cost?

## Key Results

The selected model achieved the following performance on the holdout dataset:

* **ROC-AUC:** 0.945
* **Accuracy:** 90.1%
* **Precision:** 94.6%
* **Recall:** 84.3%

Because fraud detection involves an imbalanced and cost-sensitive outcome, I used cross-validated **PR-AUC** to select the final model rather than relying on accuracy alone.

The final analysis also converted fraud probabilities into three operational decisions:

* **Approve:** low predicted fraud risk
* **Manual review:** uncertain or moderate risk
* **Decline:** high predicted fraud risk

This made it possible to evaluate the model as a decision-support system rather than only as a statistical classifier.

## What I Did

### 1. Explored application behavior

I analyzed application volume, fraud rates, product activity, and temporal patterns to identify periods with unusual behavior.

The exploratory analysis focused on features including:

* Applicant identity information
* Email characteristics
* Device and browser information
* IP and geographic signals
* Product selection
* Application timing
* Third-party fraud-risk attributes

This stage helped identify potentially useful signals while also highlighting fields that could introduce leakage or unstable model behavior.

### 2. Built a leakage-safe preprocessing pipeline

I created a reusable scikit-learn pipeline so that preprocessing steps were learned only from the training data.

The pipeline included:

* Missing-value handling
* Numerical and categorical feature separation
* Categorical encoding
* Feature transformation
* Model training
* Cross-validation
* Holdout evaluation

Keeping preprocessing inside the modeling pipeline reduced the risk of accidentally using information from the holdout set during training.

### 3. Compared classification models

I compared three model families:

* Logistic Regression
* Decision Tree
* Random Forest

Logistic Regression provided a useful interpretable baseline, while Decision Tree and Random Forest captured nonlinear relationships and interactions between fraud-related signals.

The models were compared using cross-validated PR-AUC. After selecting the best-performing model, I evaluated it once on the untouched holdout dataset.

Reported metrics included:

* PR-AUC
* ROC-AUC
* Accuracy
* Balanced accuracy
* Precision
* Recall
* Confusion matrix

### 4. Designed a fraud decision strategy

A fraud probability by itself is not a business decision.

I mapped the model scores into approve, decline, and manual-review bands using configurable probability thresholds. This allowed me to examine the tradeoff between:

* Approving legitimate applicants
* Preventing fraudulent approvals
* Sending uncertain applications to manual review
* Avoiding unnecessary declines

The thresholds are intentionally configurable because the best decision policy depends on the relative cost of fraud losses, review operations, customer friction, and false declines.

### 5. Evaluated third-party vendor data

The dataset included additional attributes supplied by an external fraud-data vendor.

To evaluate whether these features were economically useful, I compared two scenarios:

* A model trained with the vendor-provided features
* A model trained without those features

I then estimated the financial impact of each strategy using configurable assumptions for fraud losses, approval value, manual-review cost, and vendor cost.

This part of the project was important because a feature can improve predictive performance without necessarily creating enough financial value to justify its acquisition cost.

## Repository Structure

```text
fraud_detection/
    cleaning.py
    modeling.py
    profit.py

tests/
    test_cleaning.py
    test_modeling.py
    test_profit.py

prepare_application_metrics.py
plot_application_anomalies.py
clean_vendor_dataset.py
compare_fraud_models.py
evaluate_vendor_profit.py
requirements.txt
```

### Main scripts

| Script                           | Purpose                                                  |
| -------------------------------- | -------------------------------------------------------- |
| `prepare_application_metrics.py` | Creates daily application, product, and fraud metrics    |
| `plot_application_anomalies.py`  | Visualizes activity during a selected anomaly window     |
| `clean_vendor_dataset.py`        | Cleans and standardizes the vendor-enriched dataset      |
| `compare_fraud_models.py`        | Trains and compares the classification models            |
| `evaluate_vendor_profit.py`      | Compares decision economics with and without vendor data |

## Data

The original case-study workbook is not included in this repository because it contains application-level information that should not be published publicly.

To reproduce the analysis, place the following file in the repository root:

```text
FraudCaseStudy.xlsx
```

By default, the scripts expect two worksheets:

```text
p1-dataset
p2-dataset
```

The worksheet names can also be changed through the command-line arguments.

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/KianaShi/fraud-risk-analysis.git
cd fraud-risk-analysis

python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Analysis

### Generate application-level metrics

```bash
python prepare_application_metrics.py
```

### Plot an anomaly period

```bash
python plot_application_anomalies.py \
  --anomaly-start 2019-06-25 \
  --anomaly-end 2019-07-04
```

### Clean the vendor-enriched dataset

```bash
python clean_vendor_dataset.py
```

### Compare fraud models

```bash
python compare_fraud_models.py
```

### Evaluate the vendor-data strategy

```bash
python evaluate_vendor_profit.py \
  --approve-threshold 0.2 \
  --decline-threshold 0.8
```

Generated tables and figures are saved under:

```text
outputs/
```

## Testing

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic data and cover the main cleaning, modeling, and profit-analysis logic.

GitHub Actions runs the same test command on each push and pull request.

## Tools

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* unittest

## What I Learned

The most important lesson from this project was that fraud modeling is not only a classification problem.

A useful fraud system also requires careful handling of data leakage, appropriate evaluation metrics, threshold selection, operational review policies, and financial tradeoffs.

A model with slightly stronger predictive metrics may not be the best option if it creates too many false declines, requires excessive manual review, or depends on expensive external data. Evaluating the full decision process provided a much more realistic view of model value than comparing accuracy scores alone.
