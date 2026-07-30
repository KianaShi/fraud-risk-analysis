# Fraud Risk Analysis and Vendor Evaluation

## Overview

This project analyzes fraud risk in payment applications using behavioral, device, and temporal features. It includes exploratory data analysis, data preprocessing, machine learning model training, and model evaluation to identify fraudulent applications and support vendor risk assessment.

---

## Features

- Exploratory data analysis (EDA)
- Data preprocessing and feature engineering
- Fraud detection using Logistic Regression and Random Forest
- Model evaluation with ROC-AUC, Precision, Recall, and Accuracy
- Data visualization for fraud patterns and model performance

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Project Structure

```
fraud-risk-analysis/
│
├── data/
│   ├── p1_daily_metrics.csv
│   ├── p1_daily_metrics_by_product.csv
│   └── p2_dataset_clean.csv
│
├── src/
│   ├── exploratory_analysis.py
│   ├── preprocess_data.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── figures/
│
├── README.md
└── requirements.txt
```

---

## Workflow

```
Raw Data
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Cleaning & Feature Engineering
      │
      ▼
Model Training
(Logistic Regression / Random Forest)
      │
      ▼
Model Evaluation
      │
      ▼
Fraud Risk Analysis
```

---

## Results

The project compares multiple machine learning models for fraud detection.

Example evaluation metrics:

| Model | Accuracy | ROC-AUC |
|--------|---------:|--------:|
| Logistic Regression | 0.80 | 0.89 |
| Random Forest | ... | ... |

---

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- XGBoost / LightGBM comparison
- SHAP feature importance analysis
- Interactive dashboard for fraud monitoring

---

## Author

Kiana Shi
