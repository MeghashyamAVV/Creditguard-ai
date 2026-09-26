# CreditGuard AI

### Explainable Credit Risk & Loan Default Prediction

An end-to-end machine learning project for predicting loan defaults using **XGBoost**, with model evaluation and **SHAP-based interpretability** to understand the factors driving individual predictions.

## Results

Results on this dataset containing **32,581 loan applications** using an **80/20 train-test split**:

| Metric                      |     Value |
| --------------------------- | --------: |
| ROC-AUC                     | **0.951** |
| F1-score (threshold = 0.5)  | **0.818** |
| F1-score (best threshold)   | **0.844** |
| Precision (threshold = 0.5) | **0.834** |
| Recall (threshold = 0.5)    | **0.802** |

Exact metrics are regenerated each run and saved to:

`outputs/reports/metrics.json`

## Project Structure

```text
credit_risk_project/
│
├── data/
│   └── credit_risk_dataset.csv
│
├── src/
│   ├── data_preprocessing.py
│   │   └── Data cleaning, feature engineering, encoding,
│   │       and train/test splitting
│   │
│   ├── train_model.py
│   │   └── XGBoost training and RandomizedSearchCV tuning
│   │
│   ├── evaluate.py
│   │   └── Model evaluation, metrics, and visualization
│   │
│   ├── shap_analysis.py
│   │   └── SHAP global and local model interpretability
│   │
│   └── main.py
│       └── Runs the complete pipeline end-to-end
│
├── outputs/
│   ├── models/
│   │   └── Saved trained model (.joblib)
│   │
│   ├── plots/
│   │   └── ROC, PR, confusion matrix, and SHAP plots
│   │
│   └── reports/
│       ├── metrics.json
│       ├── classification_report.txt
│       └── top_risk_drivers.json
│
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

From the project directory:

```bash
cd credit_risk_project
python src/main.py
```

The complete pipeline will then run automatically.

## Pipeline

### 1. Data Preprocessing

The preprocessing stage:

* Cleans invalid records, such as `person_age > 100`
* Handles missing values through imputation
* Engineers two ratio-based features:

  * `income_to_loan_ratio`
  * `age_to_credit_hist_ratio`
* One-hot encodes categorical variables
* Splits the dataset into training and test sets

### 2. Model Training

An **XGBoost classifier** is trained with hyperparameter tuning using 5-fold cross-validated `RandomizedSearchCV`.

The search includes:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`
* `min_child_weight`
* `gamma`

Hyperparameter selection is optimized using **ROC-AUC**.

Class imbalance is addressed using XGBoost's `scale_pos_weight` parameter.

### 3. Model Evaluation

The trained model is evaluated on the held-out test set using:

* ROC-AUC
* F1-score
* Precision
* Recall
* Confusion matrix
* ROC curve
* Precision-recall curve

Two classification thresholds are reported:

* **Default threshold:** `0.5`
* **Best-F1 threshold:** threshold producing the highest F1-score on the evaluation data

### 4. Model Interpretability

**SHAP (SHapley Additive exPlanations)** is used to understand how individual features influence model predictions.

The project generates:

* Global SHAP beeswarm summary plot
* Mean absolute SHAP feature importance
* Feature importance ranking in JSON format
* Waterfall plot for a high-risk applicant
* Local explanation of an individual prediction

## Key Risk Drivers

Based on mean absolute SHAP values, the leading features influencing the model's predictions include:

1. `income_to_loan_ratio`
2. Applicant income
3. Loan-to-income percentage
4. Interest rate
5. Home ownership status

The complete feature ranking is available in:

```text
outputs/reports/top_risk_drivers.json
```

The corresponding SHAP visualization is available at:

```text
outputs/plots/shap_feature_importance.png
```

## Output Files

After running the pipeline, the `outputs/` directory contains the generated model, evaluation results, and interpretability artifacts.

```text
outputs/
├── models/
│   └── *.joblib
│
├── plots/
│   ├── ROC curve
│   ├── Precision-Recall curve
│   ├── Confusion matrix
│   ├── SHAP summary plot
│   ├── SHAP feature importance
│   └── SHAP waterfall plot
│
└── reports/
    ├── metrics.json
    ├── classification_report.txt
    └── top_risk_drivers.json
```

## Dataset

The prediction target is:

```text
loan_status
```

where:

* `1` = Default
* `0` = Non-default

The dataset contains **32,581 loan applications**.

## Threshold Selection

The project reports metrics at both the default `0.5` threshold and the threshold that produces the best F1-score.

This provides a view of how model performance changes depending on the classification threshold. In credit-risk applications, changing the threshold can affect the balance between **false positives and false negatives**, so the appropriate operating point depends on the specific business and risk requirements.

## Running on Another Dataset

To use a different dataset, replace:

```text
data/credit_risk_dataset.csv
```

The replacement CSV should contain the same required columns and target variable.

Alternatively, update the dataset path in:

```text
src/main.py
```

## Tech Stack

* **Python**
* **XGBoost**
* **Scikit-learn**
* **SHAP**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Joblib**

## Disclaimer

This project is intended for **educational and demonstration purposes**. The reported results are specific to this dataset and evaluation setup and should not be interpreted as evidence of production-level credit decisioning performance.

