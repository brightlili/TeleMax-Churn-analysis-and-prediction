# TeleMax Customer Churn Prediction
### Machine Learning Model Testing and Evaluation Using Python

---

## Project Overview

This project was completed as part of a data science lab for **TeleMax**, a telecommunications company facing a high customer churn rate. The goal was to build, evaluate, and tune multiple machine learning models to predict which customers are likely to churn — enabling the marketing and customer relations teams to take proactive retention action.

---

## Business Problem

Customer churn directly impacts TeleMax's profitability and reputation. By identifying customers at high risk of leaving, the company can intervene early with targeted offers or improved service, reducing churn and increasing long-term revenue.

---

## Dataset

**IBM Telco Customer Churn Dataset**

| Property | Value |
|---|---|
| Source | [IBM on GitHub](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) |
| Rows | 7,043 customers |
| Columns | 21 features |
| Target variable | `Churn` (Yes / No) |
| Churn rate | ~26.5% (class imbalance present) |

**Key features include:** tenure, MonthlyCharges, TotalCharges, Contract type, PaymentMethod, InternetService, and various service subscriptions.

---

## Project Objectives

- Address **class imbalance** using SMOTE (Synthetic Minority Over-sampling Technique)
- Train and compare **four classification models**: Logistic Regression, Support Vector Machine, Decision Tree, XGBoost
- Evaluate models using **confusion matrices**, **accuracy scores**, and **AUC-ROC**
- **Tune XGBoost hyperparameters** using Grid Search Cross-Validation

---

## Repository Structure

```
├── churn_analysis.ipynb     # Main Jupyter notebook with all code and outputs
├── README.md                # Project documentation (this file)
├── requirements.txt         # Python dependencies
└── Report.pdf               # Final report with methodology and findings
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- GitHub Codespaces or VS Code with a virtual environment

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/telemax-churn-prediction.git
cd telemax-churn-prediction
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Jupyter
```bash
jupyter notebook
```

Select the **Python (TeleMax)** kernel when opening `churn_analysis.ipynb`.

---

## Requirements

`requirements.txt`:
```
jupyter
notebook
pandas
numpy
matplotlib
scikit-learn
xgboost
imbalanced-learn
seaborn
ipykernel
```

---

## Methodology

### Step 1 — Data Loading & Exploration
The dataset is loaded directly from IBM's GitHub repository. Initial exploration checks the shape, data types, and the distribution of the target variable (`Churn`).

```python
import pandas as pd
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
```

### Step 2 — Class Imbalance Check
A bar chart of the `Churn` column reveals ~73.5% non-churn vs ~26.5% churn — a significant imbalance that would bias models toward the majority class.

### Step 3 — Preprocessing
Before modelling, the data is cleaned and encoded:
- `customerID` is dropped (not a predictor)
- `TotalCharges` is converted from string to numeric; rows with nulls are dropped
- `Churn` is mapped: `Yes → 1`, `No → 0`
- All remaining categorical columns are one-hot encoded using `pd.get_dummies()`

### Step 4 — SMOTE (Handling Class Imbalance)
SMOTE synthetically creates new minority-class samples to balance the dataset before training.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=100)
X_smote, y_smote = smote.fit_resample(X, y)
```

### Step 5 — Standardization
All features are scaled to the same range using `StandardScaler`, which is essential for distance-based models like SVM and Logistic Regression.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_smote_scaled = scaler.fit_transform(X_smote)
```

### Step 6 — Train/Test Split
The balanced, scaled dataset is split 70% training / 30% testing.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_smote_scaled, y_smote, test_size=0.3, random_state=100
)
```

### Step 7 — Model Training
Four models are trained on the same training set:

| Model | Key Parameters |
|---|---|
| Logistic Regression | `random_state=100` |
| Support Vector Machine | `kernel='rbf'`, `random_state=100` |
| Decision Tree | `criterion='gini'`, `max_leaf_nodes=50` |
| XGBoost | `max_depth=2`, `random_state=100` |

### Step 8 — Evaluation

**Confusion Matrix** — shows true positives, true negatives, false positives, and false negatives for each model.

**Accuracy Score** — the percentage of correctly classified customers.

**AUC-ROC** — measures the model's ability to distinguish between churners and non-churners across all thresholds. A score of 1.0 is perfect; 0.5 is random.

### Step 9 — Hyperparameter Tuning (XGBoost)
Grid Search with 5-fold cross-validation is used to find the optimal XGBoost parameters:

```python
param_grid = {
    'learning_rate': [0.1, 0.01],
    'max_depth': [5, 7],
    'n_estimators': [200, 300],
    'subsample': [0.8, 0.9]
}
grid_search = GridSearchCV(XGBClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

The best parameters are then used to retrain and evaluate the final XGBoost model.

---

## Results Summary

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~80.65% | Solid baseline; interpretable |
| Support Vector Machine | — | Strong on high-dimensional data |
| Decision Tree | — | Fast; risk of overfitting |
| XGBoost (default) | — | Best baseline performance |
| XGBoost (tuned) | — | Highest accuracy after Grid Search |

> Note: Exact scores are in the notebook outputs after running all cells.

---

## Key Concepts Explained

**SMOTE** — Instead of simply duplicating minority-class rows, SMOTE creates *synthetic* new samples by interpolating between existing minority examples. This gives the model more varied data to learn from.

**StandardScaler** — Transforms all features to have a mean of 0 and standard deviation of 1. Without this, features with large numeric ranges (like `TotalCharges`) would dominate models unfairly.

**Confusion Matrix** — A 2×2 table showing: True Negatives (correctly predicted no churn), False Positives (predicted churn but didn't), False Negatives (missed churners), True Positives (correctly predicted churn). For churn prediction, False Negatives are especially costly — those are at-risk customers we failed to identify.

**AUC-ROC** — The Area Under the Receiver Operating Characteristic Curve. It measures model performance at every possible classification threshold, not just the default 0.5. Particularly useful when class imbalance is present.

**Grid Search CV** — An exhaustive search over a defined set of hyperparameter combinations, using cross-validation to evaluate each one. Prevents overfitting to the test set during tuning.

---

## Strategic Recommendations

1. **Deploy the tuned XGBoost model** as the primary churn predictor — it consistently outperforms simpler models.
2. **Focus retention efforts on month-to-month contract customers** — this group has the highest churn risk based on feature importance.
3. **Monitor customers in their first 12 months** — tenure is a strong predictor; early intervention is most effective.
4. **Re-train the model quarterly** as customer behaviour evolves over time.
5. **Track False Negatives carefully** — missed churners represent direct revenue loss and should be minimised.

---

## Author

Developed as part of the Udemy / IBM lab: *Churn Analysis and Prediction: Machine Learning Model Testing and Evaluation Using Python*

---

## License

This project is for educational purposes.