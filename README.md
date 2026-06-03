# 📉 Customer Churn Prediction Model

**Tools:** Python · Pandas · Scikit-learn · Seaborn · Matplotlib · SQL  
**Domain:** Telecom · Machine Learning · Classification  
**Status:** Completed

---

## 📌 Project Overview

Built a machine learning pipeline to predict customer churn for a telecom company. Trained and compared **Logistic Regression** and **Random Forest** classifiers, achieving **85%+ accuracy**. Applied **SMOTE oversampling** to handle class imbalance, improving minority class (churned customers) recall by **18%**.

---

## 🎯 Objectives

- Identify customers at high risk of churning before they leave
- Determine the top features driving churn behaviour
- Handle class imbalance (churned customers are the minority)
- Deliver a model that maximizes recall for the churn class (false negatives are costly)

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall (Churn) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.2% | 0.76 | 0.71 | 0.73 | 0.84 |
| **Random Forest** | **85.4%** | **0.83** | **0.79** | **0.81** | **0.91** |
| Random Forest + SMOTE | 84.1% | 0.79 | **0.87** | 0.83 | 0.92 |

> **Best model for production:** Random Forest + SMOTE — highest recall on churn class, minimizing missed at-risk customers.

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── telecom_churn_sample.csv   # Sample dataset (Telco-style)
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.py
│   ├── 02_model_training.py
│   └── 03_model_evaluation.py
│
├── outputs/
│   └── figures/                   # ROC curves, confusion matrices, feature importance
│
├── requirements.txt
└── README.md
```

---

## 🔍 Approach

### 1. Exploratory Data Analysis
- Analyzed churn rate by contract type, tenure, monthly charges, and internet service
- Visualized distributions and correlations using Seaborn heatmaps and box plots
- Found contract type and tenure as strongest churn predictors

### 2. Data Preprocessing
- Encoded categorical variables using Label Encoding and One-Hot Encoding
- Scaled numerical features using StandardScaler
- Split data 80/20 for train/test with stratification on the target class

### 3. Handling Class Imbalance with SMOTE
- Churn class was ~26% of data — models without balancing were biased toward non-churn
- Applied SMOTE (Synthetic Minority Over-sampling Technique) on training set only
- Improved churn class recall from 0.69 → 0.87 (+18%)

### 4. Model Training & Comparison
- Trained Logistic Regression (baseline) and Random Forest
- Used `GridSearchCV` for hyperparameter tuning on Random Forest
- Evaluated with Accuracy, Precision, Recall, F1, and ROC-AUC

### 5. Interpretability
- Plotted feature importances from Random Forest
- Top drivers: **Monthly Charges, Tenure, Contract Type, Tech Support, Internet Service**

---

## ⚙️ How to Run

```bash
git clone https://github.com/madhuri-sandhi/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
python notebooks/01_eda_and_preprocessing.py
python notebooks/02_model_training.py
python notebooks/03_model_evaluation.py
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
imbalanced-learn>=0.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## 💡 Key Learnings

- SMOTE must be applied **only on training data** — applying it before splitting leaks information and inflates metrics
- Recall is the right metric for churn — a missed churner (false negative) costs more than a false alarm
- Feature importance from tree models is a fast, explainable starting point for stakeholder conversations
