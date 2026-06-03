import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
import os

# ─────────────────────────────────────────────
# Customer Churn Prediction
# Script 02: Model Training & Evaluation
# ─────────────────────────────────────────────

os.makedirs("outputs/figures", exist_ok=True)


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):

    results = {}

    # ── 1. Baseline: Logistic Regression ──
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    results["Logistic Regression"] = {
        "model": lr, "y_pred": y_pred_lr, "y_prob": y_prob_lr
    }

    # ── 2. Random Forest (tuned) ──
    print("Training Random Forest with GridSearchCV...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5]
    }
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
    print(f"Best RF params: {grid_search.best_params_}")

    results["Random Forest"] = {
        "model": best_rf, "y_pred": y_pred_rf, "y_prob": y_prob_rf
    }

    # ── 3. Random Forest + SMOTE ──
    print("Training Random Forest + SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE — Churn: {y_res.sum()} | Non-Churn: {(y_res == 0).sum()}")

    rf_smote = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    rf_smote.fit(X_res, y_res)
    y_pred_sm = rf_smote.predict(X_test)
    y_prob_sm = rf_smote.predict_proba(X_test)[:, 1]

    results["RF + SMOTE"] = {
        "model": rf_smote, "y_pred": y_pred_sm, "y_prob": y_prob_sm
    }

    # ── Print classification reports ──
    for name, r in results.items():
        print(f"\n{'='*50}")
        print(f"{name}")
        print('='*50)
        print(classification_report(y_test, r["y_pred"], target_names=["No Churn", "Churn"]))
        print(f"ROC-AUC: {roc_auc_score(y_test, r['y_prob']):.4f}")

    # ── ROC Curve comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#4472C4", "#ED7D31", "#70AD47"]
    for (name, r), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        auc = roc_auc_score(y_test, r["y_prob"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_title("ROC Curve Comparison", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Confusion matrix for best model (RF + SMOTE)
    cm = confusion_matrix(y_test, results["RF + SMOTE"]["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Confusion Matrix — RF + SMOTE", fontweight="bold")

    plt.tight_layout()
    plt.savefig("outputs/figures/roc_and_confusion.png", dpi=150)
    print("\nSaved: roc_and_confusion.png")
    plt.show()

    # ── Feature Importance ──
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": best_rf.feature_importances_
    }).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=fi, x="importance", y="feature", palette="Blues_d", ax=ax)
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig("outputs/figures/feature_importance.png", dpi=150)
    print("Saved: feature_importance.png")
    plt.show()

    return results


if __name__ == "__main__":
    # Load preprocessed data — run 01_eda_and_preprocessing.py first
    from notebooks.eda_and_preprocessing import generate_sample_data, preprocess, eda
    df = generate_sample_data(1000)
    eda(df)
    X_train, X_test, y_train, y_test, features = preprocess(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test, features)
