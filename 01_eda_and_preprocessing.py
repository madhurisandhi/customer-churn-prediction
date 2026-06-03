import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# ─────────────────────────────────────────────
# Customer Churn Prediction
# Script 01: EDA & Preprocessing
# ─────────────────────────────────────────────

sns.set_theme(style="whitegrid")
os.makedirs("outputs/figures", exist_ok=True)


def generate_sample_data(n=1000):
    """Generate a Telco-style churn dataset."""
    np.random.seed(42)
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.85, 1.0, n), 2)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{1000+i}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], n),
        "senior_citizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "partner": np.random.choice(["Yes", "No"], n),
        "dependents": np.random.choice(["Yes", "No"], n),
        "tenure": tenure,
        "phone_service": np.random.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "paperless_billing": np.random.choice(["Yes", "No"], n),
        "payment_method": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "tech_support": np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
    })

    # Churn probability influenced by real-world drivers
    churn_prob = (
        0.05
        + 0.25 * (df["contract"] == "Month-to-month").astype(float)
        + 0.10 * (df["internet_service"] == "Fiber optic").astype(float)
        + 0.08 * (df["senior_citizen"] == 1).astype(float)
        - 0.15 * (df["tenure"] > 24).astype(float)
        + 0.05 * (df["monthly_charges"] > 80).astype(float)
    )
    churn_prob = churn_prob.clip(0.02, 0.95)
    df["churn"] = (np.random.rand(n) < churn_prob).astype(int)
    return df


def eda(df):
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    print(f"\nDataset shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean() * 100:.1f}%")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Churn by contract type
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    churn_contract = df.groupby("contract")["churn"].mean().reset_index()
    sns.barplot(data=churn_contract, x="contract", y="churn",
                palette="Reds_d", ax=axes[0])
    axes[0].set_title("Churn Rate by Contract Type", fontweight="bold")
    axes[0].set_ylabel("Churn Rate")
    axes[0].set_xlabel("")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Churn by tenure bucket
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72],
                                  labels=["0–12m", "13–24m", "25–48m", "49–72m"])
    churn_tenure = df.groupby("tenure_bucket")["churn"].mean().reset_index()
    sns.barplot(data=churn_tenure, x="tenure_bucket", y="churn",
                palette="Blues_d", ax=axes[1])
    axes[1].set_title("Churn Rate by Tenure", fontweight="bold")
    axes[1].set_ylabel("Churn Rate")
    axes[1].set_xlabel("")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Monthly charges distribution
    sns.boxplot(data=df, x="churn", y="monthly_charges",
                palette=["#2E74B5", "#C00000"], ax=axes[2])
    axes[2].set_title("Monthly Charges vs Churn", fontweight="bold")
    axes[2].set_xlabel("Churned")
    axes[2].set_xticklabels(["No", "Yes"])

    plt.suptitle("Churn EDA", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figures/churn_eda.png", dpi=150)
    print("\nSaved: churn_eda.png")
    plt.show()


def preprocess(df):
    """Encode, scale, and split data."""
    df = df.copy().drop(columns=["customer_id", "tenure_bucket"], errors="ignore")

    # Binary encoding
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["gender", "partner", "dependents", "phone_service",
                "paperless_billing"]:
        df[col] = df[col].map(binary_map)

    # One-hot encode multi-category columns
    df = pd.get_dummies(df, columns=["internet_service", "contract",
                                      "payment_method", "tech_support"], drop_first=True)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ["tenure", "monthly_charges", "total_charges"]
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")

    # Correlation heatmap on numeric columns only
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmax=0.8, center=0,
                annot=False, linewidths=0.3, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figures/correlation_heatmap.png", dpi=150)
    print("Saved: correlation_heatmap.png")
    plt.show()

    return X_train, X_test, y_train, y_test, X.columns.tolist()


if __name__ == "__main__":
    df = generate_sample_data(1000)
    eda(df)
    X_train, X_test, y_train, y_test, features = preprocess(df)
    print("\nPreprocessing complete. Ready for model training.")
