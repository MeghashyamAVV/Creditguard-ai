import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

TARGET_COL = "loan_status"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Sanity filters on known bad values in this dataset
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"] <= 60]

    # Impute missing numeric values with the median
    for col in ["person_emp_length", "loan_int_rate"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["income_to_loan_ratio"] = df["person_income"] / (df["loan_amnt"] + 1)
    df["age_to_credit_hist_ratio"] = df["person_age"] / (
        df["cb_person_cred_hist_length"] + 1
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns; keeps feature names SHAP-friendly."""
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    return df


def prepare_dataset(path: str, test_size: float = 0.2, random_state: int = 42):
    
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, list(X.columns)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = prepare_dataset(
        "data/credit_risk_dataset.csv"
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Default rate (train): {y_train.mean():.3f}")
    print(f"Number of features after encoding: {len(features)}")
