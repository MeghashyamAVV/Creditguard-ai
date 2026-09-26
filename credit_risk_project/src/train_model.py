import json
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from data_preprocessing import prepare_dataset

RANDOM_STATE = 42

PARAM_DISTRIBUTIONS = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
}


def get_base_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        tree_method="hist",
    )


def train(
    data_path: str = "data/credit_risk_dataset.csv",
    model_out: str = "outputs/models/xgb_credit_risk_model.joblib",
    n_iter: int = 25,
):
    X_train, X_test, y_train, y_test, feature_names = prepare_dataset(data_path)

    # Handle class imbalance (defaults are usually the minority class)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    base_model = get_base_model(scale_pos_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print("Best params:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

    joblib.dump(
        {"model": best_model, "feature_names": feature_names}, model_out
    )

    with open("outputs/reports/best_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=2)

    return best_model, X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    train()
