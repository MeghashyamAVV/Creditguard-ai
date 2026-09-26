import time
from train_model import train
from evaluate import evaluate_model
from shap_analysis import run_shap_analysis


def main():
    t0 = time.time()

    print("=" * 60)
    print("STEP 1/3: Training XGBoost model (with hyperparameter search)")
    print("=" * 60)
    model, X_train, X_test, y_train, y_test, feature_names = train()

    print("\n" + "=" * 60)
    print("STEP 2/3: Evaluating model performance")
    print("=" * 60)
    metrics, y_proba = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("STEP 3/3: Running SHAP interpretability analysis")
    print("=" * 60)
    top_drivers, shap_values = run_shap_analysis(model, X_test)

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1 (threshold=0.5): {metrics['threshold_0.5']['f1_score']:.4f}")
    print(f"F1 (best threshold): {metrics['best_threshold']['f1_score']:.4f}")


if __name__ == "__main__":
    main()
