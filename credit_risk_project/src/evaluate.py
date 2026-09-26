import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)


def find_best_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    best_idx = np.argmax(f1_scores[:-1])  # last point has no threshold
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate_model(model, X_test, y_test, out_dir="outputs"):
    y_proba = model.predict_proba(X_test)[:, 1]

    # Default 0.5 threshold metrics
    y_pred_default = (y_proba >= 0.5).astype(int)

    # F1-optimal threshold (useful for imbalanced credit risk data)
    best_thresh, best_f1 = find_best_threshold(y_test, y_proba)
    y_pred_best = (y_proba >= best_thresh).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "threshold_0.5": {
            "f1_score": f1_score(y_test, y_pred_default),
            "precision": precision_score(y_test, y_pred_default),
            "recall": recall_score(y_test, y_pred_default),
            "accuracy": accuracy_score(y_test, y_pred_default),
        },
        "best_threshold": {
            "threshold": float(best_thresh),
            "f1_score": f1_score(y_test, y_pred_best),
            "precision": precision_score(y_test, y_pred_best),
            "recall": recall_score(y_test, y_pred_best),
            "accuracy": accuracy_score(y_test, y_pred_best),
        },
    }

    print(json.dumps(metrics, indent=2))

    with open(f"{out_dir}/reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(f"{out_dir}/reports/classification_report.txt", "w") as f:
        f.write("=== Threshold = 0.5 ===\n")
        f.write(classification_report(y_test, y_pred_default))
        f.write(f"\n=== Threshold = {best_thresh:.3f} (F1-optimal) ===\n")
        f.write(classification_report(y_test, y_pred_best))

    _plot_confusion_matrix(y_test, y_pred_default, f"{out_dir}/plots/confusion_matrix.png")
    _plot_roc_curve(y_test, y_proba, metrics["roc_auc"], f"{out_dir}/plots/roc_curve.png")
    _plot_pr_curve(y_test, y_proba, f"{out_dir}/plots/precision_recall_curve.png")

    return metrics, y_proba


def _plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix (threshold = 0.5)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_roc_curve(y_true, y_proba, auc_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_pr_curve(y_true, y_proba, path):
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
