import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def run_shap_analysis(model, X_test, out_dir="outputs", top_n=15):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- Global summary (beeswarm) plot ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plots/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Mean |SHAP value| bar chart (top risk drivers) ---
    plt.figure()
    shap.summary_plot(
        shap_values, X_test, plot_type="bar", show=False, max_display=top_n
    )
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plots/shap_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Rank features by mean |SHAP| and save as JSON ---
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranking = sorted(
        zip(X_test.columns, mean_abs_shap), key=lambda x: x[1], reverse=True
    )
    top_drivers = [{"feature": f, "mean_abs_shap": float(v)} for f, v in ranking[:top_n]]

    with open(f"{out_dir}/reports/top_risk_drivers.json", "w") as f:
        json.dump(top_drivers, f, indent=2)

    print("Top risk drivers (by mean |SHAP value|):")
    for item in top_drivers:
        print(f"  {item['feature']:<35} {item['mean_abs_shap']:.4f}")

    # --- Local explanation for a single high-risk example ---
    highest_risk_idx = int(np.argmax(model.predict_proba(X_test)[:, 1]))
    plt.figure()
    shap.plots.waterfall(shap_values[highest_risk_idx], show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plots/shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close()

    return top_drivers, shap_values
