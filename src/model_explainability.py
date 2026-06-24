import shap
import matplotlib.pyplot as plt
import os


def generate_shap_explanations(
    model,
    X_sample
):
    """
    Generate SHAP explainability plots
    """

    print("=" * 60)
    print("🔍 GENERATING SHAP EXPLANATIONS")
    print("=" * 60)

    os.makedirs(
        "reports",
        exist_ok=True
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    print("✅ SHAP values generated")

    # Summary Plot
    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "reports/shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "✅ Saved: reports/shap_summary.png"
    )

    # Bar Plot
    plt.figure()

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "reports/shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "✅ Saved: reports/shap_bar.png"
    )

    return shap_values