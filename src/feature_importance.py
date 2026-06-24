import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(
    model,
    feature_names,
    top_n=20
):

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    print("\n🏆 Top Features")
    print(
        importance_df.head(top_n)
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        importance_df["Feature"].head(top_n)[::-1],
        importance_df["Importance"].head(top_n)[::-1]
    )

    plt.title(
        "Top Feature Importance"
    )

    plt.xlabel(
        "Importance Score"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/feature_importance.png",
        dpi=300
    )

    plt.show()

    importance_df.to_csv(
        "reports/top_features.csv",
        index=False
    )

    return importance_df