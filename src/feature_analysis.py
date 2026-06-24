import pandas as pd


def analyze_top_features(
    importance_df,
    top_n=20
):

    print("\n" + "="*60)
    print("TOP FEATURE ANALYSIS")
    print("="*60)

    top_features = importance_df.head(top_n)

    for i, row in enumerate(
        top_features.itertuples(),
        start=1
    ):

        print(
            f"{i}. {row.Feature} "
            f"({row.Importance:.4f})"
        )

    top_features.to_csv(
        "reports/top_20_features.csv",
        index=False
    )

    print(
        "\n✅ Saved: reports/top_20_features.csv"
    )

    return top_features