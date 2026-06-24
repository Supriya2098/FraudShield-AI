import pandas as pd
from src.data_cleaner import DataCleaner


def load_and_clean_data(filepath):
    """
    Load raw data and perform cleaning
    """

    print("=" * 60)
    print("📥 LOADING AND CLEANING DATA")
    print("=" * 60)

    print(f"\n📂 Loading data from {filepath}...")

    df = pd.read_csv(
        filepath,
        parse_dates=["timestamp"]
    )

    print(
        f"✅ Loaded {len(df):,} records "
        f"with {len(df.columns)} columns"
    )

    cleaner = DataCleaner(df)

    df_cleaned = (
        cleaner
        .handle_missing_values()
        .remove_duplicates()
        .handle_outliers()
        .get_cleaned_data()
    )

    cleaner.generate_report()

    return df_cleaned, cleaner