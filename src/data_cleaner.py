import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Comprehensive Data Cleaning & Preparation Module
    """

    def __init__(self, df):
        self.df = df.copy()
        self.report = {}
        self.encoders = {}

    def handle_missing_values(self, strategy='median'):

        print("🔍 Handling missing values...")

        initial_missing = self.df.isnull().sum()
        self.report['missing_values_before'] = initial_missing

        numeric_cols = self.df.select_dtypes(
            include=[np.number]
        ).columns

        for col in numeric_cols:

            if self.df[col].isnull().sum() > 0:

                if strategy == 'mean':
                    self.df[col].fillna(
                        self.df[col].mean(),
                        inplace=True
                    )
                else:
                    self.df[col].fillna(
                        self.df[col].median(),
                        inplace=True
                    )

        categorical_cols = self.df.select_dtypes(
            include=['object']
        ).columns

        for col in categorical_cols:

            if self.df[col].isnull().sum() > 0:

                self.df[col].fillna(
                    self.df[col].mode()[0],
                    inplace=True
                )

        final_missing = self.df.isnull().sum()

        self.report['missing_values_after'] = final_missing

        print("✅ Missing values handled")
        print(
            f"Missing values: "
            f"{initial_missing.sum()} → {final_missing.sum()}"
        )

        return self

    def remove_duplicates(self):

        print("🔍 Removing duplicates...")

        initial_rows = len(self.df)

        self.df.drop_duplicates(
            subset=['transaction_id'],
            inplace=True
        )

        final_rows = len(self.df)

        duplicates_removed = initial_rows - final_rows

        self.report['duplicates_removed'] = duplicates_removed

        print(
            f"✅ Duplicates removed: "
            f"{duplicates_removed} records"
        )

        return self

    def handle_outliers(
        self,
        method='iqr',
        threshold=1.5
    ):

        print("🔍 Handling outliers...")

        numeric_cols = ['transaction_amount']

        outliers_count = 0

        for col in numeric_cols:

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outlier_mask = (
                (self.df[col] < lower_bound)
                |
                (self.df[col] > upper_bound)
            )

            outliers_count += outlier_mask.sum()

            self.df[col] = self.df[col].clip(
                lower=lower_bound,
                upper=upper_bound
            )

        self.report['outliers_handled'] = int(outliers_count)

        print(
            f"✅ Outliers handled: "
            f"{outliers_count} values capped"
        )

        return self

    def encode_categorical(self):

        print("🔍 Encoding categorical variables...")

        categorical_cols = self.df.select_dtypes(
            include=['object']
        ).columns

        categorical_cols = [
            col for col in categorical_cols
            if col not in [
                'transaction_id',
                'customer_id',
                'merchant_id'
            ]
        ]

        for col in categorical_cols:

            encoder = LabelEncoder()

            self.df[col + '_encoded'] = encoder.fit_transform(
                self.df[col]
            )

            self.encoders[col] = encoder

        print(
            f"✅ Encoded {len(categorical_cols)} categorical columns"
        )

        return self

    def normalize_features(self):

        print("🔍 Normalizing features...")

        numeric_cols = ['transaction_amount']

        scaler = StandardScaler()

        self.df[numeric_cols] = scaler.fit_transform(
            self.df[numeric_cols]
        )

        self.report['scaler'] = scaler

        print("✅ Features normalized")

        return self

    def generate_report(self):

        print("\n" + "=" * 60)
        print("📊 DATA QUALITY REPORT")
        print("=" * 60)

        print(f"Total Records: {len(self.df):,}")

        print(
            f"Duplicates Removed: "
            f"{self.report.get('duplicates_removed', 0)}"
        )

        missing_before = (
            self.report['missing_values_before'].sum()
            if 'missing_values_before' in self.report
            else 0
        )

        missing_after = (
            self.report['missing_values_after'].sum()
            if 'missing_values_after' in self.report
            else 0
        )

        print(
            f"Missing Values: "
            f"{missing_before} → {missing_after}"
        )

        print(
            f"Outliers Handled: "
            f"{self.report.get('outliers_handled', 0)}"
        )

        print(
            f"Fraud Cases: "
            f"{self.df['is_fraud'].sum():,}"
        )

        print(
            f"Fraud Percentage: "
            f"{self.df['is_fraud'].mean()*100:.2f}%"
        )

        print("=" * 60)

        return self.report

    def get_cleaned_data(self):
        return self.df