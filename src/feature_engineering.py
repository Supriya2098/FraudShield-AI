import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    FraudShield AI
    Feature Engineering Module
    """

    def __init__(self, df):

        self.df = df.copy()

        self.scaler = StandardScaler()

        self.features_created = []

    # --------------------------------------------------
    # Time Features
    # --------------------------------------------------

    def create_time_features(self):

        print("⏰ Creating time-based features...")

        self.df['is_night_time'] = (
            (
                self.df['transaction_hour'] >= 22
            )
            |
            (
                self.df['transaction_hour'] <= 6
            )
        ).astype(int)

        self.df['is_business_hours'] = (
            (
                self.df['transaction_hour'] >= 9
            )
            &
            (
                self.df['transaction_hour'] <= 17
            )
        ).astype(int)

        self.df['is_weekend'] = (
            self.df['transaction_day_of_week']
            .isin([5, 6])
        ).astype(int)

        self.features_created.extend([
            'is_night_time',
            'is_business_hours',
            'is_weekend'
        ])

        return self

    # --------------------------------------------------
    # Amount Features
    # --------------------------------------------------

    def create_amount_features(self):

        print("💰 Creating amount-based features...")

        self.df['amount_log'] = np.log1p(
            self.df['transaction_amount']
        )

        self.df['high_amount_flag'] = (
            self.df['transaction_amount'] > 300
        ).astype(int)

        self.df['amount_category'] = pd.cut(
            self.df['transaction_amount'],
            bins=[-np.inf, 50, 100, 200, 500, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        self.features_created.extend([
            'amount_log',
            'high_amount_flag',
            'amount_category'
        ])

        return self

    # --------------------------------------------------
    # Customer Features
    # --------------------------------------------------

    def create_customer_features(self):

        print("👤 Creating customer features...")

        customer_stats = (
            self.df
            .groupby('customer_id')['transaction_amount']
            .agg(['mean', 'std', 'max'])
        )

        customer_stats.columns = [
            'customer_avg_amount',
            'customer_std_amount',
            'customer_max_amount'
        ]

        self.df = self.df.join(
            customer_stats,
            on='customer_id'
        )

        self.df['customer_std_amount'] = (
            self.df['customer_std_amount']
            .fillna(0)
        )

        self.df['unusual_amount_for_customer'] = (
            abs(
                (
                    self.df['transaction_amount']
                    -
                    self.df['customer_avg_amount']
                )
                /
                (
                    self.df['customer_std_amount']
                    + 1e-5
                )
            ) > 3
        ).astype(int)

        self.features_created.extend([
            'customer_avg_amount',
            'customer_std_amount',
            'customer_max_amount',
            'unusual_amount_for_customer'
        ])

        return self

    # --------------------------------------------------
    # Frequency Features
    # --------------------------------------------------

    def create_frequency_features(self):

        print("📊 Creating frequency features...")

        customer_freq = (
            self.df['customer_id']
            .value_counts()
        )

        merchant_freq = (
            self.df['merchant_id']
            .value_counts()
        )

        category_freq = (
            self.df['merchant_category']
            .value_counts()
        )

        self.df['customer_transaction_count'] = (
            self.df['customer_id']
            .map(customer_freq)
        )

        self.df['merchant_transaction_count'] = (
            self.df['merchant_id']
            .map(merchant_freq)
        )

        self.df['category_transaction_count'] = (
            self.df['merchant_category']
            .map(category_freq)
        )

        self.features_created.extend([
            'customer_transaction_count',
            'merchant_transaction_count',
            'category_transaction_count'
        ])

        return self

    # --------------------------------------------------
    # Behavioral Features
    # --------------------------------------------------

    def create_behavioral_features(self):

        print("🎯 Creating behavioral features...")

        self.df['card_type_numeric'] = (
            self.df['card_type']
            .map({
                'Debit': 0,
                'Credit': 1
            })
        )

        self.df['transaction_type_numeric'] = (
            self.df['transaction_type']
            .map({
                'Online': 0,
                'Offline': 1,
                'Mobile': 2
            })
        )

        self.features_created.extend([
            'card_type_numeric',
            'transaction_type_numeric'
        ])

        return self

    # --------------------------------------------------
    # Interaction Features
    # --------------------------------------------------

    def create_interaction_features(self):

        print("🔗 Creating interaction features...")

        self.df['frequency_amount_ratio'] = (
            self.df['customer_transaction_count']
            /
            (
                self.df['transaction_amount']
                + 1
            )
        )

        self.features_created.append(
            'frequency_amount_ratio'
        )

        return self

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------

    def scale_features(self):

        print("📏 Scaling numerical features...")

        scale_cols = [
            'transaction_amount',
            'amount_log',
            'customer_avg_amount',
            'customer_std_amount',
            'customer_max_amount',
            'customer_transaction_count',
            'merchant_transaction_count',
            'category_transaction_count',
            'frequency_amount_ratio'
        ]

        scale_cols = [
            col
            for col in scale_cols
            if col in self.df.columns
        ]

        self.df[scale_cols] = (
            self.scaler.fit_transform(
                self.df[scale_cols]
            )
        )

        return self

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    def get_feature_list(self):

        return self.features_created

    def get_engineered_data(self):

        return self.df


# ==================================================
# PIPELINE
# ==================================================

def engineer_features(df):

    print("\n" + "=" * 60)
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    engineer = FeatureEngineer(df)

    df_engineered = (
        engineer
        .create_time_features()
        .create_amount_features()
        .create_customer_features()
        .create_frequency_features()
        .create_behavioral_features()
        .create_interaction_features()
        .scale_features()
        .get_engineered_data()
    )

    print(
        f"\n✅ Features Created: "
        f"{len(engineer.get_feature_list())}"
    )

    print(
        f"✅ Total Columns: "
        f"{df_engineered.shape[1]}"
    )

    return df_engineered, engineer