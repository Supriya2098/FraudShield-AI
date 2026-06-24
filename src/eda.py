import matplotlib.pyplot as plt
import seaborn as sns


class FraudEDA:

    def __init__(self, df):
        self.df = df

    def fraud_distribution(self):

        plt.figure(figsize=(8, 5))

        sns.countplot(
            x='is_fraud',
            data=self.df
        )

        plt.title("Fraud vs Genuine Transactions")
        plt.xlabel("Fraud Status")
        plt.ylabel("Count")
        plt.savefig(
        "reports/01_fraud_distribution.png",
        dpi=300,
        bbox_inches="tight"
        )

        plt.show()

        print(
            self.df['is_fraud']
            .value_counts()
        )

    def transaction_amount_distribution(self):

        plt.figure(figsize=(10, 5))

        sns.histplot(
            self.df['transaction_amount'],
            bins=50,
            kde=True
        )

        plt.title("Transaction Amount Distribution")
        plt.xlabel("Transaction Amount")
        plt.ylabel("Frequency")
        plt.savefig(
        "reports/02_transaction_amount_distribution.png",
        dpi=300,
        bbox_inches="tight"
        )

        plt.show()

        print(
            self.df['transaction_amount']
            .describe()
        )
    def fraud_amount_comparison(self):

        plt.figure(figsize=(8,5))

        sns.boxplot(
        x='is_fraud',
        y='transaction_amount',
        data=self.df
        )

        plt.title("Fraud vs Genuine Transaction Amounts")
        plt.savefig(
        "reports/03_fraud_amount_comparison.png",
        dpi=300,
        bbox_inches="tight"
     )

        plt.show()
    def fraud_by_country(self):

        fraud_country = (
        self.df[self.df['is_fraud'] == 1]
        ['country']
        .value_counts()
        .head(10)
        )

        plt.figure(figsize=(10,5))

        fraud_country.plot(
        kind='bar'
        )

        plt.title("Top Countries with Fraud Transactions")
        plt.xlabel("Country")
        plt.ylabel("Fraud Count")
        plt.savefig(
        "reports/04_fraud_by_country.png",
        dpi=300,
        bbox_inches="tight"
        )

        plt.show()

        print(fraud_country)
    def fraud_by_transaction_type(self):

        fraud_txn = (
        self.df[self.df['is_fraud'] == 1]
        ['transaction_type']
        .value_counts()
        )

        plt.figure(figsize=(8,5))

        fraud_txn.plot(
        kind='bar'
        )

        plt.title("Fraud by Transaction Type")
        plt.xlabel("Transaction Type")
        plt.ylabel("Fraud Count")
        plt.savefig(
        "reports/05_fraud_by_transaction_type.png",
        dpi=300,
        bbox_inches="tight"
        )

        plt.show()

        print(fraud_txn)  
    def fraud_by_hour(self):

        fraud_hour = (
        self.df[self.df['is_fraud'] == 1]
        ['transaction_hour']
        .value_counts()
        .sort_index()
        )

        plt.figure(figsize=(12,5))

        fraud_hour.plot(
        kind='line',
        marker='o'
        )

        plt.title("Fraud Transactions by Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Fraud Count")

        plt.grid(True)
        plt.savefig(
        "reports/06_fraud_by_hour.png",
        dpi=300,
        bbox_inches="tight"
      )

        plt.show()

        print(fraud_hour) 
    def correlation_heatmap(self):

        numeric_df = self.df.select_dtypes(
        include=['int64', 'float64']
        )

        plt.figure(figsize=(10,8))

        sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap='coolwarm',
        fmt='.2f'
        )

        plt.title("Correlation Heatmap")
        plt.savefig(
        "reports/07_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
        )

        plt.show()             
        