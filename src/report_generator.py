import os
import pandas as pd


class FraudReportGenerator:

    def __init__(self):

        os.makedirs(
            "reports",
            exist_ok=True
        )

    def generate_report(
        self,
        prediction_df
    ):

        total_transactions = len(
            prediction_df
        )

        fraud_transactions = len(
            prediction_df[
                prediction_df["prediction"] == "Fraud"
            ]
        )

        fraud_rate = (
            fraud_transactions
            / total_transactions
        ) * 100

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Total Transactions",
                    "Fraud Transactions",
                    "Fraud Rate (%)"
                ],
                "Value": [
                    total_transactions,
                    fraud_transactions,
                    round(
                        fraud_rate,
                        2
                    )
                ]
            }
        )

        summary.to_csv(
            "reports/fraud_summary.csv",
            index=False
        )

        prediction_df.to_csv(
            "reports/fraud_predictions.csv",
            index=False
        )

        top_fraud = (
            prediction_df
            .sort_values(
                by="fraud_probability",
                ascending=False
            )
            .head(20)
        )

        top_fraud.to_csv(
            "reports/top_fraud_cases.csv",
            index=False
        )

        with open(
            "reports/fraud_report.txt",
            "w"
        ) as f:

            f.write(
                "FraudShield AI Report\n"
            )

            f.write(
                "=" * 40 + "\n"
            )

            f.write(
                f"Total Transactions: {total_transactions}\n"
            )

            f.write(
                f"Fraud Transactions: {fraud_transactions}\n"
            )

            f.write(
                f"Fraud Rate: {fraud_rate:.2f}%\n"
            )

        print(
            "✅ Fraud report generated"
        )

        print(
            f"Total Transactions: {total_transactions:,}"
        )

        print(
            f"Fraud Transactions: {fraud_transactions:,}"
        )

        print(
            f"Fraud Rate: {fraud_rate:.2f}%"
        )

        return summary