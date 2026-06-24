import pickle
import pandas as pd


class FraudPredictor:

    def __init__(
        self,
        model_path="models/random_forest.pkl"
    ):

        with open(
            model_path,
            "rb"
        ) as f:

            self.model = pickle.load(f)

    def predict(
        self,
        transaction_df
    ):

        probability = self.model.predict_proba(
            transaction_df
        )[:, 1][0]

        prediction = (
            "Fraud"
            if probability >= 0.50
            else "Genuine"
        )

        return {
            "prediction": prediction,
            "fraud_probability": round(
                float(probability),
                4
            )
        }

    def batch_predict(
        self,
        transactions_df
    ):

        probabilities = self.model.predict_proba(
            transactions_df
        )[:, 1]

        predictions = [
            "Fraud"
            if p >= 0.50
            else "Genuine"
            for p in probabilities
        ]

        output = transactions_df.copy()

        output["fraud_probability"] = probabilities
        output["prediction"] = predictions

        return output