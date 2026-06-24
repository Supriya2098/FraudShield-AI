import os
import pickle
import warnings

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    FraudShield AI
    Model Training & Evaluation Module
    """

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

        self.models = {}
        self.results = {}

    def evaluate_model(
        self,
        model_name,
        y_pred,
        y_pred_proba
    ):

        accuracy = accuracy_score(
            self.y_test,
            y_pred
        )

        precision = precision_score(
            self.y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            self.y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            self.y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            self.y_test,
            y_pred_proba
        )

        self.results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }

        print("\n" + "=" * 50)
        print(model_name)
        print("=" * 50)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

    def train_logistic_regression(self):

        print("\n🔹 Training Logistic Regression...")

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        y_pred = model.predict(
            self.X_test
        )

        y_pred_proba = model.predict_proba(
            self.X_test
        )[:, 1]

        self.models["logistic_regression"] = model

        self.evaluate_model(
            "Logistic Regression",
            y_pred,
            y_pred_proba
        )

        return self

    def train_random_forest(self):

        print("\n🔹 Training Random Forest...")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        y_pred = model.predict(
            self.X_test
        )

        y_pred_proba = model.predict_proba(
            self.X_test
        )[:, 1]

        self.models["random_forest"] = model

        self.evaluate_model(
            "Random Forest",
            y_pred,
            y_pred_proba
        )

        return self

    def train_xgboost(self):

        print("\n🔹 Training XGBoost...")

        scale_pos_weight = (
            len(self.y_train[self.y_train == 0])
            /
            len(self.y_train[self.y_train == 1])
        )

        model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        y_pred = model.predict(
            self.X_test
        )

        y_pred_proba = model.predict_proba(
            self.X_test
        )[:, 1]

        self.models["xgboost"] = model

        self.evaluate_model(
            "XGBoost",
            y_pred,
            y_pred_proba
        )

        return self

    def compare_models(self):

        comparison = pd.DataFrame(
            self.results
        ).T

        print("\n")
        print("=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        print(comparison)

        return comparison

    def save_best_model(self):

        os.makedirs(
            "models",
            exist_ok=True
        )

        best_result_name = max(
            self.results,
            key=lambda x: self.results[x]["f1"]
        )

        model_mapping = {
            "Logistic Regression": "logistic_regression",
            "Random Forest": "random_forest",
            "XGBoost": "xgboost"
        }

        model_key = model_mapping[
            best_result_name
        ]

        best_model = self.models[
            model_key
        ]

        with open(
            f"models/{model_key}.pkl",
            "wb"
        ) as f:

            pickle.dump(
                best_model,
                f
            )

        print(
            f"\n✅ Best Model Saved: {model_key}"
        )

        return model_key


def train_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 60)
    print("🚀 MODEL TRAINING")
    print("=" * 60)

    trainer = ModelTrainer(
        X_train,
        X_test,
        y_train,
        y_test
    )

    trainer.train_logistic_regression()
    trainer.train_random_forest()
    trainer.train_xgboost()

    comparison = trainer.compare_models()

    best_model_name = trainer.save_best_model()

    return (
        trainer,
        comparison,
        best_model_name
    )