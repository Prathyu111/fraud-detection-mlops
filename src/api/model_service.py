import os
import mlflow
import mlflow.sklearn
import pandas as pd

from src.api.schemas import (
    PredictionResponse,
    TransactionRequest,
)
from src.features import create_features


MODEL_NAME = "fraud-detection-xgboost"
MODEL_ALIAS = "candidate"

DEFAULT_MODEL_URI = (
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    DEFAULT_MODEL_URI,
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)

DECISION_THRESHOLD = 0.9637078


class FraudModelService:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        print(
            f"Loading model from {MODEL_URI}..."
        )

        self.model = mlflow.sklearn.load_model(
            MODEL_URI
        )

        print("Model loaded successfully.")

    def predict(
        self,
        transaction: TransactionRequest,
    ) -> PredictionResponse:

        raw_transaction = pd.DataFrame(
            [transaction.model_dump()]
        )

        features = create_features(
            raw_transaction
        )

        fraud_score = float(
            self.model.predict_proba(features)[0, 1]
        )

        is_fraud = (
            fraud_score >= DECISION_THRESHOLD
        )

        return PredictionResponse(
            fraud_score=fraud_score,
            is_fraud=bool(is_fraud),
            decision_threshold=DECISION_THRESHOLD,
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
        )