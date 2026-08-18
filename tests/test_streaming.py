import numpy as np
import pandas as pd
import pytest

from src.streaming.producer import (
    create_event,
    sample_transactions,
)
from src.streaming.scoring_consumer import (
    MODEL_FEATURES,
    create_prediction,
)


def test_create_event():
    row = pd.Series(
        {
            "trans_num": "transaction-123",
            "trans_date_trans_time": (
                "2020-08-17 02:30:00"
            ),
            "category": "shopping_net",
            "amt": 450.75,
            "gender": "F",
            "city_pop": 50000,
            "lat": 32.75,
            "long": -97.33,
            "dob": "1998-05-10",
            "merch_lat": 33.1,
            "merch_long": -97.8,
            "is_fraud": 1,
        }
    )

    event = create_event(row)

    assert event["event_id"] == "transaction-123"
    assert event["amt"] == 450.75
    assert event["actual_is_fraud"] == 1


def test_sample_transactions_is_balanced(
    tmp_path,
):
    rows = []

    for index in range(4):
        rows.append(
            {
                "trans_num": f"legitimate-{index}",
                "is_fraud": 0,
            }
        )

        rows.append(
            {
                "trans_num": f"fraud-{index}",
                "is_fraud": 1,
            }
        )

    data_path = tmp_path / "transactions.csv"

    pd.DataFrame(rows).to_csv(
        data_path,
        index=False,
    )

    sample = sample_transactions(
        data_path=data_path,
        count=4,
    )

    counts = sample[
        "is_fraud"
    ].value_counts()

    assert counts[0] == 2
    assert counts[1] == 2

class FakeFraudModel:
    def __init__(self, fraud_score):
        self.fraud_score = fraud_score

    def predict_proba(self, features):
        assert features.columns.tolist() == (
            MODEL_FEATURES
        )

        return np.array(
            [
                [
                    1 - self.fraud_score,
                    self.fraud_score,
                ]
            ]
        )


def feature_event():
    return {
        "event_id": "transaction-123",
        "amt": 450.75,
        "category": "shopping_net",
        "gender": "F",
        "city_pop": 50000,
        "transaction_hour": 2,
        "day_of_week": 0,
        "customer_age": 22.27,
        "merchant_distance_km": 55.2,
        "actual_is_fraud": 1,
    }


def test_create_fraud_prediction():
    model = FakeFraudModel(
        fraud_score=0.99
    )

    prediction = create_prediction(
        model,
        feature_event(),
    )

    assert (
        prediction["event_id"]
        == "transaction-123"
    )
    assert prediction["fraud_score"] == 0.99
    assert prediction["is_fraud"] is True
    assert prediction["actual_is_fraud"] == 1
    assert prediction["model_name"]
    assert prediction["scored_at"]


def test_create_legitimate_prediction():
    model = FakeFraudModel(
        fraud_score=0.10
    )

    prediction = create_prediction(
        model,
        feature_event(),
    )

    assert prediction["is_fraud"] is False


def test_missing_model_feature_is_rejected():
    event = feature_event()
    del event["customer_age"]

    with pytest.raises(
        ValueError,
        match="customer_age",
    ):
        create_prediction(
            FakeFraudModel(0.99),
            event,
        )