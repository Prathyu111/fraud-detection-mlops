import pytest

from fastapi.testclient import TestClient

from src.api.main import app


VALID_TRANSACTION = {
    "trans_date_trans_time": "2020-08-17T02:30:00",
    "category": "shopping_net",
    "amt": 450.75,
    "gender": "F",
    "city_pop": 50000,
    "lat": 32.75,
    "long": -97.33,
    "dob": "1998-05-10",
    "merch_lat": 33.1,
    "merch_long": -97.8,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_valid_prediction(client):
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["fraud_score"] <= 1
    assert isinstance(result["is_fraud"], bool)

    assert result["decision_threshold"] == pytest.approx(
        0.9637078
    )

    assert (
        result["model_name"]
        == "fraud-detection-xgboost"
    )

    assert result["model_alias"] == "candidate"


def test_negative_amount_is_rejected(client):
    transaction = {
        **VALID_TRANSACTION,
        "amt": -25,
    }

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 422


def test_invalid_latitude_is_rejected(client):
    transaction = {
        **VALID_TRANSACTION,
        "lat": 100,
    }

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 422


def test_missing_required_field_is_rejected(client):
    transaction = VALID_TRANSACTION.copy()

    del transaction["category"]

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 422