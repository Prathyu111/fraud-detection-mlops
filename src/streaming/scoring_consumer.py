import json
import os
import time
from datetime import datetime, timezone

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)


import mlflow
import mlflow.sklearn
import pandas as pd
from confluent_kafka import Consumer, Producer


KAFKA_SERVERS = os.getenv(
    "KAFKA_SERVERS",
    "localhost:29092",
)

FEATURE_TOPIC = "fraud-features"
PREDICTION_TOPIC = "fraud-predictions"

MODEL_NAME = "fraud-detection-xgboost"
MODEL_ALIAS = "candidate"

MODEL_URI = os.getenv(
    "MODEL_URI",
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)

DECISION_THRESHOLD = 0.9637078

METRICS_PORT = int(
    os.getenv("METRICS_PORT", "8001")
)

MODEL_FEATURES = [
    "amt",
    "category",
    "gender",
    "city_pop",
    "transaction_hour",
    "day_of_week",
    "customer_age",
    "merchant_distance_km",
]

PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total number of scored transactions.",
    ["prediction"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "fraud_prediction_errors_total",
    "Total number of scoring errors.",
)

PREDICTION_SCORE = Histogram(
    "fraud_prediction_score",
    "Distribution of model fraud scores.",
    buckets=(
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.97,
        0.99,
        1.0,
    ),
)

SCORING_LATENCY = Histogram(
    "fraud_scoring_latency_seconds",
    "Time required to score a transaction.",
)

CONFUSION_MATRIX_TOTAL = Counter(
    "fraud_confusion_matrix_total",
    "Prediction outcomes compared with labels.",
    ["actual", "predicted"],
)

MODEL_INFO = Gauge(
    "fraud_model_info",
    "Information about the active fraud model.",
    ["model_name", "model_alias"],
)

def load_model():
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    print(f"Loading model from {MODEL_URI}...")

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    print("Model loaded successfully.")
    return model


def create_prediction(model, event):
    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in event
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    features = pd.DataFrame(
        [
            {
                feature: event[feature]
                for feature in MODEL_FEATURES
            }
        ]
    )

    fraud_score = float(
        model.predict_proba(features)[0, 1]
    )

    return {
        "event_id": event["event_id"],
        "fraud_score": fraud_score,
        "is_fraud": (
            fraud_score >= DECISION_THRESHOLD
        ),
        "decision_threshold": DECISION_THRESHOLD,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "actual_is_fraud": event.get(
            "actual_is_fraud"
        ),
        "scored_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def delivery_report(error, message):
    if error:
        print(
            f"Prediction delivery failed: {error}"
        )
        return

    print(
        "Published prediction "
        f"partition={message.partition()} "
        f"offset={message.offset()}"
    )


def main():
    model = load_model()

    MODEL_INFO.labels(
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
    ).set(1)

    start_http_server(METRICS_PORT)

    print(
        "Prometheus metrics available at "
        f"http://127.0.0.1:{METRICS_PORT}/metrics"
    )

    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_SERVERS,
            "group.id": "fraud-feature-scorer-v2",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )

    producer = Producer(
        {
            "bootstrap.servers": KAFKA_SERVERS,
            "client.id": "fraud-prediction-producer",
            "enable.idempotence": True,
        }
    )

    consumer.subscribe([FEATURE_TOPIC])

    print(
        f"Waiting for features from {FEATURE_TOPIC}..."
    )

    try:
        while True:
            message = consumer.poll(1.0)

            if message is None:
                continue

            if message.error():
                print(
                    f"Kafka consumer error: "
                    f"{message.error()}"
                )
                continue

            try:
                event = json.loads(
                    message.value().decode("utf-8")
                )

                scoring_started = (
                    time.perf_counter()
                )

                prediction = create_prediction(
                    model,
                    event,
                )

                scoring_duration = (
                    time.perf_counter()
                    - scoring_started
                )

                SCORING_LATENCY.observe(
                    scoring_duration
                )

                PREDICTION_SCORE.observe(
                    prediction["fraud_score"]
                )

                producer.produce(
                    topic=PREDICTION_TOPIC,
                    key=prediction[
                        "event_id"
                    ].encode("utf-8"),
                    value=json.dumps(
                        prediction
                    ).encode("utf-8"),
                    callback=delivery_report,
                )

                remaining_messages = (
                    producer.flush(10)
                )

                if remaining_messages:
                    raise RuntimeError(
                        "Prediction was not delivered"
                    )

                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                prediction_label = (
                    "fraud"
                    if prediction["is_fraud"]
                    else "legitimate"
                )

                PREDICTIONS_TOTAL.labels(
                    prediction=prediction_label
                ).inc()

                actual_label = prediction.get(
                    "actual_is_fraud"
                )

                if actual_label in (0, 1):
                    CONFUSION_MATRIX_TOTAL.labels(
                        actual=str(actual_label),
                        predicted=str(
                            int(
                                prediction[
                                    "is_fraud"
                                ]
                            )
                        ),
                    ).inc()

                print(
                    f"event_id={prediction['event_id']} "
                    f"score={prediction['fraud_score']:.4f} "
                    f"predicted_fraud="
                    f"{prediction['is_fraud']} "
                    f"actual_fraud="
                    f"{prediction['actual_is_fraud']}"
                )

            except Exception as error:
                PREDICTION_ERRORS_TOTAL.inc()

                print(
                    "Failed to score message "
                    f"partition={message.partition()} "
                    f"offset={message.offset()}: "
                    f"{error}"
                )

    except KeyboardInterrupt:
        print("Stopping scoring consumer...")

    finally:
        producer.flush()
        consumer.close()


if __name__ == "__main__":
    main()