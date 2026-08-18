import json
import os
from datetime import datetime, timezone

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

                prediction = create_prediction(
                    model,
                    event,
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

                print(
                    f"event_id={prediction['event_id']} "
                    f"score={prediction['fraud_score']:.4f} "
                    f"predicted_fraud="
                    f"{prediction['is_fraud']} "
                    f"actual_fraud="
                    f"{prediction['actual_is_fraud']}"
                )

            except Exception as error:
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