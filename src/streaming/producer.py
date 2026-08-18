import argparse
import json
from pathlib import Path

import pandas as pd

from confluent_kafka import Producer


DEFAULT_TOPIC = "fraud-transactions"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:29092"


def delivery_report(error, message):
    if error is not None:
        print(f"Delivery failed: {error}")
        return

    print(
        "Delivered event "
        f"to partition {message.partition()} "
        f"at offset {message.offset()}"
    )


def create_event(row):
    return {
        "event_id": str(row["trans_num"]),
        "trans_date_trans_time": str(
            row["trans_date_trans_time"]
        ),
        "category": str(row["category"]),
        "amt": float(row["amt"]),
        "gender": str(row["gender"]),
        "city_pop": int(row["city_pop"]),
        "lat": float(row["lat"]),
        "long": float(row["long"]),
        "dob": str(row["dob"]),
        "merch_lat": float(row["merch_lat"]),
        "merch_long": float(row["merch_long"]),

        # Available only for offline demonstration/evaluation.
        "actual_is_fraud": int(row["is_fraud"]),
    }


def sample_transactions(
    data_path: Path,
    count: int,
):
    if count < 2:
        raise ValueError(
            "Count must be at least 2."
        )

    dataframe = pd.read_csv(data_path)

    fraud = dataframe[
        dataframe["is_fraud"] == 1
    ]

    legitimate = dataframe[
        dataframe["is_fraud"] == 0
    ]

    fraud_count = count // 2
    legitimate_count = count - fraud_count

    sampled_fraud = fraud.sample(
        n=fraud_count,
        random_state=42,
    )

    sampled_legitimate = legitimate.sample(
        n=legitimate_count,
        random_state=42,
    )

    return pd.concat(
        [
            sampled_fraud,
            sampled_legitimate,
        ]
    ).sample(
        frac=1,
        random_state=42,
    )


def publish_transactions(
    data_path: Path,
    count: int,
    topic: str,
    bootstrap_servers: str,
):
    producer = Producer(
        {
            "bootstrap.servers": bootstrap_servers,
            "client.id": "fraud-transaction-producer",
            "acks": "all",
            "enable.idempotence": True,
        }
    )

    transactions = sample_transactions(
        data_path=data_path,
        count=count,
    )

    print(
        f"Publishing {len(transactions)} "
        f"transactions to {topic}..."
    )

    for _, row in transactions.iterrows():
        event = create_event(row)

        producer.produce(
            topic=topic,
            key=event["event_id"].encode("utf-8"),
            value=json.dumps(event).encode("utf-8"),
            callback=delivery_report,
        )

        producer.poll(0)

    remaining = producer.flush(timeout=30)

    if remaining:
        raise RuntimeError(
            f"{remaining} events were not delivered."
        )

    print("All events delivered successfully.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Publish sampled fraud transactions "
            "to Kafka."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/fraudTest.csv"),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
    )

    parser.add_argument(
        "--bootstrap-servers",
        default=DEFAULT_BOOTSTRAP_SERVERS,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    publish_transactions(
        data_path=args.data,
        count=args.count,
        topic=args.topic,
        bootstrap_servers=args.bootstrap_servers,
    )