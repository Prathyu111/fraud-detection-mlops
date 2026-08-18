import argparse
import json
import time

from confluent_kafka import (
    Consumer,
    KafkaException,
    KafkaError,
)


DEFAULT_TOPIC = "fraud-transactions"
DEFAULT_BOOTSTRAP_SERVERS = "localhost:29092"


def consume_transactions(
    topic: str,
    bootstrap_servers: str,
    group_id: str,
    max_messages: int,
    idle_timeout: int,
):
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )

    consumer.subscribe([topic])

    consumed = 0
    last_message_time = time.time()

    print(
        f"Waiting for up to {max_messages} "
        f"events from {topic}..."
    )

    try:
        while consumed < max_messages:
            message = consumer.poll(timeout=1.0)

            if message is None:
                if (
                    time.time() - last_message_time
                    >= idle_timeout
                ):
                    print("Idle timeout reached.")
                    break

                continue

            if message.error():
                if (
                    message.error().code()
                    == KafkaError._PARTITION_EOF
                ):
                    continue

                raise KafkaException(
                    message.error()
                )

            event = json.loads(
                message.value().decode("utf-8")
            )

            print(
                f"event_id={event['event_id']} "
                f"amount={event['amt']} "
                f"category={event['category']} "
                f"actual_fraud="
                f"{event['actual_is_fraud']} "
                f"partition={message.partition()} "
                f"offset={message.offset()}"
            )

            # Commit only after successful processing.
            consumer.commit(
                message=message,
                asynchronous=False,
            )

            consumed += 1
            last_message_time = time.time()

    finally:
        consumer.close()

    print(f"Consumed {consumed} events.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Consume transaction events from Kafka."
        )
    )

    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
    )

    parser.add_argument(
        "--bootstrap-servers",
        default=DEFAULT_BOOTSTRAP_SERVERS,
    )

    parser.add_argument(
        "--group-id",
        default="fraud-demo-consumer-v1",
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=10,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    consume_transactions(
        topic=args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        max_messages=args.max_messages,
        idle_timeout=args.idle_timeout,
    )