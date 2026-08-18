import pandas as pd

from src.streaming.producer import (
    create_event,
    sample_transactions,
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