from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    asin,
    col,
    cos,
    datediff,
    dayofweek,
    from_json,
    hour,
    least,
    lit,
    pmod,
    pow,
    radians,
    sin,
    sqrt,
    struct,
    to_date,
    to_json,
    to_timestamp,
)
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


KAFKA_SERVERS = "kafka:9092"
SOURCE_TOPIC = "fraud-transactions"
FEATURE_TOPIC = "fraud-features"

KAFKA_CHECKPOINT_PATH = (
    "/tmp/spark-checkpoints/fraud-features-kafka-v2"
)

CONSOLE_CHECKPOINT_PATH = (
    "/tmp/spark-checkpoints/fraud-features-console-v1"
)


TRANSACTION_SCHEMA = StructType(
    [
        StructField(
            "event_id",
            StringType(),
            False,
        ),
        StructField(
            "trans_date_trans_time",
            StringType(),
            False,
        ),
        StructField(
            "category",
            StringType(),
            False,
        ),
        StructField(
            "amt",
            DoubleType(),
            False,
        ),
        StructField(
            "gender",
            StringType(),
            False,
        ),
        StructField(
            "city_pop",
            IntegerType(),
            False,
        ),
        StructField(
            "lat",
            DoubleType(),
            False,
        ),
        StructField(
            "long",
            DoubleType(),
            False,
        ),
        StructField(
            "dob",
            StringType(),
            False,
        ),
        StructField(
            "merch_lat",
            DoubleType(),
            False,
        ),
        StructField(
            "merch_long",
            DoubleType(),
            False,
        ),
        StructField(
            "actual_is_fraud",
            IntegerType(),
            True,
        ),
    ]
)


def add_features(dataframe):
    transaction_time = to_timestamp(
        col("trans_date_trans_time")
    )

    lat1 = radians(col("lat"))
    lon1 = radians(col("long"))
    lat2 = radians(col("merch_lat"))
    lon2 = radians(col("merch_long"))

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    haversine_a = (
        pow(sin(delta_lat / 2), 2)
        + cos(lat1)
        * cos(lat2)
        * pow(sin(delta_lon / 2), 2)
    )

    distance = (
        lit(2)
        * lit(6371.0)
        * asin(
            least(
                lit(1.0),
                sqrt(haversine_a),
            )
        )
    )

    return (
        dataframe.withColumn(
            "transaction_timestamp",
            transaction_time,
        )
        .withColumn(
            "transaction_hour",
            hour(transaction_time),
        )
        .withColumn(
            "day_of_week",
            pmod(
                dayofweek(transaction_time) + 5,
                lit(7),
            ),
        )
        .withColumn(
            "customer_age",
            datediff(
                to_date(transaction_time),
                to_date(col("dob")),
            )
            / lit(365.25),
        )
        .withColumn(
            "merchant_distance_km",
            distance,
        )
    )


def main():
    spark = (
        SparkSession.builder
        .appName(
            "FraudTransactionFeatureProcessor"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    kafka_stream = (
        spark.readStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            KAFKA_SERVERS,
        )
        .option(
            "subscribe",
            SOURCE_TOPIC,
        )
        .option(
            "startingOffsets",
            "earliest",
        )
        .load()
    )

    parsed_events = (
        kafka_stream.select(
            from_json(
                col("value").cast("string"),
                TRANSACTION_SCHEMA,
            ).alias("event"),
            col("partition"),
            col("offset"),
            col("timestamp").alias(
                "kafka_timestamp"
            ),
        )
        .select(
            "event.*",
            "partition",
            "offset",
            "kafka_timestamp",
        )

        .withWatermark(
            "kafka_timestamp",
            "10 minutes",
        )
        .dropDuplicatesWithinWatermark(
            ["event_id"]
        )
    )

    features = add_features(
        parsed_events
    ).select(
        "event_id",
        "amt",
        "category",
        "gender",
        "city_pop",
        "transaction_hour",
        "day_of_week",
        "customer_age",
        "merchant_distance_km",
        "actual_is_fraud",
        "partition",
        "offset",
    )

    kafka_output = features.select(
        col("event_id").cast("string").alias("key"),
        to_json(
            struct(
                *[
                    col(column_name)
                    for column_name in features.columns
                ]
            )
        ).alias("value"),
    )

    kafka_query = (
        kafka_output.writeStream
        .format("kafka")
        .outputMode("append")
        .option(
            "kafka.bootstrap.servers",
            KAFKA_SERVERS,
        )
        .option(
            "topic",
            FEATURE_TOPIC,
        )
        .option(
            "checkpointLocation",
            KAFKA_CHECKPOINT_PATH,
        )
        .start()
    )

    console_query = (
        features.writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", False)
        .option(
            "checkpointLocation",
            CONSOLE_CHECKPOINT_PATH,
        )
        .start()
    )

    print(
        "Spark Structured Streaming started."
    )
    print(
        f"Publishing engineered features to {FEATURE_TOPIC}."
    )

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()