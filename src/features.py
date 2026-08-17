import numpy as np
import pandas as pd


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between customer and merchant in kilometers."""

    earth_radius_km = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def create_features(df):
    df = df.copy()

    # Convert dates
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )
    df["dob"] = pd.to_datetime(df["dob"])

    # Time-based features
    df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    # Customer age at time of transaction
    df["customer_age"] = (
        df["trans_date_trans_time"] - df["dob"]
    ).dt.days / 365.25

    # Distance between customer and merchant
    df["merchant_distance_km"] = calculate_distance(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"],
    )

    features = [
        "amt",
        "category",
        "gender",
        "city_pop",
        "transaction_hour",
        "day_of_week",
        "customer_age",
        "merchant_distance_km",
    ]

    return df[features]