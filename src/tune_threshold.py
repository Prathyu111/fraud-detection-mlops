import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline

from src.features import create_features
from src.preprocessing import build_preprocessor


print("Loading training data...")

df = pd.read_csv("data/fraudTrain.csv")

# Make sure transactions are ordered by time
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

df = df.sort_values("trans_date_trans_time")

# Earliest 80% = training
# Latest 20% = validation
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index]
validation_df = df.iloc[split_index:]

print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))

X_train = create_features(train_df)
y_train = train_df["is_fraud"]

X_validation = create_features(validation_df)
y_validation = validation_df["is_fraud"]


model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)


print("Training validation model...")

model.fit(X_train, y_train)

print("Generating fraud probabilities...")

probabilities = model.predict_proba(X_validation)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_validation,
    probabilities,
)

# thresholds has one fewer element
precision = precision[:-1]
recall = recall[:-1]

f1_scores = (
    2 * precision * recall
    / (precision + recall + 1e-10)
)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

predictions = (
    probabilities >= best_threshold
).astype(int)


print("\nBest threshold:", best_threshold)

print(
    "Precision:",
    precision_score(y_validation, predictions),
)

print(
    "Recall:",
    recall_score(y_validation, predictions),
)

print(
    "F1:",
    f1_score(y_validation, predictions),
)