import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import create_features
from src.preprocessing import build_tree_preprocessor


print("Loading training data...")

df = pd.read_csv("data/fraudTrain.csv")

df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"]
)

df = df.sort_values("trans_date_trans_time")


# Same chronological 80/20 split as Logistic Regression
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index]
validation_df = df.iloc[split_index:]


X_train = create_features(train_df)
y_train = train_df["is_fraud"]

X_validation = create_features(validation_df)
y_validation = validation_df["is_fraud"]


# Calculate class imbalance
fraud_count = y_train.sum()
legitimate_count = len(y_train) - fraud_count

scale_pos_weight = legitimate_count / fraud_count


print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))

print("Legitimate transactions:", legitimate_count)
print("Fraud transactions:", fraud_count)

print(
    "scale_pos_weight:",
    scale_pos_weight,
)


model = Pipeline(
    steps=[
        (
            "preprocessor",
            build_tree_preprocessor(),
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


print("\nTraining XGBoost model...")

model.fit(X_train, y_train)


print("Generating validation probabilities...")

probabilities = model.predict_proba(
    X_validation
)[:, 1]


precision, recall, thresholds = (
    precision_recall_curve(
        y_validation,
        probabilities,
    )
)


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


print("\nXGBoost validation results:")
print("Best threshold:", best_threshold)

print(
    "Precision:",
    precision_score(
        y_validation,
        predictions,
    ),
)

print(
    "Recall:",
    recall_score(
        y_validation,
        predictions,
    ),
)

print(
    "F1:",
    f1_score(
        y_validation,
        predictions,
    ),
)