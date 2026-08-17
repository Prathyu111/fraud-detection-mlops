import json
from pathlib import Path

import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import create_features
from src.preprocessing import build_tree_preprocessor


THRESHOLD = 0.9637078

ARTIFACT_DIR = Path("artifacts")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")


print("Loading training data...")

train_df = pd.read_csv("data/fraudTrain.csv")
test_df = pd.read_csv("data/fraudTest.csv")


X_train = create_features(train_df)
y_train = train_df["is_fraud"]

X_test = create_features(test_df)
y_test = test_df["is_fraud"]


fraud_count = y_train.sum()
legitimate_count = len(y_train) - fraud_count

scale_pos_weight = legitimate_count / fraud_count

print("scale_pos_weight:", scale_pos_weight)


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


print("Training XGBoost...")

model.fit(X_train, y_train)


print("Evaluating on test data...")

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)


metrics = {
    "decision_threshold": THRESHOLD,
    "precision": precision_score(
        y_test, predictions
    ),
    "recall": recall_score(
        y_test, predictions
    ),
    "f1": f1_score(
        y_test, predictions
    ),
    "roc_auc": roc_auc_score(
        y_test, probabilities
    ),
    "average_precision": average_precision_score(
        y_test, probabilities
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist(),
}


print("\nXGBoost Test Results:")

for name, value in metrics.items():
    print(f"{name}: {value}")


ARTIFACT_DIR.mkdir(exist_ok=True)

joblib.dump(
    model,
    ARTIFACT_DIR / "xgboost_fraud_model.joblib",
)

with open(
    ARTIFACT_DIR / "xgboost_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=2)

print("\nXGBoost model saved.")

confusion = metrics["confusion_matrix"]

true_negatives = confusion[0][0]
false_positives = confusion[0][1]
false_negatives = confusion[1][0]
true_positives = confusion[1][1]


with mlflow.start_run(run_name="xgboost-v2") as run:

    mlflow.log_params(
        {
            "model_type": "XGBoost",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "decision_threshold": THRESHOLD,
        }
    )

    mlflow.log_metrics(
        {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_positives": true_positives,
        }
    )

    mlflow.log_dict(
        metrics,
        "evaluation/metrics.json",
    )

    mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    input_example=X_test.head(5),
    serialization_format=(
        mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    ),
)

    print(
        "MLflow run ID:",
        mlflow.active_run().info.run_id,
    )

