import json
from pathlib import Path

import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.features import create_features
from src.preprocessing import build_preprocessor


TRAIN_PATH = Path("data/fraudTrain.csv")
TEST_PATH = Path("data/fraudTest.csv")
ARTIFACT_DIR = Path("artifacts")
DECISION_THRESHOLD = 0.7879241667687406

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")


def evaluate_model(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    return {
        "decision_threshold": DECISION_THRESHOLD,
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }


def main():
    print("Loading training data...")

    train_df = pd.read_csv(TRAIN_PATH)

    X_train = create_features(train_df)
    y_train = train_df["is_fraud"]

    print("Loading test data...")

    test_df = pd.read_csv(TEST_PATH)

    X_test = create_features(test_df)
    y_test = test_df["is_fraud"]

    print("Building model...")

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

    print("Training model...")

    model.fit(X_train, y_train)

    print("Evaluating model...")

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\nResults:")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    ARTIFACT_DIR.mkdir(exist_ok=True)

    joblib.dump(
        model,
        ARTIFACT_DIR / "fraud_model.joblib",
    )

    with open(
        ARTIFACT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)
        
    confusion = metrics["confusion_matrix"]

    true_negatives = confusion[0][0]
    false_positives = confusion[0][1]
    false_negatives = confusion[1][0]
    true_positives = confusion[1][1]


    with mlflow.start_run(
        run_name="logistic-regression-v1"
    ):

        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 1000,
                "decision_threshold": DECISION_THRESHOLD,
            }
        )

        mlflow.log_metrics(
            {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": (
                    metrics["average_precision"]
                ),
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
        )

        print(
            "MLflow run ID:",
            mlflow.active_run().info.run_id,
        )

        print("\nModel saved to artifacts/fraud_model.joblib")
        print("Metrics saved to artifacts/metrics.json")


if __name__ == "__main__":
    main()