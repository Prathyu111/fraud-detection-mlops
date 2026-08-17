# Fraud Detection MLOps Platform

A production-style machine-learning project for identifying potentially fraudulent financial transactions.

The project covers feature engineering, imbalanced classification, threshold optimization, experiment tracking, model registration, REST inference, automated testing, Docker deployment and CI.

## Current architecture

```text
Transaction
    ↓
FastAPI + Pydantic validation
    ↓
Feature engineering
    ↓
Registered XGBoost candidate
    ↓
Fraud score + classification
```

## Results

Evaluation was performed on 555,719 held-out transactions containing 2,145 fraud cases.

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Precision | 0.1701 | **0.8486** |
| Recall | 0.6713 | **0.7841** |
| F1 | 0.2715 | **0.8151** |
| ROC-AUC | 0.9078 | **0.9977** |
| PR-AUC | 0.1208 | **0.8750** |
| False positives | 7,024 | **300** |
| Fraud detected | 1,440 | **1,682** |

XGBoost test confusion matrix:

```text
                 Predicted
               Legit    Fraud
Actual Legit   553274     300
Actual Fraud      463    1682
```

These results apply to the public synthetic dataset used for this project and should not be interpreted as expected performance on real banking traffic.

## Features

The model uses:

- Transaction amount
- Transaction category
- Gender
- City population
- Transaction hour
- Day of week
- Customer age
- Distance between customer and merchant

Raw timestamps, birth dates and coordinates are converted into model-ready temporal, demographic and geospatial features.

Identifiers and personally identifying columns such as transaction number, card number, name and street are excluded.

## Class imbalance

Fraud represents less than 1% of the data. Accuracy would therefore be misleading because a model predicting every transaction as legitimate would still appear highly accurate.

The project instead emphasizes:

- Precision
- Recall
- F1
- ROC-AUC
- Precision-Recall AUC
- Confusion matrix
- False-positive and false-negative counts

Logistic Regression uses balanced class weights. XGBoost uses `scale_pos_weight` based on the ratio between legitimate and fraudulent training examples.

## Threshold optimization

Decision thresholds were selected using a chronological validation split.

```text
Earlier 80% of fraudTrain.csv → Model training
Later 20% of fraudTrain.csv   → Threshold selection
fraudTest.csv                 → Held-out evaluation
```

The selected XGBoost threshold is:

```text
0.9637078
```

Because class weighting affects model scores, this value is treated as a decision threshold rather than a perfectly calibrated probability.

## MLflow

MLflow tracks:

- Model type
- Hyperparameters
- Decision threshold
- Evaluation metrics
- Confusion-matrix values
- Evaluation artifacts
- Trained model

The selected XGBoost model is registered as:

```text
fraud-detection-xgboost
```

with alias:

```text
candidate
```

Start the local MLflow dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## API

The FastAPI service provides:

```text
GET  /health
POST /predict
```

Start locally:

```bash
python -m uvicorn src.api.main:app --reload
```

Open the interactive documentation:

```text
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "trans_date_trans_time": "2020-08-17T02:30:00",
  "category": "shopping_net",
  "amt": 450.75,
  "gender": "F",
  "city_pop": 50000,
  "lat": 32.75,
  "long": -97.33,
  "dob": "1998-05-10",
  "merch_lat": 33.1,
  "merch_long": -97.8
}
```

Example response:

```json
{
  "fraud_score": 0.4443696141242981,
  "is_fraud": false,
  "decision_threshold": 0.9637078,
  "model_name": "fraud-detection-xgboost",
  "model_alias": "candidate"
}
```

## Local setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements-dev.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

Place the datasets here:

```text
data/fraudTrain.csv
data/fraudTest.csv
```

The CSV files are excluded from Git.

## Training

Train the Logistic Regression baseline:

```bash
python -m src.train
```

Tune its threshold:

```bash
python -m src.tune_threshold
```

Tune XGBoost:

```bash
python -m src.tune_xgboost
```

Train and evaluate XGBoost:

```bash
python -m src.train_xgboost
```

## Testing

Run:

```bash
python -m pytest -v
```

Current tests cover:

- Health endpoint
- Successful prediction
- Response contract
- Negative transaction amount
- Invalid latitude
- Missing required fields

## Docker

Build:

```bash
docker build -t fraud-detection-api:1.0 .
```

Run:

```bash
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  fraud-detection-api:1.0
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Stop:

```bash
docker stop fraud-api
```

The container runs as a non-root user and includes an automated health check.

## Continuous integration

GitHub Actions automatically:

1. Installs Python dependencies.
2. Runs API tests.
3. Builds the Docker image.
4. Starts the container.
5. Verifies the health endpoint.

Workflow:

```text
.github/workflows/ci.yml
```

## Project structure

```text
fraud-detection-mlops/
├── .github/workflows/ci.yml
├── artifacts/
├── data/
├── deployment_model/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── model_service.py
│   │   └── schemas.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── train_xgboost.py
│   ├── tune_threshold.py
│   └── tune_xgboost.py
├── tests/test_api.py
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt
```

## Planned extensions

- Kafka transaction producer
- Spark Structured Streaming feature pipeline
- Prometheus metrics
- Grafana dashboards
- Prediction and feature-drift monitoring
- Kubernetes deployment
- Automated candidate-to-production promotion