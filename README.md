Fraud Detection MLOps Platform

An end-to-end, production-style machine-learning platform for detecting potentially fraudulent financial transactions in batch and real-time workflows.

The project includes feature engineering, imbalanced classification, threshold optimization, MLflow experiment tracking and model registration, FastAPI inference, Kafka event streaming, Spark Structured Streaming, Prometheus metrics, Grafana dashboards, automated testing, Docker deployment, and GitHub Actions CI.

Architecture

Real-time streaming path

CSV transaction simulator
          ↓
Kafka: fraud-transactions
          ↓
Spark Structured Streaming
  - JSON schema validation
  - Temporal features
  - Customer age
  - Merchant distance
  - Watermark-based deduplication
          ↓
Kafka: fraud-features
          ↓
MLflow XGBoost scoring consumer
          ↓
Kafka: fraud-predictions
          ↓
Prometheus metrics → Grafana dashboard

REST inference path

JSON transaction request
          ↓
FastAPI + Pydantic validation
          ↓
Feature engineering
          ↓
Registered XGBoost candidate
          ↓
Fraud score + classification

Model results

Evaluation was performed on 555,719 held-out transactions containing 2,145 fraud cases.

Metric

Logistic Regression

XGBoost

Precision

0.1701

0.8486

Recall

0.6713

0.7841

F1

0.2715

0.8151

ROC-AUC

0.9078

0.9977

PR-AUC

0.1208

0.8750

False positives

7,024

300

Fraud detected

1,440

1,682

XGBoost test confusion matrix:

                 Predicted
               Legit    Fraud
Actual Legit   553274     300
Actual Fraud      463    1682

These results apply to the public synthetic dataset used for this project and should not be interpreted as expected performance on real banking traffic.

Feature engineering

The model uses:

Transaction amount

Transaction category

Gender

City population

Transaction hour

Day of week

Customer age

Distance between customer and merchant

Raw timestamps, birth dates, and coordinates are converted into model-ready temporal, demographic, and geospatial features. Merchant distance is calculated with the Haversine formula.

Identifiers and personally identifying columns such as card number, customer name, street address, and transaction number are excluded from model training.

Class imbalance

Fraud represents less than 1% of the data. Accuracy would therefore be misleading because a model predicting every transaction as legitimate would still appear highly accurate.

The project instead emphasizes:

Precision

Recall

F1

ROC-AUC

Precision-Recall AUC

Confusion matrix

False-positive and false-negative counts

Logistic Regression uses balanced class weights. XGBoost uses scale_pos_weight based on the ratio between legitimate and fraudulent training examples.

Threshold optimization

Decision thresholds were selected using a chronological validation split:

Earlier 80% of fraudTrain.csv → Model training
Later 20% of fraudTrain.csv   → Threshold selection
fraudTest.csv                 → Held-out evaluation

The selected XGBoost decision threshold is:

0.9637078

Because class weighting affects model scores, this value is treated as a decision threshold rather than a perfectly calibrated probability.

MLflow experiment tracking

MLflow tracks:

Model type and hyperparameters

Decision threshold

Evaluation metrics

Confusion-matrix values

Evaluation artifacts

Trained models

The selected model is registered as:

fraud-detection-xgboost

with the alias:

candidate

Start the local MLflow dashboard:

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

FastAPI inference service

The API provides:

GET  /health
POST /predict

Start locally:

python -m uvicorn src.api.main:app --reload

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Example request:

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

Example response:

{
  "fraud_score": 0.4443696141242981,
  "is_fraud": false,
  "decision_threshold": 0.9637078,
  "model_name": "fraud-detection-xgboost",
  "model_alias": "candidate"
}

Kafka and Spark streaming

The streaming workflow uses three Kafka topics:

fraud-transactions → Raw transaction events
fraud-features     → Spark-engineered model features
fraud-predictions  → XGBoost scores and decisions

Spark performs real-time feature engineering and uses event-time watermarking, unique event IDs, Kafka offsets, and persistent checkpoints to reduce duplicate processing and support recovery.

Start Kafka and Spark:

docker compose up -d kafka spark

Create the topics if they do not already exist:

docker exec fraud-kafka /opt/kafka/bin/kafka-topics.sh \
  --create --if-not-exists \
  --topic fraud-transactions \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

docker exec fraud-kafka /opt/kafka/bin/kafka-topics.sh \
  --create --if-not-exists \
  --topic fraud-features \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

docker exec fraud-kafka /opt/kafka/bin/kafka-topics.sh \
  --create --if-not-exists \
  --topic fraud-predictions \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

Git Bash users may need to prefix these commands with MSYS_NO_PATHCONV=1.

Start the model-scoring consumer:

python -m src.streaming.scoring_consumer

Publish a balanced demonstration batch:

python -m src.streaming.producer --count 10

An optional seed can reproduce the same sample:

python -m src.streaming.producer --count 10 --seed 100

The actual_is_fraud label is included only for offline demonstration and monitoring. It would not be available during real production inference.

Prometheus and Grafana monitoring

The scoring consumer exposes Prometheus metrics at:

http://127.0.0.1:8001/metrics

Metrics include:

Total fraud and legitimate predictions

Prediction errors

Fraud-score distribution

Scoring latency

Confusion-matrix outcomes

Active model information

Start Prometheus and Grafana:

docker compose up -d prometheus grafana

Open:

Prometheus targets: http://127.0.0.1:9090/targets
Grafana:            http://127.0.0.1:3000

The provisioned Grafana dashboard is named:

Real-Time Fraud Detection Monitoring

It displays prediction totals, fraud and legitimate decisions, scoring errors, false positives, false negatives, average fraud score, P95 scoring latency, and prediction volume by class.

Prometheus configuration, Grafana data-source provisioning, and the exported dashboard JSON are stored under monitoring/ so the monitoring environment can be recreated from source control.

Local setup

Python 3.12 is recommended.

python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements-dev.txt

On macOS/Linux, activate with:

source .venv/bin/activate

Place the datasets here:

data/fraudTrain.csv
data/fraudTest.csv

The CSV files are excluded from Git.

Training

Train the Logistic Regression baseline:

python -m src.train

Tune its threshold:

python -m src.tune_threshold

Tune XGBoost:

python -m src.tune_xgboost

Train and evaluate XGBoost:

python -m src.train_xgboost

Testing

Run:

python -m pytest -v

The current 10-test suite covers:

API health and successful prediction

Response validation

Invalid transaction amounts and coordinates

Missing required fields

Kafka event construction

Balanced transaction sampling

Fraud and legitimate scoring outcomes

Missing model-feature rejection

Docker API deployment

Build:

docker build -t fraud-detection-api:1.0 .

Run:

docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  fraud-detection-api:1.0

Check health:

curl http://127.0.0.1:8000/health

Stop:

docker stop fraud-api

The API container runs as a non-root user and includes an automated health check.

Continuous integration

GitHub Actions automatically:

Installs Python dependencies.

Runs the API and streaming unit tests.

Builds the Docker API image.

Starts the container.

Verifies the health endpoint.

Workflow:

.github/workflows/ci.yml

Project structure

fraud-detection-mlops/
├── .github/workflows/ci.yml
├── artifacts/
├── data/
├── deployment_model/
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       │   └── fraud-monitoring-dashboard.json
│       └── provisioning/
│           ├── dashboards/dashboards.yml
│           └── datasources/prometheus.yml
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── model_service.py
│   │   └── schemas.py
│   ├── streaming/
│   │   ├── consumer.py
│   │   ├── producer.py
│   │   ├── scoring_consumer.py
│   │   └── spark_processor.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── train_xgboost.py
│   ├── tune_threshold.py
│   └── tune_xgboost.py
├── tests/
│   ├── test_api.py
│   └── test_streaming.py
├── compose.yaml
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt

Planned extensions

Kubernetes deployment for the FastAPI inference service

Kubernetes health probes, resource controls, scaling, and rolling updates

Automated feature-drift and prediction-drift monitoring

Alert rules for elevated errors, latency, and fraud volume

Automated candidate-to-production model promotion

Important limitations

The dataset is synthetic and does not represent live banking traffic.

Offline labels are used for demonstration; real labels would arrive later through investigation or chargeback workflows.

Model scores are not calibrated probabilities.

Local Docker Compose uses a single Kafka broker and is not a highly available production cluster.

Production deployment would require authentication, encryption, secret management, stronger data governance, and independent model validation.