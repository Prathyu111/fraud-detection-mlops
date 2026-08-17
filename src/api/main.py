from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.api.model_service import FraudModelService
from src.api.schemas import (
    PredictionResponse,
    TransactionRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = FraudModelService()

    yield


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Scores financial transactions using "
        "the registered XGBoost fraud model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    return {
        "status": "healthy",
        "model_loaded": hasattr(
            request.app.state,
            "model_service",
        ),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    transaction: TransactionRequest,
    request: Request,
):
    service = request.app.state.model_service

    return service.predict(transaction)