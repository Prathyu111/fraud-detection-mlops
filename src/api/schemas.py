from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    trans_date_trans_time: datetime
    category: str = Field(min_length=1)
    amt: float = Field(ge=0)
    gender: Literal["M", "F"]
    city_pop: int = Field(ge=0)

    lat: float = Field(ge=-90, le=90)
    long: float = Field(ge=-180, le=180)

    dob: date

    merch_lat: float = Field(ge=-90, le=90)
    merch_long: float = Field(ge=-180, le=180)


class PredictionResponse(BaseModel):
    fraud_score: float
    is_fraud: bool
    decision_threshold: float
    model_name: str
    model_alias: str