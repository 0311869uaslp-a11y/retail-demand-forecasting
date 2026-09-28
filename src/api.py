from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import FEATURES


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_demand_forecaster.joblib"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run 'python -m src.train' first."
    )

model = joblib.load(MODEL_PATH)


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "Machine learning API for next-day retail "
        "demand forecasting using XGBoost."
    ),
    version="1.0.0",
)


# =========================================================
# Request schema
# =========================================================

class PredictionRequest(BaseModel):
    store: int = Field(gt=0)
    item: int = Field(gt=0)

    year: int
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)

    day_of_week: int = Field(ge=0, le=6)
    week_of_year: int = Field(ge=1, le=53)
    quarter: int = Field(ge=1, le=4)
    is_weekend: int = Field(ge=0, le=1)

    lag_1: float
    lag_7: float
    lag_14: float
    lag_28: float

    rolling_mean_7: float
    rolling_mean_14: float
    rolling_mean_28: float

    rolling_std_7: float
    rolling_std_28: float


# =========================================================
# Endpoints
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Retail Demand Forecasting API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    input_data = pd.DataFrame(
        [request.model_dump()]
    )

    input_data = input_data[FEATURES]

    prediction = model.predict(
        input_data
    )[0]

    prediction = max(
        0.0,
        float(prediction)
    )

    return {
        "predicted_sales": round(
            prediction,
            2
        )
    }