
import time
import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import math

# Global state
model = None
reference_data = None
start_time = None
prediction_count = 0
prediction_log = []
MODEL_VERSION = "1.0.0"

class PredictRequest(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Age: float = Field(..., gt=0, lt=100)
    SibSp: int = Field(..., ge=0, le=10)
    Parch: int = Field(..., ge=0, le=10)
    Fare: float = Field(..., gt=0)
    Sex_enc: int = Field(..., ge=0, le=1)
    Embarked_enc: int = Field(..., ge=0, le=2)

    @field_validator("Age", "Fare")
    @classmethod
    def must_be_finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("Must be finite")
        return v

class PredictResponse(BaseModel):
    survived: int
    survival_probability: float
    confidence: str
    model_version: str
    processing_time_ms: float
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    prediction_count: int
    timestamp: datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, reference_data, start_time
    print("Loading model...")
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)
    reference_data = pd.read_csv("artifacts/reference_data.csv")
    start_time = time.time()
    print("Model loaded!")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Titanic Survival Predictor API",
    description="Production ML service with drift monitoring",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["Root"])
def root():
    return {"service": "Titanic Survival Predictor", "docs": "/docs", "health": "/health"}

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
        uptime_seconds=round(time.time() - start_time, 1) if start_time else 0,
        prediction_count=prediction_count,
        timestamp=datetime.now()
    )

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    global prediction_count
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    features = np.array([[
        request.Pclass, request.Age, request.SibSp, request.Parch,
        request.Fare, request.Sex_enc, request.Embarked_enc
    ]])

    survived = int(model.predict(features)[0])
    prob = float(model.predict_proba(features)[0][1])
    processing_time = round((time.time() - start) * 1000, 2)
    prediction_count += 1

    # log for monitoring
    prediction_log.append({
        "Pclass": request.Pclass, "Age": request.Age,
        "SibSp": request.SibSp, "Parch": request.Parch,
        "Fare": request.Fare, "Sex_enc": request.Sex_enc,
        "Embarked_enc": request.Embarked_enc, "prediction": survived
    })

    confidence = "high" if prob >= 0.8 or prob <= 0.2 else "medium" if prob >= 0.6 or prob <= 0.4 else "low"

    return PredictResponse(
        survived=survived,
        survival_probability=round(prob, 4),
        confidence=confidence,
        model_version=MODEL_VERSION,
        processing_time_ms=processing_time,
        timestamp=datetime.now()
    )

@app.get("/metrics", tags=["Monitoring"])
def metrics():
    survival_rate = sum(p["prediction"] for p in prediction_log) / len(prediction_log) if prediction_log else 0
    return {
        "prediction_count": prediction_count,
        "survival_rate": round(survival_rate, 3),
        "uptime_seconds": round(time.time() - start_time, 1) if start_time else 0,
        "model_version": MODEL_VERSION
    }

@app.get("/monitor", tags=["Monitoring"])
def monitor():
    if len(prediction_log) < 10:
        return {"status": "insufficient_data", "message": f"Need 10+ predictions, have {len(prediction_log)}"}

    from evidently.core.report import Report
    from evidently.presets import DataDriftPreset
    from evidently import Dataset, DataDefinition

    current_df = pd.DataFrame(prediction_log)
    data_definition = DataDefinition(
        numerical_columns=["Age", "Fare", "SibSp", "Parch"],
        categorical_columns=["Pclass", "Sex_enc", "Embarked_enc"],
    )
    ref_ds = Dataset.from_pandas(reference_data.drop(columns=["Survived"], errors="ignore"), data_definition=data_definition)
    curr_ds = Dataset.from_pandas(current_df.drop(columns=["prediction"], errors="ignore"), data_definition=data_definition)
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=curr_ds)
    result = snapshot.dict()

    drifted = 0
    total = 0
    for metric in result["metrics"]:
        if "DriftedColumnsCount" in metric.get("metric_name", ""):
            drifted = int(metric["value"]["count"])
            share = metric["value"]["share"]

    return {
        "status": "drift_detected" if share > 0.2 else "no_drift",
        "drifted_columns": drifted,
        "drift_share": round(share, 3),
        "prediction_count": len(prediction_log),
    }
