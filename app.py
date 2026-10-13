"""FastAPI backend: serves the frontend and prediction API."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
ART = BASE / "artifacts"

app = FastAPI(title="MLBC Fraud Detector")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
app.mount("/artifacts", StaticFiles(directory=BASE / "artifacts"), name="artifacts")

model = joblib.load(ART / "model.pkl")
metrics = json.load(open(ART / "metrics.json"))
schema = json.load(open(ART / "schema.json"))
fi = json.load(open(ART / "feature_importance.json"))
stats = json.load(open(ART / "stats.json"))
samples = json.load(open(ART / "samples.json"))

FEATURES = schema["feature_order"]
THRESHOLD = metrics.get("threshold", 0.3)


def to_row(payload: dict) -> list:
    row = []
    for f in FEATURES:
        if f not in payload:
            raise HTTPException(400, f"missing field: {f}")
        v = payload[f]
        try:
            row.append(float(v))
        except (TypeError, ValueError):
            raise HTTPException(400, f"invalid value for {f}: {v}")
    return row


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/meta")
def meta():
    return {
        "schema": schema,
        "metrics": metrics,
        "feature_importance": fi,
        "stats": {k: v for k, v in stats.items() if k != "class_means"},
    }


@app.get("/api/samples")
def get_samples():
    return samples


@app.post("/api/predict")
def predict(payload: dict):
    row = to_row(payload)
    x = np.array([row])
    proba = float(model.predict_proba(x)[0, 1])
    is_fraud = proba >= THRESHOLD
    if proba < 0.3:
        level = "low"
    elif proba < 0.6:
        level = "medium"
    else:
        level = "high"
    # heuristic drivers: features closer to fraud-class mean than legit-class mean
    means = stats["class_means"]
    drivers = []
    for f, v in zip(FEATURES, row):
        d0 = abs(v - means["0"][f])
        d1 = abs(v - means["1"][f])
        if d1 < d0:
            drivers.append({"feature": f, "value": v, "towards": "fraud"})
    drivers = drivers[:5]
    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": bool(is_fraud),
        "risk_level": level,
        "threshold": THRESHOLD,
        "drivers": drivers,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "n_features": len(FEATURES)}