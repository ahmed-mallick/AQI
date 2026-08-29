"""
Inference Pipeline

Loads the latest engineered feature row from the Hopsworks Feature Store
and the best model for each horizon (+24h, +48h, +72h) from the Model
Registry, and produces the "next 3 days" AQI forecast. Used by the
Streamlit dashboard (app/streamlit_app.py) and can also be run standalone:

    python -m pipelines.inference_pipeline
"""

from __future__ import annotations

import os
import json
import tempfile
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from src.config import require_keys, FORECAST_HORIZONS, MODEL_NAME_PREFIX
from src.feature_engineering import FEATURE_COLUMNS
from src.hopsworks_utils import get_or_create_feature_group, get_model_registry
from src.aqi_calculator import aqi_category

MODEL_FEATURE_COLS = [c for c in FEATURE_COLUMNS if c not in ("event_id", "datetime_utc", "city")]

_MODEL_CACHE = {}  # horizon -> (model, scaler, metrics, model_type)


def _download_model(horizon: int):
    if horizon in _MODEL_CACHE:
        return _MODEL_CACHE[horizon]

    mr = get_model_registry()
    model_meta = mr.get_best_model(
        name=f"{MODEL_NAME_PREFIX}_{horizon}h", metric="rmse", direction="min"
    )
    local_dir = model_meta.download()

    metrics_path = os.path.join(local_dir, "metrics.json")
    with open(metrics_path) as f:
        meta = json.load(f)
    best_model_name = meta["best_model"]

    scaler = joblib.load(os.path.join(local_dir, "scaler.pkl"))

    if best_model_name == "tensorflow_nn":
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(local_dir, "model.keras"))
    else:
        model = joblib.load(os.path.join(local_dir, "model.pkl"))

    _MODEL_CACHE[horizon] = (model, scaler, meta, best_model_name)
    return _MODEL_CACHE[horizon]


def get_latest_feature_row() -> pd.Series:
    fg = get_or_create_feature_group()
    df = fg.read()
    df = df.sort_values("datetime_utc").dropna(subset=MODEL_FEATURE_COLS)
    if df.empty:
        raise RuntimeError("No complete feature rows available yet. Run the feature/backfill pipeline first.")
    return df.iloc[-1]


def predict_next_3_days() -> dict:
    require_keys()
    latest_row = get_latest_feature_row()
    X = latest_row[MODEL_FEATURE_COLS].to_frame().T.astype(float)

    current_aqi = float(latest_row["aqi"])
    as_of = latest_row["datetime_utc"]

    forecasts = []
    for horizon in FORECAST_HORIZONS:
        model, scaler, meta, model_type = _download_model(horizon)
        if model_type == "ridge_regression":
            pred = model.predict(scaler.transform(X))[0]
        elif model_type == "tensorflow_nn":
            pred = model.predict(scaler.transform(X), verbose=0).flatten()[0]
        else:  # random_forest
            pred = model.predict(X)[0]

        pred = float(np.clip(pred, 0, 500))
        label, color = aqi_category(pred)
        forecasts.append({
            "horizon_hours": horizon,
            "target_datetime": as_of + timedelta(hours=horizon),
            "predicted_aqi": round(pred, 1),
            "category": label,
            "color": color,
            "model_used": model_type,
            "model_rmse": meta["all_models"][model_type]["rmse"],
        })

    return {
        "as_of": as_of,
        "current_aqi": round(current_aqi, 1),
        "current_category": aqi_category(current_aqi)[0],
        "forecasts": forecasts,
    }


if __name__ == "__main__":
    result = predict_next_3_days()
    print(json.dumps(result, indent=2, default=str))
