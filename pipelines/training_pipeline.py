from __future__ import annotations

import os
import json
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.config import require_keys, FORECAST_HORIZONS, MODEL_NAME_PREFIX
from src.feature_engineering import FEATURE_COLUMNS
from src.hopsworks_utils import get_or_create_feature_group, get_or_create_feature_view, get_model_registry

MODEL_FEATURE_COLS = [c for c in FEATURE_COLUMNS if c not in ("event_id", "datetime_utc", "city")]
TEST_FRACTION = 0.15


def load_feature_data() -> pd.DataFrame:
    fg = get_or_create_feature_group()
    get_or_create_feature_view(fg)  
    df = fg.read()
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    for h in FORECAST_HORIZONS:
        df[f"target_aqi_{h}h"] = df["aqi"].shift(-h)
    return df


def time_split(df: pd.DataFrame):
    n_test = max(1, int(len(df) * TEST_FRACTION))
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]
    return train_df, test_df


def build_nn(input_dim: int):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_horizon(df: pd.DataFrame, horizon: int, out_dir: str) -> dict:
    target_col = f"target_aqi_{horizon}h"
    horizon_df = df.dropna(subset=MODEL_FEATURE_COLS + [target_col])
    train_df, test_df = time_split(horizon_df)

    X_train, y_train = train_df[MODEL_FEATURE_COLS], train_df[target_col]
    X_test, y_test = test_df[MODEL_FEATURE_COLS], test_df[target_col]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    artifacts = {}

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["random_forest"] = evaluate(y_test, rf.predict(X_test))
    artifacts["random_forest"] = rf

    # --- Ridge Regression ---
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_s, y_train)
    results["ridge_regression"] = evaluate(y_test, ridge.predict(X_test_s))
    artifacts["ridge_regression"] = ridge

    # --- TensorFlow NN ---
    try:
        nn = build_nn(X_train_s.shape[1])
        nn.fit(X_train_s, y_train, validation_split=0.1, epochs=60, batch_size=32, verbose=0)
        results["tensorflow_nn"] = evaluate(y_test, nn.predict(X_test_s, verbose=0).flatten())
        artifacts["tensorflow_nn"] = nn
    except Exception as e:
        print(f"[horizon {horizon}h] TensorFlow model skipped: {e}")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"[horizon {horizon}h] metrics: {json.dumps(results, indent=2)}")
    print(f"[horizon {horizon}h] best model: {best_name}")

    horizon_dir = os.path.join(out_dir, f"horizon_{horizon}h")
    os.makedirs(horizon_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(horizon_dir, "scaler.pkl"))

    best_model = artifacts[best_name]
    if best_name == "tensorflow_nn":
        best_model.save(os.path.join(horizon_dir, "model.keras"))
    else:
        joblib.dump(best_model, os.path.join(horizon_dir, "model.pkl"))

    with open(os.path.join(horizon_dir, "metrics.json"), "w") as f:
        json.dump({"all_models": results, "best_model": best_name, "features": MODEL_FEATURE_COLS}, f, indent=2)

    return {"horizon": horizon, "best_model": best_name, "metrics": results, "dir": horizon_dir}


def upload_to_registry(horizon_result: dict):
    mr = get_model_registry()
    horizon = horizon_result["horizon"]
    best_metrics = horizon_result["metrics"][horizon_result["best_model"]]

    model_meta = mr.python.create_model(
        name=f"{MODEL_NAME_PREFIX}_{horizon}h",
        metrics=best_metrics,
        description=f"Best model ({horizon_result['best_model']}) for Karachi AQI +{horizon}h forecast",
    )
    model_meta.save(horizon_result["dir"])
    print(f"Uploaded {MODEL_NAME_PREFIX}_{horizon}h to the Hopsworks Model Registry.")


def main():
    require_keys()
    df = load_feature_data()
    df = build_targets(df)

    if len(df) < 100:
        raise RuntimeError(
            f"Only {len(df)} rows in the feature store — run the backfill pipeline "
            f"first (python -m pipelines.backfill_pipeline) before training."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = []
        for horizon in FORECAST_HORIZONS:
            result = train_horizon(df, horizon, tmp_dir)
            upload_to_registry(result)
            summary.append({
                "horizon": horizon,
                "best_model": result["best_model"],
                "metrics": result["metrics"][result["best_model"]],
            })

        print("\n=== Training summary ===")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
