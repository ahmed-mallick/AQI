
from __future__ import annotations

import pandas as pd
import numpy as np

from src.aqi_calculator import compute_us_aqi
from src.config import FORECAST_HORIZONS


def merge_pollution_and_weather(pollution_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Both frames must have a `datetime_utc` column. Weather is resampled/
    forward-filled to hourly if it came from a 3-hourly forecast source.
    """
    pollution_df = pollution_df.sort_values("datetime_utc")
    weather_df = weather_df.sort_values("datetime_utc")

    weather_hourly = (
        weather_df.set_index("datetime_utc")
        .resample("1h")
        .ffill()
        .reset_index()
    )

    merged = pd.merge_asof(
        pollution_df.sort_values("datetime_utc"),
        weather_hourly.sort_values("datetime_utc"),
        on="datetime_utc",
        direction="nearest",
        tolerance=pd.Timedelta("2h"),
    )
    return merged


def add_aqi_columns(df: pd.DataFrame) -> pd.DataFrame:
    aqi_rows = df.apply(
        lambda r: compute_us_aqi(r.get("pm2_5"), r.get("pm10"), r.get("o3"), r.get("no2"), r.get("so2"), r.get("co")),
        axis=1,
        result_type="expand",
    )
    return pd.concat([df, aqi_rows], axis=1)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt_local = df["datetime_utc"].dt.tz_convert("Asia/Karachi")
    df["hour"] = dt_local.dt.hour
    df["day"] = dt_local.dt.day
    df["month"] = dt_local.dt.month
    df["day_of_week"] = dt_local.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([4, 5]).astype(int) 
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    for lag_h in [1, 3, 6, 12, 24, 48]:
        df[f"aqi_lag_{lag_h}h"] = df["aqi"].shift(lag_h)
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(24, min_periods=6).mean()
    df["aqi_rolling_std_24h"] = df["aqi"].rolling(24, min_periods=6).std()
    df["aqi_change_rate_1h"] = df["aqi"].diff(1)
    df["aqi_change_rate_24h"] = df["aqi"].diff(24)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Future AQI values become the regression targets for each horizon."""
    for h in FORECAST_HORIZONS:
        df[f"target_aqi_{h}h"] = df["aqi"].shift(-h)
    return df


def build_feature_table(pollution_df: pd.DataFrame, weather_df: pd.DataFrame, with_targets: bool = True) -> pd.DataFrame:
    df = merge_pollution_and_weather(pollution_df, weather_df)
    df = add_aqi_columns(df)
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)
    if with_targets:
        df = add_targets(df)

    df["city"] = "Karachi"
    # Hopsworks primary key needs a stable unique id
    df["event_id"] = df["datetime_utc"].astype("int64") // 10**9  
    return df


FEATURE_COLUMNS = [
    "event_id", "datetime_utc", "city",
    "pm2_5", "pm10", "o3", "no2", "so2", "co", "nh3",
    "aqi", "aqi_pm2_5", "aqi_pm10", "aqi_o3", "aqi_no2", "aqi_so2", "aqi_co",
    "temp", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
    "hour", "day", "month", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h", "aqi_lag_48h",
    "aqi_rolling_mean_24h", "aqi_rolling_std_24h",
    "aqi_change_rate_1h", "aqi_change_rate_24h",
]

TARGET_COLUMNS = [f"target_aqi_{h}h" for h in FORECAST_HORIZONS]
