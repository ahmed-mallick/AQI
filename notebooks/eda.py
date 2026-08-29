"""
Exploratory Data Analysis for the Karachi AQI feature set.

Run after the backfill pipeline has populated the feature store:

    python notebooks/eda.py

Produces PNG plots in notebooks/output/ covering:
  - AQI trend over time
  - AQI distribution / category breakdown
  - Hourly and monthly seasonality
  - Correlation between AQI and weather variables
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.hopsworks_utils import get_or_create_feature_group
from src.config import require_keys, AQI_CATEGORIES

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def category_for(aqi):
    for lo, hi, label, _ in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Hazardous"


def main():
    require_keys()
    os.makedirs(OUT_DIR, exist_ok=True)

    fg = get_or_create_feature_group()
    df = fg.read().sort_values("datetime_utc")
    df["local_dt"] = df["datetime_utc"].dt.tz_convert("Asia/Karachi")
    df["category"] = df["aqi"].apply(category_for)

    print(f"Loaded {len(df)} feature rows spanning {df['local_dt'].min()} -> {df['local_dt'].max()}")

    # 1. AQI trend
    plt.figure(figsize=(12, 4))
    plt.plot(df["local_dt"], df["aqi"], linewidth=0.8)
    plt.title("Karachi AQI Over Time")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aqi_trend.png"), dpi=120)
    plt.close()

    # 2. Category breakdown
    plt.figure(figsize=(6, 5))
    df["category"].value_counts().plot(kind="bar")
    plt.title("AQI Category Frequency")
    plt.ylabel("Hours")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aqi_category_breakdown.png"), dpi=120)
    plt.close()

    # 3. Hourly seasonality
    plt.figure(figsize=(8, 4))
    df.groupby("hour")["aqi"].mean().plot(kind="bar")
    plt.title("Average AQI by Hour of Day")
    plt.xlabel("Hour (local time)")
    plt.ylabel("Mean AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aqi_by_hour.png"), dpi=120)
    plt.close()

    # 4. Monthly seasonality
    plt.figure(figsize=(8, 4))
    df.groupby("month")["aqi"].mean().plot(kind="bar")
    plt.title("Average AQI by Month")
    plt.xlabel("Month")
    plt.ylabel("Mean AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "aqi_by_month.png"), dpi=120)
    plt.close()

    # 5. Correlation heatmap
    corr_cols = ["aqi", "pm2_5", "pm10", "o3", "no2", "so2", "co", "temp", "humidity", "pressure", "wind_speed"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation: AQI, Pollutants & Weather")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    print(f"Saved 5 EDA plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
