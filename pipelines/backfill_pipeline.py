"""
Historical Data Backfill

Pulls as much history as OpenWeather's Air Pollution History endpoint has
(hourly, back to 2020-11-27) plus matching historical weather from
Open-Meteo's free archive, engineers the full feature table (including
lag/rolling/target columns) and pushes it into the Hopsworks Feature Group.

Run once before the first training run, and any time you want to extend
history further back:

    python -m pipelines.backfill_pipeline --days 365

By default it backfills the last 180 days (a good balance between having
enough training data and keeping the first run fast).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import require_keys
from src.fetch_data import (
    fetch_historical_air_pollution,
    fetch_weather_at_history_approx,
    flatten_pollution_record,
)
from src.feature_engineering import build_feature_table, FEATURE_COLUMNS
from src.hopsworks_utils import insert_features

CHUNK_DAYS = 30  # OpenWeather history calls are cheap but keep chunks reasonable


def fetch_all_history(days: int) -> pd.DataFrame:
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)

    all_records = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        print(f"Fetching pollution history {cursor.date()} -> {chunk_end.date()}")
        raw = fetch_historical_air_pollution(int(cursor.timestamp()), int(chunk_end.timestamp()))
        all_records.extend(flatten_pollution_record(r) for r in raw)
        cursor = chunk_end
        time.sleep(1)  # be polite to the API

    pollution_df = pd.DataFrame(all_records)
    if pollution_df.empty:
        raise RuntimeError(
            "No historical pollution data returned. OpenWeather's history "
            "endpoint only covers 2020-11-27 onward — check your date range."
        )

    print(f"Fetching historical weather {start.date()} -> {end.date()} from Open-Meteo")
    weather_records = fetch_weather_at_history_approx(int(start.timestamp()), int(end.timestamp()))
    weather_df = pd.DataFrame(weather_records)

    return pollution_df, weather_df


def main(days: int):
    require_keys()
    pollution_df, weather_df = fetch_all_history(days)
    # Note: the feature group only stores raw features (not future targets).
    # Targets (target_aqi_24h/48h/72h) are computed on the fly at training
    # time from the AQI time series, so the feature-group schema never
    # needs to change and the hourly pipeline can insert with the exact
    # same columns as this backfill.
    feature_df = build_feature_table(pollution_df, weather_df, with_targets=False)

    before = len(feature_df)
    feature_df = feature_df.dropna(subset=[c for c in FEATURE_COLUMNS if c != "event_id"])
    print(f"Dropped {before - len(feature_df)} edge rows lacking full lag history")

    print(f"Backfilling {len(feature_df)} rows into the Hopsworks feature group...")
    insert_features(feature_df[FEATURE_COLUMNS])
    print("Backfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical Karachi AQI features into Hopsworks")
    parser.add_argument("--days", type=int, default=180, help="How many days of history to backfill")
    args = parser.parse_args()
    main(args.days)
