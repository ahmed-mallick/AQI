"""
Hourly Feature Pipeline

Runs every hour (see .github/workflows/feature_pipeline.yml). Fetches the
last few days of pollution + weather data (enough to correctly recompute
the 48h lag/rolling features), engineers the feature table, and upserts
the rows into the Hopsworks Feature Group. Because rows are keyed by
`event_id` (unix timestamp), re-inserting overlapping rows simply updates
them rather than duplicating — this makes the job safe to re-run.

    python -m pipelines.feature_pipeline
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.config import require_keys
from src.fetch_data import (
    fetch_historical_air_pollution,
    fetch_weather_at_history_approx,
    fetch_current_air_pollution,
    fetch_current_weather,
    flatten_pollution_record,
    flatten_weather_record,
)
from src.feature_engineering import build_feature_table, FEATURE_COLUMNS
from src.hopsworks_utils import insert_features

LOOKBACK_HOURS = 72  # enough for the 48h lag feature plus a safety margin


def fetch_recent_window():
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(hours=LOOKBACK_HOURS)

    raw_pollution = fetch_historical_air_pollution(int(start.timestamp()), int(end.timestamp()))
    pollution_records = [flatten_pollution_record(r) for r in raw_pollution]

    # Always append the very latest live reading too, in case history hasn't
    # caught up to the current hour yet.
    try:
        latest = fetch_current_air_pollution()
        pollution_records.append(flatten_pollution_record(latest))
    except Exception as e:
        print(f"Warning: could not fetch current air pollution reading: {e}")

    pollution_df = pd.DataFrame(pollution_records).drop_duplicates(subset=["datetime_utc"])

    weather_records = fetch_weather_at_history_approx(int(start.timestamp()), int(end.timestamp()))
    try:
        weather_records.append(flatten_weather_record(fetch_current_weather()))
    except Exception as e:
        print(f"Warning: could not fetch current weather reading: {e}")
    weather_df = pd.DataFrame(weather_records).drop_duplicates(subset=["datetime_utc"])

    return pollution_df, weather_df


def main():
    require_keys()
    pollution_df, weather_df = fetch_recent_window()

    # with_targets=False: the most recent rows don't have future AQI yet,
    # and targets get filled in retroactively next time this window overlaps
    # them (once enough hours have passed). For simplicity in this hourly
    # job we only push feature columns, not targets.
    feature_df = build_feature_table(pollution_df, weather_df, with_targets=False)
    feature_df = feature_df.dropna(subset=[c for c in FEATURE_COLUMNS if c != "event_id"])

    if feature_df.empty:
        print("No complete rows to insert this run (not enough lag history yet).")
        return

    # Ensure numeric columns match the Hopsworks feature group's expected
    # dtypes (double), regardless of whether this run's values happen to be
    # whole numbers (which pandas/pyarrow would otherwise infer as int).
    exclude_cols = {"event_id", "datetime_utc"}

    # Match dtypes to what the Hopsworks feature group actually expects,
    # instead of guessing — avoids float/int mismatches like this.
    from src.hopsworks_utils import get_or_create_feature_group
    fg_schema = {f.name: f.type for f in get_or_create_feature_group().features}

    int_types = {"int", "bigint"}
    for col in feature_df.columns:
        if col in exclude_cols or col not in fg_schema:
            continue
        expected = fg_schema[col]
        if expected == "double" and pd.api.types.is_numeric_dtype(feature_df[col]):
            feature_df[col] = feature_df[col].astype(float)
        elif expected in int_types and pd.api.types.is_numeric_dtype(feature_df[col]):
            feature_df[col] = feature_df[col].round().astype("int64")


if __name__ == "__main__":
    main()
