"""
Thin wrapper around the OpenWeather APIs used by this project:

- Current + forecast air pollution:  /data/2.5/air_pollution[/forecast]
- Historical air pollution:          /data/2.5/air_pollution/history
- Current weather:                   /data/2.5/weather
- 5-day / 3-hour weather forecast:   /data/2.5/forecast

All functions return plain Python dicts/lists of dicts (one dict per
hourly reading) so downstream code doesn't need to know about
OpenWeather's exact JSON shape.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Dict

import requests

from src.config import OPENWEATHER_API_KEY, LATITUDE, LONGITUDE

BASE = "https://api.openweathermap.org/data/2.5"


def _get(url: str, params: dict, retries: int = 3, backoff: float = 2.0) -> dict:
    params = {**params, "appid": OPENWEATHER_API_KEY}
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"OpenWeather request failed after {retries} attempts: {last_err}")


def fetch_current_air_pollution() -> Dict:
    """Current hour pollutant concentrations for Karachi."""
    data = _get(f"{BASE}/air_pollution", {"lat": LATITUDE, "lon": LONGITUDE})
    return data["list"][0]


def fetch_air_pollution_forecast() -> List[Dict]:
    """Hourly pollutant forecast, ~4 days ahead."""
    data = _get(f"{BASE}/air_pollution/forecast", {"lat": LATITUDE, "lon": LONGITUDE})
    return data["list"]


def fetch_historical_air_pollution(start_unix: int, end_unix: int) -> List[Dict]:
    """
    Hourly historical pollutant concentrations between start_unix and
    end_unix (UTC unix timestamps). OpenWeather's history endpoint only
    has data from 2020-11-27 onward.
    """
    data = _get(
        f"{BASE}/air_pollution/history",
        {"lat": LATITUDE, "lon": LONGITUDE, "start": start_unix, "end": end_unix},
    )
    return data["list"]


def fetch_current_weather() -> Dict:
    return _get(f"{BASE}/weather", {"lat": LATITUDE, "lon": LONGITUDE, "units": "metric"})


def fetch_weather_forecast() -> List[Dict]:
    """3-hourly weather forecast for the next 5 days."""
    data = _get(f"{BASE}/forecast", {"lat": LATITUDE, "lon": LONGITUDE, "units": "metric"})
    return data["list"]


def flatten_pollution_record(rec: Dict) -> Dict:
    """Turn one air_pollution `list[]` item into a flat dict keyed by pollutant name."""
    out = {"datetime_utc": datetime.fromtimestamp(rec["dt"], tz=timezone.utc)}
    out.update(rec["components"])  # co, no, no2, o3, so2, pm2_5, pm10, nh3
    out["ow_aqi_index"] = rec["main"]["aqi"]  # OpenWeather's own 1-5 scale, kept for reference
    return out


def flatten_weather_record(rec: Dict) -> Dict:
    """Works for both current-weather and forecast-list items."""
    dt = rec.get("dt")
    out = {
        "datetime_utc": datetime.fromtimestamp(dt, tz=timezone.utc) if dt else None,
        "temp": rec["main"]["temp"],
        "humidity": rec["main"]["humidity"],
        "pressure": rec["main"]["pressure"],
        "wind_speed": rec["wind"]["speed"],
        "wind_deg": rec["wind"].get("deg"),
        "clouds": rec.get("clouds", {}).get("all"),
        "weather_main": rec["weather"][0]["main"] if rec.get("weather") else None,
    }
    return out


def fetch_weather_at_history_approx(start_unix: int, end_unix: int) -> List[Dict]:
    """
    OpenWeather's true historical-weather endpoint (One Call 3.0 `timemachine`)
    requires a separate paid subscription. As a free fallback for backfilling
    *weather* (not pollutants, which OpenWeather gives us for free), this
    project uses the Open-Meteo Archive API (no key required) for the same
    Karachi coordinates. See pipelines/backfill_pipeline.py for how this is
    combined with OpenWeather's free pollutant history.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    start_date = datetime.fromtimestamp(start_unix, tz=timezone.utc).strftime("%Y-%m-%d")
    end_date = datetime.fromtimestamp(end_unix, tz=timezone.utc).strftime("%Y-%m-%d")
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover",
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    records = []
    for i, t in enumerate(data["time"]):
        records.append({
            "datetime_utc": datetime.fromisoformat(t).replace(tzinfo=timezone.utc),
            "temp": data["temperature_2m"][i],
            "humidity": data["relative_humidity_2m"][i],
            "pressure": data["surface_pressure"][i],
            "wind_speed": data["wind_speed_10m"][i],
            "wind_deg": data["wind_direction_10m"][i],
            "clouds": data["cloud_cover"][i],
            "weather_main": None,
        })
    return records
