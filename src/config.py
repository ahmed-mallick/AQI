"""
Central configuration for the Karachi AQI Predictor.

All secrets are read from environment variables so nothing sensitive
is ever committed to git. When running locally, put them in a `.env`
file (see `.env.example`) and load it with python-dotenv. In GitHub
Actions, set them as repository secrets (Settings -> Secrets and
variables -> Actions) and they will be injected as env vars by the
workflow files in `.github/workflows/`.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Location: Karachi, Pakistan
# ---------------------------------------------------------------------------
CITY_NAME = "Karachi"
LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"

# ---------------------------------------------------------------------------
# API keys / credentials (set these as env vars or repo secrets)
# ---------------------------------------------------------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "karachi_aqi")

# ---------------------------------------------------------------------------
# Hopsworks feature store / model registry names
# ---------------------------------------------------------------------------
FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 2
FEATURE_VIEW_NAME = "karachi_aqi_fv"
FEATURE_VIEW_VERSION = 1

MODEL_NAME_PREFIX = "karachi_aqi_model"  # + _24h / _48h / _72h

# ---------------------------------------------------------------------------
# Forecast horizons (in hours) — "next 3 days"
# ---------------------------------------------------------------------------
FORECAST_HORIZONS = [24, 48, 72]

# ---------------------------------------------------------------------------
# AQI hazard thresholds (US EPA AQI scale, 0-500)
# ---------------------------------------------------------------------------
AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]
HAZARD_ALERT_THRESHOLD = 150  # trigger an alert at/above "Unhealthy"


def require_keys():
    """Fail fast with a clear message if secrets are missing."""
    missing = []
    if not OPENWEATHER_API_KEY:
        missing.append("OPENWEATHER_API_KEY")
    if not HOPSWORKS_API_KEY:
        missing.append("HOPSWORKS_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Set them in your .env file or as GitHub Actions secrets."
        )
