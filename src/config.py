import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CITY_NAME = "Karachi"
LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"


OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "karachi_aqi")


FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 2
FEATURE_VIEW_NAME = "karachi_aqi_fv"
FEATURE_VIEW_VERSION = 1

MODEL_NAME_PREFIX = "karachi_aqi_model" 


FORECAST_HORIZONS = [24, 48, 72]


AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]
HAZARD_ALERT_THRESHOLD = 150  


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
