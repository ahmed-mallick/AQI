from __future__ import annotations

import hopsworks
import pandas as pd

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
)


_project = None

def get_project():
    global _project
    if _project is None:
        _project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )
    return _project

def get_feature_store():
    return get_project().get_feature_store()


def get_or_create_feature_group(description: str = "Karachi hourly AQI features"):
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=description,
        primary_key=["event_id"],
        event_time="datetime_utc",
        online_enabled=True,
        time_travel_format="HUDI",   
    )
    return fg


def insert_features(df: pd.DataFrame):
    fg = get_or_create_feature_group()
    fg.insert(df, write_options={"wait_for_job": True})
    return fg


def get_or_create_feature_view(feature_group):
    fs = get_feature_store()
    try:
        return fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    except Exception:
        query = feature_group.select_all()
        return fs.create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            description="Feature view for Karachi AQI training/inference",
            query=query,
        )


def get_model_registry():
    return get_project().get_model_registry()


def get_dataset_api():
    return get_project().get_dataset_api()


if __name__ == "__main__":
    project = get_project()
    print(f"✅ Connected to project: {project.name}")

    fs = get_feature_store()
    print(f"✅ Feature store: {fs.name}")