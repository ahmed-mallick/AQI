import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import HAZARD_ALERT_THRESHOLD, CITY_NAME
from pipelines.inference_pipeline import predict_next_3_days, get_latest_feature_row, MODEL_FEATURE_COLS, _download_model
from src.hopsworks_utils import get_or_create_feature_group
from src.shap_utils import explain_prediction

st.set_page_config(page_title=f"{CITY_NAME} AQI Predictor", page_icon="🌫️", layout="wide")

st.title(f"🌫️ {CITY_NAME} Air Quality Index — 3-Day Forecast")
st.caption("Serverless ML pipeline: OpenWeather → Hopsworks Feature Store → RF / Ridge / TensorFlow → Streamlit")


@st.cache_data(ttl=1800, show_spinner="Fetching latest prediction...")
def load_forecast():
    return predict_next_3_days()


@st.cache_data(ttl=3600, show_spinner="Loading historical trend...")
def load_history(limit_days: int = 14):
    fg = get_or_create_feature_group()
    df = fg.read()
    df = df.sort_values("datetime_utc")
    cutoff = df["datetime_utc"].max() - pd.Timedelta(days=limit_days)
    return df[df["datetime_utc"] >= cutoff][["datetime_utc", "aqi", "pm2_5", "pm10"]]


try:
    result = load_forecast()
except Exception as e:
    st.error(
        "Could not load a forecast yet. Make sure the backfill, feature and "
        f"training pipelines have all run at least once.\n\nDetails: {e}"
    )
    st.stop()

# --- current + alert banner -------------------------------------------------
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Current AQI", f"{result['current_aqi']:.0f}", result["current_category"])
    st.caption(f"As of {result['as_of']} UTC")

max_forecast = max(f["predicted_aqi"] for f in result["forecasts"])
if max_forecast >= HAZARD_ALERT_THRESHOLD:
    st.error(
        f"⚠️ Hazard alert: AQI is forecast to reach **{max_forecast:.0f}** "
        f"(Unhealthy or worse) within the next 3 days. Consider limiting outdoor exposure."
    )

# --- 3-day forecast chart ---------------------------------------------------
st.subheader("Next 3 Days")
fc = result["forecasts"]
fig = go.Figure()
fig.add_trace(go.Bar(
    x=[f"+{f['horizon_hours']}h\n{f['target_datetime'].strftime('%a %d %b')}" for f in fc],
    y=[f["predicted_aqi"] for f in fc],
    marker_color=[f["color"] for f in fc],
    text=[f"{f['predicted_aqi']:.0f} ({f['category']})" for f in fc],
    textposition="outside",
))
fig.update_layout(yaxis_title="Predicted AQI", showlegend=False, height=400)
st.plotly_chart(fig, use_container_width=True)

cols = st.columns(len(fc))
for c, f in zip(cols, fc):
    c.metric(
        f"{f['target_datetime'].strftime('%a %d %b')} (+{f['horizon_hours']}h)",
        f"{f['predicted_aqi']:.0f}",
        f["category"],
    )
    c.caption(f"model: {f['model_used']} (RMSE {f['model_rmse']:.1f})")

# --- SHAP explanation --------------------------------------------------------
st.subheader("Why this forecast? (SHAP feature importance)")
horizon_choice = st.selectbox("Horizon", [f["horizon_hours"] for f in fc], format_func=lambda h: f"+{h}h")

try:
    latest_row = get_latest_feature_row()
    X_row = latest_row[MODEL_FEATURE_COLS].to_frame().T.astype(float)
    model, scaler, meta, model_type = _download_model(horizon_choice)

    fg = get_or_create_feature_group()
    background = fg.read().sort_values("datetime_utc").dropna(subset=MODEL_FEATURE_COLS).tail(200)[MODEL_FEATURE_COLS].astype(float)

    shap_df = explain_prediction(model, model_type, scaler, X_row, background).head(10)
    shap_fig = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=["#ff4b4b" if v > 0 else "#4b7bff" for v in shap_df["shap_value"]],
    ))
    shap_fig.update_layout(
        xaxis_title="Impact on predicted AQI",
        yaxis=dict(autorange="reversed"),
        height=400,
    )
    st.plotly_chart(shap_fig, use_container_width=True)
    st.caption("Red bars push the AQI prediction up, blue bars push it down.")
except Exception as e:
    st.info(f"SHAP explanation unavailable for this run: {e}")

# --- historical trend --------------------------------------------------------
st.subheader("Recent Trend (last 14 days)")
try:
    hist_df = load_history()
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=hist_df["datetime_utc"], y=hist_df["aqi"], mode="lines", name="AQI"))
    trend_fig.update_layout(yaxis_title="AQI", xaxis_title="Date (UTC)", height=350)
    st.plotly_chart(trend_fig, use_container_width=True)
except Exception as e:
    st.info(f"Historical trend unavailable: {e}")

st.divider()
st.caption(
    "Data: OpenWeather Air Pollution & Weather APIs · Open-Meteo (historical weather) · "
    "Feature Store & Model Registry: Hopsworks · Models: Random Forest, Ridge Regression, TensorFlow NN"
)
