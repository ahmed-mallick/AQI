# AQI Predictor — Karachi


**Author:** Ahmed Mallick | **Company:** 10Pearls (Data Science Internship)

### 🚀 Live Application
**[Click here to view the deployed Streamlit App](https://phkerauxjdnfruzfbqwyru.streamlit.app)**

### 📄 Project Report
The complete internship project report detailing the methodology, architecture, and model evaluation can be found here: 
**[Report.docx](./Report.docx)**


Predicting Karachi's Air Quality Index (AQI) for the next 3 days using a
100% serverless ML stack: OpenWeather (data) → Hopsworks (feature store +
model registry) → scikit-learn / TensorFlow (models) → GitHub Actions
(automation) → Streamlit (dashboard).

## Architecture

```
OpenWeather APIs ─┐
                  ├─> Feature Pipeline (hourly) ──> Hopsworks Feature Store
Open-Meteo (hist) ┘                                        │
                                                             ▼
                                          Training Pipeline (daily)
                                     RF / Ridge / TensorFlow, pick best
                                                             │
                                                             ▼
                                       Hopsworks Model Registry (3 models:
                                        +24h / +48h / +72h forecasts)
                                                             │
                                                             ▼
                                          Streamlit Dashboard (app/)
                                     3-day forecast + SHAP + alerts
```

## Repository layout

```
src/                      shared library code
  config.py               city coords, secrets, thresholds
  fetch_data.py           OpenWeather + Open-Meteo API calls
  aqi_calculator.py       raw pollutants -> US EPA AQI (0-500)
  feature_engineering.py  time/lag/rolling feature construction
  hopsworks_utils.py      feature store / model registry helpers
  shap_utils.py           SHAP explainers per model type

pipelines/
  backfill_pipeline.py    one-time historical backfill
  feature_pipeline.py     runs hourly (GitHub Actions)
  training_pipeline.py    runs daily (GitHub Actions)
  inference_pipeline.py   loads latest features + models -> 3-day forecast

app/
  streamlit_app.py        the dashboard

notebooks/
  eda.py                  exploratory data analysis (saves PNGs)

.github/workflows/
  feature_pipeline.yml    hourly cron
  training_pipeline.yml   daily cron
```

## 1. Setup

```bash
git clone <your-repo-url>
cd karachi_aqi_predictor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your real OPENWEATHER_API_KEY and HOPSWORKS_API_KEY
```

You already have both API keys — just drop them into `.env` (never commit
this file; it's in `.gitignore`). `HOPSWORKS_PROJECT_NAME` should match the
project you create at https://app.hopsworks.ai (free tier is enough).

## 2. Backfill historical data (run once)

```bash
python -m pipelines.backfill_pipeline --days 180
```

This pulls ~6 months of hourly pollutant history from OpenWeather's Air
Pollution History API (available from 2020-11-27 onward) plus matching
historical weather from Open-Meteo's free archive (OpenWeather's own
historical-weather endpoint needs a separate paid subscription), engineers
every feature, and writes it to the Hopsworks Feature Group. Increase
`--days` for more training data (up to ~5 years); more history generally
means a stronger model, but the first run doesn't need to be exhaustive —
you can re-run this any time to extend the window.

## 3. Train the models (run once, then daily via CI)

```bash
python -m pipelines.training_pipeline
```

For each forecast horizon (+24h, +48h, +72h) this trains a Random Forest,
a Ridge Regression, and a small TensorFlow feed-forward network on a
time-based train/test split, scores each with RMSE, MAE, and R², and
registers the best-performing model per horizon in the Hopsworks Model
Registry. Metrics are printed to stdout and saved alongside each model.

## 4. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Shows the current AQI, the +24h/+48h/+72h forecast with color-coded EPA
categories, a hazardous-AQI alert banner, a SHAP feature-importance chart
per horizon, and a 14-day historical trend.

To deploy publicly (free): push this repo to GitHub, then deploy on
[Streamlit Community Cloud](https://phkerauxjdnfruzfbqwyru.streamlit.app), pointing it at
`app/streamlit_app.py` and adding `OPENWEATHER_API_KEY`,
`HOPSWORKS_API_KEY`, and `HOPSWORKS_PROJECT_NAME` as app secrets.

## 5. Automate it (GitHub Actions)

In your GitHub repo, go to **Settings → Secrets and variables → Actions**
and add three repository secrets: `OPENWEATHER_API_KEY`,
`HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`.

Two workflows are already wired up:
- `.github/workflows/feature_pipeline.yml` — runs every hour, fetches the
  latest data, and upserts features into Hopsworks.
- `.github/workflows/training_pipeline.yml` — runs once a day, retrains
  all three model families per horizon, and republishes the best ones.

Both also support manual triggering from the GitHub Actions tab
(`workflow_dispatch`), so you can demo them on demand for your submission.

## 6. Exploratory Data Analysis

```bash
python notebooks/eda.py
```

Saves 5 plots to `notebooks/output/`: AQI trend, category breakdown,
hourly seasonality, monthly seasonality, and a correlation heatmap between
AQI, pollutants, and weather variables.

## Target variable and AQI methodology

OpenWeather's own `main.aqi` field is a coarse 1–5 index, not the 0–500
number most dashboards mean by "AQI". This project computes the standard
**US EPA AQI** from raw pollutant concentrations (PM2.5, PM10, O3, NO2,
SO2, CO) using the official EPA breakpoint tables (`src/aqi_calculator.py`),
so results are comparable to what aqicn.org / IQAir report for Karachi.

## Modeling approach

Rather than one single "3-day-ahead" model, three separate regression
targets are used — `target_aqi_24h`, `target_aqi_48h`, `target_aqi_72h` —
each trained independently so the model can specialize per horizon instead
of compounding recursive-forecast error. Features include the current
pollutant/weather readings, cyclical time encodings (hour/month
sin-cos), and AQI lag/rolling/change-rate features (1h, 3h, 6h, 12h, 24h,
48h lookback).

## Known limitations / notes for the write-up

- OpenWeather's free historical-weather endpoint isn't available, so
  historical weather comes from Open-Meteo (no key required); live/forecast
  weather still comes from OpenWeather as required by the tech stack.
- AQI sub-indices are computed from hourly (not officially-averaged 24h/8h)
  readings — a standard simplification also used by most open-source AQI
  trackers; flagged here for transparency in your report.
- The hazardous-AQI alert threshold (default 150, "Unhealthy") is
  configurable in `src/config.py`.
