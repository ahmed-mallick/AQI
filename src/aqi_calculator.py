"""
Converts raw pollutant concentrations (as returned by OpenWeather's Air
Pollution API, in ug/m3) into the standard US EPA Air Quality Index (0-500).

OpenWeather's own `main.aqi` field is a coarse 1-5 index, not the AQI most
people mean when they say "AQI" (the AQICN/EPA style 0-500 number). This
module computes the real thing from PM2.5, PM10, O3, NO2, SO2 and CO using
the official EPA breakpoint tables, so the project's target variable is
comparable to what aqicn.org / IQAir report for Karachi.
"""

from __future__ import annotations

# EPA breakpoint tables: (C_low, C_high, I_low, I_high)
# Concentrations: PM2.5 & PM10 in ug/m3 (24h avg approximated from hourly here),
# O3 in ppb (8h), CO in ppm (8h), SO2 in ppb (1h), NO2 in ppb (1h)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

O3_BREAKPOINTS_PPB = [  # 8-hour ozone
    (0, 54, 0, 50),
    (55, 70, 51, 100),
    (71, 85, 101, 150),
    (86, 105, 151, 200),
    (106, 200, 201, 300),
]

CO_BREAKPOINTS_PPM = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

SO2_BREAKPOINTS_PPB = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
]

NO2_BREAKPOINTS_PPB = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
]


def _linear_aqi(conc: float, breakpoints: list[tuple[float, float, int, int]]) -> float | None:
    if conc is None:
        return None
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= conc <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo
    # above the top breakpoint: clamp to the max category
    last = breakpoints[-1]
    if conc > last[1]:
        return float(last[3])
    return None


# --- unit conversions (ug/m3 -> ppb/ppm) at 25C, 1 atm ---
def _ugm3_to_ppb(conc_ugm3: float, molecular_weight: float) -> float:
    return conc_ugm3 * 24.45 / molecular_weight


def _ugm3_to_ppm(conc_ugm3: float, molecular_weight: float) -> float:
    return _ugm3_to_ppb(conc_ugm3, molecular_weight) / 1000.0


def compute_us_aqi(pm2_5: float, pm10: float, o3: float, no2: float, so2: float, co: float) -> dict:
    """
    Given hourly pollutant concentrations in ug/m3 (OpenWeather's native units),
    return the overall US AQI (max of sub-indices) plus each sub-index.

    Note: EPA officially uses 24h/8h averages for some pollutants; for a
    real-time/hourly pipeline we approximate using instantaneous readings,
    which is the same simplification most open-source AQI trackers make.
    """
    sub_indices = {
        "aqi_pm2_5": _linear_aqi(pm2_5, PM25_BREAKPOINTS),
        "aqi_pm10": _linear_aqi(pm10, PM10_BREAKPOINTS),
        "aqi_o3": _linear_aqi(_ugm3_to_ppb(o3, 48.0), O3_BREAKPOINTS_PPB) if o3 is not None else None,
        "aqi_no2": _linear_aqi(_ugm3_to_ppb(no2, 46.0), NO2_BREAKPOINTS_PPB) if no2 is not None else None,
        "aqi_so2": _linear_aqi(_ugm3_to_ppb(so2, 64.0), SO2_BREAKPOINTS_PPB) if so2 is not None else None,
        "aqi_co": _linear_aqi(_ugm3_to_ppm(co, 28.0), CO_BREAKPOINTS_PPM) if co is not None else None,
    }
    valid = [v for v in sub_indices.values() if v is not None]
    overall = max(valid) if valid else None
    sub_indices["aqi"] = round(overall, 1) if overall is not None else None
    return sub_indices


def aqi_category(aqi: float) -> tuple[str, str]:
    from src.config import AQI_CATEGORIES
    if aqi is None:
        return "Unknown", "#999999"
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "#7e0023"
