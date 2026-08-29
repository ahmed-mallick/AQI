from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def explain_prediction(model, model_type: str, scaler, X_row: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [feature, shap_value] sorted by
    absolute impact, for a single-row prediction.
    """
    feature_names = list(X_row.columns)

    if model_type == "random_forest":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
        values = shap_values[0] if isinstance(shap_values, list) else shap_values
        values = np.array(values).flatten()

    elif model_type == "ridge_regression":
        X_scaled = scaler.transform(X_row)
        bg_scaled = scaler.transform(background[feature_names])
        explainer = shap.LinearExplainer(model, bg_scaled)
        values = np.array(explainer.shap_values(X_scaled)).flatten()

    else:  # tensorflow_nn — model-agnostic fallback, sampled for speed
        X_scaled = scaler.transform(X_row)
        bg_scaled = scaler.transform(background[feature_names].sample(min(50, len(background)), random_state=42))
        explainer = shap.KernelExplainer(lambda x: model.predict(x, verbose=0).flatten(), bg_scaled)
        values = np.array(explainer.shap_values(X_scaled, nsamples=100)).flatten()

    df = pd.DataFrame({"feature": feature_names, "shap_value": values})
    df["abs_value"] = df["shap_value"].abs()
    return df.sort_values("abs_value", ascending=False).drop(columns="abs_value").reset_index(drop=True)
