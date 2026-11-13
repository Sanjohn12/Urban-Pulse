from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from project_paths import MODEL_BUNDLE_PATH

logger = logging.getLogger(__name__)

MODEL_PATH = MODEL_BUNDLE_PATH


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    _validate_bundle(bundle)
    return bundle


def _validate_bundle(bundle: dict) -> None:
    required = ["total_model", "pulse_model", "features", "hour_columns",
                "total_target", "target_transform", "inverse_transform"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise ValueError(f"Model bundle missing keys: {missing}")
    if not isinstance(bundle["features"], list) or len(bundle["features"]) == 0:
        raise ValueError("Bundle 'features' must be a non-empty list.")
    logger.info("Bundle validated: features=%s, target_transform=%s",
                bundle["features"], bundle["target_transform"])


def get_features(bundle: dict) -> list[str]:
    return list(bundle.get("features", []))


def _build_input(row, features: list[str]) -> pd.DataFrame:
    values = {}
    for f in features:
        raw = row.get(f, np.nan)
        try:
            v = float(raw)
            values[f] = None if (pd.isna(v) or not np.isfinite(v)) else v
        except (TypeError, ValueError):
            values[f] = None
    return pd.DataFrame([values], columns=features)


def _apply_inverse(raw: float, bundle: dict) -> float:
    inverse = bundle.get("inverse_transform", "")
    if callable(inverse):
        try:
            return float(np.asarray(inverse(np.array([raw]))).reshape(-1)[0])
        except Exception:
            pass
    transform_str = str(inverse).lower().strip()
    if transform_str == "expm1" or "log1p" in str(bundle.get("target_transform", "")).lower():
        return float(np.expm1(raw))
    return raw


def predict_total(bundle: dict, row) -> dict:
    model = bundle.get("total_model")
    features = get_features(bundle)
    if model is None or not features:
        return {"success": False, "error": "total_model or features missing from bundle."}
    X = _build_input(row, features)
    missing = [f for f in features if pd.isna(X[f].iloc[0])]
    if missing:
        return {"success": False, "error": f"Missing feature values: {missing}"}
    try:
        raw = float(np.asarray(model.predict(X)).reshape(-1)[0])
        if not np.isfinite(raw):
            return {"success": False, "error": "Model returned non-finite prediction."}
        final = max(0.0, _apply_inverse(raw, bundle))
        return {
            "success": True, "value": final,
            "raw_prediction": raw,
            "model_type": "RandomForestRegressor",
            "target_transform": str(bundle.get("target_transform", "none")),
            "note": "This is a deployment prediction, not an OOF evaluation result.",
        }
    except Exception as exc:
        return {"success": False, "error": f"Prediction failed: {exc}"}


def predict_uncertainty(bundle: dict, row) -> dict:
    """
    Computes epistemic predictive uncertainty across the ensemble of decision trees.
    Returns:
      - tree_std: Standard deviation of tree predictions (log-scale)
      - lower_ci_95: 2.5th percentile prediction (activity count)
      - upper_ci_95: 97.5th percentile prediction (activity count)
      - confidence_score: Confidence % (100% = complete tree consensus)
      - confidence_tier: 'High Confidence', 'Moderate Confidence', or 'High Uncertainty'
    """
    model = bundle.get("total_model")
    features = get_features(bundle)
    if model is None or not hasattr(model, "estimators_"):
        return {"success": False, "error": "Model estimators unavailable for uncertainty quantification."}
    
    X = _build_input(row, features)
    missing = [f for f in features if pd.isna(X[f].iloc[0])]
    if missing:
        return {"success": False, "error": f"Missing feature values: {missing}"}

    try:
        X_arr = X.values
        # Predict across all individual trees
        tree_preds_log = np.array([tree.predict(X_arr)[0] for tree in model.estimators_])
        tree_std_log = float(np.std(tree_preds_log))
        
        # Raw activity scale (expm1)
        tree_preds_raw = np.expm1(np.maximum(0.0, tree_preds_log))
        lower_ci = float(np.percentile(tree_preds_raw, 2.5))
        upper_ci = float(np.percentile(tree_preds_raw, 97.5))
        median_ci = float(np.median(tree_preds_raw))
        
        # Relative confidence score (standard error normalized by standard benchmark ~ 1.5)
        # Low tree variance -> High confidence
        confidence_pct = max(0.0, min(100.0, 100.0 * (1.0 - (tree_std_log / 2.0))))
        
        if tree_std_log <= 0.65:
            tier = "High Confidence"
            tier_color = "#10B981"  # Emerald
            tier_desc = "Strong consensus across all 300 decision trees. Stable morphology pattern."
        elif tree_std_log <= 1.25:
            tier = "Moderate Confidence"
            tier_color = "#F59E0B"  # Amber
            tier_desc = "Moderate tree disagreement. Typical for transitional urban-rural fringes."
        else:
            tier = "High Uncertainty"
            tier_color = "#EF4444"  # Crimson
            tier_desc = "Severe tree divergence. High epistemic uncertainty; manual planning survey recommended."

        return {
            "success": True,
            "tree_std_log": round(tree_std_log, 3),
            "confidence_pct": round(confidence_pct, 1),
            "confidence_tier": tier,
            "tier_color": tier_color,
            "tier_desc": tier_desc,
            "lower_ci_95": round(lower_ci, 1),
            "upper_ci_95": round(upper_ci, 1),
            "median_prediction": round(median_ci, 1),
            "tree_count": len(model.estimators_),
        }
    except Exception as exc:
        return {"success": False, "error": f"Uncertainty computation failed: {exc}"}


def compute_uncertainty_batch(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes epistemic predictive uncertainty across all grid cells in a fast vectorized manner.
    Evaluates all trees in total_model across the entire input matrix X.
    Returns DataFrame indexed by grid_id with columns:
      - tree_std_log: float
      - confidence_pct: float
      - confidence_tier: 'High Confidence', 'Moderate Confidence', or 'High Uncertainty'
      - tier_color: hex color code (#10B981, #F59E0B, #EF4444)
      - lower_ci_95: float
      - upper_ci_95: float
    """
    model = bundle.get("total_model")
    features = get_features(bundle)
    if model is None or not hasattr(model, "estimators_"):
        return pd.DataFrame()

    # Build input feature matrix
    valid_features = [f for f in features if f in df.columns]
    if len(valid_features) < len(features):
        logger.warning("Missing features in batch uncertainty: %s", set(features) - set(valid_features))
        return pd.DataFrame()

    X = df[features].apply(pd.to_numeric, errors="coerce").values
    # Predict across each tree in model.estimators_
    # tree.predict(X) has shape (N_samples,)
    # all_preds has shape (N_trees, N_samples)
    all_preds_log = np.array([tree.predict(X) for tree in model.estimators_])

    # Tree standard deviation in log-space
    tree_std_log = np.std(all_preds_log, axis=0)

    # Confidence percentage (0 to 100%)
    conf_pct = np.clip(100.0 * (1.0 - (tree_std_log / 2.0)), 0.0, 100.0)

    # 95% Confidence interval in raw activity count
    all_preds_raw = np.expm1(np.maximum(0.0, all_preds_log))
    lower_ci = np.percentile(all_preds_raw, 2.5, axis=0)
    upper_ci = np.percentile(all_preds_raw, 97.5, axis=0)

    # Assign tiers
    # High: std <= 0.65, Moderate: 0.65 < std <= 1.25, High Uncertainty: > 1.25
    tiers = np.where(tree_std_log <= 0.65, "High Confidence",
             np.where(tree_std_log <= 1.25, "Moderate Confidence", "High Uncertainty"))

    tier_colors = np.where(tree_std_log <= 0.65, "#10B981",
                  np.where(tree_std_log <= 1.25, "#F59E0B", "#EF4444"))

    grid_ids = df["grid_id"].astype(str).values if "grid_id" in df.columns else df.index.astype(str).values

    res_df = pd.DataFrame({
        "grid_id": grid_ids,
        "tree_std_log": np.round(tree_std_log, 3),
        "confidence_pct": np.round(conf_pct, 1),
        "confidence_tier": tiers,
        "tier_color": tier_colors,
        "lower_ci_95": np.round(lower_ci, 1),
        "upper_ci_95": np.round(upper_ci, 1),
    })
    return res_df


def predict_pulse(bundle: dict, row) -> dict:
    model = bundle.get("pulse_model")
    features = get_features(bundle)
    hour_columns = bundle.get("hour_columns", [])
    if model is None or not features:
        return {"success": False, "type": "unavailable", "error": "pulse_model or features missing."}
    X = _build_input(row, features)
    missing = [f for f in features if pd.isna(X[f].iloc[0])]
    if missing:
        return {"success": False, "type": "unavailable", "error": f"Missing features: {missing}"}
    try:
        pred = np.asarray(model.predict(X))
        if pred.ndim == 2 and pred.shape[1] >= 24:
            values = np.maximum(pred[0, :24].astype(float), 0.0)
        elif pred.ndim == 1 and pred.size == 24:
            values = np.maximum(pred.astype(float), 0.0)
        elif pred.ndim == 1 and pred.size == 1:
            label = str(pred[0])
            confidence = None
            if hasattr(model, "predict_proba"):
                try:
                    confidence = float(np.max(model.predict_proba(X)[0]))
                except Exception:
                    pass
            return {"success": True, "type": "class", "label": label, "confidence": confidence}
        else:
            return {"success": False, "type": "unavailable", "error": f"Unexpected prediction shape: {pred.shape}"}
        peak_hour = int(np.argmax(values))
        raw_counts = np.expm1(values)
        raw_pulse_sum = float(raw_counts.sum())
        
        # Total predicted daily activity for reconciliation
        total_model = bundle.get("total_model")
        if total_model is not None:
            try:
                tot_raw = float(np.asarray(total_model.predict(X)).reshape(-1)[0])
                predicted_daily_total = max(0.0, _apply_inverse(tot_raw, bundle))
            except Exception:
                predicted_daily_total = raw_pulse_sum
        else:
            predicted_daily_total = raw_pulse_sum

        # Reconcile hourly pulse so that sum(hourly_activity) == predicted_daily_total
        if raw_pulse_sum > 0:
            reconciled_hourly = (raw_counts / raw_pulse_sum) * predicted_daily_total
        else:
            reconciled_hourly = np.full(24, predicted_daily_total / 24.0)

        peak_value_count = float(reconciled_hourly[peak_hour])
        peak_share = float(reconciled_hourly[peak_hour] / predicted_daily_total * 100) if predicted_daily_total > 0 else 0.0

        return {
            "success": True,
            "type": "hourly",
            "values": [round(float(x), 1) for x in reconciled_hourly],
            "log_values": [round(float(x), 3) for x in values],
            "peak_hour": peak_hour,
            "peak_value": peak_value_count,
            "peak_share_pct": peak_share,
            "total": float(predicted_daily_total),
            "hour_labels": hour_columns[:24] if hour_columns else [f"activity_{i:02d}" for i in range(24)],
        }
    except Exception as exc:
        return {"success": False, "type": "unavailable", "error": f"Pulse prediction failed: {exc}"}


def get_training_bounds(bundle: dict, features: list[str]) -> dict:
    training_ids = bundle.get("training_grid_ids", [])
    return {"n_training": len(training_ids), "features": features}


def classify_pulse_shape(values: list) -> tuple[str, str]:
    """Return (label, description) derived from the 24-hour activity values.

    Period boundaries use universal clock conventions (not arbitrary thresholds).
    Bimodal detection uses a relative threshold (local peaks >= 30% of global max)
    so it adapts automatically to any activity scale without hardcoding.
    """
    v = [float(x) for x in values]
    n = len(v)
    total = sum(v)
    if total == 0 or n == 0:
        return "flat", "No significant predicted activity across any hour."
    vn = [x / total for x in v]
    peak_hour = int(v.index(max(v)))
    # Natural-day periods (universal clock conventions)
    periods = [
        ("late-night (00\u201305)",  0, min(6, n)),
        ("morning (06\u201311)",     min(6, n),  min(12, n)),
        ("afternoon (12\u201317)",   min(12, n), min(18, n)),
        ("evening (18\u201323)",     min(18, n), n),
    ]
    shares = {name: sum(vn[s:e]) for name, s, e in periods}
    dominant = max(shares, key=shares.get)
    dom_pct = round(shares[dominant] * 100)
    # Relative bimodal detection: local peaks >= 30% of global max
    threshold = 0.30 * max(v)
    local_peaks = [
        i for i in range(1, n - 1)
        if v[i] >= v[i - 1] and v[i] >= v[i + 1] and v[i] >= threshold
    ]
    if len(local_peaks) >= 2:
        ph_strs = [f"{h:02d}:00" for h in local_peaks[:3]]
        label = "bimodal/multi-peak"
        desc = (
            f"Multiple activity peaks detected across the 24-hour cycle at {', '.join(ph_strs)}, "
            f"with the primary peak occurring at {peak_hour:02d}:00."
        )
    else:
        orient = dominant.split("(")[0].strip()
        label = orient
        desc = (
            f"Peak activity occurs at {peak_hour:02d}:00, with {dom_pct}% of the 24-hour predicted activity "
            f"concentrated in the {dominant} period."
        )
    return label, desc

