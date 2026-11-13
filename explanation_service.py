from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap as _shap
    SHAP_AVAILABLE = True
except ImportError:
    _shap = None
    SHAP_AVAILABLE = False

from data_service import FEATURE_LABELS, get_feature_values
from prediction_service import get_features, _build_input, classify_pulse_shape, _apply_inverse



def get_shap_values(bundle: dict, row) -> dict:
    if not SHAP_AVAILABLE:
        return {"success": False, "error": "SHAP library is not installed."}
    model = bundle.get("total_model")
    features = get_features(bundle)
    if model is None:
        return {"success": False, "error": "total_model missing from bundle."}
    X = _build_input(row, features)
    missing = [f for f in features if pd.isna(X[f].iloc[0])]
    if missing:
        return {"success": False, "error": f"Missing feature values for SHAP: {missing}"}
    try:
        explainer = _shap.TreeExplainer(model)
        raw = np.asarray(explainer.shap_values(X))
        if raw.ndim == 3:
            values = raw[0, :, 0]
        elif raw.ndim == 2:
            values = raw[0]
        else:
            values = raw.reshape(-1)
        n = min(len(features), len(values))
        df = pd.DataFrame({
            "feature": features[:n],
            "label": [FEATURE_LABELS.get(f, f) for f in features[:n]],
            "grid_value": [float(X[f].iloc[0]) for f in features[:n]],
            "shap_value": [float(v) for v in values[:n]],
        })
        df["abs_shap"] = df["shap_value"].abs()
        df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
        df["direction"] = df["shap_value"].apply(
            lambda v: "pushes prediction higher" if v > 0 else ("pushes prediction lower" if v < 0 else "no contribution")
        )
        return {
            "success": True,
            "records": df.to_dict("records"),
            "note": "SHAP explains the saved model behaviour and does not establish causation. Values are on the log1p scale of the target.",
        }
    except Exception as exc:
        return {"success": False, "error": f"SHAP calculation failed: {exc}"}


def run_whatif_scenario(bundle: dict, row, changes: dict) -> dict:
    model = bundle.get("total_model")
    pulse_model = bundle.get("pulse_model")
    features = get_features(bundle)
    if model is None or not features:
        return {"success": False, "error": "Model not available for what-if."}
    baseline_X = _build_input(row, features)
    missing = [f for f in features if pd.isna(baseline_X[f].iloc[0])]
    if missing:
        return {"success": False, "error": f"Missing baseline features: {missing}"}
    invalid_changes = [k for k in changes if k not in features]
    if invalid_changes:
        return {"success": False, "error": f"Invalid feature names in scenario: {invalid_changes}"}
    scenario_X = baseline_X.copy()
    validated_changes = {}
    for k, v in changes.items():
        try:
            fv = float(v)
            if not np.isfinite(fv):
                return {"success": False, "error": f"Scenario value for {k!r} is not finite."}
            scenario_X[k] = fv
            validated_changes[k] = fv
        except (TypeError, ValueError):
            return {"success": False, "error": f"Invalid scenario value for {k!r}: {v!r}"}
    # Validate training bounds and feature order
    training_bounds = {
        "LConn_wt": (0.0, 6.7803),
        "MAD500_wt": (0.0, 331.5324),
        "BtA500_wt": (0.0, 87.4174),
        "Floor_Inte": (0.0, 2.7420),
        "POI_Densit": (0.0, 618.2615),
        "Transit_Ac": (0.0, 9824.3848),
        "IntDen_km2": (0.0, 353.7778),
        "land_use_e": (0.0, 0.7986),
    }
    bounds_warnings = []
    for k, v in validated_changes.items():
        if k in training_bounds:
            b_min, b_max = training_bounds[k]
            if v < b_min or v > b_max:
                bounds_warnings.append(
                    f"Scenario value for {k!r} ({v:,.2f}) is outside the training data range [{b_min:,.2f}, {b_max:,.2f}]."
                )

    try:
        raw_base = float(np.asarray(model.predict(baseline_X)).reshape(-1)[0])
        raw_scen = float(np.asarray(model.predict(scenario_X)).reshape(-1)[0])
        base_pred = max(0.0, _apply_inverse(raw_base, bundle))
        scen_pred = max(0.0, _apply_inverse(raw_scen, bundle))
        diff_abs = scen_pred - base_pred
        diff_pct = (diff_abs / base_pred * 100) if base_pred > 0 else None

        pulse_comparison = {"available": False}
        if pulse_model is not None:
            try:
                base_pulse_raw = np.asarray(pulse_model.predict(baseline_X))
                scen_pulse_raw = np.asarray(pulse_model.predict(scenario_X))

                def _extract_24h(arr):
                    if arr.ndim == 2 and arr.shape[1] >= 24:
                        return np.maximum(arr[0, :24].astype(float), 0.0).tolist()
                    elif arr.ndim == 1 and arr.size == 24:
                        return np.maximum(arr.astype(float), 0.0).tolist()
                    return None

                base_v = _extract_24h(base_pulse_raw)
                scen_v = _extract_24h(scen_pulse_raw)

                if base_v is not None and scen_v is not None:
                    assert len(base_v) == 24 and len(scen_v) == 24, "Pulse array length must be exactly 24"
                    base_pk = int(np.argmax(base_v))
                    scen_pk = int(np.argmax(scen_v))
                    base_lbl, base_desc = classify_pulse_shape(base_v)
                    scen_lbl, scen_desc = classify_pulse_shape(scen_v)

                    # Signed hourly differences (scenario - baseline)
                    signed_diffs = [s - b for b, s in zip(base_v, scen_v)]
                    pos_diffs = [(h, d) for h, d in enumerate(signed_diffs) if d > 0]
                    neg_diffs = [(h, d) for h, d in enumerate(signed_diffs) if d < 0]

                    largest_inc = max(pos_diffs, key=lambda x: x[1]) if pos_diffs else None
                    largest_dec = min(neg_diffs, key=lambda x: x[1]) if neg_diffs else None

                    pulse_comparison = {
                        "available": True,
                        "scale_note": "Pulse values represent log1p-scale activity intensity across the 24 hours (not raw count totals; they do not sum directly to the total-activity deployment prediction).",
                        "baseline_peak_hour": base_pk,
                        "scenario_peak_hour": scen_pk,
                        "peak_hour_shifted": bool(base_pk != scen_pk),
                        "baseline_peak_value": round(float(base_v[base_pk]), 2),
                        "scenario_peak_value": round(float(scen_v[scen_pk]), 2),
                        "baseline_label": base_lbl,
                        "scenario_label": scen_lbl,
                        "shape_changed": bool(base_lbl != scen_lbl),
                        "baseline_description": base_desc,
                        "scenario_description": scen_desc,
                        "baseline_values": [round(x, 2) for x in base_v],
                        "scenario_values": [round(x, 2) for x in scen_v],
                        "largest_increase": {
                            "hour": largest_inc[0],
                            "diff": round(largest_inc[1], 2),
                            "baseline_val": round(base_v[largest_inc[0]], 2),
                            "scenario_val": round(scen_v[largest_inc[0]], 2),
                        } if largest_inc else None,
                        "largest_decrease": {
                            "hour": largest_dec[0],
                            "diff": round(largest_dec[1], 2),
                            "baseline_val": round(base_v[largest_dec[0]], 2),
                            "scenario_val": round(scen_v[largest_dec[0]], 2),
                        } if largest_dec else None,
                    }
            except Exception as pe:
                logger.warning("What-if pulse calculation failed: %s", pe)

        return {
            "success": True,
            "baseline_prediction": base_pred,
            "scenario_prediction": scen_pred,
            "absolute_difference": diff_abs,
            "percent_difference": diff_pct,
            "changes_applied": validated_changes,
            "bounds_warnings": bounds_warnings,
            "pulse_comparison": pulse_comparison,
            "warning": "Model-based what-if scenario. Not a causal estimate. Changing a predictor in this tool does not guarantee the real-world outcome changes by this amount.",
        }
    except Exception as exc:
        return {"success": False, "error": f"What-if scenario failed: {exc}"}
