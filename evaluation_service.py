from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from data_service import OOF_OBSERVED_COL, OOF_PREDICTED_COL, OOF_DIFF_COL, OOF_ABS_COL


def activity_validation_metrics(observed_daily, predicted_daily, observed_hourly, predicted_hourly):
    """Paired raw-scale metrics shared by the protected experiment executor."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    yd, pdaily = np.asarray(observed_daily), np.asarray(predicted_daily)
    yh, ph = np.asarray(observed_hourly), np.asarray(predicted_hourly)
    if yd.ndim != 1 or pdaily.shape != yd.shape or yh.shape != (len(yd), 24) or ph.shape != yh.shape:
        raise ValueError("Daily/hourly evaluation shapes must match")
    if len(yd) < 2 or any(not np.isfinite(values).all() for values in (yd, pdaily, yh, ph)):
        raise ValueError("At least two finite paired observations are required")

    def metrics(actual, predicted):
        actual, predicted = actual.ravel(), predicted.ravel()
        # An undefined R² stays missing; never manufacture a perfect score.
        r2 = float(r2_score(actual, predicted)) if np.ptp(actual) > 0 else None
        return {
            "r2": r2, "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)), "n": int(actual.size),
            "bias_pred_minus_observed": float(np.mean(predicted - actual)),
        }
    delta = np.abs(np.argmax(yh, axis=1) - np.argmax(ph, axis=1))
    return {
        "daily": metrics(yd, pdaily), "hourly": metrics(yh, ph),
        "per_hour": [metrics(yh[:, hour], ph[:, hour]) for hour in range(24)],
        "peak_hour_circular_mae": float(np.mean(np.minimum(delta, 24 - delta))),
        "hourly_sum_mae": float(np.mean(np.abs(yh.sum(axis=1) - ph.sum(axis=1)))),
    }


def recalculate_pooled_metrics(gdf) -> dict:
    obs  = pd.to_numeric(gdf[OOF_OBSERVED_COL],  errors="coerce").dropna()
    pred = pd.to_numeric(gdf[OOF_PREDICTED_COL], errors="coerce")
    pred = pred[obs.index].dropna()
    obs  = obs[pred.index]
    if obs.empty:
        return {"available": False, "error": "No paired OOF obs/pred values."}
    residuals = obs - pred
    ss_res = float(((obs - pred) ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt((residuals ** 2).mean()))
    mae  = float(residuals.abs().mean())
    bias = float(residuals.mean())
    return {
        "available": True,
        "n_grids": len(obs),
        "r2": r2, "rmse": rmse, "mae": mae,
        "mean_bias": bias,
        "mean_observed": float(obs.mean()),
        "mean_predicted": float(pred.mean()),
        "note": "Pooled OOF Random Forest metrics computed from agent_eligible_grids_with_oof.csv",
    }


def get_oof_for_grid(table: pd.DataFrame, grid_id: str) -> dict:
    from data_service import normalize_id, OOF_FOLD_COL
    gid = normalize_id(grid_id)
    match = table[table["grid_id"].astype(str) == gid]
    if match.empty:
        return {"available": False, "grid_id": gid, "error": "Grid not found in evaluation table."}
    row = match.iloc[0]
    def _f(col):
        try:
            v = float(row.get(col, float("nan")))
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None
    obs   = _f(OOF_OBSERVED_COL)
    pred  = _f(OOF_PREDICTED_COL)
    diff  = _f(OOF_DIFF_COL)
    abse  = _f(OOF_ABS_COL)
    fold  = _f(OOF_FOLD_COL)
    if diff is None and obs is not None and pred is not None:
        diff = obs - pred
    if abse is None and diff is not None:
        abse = abs(diff)
    available = obs is not None and pred is not None
    interp = None
    if available and diff is not None:
        if diff > 0:
            interp = "The OOF model underestimated this grid: observed activity exceeded the prediction."
        elif diff < 0:
            interp = "The OOF model overestimated this grid: predicted activity exceeded observations."
        else:
            interp = "Observed and OOF-predicted activity match exactly in the stored data."
    return {
        "available": available, "grid_id": gid,
        "observed": obs, "predicted": pred, "diff": diff, "abs_error": abse,
        "fold": int(fold) if fold is not None else None,
        "interpretation": interp,
        "note": "OOF evaluation from cross-validation. Separate from deployment predictions.",
    }


# Standard column names for 24-hour observed and RF OOF predictions
HOURLY_OBS_COLS = [
    "activity_0", "activity_1", "activity_2", "activity_3", "activity_4", "activity_5",
    "activity_6", "activity_7", "activity_8", "activity_9", "activity10", "activity11",
    "activity12", "activity13", "activity14", "activity15", "activity16", "activity17",
    "activity18", "activity19", "activity20", "activity21", "activity22", "activity23"
]

HOURLY_RF_OOF_COLS = [f"random_forest_pred_hour_{h:02d}" for h in range(24)]


def get_hourly_oof_for_grid(row) -> dict:
    """
    Extracts 24-hour observed and spatial OOF predicted activity for a specific grid row.
    Computes hourly residuals, percentage mismatches, absolute errors, and peak hour metrics.
    Completely isolated from operational / deployment predictions.
    """
    if row is None:
        return {"available": False, "error": "No row provided."}

    # Extract observed values
    obs_vals = []
    has_obs = True
    for c in HOURLY_OBS_COLS:
        val = row.get(c, np.nan)
        try:
            val_f = float(val)
            if np.isfinite(val_f):
                obs_vals.append(max(0.0, val_f))
            else:
                has_obs = False
                break
        except (TypeError, ValueError):
            has_obs = False
            break

    # Extract RF OOF predicted values
    pred_vals = []
    has_pred = True
    for c in HOURLY_RF_OOF_COLS:
        val = row.get(c, np.nan)
        try:
            val_f = float(val)
            if np.isfinite(val_f):
                pred_vals.append(max(0.0, val_f))
            else:
                has_pred = False
                break
        except (TypeError, ValueError):
            has_pred = False
            break

    if not (has_obs and has_pred and len(obs_vals) == 24 and len(pred_vals) == 24):
        return {
            "available": False,
            "error": "Complete 24-hour observed or spatial OOF prediction values not found in row.",
        }

    obs_arr = np.array(obs_vals, dtype=float)
    pred_arr = np.array(pred_vals, dtype=float)

    # Residuals: obs - pred
    residuals = obs_arr - pred_arr
    abs_errors = np.abs(residuals)

    # Percentage mismatch safely: (obs - pred) / max(1.0, pred) * 100
    safe_denom = np.maximum(1.0, pred_arr)
    pct_mismatches = (residuals / safe_denom) * 100.0

    obs_peak_hour = int(np.argmax(obs_arr))
    pred_peak_hour = int(np.argmax(pred_arr))

    # Circular peak hour difference (0 to 12 hours)
    linear_diff = abs(obs_peak_hour - pred_peak_hour)
    circular_diff = min(linear_diff, 24 - linear_diff)

    return {
        "available": True,
        "grid_id": str(row.get("grid_id", "")),
        "district": str(row.get("district_name", row.get("district", "Unknown"))),
        "observed_hourly": [round(float(x), 2) for x in obs_arr],
        "oof_predicted_hourly": [round(float(x), 2) for x in pred_arr],
        "hourly_residuals": [round(float(x), 2) for x in residuals],
        "hourly_pct_mismatch": [round(float(x), 1) for x in pct_mismatches],
        "hourly_abs_error": [round(float(x), 2) for x in abs_errors],
        "observed_peak_hour": obs_peak_hour,
        "observed_peak_val": float(obs_arr[obs_peak_hour]),
        "oof_predicted_peak_hour": pred_peak_hour,
        "oof_predicted_peak_val": float(pred_arr[pred_peak_hour]),
        "peak_hour_diff": obs_peak_hour - pred_peak_hour,
        "peak_hour_circular_diff": circular_diff,
        "observed_hourly_sum": float(obs_arr.sum()),
        "oof_predicted_hourly_sum": float(pred_arr.sum()),
        "mean_hourly_mae": float(abs_errors.mean()),
        "note": "Evaluated strictly using 5-fold Spatial Out-Of-Fold (OOF) cross-validation.",
    }


def get_hourly_mismatch_sample(
    df: pd.DataFrame,
    district_filter: str = "All districts",
    mismatch_filter: str = "All categories",
    sample_size: int = 40,
    metric: str = "residual",
) -> tuple[pd.DataFrame, dict]:
    """
    Constructs a matrix of 24-hour temporal errors across an interactive subset of grid cells.
    metric: 'residual' (obs - pred) or 'pct_mismatch' ((obs - pred) / pred * 100).
    Filters can be applied by district and mismatch category (underpredicted, overpredicted, well-calibrated).
    """
    sub = df.copy()

    # Apply district filter
    dist_col = "district_name" if "district_name" in sub.columns else ("district" if "district" in sub.columns else None)
    if district_filter != "All districts" and dist_col:
        sub = sub[sub[dist_col] == district_filter]

    if sub.empty:
        return pd.DataFrame(), {"count": 0, "metric": metric}

    # Ensure OOF residual / pct mismatch is present for categorization
    if "oof_observed_minus_predicted" in sub.columns:
        diff_col = "oof_observed_minus_predicted"
    elif "mismatch" in sub.columns:
        diff_col = "mismatch"
    elif OOF_OBSERVED_COL in sub.columns and OOF_PREDICTED_COL in sub.columns:
        sub["oof_observed_minus_predicted"] = pd.to_numeric(sub[OOF_OBSERVED_COL], errors="coerce") - pd.to_numeric(sub[OOF_PREDICTED_COL], errors="coerce")
        diff_col = "oof_observed_minus_predicted"
    else:
        diff_col = None

    if diff_col and "mismatch_pct" not in sub.columns and OOF_PREDICTED_COL in sub.columns:
        denom = pd.to_numeric(sub[OOF_PREDICTED_COL], errors="coerce").replace(0, np.nan)
        sub["mismatch_pct"] = (pd.to_numeric(sub[diff_col], errors="coerce") / denom) * 100.0

    # Apply mismatch category filter
    if mismatch_filter != "All categories" and "mismatch_pct" in sub.columns:
        if mismatch_filter == "Underpredicted (> +10%)":
            sub = sub[sub["mismatch_pct"] > 10.0].sort_values("mismatch_pct", ascending=False)
        elif mismatch_filter == "Severe Underpredicted (> +50%)":
            sub = sub[sub["mismatch_pct"] > 50.0].sort_values("mismatch_pct", ascending=False)
        elif mismatch_filter == "Overpredicted (< -10%)":
            sub = sub[sub["mismatch_pct"] < -10.0].sort_values("mismatch_pct", ascending=True)
        elif mismatch_filter == "Severe Overpredicted (< -50%)":
            sub = sub[sub["mismatch_pct"] < -50.0].sort_values("mismatch_pct", ascending=True)
        elif mismatch_filter == "Well-Calibrated (-10% to +10%)":
            sub = sub[(sub["mismatch_pct"] >= -10.0) & (sub["mismatch_pct"] <= 10.0)]
        elif mismatch_filter == "Top Absolute Errors":
            if "oof_absolute_error" in sub.columns:
                sub = sub.sort_values("oof_absolute_error", ascending=False)
            elif diff_col:
                sub = sub.iloc[pd.to_numeric(sub[diff_col], errors="coerce").abs().argsort()[::-1]]
    else:
        if diff_col:
            # Sort by largest absolute discrepancy by default
            sub = sub.iloc[pd.to_numeric(sub[diff_col], errors="coerce").abs().argsort()[::-1]]

    # Take bounded sample
    sampled = sub.head(sample_size).copy()
    if sampled.empty:
        return pd.DataFrame(), {"count": 0, "metric": metric}

    # Build 24-hour matrix
    rows = []
    grid_labels = []
    for _, r in sampled.iterrows():
        gid = str(r.get("grid_id", ""))
        dist = str(r.get(dist_col, "")) if dist_col else ""
        label = f"Grid {gid} ({dist})" if dist else f"Grid {gid}"
        grid_labels.append(label)

        h_data = get_hourly_oof_for_grid(r)
        if h_data.get("available"):
            if metric == "pct_mismatch":
                vals = h_data["hourly_pct_mismatch"]
            else:
                vals = h_data["hourly_residuals"]
        else:
            vals = [np.nan] * 24
        rows.append(vals)

    hours = [f"{h:02d}:00" for h in range(24)]
    matrix_df = pd.DataFrame(rows, index=grid_labels, columns=hours)
    
    meta = {
        "count": len(matrix_df),
        "total_matching": len(sub),
        "metric": metric,
        "sample_size": sample_size,
    }
    return matrix_df, meta

