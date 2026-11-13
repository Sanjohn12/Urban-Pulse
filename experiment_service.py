"""
experiment_service.py
---------------------
Step 3 of Theory-Guided Self-Evolving Urban Intelligence (TSE-UI).

Transforms theoretical propositions into mathematical candidate features/interaction terms,
and evaluates their empirical validity across spatial 5-fold cross-validation.
Evaluates both overall performance (R², RMSE, MAE) and focal resolution on the anomaly clusters.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from project_paths import workspace_path
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

# Base morphological predictors established in the research
BASE_FEATURES = [
    "LConn_wt",
    "MAD500_wt",
    "BtA500_wt",
    "Floor_Inte",
    "POI_Densit",
    "Transit_Ac",
    "IntDen_km2",
    "land_use_e",
]

TARGET_COL = "total_acti"
FOLD_COL = "fold"


def fit_activity_pair(X, daily, hourly, parameters):
    """Trusted executor primitive: same RF/log1p specification, two targets."""
    daily = np.asarray(daily, dtype=float)
    hourly = np.asarray(hourly, dtype=float)
    if (not np.isfinite(daily).all() or not np.isfinite(hourly).all()
            or (daily < 0).any() or (hourly < 0).any()):
        raise ValueError("Activity targets must be finite and nonnegative")
    total_model = RandomForestRegressor(**parameters)
    hourly_model = RandomForestRegressor(**parameters)
    total_model.fit(X, np.log1p(daily))
    hourly_model.fit(X, np.log1p(hourly))
    return total_model, hourly_model


def predict_activity_pair(models, X):
    """Raw activity units; no deployment-only hourly reconciliation."""
    return (
        np.maximum(np.expm1(models[0].predict(X)), 0.0),
        np.maximum(np.expm1(models[1].predict(X)), 0.0),
    )


def fit_activity_calibration(oof_daily, oof_hourly, daily, hourly):
    """Fit only on spatially cross-fitted predictions from the outer training set."""
    total = IsotonicRegression(out_of_bounds="clip", y_min=0.0).fit(oof_daily, daily)
    per_hour = [
        IsotonicRegression(out_of_bounds="clip", y_min=0.0).fit(oof_hourly[:, hour], hourly[:, hour])
        for hour in range(24)
    ]
    return total, per_hour


def calibrate_activity_pair(calibrators, daily, hourly):
    return (
        np.maximum(calibrators[0].predict(daily), 0.0),
        np.column_stack([np.maximum(model.predict(hourly[:, hour]), 0.0)
                         for hour, model in enumerate(calibrators[1])]),
    )

# Standard theoretical candidate feature definitions
CANDIDATE_THEORY_FEATURES = {
    "Transit_Permeability_Index": {
        "formula": "Transit_Ac * land_use_e",
        "theory": "Dynamic Land-Use and Activity Rhythms (Jacobs-Crisioni et al., 2014) & Natural Movement (Hillier et al., 1993)",
        "hypothesis": "Transit accessibility generates active human presence only when coupled with land-use diversity; in monofunctional or severance areas, transit operates as through-traffic bypass.",
        "compute": lambda df: df["Transit_Ac"] * df["land_use_e"],
    },
    "Walkable_Vitality_Mix": {
        "formula": "Floor_Inte * land_use_e",
        "theory": "Sidewalk Vitality & Urban Diversity (Jacobs, 1961)",
        "hypothesis": "Floor area intensity generates sustained activity when mixed-use entropy is high, preventing ghost monofunctional density.",
        "compute": lambda df: df["Floor_Inte"] * df["land_use_e"],
    },
    "Configurational_Attractor_Synergy": {
        "formula": "BtA500_wt * POI_Densit",
        "theory": "Space Syntax Through-Movement & Attractor Theory (Hillier et al., 1993)",
        "hypothesis": "High angular betweenness centrality (BtA500) magnifies the attraction potential of POI density by funneling pedestrian journey flows through those specific corridors.",
        "compute": lambda df: df["BtA500_wt"] * df["POI_Densit"],
    },
    "Pedestrian_Permeability_Synergy": {
        "formula": "MAD500_wt * IntDen_km2",
        "theory": "Space Syntax (Hillier & Hanson, 1984) & Geometric Urban Permeability (Salingaros, 2005)",
        "hypothesis": "High street intersection density provides route choice, but pedestrian volume scales effectively only when angular reach impedance is compact.",
        "compute": lambda df: df["MAD500_wt"] * df["IntDen_km2"],
    },
}

def register_candidate_feature(name: str, formula: str, theory: str, hypothesis: str, compute_fn) -> bool:
    """Registers a dynamically synthesized theoretical candidate feature safely."""
    try:
        CANDIDATE_THEORY_FEATURES[name] = {
            "formula": formula,
            "theory": theory,
            "hypothesis": hypothesis,
            "compute": compute_fn,
        }
        return True
    except Exception as exc:
        logger.error("Failed to register candidate feature %s: %s", name, exc)
        return False

def evaluate_theoretical_hypothesis(
    dataset_path: str,
    candidate_feature_name: str,
    base_features: Optional[List[str]] = None,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates a candidate feature derived from urban theory on the 5 spatial folds.
    
    Parameters:
      dataset_path: Path to the CSV dataset.
      candidate_feature_name: Name of theoretical interaction feature.
      base_features: Optional list of baseline predictors. Defaults to all 8 standard BASE_FEATURES.
      n_estimators: Number of Random Forest trees (default: 300, matching baseline exactly).
      random_state: Seed for reproducibility (default: 42).
      
    Returns:
      - Baseline 5-fold spatial OOF metrics (R², RMSE, MAE, Mean Bias)
      - Evolved model 5-fold spatial OOF metrics
      - Multi-baseline comparisons against both Original Baseline and Stage 1 (EXP-008)
      - Anomaly resolution metrics (change in error on Archetype A and Archetype B grids)
      - Scientific verdict (ACCEPTED / REJECTED) with comprehensive multi-metric trade-off rationale
    """
    if candidate_feature_name not in CANDIDATE_THEORY_FEATURES:
        return {
            "success": False,
            "error": f"Unknown candidate feature: {candidate_feature_name}. Available: {list(CANDIDATE_THEORY_FEATURES.keys())}",
        }

    # Use specified baseline features or fallback to standard 8
    active_base_features = list(base_features) if base_features else list(BASE_FEATURES)
    if not active_base_features:
        return {"success": False, "error": "At least one baseline feature must be selected."}

    spec = CANDIDATE_THEORY_FEATURES[candidate_feature_name]
    logger.info("Evaluating theoretical feature: %s with %d base features and %d trees", candidate_feature_name, len(active_base_features), n_estimators)

    # Load spatial dataset
    try:
        dataset_path = workspace_path(dataset_path)
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        return {"success": False, "error": f"Failed to load dataset: {exc}"}

    required_cols = list(set(active_base_features + [TARGET_COL, FOLD_COL]))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return {"success": False, "error": f"Missing columns in dataset: {missing}"}

    # Clean data
    df = df.dropna(subset=required_cols).copy()
    y_raw = df[TARGET_COL].values
    y_log = np.log1p(np.maximum(0, y_raw))
    folds = df[FOLD_COL].values

    # Compute candidate feature
    df[candidate_feature_name] = spec["compute"](df)

    # Containers for out-of-fold predictions
    oof_baseline_log = np.zeros(len(df))
    oof_evolved_log = np.zeros(len(df))

    unique_folds = sorted(np.unique(folds))
    fold_metrics = []

    for fold in unique_folds:
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_train_base = df.iloc[train_idx][active_base_features].values
        X_val_base = df.iloc[val_idx][active_base_features].values
        y_train = y_log[train_idx]
        y_val_raw = y_raw[val_idx]

        # 1. Baseline Model (matching final baseline hyperparameters exactly)
        rf_base = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        rf_base.fit(X_train_base, y_train)
        oof_baseline_log[val_idx] = rf_base.predict(X_val_base)
        fold_base_pred = np.clip(np.expm1(oof_baseline_log[val_idx]), 0.0, None)

        # 2. Evolved Model (Active Base + Theory Feature, matching hyperparameters exactly)
        evolved_features = active_base_features + [candidate_feature_name]
        X_train_evo = df.iloc[train_idx][evolved_features].values
        X_val_evo = df.iloc[val_idx][evolved_features].values

        rf_evo = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        rf_evo.fit(X_train_evo, y_train)
        oof_evolved_log[val_idx] = rf_evo.predict(X_val_evo)
        fold_evo_pred = np.clip(np.expm1(oof_evolved_log[val_idx]), 0.0, None)

        # Fold-level metrics
        f_base_r2 = float(r2_score(y_val_raw, fold_base_pred))
        f_base_rmse = float(np.sqrt(mean_squared_error(y_val_raw, fold_base_pred)))
        f_base_mae = float(mean_absolute_error(y_val_raw, fold_base_pred))
        f_base_bias = float(np.mean(fold_base_pred - y_val_raw))

        f_evo_r2 = float(r2_score(y_val_raw, fold_evo_pred))
        f_evo_rmse = float(np.sqrt(mean_squared_error(y_val_raw, fold_evo_pred)))
        f_evo_mae = float(mean_absolute_error(y_val_raw, fold_evo_pred))
        f_evo_bias = float(np.mean(fold_evo_pred - y_val_raw))

        fold_metrics.append({
            "fold": int(fold),
            "n_test": len(y_val_raw),
            "baseline": {"r2": f_base_r2, "rmse": f_base_rmse, "mae": f_base_mae, "bias": f_base_bias},
            "evolved": {"r2": f_evo_r2, "rmse": f_evo_rmse, "mae": f_evo_mae, "bias": f_evo_bias},
            "delta_r2": f_evo_r2 - f_base_r2,
            "delta_rmse": f_evo_rmse - f_base_rmse,
            "delta_mae": f_evo_mae - f_base_mae,
            "delta_bias": f_evo_bias - f_base_bias,
        })

    # Convert OOF predictions back to raw scale via expm1
    oof_baseline_raw = np.clip(np.expm1(oof_baseline_log), 0.0, None)
    oof_evolved_raw = np.clip(np.expm1(oof_evolved_log), 0.0, None)

    # Compute Pooled Metrics (Raw Activity Units)
    base_r2 = float(r2_score(y_raw, oof_baseline_raw))
    base_rmse = float(np.sqrt(mean_squared_error(y_raw, oof_baseline_raw)))
    base_mae = float(mean_absolute_error(y_raw, oof_baseline_raw))
    base_bias = float(np.mean(oof_baseline_raw - y_raw))

    evo_r2 = float(r2_score(y_raw, oof_evolved_raw))
    evo_rmse = float(np.sqrt(mean_squared_error(y_raw, oof_evolved_raw)))
    evo_mae = float(mean_absolute_error(y_raw, oof_evolved_raw))
    evo_bias = float(np.mean(oof_evolved_raw - y_raw))

    # Deltas against Original In-Run Baseline
    delta_r2 = evo_r2 - base_r2
    delta_rmse = evo_rmse - base_rmse
    delta_mae = evo_mae - base_mae
    delta_bias = evo_bias - base_bias

    # District metrics analysis (Colombo vs Gampaha)
    district_results = {}
    if "district_name" in df.columns:
        for dist in sorted(df["district_name"].dropna().unique()):
            d_mask = (df["district_name"] == dist).to_numpy()
            d_actual = y_raw[d_mask]
            d_base = oof_baseline_raw[d_mask]
            d_evo = oof_evolved_raw[d_mask]

            district_results[dist] = {
                "n": int(np.sum(d_mask)),
                "baseline": {
                    "r2": float(r2_score(d_actual, d_base)),
                    "rmse": float(np.sqrt(mean_squared_error(d_actual, d_base))),
                    "mae": float(mean_absolute_error(d_actual, d_base)),
                    "bias": float(np.mean(d_base - d_actual)),
                },
                "evolved": {
                    "r2": float(r2_score(d_actual, d_evo)),
                    "rmse": float(np.sqrt(mean_squared_error(d_actual, d_evo))),
                    "mae": float(mean_absolute_error(d_actual, d_evo)),
                    "bias": float(np.mean(d_evo - d_actual)),
                },
                "delta_r2": float(r2_score(d_actual, d_evo) - r2_score(d_actual, d_base)),
                "delta_rmse": float(np.sqrt(mean_squared_error(d_actual, d_evo)) - np.sqrt(mean_squared_error(d_actual, d_base))),
                "delta_mae": float(mean_absolute_error(d_actual, d_evo) - mean_absolute_error(d_actual, d_base)),
                "delta_bias": float(np.mean(d_evo - d_actual) - np.mean(d_base - d_actual)),
            }

    # Multi-baseline comparison: Load EXP-008 from scientific memory if available
    exp008_metrics = None
    deltas_vs_exp008 = None
    try:
        from registry_service import load_scientific_memory
        mem = load_scientific_memory()
        exp008_entry = next((e for e in mem.get("experiments", []) if e.get("id") == "EXP-008"), None)
        if exp008_entry and "evolved_metrics" in exp008_entry:
            e8 = exp008_entry["evolved_metrics"]
            exp008_metrics = {
                "r2": float(e8["r2"]),
                "rmse": float(e8["rmse"]),
                "mae": float(e8["mae"]),
                "mean_bias": float(e8["mean_bias"]),
            }
            deltas_vs_exp008 = {
                "delta_r2": evo_r2 - exp008_metrics["r2"],
                "delta_rmse": evo_rmse - exp008_metrics["rmse"],
                "delta_mae": evo_mae - exp008_metrics["mae"],
                "delta_bias": evo_bias - exp008_metrics["mean_bias"],
            }
    except Exception as exc:
        logger.warning("Could not load EXP-008 reference for comparison: %s", exc)

    # Anomaly Cluster Resolution Check
    # Identify Overprediction Archetype (Obs << Pred in baseline)
    base_diff = y_raw - oof_baseline_raw
    evo_diff = y_raw - oof_evolved_raw
    
    overpred_mask = (base_diff / np.maximum(1.0, oof_baseline_raw)) < -0.5
    underpred_mask = (base_diff / np.maximum(1.0, oof_baseline_raw)) > 0.5

    overpred_count = int(np.sum(overpred_mask))
    underpred_count = int(np.sum(underpred_mask))

    base_overpred_mae = float(np.mean(np.abs(base_diff[overpred_mask]))) if overpred_count > 0 else 0.0
    evo_overpred_mae = float(np.mean(np.abs(evo_diff[overpred_mask]))) if overpred_count > 0 else 0.0
    overpred_mae_change = evo_overpred_mae - base_overpred_mae

    base_underpred_mae = float(np.mean(np.abs(base_diff[underpred_mask]))) if underpred_count > 0 else 0.0
    evo_underpred_mae = float(np.mean(np.abs(evo_diff[underpred_mask]))) if underpred_count > 0 else 0.0
    underpred_mae_change = evo_underpred_mae - base_underpred_mae

    # Multi-Metric Decision Rule:
    # 1. R² must improve (delta_r2 > 0)
    # 2. RMSE must improve or not worsen (delta_rmse <= 0)
    # 3. MAE must not deteriorate by more than 1% (delta_mae <= 0.01 * base_mae)
    # 4. Overprediction and underprediction error groups must not both substantially degrade
    r2_improved = delta_r2 > 0.0
    rmse_improved_or_flat = delta_rmse <= 0.0
    mae_not_deteriorated = delta_mae <= (0.01 * base_mae)
    anomaly_not_degraded = not (overpred_mae_change > 0 and underpred_mae_change > 0)

    accepted = r2_improved and rmse_improved_or_flat and mae_not_deteriorated and anomaly_not_degraded

    # Trade-off identification for scientific rationale
    trade_off_notes = []
    if r2_improved and delta_mae > 0:
        trade_off_notes.append(f"MAE increased slightly by {delta_mae:+.1f} (+{(delta_mae/base_mae)*100:.2f}%) despite R² gain")
    if r2_improved and overpred_mae_change > 0:
        trade_off_notes.append(f"Overprediction cluster MAE worsened by {overpred_mae_change:+.1f}")
    if r2_improved and underpred_mae_change > 0:
        trade_off_notes.append(f"Underprediction cluster MAE worsened by {underpred_mae_change:+.1f}")
    if not r2_improved and delta_rmse < 0:
        trade_off_notes.append(f"RMSE improved by {delta_rmse:.1f} but overall R² dropped by {delta_r2:.4f}")

    trade_off_str = f" [Trade-offs: {'; '.join(trade_off_notes)}]" if trade_off_notes else ""

    verdict = {
        "status": "ACCEPTED" if accepted else "REJECTED",
        "scientific_rationale": (
            f"Candidate feature '{candidate_feature_name}' demonstrated empirical advantage across criteria: "
            f"ΔR² = {delta_r2:+.4f}, ΔRMSE = {delta_rmse:+.1f}, ΔMAE = {delta_mae:+.1f}, "
            f"Overprediction MAE change = {overpred_mae_change:+.1f}, Underprediction MAE change = {underpred_mae_change:+.1f}."
            f"{trade_off_str}"
            if accepted else
            f"Candidate feature '{candidate_feature_name}' failed multi-metric acceptance criteria against baseline: "
            f"ΔR² = {delta_r2:+.4f}, ΔRMSE = {delta_rmse:+.1f}, ΔMAE = {delta_mae:+.1f}."
            f"{trade_off_str}"
        ),
    }

    return {
        "success": True,
        "learning_stage": "Theoretical Learning",
        "method": "Theory-Guided Candidate Interaction Feature",
        "feature_name": candidate_feature_name,
        "formula": spec["formula"],
        "feature_definition": {key: value for key, value in spec.items() if key != "compute"},
        "theoretical_grounding": spec["theory"],
        "hypothesis": spec["hypothesis"],
        "n_grids": len(df),
        "baseline_metrics": {
            "r2": base_r2,
            "rmse": base_rmse,
            "mae": base_mae,
            "mean_bias": base_bias,
        },
        "evolved_metrics": {
            "r2": evo_r2,
            "rmse": evo_rmse,
            "mae": evo_mae,
            "mean_bias": evo_bias,
        },
        "deltas_vs_baseline": {
            "delta_r2": delta_r2,
            "delta_rmse": delta_rmse,
            "delta_mae": delta_mae,
            "delta_bias": delta_bias,
        },
        "deltas": {
            "delta_r2": delta_r2,
            "delta_rmse": delta_rmse,
            "delta_mae": delta_mae,
            "delta_bias": delta_bias,
        },
        "exp008_stage1_metrics": exp008_metrics,
        "deltas_vs_stage1_exp008": deltas_vs_exp008,
        "anomaly_resolution": {
            "overprediction_cluster_grids": overpred_count,
            "overprediction_mae_baseline": base_overpred_mae,
            "overprediction_mae_evolved": evo_overpred_mae,
            "overprediction_mae_change": overpred_mae_change,
            "underprediction_cluster_grids": underpred_count,
            "underprediction_mae_baseline": base_underpred_mae,
            "underprediction_mae_evolved": evo_underpred_mae,
            "underprediction_mae_change": underpred_mae_change,
        },
        "fold_metrics": fold_metrics,
        "district_metrics": district_results,
        "verdict": verdict,
        "base_features": active_base_features,
        "evolved_features": active_base_features + [candidate_feature_name],
        "model_configuration": rf_base.get_params(),
        "dataset_path": str(workspace_path(dataset_path)),
        "target": TARGET_COL,
        "fold_column": FOLD_COL,
        "target_transform": "log1p",
        "inverse_transform": "expm1",
    }


def evaluate_statistical_isotonic_calibration(
    dataset_path: str,
    base_features: Optional[List[str]] = None,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Step 3 (Statistical Learning Track):
    Evaluates S1 — Fold-specific Non-linear Isotonic Calibration on Log1p Random Forest.

    Hypothesis:
      The log1p target transform normalizes extreme variances for tree construction, but
      compresses peak outputs upon expm1 inverse transformation (inducing severe negative bias
      in Q4 high-activity cells). Non-parametric monotonic isotonic calibration fitted on in-fold
      OOB predictions will recalibrate the response curve without distorting low-activity cells.

    Data Leakage Controls:
      - 5-fold spatial cross-validation partitioning is preserved identically.
      - OOB predictions (rf.oob_prediction_) originate solely from in-fold bootstrap exclusions.
      - IsotonicRegression is trained exclusively on (oob_pred_raw, y_train).
      - Calibration is then applied to the unobserved held-out spatial test fold.
    """
    active_base_features = list(base_features) if base_features else list(BASE_FEATURES)

    # 1. Load dataset
    try:
        dataset_path = workspace_path(dataset_path)
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        return {"success": False, "error": f"Failed to load dataset: {exc}"}

    required = active_base_features + [TARGET_COL, FOLD_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"success": False, "error": f"Missing required columns in dataset: {missing}"}

    df = df.dropna(subset=required).copy()
    y_raw = df[TARGET_COL].to_numpy(dtype=float)
    y_log = np.log1p(np.maximum(0.0, y_raw))
    folds = df[FOLD_COL].to_numpy()
    unique_folds = sorted(np.unique(folds))

    oof_baseline_raw = np.zeros(len(df), dtype=float)
    oof_calibrated_raw = np.zeros(len(df), dtype=float)

    fold_metrics = []

    for fold in unique_folds:
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_train = df.iloc[train_idx][active_base_features].values
        X_val = df.iloc[val_idx][active_base_features].values
        y_train_log = y_log[train_idx]
        y_train_raw = y_raw[train_idx]
        y_val_raw = y_raw[val_idx]

        # Fit Random Forest with OOB enabled
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features=1.0,
            bootstrap=True,
            oob_score=True,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train_log)

        # Baseline predictions for the held-out fold
        raw_test_pred = np.clip(np.expm1(rf.predict(X_val)), 0.0, None)
        oof_baseline_raw[val_idx] = raw_test_pred

        # In-fold training OOB predictions
        oob_log = rf.oob_prediction_
        oob_raw = np.clip(np.expm1(oob_log), 0.0, None)

        valid_oob = np.isfinite(oob_raw) & np.isfinite(y_train_raw)
        if valid_oob.sum() < 100:
            return {"success": False, "error": f"Too few valid OOB samples in fold {fold}"}

        # Fit isotonic calibrator strictly on training-fold OOB
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
        calibrator.fit(oob_raw[valid_oob], y_train_raw[valid_oob])

        # Apply to test fold
        calibrated_test_pred = np.clip(calibrator.predict(raw_test_pred), 0.0, None)
        oof_calibrated_raw[val_idx] = calibrated_test_pred

        # Fold metrics
        base_r2_f = float(r2_score(y_val_raw, raw_test_pred))
        base_rmse_f = float(np.sqrt(mean_squared_error(y_val_raw, raw_test_pred)))
        base_mae_f = float(mean_absolute_error(y_val_raw, raw_test_pred))
        base_bias_f = float(np.mean(raw_test_pred - y_val_raw))

        cal_r2_f = float(r2_score(y_val_raw, calibrated_test_pred))
        cal_rmse_f = float(np.sqrt(mean_squared_error(y_val_raw, calibrated_test_pred)))
        cal_mae_f = float(mean_absolute_error(y_val_raw, calibrated_test_pred))
        cal_bias_f = float(np.mean(calibrated_test_pred - y_val_raw))

        fold_metrics.append({
            "fold": int(fold),
            "n_test": len(y_val_raw),
            "baseline": {"r2": base_r2_f, "rmse": base_rmse_f, "mae": base_mae_f, "bias": base_bias_f},
            "calibrated": {"r2": cal_r2_f, "rmse": cal_rmse_f, "mae": cal_mae_f, "bias": cal_bias_f},
            "delta_r2": cal_r2_f - base_r2_f,
            "delta_rmse": cal_rmse_f - base_rmse_f,
            "delta_mae": cal_mae_f - base_mae_f,
        })

    # Pooled metrics
    base_r2 = float(r2_score(y_raw, oof_baseline_raw))
    base_rmse = float(np.sqrt(mean_squared_error(y_raw, oof_baseline_raw)))
    base_mae = float(mean_absolute_error(y_raw, oof_baseline_raw))
    base_bias = float(np.mean(oof_baseline_raw - y_raw))

    cal_r2 = float(r2_score(y_raw, oof_calibrated_raw))
    cal_rmse = float(np.sqrt(mean_squared_error(y_raw, oof_calibrated_raw)))
    cal_mae = float(mean_absolute_error(y_raw, oof_calibrated_raw))
    cal_bias = float(np.mean(oof_calibrated_raw - y_raw))

    delta_r2 = cal_r2 - base_r2
    delta_rmse = cal_rmse - base_rmse
    delta_mae = cal_mae - base_mae
    delta_bias = cal_bias - base_bias

    # Quartile metrics analysis
    quartile_bins = pd.qcut(df[TARGET_COL], q=4, labels=["Q1_Lowest", "Q2_Lower_Middle", "Q3_Upper_Middle", "Q4_Highest"])
    quartile_results = {}

    for q_label in ["Q1_Lowest", "Q2_Lower_Middle", "Q3_Upper_Middle", "Q4_Highest"]:
        q_mask = (quartile_bins == q_label).to_numpy()
        q_actual = y_raw[q_mask]
        q_base_pred = oof_baseline_raw[q_mask]
        q_cal_pred = oof_calibrated_raw[q_mask]

        quartile_results[q_label] = {
            "n": int(np.sum(q_mask)),
            "actual_mean": float(np.mean(q_actual)),
            "baseline": {
                "pred_mean": float(np.mean(q_base_pred)),
                "bias": float(np.mean(q_base_pred - q_actual)),
                "mae": float(mean_absolute_error(q_actual, q_base_pred)),
                "rmse": float(np.sqrt(mean_squared_error(q_actual, q_base_pred))),
                "r2": float(r2_score(q_actual, q_base_pred)),
            },
            "calibrated": {
                "pred_mean": float(np.mean(q_cal_pred)),
                "bias": float(np.mean(q_cal_pred - q_actual)),
                "mae": float(mean_absolute_error(q_actual, q_cal_pred)),
                "rmse": float(np.sqrt(mean_squared_error(q_actual, q_cal_pred))),
                "r2": float(r2_score(q_actual, q_cal_pred)),
            },
            "bias_change": float(np.mean(q_cal_pred - q_actual) - np.mean(q_base_pred - q_actual)),
            "mae_change": float(mean_absolute_error(q_actual, q_cal_pred) - mean_absolute_error(q_actual, q_base_pred)),
        }

    # District metrics analysis
    district_results = {}
    if "district_name" in df.columns:
        for dist in sorted(df["district_name"].dropna().unique()):
            d_mask = (df["district_name"] == dist).to_numpy()
            d_actual = y_raw[d_mask]
            d_base = oof_baseline_raw[d_mask]
            d_cal = oof_calibrated_raw[d_mask]

            district_results[dist] = {
                "n": int(np.sum(d_mask)),
                "baseline": {
                    "r2": float(r2_score(d_actual, d_base)),
                    "rmse": float(np.sqrt(mean_squared_error(d_actual, d_base))),
                    "mae": float(mean_absolute_error(d_actual, d_base)),
                    "bias": float(np.mean(d_base - d_actual)),
                },
                "calibrated": {
                    "r2": float(r2_score(d_actual, d_cal)),
                    "rmse": float(np.sqrt(mean_squared_error(d_actual, d_cal))),
                    "mae": float(mean_absolute_error(d_actual, d_cal)),
                    "bias": float(np.mean(d_cal - d_actual)),
                },
                "delta_r2": float(r2_score(d_actual, d_cal) - r2_score(d_actual, d_base)),
                "delta_mae": float(mean_absolute_error(d_actual, d_cal) - mean_absolute_error(d_actual, d_base)),
            }

    # Scientific Anomaly Resolution (Archetype A & Archetype B)
    base_diff = y_raw - oof_baseline_raw
    cal_diff = y_raw - oof_calibrated_raw

    overpred_mask = (base_diff / np.maximum(1.0, oof_baseline_raw)) < -0.5
    underpred_mask = (base_diff / np.maximum(1.0, oof_baseline_raw)) > 0.5

    overpred_count = int(np.sum(overpred_mask))
    underpred_count = int(np.sum(underpred_mask))

    base_overpred_mae = float(np.mean(np.abs(base_diff[overpred_mask]))) if overpred_count > 0 else 0.0
    cal_overpred_mae = float(np.mean(np.abs(cal_diff[overpred_mask]))) if overpred_count > 0 else 0.0

    base_underpred_mae = float(np.mean(np.abs(base_diff[underpred_mask]))) if underpred_count > 0 else 0.0
    cal_underpred_mae = float(np.mean(np.abs(cal_diff[underpred_mask]))) if underpred_count > 0 else 0.0

    # Evaluator verdict: Check whether R2 improved or RMSE decreased, and analyze trade-offs
    accepted = (delta_r2 > 0.0 or delta_rmse < 0.0)
    verdict = {
        "status": "ACCEPTED" if accepted else "REJECTED",
        "scientific_rationale": (
            f"Fold-specific Isotonic Calibration demonstrated empirical advantage: "
            f"ΔR² = {delta_r2:+.4f}, ΔRMSE = {delta_rmse:+.1f}, ΔMAE = {delta_mae:+.1f}, "
            f"Overall bias change = {delta_bias:+.1f}. "
            f"Q4 high-activity bias changed by {quartile_results['Q4_Highest']['bias_change']:+.1f}."
            if accepted else
            f"Fold-specific Isotonic Calibration failed to improve spatial cross-validation baseline: "
            f"ΔR² = {delta_r2:+.4f}, ΔRMSE = {delta_rmse:+.1f}, ΔMAE = {delta_mae:+.1f}."
        ),
    }

    return {
        "success": True,
        "learning_stage": "Statistical Learning",
        "method": "Fold-specific Non-linear Isotonic Calibration",
        "feature_name": "Log1p_RF_Isotonic_Calibration",
        "formula": "IsotonicRegression(out_of_bounds='clip').fit(expm1(oob_pred), y_train)",
        "theoretical_grounding": "Statistical Learning: Non-parametric monotonic recalibration for heteroscedastic scale distortion",
        "hypothesis": (
            "Log-transform compresses the dynamic range of high-activity clusters upon exponential inversion. "
            "Fold-specific isotonic calibration maps monotonic non-linearities to alleviate peak underprediction "
            "without distorting lower-activity baseline cells."
        ),
        "n_grids": len(df),
        "baseline_metrics": {
            "r2": base_r2,
            "rmse": base_rmse,
            "mae": base_mae,
            "mean_bias": base_bias,
        },
        "evolved_metrics": {
            "r2": cal_r2,
            "rmse": cal_rmse,
            "mae": cal_mae,
            "mean_bias": cal_bias,
        },
        "deltas": {
            "delta_r2": delta_r2,
            "delta_rmse": delta_rmse,
            "delta_mae": delta_mae,
            "delta_bias": delta_bias,
        },
        "fold_metrics": fold_metrics,
        "quartile_metrics": quartile_results,
        "district_metrics": district_results,
        "anomaly_resolution": {
            "overprediction_cluster_grids": overpred_count,
            "overprediction_mae_baseline": base_overpred_mae,
            "overprediction_mae_evolved": cal_overpred_mae,
            "overprediction_mae_change": cal_overpred_mae - base_overpred_mae,
            "underprediction_cluster_grids": underpred_count,
            "underprediction_mae_baseline": base_underpred_mae,
            "underprediction_mae_evolved": cal_underpred_mae,
            "underprediction_mae_change": cal_underpred_mae - base_underpred_mae,
        },
        "verdict": verdict,
        "base_features": active_base_features,
        "n_estimators": n_estimators,
        "model_configuration": rf.get_params(),
        "calibration_configuration": calibrator.get_params(),
        "dataset_path": str(workspace_path(dataset_path)),
        "target": TARGET_COL,
        "fold_column": FOLD_COL,
        "target_transform": "log1p",
        "inverse_transform": "expm1",
    }


def evaluate_data_source_experiment(
    dataset_path: str,
    feature_name: str,
    feature_source_path: Optional[str] = None,
    feature_series: Optional[pd.Series] = None,
    theoretical_grounding: str = "Spacemate Density Theory (Berghauser Pont & Haupt, 2010)",
    hypothesis: str = "Adding ground space index / building coverage disambiguates vertical high-rise compact forms from horizontal sprawling low-rise forms.",
    use_champion_model: bool = True,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Step 4: Learning from Additional Data Sources.
    
    Evaluates an additional urban data layer across spatial 5-fold cross-validation.
    Crucially adheres to Progressive Learning:
      - Builds cumulatively on the accepted Champion model (Baseline 8 + EXP-010 T2).
      - Compares performance against BOTH the Original Reference Baseline and EXP-010.
      - Uses identical 300-tree Random Forest and spatial fold partitions (N=7,764).
    """
    from registry_service import get_champion_model_state
    champ_state = get_champion_model_state()

    # Cumulative feature progression: Baseline 8 + Accepted T2
    if use_champion_model and champ_state.get("has_champion"):
        base_features = list(champ_state["features"])
        active_reference_id = champ_state["champion_id"]
        reference_r2 = champ_state["champion_metrics"]["r2"]
    else:
        base_features = list(BASE_FEATURES)
        active_reference_id = "BASELINE"
        reference_r2 = 0.5156

    logger.info("Evaluating Data Source experiment: %s on top of %s (%d base features)", feature_name, active_reference_id, len(base_features))

    try:
        dataset_path = workspace_path(dataset_path)
        if feature_source_path is not None:
            feature_source_path = workspace_path(feature_source_path)
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        return {"success": False, "error": f"Failed to load primary dataset: {exc}"}

    # Ensure required columns
    required = list(base_features) + [TARGET_COL, FOLD_COL]
    
    # If feature is already in df, use it directly
    feature_input_kind = (
        "dataset_column" if feature_name in df.columns else
        "series" if feature_series is not None else "source_file"
    )
    if feature_name in df.columns:
        pass
    elif feature_series is not None:
        df[feature_name] = feature_series
    elif feature_source_path and str(feature_source_path).endswith(('.shp', '.gpkg', '.geojson')):
        try:
            import geopandas as gpd
            src_gdf = gpd.read_file(workspace_path(feature_source_path))
            if feature_name in src_gdf.columns and "grid_id" in src_gdf.columns:
                sub = src_gdf[["grid_id", feature_name]].drop_duplicates(subset=["grid_id"])
                df = df.merge(sub, on="grid_id", how="left")
            else:
                return {"success": False, "error": f"Feature {feature_name} or grid_id not found in {feature_source_path}"}
        except Exception as exc:
            return {"success": False, "error": f"Failed to merge spatial source: {exc}"}
    elif feature_source_path and str(feature_source_path).endswith('.csv'):
        try:
            src_df = pd.read_csv(workspace_path(feature_source_path))
            if feature_name in src_df.columns and "grid_id" in src_df.columns:
                sub = src_df[["grid_id", feature_name]].drop_duplicates(subset=["grid_id"])
                df = df.merge(sub, on="grid_id", how="left")
            else:
                return {"success": False, "error": f"Feature {feature_name} or grid_id not found in {feature_source_path}"}
        except Exception as exc:
            return {"success": False, "error": f"Failed to merge CSV source: {exc}"}
    else:
        return {"success": False, "error": f"Feature {feature_name} not found in dataset or external source."}

    # Verify column exists and has no NaNs
    if feature_name not in df.columns:
        return {"success": False, "error": f"Column {feature_name} missing after resolution."}

    df[feature_name] = df[feature_name].fillna(0.0)

    # Compute candidate theoretical features if present in base_features (e.g. Pedestrian_Permeability_Synergy)
    for feat in base_features:
        if feat not in df.columns and feat in CANDIDATE_THEORY_FEATURES:
            df[feat] = CANDIDATE_THEORY_FEATURES[feat]["compute"](df)

    # Clean dataset
    all_needed = list(set(base_features + [feature_name, TARGET_COL, FOLD_COL]))
    df = df.dropna(subset=all_needed).copy()

    y_raw = df[TARGET_COL].to_numpy(dtype=float)
    y_log = np.log1p(np.maximum(0.0, y_raw))
    folds = df[FOLD_COL].to_numpy()
    unique_folds = sorted(np.unique(folds))

    evolved_features = list(base_features) + [feature_name]

    oof_base_raw = np.zeros(len(df), dtype=float)
    oof_evolved_raw = np.zeros(len(df), dtype=float)

    fold_metrics = []

    for fold in unique_folds:
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        X_train_base = df.iloc[train_idx][base_features].values
        X_val_base = df.iloc[val_idx][base_features].values
        
        X_train_evo = df.iloc[train_idx][evolved_features].values
        X_val_evo = df.iloc[val_idx][evolved_features].values

        y_tr_log = y_log[train_idx]
        y_val_raw = y_raw[val_idx]

        # Fit Base model (Champion of prior stage)
        rf_base = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        rf_base.fit(X_train_base, y_tr_log)
        base_pred = np.clip(np.expm1(rf_base.predict(X_val_base)), 0.0, None)
        oof_base_raw[val_idx] = base_pred

        # Fit Evolved model (+ candidate data feature)
        rf_evo = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        rf_evo.fit(X_train_evo, y_tr_log)
        evo_pred = np.clip(np.expm1(rf_evo.predict(X_val_evo)), 0.0, None)
        oof_evolved_raw[val_idx] = evo_pred

        # Fold metrics
        b_r2 = float(r2_score(y_val_raw, base_pred))
        b_rmse = float(np.sqrt(mean_squared_error(y_val_raw, base_pred)))
        b_mae = float(mean_absolute_error(y_val_raw, base_pred))
        b_bias = float(np.mean(base_pred - y_val_raw))

        e_r2 = float(r2_score(y_val_raw, evo_pred))
        e_rmse = float(np.sqrt(mean_squared_error(y_val_raw, evo_pred)))
        e_mae = float(mean_absolute_error(y_val_raw, evo_pred))
        e_bias = float(np.mean(evo_pred - y_val_raw))

        fold_metrics.append({
            "fold": int(fold),
            "n_test": len(val_idx),
            "baseline": {"r2": b_r2, "rmse": b_rmse, "mae": b_mae, "bias": b_bias},
            "evolved": {"r2": e_r2, "rmse": e_rmse, "mae": e_mae, "bias": e_bias},
            "delta_r2": e_r2 - b_r2,
            "delta_rmse": e_rmse - b_rmse,
            "delta_mae": e_mae - b_mae,
            "delta_bias": e_bias - b_bias,
        })

    # Pooled metrics
    base_r2 = float(r2_score(y_raw, oof_base_raw))
    base_rmse = float(np.sqrt(mean_squared_error(y_raw, oof_base_raw)))
    base_mae = float(mean_absolute_error(y_raw, oof_base_raw))
    base_bias = float(np.mean(oof_base_raw - y_raw))

    evo_r2 = float(r2_score(y_raw, oof_evolved_raw))
    evo_rmse = float(np.sqrt(mean_squared_error(y_raw, oof_evolved_raw)))
    evo_mae = float(mean_absolute_error(y_raw, oof_evolved_raw))
    evo_bias = float(np.mean(oof_evolved_raw - y_raw))

    d_r2 = evo_r2 - base_r2
    d_rmse = evo_rmse - base_rmse
    d_mae = evo_mae - base_mae
    d_bias = evo_bias - base_bias

    # Multi-reference comparisons against Original Baseline
    orig_base_r2 = 0.5156
    orig_base_rmse = 29785.44
    orig_base_mae = 14143.19

    d_r2_vs_orig = evo_r2 - orig_base_r2
    d_rmse_vs_orig = evo_rmse - orig_base_rmse
    d_mae_vs_orig = evo_mae - orig_base_mae

    # District breakdowns
    district_results = {}
    if "district_name" in df.columns:
        for dist in ["Colombo", "Gampaha"]:
            mask = df["district_name"] == dist
            if mask.sum() > 0:
                y_d = y_raw[mask]
                b_pred_d = oof_base_raw[mask]
                e_pred_d = oof_evolved_raw[mask]

                br2_d = float(r2_score(y_d, b_pred_d))
                brmse_d = float(np.sqrt(mean_squared_error(y_d, b_pred_d)))
                bmae_d = float(mean_absolute_error(y_d, b_pred_d))
                bbias_d = float(np.mean(b_pred_d - y_d))

                er2_d = float(r2_score(y_d, e_pred_d))
                ermse_d = float(np.sqrt(mean_squared_error(y_d, e_pred_d)))
                emae_d = float(mean_absolute_error(y_d, e_pred_d))
                ebias_d = float(np.mean(e_pred_d - y_d))

                district_results[dist] = {
                    "n": int(mask.sum()),
                    "baseline": {"r2": br2_d, "rmse": brmse_d, "mae": bmae_d, "bias": bbias_d},
                    "evolved": {"r2": er2_d, "rmse": ermse_d, "mae": emae_d, "bias": ebias_d},
                    "delta_r2": er2_d - br2_d,
                    "delta_rmse": ermse_d - brmse_d,
                    "delta_mae": emae_d - bmae_d,
                    "delta_bias": ebias_d - bbias_d,
                }

    # Anomaly cluster resolution
    pct_diff_base = (y_raw - oof_base_raw) / np.maximum(1.0, oof_base_raw) * 100.0
    overpred_mask = pct_diff_base < -50.0
    underpred_mask = pct_diff_base > 50.0

    over_mae_base = float(mean_absolute_error(y_raw[overpred_mask], oof_base_raw[overpred_mask])) if overpred_mask.sum() > 0 else 0.0
    over_mae_evo = float(mean_absolute_error(y_raw[overpred_mask], oof_evolved_raw[overpred_mask])) if overpred_mask.sum() > 0 else 0.0

    under_mae_base = float(mean_absolute_error(y_raw[underpred_mask], oof_base_raw[underpred_mask])) if underpred_mask.sum() > 0 else 0.0
    under_mae_evo = float(mean_absolute_error(y_raw[underpred_mask], oof_evolved_raw[underpred_mask])) if underpred_mask.sum() > 0 else 0.0

    # Multi-metric scientific verdict
    improved_folds = sum(1 for fm in fold_metrics if fm["delta_r2"] > 0)
    colombo_improved = district_results.get("Colombo", {}).get("delta_mae", 0.0) < 0
    gampaha_improved = district_results.get("Gampaha", {}).get("delta_mae", 0.0) < 0

    if d_r2 > 0 and d_rmse < 0 and improved_folds >= 3:
        status = "ACCEPTED"
        rationale = (
            f"Candidate data predictor '{feature_name}' successfully improved spatial cross-validation performance: "
            f"ΔR² = {d_r2:+.4f} (R²: {base_r2:.4f} -> {evo_r2:.4f}), ΔRMSE = {d_rmse:+.1f}, ΔMAE = {d_mae:+.1f}. "
            f"Improved in {improved_folds}/5 spatial folds. "
            f"Vs Original Baseline: ΔR² = {d_r2_vs_orig:+.4f}, ΔRMSE = {d_rmse_vs_orig:+.1f}."
        )
    else:
        status = "REJECTED"
        rationale = (
            f"Candidate data predictor '{feature_name}' failed multi-metric acceptance criteria: "
            f"ΔR² = {d_r2:+.4f}, ΔRMSE = {d_rmse:+.1f}, ΔMAE = {d_mae:+.1f} across 5 spatial folds. "
            f"Improved in only {improved_folds}/5 spatial folds."
        )

    verdict = {
        "status": status,
        "scientific_rationale": rationale,
        "improved_folds_count": improved_folds,
        "colombo_mae_improved": colombo_improved,
        "gampaha_mae_improved": gampaha_improved,
    }

    return {
        "success": True,
        "learning_stage": "New Data-Source Learning",
        "method": f"Cumulative Random Forest ({len(evolved_features)} Predictors)",
        "feature_name": feature_name,
        "formula": f"Additional Urban Layer: {feature_name}",
        "theoretical_grounding": theoretical_grounding,
        "hypothesis": hypothesis,
        "reference_model": active_reference_id,
        "use_champion_model": use_champion_model,
        "baseline_metrics": {
            "r2": base_r2,
            "rmse": base_rmse,
            "mae": base_mae,
            "mean_bias": base_bias,
        },
        "evolved_metrics": {
            "r2": evo_r2,
            "rmse": evo_rmse,
            "mae": evo_mae,
            "mean_bias": evo_bias,
        },
        "deltas": {
            "delta_r2": d_r2,
            "delta_rmse": d_rmse,
            "delta_mae": d_mae,
            "delta_bias": d_bias,
        },
        "comparisons_vs_original_baseline": {
            "original_r2": orig_base_r2,
            "delta_r2": d_r2_vs_orig,
            "delta_rmse": d_rmse_vs_orig,
            "delta_mae": d_mae_vs_orig,
        },
        "fold_metrics": fold_metrics,
        "district_metrics": district_results,
        "anomaly_resolution": {
            "overprediction_cluster_grids": int(overpred_mask.sum()),
            "overprediction_mae_baseline": over_mae_base,
            "overprediction_mae_evolved": over_mae_evo,
            "overprediction_mae_change": over_mae_evo - over_mae_base,
            "underprediction_cluster_grids": int(underpred_mask.sum()),
            "underprediction_mae_baseline": under_mae_base,
            "underprediction_mae_evolved": under_mae_evo,
            "underprediction_mae_change": under_mae_evo - under_mae_base,
        },
        "verdict": verdict,
        "base_features": base_features,
        "evolved_features": evolved_features,
        "n_estimators": n_estimators,
        "model_configuration": rf_base.get_params(),
        "dataset_path": str(workspace_path(dataset_path)),
        "feature_source_path": str(workspace_path(feature_source_path)) if feature_source_path else None,
        "feature_input_kind": feature_input_kind,
        "target": TARGET_COL,
        "fold_column": FOLD_COL,
        "target_transform": "log1p",
        "inverse_transform": "expm1",
    }


