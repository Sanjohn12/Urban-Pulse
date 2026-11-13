"""Approved paired RF experiments with spatial isolation and immutable outputs."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KDTree

import experiment_service as models
from evaluation_service import activity_validation_metrics, HOURLY_OBS_COLS
from investigation_evidence import digest, file_digest, STATIC_FEATURES
from investigation_models import ExecutionPlan, utc_now
from project_paths import DATASET_PATH, ELIGIBLE_CSV, EXPERIMENT_RUNS_DIR, workspace_path

RUNS_ROOT = EXPERIMENT_RUNS_DIR


def implementation_hashes():
    return {name: file_digest(workspace_path(name)) for name in (
        "experiment_executor.py", "experiment_service.py", "evaluation_service.py",
        "investigation_service.py", "investigation_models.py", "registry_service.py",
    )}


def read_frame(path):
    frame = pd.read_csv(workspace_path(path), dtype={"grid_id": str})
    required = ["grid_id", "fold", "spatial_block", "centroid_x", "centroid_y",
                models.TARGET_COL, *HOURLY_OBS_COLS, *models.BASE_FEATURES]
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError(f"Execution dataset missing columns: {missing}")
    frame["grid_id"] = frame.grid_id.str.strip()
    if frame.grid_id.isna().any() or frame.grid_id.eq("").any() or frame.grid_id.duplicated().any():
        raise ValueError("Execution grid IDs must be unique and nonempty")
    if frame.spatial_block.isna().any() or frame.fold.isna().any():
        raise ValueError("Spatial blocks/folds must be present")
    frame["spatial_block"] = frame.spatial_block.astype(str)
    for name in required:
        if name not in {"grid_id", "spatial_block"}:
            frame[name] = pd.to_numeric(frame[name], errors="raise")
            if not np.isfinite(frame[name]).all():
                raise ValueError(f"Execution column is non-finite: {name}")
    if (frame[["total_acti", *HOURLY_OBS_COLS]] < 0).any().any():
        raise ValueError("Observed activity must be nonnegative")
    if not np.allclose(frame.total_acti, frame[HOURLY_OBS_COLS].sum(axis=1), rtol=1e-5, atol=1e-3):
        raise ValueError("Daily observations do not equal their hourly sums")
    if frame.groupby("spatial_block").fold.nunique().gt(1).any():
        raise ValueError("A spatial block crosses folds")
    return frame


def feature_matrix(frame, proposal, base_features):
    """Interpret previously validated numeric ASTs; no eval, code or target inputs."""
    from investigation_service import validate_feature_expression
    result = frame[list(base_features)].copy()
    allowed = set(STATIC_FEATURES) & set(frame)
    functions = {"log1p": np.log1p, "sqrt": np.sqrt, "clip": np.clip}
    operators = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
                 ast.Div: np.divide, ast.Pow: np.power}

    def value(node):
        if isinstance(node, ast.Name):
            return pd.to_numeric(frame[node.id], errors="raise").to_numpy(dtype=float)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            operand = value(node.operand)
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.BinOp):
            return operators[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.Call):
            return functions[node.func.id](*[value(arg) for arg in node.args])
        raise ValueError("Unsupported feature syntax")

    for feature in proposal.features:
        validate_feature_expression(feature, allowed)
        with np.errstate(all="raise"):
            values = np.asarray(value(ast.parse(feature.expression, mode="eval").body), dtype=float)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid derived predictor: {feature.name}")
        result[feature.name] = values
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Predictors must be finite; no implicit imputation or row removal")
    return result


def purge_training(frame, training, testing, distance):
    if len(testing) < 2 or len(training) < 2:
        raise ValueError("Each evaluation partition needs at least two training/test cells")
    train = frame.iloc[training]
    test = frame.iloc[testing]
    if set(train.spatial_block) & set(test.spatial_block):
        raise ValueError("A spatial block leaks across training/evaluation")
    nearest = KDTree(test[["centroid_x", "centroid_y"]].to_numpy()).query(
        train[["centroid_x", "centroid_y"]].to_numpy(), k=1
    )[0][:, 0]
    kept = np.asarray(training)[nearest >= distance]
    if len(kept) < 2:
        raise ValueError("Spatial buffer removes too much training data")
    return kept


def partitions(frame, plan, method):
    """Preflight every split and buffer before the first RF is fitted."""
    dev = np.flatnonzero(frame.grid_id.isin(plan.development_grid_ids).to_numpy())
    if plan.validation_mode == "protected_holdout":
        outer = [("holdout", dev, np.flatnonzero(frame.grid_id.isin(plan.holdout_grid_ids).to_numpy()))]
    else:
        folds = sorted(frame.iloc[dev].fold.unique())
        if len(folds) < 3:
            raise ValueError("Spatial CV requires at least three geographic folds")
        outer = [(str(fold), dev[frame.iloc[dev].fold.to_numpy() != fold],
                  dev[frame.iloc[dev].fold.to_numpy() == fold]) for fold in folds]
    prepared = []
    for name, train, test in outer:
        train = purge_training(frame, train, test, plan.minimum_separation_m)
        inner = []
        if method == "statistical_refinement":
            groups = frame.iloc[train].fold.to_numpy()
            if len(np.unique(groups)) < plan.calibration_inner_splits:
                raise ValueError("Insufficient training-only geographic folds for calibration")
            splitter = GroupKFold(n_splits=plan.calibration_inner_splits)
            for inner_train, inner_test in splitter.split(train, groups=groups):
                itrain, itest = train[inner_train], train[inner_test]
                itrain = purge_training(frame, itrain, itest, plan.minimum_separation_m)
                inner.append((itrain, itest))
        prepared.append((name, train, test, inner))
    return prepared


def prepare_plan(state, development_ids, *, dataset_path, environment="research",
                 validation_mode="exploratory_spatial_cv", holdout_grid_ids=(),
                 holdout_attestation=None, n_estimators=300, calibration_inner_splits=3,
                 minimum_separation_m=2500.0):
    if state.proposal is None or state.proposal.method not in {"statistical_refinement", "theory_feature"}:
        raise ValueError("Execution supports an existing statistical/theory proposal only")
    if state.proposal.data_needs:
        raise ValueError("Unresolved data needs must be satisfied in a separately reviewed investigation")
    if state.proposal.method == "statistical_refinement" and state.proposal.features:
        raise ValueError("Statistical calibration cannot silently include theory feature changes")
    if environment not in {"research", "synthetic_fixture"}:
        raise ValueError("Unknown execution environment")
    if environment == "research":
        if (state.baseline.model_id != "RF_REFERENCE_SPATIAL_OOF_V1"
                or state.planner_backend != "ollama" or state.proposal_origin != "validated_ollama"):
            raise ValueError("Research execution requires a proposal from the running Ollama investigation agent")
        if n_estimators != 300:
            raise ValueError("Research runs must retain the reference 300-tree RF specification")
        if file_digest(ELIGIBLE_CSV) != state.baseline.eligible_sha256:
            raise ValueError("Reference eligible evidence changed")
    elif state.baseline.model_id == "RF_REFERENCE_SPATIAL_OOF_V1":
        raise ValueError("A research investigation cannot be relabeled as a synthetic test")
    if type(n_estimators) is not int or not 1 <= n_estimators <= 1000:
        raise ValueError("Invalid tree budget")
    frame = read_frame(dataset_path)
    dev, held = sorted(set(development_ids)), sorted(set(holdout_grid_ids))
    if not dev or set(dev) & set(held) or not (set(dev) | set(held)) <= set(frame.grid_id):
        raise ValueError("Development and holdout cohorts must be present and disjoint")
    if set(held) & set(state.protected_grid_ids):
        raise ValueError("Previously explored protected OOF cells are not a fresh holdout")
    history_ids = set(development_ids) | set(state.protected_grid_ids)
    if environment == "research":
        historical = pd.read_csv(ELIGIBLE_CSV, usecols=["grid_id"], dtype={"grid_id": str})
        history_ids |= set(historical.grid_id)
        original = read_frame(DATASET_PATH).set_index("grid_id")
        columns = [*state.baseline.features, "total_acti", *HOURLY_OBS_COLS, "fold",
                   "centroid_x", "centroid_y"]
        supplied = frame.set_index("grid_id").loc[dev]
        if not np.array_equal(supplied[columns].to_numpy(), original.loc[dev, columns].to_numpy()):
            raise ValueError("Development data differ from the approved reference source")
        if not np.array_equal(supplied.spatial_block, original.loc[dev].spatial_block):
            raise ValueError("Development spatial blocks changed")
    if validation_mode == "protected_holdout":
        attestation = holdout_attestation or {}
        if not held or set(held) & history_ids:
            raise ValueError("Confirmation requires new holdout IDs absent from all explored OOF evidence")
        if not all(isinstance(attestation.get(key), str) and attestation[key].strip()
                   for key in ("reviewer", "provenance", "not_used_for_selection")):
            raise ValueError("Fresh holdout provenance and explicit human non-exposure attestation are required")
    elif validation_mode == "exploratory_spatial_cv":
        if held:
            raise ValueError("Exploratory CV cannot claim a final holdout")
        attestation = {}
    else:
        raise ValueError("Unknown validation mode")
    frame = frame[frame.grid_id.isin(set(dev) | set(held))].reset_index(drop=True)
    candidate = feature_matrix(frame, state.proposal, state.baseline.features)
    parameters = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=2, max_features=1.0,
        random_state=42, n_jobs=1,
    ).get_params()
    core = {
        "environment": environment, "validation_mode": validation_mode,
        "dataset_path": str(workspace_path(dataset_path)),
        "dataset_sha256": file_digest(workspace_path(dataset_path)),
        "cohort_sha256": digest(sorted(frame.grid_id.tolist())),
        "split_manifest_sha256": digest(frame.sort_values("grid_id")[
            ["grid_id", "fold", "spatial_block", "centroid_x", "centroid_y"]
        ].to_dict("records")),
        "development_grid_ids": dev, "holdout_grid_ids": held,
        "model_parameters": parameters, "calibration_inner_splits": calibration_inner_splits,
        "minimum_separation_m": minimum_separation_m, "holdout_attestation": attestation,
        "implementation_sha256": implementation_hashes(),
        "software_versions": {name: importlib.metadata.version(name)
                              for name in ("numpy", "pandas", "scikit-learn")},
        "estimated_rf_fits": 1,
    }
    provisional = ExecutionPlan(execution_id="preflight", **core)
    splits = partitions(frame, provisional, "statistical_refinement")
    core["estimated_rf_fits"] = sum(4 + 4 * len(inner) if state.proposal.method == "theory_feature"
                                    else 2 + 2 * len(inner) for _, _, _, inner in splits)
    identity = digest({"proposal_configuration": state.experiment_configuration.model_dump(), **core})
    return ExecutionPlan(execution_id="RUN-" + identity, **core)


@contextmanager
def _claim_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".executor.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


def write_json(path, value):
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def claim_run(plan, approval, binding):
    root = workspace_path(RUNS_ROOT)
    if not re.fullmatch(r"RUN-[0-9a-f]{64}", plan.execution_id):
        raise ValueError("Invalid execution artifact identity")
    with _claim_lock(root):
        claims = root / "claims"
        claims.mkdir(exist_ok=True)
        held = set(plan.holdout_grid_ids)
        for path in claims.glob("*.json"):
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous["execution_id"] == plan.execution_id:
                raise ValueError("Execution already claimed; approval cannot be replayed")
            if held & set(previous["reserved_holdout_grid_ids"]):
                raise ValueError("Holdout cells were already consumed by an earlier execution")
        run = root / "runs" / plan.execution_id
        run.mkdir(parents=True, exist_ok=False)
        claim = {
            "execution_id": plan.execution_id, "reserved_holdout_grid_ids": plan.holdout_grid_ids,
            "dataset_sha256": plan.dataset_sha256, "approval": approval.model_dump(),
            "binding_digest": binding, "claimed_at": utc_now(),
        }
        write_json(claims / (plan.execution_id + ".json"), claim)
    return run


def eligible_for_acceptance(result):
    base, candidate = result["baseline_metrics"], result["candidate_metrics"]
    bd, cd = base["daily"], candidate["daily"]
    gates = {
        "independent_spatial_validation": result["independent_spatial_validation"],
        "daily_r2_improved": bd["r2"] is not None and cd["r2"] is not None and cd["r2"] > bd["r2"],
        "daily_rmse_improved": cd["rmse"] < bd["rmse"],
        "daily_mae_improved": cd["mae"] < bd["mae"],
        "hourly_rmse_not_worse": candidate["hourly"]["rmse"] <= base["hourly"]["rmse"],
        "hourly_mae_not_worse": candidate["hourly"]["mae"] <= base["hourly"]["mae"],
        "peak_timing_not_worse": candidate["peak_hour_circular_mae"] <= base["peak_hour_circular_mae"],
    }
    return {"gates": gates, "eligible": all(gates.values()),
            "recommendation": "accepted" if all(gates.values()) else "rejected"}


def execute_plan(state, approval, binding):
    """Internal host adapter. The agent has no direct tool access to this function."""
    plan = state.execution_plan
    if (plan is None or approval.action != "experiment" or approval.decision != "approved"
            or approval.binding_digest != binding):
        raise PermissionError("A matching human-approved execution plan is required")
    if plan.implementation_sha256 != implementation_hashes():
        raise ValueError("Executor implementation changed after approval")
    if any(importlib.metadata.version(name) != version for name, version in plan.software_versions.items()):
        raise ValueError("Software versions changed after approval")
    if file_digest(workspace_path(plan.dataset_path)) != plan.dataset_sha256:
        raise ValueError("Execution input changed after approval")
    # Repeat all preflight/identity checks immediately before claiming/fitting.
    rebuilt = prepare_plan(
        state, plan.development_grid_ids, dataset_path=plan.dataset_path, environment=plan.environment,
        validation_mode=plan.validation_mode, holdout_grid_ids=plan.holdout_grid_ids,
        holdout_attestation=plan.holdout_attestation, n_estimators=plan.model_parameters["n_estimators"],
        calibration_inner_splits=plan.calibration_inner_splits,
        minimum_separation_m=plan.minimum_separation_m,
    )
    if rebuilt != plan:
        raise ValueError("Execution plan no longer reproduces its approved identity")
    frame = read_frame(plan.dataset_path)
    frame = frame[frame.grid_id.isin(set(plan.development_grid_ids) | set(plan.holdout_grid_ids))].reset_index(drop=True)
    splits = partitions(frame, plan, "statistical_refinement")
    Xbase = frame[state.baseline.features]
    Xcandidate = feature_matrix(frame, state.proposal, state.baseline.features)
    daily = frame.total_acti.to_numpy(dtype=float)
    hourly = frame[HOURLY_OBS_COLS].to_numpy(dtype=float)
    run = claim_run(plan, approval, binding)
    try:
        write_json(run / "approved_plan.json", {
            "plan": plan.model_dump(), "investigation": state.model_dump(mode="json"),
            "approval": approval.model_dump(), "binding_digest": binding,
        })
        outputs, audits, fold_metrics = [], [], []

        def fit_predict(training, testing, X, role, outer_name):
            audits.append({
                "role": role, "outer_partition": outer_name,
                "training_grid_ids": frame.iloc[training].grid_id.tolist(),
                "prediction_grid_ids": frame.iloc[testing].grid_id.tolist(),
                "training_blocks": sorted(set(frame.iloc[training].spatial_block)),
            })
            pair = models.fit_activity_pair(X.iloc[training], daily[training], hourly[training],
                                            plan.model_parameters)
            return models.predict_activity_pair(pair, X.iloc[testing])

        for name, train, test, inner in splits:
            bd, bh = fit_predict(train, test, Xbase, "baseline_outer", name)
            if state.proposal.method == "theory_feature":
                m2d, m2h = fit_predict(train, test, Xcandidate, "m2_theory_outer", name)
                b_oof_d, b_oof_h = np.full(len(frame), np.nan), np.full((len(frame), 24), np.nan)
                t_oof_d, t_oof_h = np.full(len(frame), np.nan), np.full((len(frame), 24), np.nan)
                for inner_train, inner_test in inner:
                    ibd, ibh = fit_predict(inner_train, inner_test, Xbase, "m1_inner", name)
                    itd, ith = fit_predict(inner_train, inner_test, Xcandidate, "m3_inner", name)
                    b_oof_d[inner_test], b_oof_h[inner_test] = ibd, ibh
                    t_oof_d[inner_test], t_oof_h[inner_test] = itd, ith
                if not (np.isfinite(b_oof_d[train]).all() and np.isfinite(b_oof_h[train]).all()
                        and np.isfinite(t_oof_d[train]).all() and np.isfinite(t_oof_h[train]).all()):
                    raise ValueError("Training-only M1/M3 calibration OOF is incomplete")
                m1_cal = models.fit_activity_calibration(b_oof_d[train], b_oof_h[train], daily[train], hourly[train])
                m3_cal = models.fit_activity_calibration(t_oof_d[train], t_oof_h[train], daily[train], hourly[train])
                m1d, m1h = models.calibrate_activity_pair(m1_cal, bd, bh)
                cd, ch = models.calibrate_activity_pair(m3_cal, m2d, m2h)
                audits.append({"role": "m1_m3_calibrator_fit", "outer_partition": name,
                               "training_grid_ids": frame.iloc[train].grid_id.tolist()})
            else:
                cross_daily = np.full(len(frame), np.nan)
                cross_hourly = np.full((len(frame), 24), np.nan)
                for inner_train, inner_test in inner:
                    td, th = fit_predict(inner_train, inner_test, Xbase, "calibration_inner", name)
                    cross_daily[inner_test], cross_hourly[inner_test] = td, th
                if not np.isfinite(cross_daily[train]).all() or not np.isfinite(cross_hourly[train]).all():
                    raise ValueError("Training-only calibration OOF is incomplete")
                calibrators = models.fit_activity_calibration(
                    cross_daily[train], cross_hourly[train], daily[train], hourly[train],
                )
                audits.append({"role": "calibrator_fit", "outer_partition": name,
                               "training_grid_ids": frame.iloc[train].grid_id.tolist()})
                cd, ch = models.calibrate_activity_pair(calibrators, bd, bh)
            fold_metrics.append({
                "partition": name, "n_train_after_buffer": len(train), "n_test": len(test),
                "baseline": activity_validation_metrics(daily[test], bd, hourly[test], bh),
                "candidate": activity_validation_metrics(daily[test], cd, hourly[test], ch),
                **({"m1_statistical": activity_validation_metrics(daily[test], m1d, hourly[test], m1h),
                    "m2_theory": activity_validation_metrics(daily[test], m2d, hourly[test], m2h),
                    "m3_combined": activity_validation_metrics(daily[test], cd, hourly[test], ch)}
                   if state.proposal.method == "theory_feature" else {}),
            })
            for index, position in enumerate(test):
                outputs.append({
                    "grid_id": frame.iloc[position].grid_id, "partition": name,
                    "observed_daily": daily[position], "baseline_daily": bd[index], "candidate_daily": cd[index],
                    **({"m1_daily": m1d[index], "m2_daily": m2d[index]} if state.proposal.method == "theory_feature" else {}),
                    **{f"observed_hour_{hour:02d}": hourly[position, hour] for hour in range(24)},
                    **{f"baseline_hour_{hour:02d}": bh[index, hour] for hour in range(24)},
                    **{f"candidate_hour_{hour:02d}": ch[index, hour] for hour in range(24)},
                    **({f"m1_hour_{hour:02d}": m1h[index, hour] for hour in range(24)} if state.proposal.method == "theory_feature" else {}),
                    **({f"m2_hour_{hour:02d}": m2h[index, hour] for hour in range(24)} if state.proposal.method == "theory_feature" else {}),
                })
        predictions = pd.DataFrame(outputs)
        if predictions.grid_id.duplicated().any():
            raise ValueError("A cell was evaluated more than once")
        yd = predictions.observed_daily.to_numpy()
        yh = predictions[[f"observed_hour_{hour:02d}" for hour in range(24)]].to_numpy()
        base = activity_validation_metrics(
            yd, predictions.baseline_daily.to_numpy(), yh,
            predictions[[f"baseline_hour_{hour:02d}" for hour in range(24)]].to_numpy(),
        )
        candidate = activity_validation_metrics(
            yd, predictions.candidate_daily.to_numpy(), yh,
            predictions[[f"candidate_hour_{hour:02d}" for hour in range(24)]].to_numpy(),
        )
        m1 = m2 = None
        if state.proposal.method == "theory_feature":
            m1 = activity_validation_metrics(yd, predictions.m1_daily.to_numpy(), yh,
                predictions[[f"m1_hour_{hour:02d}" for hour in range(24)]].to_numpy())
            m2 = activity_validation_metrics(yd, predictions.m2_daily.to_numpy(), yh,
                predictions[[f"m2_hour_{hour:02d}" for hour in range(24)]].to_numpy())
        independent = plan.validation_mode == "protected_holdout"
        result = {
            "schema_version": "1.0", "execution_id": plan.execution_id,
            "experiment_id": state.experiment_configuration.experiment_id,
            "environment": plan.environment, "validation_mode": plan.validation_mode,
            "independent_spatial_validation": independent,
            "baseline_metrics": base, "candidate_metrics": candidate, "fold_metrics": fold_metrics,
            "m1_statistical_metrics": m1, "m2_theory_metrics": m2,
            "m3_combined_metrics": candidate if m1 is not None else None,
            "historical_reference_metrics": state.baseline.reference_metrics,
            "historical_comparison_note": "Descriptive context only; paired comparisons use identical execution partitions",
            "daily_deltas": {key: (candidate["daily"][key] - base["daily"][key])
                             if candidate["daily"][key] is not None and base["daily"][key] is not None else None
                             for key in ("r2", "rmse", "mae")},
            "candidate_model_reference": state.experiment_configuration.candidate_model_reference,
            "models_saved": False, "deployed": False, "completed_at": utc_now(),
            "limitations": [
                "No causal inference or automatic deployment.",
                "Saved OOF-derived selection is exploratory." if not independent else
                "Independence relies additionally on the human provenance/non-exposure attestation.",
            ],
        }
        result["acceptance"] = eligible_for_acceptance(result)
        predictions.to_csv(run / "predictions.csv", index=False)
        write_json(run / "fit_audit.json", audits)
        result["artifacts"] = {
            "approved_plan": {"path": str(run / "approved_plan.json"), "sha256": file_digest(run / "approved_plan.json")},
            "predictions": {"path": str(run / "predictions.csv"), "sha256": file_digest(run / "predictions.csv")},
            "fit_audit": {"path": str(run / "fit_audit.json"), "sha256": file_digest(run / "fit_audit.json")},
        }
        write_json(run / "result.json", result)
        return result, {"path": str(run / "result.json"), "sha256": file_digest(run / "result.json")}
    except Exception as exc:
        write_json(run / "failure.json", {"execution_id": plan.execution_id, "error": f"{type(exc).__name__}: {exc}",
                                           "at": utc_now(), "claim_remains_consumed": True})
        raise
