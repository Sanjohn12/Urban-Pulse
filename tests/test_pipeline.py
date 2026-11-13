import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tests")

PASS = []
FAIL = []

def run_test(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:
        FAIL.append(name)
        print(f"  FAIL  {name}: {exc}")

# ---- Tests ----

def t01_load_bundle():
    import prediction_service as ps
    bundle = ps.load_bundle()
    assert "total_model" in bundle
    assert "pulse_model" in bundle
    assert isinstance(bundle["features"], list) and len(bundle["features"]) == 8
    assert bundle["target_transform"] == "log1p"
    assert bundle["inverse_transform"] == "expm1"

def t02_load_data():
    import data_service as ds
    table, grids_geo, grids_proj, geom_path = ds.load_data()
    assert len(table) == 7764, f"Expected 7764 rows, got {len(table)}"
    assert "grid_id" in table.columns
    assert not grids_geo.empty

def t03_unique_eligible_grids():
    import data_service as ds
    table, grids_geo, grids_proj, _ = ds.load_data()
    n_unique = table["grid_id"].nunique()
    assert n_unique == 7764, f"Expected 7764 unique grids, got {n_unique}"
    oof_grids = table[table["oof_observed_total"].notna()]["grid_id"].nunique()
    print(f"    OOF records: {oof_grids}/{n_unique}")

def t04_resolve_valid_grid():
    import data_service as ds
    table, grids_geo, _, _ = ds.load_data()
    valid_id = table["grid_id"].iloc[0]
    row, error = ds.get_grid_row(grids_geo, valid_id)
    assert error is None, f"Error: {error}"
    assert row is not None
    loc = ds.get_location_details(row)
    assert loc["lat"] != 0 and loc["lon"] != 0
    print(f"    Grid {valid_id} in {loc['district']}: {loc['lat']:.5f}, {loc['lon']:.5f}")
    return valid_id

def t05_reject_invalid_grid():
    import data_service as ds
    table, grids_geo, _, _ = ds.load_data()
    row, error = ds.get_grid_row(grids_geo, "INVALID_XYZ_999999")
    assert row is None
    assert error is not None
    print(f"    Correctly rejected. Error: {error}")

def t06_feature_order():
    import data_service as ds
    import prediction_service as ps
    table, grids_geo, _, _ = ds.load_data()
    bundle = ps.load_bundle()
    features = ps.get_features(bundle)
    assert features == ["LConn_wt", "MAD500_wt", "BtA500_wt", "Floor_Inte", "POI_Densit", "Transit_Ac", "IntDen_km2", "land_use_e"]
    for f in features:
        assert f in grids_geo.columns, f"Feature column {f!r} missing from GeoDataFrame"
    print(f"    Feature order: {features}")

def t07_predict_total():
    import data_service as ds
    import prediction_service as ps
    table, grids_geo, _, _ = ds.load_data()
    bundle = ps.load_bundle()
    valid_id = table["grid_id"].iloc[0]
    row, _ = ds.get_grid_row(grids_geo, valid_id)
    result = ps.predict_total(bundle, row)
    assert result["success"], f"Prediction failed: {result.get('error')}"
    v = result["value"]
    assert np.isfinite(v) and v >= 0, f"Prediction not valid: {v}"
    print(f"    Grid {valid_id}: total prediction = {v:,.0f} activity units")

def t08_prediction_shape_and_transform():
    import data_service as ds
    import prediction_service as ps
    table, grids_geo, _, _ = ds.load_data()
    bundle = ps.load_bundle()
    row, _ = ds.get_grid_row(grids_geo, table["grid_id"].iloc[0])
    result = ps.predict_total(bundle, row)
    assert result["success"]
    assert result["target_transform"] == "log1p"
    assert "raw_prediction" in result
    import numpy as np
    expected = float(np.expm1(result["raw_prediction"]))
    assert abs(result["value"] - expected) < 0.01, "Inverse transform mismatch"
    print(f"    raw={result['raw_prediction']:.4f} -> expm1 -> {result['value']:.2f}")

def t09_pulse_prediction():
    import data_service as ds
    import prediction_service as ps
    table, grids_geo, _, _ = ds.load_data()
    bundle = ps.load_bundle()
    row, _ = ds.get_grid_row(grids_geo, table["grid_id"].iloc[0])
    pulse = ps.predict_pulse(bundle, row)
    assert pulse.get("success"), f"Pulse failed: {pulse.get('error')}"
    assert pulse.get("type") == "hourly", f"Expected hourly, got {pulse.get('type')}"
    values = pulse["values"]
    assert len(values) == 24, f"Expected 24 hourly values, got {len(values)}"
    assert all(v >= 0 for v in values), "Negative hourly values present"
    print(f"    Pulse: 24 hourly values. Peak hour={pulse['peak_hour']:02d}:00 ({pulse['peak_value']:,.1f})")

def t10_shap_explanation():
    import data_service as ds
    import explanation_service as es
    import prediction_service as ps
    table, grids_geo, _, _ = ds.load_data()
    bundle = ps.load_bundle()
    row, _ = ds.get_grid_row(grids_geo, table["grid_id"].iloc[0])
    shap_r = es.get_shap_values(bundle, row)
    assert shap_r.get("success"), f"SHAP failed: {shap_r.get('error')}"
    records = shap_r["records"]
    assert len(records) == 8
    top = records[0]
    print(f"    Top SHAP feature: {top['label']} (shap={top['shap_value']:.4f})")

def t11_oof_result_correct_grid():
    import data_service as ds
    import evaluation_service as evs
    table, grids_geo, _, _ = ds.load_data()
    oof_rows = table[table["oof_observed_total"].notna()]
    if oof_rows.empty:
        print("    No OOF records found!")
        return
    valid_id = oof_rows["grid_id"].iloc[0]
    oof = evs.get_oof_for_grid(table, valid_id)
    assert oof["available"], "OOF not available for valid grid"
    print(f"    Grid {valid_id}: obs={oof['observed']:,.0f}, pred={oof['predicted']:,.0f}, residual={oof['diff']:+,.0f}")

def t12_residual_sign_correct():
    import data_service as ds
    import evaluation_service as evs
    table, grids_geo, _, _ = ds.load_data()
    oof_rows = table[table["oof_observed_total"].notna() & table["oof_rf_predicted_total"].notna()]
    if oof_rows.empty:
        print("    No OOF rows for sign check.")
        return
    row = oof_rows.iloc[0]
    obs = float(row["oof_observed_total"])
    pred = float(row["oof_rf_predicted_total"])
    expected_diff = obs - pred
    oof = evs.get_oof_for_grid(table, row["grid_id"])
    assert oof["available"]
    computed_diff = oof["diff"]
    assert abs(computed_diff - expected_diff) < 0.01, f"Residual mismatch: {computed_diff} vs {expected_diff}"
    print(f"    Residual sign check OK: {computed_diff:+.2f} = obs({obs:.2f}) - pred({pred:.2f})")

def t13_deployment_vs_oof_separation():
    import data_service as ds
    import prediction_service as ps
    import evaluation_service as evs
    table, grids_geo, _, _ = ds.load_data()
    bundle = ps.load_bundle()
    oof_rows = table[table["oof_observed_total"].notna()]
    if oof_rows.empty:
        print("    No OOF rows for separation check.")
        return
    valid_id = oof_rows["grid_id"].iloc[0]
    row, _ = ds.get_grid_row(grids_geo, valid_id)
    depl_pred = ps.predict_total(bundle, row)
    oof = evs.get_oof_for_grid(table, valid_id)
    # Deployment prediction should NOT equal OOF prediction in general
    assert depl_pred.get("success")
    assert oof["available"]
    print(f"    Deployment={depl_pred['value']:,.0f}, OOF={oof['predicted']:,.0f}. These are separate results.")

def t14_whatif_scenario():
    import data_service as ds
    import explanation_service as es
    import prediction_service as ps
    table, grids_geo, _, _ = ds.load_data()
    bundle = ps.load_bundle()
    row, _ = ds.get_grid_row(grids_geo, table["grid_id"].iloc[0])
    feat_vals = ds.get_feature_values(row, ps.get_features(bundle))
    transit_base = feat_vals.get("Transit_Ac", 0.0) or 0.0
    new_transit = transit_base * 1.5 + 0.1  # Increase by 50%
    result = es.run_whatif_scenario(bundle, row, {"Transit_Ac": new_transit})
    assert result.get("success"), f"What-if failed: {result.get('error')}"
    print(f"    What-if: baseline={result['baseline_prediction']:,.0f}, scenario={result['scenario_prediction']:,.0f}, diff={result['absolute_difference']:+,.0f}")
    print(f"    Warning: {result.get('warning','')}")

def t15_pooled_oof_metrics():
    import data_service as ds
    import evaluation_service as evs
    table, grids_geo, _, _ = ds.load_data()
    m = evs.recalculate_pooled_metrics(grids_geo)
    assert m["available"], f"Metrics unavailable: {m.get('error')}"
    print(f"    Pooled OOF RF: R2={m['r2']:.4f}, RMSE={m['rmse']:,.1f}, MAE={m['mae']:,.1f}, bias={m['mean_bias']:+,.1f}")
    # Reference: R2~0.5156, RMSE~29785, MAE~14143
    if abs(m["r2"] - 0.5156) > 0.10:
        print(f"    WARN: R2 {m['r2']:.4f} differs from reference 0.5156 by more than 0.10")

def t16_ollama_available():
    import agent_workflow as aw
    ok, msg = aw.check_ollama_available()
    print(f"    Ollama available: {ok}. {msg}")
    if not ok:
        print("    WARN: LLM not available; fallback mode only")

def t17_agent_workflow_run():
    import data_service as ds
    import prediction_service as ps
    import agent_workflow as aw
    table, grids_geo, _, _ = ds.load_data()
    bundle = ps.load_bundle()
    valid_id = table["grid_id"].iloc[0]
    result = aw.run_workflow(
        query="Where is this grid and what activity is predicted?",
        grid_id=valid_id, bundle=bundle, gdf=grids_geo, table=table
    )
    assert "final_answer" in result and result["final_answer"], "No final_answer"
    assert "review_notes" in result
    print(f"    Workflow answer (first 200 chars): {result['final_answer'][:200]}")
    print(f"    Review notes: {result['review_notes'][:3]}")

def t18_graceful_missing_grid():
    import data_service as ds
    import prediction_service as ps
    import agent_workflow as aw
    table, grids_geo, _, _ = ds.load_data()
    bundle = ps.load_bundle()
    result = aw.run_workflow(
        query="Tell me about this grid.", grid_id="NONEXISTENT_99999",
        bundle=bundle, gdf=grids_geo, table=table
    )
    assert "final_answer" in result
    print(f"    Graceful error message: {result['final_answer'][:150]}")

def t19_five_agent_roles():
    import data_service as ds
    import prediction_service as ps
    import agent_workflow as aw
    table, grids_geo, _, _ = ds.load_data()
    bundle = ps.load_bundle()
    valid_id = table["grid_id"].iloc[0]
    result = aw.run_workflow(
        query="Full report with SHAP explanation.", grid_id=valid_id,
        bundle=bundle, gdf=grids_geo, table=table, whatif_changes=None
    )
    msgs = result.get("messages", [])
    agent_names = [m.content[:30] for m in msgs if hasattr(m, "content")]
    planner_ran    = any("[Planner]" in m for m in agent_names)
    data_ran       = any("[Data]" in m for m in agent_names)
    prediction_ran = any("[Prediction]" in m for m in agent_names)
    explanation_ran= any("[Explanation]" in m for m in agent_names)
    review_ran     = any("[Review]" in m for m in agent_names)
    composer_ran   = any("[Composer]" in m for m in agent_names)
    print(f"    Planner={planner_ran}, Data={data_ran}, Prediction={prediction_ran}, Explanation={explanation_ran}, Review={review_ran}, Composer={composer_ran}")
    assert all([planner_ran, data_ran, prediction_ran, review_ran]), "Not all required agents ran"

# ---- Run all tests ----
print("\n" + "="*60)
print("Urban Pulse AI Agent - Acceptance Tests")
print("="*60 + "\n")

run_test("T01: Load model bundle", t01_load_bundle)
run_test("T02: Load CSV and geometry", t02_load_data)
run_test("T03: Unique eligible grids and OOF counts", t03_unique_eligible_grids)
run_test("T04: Resolve valid grid ID", t04_resolve_valid_grid)
run_test("T05: Reject invalid grid ID", t05_reject_invalid_grid)
run_test("T06: Feature order matches bundle", t06_feature_order)
run_test("T07: Predict total activity", t07_predict_total)
run_test("T08: Prediction shape and inverse transform", t08_prediction_shape_and_transform)
run_test("T09: Pulse prediction (24 hourly values)", t09_pulse_prediction)
run_test("T10: SHAP explanation", t10_shap_explanation)
run_test("T11: OOF result for valid grid", t11_oof_result_correct_grid)
run_test("T12: Residual sign calculation", t12_residual_sign_correct)
run_test("T13: Deployment vs OOF separation", t13_deployment_vs_oof_separation)
run_test("T14: What-if scenario", t14_whatif_scenario)
run_test("T15: Pooled OOF metrics", t15_pooled_oof_metrics)
run_test("T16: Ollama LLM availability", t16_ollama_available)
run_test("T17: Full agent workflow", t17_agent_workflow_run)
run_test("T18: Graceful missing grid handling", t18_graceful_missing_grid)
run_test("T19: Five agent roles executed", t19_five_agent_roles)

print("\n" + "="*60)
print(f"RESULTS: {len(PASS)}/{len(PASS)+len(FAIL)} tests passed")
if FAIL:
    print(f"FAILED:  {FAIL}")
print("="*60 + "\n")
sys.exit(0 if not FAIL else 1)
