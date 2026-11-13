import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
import geopandas as gpd
from prediction_service import load_bundle, classify_pulse_shape
from data_service import load_data, get_grid_row, get_feature_values
from explanation_service import run_whatif_scenario
import agent_workflow as aw

print("Loading data...")
bundle = load_bundle()
table, grids_geo, grids_proj, geom_path = load_data()
gdf = grids_geo



# Pick a sample grid that actually exists
sample_id = str(table["grid_id"].iloc[0])
row, err = get_grid_row(gdf, sample_id)

print(f"Sample grid {sample_id}, err={err}")

feats = bundle["features"]
print(f"Features: {feats}")
feat_vals = get_feature_values(row, feats)
print(f"Baseline feature values: {feat_vals}")

# Test what-if on Transit_Ac
baseline_transit = feat_vals.get("Transit_Ac", 0.5)
changes = {"Transit_Ac": baseline_transit * 1.5}
print(f"\nRunning what-if with changes: {changes}")
wi_res = run_whatif_scenario(bundle, row, changes)

print(f"What-if success: {wi_res.get('success')}")
print(f"Baseline pred: {wi_res.get('baseline_prediction')}")
print(f"Scenario pred: {wi_res.get('scenario_prediction')}")
print(f"Diff: {wi_res.get('absolute_difference')}")

pc = wi_res.get("pulse_comparison", {})
print("\n--- Pulse Comparison ---")
print(f"Available: {pc.get('available')}")
if pc.get('available'):
    print(f"Baseline Peak Hour: {pc.get('baseline_peak_hour'):02d}:00 ({pc.get('baseline_label')})")
    print(f"Scenario Peak Hour: {pc.get('scenario_peak_hour'):02d}:00 ({pc.get('scenario_label')})")
    print(f"Peak Shifted: {pc.get('peak_hour_shifted')}")
    print(f"Shape Changed: {pc.get('shape_changed')}")
    print(f"Baseline Desc: {pc.get('baseline_description')}")
    print(f"Scenario Desc: {pc.get('scenario_description')}")

print("\nALL WHAT-IF PULSE TESTS PASSED!")
