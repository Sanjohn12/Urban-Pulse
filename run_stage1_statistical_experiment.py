import sys
import json
from pathlib import Path

# Set output encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

import experiment_service as exps
import registry_service as regs

from project_paths import DATASET_PATH

print("=" * 70)
print("STAGE 1: STATISTICAL LEARNING EXPERIMENT (S1)")
print("Fold-specific Non-linear Isotonic Calibration on Log1p Random Forest")
print("=" * 70)
print(f"Dataset path: {DATASET_PATH}")

# Run the S1 experiment with full 300 trees to match baseline exactly
result = exps.evaluate_statistical_isotonic_calibration(
    dataset_path=str(DATASET_PATH),
    n_estimators=300,
    random_state=42
)

if not result.get("success"):
    print("Experiment failed:", result.get("error"))
    sys.exit(1)

print("\n--- 1. POOLED EVALUATION RESULTS (N = 7,764) ---")
b = result["baseline_metrics"]
e = result["evolved_metrics"]
d = result["deltas"]
print(f"Baseline Log1p RF:  R2 = {b['r2']:.4f}, RMSE = {b['rmse']:,.2f}, MAE = {b['mae']:,.2f}, Mean Bias = {b['mean_bias']:+,.2f}")
print(f"Isotonic Calibrated: R2 = {e['r2']:.4f}, RMSE = {e['rmse']:,.2f}, MAE = {e['mae']:,.2f}, Mean Bias = {e['mean_bias']:+,.2f}")
print(f"Empirical Deltas:   ΔR2 = {d['delta_r2']:+.4f}, ΔRMSE = {d['delta_rmse']:+,.2f}, ΔMAE = {d['delta_mae']:+,.2f}, ΔBias = {d['delta_bias']:+,.2f}")

print("\n--- 2. FOLD-LEVEL METRICS ---")
for fm in result["fold_metrics"]:
    f_num = fm["fold"]
    b_f = fm["baseline"]
    c_f = fm["calibrated"]
    print(f"Fold {f_num} (n={fm['n_test']}):")
    print(f"  Baseline:   R2 = {b_f['r2']:.4f}, RMSE = {b_f['rmse']:,.2f}, MAE = {b_f['mae']:,.2f}, Bias = {b_f['bias']:+,.2f}")
    print(f"  Calibrated: R2 = {c_f['r2']:.4f}, RMSE = {c_f['rmse']:,.2f}, MAE = {c_f['mae']:,.2f}, Bias = {c_f['bias']:+,.2f}")
    print(f"  Delta:      ΔR2 = {fm['delta_r2']:+.4f}, ΔRMSE = {fm['delta_rmse']:+,.2f}, ΔMAE = {fm['delta_mae']:+,.2f}")

print("\n--- 3. ACTIVITY-LEVEL QUARTILE METRICS ---")
for q_name, q_data in result["quartile_metrics"].items():
    b_q = q_data["baseline"]
    c_q = q_data["calibrated"]
    print(f"{q_name} (n={q_data['n']}, Mean Actual = {q_data['actual_mean']:,.1f}):")
    print(f"  Baseline:   Bias = {b_q['bias']:+,.1f}, MAE = {b_q['mae']:,.1f}, R2 = {b_q['r2']:.4f}")
    print(f"  Calibrated: Bias = {c_q['bias']:+,.1f}, MAE = {c_q['mae']:,.1f}, R2 = {c_q['r2']:.4f}")
    print(f"  Change:     ΔBias = {q_data['bias_change']:+,.1f}, ΔMAE = {q_data['mae_change']:+,.1f}")

print("\n--- 4. DISTRICT METRICS ---")
for d_name, d_data in result["district_metrics"].items():
    b_d = d_data["baseline"]
    c_d = d_data["calibrated"]
    print(f"District: {d_name} (n={d_data['n']}):")
    print(f"  Baseline:   R2 = {b_d['r2']:.4f}, RMSE = {b_d['rmse']:,.2f}, MAE = {b_d['mae']:,.2f}, Bias = {b_d['bias']:+,.2f}")
    print(f"  Calibrated: R2 = {c_d['r2']:.4f}, RMSE = {c_d['rmse']:,.2f}, MAE = {c_d['mae']:,.2f}, Bias = {c_d['bias']:+,.2f}")
    print(f"  Delta:      ΔR2 = {d_data['delta_r2']:+.4f}, ΔMAE = {d_data['delta_mae']:+,.2f}")

print("\n--- 5. SCIENTIFIC VERDICT & RATIONALE ---")
v = result["verdict"]
print(f"Verdict:   {v['status']}")
print(f"Rationale: {v['scientific_rationale']}")

# Record into persistent scientific memory
print("\n--- 6. LOGGING TO SCIENTIFIC MEMORY ---")
entry = regs.record_experiment_result(result)
print(f"Logged as Experiment ID: {entry['id']}")
print("Scientific memory updated successfully!")

