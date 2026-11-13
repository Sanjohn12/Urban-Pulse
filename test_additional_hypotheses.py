import sys
import json
sys.stdout.reconfigure(encoding='utf-8')
from experiment_service import evaluate_theoretical_hypothesis

from project_paths import DATASET_PATH
DATASET = str(DATASET_PATH)

features_to_test = ["Configurational_Attractor_Synergy", "Walkable_Vitality_Mix"]

for feat in features_to_test:
    print(f"\n--- Testing Candidate Feature: {feat} ---")
    res = evaluate_theoretical_hypothesis(
        dataset_path=DATASET,
        candidate_feature_name=feat,
        n_estimators=30,
        random_state=42
    )
    print("Verdict:", res.get("verdict", {}).get("status"))
    print("Rationale:", res.get("verdict", {}).get("scientific_rationale"))
    print("Delta R2:", f"{res['deltas']['delta_r2']:+.4f}")
    print("Delta RMSE:", f"{res['deltas']['delta_rmse']:+.1f}")
