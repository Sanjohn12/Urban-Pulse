import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding='utf-8')

from prediction_service import load_bundle
from data_service import load_data
import agent_workflow as aw

print("Loading data bundle...")
bundle = load_bundle()
table, grids_geo, grids_proj, geom_path = load_data()
grid_id = "2690"
print(f"Testing Grid {grid_id}\n" + "="*50)

questions = [
    ("Q1", "Where is this grid, and what activity should we expect?", None),
    ("Q2", "Why does the model predict this level of activity?", None),
    ("Q3", "Describe this grid's hourly activity pattern.", None),
    ("Q4", "What happens if transit accessibility doubles?", None),
    ("Q5", "Compare the baseline and scenario hourly curves.", {"Transit_Ac": 0.40}),
]

for label, q, wi in questions:
    print(f"\n[{label}] Question: {q}")
    res = aw.run_workflow(q, grid_id, bundle, grids_geo, table, whatif_changes=wi)
    print(f"Detected Intent: {res.get('intent')}")
    print("--- Response ---")
    print(res.get("final_answer"))
    print("="*50)

print("\nALL 5 QUESTIONS TESTED.")
