import pandas as pd
import json

df = pd.read_csv('AI_Agent_Results/agent_eligible_grids_with_oof.csv')
df['grid_str'] = df['grid_id'].astype(str).str.replace(r'\.0$', '', regex=True)

hours = [f'activity_{h}' if h < 10 else f'activity{h}' for h in range(24)]
features = ['LConn_wt', 'MAD500_wt', 'BtA500_wt', 'Floor_Inte', 'POI_Densit', 'Transit_Ac', 'IntDen_km2', 'land_use_e']

target_grids = ['10000', '8970', '1000', '1355']
res = {}
for gid in target_grids:
    sub = df[df['grid_str'] == gid]
    if not sub.empty:
        row = sub.iloc[0]
        res[gid] = {
            'grid_id': gid,
            'district': str(row['district_name']),
            'total_observed': float(row['total_acti']),
            'total_rf_oof': float(row['oof_rf_predicted_total']),
            'hourly_activity': [float(row[h]) for h in hours],
            'features': {f: float(row[f]) for f in features}
        }

with open('Chapter4_Figures/Pattern_Investigation_Audit/target_grids_activity.json', 'w') as f:
    json.dump(res, f, indent=2)

print("Saved target_grids_activity.json successfully!")
for gid in target_grids:
    print(f"Grid {gid}: Total={res[gid]['total_observed']}, District={res[gid]['district']}")
