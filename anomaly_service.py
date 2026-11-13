"""
anomaly_service.py
------------------
Step 1 of Theory-Guided Self-Evolving Urban Intelligence (TSE-UI).

Diagnoses systemic empirical failure archetypes across spatial cross-validation
out-of-fold (OOF) predictions. Transforms raw errors into structured scientific
contradiction profiles ready for theory exploration.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def profile_spatial_anomalies(table: pd.DataFrame) -> Dict[str, Any]:
    """
    Profiles systemic prediction discrepancies across the 7,764 grids.
    Discovers:
      1. Overprediction Cluster (The 'Bypass / Barrier' Archetype):
         High transit & density, but actual observed activity is severely lower.
      2. Underprediction Cluster (The 'Organic / Informal Vitality' Archetype):
         Moderate/low infrastructure metrics, but actual activity is far higher.
    """
    obs_col = "oof_observed_total"
    pred_col = "oof_rf_predicted_total"
    diff_col = "oof_observed_minus_predicted"

    if obs_col not in table.columns or pred_col not in table.columns:
        return {"success": False, "error": "OOF columns missing in table."}

    df = table.dropna(subset=[obs_col, pred_col]).copy()
    df["obs"] = pd.to_numeric(df[obs_col], errors="coerce")
    df["pred"] = pd.to_numeric(df[pred_col], errors="coerce")
    df["diff"] = df["obs"] - df["pred"]
    df["pct_diff"] = (df["diff"] / df["pred"].replace(0, np.nan)) * 100.0

    # Overprediction: Model predicted way too high (obs << pred)
    over_df = df[df["pct_diff"] < -50.0].sort_values("diff", ascending=True)
    # Underprediction: Model predicted way too low (obs >> pred)
    under_df = df[df["pct_diff"] > 50.0].sort_values("diff", ascending=False)

    def _summarize_cluster(sub_df: pd.DataFrame, archetype_name: str, hypothesis_prompt: str) -> Dict[str, Any]:
        if sub_df.empty:
            return {"count": 0}
        
        feature_cols = [c for c in ["Transit_Ac", "Floor_Inte", "IntDen_km2", "land_use_e", "LConn_wt", "MAD500_wt", "BtA500_wt", "POI_Densit"] if c in sub_df.columns]
        means = sub_df[feature_cols].mean().to_dict() if feature_cols else {}
        global_means = df[feature_cols].mean().to_dict() if feature_cols else {}

        # Top representative grids
        top_examples = []
        for _, r in sub_df.head(5).iterrows():
            top_examples.append({
                "grid_id": str(r.get("grid_id")),
                "district": str(r.get("district_name", "Unknown")),
                "observed": float(r["obs"]),
                "predicted": float(r["pred"]),
                "residual": float(r["diff"]),
                "pct_mismatch": float(round(r["pct_diff"], 1)),
            })

        return {
            "archetype": archetype_name,
            "grid_count": len(sub_df),
            "pct_of_total_grids": round((len(sub_df) / len(df)) * 100, 1),
            "mean_observed": float(round(sub_df["obs"].mean(), 1)),
            "mean_predicted": float(round(sub_df["pred"].mean(), 1)),
            "mean_pct_deviation": float(round(sub_df["pct_diff"].mean(), 1)),
            "cluster_feature_means": {k: round(v, 2) for k, v in means.items()},
            "global_feature_means": {k: round(v, 2) for k, v in global_means.items()},
            "scientific_question": hypothesis_prompt,
            "top_exemplars": top_examples,
        }

    archetype_over = _summarize_cluster(
        over_df,
        archetype_name="Severe Overprediction (mechanism unverified)",
        hypothesis_prompt=(
            "Why does high physical accessibility and building density fail to generate expected human presence? "
            "Examine theories regarding transit corridors as high-speed bypasses (Space Syntax through-movement), "
            "dead-ends, spatial barriers (lagoon/water boundaries), or lack of destination permeability."
        )
    )

    archetype_under = _summarize_cluster(
        under_df,
        archetype_name="Severe Underprediction (mechanism unverified)",
        hypothesis_prompt=(
            "Why does observed human activity far exceed what physical road width and registered building GIS features predict? "
            "Examine theories regarding informal street trade (Jane Jacobs' sidewalk vitality), unmapped commercial bazaars, "
            "and pedestrian clustering in developing metropolitan contexts."
        )
    )

    return {
        "success": True,
        "total_evaluated_grids": len(df),
        "overpredicted_count": len(over_df),
        "underpredicted_count": len(under_df),
        "archetype_overprediction": archetype_over,
        "archetype_underprediction": archetype_under,
    }

if __name__ == "__main__":
    import data_service as ds
    t, _, _, _ = ds.load_data()
    summary = profile_spatial_anomalies(t)
    print("Total evaluated:", summary["total_evaluated_grids"])
    print("\n--- Archetype Overprediction ---")
    print("Grids:", summary["archetype_overprediction"]["grid_count"])
    print("Question:", summary["archetype_overprediction"]["scientific_question"])
    print("Sample:", summary["archetype_overprediction"]["top_exemplars"][:2])
    print("\n--- Archetype Underprediction ---")
    print("Grids:", summary["archetype_underprediction"]["grid_count"])
    print("Question:", summary["archetype_underprediction"]["scientific_question"])
    print("Sample:", summary["archetype_underprediction"]["top_exemplars"][:2])
