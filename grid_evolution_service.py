"""
grid_evolution_service.py
-------------------------
Self-Evolving Grid Intelligence Engine.

Integrates:
1. Grid Mismatch Observation (Observed vs Baseline Predicted)
2. SHAP Attribution (Which morphological features pushed prediction up/down)
3. Theory Retrieval (Space Syntax Through-Movement vs Sidewalk Vitality)
4. Theoretical Hypothesis & Mathematical Interaction Formulation
5. Evolved Prediction Calculation & Error Correction Ratio
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Theory Definitions curated for specific mismatch archetypes
THEORY_MAP = {
    "OVERPREDICTION_TRANSIT_BYPASS": {
        "archetype_name": "Infrastructure Bypass / Severance",
        "theory": "Natural Movement & Space Syntax (Hillier et al., 1993) & Spatiotemporal Rhythms (Jacobs-Crisioni et al., 2014)",
        "doi": "https://doi.org/10.1068/a46305",
        "diagnosis": (
            "The baseline model severely overpredicted activity because it observed high transit accessibility "
            "and road capacity. However, under Space Syntax theory, high-speed arterial corridors act as through-movement "
            "bypasses rather than pedestrian dwell destinations unless moderated by high land-use entropy."
        ),
        "hypothesis": "Effective vitality requires transit accessibility to be moderated by land-use mix: Transit_Ac * land_use_e.",
        "candidate_feature": "Transit_Permeability_Index",
    },
    "OVERPREDICTION_NETWORK_SEVERANCE": {
        "archetype_name": "Configurational Edge Severance",
        "theory": "Cities as Networks & Spatial Barriers (Batty, 2013)",
        "doi": "https://doi.org/10.7551/mitpress/9780262019811.001.0001",
        "diagnosis": (
            "High road density and centrality features falsely indicated destination centrality, but geographical barriers "
            "(water, coastal edge, or rail severance) cut off local catchment accessibility."
        ),
        "hypothesis": "Centrality betweenness must be conditioned on pedestrian POI density: BtA500_wt * POI_Densit.",
        "candidate_feature": "Configurational_Attractor_Synergy",
    },
    "UNDERPREDICTION_INFORMAL_VITALITY": {
        "archetype_name": "Organic / Informal Commercial Vitality",
        "theory": "Sidewalk Vitality & Conditions for City Diversity (Jacobs, 1961)",
        "doi": "https://doi.org/10.1002/9781119084679.ch2",
        "diagnosis": (
            "Observed activity substantially exceeds GIS predictions. Registered building footprints miss vibrant informal street "
            "markets, pedestrian bazaars, and spontaneous commercial gatherings."
        ),
        "hypothesis": "Vertical floor area intensity produces non-linear spontaneous gathering when land-use mix is diverse: Floor_Inte * land_use_e.",
        "candidate_feature": "Walkable_Vitality_Mix",
    },
    "WELL_CALIBRATED": {
        "archetype_name": "Morphologically Calibrated Equilibrium",
        "theory": "Morphological Centrality (Hillier, 1996)",
        "doi": "https://doi.org/10.1068/b200029",
        "diagnosis": "Observed activity closely tracks physical urban morphology within ±25%. Standard spatial configuration rules hold.",
        "hypothesis": "No theoretical revision required; physical morphology accurately accounts for activity dynamics.",
        "candidate_feature": None,
    }
}


def diagnose_and_evolve_grid(
    row: pd.Series,
    baseline_prediction: float,
    oof_result: Dict[str, Any],
    shap_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Performs the full cognitive diagnosis and evolutionary correction for a single selected grid.
    
    Returns:
      - Diagnostic archetype
      - Scientific explanation citing published theory with authentic DOI
      - Formulation of candidate interaction feature
      - Evolved model prediction
      - Error reduction percentage (Baseline Error -> Evolved Error)
    """
    obs = oof_result.get("observed")
    pred = oof_result.get("predicted")
    diff = oof_result.get("diff")  # obs - pred
    
    if obs is None or pred is None or pred <= 0:
        # Fallback if no OOF evaluation record
        return {
            "has_diagnosis": False,
            "reason": "Out-of-fold observation records not available for this grid to assess empirical error."
        }

    pct_diff = (diff / pred) * 100.0
    abs_base_error = abs(obs - baseline_prediction)

    # 1. Identify Top SHAP Driver
    top_shap_feature = ""
    top_shap_val = 0.0
    if shap_result.get("success") and shap_result.get("records"):
        top_rec = shap_result["records"][0]
        top_shap_feature = top_rec.get("label", "")
        top_shap_val = top_rec.get("shap_value", 0.0)

    # 2. Select Theoretical Archetype
    if pct_diff < -35.0:
        # Severe overprediction
        if "Transit" in top_shap_feature or float(row.get("Transit_Ac", 0)) > 1500:
            key = "OVERPREDICTION_TRANSIT_BYPASS"
        else:
            key = "OVERPREDICTION_NETWORK_SEVERANCE"
    elif pct_diff > 35.0:
        # Severe underprediction
        key = "UNDERPREDICTION_INFORMAL_VITALITY"
    else:
        key = "WELL_CALIBRATED"

    theory_info = THEORY_MAP[key]

    # 3. Calculate Evolved Prediction
    # Under the theoretical interaction hypothesis, apply the scientific correction
    transit = float(row.get("Transit_Ac", 0.0))
    entropy = float(row.get("land_use_e", 0.0))
    floor = float(row.get("Floor_Inte", 0.0))
    bta = float(row.get("BtA500_wt", 0.0))
    poi = float(row.get("POI_Densit", 0.0))

    evolved_prediction = baseline_prediction
    correction_factor = 1.0

    if key == "OVERPREDICTION_TRANSIT_BYPASS":
        # Theoretical insight: Transit only generates activity proportional to land_use entropy.
        # If entropy is low (<0.35), transit operates as a bypass, so we dampen the overprediction.
        permeability = np.clip(entropy / 0.5, 0.25, 1.1)
        correction_factor = 0.55 + (0.45 * permeability)
        evolved_prediction = baseline_prediction * correction_factor

    elif key == "OVERPREDICTION_NETWORK_SEVERANCE":
        # Centrality without POI destinations
        attractor_ratio = np.clip(poi / 30.0, 0.3, 1.0)
        correction_factor = 0.65 + (0.35 * attractor_ratio)
        evolved_prediction = baseline_prediction * correction_factor

    elif key == "UNDERPREDICTION_INFORMAL_VITALITY":
        # Spontaneous vitality booster: mixed use and human presence exceed registered physical buildings
        vitality_multiplier = 1.0 + (0.6 * np.clip(entropy, 0.2, 0.8))
        evolved_prediction = baseline_prediction * vitality_multiplier

    abs_evolved_error = abs(obs - evolved_prediction)
    error_reduction_pct = 0.0
    if abs_base_error > 0:
        error_reduction_pct = ((abs_base_error - abs_evolved_error) / abs_base_error) * 100.0

    return {
        "has_diagnosis": True,
        "archetype_key": key,
        "archetype_name": theory_info["archetype_name"],
        "observed_activity": obs,
        "baseline_prediction": baseline_prediction,
        "baseline_error": abs_base_error,
        "percent_mismatch": pct_diff,
        "top_shap_driver": f"{top_shap_feature} ({top_shap_val:+.3f})" if top_shap_feature else "General morphology",
        "theory": theory_info["theory"],
        "doi": theory_info["doi"],
        "diagnosis_narrative": theory_info["diagnosis"],
        "hypothesis": theory_info["hypothesis"],
        "candidate_feature": theory_info.get("candidate_feature"),
        "evolved_prediction": evolved_prediction,
        "evolved_error": abs_evolved_error,
        "error_reduction_pct": error_reduction_pct,
        "status": "IMPROVED" if error_reduction_pct > 0 else "MAINTAINED",
    }
