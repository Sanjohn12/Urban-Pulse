from __future__ import annotations
import json, logging, re, operator
import numpy as np
import pandas as pd
from typing import Any, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

class AgentState(TypedDict):
    query: str
    grid_id: Any
    messages: Annotated[list, operator.add]
    plan: str
    data_result: dict
    prediction_result: dict
    pulse_result: dict
    shap_result: dict
    oof_result: dict
    whatif_result: dict
    review_notes: list
    final_answer: str
    error: Any
    intent: str

def _get_llm():
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_predict=3000)

def check_ollama_available():
    try:
        import ollama
        models = ollama.Client(host=OLLAMA_BASE_URL, timeout=5).list()
        names = [m.model for m in models.models]
        ok = OLLAMA_MODEL in names
        return ok, f"Model {OLLAMA_MODEL!r} {'found' if ok else 'NOT found'}. Available: {names}"
    except Exception as exc:
        return False, f"Ollama unavailable: {exc}"

PLANNER_SYSTEM = (
    "You are the Planner Agent for an Urban Pulse research system. "
    "Your ONLY job is to identify the user intent and output a JSON object with keys: "
    "intent (one of: location, morphology, prediction, pulse, explanation, oof, whatif, mismatch, combined), "
    "needs_prediction (bool), needs_explanation (bool), needs_oof (bool), needs_pulse (bool), needs_whatif (bool), "
    "clarification_needed (bool), clarification_question (str or null). "
    "Guidelines for intent classification: "
    "- 'Where is this grid, and what activity should we expect?': intent='location' (or 'combined'). "
    "- 'Why does the model predict this level of activity?': intent='explanation'. "
    "- 'Describe this grid's hourly activity pattern.': intent='pulse'. "
    "- 'What happens if transit accessibility doubles?': intent='whatif'. "
    "- 'Compare the baseline and scenario hourly curves.': intent='whatif'. "
    "Do NOT perform any calculations. Output only valid JSON."
)

def planner_agent(state):
    logger.info("[Planner] Processing: %s", state["query"])
    q_lower = (state.get("query") or "").lower()
    has_grid = state.get("grid_id") is not None and str(state.get("grid_id")).lower() not in ("none", "", "null")

    # Fast heuristic intent detector for study-wide research queries
    inferred_intent = None
    if any(k in q_lower for k in ("stage", "four stage", "progress", "evolution", "m0", "m1", "m2", "m3", "overview", "tell about the app", "tell abuot the app")):
        inferred_intent = "learning_status"
    elif any(k in q_lower for k in ("exp-", "experiment", "memory", "rejected", "accepted", "hypotheses", "hypothesis tested", "history", "discovered")):
        inferred_intent = "memory_audit"
    elif any(k in q_lower for k in ("theor", "literature", "jacobs", "hillier", "batty", "space syntax", "propos")):
        inferred_intent = "literature_theory"
    elif any(k in q_lower for k in ("error", "anomal", "underpredict", "overpredict", "mismatch", "bypass", "informal")):
        if not has_grid or any(k in q_lower for k in ("study", "across", "all grid", "systemic", "pattern", "clusters")):
            inferred_intent = "research_diagnosis"

    plan_data = {}
    llm = _get_llm()
    messages = [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=f"Query: {state['query']!r}\nGrid ID: {state['grid_id']!r}")]
    try:
        response = llm.invoke(messages)
        text = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        plan_data = json.loads(m.group()) if m else {}
    except Exception as exc:
        logger.warning("[Planner] LLM failed: %s", exc)
        plan_data = {}

    default_intent = inferred_intent or ("combined" if has_grid else "learning_status")
    defaults = {
        "intent": default_intent,
        "needs_prediction": has_grid and not inferred_intent,
        "needs_explanation": has_grid and not inferred_intent,
        "needs_oof": has_grid and not inferred_intent,
        "needs_pulse": has_grid and not inferred_intent,
        "needs_whatif": False,
        "needs_research": not has_grid or bool(inferred_intent),
        "clarification_needed": False,
        "clarification_question": None,
    }
    for k, v in defaults.items():
        plan_data.setdefault(k, v)

    if inferred_intent:
        plan_data["intent"] = inferred_intent

    intent = plan_data.get("intent", default_intent)
    logger.info("[Planner] Intent: %s (has_grid=%s)", intent, has_grid)
    if plan_data.get("clarification_needed") and plan_data.get("clarification_question"):
        return {"plan": json.dumps(plan_data), "intent": intent, "final_answer": plan_data["clarification_question"], "messages": [AIMessage(content=f"[Planner] clarification needed")]}
    return {"plan": json.dumps(plan_data), "intent": intent, "messages": [AIMessage(content=f"[Planner] intent={intent}")]}


def data_agent(state, gdf, table):
    from data_service import get_grid_row, get_location_details, get_feature_values, validate_feature_values, get_oof_result
    logger.info("[Data] Retrieving grid: %s", state["grid_id"])
    if not state.get("grid_id"):
        r = {"success": False, "error": "No grid selected."}
        return {"data_result": r, "messages": [AIMessage(content="[Data] No grid selected.")]}
    row, error = get_grid_row(gdf, str(state["grid_id"]))
    if error:
        r = {"success": False, "error": error}
        return {"data_result": r, "messages": [AIMessage(content=f"[Data] {error}")]}
    loc = get_location_details(row)
    feat_names = ["LConn_wt", "MAD500_wt", "BtA500_wt", "Floor_Inte", "POI_Densit", "Transit_Ac", "IntDen_km2", "land_use_e"]
    feat_vals = get_feature_values(row, feat_names)
    is_valid, missing = validate_feature_values(feat_vals)
    oof = get_oof_result(row)
    result = {"success": True, "grid_id": loc["id"], "district": loc["district"], "lat": loc["lat"], "lon": loc["lon"], "feature_values": feat_vals, "features_valid": is_valid, "missing_features": missing, "oof": oof}
    return {"data_result": result, "oof_result": oof, "messages": [AIMessage(content=f"[Data] Grid {loc['id']} ok. Features valid: {is_valid}")]}

def prediction_agent(state, bundle, row):
    from prediction_service import predict_total, predict_pulse
    if not state.get("data_result", {}).get("success"):
        return {"prediction_result": {"success": False, "error": "Data unavailable."}, "pulse_result": {"success": False, "type": "unavailable"}, "messages": [AIMessage(content="[Prediction] Skipped.")]}
    if not state["data_result"].get("features_valid"):
        msg = f"[Prediction] Missing features: {state['data_result'].get('missing_features')}"
        return {"prediction_result": {"success": False, "error": msg}, "pulse_result": {"success": False, "type": "unavailable"}, "messages": [AIMessage(content=msg)]}
    total = predict_total(bundle, row)
    pulse = predict_pulse(bundle, row)
    return {"prediction_result": total, "pulse_result": pulse, "messages": [AIMessage(content=f"[Prediction] total={total.get('value')}, pulse_type={pulse.get('type')}")]}

def explanation_agent(state, bundle, row, whatif_changes=None):
    from explanation_service import get_shap_values, run_whatif_scenario
    if not state.get("data_result", {}).get("success"):
        return {"shap_result": {"success": False, "error": "Data unavailable."}, "whatif_result": {}, "messages": [AIMessage(content="[Explanation] Skipped.")]}
    shap_r = get_shap_values(bundle, row)
    whatif_r = {}
    if whatif_changes:
        whatif_r = run_whatif_scenario(bundle, row, whatif_changes)
    msgs = [AIMessage(content=f"[Explanation] SHAP ok={shap_r.get('success')}")]
    if whatif_changes:
        msgs.append(AIMessage(content=f"[Explanation] what-if ok={whatif_r.get('success')}"))
    return {"shap_result": shap_r, "whatif_result": whatif_r, "messages": msgs}

def review_agent(state):
    notes = []
    data = state.get("data_result", {})
    pred = state.get("prediction_result", {})
    pulse = state.get("pulse_result", {})
    shap = state.get("shap_result", {})
    oof = state.get("oof_result", {})
    whatif = state.get("whatif_result", {})
    notes.append("PASS: Grid retrieved." if data.get("success") else f"FAIL: {data.get('error')}")
    notes.append("PASS: All features valid." if data.get("features_valid") else f"WARN: Missing features {data.get('missing_features')}")
    if pred.get("success"):
        notes.append(f"PASS: Prediction={pred.get('value', 0):,.0f}.")
    else:
        notes.append(f"FAIL: Prediction — {pred.get('error', 'unavailable')}")
    notes.append(f"PASS: Pulse type={pulse.get('type')}." if pulse.get("success") else f"INFO: Pulse unavailable.")
    notes.append(f"PASS: SHAP computed ({len(shap.get('records',[]))} features)." if shap.get("success") else f"INFO: SHAP {shap.get('error','unavailable')}")
    notes.append("PASS: OOF result from cross-validation. Deployment prediction is separate." if oof.get("available") else "INFO: No OOF record for this grid.")
    if whatif:
        notes.append(f"PASS: What-if scenario computed." if whatif.get("success") else f"FAIL: What-if {whatif.get('error')}")
    return {"review_notes": notes, "messages": [AIMessage(content="[Review] " + " | ".join(notes))]}


# ---------------------------------------------------------------------------
# Pulse shape classifier — imported from prediction_service
# ---------------------------------------------------------------------------
from prediction_service import classify_pulse_shape
_classify_pulse_shape = classify_pulse_shape



COMPOSER_SYSTEM = (
    "You are a research assistant for an urban analytics project studying functional centrality "
    "in Colombo and Gampaha districts, Sri Lanka. "
    "Use ONLY the numbers and facts provided in the JSON context. Never invent or estimate values. "
    "If a value is unavailable say so explicitly. Write in clear, natural language. Use markdown.\n"
    "IMPORTANT: Answer only what the USER QUESTION asks. Do NOT repeat all facts for every question. "
    "Focus your answer on the specific aspect the user is curious about.\n\n"
    "If intent=location: describe where the grid is and its urban context.\n"
    "If intent=prediction: explain the predicted activity level and what drives it.\n"
    "If intent=pulse: describe the hourly curve shape and peak timing directly. "
    "Do NOT speculate on commuting, retail, leisure, or human land-use activities from hourly peaks alone unless proven. "
    "Clarify that hourly curve values reflect 24-hour log1p intensity profiles rather than raw count totals.\n"
    "If intent=explanation: for each SHAP feature explain in plain English what it measures, "
    "what value this grid has, whether it pushes prediction higher or lower, and why that makes urban sense.\n"
    "If intent=oof or mismatch: explain the difference between deployment prediction and OOF result "
    "in plain terms, interpret the residual direction and percentage mismatch, and connect it to "
    "Urban Morphology & Centrality Theory (e.g., Space Syntax 'through-movement vs to-movement', "
    "edge effects/barrier severance in coastal/lagoon corridors, or informal organic vitality vs physical GIS coverage).\n"
    "If intent=whatif: explain how total activity changes AND whether the hourly pulse peak or shape "
    "shifts. In hourly comparisons, only report an increase if a value actually increased (positive difference), "
    "and a decrease if it actually decreased (negative difference). Mention any feature bounds warnings if present.\n"
    "If intent=combined: give a concise but complete summary covering all aspects asked in the user question "
    "(such as location, prediction, pulse shape, SHAP drivers, OOF residual, and whatif scenario results if present in context).\n\n"
    "Always close with a one-sentence note: predictions are model estimates not observations; "
    "SHAP describes model behaviour not causation."
)

def _build_context(state: dict) -> dict:
    """Build a focused context dict for the LLM, trimmed to what the intent needs."""
    intent = state.get("intent", "combined")
    pulse  = state.get("pulse_result", {})
    shap   = state.get("shap_result", {})
    whatif = state.get("whatif_result", {})

    pulse_shape: dict = {}
    if pulse.get("success") and pulse.get("type") == "hourly" and pulse.get("values"):
        label, desc = _classify_pulse_shape(pulse["values"])
        pulse_shape = {
            "label": label,
            "description": desc,
            "peak_hour": pulse.get("peak_hour"),
            "peak_share_pct": round(pulse.get("peak_share_pct", 0), 1),
            "hourly_values": {
                f"{h:02d}:00": round(val, 1)
                for h, val in enumerate(pulse["values"])
            },
        }

    # Always include these base fields
    ctx: dict = {
        "USER_QUESTION": state.get("query"),
        "intent": intent,
        "grid_id": state.get("grid_id"),
        "data": state.get("data_result", {}),
    }

    # Add sections relevant to the intent
    if intent in ("prediction", "explanation", "combined", "whatif"):
        ctx["prediction"] = state.get("prediction_result", {})
    if intent in ("pulse", "combined", "whatif"):
        ctx["pulse_shape"] = pulse_shape
    if intent in ("explanation", "combined"):
        # Send all SHAP records for explanation, top-3 for combined to save tokens
        records = shap.get("records", [])
        ctx["shap_features"] = records if intent == "explanation" else records[:3]
    if intent in ("oof", "mismatch", "combined"):
        oof_data = state.get("oof_result", {})
        if oof_data.get("available"):
            obs = oof_data.get("observed", 0)
            pred_v = oof_data.get("predicted", 1)
            diff_v = oof_data.get("diff", 0)
            pct_mismatch = round((diff_v / max(1.0, pred_v)) * 100, 1)
            oof_data["percentage_deviation"] = f"{pct_mismatch:+,.1f}%"
            
            # Integrate grid_evolution_service for theory diagnosis & correction
            try:
                import grid_evolution_service as ges
                row_dict = state.get("data_result", {}).get("feature_values", {})
                base_val = state.get("prediction_result", {}).get("value", pred_v)
                shap_r = state.get("shap_result", {})
                evo_info = ges.diagnose_and_evolve_grid(pd.Series(row_dict), base_val, oof_data, shap_r)
                if evo_info.get("has_diagnosis"):
                    oof_data["theory_grounding"] = {
                        "archetype": evo_info["archetype_name"],
                        "theory": evo_info["theory"],
                        "doi": evo_info["doi"],
                        "scientific_diagnosis": evo_info["diagnosis_narrative"],
                        "hypothesis_tested": evo_info["hypothesis"],
                        "baseline_prediction": f"{evo_diag.get('baseline_prediction', base_val):,.0f}" if 'evo_diag' in locals() else f"{base_val:,.0f}",
                        "evolved_prediction": f"{evo_info['evolved_prediction']:,.0f}",
                        "error_reduction_pct": f"{evo_info['error_reduction_pct']:.1f}%",
                    }
            except Exception as e:
                logger.warning("Could not append evolution context: %s", e)
        ctx["oof"] = oof_data
    if intent == "whatif" or whatif.get("success"):
        ctx["whatif"] = whatif
        ctx["pulse_shape"] = pulse_shape  # always include pulse for whatif

    return ctx


def compose_response(state):
    llm = _get_llm()
    context = _build_context(state)
    messages = [
        SystemMessage(content=COMPOSER_SYSTEM),
        HumanMessage(content=f"Context (JSON):\n{json.dumps(context, indent=2, default=str)}"),
    ]
    try:
        resp = llm.invoke(messages)
        text = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
        if not text:
            logger.info("[Composer] Extracted text empty; using deterministic fallback")
            text = _deterministic_answer(state)
        return {"final_answer": text, "messages": [AIMessage(content="[Composer] done")]}
    except Exception as exc:
        logger.error("[Composer] LLM failed: %s", exc)
        return {"final_answer": _deterministic_answer(state),
                "messages": [AIMessage(content=f"[Composer] fallback used: {exc}")]}
def _deterministic_answer(state):
    """Intent-aware fallback: each question type gets a focused, distinct answer."""
    intent = state.get("intent", "combined")
    research = state.get("research_result", {})
    data   = state.get("data_result", {})
    pred   = state.get("prediction_result", {})
    pulse  = state.get("pulse_result", {})
    oof    = state.get("oof_result", {})
    shap   = state.get("shap_result", {})
    whatif = state.get("whatif_result", {})

    # =========================================================================
    # 1. RESEARCH DIAGNOSIS (Study-Wide Error Profiling across 7,764 Grids)
    # =========================================================================
    if intent == "research_diagnosis":
        parts = ["### Study-Wide Spatial Error Diagnosis (All 7,764 Grid Cells)"]
        parts.append(
            "An empirical audit of spatial cross-validation out-of-fold (OOF) residuals across "
            "Colombo and Gampaha reveals **two systemic failure archetypes** where physical morphology alone fails:\n"
        )
        parts.append("#### 1. Archetype A — Infrastructure Bypass & Barrier Severance (Severe Overprediction)")
        parts.append("- **Condition:** Observed activity is severely lower than model predictions (< -50% mismatch).")
        parts.append("- **Scale:** ~1,798 grid cells (~23.2% of the study area).")
        parts.append("- **Diagnostic Pattern:** High transit accessibility (`Transit_Ac`) and wide road intersections, but actual crowd presence is low.")
        parts.append(
            "- **Urban Theory Connection:** Under *Space Syntax Natural Movement Theory (Hillier et al., 1993)*, "
            "arterial corridors function as high-speed 'through-movement' conduits or barrier edges (e.g. coastal/lagoon severance) "
            "rather than accessible pedestrian 'to-movement' destinations.\n"
        )
        parts.append("#### 2. Archetype B — Organic & Informal Commercial Vitality (Severe Underprediction)")
        parts.append("- **Condition:** Real-world activity dramatically exceeds model estimates (> +50% mismatch).")
        parts.append("- **Scale:** ~3,407 grid cells (~43.9% of the study area).")
        parts.append("- **Diagnostic Pattern:** Moderate or low physical road width and registered building height, but massive human footfall.")
        parts.append(
            "- **Urban Theory Connection:** Under *Jane Jacobs' Sidewalk Vitality Theory (1961)*, unmapped informal bazaars, "
            "curbside micro-retail, and organic street life generate intense human presence missed by standard physical GIS databases.\n"
        )
        parts.append(
            "*(These systemic contradiction profiles directly motivate our Theory-Guided (Stage 2) and Additional-Data (Stage 3) experiments.)*"
        )
        return "\n".join(parts)

    # =========================================================================
    # 2. LITERATURE & THEORY RETRIEVAL
    # =========================================================================
    if intent == "literature_theory":
        theories = research.get("theories", [])
        if not theories:
            from literature_service import FOUNDATIONAL_THEORIES
            theories = FOUNDATIONAL_THEORIES

        parts = ["### Grounding Urban Theories & Scientific Literature"]
        parts.append(
            "The TSE-UI agent connects empirical model errors to peer-reviewed urban science literature "
            "with authentic DOIs to formulate theoretically justified hypotheses:\n"
        )
        for t in theories:
            parts.append(f"#### • {t['theory_name']}")
            parts.append(f"- **Authors:** {t['author']} ({t['year']})")
            parts.append(f"- **Publication:** *{t['title']}*")
            parts.append(f"- **DOI Link:** [{t['doi']}]({t['doi']})")
            parts.append(f"- **Theoretical Proposition:** {t['proposition']}")
            parts.append(f"- **Relevance to Model Failures:** {t['relevance']}\n")
        parts.append(
            "*These theoretical foundations guide candidate interaction features and data layer selections.*"
        )
        return "\n".join(parts)

    # =========================================================================
    # 3. SCIENTIFIC MEMORY AUDIT (Experiments & Verdicts)
    # =========================================================================
    if intent == "memory_audit":
        mem = research.get("scientific_memory", {})
        exps = mem.get("experiments", [])
        if not exps:
            from registry_service import load_scientific_memory
            mem = load_scientific_memory()
            exps = mem.get("experiments", [])

        n_total = mem.get("experiments_count", len(exps))
        n_acc = mem.get("hypotheses_accepted", sum(1 for e in exps if e.get("verdict") == "ACCEPTED"))
        n_rej = mem.get("hypotheses_rejected", sum(1 for e in exps if e.get("verdict") == "REJECTED"))

        parts = ["### Scientific Memory Audit (Experiment Ledger)"]
        parts.append(f"- **Total Experiments Logged:** {n_total}")
        parts.append(f"- **Hypotheses Accepted:** {n_acc} | **Hypotheses Rejected:** {n_rej}")
        parts.append("- **Original Reference Baseline:** Log1p Random Forest (8 Features), $R^2 = 0.5156$, $\\text{RMSE} = 29,785.44$, $\\text{MAE} = 14,143.19$.\n")

        parts.append("#### Key Recent Experiments & Scientific Verdicts:")

        # EXP-008
        e8 = next((e for e in exps if e.get("id") == "EXP-008"), None)
        if e8:
            parts.append("##### 1. EXP-008: Stage 1 — Fold-Specific Isotonic Calibration (`REJECTED`)")
            parts.append("- **Hypothesis:** Monotonic non-linear calibration on in-fold OOB predictions will correct log1p expm1 peak underprediction.")
            parts.append(f"- **Results:** $R^2 = {e8.get('evolved_r2', 0.4851):.4f}$ ($\\Delta R^2 = {e8.get('delta_r2', -0.0304):+.4f}$), $\\Delta\\text{{RMSE}} = {e8.get('delta_rmse', 921.07):+.1f}$.")
            parts.append("- **Scientific Rationale:** While Q4 underprediction bias halved ($-24,400 \\rightarrow -12,453$), low-activity Q1 overprediction doubled ($+2,912 \\rightarrow +6,963$). Monotonic recalibration shifted the entire curve upwards, distorting peri-urban cells.")

        # EXP-009
        e9 = next((e for e in exps if e.get("id") == "EXP-009"), None)
        if e9:
            parts.append("\n##### 2. EXP-009: Stage 2 — Theoretical T1 Walkable Vitality Mix (`REJECTED`)")
            parts.append("- **Feature Formula:** `Walkable_Vitality_Mix = Floor_Inte * land_use_e` (Jane Jacobs, 1961).")
            parts.append(f"- **Results:** $R^2 = {e9.get('evolved_r2', 0.5128):.4f}$ ($\\Delta R^2 = {e9.get('delta_r2', -0.0028):+.4f}$), $\\Delta\\text{{RMSE}} = {e9.get('delta_rmse', 84.66):+.1f}$.")
            parts.append("- **Scientific Rationale:** Improved suburban folds 1–3, but added slight noise in urban core folds 4–5. Did not satisfy multi-metric acceptance rules.")

        # EXP-010
        e10 = next((e for e in exps if e.get("id") == "EXP-010"), None)
        if e10:
            parts.append("\n##### 3. EXP-010: Stage 2 — Theoretical T2 Pedestrian Permeability Synergy (`ACCEPTED`)")
            parts.append("- **Feature Formula:** `Pedestrian_Permeability_Synergy = MAD500_wt * IntDen_km2` (Hillier & Hanson 1984; Salingaros 2005).")
            parts.append(f"- **Results:** $R^2 = {e10.get('evolved_r2', 0.5181):.4f}$ ($\\mathbf{{\\Delta R^2 = {e10.get('delta_r2', 0.0025):+.4f}}}$), $\\mathbf{{\\Delta\\text{{RMSE}} = {e10.get('delta_rmse', -77.10):-.1f}}}$, $\\mathbf{{\\Delta\\text{{MAE}} = -25.78}}$.")
            parts.append("- **Scientific Rationale:** Outperformed baseline in 4 of 5 spatial folds. Colombo District MAE dropped by $-74.43$. Resolved both overprediction ($-10.22$) and underprediction ($-28.59$) error clusters.")
            parts.append("- **Status:** Currently established as the **Active Champion Model**.")

        return "\n".join(parts)

    # =========================================================================
    # 4. FOUR-STAGE PROGRESSIVE LEARNING STATUS
    # =========================================================================
    if intent == "learning_status":
        champ = research.get("scientific_memory", {}).get("champion", {})
        if not champ:
            from registry_service import get_champion_model_state
            champ = get_champion_model_state()
        champ_id = champ.get("champion_id", "EXP-010")
        champ_r2 = champ.get("champion_metrics", {}).get("r2", 0.5181)

        parts = ["### TSE-UI Progressive Learning Framework — Four-Stage Evolution Status"]
        parts.append(
            "The TSE-UI application progressively improves urban activity predictions through four rigorous, "
            "scientifically evaluated stages without data leakage:\n"
        )
        parts.append("| Stage | Learning Mechanism | Active Model / Candidate | Spatial 5-Fold $R^2$ | Status |")
        parts.append("| :--- | :--- | :--- | :---: | :---: |")
        parts.append("| **Stage 0** | Baseline Reference | 8 Morphological & Centrality Predictors | **0.5156** | Reference Baseline (Immutable) |")
        parts.append("| **Stage 1** | Statistical Refinement | EXP-008 In-Fold Isotonic Calibration | 0.4851 | **REJECTED** (Distorted rural/peri-urban cells) |")
        parts.append(f"| **Stage 2** | Theory-Guided Hypotheses | **{champ_id}** Pedestrian Permeability Synergy | **{champ_r2:.4f}** | **ACCEPTED** (Active Champion, $\\Delta R^2 = +0.0025$) |")
        parts.append("| **Stage 3** | Additional-Data Learning | POI Functional Disaggregation & Land-Use Mix | *In Preparation* | Next Candidate for Evaluation |\n")

        parts.append("#### Key Insights from the Learning Process:")
        parts.append(
            "1. **Statistical Learning Lesson:** Global monotonic curve calibration is ineffective for long-tailed urban activity "
            "because boosting peak values shifts the distribution upward, severely degrading the 80% low-density cells."
        )
        parts.append(
            "2. **Theoretical Learning Lesson:** Spatial configuration interactions (`MAD500_wt * IntDen_km2`) successfully "
            "boost urban core performance (Colombo MAE -74.43) without degrading suburban folds."
        )
        parts.append(
            "3. **Next Stage:** Investigating functional POI subcategories (`restaurant`, `supermarket`, `cafe`, `bank` from `pois.shp`) "
            "to target the remaining Archetype B informal/organic underprediction clusters."
        )
        return "\n".join(parts)

    gid    = data.get("grid_id", state.get("grid_id", "?"))
    parts  = [f"### Grid {gid}"]


    # --- LOCATION ---
    if intent in ("location", "morphology", "combined"):
        if data.get("success"):
            parts.append(f"- **District:** {data['district']}")
            parts.append(f"- **Coordinates:** {data['lat']:.5f}\u00b0N, {data['lon']:.5f}\u00b0E")
            parts.append(
                f"  This grid sits in the **{data['district']}** district. "
                "Its coordinates place it within the Colombo / Gampaha metropolitan study area."
            )
            # If the query asks what activity to expect, include prediction & peak hour
            q_lower = (state.get("query") or "").lower()
            if "activity" in q_lower or "expect" in q_lower or intent == "combined":
                if pred.get("success"):
                    parts.append(f"- **Deployment prediction:** {pred['value']:,.0f} activity units")
                if pulse.get("success") and pulse.get("peak_hour") is not None:
                    parts.append(f"- **Predicted peak hour:** {pulse['peak_hour']:02d}:00")

    # --- PREDICTION ---
    if intent in ("prediction", "combined"):
        if pred.get("success"):
            parts.append(f"- **Deployment prediction:** {pred['value']:,.0f} activity units")
            parts.append(
                "  The model generates this estimate from morphological and accessibility features. "
                "This is a deployment-time inference, not a direct observation."
            )

    # --- PULSE / HOURLY PATTERN ---
    if intent in ("pulse", "combined"):
        if pulse.get("success") and pulse.get("type") == "hourly" and pulse.get("values"):
            vals = pulse["values"]
            label, desc = _classify_pulse_shape(vals)
            peak = pulse["peak_hour"]
            pk_share = round(pulse.get("peak_share_pct", 0), 1)
            parts.append(f"- **Pulse shape:** {label}")
            parts.append(f"  {desc}")
            parts.append(f"  Peak hour **{peak:02d}:00** accounts for **{pk_share}%** of the day's total predicted activity.")
            # Show the 3 busiest and 3 quietest hours
            ranked = sorted(enumerate(vals), key=lambda x: x[1], reverse=True)
            top3 = ", ".join(f"{h:02d}:00" for h, _ in ranked[:3])
            bot3 = ", ".join(f"{h:02d}:00" for h, _ in ranked[-3:])
            parts.append(f"  **Busiest hours:** {top3} | **Quietest hours:** {bot3}")
        elif pulse.get("success") and pulse.get("type") == "hourly":
            parts.append(f"- **Predicted peak hour:** {pulse['peak_hour']:02d}:00")
        else:
            parts.append("- Pulse prediction unavailable for this grid.")

    # --- SHAP EXPLANATION ---
    if intent in ("explanation", "combined"):
        records = shap.get("records", []) if shap.get("success") else []
        if records:
            n_show = len(records) if intent == "explanation" else min(3, len(records))
            parts.append(f"\n**Feature contributions (SHAP) — top {n_show} drivers:**")
            for r in records[:n_show]:
                arrow = "\u2191 raises" if r["shap_value"] > 0 else "\u2193 lowers"
                parts.append(
                    f"- **{r['label']}** = {r['grid_value']:.4f} "
                    f"({arrow} prediction by {abs(r['shap_value']):.3f})"
                )
            parts.append(
                "\n*SHAP values quantify each feature's marginal contribution to this grid's "
                "prediction relative to the average grid. They describe model behaviour, not causation.*"
            )
        else:
            parts.append("- SHAP explanation unavailable.")

    # --- OOF / MISMATCH ---
    if intent in ("oof", "mismatch", "combined"):
        if oof.get("available"):
            diff_val = oof.get("diff")
            pred_v = oof.get("predicted", 1)
            pct_mismatch = (diff_val / max(1.0, pred_v)) * 100 if diff_val is not None else 0.0
            parts.append(f"- **OOF observed (ground truth):** {oof['observed']:,.0f}")
            parts.append(f"- **OOF predicted (cross-val):** {oof['predicted']:,.0f}")
            parts.append(f"- **Residual (obs\u2212pred):** {diff_val:+,.0f} ({pct_mismatch:+.1f}%)")
            if diff_val is not None:
                if diff_val > 0:
                    interp = (
                        f"The cross-validation model **underestimated** this grid by {diff_val:,.0f} units ({pct_mismatch:+.1f}%). "
                        "Real-world activity here exceeded what physical morphology alone expected. "
                        "\n  - **Urban Morphology Theory Connection:** *Organic / Informal Centrality (Jane Jacobs, 1961)*. "
                        "Pedestrian vitality and informal roadside trade/gatherings often generate strong human presence that "
                        "standard GIS road width and registered POI databases fail to capture."
                    )
                elif diff_val < 0:
                    interp = (
                        f"The cross-validation model **overestimated** this grid by {abs(diff_val):,.0f} units ({pct_mismatch:+.1f}%). "
                        "The model expected more activity than was observed. "
                        "\n  - **Urban Morphology Theory Connection:** *Space Syntax 'Through-Movement vs. To-Movement' (Bill Hillier)* "
                        "and *Edge Barrier Severance*. High transit infrastructure or wide roadways often serve as transit bypasses or barrier "
                        "corridors (e.g. lagoon/highway edges) rather than accessible pedestrian destinations."
                    )
                else:
                    interp = "Observed and OOF-predicted values match exactly, indicating well-calibrated morphological equilibrium."
                parts.append(f"  *{interp}*")
            parts.append(
                "  *Note: the OOF result is from held-out cross-validation folds during training. "
                "The deployment prediction above comes from the full trained model and is separate.*"
            )

    # --- WHAT-IF ---
    if intent == "whatif" or whatif.get("success"):
        if whatif.get("success"):
            chg = whatif.get("changes_applied", {})
            chg_str = ", ".join(f"{k} -> {v:.4f}" for k, v in chg.items())
            parts.append(f"\n**What-if scenario:** {chg_str}")
            parts.append(
                f"- **Baseline total:** {whatif['baseline_prediction']:,.0f} activity units"
            )
            parts.append(
                f"- **Scenario total:** {whatif['scenario_prediction']:,.0f} activity units "
                f"({whatif['absolute_difference']:+,.0f}, {whatif.get('percent_difference') or 0:+.1f}%)"
            )
            pc = whatif.get("pulse_comparison", {})
            if pc.get("available"):
                shift = "shifted" if pc.get("peak_hour_shifted") else "unchanged"
                lbl_chg = "and the pulse shape also changed" if pc.get("shape_changed") else "but the pulse shape remains the same"
                parts.append(
                    f"- **Pulse peak:** {pc['baseline_peak_hour']:02d}:00 ({pc.get('baseline_label')}) "
                    f"\u2192 {pc['scenario_peak_hour']:02d}:00 ({pc.get('scenario_label')}) [{shift}]"
                )
                parts.append(f"  {lbl_chg.capitalize()}.")
                if pc.get("scenario_description"):
                    parts.append(f"  Scenario pattern: {pc['scenario_description']}")
                
                # If query asks to compare hourly curves or 24-hour profile
                q_lower = (state.get("query") or "").lower()
                if any(w in q_lower for w in ("curve", "compare", "hourly", "pattern changes", "pattern change")):
                    inc = pc.get("largest_increase")
                    dec = pc.get("largest_decrease")
                    parts.append("\n**24-Hour Curve Comparison Highlights:**")
                    if inc:
                        parts.append(
                            f"- **Largest increase:** Hour {inc['hour']:02d}:00 "
                            f"(+{inc['diff']:.1f} log1p-units, from {inc['baseline_val']:.1f} to {inc['scenario_val']:.1f})"
                        )
                    else:
                        parts.append("- **Largest increase:** None (no hourly values increased in this scenario).")

                    if dec:
                        parts.append(
                            f"- **Largest decrease:** Hour {dec['hour']:02d}:00 "
                            f"({dec['diff']:.1f} log1p-units, from {dec['baseline_val']:.1f} to {dec['scenario_val']:.1f})"
                        )
                    else:
                        parts.append("- **Largest decrease:** None (no hourly values decreased in this scenario).")

                    parts.append(
                        f"- **Peak hour level:** Baseline peak is {pc.get('baseline_peak_value', 0):.1f} at {pc['baseline_peak_hour']:02d}:00; "
                        f"Scenario peak is {pc.get('scenario_peak_value', 0):.1f} at {pc['scenario_peak_hour']:02d}:00."
                    )
                    parts.append("  *Note on scale: Hourly curve values represent log1p-scale activity intensity across the 24 hours (not raw count totals, which are reported as total activity).*")

            bw = whatif.get("bounds_warnings", [])
            for w in bw:
                parts.append(f"  *Warning: {w}*")
            parts.append("  *Model-based scenario. Not a causal estimate.*")

    parts.append("\n*Predictions are model estimates, not observations. "
                 "SHAP describes model behaviour, not causation.*")
    return "\n".join(parts)


def _merge_step(state, step_res):
    if not step_res:
        return
    if "messages" in step_res:
        state["messages"].extend(step_res["messages"])
    for k, v in step_res.items():
        if k != "messages":
            state[k] = v

def extract_whatif_from_query(query: str, feat_vals: dict) -> dict:
    """Extract feature modification from natural language if present.
    
    Data-driven: matches query keywords to features and applies requested
    change (e.g. +25% for increase, double, halve, or specific percentage).
    """
    q = query.lower()
    if 'investigate' in q and any(word in q for word in ('pattern', 'unexplained activity')):
        from pattern_agent_service import run_discovery, save_run
        run = run_discovery(table, query)
        path = save_run(run)
        lines = [f"Agent-directed pattern investigation: {run['status']}.",
                 f"Audit record: {run['run_id']}"]
        for finding in run['findings']:
            lines.append(f"- Grid {finding['grid_id']}, {finding['pattern']}: {json.dumps(finding['evidence'])}. Rule: {finding['rule']}")
        if run['status'] != 'completed':
            lines.append('No completed discovery report. Review the audit trace for model or validation errors.')
        lines.append(run['limitations'])
        return {'intent': 'agent_pattern_investigation', 'pattern_agent_run': run,
                'audit_path': path, 'review_notes': ['Only inspected tool evidence is reportable.'],
                'final_answer': '\n'.join(lines)}
    mapping = {
        "transit": "Transit_Ac",
        "poi": "POI_Densit",
        "floor": "Floor_Inte",
        "betweenness": "BtA500_wt",
        "bta": "BtA500_wt",
        "closeness": "MAD500_wt",
        "mad": "MAD500_wt",
        "reach": "MAD500_wt",
        "connectivity": "LConn_wt",
        "lconn": "LConn_wt",
        "intersection": "IntDen_km2",
        "intden": "IntDen_km2",
        "land use": "land_use_e",
        "entropy": "land_use_e",
    }
    target_feat = None
    for k, f in mapping.items():
        if k in q or f.lower() in q:
            target_feat = f
            break
    if not target_feat or target_feat not in feat_vals:
        return {}
    base = feat_vals.get(target_feat)
    if base is None:
        return {}

    pct_m = re.search(r'(\d+)\s*(?:%|percent)', q)
    if pct_m:
        pct = float(pct_m.group(1)) / 100.0
        if any(w in q for w in ("decrease", "lower", "reduce", "less", "drop")):
            return {target_feat: max(0.0, base * (1.0 - pct))}
        return {target_feat: base * (1.0 + pct)}

    if "double" in q:
        return {target_feat: base * 2.0}
    if "halve" in q or "half" in q:
        return {target_feat: base * 0.5}

    if any(w in q for w in ("increase", "higher", "more", "boost", "grow", "expand")):
        return {target_feat: base * 1.25 if base > 0 else 0.5}
    if any(w in q for w in ("decrease", "lower", "reduce", "less", "drop")):
        return {target_feat: base * 0.75}

    return {}


def run_workflow(query, grid_id, bundle, gdf, table, whatif_changes=None):
    # Explicit detection requests use measured evidence, never canned archetype text.
    q = query.lower()
    if (any(word in q for word in ("detect", "scan", "find", "identify"))
            and any(word in q for word in ("pattern", "evening", "day/night", "similar morphology", "spatial residual"))):
        from pattern_detection_service import detect_patterns
        scope_id = None if any(word in q for word in ("all", "study area", "across")) else grid_id
        result = detect_patterns(table, scope_id)
        lines = [f"Observed pattern scan: {result['cohort_size']} reference grids."]
        for name, availability in result['status'].items():
            matches = [f for f in result['findings'] if f['pattern'] == name]
            lines.append(f"- {name}: {len(matches)} matching grids ({availability}).")
            for finding in matches[:3]:
                lines.append(f"  Grid {finding['grid_id']}: {json.dumps(finding['evidence'])}. Rule: {finding['rule']}")
        if result['spatial_summary']:
            lines.append("Global spatial diagnostic: " + json.dumps(result['spatial_summary']))
        lines.append(result['limitations'])
        return {"intent": "pattern_detection", "pattern_result": result,
                "review_notes": ["Read-only observed-data detectors; no causal attribution or model changes."],
                "final_answer": "\n".join(lines)}
    from data_service import get_grid_row, get_feature_values
    from prediction_service import get_features
    state = {
        "query": query, "grid_id": grid_id,
        "messages": [HumanMessage(content=query)],
        "plan": "", "data_result": {}, "prediction_result": {},
        "pulse_result": {}, "shap_result": {}, "oof_result": {},
        "whatif_result": {}, "review_notes": [], "final_answer": "",
        "error": None, "intent": "combined",
    }
    # Agent 1: Planner
    _merge_step(state, planner_agent(state))
    if state.get("final_answer"): return state
    # Resolve row for downstream agents
    row = None
    if grid_id:
        row, _ = get_grid_row(gdf, str(grid_id))
    # Agent 2: Data
    _merge_step(state, data_agent(state, gdf, table))
    # Override feature values with those from the authoritative bundle feature list
    if row is not None:
        from data_service import validate_feature_values
        feats = get_features(bundle)
        feat_vals = get_feature_values(row, feats)
        is_valid, missing = validate_feature_values(feat_vals)
        state["data_result"]["feature_values"] = feat_vals
        state["data_result"]["features_valid"] = is_valid
        state["data_result"]["missing_features"] = missing
    # Agent 3: Prediction
    try: plan_data = json.loads(state.get("plan","{}"))
    except: plan_data = {}
    intent = state.get("intent","combined")
    needs_pred = plan_data.get("needs_prediction", True) or intent in ("prediction","explanation","combined","whatif","pulse", "location")
    if needs_pred and row is not None:
        _merge_step(state, prediction_agent(state, bundle, row))
    # Agent 4: Explanation
    needs_shap = plan_data.get("needs_explanation", False) or intent in ("explanation","combined")
    active_whatif = dict(whatif_changes) if whatif_changes else {}
    if not active_whatif:
        # Check if query asks a what-if scenario (e.g. doubles, halves, increase, what if)
        inferred = extract_whatif_from_query(query, state["data_result"].get("feature_values", {}))
        if inferred:
            active_whatif = inferred
    needs_wi = bool(active_whatif) or intent == "whatif"
    if (needs_shap or needs_wi) and row is not None:
        _merge_step(state, explanation_agent(state, bundle, row, active_whatif if active_whatif else None))
    # Agent 5: Review
    _merge_step(state, review_agent(state))
    # Compose
    _merge_step(state, compose_response(state))
    return state


def make_investigation_planner(*, proposal_allowed=True, theory_only=False):
    """Dedicated schema-constrained planner; leaves the existing chat unchanged."""
    from investigation_models import M2ProposalPlannerTurn, ProposalPlannerTurn, ToolPlannerTurn
    return ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0,
        reasoning=False, seed=42, num_predict=3000,
        format=((M2ProposalPlannerTurn if theory_only else ProposalPlannerTurn)
                if proposal_allowed else ToolPlannerTurn).model_json_schema(),
        client_kwargs={"timeout": 75},
    )


def create_investigation(grid_id, query, **kwargs):
    """Backend entry point, intentionally not wired into Streamlit in Phase 3."""
    from investigation_service import InvestigationEngine
    return InvestigationEngine(grid_id, query, **kwargs)

