import os
from project_paths import PROJECT_ROOT
BASE = str(PROJECT_ROOT)

# ---- agent_workflow.py ----
aw = open(os.path.join(BASE, "agent_workflow.py"), "w", encoding="utf-8")
aw.write("""from __future__ import annotations
import json, logging, re, operator
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
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_predict=1024)

def check_ollama_available():
    try:
        import ollama
        models = ollama.list()
        names = [m.model for m in models.models]
        ok = any(OLLAMA_MODEL in n for n in names)
        return ok, f"Model {OLLAMA_MODEL!r} {'found' if ok else 'NOT found'}. Available: {names}"
    except Exception as exc:
        return False, f"Ollama unavailable: {exc}"

PLANNER_SYSTEM = (
    "You are the Planner Agent for an Urban Pulse research system. "
    "Your ONLY job is to identify the user intent and output a JSON object with keys: "
    "intent (one of: location, morphology, prediction, pulse, explanation, oof, whatif, mismatch, combined), "
    "needs_prediction (bool), needs_explanation (bool), needs_oof (bool), needs_pulse (bool), needs_whatif (bool), "
    "clarification_needed (bool), clarification_question (str or null). "
    "Do NOT perform any calculations. Output only valid JSON."
)

def planner_agent(state):
    logger.info("[Planner] Processing: %s", state["query"])
    llm = _get_llm()
    messages = [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=f"Query: {state['query']!r}\\nGrid ID: {state['grid_id']!r}")]
    try:
        response = llm.invoke(messages)
        text = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
        m = re.search(r"\\{[^{}]+\\}", text, re.DOTALL)
        plan_data = json.loads(m.group()) if m else {}
    except Exception as exc:
        logger.warning("[Planner] LLM failed: %s", exc)
        plan_data = {}
    defaults = {"intent": "combined", "needs_prediction": True, "needs_explanation": True, "needs_oof": True, "needs_pulse": True, "needs_whatif": False, "clarification_needed": False, "clarification_question": None}
    for k, v in defaults.items():
        plan_data.setdefault(k, v)
    intent = plan_data.get("intent", "combined")
    logger.info("[Planner] Intent: %s", intent)
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
    notes.append(f"PASS: Prediction={pred.get('value',0):,.0f}." if pred.get("success") else f"FAIL: {pred.get('error')}")
    notes.append(f"PASS: Pulse type={pulse.get('type')}." if pulse.get("success") else f"INFO: Pulse unavailable.")
    notes.append(f"PASS: SHAP computed ({len(shap.get('records',[]))} features)." if shap.get("success") else f"INFO: SHAP {shap.get('error','unavailable')}")
    notes.append("PASS: OOF result from cross-validation. Deployment prediction is separate." if oof.get("available") else "INFO: No OOF record for this grid.")
    if whatif:
        notes.append(f"PASS: What-if scenario computed." if whatif.get("success") else f"FAIL: What-if {whatif.get('error')}")
    return {"review_notes": notes, "messages": [AIMessage(content="[Review] " + " | ".join(notes))]}

COMPOSER_SYSTEM = (
    "You are a research assistant explaining urban analytics results. "
    "Use ONLY the numbers and facts from the JSON provided. Do NOT invent values. "
    "If a value is unavailable, say so. Keep it concise and use plain language. "
    "Explain SHAP and OOF briefly. Use markdown. "
    "Mention: predictions are model estimates not observations. SHAP explains model behaviour not causation."
)

def compose_response(state):
    llm = _get_llm()
    context = {
        "query": state.get("query"), "grid_id": state.get("grid_id"), "intent": state.get("intent"),
        "data": state.get("data_result", {}),
        "prediction": state.get("prediction_result", {}),
        "pulse_type": state.get("pulse_result", {}).get("type"),
        "pulse_peak_hour": state.get("pulse_result", {}).get("peak_hour"),
        "pulse_peak_value": state.get("pulse_result", {}).get("peak_value"),
        "shap_top3": state.get("shap_result", {}).get("records", [])[:3],
        "oof": state.get("oof_result", {}),
        "whatif": state.get("whatif_result", {}),
        "review_notes": state.get("review_notes", []),
    }
    messages = [SystemMessage(content=COMPOSER_SYSTEM), HumanMessage(content=f"Results:\\n{json.dumps(context, indent=2, default=str)}")]
    try:
        resp = llm.invoke(messages)
        text = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
        return {"final_answer": text, "messages": [AIMessage(content="[Composer] done")]}
    except Exception as exc:
        logger.error("[Composer] LLM failed: %s", exc)
        return {"final_answer": _deterministic_answer(state), "messages": [AIMessage(content=f"[Composer] fallback used: {exc}")]}

def _deterministic_answer(state):
    data = state.get("data_result", {})
    pred = state.get("prediction_result", {})
    pulse = state.get("pulse_result", {})
    oof = state.get("oof_result", {})
    shap = state.get("shap_result", {})
    gid = data.get("grid_id", state.get("grid_id", "?"))
    parts = [f"### Grid {gid}"]
    if data.get("success"):
        parts.append(f"- **District:** {data['district']}")
        parts.append(f"- **Centre:** {data['lat']:.5f}, {data['lon']:.5f}")
    if pred.get("success"):
        parts.append(f"- **Deployment prediction:** {pred['value']:,.0f} activity units (model estimate)")
    if pulse.get("success") and pulse.get("type") == "hourly":
        parts.append(f"- **Predicted peak hour:** {pulse['peak_hour']:02d}:00")
    if oof.get("available"):
        parts.append(f"- **OOF observed:** {oof['observed']:,.0f}")
        parts.append(f"- **OOF predicted:** {oof['predicted']:,.0f}")
        parts.append(f"- **Residual (obs-pred):** {oof['diff']:+,.0f}")
        if oof.get("interpretation"): parts.append(f"  *{oof['interpretation']}*")
    if shap.get("success") and shap.get("records"):
        t = shap["records"][0]
        parts.append(f"- **Top SHAP:** {t['label']} ({t['direction']})")
    parts.append("\\n*SHAP describes model behaviour; not causation.*")
    return "\\n".join(parts)

def run_workflow(query, grid_id, bundle, gdf, table, whatif_changes=None):
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
    state.update(planner_agent(state))
    if state.get("final_answer"): return state
    # Resolve row for downstream agents
    row = None
    if grid_id:
        row, _ = get_grid_row(gdf, str(grid_id))
        if row is not None:
            feats = get_features(bundle)
            feat_vals = get_feature_values(row, feats)
            # Override feature names with bundle features (authoritative)
            from data_service import validate_feature_values
            is_valid, missing = validate_feature_values(feat_vals)
    # Agent 2: Data
    state.update(data_agent(state, gdf, table))
    if row is not None:
        feats = get_features(bundle)
        feat_vals = get_feature_values(row, feats)
        from data_service import validate_feature_values
        is_valid, missing = validate_feature_values(feat_vals)
        state["data_result"]["feature_values"] = feat_vals
        state["data_result"]["features_valid"] = is_valid
        state["data_result"]["missing_features"] = missing
    # Agent 3: Prediction
    try: plan_data = json.loads(state.get("plan","{}"))
    except: plan_data = {}
    intent = state.get("intent","combined")
    needs_pred = plan_data.get("needs_prediction", True) or intent in ("prediction","explanation","combined","whatif","pulse")
    if needs_pred and row is not None:
        state.update(prediction_agent(state, bundle, row))
    # Agent 4: Explanation
    needs_shap = plan_data.get("needs_explanation", False) or intent in ("explanation","combined")
    needs_wi = bool(whatif_changes) or intent == "whatif"
    if (needs_shap or needs_wi) and row is not None:
        state.update(explanation_agent(state, bundle, row, whatif_changes if needs_wi else None))
    # Agent 5: Review
    state.update(review_agent(state))
    # Compose
    state.update(compose_response(state))
    return state
""")
aw.close()
print("Written: agent_workflow.py")
