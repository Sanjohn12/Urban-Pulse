"""Thin host controller for the Streamlit investigation panel."""
from __future__ import annotations

import pandas as pd
import time

from agent_workflow import create_investigation
from project_paths import DATASET_PATH


def start_agent_investigation(grid_id, question, model_bundle):
    if not str(question).strip():
        raise ValueError("A research question is required")
    return create_investigation(
        str(grid_id), str(question).strip(), model_bundle=model_bundle, require_shap=True, require_theory=True,
    )


def run_agent_investigation(engine, max_steps=10):
    """Precompute bounded evidence, then reserve one real LLM call for the hypothesis."""
    diagnostics = []
    def stage(name, fn):
        started = time.monotonic()
        try:
            value = fn()
            diagnostics.append({"stage": name, "success": True,
                                "seconds": round(time.monotonic() - started, 2)})
            return value
        except Exception as exc:
            diagnostics.append({"stage": name, "success": False,
                                "seconds": round(time.monotonic() - started, 2),
                                "error": f"{type(exc).__name__}: {exc}"})
            raise
    status = engine.prompt_context()["required_evidence_status"]
    if not any(e.get("tool") == "detect_patterns" for e in engine.prompt_context()["evidence"].values()):
        stage("pattern_detection", lambda: engine.call_tool({"name": "detect_patterns"}))
    if not status["residual_collected"]:
        stage("residual_analysis", lambda: engine.call_tool({"name": "residual_analysis"}))
    status = engine.prompt_context()["required_evidence_status"]
    if not status["shap_collected"]:
        stage("shap_analysis", lambda: engine.call_tool({"name": "model_explanation"}))
    status = engine.prompt_context()["required_evidence_status"]
    if not status["usable_literature_abstract_collected"]:
        findings = [f for e in engine.prompt_context()["evidence"].values()
                    if e.get("tool") == "detect_patterns" for f in e["payload"].get("findings", [])]
        selected = next((f for f in findings if f['pattern'] in engine.state.query), None)
        query = (selected or findings[0])["literature_query"] if findings else "urban morphology land use activity transit"
        stage("openalex_literature", lambda: engine.call_tool({"name": "literature_search", "query": query}))
    state = stage("ollama_proposal", engine.advance)
    # One bounded self-repair turn is allowed only for a rejected structured
    # proposal. It receives the validator error plus exact citable excerpts.
    if state.proposal is None and state.planner_status == "invalid_response":
        state = stage("ollama_proposal_repair", engine.advance)
    diagnostics.append({"stage": "proposal_validation", "success": state.proposal is not None,
                        "planner_status": state.planner_status,
                        "errors": len(state.errors)})
    return {"state": state, "diagnostics": diagnostics}


def configure_exploratory_execution(engine):
    """Use the published spatial folds as exploratory evidence, never as an untouched holdout."""
    ids = pd.read_csv(DATASET_PATH, usecols=["grid_id"], dtype={"grid_id": str}).grid_id.tolist()
    return engine.configure_execution(
        ids, dataset_path=DATASET_PATH, environment="research",
        validation_mode="exploratory_spatial_cv", holdout_grid_ids=(),
        n_estimators=300, calibration_inner_splits=3, minimum_separation_m=2500.0,
    )


def metric_rows(state):
    evidence = (state.validation_results.execution_evidence if state.validation_results else None) or {}
    baseline = evidence.get("baseline_metrics")
    variants = [("M0 baseline", baseline), ("M1 statistical", evidence.get("m1_statistical_metrics")),
                ("M2 theory-guided", evidence.get("m2_theory_metrics")),
                ("M3 combined", evidence.get("m3_combined_metrics") or evidence.get("candidate_metrics"))]
    if not baseline:
        return []
    return [{"model": label, "target": target, "r2": metrics[target]["r2"],
             "rmse": metrics[target]["rmse"], "mae": metrics[target]["mae"]}
            for label, metrics in variants if metrics for target in ("daily", "hourly")]
