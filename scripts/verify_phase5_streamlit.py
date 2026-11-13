"""Opt-in live Phase 5 verification through Streamlit widgets.

This executes a real exploratory experiment and appends a deferred investigation
record. It never accepts or deploys a model. Run only with --execute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from project_paths import MODEL_BUNDLE_PATH, REGISTRY_PATH


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def element(elements, label):
    matches = [item for item in elements if item.label == label]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {label!r} widget, found {len(matches)}")
    return matches[0]


def run(at, timeout=180):
    at.run(timeout=timeout)
    if at.exception:
        raise RuntimeError("Streamlit exception: " + "; ".join(str(item.value) for item in at.exception))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="authorize the real exploratory run and memory append")
    parser.add_argument("--grid-id", default="1")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing live execution without --execute")

    memory_before = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    baseline_hash = sha256(MODEL_BUNDLE_PATH)
    at = AppTest.from_file(str(ROOT / "app.py"))
    run(at)
    element(at.selectbox, "Grid ID (optional)").set_value(args.grid_id)
    element(at.button, "Select this grid").click()
    run(at)
    element(at.radio, "Map view").set_value("🔬 Scientific Discovery & Evolution")
    run(at)
    for _ in range(3):
        start = element(at.button, "Start Ollama investigation")
        if not start.disabled:
            break
        run(at)
    else:
        raise RuntimeError("Start Ollama investigation remained disabled after availability retries")
    start.click()
    run(at, timeout=600)
    engine = at.session_state["investigation_engine"]
    for _ in range(3):
        if engine.state.proposal is not None:
            break
        element(at.button, "Continue agent investigation").click()
        run(at, timeout=600)
        engine = at.session_state["investigation_engine"]
    state = engine.state
    required_tools = {"residual_analysis", "model_explanation", "literature_search"}
    successful_tools = {item["tool"]["name"] for item in state.tool_trace if item.get("success")}
    if state.proposal is None:
        calls = [item["tool"]["name"] for item in state.tool_trace]
        raise RuntimeError(
            f"Agent did not produce a proposal; calls={calls}; sources={len(state.literature_sources)}; "
            f"daily={state.daily_residual_evidence is not None}; errors={state.errors}"
        )
    if not required_tools <= successful_tools:
        raise RuntimeError(f"Missing required real evidence tools: {required_tools - successful_tools}")
    proposal_summary = {
        "title": state.proposal.title, "method": state.proposal.method,
        "feature_names": [item.name for item in state.proposal.features],
        "literature_source_ids": [item.source_id for item in state.proposal.literature],
        "successful_tools": sorted(successful_tools),
    }

    element(at.button, "Prepare reproducible exploratory experiment").click()
    run(at, timeout=300)
    engine = at.session_state["investigation_engine"]
    plan = engine.state.execution_plan
    if plan is None or plan.validation_mode != "exploratory_spatial_cv" or plan.holdout_grid_ids:
        raise RuntimeError("UI did not prepare the required exploratory-only plan")

    element(at.button, "Request human experiment approval").click()
    run(at)
    request_id = at.session_state["investigation_request_id"]
    payload = engine.approval_payload(request_id)
    element(at.text_input, "Authenticated reviewer identity").set_value(
        "User-authorized Phase 5 verification"
    )
    element(at.text_area, "Approval rationale").set_value(
        "The user explicitly requested end-to-end verification of this exact exploratory workflow; no acceptance or deployment is authorized."
    )
    element(at.checkbox, "I approve this exact exploratory plan and understand it is not independent holdout evidence.").check()
    element(at.button, "Approve experiment").click()
    run(at)
    engine = at.session_state["investigation_engine"]
    if engine.state.status.value != "approved":
        raise RuntimeError("Exact experiment request was not approved")

    element(at.button, "Execute approved Python experiment").click()
    run(at, timeout=3600)
    engine = at.session_state["investigation_engine"]
    state = engine.state
    if state.status.value != "evaluated" or state.validation_results is None:
        raise RuntimeError(f"Execution did not reach evaluated: {state.status}, {state.errors}")
    evidence = state.validation_results.execution_evidence

    element(at.radio, "Human scientific decision").set_value("deferred")
    element(at.text_area, "Result-review rationale").set_value(
        "Verification run only. Results are exploratory because the hypothesis used previously inspected OOF evidence; independent confirmation remains unavailable."
    )
    element(at.text_input, "Result reviewer identity").set_value(
        "User-authorized Phase 5 verification"
    )
    element(at.button, "Finalize scientific decision").click()
    run(at)
    element(at.button, "Record outcome in scientific memory").click()
    run(at)
    engine = at.session_state["investigation_engine"]
    final = engine.state
    if final.status.value != "recorded" or final.review_status != "deferred":
        raise RuntimeError("Reviewed result was not recorded as deferred")

    memory_after = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    report = {
        "streamlit_exceptions": 0,
        "grid_id": args.grid_id,
        "investigation_id": final.investigation_id,
        "record_id": final.scientific_memory_reference["record_id"],
        "proposal": proposal_summary,
        "literature_sources_retrieved": len(final.literature_sources),
        "execution_id": final.execution_plan.execution_id,
        "validation_mode": final.execution_plan.validation_mode,
        "holdout_cells": len(final.execution_plan.holdout_grid_ids),
        "development_cells": len(final.execution_plan.development_grid_ids),
        "baseline_metrics": evidence["baseline_metrics"],
        "candidate_metrics": evidence["candidate_metrics"],
        "daily_deltas": evidence["daily_deltas"],
        "acceptance_recommendation": evidence["acceptance"],
        "human_decision": final.review_status,
        "models_saved": evidence["models_saved"],
        "deployed": evidence["deployed"],
        "memory_investigations_before": len(memory_before.get("investigations", [])),
        "memory_investigations_after": len(memory_after.get("investigations", [])),
        "legacy_experiments_unchanged": memory_before.get("experiments") == memory_after.get("experiments"),
        "baseline_model_hash_unchanged": baseline_hash == sha256(MODEL_BUNDLE_PATH),
        "approval_binding_digest": payload["request"]["binding_digest"],
    }
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
