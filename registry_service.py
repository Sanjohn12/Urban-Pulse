"""
registry_service.py
-------------------
Step 4 of Theory-Guided Self-Evolving Urban Intelligence (TSE-UI).

Maintains a persistent scientific memory registry of tested urban theories,
formulated hypotheses, experimental evaluation logs, and model versions.
"""

from __future__ import annotations
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any, List
from datetime import date, datetime
from project_paths import REGISTRY_PATH

logger = logging.getLogger(__name__)

REGISTRY_FILE = REGISTRY_PATH
_REGISTRY_LOCK = threading.RLock()


def _json_default(value):
    """Support scientific scalar/array values without silently dropping fields."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot persist {type(value).__name__}; provide a JSON feature recipe, not executable code")


def _json_snapshot(value):
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def _validate_memory(memory):
    if not isinstance(memory, dict) or not isinstance(memory.get("experiments"), list):
        raise ValueError("Scientific memory must be an object with an experiments list")
    if not all(isinstance(entry, dict) for entry in memory["experiments"]):
        raise ValueError("Each scientific memory experiment must be an object")

def load_scientific_memory() -> Dict[str, Any]:
    """Read memory without creating files or falling back to another checkout."""
    if not REGISTRY_FILE.exists():
        initial_data = {
            "project": "Theory-Guided Self-Evolving Urban Intelligence (TSE-UI)",
            "study_area": "Colombo and Gampaha, Western Province, Sri Lanka",
            "created_at": datetime.now().isoformat(),
            "experiments_count": 0,
            "hypotheses_accepted": 0,
            "hypotheses_rejected": 0,
            "experiments": [],
        }
        return initial_data

    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
        _validate_memory(memory)
        return memory
    except Exception as exc:
        raise ValueError(f"Cannot read scientific memory at {REGISTRY_FILE}: {exc}") from exc

def save_scientific_memory(data: Dict[str, Any]) -> bool:
    """Atomically replace valid memory; leave the previous file intact on failure."""
    temporary = None
    try:
        _validate_memory(data)
        payload = json.dumps(data, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False)
        with _REGISTRY_LOCK:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=REGISTRY_FILE.parent,
                prefix=f".{REGISTRY_FILE.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, REGISTRY_FILE)
        return True
    except Exception as exc:
        logger.error("Failed to write scientific memory: %s", exc)
        return False
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def record_experiment_result(experiment_result: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve the full result and legacy UI aliases, never report a failed save as success."""
    if not isinstance(experiment_result, dict) or experiment_result.get("success") is False:
        raise ValueError("Only completed experiment result dictionaries can be recorded")
    result = _json_snapshot(experiment_result)
    verdict = result.get("verdict", {})
    if not isinstance(verdict, dict):
        raise ValueError("Experiment verdict must contain status and rationale fields")
    with _REGISTRY_LOCK:
        memory = load_scientific_memory()
        used_numbers = [
            int(match.group(1))
            for item in memory["experiments"]
            if (match := re.fullmatch(r"EXP-(\d+)", str(item.get("id", ""))))
        ]
        entry = dict(result)
        entry.update({
            "id": f"EXP-{max(used_numbers, default=0) + 1:03d}",
            "timestamp": datetime.now().isoformat(),
            "learning_stage": result.get("learning_stage", "Theoretical Learning"),
            "method": result.get("method", "Theoretical Interaction Feature"),
            "theory": result.get("theoretical_grounding", result.get("theory")),
            "verdict": verdict.get("status", "UNKNOWN"),
            "verdict_details": verdict,
            "rationale": verdict.get("scientific_rationale", ""),
            "baseline_r2": result.get("baseline_metrics", {}).get("r2"),
            "evolved_r2": result.get("evolved_metrics", {}).get("r2"),
            "delta_r2": result.get("deltas", {}).get("delta_r2"),
            "delta_rmse": result.get("deltas", {}).get("delta_rmse"),
            "delta_mae": result.get("deltas", {}).get("delta_mae"),
            "delta_bias": result.get("deltas", {}).get("delta_bias"),
            # Complete canonical evidence also preserves collisions with legacy aliases.
            "experiment_result": result,
        })
        entry.setdefault("feature_definition", {
            "name": result.get("feature_name"),
            "formula": result.get("formula"),
            "theory": result.get("theoretical_grounding", result.get("theory")),
            "hypothesis": result.get("hypothesis"),
        })
        for key, default in [
            ("anomaly_resolution", {}), ("fold_metrics", []),
            ("quartile_metrics", {}), ("district_metrics", {}),
        ]:
            entry.setdefault(key, default)
        memory["experiments"].append(entry)
        memory["experiments_count"] = len(memory["experiments"])
        memory["hypotheses_accepted"] = sum(e.get("verdict") == "ACCEPTED" for e in memory["experiments"])
        memory["hypotheses_rejected"] = sum(e.get("verdict") == "REJECTED" for e in memory["experiments"])
        if not save_scientific_memory(memory):
            raise OSError(f"Experiment was not saved to {REGISTRY_FILE}")
        return entry

def get_tested_feature_names() -> dict:
    """Returns a dict of {feature_name: [verdicts]} from scientific memory.
    
    Used by the Step 3 UI to block re-testing of already-evaluated features,
    implementing the 'recorded evidence informs next investigation cycle' 
    feedback loop from the TSE-UI framework diagram.
    """
    memory = load_scientific_memory()
    result = {}
    for exp in memory.get("experiments", []):
        name = exp.get("feature_name")
        verdict = exp.get("verdict", "UNKNOWN")
        if name:
            result.setdefault(name, []).append(verdict)
    return result


def record_investigation_result(snapshot: dict, *, record_kind: str) -> dict:
    """Append a typed investigation revision without altering legacy EXP records.

    This human/host-only API is never registered as a planner tool. Existing
    experiment counters/champion selection deliberately ignore investigations.
    """
    from investigation_models import InvestigationState, InvestigationStatus, utc_now
    from investigation_evidence import digest

    state = InvestigationState.model_validate(snapshot)
    if state.proposal is None or state.experiment_configuration is None:
        raise ValueError("A validated proposal is required for an investigation record")
    if record_kind == "proposal_checkpoint":
        if state.status not in {
            InvestigationStatus.PROPOSED, InvestigationStatus.AWAITING_APPROVAL, InvestigationStatus.APPROVED,
        } or state.validation_results is not None:
            raise ValueError("Proposal checkpoints cannot claim evaluated outcomes")
    elif record_kind == "reviewed_outcome":
        if (state.status != InvestigationStatus.RECORDED or state.validation_results is None
                or state.review_status not in {"accepted", "rejected", "deferred"} or not state.human_reviews):
            raise ValueError("A human-reviewed experiment outcome is required")
    else:
        raise ValueError("Unknown investigation record kind")
    with _REGISTRY_LOCK:
        memory = load_scientific_memory()
        records = memory.get("investigations", [])
        if not isinstance(records, list):
            raise ValueError("Scientific memory investigations must be a list")
        revision = 1 + sum(record.get("investigation_id") == state.investigation_id for record in records)
        record_id = f"{state.investigation_id}-R{revision:04d}"
        reference = {
            "registry_path": str(REGISTRY_FILE), "investigation_id": state.investigation_id,
            "record_id": record_id,
        }
        payload = state.model_dump(mode="json")
        payload["scientific_memory_reference"] = reference
        records = records + [{
            "id": record_id, "investigation_id": state.investigation_id,
            "record_kind": record_kind, "recorded_at": utc_now(),
            "payload_sha256": digest(payload), "investigation": payload,
        }]
        memory["investigations"] = records
        memory["investigations_count"] = len(records)
        if not save_scientific_memory(memory):
            raise OSError(f"Investigation was not saved to {REGISTRY_FILE}")
        return reference


def get_champion_model_state() -> Dict[str, Any]:
    """Resolves the current champion model from accepted experiments in scientific memory.
    
    Preserves:
      - Original Reference Baseline (immutable 8 features, R²=0.5156)
      - Current Champion Model (e.g. Stage 3 EXP-010, R²=0.5181)
      - Cumulative feature set for progressive learning in Stage 4
    """
    base_features = [
        "LConn_wt",
        "MAD500_wt",
        "BtA500_wt",
        "Floor_Inte",
        "POI_Densit",
        "Transit_Ac",
        "IntDen_km2",
        "land_use_e",
    ]
    base_metrics = {
        "r2": 0.5156,
        "rmse": 29785.44,
        "mae": 14143.19,
        "mean_bias": -5871.73,
    }

    memory = load_scientific_memory()
    experiments = memory.get("experiments", [])

    # Find the most recent accepted experiment that improved the model
    accepted_exps = [e for e in experiments if e.get("verdict") == "ACCEPTED"]

    if not accepted_exps:
        return {
            "has_champion": False,
            "champion_id": "BASELINE",
            "stage": "Baseline Learning",
            "features": list(base_features),
            "baseline_metrics": base_metrics,
            "champion_metrics": base_metrics,
            "accepted_features": [],
        }

    # Latest accepted experiment (e.g., EXP-010)
    latest_acc = accepted_exps[-1]
    accepted_feat = latest_acc.get("feature_name")
    
    champion_features = list(base_features)
    accepted_features = []
    if accepted_feat and accepted_feat not in champion_features:
        champion_features.append(accepted_feat)
        accepted_features.append(accepted_feat)

    champ_r2 = latest_acc.get("evolved_r2", base_metrics["r2"])
    
    return {
        "has_champion": True,
        "champion_id": latest_acc.get("id", "UNKNOWN"),
        "stage": latest_acc.get("learning_stage", "Theoretical Learning"),
        "features": champion_features,
        "accepted_features": accepted_features,
        "baseline_metrics": base_metrics,
        "champion_metrics": {
            "r2": champ_r2,
            "delta_r2": latest_acc.get("delta_r2", 0.0),
            "delta_rmse": latest_acc.get("delta_rmse", 0.0),
        },
        "latest_accepted_experiment": latest_acc,
    }

