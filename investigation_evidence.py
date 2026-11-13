"""Read-only evidence adapters for the supervised investigation engine."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import anomaly_service
import data_service
import evaluation_service
from investigation_models import BaselineIdentity, utc_now
from project_paths import (
    DATASET_PATH, ELIGIBLE_CSV, MODEL_BUNDLE_PATH, PROJECT_ROOT, workspace_path,
)

BASELINE_METRICS = {
    "r2": 0.5155621220348304,
    "rmse": 29785.444397432864,
    "mae": 14143.19228997865,
}
STATIC_FEATURES = tuple(data_service.FEATURE_LABELS) + ("Bldg_Cov",)


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class InvestigationArtifacts:
    table: pd.DataFrame
    baseline: BaselineIdentity


def load_investigation_artifacts() -> InvestigationArtifacts:
    """Identify the immutable reference without fitting/loading a learned model."""
    metadata_path = workspace_path("final_rf_agent_models_metadata.json")
    comparison_path = workspace_path("spatial_cv_overall_model_comparison.csv")
    oof_path = workspace_path("spatial_cv_out_of_fold_predictions.csv")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(comparison_path)
    rows = comparison[(comparison.model == "Random Forest") & (comparison.target == "Total activity")]
    if len(rows) != 1 or int(rows.iloc[0].n_observations) != 7764:
        raise ValueError("Reference baseline comparison/cohort is not the audited baseline")
    for key, expected in BASELINE_METRICS.items():
        column = {"r2": "R2", "rmse": "RMSE", "mae": "MAE"}[key]
        if not math.isclose(float(rows.iloc[0][column]), expected, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError(f"Reference baseline {key} differs from the audited result")
    primary = pd.read_csv(DATASET_PATH)
    table = pd.read_csv(ELIGIBLE_CSV)
    for frame in (primary, table):
        frame["grid_id"] = frame.grid_id.map(data_service.normalize_id)
        if frame.grid_id.duplicated().any() or frame.grid_id.eq("").any():
            raise ValueError("Missing/duplicate grid IDs in investigation artifacts")
    if len(primary) != 7764 or set(primary.grid_id) != set(table.grid_id):
        raise ValueError("Investigation inputs do not match the audited cohort")
    if metadata["features"] != list(data_service.FEATURE_LABELS):
        raise ValueError("Reference model feature order changed")
    matched = table.merge(
        primary[["grid_id", "fold", "total_acti", "spatial_block"]],
        on="grid_id", validate="one_to_one", suffixes=("", "_primary"),
    )
    if not np.array_equal(matched.oof_fold, matched.fold_primary):
        raise ValueError("Eligible OOF folds do not match the primary dataset")
    if not np.array_equal(matched.oof_observed_total, matched.total_acti_primary):
        raise ValueError("Eligible observations do not match the primary dataset")
    if primary.groupby("spatial_block").fold.nunique().gt(1).any():
        raise ValueError("A spatial block crosses folds")
    measured = evaluation_service.recalculate_pooled_metrics(matched)
    for key, expected in BASELINE_METRICS.items():
        if not math.isclose(measured[key], expected, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError(f"Eligible OOF {key} does not match the reference baseline")
    model_hash = file_digest(MODEL_BUNDLE_PATH)
    if model_hash != "f39bb7798852b5215a9f97f5655b28021f226a6d8fbcc8c1d2ff7d86f2664404":
        raise ValueError("Model bundle is not the audited immutable reference")
    provenance = {
        "model": str(MODEL_BUNDLE_PATH.relative_to(PROJECT_ROOT)),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "eligible": str(ELIGIBLE_CSV.relative_to(PROJECT_ROOT)),
        "oof": str(oof_path.relative_to(PROJECT_ROOT)),
        "metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        "comparison_sha256": file_digest(comparison_path),
        "sample_role": "exploratory; previously inspected OOF; no untouched holdout certified",
    }
    baseline = BaselineIdentity(
        model_id="RF_REFERENCE_SPATIAL_OOF_V1", model_sha256=model_hash,
        dataset_sha256=file_digest(DATASET_PATH), eligible_sha256=file_digest(ELIGIBLE_CSV),
        oof_sha256=file_digest(oof_path), metadata_sha256=file_digest(metadata_path),
        cohort_sha256=digest(sorted(primary.grid_id.tolist())),
        fold_manifest_sha256=digest(
            primary.sort_values("grid_id")[["grid_id", "fold", "spatial_block"]].to_dict("records")
        ),
        n_cells=len(primary), features=metadata["features"],
        model_parameters=metadata["parameters"], reference_metrics=BASELINE_METRICS,
        provenance=provenance,
    )
    return InvestigationArtifacts(matched, baseline)


def select_row(table, grid_id):
    selected = table[table.grid_id == grid_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one eligible row for grid {grid_id}")
    return selected.iloc[0]


def residual_evidence(table, grid_id) -> dict:
    """Use existing evaluators, but verify the stored residual before exposing it."""
    row = select_row(table, grid_id)
    daily = evaluation_service.get_oof_for_grid(table, grid_id)
    if not daily["available"]:
        raise ValueError("Paired daily OOF observations/predictions are missing")
    observed, predicted = daily["observed"], daily["predicted"]
    if min(observed, predicted) < 0:
        raise ValueError("Daily activity must be nonnegative")
    if daily["fold"] is None:
        raise ValueError("Spatial fold is missing")
    expected = observed - predicted
    if daily["diff"] is None or not math.isclose(daily["diff"], expected, abs_tol=1e-6):
        raise ValueError("Stored daily residual is inconsistent")
    daily["diff"] = expected
    daily["abs_error"] = abs(expected)
    hourly_columns = evaluation_service.HOURLY_OBS_COLS + evaluation_service.HOURLY_RF_OOF_COLS
    hourly_valid = all(
        col in row and pd.notna(row[col]) and math.isfinite(float(row[col])) and float(row[col]) >= 0
        for col in hourly_columns
    )
    hourly_valid = hourly_valid and math.isclose(
        sum(float(row[col]) for col in evaluation_service.HOURLY_OBS_COLS), observed,
        rel_tol=1e-5, abs_tol=1e-3,
    )
    hourly = evaluation_service.get_hourly_oof_for_grid(row) if hourly_valid else {
        "available": False, "error": "Hourly data are missing, invalid, or inconsistent with the daily observation",
    }
    # Scope first: never run the group diagnostic on withheld rows.
    profile = anomaly_service.profile_spatial_anomalies(table)
    if not profile.get("success"):
        raise ValueError(profile.get("error", "Anomaly diagnostic failed"))
    # Do not expose other locations or canned causal stories as observed facts.
    diagnostic = {
        "total_evaluated_grids": profile["total_evaluated_grids"],
        "overpredicted_count": profile["overpredicted_count"],
        "underpredicted_count": profile["underpredicted_count"],
        "group_definition": "relative residual thresholds; not a statistical spatial cluster test",
    }
    return {
        "daily": daily, "hourly": hourly, "scoped_anomaly_counts": diagnostic,
        "residual_convention": "observed_minus_predicted",
        "scope": "explored_saved_spatial_oof",
        "independent_validation": False,
    }


def location_evidence(table, grid_id) -> dict:
    row = select_row(table, grid_id)
    features = {}
    for column in STATIC_FEATURES:
        if column in row and pd.notna(row[column]) and math.isfinite(float(row[column])):
            features[column] = float(row[column])
    return {
        "grid_id": grid_id, "district": str(row.get("district_name", "not recorded")),
        "features": features,
        "missing_features": [name for name in STATIC_FEATURES if name not in features],
        "source": "eligible attribute table; not a field survey or causal diagnosis",
    }


def literature_records(papers: list[dict], query: str) -> dict[str, dict]:
    """Retain provider metadata verbatim; do not turn metadata into verified theory."""
    from urllib.parse import urlparse

    records = {}
    for paper in papers:
        url = paper.get("doi", "")
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {"doi.org", "openalex.org"}:
            continue
        if not parsed.path.strip("/") or not paper.get("title"):
            continue
        source_id = "L-" + digest({"url": url, "title": paper["title"]})[:20]
        records[source_id] = {
            "source_id": source_id, "provider": "OpenAlex",
            "query": query, "retrieved_at": utc_now(), "url": url,
            "title": paper["title"], "authors": paper.get("authors"),
            "year": paper.get("year"), "abstract": paper.get("abstract", ""),
            "metadata_sha256": digest(paper), "metadata": paper,
            "verification": "provider_metadata_only; relevance and bibliographic accuracy need human review",
        }
    return records
