from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any
from project_paths import PROJECT_ROOT, ELIGIBLE_CSV, ELIGIBLE_GPKG, GRID_SHAPEFILE
import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = PROJECT_ROOT
CSV_PATH = ELIGIBLE_CSV
GPKG_PATH = ELIGIBLE_GPKG
SHP_PATH = GRID_SHAPEFILE

OOF_OBSERVED_COL  = "oof_observed_total"
OOF_PREDICTED_COL = "oof_rf_predicted_total"
OOF_DIFF_COL      = "oof_observed_minus_predicted"
OOF_ABS_COL       = "oof_absolute_error"
OOF_FOLD_COL      = "oof_fold"

FEATURE_LABELS = {
    "LConn_wt":   "Local road-network connectivity",
    "MAD500_wt":  "Street-network reach (500 m)",
    "BtA500_wt":  "Street-network betweenness (500 m)",
    "Floor_Inte": "Building floor-area intensity",
    "POI_Densit": "POI density",
    "Transit_Ac": "Transit accessibility",
    "IntDen_km2": "Intersection density per sq km",
    "land_use_e": "Land-use diversity",
}

FEATURE_DESCRIPTIONS = {
    "LConn_wt":   "How well local streets are connected within the grid.",
    "MAD500_wt":  "How many destinations reachable within 500 m walk.",
    "BtA500_wt":  "How often streets appear on shortest paths within 500 m.",
    "Floor_Inte": "Total floor area divided by grid area.",
    "POI_Densit": "Points of interest per unit area.",
    "Transit_Ac": "Proximity and frequency of transit stops.",
    "IntDen_km2": "Street intersections per sq km.",
    "land_use_e": "Shannon entropy of land-use types.",
}


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]
    return value


def _find_geometry_file():
    if GPKG_PATH.exists():
        return GPKG_PATH
    if SHP_PATH.exists():
        return SHP_PATH
    return None


def _merge_attributes(gdf, attrs):
    merged = gdf.merge(attrs, on="grid_id", how="left", suffixes=("", "_csv"))
    for col in list(merged.columns):
        if col.endswith("_csv"):
            original = col[:-4]
            if original in merged.columns:
                merged[original] = merged[col].combine_first(merged[original])
            else:
                merged[original] = merged[col]
    merged.drop(columns=[c for c in merged.columns if c.endswith("_csv")], inplace=True, errors="ignore")
    return merged


def _ensure_oof_diff(gdf):
    if OOF_DIFF_COL not in gdf.columns:
        if OOF_OBSERVED_COL in gdf.columns and OOF_PREDICTED_COL in gdf.columns:
            gdf[OOF_DIFF_COL] = (
                pd.to_numeric(gdf[OOF_OBSERVED_COL], errors="coerce")
                - pd.to_numeric(gdf[OOF_PREDICTED_COL], errors="coerce")
            )


def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Eligible grid CSV not found: {CSV_PATH}")
    geom_path = _find_geometry_file()
    if geom_path is None:
        raise FileNotFoundError(f"Grid geometry not found. Checked: {GPKG_PATH} and {SHP_PATH}")
    table = pd.read_csv(CSV_PATH, low_memory=False)
    if "grid_id" not in table.columns:
        raise ValueError("CSV must contain grid_id.")
    table["grid_id"] = table["grid_id"].map(normalize_id)
    table = table.drop_duplicates("grid_id")
    grids = gpd.read_file(geom_path)
    if "grid_id" not in grids.columns:
        if "id" in grids.columns:
            grids = grids.rename(columns={"id": "grid_id"})
        else:
            raise ValueError("Grid geometry must contain grid_id or id.")
    grids["grid_id"] = grids["grid_id"].map(normalize_id)
    if grids.crs is None:
        raise ValueError("Grid geometry file has no CRS.")
    eligible_ids = set(table["grid_id"])
    grids = grids[grids["grid_id"].isin(eligible_ids)].copy()
    if grids.empty:
        raise ValueError("No matching grid IDs between CSV and geometry.")
    grids_geo  = grids.to_crs("EPSG:4326")
    grids_proj = grids.to_crs("EPSG:32644")
    attrs = table.drop(columns=["geometry"], errors="ignore")
    grids_geo  = _merge_attributes(grids_geo, attrs)
    grids_proj = _merge_attributes(grids_proj, attrs)
    _ensure_oof_diff(grids_geo)
    _ensure_oof_diff(grids_proj)
    return table, grids_geo, grids_proj, geom_path


def get_grid_row(gdf, grid_id):
    gid = normalize_id(grid_id)
    if not gid:
        return None, "Grid ID is empty or invalid."
    matches = gdf[gdf["grid_id"].astype(str) == gid]
    if matches.empty:
        return None, f"Grid ID {gid!r} is not in the eligible grid dataset."
    return matches.iloc[0], None


def get_location_details(row):
    centroid = row.geometry.centroid
    district = row.get("district_name", row.get("district", "Not recorded"))
    if pd.isna(district):
        district = "Not recorded"
    return {"id": normalize_id(row.get("grid_id")), "district": str(district), "lat": float(centroid.y), "lon": float(centroid.x)}


def get_feature_values(row, features):
    result = {}
    for f in features:
        raw = row.get(f, np.nan)
        try:
            v = float(raw)
            result[f] = None if (pd.isna(v) or not np.isfinite(v)) else v
        except (TypeError, ValueError):
            result[f] = None
    return result


def validate_feature_values(values):
    missing = [k for k, v in values.items() if v is None]
    return len(missing) == 0, missing


def get_oof_result(row):
    def _f(col):
        try:
            v = row.get(col, np.nan)
            f = float(v)
            return f if np.isfinite(f) else None
        except (TypeError, ValueError):
            return None
    observed  = _f(OOF_OBSERVED_COL)
    predicted = _f(OOF_PREDICTED_COL)
    diff      = _f(OOF_DIFF_COL)
    abs_error = _f(OOF_ABS_COL)
    fold      = _f(OOF_FOLD_COL)
    if diff is None and observed is not None and predicted is not None:
        diff = observed - predicted
    if abs_error is None and diff is not None:
        abs_error = abs(diff)
    return {
        "available": observed is not None and predicted is not None,
        "observed": observed, "predicted": predicted,
        "diff": diff, "abs_error": abs_error,
        "fold": int(fold) if fold is not None else None,
    }


def find_clicked_grid(gdf, lat, lon):
    from shapely.geometry import Point
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").iloc[0]
    matches = gdf[gdf.geometry.intersects(point)]
    if not matches.empty:
        return normalize_id(matches.iloc[0]["grid_id"])
    proj = gdf.to_crs("EPSG:32644")
    pt_proj = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:32644").iloc[0]
    distances = proj.geometry.distance(pt_proj)
    nearest_idx = distances.idxmin()
    if distances.loc[nearest_idx] <= 80:
        return normalize_id(gdf.loc[nearest_idx, "grid_id"])
    return None


def compute_study_wide_metrics(gdf):
    diff = pd.to_numeric(gdf.get(OOF_DIFF_COL, pd.Series(dtype=float)), errors="coerce")
    obs  = pd.to_numeric(gdf.get(OOF_OBSERVED_COL, pd.Series(dtype=float)), errors="coerce")
    pred = pd.to_numeric(gdf.get(OOF_PREDICTED_COL, pd.Series(dtype=float)), errors="coerce")
    valid = diff.dropna()
    if valid.empty:
        return {"n_total": len(gdf), "n_with_oof": 0, "n_positive": 0, "n_negative": 0, "n_zero": 0, "mean_diff": None, "mean_obs": None, "mean_pred": None, "top_under": [], "top_over": []}
    cols_avail = [c for c in ["grid_id", OOF_OBSERVED_COL, OOF_PREDICTED_COL, OOF_DIFF_COL] if c in gdf.columns]
    summary = gdf[cols_avail].copy()
    summary[OOF_DIFF_COL] = valid.values
    summary = summary.dropna(subset=[OOF_DIFF_COL])
    return {
        "n_total": len(gdf), "n_with_oof": int(valid.notna().sum()),
        "n_positive": int((valid > 0).sum()), "n_negative": int((valid < 0).sum()), "n_zero": int((valid == 0).sum()),
        "mean_diff": float(valid.mean()),
        "mean_obs": float(obs.dropna().mean()) if not obs.dropna().empty else None,
        "mean_pred": float(pred.dropna().mean()) if not pred.dropna().empty else None,
        "top_under": summary.nlargest(5, OOF_DIFF_COL).to_dict("records"),
        "top_over":  summary.nsmallest(5, OOF_DIFF_COL).to_dict("records"),
    }
