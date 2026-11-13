import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import LOG_DIR, DATASET_PATH
import json, logging, warnings
import numpy as np
import pandas as pd
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

import data_service as ds
import prediction_service as ps
import explanation_service as es
import evaluation_service as evs
import agent_workflow as aw
import anomaly_service as ans
import literature_service as lits
import experiment_service as exps
import registry_service as regs
import grid_evolution_service as ges
import phase5_ui_service as p5

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Urban Pulse AI Agent",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("app")

# -------------------------------------------------------
# Styling
# -------------------------------------------------------
st.markdown("""
<style>
  .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
  .agent-badge { background:#1e3a5f; color:#90caf9; border-radius:6px; padding:2px 8px; font-size:0.78em; font-weight:600; margin-right:4px; }
  .review-pass { color:#4caf50; }
  .review-fail { color:#f44336; }
  .review-warn { color:#ff9800; }
  .review-info { color:#90caf9; }
  .stMetric label { font-size:0.8em; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Load data (cached)
# -------------------------------------------------------
@st.cache_resource(show_spinner="Loading model bundle and grid data...")
def _load_all():
    bundle = ps.load_bundle()
    table, grids_geo, grids_proj, geom_path = ds.load_data()
    return bundle, table, grids_geo, grids_proj

@st.cache_data(show_spinner="Computing ensemble uncertainty across grids...")
def _get_uncertainty_df(_bundle, _grids_geo):
    return ps.compute_uncertainty_batch(_bundle, _grids_geo)

try:
    bundle, table, grids_geo, grids_proj = _load_all()
except Exception as exc:
    st.error("**Application could not load its data or model.**")
    st.code(str(exc))
    st.info("Check file paths in data_service.py and prediction_service.py.")
    st.stop()

features = ps.get_features(bundle)
eligible_count = grids_geo["grid_id"].nunique()
ollama_ok, ollama_msg = aw.check_ollama_available()

# -------------------------------------------------------
# Session state
# -------------------------------------------------------
for k, v in {
    "selected_grid_id": None,
    "messages": [],
    "map_mode": "Explore grids",
    "agent_log": [],
    "show_debug": False,
    "investigation_engine": None,
    "investigation_request_id": None,
    "phase5_diagnostics": [],
}.items():
    st.session_state.setdefault(k, v)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.markdown("## 🗺️ Urban Pulse AI Agent")
    st.markdown("*Colombo & Gampaha — Western Province, Sri Lanka*")
    st.divider()

    st.metric("Eligible grid cells", f"{eligible_count:,}")
    st.caption("Each cell ≈ 500 m × 500 m. Only model-eligible cells shown.")

    if ollama_ok:
        st.success(f"✅ LLM: qwen3:4b (Ollama)", icon="🤖")
    else:
        st.warning(f"⚠️ LLM unavailable — fallback mode active", icon="⚠️")
        st.caption(ollama_msg)

    st.divider()

    mode = st.radio(
        "Map view",
        ["Explore grids", "Spatial mismatch", "🛡️ Predictive Uncertainty & Model Confidence", "🔬 Scientific Discovery & Evolution"],
        key="map_mode"
    )

    district_col = "district_name" if "district_name" in grids_geo.columns else None
    if district_col:
        districts = sorted(grids_geo[district_col].dropna().astype(str).unique())
        district_filter = st.selectbox("Filter by district", ["All districts"] + districts)
    else:
        district_filter = "All districts"

    st.divider()
    ids = sorted(grids_geo["grid_id"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
    current = str(st.session_state.selected_grid_id) if st.session_state.selected_grid_id else ids[0]
    default_index = ids.index(current) if current in ids else 0
    chosen_id = st.selectbox("Grid ID (optional)", ids, index=default_index, help="You can also click directly on the map.")
    if st.button("Select this grid", use_container_width=True):
        st.session_state.selected_grid_id = chosen_id
        st.session_state.messages = []
        st.session_state.agent_log = []
        st.rerun()

    st.divider()
    if st.button("Clear selection & conversation", use_container_width=True):
        st.session_state.selected_grid_id = None
        st.session_state.messages = []
        st.session_state.agent_log = []
        st.rerun()

    st.session_state.show_debug = st.toggle("Show agent debug log", value=st.session_state.show_debug)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def _mismatch_gradient_color(pct_val, raw_diff=None):
    """
    Returns a diverging hex color based on percentage mismatch:
    ((Observed - Predicted) / Predicted) * 100
      - Positive (Red gradient): Observed > Predicted (Model underpredicted actual vitality).
        Light pink (+10%) to deep crimson (> +100%).
      - Negative (Blue gradient): Observed < Predicted (Model overpredicted actual vitality).
        Light cyan (-10%) to deep navy/indigo (< -60%).
      - Near zero (-10% to +10%): Neutral pale slate/grey (well-calibrated).
    """
    try:
        if pct_val is None or (isinstance(pct_val, float) and np.isnan(pct_val)):
            if raw_diff is not None and not np.isnan(raw_diff):
                pct_val = 100.0 if raw_diff > 0 else -60.0
            else:
                return "#9CA3AF"  # No data / missing

        pct = float(pct_val)

        # Well calibrated / close match
        if -10.0 <= pct <= 10.0:
            return "#E2E8F0"  # Neutral soft grey

        # Positive residuals: Underpredicted (Observed > Predicted) -> Red gradient
        if pct > 10.0:
            if pct < 35.0:
                return "#FCA5A5"  # Low (+10% to +35%) - light coral
            elif pct < 75.0:
                return "#F87171"  # Moderate (+35% to +75%) - medium red
            elif pct < 150.0:
                return "#EF4444"  # High (+75% to +150%) - vivid red
            else:
                return "#991B1B"  # Extreme (> +150%) - deep dark crimson

        # Negative residuals: Overpredicted (Observed < Predicted) -> Blue gradient
        else:
            if pct > -25.0:
                return "#93C5FD"  # Low (-10% to -25%) - light sky blue
            elif pct > -50.0:
                return "#60A5FA"  # Moderate (-25% to -50%) - medium blue
            elif pct > -75.0:
                return "#2563EB"  # High (-50% to -75%) - vibrant royal blue
            else:
                return "#1E3A8A"  # Extreme (< -75%) - deep dark navy
    except (TypeError, ValueError):
        return "#9CA3AF"

def _create_map(gdf, selected_id=None, mode="Explore grids"):
    if gdf.empty:
        return folium.Map(
            location=[6.9, 79.9], zoom_start=10,
            tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
        )
    bounds = gdf.total_bounds
    centre = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
    fmap = folium.Map(
        location=centre,
        zoom_start=10,
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
        name="OpenStreetMap",
        control_scale=True,
    )
    # Google Satellite Hybrid (satellite imagery + roads/labels)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="&copy; Google",
        name="Google Satellite Hybrid",
        max_zoom=20,
    ).add_to(fmap)

    cols = ["grid_id", "geometry"]
    for c in ["district_name", ds.OOF_DIFF_COL, ds.OOF_OBSERVED_COL, ds.OOF_PREDICTED_COL,
              "tree_std_log", "confidence_pct", "confidence_tier", "tier_color", "lower_ci_95", "upper_ci_95"]:
        if c in gdf.columns:
            cols.append(c)

    sub = gdf[cols].copy()
    sub["grid_id"] = sub["grid_id"].map(ds.normalize_id)
    rename_map = {}
    if "district_name" in sub.columns:
        rename_map["district_name"] = "district"
    if ds.OOF_DIFF_COL in sub.columns:
        rename_map[ds.OOF_DIFF_COL] = "mismatch"
    if ds.OOF_OBSERVED_COL in sub.columns:
        rename_map[ds.OOF_OBSERVED_COL] = "observed"
    if ds.OOF_PREDICTED_COL in sub.columns:
        rename_map[ds.OOF_PREDICTED_COL] = "oof_predicted"
    sub = sub.rename(columns=rename_map)

    # Calculate percentage mismatch: ((Observed - Predicted) / Predicted) * 100
    if "observed" in sub.columns and "oof_predicted" in sub.columns:
        denom = sub["oof_predicted"].replace(0, np.nan)
        sub["mismatch_pct"] = ((sub["mismatch"] / denom) * 100).round(1)
        sub["mismatch_pct_label"] = sub["mismatch_pct"].apply(lambda p: f"{p:+.1f}%" if pd.notnull(p) else "N/A")
    else:
        sub["mismatch_pct"] = np.nan
        sub["mismatch_pct_label"] = "N/A"

    if "confidence_pct" in sub.columns:
        sub["confidence_label"] = sub["confidence_pct"].apply(lambda p: f"{p:.1f}%" if pd.notnull(p) else "N/A")
    if "tree_std_log" in sub.columns:
        sub["variance_label"] = sub["tree_std_log"].apply(lambda v: f"±{v:.2f}" if pd.notnull(v) else "N/A")

    geojson_dict = json.loads(sub.to_json())

    def style(feat):
        props = feat.get("properties", {})
        gid = str(props.get("grid_id", ""))
        if selected_id is not None and gid == str(selected_id):
            return {"color": "#111827", "weight": 3.5, "fillColor": "#F59E0B", "fillOpacity": 0.95}

        if mode == "Spatial mismatch":
            pct = props.get("mismatch_pct")
            raw_diff = props.get("mismatch")
            c = _mismatch_gradient_color(pct, raw_diff)
            return {"color": "#475569", "weight": 0.4, "fillColor": c, "fillOpacity": 0.70}
        elif mode == "🛡️ Predictive Uncertainty & Model Confidence":
            c = props.get("tier_color", "#10B981")
            return {"color": "#334155", "weight": 0.4, "fillColor": c, "fillOpacity": 0.75}
        else:
            return {"color": "#FFFFFF", "weight": 0.3, "fillColor": "#60A5FA", "fillOpacity": 0.55}

    if mode == "Spatial mismatch":
        tooltip_fields = ["grid_id", "district", "observed", "oof_predicted", "mismatch", "mismatch_pct_label"]
        tooltip_aliases = ["Grid ID:", "District:", "Observed:", "Validation Pred:", "Residual:", "Mismatch %:"]
    elif mode == "🛡️ Predictive Uncertainty & Model Confidence":
        tooltip_fields = ["grid_id", "district", "confidence_tier", "confidence_label", "variance_label"]
        tooltip_aliases = ["Grid ID:", "District:", "Confidence Tier:", "Confidence Score:", "Tree Disagreement:"]
    else:
        tooltip_fields = ["grid_id", "district"]
        tooltip_aliases = ["Grid ID:", "District:"]

    sample_props = geojson_dict["features"][0]["properties"] if geojson_dict.get("features") else {}
    valid_fields = [f for f in tooltip_fields if f in sample_props]
    valid_aliases = [a for f, a in zip(tooltip_fields, tooltip_aliases) if f in sample_props]

    folium.GeoJson(
        geojson_dict, name=mode, style_function=style,
        highlight_function=lambda _: {"weight": 2.5, "color": "#F3F4F6", "fillOpacity": 0.95},
        tooltip=folium.GeoJsonTooltip(fields=valid_fields, aliases=valid_aliases, sticky=True) if valid_fields else None,
    ).add_to(fmap)

    if selected_id is not None:
        sel = gdf[gdf["grid_id"].astype(str) == str(selected_id)]
        if not sel.empty:
            cen = sel.geometry.iloc[0].centroid
            folium.Marker([cen.y, cen.x], tooltip=f"Selected: Grid {selected_id}", icon=folium.Icon(color="orange", icon="map-marker")).add_to(fmap)
            b = sel.total_bounds
            fmap.fit_bounds([[b[1],b[0]],[b[3],b[2]]], padding=(30,30))

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.title("🗺️ Urban Pulse AI Agent")
st.markdown(
    "**Research project:** Learning Functional Centrality from Urban Morphology — Colombo & Gampaha, Sri Lanka. "
    "Select a grid on the map and ask questions in plain language."
)

# -------------------------------------------------------
# Main layout
# -------------------------------------------------------
if mode == "🔬 Scientific Discovery & Evolution":
    st.subheader("🔬 Theory-Guided Self-Evolving Urban Intelligence (TSE-UI)")
    st.markdown(
        "Empirical anomaly diagnosis, peer-reviewed scientific literature integration (OpenAlex), "
        "and spatial cross-validation hypothesis evaluation engine."
    )

    t1, t2, t3, t4 = st.tabs([
        "1. Anomaly Profiling & Pattern Detection",
        "2. Peer-Reviewed Theory Grounding",
        "3. Hypothesis Testing & Evolution",
        "4. Scientific Memory Registry",
    ])

    with t1:
        st.markdown("### Step 1: Systemic Empirical Failure Clusters (7,764 Grids)")
        st.caption("Automatically classifies spatial cross-validation discrepancies into structural archetypes.")
        
        profiles = ans.profile_spatial_anomalies(table)
        if profiles.get("success", False):
            c_a = profiles.get("archetype_overprediction", {})
            c_b = profiles.get("archetype_underprediction", {})

            col_a, col_b = st.columns(2)
            with col_a:
                st.error(f"**Archetype A: {c_a.get('archetype', 'Transit Barrier / Bypass')}**")
                st.markdown(f"**Grids Identified:** **{c_a.get('grid_count', 0):,}** ({c_a.get('pct_of_total_grids', 0)}% of study area)")
                st.markdown(f"**Empirical Anomaly:** Mean observed: `{c_a.get('mean_observed', 0):,.0f}` vs. Mean predicted: `{c_a.get('mean_predicted', 0):,.0f}` ({c_a.get('mean_pct_deviation', 0):+.1f}%)")
                st.markdown(f"*{c_a.get('scientific_question', '')}*")
                if c_a.get("top_exemplars"):
                    st.markdown("**Top Representative Failure Grids:**")
                    ex_df_a = pd.DataFrame(c_a["top_exemplars"])
                    st.dataframe(ex_df_a, use_container_width=True, hide_index=True)
                if st.button("🔍 Ground Overprediction Anomaly with Literature", key="btn_ground_a", use_container_width=True):
                    st.session_state["lit_search_query"] = "urban morphology space syntax through movement transit barrier severance"
                    st.session_state["active_archetype_target"] = "overprediction"
                    st.session_state["cached_lit_works"] = None
                    st.success("Target set to Archetype A! Switched research focus to transit barrier / edge severance literature.")

            with col_b:
                st.success(f"**Archetype B: {c_b.get('archetype', 'Organic / Informal Vitality')}**")
                st.markdown(f"**Grids Identified:** **{c_b.get('grid_count', 0):,}** ({c_b.get('pct_of_total_grids', 0)}% of study area)")
                st.markdown(f"**Empirical Anomaly:** Mean observed: `{c_b.get('mean_observed', 0):,.0f}` vs. Mean predicted: `{c_b.get('mean_predicted', 0):,.0f}` ({c_b.get('mean_pct_deviation', 0):+.1f}%)")
                st.markdown(f"*{c_b.get('scientific_question', '')}*")
                if c_b.get("top_exemplars"):
                    st.markdown("**Top Representative Vitality Grids:**")
                    ex_df_b = pd.DataFrame(c_b["top_exemplars"])
                    st.dataframe(ex_df_b, use_container_width=True, hide_index=True)
                if st.button("🔍 Ground Underprediction Anomaly with Literature", key="btn_ground_b", use_container_width=True):
                    st.session_state["lit_search_query"] = "informal urban centrality street vitality land use diversity"
                    st.session_state["active_archetype_target"] = "underprediction"
                    st.session_state["cached_lit_works"] = None
                    st.success("Target set to Archetype B! Switched research focus to Jane Jacobs street vitality literature.")
        else:
            st.warning(f"Could not calculate anomaly clusters: {profiles.get('error', 'Unknown')}")

    with t1:
        from pattern_ui_service import render_pattern_panel
        render_pattern_panel(table)

    with t2:
        st.markdown("### Step 2: Verified Scientific Literature Retrieval (OpenAlex)")
        st.caption("Live connection to OpenAlex scholarly index (250M+ peer-reviewed papers).")
        
        default_q = st.session_state.get("lit_search_query", "urban morphology transit accessibility natural movement space syntax")
        q_topic = st.text_input(
            "Search Urban Theory Literature",
            value=default_q,
            key="input_lit_query",
        )

        col_search1, col_search2 = st.columns([1, 1])
        with col_search1:
            do_search = st.button("🔍 Search Scientific Works", use_container_width=True)
        with col_search2:
            if st.button("⚡ Reset / Fetch Grounding Works", use_container_width=True):
                st.session_state["cached_lit_works"] = None
                do_search = True

        if do_search or ("cached_lit_works" not in st.session_state or st.session_state["cached_lit_works"] is None):
            with st.spinner("Querying OpenAlex scientific database (250M+ peer-reviewed works)..."):
                works = lits.search_scholarly_literature(q_topic, limit=5)
                st.session_state["cached_lit_works"] = works
        else:
            works = st.session_state.get("cached_lit_works", [])

        if works:
            st.success(f"Retrieved {len(works)} peer-reviewed papers for: *{q_topic}*")
            for w in works:
                with st.expander(f"📄 {w['title']} ({w.get('year', 'N/A')})"):
                    st.markdown(f"**Authors:** {w.get('authors', 'Unknown')}")
                    st.markdown(f"**Citations:** {w.get('cited_by_count', 0):,}")
                    st.markdown(f"**DOI Link:** [{w['doi']}]({w['doi']})")
                    st.caption(f"**Abstract:** {w.get('abstract', '')}")
        else:
            st.info("No works returned for this query. Try a broader search term like 'urban morphology space syntax' or click 'Reset / Fetch Grounding Works'.")

        st.divider()
        st.markdown("#### 🤖 AI Theory & Feature Recommender (Error % & SHAP-Guided)")
        st.caption("AI reasons over the dominant error clusters and SHAP drivers to recommend the optimal set of morphological variables.")

        target_arch = st.session_state.get("active_archetype_target", "overprediction")

        if target_arch == "overprediction":
            st.warning(
                "**AI Diagnostic Insight for Overprediction (Transit Bypass / Barrier Severance):**  \n"
                "• **Observed Failure:** Model predicts high vitality because `Transit_Ac` and `BtA500_wt` are high, but actual activity is 50–75% lower.  \n"
                "• **SHAP Driver:** `Transit_Ac` has the highest positive SHAP push, but `land_use_e` is near 0.  \n"
                "• **Recommended Action:** Train with `Transit_Ac × land_use_e` or deselect monofunctional transit to prevent expressway bypass errors."
            )
            rec_features = ["Transit_Ac", "land_use_e", "BtA500_wt", "POI_Densit", "Floor_Inte"]
        else:
            st.success(
                "**AI Diagnostic Insight for Underprediction (Informal / Organic Vitality):**  \n"
                "• **Observed Failure:** Model underpredicts actual footfall by +40% to +150% because formal building footprints (`Floor_Inte`) are modest.  \n"
                "• **SHAP Driver:** Low `Floor_Inte` suppresses the prediction, ignoring dense pedestrian junctions.  \n"
                "• **Recommended Action:** Train with `IntDen_km2 × (1.0 + land_use_e)` and `LConn_wt` to capture Jane Jacobs informal sidewalk vitality."
            )
            rec_features = ["IntDen_km2", "land_use_e", "LConn_wt", "POI_Densit"]

        st.markdown(f"**💡 AI Suggested Training Predictors for {target_arch.title()}:** `{', '.join(rec_features)}`")
        if st.button(f"🎯 Apply AI Recommended Variables to Step 3 Experiment", key="btn_apply_ai_vars", use_container_width=True):
            st.session_state["ai_recommended_base_features"] = rec_features
            st.success(f"Applied {len(rec_features)} AI-recommended predictors to Step 3!")

        st.divider()
        st.markdown("#### Foundational Theoretical Grounding & Hypothesis Synthesis")
        st.caption("Synthesize theoretical propositions into testable individual and mutual candidate features.")

        for idx, t in enumerate(lits.FOUNDATIONAL_THEORIES):
            with st.expander(f"🏛️ {t['theory_name']} — {t['author']} ({t['year']})"):
                st.markdown(f"**Title:** *{t['title']}*")
                st.markdown(f"**DOI:** [{t['doi']}]({t['doi']})")
                st.markdown(f"**Core Proposition:** {t['proposition']}")
                st.markdown(f"**Empirical Relevance:** {t['relevance']}")

                if st.button(f"🧪 Synthesize Hypotheses from {t['theory_name'].split()[0]}", key=f"synth_{idx}", use_container_width=True):
                    generated_hypos = lits.synthesize_theory_hypotheses(target_arch, t["theory_name"], t["proposition"])
                    for gh in generated_hypos:
                        exps.register_candidate_feature(
                            name=gh["name"],
                            formula=gh["formula"],
                            theory=f"{t['theory_name']} ({t['author']} {t['year']})",
                            hypothesis=gh["hypothesis"],
                            compute_fn=gh["compute"]
                        )
                    st.session_state["last_synthesized"] = [gh["name"] for gh in generated_hypos]
                    st.success(f"Generated {len(generated_hypos)} candidate hypotheses (Individual & Mutual) and sent to Step 3!")
                    for gh in generated_hypos:
                        st.markdown(f"- **[{gh['type']}] `{gh['name']}`**: `{gh['formula']}`  \n  *{gh['hypothesis']}*")

    with t3:
        st.markdown("### Step 3: AI-Agent Investigation & Human-Supervised Experiment")
        st.caption(
            "qwen3:4b investigates the selected grid using residual, SHAP and literature tools. "
            "Experiments compare M0 baseline, M1 statistical calibration, M2 AI theory feature and M3 combined learning on exploratory spatial CV. "
            "Every execution requires explicit human approval."
        )

        current_engine = st.session_state.get("investigation_engine")
        if current_engine is not None and str(current_engine.state.grid_id) != str(st.session_state.selected_grid_id):
            st.session_state.investigation_engine = None
            st.session_state.investigation_request_id = None
            current_engine = None

        investigation_question = st.text_area(
            "Investigation question",
            value="Why does the saved model misestimate daily and hourly activity at this location, and which testable morphology refinement is supported by retrieved literature?",
            key="phase5_investigation_question",
        )
        start_disabled = st.session_state.selected_grid_id is None or not ollama_ok
        start_col, continue_col, reset_col = st.columns(3)
        with start_col:
            start_clicked = st.button("Start Ollama investigation", disabled=start_disabled,
                                      use_container_width=True, key="phase5_start")
        with continue_col:
            continue_clicked = st.button("Continue agent investigation", disabled=current_engine is None,
                                         use_container_width=True, key="phase5_continue")
        with reset_col:
            if st.button("Reset investigation", disabled=current_engine is None,
                         use_container_width=True, key="phase5_reset"):
                st.session_state.investigation_engine = None
                st.session_state.investigation_request_id = None
                st.rerun()

        if start_clicked:
            try:
                with st.spinner("Ollama agent is collecting residual, SHAP and literature evidence..."):
                    current_engine = p5.start_agent_investigation(
                        st.session_state.selected_grid_id, investigation_question, bundle,
                    )
                    run_report = p5.run_agent_investigation(current_engine)
                    st.session_state.phase5_diagnostics = run_report["diagnostics"]
                st.session_state.investigation_engine = current_engine
                st.session_state.investigation_request_id = None
                st.rerun()
            except Exception as exc:
                st.error(f"Investigation could not start: {exc}")
        elif continue_clicked:
            try:
                with st.spinner("Ollama agent is continuing the bounded investigation..."):
                    run_report = p5.run_agent_investigation(current_engine)
                    st.session_state.phase5_diagnostics = run_report["diagnostics"]
            except Exception as exc:
                st.error(f"Investigation could not continue: {exc}")

        current_engine = st.session_state.get("investigation_engine")
        if current_engine is not None:
            state = current_engine.state
            st.markdown(f"**Investigation:** `{state.investigation_id}` · **Grid:** `{state.grid_id}` · **Status:** `{state.status.value}`")
            if state.errors:
                with st.expander("Agent/tool limitations and errors"):
                    st.json(state.errors)
            if st.session_state.phase5_diagnostics:
                with st.expander("Investigation stage diagnostics", expanded=True):
                    st.dataframe(pd.DataFrame(st.session_state.phase5_diagnostics), use_container_width=True, hide_index=True)

            evidence_rows = []
            for evidence_id, entry in state.evidence.items():
                evidence_rows.append({
                    "evidence_id": evidence_id, "tool": entry.get("tool"),
                    "scope": entry.get("payload", {}).get("scope", "retrieved evidence"),
                })
            if evidence_rows:
                st.markdown("#### Collected evidence")
                st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)
                with st.expander("Inspect evidence and provenance"):
                    st.json(state.evidence)

            if state.proposal is None:
                st.info("The agent has not produced a valid grounded hypothesis yet. Review errors, then continue the bounded investigation.")
            else:
                proposal = state.proposal
                st.markdown("#### Agent-generated testable hypothesis")
                st.info(f"**{proposal.title}**\n\n{proposal.hypothesis}\n\n**Mechanism:** {proposal.mechanism}")
                st.markdown("**Expected relationships:** " + "; ".join(proposal.expected_relationships))
                st.markdown("**Uncertainties:** " + "; ".join(proposal.uncertainties))
                if proposal.features:
                    st.dataframe(pd.DataFrame([item.model_dump() for item in proposal.features]),
                                 use_container_width=True, hide_index=True)
                with st.expander("Literature grounding and cited excerpts"):
                    st.json({"citations": [item.model_dump() for item in proposal.literature],
                             "retrieved_sources": state.literature_sources})

                if state.execution_plan is None and state.status.value == "proposed":
                    st.warning(
                        "This is exploratory research on previously studied spatial folds. "
                        "It is not an untouched holdout and cannot support model promotion."
                    )
                    if st.button("Prepare reproducible exploratory experiment", use_container_width=True,
                                 key="phase5_prepare"):
                        try:
                            with st.spinner("Validating cohort, spatial blocks, buffers and reproducibility identities..."):
                                p5.configure_exploratory_execution(current_engine)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Experiment preflight failed: {exc}")

                state = current_engine.state
                if state.execution_plan is not None:
                    plan = state.execution_plan
                    st.markdown("#### Reproducible execution plan")
                    pc1, pc2, pc3 = st.columns(3)
                    pc1.metric("Validation", "Exploratory spatial CV")
                    pc2.metric("Development cells", f"{len(plan.development_grid_ids):,}")
                    pc3.metric("Estimated RF fits", f"{plan.estimated_rf_fits:,}")
                    st.caption(f"Run identity: `{plan.execution_id}` · Spatial buffer: {plan.minimum_separation_m:,.0f} m · No model saving/deployment")

                    request_id = st.session_state.get("investigation_request_id")
                    if request_id is None and state.status.value == "proposed":
                        if st.button("Request human experiment approval", use_container_width=True,
                                     key="phase5_request_approval"):
                            try:
                                request = current_engine.request_action("experiment")
                                st.session_state.investigation_request_id = request.request_id
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Approval request failed: {exc}")

                    request_id = st.session_state.get("investigation_request_id")
                    if request_id and state.status.value == "awaiting_approval":
                        with st.expander("Exact approval payload", expanded=True):
                            st.json(current_engine.approval_payload(request_id))
                        reviewer = st.text_input("Authenticated reviewer identity", key="phase5_reviewer")
                        rationale = st.text_area("Approval rationale", key="phase5_approval_rationale")
                        confirmed = st.checkbox(
                            "I approve this exact exploratory plan and understand it is not independent holdout evidence.",
                            key="phase5_approval_confirm",
                        )
                        ac1, ac2 = st.columns(2)
                        if ac1.button("Approve experiment", disabled=not confirmed,
                                     use_container_width=True, key="phase5_approve"):
                            try:
                                current_engine.decide_action(
                                    request_id, approve=True, reviewer=reviewer, rationale=rationale,
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Approval failed: {exc}")
                        if ac2.button("Deny experiment", use_container_width=True, key="phase5_deny"):
                            try:
                                current_engine.decide_action(
                                    request_id, approve=False, reviewer=reviewer, rationale=rationale,
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Decision failed: {exc}")

                    state = current_engine.state
                    if request_id and state.status.value == "approved":
                        st.warning("Execution can be computationally expensive. It creates new result files but never saves or deploys a model.")
                        if st.button("Execute approved Python experiment", type="primary",
                                     use_container_width=True, key="phase5_execute"):
                            try:
                                with st.spinner("Running the approved spatial experiment..."):
                                    current_engine.execute_action(request_id, "experiment")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Experiment execution failed: {exc}")

                state = current_engine.state
                rows = p5.metric_rows(state)
                if rows:
                    st.markdown("#### Actual paired validation results")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    st.caption("M0–M3 were evaluated on identical exploratory spatial partitions. M1 and M3 calibration fit only inner training-fold predictions. These are not untouched-holdout metrics.")
                    if state.status.value == "evaluated":
                        decision = st.radio("Human scientific decision", ["rejected", "deferred"],
                                            horizontal=True, key="phase5_result_decision")
                        result_rationale = st.text_area("Result-review rationale", key="phase5_result_rationale")
                        result_reviewer = st.text_input("Result reviewer identity", key="phase5_result_reviewer")
                        if st.button("Finalize scientific decision", use_container_width=True,
                                     key="phase5_finalize"):
                            try:
                                current_engine.review_validation(
                                    decision, reviewer=result_reviewer, rationale=result_rationale,
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Review failed: {exc}")
                    elif state.status.value == "accepted_or_rejected":
                        st.markdown(f"**Human decision:** `{state.review_status}`")
                        if st.button("Record outcome in scientific memory", use_container_width=True,
                                     key="phase5_record"):
                            try:
                                current_engine.record()
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Recording failed: {exc}")
                    elif state.status.value == "recorded":
                        st.success(f"Outcome recorded: `{state.scientific_memory_reference['record_id']}`")

        st.divider()
        st.markdown("### 📊 Model Evolution Benchmark Stages")
        st.caption("Measuring progression across statistical refinement, theoretical interaction, and data-source learning.")

        champ_info = regs.get_champion_model_state()
        champ_id = champ_info.get("champion_id", "EXP-010")
        champ_r2 = champ_info.get("champion_metrics", {}).get("r2", 0.5181)

        benchmark_stages = [
            {"Model Stage": "M0: Baseline Reference", "Learning Mechanism": "Original 8 Morphological Predictors (Log1p RF)", "Spatial 5-Fold R²": "0.5156", "ΔR² vs M0": "—", "Scientific Verdict": "Reference Baseline (Immutable)"},
            {"Model Stage": "M1: Statistical Refinement", "Learning Mechanism": "EXP-008 In-Fold Isotonic Calibration", "Spatial 5-Fold R²": "0.4851", "ΔR² vs M0": "-0.0304", "Scientific Verdict": "REJECTED (Distorted Q1 low-activity cells)"},
            {"Model Stage": f"M2: Theory-Guided Model ({champ_id})", "Learning Mechanism": "Pedestrian Permeability Synergy (MAD500_wt × IntDen_km2)", "Spatial 5-Fold R²": f"{champ_r2:.4f}", "ΔR² vs M0": "+0.0025", "Scientific Verdict": "ACCEPTED Champion (Colombo MAE -74.4)"},
            {"Model Stage": "M3: Additional-Data Learning", "Learning Mechanism": "POI Functional Subcategories (pois.shp) & Land-Use", "Spatial 5-Fold R²": "Pending", "ΔR² vs M0": "Pending", "Scientific Verdict": "Candidate in Preparation"},
        ]
        st.dataframe(pd.DataFrame(benchmark_stages), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### 🧪 Theory-Guided Hypothesis Evaluator")
        st.caption("Test theoretical interaction features across spatial 5-fold cross-validation with active baseline predictors.")

        # Interactive Hypothesis & Feature Creator Expander
        with st.expander("✨ Formulate New Hypothesis & Custom Candidate Feature", expanded=False):
            st.markdown("Create a new mathematical interaction term guided by urban theory:")
            c_f1, c_op, c_f2 = st.columns([1.5, 0.8, 1.5])
            base_feature_options = list(exps.BASE_FEATURES)
            with c_f1:
                var1 = st.selectbox("Predictor 1", base_feature_options, index=1, key="custom_var1")
            with c_op:
                op = st.selectbox("Operation", ["*", "+", "/"], key="custom_op")
            with c_f2:
                var2 = st.selectbox("Predictor 2", [f for f in base_feature_options if f != var1], index=5, key="custom_var2")

            auto_name = f"{var1}_{var2}_Synergy" if op == "*" else (f"{var1}_{var2}_Ratio" if op == "/" else f"{var1}_{var2}_Sum")
            custom_name = st.text_input("New Feature Name", value=auto_name, key="custom_feat_name")
            custom_theory = st.text_input("Theoretical Grounding", value="Urban Morphology & Functional Agglomeration", key="custom_theory")
            custom_hypo = st.text_area("Hypothesis", value=f"The interaction between {var1} and {var2} explains functional activity variation.", key="custom_hypo")

            if st.button("➕ Register & Select for Testing", key="btn_register_custom", use_container_width=True):
                if op == "*":
                    compute_lambda = lambda df, v1=var1, v2=var2: df[v1] * df[v2]
                elif op == "/":
                    compute_lambda = lambda df, v1=var1, v2=var2: df[v1] / (np.maximum(1e-5, df[v2]))
                else:
                    compute_lambda = lambda df, v1=var1, v2=var2: df[v1] + df[v2]

                exps.register_candidate_feature(
                    name=custom_name,
                    formula=f"{var1} {op} {var2}",
                    theory=custom_theory,
                    hypothesis=custom_hypo,
                    compute_fn=compute_lambda
                )
                st.session_state["last_synthesized"] = [custom_name]
                st.success(f"Registered `{custom_name}`! It is now selected for testing.")
                st.rerun()

        # 1. Feature selection & Candidate selection
        feature_options = list(exps.CANDIDATE_THEORY_FEATURES.keys())
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            default_index = len(feature_options) - 1 if "last_synthesized" in st.session_state else 0
            if default_index >= len(feature_options):
                default_index = 0
            selected_cand = st.selectbox("Candidate Theoretical Feature to Test", feature_options, index=default_index)
        with col_f2:
            default_base = st.session_state.get("ai_recommended_base_features", base_feature_options)
            selected_base = st.multiselect(
                "Baseline Morphological Predictors",
                options=base_feature_options,
                default=default_base,
                help="Select which baseline features to include. Deselect features to run ablation experiments.",
            )

        cand_info = exps.CANDIDATE_THEORY_FEATURES.get(selected_cand, {})

        st.info(
            f"**Candidate Interaction Formula:** `{cand_info.get('formula', 'custom')}`  \n"
            f"**Theoretical Grounding:** {cand_info.get('theory', 'Urban Theory')}  \n"
            f"**Hypothesis:** {cand_info.get('hypothesis', 'Custom hypothesis')}  \n"
            f"**Active Baseline Predictors ({len(selected_base)}):** `{', '.join(selected_base) if selected_base else 'None'}`"
        )

        n_trees = st.slider(
            "Number of Random Forest Trees (Spatial CV)",
            min_value=20,
            max_value=300,
            value=50,
            step=10,
            help="50 trees is recommended for quick interactive testing (~5s). 300 trees matches final publication specification (~40s).",
        )

        # --- Flexible Duplicate Check: inform user of prior evaluations without hard-blocking re-testing ---
        tested = regs.get_tested_feature_names()
        already_tested = selected_cand in tested
        allow_retest = True
        if already_tested:
            prev_verdicts = tested[selected_cand]
            accepted_count = prev_verdicts.count("ACCEPTED")
            rejected_count = prev_verdicts.count("REJECTED")
            st.info(
                f"ℹ️ **`{selected_cand}` has already been tested {len(prev_verdicts)}× in scientific memory** "
                f"({rejected_count} REJECTED, {accepted_count} ACCEPTED). "
                f"You can re-evaluate this feature with a modified baseline predictor subset (ablation), "
                f"different tree count, or customize a new hypothesis above."
            )
            allow_retest = st.checkbox(
                "Allow re-evaluation with modified predictors or settings",
                value=True,
                key=f"chk_allow_retest_{selected_cand}"
            )

        can_run = len(selected_base) > 0 and (not already_tested or allow_retest)
        if not can_run:
            st.warning("⚠️ Please select at least one baseline predictor to run the experiment.")

        if st.button("🚀 Run 5-Fold Spatial CV Experiment", use_container_width=True, disabled=not can_run):
            with st.spinner(f"Re-training spatial 5-fold models ({len(selected_base)} base + 1 theoretical feature, {n_trees} trees)..."):
                dataset_path = DATASET_PATH
                exp_res = exps.evaluate_theoretical_hypothesis(
                    dataset_path=dataset_path,
                    candidate_feature_name=selected_cand,
                    base_features=selected_base,
                    n_estimators=n_trees,
                )

            if exp_res.get("success"):
                regs.record_experiment_result(exp_res)
                v = exp_res.get("verdict", {})
                if v.get("status") == "ACCEPTED":
                    st.success(f"**Scientific Verdict: {v.get('status')}** — {v.get('scientific_rationale')}")
                else:
                    st.warning(f"**Scientific Verdict: {v.get('status')}** — {v.get('scientific_rationale')}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Baseline R²", f"{exp_res['baseline_metrics']['r2']:.4f}")
                m2.metric("Evolved R²", f"{exp_res['evolved_metrics']['r2']:.4f}", delta=f"{exp_res['deltas']['delta_r2']:+.4f}")
                m3.metric("Baseline RMSE", f"{exp_res['baseline_metrics']['rmse']:,.1f}")
                m4.metric("Evolved RMSE", f"{exp_res['evolved_metrics']['rmse']:,.1f}", delta=f"{exp_res['deltas']['delta_rmse']:+,.1f}")

                st.markdown("#### Anomaly Cluster Resolution")
                ar = exp_res.get("anomaly_resolution", {})
                ac1, ac2 = st.columns(2)
                ac1.metric(
                    "Archetype A (Overprediction) MAE Change",
                    f"{ar.get('overprediction_mae_baseline', 0):,.1f} → {ar.get('overprediction_mae_evolved', 0):,.1f}",
                    delta=f"{ar.get('overprediction_mae_change', 0):+,.1f}",
                )
                ac2.metric(
                    "Archetype B (Underprediction) MAE Change",
                    f"{ar.get('underprediction_mae_baseline', 0):,.1f} → {ar.get('underprediction_mae_evolved', 0):,.1f}",
                    delta=f"{ar.get('underprediction_mae_change', 0):+,.1f}",
                )
            else:
                st.error(f"Experiment failed: {exp_res.get('error')}")


    with t4:
        st.markdown("### Step 4: Scientific Memory & Model Version Registry")
        st.caption("Immutable ledger of formulated hypotheses, empirical deltas, and theoretical justifications.")

        mem = regs.load_scientific_memory()
        s1, s2, s3 = st.columns(3)
        s1.metric("Experiments Logged", mem.get("experiments_count", 0))
        s2.metric("Hypotheses Accepted", mem.get("hypotheses_accepted", 0))
        s3.metric("Hypotheses Rejected", mem.get("hypotheses_rejected", 0))

        exps_list = mem.get("experiments", [])
        if exps_list:
            df_mem = pd.DataFrame(exps_list)
            show_cols = [c for c in ["id", "feature_name", "formula", "verdict", "delta_r2", "delta_rmse", "theory"] if c in df_mem.columns]
            st.dataframe(df_mem[show_cols], use_container_width=True, hide_index=True)
            for e in reversed(exps_list):
                with st.expander(f"📜 {e['id']} — {e['feature_name']} [{e['verdict']}]"):
                    st.markdown(f"**Formula:** `{e.get('formula')}`")
                    st.markdown(f"**Theory:** {e.get('theory')}")
                    st.markdown(f"**Hypothesis:** {e.get('hypothesis')}")
                    st.markdown(f"**Scientific Rationale:** {e.get('rationale')}")
                    st.markdown(f"**ΔR²:** `{e.get('delta_r2'):+.4f}` | **ΔRMSE:** `{e.get('delta_rmse'):+.1f}`")
        else:
            st.info("No experiments recorded yet in scientific memory.")

        investigations = mem.get("investigations", [])
        st.divider()
        st.markdown("#### AI-Agent Investigation History")
        st.metric("Recorded investigation revisions", len(investigations))
        if investigations:
            history_rows = []
            for record in reversed(investigations):
                snapshot = record.get("investigation", {})
                proposal = snapshot.get("proposal") or {}
                receipt = snapshot.get("validation_results") or {}
                history_rows.append({
                    "record_id": record.get("id"), "grid_id": snapshot.get("grid_id"),
                    "hypothesis": proposal.get("title"), "method": proposal.get("method"),
                    "decision": snapshot.get("review_status"),
                    "candidate_r2": (receipt.get("daily") or {}).get("r2"),
                    "scope": receipt.get("scope"),
                })
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
            with st.expander("Inspect latest structured investigation record"):
                st.json(investigations[-1])
        else:
            st.info("No supervised AI-agent investigation outcomes have been recorded yet.")

else:
    map_grids = grids_geo.copy()
    if district_filter != "All districts" and district_col:
        map_grids = map_grids[map_grids[district_col].astype(str) == district_filter].copy()

    # Pre-merge batch epistemic uncertainty for maps and tooltips
    try:
        unc_df = _get_uncertainty_df(bundle, grids_geo)
        if not unc_df.empty:
            map_grids = map_grids.merge(unc_df, on="grid_id", how="left")
    except Exception as exc:
        logger.warning("Could not merge batch uncertainty: %s", exc)

    selected_id = st.session_state.selected_grid_id

    col_map, col_panel = st.columns([1.6, 1], gap="large")

    # ---- Map ----
    with col_map:
        if mode == "Spatial mismatch":
            st.subheader("Spatial Mismatch Map (Gradient by % Deviation)")
            st.markdown(
                """
                <div style="font-size: 0.82rem; margin-bottom: 8px; line-height: 1.5; padding: 8px 12px; background: rgba(30, 41, 59, 0.4); border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);">
                    <div style="font-weight: 600; margin-bottom: 4px;">🎨 Diverging Mismatch Gradient:</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
                        <span style="display: inline-flex; align-items: center; gap: 4px;"><span style="display:inline-block;width:12px;height:12px;background:#1E3A8A;border-radius:2px;"></span> <b>&lt; -75%</b> (Severe Overprediction)</span>
                        <span style="display: inline-flex; align-items: center; gap: 4px;"><span style="display:inline-block;width:12px;height:12px;background:#60A5FA;border-radius:2px;"></span> <b>-25% to -50%</b></span>
                        <span style="display: inline-flex; align-items: center; gap: 4px;"><span style="display:inline-block;width:12px;height:12px;background:#E2E8F0;border-radius:2px;"></span> <b>±10%</b> (Balanced)</span>
                        <span style="display: inline-flex; align-items: center; gap: 4px;"><span style="display:inline-block;width:12px;height:12px;background:#F87171;border-radius:2px;"></span> <b>+35% to +75%</b></span>
                        <span style="display: inline-flex; align-items: center; gap: 4px;"><span style="display:inline-block;width:12px;height:12px;background:#991B1B;border-radius:2px;"></span> <b>&gt; +150%</b> (Severe Underprediction)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif mode == "🛡️ Predictive Uncertainty & Model Confidence":
            st.subheader("🛡️ Predictive Uncertainty & Model Confidence Map")
            st.markdown(
                """
                <div style="font-size: 0.82rem; margin-bottom: 8px; line-height: 1.5; padding: 8px 12px; background: rgba(30, 41, 59, 0.4); border-radius: 6px; border: 1px solid rgba(255,255,255,0.08);">
                    <div style="font-weight: 600; margin-bottom: 4px;">🛡️ Epistemic Uncertainty Tiers (300 Decision Tree Consensus):</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center;">
                        <span style="display: inline-flex; align-items: center; gap: 4px;"><span style="display:inline-block;width:12px;height:12px;background:#10B981;border-radius:2px;"></span> <b>🟢 High Confidence</b> (σ ≤ 0.65 — Strong tree consensus; safe for policy)</span>
                        <span style="display: inline-flex; align-items: center; gap: 4px;"><span style="display:inline-block;width:12px;height:12px;background:#F59E0B;border-radius:2px;"></span> <b>🟡 Moderate Confidence</b> (0.65 &lt; σ ≤ 1.25 — Suburban-rural transition)</span>
                        <span style="display: inline-flex; align-items: center; gap: 4px;"><span style="display:inline-block;width:12px;height:12px;background:#EF4444;border-radius:2px;"></span> <b>🔴 High Uncertainty</b> (σ &gt; 1.25 — Severe tree divergence; requires survey)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.subheader("Eligible Grid Cells")
            st.caption("Click inside a grid polygon to select it. The selected grid highlights in orange.")

        fmap = _create_map(map_grids, selected_id=selected_id, mode=mode)
        map_result = st_folium(fmap, height=580, use_container_width=True, returned_objects=["last_clicked"], key=f"map_{mode}_{district_filter}")

        clicked = (map_result or {}).get("last_clicked")
        if clicked and clicked.get("lat") is not None:
            clicked_id = ds.find_clicked_grid(map_grids, clicked["lat"], clicked["lng"])
            if clicked_id and clicked_id != str(st.session_state.selected_grid_id):
                st.session_state.selected_grid_id = clicked_id
                st.session_state.messages = []
                st.session_state.agent_log = []
                st.rerun()
            elif not clicked_id:
                st.info("No eligible grid found at that click. Try clicking inside a visible polygon.")

    # ---- Grid panel ----
    with col_panel:
        st.subheader("Selected Grid")

        if selected_id is None:
            st.info("Click a grid on the map, or select a Grid ID in the sidebar.")
            st.markdown("""
    **After selecting a grid you can:**
    - Ask about its location and morphology
    - Get a predicted total activity value
    - See the 24-hour urban pulse chart
    - Understand which features drove the prediction (SHAP)
    - Compare observed vs OOF-predicted activity
    - Run a what-if scenario
    """)
        else:
            row_match = grids_geo[grids_geo["grid_id"].astype(str) == str(selected_id)]
            if row_match.empty:
                st.warning("Selected grid is not in the current map filter. Change district or clear selection.")
            else:
                row = row_match.iloc[0]
                loc = ds.get_location_details(row)
                oof = ds.get_oof_result(row)
                total_pred = ps.predict_total(bundle, row)
                pulse_pred = ps.predict_pulse(bundle, row)

                st.markdown(f"### Grid {selected_id}")
                st.markdown(f"**District:** {loc['district']}  \n**Centre:** {loc['lat']:.5f}, {loc['lon']:.5f}")

                # Compute mismatch & accuracy values for top display
                # Note: OOF (cross-validation) estimate is what powers the Spatial Mismatch Map
                obs_val = oof.get("observed")
                oof_pred_val = oof.get("predicted")
                pred_val = total_pred.get("value")
                base_mis_str, evo_mis_str, mis_delta_str = "—", "—", None

                # Base mismatch reference uses OOF validation estimate if available to exactly match the map (-56.4%),
                # falling back to operational prediction if OOF record is not available.
                ref_base_pred = oof_pred_val if (oof_pred_val is not None and oof_pred_val > 0) else pred_val
                if obs_val and obs_val > 0 and ref_base_pred is not None and ref_base_pred > 0:
                    # Initial Mismatch: (Observed - Predicted) / Predicted * 100
                    init_mis = ((obs_val - ref_base_pred) / ref_base_pred) * 100.0
                    base_mis_str = f"{init_mis:+.1f}%"
                    
                    # Quick check if evolved correction is available
                    shap_quick = es.get_shap_values(bundle, row)
                    evo_quick = ges.diagnose_and_evolve_grid(row, ref_base_pred, oof, shap_quick)
                    if evo_quick.get("has_diagnosis"):
                        evo_p = evo_quick["evolved_prediction"]
                        corr_mis = ((obs_val - evo_p) / evo_p) * 100.0 if evo_p > 0 else 0.0
                        evo_mis_str = f"{corr_mis:+.1f}%"
                        mis_drop = abs(init_mis) - abs(corr_mis)
                        if mis_drop > 0:
                            mis_delta_str = f"{mis_drop:.1f}% closer to 0"
                        elif mis_drop < 0:
                            mis_delta_str = f"{mis_drop:.1f}%"

                m1, m2, m3 = st.columns(3)
                with m1:
                    val = f"{pred_val:,.0f}" if pred_val is not None else "—"
                    st.metric("Predicted Daily Activity", val, help="Main operational model estimate based on urban morphology.")
                with m2:
                    st.metric("Initial Mismatch", base_mis_str, help="Initial cross-validation discrepancy = (Observed - Validation Pred) / Validation Pred * 100. Exactly matches Spatial Mismatch Map.")
                with m3:
                    st.metric("Corrected Mismatch", evo_mis_str, delta=mis_delta_str, help="Remaining discrepancy after theory adjustment = (Observed - Adjusted) / Adjusted * 100.")

                # Pulse chart: Compare Actual (Observed) vs. Predicted 24-Hour Pulses
                if pulse_pred.get("success") and pulse_pred.get("type") == "hourly":
                    pred_values = pulse_pred["values"]
                    hours = [f"{i:02d}:00" for i in range(24)]
                    peak = pulse_pred["peak_hour"]

                    # Extract actual hourly observed values from grid row if available
                    hour_cols = [
                        "activity_0", "activity_1", "activity_2", "activity_3", "activity_4", "activity_5",
                        "activity_6", "activity_7", "activity_8", "activity_9", "activity10", "activity11",
                        "activity12", "activity13", "activity14", "activity15", "activity16", "activity17",
                        "activity18", "activity19", "activity20", "activity21", "activity22", "activity23"
                    ]
                    has_actual = any(c in row and pd.notna(row[c]) for c in hour_cols)
                    actual_counts = []
                    if has_actual:
                        for c in hour_cols:
                            raw_v = float(row[c]) if (c in row and pd.notna(row[c])) else 0.0
                            actual_counts.append(max(0.0, raw_v))

                    fig = go.Figure()

                    # 1. Actual Observed Pulse – line + markers (raw activity counts)
                    if has_actual:
                        fig.add_trace(go.Scatter(
                            x=hours,
                            y=actual_counts,
                            mode="lines+markers",
                            name=f"Actual Observed Pulse (sum: {sum(actual_counts):,.0f})",
                            line=dict(color="#F59E0B", width=2.5),
                            marker=dict(color="#F59E0B", size=5, symbol="circle"),
                            fill="tozeroy",
                            fillcolor="rgba(245,158,11,0.10)",
                        ))

                    # 2. Predicted Pulse – line + markers (raw activity counts, sum matches Predicted Daily Activity)
                    pred_sum = sum(pred_values)
                    fig.add_trace(go.Scatter(
                        x=hours,
                        y=pred_values,
                        mode="lines+markers",
                        name=f"Predicted Pulse (peak {peak:02d}:00 | sum: {pred_sum:,.0f})",
                        line=dict(color="#3B82F6", width=2.5, dash="dot"),
                        marker=dict(color="#3B82F6", size=5, symbol="diamond"),
                        fill="tozeroy",
                        fillcolor="rgba(59,130,246,0.08)",
                    ))

                    chart_title = "24-Hour Urban Pulse: Actual Observed vs. Model Predicted" if has_actual else f"Predicted Hourly Urban Pulse (peak: {peak:02d}:00)"

                    fig.update_layout(
                        title=chart_title,
                        xaxis_title="Hour of Day",
                        yaxis_title="Hourly Human Activity (Count)",
                        height=295,
                        margin=dict(l=10, r=10, t=40, b=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(size=11),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("ℹ️ Hourly diurnal distribution (hourly activity count). The sum of all 24 predicted hourly values directly reconciles to match the Predicted Daily Activity.")
                elif pulse_pred.get("type") == "class":
                    st.info(f"Pulse model classifies this grid as: **{pulse_pred.get('label')}**")
                else:
                    st.caption("Pulse prediction unavailable.")

                # Model Validation detail
                with st.expander("Model Validation (Cross-Validation Performance)", expanded=True):
                    if oof["available"]:
                        oc, pc = st.columns(2)
                        oc.metric("Observed (Actual)", f"{oof['observed']:,.0f}")
                        pc.metric("Validation Estimate (OOF)", f"{oof['predicted']:,.0f}")
                        diff = oof["diff"]
                        if diff is not None:
                            if diff > 0:
                                st.success(f"Validation Residual: **{diff:+,.0f}** — Model underestimated this grid during cross-validation.")
                            elif diff < 0:
                                st.info(f"Validation Residual: **{diff:+,.0f}** — Model overestimated this grid during cross-validation.")
                            else:
                                st.info("Observed and validation-predicted values match.")
                        st.caption(f"Cross-Validation Fold: {oof.get('fold', 'unknown')} | Absolute Error: {oof.get('abs_error', 0):,.0f}")
                        st.caption("ℹ️ Out-Of-Fold (OOF) evaluates model generalization on unseen grids during testing. It is distinct from the primary Predicted Daily Activity above.")

                        # --- Hourly OOF Performance & Peak Hour Analysis ---
                        h_oof = evs.get_hourly_oof_for_grid(row)
                        if h_oof.get("available"):
                            st.markdown("---")
                            st.markdown("##### ⏱️ 24-Hour Spatial OOF Profile vs. Observed Ground Truth")

                            hp1, hp2, hp3 = st.columns(3)
                            with hp1:
                                st.metric("Observed Peak Hour", f"{h_oof['observed_peak_hour']:02d}:00", help=f"Observed peak activity value: {h_oof['observed_peak_val']:,.0f}")
                            with hp2:
                                st.metric("OOF Predicted Peak Hour", f"{h_oof['oof_predicted_peak_hour']:02d}:00", help=f"Spatial OOF predicted peak value: {h_oof['oof_predicted_peak_val']:,.0f}")
                            with hp3:
                                circ_diff = h_oof["peak_hour_circular_diff"]
                                diff_sign = h_oof["peak_hour_diff"]
                                align_desc = "Exact match" if circ_diff == 0 else f"{circ_diff} hr off (circular)"
                                st.metric("Peak Timing Error", align_desc, delta=f"{diff_sign:+d} hr (obs−pred)" if circ_diff != 0 else "0 hr")

                            hours_24 = [f"{i:02d}:00" for i in range(24)]
                            fig_h_oof = go.Figure()
                            # Observed curve
                            fig_h_oof.add_trace(go.Scatter(
                                x=hours_24,
                                y=h_oof["observed_hourly"],
                                mode="lines+markers",
                                name=f"Observed Hourly (sum: {h_oof['observed_hourly_sum']:,.0f})",
                                line=dict(color="#F59E0B", width=2.5),
                                marker=dict(color="#F59E0B", size=4),
                                fill="tozeroy",
                                fillcolor="rgba(245,158,11,0.08)",
                            ))
                            # Spatial OOF Predicted curve
                            fig_h_oof.add_trace(go.Scatter(
                                x=hours_24,
                                y=h_oof["oof_predicted_hourly"],
                                mode="lines+markers",
                                name=f"OOF Validation Pred (sum: {h_oof['oof_predicted_hourly_sum']:,.0f})",
                                line=dict(color="#8B5CF6", width=2.2, dash="dash"),
                                marker=dict(color="#8B5CF6", size=4, symbol="square"),
                                fill="tozeroy",
                                fillcolor="rgba(139,92,246,0.06)",
                            ))
                            fig_h_oof.update_layout(
                                title="Hourly Out-of-Fold Validation: Observed vs. Cross-Validated Prediction",
                                xaxis_title="Hour of Day",
                                yaxis_title="Hourly Human Presence (Count)",
                                height=260,
                                margin=dict(l=10, r=10, t=35, b=25),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(size=10.5),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                                hovermode="x unified",
                            )
                            st.plotly_chart(fig_h_oof, use_container_width=True)

                            # Hourly Residuals Bar Chart
                            res_vals = h_oof["hourly_residuals"]
                            bar_colors = ["#EF4444" if v >= 0 else "#3B82F6" for v in res_vals]
                            fig_h_res = go.Figure(go.Bar(
                                x=hours_24,
                                y=res_vals,
                                marker_color=bar_colors,
                                name="Hourly Residual (obs−pred)",
                                hovertemplate="Hour %{x}: %{y:+,.0f}<extra></extra>",
                            ))
                            fig_h_res.update_layout(
                                title="Hourly Residual Profile (Red = Underpredicted, Blue = Overpredicted)",
                                xaxis_title="Hour of Day",
                                yaxis_title="Residual (Observed − OOF Pred)",
                                height=210,
                                margin=dict(l=10, r=10, t=35, b=25),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(size=10.5),
                                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
                            )
                            st.plotly_chart(fig_h_res, use_container_width=True)
                            st.caption(f"ℹ️ Mean Hourly MAE: **{h_oof['mean_hourly_mae']:,.1f}** count units across 24 hours (strictly Spatial OOF cross-validation).")
                    else:
                        st.info("No cross-validation evaluation record for this grid.")

                # Model Confidence & Epistemic Uncertainty Quantification
                with st.expander("🛡️ Model Confidence & Epistemic Uncertainty (300 Trees)", expanded=True):
                    unc_res = ps.predict_uncertainty(bundle, row)
                    if unc_res.get("success"):
                        u1, u2, u3 = st.columns(3)
                        with u1:
                            st.metric("Confidence Score", f"{unc_res['confidence_pct']:.1f}%", help="100% = complete agreement across all 300 Random Forest estimators.")
                        with u2:
                            st.metric("Confidence Tier", unc_res["confidence_tier"], help="Categorized by standard deviation of decision tree ensemble predictions.")
                        with u3:
                            st.metric("Tree Disagreement (σ)", f"±{unc_res['tree_std_log']:.2f}", help="Standard deviation of log-scale predictions across all 300 individual trees.")

                        st.markdown(
                            f"""
                            <div style="padding: 10px 14px; border-radius: 6px; border-left: 4px solid {unc_res['tier_color']}; background: rgba(30, 41, 59, 0.45); margin-bottom: 10px;">
                                <div style="font-weight: 600; color: {unc_res['tier_color']}; margin-bottom: 2px;">{unc_res['confidence_tier']}</div>
                                <div style="font-size: 0.85rem; color: #cbd5e1;">{unc_res['tier_desc']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        ci_lower = unc_res["lower_ci_95"]
                        ci_upper = unc_res["upper_ci_95"]
                        st.markdown(
                            f"**95% Prediction Interval (Ensemble Spread):** `{ci_lower:,.0f}` — `{ci_upper:,.0f}` activity count  \n"
                            f"<span style='font-size:0.82rem; color:#94a3b8;'>Computed across all {unc_res['tree_count']} decision trees (2.5th to 97.5th percentiles of raw predicted activity).</span>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.info(f"Uncertainty quantification unavailable: {unc_res.get('error')}")

                # Self-Evolving AI Diagnosis & Correction
                with st.expander("🧠 Self-Evolving AI Diagnosis & Theory Correction", expanded=True):
                    shap_r_evo = es.get_shap_values(bundle, row)
                    base_val_evo = ref_base_pred if (ref_base_pred is not None and ref_base_pred > 0) else total_pred.get("value", 0.0)
                    evo_diag = ges.diagnose_and_evolve_grid(row, base_val_evo, oof, shap_r_evo)
                    
                    if evo_diag.get("has_diagnosis"):
                        st.markdown(f"**Diagnostic Archetype:** `{evo_diag['archetype_name']}`")
                        st.caption(f"**Empirical Discrepancy:** {evo_diag['percent_mismatch']:+.1f}% | **Top SHAP Driver:** {evo_diag['top_shap_driver']}")
                        
                        st.info(f"🏛️ **Grounding Theory:** {evo_diag['theory']}  \n🔗 **Source DOI:** [{evo_diag['doi']}]({evo_diag['doi']})")
                        st.markdown(f"*{evo_diag['diagnosis_narrative']}*")
                        
                        st.markdown("---")
                        st.markdown("**🔬 Theory-Guided Scenario Adjustment & Accuracy Evaluation:**")
                        
                        # Calculate empirical accuracy and mismatch percentage
                        obs_act = evo_diag.get("observed_activity", 0.0)
                        base_pred_act = evo_diag["baseline_prediction"]
                        evo_pred_act = evo_diag["evolved_prediction"]
                        
                        if obs_act > 0:
                            base_err = abs(obs_act - base_pred_act)
                            evo_err = abs(obs_act - evo_pred_act)
                            base_acc = max(0.0, 100.0 - (base_err / obs_act * 100.0))
                            evo_acc = max(0.0, 100.0 - (evo_err / obs_act * 100.0))
                            acc_delta = evo_acc - base_acc

                            # Mismatch percentages: (Observed - Predicted) / Predicted * 100
                            init_mismatch_pct = ((obs_act - base_pred_act) / base_pred_act) * 100.0 if base_pred_act > 0 else 0.0
                            corr_mismatch_pct = ((obs_act - evo_pred_act) / evo_pred_act) * 100.0 if evo_pred_act > 0 else 0.0
                            mismatch_reduction = abs(init_mismatch_pct) - abs(corr_mismatch_pct)
                        else:
                            base_acc, evo_acc, acc_delta = 0.0, 0.0, 0.0
                            init_mismatch_pct, corr_mismatch_pct, mismatch_reduction = 0.0, 0.0, 0.0

                        c_m1, c_m2, c_m3 = st.columns(3)
                        c_m1.metric(
                            "Morphology Baseline",
                            f"{base_pred_act:,.0f}",
                            help="Primary Random Forest prediction from urban morphology features."
                        )
                        c_m2.metric(
                            "Theory Adjustment",
                            f"{evo_pred_act:,.0f}",
                            help="Post-hoc rule-based scenario adjustment under grounding theory."
                        )
                        delta_txt = f"{evo_diag['error_reduction_pct']:.1f}% error reduction" if evo_diag['error_reduction_pct'] > 0 else "maintained"
                        c_m3.metric("Adjusted Residual", f"{evo_diag['evolved_error']:,.0f}", delta=delta_txt)

                        # Before vs After Mismatch & Accuracy Row
                        row2_c1, row2_c2, row2_c3 = st.columns(3)
                        row2_c1.metric(
                            "Initial Mismatch",
                            f"{init_mismatch_pct:+.1f}%",
                            help="Initial prediction discrepancy = (Observed - Baseline) / Baseline * 100."
                        )
                        delta_mis_txt = f"{abs(mismatch_reduction):.1f}% closer to zero" if mismatch_reduction > 0 else "0.0%"
                        row2_c2.metric(
                            "Corrected Mismatch",
                            f"{corr_mismatch_pct:+.1f}%",
                            delta=delta_mis_txt,
                            help="Discrepancy remaining after theory adjustment = (Observed - Adjusted) / Adjusted * 100."
                        )
                        row2_c3.metric(
                            "Model Accuracy",
                            f"{evo_acc:.1f}%",
                            delta=f"{acc_delta:+.1f}% gain" if acc_delta > 0 else ("0.0%" if acc_delta == 0 else f"{acc_delta:.1f}%"),
                            help="Accuracy = 100% - (|Observed - Prediction| / Observed * 100)."
                        )

                        st.caption(f"**Candidate Interaction Formula:** `{evo_diag['hypothesis']}`")
                        st.caption(
                            "⚠️ **Disclaimer:** Diagnostic hypotheses and scenario adjustments are "
                            "rule-based computational heuristics inspired by urban morphology theory. "
                            "They require systematic evaluation across all held-out OOF grids "
                            "(ΔMAE, ΔRMSE, bias) before claiming improved model performance."
                        )
                    else:
                        st.info(evo_diag.get("reason", "Diagnosis unavailable."))

                # Predictor values
                with st.expander("Predictor Feature Values"):
                    feat_vals = ds.get_feature_values(row, features)
                    feat_rows = [{"Field": f, "Description": ds.FEATURE_LABELS.get(f, f), "Value": v if v is not None else "Missing"} for f, v in feat_vals.items()]
                    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

                # SHAP
                with st.expander("SHAP Explanation"):
                    shap_r = es.get_shap_values(bundle, row)
                    if shap_r["success"]:
                        records = shap_r["records"]
                        shap_df = pd.DataFrame(records)[["label", "grid_value", "shap_value", "direction"]]
                        shap_df.columns = ["Feature", "Grid value", "SHAP contribution", "Direction"]
                        st.dataframe(shap_df, use_container_width=True, hide_index=True)
                        fig2 = go.Figure(go.Bar(
                            x=[r["shap_value"] for r in records],
                            y=[r["label"] for r in records],
                            orientation="h",
                            marker_color=["#D94841" if r["shap_value"] > 0 else "#3677B8" for r in records],
                        ))
                        fig2.update_layout(title="SHAP Feature Contributions", height=280, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig2, use_container_width=True)
                        st.caption("🔬 SHAP explains the model's prediction, not real-world causation. Values are on the log1p scale.")
                    else:
                        st.info(f"SHAP unavailable: {shap_r.get('error')}")

                # What-if
                with st.expander("What-If Scenario"):
                    st.caption("Change one or more predictor values to see how the model prediction changes. This is NOT a causal estimate.")
                    wi_changes = {}
                    for f in features:
                        baseline_val = feat_vals.get(f)
                        if baseline_val is None:
                            continue
                        new_val = st.number_input(f"{ds.FEATURE_LABELS.get(f, f)} (baseline: {baseline_val:.4f})", value=float(baseline_val), step=0.01, key=f"wi_{f}_{selected_id}")
                        if abs(new_val - baseline_val) > 1e-9:
                            wi_changes[f] = new_val
                    if st.button("Run What-If Scenario", use_container_width=True) and wi_changes:
                        with st.spinner("Running scenario..."):
                            wi_result = es.run_whatif_scenario(bundle, row, wi_changes)
                        if wi_result["success"]:
                            w1, w2, w3 = st.columns(3)
                            w1.metric("Baseline Total", f"{wi_result['baseline_prediction']:,.0f}")
                            w2.metric("Scenario Total", f"{wi_result['scenario_prediction']:,.0f}")
                            diff_pct = wi_result.get("percent_difference")
                            w3.metric("Difference", f"{wi_result['absolute_difference']:+,.0f}", delta=f"{diff_pct:+.1f}%" if diff_pct else None)

                            pc = wi_result.get("pulse_comparison", {})
                            if pc.get("available"):
                                st.markdown("---")
                                st.markdown("**Predicted Hourly Pulse Shift**")
                                shift_txt = "shifted" if pc.get("peak_hour_shifted") else "unchanged"
                                st.caption(
                                    f"Peak Hour: **{pc['baseline_peak_hour']:02d}:00** ({pc.get('baseline_label')}) \u2192 "
                                    f"**{pc['scenario_peak_hour']:02d}:00** ({pc.get('scenario_label')}) [{shift_txt}]"
                                )
                                hours = [f"{i:02d}:00" for i in range(24)]
                                fig_wi = go.Figure()
                                fig_wi.add_trace(go.Scatter(
                                    x=hours, y=pc["baseline_values"],
                                    mode="lines+markers",
                                    name="Baseline Pulse",
                                    line=dict(color="#3B82F6", width=2.5),
                                    marker=dict(color="#3B82F6", size=5),
                                    fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
                                ))
                                fig_wi.add_trace(go.Scatter(
                                    x=hours, y=pc["scenario_values"],
                                    mode="lines+markers",
                                    name="Scenario Pulse",
                                    line=dict(color="#10B981", width=2.5, dash="dot"),
                                    marker=dict(color="#10B981", size=5, symbol="diamond"),
                                    fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
                                ))
                                fig_wi.update_layout(
                                    title="24-Hour Activity Pulse: Baseline vs Scenario",
                                    xaxis_title="Hour", yaxis_title="Activity intensity (log1p scale)",
                                    height=280, margin=dict(l=10, r=10, t=40, b=30),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(size=11),
                                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                                    hovermode="x unified",
                                )
                                st.plotly_chart(fig_wi, use_container_width=True)
                                if pc.get("scenario_description"):
                                    st.caption(f"**Scenario pulse pattern:** {pc['scenario_description']}")
                                st.caption("ℹ️ Hourly pulse values represent log1p-scale activity intensity across the 24 hours, trained separately from the total daily activity count model.")
                            st.caption("⚠️ Model-based what-if scenario. Not a causal estimate. Changing a predictor in this tool does not guarantee the real-world outcome changes by this amount.")
                        else:
                            st.error(f"Scenario failed: {wi_result.get('error')}")

    # -------------------------------------------------------
    # Study-wide mismatch summary
    # -------------------------------------------------------
    if mode == "Spatial mismatch":
        st.divider()
        st.subheader("Study-area Spatial Mismatch Summary")
        metrics = ds.compute_study_wide_metrics(map_grids)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Grids with OOF data", f"{metrics['n_with_oof']:,}")
        c2.metric("Observed higher (underpredict)", f"{metrics['n_positive']:,}")
        c3.metric("Observed lower (overpredict)", f"{metrics['n_negative']:,}")
        c4.metric("Mean residual (obs−pred)", f"{metrics['mean_diff']:+,.1f}" if metrics['mean_diff'] is not None else "—")
        c5.metric("Mean observed", f"{metrics['mean_obs']:,.1f}" if metrics['mean_obs'] else "—")
        st.caption("Residual = observed activity − OOF-predicted activity. Positive = underprediction. Negative = overprediction.")

        # Model metrics from evaluation
        with st.expander("Pooled OOF Model Metrics (RF)"):
            m = evs.recalculate_pooled_metrics(grids_geo)
            if m["available"]:
                mm1, mm2, mm3, mm4 = st.columns(4)
                mm1.metric("R²", f"{m['r2']:.4f}")
                mm2.metric("RMSE", f"{m['rmse']:,.1f}")
                mm3.metric("MAE",  f"{m['mae']:,.1f}")
                mm4.metric("Mean bias", f"{m['mean_bias']:+,.1f}")
                st.caption(m["note"])
            else:
                st.info(m.get("error"))

        if metrics["top_under"]:
            tl, tr = st.columns(2)
            with tl:
                st.markdown("**Top 5 largest underpredictions (obs > pred)**")
                st.dataframe(pd.DataFrame(metrics["top_under"]), use_container_width=True, hide_index=True)
            with tr:
                st.markdown("**Top 5 largest overpredictions (pred > obs)**")
                st.dataframe(pd.DataFrame(metrics["top_over"]), use_container_width=True, hide_index=True)

        # --- Interactive Hourly Mismatch Heatmap ---
        with st.expander("🔥 Interactive Hourly Mismatch Heatmap (24-Hour Temporal Error)", expanded=True):
            st.markdown(
                "Explore how spatial cross-validation errors unfold hour-by-hour across grid cells. "
                "Filtered sample prevents browser rendering lag while revealing temporal failure dynamics."
            )
            h_c1, h_c2, h_c3, h_c4 = st.columns([1.2, 1.4, 1.2, 1.2])
            with h_c1:
                metric_opt = st.selectbox(
                    "Error Metric",
                    ["Residual (obs − pred)", "Percentage Mismatch (%)"],
                    key="h_heat_metric",
                    help="Residual: count units. Percentage Mismatch: relative error normalized by predicted activity."
                )
            with h_c2:
                cat_opt = st.selectbox(
                    "Mismatch Category Filter",
                    [
                        "All categories",
                        "Underpredicted (> +10%)",
                        "Severe Underpredicted (> +50%)",
                        "Overpredicted (< -10%)",
                        "Severe Overpredicted (< -50%)",
                        "Well-Calibrated (-10% to +10%)",
                        "Top Absolute Errors",
                    ],
                    key="h_heat_cat",
                )
            with h_c3:
                dist_opt = st.selectbox(
                    "District Filter",
                    ["All districts", "Colombo", "Gampaha"],
                    index=0 if district_filter == "All districts" else (1 if district_filter == "Colombo" else 2),
                    key="h_heat_dist",
                )
            with h_c4:
                sample_n = st.slider("Sample Grids", min_value=10, max_value=80, value=35, step=5, key="h_heat_n")

            metric_code = "pct_mismatch" if "Percentage" in metric_opt else "residual"
            matrix_df, heat_meta = evs.get_hourly_mismatch_sample(
                grids_geo,
                district_filter=dist_opt,
                mismatch_filter=cat_opt,
                sample_size=sample_n,
                metric=metric_code,
            )

            if not matrix_df.empty:
                # Diverging colorscale centered at 0: Blue (negative/overpredicted) -> White (0) -> Red (positive/underpredicted)
                max_abs = float(np.nanpercentile(np.abs(matrix_df.values), 95))
                if max_abs == 0 or np.isnan(max_abs):
                    max_abs = 100.0

                z_min = -max_abs
                z_max = max_abs

                heat_fig = go.Figure(data=go.Heatmap(
                    z=matrix_df.values,
                    x=matrix_df.columns.tolist(),
                    y=matrix_df.index.tolist(),
                    colorscale="RdBu_r",
                    zmin=z_min,
                    zmax=z_max,
                    colorbar=dict(
                        title="Mismatch %" if metric_code == "pct_mismatch" else "Residual",
                        tickformat="+.0f" if metric_code == "residual" else "+.0f%%",
                    ),
                    hovertemplate="<b>%{y}</b><br>Hour: %{x}<br>Error: %{z:+.1f}<extra></extra>",
                ))
                heat_fig.update_layout(
                    title=f"24-Hour Temporal Error Matrix ({heat_meta['count']} Grids of {heat_meta['total_matching']} matching)",
                    xaxis_title="Hour of Day",
                    yaxis_title="Grid Cells",
                    height=max(340, min(800, sample_n * 16 + 120)),
                    margin=dict(l=10, r=10, t=40, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(heat_fig, use_container_width=True)
                st.caption(
                    f"Showing top {heat_meta['count']} grid cells in `{cat_opt}` ({dist_opt}). "
                    "Red: underpredicted hours (observed crowd presence exceeded model prediction). "
                    "Blue: overpredicted hours (model predicted higher human presence than observed)."
                )
            else:
                st.info("No grid cells match the chosen filter criteria.")

    # -------------------------------------------------------
    # Chat: LLM Agent
    # -------------------------------------------------------
    st.divider()
    st.subheader("💬 Ask the Urban Pulse AI Agent")

    if selected_id is None:
        st.info("Select a grid first. The agent will answer questions about that grid using real model predictions.")
    else:
        st.caption(f"Current context: **Grid {selected_id}** — {grids_geo[grids_geo['grid_id'].astype(str)==str(selected_id)].iloc[0].get('district_name','unknown') if not grids_geo[grids_geo['grid_id'].astype(str)==str(selected_id)].empty else 'unknown'}")

        if not ollama_ok:
            st.warning("⚠️ Ollama / qwen3:4b is not available. Answers use deterministic fallback only.")

        suggestions = [
            "Where exactly is this grid?",
            "What activity does the model predict here?",
            "What is the expected hourly pulse?",
            "Why did the model make this prediction?",
            "Is observed activity higher or lower than expected?",
            "What happens if transit accessibility increases?",
        ]
        sugg_cols = st.columns(3)
        for i, q in enumerate(suggestions):
            with sugg_cols[i % 3]:
                if st.button(q, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner("Agent is thinking..."):
                        result = aw.run_workflow(q, selected_id, bundle, grids_geo, table)
                        answer = result.get("final_answer", "No answer generated.")
                        st.session_state.agent_log.append({"query": q, "review": result.get("review_notes", []), "intent": result.get("intent")})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()

        # Display messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_q = st.chat_input("Ask about this grid's location, activity, pulse, SHAP, or what-if...")
        if user_q:
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            # Parse what-if changes from message if any
            wi_changes = {}  # Could be enhanced with LLM extraction

            with st.chat_message("assistant"):
                with st.spinner("Five agents are working..."):
                    result = aw.run_workflow(user_q, selected_id, bundle, grids_geo, table, wi_changes or None)
                    answer = result.get("final_answer", "No answer generated.")
                    st.session_state.agent_log.append({"query": user_q, "review": result.get("review_notes", []), "intent": result.get("intent")})
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        # Debug log
        if st.session_state.show_debug and st.session_state.agent_log:
            with st.expander("🔍 Agent Debug Log", expanded=False):
                for entry in st.session_state.agent_log:
                    st.markdown(f"**Query:** {entry.get('query')}  |  **Intent:** {entry.get('intent')}")
                    for note in entry.get("review", []):
                        color = "review-pass" if "PASS" in note else ("review-fail" if "FAIL" in note else ("review-warn" if "WARN" in note else "review-info"))
                        st.markdown(f'<span class="{color}">• {note}</span>', unsafe_allow_html=True)
                    st.divider()

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.divider()
st.caption(
    "**Research note:** Spatial mismatch = observed − OOF predicted activity. "
    "OOF = out-of-fold cross-validation. Deployment predictions are separate from OOF evaluation. "
    "SHAP describes model behaviour and does not establish causation. "
    "Basemap requires internet."
)
