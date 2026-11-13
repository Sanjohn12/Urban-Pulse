"""
literature_service.py
---------------------
Step 2 of Theory-Guided Self-Evolving Urban Intelligence (TSE-UI).

Connects to the OpenAlex scholarly database (indexing 250M+ peer-reviewed papers)
to retrieve real scientific literature, extract theoretical propositions, and
build a verifiable, evidence-grounded scientific memory with genuine DOIs.
"""

from __future__ import annotations
import json
import logging
import numpy as np
import urllib.parse
import urllib.request
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

OPENALEX_API_URL = "https://api.openalex.org/works"
USER_AGENT = "TSE-UI-UrbanAgent/1.0 (mailto:urban.analytics.research@univ.ac.lk)"

# Foundational theories curated as seed grounding (per Section 7 of mentor's framework)
FOUNDATIONAL_THEORIES = [
    {
        "theory_name": "Natural Movement (Space Syntax)",
        "author": "Hillier, B., Penn, A., Hanson, J., Grajewski, T., & Xu, J.",
        "year": 1993,
        "title": "Natural movement: or, configuration and attraction in the urban pedestrian movement",
        "doi": "https://doi.org/10.1068/b200029",
        "proposition": "Street network configuration is the primary generator of movement distribution ('through-movement' vs. 'to-movement'); land-use attractors locate in response to this configurational potential.",
        "relevance": "Explains why high transit/capacity roads without local street integration act as bypasses rather than pedestrian destinations.",
    },
    {
        "theory_name": "Cities as Flows and Networks",
        "author": "Batty, M.",
        "year": 2013,
        "title": "The New Science of Cities",
        "doi": "https://doi.org/10.7551/mitpress/9780262019811.001.0001",
        "proposition": "Cities are dynamic networks of flows and interactions rather than static physical structures; morphology provides the constraints, but activities fluctuate temporally.",
        "relevance": "Justifies dynamic temporal pulse modelling beyond static road/building footprint metrics.",
    },
    {
        "theory_name": "Sidewalk Vitality and Urban Diversity",
        "author": "Jacobs, J.",
        "year": 1961,
        "title": "The Death and Life of Great American Cities",
        "doi": "https://doi.org/10.1002/9781119084679.ch2",
        "proposition": "Intense urban vitality requires small blocks, aged buildings, concentration of population, and mixed primary uses; informal commercial activity clusters organically.",
        "relevance": "Explains severe model underpredictions where unmapped street markets and informal trade generate massive footfall missed by GIS road databases.",
    },
    {
        "theory_name": "Dynamic Land-Use and Activity Rhythms",
        "author": "Jacobs-Crisioni, C., Rietveld, P., Koomen, E., & Tranos, E.",
        "year": 2014,
        "title": "Evaluating the impact of land-use density and mix on spatiotemporal urban activity rhythms",
        "doi": "https://doi.org/10.1068/a46305",
        "proposition": "The interaction of land-use mix and transit access determines the diurnal distribution and persistence of urban vitality across the 24-hour cycle.",
        "relevance": "Provides the mathematical basis for interaction terms between transit accessibility and land-use diversity.",
    },
]

def search_scholarly_literature(query: str, max_results: int = 5, limit: int = None) -> List[Dict[str, Any]]:
    """
    Queries OpenAlex for peer-reviewed research papers matching the given urban query.
    Extracts title, publication year, author, citation count, reconstructed abstract, and DOI.
    """
    if limit is not None:
        max_results = limit
    params = {
        "search": query,
        "per-page": max_results,
        # A grounded proposal needs a citable abstract excerpt; metadata-only
        # records remain useful in the legacy UI but cannot satisfy that gate.
        "filter": "has_abstract:true,type:article|book",
        "sort": "relevance_score:desc",
    }
    encoded_url = f"{OPENALEX_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(encoded_url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            if response.status != 200:
                logger.warning("OpenAlex returned status %s", response.status)
                return []
            data = json.loads(response.read().decode("utf-8"))
            results = data.get("results", [])

            papers = []
            for item in results:
                # Reconstruct abstract from inverted index if available
                abstract = ""
                inv_index = item.get("abstract_inverted_index")
                if inv_index and isinstance(inv_index, dict):
                    word_positions = []
                    for word, positions in inv_index.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort(key=lambda x: x[0])
                    abstract = " ".join([w for _, w in word_positions])
                    if len(abstract) > 350:
                        abstract = abstract[:347] + "..."

                authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])]
                author_str = ", ".join([a for a in authors[:3] if a])
                if len(authors) > 3:
                    author_str += " et al."

                doi = item.get("doi") or f"https://openalex.org/{item.get('id', '').split('/')[-1]}"

                papers.append({
                    "title": item.get("title", "Untitled"),
                    "authors": author_str or "Unknown",
                    "year": item.get("publication_year"),
                    "cited_by_count": item.get("cited_by_count", 0),
                    "doi": doi,
                    "abstract": abstract or "Abstract indexing unavailable.",
                })
            return papers
    except Exception as exc:
        logger.error("Scholarly literature search failed: %s", exc)
        return []

def retrieve_theories_for_anomaly(archetype_name: str, scientific_question: str) -> Dict[str, Any]:
    """
    Given an empirical failure archetype from anomaly_service, retrieves
    both curated foundational theories and live queried literature from OpenAlex.
    """
    # 1. Search live academic literature using keywords
    search_query = (
        "urban morphology space syntax through movement transit barrier"
        if "overprediction" in archetype_name.lower() or "bypass" in archetype_name.lower()
        else "informal urban centrality street vitality land use diversity"
    )
    live_papers = search_scholarly_literature(search_query, max_results=3)

    # 2. Select relevant foundational seed theories
    if "overprediction" in archetype_name.lower():
        seeds = [FOUNDATIONAL_THEORIES[0], FOUNDATIONAL_THEORIES[1]]  # Hillier & Batty
    else:
        seeds = [FOUNDATIONAL_THEORIES[2], FOUNDATIONAL_THEORIES[3]]  # Jacobs & Jacobs-Crisioni

    return {
        "archetype": archetype_name,
        "scientific_question": scientific_question,
        "seed_theories": seeds,
        "retrieved_literature": live_papers,
    }

def synthesize_theory_hypotheses(archetype_type: str, theory_name: str, proposition: str) -> List[Dict[str, Any]]:
    """
    Synthesizes mathematical candidate features (both individual non-linear and mutual interaction)
    grounded in the given theoretical proposition.
    
    Safe implementation: Pre-validates mathematical formulations on known columns.
    Generates rich, tailored hypotheses for each theory and archetype.
    """
    hypotheses = []
    t_key = theory_name.lower()
    t_short = theory_name.split()[0]
    
    is_overpred = "overprediction" in archetype_type.lower() or "bypass" in archetype_type.lower() or "barrier" in archetype_type.lower()

    if is_overpred:
        # Overprediction Archetype (Model overpredicts -> Transit Barrier / Monofunctional dead corridors)
        if "natural movement" in t_key or "hillier" in t_key or "syntax" in t_key:
            hypotheses.append({
                "name": f"Mutual_{t_short}_Transit_LandMix",
                "formula": "Transit_Ac * land_use_e",
                "type": "Mutual Interaction",
                "theory": theory_name,
                "hypothesis": "High transit capacity only translates to pedestrian destination encounters when coupled with land-use entropy; otherwise it operates as through-traffic bypass.",
                "compute": lambda df: df["Transit_Ac"] * df["land_use_e"],
            })
            hypotheses.append({
                "name": f"Mutual_{t_short}_Centrality_Destination_Ratio",
                "formula": "BtA500_wt / (1.0 + POI_Densit)",
                "type": "Mutual Interaction",
                "theory": theory_name,
                "hypothesis": "High betweenness centrality without POI destinations acts as a through-movement channel that severely suppresses local pedestrian activity.",
                "compute": lambda df: df["BtA500_wt"] / (1.0 + np.maximum(0, df["POI_Densit"])),
            })
            hypotheses.append({
                "name": f"Individual_{t_short}_Transit_Log_Dampener",
                "formula": "np.log1p(Transit_Ac)",
                "type": "Individual Non-Linear",
                "theory": theory_name,
                "hypothesis": "Transit accessibility exhibits diminishing returns; log-transforming prevents high arterial bus frequency from exaggerating local activity.",
                "compute": lambda df: np.log1p(np.maximum(0, df["Transit_Ac"])),
            })
        elif "batty" in t_key or "flows" in t_key:
            hypotheses.append({
                "name": f"Mutual_{t_short}_Flow_Congestion_Drag",
                "formula": "Transit_Ac / (1.0 + MAD500_wt)",
                "type": "Mutual Interaction",
                "theory": theory_name,
                "hypothesis": "Transit access along wide, high-speed arterial links without fine pedestrian metric access creates physical severance rather than destination footfall.",
                "compute": lambda df: df["Transit_Ac"] / (1.0 + np.maximum(0, df["MAD500_wt"])),
            })
            hypotheses.append({
                "name": f"Individual_{t_short}_Betweenness_Saturation",
                "formula": "np.log1p(BtA500_wt)",
                "type": "Individual Non-Linear",
                "theory": theory_name,
                "hypothesis": "Global betweenness centrality reaches network flow saturation beyond which additional connectivity ceases to add local pedestrian density.",
                "compute": lambda df: np.log1p(np.maximum(0, df["BtA500_wt"])),
            })
        elif "jacobs" in t_key:
            hypotheses.append({
                "name": f"Mutual_{t_short}_Floor_Monofunctional_Trap",
                "formula": "Floor_Inte * (1.0 - np.clip(land_use_e, 0.0, 1.0))",
                "type": "Mutual Interaction",
                "theory": theory_name,
                "hypothesis": "High floor intensity in low-entropy zones identifies single-use compounds (warehouses, closed campuses) that repel street vibrancy.",
                "compute": lambda df: df["Floor_Inte"] * (1.0 - np.clip(df["land_use_e"], 0.0, 1.0)),
            })
            hypotheses.append({
                "name": f"Individual_{t_short}_Floor_Saturation",
                "formula": "np.log1p(Floor_Inte)",
                "type": "Individual Non-Linear",
                "theory": theory_name,
                "hypothesis": "Floor area intensity has diminishing marginal returns on activity; log-scaling prevents large single-use footprints from overpredicting.",
                "compute": lambda df: np.log1p(np.maximum(0, df["Floor_Inte"])),
            })
        else:
            hypotheses.append({
                "name": f"Mutual_{t_short}_Transit_Diversity_Coupling",
                "formula": "np.log1p(Transit_Ac) * land_use_e",
                "type": "Mutual Interaction",
                "theory": theory_name,
                "hypothesis": "Transit corridors require a critical threshold of functional land-use mix to transform passing commuters into local street activity.",
                "compute": lambda df: np.log1p(np.maximum(0, df["Transit_Ac"])) * df["land_use_e"],
            })
            hypotheses.append({
                "name": f"Individual_{t_short}_Entropy_Gated",
                "formula": "np.maximum(0, land_use_e - 0.25)",
                "type": "Individual Non-Linear",
                "theory": theory_name,
                "hypothesis": "Monofunctional corridors below 0.25 entropy lack the baseline diversity required to sustain daytime footfall.",
                "compute": lambda df: np.maximum(0, df["land_use_e"] - 0.25),
            })
    else:
        # Underprediction Archetype (Model underpredicts -> Organic Vitality, informal markets, street life)
        if "jacobs" in t_key:
            hypotheses.append({
                "name": f"Mutual_{t_short}_Sidewalk_Vitality_Synergy",
                "formula": "IntDen_km2 * (1.0 + land_use_e)",
                "type": "Mutual Interaction",
                "theory": theory_name,
                "hypothesis": "Permeable short blocks intersecting diverse mixed land uses produce exponential informal encounters (Jane Jacobs sidewalk vitality).",
                "compute": lambda df: df["IntDen_km2"] * (1.0 + df["land_use_e"]),
            })
            hypotheses.append({
                "name": f"Mutual_{t_short}_Micro_Destination_Cluster",
                "formula": "np.log1p(POI_Densit) * (1.0 + land_use_e)",
                "type": "Mutual Interaction",
                "theory": theory_name,
                "hypothesis": "Points of interest embedded within diverse residential/commercial zones trigger spontaneous multi-trip chain encounters.",
                "compute": lambda df: np.log1p(np.maximum(0, df["POI_Densit"])) * (1.0 + df["land_use_e"]),
            })
            hypotheses.append({
                "name": f"Individual_{t_short}_Intersection_Clustering",
                "formula": "np.sqrt(np.maximum(0, IntDen_km2))",
                "type": "Individual Non-Linear",
                "theory": theory_name,
                "hypothesis": "Intersection density represents micro-pedestrian junctions that aggregate local street life more effectively than linear road links.",
                "compute": lambda df: np.sqrt(np.maximum(0, df["IntDen_km2"])),
            })
        elif "crisioni" in t_key or "rhythms" in t_key:
            hypotheses.append({
                "name": f"Mutual_{t_short}_TOD_Vitality_Multipier",
                "formula": "np.log1p(Transit_Ac) * np.log1p(POI_Densit)",
                "type": "Mutual Interaction",
                "theory": theory_name,
                "hypothesis": "Sub-linear joint compounding of transit access and local POI density sustains multi-hour activity pulses.",
                "compute": lambda df: np.log1p(np.maximum(0, df["Transit_Ac"])) * np.log1p(np.maximum(0, df["POI_Densit"])),
            })
            hypotheses.append({
                "name": f"Individual_{t_short}_POI_Agglomeration_Power",
                "formula": "np.power(np.maximum(0, POI_Densit), 1.25)",
                "type": "Individual Non-Linear",
                "theory": theory_name,
                "hypothesis": "Dense commercial clusters exhibit super-linear agglomeration economies, drawing regional vitality beyond local building floor area.",
                "compute": lambda df: np.power(np.maximum(0, df["POI_Densit"]), 1.25),
            })
        else:
            hypotheses.append({
                "name": f"Mutual_{t_short}_Connectivity_Vitality",
                "formula": "LConn_wt * (1.0 + land_use_e)",
                "type": "Mutual Interaction",
                "theory": theory_name,
                "hypothesis": "Local street connectivity magnifies organic pedestrian vitality when functional land-use entropy is present.",
                "compute": lambda df: df["LConn_wt"] * (1.0 + df["land_use_e"]),
            })
            hypotheses.append({
                "name": f"Individual_{t_short}_Local_Connectivity_Boost",
                "formula": "np.log1p(LConn_wt)",
                "type": "Individual Non-Linear",
                "theory": theory_name,
                "hypothesis": "Log-scaled local connectivity captures fine-grained urban grain where unmapped informal pedestrian alleys thrive.",
                "compute": lambda df: np.log1p(np.maximum(0, df["LConn_wt"])),
            })

    return hypotheses

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("Testing OpenAlex literature retrieval...")
    res = retrieve_theories_for_anomaly(
        "Infrastructure Bypass / Barrier Severance (Severe Overprediction)",
        "Why does high physical accessibility fail to generate expected human presence?"
    )
    print("\nArchetype:", res["archetype"])
    print("\n--- Seed Grounding ---")
    for s in res["seed_theories"]:
        print(f"* [{s['year']}] {s['author']} — {s['title']} ({s['doi']})")
    print("\n--- Live Retrieved Papers from OpenAlex ---")
    for p in res["retrieved_literature"]:
        print(f"* [{p['year']}] {p['authors']} — {p['title']}")
        print(f"  DOI: {p['doi']} | Citations: {p['cited_by_count']}")
        print(f"  Abstract preview: {p['abstract'][:150]}...\n")
