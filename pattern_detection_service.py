"""Read-only, exploratory detectors. Observed outcomes never become model inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

FEATURES = ['LConn_wt', 'MAD500_wt', 'BtA500_wt', 'Floor_Inte',
            'POI_Densit', 'Transit_Ac', 'IntDen_km2', 'land_use_e']
HOURS = [f'activity_{h}' if h < 10 else f'activity{h}' for h in range(24)]
QUESTIONS = {
    'morphology_mismatch': 'When does connectivity or building intensity fail to explain destination activity?',
    'temporal_contrast': 'What urban functions explain commuting peaks or strong day-night contrasts?',
    'evening_persistence': 'How might mixed destinations explain sustained evening activity?',
    'matched_profiles': 'Why do locations with similar morphology have different daily rhythms?',
    'spatial_context': 'Could neighbourhood context or omitted spatial processes explain residual patterns?',
}
SEARCHES = {
    'morphology_mismatch': 'urban network connectivity destination attraction activity',
    'temporal_contrast': 'urban activity temporal rhythms commuting land use',
    'evening_persistence': 'urban mixed land use evening activity duration',
    'matched_profiles': 'urban morphology functional differences temporal activity profiles',
    'spatial_context': 'urban activity spatial dependence neighbourhood model residuals',
}


def detect_patterns(table, grid_id=None, patterns=None):
    """Scan the supplied cohort only; grid_id limits returned findings, not reference thresholds.

    Fixed exploratory rules are disclosed in every finding. No causal classification,
    training, file writes, or external calls. Missing inputs yield unavailable detectors.
    """
    active = set(QUESTIONS if patterns is None else patterns)
    if not active or not active <= set(QUESTIONS):
        raise ValueError('Choose one or more supported patterns')
    df = table.copy().reset_index(drop=True)
    if 'grid_id' not in df or df.grid_id.isna().any():
        raise ValueError('Nonmissing grid IDs are required')
    df['grid_id'] = df.grid_id.astype(str).str.replace(r'\.0$', '', regex=True)
    if df.grid_id.duplicated().any():
        raise ValueError('Grid IDs must be unique')
    selected = None if grid_id is None else str(grid_id).removesuffix('.0')
    if selected is not None and selected not in set(df.grid_id):
        raise ValueError('Selected grid is outside the supplied cohort')
    findings, status = [], {}

    def numeric(columns):
        return df.reindex(columns=columns).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

    def add(kind, i, evidence, rule):
        if kind in active and (selected is None or df.at[i, 'grid_id'] == selected):
            findings.append(dict(pattern=kind, grid_id=df.at[i, 'grid_id'], evidence=evidence,
                                 rule=rule, research_question=QUESTIONS[kind], literature_query=SEARCHES[kind]))

    obs, pred = numeric(['oof_observed_total', 'oof_rf_predicted_total']).to_numpy().T
    residual = obs - pred
    valid_oof = np.isfinite(obs) & np.isfinite(pred) & (obs >= 0) & (pred > 0)
    morph = numeric(['LConn_wt', 'Floor_Inte'])
    thresholds = morph.quantile(.75)
    status['morphology_mismatch'] = 'available' if valid_oof.any() and morph.notna().any().any() else 'unavailable: morphology or OOF missing'
    for i in (np.flatnonzero(valid_oof & (obs < .5 * pred)) if 'morphology_mismatch' in active else []):
        high = [c for c in morph if pd.notna(morph.at[i, c]) and morph.at[i, c] > thresholds[c]]
        if high:
            add('morphology_mismatch', i, {'observed': float(obs[i]), 'oof_predicted': float(pred[i]),
                'high_features': {c: float(morph.at[i, c]) for c in high},
                'reference_q75': {c: float(thresholds[c]) for c in high}},
                'At least one morphology value > cohort Q75; observed < 50% of OOF prediction.')

    h = numeric(HOURS).to_numpy()
    totals = h.sum(axis=1)
    valid_h = np.isfinite(h).all(axis=1) & (h >= 0).all(axis=1) & (totals > 0)
    shares = np.divide(h, totals[:, None], out=np.zeros_like(h), where=valid_h[:, None])
    for kind in ['temporal_contrast', 'evening_persistence']:
        status[kind] = 'available' if valid_h.any() else 'unavailable: complete positive observed hourly profile required'
    for i in (np.flatnonzero(valid_h) if active & {'temporal_contrast', 'evening_persistence'} else []):
        v = h[i]
        peaks = [j for j in range(1, 23) if v[j] > v[j-1] and v[j] >= v[j+1] and v[j] >= .3*v.max()]
        twin = any(6 <= j <= 9 for j in peaks) and any(16 <= j <= 19 for j in peaks)
        day_share = float(shares[i, 7:19].sum())
        if twin or day_share >= .8:
            add('temporal_contrast', i, {'morning_evening_peaks': bool(twin), 'peak_hours': peaks,
                'day_share_07_18': day_share},
                'Separated local peaks in 06-09 and 16-19, each >=30% daily maximum; OR 07-18 share >=80%.')
        evening = float(shares[i, 18:24].sum())
        run = longest = 0
        for hour_active in shares[i, 18:24] >= 1/24:
            run = run+1 if hour_active else 0
            longest = max(longest, run)
        if evening >= .3 and longest >= 4:
            add('evening_persistence', i, {'evening_share': evening, 'consecutive_hours': longest},
                '18-23 share >=30%; at least four consecutive evening hours each >=1/24 daily activity.')

    x = numeric(FEATURES)
    good = x.notna().all(axis=1).to_numpy() & valid_h
    positions = np.flatnonzero(good)
    status['matched_profiles'] = 'unavailable: need two complete morphology and hourly records'
    if 'matched_profiles' in active and len(positions) >= 2:
        values = x.iloc[positions].to_numpy()
        scale = values.std(axis=0)
        scale[scale == 0] = 1
        z = (values-values.mean(axis=0))/scale
        distances, neighbours = cKDTree(z).query(z, k=2)
        status['matched_profiles'] = 'available'
        for a, i in enumerate(positions):
            b = next((int(b) for b in neighbours[a] if b != a), None)
            if b is None:
                continue
            j = positions[b]
            distance = float(np.linalg.norm(z[a]-z[b])/np.sqrt(len(FEATURES)))
            tv = float(np.abs(shares[i]-shares[j]).sum()/2)
            if distance <= .5 and tv >= .3:
                add('matched_profiles', i, {'matched_grid_id': df.at[j, 'grid_id'],
                    'standardized_rms_distance': distance, 'profile_total_variation': tv},
                    'Nearest morphology neighbour; standardized RMS distance <=0.5 and profile total variation >=0.3. Observational match, not causal control.')

    # Projected metre coordinates only, never latitude/longitude degrees.
    coords = numeric(['centroid_x', 'centroid_y'])
    if coords.isna().all().all() and {'left', 'right', 'top', 'bottom'} <= set(df):
        bounds = numeric(['left', 'right', 'top', 'bottom'])
        coords = pd.DataFrame({'x': (bounds.left+bounds.right)/2, 'y': (bounds.top+bounds.bottom)/2})
    good = coords.notna().all(axis=1).to_numpy() & valid_oof
    positions = np.flatnonzero(good)
    status['spatial_context'] = 'unavailable: need >=4 valid projected locations and nonconstant residuals'
    spatial = {}
    if 'spatial_context' in active and len(positions) >= 4 and np.std(residual[positions]) > 0:
        points = coords.iloc[positions].to_numpy()
        pairs = cKDTree(points).query_pairs(750, output_type='ndarray')
        if len(pairs):
            rows, cols = np.concatenate([pairs[:, 0], pairs[:, 1]]), np.concatenate([pairs[:, 1], pairs[:, 0]])
            weights = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(points), len(points)))
            r = residual[positions]; r = r-r.mean()
            factor = len(r)/weights.sum()
            def moran(v):
                return float(factor * (v @ (weights @ v))/(v @ v))
            measured = moran(r)
            rng = np.random.default_rng(42)
            simulated = np.array([moran(rng.permutation(r)) for _ in range(199)])
            expected = -1/(len(r)-1)
            p = float((1+np.sum(np.abs(simulated-expected) >= abs(measured-expected)))/200)
            spatial = {'morans_i': measured, 'permutation_p_two_sided': p, 'permutations': 199,
                       'radius_m': 750, 'n_grids': len(r), 'isolates': int(np.sum(weights.sum(axis=1).A1 == 0)),
                       'note': 'Global exploratory test; does not identify significant local clusters or a causal mechanism.'}
            status['spatial_context'] = 'available'
            connectivity = numeric(['LConn_wt']).iloc[:, 0]
            low, high = connectivity.quantile(.25), connectivity.quantile(.75)
            activity_high = float(np.quantile(obs[valid_oof], .75))
            for a, i in enumerate(positions):
                nearby = positions[weights[a].indices]
                mean = connectivity.iloc[nearby].mean()
                if len(nearby) and connectivity.iloc[i] < low and obs[i] > activity_high and mean > high:
                    add('spatial_context', i, {'observed': float(obs[i]), 'local_connectivity': float(connectivity.iloc[i]),
                        'neighbour_mean_connectivity': float(mean), 'neighbour_count': len(nearby)},
                        'Local connectivity <Q25, activity >Q75, neighbour mean connectivity >Q75 within 750m. No spillover causality established.')
    return {'version': '1.0', 'scope': 'exploratory_observed_patterns', 'cohort_size': len(df),
            'selected_grid_id': selected, 'status': {k: v for k, v in status.items() if k in active}, 'findings': findings, 'spatial_summary': spatial,
            'limitations': 'Fixed exploratory thresholds; overlapping patterns; missing inputs excluded. Findings are associations, not mechanisms. No training or model replacement.'}
