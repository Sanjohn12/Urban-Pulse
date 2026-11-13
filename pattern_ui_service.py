"""On-demand detector panel within the existing anomaly tab."""
import json
import pandas as pd
import streamlit as st
from pattern_detection_service import detect_patterns


@st.cache_data(show_spinner=False)
def scan(table):
    return detect_patterns(table)


def render_pattern_panel(table):
    st.markdown('### Observed Pattern Detection')
    st.caption('Exploratory rules using observed activity. Patterns may overlap; explanations require investigation.')
    st.text_area('Research guidance for the AI agent',
        value='Investigate where urban morphology does not explain observed activity. Choose relevant pattern tools, inspect candidate evidence, and report supported findings without claiming causes.',
        key='pattern_agent_guidance', max_chars=2000)
    if st.button('Let the AI agent investigate patterns', key='run_pattern_agent'):
        from pattern_agent_service import run_discovery, save_run
        with st.status('Agent choosing analyses and examining evidence...', expanded=True) as progress:
            try:
                run = run_discovery(table, st.session_state.pattern_agent_guidance,
                    progress=lambda e: st.write(f"Step {e['step']}: " +
                        (e.get('decision', {}).get('action', 'model request')) +
                        (' completed' if e['success'] else ': ' + e['error'])))
                st.session_state.pattern_agent_run = run
                st.session_state.pattern_agent_log = save_run(run)
                if run['status'] == 'completed':
                    st.session_state.pattern_scan = {
                        'source': 'agent_directed', 'agent_run_id': run['run_id'],
                        'status': {k: v['status'] for k, v in run['scans'].items()},
                        'scan_counts': {k: v['count'] for k, v in run['scans'].items()},
                        'findings': [dict(f, agent_run_id=run['run_id']) for f in run['findings']],
                        'spatial_summary': run['scans'].get('spatial_context', {}).get('spatial_summary', {}),
                        'limitations': run['limitations']}
                progress.update(label='Agent run: ' + run['status'],
                                state='complete' if run['status'] == 'completed' else 'error')
            except Exception as exc:
                st.error(f'Agent run failed: {exc}')
                progress.update(label='Agent run failed', state='error')
    agent_run = st.session_state.get('pattern_agent_run')
    if agent_run:
        st.caption(f"Last AI run: {agent_run['run_id']} — {agent_run['status']}. Backend: {agent_run['backend']}.")
        with st.expander('Agent decisions and calculated evidence'):
            st.caption('Model-written reasons are unverified decision notes. Findings and scan counts come from validated tool results.')
            st.json(agent_run)
        st.download_button('Download agent audit log', json.dumps(agent_run, indent=2),
                           file_name=agent_run['run_id']+'.json', mime='application/json')
        if agent_run['status'] != 'completed':
            st.warning('This AI run did not complete. It is not an agent-detected finding report. Any previous results below retain their original source.')
    if st.button('Scan existing data for all five patterns', key='scan_patterns'):
        with st.spinner('Calculating observed patterns and spatial diagnostics...'):
            try:
                st.session_state.pattern_scan = scan(table)
            except Exception as exc:
                st.error(f'Pattern scan failed: {exc}')
    result = st.session_state.get('pattern_scan')
    if not result:
        return
    findings = result['findings']
    st.info('Source: AI-selected and inspected findings (researcher-defined tools).' if result.get('source') == 'agent_directed'
            else 'Source: direct rule-based scan, not an AI-directed investigation.')
    st.dataframe(pd.DataFrame([{'pattern': k, 'status': v,
        'matching_grids': result.get('scan_counts', {}).get(k, sum(f['pattern'] == k for f in findings)),
        'findings_in_this_report': sum(f['pattern'] == k for f in findings)} for k, v in result['status'].items()]),
        hide_index=True, use_container_width=True)
    st.caption(result['limitations'])
    if result['spatial_summary']:
        with st.expander('Global spatial residual test'):
            st.json(result['spatial_summary'])
    if not findings:
        st.info('No findings met the configured rules. See availability above.')
        return
    index = st.selectbox('Finding to investigate', range(len(findings)),
        format_func=lambda i: f"Grid {findings[i]['grid_id']} — {findings[i]['pattern']}")
    finding = findings[index]
    st.json(finding)
    row = table[table.grid_id.astype(str).str.replace(r'\.0$', '', regex=True) == finding['grid_id']].iloc[0]
    from pattern_detection_service import HOURS
    if all(c in row for c in HOURS):
        values = pd.to_numeric(row[HOURS], errors='coerce')
        st.line_chart(pd.DataFrame({'Observed activity': values.to_numpy()}, index=range(24)))
    if st.button('Use finding in Literature and Investigation', key='handoff_pattern'):
        st.session_state.selected_grid_id = finding['grid_id']
        st.session_state.lit_search_query = finding['literature_query']
        st.session_state.input_lit_query = finding['literature_query']
        st.session_state.cached_lit_works = None
        st.session_state.phase5_investigation_question = (
            finding['research_question'] + '\nMeasured exploratory finding: ' + json.dumps(finding))
        st.session_state.investigation_engine = None
        st.session_state.investigation_request_id = None
        st.session_state.messages = []
        st.success('Finding selected. Open Literature to search, or Hypothesis Testing to start the investigation.')
