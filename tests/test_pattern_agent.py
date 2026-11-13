"""Agent selection, evidence validation, failure and audit tests without model/network."""
from pathlib import Path
import sys
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_patterns import fixture
from pattern_agent_service import run_discovery, save_run, decision_schema


class Planner:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.contexts = []

    def invoke(self, messages):
        self.contexts.append(json.loads(messages[-1].content))
        value = next(self.choices)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(content=json.dumps(value))


def choice(action, **kwargs):
    return dict(action=action, reason='Examine measured activity differences.', **kwargs)


class AgentTests(unittest.TestCase):
    def test_live_schema_excludes_repeated_scans_and_invented_ids(self):
        schema = decision_schema({'evening_persistence': {}}, {'evening_persistence:1': {}}, {}, 4)
        options = {v['properties']['action']['const']: v for v in schema['oneOf']}
        self.assertNotIn('finish', options)
        self.assertNotIn('evening_persistence', options['scan']['properties']['pattern']['enum'])
        self.assertEqual(options['inspect']['properties']['finding_id']['enum'], ['evening_persistence:1'])
        self.assertNotIn('finding_ids', options['inspect']['properties'])
        final = decision_schema({'evening_persistence': {}}, {'evening_persistence:1': {}}, {'evening_persistence:1': {}}, 1)
        self.assertEqual(len(final['oneOf']), 1)
        self.assertEqual(final['oneOf'][0]['properties']['action']['const'], 'finish')

    def test_agent_selects_follows_up_and_reports_inspected_evidence(self):
        planner = Planner([choice('scan', pattern='evening_persistence'),
                           choice('inspect', finding_id='evening_persistence:1'),
                           choice('scan', pattern='matched_profiles'),
                           choice('inspect', finding_id='matched_profiles:1'),
                           choice('finish', finding_ids=['evening_persistence:1', 'matched_profiles:1'])])
        result = run_discovery(fixture(), 'Investigate evening activity and comparable morphology.', planner=planner)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(result['findings']), 2)
        self.assertEqual(result['backend'], 'injected_test_planner')
        self.assertIn('matched_location', result['inspections']['matched_profiles:1'])
        self.assertIn('evening_persistence:1', planner.contexts[2]['inspections'])
        self.assertNotIn('temporal_contrast', result['scans'])
        self.assertTrue(all(e['success'] for e in result['trace']))

    def test_fabricated_uninspected_and_repeat_requests_rejected(self):
        planner = Planner([choice('scan', pattern='evening_persistence'),
                           choice('finish', finding_ids=['evening_persistence:1']),
                           choice('inspect', finding_id='invented:999'),
                           choice('scan', pattern='evening_persistence')])
        result = run_discovery(fixture(), 'Find patterns', planner=planner, max_steps=4)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['findings'], [])
        self.assertTrue(all(not e['success'] for e in result['trace'][1:]))

    def test_model_failure_no_scripted_fallback(self):
        result = run_discovery(fixture(), 'Find patterns', planner=Planner([RuntimeError('offline')]))
        self.assertEqual(result['status'], 'model_unavailable')
        self.assertEqual(result['scans'], {})
        self.assertEqual(result['findings'], [])

    def test_no_findings_and_append_only_audit(self):
        df = fixture()
        from pattern_detection_service import HOURS
        df[HOURS] = 10.
        planner = Planner([choice('scan', pattern='evening_persistence'), choice('finish', finding_ids=[])])
        result = run_discovery(df, 'Look for evening persistence', planner=planner)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['findings'], [])
        with tempfile.TemporaryDirectory() as directory:
            with patch('pattern_agent_service.workspace_path', return_value=Path(directory)):
                path = save_run(result)
                self.assertEqual(json.loads(Path(path).read_text())['guidance'], result['guidance'])
                with self.assertRaises(FileExistsError):
                    save_run(result)

    def test_selected_detector_does_not_run_spatial_or_matching_tools(self):
        from pattern_detection_service import detect_patterns
        with patch('pattern_detection_service.cKDTree', side_effect=AssertionError('Unrequested spatial work')):
            result = detect_patterns(fixture(), patterns=['evening_persistence'])
        self.assertEqual(list(result['status']), ['evening_persistence'])
        self.assertTrue(all(f['pattern'] == 'evening_persistence' for f in result['findings']))

    def test_guided_ui_and_handoff_keep_audit_reference(self):
        from streamlit.testing.v1 import AppTest
        planner = Planner([choice('scan', pattern='evening_persistence'),
                           choice('inspect', finding_id='evening_persistence:1'),
                           choice('finish', finding_ids=['evening_persistence:1'])])
        run = run_discovery(fixture(), 'Look for evening persistence', planner=planner)
        app = AppTest.from_string('''
import sys
sys.path.insert(0, 'tests')
from test_patterns import fixture
from pattern_ui_service import render_pattern_panel
render_pattern_panel(fixture())
''').run(timeout=30)
        with patch('pattern_agent_service.run_discovery', return_value=run), patch('pattern_agent_service.save_run', return_value='test-audit.json'):
            app.button(key='run_pattern_agent').click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['pattern_scan']['agent_run_id'], run['run_id'])
        app.button(key='handoff_pattern').click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn(run['run_id'], app.session_state['phase5_investigation_question'])


if __name__ == '__main__':
    unittest.main()
