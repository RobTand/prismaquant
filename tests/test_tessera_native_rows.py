"""Native caller boundaries; real lifecycle proof is retained beside migration evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prismaquant import flux_jobs as jobs
from prismaquant import tessera_campaign_rows as rows
from tools import dispatch_tessera_campaign as campaign
from tools.tessera_fleet import common, status


class NativeRowsTests(unittest.TestCase):
    def row(self, **extra):
        return {'cwd': '/mnt/shared/source', 'argv': ['/bin/true'],
                'demand': {'cpu': 1, 'mem_gb': 1}, 'env': {}, 'tags': [],
                'timeout_s': 60, **extra}

    def test_preflight_all_rows_before_launch(self):
        with patch.object(jobs, 'validate_job', return_value={}), patch.object(jobs, 'submit_job') as submit:
            with self.assertRaisesRegex(ValueError, 'gpus'):
                rows.submit_rows([self.row(), self.row(demand={'cpu': 1, 'mem_gb': 1, 'gpu': 1})],
                                 '/mnt/shared/never-created')
            submit.assert_not_called()

    def test_contracts_refuse_before_native_submission(self):
        cases = ({'data_manifest': 'inputs.json'}, {'progress_phases': ['pricing=900']},
                 {'residency': 'stage'}, {'produced_outputs': []}, {'pilot': {}},
                 {'container_images': ['pinned']}, {'gpu_memory_gb': 1},
                 {'priority': -10}, {'made_up': 1})
        with patch.object(jobs, 'submit_job') as submit:
            for extra in cases:
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    rows.submit_rows([self.row(**extra)], '/mnt/shared/never-created')
            submit.assert_not_called()

    def test_missing_finite_walltime_refuses(self):
        row = self.row()
        del row['timeout_s']
        with self.assertRaises(KeyError):
            rows.validate_row(row)

    def test_cpu_and_memory_demands_are_not_clamped(self):
        for demand in ({'cpu': 5, 'mem_gb': 1}, {'cpu': 1, 'mem_gb': 17},
                       {'cpu': 1, 'mem_gb': 1, 'ssd_gb': 2}):
            with self.subTest(demand=demand), self.assertRaises(ValueError):
                rows.validate_row(self.row(demand=demand))

    def test_unknown_or_combined_placement_refuses(self):
        for tags in (['gfx1201'], ['gb10', 'sparky']):
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                rows.validate_row(self.row(tags=tags))

    def test_local_source_refuses_even_for_cpu(self):
        with self.assertRaisesRegex(ValueError, 'shared'):
            rows.validate_row(self.row(cwd='/home/rob/prismaquant'))

    def test_application_environment_never_dropped(self):
        with patch.object(jobs, '_source_identity', return_value={}):
            for env in ({'SCIENTIFIC_SETTING': 'a'}, {'PRISMABUILD_ACTION_KEY': 'x'},
                        {'API_TOKEN': 'x'}, {'OMP_NUM_THREADS': '4'}):
                with self.subTest(env=env), self.assertRaisesRegex(ValueError, 'environment'):
                    rows.validate_row(self.row(env=env))

    def test_unsupported_retry_configuration_is_not_ignored(self):
        with self.assertRaises(ValueError):
            rows.validate_row(self.row(retry={'max_attempts': 2}))

    def test_saved_producer_required(self):
        submission = jobs.FluxSubmission('1', Path('/missing'), 'sha')
        with patch.object(jobs, 'load_submission', side_effect=jobs.SubmissionUnknown('unknown')):
            with self.assertRaises(jobs.SubmissionUnknown):
                rows.submit_rows([self.row()], '/mnt/shared/never-created', afterok=[submission])

    def test_query_error_never_passes_merge(self):
        for outcome in ('failed', 'running', 'unknown', 'canceled', 'dependency-failed', 'cache_hit'):
            with patch.object(rows, 'collect_rows', return_value=[{'status': outcome,
                    'payload_exit': 0, 'query_error': None}]):
                with self.subTest(outcome=outcome), self.assertRaisesRegex(RuntimeError, 'merge'):
                    rows.require_native_success('/unused', 1)
        with patch.object(rows, 'collect_rows', return_value=[{'status': 'succeeded',
                'payload_exit': 0, 'query_error': 'transport unavailable'}]):
            with self.assertRaises(RuntimeError):
                rows.require_native_success('/unused', 1)

    def test_positive_native_completion_keeps_scientific_validation_separate(self):
        fact = {'status': 'succeeded', 'payload_exit': 0, 'query_error': None}
        with patch.object(rows, 'collect_rows', return_value=[fact]):
            self.assertEqual(rows.require_native_success('/unused', 1), [fact])
            with self.assertRaises(RuntimeError):
                rows.require_native_success('/unused', 2)

    def test_receipt_written_early_cannot_override_payload_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps([self.row()]))
            receipt = root / 'receipts.json'
            receipt.write_text('{"rows":[{"status":"succeeded"}],"returncode":0}')
            failure = {'status': 'failed', 'payload_exit': 7, 'query_error': None}
            with patch.object(rows, 'submit_rows'), patch.object(rows, 'collect_rows', return_value=[failure]):
                self.assertEqual(campaign._flux_campaign(manifest, wait_s=0, receipts=receipt), 1)
            self.assertEqual(json.loads(receipt.read_text())['rows'], [failure])

    def test_merge_intent_cannot_substitute_another_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            (state / 'manifest.json').write_text(json.dumps({'rows': [self.row()]}))
            with self.assertRaisesRegex(RuntimeError, 'scientific manifest'):
                rows.require_native_success(state, 1, rows=[self.row(argv=['/bin/false'])])

    def test_status_ignores_historical_pb_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / '.pb-state'
            state.mkdir()
            (state / 'export-submissions.json').write_text('{"rows":[{"action_key":"old"}]}')
            with patch.object(status, 'collect_rows') as collect:
                self.assertEqual(status.main(['--workspace', temporary]), status.EXIT_NOTHING)
                collect.assert_not_called()

    def test_read_manifest_is_immutable_even_when_execution_refuses(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / 'read-manifest.json'
            original = b'{"row":"scientific-inputs"}\n'
            campaign._publish_intent_bytes(manifest, original)
            campaign._publish_intent_bytes(manifest, original)
            with self.assertRaisesRegex(RuntimeError, 'fresh workspace'):
                campaign._publish_intent_bytes(manifest, b'{"row":"changed-inputs"}\n')
            self.assertEqual(manifest.read_bytes(), original)

    def test_read_manifest_symlink_never_rebinds_intent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'target'
            target.write_bytes(b'exact-inputs')
            manifest = root / 'manifest'
            manifest.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, 'fresh workspace'):
                campaign._publish_intent_bytes(manifest, b'exact-inputs')
            self.assertTrue(manifest.is_symlink())
            self.assertEqual(target.read_bytes(), b'exact-inputs')

    def test_campaign_status_never_submits(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(jobs, 'submit_job', side_effect=AssertionError('submit')):
                self.assertEqual(campaign.main(['status', '--workspace', temporary]), status.EXIT_NOTHING)

    def test_stage_barrier_refuses_missing_rows_and_failed_payload(self):
        failure = {'status': 'failed', 'payload_exit': 7, 'query_error': None}
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(rows, 'submit_rows'), patch.object(rows, 'collect_rows', return_value=[failure]):
                with self.assertRaisesRegex(ValueError, 'downstream merge'):
                    common.run_stage([self.row()], Path(temporary), 'encode', wait_s=0)
            self.assertEqual(json.loads((Path(temporary) / '.flux-state/encode/endings.json').read_text())['rows'], [failure])


if __name__ == '__main__':
    unittest.main()
