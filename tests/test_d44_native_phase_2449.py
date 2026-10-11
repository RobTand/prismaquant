"""CPU behavior tests for native completion, lifetime cost and disk admission."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.d44_native import next_wave as wave
import test_d44_native_controller_2449 as policy


class PhaseTests(unittest.TestCase):
    setUp = policy.ControllerTests.setUp
    admit = policy.ControllerTests.admit
    record = policy.ControllerTests.record

    def test_completion_resumes_without_another_invocation(self):
        self.controller.autonomous = True
        waited = []

        def complete():
            key = self.calls[-1]
            waited.append(key)
            self.statuses[key] = ('done', None)
            return True

        self.controller.await_completion = complete
        self.assertEqual(self.controller.run(), 0)
        self.assertEqual(self.calls, self.controller.keys)
        self.assertEqual(waited, self.controller.keys)
        policy.ControllerTests.setUp(self)
        self.controller.autonomous = True
        def fail_child():
            self.statuses[self.calls[-1]] = ('failed', None)
            return True
        self.controller.await_completion = fail_child
        self.assertEqual(self.controller.run(), 1)
        self.assertEqual(self.calls, self.controller.keys[:1], 'A failed child must not trigger a scientific retry.')

    def test_disk_failure_prevents_child_record(self):
        with patch.object(wave, 'routed_diskcheck', return_value={'pass': False}, create=True):
            self.assertEqual(self.controller.run(), 1)
        self.assertEqual(self.calls, [])

    def test_fresh_disk_check_precedes_every_child(self):
        checks = []
        self.controller.autonomous = True
        self.controller.await_completion = lambda: self.finish()

        def disk(*args, **kwargs):
            checks.append(len(self.calls))
            return {'pass': True}

        with patch.object(wave, 'routed_diskcheck', side_effect=disk, create=True):
            self.assertEqual(self.controller.run(), 0)
        self.assertEqual(checks, list(range(4)))
        policy.ControllerTests.setUp(self)
        def ship_during_disk(*args, **kwargs):
            policy.ControllerTests.ship(self, priority=10)
            return {'pass': True}
        with patch.object(wave, 'routed_diskcheck', side_effect=ship_during_disk):
            self.assertEqual(self.controller.run(), 3)
        self.assertEqual(self.calls, [], 'Ship work that arrives during D1 must stop admission.')

    def finish(self):
        self.statuses[self.calls[-1]] = ('done', None)
        return True

    def test_incomplete_disk_result_prevents_child_record(self):
        with patch.object(wave, 'routed_diskcheck', side_effect=ValueError('Incomplete D1 evidence'), create=True):
            self.assertEqual(self.controller.run(), 1)
        self.assertEqual(self.calls, [])
        import datetime
        child = {'params': {'placement': {'required_tags': ['x86']}}}
        body = {'tool': 'fleet-diskcheck', 'need_gb': 0.1,
                'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'pass': True, 'hosts': {'dl380g10': {'pass': True, 'filesystems': {}}}}
        with patch.object(wave.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(body))):
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                self.disk_patch.temp_original(child, need_gb=0.1, command='unexecuted-fixture')
        body['at'] = '2000-01-01T00:00:00+00:00'
        with patch.object(wave.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(body))):
            with self.assertRaisesRegex(ValueError, 'stale'):
                self.disk_patch.temp_original(child, need_gb=0.1, command='unexecuted-fixture')

    def native_gpu_cost_fixture(self):
        pc = wave.routed_api()
        self.controller.queue = pc.pool.PoolQueue(self.queue)
        self.controller.cas = SimpleNamespace(read_action_request=lambda _: {
            'params': {'demand': {'gpu': 1}, 'execution_timeout_s': 1800,
                       'retry_policy': {'max_attempts': 1}}})
        return self.controller.queue

    def test_ambiguous_native_ending_refuses_next_admission(self):
        key = self.record(0, status='done')
        queue = self.native_gpu_cost_fixture()
        row = {'action_key': key, 'published_unix': 123.0}
        wave.durable_json(queue.item_path('done', key), row)
        wave.durable_json(queue.item_path('failed', key), row)
        self.assertTrue(queue.current_ending(key)['ambiguous'])
        self.assertEqual(self.controller.run(), 1)
        stored = json.loads(self.controller.ledger.path.read_text())
        self.assertEqual(stored['reservations'], {key: 0.5})
        self.assertEqual(stored['charges'], {})
        self.assertEqual(self.calls, [])
        self.assertIn('Ambiguous', json.loads(self.controller.wait_path.read_text())['failure'])

    def test_missing_native_partial_history_retains_reservation(self):
        key = self.record(0, status='live')
        queue = self.native_gpu_cost_fixture()
        row = {'action_key': key, 'published_unix': 123.0, 'max_attempts': 2,
               'retry_safe': True, 'attempts': 0, 'attempt_history': [],
               'claimed_unix': 123.0, 'claimed_host': 'sparky', 'finished_unix': 483.0}
        row['attempt_history'] = queue.archive_attempt(
            row, attempt=1, status='failed', disposition='ready', detail={'elapsed_s': 360.0})
        row.update(attempts=1, attempt_history_missing_before=1)
        wave.durable_json(queue.item_path('ready', key), row)
        self.assertEqual(self.controller.run(), 1)
        stored = json.loads(self.controller.ledger.path.read_text())
        self.assertEqual(stored['reservations'], {key: 0.5})
        self.assertEqual(stored['charges'], {})
        self.assertEqual(self.calls, [])
        self.assertIn('Unknown partial', json.loads(self.controller.wait_path.read_text())['failure'])

    def test_native_cleanup_tombstone_retains_capacity(self):
        pc = wave.routed_api()
        key = self.record(0)
        queue = pc.pool.PoolQueue(self.queue)
        tombstone = self.queue / 'claimed' / f'{key}.1.sparky.tombstone'
        wave.durable_json(tombstone, {'action_key': key, 'published_unix': 123.0})
        wave.durable_json(queue.item_path('done', key), {'action_key': key, 'published_unix': 123.0})
        self.status_patch.stop()
        self.assertEqual(wave.routed_status(key), ('live', None))
        self.assertTrue(pc._pool_slot_occupied(queue, key))
        self.assertEqual(self.controller.capacity(), 0)
        tombstone.unlink()
        self.assertFalse(pc._pool_slot_occupied(queue, key))
        self.assertEqual(wave.routed_status(key), ('done', None))
        self.assertEqual(self.controller.capacity(), 1)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='d44-ledger-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seed = self.root / 'seed.json'
        self.seed.write_text(json.dumps({
            'schema': 'fleet.d44.native_phase_ledger_seed.v1',
            'authoritative_seed_GPU_hours': 21.187425203522047,
            'already_charged_native_action_keys': ['a' * 64],
            'already_charged_preflight_action_keys': [],
            'already_charged_known_historical_action_keys': [],
            'global_ceiling_GPU_hours': 110,
            'GPU_hour_multiplier': 1}))
        self.path = self.root / 'ledger.json'

    def ledger(self):
        return wave.RoutedLedger(self.path, seed_path=self.seed)

    def test_reservation_checks_lifetime_balance(self):
        ledger = self.ledger()
        ledger.reserve('b' * 64, 1800, 1)
        self.assertAlmostEqual(ledger.total(), 21.687425203522047)
        with self.assertRaisesRegex(ValueError, '110'):
            ledger.reserve('c' * 64, 110 * 3600, 1)
        self.assertNotIn('c' * 64, json.loads(self.path.read_text())['reservations'])
        self.admit = lambda *a, **kw: self.fail('Native admission bypassed the lifetime ceiling.')
        policy.ControllerTests.setUp(self)
        self.controller.cas = SimpleNamespace(read_action_request=lambda key: {
            'action_key': key, 'params': {'demand': {'gpu': 1}, 'execution_timeout_s': 1800,
                                         'retry_policy': {'max_attempts': 1}}})
        self.controller.ledger.value['seed']['authoritative_seed_GPU_hours'] = 109.9
        self.controller.ledger.save()
        self.assertEqual(self.controller.run(), 1)
        self.assertEqual(self.calls, [], 'The lifetime reservation must refuse before native admission.')

    def test_exact_lifetime_boundary_and_restart(self):
        seed = json.loads(self.seed.read_text())
        seed['authoritative_seed_GPU_hours'] = 109.5
        self.seed.write_text(json.dumps(seed))
        key = 'b' * 64
        ledger = self.ledger()
        ledger.reserve(key, 1800, 1)
        self.assertEqual(ledger.total(), 110.0)
        ledger = self.ledger()
        ledger.reserve(key, 1800, 1)
        self.assertEqual(ledger.total(), 110.0)
        stored = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, '110'):
            ledger.reserve('c' * 64, 0.001, 1)
        self.assertEqual(self.path.read_bytes(), stored)

    def test_success_failure_partial_and_duplicate_restart(self):
        import sys
        sys.path.insert(0, '/mnt/shared/prismabuild-fleet/repo/tools')
        import pbcampaign as pc
        ledger = self.ledger()
        key = 'b' * 64
        ledger.reserve(key, 1800, 1)
        queue = pc.pool.PoolQueue(self.root / 'queue')
        for name in ('ready', 'claimed', 'done', 'failed', 'withdrawn'):
            queue.dir(name).mkdir(parents=True, exist_ok=True)
        row = {'action_key': key, 'published_unix': 123.0, 'max_attempts': 2,
               'retry_safe': True, 'attempts': 0, 'attempt_history': [],
               'claimed_unix': 123.0, 'claimed_host': 'sparky', 'finished_unix': 483.0}
        row['attempt_history'] = queue.archive_attempt(row, attempt=1, status='failed',
            disposition='ready', detail={'elapsed_s': 360.0, 'stdout': 'Partial CPU fixture result.'})
        row['attempts'] = 1
        wave.durable_json(queue.item_path('ready', key), row)
        controller = wave.RoutedWave.__new__(wave.RoutedWave)
        controller.state = {'waves': [{'members': [{'key': key}], 'closed': False}]}
        controller.ledger, controller.queue = ledger, queue
        controller.cas = SimpleNamespace(read_action_request=lambda _: {
            'params': {'demand': {'gpu': 1}, 'execution_timeout_s': 1800}})
        controller.persist = lambda *a, **kw: None
        with patch.object(wave, 'routed_status', return_value=('live', None)):
            controller.checkpoint_costs()
        self.assertAlmostEqual(ledger.total(), 21.687425203522047)
        row['finished_unix'] = 663.0
        row['attempt_history'] = queue.archive_attempt(row, attempt=2, status='executed',
            disposition='done', detail={'elapsed_s': 180.0, 'stdout': 'Successful CPU fixture result.'})
        row.update(attempts=2, status='executed')
        queue.item_path('ready', key).unlink()
        wave.durable_json(queue.item_path('done', key), row)
        controller.ledger = self.ledger()
        actual_writer = wave.durable_json

        def interrupted(path, value):
            actual_writer(path, value)
            if path == self.path and len(value['charges']) == 2:
                raise OSError('Crash after the second charge checkpoint.')

        with patch.object(wave, 'routed_status', return_value=('done', None)):
            with patch.object(wave, 'durable_json', side_effect=interrupted):
                with self.assertRaisesRegex(OSError, 'Crash'):
                    controller.checkpoint_costs()
            controller.ledger = self.ledger()
            controller.checkpoint_costs()
            controller.checkpoint_costs()
        self.assertAlmostEqual(controller.ledger.total(), 21.337425203522047)
        self.assertEqual(len(json.loads(self.path.read_text())['charges']), 2)
        self.assertEqual(json.loads(self.path.read_text())['reservations'], {})

    @patch.dict('os.environ', {'PRISMAQUANT_DEV_MODE': '1'})
    def test_seeded_cache_does_not_charge_twice(self):
        ledger = self.ledger()
        ledger.checkpoint('a' * 64, [], terminal=True, gpu=1)
        self.assertEqual(ledger.total(), 21.187425203522047)
        changed = json.loads(self.seed.read_text())
        changed['authoritative_seed_GPU_hours'] = 0.0
        self.seed.write_text(json.dumps(changed))
        self.assertEqual(self.ledger().total(), 21.187425203522047)
        cpu_path = self.root / 'CPU-only.json'
        wave.RoutedLedger(cpu_path, cpu=True)
        with self.assertRaisesRegex(ValueError, 'CPU-only'):
            wave.RoutedLedger(cpu_path, seed_path=self.seed)

    def test_changed_seed_development_mode_uses_shared_stamp(self):
        import contextlib
        import io
        import os
        self.ledger()
        stored = self.path.read_bytes()
        changed = json.loads(self.seed.read_text())
        changed['authoritative_seed_GPU_hours'] = 0.0
        self.seed.write_text(json.dumps(changed))
        out = io.StringIO()
        with patch.dict(os.environ, {'PRISMAQUANT_DEV_MODE': '1'}), contextlib.redirect_stdout(out):
            ledger = self.ledger()
        self.assertEqual(ledger.total(), 21.187425203522047)
        self.assertEqual(self.path.read_bytes(), stored)
        self.assertEqual(out.getvalue().count('[DEV-MODE]'), 1)
        self.assertIn('d44 ledger seed', out.getvalue())
        self.assertIn('expected 21.187425203522047', out.getvalue())
        self.assertIn('actual 0.0', out.getvalue())

    def test_changed_seed_certified_mode_refuses_without_balance_change(self):
        import os
        self.ledger()
        stored = self.path.read_bytes()
        changed = json.loads(self.seed.read_text())
        changed['authoritative_seed_GPU_hours'] = 0.0
        self.seed.write_text(json.dumps(changed))
        with patch.dict(os.environ, {'PRISMAQUANT_DEV_MODE': '0'}):
            with self.assertRaisesRegex(ValueError, 'd44 ledger seed'):
                self.ledger()
        self.assertEqual(self.path.read_bytes(), stored)

    def test_unknown_cost_preserves_reservation(self):
        ledger = self.ledger()
        key = 'b' * 64
        ledger.reserve(key, 1800, 1)
        with self.assertRaisesRegex(ValueError, 'cost'):
            ledger.checkpoint(key, [], terminal=True, gpu=1)
        self.assertIn(key, json.loads(self.path.read_text())['reservations'])
        ledger = self.ledger()
        with self.assertRaisesRegex(ValueError, 'cost'):
            ledger.checkpoint(key, [{'action_key': key, 'published_unix': 1.0,
                'attempt': 1, 'status': 'failed', 'detail': {}}], terminal=True, gpu=1)
        self.assertIn(key, json.loads(self.path.read_text())['reservations'])

    def test_overrun_retains_charge_and_refuses_next_admission(self):
        ledger = self.ledger()
        key = 'b' * 64
        ledger.reserve(key, 1800, 1)
        ledger.checkpoint(key, [{'action_key': key, 'published_unix': 1.0,
            'attempt': 1, 'status': 'failed', 'detail': {'elapsed_s': 100 * 3600}}],
            terminal=True, gpu=1)
        with self.assertRaisesRegex(ValueError, '110'):
            ledger.reserve('c' * 64, 1, 1)
        self.assertAlmostEqual(ledger.total(), 121.18742520352205)


if __name__ == '__main__':
    unittest.main(verbosity=2)
