"""Direct behavior proof for the one-slot controller and ship queue boundary."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.d44_native import next_wave as wave


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='d44-policy-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / 'queue'
        for name in ('ready', 'claimed', 'done', 'failed', 'withdrawn'):
            (self.queue / name).mkdir(parents=True)
        self.patch = patch.object(wave, 'QUEUE', self.queue)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.calls = []
        self.statuses = {}
        self.controller = wave.RoutedWave.__new__(wave.RoutedWave)
        self.controller.keys = [f'{i:064x}' for i in range(1, 5)]
        self.controller.ordinal = {key: i for i, key in enumerate(self.controller.keys)}
        self.controller.state = {'waves': []}
        self.controller.journal = []
        self.controller.state_path = self.root / 'wave-state.json'
        self.controller.wait_path = self.root / 'wait.json'
        self.controller.journal_path = self.root / 'sub-keys.txt'
        self.controller.resume_command = 'CPU policy proof'
        self.controller.args = SimpleNamespace()
        self.controller.request = {}
        self.controller.plan = {}
        self.controller.data_plans = [None] * 4
        self.controller.cas = SimpleNamespace(read_action_request=lambda key: {
            'action_key': key, 'params': {'demand': {'gpu': 0}, 'execution_timeout_s': 30}})
        self.controller.autonomous = False
        self.controller.disk_need_gb = 0.1
        self.controller.ledger = wave.RoutedLedger(self.root / 'lifetime-cost.json', cpu=True)
        self.disk_patch = patch.object(wave, 'routed_diskcheck', return_value={'pass': True})
        self.disk_patch.start()
        self.addCleanup(self.disk_patch.stop)
        self.controller.pc = SimpleNamespace(child_record=self.admit, close_group=lambda *a, **kw: 0)
        self.controller.queue = SimpleNamespace(root=self.queue)
        self.controller.reconcile = lambda: None
        self.status_patch = patch.object(wave, 'routed_status', lambda key: self.statuses.get(key, ('unknown', None)))
        self.status_patch.start()
        self.addCleanup(self.status_patch.stop)

    def record(self, index, status='live', host=None):
        key = self.controller.keys[index]
        self.statuses[key] = (status, host)
        self.controller.accept(key)
        return key

    def admit(self, child, **kwargs):
        key = child['action_key']
        self.calls.append(key)
        self.statuses[key] = ('live', 'sparky')
        return {'action_key': key, 'status': 'submitted'}

    def ship(self, key='a' * 64, priority=0, gpu=1, state='ready'):
        path = self.queue / state / (key + '.json')
        path.write_text(json.dumps({'action_key': key, 'priority': priority,
                                   'resources': {'gpu': gpu, 'cpu': 1},
                                   'priority_reason': 'ship-path kernel and serve check'}))
        return path

    def test_one_claim_on_spark_blocks_portable_admission(self):
        self.record(0, host='sparky')
        self.assertEqual(self.controller.capacity(), 0,
                         'A portable child must not take a second slot on an occupied Spark.')

    def test_unknown_portable_key_blocks_every_spark(self):
        self.record(0, status='unknown')
        self.assertEqual(self.controller.capacity(), 0,
                         'An unknown portable key consumes the only slot on every possible Spark.')

    def test_pending_admission_consumes_capacity(self):
        self.controller.state['pending_submission'] = {'key': self.controller.keys[0], 'batch': 'child-00000'}
        self.assertEqual(self.controller.capacity(), 0,
                         'Pending admission custody must consume the only portable slot.')
        self.assertEqual(self.controller.run(), 3)
        self.assertEqual(self.calls, [])

    def test_ship_queue_blocks_first_admission(self):
        self.ship(priority=10)
        self.assertEqual(self.controller.run(), 3)
        self.assertEqual(self.calls, [], 'A queued PACT job must prevent child_record.')

    def test_ship_arrival_stops_next_admission(self):
        def admit_and_finish(child, **kwargs):
            result = self.admit(child, **kwargs)
            self.statuses[child['action_key']] = ('done', None)
            self.ship(priority=0)
            return result
        self.controller.pc.child_record = admit_and_finish
        self.assertEqual(self.controller.run(), 3)
        self.assertEqual(self.calls, self.controller.keys[:1],
                         'A ship job between admissions must stop the next child_record call.')

    def test_unreadable_queue_stops_admission(self):
        (self.queue / 'ready' / ('b' * 64 + '.json')).write_text('{bad')
        self.assertEqual(self.controller.run(), 3)
        self.assertEqual(self.calls, [], 'A partial queue census must not authorize admission.')

    def test_claimed_ship_and_transition_remain_a_barrier(self):
        self.ship(state='claimed')
        self.assertEqual(self.controller.run(), 3)
        self.assertEqual(self.calls, [])

    def test_ship_tombstone_remains_a_barrier(self):
        path = self.ship(state='claimed')
        path.rename(path.with_suffix('.1.sparky.tombstone'))
        self.assertEqual(self.controller.run(), 3)
        self.assertEqual(self.calls, [])

    def test_cpu_and_low_priority_work_do_not_block(self):
        self.ship(key='a' * 64, gpu=0, priority=10)
        self.ship(key='b' * 64, priority=-10)
        self.assertEqual(self.controller.run(), 3)
        self.assertEqual(self.calls, self.controller.keys[:1])

    def test_own_native_child_does_not_become_ship_work(self):
        self.ship(key=self.controller.keys[0], priority=0)
        self.statuses[self.controller.keys[0]] = ('done', None)
        self.record(0, status='done')
        self.assertEqual(self.controller.run(), 3)
        self.assertEqual(self.calls, self.controller.keys[1:2])

    def test_terminal_custody_releases_capacity(self):
        self.record(0, status='done')
        self.assertEqual(self.controller.capacity(), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
