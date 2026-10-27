"""A stage leader exiting cannot release ownership of its surviving writers."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from prismaquant import cluster_campaign as campaign
from test_cluster_campaign_runner import _wait_until


OWNER = 'a' * 64
CHILD = """
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
while True:
    with open(sys.argv[2], 'ab', buffering=0) as stream:
        stream.write(b'x')
    time.sleep(.01)
"""
LEADER = """
import os, subprocess, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(str(os.getpid()))
environment = dict(os.environ)
if sys.argv[6] == 'unknown':
    environment.pop('PRISMAQUANT_CLUSTER_OWNER', None)
subprocess.Popen([sys.executable, '-c', sys.argv[5], sys.argv[2], sys.argv[3]], env=environment)
while not Path(sys.argv[2]).exists():
    time.sleep(.01)
Path(sys.argv[4]).write_bytes(b'done')
if sys.argv[7] != 'normal':
    while True:
        time.sleep(1)
"""


def _case(tmp_path, *, mode='normal', ownership='owned'):
    leader, child, writes, receipt = [tmp_path / name for name in
                                       ('leader.pid', 'child.pid', 'writes', 'receipt')]
    request = {'schema': campaign.WORKER_REQUEST_SCHEMA_V2, 'owner_token': OWNER,
               'work_root': str(tmp_path), 'stage_id': 'group', 'cwd': str(tmp_path),
               'env': {}, 'lock_path': str(tmp_path / 'stage.lock'),
               'timeout_seconds': 120, 'verify_only': False,
               'receipts': [{'path': str(receipt),
                             'sha256': hashlib.sha256(b'done').hexdigest()}],
               'argv': [sys.executable, '-c', LEADER, str(leader), str(child),
                        str(writes), str(receipt), CHILD, ownership, mode]}
    return request, leader, child, writes


def _live(pid):
    snapshot = campaign._proc_snapshot(pid)
    return snapshot is not None and snapshot[1] != 'Z'


def _cleanup(*pidfiles):
    for path in pidfiles:
        if path.exists():
            try:
                os.kill(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def _interrupt_after_child_ready(monkeypatch, ready, mode):
    original = subprocess.Popen

    class InterruptibleChild(original):
        interrupted = False

        def wait(self, timeout=None):
            if not self.interrupted:
                self.interrupted = True
                _wait_until(ready.exists, unmet='stage descendant never became ready',
                            alive=lambda: self.poll() is None)
                if mode == 'cancel':
                    raise KeyboardInterrupt
                raise subprocess.TimeoutExpired(self.args, timeout)
            return super().wait(timeout=timeout)

    monkeypatch.setattr(subprocess, 'Popen', InterruptibleChild)


@pytest.mark.parametrize('mode', ['normal', 'timeout', 'cancel'])
def test_worker_waits_for_the_group_before_receipt_success_or_unlock(tmp_path, monkeypatch, mode):
    request, leader, child, writes = _case(tmp_path, mode=mode)
    if mode != 'normal':
        _interrupt_after_child_ready(monkeypatch, Path(request['receipts'][0]['path']), mode)
    try:
        result = campaign.execute_worker_request(request)
        assert result == (0 if mode == 'normal' else campaign.EXIT_STAGE_FAILED)
        assert child.exists()
        assert not _live(int(child.read_text())), 'worker returned while a descendant could still write'
        assert Path(request['lock_path']).read_bytes() == b''
        before = writes.read_bytes() if writes.exists() else b''
        assert campaign.execute_worker_request({**request, 'verify_only': True}) == 0
        assert (writes.read_bytes() if writes.exists() else b'') == before
    finally:
        _cleanup(child, leader)


def test_recorded_group_teardown_finds_descendants_after_the_leader_exits(tmp_path):
    request, leader, child, _ = _case(tmp_path)
    process = subprocess.Popen(request['argv'], env={campaign._RESERVED_OWNER_ENV: OWNER},
                               start_new_session=True)
    snapshot = campaign._proc_snapshot(process.pid)
    assert snapshot is not None
    try:
        process.wait(timeout=120)
        assert _live(int(child.read_text()))
        assert campaign._terminate_recorded_owned_process(process.pid, snapshot[0], OWNER)
        assert not _live(int(child.read_text()))
    finally:
        _cleanup(child, leader)
        process.wait(timeout=120)


def test_unknown_descendant_quarantines_the_lock_and_refuses_probes_and_retries(tmp_path, monkeypatch):
    request, leader, child, _ = _case(tmp_path, ownership='unknown')
    monkeypatch.setattr(campaign, '_RECOVERY_KILL_GRACE_SECONDS', .05)
    try:
        with pytest.raises(campaign.CampaignContractError, match='quarantined'):
            campaign.execute_worker_request(request)
        assert _live(int(child.read_text())), 'an unowned descendant must not be killed'
        lock = Path(request['lock_path'])
        record = json.loads(lock.read_text())
        assert record['schema'] == campaign.WORKER_LOCK_SCHEMA
        assert record['status'] == 'quarantined'
        for verify_only in (True, False):
            with pytest.raises(campaign.CampaignContractError, match='unresolved process group'):
                campaign.execute_worker_request({**request, 'verify_only': verify_only})
        assert json.loads(lock.read_text()) == record
    finally:
        _cleanup(child, leader)


def test_retained_running_lock_cannot_be_cleared_while_a_descendant_survives(tmp_path):
    request, leader, child, _ = _case(tmp_path)
    process = subprocess.Popen(request['argv'], env={campaign._RESERVED_OWNER_ENV: OWNER},
                               start_new_session=True)
    snapshot = campaign._proc_snapshot(process.pid)
    assert snapshot is not None
    lock = Path(request['lock_path'])
    try:
        process.wait(timeout=120)
        record = {'schema': campaign.WORKER_LOCK_SCHEMA, 'status': 'running',
                  'stage_id': 'group', 'owner_token': OWNER, 'pid': process.pid,
                  'pid_start_ticks': snapshot[0]}
        lock.write_text(json.dumps(record))
        with pytest.raises(campaign.CampaignContractError, match='unresolved process group'):
            campaign.execute_worker_request({**request, 'verify_only': True})
        assert campaign._terminate_recorded_owned_process(process.pid, snapshot[0], OWNER)
        assert campaign.execute_worker_request({**request, 'verify_only': True}) == 0
        assert lock.read_bytes() == b''
    finally:
        _cleanup(child, leader)
        process.wait(timeout=120)
