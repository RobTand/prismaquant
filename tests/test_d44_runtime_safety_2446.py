"""CPU refusals for D44 native status, CLI dispatch and configured D1 targets."""
import datetime
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools.d44_native import next_wave as wave


@pytest.fixture
def native_queue(tmp_path, monkeypatch):
    queue = wave.native_client().PoolQueue(tmp_path / 'queue')
    for state in ('ready', 'claimed', 'done', 'failed', 'withdrawn'):
        queue.records.dir(state).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(wave, 'QUEUE', queue.root)
    return queue


@pytest.mark.parametrize('done_generation, failed_generation, expected', [
    (100, 200, 'failed'), (200, 100, 'done'), (100, 100, 'unknown'),
    (None, 200, 'unknown'), (None, None, 'unknown'),
])
def test_status_uses_native_current_generation(native_queue,
                                               done_generation, failed_generation, expected):
    key = 'a' * 64
    queue = native_queue
    for state, generation in [('done', done_generation), ('failed', failed_generation)]:
        row = {'action_key': key}
        if generation is not None:
            row['published_unix'] = generation
        wave.durable_json(queue.records.item_path(state, key), row)
    ending = queue.current_ending(key)
    if expected != 'unknown':
        assert ending['state'] == expected
    else:
        assert ending['ambiguous']
    assert wave.routed_status(key) == (expected, None)


def test_unreadable_competitor_retains_custody(native_queue):
    key = 'a' * 64
    queue = native_queue
    wave.durable_json(queue.records.item_path('done', key),
                      {'action_key': key, 'published_unix': 100})
    queue.records.item_path('failed', key).write_text('{broken')
    assert queue.current_ending(key)['unreadable']
    assert wave.routed_status(key) == ('unknown', None)


@pytest.mark.parametrize('arguments, code', [
    (['--help'], 0), ([], 2), (['--unknown'], 2), (['--catalog-only'], 2),
    (['--routed-plan'], 2), (['--legacy', '--catalog-only'], 2),
    (['--routed-plan', 'plan.json', '--unknown'], 2),
])
def test_cli_help_and_invalid_invocations_never_submit(tmp_path, monkeypatch, arguments, code):
    monkeypatch.chdir(tmp_path)
    submission = Mock(side_effect=AssertionError('No submission path may be reached.'))
    monkeypatch.setattr(wave, 'main', submission)
    monkeypatch.setattr(wave, 'prepare_routed_catalog', submission)
    monkeypatch.setattr(wave, 'RoutedWave', submission)
    with pytest.raises(SystemExit) as error:
        wave.routed_main(arguments)
    assert error.value.code == code
    submission.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_diskcheck_targets_exact_configured_host_and_mounts(monkeypatch):
    placement = {'required_host': 'worker-17', 'required_tags': ['accelerated']}
    targets = {'targets': [{'placement': placement, 'hosts': ['inventory-17'],
                            'paths': ['/scratch/jobs', '/results']}]}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    body = {'tool': 'fleet-diskcheck', 'need_gb': 2.5, 'at': now, 'pass': True,
            'hosts': {'inventory-17': {'pass': True, 'filesystems': {
                path: {'pass': True, 'avail_gib': 100, 'free_pct': 90,
                       'required_gib': 2.5, 'inodes_free_pct': 95}
                for path in targets['targets'][0]['paths']}}}}
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=json.dumps(body)))
    monkeypatch.setattr(wave.subprocess, 'run', run)
    result = wave.routed_diskcheck({'params': {'placement': placement}}, need_gb=2.5,
                                  command='configured-diskcheck', targets=targets)
    assert result['pass'] is True
    argv = run.call_args.args[0]
    assert argv == ['configured-diskcheck', '--need-gb', '2.5', '--hosts', 'inventory-17',
                    '--paths', '/scratch/jobs,/results']
    with pytest.raises(ValueError, match='no unique configured targets'):
        wave.routed_diskcheck({'params': {'placement': {'required_host': 'another-worker'}}},
                              need_gb=2.5, command='configured-diskcheck', targets=targets)
    assert run.call_count == 1


def test_launcher_queue_discovery_does_not_require_staged_sdk(tmp_path, monkeypatch):
    monkeypatch.setattr(wave, 'QUEUE', None)
    monkeypatch.setenv(wave.QUEUE_ROOT_ENV_VAR, str(tmp_path / 'launched-queue'))
    monkeypatch.setattr(wave, 'native_client', Mock(side_effect=AssertionError('No SDK needed for launch context.')))
    assert wave.routed_queue_root() == tmp_path / 'launched-queue'
    wave.native_client.assert_not_called()
    monkeypatch.delenv(wave.QUEUE_ROOT_ENV_VAR)
    with pytest.raises(ValueError, match='launcher-published queue root'):
        wave.routed_queue_root()


def test_native_loader_uses_configured_public_tool(tmp_path, monkeypatch):
    import sys

    tool = tmp_path / 'tools' / 'fleet' / 'pbcampaign.py'
    tool.parent.mkdir(parents=True)
    tool.write_text('configured_public_tool = True\n')
    monkeypatch.setattr(wave, 'lease_helper_root', lambda: str(tmp_path))
    monkeypatch.setattr(wave, 'native_client', lambda: None)
    monkeypatch.delitem(sys.modules, 'pbcampaign', raising=False)
    before = sys.path[:]
    try:
        module = wave.routed_api()
        assert module.configured_public_tool is True
        assert module.__file__ == str(tool)
        assert sys.path == before
    finally:
        sys.modules.pop('pbcampaign', None)


def test_native_loader_refuses_missing_generation(monkeypatch):
    monkeypatch.setattr(wave, 'lease_helper_root', lambda: None)
    monkeypatch.setattr(wave, 'native_client', lambda: None)
    with pytest.raises(ValueError, match='PRISMABUILD_READER_HELPER_ROOT'):
        wave.routed_api()


def test_native_loader_refuses_divergent_preimport(tmp_path, monkeypatch):
    import sys

    package = tmp_path / 'src' / 'prismabuild'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    monkeypatch.setattr(wave, 'lease_helper_root', lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, 'prismabuild', SimpleNamespace(__file__=str(tmp_path / 'elsewhere.py')))
    with pytest.raises(ValueError, match='divergent preimported'):
        wave.native_client()


def test_missing_disk_targets_refuse_before_check(monkeypatch):
    monkeypatch.delenv('D44_DISK_TARGETS', raising=False)
    run = Mock(side_effect=AssertionError('No unconfigured D1 approval.'))
    monkeypatch.setattr(wave.subprocess, 'run', run)
    with pytest.raises(ValueError, match='configured placement/storage targets'):
        wave.routed_diskcheck({'params': {'placement': {'required_tags': ['x86']}}}, need_gb=1)
    run.assert_not_called()
