"""The Tessera fleet drivers moved from PrismaBuild (RobTand/prismabuild#1076).

PrismaBuild names no client, so its whole-model Tessera dispatcher, the two
per-shard dispatchers and the status screen live in ``tools/tessera_fleet``.
They reach PrismaBuild only through the published ``pbcampaign.py`` and
``pbwait.py`` commands, which ``tests/test_prismabuild_boundary.py`` holds
them to. These tests cover what the move kept from the PrismaBuild suite
(partitioning, input verification, the assembly barrier, the shard domain)
and what the rewire changed (named preparation hosts, results read from the
worker's own records, one sealed workspace per dispatch).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.tessera_fleet import common, dispatch_ladder, dispatch_model, dispatch_shards
from tools.tessera_fleet import model_worker as model
from tools.tessera_fleet import status

PRODUCER = SimpleNamespace(
    BODY_LAYER=re.compile(r'^model\.layers\.(\d+)\.'),
    partition_owner=lambda name, count: int(name.split('.')[2]) % count if '.layers.' in name else 0)

DISPATCHERS = (dispatch_shards, dispatch_ladder)


# --- the worker adapter, unchanged by the move --------------------------------

def test_single_source_file_still_yields_24_layer_actions():
    tensors = {f'model.layers.{i}.weight': 'model.safetensors' for i in range(24)}
    tensors['model.embed_tokens.weight'] = 'model.safetensors'
    assert model.partitions(tensors, PRODUCER) == 24
    spec = dict(cpus=1, mem_gb=16, assembly_mem_gb=4, tags=['gb10'])
    rows = [dispatch_model.campaign_row('/checkout', spec, 'encode', index=i) for i in range(24)]
    assert len({tuple(row['argv']) for row in rows}) == 24
    assert all(row['tags'] == ['gb10'] and row['demand']['gpu'] == 1 for row in rows)
    assert not any('here' in row for row in rows)


def test_sparse_layers_never_generate_empty_partitions():
    names = ['model.layers.0.weight', 'model.layers.2.weight']
    count = model.partitions(names, PRODUCER)
    assert {PRODUCER.partition_owner(n, count) for n in names} == set(range(count))


def _source(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'config.json').write_text('{}')
    (source / 'weights').write_bytes(b'original')
    identity = {'auxiliary_sha256': {}, 'config_sha256': model.digest_file(source / 'config.json'),
                'files': {'weights': model.digest_file(source / 'weights')}}
    return source, identity


def test_input_hash_is_reused_only_for_unchanged_verified_files(tmp_path, monkeypatch):
    source, identity = _source(tmp_path)
    cache = tmp_path / 'cache'
    model.verified_inputs(source, identity, cache)
    original = model.digest_file
    calls = []

    def digest(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(model, 'digest_file', digest)
    model.verified_inputs(source, identity, cache)
    assert calls == []
    (source / 'weights').write_bytes(b'modified')
    with pytest.raises(ValueError, match='source changed'):
        model.verified_inputs(source, identity, cache)


def test_mid_export_source_mutation_is_detected(tmp_path):
    source, identity = _source(tmp_path)
    stamps = model.verified_inputs(source, identity, tmp_path / 'cache')
    (source / 'weights').write_bytes(b'changed')
    with pytest.raises(ValueError, match='during export'):
        model.check_stamps(source, stamps)


def test_output_must_match_its_barrier_record_and_actual_payload(tmp_path):
    (tmp_path / 'wire').write_bytes(b'wire')
    record = {'files': {'wire': model.digest_file(tmp_path / 'wire')}, 'contract': 'a'}
    model.atomic_json(tmp_path / 'pb-result.json', record)
    assert model.verify_output_record(tmp_path, record) == record
    with pytest.raises(ValueError, match='differs'):
        model.verify_output_record(tmp_path, dict(record, contract='b'))
    (tmp_path / 'wire').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='output changed'):
        model.verify_output_record(tmp_path, record)


def test_unlisted_output_cannot_be_reused(tmp_path):
    (tmp_path / 'wire').write_bytes(b'wire')
    record = {'files': {'wire': model.digest_file(tmp_path / 'wire')}}
    model.atomic_json(tmp_path / 'pb-result.json', record)
    (tmp_path / 'unexpected.safetensors').write_bytes(b'stale checkpoint')
    with pytest.raises(ValueError, match='population changed'):
        model.verify_output_record(tmp_path, record)


def test_mutated_plan_refuses_before_any_encode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'plan.json').write_text('{}')
    model.atomic_json('job.json', {'plan_sha256': '0' * 64})
    with pytest.raises(ValueError, match='plan changed'):
        model.main(['encode'])


def test_prepare_records_its_identity_under_the_host_it_was_pinned_to(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source, _ = _source(tmp_path)
    (tmp_path / 'plan.json').write_text('{}')
    spec = {'plan_sha256': model.digest_file(tmp_path / 'plan.json'), 'image': 'x@sha256:' + '0' * 64,
            'source': str(source), 'out': str(tmp_path / 'out')}
    model.atomic_json('job.json', spec)
    parts = SimpleNamespace(source_identity=lambda path: {'tensors': {'model.layers.0.w': 'weights'}})
    monkeypatch.setattr(model, 'verify_image', lambda image: 'id')
    monkeypatch.setattr(model, 'producer_parts', lambda root: parts)
    monkeypatch.setattr(model, 'partitions', lambda tensors, producer: 1)
    with pytest.raises(ValueError, match='needs --host'):
        model.main(['prepare'])
    assert model.main(['prepare', '--host', 'sparky']) == 0
    assert model.read_json(model.prepared_path(spec, 'sparky')) == {
        'source_identity': {'tensors': {'model.layers.0.w': 'weights'}}, 'count': 1}


# --- the whole-model dispatcher, rewired onto pbcampaign ----------------------

def _campaign(monkeypatch, *, statuses, returncode=0):
    calls = []

    def run(rows, workspace, stage, *, wait_s, detach=False):
        rows = list(rows)
        calls.append((stage, rows, detach))
        header = 'key           status     transport  job  host    elapsed  rc  receipt  note'
        lines = [header] + [f'{("%012x" % i)}  {s:<9}  pool       -    sparky  1.0      0   yes      -'
                            for i, s in enumerate(statuses)]
        return returncode, '\n'.join(lines) + '\n'
    monkeypatch.setattr(common, 'run_pbcampaign', run)
    return calls


def test_a_failed_partition_cannot_pass_the_assembly_barrier(tmp_path, monkeypatch):
    _campaign(monkeypatch, statuses=['failed'], returncode=1)
    with pytest.raises(ValueError, match='0 of 1 action'):
        common.run_stage([{}], tmp_path, 'encode', wait_s=1)
    endings = json.loads((tmp_path / common.STATE / 'encode-endings.json').read_text())
    assert endings['returncode'] == 1 and endings['rows'][0]['status'] == 'failed'
    assert not (tmp_path / 'barrier.json').exists()


def test_a_stage_that_ran_fewer_rows_than_it_submitted_is_refused(tmp_path, monkeypatch):
    _campaign(monkeypatch, statuses=['executed'])
    with pytest.raises(ValueError, match='1 of 2 action'):
        common.run_stage([{}, {}], tmp_path, 'encode', wait_s=1)
    _campaign(monkeypatch, statuses=['executed', 'cache_hit'])
    assert len(common.run_stage([{}, {}], tmp_path, 'encode', wait_s=1)) == 2


def test_preparation_runs_on_every_named_host_and_they_must_agree(tmp_path, monkeypatch):
    calls = _campaign(monkeypatch, statuses=['executed', 'executed'])
    spec = {'out': str(tmp_path / 'out'), 'prepare_hosts': ['sparklina', 'sparky'],
            'tags': ['gb10'], 'cpus': 1, 'mem_gb': 16, 'assembly_mem_gb': 4}
    for host, count in (('sparklina', 3), ('sparky', 3)):
        model.atomic_json(model.prepared_path(spec, host), {'count': count})
    assert dispatch_model.prepare_source_identity(spec, tmp_path, 1) == {'count': 3}
    rows = calls[0][1]
    assert [row['tags'] for row in rows] == [['gb10', 'sparklina'], ['gb10', 'sparky']]
    assert [row['argv'][-2:] for row in rows] == [['--host', 'sparklina'], ['--host', 'sparky']]
    model.atomic_json(model.prepared_path(spec, 'sparky'), {'count': 4})
    with pytest.raises(ValueError, match='disagree'):
        dispatch_model.prepare_source_identity(spec, tmp_path, 1)


def test_a_part_record_of_another_export_is_refused(tmp_path):
    spec = {'parts': str(tmp_path), 'count': 1, 'contract': 'a'}
    model.atomic_json(tmp_path / 'part-00000' / 'pb-result.json', {'index': 0, 'contract': 'b'})
    with pytest.raises(ValueError, match='another export'):
        dispatch_model.part_records(spec)
    model.atomic_json(tmp_path / 'part-00000' / 'pb-result.json', {'index': 0, 'contract': 'a'})
    assert dispatch_model.part_records(spec) == [{'index': 0, 'contract': 'a'}]


def test_resume_refuses_an_input_override(tmp_path):
    for override in (['--plan', 'changed.json'], ['--prepare-host', 'sparky']):
        with pytest.raises(SystemExit) as exc:
            dispatch_model.main(['--workspace', str(tmp_path), '--resume', *override])
        assert exc.value.code == 2


def test_a_new_dispatch_names_its_preparation_hosts(tmp_path, capsys):
    argv = ['--workspace', str(tmp_path / 'w'), '--source', '/mnt/shared/s', '--plan', 'p',
            '--encoder-checkout', '.', '--encoder-revision', '0' * 40,
            '--image', 'x@sha256:' + '0' * 64, '--out', '/mnt/shared/o']
    with pytest.raises(SystemExit) as exc:
        dispatch_model.main(argv)
    assert exc.value.code == 2
    assert '--prepare-host is required' in capsys.readouterr().err


# --- the per-shard dispatchers ------------------------------------------------

@pytest.mark.parametrize('dispatcher', DISPATCHERS, ids=lambda d: d.__name__)
@pytest.mark.parametrize('text', ['abc', '', '1-', '-3', '1,2', '1-2-3', '1 2',
                                  '0', '121', '5-3', '0-4', '119-121'])
def test_a_value_that_names_no_shard_of_this_run_is_refused(dispatcher, text):
    with pytest.raises(argparse.ArgumentTypeError, match='within 1-120'):
        common.shard_range(dispatcher.OF_SHARDS)(text)


@pytest.mark.parametrize('text,expected', [('1', range(1, 2)), ('61', range(61, 62)),
                                           ('1-120', range(1, 121)), ('7-9', range(7, 10))])
def test_the_range_an_operator_means_is_what_the_loop_gets(text, expected):
    assert common.shard_range(120)(text) == expected


@pytest.mark.parametrize('dispatcher', DISPATCHERS, ids=lambda d: d.__name__)
def test_a_bad_shards_value_is_a_usage_error(dispatcher, capsys):
    with pytest.raises(SystemExit) as exc:
        dispatcher.main(['--shards', 'abc', '--dry-run'])
    assert exc.value.code == 2
    assert 'within 1-120' in capsys.readouterr().err


def _checkout(tmp_path):
    checkout = tmp_path / 'checkout'
    (checkout / 'tessera' / 'src' / 'tessera').mkdir(parents=True)
    (checkout / 'tessera' / 'src' / 'tessera' / 'encode.py').write_text('ENCODER = 1\n')
    (checkout / dispatch_shards.WRAPPER).write_text('print("export")\n')
    (checkout / dispatch_ladder.WRAPPER).write_text('print("ladder")\n')
    return checkout


def _listing(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob('*'))}


@pytest.mark.parametrize('dispatcher', DISPATCHERS, ids=lambda d: d.__name__)
def test_a_dry_run_prints_rows_and_writes_nothing(dispatcher, tmp_path, capsys, monkeypatch):
    checkout = _checkout(tmp_path)
    before = _listing(tmp_path)
    monkeypatch.setattr(common, 'run_pbcampaign', lambda *a, **k: pytest.fail('submitted'))
    assert dispatcher.main(['--shards', '3-4', '--dry-run', '--checkout', str(checkout)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert _listing(tmp_path) == before
    assert [row['argv'][row['argv'].index('--shard') + 1] for row in rows] == ['3', '4']
    for row in rows:
        # Relative to the sealed tree the action runs in, never the submitter's.
        assert row['env']['PYTHONPATH'] == 'tessera/src'
        assert row['tags'] == ['gb10'] and row['demand']['gpu'] == 1
        assert not row['env']['TMPDIR'].startswith('/tmp')


def test_a_submission_seals_wrapper_plan_and_encoder_into_one_workspace(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path)
    plan = tmp_path / 'plan.json'
    plan.write_text('{"w": 896}')
    submitted = []

    def submit(rows, workspace, stage, *, wait_s):
        submitted.append((rows, Path(workspace), stage))
        return [{'action_key': 'a' * 64}] * len(rows)
    monkeypatch.setattr(common, 'submit_detached', submit)
    workspace = tmp_path / 'ws'
    assert dispatch_shards.main(['--shards', '1-2', '--workspace', str(workspace),
                                 '--checkout', str(checkout), '--plan', str(plan)]) == 0
    rows, staged, stage = submitted[0]
    assert stage == 'export' and staged == workspace.resolve()
    assert (workspace / 'plan.json').read_text() == plan.read_text()
    assert (workspace / dispatch_shards.WRAPPER).read_text() == 'print("export")\n'
    assert (workspace / 'tessera/src/tessera/encode.py').read_text() == 'ENCODER = 1\n'
    assert subprocess.run(['git', '-C', str(workspace), 'rev-parse', 'HEAD'],
                          capture_output=True).returncode == 0
    assert (workspace / '.git/info/exclude').read_text() == f'/{common.STATE}/\n'
    assert all(row['cwd'] == str(workspace.resolve()) for row in rows)
    assert all(row['argv'][row['argv'].index('--plan') + 1] == 'plan.json' for row in rows)
    with pytest.raises(ValueError, match='already exists'):
        dispatch_shards.main(['--shards', '1', '--workspace', str(workspace),
                              '--checkout', str(checkout), '--plan', str(plan)])


def test_two_ladder_dispatches_with_different_probes_are_separate_trees(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path)
    monkeypatch.setattr(common, 'submit_detached',
                        lambda rows, workspace, stage, *, wait_s: [{'action_key': 'a'}] * len(rows))
    for index in range(2):
        probe = tmp_path / f'probe-{index}.py'
        probe.write_text(f'print({index})\n')
        workspace = tmp_path / f'ws{index}'
        assert dispatch_ladder.main(['--shards', '1', '--workspace', str(workspace),
                                     '--checkout', str(checkout), '--wrapper', str(probe)]) == 0
        assert (workspace / dispatch_ladder.WRAPPER).read_text() == f'print({index})\n'
    assert (checkout / dispatch_ladder.WRAPPER).read_text() == 'print("ladder")\n'


def test_detached_submissions_are_read_off_the_json_lines():
    text = ('pbcampaign: something on stderr\n'
            '{"action_key": "' + 'b' * 64 + '", "status": "submitted"}\n'
            '{"not": "a row"}\nnot json {\n')
    assert common.detached_submissions(text) == [{'action_key': 'b' * 64, 'status': 'submitted'}]


# --- the status screen --------------------------------------------------------

def test_status_with_nothing_recorded_says_so(tmp_path, capsys):
    assert status.main(['--workspace', str(tmp_path)]) == status.EXIT_NOTHING


def test_status_asks_pbwait_for_the_recorded_keys(tmp_path, capsys):
    common.atomic_json(tmp_path / common.STATE / 'export-submissions.json',
                       {'returncode': 0, 'rows': [{'action_key': 'a' * 64},
                                                  {'action_key': 'b' * 64}]})
    fake = tmp_path / 'pbwait.py'
    fake.write_text(
        'import sys\n'
        'assert sys.argv[1:3] == ["--wait-s", "0"], sys.argv\n'
        'print("key           status     transport  host")\n'
        'print(sys.argv[3][:12] + "  executed   pool       sparky")\n'
        'print(sys.argv[4][:12] + "  waiting    pool       -")\n')
    assert status.main(['--workspace', str(tmp_path), '--pbwait', str(fake)]) == status.EXIT_RUNNING
    assert 'export: 2 recorded, 1 executed, 1 waiting' in capsys.readouterr().out


@pytest.mark.parametrize('statuses,expected', [
    (['executed', 'cache_hit'], status.EXIT_DONE),
    (['executed', 'waiting'], status.EXIT_RUNNING),
    (['failed', 'waiting'], status.EXIT_FAILED),
    (['executed'], status.EXIT_FAILED),
])
def test_the_exit_status_is_the_worst_ending(statuses, expected):
    assert status.exit_status([{'status': s} for s in statuses], 2) == expected


# --- nothing here reaches PrismaBuild internals -------------------------------

@pytest.mark.parametrize('module', ['tessera_fleet.dispatch_model', 'tessera_fleet.dispatch_shards',
                                    'tessera_fleet.dispatch_ladder', 'tessera_fleet.status',
                                    'tessera_fleet.model_worker', 'render_identity'])
def test_help_runs_without_the_fleet(module):
    """``--help`` imports each driver the way an operator runs it, and exits 0."""
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, '-m', f'tools.{module}', '--help'],
                               cwd=root, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert 'usage:' in completed.stdout
