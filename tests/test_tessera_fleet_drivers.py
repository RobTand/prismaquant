"""The Tessera whole-model dispatcher and native status checks.

They live in ``tools/tessera_fleet`` and reach the execution system only through its published
submit and wait commands. These tests cover partitioning, input verification, the assembly barrier,
the shard domain, named preparation hosts, results read from the worker's own records and one
sealed workspace per dispatch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from prismaquant.export_partition import whole_layer_partitions
from tools.tessera_fleet import common, dispatch_model
from tools.tessera_fleet import model_worker as model

PRODUCER = SimpleNamespace(
    BODY_LAYER=re.compile(r'^model\.layers\.(\d+)\.'),
    partition_owner=lambda name, count: int(name.split('.')[2]) % count if '.layers.' in name else 0)


# --- the worker adapter, unchanged by the move --------------------------------

def test_single_source_file_still_yields_24_layer_actions():
    tensors = {f'model.layers.{i}.weight': 'model.safetensors' for i in range(24)}
    tensors['model.embed_tokens.weight'] = 'model.safetensors'
    assert len(whole_layer_partitions(tensors, PRODUCER)) == 24
    spec = dict(cpus=1, mem_gb=16, assembly_mem_gb=4, tags=['gb10'])
    rows = [dispatch_model.model_export_row('/checkout', spec, 'encode', index=i) for i in range(24)]
    assert len({tuple(row['argv']) for row in rows}) == 24
    assert all(row['tags'] == ['gb10'] and row['demand']['gpu'] == 1 for row in rows)
    assert not any('here' in row for row in rows)


def test_sparse_layers_never_generate_empty_partitions():
    names = ['model.layers.0.weight', 'model.layers.2.weight']
    count = len(whole_layer_partitions(dict.fromkeys(names, 'weights'), PRODUCER))
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
    # Same size on purpose: the stamp then differs only by time.  Rewriting
    # within one filesystem timestamp tick would leave it unchanged, so the new
    # mtime is set explicitly instead of left to the clock.
    before = (source / 'weights').stat()
    (source / 'weights').write_bytes(b'modified')
    os.utime(source / 'weights', ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match='source changed'):
        model.verified_inputs(source, identity, cache)


@pytest.mark.parametrize('small_limit', [1024, 0])
@pytest.mark.parametrize('clock_offset', [0, -10_000_000_000])
def test_same_stamp_rewrites_refuse_at_worker_boundaries(tmp_path, monkeypatch, small_limit, clock_offset):
    source, identity = _source(tmp_path)
    weights = source / 'weights'
    real_stamp = model.stamp
    fixed_ctime = {p: p.stat().st_ctime_ns for p in source.iterdir()}
    clock = [max(p.stat().st_ctime_ns for p in source.iterdir()) + clock_offset]
    monkeypatch.setattr(model, 'time', SimpleNamespace(time_ns=lambda: clock[0]), raising=False)
    monkeypatch.setattr(model, 'SMALL_FILE_BYTES', small_limit, raising=False)

    def stamp(path, **kwargs):
        value = real_stamp(path, **kwargs)
        value[4] = fixed_ctime[Path(path)]
        return value

    monkeypatch.setattr(model, 'stamp', stamp)

    def rewrite():
        before = model.stamp(weights)[:5]
        info = weights.stat()
        assert weights.read_bytes() == b'original'
        weights.write_bytes(b'modified')
        os.utime(weights, ns=(info.st_atime_ns, info.st_mtime_ns))
        assert model.stamp(weights)[:5] == before

    cache = tmp_path / 'cache'
    model.verified_inputs(source, identity, cache)
    snapshots = model.verified_inputs(source, identity, cache)
    model.check_stamps(source, snapshots)
    rewrite()
    with pytest.raises(ValueError) as error:
        model.verified_inputs(source, identity, cache)
    assert type(error.value) is ValueError
    assert str(error.value) == 'source changed: weights'
    assert error.value.__cause__ is None
    clock[0] += 10_000_000_000
    with pytest.raises(ValueError) as error:
        model.check_stamps(source, snapshots)
    assert type(error.value) is ValueError
    assert str(error.value) == 'source changed during export: weights'
    assert error.value.__cause__ is None

    weights.write_bytes(b'original')
    clock[0] = weights.stat().st_mtime_ns + clock_offset
    monkeypatch.chdir(tmp_path)
    Path('plan.json').write_text('{}')
    spec = {'plan_sha256': model.digest_file('plan.json'), 'image': 'x@sha256:' + '0' * 64,
            'source': str(source), 'out': str(tmp_path / 'out')}
    model.atomic_json('job.json', spec)

    def source_identity(path):
        assert path == source
        rewrite()
        clock[0] += 10_000_000_000
        return {'tensors': {'model.layers.0.w': 'weights'}}

    parts = SimpleNamespace(source_identity=source_identity, BODY_LAYER=PRODUCER.BODY_LAYER,
                            partition_owner=PRODUCER.partition_owner)
    monkeypatch.setattr(model, 'verify_image', lambda image: 'id')
    monkeypatch.setattr(model, 'producer_parts', lambda root: parts)
    with pytest.raises(ValueError) as error:
        model.main(['prepare', '--host', 'sparky'])
    assert type(error.value) is ValueError
    assert str(error.value) == 'source changed during preparation'
    assert error.value.__cause__ is None
    assert not model.prepared_path(spec, 'sparky').exists()

    weights.write_bytes(b'original')
    clock[0] = max(p.stat().st_ctime_ns for p in source.iterdir()) + 10_000_000_000
    snapshots = model.verified_inputs(source, identity, cache)
    real_digest = model.digest_file
    calls = []

    def digest(path):
        calls.append(Path(path).name)
        return real_digest(path)

    monkeypatch.setattr(model, 'digest_file', digest)
    model.verified_inputs(source, identity, cache)
    assert calls == []
    historical = {name: dict(record, stamp=record['stamp'][:5]) for name, record in snapshots.items()}
    model.check_stamps(source, historical)
    assert 'weights' in calls
    cache_path = cache / (model.digest_json([str(source.resolve()),
                         {**identity['files'], 'config.json': identity['config_sha256']}]) + '.json')
    model.atomic_json(cache_path, historical)
    calls.clear()
    model.verified_inputs(source, identity, cache)
    assert 'weights' in calls
    rewrite()
    with pytest.raises(ValueError, match='^source changed during export: weights$'):
        model.check_stamps(source, historical)


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
    parts = SimpleNamespace(
        source_identity=lambda path: {'tensors': {'model.layers.0.w': 'weights'}},
        BODY_LAYER=PRODUCER.BODY_LAYER, partition_owner=PRODUCER.partition_owner)
    monkeypatch.setattr(model, 'verify_image', lambda image: 'id')
    monkeypatch.setattr(model, 'producer_parts', lambda root: parts)
    with pytest.raises(ValueError, match='needs --host'):
        model.main(['prepare'])
    assert model.main(['prepare', '--host', 'sparky']) == 0
    assert model.read_json(model.prepared_path(spec, 'sparky')) == {
        'source_identity': {'tensors': {'model.layers.0.w': 'weights'}}, 'count': 1}


# --- the whole-model dispatcher, rewired onto the campaign submitter ----------------------

def _campaign(monkeypatch, *, statuses, returncode=0):
    calls = []
    def submit(rows, state, *, afterok=()):
        calls.append((Path(state).name, list(rows), False))
        return []
    def collect(state, *, wait=False):
        return [{'job_id': str(index + 1), 'status': status,
                 'payload_exit': 0 if status == 'succeeded' else 7,
                 'query_error': None} for index, status in enumerate(statuses)]
    monkeypatch.setattr(common.campaign_rows, 'submit_rows', submit)
    monkeypatch.setattr(common.campaign_rows, 'collect_rows', collect)
    return calls


def test_a_failed_partition_cannot_pass_the_assembly_barrier(tmp_path, monkeypatch):
    _campaign(monkeypatch, statuses=['failed'], returncode=1)
    with pytest.raises(ValueError, match='0 of 1 action'):
        common.run_stage([{}], tmp_path, 'encode', wait_s=1)
    endings = json.loads((tmp_path / common.STATE / 'encode' / 'endings.json').read_text())
    assert endings['rows'][0]['status'] == 'failed'
    assert not (tmp_path / 'barrier.json').exists()


def test_a_stage_that_ran_fewer_rows_than_it_submitted_is_refused(tmp_path, monkeypatch):
    _campaign(monkeypatch, statuses=['succeeded'])
    with pytest.raises(ValueError, match='1 of 2 action'):
        common.run_stage([{}, {}], tmp_path, 'encode', wait_s=1)
    _campaign(monkeypatch, statuses=['succeeded', 'succeeded'])
    assert len(common.run_stage([{}, {}], tmp_path, 'encode', wait_s=1)) == 2


def test_preparation_runs_on_every_named_host_and_they_must_agree(tmp_path, monkeypatch):
    calls = _campaign(monkeypatch, statuses=['succeeded', 'succeeded'])
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


@pytest.mark.parametrize('statuses,expected', [
    (['succeeded', 'succeeded'], common.EXIT_DONE),
    (['succeeded', 'running'], common.EXIT_RUNNING),
    (['failed', 'running'], common.EXIT_FAILED),
    (['succeeded'], common.EXIT_FAILED),
])
def test_the_exit_status_is_the_worst_ending(statuses, expected):
    assert common.exit_status([{'status': s} for s in statuses], 2) == expected


# --- nothing here reaches PrismaBuild internals -------------------------------

@pytest.mark.parametrize('module', ['tessera_fleet.dispatch_model',
                                    'tessera_fleet.model_worker', 'render_identity'])
def test_help_runs_without_the_fleet(module):
    """``--help`` imports each driver the way an operator runs it, and exits 0."""
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, '-m', f'tools.{module}', '--help'],
                               cwd=root, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert 'usage:' in completed.stdout


def test_the_unused_show_status_stays_deleted():
    assert not {"show_status", "EXIT_NOTHING"} & set(vars(common))


def test_dispatch_workspace_commit_includes_ignored_payload_and_detaches(tmp_path):
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    (tmp_path / ".git" / "info" / "exclude").write_text("*.bin\n")
    (tmp_path / "weights.bin").write_bytes(b"staged-weights")
    common.initialize_git(tmp_path, message="test staged workspace")
    git = ["git", "-C", str(tmp_path)]
    assert subprocess.check_output([*git, "show", "HEAD:weights.bin"]) == b"staged-weights"
    assert subprocess.check_output([*git, "status", "--porcelain"]) == b""
    assert subprocess.run([*git, "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode == 1
