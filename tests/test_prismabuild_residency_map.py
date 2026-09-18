"""The stage tier is only worth building if the consumer actually reads it.

Every test here mutates the driver -- the map PrismaBuild writes, or the bytes
on the stage -- rather than the fixture, so a passing assertion says the
reader's own check bit, not that a fixture happened to agree with itself.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from prismaquant import residency_map
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.residency_map import (
    ENV_VAR, SCHEMA, bind_residency_manifest, residency_map_key,
    residency_report, residency_resolver, reset_residency_resolver_for_tests,
)


MANIFEST = 'a' * 64
LEAD = 'b' * 64


FMT = 'TESSERA_E4M3_K1_R1024'


@pytest.fixture(autouse=True)
def _forget_resolver(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    yield
    reset_residency_resolver_for_tests()


def _pool_cache(tmp_path, count=1, budget=1 << 20):
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    paths, tensors = {}, {}
    for index in range(count):
        key = (f'unit{index}', FMT)
        tensors[key] = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) + index
        paths[key] = pool / f'unit{index}.pt'
        torch.save(tensors[key], paths[key])
    cache = ProductionWeightCache(weights={k: str(v) for k, v in paths.items()}, levers={})
    cache.enable_lru(budget)
    return cache, paths, tensors


def _stage(tmp_path, paths, *, corrupt=()):
    """Copy the pool files onto a stage root; ``corrupt`` keeps size, not bytes."""
    root = tmp_path / 'stage' / 'prewarm'
    root.mkdir(parents=True)
    staged = {}
    for key, path in paths.items():
        blob = path.read_bytes()
        if key in corrupt:
            blob = blob[:-1] + bytes([blob[-1] ^ 0xFF])
        target = root / path.name
        target.write_bytes(blob)
        staged[key] = target
    return root, staged


def _write_map(tmp_path, root, paths, staged, *, digest_of=None, name='residency.json',
               schema=SCHEMA, entries=None, manifest_sha256=MANIFEST, extra=None):
    digest_of = digest_of or paths
    body = {
        'schema': schema,
        'tier_id': 'prismabuild-stage:dl380g10',
        'stage_root': str(root),
        'manifest_sha256': manifest_sha256,
        'leads': [LEAD],
        'generation': len(paths),
        'entries': {
            residency_map_key(str(paths[key]), 0): {
                'stage_path': str(staged[key]),
                'bytes': paths[key].stat().st_size,
                'offset': 0,
                'sha256': hashlib.sha256(Path(digest_of[key]).read_bytes()).hexdigest(),
            }
            for key in paths
        } if entries is None else entries,
        **(extra or {}),
    }
    path = tmp_path / name
    path.write_text(json.dumps(body))
    return path


def _bind(digest=MANIFEST):
    """The reader only serves the read set the pass was submitted with."""
    bind_residency_manifest(digest)


def _prepare(cache, paths, *, expected=None):
    """Enable the digest-fused load the joint stages use."""
    bound = max(path.stat().st_size for path in paths.values())
    if expected is None:
        cache.enable_file_load_receipts(max_file_bytes=bound)
    else:
        cache.require_file_load_sha256(expected, max_file_bytes=bound)


# --------------------------------------------------------------------------
# inert without the variable
# --------------------------------------------------------------------------

def test_resolver_is_inert_and_the_record_gains_no_key_without_the_variable(tmp_path):
    cache, paths, tensors = _pool_cache(tmp_path)
    root, staged = _stage(tmp_path, paths)
    _write_map(tmp_path, root, paths, staged)
    assert os.environ.get(ENV_VAR) is None
    assert residency_resolver() is None
    assert residency_report() is None
    _prepare(cache, paths)
    key, path = next(iter(paths.items()))
    assert cache.prefetch([key], max_workers=1) == 1
    tensor = cache.get(*key)
    assert torch.equal(tensor, tensors[key])
    # The bytes came from the declared path even though a valid map exists.
    assert cache.file_load_receipt(key, tensor)['path'] == str(path)
    assert residency_report() is None


def test_an_unset_variable_builds_no_resolver_at_all(tmp_path, monkeypatch):
    cache, paths, tensors = _pool_cache(tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError('a resolver was built with no map named')

    monkeypatch.setattr(residency_map, 'ResidencyResolver', refuse)
    _prepare(cache, paths)
    key = next(iter(paths))
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensors[key])


# --------------------------------------------------------------------------
# the redirect
# --------------------------------------------------------------------------

def test_a_staged_render_is_read_from_the_stage_and_counted(tmp_path, monkeypatch):
    cache, paths, tensors = _pool_cache(tmp_path, count=2)
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch(list(paths), max_workers=1) == 2
    for key in paths:
        tensor = cache.get(*key)
        assert torch.equal(tensor, tensors[key])
        assert cache.file_load_receipt(key, tensor)['path'] == str(staged[key])
    report = residency_report()
    assert report['hits'] == 2 and report['misses'] == 0 and report['fallbacks'] == []
    assert report['bytes_from_stage'] == sum(p.stat().st_size for p in paths.values())
    assert report['bytes_from_pool'] == 0
    assert report['entries'] == 2 and report['map_path'] == str(map_path)
    assert report['map_sha256'] == hashlib.sha256(map_path.read_bytes()).hexdigest()
    assert report['tier_id'] == 'prismabuild-stage:dl380g10'
    assert 'refused' not in report


def test_a_path_the_map_does_not_name_is_a_miss_read_from_the_pool(tmp_path, monkeypatch):
    cache, paths, tensors = _pool_cache(tmp_path, count=2)
    named, other = sorted(paths)
    root, staged = _stage(tmp_path, {named: paths[named]})
    map_path = _write_map(tmp_path, root, {named: paths[named]}, staged)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch([named, other], max_workers=1) == 2
    for key in (named, other):
        assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert report['hits'] == 1 and report['misses'] == 1 and report['fallbacks'] == []
    assert report['bytes_from_stage'] == paths[named].stat().st_size
    assert report['bytes_from_pool'] == paths[other].stat().st_size


def test_the_reader_starts_on_the_first_resident_entry(tmp_path, monkeypatch):
    """A partial map serves what it holds; the rest reads the pool now."""
    cache, paths, tensors = _pool_cache(tmp_path, count=3)
    first = sorted(paths)[0]
    root, staged = _stage(tmp_path, {first: paths[first]})
    map_path = _write_map(tmp_path, root, {first: paths[first]}, staged)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch(sorted(paths), max_workers=1) == 3
    for key in paths:
        assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert report['entries'] == 1 and report['hits'] == 1 and report['misses'] == 2


def test_a_map_replaced_mid_run_is_re_read(tmp_path, monkeypatch):
    cache, paths, tensors = _pool_cache(tmp_path, count=2)
    early, late = sorted(paths)
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, {early: paths[early]},
                          {early: staged[early]})
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch([early], max_workers=1) == 1
    grown = _write_map(tmp_path, root, paths, staged, name='grown.json')
    os.replace(grown, map_path)
    assert cache.prefetch([late], max_workers=1) == 1
    for key in paths:
        assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert report['entries'] == 2 and report['hits'] == 2 and report['misses'] == 0


# --------------------------------------------------------------------------
# refused entries
# --------------------------------------------------------------------------

def test_a_map_digest_that_differs_refuses_that_entry_before_opening_it(tmp_path, monkeypatch):
    cache, paths, tensors = _pool_cache(tmp_path)
    key, path = next(iter(paths.items()))
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    body = json.loads(map_path.read_text())
    body['entries'][str(path)]['sha256'] = '0' * 64
    map_path.write_text(json.dumps(body))
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    expected = {key: hashlib.sha256(path.read_bytes()).hexdigest()}
    _prepare(cache, paths, expected=expected)
    assert cache.prefetch([key], max_workers=1) == 1
    tensor = cache.get(*key)
    assert torch.equal(tensor, tensors[key])
    assert cache.file_load_receipt(key, tensor)['path'] == str(path)
    report = residency_report()
    assert report['hits'] == 0 and report['bytes_from_stage'] == 0
    assert report['bytes_from_pool'] == path.stat().st_size
    assert len(report['fallbacks']) == 1
    assert report['fallbacks'][0]['path'] == str(path)
    assert 'digest' in report['fallbacks'][0]['reason']


def test_staged_bytes_that_differ_are_refused_after_the_read(tmp_path, monkeypatch):
    """Same size, same published digest, different bytes: the read refuses."""
    cache, paths, tensors = _pool_cache(tmp_path)
    key, path = next(iter(paths.items()))
    root, staged = _stage(tmp_path, paths, corrupt={key})
    assert staged[key].stat().st_size == path.stat().st_size
    assert staged[key].read_bytes() != path.read_bytes()
    map_path = _write_map(tmp_path, root, paths, staged, digest_of=paths)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch([key], max_workers=1) == 1
    tensor = cache.get(*key)
    assert torch.equal(tensor, tensors[key])
    assert cache.file_load_receipt(key, tensor)['path'] == str(path)
    report = residency_report()
    assert report['hits'] == 0 and report['bytes_from_stage'] == 0
    assert report['bytes_from_pool'] == path.stat().st_size
    assert len(report['fallbacks']) == 1
    assert 'differ from the map digest' in report['fallbacks'][0]['reason']


def test_a_missing_staged_copy_refuses_that_entry(tmp_path, monkeypatch):
    cache, paths, tensors = _pool_cache(tmp_path)
    key, path = next(iter(paths.items()))
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    staged[key].unlink()
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert report['hits'] == 0 and report['bytes_from_pool'] == path.stat().st_size
    assert 'unreadable' in report['fallbacks'][0]['reason']


def test_a_staged_copy_of_another_size_refuses_that_entry(tmp_path, monkeypatch):
    cache, paths, tensors = _pool_cache(tmp_path)
    key, path = next(iter(paths.items()))
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    staged[key].write_bytes(path.read_bytes() + b'\0')
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert report['hits'] == 0 and 'size differs' in report['fallbacks'][0]['reason']


# --------------------------------------------------------------------------
# refused maps
# --------------------------------------------------------------------------

def _refusal(tmp_path, monkeypatch, map_path):
    cache, paths, tensors = _pool_cache(tmp_path.joinpath('tree'))
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    key = next(iter(paths))
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensors[key])
    return residency_report()


def test_a_map_of_another_schema_is_refused_whole_with_a_reason(tmp_path, monkeypatch):
    tmp_path.joinpath('tree').mkdir()
    cache, paths, _ = _pool_cache(tmp_path / 'other')
    root, staged = _stage(tmp_path / 'other', paths)
    map_path = _write_map(tmp_path, root, paths, staged,
                          schema='prismaquant.prismabuild.residency_map.v2')
    report = _refusal(tmp_path, monkeypatch, map_path)
    assert report['entries'] == 0 and report['hits'] == 0
    assert 'residency_map.v2' in report['refused'] and SCHEMA in report['refused']
    assert report['map_sha256'] is None


def test_a_ranged_roster_is_refused_whole_and_the_shape_is_named(tmp_path, monkeypatch):
    """The revision-2 shape (a list of range entries) is not this schema."""
    tmp_path.joinpath('tree').mkdir()
    cache, paths, _ = _pool_cache(tmp_path / 'other')
    root, staged = _stage(tmp_path / 'other', paths)
    key = next(iter(paths))
    map_path = _write_map(tmp_path, root, paths, staged, entries=[
        {'index': 0, 'path': str(paths[key]), 'offset': 0,
         'bytes': paths[key].stat().st_size, 'resident': True}])
    report = _refusal(tmp_path, monkeypatch, map_path)
    assert report['entries'] == 0 and 'ranged roster' in report['refused']


def test_a_map_that_is_not_there_yet_is_recorded_not_fatal(tmp_path, monkeypatch):
    tmp_path.joinpath('tree').mkdir()
    report = _refusal(tmp_path, monkeypatch, tmp_path / 'not-written-yet.json')
    assert report['entries'] == 0 and 'unreadable' in report['refused']


def test_a_staged_path_outside_the_declared_root_is_refused_whole(tmp_path, monkeypatch):
    tmp_path.joinpath('tree').mkdir()
    cache, paths, _ = _pool_cache(tmp_path / 'other')
    root, staged = _stage(tmp_path / 'other', paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    body = json.loads(map_path.read_text())
    key = next(iter(body['entries']))
    body['entries'][key]['stage_path'] = str(tmp_path / 'elsewhere.pt')
    map_path.write_text(json.dumps(body))
    report = _refusal(tmp_path, monkeypatch, map_path)
    assert report['entries'] == 0 and 'staged outside' in report['refused']


def test_a_map_that_appears_later_is_adopted_on_the_next_read(tmp_path, monkeypatch):
    cache, paths, tensors = _pool_cache(tmp_path, count=2)
    early, late = sorted(paths)
    root, staged = _stage(tmp_path, paths)
    map_path = tmp_path / 'residency.json'
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch([early], max_workers=1) == 1
    assert residency_report()['refused'].startswith('residency map is unreadable')
    written = _write_map(tmp_path, root, paths, staged, name='written.json')
    os.replace(written, map_path)
    assert cache.prefetch([late], max_workers=1) == 1
    for key in paths:
        assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert 'refused' not in report and report['hits'] == 1 and report['entries'] == 2


def test_a_map_for_another_read_set_is_refused_whole(tmp_path, monkeypatch):
    tmp_path.joinpath('tree').mkdir()
    cache, paths, _ = _pool_cache(tmp_path / 'other')
    root, staged = _stage(tmp_path / 'other', paths)
    map_path = _write_map(tmp_path, root, paths, staged, manifest_sha256='c' * 64)
    report = _refusal(tmp_path, monkeypatch, map_path)
    assert report['entries'] == 0 and report['hits'] == 0
    assert 'names data manifest' in report['refused']


def test_an_unbound_read_set_gets_no_redirect(tmp_path, monkeypatch):
    cache, paths, tensors = _pool_cache(tmp_path)
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _prepare(cache, paths)
    key = next(iter(paths))
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert report['entries'] == 0 and report['hits'] == 0
    assert 'no data manifest digest' in report['refused']


def test_an_unknown_field_refuses_the_map_rather_than_being_ignored(tmp_path, monkeypatch):
    tmp_path.joinpath('tree').mkdir()
    cache, paths, _ = _pool_cache(tmp_path / 'other')
    root, staged = _stage(tmp_path / 'other', paths)
    map_path = _write_map(tmp_path, root, paths, staged, extra={'mount_prefix': '/mnt/shared'})
    report = _refusal(tmp_path, monkeypatch, map_path)
    assert report['entries'] == 0 and "'mount_prefix'" in report['refused']


def test_an_unknown_entry_field_refuses_the_map(tmp_path, monkeypatch):
    tmp_path.joinpath('tree').mkdir()
    cache, paths, _ = _pool_cache(tmp_path / 'other')
    root, staged = _stage(tmp_path / 'other', paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    body = json.loads(map_path.read_text())
    key = next(iter(body['entries']))
    body['entries'][key]['resident'] = True
    map_path.write_text(json.dumps(body))
    report = _refusal(tmp_path, monkeypatch, map_path)
    assert report['entries'] == 0 and "'resident'" in report['refused']


def test_an_entry_offset_that_disagrees_with_its_key_refuses_the_map(tmp_path, monkeypatch):
    tmp_path.joinpath('tree').mkdir()
    cache, paths, _ = _pool_cache(tmp_path / 'other')
    root, staged = _stage(tmp_path / 'other', paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    body = json.loads(map_path.read_text())
    key = next(iter(body['entries']))
    body['entries'][key]['offset'] = 4096
    map_path.write_text(json.dumps(body))
    report = _refusal(tmp_path, monkeypatch, map_path)
    assert report['entries'] == 0 and 'disagrees with its key' in report['refused']


def test_a_byte_range_entry_is_refused_for_a_whole_file_read(tmp_path, monkeypatch):
    """A range of a shard is a valid map entry and a wrong answer here."""
    cache, paths, tensors = _pool_cache(tmp_path)
    key, path = next(iter(paths.items()))
    root, staged = _stage(tmp_path, paths)
    partial = root / 'partial.range'
    blob = path.read_bytes()[:64]
    partial.write_bytes(blob)
    map_path = _write_map(tmp_path, root, paths, staged, entries={
        residency_map_key(str(path), 0): {
            'stage_path': str(partial), 'bytes': len(blob), 'offset': 0,
            'sha256': hashlib.sha256(blob).hexdigest()}})
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert report['hits'] == 0 and report['bytes_from_pool'] == path.stat().st_size
    assert 'byte range' in report['fallbacks'][0]['reason']


def test_an_entry_at_another_offset_is_not_a_whole_file_answer(tmp_path, monkeypatch):
    cache, paths, tensors = _pool_cache(tmp_path)
    key, path = next(iter(paths.items()))
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, paths, staged, entries={
        residency_map_key(str(path), 4096): {
            'stage_path': str(staged[key]), 'bytes': path.stat().st_size,
            'offset': 4096,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}})
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert report['entries'] == 1 and report['hits'] == 0 and report['misses'] == 1


def test_a_resolved_spelling_still_finds_the_manifests_own_spelling(tmp_path, monkeypatch):
    """The manifest never resolves symlinks; several readers do."""
    cache, paths, tensors = _pool_cache(tmp_path)
    key, path = next(iter(paths.items()))
    root, staged = _stage(tmp_path, paths)
    link = tmp_path / 'link'
    link.symlink_to(tmp_path / 'pool')
    declared = link / path.name
    map_path = _write_map(tmp_path, root, {key: declared}, staged,
                          digest_of={key: path})
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    _prepare(cache, paths)
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert report['hits'] == 1 and report['bytes_from_stage'] == path.stat().st_size


# --------------------------------------------------------------------------
# the wire reader
# --------------------------------------------------------------------------

def _wire_cell(tmp_path, *, blob=b'tessera-wire-bytes-0123456789'):
    pool = tmp_path / 'pool'
    pool.mkdir(exist_ok=True)
    wire = pool / 'cell.wire'
    wire.write_bytes(blob)
    return {'wire': str(wire), 'record': {
        'blob_bytes': len(blob), 'blob_sha256': hashlib.sha256(blob).hexdigest()}}, wire, blob


def test_a_staged_wire_is_read_from_the_stage(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire_cell(tmp_path)
    paths = {'w': wire}
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = residency_report()
    assert report['hits'] == 1 and report['bytes_from_stage'] == len(blob)


def test_a_staged_wire_whose_bytes_differ_falls_back_to_the_pool(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire_cell(tmp_path)
    paths = {'w': wire}
    root, staged = _stage(tmp_path, paths, corrupt={'w'})
    map_path = _write_map(tmp_path, root, paths, staged, digest_of=paths)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = residency_report()
    assert report['hits'] == 0 and report['bytes_from_pool'] == len(blob)
    assert 'receipt digest' in report['fallbacks'][0]['reason']


def test_the_wire_reader_is_unchanged_without_the_variable(tmp_path):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire_cell(tmp_path)
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == hashlib.sha256(blob).hexdigest()
    assert residency_report() is None


# --------------------------------------------------------------------------
# the submitter flag
# --------------------------------------------------------------------------

def _argv(residency):
    import types
    from tools.dispatch_tessera_campaign import _pbrun_argv
    spec = Path('/dev/null')
    args = types.SimpleNamespace(
        spec=spec, pbrun='/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py',
        demand='gpu=1,mem_gb=101', cpus=6, tag='sparky', priority=-10,
        timeout_s=None, container_arg=None, residency=residency, head_grace_s=None)
    return _pbrun_argv(args, manifest=Path('/tmp/m.json.gz'), inner=['--resume'],
                       container_spec={'image': 'x'})


def test_the_submitter_passes_residency_stage_through_to_pbrun():
    argv = _argv('stage')
    assert '--residency' in argv
    assert argv[argv.index('--residency') + 1] == 'stage'
    # A pbrun option, so it precedes the action separator.
    assert argv.index('--residency') < argv.index('--')


def test_the_submitter_omits_the_flag_by_default():
    assert '--residency' not in _argv(None)
