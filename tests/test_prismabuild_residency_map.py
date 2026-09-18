"""The stage tier is only worth building if the consumer actually reads it.

Every test here mutates the driver -- the map PrismaBuild writes, or the bytes
on the stage -- rather than the fixture, so a passing assertion says the
reader's own check bit, not that a fixture happened to agree with itself.
"""
import hashlib
import json
import os
import shlex
from pathlib import Path

import pytest
import torch

from prismaquant import residency_map
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.residency_map import (
    ENV_VAR, SCHEMA, bind_residency_manifest, residency_map_key,
    residency_report, residency_resolver, reset_residency_resolver_for_tests,
)
# The submitter tests at the end of this file run one real submit-joint dry
# run rather than a hand-built namespace, so what they assert is the argv the
# tool builds.
from test_glm_joint_data_manifest_at_submit import scratch, shared_mount  # noqa: F401


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
        # The receipt is about the declared object, so its path and the
        # signature its lifetime fence re-checks stay the pool's even when the
        # bytes came from the stage. Which copy served them is the residency
        # report's business, not the content receipt's.
        assert cache.file_load_receipt(key, tensor)['path'] == str(paths[key])
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
    body['entries'][residency_map_key(str(path), 0)]['sha256'] = '0' * 64
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


def test_a_declared_file_that_cannot_be_stated_is_a_fallback_not_a_shortcut(
    tmp_path, monkeypatch,
):
    """The declared file's own length is what binds a whole-file entry.

    Without it a byte-range entry would be served as a whole file, and a pool
    path that has gone away would succeed from the stage where reading it
    directly fails closed. The map redirects a read; it does not stand in for
    one that no longer has an object to redirect.
    """
    cache, paths, tensors = _pool_cache(tmp_path)
    key, path = next(iter(paths.items()))
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    resolver = residency_resolver()
    path.unlink()
    assert resolver.staged_read(str(path)) is None
    report = residency_report()
    assert report['hits'] == 0 and report['bytes_from_stage'] == 0
    assert report['fallbacks'][0]['path'] == str(path)
    assert report['fallbacks'][0]['reason'] == (
        'declared file is unstatable, cannot bind the entry to it')


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


def test_a_staged_copy_released_between_the_stat_and_the_open_falls_back(tmp_path, monkeypatch):
    """PrismaBuild recomposes after every egress; a vanished copy is a miss."""
    cache, paths, tensors = _pool_cache(tmp_path)
    key, path = next(iter(paths.items()))
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    original = Path.open

    def vanishing(candidate, *args, **kwargs):
        if candidate == staged[key]:
            candidate.unlink()
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', vanishing)
    _prepare(cache, paths)
    assert cache.prefetch([key], max_workers=1) == 1
    assert torch.equal(cache.get(*key), tensors[key])
    report = residency_report()
    assert report['hits'] == 0 and report['bytes_from_pool'] == path.stat().st_size
    assert 'unreadable' in report['fallbacks'][0]['reason']


def test_a_staged_wire_released_between_the_stat_and_the_open_falls_back(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    cell, wire, blob = _wire_cell(tmp_path)
    paths = {'w': wire}
    root, staged = _stage(tmp_path, paths)
    map_path = _write_map(tmp_path, root, paths, staged)
    monkeypatch.setenv(ENV_VAR, str(map_path))
    _bind()
    original = os.open

    def vanishing(candidate, *args, **kwargs):
        if str(candidate) == str(staged['w']):
            staged['w'].unlink()
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(os, 'open', vanishing)
    read, digest = _read_verified_wire_blob(cell)
    assert read == blob and digest == cell['record']['blob_sha256']
    report = residency_report()
    assert report['hits'] == 0 and report['bytes_from_pool'] == len(blob)
    assert 'unreadable' in report['fallbacks'][0]['reason']


# --------------------------------------------------------------------------
# the submitter flag
# --------------------------------------------------------------------------

def _argv(residency, *, on_args=None):
    import types
    from tools.dispatch_tessera_campaign import _pbrun_argv
    spec = Path('/dev/null')
    args = types.SimpleNamespace(
        spec=spec, pbrun='/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py',
        demand='gpu=1,mem_gb=101', cpus=6, tag='sparky', priority=-10,
        timeout_s=None, container_arg=None, residency=on_args, head_grace_s=None)
    return _pbrun_argv(args, manifest=Path('/dev/null'), inner=['--resume'],
                       residency=residency, container_spec={'image': 'x'})


def test_the_submitter_passes_residency_stage_through_to_pbrun():
    argv = _argv('stage')
    assert '--residency' in argv
    assert argv[argv.index('--residency') + 1] == 'stage'
    # A pbrun option, so it precedes the action separator.
    assert argv.index('--residency') < argv.index('--')


def test_the_submitter_omits_the_flag_by_default():
    assert '--residency' not in _argv(None)


def test_the_flag_on_the_namespace_alone_emits_nothing():
    """Only the caller knows the entry point, so only the caller may ask.

    ``--residency`` is shared by four subcommands and read through the
    resolver by one. If the argv builder took it off the namespace, adding a
    subcommand would silently opt it into a tier reservation nothing consumes.
    """
    assert '--residency' not in _argv(None, on_args='stage')


def test_a_binding_that_is_not_a_digest_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / 'residency.json'))
    with pytest.raises(ValueError, match='64 lowercase hex'):
        bind_residency_manifest('not-a-digest')


def test_a_non_joint_submission_refuses_the_flag_and_says_why(monkeypatch):
    """A reservation nothing reads is the failure this change exists to stop.

    ``--residency stage`` reserves cluster-scoped tier capacity and narrows
    placement to the boxes that mount the tier. Allocation, export and AQUA
    read every byte from the pool, so for them the flag would buy that
    narrowing and be consumed by nothing at all.
    """
    import types
    from tools.dispatch_tessera_campaign import (
        ALLOCATION_ENTRY_POINT, AQUA_ENTRY_POINT, EXPORT_ENTRY_POINT,
        _submit_gpu_action,
    )
    args = types.SimpleNamespace(residency='stage', spec='/dev/null')

    def unreachable():
        raise AssertionError('the manifest was built for a refused submission')

    for entry_point in (ALLOCATION_ENTRY_POINT, EXPORT_ENTRY_POINT, AQUA_ENTRY_POINT):
        with pytest.raises(RuntimeError, match='joint-only') as refusal:
            _submit_gpu_action(args, entry_point=entry_point, command='handoff',
                               inner=[], plan={}, build=unreachable)
        # The message names the row that asked, so the operator sees which
        # submission to resubmit without it.
        assert entry_point in str(refusal.value)
        assert 'from the pool' in str(refusal.value)


def test_a_non_joint_submission_without_the_flag_passes_the_gate():
    """The gate bites on the flag, not on the entry point."""
    import types
    from tools.dispatch_tessera_campaign import ALLOCATION_ENTRY_POINT, _submit_gpu_action
    args = types.SimpleNamespace(residency=None, spec=str(Path('/dev/null')))
    # Past the gate it fails on the spec, which is the next thing it reads --
    # any failure but the residency refusal proves the gate let it through.
    with pytest.raises(Exception) as outcome:
        _submit_gpu_action(args, entry_point=ALLOCATION_ENTRY_POINT,
                           command='handoff', inner=[], plan={},
                           build=lambda: None)
    assert 'joint-only' not in str(outcome.value)


def _dry_run_joint(scratch, monkeypatch, capsys, *, residency):
    """One real submit-joint dry run, so the argv asserted is the argv built."""
    import dispatch_tessera_campaign as dispatch
    from experiments import glm_data_manifests
    import test_glm_joint_data_manifest_at_submit as fixtures

    fixture = fixtures._workspace(scratch)
    spec = scratch / 'spec.joint.json'
    spec.write_text(json.dumps({'container': {'image': 'x'}}))
    monkeypatch.setattr(dispatch, '_manifest_producer', lambda: glm_data_manifests)
    argv = [
        'submit-joint', 'prepare',
        '--plan', str(fixture['plan']),
        *fixtures._scope_args(fixture),
        '--spec', str(spec),
        '--demand', 'gpu=1,mem_gb=104',
        '--cpus', '6', '--tag', 'gb10', '--priority', '-10',
        '--manifest-dir', str(scratch / 'manifests'),
        '--dry-run',
    ]
    if residency is not None:
        argv += ['--residency', residency]
    assert dispatch.main(argv) == 0
    printed = capsys.readouterr().out
    lines = printed.splitlines()
    at = next(n for n, line in enumerate(lines) if line.startswith('[dry-run] '))
    # The command is shlex-quoted and the container spec inside it is itself
    # JSON, so the summary is the object printed *after* that line, not the
    # first brace in the output.
    command = shlex.split(lines[at][len('[dry-run] '):])
    rest = '\n'.join(lines[at + 1:])
    summary, _ = json.JSONDecoder().raw_decode(rest[rest.index('{'):])
    return command, summary


def test_the_joint_row_carries_the_manifest_digest_it_was_submitted_with(
    scratch, shared_mount, monkeypatch, capsys,
):
    """The identity half of the design: unbound, the reader serves nothing.

    ``bind_manifest_sha256`` refuses every map until the pass says which read
    set it holds, and the only place that digest exists at submit time is
    here. Without this argv the resolver is inert on exactly the gated stage.
    """
    argv, summary = _dry_run_joint(scratch, monkeypatch, capsys, residency='stage')
    assert '--data-manifest-sha256' in argv
    digest = argv[argv.index('--data-manifest-sha256') + 1]
    # The digest is the manifest's own -- the one pbrun seals into the action
    # key -- so the action is told a fact it already runs under.
    assert digest == summary['manifest_sha256']
    # It is an argument of the action, not of pbrun.
    assert argv.index('--') < argv.index('--data-manifest-sha256')
    assert argv.index('--residency') < argv.index('--')
    assert summary['resource_demand']['residency'] == 'stage'


def test_a_joint_row_that_asks_for_no_stage_keeps_the_argv_it_has_today(
    scratch, shared_mount, monkeypatch, capsys,
):
    """No flag, no digest, no demand key: the action key is unchanged."""
    argv, summary = _dry_run_joint(scratch, monkeypatch, capsys, residency=None)
    assert '--data-manifest-sha256' not in argv
    assert '--residency' not in argv
    assert 'residency' not in summary['resource_demand']
