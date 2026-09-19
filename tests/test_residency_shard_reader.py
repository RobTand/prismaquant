"""A shard's tensors come off the stage, ranges included, or they come off the pool.

The stage holds byte ranges, not whole shards: PrismaBuild's mover writes each
declared range as a file of its own starting at byte 0 of the range, so 34 of
the live run's 55 staged shard entries are unreadable by a path rewrite and were
read from the pool by every source open (PQ #732).

Every test here mutates the driver -- the map PrismaBuild writes, or the bytes
on the stage -- rather than the fixture, and every served tensor is compared
bit for bit against what ``safetensors.safe_open`` returns from the pool file.
A test that only asserted "a staged read happened" would pass on a reader that
served the wrong bytes.
"""
import hashlib
import json
import os
import threading

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import layer_streaming, residency_shard_reader
from prismaquant.residency_map import (
    ENV_VAR, SCHEMA, bind_residency_manifest, residency_map_key,
    residency_resolver, reset_residency_resolver_for_tests,
)
from prismaquant.residency_shard_reader import staged_shard_opener


MANIFEST = 'c' * 64
LEAD = 'd' * 64


@pytest.fixture(autouse=True)
def _forget_resolver(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    yield
    reset_residency_resolver_for_tests()


def _shard(tmp_path):
    """One real safetensors file, wide enough that dtype handling is exercised.

    The shapes are chosen so the tensors land at distinct spans; the reader is
    asked for each of them by name and its answer is compared with the pool's.
    """
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    tensors = {
        'f32': torch.linspace(-3, 3, 256, dtype=torch.float32).reshape(8, 32),
        'bf16': (torch.arange(512, dtype=torch.int32).reshape(16, 32)
                 .to(torch.bfloat16)),
        'f16': torch.linspace(-1, 1, 128, dtype=torch.float16).reshape(8, 16),
        'f8': (torch.linspace(-2, 2, 64, dtype=torch.float32)
               .to(torch.float8_e4m3fn).reshape(8, 8)),
        'i8': torch.arange(-32, 32, dtype=torch.int8).reshape(8, 8),
        'u8': torch.arange(64, dtype=torch.uint8).reshape(8, 8),
        'flag': (torch.arange(64) % 3 == 0).reshape(8, 8),
        'empty': torch.zeros((0, 4), dtype=torch.float32),
    }
    path = pool / 'model-00001-of-00002.safetensors'
    save_file(tensors, str(path))
    return path, tensors


def _header(path):
    """The tensor spans as absolute file offsets, the way the reader sees them."""
    raw = path.read_bytes()
    size = int.from_bytes(raw[:8], 'little')
    body = json.loads(raw[8:8 + size])
    base = 8 + size
    return {name: (base + row['data_offsets'][0], base + row['data_offsets'][1])
            for name, row in body.items() if name != '__metadata__'}


def _stage_root(tmp_path):
    root = tmp_path / 'stage' / 'prewarm'
    root.mkdir(parents=True, exist_ok=True)
    return root


def _stage_range(root, path, offset, length, *, blob=None):
    """Stage one byte range the way PrismaBuild's mover writes it.

    ``tools/fleet/stage_move.py``: a range is its own file under
    ``<name>.pbrange/<offset>-<length>`` holding exactly that range, read by
    the consumer from position 0.
    """
    directory = root / f'{path.name}.pbrange'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'{offset}-{length}'
    target.write_bytes(path.read_bytes()[offset:offset + length] if blob is None
                       else blob)
    return target


def _stage_whole(root, path):
    target = root / path.name
    target.write_bytes(path.read_bytes())
    return target


def _write_map(tmp_path, root, rows, *, name='residency.json',
               manifest_sha256=MANIFEST):
    """``rows`` is [(declared Path, offset, bytes, staged Path)]."""
    entries = {}
    for declared, offset, size, staged in rows:
        entries[residency_map_key(str(declared), offset)] = {
            'stage_path': str(staged),
            'bytes': size,
            'offset': offset,
            'sha256': hashlib.sha256(
                declared.read_bytes()[offset:offset + size]).hexdigest(),
        }
    body = {
        'schema': SCHEMA,
        'tier_id': 'prismabuild-stage:dl380g10',
        'stage_root': str(root),
        'manifest_sha256': manifest_sha256,
        'leads': [LEAD],
        'generation': 3,
        'entries': entries,
    }
    path = tmp_path / name
    path.write_text(json.dumps(body))
    return path


def _bind(monkeypatch, map_path):
    monkeypatch.setenv(ENV_VAR, str(map_path))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    return residency_resolver()


def _identical(reader, pool_path, tensors):
    """Every tensor the reader hands back equals the pool's, bit for bit."""
    with safe_open(str(pool_path), framework='pt') as reference:
        for name in sorted(tensors):
            got, want = reader.get_tensor(name), reference.get_tensor(name)
            assert got.dtype == want.dtype, name
            assert got.shape == want.shape, name
            assert got.device == want.device, name
            if got.numel():
                assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), name


# -- the inert and unmapped paths -------------------------------------------

def test_without_a_map_the_opener_is_the_callers_own(tmp_path):
    path, _ = _shard(tmp_path)
    sentinel = object()
    assert staged_shard_opener(path, sentinel) is sentinel


def test_inert_source_open_calls_safe_open_with_the_same_arguments(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        return safe_open(*args, **kwargs)

    monkeypatch.setattr(layer_streaming, 'safe_open', recorder)
    with layer_streaming._source_safe_open(str(path), framework='pt') as handle:
        assert set(handle.keys()) == set(tensors)
    assert calls == [((str(path),), {'framework': 'pt'})]


def test_a_file_the_map_never_names_is_the_callers_own_opener(tmp_path, monkeypatch):
    path, _ = _shard(tmp_path)
    other = path.with_name('model-00002-of-00002.safetensors')
    other.write_bytes(path.read_bytes())
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, other)
    _bind(monkeypatch, _write_map(
        tmp_path, root, [(other, 0, other.stat().st_size, staged)]))
    sentinel = object()
    assert staged_shard_opener(path, sentinel) is sentinel
    assert staged_shard_opener(other, sentinel) is not sentinel


# -- whole-file and range entries -------------------------------------------

def test_a_whole_file_entry_serves_every_tensor(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, 0, path.stat().st_size, staged)]))
    spans = _header(path)
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        _identical(reader, path, tensors)
    report = resolver.report()
    served = [name for name in tensors if spans[name][1] > spans[name][0]]
    assert report['range_hits'] == len(served)
    assert report['bytes_from_stage'] == sum(
        spans[name][1] - spans[name][0] for name in served)
    assert report['fallback_count'] == 0


def test_only_the_tensors_a_range_covers_come_off_the_stage(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    spans = _header(path)
    root = _stage_root(tmp_path)
    # One range over the first two tensors by offset, nothing over the rest.
    ordered = sorted((span, name) for name, span in spans.items()
                     if span[1] > span[0])
    covered = [name for _, name in ordered[:2]]
    start, end = ordered[0][0][0], ordered[1][0][1]
    staged = _stage_range(root, path, start, end - start)
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, start, end - start, staged)]))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        _identical(reader, path, tensors)
    report = resolver.report()
    assert report['range_hits'] == len(covered)
    assert report['range_misses'] == len([n for n in tensors
                                          if spans[n][1] > spans[n][0]]) - len(covered)
    assert report['bytes_from_stage'] == sum(spans[n][1] - spans[n][0] for n in covered)
    assert report['bytes_from_pool'] > 0
    assert report['fallback_count'] == 0


def test_a_range_covering_part_of_a_tensor_serves_none_of_it(tmp_path, monkeypatch, capsys):
    path, tensors = _shard(tmp_path)
    spans = _header(path)
    root = _stage_root(tmp_path)
    start, end = spans['bf16']
    short = (end - start) // 2
    staged = _stage_range(root, path, start, short)
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, start, short, staged)]))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        _identical(reader, path, tensors)
    report = resolver.report()
    # Nothing is served: a partial cover is a miss, not a partial read, and it
    # is silent because a half-staged shard is the ordinary mid-flight state.
    assert report['range_hits'] == 0
    assert report['bytes_from_stage'] == 0
    assert report['fallback_count'] == 0
    assert '[residency] fallback' not in capsys.readouterr().out


def test_a_tensor_straddling_two_ranges_reads_the_pool(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    spans = _header(path)
    root = _stage_root(tmp_path)
    start, end = spans['bf16']
    split = start + (end - start) // 2
    rows = [(path, start, split - start, _stage_range(root, path, start, split - start)),
            (path, split, end - split, _stage_range(root, path, split, end - split))]
    resolver = _bind(monkeypatch, _write_map(tmp_path, root, rows))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        _identical(reader, path, tensors)
    report = resolver.report()
    assert report['range_hits'] == 0
    assert report['bytes_from_stage'] == 0
    assert report['fallback_count'] == 0


def test_a_staged_range_of_the_wrong_size_falls_back_and_says_so(tmp_path, monkeypatch, capsys):
    path, tensors = _shard(tmp_path)
    spans = _header(path)
    root = _stage_root(tmp_path)
    start, end = spans['f32']
    staged = _stage_range(root, path, start, end - start)
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, start, end - start, staged)]))
    staged.write_bytes(staged.read_bytes()[:-8])
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        _identical(reader, path, tensors)
    report = resolver.report()
    assert report['range_hits'] == 0
    assert report['fallback_count'] >= 1
    reasons = {row['reason'] for row in report['fallbacks']}
    assert 'staged copy size differs from the map' in reasons
    assert '[residency] fallback' in capsys.readouterr().out


def test_an_entry_running_past_the_declared_file_refuses(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    root = _stage_root(tmp_path)
    size = path.stat().st_size
    staged = _stage_whole(root, path)
    map_path = _write_map(tmp_path, root, [(path, 0, size, staged)])
    # The declared file shrinks under a map that still claims its old length:
    # the entry no longer fits the file it stands for.
    body = json.loads(map_path.read_text())
    body['entries'][residency_map_key(str(path), 0)]['bytes'] = size + 4096
    staged.write_bytes(staged.read_bytes() + b'\0' * 4096)
    map_path.write_text(json.dumps(body))
    resolver = _bind(monkeypatch, map_path)
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        _identical(reader, path, tensors)
    report = resolver.report()
    assert report['range_hits'] == 0
    assert 'map entry runs past the declared file' in {
        row['reason'] for row in report['fallbacks']}


def test_a_staged_range_that_is_not_a_regular_file_refuses(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    spans = _header(path)
    root = _stage_root(tmp_path)
    start, end = spans['f32']
    staged = _stage_range(root, path, start, end - start)
    map_path = _write_map(tmp_path, root, [(path, start, end - start, staged)])
    staged.unlink()
    staged.mkdir()
    resolver = _bind(monkeypatch, map_path)
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        _identical(reader, path, tensors)
    assert resolver.report()['range_hits'] == 0
    assert 'staged copy is not a regular file' in {
        row['reason'] for row in resolver.report()['fallbacks']}


# -- the descriptor-alias seam ----------------------------------------------

class _AliasOwner:
    """The one thing about the real capture owner this reader has to survive.

    ``tessera_calibration_cache._CaptureSourceSafeOpen`` calls the opener with
    ``/proc/self/fd/<fd>`` rather than the declared path, so an opener that
    looked the map up by the path it is handed would never match an entry.
    """

    def __init__(self):
        self.handed = None

    def safe_open(self, factory, path, *args, **kwargs):
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        self.fd = fd
        self.handed = f'/proc/self/fd/{fd}'
        return factory(self.handed, *args, **kwargs)

    def close(self):
        os.close(self.fd)


def test_the_reader_serves_through_a_descriptor_alias(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, 0, path.stat().st_size, staged)]))
    owner = _AliasOwner()
    try:
        context = layer_streaming._source_safe_open(
            str(path), framework='pt', source_authentication=owner)
        with context as reader:
            assert owner.handed is not None and owner.handed.startswith('/proc/self/fd/')
            _identical(reader, path, tensors)
    finally:
        owner.close()
    assert resolver.report()['range_hits'] > 0


# -- what the reader delegates ----------------------------------------------

def test_keys_metadata_and_slices_come_from_the_declared_file(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, 0, path.stat().st_size, staged)]))
    with safe_open(str(path), framework='pt') as reference:
        with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
            assert set(reader.keys()) == set(reference.keys())
            assert reader.metadata() == reference.metadata()
            for name in sorted(tensors):
                got, want = reader.get_slice(name), reference.get_slice(name)
                assert list(got.get_shape()) == list(want.get_shape())
                assert got.get_dtype() == want.get_dtype()


def test_a_staged_read_reports_hits_and_range_hits_together(tmp_path, monkeypatch):
    path, tensors = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, 0, path.stat().st_size, staged)]))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        reader.get_tensor('f32')
    report = resolver.report()
    assert report['range_hits'] == 1
    assert report['hits'] == 1


# -- the resolver's own contract --------------------------------------------

def test_staged_read_still_refuses_a_byte_range(tmp_path, monkeypatch):
    """The whole-file contract is untouched; its two consumers depend on it."""
    path, _ = _shard(tmp_path)
    spans = _header(path)
    root = _stage_root(tmp_path)
    start, end = spans['f32']
    staged = _stage_range(root, path, start, end - start)
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, 0, end - start, staged)]))
    assert resolver.staged_read(path) is None
    assert 'map entry is a byte range, not the whole declared file' in {
        row['reason'] for row in resolver.report()['fallbacks']}


def test_staged_range_refuses_a_backwards_span(tmp_path, monkeypatch):
    path, _ = _shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, 0, path.stat().st_size, staged)]))
    with pytest.raises(ValueError):
        resolver.staged_range(path, 64, 8)


def test_staged_range_is_rebuilt_when_the_map_is_replaced(tmp_path, monkeypatch):
    """A recomposed map must not be answered out of the previous map's index."""
    path, tensors = _shard(tmp_path)
    spans = _header(path)
    root = _stage_root(tmp_path)
    start, end = spans['f32']
    staged = _stage_range(root, path, start, end - start)
    map_path = _write_map(tmp_path, root, [(path, start, end - start, staged)])
    resolver = _bind(monkeypatch, map_path)
    assert resolver.staged_range(path, start, end) is not None
    body = json.loads(map_path.read_text())
    body['entries'] = {}
    body['generation'] = 4
    map_path.write_text(json.dumps(body))
    os.utime(map_path, ns=(0, 0))
    assert resolver.staged_range(path, start, end) is None
    assert resolver.stages(path) is False


def test_the_reader_reads_spans_larger_than_one_page(tmp_path, monkeypatch):
    """The read loop, on a span that is not a single small pread."""
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    big = {'w': torch.arange(1 << 20, dtype=torch.int32).to(torch.bfloat16).reshape(1024, 1024)}
    path = pool / 'model-00001-of-00001.safetensors'
    save_file(big, str(path))
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, 0, path.stat().st_size, staged)]))
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        _identical(reader, path, big)
    assert resolver.report()['range_hits'] == 1
    assert resolver.report()['bytes_from_stage'] == big['w'].numel() * 2


def test_the_reader_module_reads_safetensors_own_dtype_table():
    """No hand-rolled dtype map: the format's own spelling or no staged read."""
    assert residency_shard_reader._SAFETENSORS_DTYPES is not None
    assert residency_shard_reader._SAFETENSORS_DTYPES['BF16'] is torch.bfloat16


# -- the span read: one stream or the mount's own (PQ #746) ------------------

NFS_OPTIONS = ('rw,noatime,vers=4.2,rsize=1048576,wsize=1048576,namlen=255,hard,'
               'proto=rdma,nconnect=16,port=20049,timeo=600,retrans=2,sec=sys')


@pytest.fixture(autouse=True)
def _forget_read_shape():
    residency_shard_reader.reset_read_shape_cache_for_tests()
    yield
    residency_shard_reader.reset_read_shape_cache_for_tests()


def _mount_table(tmp_path, rows, *, name='mounts'):
    """A mount table of our own, in the kernel's own spelling."""
    path = tmp_path / name
    path.write_text(''.join(f'{device} {point} {fstype} {options} 0 0\n'
                            for device, point, fstype, options in rows))
    return path


def _use_table(monkeypatch, table):
    monkeypatch.setattr(residency_shard_reader, 'MOUNTS_PATH', str(table))
    residency_shard_reader.reset_read_shape_cache_for_tests()


def test_the_read_shape_is_the_mounts_own_two_numbers(tmp_path, monkeypatch):
    """nconnect streams, rsize chunks -- neither one a constant of ours."""
    _use_table(monkeypatch, _mount_table(tmp_path, [
        ('/dev/root', '/', 'ext4', 'rw'),
        ('10.100.99.3:/stage/prewarm', '/stage/prewarm', 'nfs4', NFS_OPTIONS),
    ]))
    assert residency_shard_reader._read_shape(
        '/stage/prewarm/model-00001.safetensors.pbrange/0-100') == (16, 1048576)


def test_the_nfs_mount_on_top_of_its_autofs_trigger_is_the_one_read(tmp_path, monkeypatch):
    """The live table, not a tidied one: two rows, same point, NFS on top.

    The first sweep of this change measured nothing because the autofs row
    answered first and publishes no transport count. The table below is the
    one /stage/prewarm actually has.
    """
    _use_table(monkeypatch, _mount_table(tmp_path, [
        ('systemd-1', '/stage/prewarm', 'autofs',
         'rw,relatime,fd=97,pgrp=1,timeout=0,minproto=5,maxproto=5,direct'),
        ('10.100.99.3:/stage/prewarm', '/stage/prewarm', 'nfs4', NFS_OPTIONS),
    ]))
    assert residency_shard_reader._mount_options('/stage/prewarm/x')[1] == 'nfs4'
    assert residency_shard_reader._read_shape('/stage/prewarm/x') == (16, 1048576)


def test_a_mount_that_publishes_no_transport_count_stays_serial(tmp_path, monkeypatch):
    """No explicit, no split: the read is the one it has always been."""
    for options in ('rw,rsize=1048576,vers=4.2',          # NFS without nconnect
                    'rw,nconnect=16,vers=4.2'):           # NFS without rsize
        _use_table(monkeypatch, _mount_table(tmp_path, [
            ('10.100.99.3:/stage/prewarm', '/stage/prewarm', 'nfs4', options)]))
        assert residency_shard_reader._read_shape('/stage/prewarm/x') is None
    _use_table(monkeypatch, _mount_table(tmp_path, [
        ('/dev/sda1', '/stage/prewarm', 'ext4', 'rw,rsize=1048576,nconnect=16')]))
    assert residency_shard_reader._read_shape('/stage/prewarm/x') is None


def test_the_longest_mount_point_wins_and_octal_is_decoded(tmp_path, monkeypatch):
    """The kernel resolves a path to its longest matching mount point."""
    _use_table(monkeypatch, _mount_table(tmp_path, [
        ('10.100.99.3:/shared', '/stage', 'nfs4', 'rw,rsize=4096,nconnect=2'),
        ('10.100.99.3:/prewarm', '/stage/pre\\040warm', 'nfs4', NFS_OPTIONS),
    ]))
    assert residency_shard_reader._read_shape('/stage/other') == (2, 4096)
    assert residency_shard_reader._read_shape('/stage/pre warm/x') == (16, 1048576)


def test_the_environment_can_put_the_read_back_on_one_stream(tmp_path, monkeypatch):
    """The before arm of an A/B, and the way out if a mount misreports."""
    table = _mount_table(tmp_path, [
        ('10.100.99.3:/stage/prewarm', '/stage/prewarm', 'nfs4', NFS_OPTIONS)])
    monkeypatch.setenv(residency_shard_reader.STREAMS_ENV, '1')
    _use_table(monkeypatch, table)
    assert residency_shard_reader._read_shape('/stage/prewarm/x') is None
    monkeypatch.setenv(residency_shard_reader.STREAMS_ENV, '4')
    residency_shard_reader.reset_read_shape_cache_for_tests()
    assert residency_shard_reader._read_shape('/stage/prewarm/x') == (4, 1048576)


def test_the_cuts_cover_the_span_exactly_and_align_to_the_chunk():
    """Disjoint, contiguous, summing to the span, cut on the mount's own grid."""
    for offset in (0, 1, 4095, 4096, 1048576, 1048577):
        for count in (1, 4095, 4096, 4097, 100000, 1 << 21):
            for chunk in (4096, 1 << 20):
                cuts = residency_shard_reader._cuts(offset, count, chunk)
                assert cuts[0][0] == offset
                assert sum(size for _, size in cuts) == count
                for (at, size), (next_at, _) in zip(cuts, cuts[1:]):
                    assert at + size == next_at
                    assert (at + size) % chunk == 0
                assert cuts[-1][0] + cuts[-1][1] == offset + count


def _big_shard(tmp_path):
    """A shard whose one tensor is many chunks wide at the sizes below."""
    pool = tmp_path / 'pool'
    pool.mkdir(parents=True, exist_ok=True)
    tensors = {'w': torch.arange(1 << 19, dtype=torch.int32).to(torch.float32).reshape(512, 1024)}
    path = pool / 'model-00001-of-00001.safetensors'
    save_file(tensors, str(path))
    return path, tensors


def test_a_span_read_on_several_streams_is_the_bytes_one_stream_read(tmp_path, monkeypatch):
    """The claim the whole change rests on: same bytes, more streams."""
    path, tensors = _big_shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    _use_table(monkeypatch, _mount_table(tmp_path, [
        ('10.100.99.3:/prewarm', str(root), 'nfs4', 'rw,rsize=4096,nconnect=4')]))
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, 0, path.stat().st_size, staged)]))
    calls = []
    real = os.preadv
    monkeypatch.setattr(os, 'preadv', lambda fd, bufs, off: (calls.append(off), real(fd, bufs, off))[1])
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        _identical(reader, path, tensors)
    assert resolver.report()['range_hits'] == 1
    # The span really was cut: one stream would have issued one read.
    assert len(calls) > 4


def test_a_failed_chunk_falls_the_whole_tensor_back_to_the_pool(tmp_path, monkeypatch, capsys):
    """A partial buffer is never handed out, and the pool's bytes are."""
    path, tensors = _big_shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    _use_table(monkeypatch, _mount_table(tmp_path, [
        ('10.100.99.3:/prewarm', str(root), 'nfs4', 'rw,rsize=4096,nconnect=4')]))
    resolver = _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, 0, path.stat().st_size, staged)]))
    real, seen = os.preadv, []

    def flaky(fd, bufs, off):
        seen.append(off)
        if len(seen) > 3:
            raise OSError(5, 'Input/output error')
        return real(fd, bufs, off)

    monkeypatch.setattr(os, 'preadv', flaky)
    with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
        _identical(reader, path, tensors)
    report = resolver.report()
    assert report['fallback_count'] == 1
    assert report['bytes_from_stage'] == 0
    assert report['bytes_from_pool'] == tensors['w'].numel() * 4


def test_one_process_wide_pool_carries_every_readers_chunks(tmp_path, monkeypatch):
    """The transports are the client's, not each reader's."""
    path, _ = _big_shard(tmp_path)
    root = _stage_root(tmp_path)
    staged = _stage_whole(root, path)
    _use_table(monkeypatch, _mount_table(tmp_path, [
        ('10.100.99.3:/prewarm', str(root), 'nfs4', 'rw,rsize=4096,nconnect=4')]))
    _bind(monkeypatch, _write_map(
        tmp_path, root, [(path, 0, path.stat().st_size, staged)]))
    threads = set()
    real = os.preadv

    def named(fd, bufs, off):
        threads.add(threading.current_thread().name)
        return real(fd, bufs, off)

    monkeypatch.setattr(os, 'preadv', named)
    for _ in range(2):
        with layer_streaming._source_safe_open(str(path), framework='pt') as reader:
            reader.get_tensor('w')
    workers = {name for name in threads if name.startswith('pq-staged-read')}
    assert 0 < len(workers) <= 4
