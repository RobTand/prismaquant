"""Migrated shard, whole-entry, metadata and IO lifetime regression oracles."""
import copy
import json
import os
from pathlib import Path
import sys
import threading

import pytest
import torch
from safetensors.torch import save

from prismaquant import io_engine, residency_map, residency_shard_reader, stage_b_prep_io
from prismaquant.staged_manifest import ManifestRefused, load_manifest, publish_json, materialize
from prismaquant.staged_tier_policy import staged_tier_policy_test_context, TierPolicyRefused
from prismaquant.staged_whole_file import read_staged_entry, read_staged_whole_file
from staged_native_fixtures import fixture, sha, bind


@pytest.fixture(autouse=True)
def reset():
    residency_map.reset_residency_resolver_for_tests()
    stage_b_prep_io.reset_staged_reads_for_tests()
    yield
    residency_map.reset_residency_resolver_for_tests()
    stage_b_prep_io.reset_staged_reads_for_tests()
    assert not any(n == 'prismabuild' or n.startswith('prismabuild.') for n in sys.modules)


def forbidden(*args, **kwargs):
    raise AssertionError('canonical opener called')


def test_owned_whole_io_and_metadata_without_canonical_file(tmp_path, monkeypatch):
    raw = b'independently-known\x00\xff\n'
    source, data, binding, _ = fixture(tmp_path, raw)
    resolver = bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        assert read_staged_whole_file(source, sha(raw), label='fixture') == raw
        got, receipt, _ = io_engine.read_file(source, len(raw), staged=resolver.staged_read(source))
        assert got == raw and receipt['sha256'] == sha(raw)
        value, _ = io_engine.load_file(source, len(raw), binding=sha(raw),
                                      decode=lambda raw, receipt, staged: (raw, None))
        assert value == raw
        reads = stage_b_prep_io.bind_staged_reads(manifest_sha256=binding['read_contract_sha256'], allowed_tiers='ssd')
        assert reads.whole(source, where='metadata') == raw
        sealed = read_staged_entry(resolver, source, resolver.staged_read(source), label='sealed', sealed=True)
        try:
            sealed.require_sealed()
            assert bytes(sealed) == raw
        finally:
            sealed.close()
    assert not source.exists()
    assert resolver.report()['bytes_from_pool'] == 0


def test_shard_dtypes_metadata_slices_owned_buffers_and_nonzero_offsets(tmp_path, monkeypatch):
    tensors = {'f32': torch.arange(8192, dtype=torch.float32).reshape(64, 128),
               'bf16': torch.arange(128).to(torch.bfloat16),
               'f16': torch.arange(32).to(torch.float16),
               'i8': torch.arange(-32, 32, dtype=torch.int8),
               'u8': torch.arange(64, dtype=torch.uint8),
               'flag': torch.tensor([True, False, True]), 'empty': torch.zeros((0, 4))}
    raw = save(tensors, metadata={'oracle': 'known fixture'})
    base = 8 + int.from_bytes(raw[:8], 'little')
    source, data, binding, _ = fixture(tmp_path, raw, ranges=[(0, base), (base, len(raw))])
    resolver = bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        with residency_shard_reader.staged_shard_opener(source, forbidden)(str(source), framework='pt') as reader:
            assert set(reader.keys()) == set(tensors)
            assert reader.metadata() == {'oracle': 'known fixture'}
            retained = []
            for name, expected in tensors.items():
                got = reader.get_tensor(name)
                assert got.dtype == expected.dtype and got.shape == expected.shape
                assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
                retained.append(got)
            assert reader.get_slice('f32').get_shape() == [64, 128]
            assert torch.equal(reader.get_slice('f32')[:2], tensors['f32'][:2])
        for got, expected in zip(retained, tensors.values()):
            assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
    assert resolver.report()['bytes_from_pool'] == 0


@pytest.mark.parametrize('fault', ['missing', 'symlink', 'parent-symlink', 'length', 'digest', 'change'])
def test_whole_entry_refuses_without_canonical_read(tmp_path, monkeypatch, fault):
    source, data, binding, _ = fixture(tmp_path)
    obj = Path(data['entries'][0]['object']['path'])
    resolver = bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        if fault == 'change':
            assert read_staged_whole_file(source, data['entries'][0]['source']['sha256'], label='before') == b'known-fixture\x00\xff'
        if fault == 'missing':
            obj.unlink()
        elif fault == 'symlink':
            dest = obj.with_name('replacement'); obj.rename(dest); obj.symlink_to(dest)
        elif fault == 'parent-symlink':
            parent = obj.parent; dest = parent.with_name('replacement'); parent.rename(dest); parent.symlink_to(dest, target_is_directory=True)
        elif fault == 'length':
            obj.write_bytes(b'short')
        else:
            blob = bytearray(obj.read_bytes()); blob[14] ^= 1; obj.write_bytes(blob)
        with pytest.raises((OSError, ValueError, RuntimeError, TierPolicyRefused)):
            read_staged_whole_file(source, data['entries'][0]['source']['sha256'], label=fault)
    assert not source.exists()


@pytest.mark.parametrize('fault', ['manifest-digest', 'null-digest', 'overlap', 'object-bounds', 'source-bounds', 'zero', 'source-conflict'])
def test_manifest_refusal(tmp_path, fault):
    source, data, binding, _ = fixture(tmp_path)
    if fault == 'manifest-digest':
        with pytest.raises(ManifestRefused, match='digest mismatch'):
            load_manifest(binding['path'], '0' * 64)
        return
    row = data['entries'][0]
    if fault == 'null-digest': row['object']['sha256'] = None
    elif fault == 'overlap': data['entries'].append(copy.deepcopy(row))
    elif fault == 'object-bounds': row['object_offset'] = row['object']['bytes']
    elif fault == 'source-bounds': row['source_offset'] = 1
    elif fault == 'zero': row['length'] = 0
    else:
        second = copy.deepcopy(row); second['source']['sha256'] = '0' * 64; data['entries'].append(second)
    target = tmp_path / 'bad.json'
    digest = publish_json(target, data)
    with pytest.raises(ManifestRefused):
        load_manifest(target, digest)


def test_header_and_payload_gaps_refuse(tmp_path, monkeypatch):
    raw = save({'weight': torch.arange(16, dtype=torch.float32)})
    base = 8 + int.from_bytes(raw[:8], 'little')
    source, data, binding, _ = fixture(tmp_path, raw, ranges=[(base, len(raw))])
    bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        with residency_shard_reader.staged_shard_opener(source, forbidden)(str(source), framework='pt') as reader:
            with pytest.raises(TierPolicyRefused): reader.keys()
    residency_map.reset_residency_resolver_for_tests()
    other = tmp_path / 'other'; other.mkdir()
    source, _, binding, _ = fixture(other, raw, ranges=[(0, base), (base + 4, len(raw))])
    bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        with residency_shard_reader.staged_shard_opener(source, forbidden)(str(source), framework='pt') as reader:
            assert reader.keys() == ['weight']
            with pytest.raises(TierPolicyRefused): reader.get_tensor('weight')


def test_materializer_publishes_actual_hashes_and_no_partial_manifest(tmp_path):
    source = tmp_path / 'source'; source.write_bytes(b'prefix-KNOWN-extent-tail')
    dest = tmp_path / 'phase'; dest.mkdir()
    out = tmp_path / 'manifest.json'
    src = {'path': str(source), 'bytes': source.stat().st_size, 'sha256': sha(source.read_bytes())}
    kwargs = dict(destination=dest, manifest_path=out, read_contract_sha256='3' * 64, phase_id='offline',
        producer={'source_commit': '1' * 40, 'receipt_sha256': '2' * 64},
        lifetime={'mode': 'retained_shared_phase', 'namespace': str(dest), 'reader_scopes': ['owned'],
                  'retention': 'until_all_readers_stopped_or_fenced'})
    digest = materialize([{'source': src, 'source_offset': 7, 'length': 12}], **kwargs)
    manifest = load_manifest(out, digest)
    obj = manifest.extents[0]
    assert Path(obj.object_path).read_bytes() == b'KNOWN-extent'
    assert obj.object_sha256 == sha(b'KNOWN-extent')
    assert obj.source_sha256 == sha(b'prefix-KNOWN-extent-tail')
    assert not list(dest.glob('.extent-*'))
    bad = tmp_path / 'bad.json'; kwargs['manifest_path'] = bad
    with pytest.raises(ManifestRefused, match='digest mismatch'):
        materialize([{'source': src, 'source_offset': 0, 'length': 6, 'sha256': '0' * 64}], **kwargs)
    assert not bad.exists() and not list(dest.glob('.extent-*'))


def test_window_fork_refusal_and_closed_descriptors(tmp_path, monkeypatch):
    from prismaquant.staged_lease import acquire_entry_window
    source, _, binding, _ = fixture(tmp_path)
    resolver = bind(monkeypatch, binding)
    with staged_tier_policy_test_context('ssd'):
        window, key = acquire_entry_window(resolver, source, resolver.staged_read(source))
        with window:
            fd, _ = window.open(key)
            pid = os.fork()
            if pid == 0:
                try:
                    window.open(key)
                except RuntimeError:
                    os._exit(0)
                os._exit(9)
            assert os.waitpid(pid, 0)[1] == 0
            assert os.pread(fd, 5, 13) == b'known'
        with pytest.raises(OSError): os.fstat(fd)
    assert not resolver._closed
