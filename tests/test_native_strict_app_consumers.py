"""Retained PWC, activation, wire and checkpoint obligations without PB fixtures."""
from contextlib import closing
import os
from pathlib import Path

import pytest
import torch

from prismaquant import io_engine, joint_adjoint_checkpoints as checkpoints
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.staged_lease import LeaseWindow
from prismaquant.staged_manifest import load_staged_manifest
from prismaquant.staged_tier_policy import staged_tier_policy_test_context, TierPolicyRefused
from staged_native_fixtures import bind, native_staged_state, publish_inputs, sha
from test_native_strict_reader_contract import _forbid_source


pytestmark = pytest.mark.usefixtures('native_staged_state')


def _pwc(tmp_path):
    key = ('unit0', 'TESSERA_E4M3_K1_R1024')
    expected = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    source = tmp_path / 'render.pt'
    torch.save(expected, source)
    raw = source.read_bytes()
    cache = ProductionWeightCache(weights={key: str(source)}, levers={})
    cache.enable_lru(1 << 20)
    return source, raw, cache, key, expected


def test_pwc_reads_digest_bound_known_tensor_and_retains_owned_mapping(tmp_path, monkeypatch):
    source, raw, cache, key, expected = _pwc(tmp_path)
    cache.require_file_load_sha256({key: sha(raw)}, max_file_bytes=len(raw))
    resolver = bind(monkeypatch, publish_inputs(tmp_path, [source]))
    opened = _forbid_source(monkeypatch, source)
    with staged_tier_policy_test_context('ssd'):
        assert cache.prefetch([key], max_workers=1) == 1
        got = cache.get(*key)
        assert got.dtype == expected.dtype and got.shape == expected.shape
        assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
        receipt = cache.file_load_receipt(key, got)
        assert receipt['sha256'] == sha(raw) and receipt['serving_tier'] == 'stage'
        assert resolver._windows == 0
        source.unlink()
        # The decoded tensor's own mapping survives the staged window.
        resolver.close()
        assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
        assert cache.file_load_receipt(key, got) == receipt
        got[0, 0] += 1
        with pytest.raises(RuntimeError, match='tensor or source file changed'):
            cache.file_load_receipt(key, got)
    assert opened == [] and resolver.report()['bytes_from_pool'] == 0
    assert resolver.report()['bytes_from_stage'] == len(raw)


@pytest.mark.parametrize('fault', ['missing-digest', 'unbounded', 'wrong-digest'])
def test_pwc_requires_explicit_digest_and_bound_before_read(tmp_path, monkeypatch, fault):
    source, raw, cache, key, _ = _pwc(tmp_path)
    if fault == 'missing-digest':
        cache.enable_file_load_receipts(max_file_bytes=len(raw))
    elif fault == 'wrong-digest':
        cache.require_file_load_sha256({key: '0' * 64}, max_file_bytes=len(raw))
    resolver = bind(monkeypatch, publish_inputs(tmp_path, [source]))
    source.unlink()
    opened = _forbid_source(monkeypatch, source)
    with staged_tier_policy_test_context('ssd'):
        with pytest.raises((TierPolicyRefused, RuntimeError, ValueError)):
            cache.prefetch([key], max_workers=1)
    assert opened == [] and resolver._windows == 0
    assert resolver.report()['bytes_from_pool'] == 0


@pytest.mark.parametrize('error_type', [MemoryError, OSError])
def test_pwc_allocation_failure_releases_actual_descriptor_window(tmp_path, monkeypatch, error_type):
    source, raw, cache, key, _ = _pwc(tmp_path)
    cache.require_file_load_sha256({key: sha(raw)}, max_file_bytes=len(raw))
    resolver = bind(monkeypatch, publish_inputs(tmp_path, [source]))
    descriptors = []
    original = LeaseWindow.open
    def opened(window, key):
        fd, serving = original(window, key)
        descriptors.append(fd)
        return fd, serving
    def fail_allocation(buffer, size):
        assert resolver._windows == 1
        assert os.fstat(descriptors[-1]).st_size == len(raw)
        raise error_type('owned allocation refused')
    monkeypatch.setattr(LeaseWindow, 'open', opened)
    monkeypatch.setattr(io_engine.SealedBuffer, '__init__', fail_allocation)
    with staged_tier_policy_test_context('ssd'):
        with pytest.raises(error_type, match='owned allocation refused'):
            cache.prefetch([key], max_workers=1)
    assert len(descriptors) == 1 and resolver._windows == 0
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    assert resolver.report()['bytes_from_pool'] == 0


@pytest.mark.parametrize('fault', ['none', 'corrupt', 'unmapped'])
def test_wire_known_bytes_and_digest_or_refusal_without_source_read(tmp_path, monkeypatch, fault):
    from prismaquant.tessera_joint_aura import _read_verified_wire_blob
    raw = b'independent-wire-content\x00\xff'
    source = tmp_path / 'cell.wire'
    source.write_bytes(raw)
    other = tmp_path / 'other.wire'
    other.write_bytes(b'unrelated-declared-input')
    binding = publish_inputs(tmp_path, [other if fault == 'unmapped' else source])
    resolver = bind(monkeypatch, binding)
    if fault == 'corrupt':
        obj = Path(load_staged_manifest(binding['path'], binding['sha256']).extents[0].object_path)
        obj.chmod(0o600)
        obj.write_bytes(b'X' + raw[1:])
    source.unlink()
    opened = _forbid_source(monkeypatch, source)
    cell = {'wire': str(source), 'record': {'blob_bytes': len(raw), 'blob_sha256': sha(raw)}}
    with staged_tier_policy_test_context('ssd'):
        if fault == 'none':
            got, digest = _read_verified_wire_blob(cell)
            assert got == raw and digest == sha(raw)
            assert resolver.report()['bytes_from_stage'] == len(raw)
        else:
            with pytest.raises((TierPolicyRefused, RuntimeError, ValueError)):
                _read_verified_wire_blob(cell)
            assert resolver.report()['bytes_from_stage'] == 0
    assert opened == [] and resolver._windows == 0
    assert resolver.report()['bytes_from_pool'] == 0


@pytest.mark.parametrize('fault', ['none', 'unmapped', 'corrupt'])
def test_verified_activation_retains_tensor_bytes_and_read_accounting(tmp_path, monkeypatch, fault):
    from prismaquant.perturbed_x_cache import load_verified_activation_cache_entry
    source = tmp_path / 'capture.pt'
    expected = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torch.save({'inputs': expected}, source)
    raw = source.read_bytes()
    other = tmp_path / 'other'
    other.write_bytes(b'other-input')
    binding = publish_inputs(tmp_path, [other if fault == 'unmapped' else source])
    resolver = bind(monkeypatch, binding)
    if fault == 'corrupt':
        obj = Path(load_staged_manifest(binding['path'], binding['sha256']).extents[0].object_path)
        obj.chmod(0o600)
        obj.write_bytes(b'X' + raw[1:])
    opened = _forbid_source(monkeypatch, source)
    policy = {'schema': 'prismaquant.verified_activation_load.v1',
              'max_buffer_bytes': 8 << 20, 'max_scratch_bytes': 8 << 20}
    with staged_tier_policy_test_context('ssd'):
        if fault == 'none':
            got, _ = load_verified_activation_cache_entry(source, expected_sha256=sha(raw),
                policy=policy, max_storage_bytes=4 << 20)
            assert torch.equal(got['inputs'].view(torch.uint8), expected.view(torch.uint8))
            assert resolver.report()['bytes_from_stage'] == len(raw)
        else:
            with pytest.raises((TierPolicyRefused, RuntimeError, ValueError)):
                load_verified_activation_cache_entry(source, expected_sha256=sha(raw),
                    policy=policy, max_storage_bytes=4 << 20)
            assert resolver.report()['bytes_from_stage'] == 0
    assert opened == [] and resolver._windows == 0
    assert resolver.report()['bytes_from_pool'] == 0


def _checkpoint(tmp_path):
    tensors = {(probe, batch): torch.arange(12, dtype=torch.float32).reshape(3, 4)
               + 100 * probe + 10 * batch for probe in range(2) for batch in range(2)}
    state = {'enabled': True, 'accumulators': [{'slot': ['fixture-kv', 1, 0],
              'tensor': torch.tensor([3.0, -2.0], dtype=torch.float32)}],
             'counters': {'n_grafted': 3, 'n_harvested': 2, 'n_seeded': 1, 'n_no_grad': 0},
             'nondifferentiable': []}
    shared_pass = {batch: {'captured': None, 'fixture_batch': batch} for batch in range(2)}
    space = checkpoints.adjoint_space(tmp_path)
    record = checkpoints.write_adjoint_checkpoint(space, boundary=5,
        session={'generation': 'g' * 32, 'kind': 'adjoint_checkpoint'}, cotangents=tensors,
        shared_adjoint={key: state for key in tensors}, shared_pass=shared_pass)
    entries = record['activation_entries'] + record['shared_state_entries']
    manifest = Path(record['activation_entries'][0]['path']).parent.parent / 'checkpoint.json'
    return space, record, tensors, state, shared_pass, [manifest, *(Path(row['path']) for row in entries)]


@pytest.mark.parametrize('local_scratch', [False, True])
def test_checkpoint_tensor_and_shared_state_roundtrip_without_pool(tmp_path, monkeypatch, local_scratch):
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    space, record, expected, state, passes, files = _checkpoint(tmp_path)
    binding = publish_inputs(tmp_path, files)
    resolver = bind(monkeypatch, binding)
    opened = [_forbid_source(monkeypatch, filename) for filename in files]
    arena = None
    def factory(entries):
        nonlocal arena
        arena = ExactCotangentScratch(entries, directory=tmp_path, max_bytes=1 << 20)
        return arena
    held, peak = [0], [0]
    def charge(delta):
        held[0] += delta
        peak[0] = max(peak[0], held[0])
    try:
        with staged_tier_policy_test_context('ssd'):
            got, shared, got_pass = checkpoints.load_adjoint_checkpoint(space, record,
                cotangent_factory=factory if local_scratch else None,
                max_resident_bytes=192, residency_check=charge)
            assert set(got) == set(expected) == set(shared)
            for key, want in expected.items():
                assert torch.equal(got[key].view(torch.uint8), want.view(torch.uint8))
                assert shared[key]['enabled'] is True
                assert shared[key]['counters'] == state['counters']
                assert shared[key]['nondifferentiable'] == []
                assert shared[key]['accumulators'][0]['slot'] == ['fixture-kv', 1, 0]
                assert torch.equal(shared[key]['accumulators'][0]['tensor'], torch.tensor([3.0, -2.0]))
            assert got_pass == passes
    finally:
        if arena is not None:
            arena.close()
    assert held == [0] and 0 < peak[0] <= 192
    assert resolver._windows == 0 and resolver.report()['bytes_from_pool'] == 0
    assert all(not calls for calls in opened)


@pytest.mark.parametrize('fault', ['manifest-unmapped', 'state-unmapped', 'state-corrupt'])
def test_checkpoint_missing_or_corrupt_scientific_state_refuses(tmp_path, monkeypatch, fault):
    space, record, _, _, _, files = _checkpoint(tmp_path)
    state_path = Path(record['shared_state_entries'][0]['path'])
    selected = [p for p in files if not (
        fault == 'manifest-unmapped' and p == files[0] or fault == 'state-unmapped' and p == state_path)]
    binding = publish_inputs(tmp_path, selected)
    resolver = bind(monkeypatch, binding)
    if fault == 'state-corrupt':
        extent = next(e for e in load_staged_manifest(binding['path'], binding['sha256']).extents
                      if e.source_path == str(state_path))
        obj = Path(extent.object_path)
        raw = obj.read_bytes()
        obj.chmod(0o600)
        obj.write_bytes(b'X' + raw[1:])
    opened = [_forbid_source(monkeypatch, filename) for filename in files]
    with staged_tier_policy_test_context('ssd'):
        with pytest.raises((TierPolicyRefused, RuntimeError, ValueError)):
            checkpoints.load_adjoint_checkpoint(space, record, max_resident_bytes=192)
    assert resolver._windows == 0 and resolver.report()['bytes_from_pool'] == 0
    assert all(not calls for calls in opened)


def test_checkpoint_stream_close_releases_charges_before_owned_inputs_close(tmp_path, monkeypatch):
    space, record, expected, _, _, files = _checkpoint(tmp_path)
    resolver = bind(monkeypatch, publish_inputs(tmp_path, files))
    held, peak = [0], [0]
    def charge(delta):
        held[0] += delta
        peak[0] = max(peak[0], held[0])
    session = checkpoints.checkpoint_entry_session(record)
    with staged_tier_policy_test_context('ssd'):
        with closing(checkpoints.stream_exact_entry_tensors(record['activation_entries'],
                expected_session=session, max_resident_bytes=192, residency_check=charge)) as stream:
            entry, got = next(stream)
            coordinates = entry['metadata']['identity']['coordinates']
            coordinate = (coordinates['probe'], coordinates['batch'])
            assert torch.equal(got.view(torch.uint8), expected[coordinate].view(torch.uint8))
            assert 0 < held[0] <= 192
        assert held == [0] and 0 < peak[0] <= 192
        assert resolver._windows == 0
        resolver.close()
