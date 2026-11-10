"""Retained strict refusals and explicit offline reads, without PB transport."""
from pathlib import Path

import pytest
import torch

from prismaquant import io_engine, layer_streaming, perturbed_x_cache as pxc
from prismaquant import joint_adjoint_checkpoints as checkpoints
from prismaquant.staged_manifest import load_staged_manifest
from prismaquant.staged_tier_policy import active_policy, TierPolicyRefused, staged_tier_policy_test_context
from prismaquant.staged_whole_file import read_staged_whole_file
from prismaquant.tessera_joint_aura import _read_verified_wire_blob
from staged_native_fixtures import bind, native_staged_state, publish_inputs, sha
from test_native_strict_reader_contract import _forbid_source, _shard
from test_native_strict_app_consumers import _checkpoint

pytestmark = pytest.mark.usefixtures('native_staged_state')


@pytest.mark.parametrize('tiers', ['ram', 'ssd'])
def test_native_undeclared_input_never_uses_existing_canonical_payload(tmp_path, monkeypatch, tiers):
    source = tmp_path / 'canonical'
    source.write_bytes(b'existing-but-undeclared')
    other = tmp_path / 'declared'
    other.write_bytes(b'only-admitted-input')
    resolver = bind(monkeypatch, publish_inputs(tmp_path, [other]))
    opened = _forbid_source(monkeypatch, source)
    with staged_tier_policy_test_context(tiers):
        with pytest.raises(TierPolicyRefused):
            read_staged_whole_file(source, sha(b'existing-but-undeclared'), label='undeclared')
    assert opened == [] and resolver._windows == 0
    assert resolver.report()['bytes_from_pool'] == resolver.report()['bytes_from_stage'] == 0


def test_ram_only_scope_cannot_adopt_native_ssd_material(tmp_path, monkeypatch):
    source = tmp_path / 'wire'
    raw = b'known-declared-ssd-wire'
    source.write_bytes(raw)
    resolver = bind(monkeypatch, publish_inputs(tmp_path, [source]))
    opened = _forbid_source(monkeypatch, source)
    with staged_tier_policy_test_context('ram'):
        with pytest.raises(TierPolicyRefused, match='ssd-not-allowed'):
            _read_verified_wire_blob({'wire': str(source),
                'record': {'blob_bytes': len(raw), 'blob_sha256': sha(raw)}})
    assert opened == [] and resolver._windows == 0
    assert resolver.report()['bytes_from_pool'] == resolver.report()['bytes_from_stage'] == 0


def test_equal_sized_sources_keep_their_own_bytes_and_never_adopt_a_changed_object(tmp_path, monkeypatch):
    payloads = [b'owned-A-0123456789', b'owned-B-0123456789']
    sources = [tmp_path / 'a.wire', tmp_path / 'b.wire']
    for path, raw in zip(sources, payloads):
        path.write_bytes(raw)
    binding = publish_inputs(tmp_path, sources)
    resolver = bind(monkeypatch, binding)
    opened = [_forbid_source(monkeypatch, path) for path in sources]
    for path in sources:
        path.unlink()
    def read(index):
        return _read_verified_wire_blob({'wire': str(sources[index]),
            'record': {'blob_bytes': len(payloads[index]), 'blob_sha256': sha(payloads[index])}})
    with staged_tier_policy_test_context('ssd'):
        for index, raw in enumerate(payloads):
            assert read(index) == (raw, sha(raw))
        before = resolver.report()['bytes_from_stage']
        extent = load_staged_manifest(binding['path'], binding['sha256']).extents[0]
        obj = Path(extent.object_path)
        replacement = obj.with_name('wrong-object')
        replacement.write_bytes(payloads[1])
        replacement.replace(obj)
        with pytest.raises((RuntimeError, TierPolicyRefused), match='changed|digest|checksum'):
            read(0)
        assert resolver.report()['bytes_from_stage'] == before
    assert all(not rows for rows in opened) and resolver._windows == 0
    assert resolver.report()['bytes_from_pool'] == 0


@pytest.mark.parametrize('fault', ['unmapped', 'corrupt', 'missing'])
def test_exact_group_integrity_failure_exposes_nothing_and_releases_charges(tmp_path, monkeypatch, fault):
    _, record, _, _, _, files = _checkpoint(tmp_path)
    entries = record['activation_entries']
    victim = Path(entries[2]['path'])
    binding = publish_inputs(tmp_path, [p for p in files if fault != 'unmapped' or p != victim])
    resolver = bind(monkeypatch, binding)
    opened = [_forbid_source(monkeypatch, path) for path in files]
    if fault != 'unmapped':
        extent = next(e for e in load_staged_manifest(binding['path'], binding['sha256']).extents
                      if e.source_path == str(victim))
        obj = Path(extent.object_path)
        if fault == 'missing':
            obj.unlink()
        else:
            raw = obj.read_bytes()
            obj.chmod(0o600)
            obj.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    refs = [checkpoints.reference_from_record(row) for row in entries]
    held, exposed = [0], []
    def charge(delta):
        held[0] += delta
    with staged_tier_policy_test_context('ssd'):
        with pytest.raises((TierPolicyRefused, RuntimeError, OSError)):
            with pxc.prefetch_exact_activation_cache_entries(refs, max_tensor_bytes=192,
                    expected_session=checkpoints.checkpoint_entry_session(record),
                    residency_check=charge) as window:
                exposed.append(window.get(refs[0]))
    assert exposed == [] and held == [0] and resolver._windows == 0
    assert all(not calls for calls in opened)
    assert resolver.report()['bytes_from_pool'] == 0


def test_forked_shard_cannot_use_payload_but_keeps_known_header_metadata(tmp_path, monkeypatch):
    import os
    source, tensors = _shard(tmp_path)
    resolver = bind(monkeypatch, publish_inputs(tmp_path, [source]))
    with staged_tier_policy_test_context('ssd'):
        with layer_streaming._source_safe_open(str(source), framework='pt') as reader:
            assert torch.equal(reader.get_tensor('f32'), tensors['f32'])
            with monkeypatch.context() as child:
                pid = os.getpid()
                child.setattr(os, 'getpid', lambda: pid + 100000)
                assert set(reader.keys()) == set(tensors)
                with pytest.raises(RuntimeError, match='forked child'):
                    reader.get_tensor('bf16')
                with pytest.raises(RuntimeError, match='forked child'):
                    reader.get_slice('f32')
        assert resolver._windows == 0


@pytest.mark.parametrize('bound', [False, True])
def test_explicit_offline_unstaged_read_preserves_known_bytes_and_pool_accounting(tmp_path, monkeypatch, bound):
    payload = b'explicit-offline-known-input'
    source = tmp_path / 'offline'
    source.write_bytes(payload)
    resolver = None
    if bound:
        declared = tmp_path / 'declared'
        declared.write_bytes(b'a-different-input')
        resolver = bind(monkeypatch, publish_inputs(tmp_path, [declared]))
    assert active_policy() is None
    value, observed = io_engine.load_file(source, len(payload), binding=None,
        decode=lambda raw, receipt, staged: (bytes(raw), None))
    assert value == payload and observed[0]['sha256'] == sha(payload)
    if resolver is not None:
        assert resolver.report()['bytes_from_pool'] == len(payload)
        assert resolver.report()['bytes_from_stage'] == 0 and resolver._windows == 0


def test_immutable_reader_has_no_configured_wait():
    from prismaquant import residency_shard_reader as reader
    assert not hasattr(reader, 'STAGED_RANGE_WAIT_S')
    assert not hasattr(reader, 'staged_range_wait_s')
