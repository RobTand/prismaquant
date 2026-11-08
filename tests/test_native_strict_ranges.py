"""Native whole-input offsets, bounded metadata and exact content identity."""
from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch

from prismaquant import perturbed_x_cache as pxc
from prismaquant.staged_lease import load_sealed_manifest
from prismaquant.staged_manifest import MAX_BYTES, ManifestRefused, load_staged_manifest
from prismaquant.staged_tier_policy import staged_tier_policy_test_context
from staged_native_fixtures import bind, fixture, native_staged_state, publish_inputs, sha
from test_native_strict_reader_contract import _forbid_source

pytestmark = pytest.mark.usefixtures('native_staged_state')


@pytest.mark.parametrize('consumer', ['wire', 'activation', 'exact'])
def test_whole_consumers_read_only_the_declared_nonzero_object_extent(tmp_path, monkeypatch, consumer):
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4) - 4
    original = tmp_path / 'original.pt'
    session = {'generation': 'range-owned', 'kind': 'fixture'}
    if consumer == 'exact':
        ref = pxc.write_exact_activation_cache_entry(tmp_path / 'original', 'known', expected,
            identity={'session': session, 'slot': 'known'}, max_tensor_bytes=48,
            max_file_bytes=1 << 20)
        raw = Path(ref.path).read_bytes()
    elif consumer == 'activation':
        torch.save({'inputs': expected}, original)
        raw = original.read_bytes()
    else:
        raw = b'independent-wire\x00\xff-nonzero-offset'
    source, _, binding, _ = fixture(tmp_path, payload=raw)
    source.write_bytes(raw)
    resolver = bind(monkeypatch, binding)
    opened = _forbid_source(monkeypatch, source)
    extent = load_staged_manifest(binding['path'], binding['sha256']).extents[0]
    assert extent.object_offset == 13 and extent.object_bytes > len(raw)
    with staged_tier_policy_test_context('ssd'):
        if consumer == 'wire':
            from prismaquant.tessera_joint_aura import _read_verified_wire_blob
            got, digest = _read_verified_wire_blob({'wire': str(source),
                'record': {'blob_bytes': len(raw), 'blob_sha256': sha(raw)}})
            assert got == raw and digest == sha(raw)
        elif consumer == 'activation':
            got, receipt = pxc.load_verified_activation_cache_entry(source,
                expected_sha256=sha(raw), policy={'max_buffer_bytes': 8 << 20,
                'max_scratch_bytes': 8 << 20}, max_storage_bytes=1 << 20)
            assert torch.equal(got['inputs'].view(torch.uint8), expected.view(torch.uint8))
            assert receipt['source_read_bytes'] == len(raw)
        else:
            ref = replace(ref, path=str(source))
            with pxc.prefetch_exact_activation_cache_entries([ref], max_tensor_bytes=48,
                    expected_session=session) as window:
                assert torch.equal(window.get(ref).view(torch.uint8), expected.view(torch.uint8))
    assert opened == [] and resolver._windows == 0
    assert resolver.report()['bytes_from_pool'] == 0
    assert resolver.report()['bytes_from_stage'] == len(raw)


def test_read_contract_size_refuses_before_any_payload_read(tmp_path, monkeypatch):
    source = tmp_path / 'input'
    source.write_bytes(b'known')
    binding = publish_inputs(tmp_path, [source])
    bind(monkeypatch, binding)
    contract = Path(binding['read_contract_path'])
    with contract.open('r+b') as handle:
        handle.truncate(MAX_BYTES + 1)
    signature = contract.stat()
    real_pread, read = os.pread, []
    def bounded(fd, size, offset):
        actual = os.fstat(fd)
        if (actual.st_dev, actual.st_ino) == (signature.st_dev, signature.st_ino):
            read.append((size, offset))
            raise AssertionError('oversized contract payload was read')
        return real_pread(fd, size, offset)
    monkeypatch.setattr(os, 'pread', bounded)
    with pytest.raises(ManifestRefused, match='exceeds bound'):
        load_sealed_manifest(binding['read_contract_sha256'])
    assert read == []


def test_read_contract_changed_digest_refuses_before_decode(tmp_path, monkeypatch):
    source = tmp_path / 'input'
    source.write_bytes(b'known')
    binding = publish_inputs(tmp_path, [source])
    bind(monkeypatch, binding)
    contract = Path(binding['read_contract_path'])
    contract.write_bytes(b'invalid and changed JSON')
    from prismaquant.staged_lease import ReadsetUnbound
    with pytest.raises(ReadsetUnbound, match='digest mismatch'):
        load_sealed_manifest(binding['read_contract_sha256'])
