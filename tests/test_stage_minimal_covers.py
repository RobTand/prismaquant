"""A native entry depends only on its declared immutable object.

PB lead fragments and mover material sidecars are retired. These tests retain
known current bytes, unrelated-input independence and fail-closed coverage.
"""
import os
from pathlib import Path

import pytest

from prismaquant.residency_map import StagedReadRefused
from prismaquant.staged_lease import acquire_entry_window
from prismaquant.staged_manifest import load_staged_manifest, publish_staged_json, ManifestRefused
from prismaquant.staged_tier_policy import staged_tier_policy_test_context
from staged_native_fixtures import bind, native_staged_state, publish_inputs


@pytest.fixture(autouse=True)
def _policy(native_staged_state):
    with staged_tier_policy_test_context('ssd'):
        yield


def _two_entries(tmp_path, monkeypatch, *, retire_head=False):
    payloads = [b'\x01head-bytes\x00' * 64, b'\x02current-bytes' * 64]
    sources = [tmp_path / 'head.bin', tmp_path / 'current.bin']
    for source, raw in zip(sources, payloads):
        source.write_bytes(raw)
    binding = publish_inputs(tmp_path, sources)
    manifest = load_staged_manifest(binding['path'], binding['sha256'])
    resolver = bind(monkeypatch, binding)
    # Fault injection only, not a supported per-object phase retirement API.
    if retire_head:
        Path(manifest.extents[0].object_path).unlink()
    for source in sources:
        source.unlink()
    return resolver, sources, manifest, payloads, binding


def _read(resolver, source, length):
    entry = resolver.staged_range(source, 0, length)
    assert entry is not None
    window, key = acquire_entry_window(resolver, source, entry)
    with window:
        fd, _serving = window.open(key)
        try:
            raw = os.read(fd, length)
            window.check_fd(fd)
            return raw
        finally:
            window.close_fd(fd)


def test_a_retired_unrelated_lead_does_not_poison_a_current_entry(tmp_path, monkeypatch):
    resolver, sources, _manifest, payloads, _binding = _two_entries(
        tmp_path, monkeypatch, retire_head=True)
    assert _read(resolver, sources[1], len(payloads[1])) == b'\x02current-bytes' * 64
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0


def test_a_current_entry_also_acquires_while_the_head_is_live(tmp_path, monkeypatch):
    resolver, sources, _manifest, payloads, _binding = _two_entries(tmp_path, monkeypatch)
    for source, raw in zip(sources, payloads):
        assert _read(resolver, source, len(raw)) == raw
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0


def test_missing_requested_coverage_still_refuses_without_pool_bytes(tmp_path, monkeypatch):
    resolver, sources, manifest, payloads, _binding = _two_entries(tmp_path, monkeypatch)
    Path(manifest.extents[1].object_path).unlink()
    with pytest.raises(StagedReadRefused):
        _read(resolver, sources[1], len(payloads[1]))
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0


def test_contradictory_actual_proof_refuses_as_integrity(tmp_path, monkeypatch):
    resolver, sources, _manifest, payloads, binding = _two_entries(tmp_path, monkeypatch)
    import json
    body = json.loads(Path(binding['path']).read_bytes())
    conflicting = dict(body['entries'][1], sha256='0' * 64)
    body['entries'].append(conflicting)
    changed = tmp_path / 'contradictory.json'
    digest = publish_staged_json(changed, body)
    with pytest.raises(ManifestRefused):
        load_staged_manifest(changed, digest)
    # A refused alternative cannot replace the already authenticated input.
    assert _read(resolver, sources[1], len(payloads[1])) == payloads[1]
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0
