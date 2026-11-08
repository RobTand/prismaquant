"""Immutable readiness replaces PB mover rediscovery and sidecar polling.

Complete-before-consume inputs serve their known header/body bytes. Missing or
changed inputs refuse. Real warm/afterok exclusion is proved separately in the
native dependency proof, not simulated by a mutable mover here.
"""
import os
from pathlib import Path

import pytest

from prismaquant.residency_map import StagedReadRefused
from prismaquant.residency_shard_reader import await_staged_spans
from prismaquant.staged_lease import acquire_entry_window, stage_covers_are_published
from prismaquant.staged_manifest import load_staged_manifest
from prismaquant.staged_tier_policy import staged_tier_policy_test_context
from staged_native_fixtures import bind, native_staged_state, publish_inputs

HEADER = b'known-header\x00\xff' * 64
BODY = b'known-body\x01\xfe' * 64


@pytest.fixture(autouse=True)
def _policy(native_staged_state):
    with staged_tier_policy_test_context('ssd'):
        yield


def _inputs(tmp_path, monkeypatch):
    paths = [tmp_path / 'header.bin', tmp_path / 'body.bin']
    for path, raw in zip(paths, [HEADER, BODY]):
        path.write_bytes(raw)
    binding = publish_inputs(tmp_path, paths)
    manifest = load_staged_manifest(binding['path'], binding['sha256'])
    resolver = bind(monkeypatch, binding)
    for path in paths:
        path.unlink()
    return resolver, paths, manifest


def _read(resolver, source, size):
    entry = resolver.staged_range(source, 0, size)
    assert entry is not None
    window, key = acquire_entry_window(resolver, source, entry)
    with window:
        fd, _serving = window.open(key)
        try:
            got = os.read(fd, size)
            window.check_fd(fd)
            return got
        finally:
            window.close_fd(fd)


def test_published_inputs_serve_both_known_extents(tmp_path, monkeypatch):
    resolver, paths, _manifest = _inputs(tmp_path, monkeypatch)
    assert _read(resolver, paths[0], len(HEADER)) == HEADER
    assert _read(resolver, paths[1], len(BODY)) == BODY
    items = [(p, resolver.staged_range(p, 0, size))
             for p, size in zip(paths, [len(HEADER), len(BODY)])]
    assert stage_covers_are_published(resolver, items)
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0


@pytest.mark.parametrize('fault', ['missing', 'changed'])
def test_a_ready_declaration_cannot_make_bad_object_bytes_readable(tmp_path, monkeypatch, fault):
    resolver, paths, manifest = _inputs(tmp_path, monkeypatch)
    assert _read(resolver, paths[0], len(HEADER)) == HEADER
    assert _read(resolver, paths[1], len(BODY)) == BODY
    body = Path(next(e.object_path for e in manifest.extents if e.source_path == str(paths[1])))
    if fault == 'missing':
        body.unlink()
    else:
        body.chmod(0o600)
        body.write_bytes(b'!' * len(BODY))
    refusal = FileNotFoundError if fault == 'missing' else StagedReadRefused
    with pytest.raises(refusal):
        _read(resolver, paths[1], len(BODY))
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0


def test_native_readiness_checks_exact_declared_spans(tmp_path, monkeypatch):
    resolver, paths, _manifest = _inputs(tmp_path, monkeypatch)
    from prismaquant.residency_map import RANGE_HIT, RANGE_UNDECLARED
    spans = [(path, 0, size, size) for path, size in zip(paths, [len(HEADER), len(BODY)])]
    assert await_staged_spans(resolver, spans, deadline=None) == RANGE_HIT
    assert await_staged_spans(resolver, [*spans, (tmp_path / 'undeclared', 0, 1, 1)], deadline=None) == RANGE_UNDECLARED
    assert _read(resolver, paths[1], len(BODY)) == BODY
    assert resolver.report()['bytes_from_pool'] == 0 and resolver._windows == 0
