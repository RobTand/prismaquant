"""Every covering map entry is asked before a span is refused (PQ #902).

PrismaBuild allows two manifest entries to overlap and stages each as a file
of its own, and its composed map can name an entry whose staged file an
eviction already unlinked. The resolver used to take the lowest-offset covering
entry and stop, so a stale neighbour entry hid the layer's own healthy range
and the strict tier policy ended the run on bytes that were on the stage.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from prismaquant.residency_map import (
    ENV_VAR, RANGE_HIT, RANGE_REFUSED, SCHEMA, bind_residency_manifest,
    residency_map_key, residency_resolver, reset_residency_resolver_for_tests,
)

MANIFEST = 'c' * 64
HEADER_BYTES = 4096
OWN = (1000, 600)      # the layer's own entry, inside the header entry
SPAN = (1100, 1300)    # a tensor both entries cover


@pytest.fixture(autouse=True)
def _forget_resolver(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    yield
    reset_residency_resolver_for_tests()


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A shard with a neighbour's header entry and the layer's own entry."""
    shard = tmp_path / 'pool' / 'model-00001.safetensors'
    shard.parent.mkdir()
    shard.write_bytes(bytes(range(256)) * 64)
    root = tmp_path / 'stage'
    files, entries = {}, {}
    for name, (offset, size) in (('header', (0, HEADER_BYTES)), ('own', OWN)):
        target = root / f'{shard.name}.pbrange' / f'{offset}-{size}'
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = shard.read_bytes()[offset:offset + size]
        target.write_bytes(blob)
        files[name] = target
        entries[residency_map_key(str(shard), offset)] = {
            'stage_path': str(target), 'bytes': size, 'offset': offset,
            'sha256': hashlib.sha256(blob).hexdigest()}
    map_path = tmp_path / 'residency.json'
    map_path.write_text(json.dumps({
        'schema': SCHEMA, 'tier_id': 'prismabuild-stage:fixture',
        'stage_root': str(root), 'manifest_sha256': MANIFEST,
        'leads': ['d' * 64], 'generation': 1, 'entries': entries}))
    monkeypatch.setenv(ENV_VAR, str(map_path))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    return {'shard': shard, 'files': files, 'resolver': residency_resolver()}


def test_the_lowest_offset_entry_still_serves_when_it_is_healthy(staged):
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], *SPAN)
    assert outcome == RANGE_HIT
    assert entry['offset'] == 0
    assert staged['resolver'].report()['range_rows_passed_over'] == 0


def test_an_entry_whose_staged_file_is_gone_does_not_hide_the_one_behind_it(
        staged, capsys):
    staged['files']['header'].unlink()
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], *SPAN)
    assert outcome == RANGE_HIT
    assert (entry['offset'], entry['bytes']) == OWN
    assert entry['stage_path'] == str(staged['files']['own'])
    report = staged['resolver'].report()
    assert report['fallback_count'] == 0
    assert report['range_rows_passed_over'] == 1
    assert '[residency] fallback' not in capsys.readouterr().out


def test_a_truncated_first_entry_does_not_hide_the_one_behind_it(staged):
    header = staged['files']['header']
    header.write_bytes(header.read_bytes()[:-8])
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], *SPAN)
    assert outcome == RANGE_HIT
    assert (entry['offset'], entry['bytes']) == OWN


def test_a_span_every_covering_entry_fails_refuses_with_the_first_reason(
        staged, capsys):
    staged['files']['header'].unlink()
    own = staged['files']['own']
    own.write_bytes(own.read_bytes()[:-8])
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], *SPAN)
    assert (entry, outcome) == (None, RANGE_REFUSED)
    report = staged['resolver'].report()
    assert report['fallback_count'] == 1
    assert report['range_rows_passed_over'] == 0
    assert report['fallbacks'][0]['reason'].startswith('staged copy is unreadable')
    assert '[residency] fallback' in capsys.readouterr().out


def test_a_span_only_the_stale_entry_covers_still_refuses(staged):
    staged['files']['header'].unlink()
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], 10, 900)
    assert (entry, outcome) == (None, RANGE_REFUSED)
    assert staged['resolver'].report()['fallback_count'] == 1
