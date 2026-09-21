"""Every covering map entry is asked before a span is refused (PQ #902, #903).

PrismaBuild allows two manifest entries to overlap and stages each as a file
of its own, and its composed map can name an entry whose staged file an
eviction already unlinked. The resolver used to take the lowest-offset covering
entry and stop, so a stale neighbour entry hid the layer's own healthy range
and the strict tier policy ended the run on bytes that were staged.

PQ #903: a covering entry whose staged file is merely missing (ENOENT) is not
a refusal when the span is declared -- it is the same waitable miss as no
covering entry at all. Wrong-size, non-regular, permission and other integrity
failures still refuse at once, classified from errno, never by matching the
translated strerror text.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os

import pytest

from prismaquant.residency_map import (
    ENV_VAR, RANGE_HIT, RANGE_REFUSED, RANGE_UNCOVERED, RANGE_UNDECLARED,
    SCHEMA, bind_residency_manifest,
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


def test_a_span_every_covering_entry_fails_refuses_with_the_first_hard_reason(
        staged, capsys):
    # Header is missing (waitable on its own) but the own entry behind it is
    # truncated (hard): integrity first, so the span still refuses at once and
    # reports the hard reason, not the missing one.
    staged['files']['header'].unlink()
    own = staged['files']['own']
    own.write_bytes(own.read_bytes()[:-8])
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], *SPAN)
    assert (entry, outcome) == (None, RANGE_REFUSED)
    report = staged['resolver'].report()
    assert report['fallback_count'] == 1
    assert report['range_rows_passed_over'] == 0
    assert report['fallbacks'][0]['reason'].startswith('staged copy size differs')
    assert '[residency] fallback' in capsys.readouterr().out


def test_a_span_only_the_stale_entry_covers_is_a_waitable_miss(staged, capsys):
    # PQ #903: the only covering staged file is missing. The sealed readset is
    # unbound in this fixture (unknown, never False), so the outcome is the
    # same waitable miss as no covering entry: UNCOVERED, silent, no fallback.
    # The declared + bound case (wait, then serve) is pinned by the strict
    # mid-flight integration tests with real PB writers and lease paths.
    staged['files']['header'].unlink()
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], 10, 900)
    assert (entry, outcome) == (None, RANGE_UNCOVERED)
    report = staged['resolver'].report()
    assert report['fallback_count'] == 0
    assert report['range_misses'] >= 1
    assert '[residency] fallback' not in capsys.readouterr().out


def test_a_missing_span_the_readset_never_declared_is_undeclared(staged, monkeypatch):
    staged['files']['header'].unlink()
    monkeypatch.setattr(
        staged['resolver'], '_declares', lambda path, start, end: False)
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], 10, 900)
    assert (entry, outcome) == (None, RANGE_UNDECLARED)
    assert staged['resolver'].report()['fallback_count'] == 0


def test_a_nonregular_staged_file_still_refuses_at_once(staged, capsys):
    target = staged['files']['header']
    target.unlink()
    target.mkdir()
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], 10, 900)
    assert (entry, outcome) == (None, RANGE_REFUSED)
    report = staged['resolver'].report()
    assert report['fallback_count'] == 1
    assert report['fallbacks'][0]['reason'] == 'staged copy is not a regular file'
    assert '[residency] fallback' in capsys.readouterr().out


def test_a_permission_failure_still_refuses_at_once(staged, monkeypatch, capsys):
    # Classified from errno, never by matching strerror text: EACCES is hard
    # even though the file is "unreadable" like a missing one is.
    real_lstat = os.lstat
    stage_path = str(staged['files']['header'])

    def _denied(path, *args, **kwargs):
        if os.fspath(path) == stage_path:
            raise OSError(errno.EACCES, os.strerror(errno.EACCES))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, 'lstat', _denied)
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], 10, 900)
    assert (entry, outcome) == (None, RANGE_REFUSED)
    report = staged['resolver'].report()
    assert report['fallback_count'] == 1
    assert report['fallbacks'][0]['reason'].startswith('staged copy is unreadable')
    assert '[residency] fallback' in capsys.readouterr().out


def test_a_missing_cause_is_typed_not_substring_matched(staged, monkeypatch):
    # A strerror that happens to mention a missing file must not turn a hard
    # refusal waitable: only errno.ENOENT is missing.
    real_lstat = os.lstat
    stage_path = str(staged['files']['header'])

    def _confusing(path, *args, **kwargs):
        if os.fspath(path) == stage_path:
            raise OSError(errno.EACCES, 'No such file or directory')
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, 'lstat', _confusing)
    entry, outcome = staged['resolver'].staged_range_outcome(staged['shard'], 10, 900)
    assert (entry, outcome) == (None, RANGE_REFUSED)
