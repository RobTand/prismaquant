"""Retained namespace and acquire/open failure obligations without PB owners."""
import errno
import json
import os
from pathlib import Path

import pytest

from prismaquant import residency_map
from prismaquant.staged_lease import acquire_entry_window
from prismaquant.staged_manifest import ManifestRefused, load_staged_manifest, publish_staged_json
from prismaquant.staged_tier_policy import staged_tier_policy_test_context
from staged_native_fixtures import bind, native_staged_state, publish_inputs, sha
from test_native_strict_reader_contract import _forbid_source

pytestmark = pytest.mark.usefixtures('native_staged_state')


def test_retained_phase_cannot_adopt_an_object_outside_its_namespace(tmp_path, monkeypatch):
    source = tmp_path / 'declared'
    raw = b'known-owned-namespace-content'
    source.write_bytes(raw)
    binding = publish_inputs(tmp_path, [source])
    original = load_staged_manifest(binding['path'], binding['sha256'])
    assert Path(original.extents[0].object_path).read_bytes() == raw
    outside = tmp_path / 'outside-retained-phase'
    outside.write_bytes(raw)
    document = json.loads(Path(binding['path']).read_bytes())
    document['entries'][0]['object']['path'] = str(outside)
    replacement = tmp_path / 'outside-manifest.json'
    digest = publish_staged_json(replacement, document)
    opened = _forbid_source(monkeypatch, outside)
    with pytest.raises(ManifestRefused, match='outside retained phase'):
        load_staged_manifest(replacement, digest)
    assert opened == []
    assert sha(raw) == original.extents[0].sha256


@pytest.mark.parametrize('fault', ['missing', 'permission'])
def test_open_failure_after_acquisition_releases_only_its_owned_window(tmp_path, monkeypatch, fault):
    sources = [tmp_path / name for name in ('victim', 'retained-reader')]
    payloads = [b'victim-known-bytes', b'independent-surviving-reader-bytes']
    for source, raw in zip(sources, payloads):
        source.write_bytes(raw)
    resolver = bind(monkeypatch, publish_inputs(tmp_path, sources))
    opened = [_forbid_source(monkeypatch, source) for source in sources]
    for source in sources:
        source.unlink()
    entries = [resolver.staged_read(source) for source in sources]
    victim = Path(entries[0]['stage_path'])
    with staged_tier_policy_test_context('ssd'):
        stable, stable_key = acquire_entry_window(resolver, sources[1], entries[1])
        failed, failed_key = acquire_entry_window(resolver, sources[0], entries[0])
        with stable:
            fd, _ = stable.open(stable_key)
            assert os.pread(fd, len(payloads[1]), 0) == payloads[1]
            with failed:
                assert resolver._windows == 2
                if fault == 'missing':
                    victim.unlink()
                else:
                    original_open = residency_map.open_regular

                    def denied(filename):
                        if Path(filename) == victim:
                            raise PermissionError(errno.EACCES, 'owned object open refused', str(filename))
                        return original_open(filename)

                    monkeypatch.setattr(residency_map, 'open_regular', denied)
                with pytest.raises(FileNotFoundError if fault == 'missing' else PermissionError):
                    failed.open(failed_key)
                assert failed._fds == {}
            assert resolver._windows == 1
            assert os.pread(fd, len(payloads[1]), 0) == payloads[1]
        assert resolver._windows == 0
        with pytest.raises(OSError):
            os.fstat(fd)
    assert resolver.report()['bytes_from_pool'] == 0
    assert all(not calls for calls in opened)
    resolver.close()
    assert resolver._fds == {}
