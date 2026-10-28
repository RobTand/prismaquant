"""PQ #2008: direct publication must retain the serialized entry's stat fence.

This is an explicit capture operation, not immutable-provider qualification.
Only metadata-observable drift is exercised; no same-signature byte-tamper or
atomic multi-file snapshot guarantee is asserted.
"""
import os
from pathlib import Path

import pytest

from prismaquant import tessera_calibration_cache as cc
from prismaquant.file_identity import file_stat_signature
from test_capture_single_pass_source_1896 import _entry_reads
from test_tessera_calibration_cache import capture  # noqa: F401 (fixture)


def test_publish_refuses_drift_between_entry_write_and_seal(capture, tmp_path, monkeypatch):  # noqa: F811
    _root, census_path, census, identity, acts, hessians, _manifest = capture
    root = tmp_path / 'direct-publish'
    reads = _entry_reads(monkeypatch, root)
    original_write = cc._write_capture_entry
    changed = []

    def write_then_drift(*args, **kwargs):
        # Use the real serializer/digest/fingerprint owner. Deterministically
        # change metadata at its handoff to the publisher, after the write-time
        # fingerprint exists but before the publisher can seal this entry.
        path, record, fingerprint = original_write(*args, **kwargs)
        if not changed:
            before = path.stat()
            os.utime(path, ns=(before.st_atime_ns,
                               before.st_mtime_ns + 1_000_000_000))
            assert file_stat_signature(path.stat()) != file_stat_signature(before)
            changed.append(path)
        return path, record, fingerprint

    monkeypatch.setattr(cc, '_write_capture_entry', write_then_drift)
    with pytest.raises(RuntimeError, match='changed since its writer'):
        cc.publish_capture(root, census_path=census_path, identity=identity,
                           acts=acts, hessians=hessians,
                           counts=census['counts'], maxima=census['max_abs'])

    assert changed
    assert reads == []  # Refuse by stat fence, not a payload hash/load reread.
    assert not (Path(root) / 'capture_manifest.json').exists()
