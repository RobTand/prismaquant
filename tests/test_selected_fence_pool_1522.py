"""The selected-cache export re-hashes drifted overlay wires on the bounded pool (PQ #1522).

The rooted selected cache rebinds every selected overlay cell to its catalog
row (``require_selected_catalog_cell``). A4 selects all 4,320 ctime-drifted
overlay wires, and that rebind re-hashed each one on the calling thread
(18.1 GB). These tests pin the pool #1519 built for the loader's intake on
this path too, and the refusals it must keep.
"""
import os
import re
import time
from pathlib import Path

import pytest

from prismaquant import joint_catalog_extension as jce
from test_joint_catalog_extension import _touch_ctime
from test_overlay_fence_pool_1519 import _DigestProbe, _pool
from test_tessera_selected_cache import _rooted_case


def _drift_every_wire(case):
    time.sleep(0.01)
    wires = [Path(row['wire']) for row in case.rows]
    for wire in wires:
        _touch_ctime(wire)
    return wires


@pytest.fixture
def pooled(monkeypatch):
    # Pin the pool to a size the runner can overlap; the default is the
    # assigned CPU set, which the #1519 test already bounds.
    size = _pool()
    real = jce._fence_hash_workers
    monkeypatch.setattr(jce, '_fence_hash_workers', lambda requested=None: real(requested or size))
    return size


def test_drifted_selected_wires_are_rehashed_on_the_pool(tmp_path, monkeypatch, pooled):
    case = _rooted_case(tmp_path, monkeypatch)
    wires = _drift_every_wire(case)
    assert len(wires) >= 2
    digests = _DigestProbe(monkeypatch, case)
    jce.FENCE_REHASHED.clear()
    manifest = case.build()
    assert digests.peak >= 2, f'peak concurrent re-hashes {digests.peak}: the pool is not used'
    assert all(name.startswith('overlay-fence-hash') for name in digests.threads), digests.threads
    assert sorted(digests.paths) == sorted(str(w) for w in wires), 'each drifted wire hashed exactly once'
    assert jce.FENCE_REHASHED == {'selected current wire fence': len(wires)}
    assert set(manifest['encoder_adoptions']) == {row['qname'] for row in case.rows}


def test_a_mismatched_selected_rehash_refuses_and_names_the_file(tmp_path, monkeypatch, pooled):
    case = _rooted_case(tmp_path, monkeypatch)
    wires = _drift_every_wire(case)
    bad = wires[-1]
    before = bad.stat()
    body = bytearray(bad.read_bytes())
    body[-1] ^= 1
    bad.write_bytes(bytes(body))
    os.utime(bad, ns=(before.st_atime_ns, before.st_mtime_ns))
    _DigestProbe(monkeypatch, case)
    with pytest.raises(ValueError, match=re.escape(str(bad))) as refused:
        case.build()
    assert 'content re-hash after stat drift' in str(refused.value)


def test_a_selected_size_change_refuses_without_hashing(tmp_path, monkeypatch, pooled):
    case = _rooted_case(tmp_path, monkeypatch)
    wires = _drift_every_wire(case)
    with open(wires[0], 'ab') as handle:
        handle.write(b'grown')
    digests = _DigestProbe(monkeypatch, case)
    with pytest.raises(ValueError, match='selected current wire fence differs'):
        case.build()
    assert str(wires[0]) not in digests.paths, 'a size change must refuse before any hash'


def test_an_undigested_selected_render_drift_keeps_the_strict_fence(tmp_path, monkeypatch, pooled):
    case = _rooted_case(tmp_path, monkeypatch)
    time.sleep(0.01)
    render = Path(case.rows[0]['render'])
    _touch_ctime(render)
    digests = _DigestProbe(monkeypatch, case)
    with pytest.raises(ValueError, match='selected current render fence differs'):
        case.build()
    assert str(render) not in digests.paths
