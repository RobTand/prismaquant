"""The overlay fence re-hashes drifted wires on a bounded pool and reports as it admits (#1519).

A4 r7 spent its last 600 s in one thread re-hashing stat-drifted overlay wires
(``joint_catalog_extension._artifact_fence``), reported nothing, and was killed
``no_progress``. These tests pin the pool, the refusals it must keep, the
single hash per file and the loader's cadenced overlay progress.
"""
import hashlib
import re
import threading
import time
from pathlib import Path

import pytest

from prismaquant import joint_catalog_extension as jce
from tests.test_joint_catalog_extension import _overlay_case, _touch_ctime
from tests.test_joint_quanta_join import campaign, probe  # noqa: F401 - fixtures


class _DigestProbe:
    """Wrap ``hashlib.file_digest``: count calls per path and the peak concurrency."""

    def __init__(self, monkeypatch, delay_s=0.05):
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.paths = []
        self.threads = set()
        real = hashlib.file_digest

        def probe_digest(handle, digest):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.paths.append(str(handle.name))
                self.threads.add(threading.current_thread().name)
            try:
                time.sleep(delay_s)
                return real(handle, digest)
            finally:
                with self.lock:
                    self.active -= 1

        monkeypatch.setattr(jce.hashlib, 'file_digest', probe_digest)


def _drift_every_wire(case):
    time.sleep(0.01)
    wires = [Path(row['wire']) for row in case.rows]
    for wire in wires:
        _touch_ctime(wire)
    return wires


def test_drifted_wires_are_hashed_concurrently_and_admitted_in_order(tmp_path, campaign, probe, monkeypatch):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    wires = _drift_every_wire(case)
    assert len(wires) >= 4
    digests = _DigestProbe(monkeypatch)
    jce.FENCE_REHASHED.clear()
    seen = []
    result = jce.attach_candidate_overlay(data, bound, verify_payloads=False, hash_workers=4,
                                          progress=lambda admitted, unit: seen.append((admitted, unit)))
    assert digests.peak >= 2, f'peak concurrent re-hashes {digests.peak}: the pool is not used'
    assert all(name.startswith('overlay-fence-hash') for name in digests.threads)
    assert sorted(digests.paths) == sorted(str(w) for w in wires), 'each drifted wire hashed exactly once'
    assert jce.FENCE_REHASHED == {'overlay current wire fence': len(wires)}
    assert len(result.cells) == 12
    # Catalog order, cumulative, one report per admitted cell.
    selected = [f"{row['qname']}@{row['format']}" for row in case.rows]
    assert seen == list(enumerate(selected, start=1))


def test_a_mismatched_rehash_refuses_and_names_the_file(tmp_path, campaign, probe, monkeypatch):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    wires = _drift_every_wire(case)
    bad = wires[len(wires) // 2]
    before = bad.stat()
    body = bytearray(bad.read_bytes())
    body[-1] ^= 1
    bad.write_bytes(bytes(body))
    import os
    os.utime(bad, ns=(before.st_atime_ns, before.st_mtime_ns))
    digests = _DigestProbe(monkeypatch)
    with pytest.raises(ValueError, match=re.escape(str(bad))) as refused:
        jce.attach_candidate_overlay(data, bound, verify_payloads=False, hash_workers=4)
    assert 'content re-hash after stat drift' in str(refused.value)
    assert digests.peak >= 2
    bad_row = next(row for row in case.rows if Path(row['wire']) == bad)
    assert (bad_row['qname'], bad_row['format']) not in data.cells, 'an unproven cell was admitted'


def test_a_size_change_refuses_without_hashing(tmp_path, campaign, probe, monkeypatch):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    wires = _drift_every_wire(case)
    with open(wires[0], 'ab') as handle:
        handle.write(b'grown')
    digests = _DigestProbe(monkeypatch)
    with pytest.raises(ValueError, match='overlay current wire fence differs'):
        jce.attach_candidate_overlay(data, bound, verify_payloads=False, hash_workers=4)
    assert str(wires[0]) not in digests.paths, 'a size change must refuse before any hash'


def test_an_undigested_render_drift_keeps_the_strict_fence(tmp_path, campaign, probe, monkeypatch):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    time.sleep(0.01)
    render = Path(case.rows[0]['render'])
    _touch_ctime(render)
    digests = _DigestProbe(monkeypatch)
    with pytest.raises(ValueError, match='overlay current render fence differs'):
        jce.attach_candidate_overlay(data, bound, verify_payloads=False, hash_workers=4)
    assert str(render) not in digests.paths


def test_verify_payloads_hashes_each_drifted_file_once(tmp_path, campaign, probe, monkeypatch):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    wires = _drift_every_wire(case)
    renders = [Path(row['render']) for row in case.rows]
    digests = _DigestProbe(monkeypatch, delay_s=0)
    jce.FENCE_REHASHED.clear()
    result = jce.attach_candidate_overlay(data, bound, verify_payloads=True, hash_workers=4)
    assert sorted(digests.paths) == sorted(str(p) for p in wires + renders), \
        'the drifted fence and payload verification share one wire hash'
    assert jce.FENCE_REHASHED == {'overlay current wire fence': len(wires)}
    for row in case.rows:
        cell = result.cells[row['qname'], row['format']]
        assert cell['render_file_sha256'] == hashlib.sha256(Path(row['render']).read_bytes()).hexdigest()


def test_deferred_render_hashes_are_not_read(tmp_path, campaign, probe, monkeypatch):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    digests = _DigestProbe(monkeypatch, delay_s=0)
    result = jce.attach_candidate_overlay(data, bound, verify_payloads=True, defer_render_hashes=True,
                                          hash_workers=2)
    assert sorted(digests.paths) == sorted(row['wire'] for row in case.rows)
    assert all('render_file_sha256' not in result.cells[row['qname'], row['format']] for row in case.rows)


def test_fence_pool_size_is_bounded_by_the_assigned_cpus():
    import os
    assigned = len(os.sched_getaffinity(0))
    assert jce._fence_hash_workers() == assigned
    with pytest.raises(ValueError, match='assigned CPU affinity'):
        jce._fence_hash_workers(assigned + 1)
    with pytest.raises(ValueError, match='assigned CPU affinity'):
        jce._fence_hash_workers(0)
