"""The overlay fence re-hashes drifted wires through the IO engine and reports as it admits (#1519, #1531).

A4 r7 spent its last 600 s in one thread re-hashing stat-drifted overlay wires
(``joint_catalog_extension._artifact_fence``), reported nothing, and was killed
``no_progress``. #1519 moved the re-hash onto a pool of the fence's own; #1531
moved it onto the process's one IO engine (``io_engine.read_stream``, #1294),
which decides how many files are read at once. These tests pin the contract,
not a pool: the reads go through the engine, cells are admitted in catalog
order however the reads complete, one bad digest refuses and admits nothing
after it, each file is hashed once, and progress counts every admission.
"""
import hashlib
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import joint_catalog_extension as jce
from tests.test_joint_catalog_extension import _overlay_case, _touch_ctime
from tests.test_joint_quanta_join import campaign, probe  # noqa: F401 - fixtures


class _DigestProbe:
    """Wrap ``hashlib.file_digest``: count calls per overlay artifact and record completion order.

    Only the case's own wires and renders are counted: adoption proofs hash
    their result files through the same function. ``delay_s`` is seconds, or
    a function of the path giving them.
    """

    def __init__(self, monkeypatch, case, delay_s=0.05):
        watch = {str(row[field]) for row in case.rows for field in ('wire', 'render')}
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.paths = []
        self.completed = []
        self.threads = set()
        real = hashlib.file_digest
        delay = delay_s if callable(delay_s) else (lambda _path: delay_s)

        def probe_digest(handle, digest):
            if str(handle.name) not in watch:
                return real(handle, digest)
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.paths.append(str(handle.name))
                self.threads.add(threading.current_thread().name)
            try:
                time.sleep(delay(str(handle.name)))
                return real(handle, digest)
            finally:
                with self.lock:
                    self.active -= 1
                    self.completed.append(str(handle.name))

        monkeypatch.setattr(hashlib, 'file_digest', probe_digest)


@pytest.fixture
def engine(monkeypatch):
    """A fresh IO engine four reads wide, whatever this runner's CPU set, and its streams.

    Width is the engine's to decide in production; a fixed width here lets a
    test make reads complete out of order on any runner.
    """
    from prismaquant import io_engine
    fresh = io_engine.IOEngine()
    fresh.width = 4
    streams = []
    real = io_engine.read_stream

    def spy(*args, **kwargs):
        stream = real(*args, **kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setattr(io_engine, 'ENGINE', fresh)
    monkeypatch.setattr(io_engine, 'read_stream', spy)
    yield SimpleNamespace(engine=fresh, streams=streams)
    if fresh._pool is not None:
        fresh._pool.shutdown(wait=True)


def _read_through_engine(engine, digests, files):
    """Every hashed file was read by one engine stream, on the engine's own threads."""
    fence = [s for s in engine.streams if s.counters['entries'] == len(files)]
    assert len(fence) == 1, f'{len(fence)} engine streams of {len(files)} entries'
    counters = fence[0].counters
    assert counters['entries_read'] == len(files)
    assert counters['pool_width'] == engine.engine.width
    assert digests.threads and all(name.startswith('pq-io') for name in digests.threads), digests.threads


def _drift_every_wire(case):
    time.sleep(0.01)
    wires = [Path(row['wire']) for row in case.rows]
    for wire in wires:
        _touch_ctime(wire)
    return wires


def test_drifted_wires_are_admitted_in_catalog_order_when_reads_complete_out_of_order(
        tmp_path, campaign, probe, monkeypatch, engine):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    wires = _drift_every_wire(case)
    assert len(wires) >= 4
    # The first wire in catalog order is the slowest to hash, the last the fastest.
    slowest = {str(w): 0.05 * (len(wires) - i) for i, w in enumerate(wires)}
    digests = _DigestProbe(monkeypatch, case, delay_s=lambda path: slowest.get(path, 0))
    jce.FENCE_REHASHED.clear()
    seen = []
    result = jce.attach_candidate_overlay(data, bound, verify_payloads=False,
                                          progress=lambda admitted, unit: seen.append((admitted, unit)))
    assert digests.completed != [str(w) for w in wires], 'the reads completed in order; nothing was tested'
    _read_through_engine(engine, digests, wires)
    assert sorted(digests.paths) == sorted(str(w) for w in wires), 'each drifted wire hashed exactly once'
    assert jce.FENCE_REHASHED == {'overlay current wire fence': len(wires)}
    assert len(result.cells) == 12
    # Catalog order, cumulative, one report per admitted cell.
    selected = [f"{row['qname']}@{row['format']}" for row in case.rows]
    assert seen == list(enumerate(selected, start=1))


def test_a_mismatched_rehash_refuses_and_names_the_file(tmp_path, campaign, probe, monkeypatch, engine):
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
    _DigestProbe(monkeypatch, case)
    seen = []
    with pytest.raises(ValueError, match=re.escape(str(bad))) as refused:
        jce.attach_candidate_overlay(data, bound, verify_payloads=False,
                                     progress=lambda admitted, unit: seen.append(unit))
    assert 'content re-hash after stat drift' in str(refused.value)
    # Every cell before the bad one in catalog order is admitted and reported;
    # the bad one and every cell after it are not.
    order = [(row['qname'], row['format']) for row in case.rows]
    cut = next(i for i, row in enumerate(case.rows) if Path(row['wire']) == bad)
    assert seen == [f'{name}@{fmt}' for name, fmt in order[:cut]]
    assert all(pair in data.cells for pair in order[:cut])
    assert not any(pair in data.cells for pair in order[cut:]), 'an unproven cell was admitted'


def test_a_size_change_refuses_without_hashing(tmp_path, campaign, probe, monkeypatch):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    wires = _drift_every_wire(case)
    with open(wires[0], 'ab') as handle:
        handle.write(b'grown')
    digests = _DigestProbe(monkeypatch, case)
    with pytest.raises(ValueError, match='overlay current wire fence differs'):
        jce.attach_candidate_overlay(data, bound, verify_payloads=False)
    assert str(wires[0]) not in digests.paths, 'a size change must refuse before any hash'


def test_an_undigested_render_drift_keeps_the_strict_fence(tmp_path, campaign, probe, monkeypatch):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    time.sleep(0.01)
    render = Path(case.rows[0]['render'])
    _touch_ctime(render)
    digests = _DigestProbe(monkeypatch, case)
    with pytest.raises(ValueError, match='overlay current render fence differs'):
        jce.attach_candidate_overlay(data, bound, verify_payloads=False)
    assert str(render) not in digests.paths


def test_verify_payloads_hashes_each_drifted_file_once(tmp_path, campaign, probe, monkeypatch):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    wires = _drift_every_wire(case)
    renders = [Path(row['render']) for row in case.rows]
    digests = _DigestProbe(monkeypatch, case, delay_s=0)
    jce.FENCE_REHASHED.clear()
    result = jce.attach_candidate_overlay(data, bound, verify_payloads=True)
    assert sorted(digests.paths) == sorted(str(p) for p in wires + renders), \
        'the drifted fence and payload verification share one wire hash'
    assert jce.FENCE_REHASHED == {'overlay current wire fence': len(wires)}
    for row in case.rows:
        cell = result.cells[row['qname'], row['format']]
        assert cell['render_file_sha256'] == hashlib.sha256(Path(row['render']).read_bytes()).hexdigest()


def test_deferred_render_hashes_are_not_read(tmp_path, campaign, probe, monkeypatch):
    case = _overlay_case(tmp_path, campaign, probe)
    data, bound = case.bind(lambda rows, scalar_costs: None)
    digests = _DigestProbe(monkeypatch, case, delay_s=0)
    result = jce.attach_candidate_overlay(data, bound, verify_payloads=True, defer_render_hashes=True)
    assert sorted(digests.paths) == sorted(row['wire'] for row in case.rows)
    assert all('render_file_sha256' not in result.cells[row['qname'], row['format']] for row in case.rows)
