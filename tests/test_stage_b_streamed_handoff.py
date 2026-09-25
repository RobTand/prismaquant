"""The streamed band-serial handoff (PQ #1251).

A producer quantum's handoff writer runs on a thread of its own while the
quantum's passes run: each ``(probe, batch)`` entry is written once a final
pass has stored it, instead of after the last pass. What must not change:

* the bytes. Every entry, ``owner-states.pkl`` and ``handoff.json`` are
  what the all-final writer (``HandoffEmitter.emit``) writes for the same
  plane, whenever the slots become final;
* the record is written last, so a failure on either thread, or a kill,
  leaves no ``handoff.json`` and no consumer can bind a partial handoff;
* a slot is read only after its final pass stored it, and a cotangent
  scratch slot is write-once from then on.

The quantum-level tests run the real Stage B core on the band-serial
fixture of ``test_quantum_band_serial``; the rest drive the emitter on the
tiny records of ``test_quantum_executable_readset``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant.joint_quantum_handoff import (  # noqa: E402
    HANDOFF_DIRECTORY,
    HANDOFF_RECORD_NAME,
    HandoffEmitter,
    handoff_root,
)
from test_quantum_executable_readset import (  # noqa: E402
    N_BATCHES,
    N_PROBES,
    _bind_slice,
    _tiny_receipt,
    _tiny_records,
)
# The quantum-level tests run offline, as test_quantum_band_serial's do.
from test_joint_cost_quantum_runtime import _offline_tier_policy  # noqa: E402,F401

PREFETCH = 2


class _Owner:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return dict(self.state)


def _producer(tmp_path):
    """Layer 3 of the tiny campaign, bound to its Stage A slice."""
    records, _parent = _tiny_records(tmp_path)
    receipt = _tiny_receipt(tmp_path, records[0]["campaign"])
    record = next(_bind_slice(record, receipt, tmp_path / "adjoint-slices")[0]
                  for record in records if record["layer"] == 3)
    return record, json.loads(Path(record["adjoint"]["slice_path"]).read_text())


def _storage(tmp_path):
    from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA
    return {"schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(tmp_path / "exact"),
            "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
            "max_artifact_bytes": 1 << 20, "prefetch_batches": PREFETCH}


def _emitter(tmp_path, record, adjoint_slice):
    return HandoffEmitter(record=record, adjoint_slice=adjoint_slice,
                          boundary_storage=_storage(tmp_path), capture_batch=1)


def _plane():
    return {(p, b): torch.full((2, 4), 10.0 * p + b)
            for p in range(N_PROBES) for b in range(N_BATCHES)}


def _owners():
    return [[_Owner({"scale": float(p + b)}) for b in range(N_BATCHES)]
            for p in range(N_PROBES)]


def _keys():
    return [(p, b) for p in range(N_PROBES) for b in range(N_BATCHES)]


def _generations(record):
    root = handoff_root(record["output_space"]["root"])
    return sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []


def _statuses(record):
    return [json.loads((generation / "generation.json").read_text())["status"]
            for generation in _generations(record)]


def _records(record):
    return [generation / HANDOFF_RECORD_NAME for generation in _generations(record)
            if (generation / HANDOFF_RECORD_NAME).exists()]


def _writers():
    return [thread for thread in threading.enumerate()
            if thread.name == "handoff-writer" and thread.is_alive()]


def _digest(generation):
    """Every file of one handoff generation: relative path -> sha256."""
    return {str(path.relative_to(generation)):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(generation.rglob("*")) if path.is_file()}


def _pin_generations(monkeypatch, value):
    """Every generation this test binds gets ``value`` as its uuid."""
    import uuid

    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=value))


# -- the bytes ------------------------------------------------------------------

def test_streamed_bytes_equal_the_all_final_bytes(tmp_path, monkeypatch):
    """Finality arriving slot by slot, interleaved with the writes, changes no
    byte of any handoff file, and the published block is the same."""
    record, adjoint_slice = _producer(tmp_path)
    _pin_generations(monkeypatch, 1251)
    root = handoff_root(record["output_space"]["root"])

    serial = _emitter(tmp_path, record, adjoint_slice).emit(
        grad_plane=_plane(), cotangent_owners=_owners(), n_probes=N_PROBES,
        n_batches=N_BATCHES, kda_capture_kernel=None)
    generation = Path(serial["path"]).parent
    expected = _digest(generation)
    assert len([name for name in expected if name.startswith("entries/")]) == (
        N_PROBES * N_BATCHES)
    moved = tmp_path / "all-final"
    root.rename(moved)

    plane = {}
    emitter = _emitter(tmp_path, record, adjoint_slice)
    with emitter.stream(grad_plane=plane, n_probes=N_PROBES, n_batches=N_BATCHES,
                        kda_capture_kernel=None) as stream:
        entries = generation / "entries"
        for key in _keys():
            plane[key] = _plane()[key]
            stream.mark_final([key])
            # The writer writes each entry before the next slot is final.
            deadline = time.monotonic() + 30
            while (len(list(entries.glob("*.pt"))) < _keys().index(key) + 1
                   and time.monotonic() < deadline):
                time.sleep(0.01)
            assert len(list(entries.glob("*.pt"))) == _keys().index(key) + 1
        streamed = stream.finish(_owners())
    assert streamed == serial
    assert _digest(generation) == expected
    telemetry = stream.telemetry
    assert telemetry["entries"] == N_PROBES * N_BATCHES
    assert telemetry["entries_before_finish"] == N_PROBES * N_BATCHES
    assert telemetry["entries_after_finish"] == 0
    assert telemetry["cancelled"] is False and telemetry["error"] is None
    assert not _writers()


def test_a_scratch_plane_through_the_tee_ring_writes_the_same_bytes(
        tmp_path, monkeypatch):
    """A cotangent scratch plane, with final rows through a one-slot tee ring:
    one row per call fits the ring, and the second of two rows in one call
    finds it full, so the writer reads that slot back from the scratch. The
    files are the dict plane's, byte for byte."""
    import mmap

    from prismaquant.joint_replay_spill import _aligned_buffer
    from prismaquant.perturbed_x_cache import ExactCotangentScratch

    record, adjoint_slice = _producer(tmp_path)
    _pin_generations(monkeypatch, 1251)
    root = handoff_root(record["output_space"]["root"])
    serial = _emitter(tmp_path, record, adjoint_slice).emit(
        grad_plane=_plane(), cotangent_owners=_owners(), n_probes=N_PROBES,
        n_batches=N_BATCHES, kda_capture_kernel=None)
    generation = Path(serial["path"]).parent
    expected = _digest(generation)
    root.rename(tmp_path / "all-final")

    records = [{"name": f"cotangent-{p}-{b}", "shape": [2, 4],
                "dtype": "torch.float32", "tensor_bytes": 32} for p, b in _keys()]
    (tmp_path / "scratch").mkdir()
    scratch = ExactCotangentScratch(records, directory=tmp_path / "scratch",
                                    max_bytes=1 << 20)
    try:
        emitter = _emitter(tmp_path, record, adjoint_slice)
        stream = emitter.stream(grad_plane=scratch, n_probes=N_PROBES,
                                n_batches=N_BATCHES, kda_capture_kernel=None)
        stream.attach_tee([_aligned_buffer(32, mmap.PAGESIZE, False).zero_()])
        with stream:
            keys = _keys()
            first, rest = keys[:1], keys[1:]
            for key in first:
                scratch[key] = _plane()[key]
                stream.mark_final([key], [_plane()[key]])
            deadline = time.monotonic() + 30
            while stream.telemetry["tee_hits"] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            for pair in (rest[i:i + 2] for i in range(0, len(rest), 2)):
                for key in pair:
                    scratch[key] = _plane()[key]
                stream.mark_final(pair, [_plane()[key] for key in pair])
            streamed = stream.finish(_owners())
        with pytest.raises(RuntimeError, match="sealed"):
            scratch[keys[0]] = torch.zeros(2, 4)
    finally:
        scratch.close()
    assert streamed == serial
    assert _digest(generation) == expected
    telemetry = stream.telemetry
    assert telemetry["tee_slots"] == 1
    assert telemetry["tee_hits"] + telemetry["scratch_reads"] == N_PROBES * N_BATCHES
    assert telemetry["tee_hits"] >= 1 and telemetry["tee_misses"] >= 1
    assert telemetry["tee_misses"] == telemetry["scratch_reads"]


def test_a_handoff_generation_digest_line(tmp_path, monkeypatch, capsys):
    """The all-final writer's file digests, printed for a before/after
    comparison across trees (PQ #1251): run with a fixed ``--basetemp``."""
    record, adjoint_slice = _producer(tmp_path)
    _pin_generations(monkeypatch, 1251)
    published = _emitter(tmp_path, record, adjoint_slice).emit(
        grad_plane=_plane(), cotangent_owners=_owners(), n_probes=N_PROBES,
        n_batches=N_BATCHES, kda_capture_kernel=None)
    generation = Path(published["path"]).parent
    document = json.loads(Path(published["path"]).read_bytes())
    line = {"published": published, "files": _digest(generation),
            "entries": {entry["name"]: entry["sha256"]
                        for entry in document["activation_entries"]}}
    with capsys.disabled():
        print("HANDOFF-IDENTITY " + json.dumps(line, sort_keys=True), flush=True)


# -- finality --------------------------------------------------------------------

def test_the_finish_refuses_a_slot_no_final_pass_stored(tmp_path):
    record, adjoint_slice = _producer(tmp_path)
    plane = _plane()
    emitter = _emitter(tmp_path, record, adjoint_slice)
    with pytest.raises(RuntimeError, match="never stored by a final pass"):
        with emitter.stream(grad_plane=plane, n_probes=N_PROBES, n_batches=N_BATCHES,
                            kda_capture_kernel=None) as stream:
            stream.mark_final(_keys()[:-1])
            stream.finish(_owners())
    assert _records(record) == [] and _statuses(record) == ["failed"]
    assert stream.telemetry["cancelled"] is True
    assert not _writers()


def test_a_slot_stored_twice_or_outside_the_plane_refuses(tmp_path):
    record, adjoint_slice = _producer(tmp_path)
    emitter = _emitter(tmp_path, record, adjoint_slice)
    with pytest.raises(RuntimeError, match="twice"):
        with emitter.stream(grad_plane=_plane(), n_probes=N_PROBES,
                            n_batches=N_BATCHES, kda_capture_kernel=None) as stream:
            with pytest.raises(RuntimeError, match="outside the plane"):
                stream.mark_final([(N_PROBES, 0)])
            stream.mark_final([(0, 0)])
            stream.mark_final([(0, 0)])
    assert _records(record) == [] and _statuses(record) == ["failed"]
    assert not _writers()
    # A stream is opened once per emitter.
    with pytest.raises(RuntimeError, match="once"):
        emitter.stream(grad_plane=_plane(), n_probes=N_PROBES, n_batches=N_BATCHES,
                       kda_capture_kernel=None)


def test_a_scratch_slot_is_write_once_after_its_final_store(tmp_path):
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    records = [{"name": f"cotangent-0-{i}", "shape": [2, 4],
                "dtype": "torch.float32", "tensor_bytes": 32} for i in range(2)]
    scratch = ExactCotangentScratch(records, directory=tmp_path, max_bytes=1 << 20)
    try:
        with pytest.raises(RuntimeError, match="does not hold"):
            scratch.seal([(0, 0)])
        scratch[0, 0] = torch.ones(2, 4)
        scratch.seal([(0, 0)])
        with pytest.raises(RuntimeError, match="sealed"):
            scratch[0, 0] = torch.zeros(2, 4)
        assert torch.equal(scratch[0, 0], torch.ones(2, 4))
        scratch[0, 1] = torch.zeros(2, 4)       # an unsealed slot still rolls
    finally:
        scratch.close()


# -- failure: no record, on either thread ------------------------------------------

def test_a_writer_failure_reaches_the_main_thread_and_leaves_no_record(tmp_path):
    record, adjoint_slice = _producer(tmp_path)
    source = _plane()

    class Plane(dict):
        def __getitem__(self, key):
            if key == (1, 1):
                raise RuntimeError("fixture: plane read failed")
            return source[key]

    emitter = _emitter(tmp_path, record, adjoint_slice)
    with pytest.raises(RuntimeError, match="plane read failed"):
        with emitter.stream(grad_plane=Plane(), n_probes=N_PROBES,
                            n_batches=N_BATCHES, kda_capture_kernel=None) as stream:
            for key in _keys():
                stream.mark_final([key])
                time.sleep(0.01)
            stream.finish(_owners())
    assert _records(record) == [] and _statuses(record) == ["failed"]
    assert "plane read failed" in stream.telemetry["error"]
    assert not _writers()


def test_leaving_without_a_finish_cancels_the_writer(tmp_path):
    """The quantum failed first: the writer stops between entries, its owner
    exits on its failure path, and the quantum's own error propagates."""
    record, adjoint_slice = _producer(tmp_path)
    emitter = _emitter(tmp_path, record, adjoint_slice)
    with pytest.raises(ValueError, match="fixture: the passes failed"):
        with emitter.stream(grad_plane=_plane(), n_probes=N_PROBES,
                            n_batches=N_BATCHES, kda_capture_kernel=None) as stream:
            stream.mark_final(_keys()[:N_BATCHES])
            raise ValueError("fixture: the passes failed")
    assert _records(record) == [] and _statuses(record) == ["failed"]
    assert stream.telemetry["cancelled"] is True
    assert stream.telemetry["error"] is None
    assert not _writers()


_KILLED_CHILD = r"""
import json, sys, time
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [sys.argv[2], sys.argv[2] + "/tests", sys.argv[2] + "/tools"]
import torch
from test_stage_b_streamed_handoff import (
    N_BATCHES, N_PROBES, _emitter, _keys, _plane, _producer)
record, adjoint_slice = _producer(root)
plane = {}
emitter = _emitter(root, record, adjoint_slice)
with emitter.stream(grad_plane=plane, n_probes=N_PROBES, n_batches=N_BATCHES,
                    kda_capture_kernel=None) as stream:
    for key in _keys()[:N_BATCHES + 1]:
        plane[key] = _plane()[key]
        stream.mark_final([key])
    from prismaquant.joint_quantum_handoff import handoff_root
    print("MARKED " + str(handoff_root(record["output_space"]["root"])), flush=True)
    time.sleep(3600)
"""


def test_a_killed_producer_leaves_no_handoff_a_consumer_can_bind(tmp_path):
    """SIGKILL mid-emit: some entries are on disk, the record is not.

    A consumer binds a handoff only through its ``handoff.json`` path and
    sha256, which the producer reports after the record group is written,
    so a producer killed before that leaves nothing to bind.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", _KILLED_CHILD, str(tmp_path), str(ROOT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(ROOT), str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")])})
    try:
        line = child.stdout.readline().split()
        assert line and line[0] == "MARKED", child.stderr.read()
        record_root = Path(line[1])
        deadline = time.monotonic() + 60
        entries = []
        while time.monotonic() < deadline:
            entries = list(record_root.glob("*/entries/*.pt"))
            if len(entries) == N_BATCHES + 1:
                break
            time.sleep(0.05)
        assert len(entries) == N_BATCHES + 1, entries
        child.send_signal(signal.SIGKILL)
        assert child.wait(timeout=60) == -signal.SIGKILL
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    generations = [path for path in record_root.iterdir() if path.is_dir()]
    assert len(generations) == 1
    generation = generations[0]
    assert not (generation / HANDOFF_RECORD_NAME).exists()
    assert not (generation / "owner-states.pkl").exists()
    # The generation never completed: its status says so.
    status = json.loads((generation / "generation.json").read_text())["status"]
    assert status == "running"
    assert record_root.name == HANDOFF_DIRECTORY


# -- the scratch under two threads ---------------------------------------------------

def _direct_io_supported(directory):
    import tempfile
    from prismaquant.perturbed_x_cache import _direct_io_block
    with tempfile.TemporaryFile(dir=directory) as handle:
        try:
            return 8192 % _direct_io_block(handle.fileno()) == 0
        except (OSError, RuntimeError):
            return False


def test_a_second_thread_reads_sealed_slots_while_the_pass_writes(tmp_path):
    """The handoff writer reads into its own grid buffer while the pass
    writes other slots through the shared bounce buffer: every byte exact."""
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    count = 12
    records = [{"name": f"cotangent-0-{i}", "shape": [2, 1024],
                "dtype": "torch.float32", "tensor_bytes": 8192} for i in range(count)]
    scratch = ExactCotangentScratch(records, directory=tmp_path, max_bytes=1 << 20)
    direct = _direct_io_supported(tmp_path)
    try:
        def value(i, round_):
            return (torch.arange(2048, dtype=torch.int32).reshape(2, 1024)
                    * (i + 1) + round_).view(torch.float32)

        # Off the grid: a source one element into its storage.
        def off_grid(tensor):
            base = torch.empty(tensor.numel() + 1, dtype=tensor.dtype)
            view = base[1:].view(tensor.shape)
            view.copy_(tensor)
            return view

        final = {}
        for i in range(0, count, 2):
            scratch[0, i] = off_grid(value(i, 0))
            final[(0, i)] = value(i, 0)
        scratch.seal(list(final))
        failures = []

        def reader():
            try:
                buffer = scratch.aligned_buffer((0, 0))
                for _round in range(40):
                    for key, expected in final.items():
                        out = scratch.read_into(key, buffer, bounce=False)
                        if not torch.equal(out.view(torch.int32),
                                           expected.view(torch.int32)):
                            failures.append(key)
            except BaseException as exc:            # noqa: BLE001
                failures.append(repr(exc))

        thread = threading.Thread(target=reader)
        thread.start()
        for round_ in range(40):
            for i in range(1, count, 2):
                scratch[0, i] = off_grid(value(i, round_))
                assert torch.equal(scratch[0, i].view(torch.int32),
                                   value(i, round_).view(torch.int32))
        thread.join()
        assert failures == []
        if direct:
            # A reader that refuses the bounce buffer refuses an off-grid target.
            with pytest.raises(RuntimeError, match="off the direct-I/O grid"):
                scratch.read_into((0, 0), off_grid(value(0, 0)), bounce=False)
    finally:
        scratch.close()
    assert list(tmp_path.iterdir()) == []


def test_close_waits_for_a_read_in_flight(tmp_path, monkeypatch):
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    records = [{"name": "cotangent-0-0", "shape": [2, 1024],
                "dtype": "torch.float32", "tensor_bytes": 8192}]
    scratch = ExactCotangentScratch(records, directory=tmp_path, max_bytes=1 << 20)
    scratch[0, 0] = torch.ones(2, 1024)
    entered, release = threading.Event(), threading.Event()
    real = scratch._read_slot

    def slow(key, out, *, bounce):
        entered.set()
        release.wait(30)
        return real(key, out, bounce=bounce)

    monkeypatch.setattr(scratch, "_read_slot", slow)
    result = {}
    reader = threading.Thread(target=lambda: result.setdefault(
        "out", scratch.read_into((0, 0), torch.empty(2, 1024))))
    reader.start()
    assert entered.wait(30)
    closer = threading.Thread(target=scratch.close)
    closer.start()
    time.sleep(0.2)
    assert closer.is_alive(), "close did not wait for the read in flight"
    release.set()
    reader.join(30)
    closer.join(30)
    assert not closer.is_alive()
    assert torch.equal(result["out"], torch.ones(2, 1024))
    assert scratch._file is None


# -- the quantum: entries are written while the final passes run -------------------

def _band_serial():
    import test_quantum_band_serial as band
    return band


@pytest.mark.parametrize("plane", ["dict", "scratch"])
def test_the_handoff_is_written_while_the_final_passes_run(tmp_path, monkeypatch, plane):
    """Red before PQ #1251: every handoff write followed the last pass.

    Each final store here waits, bounded, for the writer to write the slots
    it stored, which the writer can only do while the passes are still
    running. The handoff is then the one the chain-equality test checks.
    """
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.joint_adjoint_checkpoints import PlaneHostStaging

    band = _band_serial()
    if plane == "scratch":
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("PRISMAQUANT_STAGE_B_COTANGENT_ROOT", str(scratch))
        monkeypatch.setenv("PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES", str(1 << 20))
    single, receipt, output_root = band._campaign(tmp_path, monkeypatch)

    events = []
    written = threading.Condition()
    handoff_keys = set()
    real_store, real_write = PlaneHostStaging.store, StreamedBoundaryArtifacts.write

    def write(self, tensor, *, batch_index, boundary_index, probe_index=None, **kw):
        reference = real_write(self, tensor, batch_index=batch_index,
                               boundary_index=boundary_index,
                               probe_index=probe_index, **kw)
        if Path(self.config["directory"]).name == HANDOFF_DIRECTORY:
            with written:
                events.append(("handoff", (probe_index, batch_index)))
                handoff_keys.add((probe_index, batch_index))
                written.notify_all()
        return reference

    final = set()

    def written_prefix(plane):
        # The writer writes in canonical order, probe-major, so it can write
        # exactly the canonical prefix of the slots stored final so far.
        universe = sorted(plane._slots if hasattr(plane, "_slots") else plane.keys())
        prefix = set()
        for key in universe:
            if key not in final:
                break
            prefix.add(key)
        return prefix

    def store(self, keys, rows, gradient):
        keys = list(keys)
        real_store(self, keys, rows, gradient)
        with written:
            events.append(("store", tuple(keys)))
        if getattr(self, "on_store", None) is not None:
            with written:
                final.update(keys)
                prefix = written_prefix(self.plane)
                assert written.wait_for(lambda: prefix <= handoff_keys, timeout=60), (
                    "the writer did not write the stored slots while the passes ran")

    monkeypatch.setattr(StreamedBoundaryArtifacts, "write", write)
    monkeypatch.setattr(PlaneHostStaging, "store", store)
    _payload, record, counters = band._quantum(
        tmp_path, monkeypatch, single=single, receipt=receipt,
        output_root=output_root, layer=2, handoff_emitter=band._emitter)

    kinds = [kind for kind, _key in events]
    assert "handoff" in kinds and "store" in kinds
    last_store = len(kinds) - 1 - kinds[::-1].index("store")
    assert kinds.index("handoff") < last_store, (
        "every handoff entry was written after the last pass stored")
    emitted = counters["handoff_emit"]
    n_entries = emitted["entries"]
    assert len(handoff_keys) == n_entries > 0
    assert emitted["entries_before_finish"] == n_entries
    assert emitted["entries_after_finish"] == 0
    assert emitted["cancelled"] is False and emitted["error"] is None
    if plane == "scratch":
        # Each entry came from the tee ring or, when it was full, a scratch
        # read on the writer's own buffer.
        assert emitted["tee_slots"] >= 2
        assert emitted["tee_hits"] + emitted["scratch_reads"] == n_entries
        assert emitted["tee_misses"] == emitted["scratch_reads"]
        assert emitted["tee_hits"] > 0
        assert list(scratch.iterdir()) == []
    else:
        assert emitted["tee_slots"] == 0 and emitted["scratch_reads"] == 0
    assert len(_records(record)) == 1
    assert not _writers()


def test_a_pass_failure_cancels_the_writer_and_publishes_no_record(
        tmp_path, monkeypatch):
    """The quantum fails in a final pass after the writer wrote entries: its
    error propagates, the writer is joined, and no record exists."""
    from prismaquant.joint_adjoint_checkpoints import PlaneHostStaging

    band = _band_serial()
    single, receipt, output_root = band._campaign(tmp_path, monkeypatch)
    real_store = PlaneHostStaging.store

    def store(self, keys, rows, gradient):
        keys = list(keys)
        if getattr(self, "on_store", None) is not None and keys[0][0] == 1:
            raise ValueError("fixture: a final pass failed")
        real_store(self, keys, rows, gradient)

    monkeypatch.setattr(PlaneHostStaging, "store", store)
    with pytest.raises(ValueError, match="a final pass failed"):
        band._quantum(tmp_path, monkeypatch, single=single, receipt=receipt,
                      output_root=output_root, layer=2, handoff_emitter=band._emitter)
    root = output_root / "layer-quanta" / "layer-002" / HANDOFF_DIRECTORY
    assert not list(root.glob(f"*/{HANDOFF_RECORD_NAME}"))
    statuses = [json.loads(path.read_text())["status"]
                for path in root.glob("*/generation.json")]
    assert statuses == ["failed"]
    assert not _writers()
