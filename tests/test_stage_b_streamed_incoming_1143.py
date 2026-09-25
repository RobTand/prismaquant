"""A spill consumer streams each probe's incoming plane in its spill phase (PQ #1143).

A band-serial consumer used to read its whole incoming plane in
``handoff-load``: 2048 entries, 32 GiB on a GLM row, written into the
cotangent scratch and then read back, one slot at a time, by each probe's
capture. Under the one-pass spill each incoming slot is read exactly once,
by its probe's final pass. So a spill-sealed consumer's derived read plan
stages probe p's plane entries in ``spill-pP``, beside the boundaries the
capture reads with them, and the capture reads them from a verified exact
stream instead of the scratch.

The tests here cover the derived read plan, the dispatcher's capture units,
the stream (order, bytes, budget, refusals) and the plane staging that reads
it. ``tests/test_band_serial_spill.py`` and
``tests/test_band_serial_batched_regime.py`` run the real quantum through
this path at capture batch 1 and 4 and compare it with chain mode.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import test_band_serial_dispatch as dispatch_tests  # noqa: E402
from prismaquant import joint_layer_quanta as jl  # noqa: E402
from prismaquant.joint_quantum_handoff import (  # noqa: E402
    HANDOFF_LOAD_PHASE,
    band_serial_manifest,
    band_serial_manifest_bytes,
    handoff_read_entries,
    load_quantum_handoff,
)
from test_quantum_executable_readset import N_BATCHES, N_PROBES  # noqa: E402

_offline_tier_policy = dispatch_tests._offline_tier_policy


def _spill_consumer(tmp_path):
    """Layer 2 of the two-band fixture, sealed for the spill, and layer 3's handoff."""
    bound, receipt = dispatch_tests._campaign(tmp_path, replay_mode="spill")
    published = dispatch_tests._emit(bound[3])
    consumer = bound[2]
    adjoint_slice = dispatch_tests._slice(consumer)
    handoff = load_quantum_handoff(published["path"], published["sha256"],
                                   record=consumer, adjoint_slice=adjoint_slice,
                                   kda_capture_kernel=None)
    return consumer, adjoint_slice["checkpoint"], handoff, receipt


def _derived(consumer, handoff, checkpoint):
    return band_serial_manifest(
        dispatch_tests._sealed(consumer), handoff, checkpoint,
        sealed_manifest_sha256=consumer["executable_readset"]["manifest_sha256"])


def _rows(manifest, name):
    phase = next(p for p in manifest["read_plan"]["phases"] if p["name"] == name)
    return [manifest["entries"][index] for index in phase["entry_indices"]]


def _plane_row(handoff, probe, batch):
    from prismaquant.joint_quantum_handoff import _plane_coordinates
    (entry,) = [entry for entry in handoff["activation_entries"]
                if _plane_coordinates(entry)[:2] == (probe, batch)]
    return {"path": entry["path"], "offset": 0, "bytes": int(entry["file_bytes"]),
            "sha256": entry["sha256"]}


def _spill_phase(probe):
    return jl.executable_spill_phase_name(probe)


# -- the derived read plan ----------------------------------------------------

def test_a_spill_consumer_stages_each_probes_incoming_in_its_spill_phase(tmp_path):
    """``handoff-load`` keeps the owner states and the shared-pass entries;
    ``spill-pP`` stages probe P's plane entries, each beside the boundary
    entry of its batch. The phase sequence does not change."""
    consumer, checkpoint, handoff, _receipt = _spill_consumer(tmp_path)
    sealed = dispatch_tests._sealed(consumer)
    assert sealed["annotations"]["replay_mode"] == "spill"
    derived = _derived(consumer, handoff, checkpoint)

    sealed_names = [phase["name"] for phase in sealed["read_plan"]["phases"]]
    names = [phase["name"] for phase in derived["read_plan"]["phases"]]
    assert names == ["head", HANDOFF_LOAD_PHASE, *sealed_names[4:]]
    plane = {_plane_row(handoff, p, b)["path"]: (p, b)
             for p in range(N_PROBES) for b in range(N_BATCHES)}

    load = _rows(derived, HANDOFF_LOAD_PHASE)
    assert not [row for row in load if row["path"] in plane], (
        "handoff-load still stages the incoming plane")
    # The rest of what the head reads, in the head's order.
    assert load == [row for row in handoff_read_entries(handoff, checkpoint)
                    if row["path"] not in plane]

    for probe in range(N_PROBES):
        boundaries = _rows(sealed, _spill_phase(probe))
        assert len(boundaries) == N_BATCHES
        expected = []
        for batch in range(N_BATCHES):
            expected += [boundaries[batch], _plane_row(handoff, probe, batch)]
        assert _rows(derived, _spill_phase(probe)) == expected, probe

    # Every other phase stages the sealed bytes, unchanged.
    for name in ["head", *sealed_names[4:]]:
        if not name.startswith("spill-p"):
            assert _rows(derived, name) == _rows(sealed, name), name
    # Each plane entry is staged once, in its own probe's spill phase.
    staged = [row["path"] for phase in derived["read_plan"]["phases"]
              for row in _rows(derived, phase["name"]) if row["path"] in plane]
    assert sorted(staged) == sorted(plane)
    assert derived["annotations"]["band_serial"]["streamed_incoming"] == {
        _spill_phase(probe): N_BATCHES for probe in range(N_PROBES)}
    assert derived["entry_count"] == len(derived["entries"])
    assert derived["total_bytes"] == sum(e["bytes"] for e in derived["entries"])
    assert derived["read_plan"]["read_bytes"] == sum(
        phase["bytes"] for phase in derived["read_plan"]["phases"])
    assert len({(e["path"], e["offset"]) for e in derived["entries"]}) == len(
        derived["entries"])
    assert band_serial_manifest_bytes(
        consumer, handoff, checkpoint,
        output_root=tmp_path) == jl.seal_manifest_bytes(derived)


def test_a_windowed_consumer_keeps_its_whole_plane_in_handoff_load(tmp_path):
    """A windowed row reads every incoming slot in every pass, so its plane
    stays in ``handoff-load`` and nothing is annotated as streamed."""
    bound, _receipt = dispatch_tests._campaign(tmp_path)
    published = dispatch_tests._emit(bound[3])
    consumer = bound[2]
    checkpoint = dispatch_tests._slice(consumer)["checkpoint"]
    handoff = load_quantum_handoff(published["path"], published["sha256"],
                                   record=consumer,
                                   adjoint_slice=dispatch_tests._slice(consumer),
                                   kda_capture_kernel=None)
    derived = _derived(consumer, handoff, checkpoint)
    assert _rows(derived, HANDOFF_LOAD_PHASE) == handoff_read_entries(
        handoff, checkpoint)
    assert "streamed_incoming" not in derived["annotations"]["band_serial"]


def test_the_streamed_plan_passes_the_pb_phase_planner(tmp_path):
    from test_quantum_executable_readset import _pb
    core, tiers, _plans = _pb()
    consumer, checkpoint, handoff, _receipt = _spill_consumer(tmp_path)
    derived = _derived(consumer, handoff, checkpoint)
    prefix = str(tmp_path)
    for entry in derived["entries"]:
        if entry["path"].startswith("/fixture"):
            entry["path"] = "/mnt/shared/fixture" + entry["path"][len("/fixture"):]
        elif entry["path"].startswith(prefix):
            entry["path"] = "/mnt/shared/fixture-run" + entry["path"][len(prefix):]
    validated = core.validate_data_manifest(derived)
    ranges = tiers.manifest_phase_ranges(validated)
    assert [item["name"] for item in ranges] == [
        phase["name"] for phase in derived["read_plan"]["phases"]]
    assert ranges[-1]["end_bytes"] == derived["read_plan"]["read_bytes"]
    # The movers walk each phase's entries in the order the pass reads them.
    walked = [entry["path"] for entry in tiers.manifest_read_entries(validated)]
    expanded = [derived["entries"][index]["path"]
                for phase in derived["read_plan"]["phases"]
                for index in phase["entry_indices"]]
    assert walked == expanded


def test_the_dispatcher_counts_a_spill_capture_by_its_boundary_entries(tmp_path):
    """The capture's units are its stored batches: the moved plane entries
    add read bytes to ``spill-pP``, never capture groups."""
    import dispatch_joint_quanta as dispatch

    consumer, checkpoint, handoff, _receipt = _spill_consumer(tmp_path)
    sealed_facts, sealed_annotations = dispatch._read_plan_phase_facts(
        jl.seal_manifest_bytes(dispatch_tests._sealed(consumer)), where="sealed")
    derived_facts, derived_annotations = dispatch._read_plan_phase_facts(
        band_serial_manifest_bytes(consumer, handoff, checkpoint,
                                   output_root=tmp_path), where="derived")
    for probe in range(N_PROBES):
        name = _spill_phase(probe)
        assert derived_facts[name]["entries"] == 2 * N_BATCHES
        assert derived_facts[name]["bytes"] > sealed_facts[name]["bytes"]
        for capture_batch in (1, 2):
            sealed_work = dispatch.compute_phase_work(
                name, replay_mode="spill",
                entries=dispatch.phase_work_entries(
                    name, sealed_facts[name], sealed_annotations),
                n_probes=N_PROBES, capture_batch=capture_batch)
            derived_work = dispatch.compute_phase_work(
                name, replay_mode="spill",
                entries=dispatch.phase_work_entries(
                    name, derived_facts[name], derived_annotations),
                n_probes=N_PROBES, capture_batch=capture_batch)
            assert derived_work == sealed_work
            assert derived_work[0]["units"] == math.ceil(N_BATCHES / capture_batch)


# -- the stream ----------------------------------------------------------------

def _entry_bytes(handoff):
    (size,) = {int(entry["tensor_bytes"]) for entry in handoff["activation_entries"]}
    return size


class _Budget:
    """``reserve_resident``'s contract: refuses past the cap, never below zero."""

    def __init__(self, cap):
        self.cap, self.held, self.peak = cap, 0, 0

    def __call__(self, delta):
        value = self.held + delta
        if value < 0 or value > self.cap:
            raise RuntimeError("exact boundary tensor residency budget exceeded")
        self.held = value
        self.peak = max(self.peak, value)


def _spy_reads(monkeypatch):
    from prismaquant import joint_adjoint_checkpoints as checkpoints
    opened = []
    read_tensors = checkpoints.read_exact_entry_tensors
    read_payload = checkpoints._read_shared_state_payload

    def spy_tensors(records, **kwargs):
        records = list(records)
        opened.extend(entry["path"] for entry in records)
        return read_tensors(records, **kwargs)

    def spy_payload(path, entry):
        opened.append(str(path))
        return read_payload(path, entry)

    monkeypatch.setattr(checkpoints, "read_exact_entry_tensors", spy_tensors)
    monkeypatch.setattr(checkpoints, "_read_shared_state_payload", spy_payload)
    return opened


def test_each_probe_streams_what_its_spill_phase_stages(tmp_path, monkeypatch):
    """The head reads what ``handoff-load`` stages and no plane entry; each
    probe's stream reads what ``spill-pP`` stages for it, in order, with the
    producer's bytes, inside the budget it is given."""
    from prismaquant.joint_quantum_handoff import HandoffIncoming, load_handoff_inputs

    consumer, checkpoint, handoff, _receipt = _spill_consumer(tmp_path)
    derived = _derived(consumer, handoff, checkpoint)
    plane_paths = {_plane_row(handoff, p, b)["path"]
                   for p in range(N_PROBES) for b in range(N_BATCHES)}
    opened = _spy_reads(monkeypatch)
    plane, owners, shared_pass = load_handoff_inputs(
        handoff, checkpoint, n_probes=N_PROBES, n_batches=N_BATCHES,
        stream_incoming=True)
    assert dict(plane) == {}
    assert sorted(owners) == [(p, b) for p in range(N_PROBES)
                              for b in range(N_BATCHES)]
    assert sorted(shared_pass) == list(range(N_BATCHES))
    assert opened == [row["path"] for row in _rows(derived, HANDOFF_LOAD_PHASE)]

    incoming = HandoffIncoming(handoff, n_probes=N_PROBES, n_batches=N_BATCHES)
    size = _entry_bytes(handoff)
    assert incoming.max_entry_bytes == size
    for probe in range(N_PROBES):
        opened.clear()
        budget = _Budget(2 * size)
        with incoming.open(probe, max_resident_bytes=budget.cap,
                           residency_check=budget) as stream:
            for batch in range(N_BATCHES):
                assert stream.layout((probe, batch)) == ((2, 4), torch.float32)
                (tensor,) = stream.take([(probe, batch)])
                # dispatch_tests._emit's plane: 10 * probe + batch.
                assert torch.equal(tensor, torch.full((2, 4), 10.0 * probe + batch))
        assert opened == [row["path"] for row in _rows(derived, _spill_phase(probe))
                          if row["path"] in plane_paths]
        assert budget.held == 0
        assert 0 < budget.peak <= budget.cap
        assert stream.telemetry["entries"] == N_BATCHES
        assert stream.telemetry["tensor_bytes"] == N_BATCHES * size


def test_a_stream_refuses_a_budget_under_one_entry(tmp_path):
    from prismaquant.joint_quantum_handoff import HandoffIncoming

    _consumer, _checkpoint, handoff, _receipt = _spill_consumer(tmp_path)
    incoming = HandoffIncoming(handoff, n_probes=N_PROBES, n_batches=N_BATCHES)
    budget = _Budget(_entry_bytes(handoff) - 1)
    with pytest.raises(RuntimeError, match="exceeds the stream's resident budget"):
        with incoming.open(0, max_resident_bytes=budget.cap,
                           residency_check=budget) as stream:
            stream.take([(0, 0)])
    assert budget.held == 0


def test_a_stream_is_taken_in_capture_order_and_to_its_end(tmp_path):
    from prismaquant.joint_quantum_handoff import HandoffIncoming

    _consumer, _checkpoint, handoff, _receipt = _spill_consumer(tmp_path)
    incoming = HandoffIncoming(handoff, n_probes=N_PROBES, n_batches=N_BATCHES)
    size = _entry_bytes(handoff)

    budget = _Budget(2 * size)
    with pytest.raises(RuntimeError, match="capture order"):
        with incoming.open(1, max_resident_bytes=budget.cap,
                           residency_check=budget) as stream:
            stream.take([(1, 1)])
    assert budget.held == 0
    with pytest.raises(RuntimeError, match="capture order"):
        with incoming.open(1, max_resident_bytes=budget.cap,
                           residency_check=budget) as stream:
            stream.take([(0, 0)])
    assert budget.held == 0
    # A pass that ends cleanly has read every incoming entry of its probe.
    with pytest.raises(RuntimeError, match="unread"):
        with incoming.open(1, max_resident_bytes=budget.cap,
                           residency_check=budget) as stream:
            stream.take([(1, 0)])
    assert budget.held == 0
    # A pass that fails keeps its own error; the stream only closes.
    with pytest.raises(KeyError, match="the pass failed"):
        with incoming.open(1, max_resident_bytes=budget.cap,
                           residency_check=budget) as stream:
            stream.take([(1, 0)])
            raise KeyError("the pass failed")
    assert budget.held == 0


def test_a_changed_incoming_entry_refuses_at_its_read(tmp_path):
    """The head does not read the plane, so a changed entry refuses where
    its probe's pass reads it, after the entries before it."""
    from prismaquant.joint_quantum_handoff import HandoffIncoming, load_handoff_inputs

    _consumer, checkpoint, handoff, _receipt = _spill_consumer(tmp_path)
    changed = Path(_plane_row(handoff, 1, N_BATCHES - 1)["path"])
    payload = bytearray(changed.read_bytes())
    payload[-1] ^= 0xFF
    changed.chmod(0o644)
    changed.write_bytes(bytes(payload))
    load_handoff_inputs(handoff, checkpoint, n_probes=N_PROBES,
                        n_batches=N_BATCHES, stream_incoming=True)
    incoming = HandoffIncoming(handoff, n_probes=N_PROBES, n_batches=N_BATCHES)
    size = _entry_bytes(handoff)
    budget = _Budget(2 * size)
    with incoming.open(0, max_resident_bytes=budget.cap,
                       residency_check=budget) as stream:
        for batch in range(N_BATCHES):
            stream.take([(0, batch)])
    with pytest.raises(Exception) as refused:
        with incoming.open(1, max_resident_bytes=budget.cap,
                           residency_check=budget) as stream:
            for batch in range(N_BATCHES):
                (tensor,) = stream.take([(1, batch)])
                assert batch < N_BATCHES - 1, "the changed entry was read"
                assert torch.equal(tensor, torch.full((2, 4), 10.0 + batch))
    assert not isinstance(refused.value, AssertionError), refused.value
    assert budget.held == 0


# -- the plane staging ----------------------------------------------------------

def _scratch(tmp_path, handoff, name):
    from prismaquant.perturbed_x_cache import ExactCotangentScratch
    rows = [{"name": f"cotangent-{p}-{b}", "shape": [2, 4], "dtype": "torch.float32",
             "tensor_bytes": _entry_bytes(handoff)}
            for p in range(N_PROBES) for b in range(N_BATCHES)]
    directory = tmp_path / name
    directory.mkdir()
    return ExactCotangentScratch(rows, directory=directory, max_bytes=1 << 20)


@pytest.mark.parametrize("plane_kind", ["dict", "scratch"])
@pytest.mark.parametrize("group", [1, 2])
def test_the_staging_reads_a_stream_as_it_reads_the_plane(tmp_path, plane_kind, group):
    """Bitwise: the stacked incoming rows a capture group moves to the
    device are the same from the stream as from a plane holding the same
    entries, for a kept plane and for the cotangent scratch, in one-batch
    groups and in stacked groups."""
    from prismaquant.joint_adjoint_checkpoints import PlaneHostStaging
    from prismaquant.joint_quantum_handoff import HandoffIncoming

    _consumer, _checkpoint, handoff, _receipt = _spill_consumer(tmp_path)
    incoming = HandoffIncoming(handoff, n_probes=N_PROBES, n_batches=N_BATCHES)
    size = _entry_bytes(handoff)
    for probe in range(N_PROBES):
        keys = [(probe, batch) for batch in range(N_BATCHES)]
        if plane_kind == "dict":
            loaded, streamed_plane = {}, {}
        else:
            loaded = _scratch(tmp_path, handoff, f"loaded-{group}-{probe}")
            streamed_plane = _scratch(tmp_path, handoff, f"streamed-{group}-{probe}")
        budget = _Budget(2 * size)
        # The plane as the head loads it today: every entry, verified.
        with incoming.open(probe, max_resident_bytes=budget.cap,
                           residency_check=budget) as stream:
            for key in keys:
                (loaded[key],) = stream.take([key])
        plain = PlaneHostStaging(loaded, keys, group_batches=group,
                                 dtype=torch.bfloat16)
        with incoming.open(probe, max_resident_bytes=budget.cap,
                           residency_check=budget) as stream:
            streamed = PlaneHostStaging(streamed_plane, keys, group_batches=group,
                                        dtype=torch.bfloat16, incoming=stream)
            assert streamed.capacity_bytes == plain.capacity_bytes
            with plain.held(), streamed.held():
                for start in range(0, N_BATCHES, group):
                    chunk = keys[start:start + group]
                    expected = plain.incoming(chunk, device="cpu")
                    actual = streamed.incoming(chunk, device="cpu")
                    assert actual.dtype == expected.dtype
                    assert actual.shape == expected.shape
                    assert torch.equal(actual.view(torch.uint8),
                                       expected.view(torch.uint8)), chunk
        assert budget.held == 0
        for owner in (loaded, streamed_plane):
            if hasattr(owner, "close"):
                owner.close()


def test_a_staging_refuses_a_stream_whose_rows_differ_from_the_scratch(tmp_path):
    from prismaquant.joint_adjoint_checkpoints import PlaneHostStaging
    from prismaquant.joint_quantum_handoff import HandoffIncoming
    from prismaquant.perturbed_x_cache import ExactCotangentScratch

    _consumer, _checkpoint, handoff, _receipt = _spill_consumer(tmp_path)
    incoming = HandoffIncoming(handoff, n_probes=N_PROBES, n_batches=N_BATCHES)
    rows = [{"name": f"cotangent-{p}-{b}", "shape": [4, 2], "dtype": "torch.float32",
             "tensor_bytes": _entry_bytes(handoff)}
            for p in range(N_PROBES) for b in range(N_BATCHES)]
    scratch = ExactCotangentScratch(rows, directory=tmp_path, max_bytes=1 << 20)
    budget = _Budget(2 * _entry_bytes(handoff))
    try:
        with pytest.raises(RuntimeError, match="the scratch's slot"):
            with incoming.open(0, max_resident_bytes=budget.cap,
                               residency_check=budget) as stream:
                PlaneHostStaging(scratch, [(0, b) for b in range(N_BATCHES)],
                                 group_batches=1, dtype=torch.bfloat16,
                                 incoming=stream)
    finally:
        scratch.close()
    assert budget.held == 0


def test_the_readset_digest_the_quantum_checks_is_the_streamed_plan(tmp_path):
    """The dispatcher stages, and the quantum re-derives, one wire."""
    from prismaquant.joint_quantum_handoff import require_band_serial_readset

    consumer, checkpoint, handoff, _receipt = _spill_consumer(tmp_path)
    wire = band_serial_manifest_bytes(consumer, handoff, checkpoint,
                                      output_root=tmp_path)
    body = json.loads(gzip.decompress(wire))
    assert "streamed_incoming" in body["annotations"]["band_serial"]
    require_band_serial_readset(
        consumer, handoff, checkpoint, output_root=tmp_path,
        data_manifest_sha256=hashlib.sha256(wire).hexdigest())
