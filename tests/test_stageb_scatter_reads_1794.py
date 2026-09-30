"""PQ #1794: bounded physical-span coalescing without operand reordering."""
from __future__ import annotations

import pytest
import torch

import prismaquant.joint_replay_spill as spill
from test_stageb_spill_integrity_1369 import _captured, _payload_range


def _permuted(session):
    window = session._windows[0]
    owner, (records, inputs, gradients, used) = window.plan[0]
    assert len(inputs) == len(gradients) == 2
    before, _ = session._read_chunk(window, 0, window.plan[0])
    physical = [session._physical(window.x_runs[owner], window.x_starts[owner],
                                  window.entries[owner][index].logical, 64)
                for index, _ in inputs]
    for index, _ in gradients:
        name, _, _, logical, nbytes, _ = window.records[index]
        physical.append(session._physical(window.g_runs[(name, 0)],
                                           window.g_starts[(name, 0)], logical, nbytes))
    ordered = sorted(physical)
    positions = [3 - ordered.index(offset) for offset in physical]
    block = session._block
    original = list(inputs) + list(gradients)
    remapped = [(index, positions[i] * block + offset % block)
                for i, (index, offset) in enumerate(original)]
    item = (owner, (records, tuple(remapped[:2]), tuple(remapped[2:]), used))
    return window, item, before, original, remapped


@pytest.mark.parametrize("scatter", [False, True])
def test_permuted_destinations_coalesce_only_when_selected(tmp_path, monkeypatch, scatter):
    with _captured(tmp_path, monkeypatch, probes=1) as session:
        window, item, original, source, remapped = _permuted(session)
        # The old implementation ignores this requested policy: the RED is
        # actual four-span behavior, not a missing-constructor TypeError.
        session._scatter_reads = scatter
        calls = []
        real_read = session._scratch.read_into

        def traced(offset, views):
            calls.append((offset, tuple(len(view) for view in views)))
            return real_read(offset, views)

        monkeypatch.setattr(session._scratch, "read_into", traced)
        before = session.telemetry["reads"]
        restored, _ = session._read_chunk(window, 0, item)
        for (_, old), (_, new) in zip(source, remapped):
            assert torch.equal(original[old:old + 64], restored[new:new + 64])
        assert session.telemetry["reads"] - before == (1 if scatter else 4)
        assert len(calls) == (1 if scatter else 4)
        assert sum(sum(lengths) for _, lengths in calls) == 4 * session._block


def test_scatter_calls_obey_byte_and_iovec_limits(tmp_path, monkeypatch):
    with _captured(tmp_path, monkeypatch, probes=1) as session:
        window, item, original, source, remapped = _permuted(session)
        session._scatter_reads = True
        monkeypatch.setattr(spill, "READ_CALL_BYTES", 2 * session._block)
        monkeypatch.setattr(spill, "READ_IOV_MAX", 2, raising=False)
        calls = []
        real_read = session._scratch.read_into

        def traced(offset, views):
            calls.append((offset, tuple(len(view) for view in views)))
            return real_read(offset, views)

        monkeypatch.setattr(session._scratch, "read_into", traced)
        before = session.telemetry["reads"]
        restored, _ = session._read_chunk(window, 0, item)
        assert session.telemetry["reads"] - before == 1
        assert len(calls) == 2
        assert all(len(lengths) <= 2 and sum(lengths) <= 2 * session._block
                   for _, lengths in calls)
        for (_, old), (_, new) in zip(source, remapped):
            assert torch.equal(original[old:old + 64], restored[new:new + 64])


def test_scatter_still_rejects_corrupt_operand(tmp_path, monkeypatch):
    with _captured(tmp_path, monkeypatch, probes=1) as session:
        window, item, _, _, _ = _permuted(session)
        session._scatter_reads = True
        low, _ = _payload_range(session, "x", 0)
        block = session._block
        at = low - low % block
        sector = spill._aligned_buffer(block, block, False)
        view = memoryview(sector.numpy())
        session._scratch.read_into(at, [view])
        sector[low - at] ^= 1
        session._scratch.write(at, [view], call_bytes=block)
        with pytest.raises(spill.TerminalReadError, match="checksum mismatch"):
            session._read_chunk(window, 0, item)


@pytest.mark.parametrize("selected", [False, True])
def test_explicit_constructor_selection(tmp_path, monkeypatch, selected):
    with _captured(tmp_path, monkeypatch, probes=1) as original:
        with spill.StageBReplaySpill(
                root=original._scratch.root, max_bytes=1 << 20,
                geometry=original.geometry, window_names=[("a",)], n_probes=1,
                dtype=torch.bfloat16, device="cpu", threads=False,
                scatter_reads=selected) as session:
            assert session._scatter_reads is selected
            assert session.telemetry["scatter_reads"] is selected


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_invalid_selection_refuses_before_allocation(bad):
    with pytest.raises(ValueError, match="scatter_reads must be a boolean"):
        spill.StageBReplaySpill(root=None, max_bytes=0, geometry=None,
                               window_names=(), n_probes=1, dtype=torch.bfloat16,
                               device="cpu", scatter_reads=bad)


@pytest.mark.parametrize("envelopes", [
    [(0, 4096, 0), (0, 4096, 4096)],  # Same physical bytes twice.
    [(0, 4096, 0), (4096, 8192, 0)],  # Destructive destination alias.
    [(0, 4096, 1)],  # Unaligned destination.
])
def test_invalid_scatter_envelopes_refuse(envelopes):
    with pytest.raises(RuntimeError):
        spill._scatter_read_calls(envelopes, block=4096, call_bytes=8192, max_iov=2)


def test_scatter_gap_and_abutting_destinations():
    assert spill._scatter_read_calls(
        [(0, 4096, 0), (4096, 8192, 4096)],
        block=4096, call_bytes=8192, max_iov=1) == (1, [(0, [(0, 8192)])])
    assert spill._scatter_read_calls(
        [(0, 4096, 4096), (8192, 12288, 0)],
        block=4096, call_bytes=8192, max_iov=2) == (
            2, [(0, [(4096, 4096)]), (8192, [(0, 4096)])])
    assert spill._scatter_read_calls(
        [(0, 12288, 0)], block=4096, call_bytes=8192, max_iov=1) == (
            1, [(0, [(0, 8192)]), (8192, [(8192, 4096)])])


def test_slot_padding_is_written_not_just_reserved(tmp_path, monkeypatch):
    with _captured(tmp_path, monkeypatch, probes=1) as session:
        payload = session.telemetry["x_bytes_written"] + session.telemetry["g_bytes_written"]
        actual = session.telemetry["file_bytes_written"]
        assert actual == session._scratch.allocated == 4 * session._block
        assert actual > payload == 4 * 64
        assert session._scratch.capacity >= actual
        assert spill._slot(0, 64, 64, session._block) == (0, 64, session._block)
