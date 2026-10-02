"""Bound compact metadata without changing operand/read/last-use order."""
from functools import partial

import pytest
import torch

from experiments.spill_readplan_metadata import metadata_session, plan_digest
from prismaquant import io_engine
from prismaquant import joint_replay_spill as spill
from test_stageb_spill_integrity_1369 import (
    _captured, test_disk_bit_flip_is_rejected_and_restored_chunk_is_bitwise as _bit_flip)


def test_declared_geometry_refuses_before_read_plan_growth():
    session, window = metadata_session(4, 3, packed=True)
    session.geometry.max_parts = len(window.records) - 1
    with pytest.raises(RuntimeError, match="geometry"):
        session._plan(window)
    assert window.last_ref == {} and not window.plan


def test_replay_readers_retain_indexes_instead_of_expanded_chunks(monkeypatch):
    session, window = metadata_session(4, 3, packed=True)
    window.plan = session._plan(window)
    captured = []
    class Stream:
        wait_sink = None
    def capture(entries, **kwargs):
        captured.extend(entries)
        return Stream()
    monkeypatch.setattr(io_engine, "read_stream", capture)
    session._open_replay_stream()
    assert len(captured) == len(window.plan) * session.n_probes
    assert all(type(entry.reader.args[-1]) is int for entry in captured)
    assert all(entry.key is entry.group for entry in captured)


@pytest.mark.parametrize("read_bytes", [96 << 10, 128 << 10, 1 << 20])
def test_all_chunk_fields_offsets_last_uses_and_order_match_existing_plan(read_bytes):
    baseline, left = metadata_session(7, 17, packed=False, read_bytes=read_bytes)
    candidate, right = metadata_session(7, 17, packed=True, read_bytes=read_bytes)
    left.plan = baseline._plan(left)
    right.plan = candidate._plan(right)
    assert plan_digest(left) == plan_digest(right)
    assert list(right.plan) == list(left.plan)
    assert dict(right.last_ref) == dict(left.last_ref)
    assert right.plan[-1] == left.plan[-1]
    assert list(right.plan[::-1]) == list(left.plan[::-1])
    with pytest.raises(IndexError):
        _ = right.plan[len(right.plan)]


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_policy_requires_explicit_bool_before_scratch_creation(value):
    with pytest.raises(ValueError, match="packed_read_plan.*boolean"):
        spill.StageBReplaySpill(root=None, max_bytes=0, geometry=None,
                                window_names=[], n_probes=1,
                                dtype=torch.bfloat16, device="cpu", packed_read_plan=value)


@pytest.mark.parametrize("threads", [False, True])
@pytest.mark.parametrize("scatter_reads", [False, True])
def test_real_spill_replay_remains_bitwise_and_reader_lifetime_closes(
        tmp_path, monkeypatch, threads, scatter_reads):
    original = spill.StageBReplaySpill
    monkeypatch.setattr(spill, "StageBReplaySpill", partial(original, packed_read_plan=True))
    with _captured(tmp_path, monkeypatch, threads=threads, scatter_reads=scatter_reads) as session:
        window = session._windows[0]
        assert not isinstance(window.records, list)
        assert not isinstance(window.plan, list)
        observed = []
        class Lease:
            from torch import nn
            modules = {"a": nn.Linear(8, 8, bias=False)}
            def _observe_invocation(self, name, weight, x, gradient):
                observed.append((name, x.clone(), gradient.clone()))
        for probe in range(2):
            session.replay(0, probe, Lease())
        for index, (name, x, gradient) in enumerate(observed):
            probe, invocation = divmod(index, 2)
            assert name == "a"
            assert torch.equal(x, (torch.arange(32).reshape(4, 8) + invocation * 64).to(torch.bfloat16))
            assert torch.equal(gradient, torch.full_like(x, probe + invocation + 1))
        assert session.telemetry["checksum_bytes_verified"] == 8 * 64
        session.close_replay_stream()
        assert session._replay_stream is None


@pytest.mark.parametrize("kind,probe", [("x", 0), ("g", 0), ("g", 1)])
def test_integrity_and_terminal_failure_are_not_weakened(tmp_path, monkeypatch, kind, probe):
    original = spill.StageBReplaySpill
    monkeypatch.setattr(spill, "StageBReplaySpill", partial(original, packed_read_plan=True))
    _bit_flip(tmp_path, monkeypatch, kind, probe, False)


def test_published_metadata_is_frozen_and_views_cannot_change_offsets():
    session, window = metadata_session(3, 4, packed=True)
    window.plan = session._plan(window)
    owner, (records, inputs, gradients, used) = window.plan[0]
    before = plan_digest(window)
    with pytest.raises(TypeError):
        inputs[0] = (0, 123)
    with pytest.raises(RuntimeError, match="frozen"):
        window.plan.append((owner, (records, inputs, gradients, used)))
    with pytest.raises(RuntimeError, match="frozen"):
        window.last_ref[owner, 0] = 123
    assert plan_digest(window) == before


def test_empty_operands_preserve_every_record_and_last_use():
    digests = []
    for packed in (False, True):
        session, window = metadata_session(2, 3, packed=packed)
        replacement = spill._PackedRecords(window.names, max_records=session.geometry.max_parts)
        for name, owner, entry, _logical, _bytes, (_shape, stride, residue) in window.records:
            replacement.append((name, owner, entry, 0, 0, ((0, 2048), stride, residue)))
        window.records = replacement
        for entries in window.entries.values():
            for entry in entries:
                entry.logical = entry.nbytes = 0
                entry.layout = ((0, 1024), (1024, 1), 0)
        window.plan = session._plan(window)
        assert len(window.plan) == 1
        assert len(window.plan[0][1][0]) == 6
        assert window.plan[0][1][3] == 0
        digests.append(plan_digest(window))
    assert digests[0] == digests[1]


def test_declared_capture_cap_refuses_before_another_input_or_write(tmp_path, monkeypatch):
    original = spill.StageBReplaySpill
    monkeypatch.setattr(spill, "StageBReplaySpill", partial(original, packed_read_plan=True))
    with _captured(tmp_path, monkeypatch, probes=1) as session:
        session._probe = 0
        session._records_seen = session.geometry.max_parts
        window = session._windows[0]
        entries, records = len(window.entries["a"]), len(window.records)
        def refused_stage(*args):
            pytest.fail("geometry refusal attempted another payload write")
        monkeypatch.setattr(session, "_stage", refused_stage)
        x = torch.zeros(4, 8, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="geometry"):
            session._record("a", x, x)
        assert len(window.entries["a"]) == entries and len(window.records) == records
