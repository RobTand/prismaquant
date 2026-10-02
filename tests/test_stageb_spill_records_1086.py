"""PQ #1086: compact firing-order metadata without changing spill replay."""
from functools import partial

import pytest
import torch
from torch import nn

import prismaquant.joint_replay_spill as spill
from test_stageb_spill_integrity_1369 import (
    _captured, test_disk_bit_flip_is_rejected_and_restored_chunk_is_bitwise as _bit_flip)


@pytest.fixture
def packed(monkeypatch):
    monkeypatch.setattr(spill, "StageBReplaySpill",
                        partial(spill.StageBReplaySpill, packed_records=True))


@pytest.mark.parametrize("threads", [False, True])
@pytest.mark.parametrize("scatter_reads", [False, True])
def test_packed_records_replay_the_same_live_operands(
        tmp_path, monkeypatch, packed, threads, scatter_reads):

    class Lease:
        modules = {"a": nn.Linear(8, 8, bias=False)}

        def __init__(self):
            self.observed = []

        def _observe_invocation(self, name, weight, x, gradient):
            assert name == "a" and weight is self.modules[name].weight
            self.observed.append((x.clone(), gradient.clone()))

    with _captured(tmp_path, monkeypatch, threads=threads,
                   scatter_reads=scatter_reads) as session:
        records = session._windows[0].records
        assert not isinstance(records, list)
        assert [row[:5] for row in records] == [
            ("a", "a", 0, 0, 64), ("a", "a", 1, 64, 64)]
        for probe in range(2):
            lease = Lease()
            session.replay(0, probe, lease)
            assert len(lease.observed) == 2
            for invocation, (x, gradient) in enumerate(lease.observed):
                expected_x = (torch.arange(32).reshape(4, 8) + invocation * 64).to(
                    torch.bfloat16)
                assert torch.equal(x, expected_x)
                assert torch.equal(gradient, torch.full_like(x, probe + invocation + 1))
        assert session.telemetry["checksum_bytes_verified"] == 8 * 64


@pytest.mark.parametrize("kind,probe", [("x", 0), ("g", 0), ("g", 1)])
@pytest.mark.parametrize("threads", [False, True])
def test_packed_records_preserve_operand_corruption_gate(
        tmp_path, monkeypatch, packed, kind, probe, threads):
    _bit_flip(tmp_path, monkeypatch, kind, probe, threads)


@pytest.mark.parametrize("mismatch", ["entry", "logical", "bytes", "shape", "stride"])
def test_packed_records_still_refuse_probe_order_and_layout_drift(
        tmp_path, monkeypatch, packed, mismatch):
    with _captured(tmp_path, monkeypatch, probes=1) as session:
        window = session._windows[0]
        session._probe = 1
        window.entry_cursor = {}
        window.record_cursor = 1 if mismatch == "entry" else 0
        window.g_logical = {"a": 64} if mismatch == "logical" else {}
        x = torch.arange(32).reshape(4, 8).to(torch.bfloat16)
        g = torch.ones_like(x)
        if mismatch == "bytes":
            g = g[:2]
        elif mismatch == "shape":
            g = g.reshape(2, 16)
        elif mismatch == "stride":
            g = torch.ones((8, 4), dtype=torch.bfloat16).t()
        with pytest.raises(RuntimeError, match="invocation order or gradient layout"):
            session._record("a", x, g)


def test_packed_record_sequence_preserves_random_access_and_order():
    records = spill._PackedRecords(("gate", "up"), max_records=4)
    layout = ((4, 8), (8, 1), 128)
    expected = [("gate", "gate", 0, 0, 64, layout),
                ("up", "gate", 0, 0, 64, layout),
                ("gate", "gate", 1, 64, 64, layout)]
    for row in expected:
        records.append(row)
    assert list(records) == expected
    assert records[-1] == expected[-1]
    assert records[1:] == expected[1:]
    assert records[::-1] == expected[::-1]
    assert len(records._layouts) == 1
    assert records._rows.itemsize == 8
    assert len(records._rows) == len(expected) * 6
    with pytest.raises(IndexError):
        _ = records[len(expected)]
    with pytest.raises(IndexError):
        _ = records[-len(expected) - 1]


def test_packed_record_bounds_and_roster_fail_before_append():
    records = spill._PackedRecords(("a",), max_records=1)
    row = ("a", "a", 0, 0, 64, ((4, 8), (8, 1), 0))
    with pytest.raises(ValueError, match="roster"):
        records.append(("foreign", *row[1:]))
    with pytest.raises(ValueError, match="unsigned"):
        records.append((*row[:2], -1, *row[3:]))
    assert len(records) == 0 and not records._layouts
    records.append(row)
    with pytest.raises(RuntimeError, match="geometry"):
        records.append(row)
    assert list(records) == [row]


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_packed_records_policy_refuses_non_boolean(value):
    with pytest.raises(ValueError, match="packed_records.*boolean"):
        spill.StageBReplaySpill(root=None, max_bytes=0, geometry=None,
                                window_names=[], n_probes=1,
                                dtype=torch.bfloat16, device="cpu",
                                packed_records=value)


def test_default_record_storage_stays_a_list(tmp_path, monkeypatch):
    with _captured(tmp_path, monkeypatch) as session:
        assert isinstance(session._windows[0].records, list)
