"""Row usage and failure behaviour of the batched cost measurement.

Two defects are pinned here.

ROW TRUNCATION. ``measure_batched_gpu`` used to truncate every Linear in a
chunk to ``min(rows)`` over the chunk's members. On a MoE layer that minimum is
set by the single least-routed expert, so from DSv4-Flash layer 3 onward every
one of the 768 expert Linears had its ``output_mse`` computed from ONE token
row while up to 63 cached rows were thrown away. Each Linear now gets its own
rows, via one BMM per distinct row count (no padding, no waste).

FAIL-OPEN ERRORS. A measurement exception used to be swallowed into a silent
per-chunk ``{"error": ...}``, turning up to ``chunk_size`` priced rows into
holes the allocator reads as "format not offered here". It now aborts.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from prismaquant import format_registry as fr
from prismaquant import measure_quant_cost as mqc


class _FakeActIndex:
    """Minimal ActivationIndex stand-in with per-name row counts."""

    def __init__(self, rows_by_name: dict[str, int], in_f: int, seed: int = 5):
        g = torch.Generator().manual_seed(seed)
        self._acts = {
            n: torch.randn(r, in_f, generator=g)
            for n, r in rows_by_name.items()
        }

    def __contains__(self, name):
        return name in self._acts

    def __len__(self):
        return len(self._acts)

    def load(self, name):
        return self._acts[name]

    def prefetch_bytes(self, name):
        a = self._acts[name]
        return a.numel() * a.element_size() + a.shape[0] * 8

    def load_with_row_indices(self, name):
        a = self._acts[name]
        return a, torch.arange(a.shape[0])


def _model(n: int, out_f: int, in_f: int, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    mod = nn.Module()
    for i in range(n):
        lin = nn.Linear(in_f, out_f, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(out_f, in_f, generator=g) * 0.05)
        mod.add_module(f"lin{i}", lin)
    return mod


def _names(n):
    return [f"lin{i}" for i in range(n)]


def _measure(rows_by_name, in_f=64, out_f=32, fmt="INT8_W8A16", chunk_size=256):
    n = len(rows_by_name)
    model = _model(n, out_f, in_f)
    acts = _FakeActIndex(rows_by_name, in_f)
    spec = fr.get_format(fmt)
    return mqc.measure_batched_gpu(
        model, acts, set(rows_by_name), [spec], "cpu", torch.float32,
        chunk_size=chunk_size)


def test_ragged_rows_do_not_truncate_to_the_chunk_minimum():
    """The one-row Linear must not drag the 64-row Linears down to 1 row."""
    rows = {"lin0": 64, "lin1": 64, "lin2": 1, "lin3": 17}
    got = _measure(rows)
    for i, r in enumerate([64, 64, 1, 17]):
        assert got[f"lin{i}"]["INT8_W8A16"]["n_activation_rows"] == r


def test_well_covered_rows_match_a_homogeneous_chunk():
    """A 64-row Linear is scored identically whether or not a 1-row Linear
    shares its chunk. Under the old chunk-min truncation it was not."""
    in_f, out_f = 64, 32
    homogeneous = _measure({"lin0": 64, "lin1": 64}, in_f, out_f)
    ragged = _measure({"lin0": 64, "lin1": 64, "lin2": 1}, in_f, out_f)
    for k in ("output_mse", "rel_output_mse", "weight_mse"):
        assert homogeneous["lin0"]["INT8_W8A16"][k] == ragged["lin0"]["INT8_W8A16"][k], k


def test_sparse_experts_keep_their_honest_estimate():
    """A genuinely 1-row Linear still reports a 1-row estimate — the defect
    was artificial truncation, not the existence of sparse rows."""
    got = _measure({"lin0": 1, "lin1": 64})
    assert got["lin0"]["INT8_W8A16"]["n_activation_rows"] == 1
    assert got["lin1"]["INT8_W8A16"]["n_activation_rows"] == 64


def test_row_cap_env_is_respected(monkeypatch):
    monkeypatch.setattr(mqc, "_ACT_ROW_CAP", 8)
    got = _measure({"lin0": 64, "lin1": 3})
    assert got["lin0"]["INT8_W8A16"]["n_activation_rows"] == 8
    assert got["lin1"]["INT8_W8A16"]["n_activation_rows"] == 3


def test_uniform_rows_are_unchanged_by_the_bucketing():
    """Layers whose Linears all share a row count (DSv4-Flash 0-2, all 64)
    must be bit-unchanged: one bucket, one BMM, the old code path's shape."""
    got = _measure({f"lin{i}": 64 for i in range(6)})
    assert {v["INT8_W8A16"]["n_activation_rows"] for v in got.values()} == {64}


def test_measurement_failure_aborts_the_shard(monkeypatch):
    """Fail loud: an exception must not become a silent per-chunk error row."""
    monkeypatch.setattr(mqc, "_COST_FAIL_FAST", True)

    def boom(*a, **kw):
        raise RuntimeError("synthetic OOM")

    monkeypatch.setattr(mqc, "_batched_quantize", boom)
    with pytest.raises(RuntimeError, match="synthetic OOM"):
        _measure({"lin0": 8, "lin1": 8})


def test_measurement_failure_stamps_refusable_rows_when_not_fail_fast(
        monkeypatch):
    """With the escape hatch set, rows carry the flag the merge gate refuses."""
    monkeypatch.setattr(mqc, "_COST_FAIL_FAST", False)

    def boom(*a, **kw):
        raise RuntimeError("synthetic OOM")

    monkeypatch.setattr(mqc, "_batched_quantize", boom)
    got = _measure({"lin0": 8, "lin1": 8})
    for n in ("lin0", "lin1"):
        assert got[n]["INT8_W8A16"]["cost_measurement_failed"] is True
        assert "synthetic OOM" in got[n]["INT8_W8A16"]["error"]


def test_n_activation_rows_takes_the_minimum_across_draws():
    bucket: dict = {}
    mqc._accumulate_result(bucket, "x", "F", 1.0, 1.0, 1.0,
                           n_activation_rows=64)
    mqc._accumulate_result(bucket, "x", "F", 1.0, 1.0, 1.0,
                           n_activation_rows=3)
    out = mqc._finalize_results(bucket)
    assert out["x"]["F"]["n_activation_rows"] == 3
