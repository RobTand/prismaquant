"""Per-channel Fisher marginal emission in the incremental probe.

The load-bearing check is an identity that holds BY CONSTRUCTION:

    sum(fisher_row) == sum(fisher_col) == h_trace_raw

Both marginals are reductions of the same `chunk_h` the probe already
materializes, and `h_trace_raw` on that path is `chunk_h.sum()`. Any
transposed axis, dropped chunk, wrong-shaped accumulator or misplaced
normalization breaks the identity, so it is a cheap total wiring check.

CPU-only and synthetic on purpose: the probe's hot path is GPU, but the
marginal math is dtype/device agnostic and this suite must not contend
for the box.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from prismaquant.incremental_probe import _CONTENT_META_KEYS, _content_meta_compatible, _expected_probe_shard_meta, _marginal_accumulate, _marginal_chunk, _marginal_flush, _marginal_zeros, _marginals_enabled
from prismaquant.probe_statistics import FISHER_MARGINAL_KEYS as _MARGINAL_KEYS, merge_marginals

NAME = "model.layers.0.mlp.down_proj"


def _drive_probe_hook(mod: nn.Linear, batches, *, seed: int = 0) -> dict:
    """Run `mod` forward+backward under a full-backward hook that mirrors
    the phase-3 accumulation site in `incremental_probe` line-for-line:
    same `gy2_sq`/`x2_sq`/`chunk_h` construction, the same
    `_marginal_chunk` + `_marginal_accumulate` calls, the same
    `chunk_h.sum()` trace, and the same one-flush-per-layer drain.

    Returns the finished per-Linear stats entry.
    """
    emit_marginals = _marginals_enabled()
    stats: dict[str, dict] = {
        NAME: {
            "h_trace_raw": 0.0,
            "h_w2_sum_raw": 0.0,
            "in_features": mod.in_features,
            "out_features": mod.out_features,
        }
    }
    if emit_marginals:
        stats[NAME].update(_marginal_zeros(mod.out_features, mod.in_features))

    device_marginals: dict[str, list[torch.Tensor]] = {}
    saved: dict[str, torch.Tensor] = {}

    def fwd(module, inp, out):
        saved["x"] = (inp[0] if isinstance(inp, tuple) else inp).detach()

    def bwd(module, grad_input, grad_output):
        gy = grad_output[0]
        x = saved.pop("x", None)
        if x is None or gy is None:
            return
        gy2 = gy.reshape(-1, gy.size(-1))
        x2 = x.reshape(-1, x.size(-1))
        gy2_sq = gy2.pow(2)
        x2_sq = x2.pow(2)
        chunk_h = (gy2_sq.t() @ x2_sq).float()
        if emit_marginals:
            _marginal_accumulate(
                device_marginals, NAME,
                _marginal_chunk(gy2_sq, x2_sq, x2, chunk_h))
        stats[NAME]["h_trace_raw"] += float(chunk_h.sum().item())
        w = module.weight
        stats[NAME]["h_w2_sum_raw"] += float(
            (chunk_h * w.detach().float().pow(2)).sum().item())

    h1 = mod.register_forward_hook(fwd)
    h2 = mod.register_full_backward_hook(bwd)
    try:
        g = torch.Generator().manual_seed(seed)
        for x in batches:
            out = mod(x)
            # Any scalar with a nonzero cotangent at `out` drives the
            # hook; a fixed random target keeps it deterministic.
            tgt = torch.randn(out.shape, generator=g, dtype=out.dtype)
            (out * tgt).sum().backward()
            mod.zero_grad(set_to_none=True)
    finally:
        h1.remove()
        h2.remove()
    # Single device->host drain, as the per-layer flush does.
    _marginal_flush(device_marginals, stats)
    return stats[NAME]


def _batches(n_batch: int, tokens: int, in_features: int, *, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(1, tokens, in_features, generator=g,
                    dtype=torch.float32, requires_grad=True)
        for _ in range(n_batch)
    ]


# --- A: the marginal identity -----------------------------------------

@pytest.mark.parametrize("n_batch", [1, 3])
def test_marginal_sums_equal_h_trace(monkeypatch, n_batch):
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "1")
    mod = nn.Linear(24, 10, bias=False)
    s = _drive_probe_hook(mod, _batches(n_batch, 16, 24))

    row = s["fisher_row"]
    col = s["fisher_col"]
    assert row.shape == (10,)
    assert col.shape == (24,)
    assert row.dtype == np.float32 and col.dtype == np.float32
    np.testing.assert_allclose(row.sum(), s["h_trace_raw"], rtol=1e-5)
    np.testing.assert_allclose(col.sum(), s["h_trace_raw"], rtol=1e-5)
    np.testing.assert_allclose(row.sum(), col.sum(), rtol=1e-5)


def test_marginal_shapes_finite_nonnegative(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "1")
    mod = nn.Linear(24, 10, bias=False)
    s = _drive_probe_hook(mod, _batches(2, 16, 24))

    expected = {
        "fisher_row": (10,), "fisher_col": (24,),
        "g_sq_sum": (10,), "act_sq_sum": (24,), "act_absmax": (24,),
    }
    for key, shape in expected.items():
        v = s[key]
        assert isinstance(v, np.ndarray), key
        assert v.dtype == np.float32, key
        assert v.shape == shape, key
        assert np.isfinite(v).all(), key
        assert (v >= 0).all(), key


def test_pure_factors_match_direct_reductions(monkeypatch):
    """g_sq_sum / act_sq_sum / act_absmax are the marginals' unmixed
    factors; check them against a direct reduction of the same inputs."""
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "1")
    batches = _batches(2, 16, 24)
    mod = nn.Linear(24, 10, bias=False)
    s = _drive_probe_hook(mod, batches)

    x_all = torch.cat([b.detach().reshape(-1, 24) for b in batches], dim=0)
    np.testing.assert_allclose(
        s["act_sq_sum"], x_all.pow(2).sum(dim=0).numpy(), rtol=1e-5)
    np.testing.assert_allclose(
        s["act_absmax"], x_all.abs().amax(dim=0).numpy(), rtol=1e-6)


def test_identity_survives_a_single_chunk_helper_call():
    """The helper alone, with no hook plumbing: the two marginals of a
    chunk sum to the chunk's own trace."""
    torch.manual_seed(3)
    gy2_sq = torch.rand(11, 7).pow(2)
    x2_sq = torch.rand(11, 5).pow(2)
    x2 = torch.randn(11, 5)
    chunk_h = (gy2_sq.t() @ x2_sq).float()
    row, col, g_sq, act_sq, absmax = _marginal_chunk(
        gy2_sq, x2_sq, x2, chunk_h)

    assert row.shape == (7,) and col.shape == (5,)
    assert g_sq.shape == (7,) and act_sq.shape == (5,)
    assert absmax.shape == (5,)
    torch.testing.assert_close(row.sum(), chunk_h.sum(), rtol=1e-5, atol=0)
    torch.testing.assert_close(col.sum(), chunk_h.sum(), rtol=1e-5, atol=0)
    torch.testing.assert_close(absmax, x2.abs().amax(dim=0))


# --- B: the merge rule ------------------------------------------------

def test_merge_marginals_sums_add_and_absmax_maxes():
    a = {
        "fisher_row": np.array([1.0, 2.0], dtype=np.float32),
        "fisher_col": np.array([3.0], dtype=np.float32),
        "g_sq_sum": np.array([4.0, 5.0], dtype=np.float32),
        "act_sq_sum": np.array([6.0], dtype=np.float32),
        "act_absmax": np.array([9.0], dtype=np.float32),
    }
    b = {
        "fisher_row": np.array([10.0, 20.0], dtype=np.float32),
        "fisher_col": np.array([30.0], dtype=np.float32),
        "g_sq_sum": np.array([40.0, 50.0], dtype=np.float32),
        "act_sq_sum": np.array([60.0], dtype=np.float32),
        "act_absmax": np.array([2.0], dtype=np.float32),
    }
    merge_marginals(a, b)
    np.testing.assert_allclose(a["fisher_row"], [11.0, 22.0])
    np.testing.assert_allclose(a["fisher_col"], [33.0])
    np.testing.assert_allclose(a["g_sq_sum"], [44.0, 55.0])
    np.testing.assert_allclose(a["act_sq_sum"], [66.0])
    # 9 > 2: a bound merges by MAXIMUM, never by sum (11.0 would be the
    # bug this asserts against).
    np.testing.assert_allclose(a["act_absmax"], [9.0])

    c = {"act_absmax": np.array([1.0], dtype=np.float32)}
    merge_marginals(c, {"act_absmax": np.array([5.0], dtype=np.float32)})
    np.testing.assert_allclose(c["act_absmax"], [5.0])


def test_merge_marginals_asymmetric_presence():
    """A partial stats entry may predate the flag, or carry keys the
    destination lacks; neither side may raise or silently drop."""
    dst: dict = {}
    src = {"fisher_row": np.array([1.0, 2.0], dtype=np.float32)}
    merge_marginals(dst, src)
    np.testing.assert_allclose(dst["fisher_row"], [1.0, 2.0])
    assert "fisher_col" not in dst
    # Copied, not aliased: mutating the source must not move `dst`.
    src["fisher_row"][0] = 99.0
    np.testing.assert_allclose(dst["fisher_row"], [1.0, 2.0])

    before = dict(dst)
    merge_marginals(dst, {})
    np.testing.assert_allclose(dst["fisher_row"], before["fisher_row"])


def test_identity_survives_the_cross_shard_merge(monkeypatch):
    """Two shards' partial stats merged: the identity must still hold on
    the merged entry, which is what `merge_marginals` at the shard-merge
    site is responsible for."""
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "1")
    mod = nn.Linear(20, 9, bias=False)
    part1 = _drive_probe_hook(mod, _batches(2, 12, 20, seed=1), seed=1)
    part2 = _drive_probe_hook(mod, _batches(2, 12, 20, seed=2), seed=2)

    merged = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in part1.items()}
    merge_marginals(merged, part2)
    h_trace = part1["h_trace_raw"] + part2["h_trace_raw"]

    np.testing.assert_allclose(
        merged["fisher_row"].sum(), h_trace, rtol=1e-5)
    np.testing.assert_allclose(
        merged["fisher_col"].sum(), h_trace, rtol=1e-5)
    np.testing.assert_allclose(
        merged["act_absmax"],
        np.maximum(part1["act_absmax"], part2["act_absmax"]))


# --- C: the flag is a true no-op --------------------------------------

def test_flag_off_is_byte_identical_and_emits_nothing(monkeypatch):
    batches = _batches(3, 16, 24)

    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "1")
    mod_on = nn.Linear(24, 10, bias=False)
    on = _drive_probe_hook(mod_on, batches)

    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "0")
    mod_off = nn.Linear(24, 10, bias=False)
    mod_off.load_state_dict(mod_on.state_dict())
    off = _drive_probe_hook(mod_off, batches)

    # Bit-identical, not merely close: the marginal reductions are
    # separate ops and must not perturb the existing accumulators.
    assert off["h_trace_raw"] == on["h_trace_raw"]
    assert off["h_w2_sum_raw"] == on["h_w2_sum_raw"]
    for key in _MARGINAL_KEYS:
        assert key not in off, key
        assert key in on, key


def test_flag_default_is_on(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_PROBE_MARGINALS", raising=False)
    assert _marginals_enabled() is True
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "0")
    assert _marginals_enabled() is False


def test_marginal_zeros_are_merge_identities():
    z = _marginal_zeros(4, 6)
    assert set(z) == set(_MARGINAL_KEYS)
    assert z["fisher_row"].shape == (4,) and z["fisher_col"].shape == (6,)
    real = {
        "fisher_row": np.arange(4, dtype=np.float32),
        "fisher_col": np.arange(6, dtype=np.float32),
        "g_sq_sum": np.arange(4, dtype=np.float32),
        "act_sq_sum": np.arange(6, dtype=np.float32),
        "act_absmax": np.arange(6, dtype=np.float32),
    }
    merge_marginals(z, real)
    for key, want in real.items():
        np.testing.assert_allclose(z[key], want, err_msg=key)


def test_flush_uses_one_transfer_and_accumulates_across_calls():
    """`_marginal_flush` drains and clears; a second flush for the same
    Linear must fold into the existing arrays, not replace them."""
    slot: dict[str, list[torch.Tensor]] = {}
    stats = {NAME: dict(_marginal_zeros(2, 3))}
    vecs = [
        torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]),
        torch.tensor([6.0, 7.0]), torch.tensor([8.0, 9.0, 10.0]),
        torch.tensor([2.0, 1.0, 4.0]),
    ]
    _marginal_accumulate(slot, NAME, vecs)
    _marginal_flush(slot, stats)
    assert slot == {}
    np.testing.assert_allclose(stats[NAME]["fisher_row"], [1.0, 2.0])

    _marginal_accumulate(slot, NAME, vecs)
    _marginal_flush(slot, stats)
    np.testing.assert_allclose(stats[NAME]["fisher_row"], [2.0, 4.0])
    np.testing.assert_allclose(stats[NAME]["fisher_col"], [6.0, 8.0, 10.0])
    # absmax across two identical flushes stays the bound, not 2x.
    np.testing.assert_allclose(stats[NAME]["act_absmax"], [2.0, 1.0, 4.0])


# --- resume gating: a flag-off shard must not be reused flag-on -------

class _Args:
    """Minimal stand-in for the argparse namespace the shard-meta
    builder reads."""
    model = "/models/qwen3-0p6b"
    dataset = "ultrachat_200k"
    nsamples = 8
    seqlen = 512
    dtype = "bf16"
    device = "cuda"
    device_map = None
    importance_weighting = True
    h_detail_dir = None
    activation_rows_limit = 256


def _shard_meta():
    return _expected_probe_shard_meta(
        _Args(), linear_include="re:.*", shard_idx=0,
        activation_cache_dir="/work/act")


def test_shard_meta_carries_the_marginal_flag(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "1")
    assert _shard_meta()["emit_marginals"] is True
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "0")
    assert _shard_meta()["emit_marginals"] is False


def test_flag_off_shard_is_not_poolable_into_a_flag_on_run(monkeypatch):
    """Marginal emission decides WHICH KEYS a stats entry carries, so it
    is a content axis, not a grouping axis: the LPS-invariant linear
    cache must refuse to pool across it."""
    assert "emit_marginals" in _CONTENT_META_KEYS

    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "0")
    off_shard_meta = {"incremental_shard": _shard_meta()}
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "1")
    on_anchor = _shard_meta()

    assert not _content_meta_compatible(off_shard_meta, on_anchor)
    assert _content_meta_compatible(
        {"incremental_shard": _shard_meta()}, on_anchor)


def test_pre_flag_shard_is_not_poolable(monkeypatch):
    """A shard written before this key existed has no `emit_marginals`
    at all; `None != True` must rebuild loudly rather than silently
    contribute marginal-less entries."""
    monkeypatch.setenv("PRISMAQUANT_PROBE_MARGINALS", "1")
    anchor = _shard_meta()
    legacy = {k: v for k, v in anchor.items() if k != "emit_marginals"}
    assert not _content_meta_compatible({"incremental_shard": legacy}, anchor)


def test_accumulate_maxes_absmax_within_a_layer():
    slot: dict[str, list[torch.Tensor]] = {}
    _marginal_accumulate(slot, NAME, [
        torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]),
        torch.tensor([1.0]), torch.tensor([5.0]),
    ])
    _marginal_accumulate(slot, NAME, [
        torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]),
        torch.tensor([1.0]), torch.tensor([3.0]),
    ])
    assert float(slot[NAME][0]) == 2.0          # fisher_row sums
    assert float(slot[NAME][4]) == 5.0          # act_absmax maxes
