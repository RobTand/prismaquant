"""Fisher h_trace must be normalized by the GLOBAL calib token count.

Synthetic reproducer: two identical Linears with identical gradient
statistics over the same calibration run. The "dense" one sees every token
(most carrying zero gradient); the "expert" one only sees the tokens routed
to it — the very tokens that carry the gradient. Tokens never routed to an
expert contribute zero gradient, so both rows have the SAME empirical
Fisher. The old per-row `h_trace_raw / n_tokens_seen` divided the expert by
its routed count only, inflating it by (global/routed).

Also pins:
  - scalar/detail denominator agreement: the h-detail blob (which feeds
    `predicted_dloss` via measure_quant_cost) and the scalar `h_trace`
    must share the one global-token denominator — sum(h_diag) == h_trace;
  - the sensitivity backend (`FisherAccumulator.finalize`) applies the
    same convention end-to-end, scalar and blob;
  - the allocator's load-time renormalization: recompute from raw
    accumulators when meta carries the token count, hard-fail when it
    does not (escape hatch: --allow-legacy-fisher-norm);
  - per-row denominator stamps win over the probe-wide meta count, so a
    MERGED body+visual probe keeps each pass on its own writer's
    denominator (the visual pass is finalized at its own calibration
    size, and merge_probe_pickles keeps the BODY's meta);
  - the h-detail units GATE: a blob whose recorded norm_tokens does not
    match its row's scalar denominator (a v3 per-routed-token blob, or a
    dir written at a different calibration size) is refused, not
    silently mixed into the same knapsack.
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from prismaquant.allocator import clip_probe_fisher_outliers
from prismaquant.probe_statistics import renormalize_probe_fisher
from prismaquant.probe_statistics import finalize_fisher_stats
from prismaquant.measure_quant_cost import HDetailIndex
from prismaquant.probe_statistics import h_detail_expected_norm_tokens
from prismaquant.sensitivity_probe import FisherAccumulator
from prismaquant.probe_statistics import h_detail_blob


def _h_trace_raw(x: torch.Tensor, gy: torch.Tensor) -> float:
    """The probe's per-token trace accumulator: sum_t ||gy_t||^2 * ||x_t||^2."""
    return float((gy.pow(2).sum(dim=1) * x.pow(2).sum(dim=1)).sum().item())


def test_h_trace_equal_for_equal_gradient_mass_after_global_normalization():
    torch.manual_seed(0)
    global_tokens, routed_tokens = 32, 4  # the expert is routed 1/8 of tokens
    dense = nn.Linear(16, 8, bias=False)
    expert = nn.Linear(16, 8, bias=False)
    expert.load_state_dict(dense.state_dict())

    x = torch.randn(global_tokens, 16)
    # Identical per-token gradients on the routed tokens; zero elsewhere
    # (a token never routed to the expert contributes zero gradient).
    gy = torch.zeros(global_tokens, 8)
    gy[:routed_tokens] = torch.randn(routed_tokens, 8)

    y_dense = dense(x)                      # dense row: all 32 tokens
    y_dense.backward(gy)
    y_expert = expert(x[:routed_tokens])    # expert row: its 4 routed tokens
    y_expert.backward(gy[:routed_tokens])

    stats = {
        "dense": {"h_trace_raw": _h_trace_raw(x, gy),
                  "h_w2_sum_raw": 0.0, "n_tokens_seen": global_tokens},
        "expert": {"h_trace_raw": _h_trace_raw(x[:routed_tokens],
                                               gy[:routed_tokens]),
                   "h_w2_sum_raw": 0.0, "n_tokens_seen": routed_tokens},
    }
    # Same gradient mass -> same raw accumulator (and same weight grads).
    assert stats["expert"]["h_trace_raw"] == pytest.approx(
        stats["dense"]["h_trace_raw"])
    assert torch.allclose(dense.weight.grad, expert.weight.grad)

    # OLD normalization (per-row n_tokens_seen), computed inline for
    # contrast: the expert looks (global/routed) = 8x more sensitive.
    old = {n: s["h_trace_raw"] / s["n_tokens_seen"] for n, s in stats.items()}
    assert old["expert"] == pytest.approx(
        old["dense"] * global_tokens / routed_tokens)

    finalize_fisher_stats(stats, global_tokens)
    assert stats["expert"]["h_trace"] == pytest.approx(stats["dense"]["h_trace"])
    assert stats["dense"]["h_trace"] == pytest.approx(
        stats["dense"]["h_trace_raw"] / global_tokens)
    # n_tokens_seen stays raw (routed count) for the h_detail consumers.
    assert stats["expert"]["n_tokens_seen"] == routed_tokens
    assert stats["expert"]["h_trace_norm_tokens"] == global_tokens


def test_rows_without_token_counter_use_global_count_not_one():
    """Stat rows filled by the batched MoE-block flush path never increment
    n_tokens_seen; the old finalize divided them by max(0, 1) == 1."""
    stats = {"expert": {"h_trace_raw": 64.0, "h_w2_sum_raw": 0.0,
                        "n_tokens_seen": 0}}
    finalize_fisher_stats(stats, 32)
    assert stats["expert"]["h_trace"] == pytest.approx(2.0)


def test_scalar_and_h_detail_share_the_global_denominator():
    """The h-detail blob (predicted_dloss fallback input) and the scalar
    h_trace must be divided by the SAME global token count. Identity:
    h_diag[i,j] = sum_t gy_ti^2 x_tj^2 / N  =>  sum(h_diag) == h_trace.
    Dividing the expert blob by its routed count instead (the pre-fix
    writer behavior) leaves the detail (global/routed)x hotter than the
    scalar — the mixed-units error, relocated into the knapsack."""
    torch.manual_seed(1)
    global_tokens, routed_tokens = 32, 4
    x_r = torch.randn(routed_tokens, 16)
    gy_r = torch.randn(routed_tokens, 8)

    # Expert row: raw accumulators over its routed tokens only.
    h_full_raw = gy_r.pow(2).t() @ x_r.pow(2)       # token-summed [out, in]
    stats = {"expert": {"h_trace_raw": _h_trace_raw(x_r, gy_r),
                        "h_w2_sum_raw": 0.0, "n_tokens_seen": routed_tokens}}
    finalize_fisher_stats(stats, global_tokens)

    blob = h_detail_blob(h_full_raw, global_tokens, "expert", kind="linear")
    h_diag = HDetailIndex.h_diag_from_blob(blob)
    assert blob["norm_tokens"] == global_tokens
    assert float(h_diag.sum()) == pytest.approx(
        stats["expert"]["h_trace"], rel=1e-5)

    # Contrast: a per-routed-token blob disagrees with the scalar by
    # exactly (global/routed).
    stale = HDetailIndex.h_diag_from_blob(
        h_detail_blob(h_full_raw, routed_tokens, "expert", kind="linear"))
    assert float(stale.sum()) == pytest.approx(
        stats["expert"]["h_trace"] * global_tokens / routed_tokens, rel=1e-5)


def test_sensitivity_backend_scalar_and_blob_agree_end_to_end():
    """FisherAccumulator.finalize: an unpacked expert Linear routed a
    fraction of the calib tokens lands at the same h_trace as a dense
    twin with identical gradient mass, and its h-detail blob sums to the
    scalar (shared global denominator), while n_tokens_seen stays raw."""
    torch.manual_seed(2)
    global_tokens, routed_tokens = 24, 3
    model = nn.Module()
    model.dense = nn.Linear(6, 5, bias=False)
    model.expert = nn.Linear(6, 5, bias=False)
    model.expert.load_state_dict(model.dense.state_dict())
    for p in model.parameters():
        p.requires_grad_(False)

    x = torch.randn(global_tokens, 6)
    v = torch.zeros(global_tokens, 5)
    v[:routed_tokens] = torch.randn(routed_tokens, 5)

    with tempfile.TemporaryDirectory() as td:
        h_dir = Path(td) / "h"
        acc = FisherAccumulator(model, ["dense", "expert"], {},
                                h_detail_dir=h_dir)
        xg = x.detach().requires_grad_(True)
        xr = x[:routed_tokens].detach().requires_grad_(True)
        loss = (model.dense(xg) * v).sum() + \
            (model.expert(xr) * v[:routed_tokens]).sum()
        loss.backward()
        acc.finalize(None, global_tokens=global_tokens)
        acc.remove_hooks()

        d, e = acc.stats["dense"], acc.stats["expert"]
        assert d["n_tokens_seen"] == global_tokens
        assert e["n_tokens_seen"] == routed_tokens
        assert e["h_trace"] == pytest.approx(d["h_trace"], rel=1e-4)
        assert e["h_trace_norm_tokens"] == global_tokens

        index = HDetailIndex(h_dir, ["dense", "expert"])
        for name, s in (("dense", d), ("expert", e)):
            blob = index.load_blob(name)
            assert blob["h_detail_version"] == 4
            assert blob["norm_tokens"] == global_tokens
            assert float(blob["h_diag"].sum()) == pytest.approx(
                s["h_trace"], rel=1e-4)

def _legacy_probe_stats():
    """A legacy-finalized probe: expert row divided by its routed count."""
    return {
        "dense": {"h_trace_raw": 64.0, "h_w2_sum_raw": 8.0,
                  "h_trace": 2.0, "h_w2_sum": 0.25, "n_tokens_seen": 32},
        "expert": {"h_trace_raw": 64.0, "h_w2_sum_raw": 8.0,
                   "h_trace": 16.0, "h_w2_sum": 2.0, "n_tokens_seen": 4},
    }


def test_allocator_renormalizes_from_meta_and_is_idempotent():
    stats = _legacy_probe_stats()
    assert renormalize_probe_fisher(
        stats, {"nsamples": 4, "seqlen": 8}) == 32
    assert stats["expert"]["h_trace"] == pytest.approx(2.0)
    assert stats["expert"]["h_w2_sum"] == pytest.approx(0.25)
    assert stats["dense"]["h_trace"] == pytest.approx(2.0)
    assert stats["expert"]["h_trace_norm_tokens"] == 32
    # fisher_norm_tokens (stamped by the fixed finalize) wins over
    # nsamples x seqlen, and re-running changes nothing.
    assert renormalize_probe_fisher(
        stats, {"fisher_norm_tokens": 32, "nsamples": 999, "seqlen": 999}) == 32
    assert stats["expert"]["h_trace"] == pytest.approx(2.0)


def test_allocator_hard_fails_on_probe_without_token_meta():
    stats = _legacy_probe_stats()
    with pytest.raises(SystemExit, match="allow-legacy-fisher-norm"):
        renormalize_probe_fisher(stats, {})
    # Values untouched by the failed attempt.
    assert stats["expert"]["h_trace"] == pytest.approx(16.0)


def test_allocator_legacy_escape_hatch_keeps_stored_values():
    stats = _legacy_probe_stats()
    assert renormalize_probe_fisher(stats, {}, allow_legacy=True) is None
    assert stats["expert"]["h_trace"] == pytest.approx(16.0)
    assert stats["dense"]["h_trace"] == pytest.approx(2.0)


def test_allocator_no_raw_rows_is_silent_no_op():
    """Probes without raw accumulators (nothing to renormalize, nothing to
    detect) must not trip the hard fail."""
    stats = {"dense": {"h_trace": 2.0, "n_tokens_seen": 32}}
    assert renormalize_probe_fisher(stats, {}) is None
    assert stats["dense"]["h_trace"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Merged body+visual probe: the visual pass is finalized at its OWN global
# token count, and merge_probe_pickles keeps the FIRST (body) shard's meta.
# Renormalizing every row from the body's count would rescale visual rows by
# body_tokens/visual_tokens (32x under pipeline defaults: 32x1024 vs 8x128).
# ---------------------------------------------------------------------------
_BODY_TOKENS = 32 * 1024
_VISUAL_TOKENS = 8 * 128


def _merged_body_visual_probe():
    """What incremental_probe writes: body rows finalized at the body's
    global count, visual rows at the visual pass', merged into one stats
    dict under the BODY's meta."""
    body = {"model.layers.0.self_attn.q_proj": {
        "h_trace_raw": 64.0, "h_w2_sum_raw": 8.0,
        "n_tokens_seen": _BODY_TOKENS}}
    visual = {"visual.blocks.0.attn.qkv": {
        "h_trace_raw": 64.0, "h_w2_sum_raw": 8.0,
        "n_tokens_seen": _VISUAL_TOKENS}}
    finalize_fisher_stats(body, _BODY_TOKENS)
    finalize_fisher_stats(visual, _VISUAL_TOKENS)
    return {**body, **visual}, {"fisher_norm_tokens": _BODY_TOKENS,
                                "nsamples": 32, "seqlen": 1024}


def test_merged_visual_rows_keep_their_own_writer_denominator():
    stats, meta = _merged_body_visual_probe()
    body_written = stats["model.layers.0.self_attn.q_proj"]["h_trace"]
    visual_written = stats["visual.blocks.0.attn.qkv"]["h_trace"]
    # The writer's values: same raw mass, different denominators.
    assert visual_written == pytest.approx(64.0 / _VISUAL_TOKENS)
    assert visual_written == pytest.approx(
        body_written * _BODY_TOKENS / _VISUAL_TOKENS)

    assert renormalize_probe_fisher(stats, meta) == _BODY_TOKENS
    # Both rows come out exactly as their writer computed them.
    assert stats["model.layers.0.self_attn.q_proj"]["h_trace"] == \
        pytest.approx(body_written)
    assert stats["visual.blocks.0.attn.qkv"]["h_trace"] == \
        pytest.approx(visual_written)
    assert stats["visual.blocks.0.attn.qkv"]["h_trace_norm_tokens"] == \
        _VISUAL_TOKENS
    assert stats["visual.blocks.0.attn.qkv"]["h_w2_sum"] == \
        pytest.approx(8.0 / _VISUAL_TOKENS)


def test_merged_probe_renormalization_is_idempotent():
    stats, meta = _merged_body_visual_probe()
    renormalize_probe_fisher(stats, meta)
    once = {n: (s["h_trace"], s["h_w2_sum"]) for n, s in stats.items()}
    renormalize_probe_fisher(stats, meta)
    renormalize_probe_fisher(stats, meta)
    assert {n: (s["h_trace"], s["h_w2_sum"])
            for n, s in stats.items()} == once


def test_row_stamp_carries_a_probe_whose_meta_lost_the_token_count():
    """Row stamps alone are enough — no hard fail, no rescale."""
    stats, _ = _merged_body_visual_probe()
    before = {n: s["h_trace"] for n, s in stats.items()}
    # The body denominator is the one most rows share, so it is reported.
    assert renormalize_probe_fisher(stats, {}) == _BODY_TOKENS
    assert {n: s["h_trace"] for n, s in stats.items()} == before


def test_unstamped_rows_still_hard_fail_when_meta_is_empty():
    """The gate must survive the per-row stamp path: a row with raw
    accumulators, no stamp and no meta count is still fatal."""
    stats, _ = _merged_body_visual_probe()
    stats["legacy.row"] = {"h_trace_raw": 64.0, "h_trace": 16.0,
                           "n_tokens_seen": 4}
    with pytest.raises(SystemExit, match="allow-legacy-fisher-norm"):
        renormalize_probe_fisher(stats, {})
    assert stats["legacy.row"]["h_trace"] == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# h-detail units gate (HDetailIndex): a blob must be on its row's scalar
# denominator. Same hard-refusal idiom as prepare_cost_context's
# packed_fisher_estimator gate.
# ---------------------------------------------------------------------------
def _write_blob(h_dir: Path, name: str, norm_tokens: int, *,
                version: int = 4) -> None:
    h_dir.mkdir(parents=True, exist_ok=True)
    blob = h_detail_blob(torch.ones(2, 3), norm_tokens, name, kind="linear")
    if version < 4:                      # emulate the pre-v4 writer
        blob.pop("norm_tokens")
        blob["h_detail_version"] = version
    torch.save(blob, h_dir / (HDetailIndex._FNAME_SUB.sub("__", name) + ".pt"))


def _probe_for(stats: dict, meta: dict) -> dict:
    return {"stats": stats, "meta": meta}


def test_h_detail_expected_norm_tokens_prefers_row_stamps():
    stats, meta = _merged_body_visual_probe()
    exp = h_detail_expected_norm_tokens(_probe_for(stats, meta))
    assert exp["model.layers.0.self_attn.q_proj"] == _BODY_TOKENS
    assert exp["visual.blocks.0.attn.qkv"] == _VISUAL_TOKENS
    # Unstamped rows fall back to the probe-wide meta count.
    exp2 = h_detail_expected_norm_tokens(
        _probe_for({"a": {"h_trace_raw": 1.0}}, {"nsamples": 4, "seqlen": 8}))
    assert exp2["a"] == 32
    # No probe / no token info -> no expectations -> gate inert.
    assert h_detail_expected_norm_tokens(None) == {}
    assert h_detail_expected_norm_tokens(_probe_for({"a": {}}, {})) == {}


def test_h_detail_index_accepts_matching_norm_tokens(tmp_path):
    stats, meta = _merged_body_visual_probe()
    names = list(stats)
    for n in names:
        _write_blob(tmp_path, n, stats[n]["h_trace_norm_tokens"])
    index = HDetailIndex(
        tmp_path, names,
        expected_norm_tokens=h_detail_expected_norm_tokens(
            _probe_for(stats, meta)))
    assert len(index) == 2
    for n in names:
        assert index.load(n).shape == (2, 3)
        assert index.load_blob(n)["norm_tokens"] == \
            stats[n]["h_trace_norm_tokens"]


def test_h_detail_index_refuses_v3_blob(tmp_path):
    """A v3 blob has no norm_tokens: it was divided by the row's OWN token
    count, per-ROUTED-token on unpacked expert rows."""
    stats, meta = _merged_body_visual_probe()
    name = "model.layers.0.self_attn.q_proj"
    _write_blob(tmp_path, name, _BODY_TOKENS, version=3)
    with pytest.raises(SystemExit) as ei:
        HDetailIndex(tmp_path, [name],
                     expected_norm_tokens=h_detail_expected_norm_tokens(
                         _probe_for(stats, meta)))
    msg = str(ei.value)
    assert str(tmp_path) in msg          # names the blob dir
    assert "no norm_tokens stamp" in msg
    assert "per-ROUTED-token" in msg     # says why the units are wrong
    assert "Regenerate" in msg           # tells the operator what to do


def test_h_detail_index_refuses_wrong_calibration_size(tmp_path):
    """A v4 blob written at a different calibration size is caught too —
    the reason to compare norm_tokens rather than the version integer."""
    stats, meta = _merged_body_visual_probe()
    name = "model.layers.0.self_attn.q_proj"
    _write_blob(tmp_path, name, _BODY_TOKENS // 2)
    with pytest.raises(SystemExit, match="different calibration size"):
        HDetailIndex(tmp_path, [name],
                     expected_norm_tokens=h_detail_expected_norm_tokens(
                         _probe_for(stats, meta)))


def test_h_detail_index_refuses_visual_blob_on_the_body_denominator(tmp_path):
    """The merged-probe case the gate must get right in BOTH directions: a
    visual blob stamped with the BODY's count disagrees with its row's
    scalar and is refused, while the correctly-stamped one is accepted."""
    stats, meta = _merged_body_visual_probe()
    name = "visual.blocks.0.attn.qkv"
    expected = h_detail_expected_norm_tokens(_probe_for(stats, meta))
    _write_blob(tmp_path, name, _BODY_TOKENS)
    with pytest.raises(SystemExit, match="different calibration size"):
        HDetailIndex(tmp_path, [name], expected_norm_tokens=expected)
    _write_blob(tmp_path, name, _VISUAL_TOKENS)
    assert len(HDetailIndex(tmp_path, [name],
                            expected_norm_tokens=expected)) == 1


def test_h_detail_gate_is_off_without_expectations(tmp_path):
    """Archived/diagnostic readers that pass no probe keep working."""
    name = "model.layers.0.self_attn.q_proj"
    _write_blob(tmp_path, name, _BODY_TOKENS, version=3)
    index = HDetailIndex(tmp_path, [name])
    assert index.load(name).shape == (2, 3)


def test_h_detail_gate_has_no_env_override_and_names_the_safe_escape():
    """Unlike the packed_fisher_estimator gate there is deliberately no env
    override: h-detail is optional, so the safe escape is to drop
    --h-detail-dir (scalar-proxy fallback), not to admit wrong units."""
    src = Path(HDetailIndex.__module__.replace(".", "/") + ".py")
    text = (Path(__file__).resolve().parents[1] / src).read_text()
    assert "ALLOW_STALE_H_DETAIL" not in text
    assert "drop --h-detail-dir to fall back to the scalar proxy" in text


def test_h_detail_gate_fires_on_every_read_not_just_construction(tmp_path):
    """Construction checks one blob; a dir that goes stale per-row must
    still be caught when that row is read."""
    stats, meta = _merged_body_visual_probe()
    names = list(stats)
    expected = h_detail_expected_norm_tokens(_probe_for(stats, meta))
    for n in names:
        _write_blob(tmp_path, n, stats[n]["h_trace_norm_tokens"])
    index = HDetailIndex(tmp_path, names, expected_norm_tokens=expected)
    # Corrupt the second row's blob after the index was built.
    _write_blob(tmp_path, names[1], 7)
    with pytest.raises(SystemExit, match="different calibration size"):
        index.load(names[1])
    with pytest.raises(SystemExit, match="different calibration size"):
        index.load_blob(names[1])
    assert index.load(names[0]).shape == (2, 3)   # untouched row still fine


# ---------------------------------------------------------------------------
# Robust Fisher clip (opt-in research lever, PRISMAQUANT_FISHER_CAP_MULTIPLIER).
# Default OFF must be a byte-identical no-op; ON must apply the reference
# tool's K x per-role-median cap with the reference tool's role grouping
# (dense `layers.<N>.<container>.<role>` only — experts are NOT bucketed).
# ---------------------------------------------------------------------------

def _clip_probe_stats():
    """Three q_proj rows (median 1.0) with one 10x outlier, three down_proj
    rows (median 100.0) with one 10x outlier, plus rows the tool's role
    regex deliberately does not match."""
    def row(h):
        return {"h_trace": h, "h_w2_sum": h * 2.0, "h_trace_raw": h * 32.0}
    return {
        "model.layers.0.self_attn.q_proj": row(0.5),
        "model.layers.1.self_attn.q_proj": row(1.0),
        "model.layers.2.self_attn.q_proj": row(10.0),
        "model.layers.0.mlp.down_proj": row(50.0),
        "model.layers.1.mlp.down_proj": row(100.0),
        "model.layers.2.mlp.down_proj": row(1000.0),
        # Not matched by the reference role regex (extra container segment):
        "model.layers.0.mlp.experts.gate_proj": row(9999.0),
        "model.layers.0.mlp.experts.3.up_proj": row(9999.0),
        "lm_head": row(9999.0),
    }


def test_fisher_clip_unset_is_byte_identical_no_op(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_FISHER_CAP_MULTIPLIER", raising=False)
    stats = _clip_probe_stats()
    before = copy.deepcopy(stats)
    meta = {}
    assert clip_probe_fisher_outliers(stats, meta) is None
    assert stats == before
    assert meta == {}          # no provenance stamp when inactive


def test_fisher_clip_empty_env_value_is_also_off(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_FISHER_CAP_MULTIPLIER", "  ")
    stats = _clip_probe_stats()
    before = copy.deepcopy(stats)
    assert clip_probe_fisher_outliers(stats, {}) is None
    assert stats == before


def test_fisher_clip_applies_per_role_k_times_median(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_FISHER_CAP_MULTIPLIER", "3")
    stats = _clip_probe_stats()
    meta = {}
    info = clip_probe_fisher_outliers(stats, meta)

    # Per-role medians, computed independently per bucket.
    assert info["role_median"]["q_proj"] == pytest.approx(1.0)
    assert info["role_median"]["down_proj"] == pytest.approx(100.0)
    assert info["role_cap"]["q_proj"] == pytest.approx(3.0)
    assert info["role_cap"]["down_proj"] == pytest.approx(300.0)
    assert info["cap_multiplier"] == pytest.approx(3.0)

    # Only the outliers move; sub-cap rows are untouched.
    assert stats["model.layers.2.self_attn.q_proj"]["h_trace"] == pytest.approx(3.0)
    assert stats["model.layers.2.mlp.down_proj"]["h_trace"] == pytest.approx(300.0)
    assert stats["model.layers.0.self_attn.q_proj"]["h_trace"] == pytest.approx(0.5)
    assert stats["model.layers.1.self_attn.q_proj"]["h_trace"] == pytest.approx(1.0)
    assert stats["model.layers.0.mlp.down_proj"]["h_trace"] == pytest.approx(50.0)
    assert stats["model.layers.1.mlp.down_proj"]["h_trace"] == pytest.approx(100.0)

    # A high down_proj row is NOT clipped against the q_proj median: the
    # buckets are what make this robust rather than a global truncation.
    assert stats["model.layers.0.mlp.down_proj"]["h_trace"] > info["role_cap"]["q_proj"]

    # h_w2_sum rescales by the same ratio; the raw accumulator does not move.
    assert stats["model.layers.2.self_attn.q_proj"]["h_w2_sum"] == pytest.approx(
        20.0 * (3.0 / 10.0))
    assert stats["model.layers.2.self_attn.q_proj"]["h_trace_raw"] == pytest.approx(320.0)

    assert info["n_clipped"] == 2
    assert info["n_considered"] == 6
    assert meta["clip_history"][-1]["schema"] == "prismaquant.robust_fisher_clip.v1"


def test_fisher_clip_skips_rows_outside_the_tool_role_grouping(monkeypatch):
    """Experts / sidecars / bare names are not bucketed at all — the 4B
    result was measured on the dense grouping only."""
    monkeypatch.setenv("PRISMAQUANT_FISHER_CAP_MULTIPLIER", "1.0")
    stats = _clip_probe_stats()
    info = clip_probe_fisher_outliers(stats, {})
    for name in ("model.layers.0.mlp.experts.gate_proj",
                 "model.layers.0.mlp.experts.3.up_proj",
                 "lm_head"):
        assert stats[name]["h_trace"] == pytest.approx(9999.0)
    assert set(info["role_median"]) == {"q_proj", "down_proj"}


def test_fisher_clip_rejects_nonsense_multipliers(monkeypatch):
    for bad in ("0", "-2", "nope", "nan"):
        monkeypatch.setenv("PRISMAQUANT_FISHER_CAP_MULTIPLIER", bad)
        with pytest.raises(SystemExit, match="FISHER_CAP_MULTIPLIER"):
            clip_probe_fisher_outliers(_clip_probe_stats(), {})


def test_fisher_clip_explicit_argument_overrides_unset_env(monkeypatch):
    monkeypatch.delenv("PRISMAQUANT_FISHER_CAP_MULTIPLIER", raising=False)
    stats = _clip_probe_stats()
    info = clip_probe_fisher_outliers(stats, {}, cap_multiplier=3.0)
    assert info["n_clipped"] == 2
