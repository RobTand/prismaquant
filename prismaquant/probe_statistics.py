"""Shared Fisher rows, marginal reduction, normalization and detail contracts."""
from __future__ import annotations

import numpy as np
import torch


FISHER_MARGINAL_KEYS = (
    "fisher_row", "fisher_col", "g_sq_sum", "act_sq_sum", "act_absmax",
)

FISHER_MAX_KEYS = frozenset({"act_absmax"})

PACKED_FISHER_MARGINAL_KEYS = (
    "expert_g_sq_sum", "expert_act_sq_sum", "expert_act_absmax",
    "expert_tokens")

def merge_marginals(dst: dict, src) -> None:
    """Fold per-channel marginals from `src` into `dst` in place.

    Sums add elementwise; act_absmax merges by MAXIMUM. Used both for
    the per-layer host flush and for the cross-shard partial-stats
    merge, so the two cannot drift apart on the max-vs-sum rule.
    """
    for key in FISHER_MARGINAL_KEYS:
        new = src.get(key)
        if new is None:
            continue
        new = np.asarray(new, dtype=np.float32)
        old = dst.get(key)
        if old is None:
            dst[key] = new.copy()
        elif key in FISHER_MAX_KEYS:
            dst[key] = np.maximum(old, new)
        else:
            dst[key] = old + new

def h_detail_blob(h_raw: torch.Tensor, global_tokens: int, name: str, *,
                  kind: str = "linear",
                  g2_per_token: torch.Tensor | None = None) -> dict:
    """Normalize a token-SUMMED Fisher diagonal accumulator to per-token
    units and wrap it in the canonical h-detail blob schema.

    ``global_tokens`` must be the GLOBAL calibration token count — the
    same denominator `finalize_fisher_stats` applies to the scalar
    ``h_trace`` — never a per-row routed-token count. v4 pins this:
    passing an unpacked expert Linear's own ``n_tokens_seen`` here left
    the detail blob (global/routed)× hotter than the scalar it must
    agree with, so `predicted_dloss` fallback rows priced expert rows on
    a different scale than the rest of the knapsack.

    Every h-detail writer (this module's `FisherAccumulator.finalize` and
    `incremental_probe`'s two writer sites) goes through this helper so
    the on-disk units are per-token everywhere and stamped with an
    explicit ``units: "per_token"`` marker (audit M9 — the incremental
    writer used to save the raw token-summed ``H``, leaving the consumer
    `HDetailIndex.h_diag_from_blob` ~n_tokens× off depending on which
    probe produced the directory). ``g2_per_token`` is already a
    per-token vector and is stored as-is.
    """
    tokens = max(int(global_tokens), 1)
    h = h_raw.detach().to("cpu", torch.float32) / tokens
    blob = {
        "h_diag": h,
        "name": name,
        "kind": kind,
        "shape": list(h.shape),
        "units": "per_token",
        # v3: unit marker; no route_prob term (M4).
        # v4: denominator is the GLOBAL calib token count for every row
        #     (v3 unpacked-expert blobs were per-ROUTED-token).
        "h_detail_version": 4,
        "norm_tokens": tokens,
    }
    if g2_per_token is not None:
        blob["g2_per_token"] = g2_per_token
    return blob

def finalize_fisher_stats(merged_stats: dict, global_tokens: int) -> None:
    """Normalize raw Fisher accumulators into ``h_trace`` (etc.), in place.

    Fisher normalization must share ONE denominator across every row: the
    global calibration token count (nsamples x seqlen). Dense trunk Linears
    accumulate exactly that many tokens in ``n_tokens_seen``, so their
    values are unchanged by dividing by the global count. Per-expert
    Linears, however, only see their ROUTED tokens; a per-row
    ``h_trace_raw / n_tokens_seen`` inflates a rarely-routed expert's
    Fisher by (global/routed) — exactly inverted importance weighting (the
    least-used experts look the most sensitive). Tokens never routed to an
    expert contribute zero gradient, so the empirical Fisher over the
    calibration set divides by the GLOBAL count for every row.

    HISTORY — this deliberately REVERSES a documented convention. Audit M4
    removed an explicit ÷route_prob on the grounds that dividing by the
    routed-token count was "the one implicit ÷token-fraction the MoE
    convention prescribes", and both backends then shipped per-routed-token
    as the single normalization (pinned by the original
    tests/test_packed_expert_per_token_fisher.py). M4's *agreement* goal
    stands — one division, both backends, route_prob as metadata only —
    but per-routed-token was the wrong denominator for the mean-Δloss
    objective the allocator optimizes: it is the same 1/p_e inflation M4
    removed, merely implicit. CLAUDE.md §3 records the same reversal.

    ``n_tokens_seen`` stays raw (routed count, metadata). Every h-detail
    blob is normalized by the SAME global count (`h_detail_blob` v4) so
    scalar and per-weight detail Fisher share one denominator.
    """
    for s in merged_stats.values():
        s["h_trace"] = s.get("h_trace_raw", 0.0) / global_tokens
        s["h_w2_sum"] = s.get("h_w2_sum_raw", 0.0) / global_tokens
        # Per-expert Fisher trace (only present on packed-3D stat entries;
        # dense Linears have no per-expert dimension). Normalize by the
        # same token count so it shares units with `h_trace`.
        per = s.get("h_trace_per_expert_raw")
        if per is not None:
            s["h_trace_per_expert"] = [float(v) / global_tokens for v in per]
        # Per-GROUP Fisher trace (grouped-BMM operands like DSv4's wo_a).
        # Same rule, same denominator: the group axis is a contraction
        # structure ON one logical tensor, and every group sees every
        # calibration token (attention output projection — nothing is
        # routed), so the global count is exact for it, not just
        # consistent.
        per_group = s.get("h_trace_per_group_raw")
        if per_group is not None:
            s["h_trace_per_group"] = [
                float(v) / global_tokens for v in per_group]
        s["h_trace_norm_tokens"] = global_tokens

def renormalize_probe_fisher(stats: dict, meta: dict, *,
                             allow_legacy: bool = False) -> int | None:
    """Recompute h_trace/h_w2_sum/h_trace_per_expert from the stored RAW
    accumulators with the one correct shared denominator: the global calib
    token count.

    Older incremental probes finalized `h_trace = h_trace_raw /
    n_tokens_seen` PER ROW, where per-expert Linears only count their
    ROUTED tokens — inflating a rarely-routed expert's Fisher by
    (global/routed), i.e. inverted importance weighting. The typical
    inflation is ~n_experts/top_k (≈32x on a 256-expert top-8 model);
    the degenerate 1-routed-token case reaches ~33,000x at a 32k-token
    calibration. Dense rows saw exactly the global count, so their values
    are unchanged; probes written by the fixed finalize renormalize to
    identical values — the recompute is idempotent (see PER-ROW STAMPS).

    PER-ROW STAMPS WIN. A row that already carries ``h_trace_norm_tokens``
    was written by a fixed finalize (the field was introduced with it) and
    is already divided by the global count of the pass that PRODUCED it —
    which is not always this probe's meta count. `incremental_probe` merges
    the multimodal visual pass (finalized at ``n_samples x max_text_len``,
    e.g. 8x128) into the body probe, and `merge_probe_pickles` keeps the
    FIRST shard's meta (the body's, e.g. 32x1024), so recomputing every row
    from the meta count would rescale the visual rows by ~32x away from
    what their writer computed — and would not be idempotent. Honouring the
    row stamp keeps each row on its own writer's denominator and makes the
    recompute a true no-op for any probe written by the fixed finalize.

    Returns the probe-wide token count (the meta fallback used for
    unstamped rows), or None when nothing could be renormalized. A row
    with raw accumulators, no stamp, and no usable meta count is a HARD
    ERROR (SystemExit) — silently allocating on routed-token-normalized
    Fisher mis-spends the bit budget; `allow_legacy=True`
    (--allow-legacy-fisher-norm) downgrades it to a warning and keeps the
    probe's stored h_trace values.
    """
    meta = meta or {}
    meta_tokens = int(meta.get("fisher_norm_tokens", 0) or 0)
    if meta_tokens <= 0:
        meta_tokens = (int(meta.get("nsamples", 0) or 0)
                       * int(meta.get("seqlen", 0) or 0))

    renormed = 0
    unresolvable = 0
    denominators: dict[int, int] = {}
    for entry in stats.values():
        if not isinstance(entry, dict) or "h_trace_raw" not in entry:
            continue
        # The row's own writer-stamped denominator, else the probe-wide one.
        row_tokens = int(entry.get("h_trace_norm_tokens", 0) or 0)
        if row_tokens <= 0:
            row_tokens = meta_tokens
        if row_tokens <= 0:
            unresolvable += 1
            continue
        entry["h_trace"] = float(entry["h_trace_raw"]) / row_tokens
        if "h_w2_sum_raw" in entry:
            entry["h_w2_sum"] = (
                float(entry["h_w2_sum_raw"]) / row_tokens)
        per_raw = entry.get("h_trace_per_expert_raw")
        if per_raw is not None:
            entry["h_trace_per_expert"] = [
                float(v) / row_tokens for v in per_raw]
        # Keep the probe-side stamp truthful on legacy pickles.
        entry["h_trace_norm_tokens"] = row_tokens
        denominators[row_tokens] = denominators.get(row_tokens, 0) + 1
        renormed += 1

    if unresolvable:
        msg = (
            "probe rows carry raw Fisher accumulators but no usable token "
            "count: neither a per-row h_trace_norm_tokens stamp nor "
            f"meta.fisher_norm_tokens / nsamples+seqlen ({unresolvable} of "
            f"{len(stats)} rows). h_trace cannot be renormalized by the "
            "global calib token count; legacy per-row normalization inflates "
            "routed-expert rows by (global/routed) — typically "
            "~n_experts/top_k — and the allocator would mis-spend the bit "
            "budget on rarely-routed experts. Re-run the probe with current "
            "code (it stamps meta.fisher_norm_tokens), or pass "
            "--allow-legacy-fisher-norm to proceed with the stored legacy "
            "values anyway.")
        if not allow_legacy:
            raise SystemExit(f"[alloc] ERROR: {msg}")
        print(f"[alloc] WARNING (--allow-legacy-fisher-norm): {msg}",
              flush=True)

    if renormed:
        mix = ", ".join(f"{n}x{cnt} rows"
                        for n, cnt in sorted(denominators.items()))
        print(f"[alloc] Fisher renormalized: {renormed}/{len(stats)} rows "
              f"→ h_trace_raw / global calib tokens [{mix}] "
              "(per-expert routed-count normalization corrected)", flush=True)
        if meta_tokens > 0:
            return meta_tokens
        # Meta-less probe carried entirely by row stamps: report the
        # denominator the most rows share (the body pass').
        return max(denominators.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return None

def h_detail_expected_norm_tokens(probe: dict | None) -> dict[str, int]:
    """Per-row Fisher denominator each h-detail blob is required to carry.

    An h-detail blob and the scalar ``h_trace`` for the SAME row must be
    divided by the SAME token count, or `predicted_dloss` prices that row
    on a different scale than the rest of the knapsack. The blob records
    its denominator as ``norm_tokens`` (schema v4) and the probe stat
    records the scalar's as ``h_trace_norm_tokens``; this builds the
    row -> expected map so `HDetailIndex` can refuse a mismatch.

    Row stamps win over the probe-wide ``meta.fisher_norm_tokens`` so a
    MERGED probe validates per row: the multimodal visual pass is
    finalized at its own (smaller) calibration size, and its blobs are
    stamped to match, so comparing them against the body's count would
    reject correct blobs. Rows with neither a stamp nor a usable meta
    count get no expectation and are left unchecked (nothing to compare).
    """
    if not isinstance(probe, dict):
        return {}
    meta = probe.get("meta") or {}
    fallback = int(meta.get("fisher_norm_tokens", 0) or 0)
    if fallback <= 0:
        fallback = (int(meta.get("nsamples", 0) or 0)
                    * int(meta.get("seqlen", 0) or 0))
    out: dict[str, int] = {}
    for name, entry in (probe.get("stats") or {}).items():
        if not isinstance(entry, dict):
            continue
        n = int(entry.get("h_trace_norm_tokens", 0) or 0) or fallback
        if n > 0:
            out[str(name)] = n
    return out
