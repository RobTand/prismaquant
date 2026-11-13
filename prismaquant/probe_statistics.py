"""Shared Fisher rows, marginal reduction, normalization and detail contracts."""
from __future__ import annotations

import numpy as np
import torch

_U16_MASK = (1 << 16) - 1
_U32_MASK = (1 << 32) - 1
_FMIX32_C1 = 0x85EBCA6B
_FMIX32_C2 = 0xC2B2AE35

# The rule that selects the act/ rows. The key is fixed: the first eight
# BLAKE2b bytes of this name, read as little-endian uint32 k0, k1
# (tests/test_body_fisher_sweep.py derives them again).
ACT_ROW_SELECTION = "blake2b64-fixed-key-fmix32x2-prp-global-row-v1"
_ACT_ROW_KEY = (0x497DDF44, 0xCCAD2C68)


def _mullo32_tensor(values: torch.Tensor, constant: int) -> torch.Tensor:
    """Vectorized low uint32 product with all int64 intermediates < 2**49."""
    low = torch.bitwise_and(values, _U16_MASK)
    high = torch.bitwise_right_shift(values, 16)
    return torch.bitwise_and(
        low * int(constant)
        + torch.bitwise_left_shift(high * int(constant & _U16_MASK), 16),
        _U32_MASK,
    )


def fmix32_tensor(values: torch.Tensor) -> torch.Tensor:
    """Murmur3 fmix32 on int64 tensors: a permutation of the uint32 domain."""
    x = torch.bitwise_and(values, _U32_MASK)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    x = _mullo32_tensor(x, _FMIX32_C1)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 13))
    x = _mullo32_tensor(x, _FMIX32_C2)
    x = torch.bitwise_xor(x, torch.bitwise_right_shift(x, 16))
    return torch.bitwise_and(x, _U32_MASK)


def keep_act_rows(snaps: dict, indices: dict, name: str, flat: torch.Tensor,
                  rows: torch.Tensor, limit: int, *, to_cpu: bool = False) -> None:
    """Keep the ``limit`` act/ rows of ``name`` with the lowest priority.

    ``rows`` are the global row numbers (sample * seqlen + token) of
    ``flat``. The priority is a keyed permutation of the row number, so it
    reads no RNG state, gives the same result on each device, and has no
    ties. Thus the kept set, sorted by row, is the same for each split of
    the rows into hook calls or calibration slices (a top-K merge).
    """
    if snaps.get(name):
        flat = torch.cat([snaps[name][0].to(flat.device), flat])
        rows = torch.cat([indices[name][0].to(rows.device), rows])
    if flat.size(0) > limit:
        k0, k1 = _ACT_ROW_KEY
        priority = fmix32_tensor(
            torch.bitwise_xor(fmix32_tensor(torch.bitwise_xor(rows, k0)), k1))
        pick = torch.topk(priority, int(limit), largest=False).indices
        flat, rows = flat.index_select(0, pick), rows[pick]
    rows, order = torch.sort(rows)
    flat = flat.index_select(0, order)
    snaps[name] = [flat.to("cpu") if to_cpu else flat]
    indices[name] = [rows.to("cpu") if to_cpu else rows]


FISHER_MARGINAL_KEYS = (
    "fisher_row", "fisher_col", "g_sq_sum", "act_sq_sum", "act_absmax",
)

FISHER_MAX_KEYS = frozenset({"act_absmax"})

PACKED_FISHER_MARGINAL_KEYS = (
    "expert_g_sq_sum", "expert_act_sq_sum", "expert_act_absmax",
    "expert_tokens")


def fisher_norm_tokens(row: dict, fallback: int) -> int:
    """Use a writer's positive row stamp before the legacy probe count."""
    stamp = int(row.get("h_trace_norm_tokens", 0) or 0)
    return stamp if stamp > 0 else fallback


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

    ``n_tokens_seen`` stays raw (routed count, metadata).
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
