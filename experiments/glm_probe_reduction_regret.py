#!/usr/bin/env python
"""Offline, CPU-only allocation-regret study for GLM-5.3-Flash Tessera anchor
schedules (workstream W2, 2026-09-10).

Question. The census campaign encodes every routed expert of every layer at
three E4M3 rates (832/960/1088 q256), yet the allocator decides ONE rate per
packed expert stack (``allocator_candidates.aggregate_packed_serving_groups``
makes a packed group one multi-choice-knapsack item).  How much allocation
regret does a cheaper anchor schedule cost, scored against the census?

Data.  The completed ``cost.pkl`` payloads under the census workspace (read
only; measured cells only, ``output_mse_measured == True``) and the census
``counts`` (routed tokens per unit) as a stated proxy for the absent Fisher
weight.  Currency: ``output_mse_under_route_activation_contract`` (scalar
route MSE).  This is NOT the campaign's final joint-AURA objective and no KL
or served-quality claim follows from it.

Schedules (predicted stack costs; dense units are a census in every arm):
  a   three anchors, census (today)
  c   endpoints 832/1088 census; 960 by the per-unit log-linear chord
  c2  chord + pooled other-layer median curvature per projection
  b   960 census + dual-rate (832/1088) sample of s% of experts per stack,
      three estimators: HT ratio (b_ratio), per-(stack, projection)
      intercept transfer law with the slope pooled from OTHER layers (b_law),
      and the model-assisted difference estimator (b_law + HT residual
      correction, b_diff)
  d   k sampled experts per stack at all three rates, Horvitz-Thompson stack
      total (the code's own ``--stack-sample`` path)

Allocation.  Exact multi-choice knapsack DP (numpy) over the 28 measured
stacks + 45 fused gate/up pairs (per-member q256 inside one family, as the
runtime contract licenses) + 45 dense down_proj units, charging
``wire_bytes``, to the byte budget scaled to the measured subset.  Regret =
(truth objective of the allocation decided on predicted costs - omniscient
truth objective) / omniscient, in percent.  Sampling arms are repeated over
seeds and reported as a distribution (mean / p90 / max).

Dense cross-family screen (Step 3): leave-one-LAYER-out prediction of the
BF16 curve from the E4M3 curve + features, regret on the dense-only
allocation.

Runs through PrismaBuild on dl380g10 (CPU only).  Never writes to /tmp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np

BASE = "/mnt/shared/tessera-measurements/glm-canonical-census-20260908"
WORKSPACE = f"{BASE}/first-proof-anchor-preparation-05/workspace"
RATES = (832, 960, 1088)
EXPERT_FAMILY = "TESSERA_E4M3_K1"
DENSE_FAMILIES = ("TESSERA_E4M3_K1", "TESSERA_BF16_K1", "TESSERA_E2M1_K2")
CURRENCY = "output_mse_under_route_activation_contract"
BYTE_TARGET = 155_668_854_528
TOTAL_QUANTIZABLE_PARAMS = 305_915_756_544
DENSE_TOKENS = 262_144
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.(\w+)$")
DENSE_PROJ_RE = re.compile(r"^(.*)\.(\w+)$")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@dataclass
class Stack:
    layer: int
    experts: list                # sorted expert ids
    projections: tuple           # ("down_proj", "gate_proj", "up_proj")
    mse: np.ndarray              # [E, P, R]
    bytes: np.ndarray            # [E, P, R]
    enc_s: np.ndarray            # [E, P, R] encode seconds of the base anchors
    counts: np.ndarray           # [E] routed tokens (per unit; asserted equal across P)
    extra_enc_s: float           # encode seconds of adaptive extra anchors (not in RATES)
    extra_cells: int


@dataclass
class DenseUnit:
    qname: str
    layer: int
    proj: str
    shape: tuple
    counts: int
    # family -> {rate: (mse, bytes, enc_s)}
    curves: dict
    prefix: str = ""             # qname minus the projection (module the fused pair lives in)


def load_rows(rows_dir: str, limit: int | None):
    rows = sorted(d for d in os.listdir(rows_dir) if d.startswith("row-"))
    payloads = []
    for name in rows:
        p = os.path.join(rows_dir, name, "cost.pkl")
        if os.path.exists(p):
            payloads.append(p)
    if limit:
        payloads = payloads[:limit]
    return payloads


def build_population(payloads, census, log):
    counts = census["counts"]
    shapes = census["unit_shapes"]
    stacks: dict[int, dict] = {}
    dense: dict[str, DenseUnit] = {}
    input_hashes = {}
    encode_ledger = defaultdict(lambda: [0, 0.0])   # (class, family, rate) -> [cells, seconds]
    for p in payloads:
        input_hashes[os.path.relpath(p, WORKSPACE)] = sha256_file(p)
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        assert d["currency"] == CURRENCY, (p, d["currency"])
        for qname, cells in d["costs"].items():
            m = LAYER_RE.search(qname)
            layer = int(m.group(1))
            em = EXPERT_RE.search(qname)
            measured = {f: c for f, c in cells.items() if c.get("output_mse_measured")}
            for f, c in measured.items():
                assert c["cost_source"] == "tessera_campaign_measured", (qname, f)
                assert c["tessera_provenance"] == "measured"
                cls = "expert" if em else "dense"
                key = (cls, c["tessera_family"], int(c["tessera_body_rate_q256"]))
                encode_ledger[key][0] += 1
                encode_ledger[key][1] += float(c["encode_seconds"])
            if em:
                e, proj = int(em.group(1)), em.group(2)
                st = stacks.setdefault(layer, {})
                rec = st.setdefault(e, {})
                per = {}
                extra_s, extra_n = 0.0, 0
                for f, c in measured.items():
                    assert c["tessera_family"] == EXPERT_FAMILY, (qname, f)
                    r = int(c["tessera_body_rate_q256"])
                    if r in RATES:
                        per[r] = (float(c["output_mse"]), int(c["wire_bytes"]),
                                  float(c["encode_seconds"]))
                    else:
                        extra_s += float(c["encode_seconds"]); extra_n += 1
                assert set(per) == set(RATES), (qname, sorted(per))
                rec[proj] = (per, int(counts[qname]), extra_s, extra_n)
            else:
                pm = DENSE_PROJ_RE.search(qname)
                prefix, proj = pm.group(1), pm.group(2)
                curves = defaultdict(dict)
                for f, c in measured.items():
                    curves[c["tessera_family"]][int(c["tessera_body_rate_q256"])] = (
                        float(c["output_mse"]), int(c["wire_bytes"]), float(c["encode_seconds"]))
                dense[qname] = DenseUnit(qname, layer, proj, tuple(shapes[qname]),
                                         int(counts[qname]), dict(curves), prefix)
    out_stacks = {}
    for layer, st in sorted(stacks.items()):
        experts = sorted(st)
        projections = tuple(sorted(st[experts[0]]))
        E, P, R = len(experts), len(projections), len(RATES)
        mse = np.zeros((E, P, R)); byt = np.zeros((E, P, R), dtype=np.int64)
        enc = np.zeros((E, P, R)); cnt = np.zeros(E)
        extra_s, extra_n = 0.0, 0
        for i, e in enumerate(experts):
            assert tuple(sorted(st[e])) == projections, (layer, e)
            cs = set()
            for j, proj in enumerate(projections):
                per, c, xs, xn = st[e][proj]
                cs.add(c); extra_s += xs; extra_n += xn
                for k, r in enumerate(RATES):
                    mse[i, j, k], byt[i, j, k], enc[i, j, k] = per[r]
            assert len(cs) == 1, (layer, e, cs)
            cnt[i] = cs.pop()
        out_stacks[layer] = Stack(layer, experts, projections, mse, byt, enc, cnt, extra_s, extra_n)
    log(f"loaded {len(out_stacks)} stacks, {len(dense)} dense units from {len(payloads)} payloads")
    return out_stacks, dense, input_hashes, {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in encode_ledger.items()}


# --------------------------------------------------------------------------
# Items and the exact multi-choice knapsack
# --------------------------------------------------------------------------

@dataclass
class Item:
    key: str
    kind: str                    # "stack" | "fused" | "dense"
    labels: list                 # option labels
    bytes: np.ndarray            # [O] int64
    truth: np.ndarray            # [O] float64 (weighted truth cost)


def stack_truth(stack: Stack, w: np.ndarray) -> np.ndarray:
    """[R] weighted truth cost: sum_e w_e * sum_p mse_{e,p}(r)."""
    return np.einsum("e,epr->r", w, stack.mse)


def stack_bytes(stack: Stack) -> np.ndarray:
    return stack.bytes.sum(axis=(0, 1))


def dense_items(dense: dict, census_groups: dict, weight_mode: str) -> list:
    """Fused gate/up pairs (per-member rate inside one family) + down units."""
    def w_of(u: DenseUnit):
        return 1.0 if weight_mode == "uniform" else u.counts / DENSE_TOKENS

    def opts(u: DenseUnit):
        out = []
        for fam in DENSE_FAMILIES:
            for r, (mse, b, _) in sorted(u.curves.get(fam, {}).items()):
                out.append((fam, r, mse * w_of(u), b))
        return out

    items = []
    by_layer = defaultdict(dict)
    for u in dense.values():
        by_layer[(u.layer, u.prefix)][u.proj] = u
    for (layer, prefix), units in sorted(by_layer.items()):
        g, up, dn = units.get("gate_proj"), units.get("up_proj"), units.get("down_proj")
        if g is not None and up is not None:
            labels, byts, costs = [], [], []
            for fam in DENSE_FAMILIES:
                og = [o for o in opts(g) if o[0] == fam]
                ou = [o for o in opts(up) if o[0] == fam]
                for a in og:
                    for b in ou:
                        labels.append(f"{fam}:{a[1]}/{b[1]}")
                        byts.append(a[3] + b[3]); costs.append(a[2] + b[2])
            items.append(Item(f"g:{prefix}.gate_up", "fused", labels,
                              np.array(byts, dtype=np.int64), np.array(costs)))
        elif g is not None or up is not None:
            raise RuntimeError(f"layer {layer}: incomplete fused pair in the completed rows")
        if dn is not None:
            o = opts(dn)
            items.append(Item(f"d:{prefix}.down", "dense", [f"{f}:{r}" for f, r, _, _ in o],
                              np.array([b for *_, b in o], dtype=np.int64),
                              np.array([c for _, _, c, _ in o])))
    return items


def _dp_pass(dp0, items: list, costs: list, S: int, resolution: int):
    """Run the 'at most s slots' multi-choice DP over ``items`` on top of the
    table ``dp0`` (min cost of everything already folded in, per slot)."""
    dp = dp0
    choice = np.zeros((len(items), S + 1), dtype=np.int8)
    INF = np.inf
    for i, (it, c) in enumerate(zip(items, costs)):
        new = np.full(S + 1, INF)
        arg = np.zeros(S + 1, dtype=np.int8)
        for o in range(len(it.labels)):
            nb = int(-(-int(it.bytes[o]) // resolution))
            if nb > S:
                continue
            cand = np.full(S + 1, INF)
            cand[nb:] = dp[: S + 1 - nb] + float(c[o])
            better = cand < new
            new[better] = cand[better]
            arg[better] = o
        dp = new
        choice[i] = arg
    return dp, choice


def _backtrack(choice, items, b, resolution):
    sel = []
    for i in range(len(items) - 1, -1, -1):
        o = int(choice[i, b])
        sel.append(o)
        b -= int(-(-int(items[i].bytes[o]) // resolution))
    sel.reverse()
    return sel, b


class DensePrefix:
    """The DP over the fixed (census) dense items, computed once at the largest
    slot count and sliced per budget -- dp[s] depends only on smaller s."""

    def __init__(self, items, S_max, resolution):
        self.items = items
        self.resolution = resolution
        self.dp, self.choice = _dp_pass(np.zeros(S_max + 1), items,
                                        [it.truth for it in items], S_max, resolution)


def solve_mckp(items: list, costs: list, budget: int, resolution: int, prefix=None):
    """Exact multi-choice knapsack: min sum cost s.t. sum bytes <= budget.

    Item bytes are rounded UP to ``resolution`` so every DP-feasible choice is
    feasible in true bytes.  ``prefix`` folds in a fixed item set first; its
    choices are appended after ``items``' choices in the returned selection.
    Returns (choice indices, true bytes) or None if infeasible.
    """
    S = budget // resolution
    if prefix is not None:
        assert prefix.resolution == resolution and S < len(prefix.dp)
        dp0 = prefix.dp[: S + 1]
    else:
        dp0 = np.zeros(S + 1)
    dp, choice = _dp_pass(dp0, items, costs, S, resolution)
    if not np.isfinite(dp[S]):
        return None
    sel, b = _backtrack(choice, items, S, resolution)
    all_items = list(items)
    if prefix is not None:
        psel, _ = _backtrack(prefix.choice, prefix.items, b, resolution)
        sel = sel + psel
        all_items = all_items + list(prefix.items)
    true_bytes = int(sum(int(all_items[i].bytes[o]) for i, o in enumerate(sel)))
    assert true_bytes <= budget
    return sel, true_bytes


def objective(items, sel):
    return float(sum(float(items[i].truth[o]) for i, o in enumerate(sel)))


def lambda_star(items, budget):
    """Lagrangian threshold: the price lambda whose convex-hull greedy meets the
    budget (bisection).  Diagnostic only (the DP is exact)."""
    def spend(lam):
        tot = 0
        for it in items:
            o = int(np.argmin(it.truth + lam * it.bytes))
            tot += int(it.bytes[o])
        return tot
    lo, hi = 0.0, 1.0
    while spend(hi) > budget and hi < 1e30:
        hi *= 10
    for _ in range(200):
        mid = math.sqrt(lo * hi) if lo > 0 else 0.5 * (lo + hi)
        if spend(mid) > budget:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------
# Sampling: reuse the campaign's own draw + Horvitz-Thompson estimator
# --------------------------------------------------------------------------

def _load_campaign_sampling():
    try:
        from prismaquant import tessera_campaign as tc  # noqa
        return tc, "prismaquant.tessera_campaign"
    except Exception as exc:  # pragma: no cover - PB run has the real one
        return None, f"local_fallback ({type(exc).__name__}: {exc})"


TC, SAMPLING_SOURCE = _load_campaign_sampling()


def _local_draw(weights, n, *, seed, stack):
    """Fallback mirror of draw_stack_sample (randomized systematic PPS with a
    take-all stratum).  Used only when the campaign module cannot import."""
    names = sorted(weights)
    sizes = {k: float(weights[k]) for k in names}
    frame = [k for k in names if sizes[k] > 0]
    if n >= len(frame):
        return {"units": frame, "inclusion_probability": {k: 1.0 for k in frame}, "method": "census"}
    certainty, rest, remaining = [], list(frame), n
    while rest and remaining > 0:
        tot = sum(sizes[k] for k in rest)
        over = [k for k in rest if remaining * sizes[k] / tot >= 1.0]
        if not over:
            break
        certainty += over; rest = [k for k in rest if k not in set(over)]; remaining -= len(over)
    if remaining == 1:
        raise RuntimeError(f"stack {stack}: one random draw")
    rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{stack}".encode()).hexdigest()[:8], 16))
    pi = {k: 1.0 for k in certainty}
    drawn = list(certainty)
    if remaining > 0 and rest:
        tot = sum(sizes[k] for k in rest)
        perm = list(rng.permutation(rest))
        pi.update({k: remaining * sizes[k] / tot for k in rest})
        start = rng.random(); cum = 0.0; step = 0
        for k in perm:
            cum += pi[k]
            while step < remaining and start + step < cum:
                drawn.append(k); step += 1
    return {"units": sorted(set(drawn)), "inclusion_probability": pi,
            "method": "local_randomized_systematic_pps"}


def draw(weights: dict, n: int, *, seed: int, stack: str):
    fn = TC.draw_stack_sample if TC is not None else _local_draw
    try:
        return fn(weights, n, seed=seed, stack=stack), n
    except RuntimeError as exc:
        if "exactly one" in str(exc) or "one random draw" in str(exc):
            return fn(weights, n + 1, seed=seed, stack=stack), n + 1
        raise


def ht_total(sampled: list, pi: dict, y: dict) -> float:
    """Horvitz-Thompson total sum_S y_e / pi_e, through the campaign's estimator
    when importable (it also validates m >= 2 for the random stratum)."""
    if TC is not None:
        sample = TC.StackExpertSample(
            packed_qname="study", packed_experts_module="study", packed_param="gate_up_proj",
            num_experts=len(pi), stack_h_trace=1.0,
            h_trace_per_expert=tuple(1.0 for _ in pi), sampled_experts=tuple(sampled),
            inclusion_prob=pi, members={e: () for e in sampled}, seed=0)
        t, _se, _m = TC._horvitz_thompson_stack(sample, y)
        return t
    return math.fsum(y[e] / pi[e] for e in sampled)


# --------------------------------------------------------------------------
# Predictors for the stack cost at each rate
# --------------------------------------------------------------------------

def chord_960(stack: Stack) -> np.ndarray:
    """[E, P] log2 mse at 960 from the 832/1088 chord (linear in q256)."""
    l832, l1088 = np.log2(stack.mse[:, :, 0]), np.log2(stack.mse[:, :, 2])
    t = (960 - 832) / (1088 - 832)
    return l832 + t * (l1088 - l832)


def curvature_prior(stacks: dict, hold_out: int) -> np.ndarray:
    """[P] median of (log2 mse(960) - chord(960)) over OTHER layers, per projection."""
    res = []
    for L, s in stacks.items():
        if L == hold_out:
            continue
        res.append(np.log2(s.mse[:, :, 1]) - chord_960(s))
    res = np.concatenate(res, axis=0)
    return np.median(res, axis=0)


def pooled_slopes(stacks: dict, hold_out: int) -> np.ndarray:
    """[P, R] OLS slope of log2 mse(r) on log2 mse(960) pooled over OTHER layers."""
    xs, ys = [], []
    for L, s in stacks.items():
        if L == hold_out:
            continue
        xs.append(np.log2(s.mse[:, :, 1])); ys.append(np.log2(s.mse))
    x = np.concatenate(xs, 0)                       # [N, P]
    y = np.concatenate(ys, 0)                       # [N, P, R]
    P, R = y.shape[1], y.shape[2]
    b = np.ones((P, R))
    for p in range(P):
        xc = x[:, p] - x[:, p].mean()
        for k in range(R):
            if RATES[k] == 960:
                continue
            yc = y[:, p, k] - y[:, p, k].mean()
            b[p, k] = float((xc * yc).sum() / (xc * xc).sum())
    return b


def per_layer_slopes(stacks: dict) -> dict:
    out = {}
    for L, s in stacks.items():
        x = np.log2(s.mse[:, :, 1]); y = np.log2(s.mse)
        b = np.ones((y.shape[1], y.shape[2]))
        for p in range(y.shape[1]):
            xc = x[:, p] - x[:, p].mean()
            for k in range(y.shape[2]):
                if RATES[k] == 960:
                    continue
                yc = y[:, p, k] - y[:, p, k].mean()
                b[p, k] = float((xc * yc).sum() / (xc * xc).sum())
        out[L] = b
    return out


def predict_stack(stack: Stack, w: np.ndarray, schedule: str, *, seed: int,
                  param, pooled) -> tuple[np.ndarray, dict]:
    """Return ([R] predicted weighted cost, info)."""
    truth = stack_truth(stack, w)
    E = len(stack.experts)
    info = {}
    if schedule == "a":
        return truth.copy(), info
    if schedule in ("c", "c2"):
        l960 = chord_960(stack)
        if schedule == "c2":
            l960 = l960 + pooled["curv"][None, :]
        pred = truth.copy()
        pred[1] = float(np.einsum("e,ep->", w, 2.0 ** l960))
        return pred, info
    # sampled arms
    names = {str(e): float(w[i]) for i, e in enumerate(stack.experts)}
    if schedule.startswith("b"):
        n = max(2, int(round(param * E / 100.0)))
    else:
        n = int(param)
    d, n_used = draw(names, n, seed=seed, stack=f"layer{stack.layer}")
    sampled = [int(u) for u in d["units"]]
    pi = {int(k): float(v) for k, v in d["inclusion_probability"].items() if float(v) > 0}
    idx = {e: i for i, e in enumerate(stack.experts)}
    info = {"n_requested": n, "n_used": n_used, "m_sampled": len(sampled)}
    y = {r: {e: float(w[idx[e]] * stack.mse[idx[e], :, k].sum()) for e in sampled}
         for k, r in enumerate(RATES)}
    if schedule == "d":
        pred = np.array([ht_total(sampled, pi, y[r]) for r in RATES])
        return pred, info
    # b arms: 960 is a census
    pred = truth.copy()
    if schedule == "b_ratio":
        t960 = ht_total(sampled, pi, y[960])
        for k, r in enumerate(RATES):
            if r == 960:
                continue
            pred[k] = truth[1] * ht_total(sampled, pi, y[r]) / t960
        return pred, info
    # transfer law: log2 mse_{e,p}(r) = a_{p,r} + b_{p,r} * log2 mse_{e,p}(960)
    b = pooled["slope"]                                 # [P, R]
    l960 = np.log2(stack.mse[:, :, 1])                  # [E, P]
    srows = np.array([idx[e] for e in sampled])
    for k, r in enumerate(RATES):
        if r == 960:
            continue
        lr = np.log2(stack.mse[srows, :, k])            # [m, P]
        a = (lr - b[:, k][None, :] * l960[srows]).mean(axis=0)   # [P]
        lhat = a[None, :] + b[:, k][None, :] * l960     # [E, P]
        yhat = (2.0 ** lhat).sum(axis=1)                # [E]
        total = float((w * yhat).sum())
        if schedule == "b_diff":
            resid = {e: float(w[idx[e]] * (stack.mse[idx[e], :, k].sum() - yhat[idx[e]]))
                     for e in sampled}
            total += ht_total(sampled, pi, resid)
        pred[k] = total
    return pred, info


# --------------------------------------------------------------------------
# One evaluation = one (weight mode, schedule, param, seed) -> allocations
# --------------------------------------------------------------------------

G = {}   # populated in the parent before forking


def evaluate(job):
    weight_mode, schedule, param, seed, h_seed = job
    stacks = G["stacks"]
    weights = G["weights"][(weight_mode, h_seed)]
    base_items = G["dense_items"][weight_mode]
    budgets = G["budgets"]
    items = []
    preds = []
    infos = {}
    for L in sorted(stacks):
        s = stacks[L]
        w = weights[L]
        pooled = G["pooled"][(weight_mode, L)]
        pred, info = predict_stack(s, w, schedule, seed=seed, param=param, pooled=pooled)
        items.append(Item(f"s:layer{L}", "stack", [f"{EXPERT_FAMILY}:{r}" for r in RATES],
                          stack_bytes(s), stack_truth(s, w)))
        preds.append(pred)
        if info:
            infos[L] = info
    prefix = G["dense_prefix"][weight_mode]
    n_stack = len(stacks)
    stack_items = items
    items = items + base_items
    pred_costs = preds
    out = {"job": {"weight_mode": weight_mode, "schedule": schedule, "param": param,
                   "seed": seed, "h_seed": h_seed},
           "sampling_source": SAMPLING_SOURCE, "budgets": {}, "pred_log2_error": {}}
    # pointwise error of the predicted stack cost per layer per rate
    for i, L in enumerate(sorted(stacks)):
        out["pred_log2_error"][str(L)] = [float(np.log2(preds[i][k] / items[i].truth[k]))
                                          for k in range(len(RATES))]
    for tag, budget in budgets.items():
        opt = G["opt"][(weight_mode, h_seed, tag)]
        if opt is None:
            out["budgets"][tag] = {"infeasible": True}
            continue
        r = solve_mckp(stack_items, pred_costs, budget, G["resolution"], prefix=prefix)
        sel, used = r
        obj = objective(items, sel)
        regret = (obj - opt["objective"]) / opt["objective"] * 100.0
        agree = [int(sel[i] == opt["sel"][i]) for i in range(n_stack)]
        out["budgets"][tag] = {
            "regret_pct": regret, "objective": obj, "bytes": used,
            "stack_rates": [RATES[sel[i]] for i in range(n_stack)],
            "stack_agreement": agree,
            "stack_agreement_frac": float(np.mean(agree)),
            "all_agree": bool(sum(sel[i] != opt["sel"][i] for i in range(len(items))) == 0),
        }
    out["sample_info"] = infos
    return out


# --------------------------------------------------------------------------
# Dense cross-family screen
# --------------------------------------------------------------------------

def dense_cross_family(dense: dict, budgets_dense: dict, resolution: int, log):
    """Leave-one-LAYER-out: predict the BF16 curve of the held-out layer's units
    from E4M3 + features; regret on the dense-only allocation."""
    units = sorted(dense.values(), key=lambda u: (u.layer, u.proj))
    projs = sorted({u.proj for u in units})
    layers = sorted({u.layer for u in units})

    def features(u: DenseUnit, r_idx: int):
        f = [1.0, math.log2(u.curves["TESSERA_E4M3_K1"][RATES[r_idx]][0])]
        f += [1.0 if u.proj == p else 0.0 for p in projs[1:]]
        f += [math.log2(u.shape[0]), math.log2(u.shape[1]), u.layer / max(layers)]
        f += [1.0 if k == r_idx else 0.0 for k in range(1, len(RATES))]
        return f

    X, Y, owner = [], [], []
    for u in units:
        for k in range(len(RATES)):
            X.append(features(u, k)); Y.append(math.log2(u.curves["TESSERA_BF16_K1"][RATES[k]][0]))
            owner.append(u.layer)
    X, Y, owner = np.array(X), np.array(Y), np.array(owner)
    pred_log2 = {}
    errs = []
    for L in layers:
        tr = owner != L
        beta, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
        yhat = X[~tr] @ beta
        errs.extend((yhat - Y[~tr]).tolist())
        rows = np.where(~tr)[0]
        for i, rix in enumerate(rows):
            pred_log2[rix] = float(yhat[i])
    errs = np.abs(np.array(errs))
    # build predicted dense population
    pred_dense = {}
    rix = 0
    for u in units:
        curves = {fam: dict(c) for fam, c in u.curves.items()}
        for k in range(len(RATES)):
            mse, b, s = curves["TESSERA_BF16_K1"][RATES[k]]
            curves["TESSERA_BF16_K1"][RATES[k]] = (2.0 ** pred_log2[rix], b, s)
            rix += 1
        pred_dense[u.qname] = DenseUnit(u.qname, u.layer, u.proj, u.shape, u.counts, curves)
    result = {"n_units": len(units), "n_layers": len(layers),
              "bf16_from_e4m3_loo_layer_abs_log2_error": {
                  "mean": float(errs.mean()), "median": float(np.median(errs)),
                  "p90": float(np.quantile(errs, 0.9)), "max": float(errs.max())},
              "budgets": {}}
    for wm in ("uniform", "counts"):
        items_t = dense_items(dense, None, wm)
        items_p = dense_items(pred_dense, None, wm)
        assert [it.labels for it in items_t] == [it.labels for it in items_p]
        for tag, budget in budgets_dense.items():
            rt = solve_mckp(items_t, [it.truth for it in items_t], budget, resolution)
            if rt is None:
                result["budgets"][f"{wm}:{tag}"] = {"infeasible": True}
                continue
            rp = solve_mckp(items_t, [it.truth for it in items_p], budget, resolution)
            ot, op = objective(items_t, rt[0]), objective(items_t, rp[0])
            fam_t = [it.labels[o].split(":")[0] for it, o in zip(items_t, rt[0])]
            fam_p = [it.labels[o].split(":")[0] for it, o in zip(items_t, rp[0])]
            result["budgets"][f"{wm}:{tag}"] = {
                "regret_pct": (op - ot) / ot * 100.0, "bytes_budget": budget,
                "bytes_truth": rt[1], "bytes_pred": rp[1],
                "items_agree_frac": float(np.mean([a == b for a, b in zip(rt[0], rp[0])])),
                "family_hist_truth": {f: fam_t.count(f) for f in set(fam_t)},
                "family_hist_pred": {f: fam_p.count(f) for f in set(fam_p)},
            }
    log(f"dense cross-family screen: {result['bf16_from_e4m3_loo_layer_abs_log2_error']}")
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory (never /tmp)")
    ap.add_argument("--rows-limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--sweep-seeds", type=int, default=8)
    ap.add_argument("--h-seeds", type=int, default=3)
    ap.add_argument("--h-cv", type=float, default=1.5)
    ap.add_argument("--h-rho", type=float, default=0.5)
    ap.add_argument("--resolution", type=int, default=256 * 1024)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample-pcts", default="1,3,5,10")
    ap.add_argument("--sample-ks", default="8,32,96,216")
    ap.add_argument("--budget-deltas", default="-10,-5,-2.5,0,2.5,5,10")
    ap.add_argument("--sweep-points", type=int, default=21)
    args = ap.parse_args()
    assert not args.out.startswith("/tmp"), "never /tmp"
    os.makedirs(args.out, exist_ok=True)
    os.environ["TMPDIR"] = args.out
    t0 = time.time()
    log_lines = []

    def log(msg):
        line = f"[{time.time() - t0:8.1f}s] {msg}"
        print(line, flush=True); log_lines.append(line)

    log(f"sampling source: {SAMPLING_SOURCE}")
    with open(f"{WORKSPACE}/census.json") as fh:
        census = json.load(fh)
    payloads = load_rows(f"{WORKSPACE}/rows", args.rows_limit)
    stacks, dense, input_hashes, encode_ledger = build_population(payloads, census, log)
    input_hashes["census.json"] = sha256_file(f"{WORKSPACE}/census.json")

    # bytes per rate are a function of shape only -- assert, and record
    for L, s in stacks.items():
        for p in range(s.bytes.shape[1]):
            for k in range(s.bytes.shape[2]):
                assert len(set(s.bytes[:, p, k].tolist())) == 1, (L, p, k)

    # weights: uniform, counts (routed-token fraction), and lognormal-h sensitivity
    rng = np.random.default_rng(20260910)
    weights = {}
    for L, s in stacks.items():
        weights.setdefault(("uniform", 0), {})[L] = np.ones(len(s.experts))
        weights.setdefault(("counts", 0), {})[L] = s.counts / DENSE_TOKENS
    sigma = math.sqrt(math.log(1 + args.h_cv ** 2))
    for hs in range(1, args.h_seeds + 1):
        weights[("lognormal_h", hs)] = {}
        for L, s in stacks.items():
            zc = np.log(np.maximum(s.counts, 1.0)); zc = (zc - zc.mean()) / (zc.std() + 1e-12)
            z = args.h_rho * zc + math.sqrt(1 - args.h_rho ** 2) * rng.standard_normal(len(zc))
            h = np.exp(sigma * z - sigma ** 2 / 2) * (s.counts / DENSE_TOKENS)
            weights[("lognormal_h", hs)][L] = h
    count_cv = {str(L): float(s.counts.std() / s.counts.mean()) for L, s in stacks.items()}

    # pooled (other-layer) parameters for c2 / b_law / b_diff, keyed by held-out layer
    pooled = {}
    for wm in ("uniform", "counts", "lognormal_h"):
        for L in stacks:
            pooled[(wm, L)] = {"curv": curvature_prior(stacks, L), "slope": pooled_slopes(stacks, L)}
    slopes_by_layer = per_layer_slopes(stacks)
    slope_sd = np.std(np.stack(list(slopes_by_layer.values())), axis=0)

    # budgets scaled to the measured subset
    params_stacks = sum(int(np.prod(census["unit_shapes"][f"model.language_model.layers.{L}.mlp.experts.{e}.{p}"]))
                        for L, s in stacks.items() for e in s.experts for p in s.projections)
    params_dense = sum(int(np.prod(u.shape)) for u in dense.values())
    bpp = BYTE_TARGET * 8 / TOTAL_QUANTIZABLE_PARAMS
    subset_budget = int(bpp * (params_stacks + params_dense) / 8)
    dense_budget = int(bpp * params_dense / 8)
    deltas = [float(x) for x in args.budget_deltas.split(",")]
    budgets = {f"{d:+g}%": int(subset_budget * (1 + d / 100)) for d in deltas}
    budgets_dense = {f"{d:+g}%": int(dense_budget * (1 + d / 100)) for d in deltas}
    log(f"bpp target {bpp:.6f}; subset params {params_stacks + params_dense:,} "
        f"(stacks {params_stacks:,}, dense {params_dense:,}); subset budget {subset_budget:,} B")

    # items per weight mode; menu span; omniscient allocations
    G["stacks"] = stacks; G["weights"] = weights; G["pooled"] = pooled
    G["resolution"] = args.resolution; G["budgets"] = budgets
    G["dense_items"] = {wm: dense_items(dense, None, "uniform" if wm == "uniform" else "counts")
                        for wm in ("uniform", "counts", "lognormal_h")}
    G["opt"] = {}
    menu = {}
    S_max = 0
    for (wm, hs), wts in weights.items():
        items = [Item(f"s:layer{L}", "stack", [f"{EXPERT_FAMILY}:{r}" for r in RATES],
                      stack_bytes(stacks[L]), stack_truth(stacks[L], wts[L])) for L in sorted(stacks)]
        max_spend = int(sum(int(it.bytes.max()) for it in items + G["dense_items"][wm]))
        S_max = max(S_max, int(max(max(budgets.values()), max_spend) // args.resolution))
    G["dense_prefix"] = {wm: DensePrefix(G["dense_items"][wm], S_max, args.resolution)
                         for wm in ("uniform", "counts", "lognormal_h")}
    log(f"dense DP prefixes built at {S_max + 1} slots")
    for (wm, hs), wts in weights.items():
        items = [Item(f"s:layer{L}", "stack", [f"{EXPERT_FAMILY}:{r}" for r in RATES],
                      stack_bytes(stacks[L]), stack_truth(stacks[L], wts[L])) for L in sorted(stacks)]
        items += G["dense_items"][wm]
        min_spend = int(sum(int(it.bytes.min()) for it in items))
        max_spend = int(sum(int(it.bytes.max()) for it in items))
        menu[f"{wm}:{hs}"] = {"min_spend": min_spend, "max_spend": max_spend}
        stack_items = items[:len(stacks)]
        for tag, budget in budgets.items():
            r = solve_mckp(stack_items, [it.truth for it in stack_items], budget, args.resolution,
                           prefix=G["dense_prefix"][wm])
            if r is None:
                G["opt"][(wm, hs, tag)] = None
                continue
            sel, used = r
            lam = lambda_star(items, budget)
            # marginal-price diagnostic per stack (truth): distance of the chosen
            # rung's adjacent marginal prices from lambda*, in log10
            diag = {}
            for i, L in enumerate(sorted(stacks)):
                it = items[i]
                mp = [(it.truth[k] - it.truth[k + 1]) / (it.bytes[k + 1] - it.bytes[k]) for k in range(2)]
                diag[str(L)] = {"rate": RATES[sel[i]],
                                "log10_marginal_price_over_lambda": [float(np.log10(m / lam)) for m in mp]}
            G["opt"][(wm, hs, tag)] = {"sel": sel, "objective": objective(items, sel), "bytes": used,
                                      "budget": budget, "saturated": budget >= max_spend,
                                      "below_min": budget < min_spend, "lambda_star": lam,
                                      "stack_rates": [RATES[sel[i]] for i in range(len(stacks))],
                                      "marginal_price_diag": diag}
    log(f"menu span: {menu}")
    for k, v in sorted(G["opt"].items(), key=str):
        if v:
            log(f"omniscient {k}: bytes {v['bytes']:,}/{v['budget']:,} obj {v['objective']:.6e} "
                f"saturated={v['saturated']} rates={v['stack_rates']}")

    # jobs
    pcts = [float(x) for x in args.sample_pcts.split(",")]
    ks = [int(x) for x in args.sample_ks.split(",")]
    jobs = []
    for wm in ("uniform", "counts"):
        jobs.append((wm, "a", None, 0, 0))
        jobs.append((wm, "c", None, 0, 0))
        jobs.append((wm, "c2", None, 0, 0))
        for seed in range(args.seeds):
            for s in pcts:
                for est in ("b_ratio", "b_law", "b_diff"):
                    jobs.append((wm, est, s, seed, 0))
            for k in ks:
                jobs.append((wm, "d", k, seed, 0))
    for hs in range(1, args.h_seeds + 1):
        jobs.append(("lognormal_h", "c", None, 0, hs))
        jobs.append(("lognormal_h", "c2", None, 0, hs))
        for seed in range(min(args.seeds, 20)):
            for s in pcts:
                for est in ("b_ratio", "b_law", "b_diff"):
                    jobs.append(("lognormal_h", est, s, seed, hs))
            for k in ks:
                jobs.append(("lognormal_h", "d", k, seed, hs))
    log(f"{len(jobs)} evaluations x {len(budgets)} budgets over {len(stacks)} held-out layers")
    ctx = get_context("fork")
    partial_dir = os.path.join(args.out, "partial")
    os.makedirs(partial_dir, exist_ok=True)
    groups = defaultdict(list)
    for j in jobs:
        groups[(j[0], j[1], j[2])].append(j)
    results = []
    with ctx.Pool(args.workers) as pool:
        for gi, (gk, gjobs) in enumerate(sorted(groups.items(), key=str)):
            t1 = time.time()
            rs = pool.map(evaluate, gjobs, chunksize=2)
            results.extend(rs)
            name = f"{gk[0]}_{gk[1]}" + ("" if gk[2] is None else f"_{gk[2]:g}")
            with open(os.path.join(partial_dir, f"{name}.json"), "w") as fh:
                json.dump({"group": list(gk), "n": len(rs), "results": rs}, fh, default=float)
            reg = {t: float(np.mean([r["budgets"][t]["regret_pct"] for r in rs if "regret_pct" in r["budgets"][t]] or [float("nan")]))
                   for t in budgets}
            log(f"group {gi + 1}/{len(groups)} {name}: {len(rs)} evals x {len(stacks)} layers in "
                f"{time.time() - t1:.1f}s; mean regret% " + " ".join(f"{t}={v:.4f}" for t, v in reg.items()))
    log("main grid done")

    # continuous budget sweep (aggregate, fewer seeds), uniform + counts
    sweep = {}
    for wm in ("uniform", "counts"):
        m = menu[f"{wm}:0"]
        grid = np.linspace(m["min_spend"], m["max_spend"], args.sweep_points)
        sweep_budgets = {f"sweep{i}": int(b) for i, b in enumerate(grid)}
        items = [Item(f"s:layer{L}", "stack", [], stack_bytes(stacks[L]),
                      stack_truth(stacks[L], weights[(wm, 0)][L])) for L in sorted(stacks)] + G["dense_items"][wm]
        for it in items[:len(stacks)]:
            it.labels = [f"{EXPERT_FAMILY}:{r}" for r in RATES]
        G["budgets"] = sweep_budgets
        for tag, budget in sweep_budgets.items():
            r = solve_mckp(items[:len(stacks)], [it.truth for it in items[:len(stacks)]], budget,
                           args.resolution, prefix=G["dense_prefix"][wm])
            G["opt"][(wm, 0, tag)] = None if r is None else {
                "sel": r[0], "objective": objective(items, r[0]), "bytes": r[1]}
        sjobs = [(wm, "c", None, 0, 0), (wm, "c2", None, 0, 0)]
        for seed in range(args.sweep_seeds):
            for s in pcts:
                for est in ("b_ratio", "b_law", "b_diff"):
                    sjobs.append((wm, est, s, seed, 0))
            for k in ks:
                sjobs.append((wm, "d", k, seed, 0))
        with ctx.Pool(args.workers) as pool:
            sres = pool.map(evaluate, sjobs, chunksize=2)
        with open(os.path.join(partial_dir, f"sweep_{wm}.json"), "w") as fh:
            json.dump({"budgets": sweep_budgets, "results": sres}, fh, default=float)
        log(f"sweep {wm}: {len(sres)} evals x {len(sweep_budgets)} budgets written")
        sweep[wm] = {"budgets": sweep_budgets, "bpp": {t: b * 8 / (params_stacks + params_dense)
                                                       for t, b in sweep_budgets.items()},
                     "results": [{"job": r["job"], "budgets": {t: {"regret_pct": v.get("regret_pct"),
                                                                   "stack_agreement_frac": v.get("stack_agreement_frac")}
                                                               for t, v in r["budgets"].items()}} for r in sres]}
    G["budgets"] = budgets
    log("sweep done")

    dense_result = dense_cross_family(dense, budgets_dense, args.resolution, log)

    # encode accounting -----------------------------------------------------
    def ledger(cls, fam, rate):
        v = encode_ledger.get(f"{cls}|{fam}|{rate}", [0, 0.0]); return v[0], v[1]
    n_stack = len(stacks); E = 288
    expert_cells = {r: ledger("expert", EXPERT_FAMILY, r) for r in RATES}
    extra_cells = sum(s.extra_cells for s in stacks.values())
    extra_secs = sum(s.extra_enc_s for s in stacks.values())
    dense_cells = sum(v[0] for k, v in encode_ledger.items() if k.startswith("dense|"))
    dense_secs = sum(v[1] for k, v in encode_ledger.items() if k.startswith("dense|"))
    per_rate_cells = {r: expert_cells[r][0] for r in RATES}
    per_rate_secs = {r: expert_cells[r][1] for r in RATES}
    schedules_enc = {}
    schedules_enc["a_today_actual"] = {"expert_cells": sum(per_rate_cells.values()) + extra_cells,
                                       "expert_gpu_s": sum(per_rate_secs.values()) + extra_secs}
    schedules_enc["a_three_anchor"] = {"expert_cells": sum(per_rate_cells.values()),
                                       "expert_gpu_s": sum(per_rate_secs.values())}
    schedules_enc["c_endpoints"] = {"expert_cells": per_rate_cells[832] + per_rate_cells[1088],
                                    "expert_gpu_s": per_rate_secs[832] + per_rate_secs[1088]}
    for s in pcts:
        n = max(2, int(round(s * E / 100)))
        frac = n / E
        schedules_enc[f"b_{s:g}pct(n={n})"] = {
            "expert_cells": per_rate_cells[960] + frac * (per_rate_cells[832] + per_rate_cells[1088]),
            "expert_gpu_s": per_rate_secs[960] + frac * (per_rate_secs[832] + per_rate_secs[1088])}
    for k in ks:
        frac = k / E
        schedules_enc[f"d_k{k}"] = {"expert_cells": frac * sum(per_rate_cells.values()),
                                    "expert_gpu_s": frac * sum(per_rate_secs.values())}
    for v in schedules_enc.values():
        v["expert_gpu_h_measured_subset"] = v["expert_gpu_s"] / 3600
        v["expert_gpu_h_extrapolated_42_stacks"] = v["expert_gpu_s"] / 3600 * 42 / n_stack
        v["saving_vs_three_anchor_gpu_h_42_stacks"] = (
            (schedules_enc["a_three_anchor"]["expert_gpu_s"] - v["expert_gpu_s"]) / 3600 * 42 / n_stack)
    encode = {"per_rate_expert_cells": per_rate_cells, "per_rate_expert_gpu_s": per_rate_secs,
              "adaptive_extra_expert_cells": extra_cells, "adaptive_extra_expert_gpu_s": extra_secs,
              "dense_cells_all_families": dense_cells, "dense_gpu_s_all_families": dense_secs,
              "ledger": encode_ledger, "schedules": schedules_enc,
              "accounting": "encode_seconds per measured cell; expert cells use "
                            "batch_wall_time_divided_by_batch_size (encoding_batch_size 8)"}

    # summary tables --------------------------------------------------------
    def summarize(rs, tag):
        vals = np.array([r["budgets"][tag]["regret_pct"] for r in rs if "regret_pct" in r["budgets"][tag]])
        agree = np.array([r["budgets"][tag]["stack_agreement_frac"] for r in rs if "regret_pct" in r["budgets"][tag]])
        if len(vals) == 0:
            return {"n": 0}
        return {"n": int(len(vals)), "mean": float(vals.mean()), "p50": float(np.median(vals)),
                "p90": float(np.quantile(vals, 0.9)), "max": float(vals.max()),
                "frac_zero": float((vals <= 1e-12).mean()),
                "stack_agreement_mean": float(agree.mean())}
    summary = {}
    for wm in ("uniform", "counts", "lognormal_h"):
        summary[wm] = {}
        keyed = defaultdict(list)
        for r in results:
            j = r["job"]
            if j["weight_mode"] != wm:
                continue
            keyed[(j["schedule"], j["param"])].append(r)
        for (sch, param), rs in sorted(keyed.items(), key=str):
            name = sch if param is None else f"{sch}@{param:g}"
            summary[wm][name] = {tag: summarize(rs, tag) for tag in budgets}
            # per-layer pointwise error (log2) across seeds
            per_layer = {}
            for L in sorted(stacks):
                arr = np.array([r["pred_log2_error"][str(L)] for r in rs])
                per_layer[str(L)] = {"mean_abs_log2": [float(x) for x in np.abs(arr).mean(axis=0)],
                                     "max_abs_log2": [float(x) for x in np.abs(arr).max(axis=0)]}
                for tag in budgets:
                    ag = [r["budgets"][tag]["stack_agreement"][sorted(stacks).index(L)]
                          for r in rs if "stack_agreement" in r["budgets"][tag]]
                    per_layer[str(L)][f"agree_{tag}"] = float(np.mean(ag)) if ag else None
            summary[wm][name]["per_layer"] = per_layer
    # sweep summary
    sweep_summary = {}
    for wm, sw in sweep.items():
        keyed = defaultdict(list)
        for r in sw["results"]:
            j = r["job"]; keyed[j["schedule"] if j["param"] is None else f"{j['schedule']}@{j['param']:g}"].append(r)
        sweep_summary[wm] = {"bpp": [sw["bpp"][t] for t in sw["budgets"]], "curves": {}}
        for name, rs in sorted(keyed.items()):
            curve = []
            for t in sw["budgets"]:
                vals = [r["budgets"][t]["regret_pct"] for r in rs if r["budgets"][t]["regret_pct"] is not None]
                curve.append({"mean": float(np.mean(vals)) if vals else None,
                              "p90": float(np.quantile(vals, 0.9)) if vals else None})
            sweep_summary[wm]["curves"][name] = curve

    omniscient = {f"{wm}:{hs}:{tag}": v for (wm, hs, tag), v in G["opt"].items()
                  if not tag.startswith("sweep")}
    result = {
        "schema": "prismaquant.glm_probe_reduction_regret.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv, "sampling_source": SAMPLING_SOURCE,
        "inputs": {"workspace": WORKSPACE, "payloads": len(payloads), "sha256": input_hashes,
                   "input_list_sha256": hashlib.sha256(
                       json.dumps(sorted(input_hashes.items())).encode()).hexdigest()},
        "population": {"stacks": len(stacks), "layers": sorted(stacks), "experts_per_stack": E,
                       "dense_units": len(dense), "params_stacks": params_stacks,
                       "params_dense": params_dense, "count_cv_per_stack": count_cv,
                       "count_cv_median": float(np.median(list(count_cv.values())))},
        "objective_note": ("unweighted sum of output_mse (weight_mode=uniform) or routed-token-"
                           "fraction weighted sum counts/262144 (weight_mode=counts) as a stated "
                           "proxy for h_trace, which no GLM probe provides; lognormal_h is a "
                           "hypothesis sensitivity (CV %.2f, rho %.2f to log counts), NOT a "
                           "measurement. Bytes = wire_bytes. Currency = %s. The campaign's final "
                           "selection objective is joint AURA (tessera_joint_aura.py), not this."
                           % (args.h_cv, args.h_rho, CURRENCY)),
        "budget": {"byte_target_full_model": BYTE_TARGET, "bpp": bpp,
                   "subset_budget": subset_budget, "budgets": budgets, "menu_span": menu,
                   "dense_budget": dense_budget, "dense_budgets": budgets_dense,
                   "resolution_bytes": args.resolution},
        "omniscient": omniscient,
        "transfer_law": {"pooled_slope_sd_across_layers_P_by_R": slope_sd.tolist(),
                         "per_layer_slopes": {str(L): b.tolist() for L, b in slopes_by_layer.items()},
                         "projections": list(next(iter(stacks.values())).projections), "rates": list(RATES)},
        "summary": summary, "sweep": sweep_summary, "dense_cross_family": dense_result,
        "encode": encode,
        "raw": results,
        "log": log_lines,
        "wall_seconds": time.time() - t0,
    }
    out_path = os.path.join(args.out, "regret_study.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    digest = sha256_file(out_path)
    with open(os.path.join(args.out, "regret_study.sha256"), "w") as fh:
        fh.write(f"{digest}  regret_study.json\n")
    log(f"wrote {out_path} sha256 {digest}")

    # human table
    print("\n=== regret % (mean / p90 / max over seeds; stack agreement) ===")
    for wm in ("uniform", "counts", "lognormal_h"):
        print(f"--- weights: {wm}")
        for name, per in summary[wm].items():
            cells = []
            for tag in budgets:
                s = per[tag]
                if s.get("n", 0) == 0:
                    cells.append(f"{tag}: n/a")
                else:
                    cells.append(f"{tag}: {s['mean']:.4f}/{s['p90']:.4f}/{s['max']:.4f} ag={s['stack_agreement_mean']:.3f}")
            print(f"{name:14s} " + " | ".join(cells))
    print("\n=== encode ===")
    for k, v in schedules_enc.items():
        print(f"{k:22s} cells {v['expert_cells']:10.0f}  gpu_h(28) {v['expert_gpu_h_measured_subset']:7.2f}  "
              f"gpu_h(42) {v['expert_gpu_h_extrapolated_42_stacks']:7.2f}  saving(42) {v['saving_vs_three_anchor_gpu_h_42_stacks']:7.2f}")
    print("\n=== dense cross-family ===")
    print(json.dumps({k: v for k, v in dense_result.items() if k != "budgets"}, indent=1))
    for k, v in dense_result["budgets"].items():
        print(k, v)


if __name__ == "__main__":
    main()
