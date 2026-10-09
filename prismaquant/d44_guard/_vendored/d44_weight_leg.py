"""Frozen D44 weight-leg statistics: pure, deterministic, stdlib only.

Implements frozen-method points 6 and 9 (CEO freeze 2026-10-06T21:15:53Z,
frozen_D44_method points 6_primary_metric and 9_weight_leg):

Primary metric C_gap = sum(A-N)/sum(A-X) over ALL routed units, where
A/N/X are raw full-unnormalized HELD SSE totals (A8S/new/EXL3). The
A8S-mass-weighted C is a secondary readout, never a gate. Per-layer
fraction f_l = (A_l-N_l)/(A_l-X_l).

Bootstrap: exactly 2000 resamples, seed 20261006, stratified by layer,
unit = WHOLE experts (each expert's gate/up/down units move jointly).
Each layer resamples its own expert count with replacement from its own
experts only; raw SSE triples are resampled and C_gap and every f_l are
recomputed from the resampled raw sums. There are no three independent
role resamples and no per-expert fraction averaging anywhere.

Quantiles are nearest-rank: q(sorted samples) = s[ceil(q*n)-1].

A missing unit is unmeasured, never PASS and never excluded: any layer
with missing/unmeasured units forces status UNMEASURED. A measured but
degenerate layer (nonpositive or nonfinite gap denominator) is an
explicit FAIL reason, never a silent exclusion.
"""
from __future__ import annotations
import math
import random

LAYERS = (3, 40, 41, 42, 43, 44)
RESAMPLES = 2000
SEED = 20261006
C_GAP_MIN = 0.5
P10_MIN = 0.35
POSITIVE_LAYERS_MIN = 4
EXCEPTION_LAYER = 3
EXCEPTION_C_GAP_MIN = 0.5
EXPERTS_PER_LAYER = 288


def ratio(A, N, X):
    """(value, reason). value None iff unmeasured or undefined denominator."""
    for name, v in (('A', A), ('N', N), ('X', X)):
        if v is None:
            return None, 'unmeasured ' + name
        if not math.isfinite(v):
            return None, 'nonfinite ' + name
    d = A - X
    if not math.isfinite(d) or d <= 0:
        return None, 'nonpositive or nonfinite gap denominator A-X'
    return (A - N) / d, None


def ratio_signed(A, N, X):
    """Bootstrap-draw ratio over every mathematically defined draw, including
    signed (negative) denominators. Only zero or nonfinite denominators are
    undefined; those draws stay explicit and can never silently qualify a gate."""
    for name, v in (('A', A), ('N', N), ('X', X)):
        if v is None:
            return None, 'unmeasured ' + name
        if not math.isfinite(v):
            return None, 'nonfinite ' + name
    d = A - X
    if d == 0:
        return None, 'zero gap denominator'
    if not math.isfinite(d):
        return None, 'nonfinite gap denominator'
    return (A - N) / d, None


def quantile(sorted_samples, q):
    """Nearest-rank quantile on an ascending-sorted list."""
    n = len(sorted_samples)
    return sorted_samples[min(n - 1, max(0, math.ceil(q * n) - 1))]


def bootstrap_experts(experts_by_layer, *, resamples=RESAMPLES, seed=SEED):
    """Stratified whole-expert bootstrap.

    experts_by_layer: {layer: [(A,N,X), ...]} whole-expert raw SSE triples
    (the expert's gate/up/down unit SSEs already summed jointly). Each
    resample draws len(pool) experts with replacement per layer from that
    layer's experts only, then recomputes raw sums, C_gap and every f_l.
    """
    layers = sorted(experts_by_layer)
    rng = random.Random(seed)
    c_samples, f_samples = [], {l: [] for l in layers}
    c_undef, f_undef = {}, {l: {} for l in layers}
    for _ in range(resamples):
        sums = {}
        for l in layers:
            pool = experts_by_layer[l]
            k = len(pool)
            A, N, X = [], [], []
            for _ in range(k):
                ea, en, ex = pool[rng.randrange(k)]
                A.append(ea); N.append(en); X.append(ex)
            sums[l] = (math.fsum(A), math.fsum(N), math.fsum(X))
            v, r = ratio_signed(*sums[l])
            f_samples[l].append(v)
            if v is None:
                f_undef[l][r] = f_undef[l].get(r, 0) + 1
        tot = tuple(math.fsum(sums[l][i] for l in layers) for i in range(3))
        v, r = ratio_signed(*tot)
        c_samples.append(v)
        if v is None:
            c_undef[r] = c_undef.get(r, 0) + 1
    c_ord = sorted(v for v in c_samples if v is not None)
    def p90s(l):
        o = sorted(v for v in f_samples[l] if v is not None)
        return quantile(o, 0.90) if o else None
    return {'resamples': resamples, 'seed': seed, 'stratified_by': 'layer',
            'unit': 'whole experts (gate/up/down joint); raw SSE resampled, C_gap and f_l recomputed',
            'quantile': 'nearest-rank over signed ratios of ALL 2000 draws; undefined-denominator draws are counted, never discarded silently',
            'c_gap_samples': c_samples,
            'c_gap_defined_draws': len(c_ord),
            'c_gap_undefined_draws': resamples - len(c_ord),
            'c_gap_undefined_reasons': c_undef,
            'layer_f_undefined_draws': f_undef,
            'C_gap_p10': quantile(c_ord, 0.10) if c_ord else None,
            'C_gap_p50': quantile(c_ord, 0.50) if c_ord else None,
            'C_gap_p90': quantile(c_ord, 0.90) if c_ord else None,
            'layer_f_p90': {l: p90s(l) for l in layers},
            'layer_f_samples': f_samples}


def weight_leg(per_layer, *, resamples=RESAMPLES, seed=SEED,
               c_gap_min=C_GAP_MIN, p10_min=P10_MIN,
               positive_layers_min=POSITIVE_LAYERS_MIN,
               exception_layer=EXCEPTION_LAYER, exception_c_gap_min=EXCEPTION_C_GAP_MIN):
    """Full frozen weight-leg decision.

    per_layer: {layer: {'triples':[(A,N,X),...] whole-expert measured triples,
                        'missing_units':[qname,...]}}.
    Returns a dict fragment with status in
    PASS|FAIL|UNMEASURED|PASS-except-L3 plus reasons, primary C_gap,
    per-layer f, bootstrap metadata and the exception readout.
    """
    layers = sorted(per_layer)
    missing = {l: list(per_layer[l]['missing_units']) for l in layers if per_layer[l]['missing_units']}
    unmeasured = sorted(l for l in layers if not per_layer[l]['triples'] and not per_layer[l]['missing_units'])
    if missing or unmeasured:
        return {'status': 'UNMEASURED',
                'reasons': [f'layer {l}: {len(qs)} unmeasured unit(s)' for l, qs in sorted(missing.items())]
                           + [f'layer {l}: no measured experts' for l in unmeasured],
                'missing_units': missing, 'C_gap': None, 'C_gap_reason': 'unmeasured routed units present',
                'raw_totals': None, 'positive_layers': 0,
                'bootstrap': None, 'layer_f': {}, 'clearly_adverse_layers': [],
                'C_gap_without_exception_layer': None, 'C_gap_without_exception_layer_reason': 'unmeasured routed units present'}
    tot = tuple(math.fsum(t[i] for l in layers for t in per_layer[l]['triples']) for i in range(3))
    cg, cg_reason = ratio(*tot)
    boot = bootstrap_experts({l: per_layer[l]['triples'] for l in layers}, resamples=resamples, seed=seed)
    layer_f, undefined = {}, {}
    for l in layers:
        s = tuple(math.fsum(t[i] for t in per_layer[l]['triples']) for i in range(3))
        layer_f[l], undefined[l] = ratio(*s)
    undefined = {l: r for l, r in undefined.items() if r is not None}
    adverse = sorted(l for l in layers if layer_f[l] is not None and layer_f[l] < 0 and boot['layer_f_p90'][l] is not None and boot['layer_f_p90'][l] < 0)
    positive = sum(1 for l in layers if layer_f[l] is not None and layer_f[l] > 0)
    exc = [l for l in layers if l != exception_layer]
    exc_tot = tuple(math.fsum(t[i] for l in exc for t in per_layer[l]['triples']) for i in range(3))
    cg_exc, cg_exc_reason = ratio(*exc_tot)
    reasons = []
    degenerate = boot['c_gap_undefined_draws'] > 0 or any(f_undef[d] for f_undef in boot['layer_f_undefined_draws'].values() for d in f_undef)
    if degenerate:
        reasons.append(f'{boot["c_gap_undefined_draws"]} bootstrap C_gap draw(s) with undefined denominators: {boot["c_gap_undefined_reasons"]}; '
                       + '; '.join(f'layer {l}: {sum(d.values())} undefined draw(s)' for l, d in sorted(boot['layer_f_undefined_draws'].items()) if d)
                       + ' — such draws cannot qualify a gate')
    if cg is None:
        status = 'FAIL'
        reasons.append('primary C_gap undefined: ' + cg_reason)
    else:
        if undefined:
            for l, r in sorted(undefined.items()):
                reasons.append(f'layer {l}: f_l undefined: {r}')
        if cg < c_gap_min:
            reasons.append(f'C_gap {cg!r} < {c_gap_min}')
        if boot['C_gap_p10'] < p10_min:
            reasons.append(f'bootstrap C_gap p10 {boot["C_gap_p10"]!r} < {p10_min}')
        if positive < positive_layers_min:
            reasons.append(f'{positive} positive-f layers < {positive_layers_min}')
        if adverse:
            reasons.append(f'clearly adverse layers (f_l<0 and layer bootstrap p90<0): {adverse}')
        if not undefined and not adverse and not degenerate and cg >= c_gap_min and boot['C_gap_p10'] >= p10_min and positive >= positive_layers_min:
            status = 'PASS'
        elif adverse == [exception_layer] and not degenerate and cg_exc is not None and cg_exc >= exception_c_gap_min and not undefined:
            status = 'PASS-except-L3'
            reasons.append('only the exception layer is clearly adverse and C_gap without it meets the minimum; routed to CEO, not an ordinary PASS')
        else:
            status = 'FAIL'
    return {'status': status, 'reasons': reasons,
            'C_gap': cg, 'C_gap_reason': cg_reason,
            'raw_totals': {'A8S': tot[0], 'new': tot[1], 'EXL3': tot[2]},
            'C_gap_without_exception_layer': cg_exc,
            'C_gap_without_exception_layer_reason': cg_exc_reason,
            'layer_f': layer_f, 'layer_f_undefined': undefined,
            'layer_bootstrap_p90': boot['layer_f_p90'],
            'positive_layers': positive, 'clearly_adverse_layers': adverse,
            'bootstrap': boot}
