#!/usr/bin/env python3
"""Frozen D44 FIT-prefix selector: training-rest pooling, sigma/k grid, selection output.

Single home of the frozen conditioning math (criteria 1-3 of the CEO D44 method,
frozen 2026-10-06T21:15:53Z, method file copied byte-identical to
d44-frozen-method.json beside this module):

  prefix_fold        H_rest = H_fit - X_prefix.T @ X_prefix      (1_selector.training)
  pooled_covariance  sum(raw grams) / sum(actual routed rows)    (3_pool.denominator)
  conditioned_hessian (Hraw + k*poolraw/poolcount) / (ne + k)    (3_pool.formula)
  rawcount_hessian   Hraw + k*poolraw/poolcount                  (raw-count scale handed
                      to the ActivationSource encoder; k pseudo-rows at the pool mean)

Rules implemented here and nowhere else:
  * One 512-row FIT-prefix fold per (layer, role) cell, same rule at every one of the
    six routed layers; validation is the prefix, training is the rest.
  * Selection objective: summed RAW validation SSE (d44.output_sse float64 contraction)
    over experts with FIT count strictly greater than 1024; no per-expert mean weighting.
  * Pool per (layer, role) over raw routed expert moments; denominator is the SUM of
    actual routed rows in the pool, not a global token count; every positive-FIT expert
    is a member including the target itself (self-inclusion); zero-FIT experts are
    excluded; training pools use training-rest moments/counts, the final encoding pool
    uses full-FIT moments/counts and is built in the same pass.
  * Low-count experts are excluded from the choice only: they receive the chosen
    sigma/k and remain in the bar.
  * Sigma is the existing ActivationSource ldlq_sigma knob; no second damping term is
    added and the old removed damping {0.001, 0.01, 0.1} stays removed.
  * Fail closed: a cell with no eligible expert, or a missing fit/prefix dependency,
    refuses rather than picking an arbitrary winner.

Everything executes inside admitted PB payloads; this module only computes and
writes receipts. No HELD scoring, no G3 scoring, no scientific submission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from functools import lru_cache
import stage1 as S

HERE = Path(__file__).resolve().parent
METHOD_FILE = HERE / 'd44-frozen-method.json'
SELECTION_SCHEMA = 'd44.frozen_condition_selection.v1'
POOL_SCHEMA = 'd44.frozen_cell_pool.v2'  # v2: pools persist RAW member sums + counts; v1 stored a mean and consumers redivided (P1 defect, superseded)
GRID_UNIT_SCHEMA = 'd44.frozen_grid_unit.v1'
SELFCHECK_SCHEMA = 'd44.frozen_training_selfcheck.v1'
FROZEN_FIT_MIN = 1024  # 1_selector.objective: fit count strictly greater than 1024
PREFIX_ROWS = 512      # 1_selector.rule: 512-row FIT prefix fold
ROLE_SUFFIX = {'gate': 'gate_proj', 'up': 'up_proj', 'down': 'down_proj'}
REMOVED_SHRINKAGE_K = (1000, 4000)  # old experiment grid, dropped by the frozen method


def method_sha256():
    return S.sha(METHOD_FILE)


@lru_cache(maxsize=1)
def method():
    m = S.load(METHOD_FILE)
    for key in ('1_selector', '2_grid', '3_pool', '7_roster'):
        S.require(key in m, f'frozen method file lacks {key}')
    S.require(m['5_bar_scope'].get('routed_only') is True, 'frozen method must stay routed-only here')
    for dropped in m['2_grid']['removed_damping']:
        S.require(float(dropped) not in [float(x) for x in m['2_grid']['sigma']],
                  'removed damping reentered the sigma grid')
    S.require(not (set(REMOVED_SHRINKAGE_K) & {int(k) for k in m['2_grid']['shrinkage_k']}),
              'old shrinkage k reentered the grid')
    return m


def layers():
    return [int(x) for x in method()['7_roster']['layers']]


def roles():
    return [str(r) for r in method()['7_roster']['roles']]


def sigmas():
    return [float(x) for x in method()['2_grid']['sigma']]


def shrinkage_ks():
    return [int(k) for k in method()['2_grid']['shrinkage_k']]


def grid():
    """Frozen candidate grid, deterministic order: k outer, sigma inner."""
    return [(k, s) for k in shrinkage_ks() for s in sigmas()]


def selection_binding(selection):
    """Hard numeric binding of a selection to the live frozen method.

    The whole-file method_sha256 is a metadata-only drift stamp, never a
    refusal (a D32-style identity wall would reject byte-identical numerics
    over an unrelated byte change). The frozen NUMERICS — sigma/k grid, strict
    fit minimum, fold width, cell roster — plus split/corpus comparability are
    refusals. Returns the drift stamp (None when the method file is unchanged).
    """
    S.require(selection.get('schema') == SELECTION_SCHEMA, 'not a frozen condition selection')
    S.require(selection.get('grid', {}).get('sigma') == sigmas(), 'selection sigma grid differs from the frozen method')
    S.require(selection.get('grid', {}).get('shrinkage_k') == shrinkage_ks(), 'selection k grid differs from the frozen method')
    S.require(selection.get('fit_min_strict') == FROZEN_FIT_MIN, 'selection fit minimum differs from the frozen method')
    S.require(selection.get('fold', {}).get('prefix_rows') == PREFIX_ROWS, 'selection fold width differs from the frozen method')
    required_cells = {(layer, role) for layer in layers() for role in roles()}
    cell_keys = [(int(c['layer']), c['role']) for c in selection.get('cells', [])]
    S.require(len(cell_keys) == len(required_cells) and len(set(cell_keys)) == len(cell_keys),
              'selection requires one unique cell for each layer and role')
    S.require(set(cell_keys) == required_cells, 'selection cell roster differs from the frozen method')
    # A declared-good grid must not authorize an off-grid choice. The original
    # values are validated BEFORE conversion: int(1024.5) would silently become
    # 1024, so k must already be an integer value and sigma finite. Every
    # cell's chosen pair and every per-unit choice must sit inside the frozen
    # 12-pair grid, and each unit's choice must equal its own cell's choice.
    # sigma=0.001, k=1000/4000, fractional k, and mismatched unit choices are
    # refused here, before any encoder invocation.
    frozen = {(k, s) for k, s in grid()}
    def check_pair(raw_k, raw_sigma, where):
        try:
            k_is_int = float(raw_k).is_integer()
        except (TypeError, ValueError):
            k_is_int = False
        S.require(k_is_int, f'{where}: k {raw_k!r} is not an integer value')
        S.require(math.isfinite(float(raw_sigma)), f'{where}: sigma {raw_sigma!r} is not finite')
        pair = (int(raw_k), float(raw_sigma))
        S.require(pair in frozen, f'{where}: chosen {pair} outside the frozen 12-pair grid')
        return pair
    by_unit = {}
    for c in selection.get('cells', []):
        cell_pair = check_pair(c['chosen']['k'], c['chosen']['sigma'], f"{c['layer']}:{c['role']}")
        for q in c.get('unit_roster', []):
            S.require(q not in by_unit, f'{q}: unit sits in two cells; its choice is not unique')
            by_unit[q] = cell_pair
    S.require(set(selection.get('unit_chosen', {})) == set(by_unit),
              'unit choices do not cover the cell rosters exactly (missing or extra units)')
    for q, choice in selection.get('unit_chosen', {}).items():
        unit_pair = check_pair(choice['k'], choice['sigma'], q)
        S.require(unit_pair == by_unit[q], f'{q}: per-unit choice differs from its cell choice')
    live = method_sha256()
    return None if selection.get('method_sha256') == live else {
        'selection_method_sha256': selection.get('method_sha256'), 'live_method_sha256': live}


def pool_method_drift(pool):
    """Metadata-only drift of a pool's method stamp against the live method file."""
    live = method_sha256()
    return None if pool.get('method_sha256') == live else {
        'pool_method_sha256': pool.get('method_sha256'), 'live_method_sha256': live}


def cell_id(layer, role):
    return f'{int(layer)}:{role}'


def layer_dir(layer):
    return 'L%03d' % int(layer)


def role_for_qname(q):
    """Routed role of an expert qname; refuses dense/shared or unknown names."""
    base = q.removesuffix('.weight')
    if '.experts.' not in base:
        raise ValueError(f'{q}: not a routed expert qname')
    for role, suffix in ROLE_SUFFIX.items():
        if base.endswith('.' + suffix):
            return role
    raise ValueError(f'{q}: no frozen routed role suffix')


def routed_cells():
    """{(layer, role): [row, ...]} over the actual decision-layer routed roster."""
    m, _, routed, _ = S.population()
    cells = {(layer, role): [] for layer in layers() for role in roles()}
    for r in routed:
        layer, role = int(r['layer']), role_for_qname(r['qname'])
        S.require((layer, role) in cells, f"{r['qname']}: routed unit outside the frozen cell roster")
        cells[(layer, role)].append(r)
    for cell, rows in cells.items():
        S.require(bool(rows), f'{cell}: empty routed cell roster')
    return cells


# ---------------------------------------------------------------- frozen math, single home
def prefix_fold(Hfit, Xprefix, nfit):
    """1_selector.training: H_rest = H_fit - X_prefix.T @ X_prefix, in float64.

    Returns (H_rest float64, n_rest) with n_rest = nfit - prefix rows."""
    import torch
    Hfit = torch.as_tensor(Hfit)
    Xprefix = torch.as_tensor(Xprefix)
    nprefix = int(Xprefix.shape[0])
    S.require(0 < nprefix <= int(nfit), 'prefix must be a nonempty subset of the fit rows')
    S.require(Hfit.ndim == 2 and Hfit.shape[0] == Hfit.shape[1] == Xprefix.shape[1],
              'prefix fold geometry mismatch')
    H64 = Hfit.to(dtype=torch.float64)
    X64 = Xprefix.to(dtype=torch.float64)
    Hrest = H64 - X64.T @ X64
    return Hrest, int(nfit) - nprefix


def pooled_covariance(rawHs, counts):
    """3_pool.denominator: sum(raw member grams) / sum(actual member rows).

    The single home of the pool-MEAN definition. Nothing on the persistence or
    encoding path consumes its output: pool files persist raw sums plus counts
    and the frozen division happens only in rawcount_hessian/conditioned_hessian.
    Zero-count members must carry an exactly zero gram and contribute nothing.
    Returns (mean gram float64, summed actual rows)."""
    import torch
    S.require(len(rawHs) == len(counts) and bool(counts), 'pool needs members with counts')
    total, acc = 0, None
    for h, n in zip(rawHs, counts):
        n = int(n)
        S.require(n >= 0, 'negative pool member count')
        h = torch.as_tensor(h)
        if n == 0:
            S.require(bool((h == 0).all().item()), 'zero-count pool member carries nonzero moment')
            continue
        h64 = h.to(dtype=torch.float64)
        acc = h64.clone() if acc is None else acc + h64
        total += n
    S.require(total > 0, 'empty pool: no positive-count member')
    return acc / float(total), total


def rawcount_hessian(Hraw, ne, poolraw_sum, poolcount, k):
    """Raw-count-scale shrinkage sum: Hraw + k * (poolraw_sum / poolcount).

    ``poolraw_sum`` is the RAW SUM over the pooled members' grams — the
    ``gram_raw_sum`` a v2 pool file persists — and ``poolcount`` its summed
    actual routed rows. This function and conditioned_hessian are the ONLY
    places that division happens. The result is the matrix handed to the
    ActivationSource encoder, restoring the raw count (row-sum) scale the
    served path uses; k=0 reduces to Hraw bitwise."""
    import torch
    ne, poolcount = int(ne), int(poolcount)
    S.require(ne > 0 and poolcount > 0 and float(k) >= 0.0,
              'conditioning refused by name: zero-FIT expert (ne=0) or empty pool makes '
              '(Hraw + k*poolraw_sum/poolcount)/(ne+k) undefined; unmeasured, never invented')
    poolmean = torch.as_tensor(poolraw_sum).to(dtype=torch.float64) / float(poolcount)
    return torch.as_tensor(Hraw).to(dtype=torch.float64) + float(k) * poolmean


def conditioned_hessian(Hraw, ne, poolraw_sum, poolcount, k):
    """3_pool.formula: (Hraw + k*poolraw_sum/poolcount) / (ne + k)."""
    ne, k = int(ne), int(k)
    return rawcount_hessian(Hraw, ne, poolraw_sum, poolcount, k) / float(ne + k)


def fit_provenance(cap, fp):
    prov = dict(cap.split['roles']['fit']['provenance'])
    prov.update(split_sha256=fp['split_sha256'], source_draw=fp['draw'],
                capture_role='fit', capture_file_sha256=fp['sha256'])
    return prov


def load_pool(path, expect_cell=None):
    import torch
    p = Path(path)
    S.require(p.exists(), f'{p}: frozen cell pool missing')
    receipt_path = p.with_suffix('.receipt.json')
    S.require(receipt_path.exists(), f'{receipt_path}: pool receipt missing')
    receipt = S.load(receipt_path)
    S.require(receipt['pool_sha256'] == S.sha(p), f'{p}: pool bytes differ from receipt digest')
    pool = torch.load(p, map_location='cpu', weights_only=True)
    S.require(pool['schema'] == POOL_SCHEMA, f'{p}: not a frozen cell pool')
    for side in ('fit', 'rest'):
        S.require('gram_raw_sum' in pool[side] and int(pool[side]['count']) > 0,
                  f'{p}: pool side {side} lacks gram_raw_sum or a positive count')
    if expect_cell is not None:
        S.require((int(pool['layer']), pool['role']) == tuple(expect_cell), f'{p}: wrong cell pool')
    return pool, receipt


# ---------------------------------------------------------------- PB payload phases
def _write_pool(a, layer, role):
    import torch
    torch, _, _, _ = S.setup()
    started = time.monotonic()
    cells = routed_cells()
    rows = cells[(int(layer), role)]
    cap = S.Captures(a.capture)
    cached = S.population()[1]
    fit_acc = rest_acc = None
    fit_total = rest_total = 0
    members = 0
    member_nfit, member_nrest, zero_fit = {}, {}, []
    for r in rows:
        q = r['qname']
        fit_entry = cap.units[q][1].get('fit')
        S.require(fit_entry is not None, f'{q}: FIT entry absent from the capture manifest (unmeasured; never mapped to zero)')
        manifest_nfit = int(fit_entry['count'])
        S.require(manifest_nfit >= 0, f'{q}: invalid negative FIT count (never zero)')
        if manifest_nfit == 0:
            # Frozen 3_pool.zero_FIT_experts: only an explicitly recorded actual
            # count==0 is excluded from pool membership. Roster entries stay at
            # 0; no H is fabricated.
            zero_fit.append(q)
            member_nfit[q], member_nrest[q] = 0, 0
            continue
        width = cached[q]['identity']['source']['shape'][1]
        d, _ = cap.get(q, 'fit', width)
        Hfit, nfit, Xp = d['hessian'], int(d['count']), d['inputs']
        S.require(nfit == manifest_nfit and nfit > 0, f'{q}: capture count disagrees with the manifest count')
        Hrest, nrest = prefix_fold(Hfit, Xp, nfit)
        # Per-expert fold identity: rest + prefix == fit, on the raw count scale.
        X64 = Xp.to(dtype=torch.float64)
        Hfit64 = Hfit.to(dtype=torch.float64)
        err = float((Hrest + X64.T @ X64 - Hfit64).abs().max().item())
        rel = err / max(float(Hfit64.abs().max().item()), 1e-30)
        S.require(rel <= 1e-5, f'{q}: prefix fold does not reconstruct the fit gram (rel {rel:g})')
        fit_acc = Hfit64 if fit_acc is None else fit_acc + Hfit64
        rest_acc = Hrest if rest_acc is None else rest_acc + Hrest
        fit_total, rest_total = fit_total + nfit, rest_total + nrest
        members += 1
        member_nfit[q], member_nrest[q] = nfit, nrest
        del Hfit, Hrest, Hfit64, X64
    S.require(members > 0, f'{cell_id(layer, role)}: no positive-FIT expert; pool refused')
    S.require(fit_total == sum(member_nfit.values()) and rest_total == sum(member_nrest.values()),
              f'{cell_id(layer, role)}: pool count sums disagree with member counts')
    # Persist RAW member sums plus counts (schema v2). The frozen division
    # poolraw_sum/poolcount happens only inside rawcount_hessian/conditioned_hessian;
    # v1 persisted pooled_covariance's mean and consumers redivided (P1, superseded).
    count_fit, count_rest = fit_total, rest_total
    payload = {'schema': POOL_SCHEMA, 'layer': int(layer), 'role': role,
               'method_sha256': method_sha256(), 'split_sha256': cap.split_sha256,
               'fit': {'gram_raw_sum': fit_acc, 'count': int(count_fit), 'members': members},
               'rest': {'gram_raw_sum': rest_acc, 'count': int(count_rest), 'members': members},
               'member_nfit': member_nfit, 'member_nrest': member_nrest,
               'zero_fit_excluded': zero_fit,
               'zero_fit_membership': 'excluded members keep roster entries at count 0; chosen settings still apply, no silent unit drop',
               'self_inclusion': 'every positive-FIT expert of the cell, the target included, is a member',
               'denominator': 'sum of actual routed rows over pooled members; never a global token count',
               'countsum': {'fit_total': int(count_fit), 'rest_total': int(count_rest)},
               'action_key': os.environ.get('PRISMABUILD_ACTION_KEY'),
               'elapsed_seconds': time.monotonic() - started}
    out = Path(a.root) / 'pool' / (layer_dir(layer) + f'-{role}.pool.pt')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    receipt = {'schema': POOL_SCHEMA, 'layer': int(layer), 'role': role, 'pool_path': str(out),
               'pool_sha256': S.sha(out), 'pool_bytes': out.stat().st_size,
               'fit_count': int(count_fit), 'rest_count': int(count_rest),
               'members': members, 'zero_fit_excluded': zero_fit,
               'method_sha256': method_sha256(), 'split_sha256': cap.split_sha256,
               'action_key': os.environ.get('PRISMABUILD_ACTION_KEY'),
               'elapsed_seconds': payload['elapsed_seconds']}
    S.save(out.with_suffix('.receipt.json'), receipt)
    print(json.dumps({'cell': cell_id(layer, role), 'pool': str(out), 'fit_count': int(count_fit),
                      'rest_count': int(count_rest), 'seconds': payload['elapsed_seconds']}), flush=True)


def pool(a):
    if a.batch:
        for t in S.load(a.batch)['tasks']:
            p = t['payload']
            _write_pool(a, int(p['layer']), p['role'])
        return
    S.require(a.layer is not None and a.role is not None, '--layer and --role (or --batch) required')
    S.require(a.role in roles(), f'{a.role}: role outside the frozen roster')
    _write_pool(a, a.layer, a.role)


def evaluate_expert(q, row, cached, cap, device, enc, pool_cache, *, dry_run=False):
    """One expert, the whole frozen grid: training-rest conditioning, full FIT is a later phase."""
    import torch
    D44 = import_d44()
    cell = (int(row['layer']), role_for_qname(q))
    # Authoritative count pre-read from the parsed layer manifest. A missing
    # entry is UNMEASURED (refused by name, never mapped to zero); only an
    # explicitly recorded count==0 takes the zero-FIT roster path.
    fit_entry = cap.units[q][1].get('fit')
    S.require(fit_entry is not None, f'{q}: FIT entry absent from the capture manifest (unmeasured; never mapped to zero)')
    manifest_nfit = int(fit_entry['count'])
    S.require(manifest_nfit >= 0, f'{q}: invalid negative FIT count (never zero)')
    if manifest_nfit == 0:
        return {'qname': q, 'layer': cell[0], 'role': cell[1], 'nfit': 0, 'nprefix': 0, 'nrest': 0,
                'eligible': False, 'grid_rows': [], 'zero_fit': True, 'fit': None,
                'split_sha256': cap.split_sha256, 'method_sha256': method_sha256(),
                'note': 'explicit zero-FIT: excluded from choice and pool; roster retained, no H fabricated',
                'action_key': os.environ.get('PRISMABUILD_ACTION_KEY'), 'device': device, 'dry_run': dry_run}
    width = cached[q]['identity']['source']['shape'][1]
    d, fp = cap.get(q, 'fit', width)
    Hfit, nfit, Xp = d['hessian'], int(d['count']), d['inputs']
    Hrest, nrest = prefix_fold(Hfit, Xp, nfit)
    Hval = Xp.to(dtype=torch.float64).T @ Xp.to(dtype=torch.float64)  # raw validation gram
    eligible = nfit > FROZEN_FIT_MIN
    pool, receipt = load_pool(pool_cache[cell], expect_cell=cell)
    S.require(pool['split_sha256'] == cap.split_sha256, f'{q}: pool and capture splits differ')
    # The pool's method stamp is metadata-only drift; split/member comparability stays hard.
    method_drift = pool_method_drift(pool)
    S.require(pool['member_nfit'][q] == nfit, f'{q}: expert fit count drifted since the pool was built')
    grid_rows, prov = [], fit_provenance(cap, fp)
    if eligible:
        w, stamp = S.source(row, cached, device)
        poolraw, poolcount = pool['rest']['gram_raw_sum'], int(pool['rest']['count'])
        for k, sigma in grid():
            Hc = rawcount_hessian(Hrest, nrest, poolraw, poolcount, k).to(dtype=torch.float32).contiguous()
            kw = S.fixed_kwargs(enc, q, Hc, prov, cached, device, ldlq_sigma=sigma)
            if dry_run:
                grid_rows.append({'k': k, 'sigma': sigma, 'kwargs_built': True})
                del Hc, kw
                continue
            [(render, blob)] = enc.encode([w], [kw], 'NONE')
            del blob
            sse = D44.output_sse(w, render, Hval)  # shared float64 contraction, RAW sum
            S.require(math.isfinite(sse) and sse >= 0, f'{q}: invalid validation SSE')
            grid_rows.append({'k': k, 'sigma': sigma, 'validation_sse': sse})
            del render, Hc, kw
        del w
    return {'qname': q, 'layer': cell[0], 'role': cell[1], 'nfit': nfit,
            'nprefix': int(Xp.shape[0]), 'nrest': nrest, 'eligible': eligible,
            'grid_rows': grid_rows, 'fit': fp, 'split_sha256': cap.split_sha256,
            'method_sha256': method_sha256(), 'method_drift': method_drift, 'pool_sha256': receipt['pool_sha256'],
            'action_key': os.environ.get('PRISMABUILD_ACTION_KEY'), 'device': device, 'dry_run': dry_run}


def import_d44():
    import d44 as D
    return D


def select_unit(a):
    import torch
    torch, CD2, _, _ = S.setup()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    m, cached, _, _ = S.population()
    by = {r['qname']: r for r in m['rows']}
    cap = S.Captures(a.capture)
    names = [t['payload']['qname'] for t in S.load(a.batch)['tasks']] if a.batch else [a.qname]
    S.require(bool(names) and all(names), 'select-unit needs explicit qnames')
    pool_cache = {}
    for q in names:
        S.require(q in by, f'{q}: outside the actual body roster')
        cell = (int(by[q]['layer']), role_for_qname(q))
        if cell not in pool_cache:
            pool_cache[cell] = Path(getattr(a, 'pool_root', None) or a.root) / 'pool' / (layer_dir(cell[0]) + f'-{cell[1]}.pool.pt')
    enc = CD2.Encoder(S.FMT)
    for q in names:
        row = by[q]
        result = evaluate_expert(q, row, cached, cap, a.device, enc, pool_cache, dry_run=getattr(a, 'dry_run', False))
        directory = 'preflight' if result['dry_run'] else 'grid'
        out = Path(a.root) / directory / (import_d44().stem(q) + '.json')
        schema = 'd44.selector_entry_preflight.v1' if result['dry_run'] else GRID_UNIT_SCHEMA
        S.save(out, {'schema': schema, **result})
        print(json.dumps({'qname': q, 'eligible': result['eligible'],
                          'grid_rows': len(result['grid_rows']), 'out': str(out)}), flush=True)
        del result
    S.require(time.monotonic() - started < 1800, 'select-unit exceeded the real 30-minute hard bound')
    print(json.dumps({'units': len(names), 'elapsed_seconds': time.monotonic() - started}), flush=True)


def objective_total(grid_rows, eligible_names, key='validation_sse'):
    """Summed RAW validation SSE over eligible experts; no per-expert mean weighting."""
    totals = {}
    for r in grid_rows:
        q = r['qname']
        if q not in eligible_names:
            continue
        for row in r['grid_rows']:
            cell_key = (int(row['k']), float(row['sigma']))
            totals[cell_key] = totals.get(cell_key, 0.0) + float(row[key])
    return totals


def reduce(a):
    S.setup()  # pins tessera/PrismaQuant paths before any owner import
    started = time.monotonic()
    m = method()
    cells = routed_cells()
    cap = S.Captures(a.capture)
    S.require(a.out is not None and not Path(a.out).exists(), '--out required and must not already exist')
    out_cells, unit_chosen, method_drifts = [], {}, []
    for (layer, role), rows in sorted(cells.items()):
        pool_path = Path(a.root) / 'pool' / (layer_dir(layer) + f'-{role}.pool.pt')
        pool, receipt = load_pool(pool_path, expect_cell=(layer, role))
        S.require(pool['split_sha256'] == cap.split_sha256, f'{cell_id(layer, role)}: pool split differs')
        eligible, lowcount, grid_files = [], [], []
        for r in rows:
            q = r['qname']
            path = Path(a.root) / 'grid' / (import_d44().stem(q) + '.json')
            S.require(path.exists(), f'{cell_id(layer, role)}: missing grid unit {q}; refusing to select')
            g = S.load(path)
            S.require(g['qname'] == q and g['schema'] == GRID_UNIT_SCHEMA, f'{q}: wrong grid unit record')
            S.require(g['split_sha256'] == cap.split_sha256, f'{q}: grid unit from another split')
            # Grid-unit method stamps are metadata-only drift; grid completeness
            # against the live grid() below is the hard method binding.
            S.require(int(g['nfit']) == int(pool['member_nfit'][q]), f'{q}: fit count drifted against the pool')
            drifted = g.get('method_drift') or pool_method_drift({'method_sha256': g.get('method_sha256')})
            if drifted:
                method_drifts.append({'qname': q, 'grid': drifted})
            grid_files.append(g)
            (eligible if g['eligible'] else lowcount).append(q)
        if not eligible:
            raise ValueError(f'{cell_id(layer, role)}: no expert with FIT count > {FROZEN_FIT_MIN}; fail closed')
        want = {(k, s) for k, s in grid()}
        for g in grid_files:
            if not g['eligible']:
                S.require(not g['grid_rows'], f"{g['qname']}: ineligible expert carries grid rows")
                continue
            have = {(int(row['k']), float(row['sigma'])) for row in g['grid_rows']}
            S.require(have == want, f"{g['qname']}: incomplete candidate grid {sorted(have)} vs {sorted(want)}")
        totals = objective_total(grid_files, set(eligible))
        S.require(set(totals) == want, f'{cell_id(layer, role)}: incomplete candidate grid {sorted(totals)} vs {sorted(want)}')
        for key, total in totals.items():
            S.require(math.isfinite(total) and total >= 0, f'{cell_id(layer, role)}: invalid grid total {key}')
        (chosen_k, chosen_sigma) = min(sorted(totals), key=lambda ks: (totals[ks], ks[0], ks[1]))
        candidate_grid = [{'k': k, 'sigma': s, 'validation_sse_summed_raw': totals[(k, s)]}
                          for k, s in sorted(totals)]
        pool_fit_digest = {'path': str(pool_path), 'sha256': receipt['pool_sha256'],
                           'bytes': receipt['pool_bytes'], 'count': int(pool['fit']['count']),
                           'members': int(pool['fit']['members'])}
        for r in rows:
            unit_chosen[r['qname']] = {'sigma': chosen_sigma, 'k': chosen_k}
        out_cells.append({'layer': layer, 'role': role, 'chosen': {'sigma': chosen_sigma, 'k': chosen_k},
                          'candidate_grid': candidate_grid, 'eligible_expert_count': len(eligible),
                          'eligible_qnames': sorted(eligible),
                          'unit_roster': sorted(r['qname'] for r in rows),
                          'lowcount_excluded_from_choice': [{'qname': q, 'nfit': int(pool['member_nfit'][q])}
                                                            for q in sorted(lowcount)],
                          'pool_fit': pool_fit_digest,
                          'pool_rest_count': int(pool['rest']['count']),
                          'countsum': {'fit_total': int(pool['fit']['count']), 'rest_total': int(pool['rest']['count']),
                                       'member_fit_sum': int(sum(pool['member_nfit'].values())),
                                       'member_rest_sum': int(sum(pool['member_nrest'].values()))},
                          'self_inclusion': pool['self_inclusion'],
                          'denominator': pool['denominator'],
                          'zero_fit_excluded': pool['zero_fit_excluded']})
    selection = {'schema': SELECTION_SCHEMA, 'method_sha256': method_sha256(),
                 'split_sha256': cap.split_sha256,
                 'objective': m['1_selector']['objective'],
                 'scope': m['1_selector']['scope'],
                 'fold': {'rule': m['1_selector']['rule'], 'prefix_rows': PREFIX_ROWS,
                          'training': m['1_selector']['training'], 'validation': m['1_selector']['validation']},
                 'fit_min_strict': FROZEN_FIT_MIN,
                 'grid': {'sigma': sigmas(), 'shrinkage_k': shrinkage_ks(),
                          'removed_damping': m['2_grid']['removed_damping'],
                          'removed_shrinkage_k': list(REMOVED_SHRINKAGE_K),
                          'sigma_is': 'the existing ActivationSource ldlq_sigma knob; no second damping'},
                 'pooling': {'formula': m['3_pool']['formula'], 'denominator': m['3_pool']['denominator'],
                             'self_inclusion': m['3_pool']['self_inclusion'],
                             'zero_fit_exclusion': m['3_pool']['zero_FIT_experts'],
                             'training_pool': 'training-rest moments over actual rest routed-row counts; validation prefixes kept out of every member training moment',
                             'final_pool': 'full-FIT moments over actual FIT routed-row counts, built in the same pass'},
                 'final': {'reencode': m['1_selector']['final'], 'heldout_scored': False,
                           'scientific_selection_submitted': False},
                 'cells': out_cells, 'unit_chosen': unit_chosen,
                 'method_drift_at_reduce': method_drifts,
                 'fit_only_proof': {'heldout_grams_used': 0,
                                    'statement': 'selection consumed FIT-role captures only; the held-out role was never opened by the selector'},
                 'lowcount_policy': 'excluded from the choice only; encoded with the chosen sigma/k and kept in the bar',
                 'reduced_seconds': time.monotonic() - started,
                 'action_key': os.environ.get('PRISMABUILD_ACTION_KEY')}
    S.save(a.out, selection)
    print(json.dumps({'out': str(a.out), 'cells': len(out_cells),
                      'chosen': {f'{c["layer"]}:{c["role"]}': c['chosen'] for c in out_cells}}), flush=True)


# ---------------------------------------------------------------- selfcheck (deterministic invariants)
def selfcheck(a):
    import torch
    S.setup()  # pins tessera/PrismaQuant paths before any owner import
    torch.manual_seed(0)
    checks, failures = {}, []

    def record(name, fn):
        try:
            detail = fn()
            checks[name] = {'passed': True, 'detail': detail}
        except Exception as exc:  # noqa: BLE001 - the check IS the observation
            checks[name] = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
            failures.append(name)

    def t_case():
        # One-expert pool invariant: with the target as the only member (pool raw
        # sum == Hraw, poolcount == ne), conditioned_hessian == Hraw/ne for EVERY
        # k in the grid. float64 eps = 2**-52 ~ 2.22e-16; the k>0 path costs three
        # correctly rounded ops (pool division, k-scale+add, final divide) beyond
        # the single division of Hraw/ne, so the derived bound is 6*eps. k=0 is a
        # single division and must match bitwise.
        eps = sys.float_info.epsilon
        bound = 6 * eps
        H = torch.randn(8, 8, dtype=torch.float64).abs()
        H = H + H.T
        ne = 37
        worst = 0.0
        for k, _ in grid():
            if k == 0:
                continue
            got = conditioned_hessian(H, ne, H, ne, k)  # raw-sum inputs: the pool is the target itself
            want = H / float(ne)
            worst = max(worst, float((got - want).abs().max().item()) / float(want.abs().max().item()))
        S.require(worst <= bound, f'one-expert pool error {worst:g} exceeds derived float64 bound {bound:g}')
        S.require(torch.equal(conditioned_hessian(H, ne, H, ne, 0), H / float(ne)),
                  'k=0 must reduce bitwise to Hraw/ne')
        return {'max_relative_error_k_positive': worst, 'derived_bound': bound, 'float64_eps': eps,
                'k0_bitwise_exact': True,
                'note': 'observed k>0 error is numerical (correctly rounded), not bitwise equality'}

    # The failing-before line is an OBSERVED PB run, not a reconstructed
    # variant: PB action 01b7c4c9238a9e45d997e01cd043fb451f00c9f62a0e52fc160d3683071f5810
    # executed the exact rejected source (CAS git bundle 75104bd2, commit
    # 619f7c80: v1 pooled-mean persistence under 'gram' + v1 consumer
    # redivision) on 3 real L3 up experts and measured relative error
    # 0.19540154460560835 against the frozen formula; the same fixture through
    # the corrected code diverged 0.0. No reconstructed wrong-variant proof is
    # kept here; the red line lives in that action's terminal log.


    def diagnostic_unweighted_pool_denominator():
        # Manufactured diagnostic (NOT an observed failure): an unweighted
        # mean-of-grams pool violates the one-expert invariant.
        H = torch.randn(6, 6, dtype=torch.float64).abs()
        H = H + H.T
        H2 = torch.randn(6, 6, dtype=torch.float64).abs()
        H2 = H2 + H2.T
        ne = 11
        mean_of_grams = (H + H2) / 2.0           # WRONG: denominator = 2 members
        row_weighted, _ = pooled_covariance([H, H2], [ne, 5 * ne])
        S.require(not torch.allclose(mean_of_grams, row_weighted, rtol=1e-9),
                  'mean-of-grams accidentally equals row-weighted pool')
        got = conditioned_hessian(H, ne, mean_of_grams * 2.0, ne + 5 * ne, 4096)  # unweighted sum fed as raw sum
        want = H / float(ne)
        rel = float((got - want).abs().max().item()) / float(want.abs().max().item())
        S.require(rel > 1e-6, 'unweighted pool diagnostic unexpectedly passed the invariant')
        return {'relative_error_of_wrong_variant': rel, 'manufactured': True}

    def diagnostic_target_exclusion():
        # Manufactured diagnostic (NOT an observed failure): a pool that drops
        # the target cannot return Hraw/ne at k>0 for a one-expert cell.
        H = torch.randn(5, 5, dtype=torch.float64).abs()
        H = H + H.T
        ne = 7
        got = conditioned_hessian(H, ne, torch.zeros_like(H), 1, 1024)  # target excluded -> zero raw sum
        want = H / float(ne)
        rel = float((got - want).abs().max().item()) / float(want.abs().max().item())
        S.require(rel > 1e-6, 'target-excluding pool diagnostic unexpectedly passed')
        return {'relative_error_of_wrong_variant': rel, 'manufactured': True}

    def zero_fit_conditioning_refused():
        # ne=0 (or an empty pool) makes the frozen formula undefined; the answer
        # is a named refusal, never a number.
        H = torch.randn(4, 4, dtype=torch.float64).abs()
        H = H + H.T
        refused = []
        for ne, k, pc in ((0, 0, 5), (0, 1024, 5), (37, 0, 0)):
            try:
                conditioned_hessian(H, ne, H, pc, k)
            except ValueError as exc:
                S.require('refused by name' in str(exc) and 'unmeasured' in str(exc),
                          f'unexpected refusal text: {exc}')
                refused.append((ne, k, pc))
                continue
            raise AssertionError(f'ne={ne},k={k},poolcount={pc}: undefined conditioning was not refused')
        return {'refused_cases': refused, 'behavior': 'unmeasured by name; no numeric answer invented'}

    def serialized_pool_round_trip():
        # Actual serialized pool -> actual consumer: accumulate raw sums the way
        # _write_pool does, torch.save + receipt, load_pool, condition members
        # with UNEQUAL counts, and a one-expert cell; compare against the direct
        # frozen formula.
        import tempfile
        means = [torch.randn(5, 5, dtype=torch.float64) for _ in range(3)]
        counts = [10, 990, 7]  # unequal on purpose
        raws = [m * float(n) for m, n in zip(means, counts)]
        acc = None
        total = 0
        for h, n in zip(raws, counts):  # same accumulation shape as _write_pool
            acc = h.clone() if acc is None else acc + h
            total += n
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'L000-up.pool.pt'
            with path.open('xb') as f:
                import torch as _torch
                _torch.save({'schema': POOL_SCHEMA, 'layer': 0, 'role': 'up', 'split_sha256': 'fixture',
                             'fit': {'gram_raw_sum': acc, 'count': int(total), 'members': 3},
                             'rest': {'gram_raw_sum': acc.clone(), 'count': int(total), 'members': 3}}, f)
            receipt = {'schema': POOL_SCHEMA, 'pool_path': str(path), 'pool_sha256': S.sha(path),
                       'pool_bytes': path.stat().st_size}
            S.save(path.with_suffix('.receipt.json'), receipt)
            pool, loaded_receipt = load_pool(path, expect_cell=(0, 'up'))
            S.require(torch.equal(pool['rest']['gram_raw_sum'], acc), 'serialized raw sum lost bits')
            worst = 0.0
            for m, n in zip(means, counts):
                H = m * float(n)
                got = conditioned_hessian(H, n, pool['rest']['gram_raw_sum'], int(pool['rest']['count']), 4096)
                want = (H + 4096.0 * (acc / float(total))) / float(n + 4096)
                worst = max(worst, float((got - want).abs().max().item()))
            S.require(worst == 0.0, 'serialized consumer path diverged from the direct formula')
            # One-expert serialized cell: target is the only member.
            H1 = means[0] * 37.0
            pool1 = {'rest': {'gram_raw_sum': H1.clone(), 'count': 37}}
            got1 = conditioned_hessian(H1, 37, pool1['rest']['gram_raw_sum'], 37, 1024)
            rel = float((got1 - H1 / 37.0).abs().max().item()) / float((H1 / 37.0).abs().max().item())
            S.require(rel <= 6 * sys.float_info.epsilon, f'one-expert serialized round trip exceeds 6*eps ({rel:g})')
        return {'members': 3, 'unequal_counts': counts, 'max_abs_divergence': worst,
                'one_expert_relative_error': rel}

    def fold_identity():
        W, ne = 32, 600  # ne > PREFIX_ROWS so the prefix is a strict subset
        X = torch.randn(ne, W, dtype=torch.float64)
        Hfit = X.T @ X
        Hrest, nrest = prefix_fold(Hfit.to(dtype=torch.float32), X[:PREFIX_ROWS].to(dtype=torch.float32), ne)
        err = float((Hrest + X[:PREFIX_ROWS].T @ X[:PREFIX_ROWS] - Hfit).abs().max().item())
        S.require(nrest == ne - PREFIX_ROWS, 'rest row count wrong')
        # Hfit reaches prefix_fold at capture dtype float32: the round trip is
        # exact up to that input quantization.
        rel = err / float(Hfit.abs().max().item())
        S.require(rel <= 1e-5, f'fold round trip exceeds float32 input quantization (rel {rel:g})')
        return {'max_abs_error': err, 'relative_error': rel, 'nrest': nrest}

    def pool_denominator():
        # Members hand pooled_covariance their RAW grams (already row sums):
        # G1 over 10 rows, G2 over 990 rows -> pool mean (10*G1 + 990*G2)/1000.
        G1 = torch.randn(4, 4, dtype=torch.float64)
        G2 = torch.randn(4, 4, dtype=torch.float64)
        H1, H2 = G1 * 10.0, G2 * 990.0  # raw member grams
        got, total = pooled_covariance([H1, H2], [10, 990])
        want = (H1 + H2) / 1000.0
        S.require(total == 1000 and torch.allclose(got, want, rtol=0, atol=1e-12),
                  'pool covariance is not sum(grams)/sum(rows)')
        S.require(torch.allclose(got, (G1 * 10.0 + G2 * 990.0) / 1000.0, rtol=0, atol=1e-12),
                  'pool denominator is not the summed actual rows')
        return {'total_rows': total}

    def objective_raw_sum():
        # The selector objective sums RAW SSE; per-expert mean weighting must differ.
        rows = [{'qname': 'a', 'grid_rows': [{'k': 0, 'sigma': 1.0, 'validation_sse': 100.0}]},
                {'qname': 'b', 'grid_rows': [{'k': 0, 'sigma': 1.0, 'validation_sse': 3.0}]}]
        raw = objective_total(rows, {'a', 'b'})[(0, 1.0)]
        weighted = 100.0 / 2.0 + 3.0 / 2.0
        S.require(raw == 103.0, 'objective is not the raw sum')
        S.require(abs(raw - weighted) > 1e-9, 'raw sum is indistinguishable from mean weighting in the fixture')
        return {'raw_sum': raw, 'mean_weighted_would_be': weighted}

    def role_map():
        pairs = {'model.language_model.layers.3.mlp.experts.0.up_proj': 'up',
                 'model.language_model.layers.44.mlp.experts.287.down_proj': 'down',
                 'model.language_model.layers.40.mlp.experts.17.gate_proj': 'gate'}
        for q, role in pairs.items():
            S.require(role_for_qname(q) == role, f'{q}: role map wrong')
        for bad in ('model.language_model.layers.3.mlp.down_proj', 'model.layers.3.mlp.experts.0.something'):
            try:
                role_for_qname(bad)
            except ValueError:
                continue
            raise AssertionError(f'{bad}: non-routed name accepted')
        return {'mapped': len(pairs)}

    def grid_config():
        m = method()
        S.require(grid() == [(k, s) for k in (0, 1024, 4096, 16384) for s in (1.0, 3.0, 10.0)],
                  'grid does not match the frozen user grid')
        S.require(float(m['2_grid']['removed_damping'][0]) == 0.001, 'removed damping list changed')
        return {'sigma': sigmas(), 'k': shrinkage_ks(), 'candidates': len(grid())}

    def sigma_single_knob():
        # Sigma must flow through the EXISTING ldlq_sigma owner and add no second
        # damping: regularize_hessian(H, s) == H + s*mean(diag)*I exactly, and the
        # only cached kwarg the override touches is ldlq_sigma.
        from tessera.compensate import regularize_hessian
        H = torch.randn(12, 12, dtype=torch.float32).abs()
        H = H + H.T + 0.1 * torch.eye(12)
        for s in sigmas():
            reg = regularize_hessian(H.clone(), sigma_reg=float(s))
            want = H + float(s) * float(H.diagonal().mean()) * torch.eye(12)
            S.require(torch.allclose(reg, want, rtol=1e-6), f'ldlq_sigma {s} is not the single damping owner')
        _, cached, _, _ = S.population()
        base = cached[_first_routed_qname()]['identity']['calibration']['settings']
        S.require(set(base) == {'ldlq_sigma', 'ldlq_block', 'refit_objective', 'refit_objective_trailing',
                                'refit_reach_floor', 'refit_gauss_seidel', 'hessian'},
                  'cached producer settings shape changed')
        S.require(base['ldlq_block'] == 32, 'ldlq_block drifted from the frozen block 32')
        return {'cached_ldlq_sigma': base['ldlq_sigma'], 'override_grid': sigmas()}

    def encoder_seam():
        # Real CD2 encoder seam on a deterministic fixture at an override sigma:
        # exercises the ACTUAL consumer path fixed_kwargs(ldlq_sigma=...) ->
        # encode -> wire_facts with the raw-sum pool representation the fixed
        # code persists. Fixture weights; not a replacement weight. This check
        # validates the knob/seam only, never the pool representation.
        torch, CD2, _, _ = S.setup()
        _, cached, _, dense = S.population()
        q = 'd44_training.seam_fixture'
        enc = CD2.Encoder(S.FMT)
        w = torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128).sin().to(torch.bfloat16) / 16
        # Generic SPD Hessian with off-diagonal mass: a uniform-diagonal H would
        # make LDL compensation scale-uniform and sigma legitimately inert.
        Gf = torch.randn(128, 32, generator=torch.Generator().manual_seed(7))
        Hfit = (Gf @ Gf.T / 32.0 + 0.5 * torch.eye(128)).to(dtype=torch.float32)
        Xp = torch.eye(128, dtype=torch.float32)[:32]
        Hrest, nrest = prefix_fold(Hfit, Xp, 64)
        # One-member fixture: the raw sum IS Hrest. pooled_covariance's mean
        # must never enter the raw-sum consumer (that was the v1 defect).
        pool_raw, pool_count = Hrest.clone(), nrest
        blob_shas = {}
        for s in (1.0, 3.0):
            Hc = rawcount_hessian(Hrest, nrest, pool_raw, pool_count, 1024).to(dtype=torch.float32).contiguous()
            fixture_prov = {'fixture': 'deterministic CPU only, fit role', 'fit_tokens': 64,
                            'text_sha256': S.blob_sha(b'd44_training seam fixture text'),
                            'fit_ids_sha256': S.blob_sha(torch.arange(64, dtype=torch.int32).numpy().tobytes())}
            kw = S.fixed_kwargs(enc, q, Hc, fixture_prov, {q: cached[dense[0]['qname']]}, 'cpu', ldlq_sigma=s)
            [(render, blob)] = enc.encode([w], [kw], 'NONE')
            S.wire_facts(blob, cached[dense[0]['qname']])
            # The fixed-length check is per real-unit geometry (stage1.encode); a
            # 16x128 fixture blob legitimately differs from the served unit's length.
            S.require(render.shape == w.shape and torch.isfinite(render).all().item(), 'seam decode invalid')
            blob_shas[s] = S.blob_sha(blob)
            del render, blob
        S.require(blob_shas[1.0] != blob_shas[3.0], 'sigma override produced identical bytes; knob inert')
        return {'sigma_bytes_differ': True, 'sigmas': sorted(blob_shas)}

    def schema_example():
        cell = {'layer': 3, 'role': 'up', 'chosen': {'sigma': 3.0, 'k': 1024},
                'candidate_grid': [{'k': 0, 'sigma': 1.0, 'validation_sse_summed_raw': 0.0}],
                'eligible_expert_count': 1, 'eligible_qnames': ['model.language_model.layers.3.mlp.experts.0.up_proj'],
                'lowcount_excluded_from_choice': [], 'pool_fit': {'path': 'pool', 'sha256': '0' * 64, 'bytes': 0, 'count': 1, 'members': 1},
                'pool_rest_count': 1, 'countsum': {'fit_total': 1, 'rest_total': 1, 'member_fit_sum': 1, 'member_rest_sum': 1},
                'self_inclusion': 'target included', 'denominator': 'sum of actual routed rows', 'zero_fit_excluded': []}
        example = {'schema': SELECTION_SCHEMA, 'method_sha256': '0' * 64, 'split_sha256': '0' * 64,
                   'cells': [cell], 'unit_chosen': {cell['eligible_qnames'][0]: cell['chosen']}}
        S.require(example['schema'] == SELECTION_SCHEMA and all(k in example for k in
                  ('method_sha256', 'split_sha256', 'cells', 'unit_chosen')), 'selection schema skeleton incomplete')
        return {'fields': sorted(example.keys()), 'shared_with': ['eng-d44-frozen-bar', 'eng-d44-frozen-g3', 'eng-d44-frozen-v0']}

    def selection_binding_guard():
        # Malformed and valid conditions through the REAL consumer guard
        # (selection_binding), before any encoder invocation. Fixture
        # selections; the guard under test is the shipped function.
        def shape(chosen, unit_chosen, rosters=True):
            cells = []
            for i, (layer, role) in enumerate([(l, r) for l in layers() for r in roles()]):
                cell = {'layer': layer, 'role': role, 'chosen': dict(chosen)}
                cell['unit_roster'] = ['q'] if rosters and i == 0 else []
                cells.append(cell)
            return {'schema': SELECTION_SCHEMA, 'method_sha256': method_sha256(),
                    'grid': {'sigma': sigmas(), 'shrinkage_k': shrinkage_ks()},
                    'fit_min_strict': FROZEN_FIT_MIN, 'fold': {'prefix_rows': PREFIX_ROWS},
                    'cells': cells, 'unit_chosen': dict(unit_chosen)}
        S.require(selection_binding(shape({'sigma': 3.0, 'k': 1024}, {'q': {'sigma': 3.0, 'k': 1024}})) is None,
                  'valid-shaped selection refused by the real guard')
        refused = 0
        cases = [({'sigma': 0.001, 'k': 1024}, {'q': {'sigma': 0.001, 'k': 1024}}),
                 ({'sigma': 3.0, 'k': 1000}, {'q': {'sigma': 3.0, 'k': 1000}}),
                 ({'sigma': 3.0, 'k': 4000}, {'q': {'sigma': 3.0, 'k': 4000}}),
                 ({'sigma': 1.0, 'k': -1}, {'q': {'sigma': 1.0, 'k': -1}}),
                 ({'sigma': float('nan'), 'k': 1024}, {'q': {'sigma': float('nan'), 'k': 1024}}),
                 ({'sigma': 3.0, 'k': 1024.5}, {'q': {'sigma': 3.0, 'k': 1024.5}}),
                 ({'sigma': 3.0, 'k': 1024}, {'q': {'sigma': 1.0, 'k': 0}}),
                 ({'sigma': 3.0, 'k': 1024}, {})]
        for chosen, units in cases:
            try:
                selection_binding(shape(chosen, units))
            except ValueError:
                refused += 1
                continue
            raise AssertionError(f'chosen {chosen} with units {units} passed the guard')
        S.require(refused == len(cases), 'malformed conditions not all refused')
        return {'malformed_refused': refused, 'valid_accepted': True}

    record('one_expert_pool_invariant', t_case)
    def rejected_producer_consumer_fixture():
        # Bounded valid fixture for the EXACT rejected and current producer
        # path. Reader-boundary mocks feed legitimate small tensors/counts;
        # the writer/serializer/consumer functions that run are the actual
        # rejected ones (CAS bundle 75104bd2, commit 619f7c80) and the current
        # ones. No selection runs here, so no grid pairs are involved.
        import argparse
        import subprocess
        import tempfile
        bundle = '/mnt/shared/prismabuild-fleet/cas/blobs/75/75104bd2b351acb0aec7fd594169770b247ffc0fdc53d7f629cd1080ba2c8743'
        S.require(Path(bundle).exists(), 'rejected-source bundle missing from CAS')
        W, nfits = 32, [600, 900]
        gen = torch.Generator().manual_seed(11)
        members = []
        for i, ne in enumerate(nfits):
            X = torch.randn(ne, W, generator=gen, dtype=torch.float64)
            members.append({'q': f'fixture.l3.experts.{i}.up_proj',
                            'Hfit': (X.T @ X).to(dtype=torch.float32),
                            'Xp': X[:PREFIX_ROWS].to(dtype=torch.float32), 'nfit': ne})
        suffix = {'gate': 'gate_proj', 'up': 'up_proj', 'down': 'down_proj'}
        rows, cached = [], {}
        _, live_cached, _, _ = S.population()
        live_settings = live_cached[_first_routed_qname()]['identity']['calibration']['settings']
        for m in members:
            rows.append({'qname': m['q'], 'layer': 3, 'kind': 'routed', 'pick': {'a8s': S.FMT}})
            cached[m['q']] = {'identity': {'source': {'shape': [16, W]},
                                           'calibration': {'settings': dict(live_settings)}}}
        for layer in layers():
            for role in roles():
                if (layer, role) == (3, 'up'):
                    continue
                q = f'fixture.l{layer}.experts.0.{suffix[role]}'
                rows.append({'qname': q, 'layer': layer, 'kind': 'routed', 'pick': {'a8s': S.FMT}})
                cached[q] = {'identity': {'source': {'shape': [16, W]},
                                          'calibration': {'settings': dict(live_settings)}}}
        manifest = {'rows': rows}
        by_q = {m['q']: m for m in members}
        class FixtureCap:
            def __init__(self):
                self.root = Path('.')
                self.split_sha256 = 'fixture'
                self.units = {m['q']: (Path('.'), {'fit': {'count': m['nfit']}}) for m in members}
            def get(self, q, role, width):
                S.require(role == 'fit', 'fixture serves the fit role only')
                m = by_q[q]
                return ({'hessian': m['Hfit'], 'inputs': m['Xp'], 'count': m['nfit']},
                        {'split_sha256': 'fixture', 'draw': {}, 'sha256': 'fixture', 'count': m['nfit']})
        with tempfile.TemporaryDirectory() as td:
            v1dir = str(Path(td) / 'v1src')
            subprocess.run(['git', 'clone', '-q', '-b', 'prismabuild-snapshot', bundle, v1dir], check=True)
            import importlib.util
            sp = importlib.util.spec_from_file_location('d44_training_rejected', v1dir + '/d44_training.py')
            v1 = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(v1)
            fixture_cap = FixtureCap()
            real_population, real_captures = S.population, S.Captures
            S.population = lambda: (manifest, cached, [r for r in rows], [])
            S.Captures = lambda capture: fixture_cap
            try:
                red_root = Path(td) / 'red'
                v1._write_pool(argparse.Namespace(root=red_root, capture='fixture', batch=None), 3, 'up')
                red_pool, _ = v1.load_pool(red_root / 'pool' / 'L003-up.pool.pt', expect_cell=(3, 'up'))
                green_root = Path(td) / 'green'
                _write_pool(argparse.Namespace(root=green_root, capture='fixture', batch=None), 3, 'up')
                green_pool, _ = load_pool(green_root / 'pool' / 'L003-up.pool.pt', expect_cell=(3, 'up'))
            finally:
                S.population, S.Captures = real_population, real_captures
            rests = []
            for m in members:
                Hr, nr = prefix_fold(m['Hfit'], m['Xp'], m['nfit'])
                rests.append((Hr, nr))
            rawsum = rests[0][0] + rests[1][0]
            total = rests[0][1] + rests[1][1]
            h0, n0 = rests[0]
            red = v1.conditioned_hessian(h0, n0, red_pool['rest']['gram'], int(red_pool['rest']['count']), 1024)
            true = (h0 + 1024.0 * (rawsum / float(total))) / float(n0 + 1024)
            rel = float((red - true).abs().max().item()) / float(true.abs().max().item())
            S.require(rel > 1e-3, 'rejected producer/consumer fixture did not reproduce the defect')
            green = conditioned_hessian(h0, n0, green_pool['rest']['gram_raw_sum'], int(green_pool['rest']['count']), 1024)
            div = float((green - true).abs().max().item())
            S.require(div == 0.0, 'current producer/consumer fixture diverged from the frozen formula')
        return {'fixture': '2 members, W=32, counts [600, 900]', 'red_relative_error': rel,
                'green_max_abs_divergence': div, 'rejected_commit': '619f7c80b8261653f980fe35f940b2ef4ff9e1ac'}
    record('rejected_producer_consumer_fixture', rejected_producer_consumer_fixture)
    record('diagnostic_unweighted_pool_denominator', diagnostic_unweighted_pool_denominator)
    record('diagnostic_target_exclusion', diagnostic_target_exclusion)
    record('serialized_pool_round_trip', serialized_pool_round_trip)
    record('zero_fit_conditioning_refused_by_name', zero_fit_conditioning_refused)
    record('prefix_fold_identity', fold_identity)
    record('pool_denominator_sum_of_rows', pool_denominator)
    record('objective_is_raw_sum', objective_raw_sum)
    record('role_for_qname', role_map)
    record('grid_from_frozen_config', grid_config)
    record('sigma_single_existing_knob', sigma_single_knob)
    record('encoder_seam_sigma_override', encoder_seam)
    record('selection_schema_example', schema_example)
    record('selection_binding_guard', selection_binding_guard)
    ok = not failures
    S.save(a.out, {'schema': SELFCHECK_SCHEMA, 'passed': ok, 'checks': checks, 'failures': failures,
                   'method_sha256': method_sha256(),
                   'note': 'failing-before lives in PB action 01b7c4c9 (exact rejected source, observed red 0.1954/green 0.0); diagnostic_* are manufactured reference-wrong variants, not observed failures',
                   'action_key': os.environ.get('PRISMABUILD_ACTION_KEY')})
    print(json.dumps({'passed': ok, 'checks': {k: v['passed'] for k, v in checks.items()}}), flush=True)
    S.require(ok, f'selfcheck failures: {failures}')


def _first_routed_qname():
    _, _, routed, _ = S.population()
    return routed[0]['qname']


# ---------------------------------------------------------------- readset + same-entry CPU preflight
def readset(a):
    S.setup()  # pins tessera/PrismaQuant paths before any owner import
    cap = S.Captures(a.capture)
    m, cached, _, _ = S.population()
    by = {r['qname']: r for r in m['rows']}
    entries = {}

    def add(path, offset=0, size=None, digest=None):
        path = Path(path)
        n = path.stat().st_size - offset if size is None else size
        entries[(str(path), offset, n)] = {'path': str(path), 'offset': offset, 'bytes': n, 'sha256': digest}

    for p in (METHOD_FILE, S.CACHE, S.MANIFEST, Path(a.capture) / 'split-manifest.json',
              cap.split['source_draw']['path'], S.MODEL / 'model.safetensors.index.json'):
        add(p)
    names = [a.qname] if a.qname else []
    if a.cells:
        for cid in a.cells:
            layer, role = cid.split(':')
            S.require(role in roles(), f'{cid}: role outside frozen roster')
            names.extend(r['qname'] for r in routed_cells()[(int(layer), role)])
    S.require(bool(names), 'readset needs --qname or --cells')
    for layer in sorted({int(by[q]['layer']) for q in set(names)}):
        p = Path(a.capture) / 'layers' / (layer_dir(layer) + '/manifest.json')
        S.require(p.exists(), f'layer {layer}: capture manifest not landed yet (priority0 chain still walking to L044)')
        add(p)
    wm = S.load(S.MODEL / 'model.safetensors.index.json')['weight_map']
    for q in sorted(set(names)):
        r = by[q]
        S.require(r['pick']['a8s'] == S.FMT, f'{q}: outside fixed format')
        e = cap.units[q][1]['fit']  # FIT only: the held-out entry is never declared
        add(Path(a.capture) / e['file'], size=e['bytes'], digest=e['sha256'])
        add(S.MODEL / wm[q + '.weight'])
        loc = r['formats'][S.FMT]['wire']
        off, n = S.setup()[2].member_location(loc)
        add(S.A8 / loc['shard'], off, n, cached[q]['blob_sha256'])
        pool_path = Path(a.root) / 'pool' / (layer_dir(int(r['layer'])) + f'-{role_for_qname(q)}.pool.pt')
        if pool_path.exists():
            add(pool_path)  # PB materializes/hash-checks entries lacking a digest
            receipt_path = pool_path.with_suffix('.receipt.json')  # actual consumed sidecar, not H/weights only
            if receipt_path.exists():
                add(receipt_path)
    if a.selection and Path(a.selection).exists():
        add(a.selection)
    data = list(entries.values())
    S.save(a.out, {'schema': 'prismaquant.prismabuild.data_manifest.v1', 'mount_prefix': '/mnt/shared',
                   'entries': data, 'entry_count': len(data), 'total_bytes': sum(e['bytes'] for e in data),
                   'produced_by': {'tool': 'd44_training.readset; FIT-role entries only'},
                   'annotations': {'fit_only': True, 'heldout_entries': 0}})


def preflight(a):
    """Same-entry CPU preflight: read and digest every declared entry, in this entry."""
    started = time.monotonic()
    dm = S.load(a.readset)
    S.require(dm['schema'] == 'prismaquant.prismabuild.data_manifest.v1', 'wrong data manifest schema')
    verified, total, bad = 0, 0, []
    for e in dm['entries']:
        try:
            got = S.data_entry(e['path'], offset=e['offset'], size=e['bytes'])
            if e['sha256']:
                S.require(got['sha256'] == e['sha256'], f"{e['path']}: digest differs from declaration")
            verified += 1
            total += e['bytes']
        except Exception as exc:  # noqa: BLE001
            bad.append({'entry': e['path'], 'error': str(exc)})
    S.require(not bad, f'preflight failures: {bad}')
    S.save(a.out, {'schema': 'd44.frozen_training_preflight.v1', 'verified_entries': verified,
                   'verified_bytes': total, 'failures': bad, 'readset': str(a.readset),
                   'elapsed_seconds': time.monotonic() - started,
                   'action_key': os.environ.get('PRISMABUILD_ACTION_KEY')})
    print(json.dumps({'verified_entries': verified, 'verified_bytes': total}), flush=True)


# ---------------------------------------------------------------- logical plans (PB owns placement/batching)
def _common_argv(a, stage):
    return [S.PYTHON if a.device == 'cpu' else 'python3', 'd44_training.py', stage,
            '--capture', str(a.capture), '--root', str(a.root), '--device', a.device]


def plan(a):
    """Emit prismabuild.logical_request.v1 for the frozen selection phases.

    pool:    one logical task per (layer, role) cell.
    select:  one logical task per routed expert (12 grid encodes each), estimated
             from REAL measured unit timings of the same geometry, never invented.
    """
    S.setup()  # pins tessera/PrismaQuant paths before any owner import
    _, cached, _, _ = S.population()
    cap = S.Captures(a.capture)
    timings = S.load(a.timings)
    S.require(timings.get('observed') is True and timings.get('device') == a.device and timings.get('action_key'),
              'real same-device PB runtime receipts required, no invented estimates')
    cells = routed_cells()
    if getattr(a, 'cells', None):
        want = set()
        for cid in a.cells:
            S.require(':' in cid, f'{cid}: cell id must read <layer>:<role>')
            layer_s, role = cid.split(':')
            S.require(role in roles(), f'{cid}: role outside frozen roster')
            want.add((int(layer_s), role))
        S.require(want <= set(cells), f'{sorted(want - set(cells))}: cells outside the frozen roster')
        cells = {cell: cells[cell] for cell in sorted(want)}
    # Missing capture data is a named dependency, never a zero and never a
    # silent roster cut: the full campaign waits for complete input data.
    for cell, rows in cells.items():
        layer = cell[0]
        S.require((Path(a.capture) / 'layers' / (layer_dir(layer) + '/manifest.json')).exists(),
                  f'layer {layer}: capture manifest not landed yet (priority0 chain still walks to L044)')
        for r in rows:
            S.require(r['qname'] in cap.units,
                      f"{r['qname']}: capture data missing for layer {layer} (priority0 chain has not landed it); the full campaign waits for complete input, and the unit stays on the fixed roster")
    wm = S.load(S.MODEL / 'model.safetensors.index.json')['weight_map']
    setup_entries = [S.data_entry(p) for p in (METHOD_FILE, S.CACHE, S.MANIFEST,
                     Path(a.capture) / 'split-manifest.json', cap.split['source_draw']['path'],
                     S.MODEL / 'model.safetensors.index.json')]
    setup_entries.extend(S.data_entry(p) for p in sorted(Path(a.capture).glob('layers/L*/manifest.json')))
    tasks, residency = [], {}
    if a.phase == 'pool':
        cell_seconds = float(timings['pool_cell_seconds'])
        for (layer, role) in sorted(cells):
            key = layer_dir(layer)
            residency[key] = {'key': key, 'setup_seconds': cell_seconds, 'setup_evidence': timings['action_key']}
            reads = list(setup_entries)
            for r in cells[(layer, role)]:
                e = cap.units[r['qname']][1]['fit']
                reads.append({'path': str(Path(a.capture) / e['file']), 'offset': 0, 'bytes': e['bytes'], 'sha256': e['sha256']})
            tid = f"pool-{layer_dir(layer)}-{role}"
            tasks.append({'id': tid, 'output_id': tid,
                          'payload': {'layer': layer, 'role': role, 'reads': reads},
                          'residency_key': key, 'estimated_seconds': cell_seconds,
                          'estimate_evidence': timings['action_key']})
        stage = 'pool'
        extra_argv = []
    elif a.phase == 'select':
        geometries = timings['geometries']
        for (layer, role), rows in sorted(cells.items()):
            key = layer_dir(layer)
            setup = max(geometries['x'.join(map(str, cached[r['qname']]['identity']['source']['shape']))]['setup_seconds']
                        for r in rows)
            residency[key] = {'key': key, 'setup_seconds': setup, 'setup_evidence': timings['action_key']}
            for r in rows:
                q = r['qname']
                geometry = 'x'.join(map(str, cached[q]['identity']['source']['shape']))
                t = geometries[geometry]
                S.require(t['unit_seconds'] > 0 and t['setup_seconds'] > 0, f'{q}: invalid measured runtime')
                reads = list(setup_entries)
                e = cap.units[q][1]['fit']
                reads.append({'path': str(Path(a.capture) / e['file']), 'offset': 0, 'bytes': e['bytes'], 'sha256': e['sha256']})
                reads.append(S.data_entry(S.MODEL / wm[q + '.weight']))
                # The actual pool and its receipt sidecar are consumed inputs,
                # declared with real digests; a missing pool refuses at plan
                # time (run the pool phase first) rather than at encode time.
                pool_file = Path(a.pool_root or a.root) / 'pool' / (layer_dir(layer) + f'-{role}.pool.pt')
                S.require(pool_file.exists(), f'{cell_id(layer, role)}: pool absent at plan time; run the pool phase first')
                reads.append(S.data_entry(pool_file))
                receipt_file = pool_file.with_suffix('.receipt.json')
                S.require(receipt_file.exists(), f'{cell_id(layer, role)}: pool receipt sidecar absent at plan time')
                reads.append(S.data_entry(receipt_file))
                tid = 'sel-' + hashlib.sha256(q.encode()).hexdigest()[:20]
                tasks.append({'id': tid, 'output_id': tid, 'payload': {'qname': q, 'reads': reads},
                              'residency_key': key,
                              'estimated_seconds': len(grid()) * t['unit_seconds'],
                              'estimate_evidence': timings['action_key']})
        stage = 'select-unit'
        extra_argv = ['--timings', str(a.timings)] + (['--pool-root', str(a.pool_root)] if getattr(a, 'pool_root', None) else [])
    else:
        raise ValueError(f'{a.phase}: unsupported plan phase')
    launch_argv = _common_argv(a, stage)
    if a.device == 'cuda' and a.phase == 'select':
        launch_argv = [S.PYTHON, 'encode_launch.py', '--stage', 'select-unit',
                       '--device', 'cuda', '--capture', str(a.capture), '--root', str(a.root)]
    common = {'argv': [*launch_argv, *extra_argv, '--batch', '{pb.task_batch}'],
              'cwd': str(HERE), 'demand': {'cpu': 1, 'mem_gb': a.mem_gb, **({'gpu': 1} if a.device == 'cuda' else {})},
              'gpu_memory_gb': a.gpu_memory_gb if a.device == 'cuda' else None, 'data_manifest': None,
              'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                      'PYTHONDONTWRITEBYTECODE': '1',
                      **({'TMPDIR': '/tmp', 'PRISMAQUANT_TMPDIR': '/tmp'} if a.device == 'cpu' else {})},
              'tags': ['gb10'] if a.device == 'cuda' else ['x86'],
              'timeout_s': 3600 if a.phase == 'pool' else 1800}
    S.require(a.device != 'cuda' or a.gpu_memory_gb is not None, 'explicit GPU subset budget required')
    S.save(a.out, {'schema': 'prismabuild.logical_request.v1', 'common': common,
                   'roster': {'schema': 'prismabuild.logical_task_roster.v1', 'tasks': tasks},
                   'batch_policy': {'schema': 'prismabuild.roster_batch_policy.v1',
                                    'residencies': list(residency.values()),
                                    'max_setup_fraction': a.max_setup_fraction,
                                    'max_estimated_wall_seconds': 1800},
                   'task_data_manifest': {'schema': 'prismabuild.task_data_manifest.v1', 'payload_field': 'reads',
                                          'mount_prefix': '/mnt/shared', 'residency_tier': None,
                                          'residency_ram': 'auto', 'mover_readers': 4, 'mover_mem_gb': 1}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('pool', 'select-unit', 'reduce', 'selfcheck', 'readset', 'preflight', 'plan'):
        s = sub.add_parser(name)
        s.add_argument('--capture', type=Path, default=S.CAPTURE)
        s.add_argument('--root', type=Path, default=S.OUTPUT)
        s.add_argument('--out', type=Path)
        s.add_argument('--batch', type=Path)
        s.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    for name in ('pool', 'select-unit', 'readset'):
        sub.choices[name].add_argument('--qname')
    sub.choices['pool'].add_argument('--layer', type=int)
    sub.choices['pool'].add_argument('--role')
    sub.choices['readset'].add_argument('--cells', action='append')
    sub.choices['readset'].add_argument('--selection', type=Path)
    sub.choices['preflight'].add_argument('--readset', type=Path, required=True)
    sub.choices['plan'].add_argument('--phase', choices=('pool', 'select'), required=True)
    sub.choices['plan'].add_argument('--timings', required=True)
    sub.choices['select-unit'].add_argument('--dry-run', action='store_true')
    sub.choices['plan'].add_argument('--mem-gb', type=int, required=True)
    sub.choices['plan'].add_argument('--gpu-memory-gb', type=int)
    sub.choices['plan'].add_argument('--max-setup-fraction', type=float, required=True)
    sub.choices['select-unit'].add_argument('--timings', type=Path)
    sub.choices['select-unit'].add_argument('--pool-root', type=Path)
    sub.choices['plan'].add_argument('--pool-root', type=Path)
    sub.choices['plan'].add_argument('--cells', action='append')
    a = p.parse_args()
    if a.command in ('select-unit',):
        S.require(bool(a.batch) != bool(a.qname), 'exactly one --batch or --qname required')
    if a.command in ('selfcheck', 'preflight', 'reduce', 'readset', 'plan'):
        S.require(a.out is not None, '--out required')
    globals()[a.command.replace('-', '_')](a)


if __name__ == '__main__':
    main()
