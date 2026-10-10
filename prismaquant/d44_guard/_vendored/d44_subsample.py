#!/usr/bin/env python3
"""D44 seeded-subsample selector (CEO decision dec-1007-052055-aae6, option B).

Frozen files stay byte-identical; this module only reuses their functions.
  sample       preregistered 32-expert sample per (layer, role) cell (FIT counts only)
  reduce       frozen selector objective summed over the sample only -> a
               d44.frozen_condition_selection.v1 file that stage1 --condition-selection reads
  regret       selector-only L40 check on finished grid units (FIT prefix rows only, no HELD)
  encode-cost  GPU timing of the final full-FIT conditioned encode, FIT role only.
               It never opens the held-out role and writes no receipt or blob a
               scientific stage can read: cost evidence only.
Preregistered rule (record eng-d44-campaign-opus, subsample_preregistration):
  n = 32 eligible experts (FIT count > 1024) per cell, all if fewer; rank eligible
  qnames by sha256(("d44-subsample-20261007:" + qname).encode()).hexdigest() ascending.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import stage1 as S
import d44_training as T

SAMPLE_SCHEMA = 'd44.subsample_preregistered.v1'
SAMPLE_N = 32
SAMPLE_SALT = 'd44-subsample-20261007:'
REGRET_SEED = 20261007
REGRET_RESAMPLES = 2000


def fit_only_captures(root):
    """Capture reader that never opens HELD data, not even HELD token ids (review C1).

    stage1.Captures.__init__ loads the whole calibration draw and hashes the HELD
    token slice to re-derive the split identity. This reader takes the split
    identity from the split manifest's own field, requires every layer manifest
    to name it, and serves only the FIT role through the frozen Captures.get
    checks (file digest, shapes, counts, prefix coordinates, finiteness).
    """
    from g3_residency import read_file
    cap = S.Captures.__new__(S.Captures)
    cap.root = Path(root)
    raw = read_file(cap.root / 'split-manifest.json')
    cap.split = json.loads(raw)
    cap.split_sha256 = cap.split['split_sha256']
    cap.split_manifest_file_sha256 = S.blob_sha(raw)
    cap.split_identity = 'manifest field; not re-derived from token ids (FIT-only reader)'
    cap.units = {}
    for path in sorted(cap.root.glob('layers/L*/manifest.json')):
        m = S.load(path)
        S.require(m['split_sha256'] == cap.split_sha256, 'Layer captures name different actual split body')
        if 'split_manifest_file_sha256' in m:
            S.require(m['split_manifest_file_sha256'] == cap.split_manifest_file_sha256, 'Layer-bound split JSON own-byte integrity')
        for q, roles in m['units'].items():
            S.require(q not in cap.units, 'Duplicated captured unit')
            cap.units[q] = (path.parent, roles)
    frozen_get = S.Captures.get
    def get(q, role, width):
        S.require(role == 'fit', f'{q}: FIT-only reader refuses role {role!r}')
        return frozen_get(cap, q, role, width)
    cap.get = get
    return cap


def require_finite_totals(totals, where):
    """The frozen reduce refusal (d44_training.py reduce): every total finite and >= 0 (review S1)."""
    for key, total in totals.items():
        S.require(math.isfinite(total) and total >= 0, f'{where}: invalid grid total {key}')


def expected_sample(cap):
    """The preregistered rule, recomputed from capture FIT counts."""
    out = {}
    for (layer, role), rows in sorted(T.routed_cells().items()):
        eligible = sorted(r['qname'] for r in rows if fit_count(cap, r['qname']) > T.FROZEN_FIT_MIN)
        S.require(bool(eligible), f'{T.cell_id(layer, role)}: no eligible expert; fail closed')
        out[(layer, role)] = (len(eligible), sorted(eligible, key=rank_key)[:SAMPLE_N])
    return out


def load_sample(path, cap):
    """Validate the complete preregistered draw (review S2): schema, split, n, salt, cell set, members."""
    smp = S.load(path)
    S.require(smp.get('schema') == SAMPLE_SCHEMA, 'wrong sample schema')
    S.require(smp.get('split_sha256') == cap.split_sha256, 'sample split differs from the capture')
    S.require(smp.get('n') == SAMPLE_N, f"sample n {smp.get('n')!r} is not {SAMPLE_N}")
    S.require(smp.get('salt') == SAMPLE_SALT, 'sample salt differs from the preregistered prefix')
    want = expected_sample(cap)
    keys = [(int(c['layer']), c['role']) for c in smp['cells']]
    S.require(len(keys) == len(set(keys)) and set(keys) == set(want), 'sample cell set differs from the live routed cells')
    for c in smp['cells']:
        key = (int(c['layer']), c['role'])
        S.require(int(c['eligible_count']) == want[key][0], f'{T.cell_id(*key)}: eligible count differs')
        S.require(list(c['sample']) == want[key][1], f'{T.cell_id(*key)}: sampled qnames differ from the preregistered rule')
    return smp


def rank_key(q):
    return hashlib.sha256((SAMPLE_SALT + q).encode()).hexdigest()


def fit_count(cap, q):
    e = cap.units[q][1].get('fit')
    S.require(e is not None, f'{q}: FIT entry absent (unmeasured; never mapped to zero)')
    return int(e['count'])


def pool_path(root, layer, role):
    return Path(root) / 'pool' / (T.layer_dir(layer) + f'-{role}.pool.pt')


def sample(a):
    S.setup()
    cap = S.Captures(a.capture)
    cells = T.routed_cells()
    want = expected_sample(cap)
    out = []
    for (layer, role), rows in sorted(cells.items()):
        n_eligible, chosen = want[(layer, role)]
        out.append({'layer': layer, 'role': role, 'roster_count': len(rows), 'eligible_count': n_eligible,
                    'sample': chosen, 'sample_rank_keys': [rank_key(q) for q in chosen]})
    S.save(a.out, {'schema': SAMPLE_SCHEMA, 'n': SAMPLE_N, 'salt': SAMPLE_SALT,
                   'rule': 'rank eligible qnames (FIT count > 1024) by sha256(salt + qname) ascending; take the first n',
                   'eligibility_source': 'capture manifest FIT counts only; no grid value, no HELD read',
                   'split_sha256': cap.split_sha256, 'cells': out})
    print(json.dumps({'out': str(a.out), 'cells': len(out), 'units': sum(len(c['sample']) for c in out)}), flush=True)


def _load_grid(root, q, cap):
    path = Path(root) / 'grid' / (q.replace('.', '__') + '.json')
    if not path.exists():
        return None
    g = S.load(path)
    S.require(g['qname'] == q and g['schema'] == T.GRID_UNIT_SCHEMA and not g['dry_run'], f'{q}: wrong grid unit record')
    S.require(g['split_sha256'] == cap.split_sha256, f'{q}: grid unit from another split')
    want = {(k, s) for k, s in T.grid()}
    have = {(int(r['k']), float(r['sigma'])) for r in g['grid_rows']}
    S.require(g['eligible'] and have == want, f'{q}: incomplete or ineligible grid')
    return g


def _argmin(totals):
    return min(sorted(totals), key=lambda ks: (totals[ks], ks[0], ks[1]))


def reduce(a):
    S.setup()
    started = time.monotonic()
    m = T.method()
    cap = S.Captures(a.capture)
    smp = load_sample(a.sample, cap)
    by_cell = {(int(c['layer']), c['role']): c for c in smp['cells']}
    cells = T.routed_cells()
    out_cells, unit_chosen = [], {}
    for (layer, role), rows in sorted(cells.items()):
        pp = pool_path(a.pool_root, layer, role)
        pool, receipt = T.load_pool(pp, expect_cell=(layer, role))
        S.require(pool['split_sha256'] == cap.split_sha256, f'{T.cell_id(layer, role)}: pool split differs')
        sampled = by_cell[(layer, role)]['sample']
        grids = []
        for q in sampled:
            g = _load_grid(a.root, q, cap)
            S.require(g is not None, f'{q}: sampled grid unit missing; refusing to select')
            S.require(int(g['nfit']) == int(pool['member_nfit'][q]), f'{q}: fit count drifted against the pool')
            grids.append(g)
        totals = T.objective_total(grids, set(sampled))
        S.require(set(totals) == {(k, s) for k, s in T.grid()}, f'{T.cell_id(layer, role)}: incomplete grid totals')
        require_finite_totals(totals, T.cell_id(layer, role))
        k, sigma = _argmin(totals)
        for r in rows:
            unit_chosen[r['qname']] = {'sigma': sigma, 'k': k}
        out_cells.append({'layer': layer, 'role': role, 'chosen': {'sigma': sigma, 'k': k},
                          'candidate_grid': [{'k': kk, 'sigma': ss, 'validation_sse_summed_raw': totals[(kk, ss)]}
                                             for kk, ss in sorted(totals)],
                          'objective_scope': 'preregistered subsample', 'sampled_qnames': sorted(sampled),
                          'eligible_expert_count': by_cell[(layer, role)]['eligible_count'],
                          'unit_roster': sorted(r['qname'] for r in rows),
                          'pool_fit': {'path': str(pp), 'sha256': receipt['pool_sha256'], 'bytes': receipt['pool_bytes'],
                                       'count': int(pool['fit']['count']), 'members': int(pool['fit']['members'])},
                          'pool_rest_count': int(pool['rest']['count']),
                          'self_inclusion': pool['self_inclusion'], 'denominator': pool['denominator'],
                          'zero_fit_excluded': pool['zero_fit_excluded']})
    selection = {'schema': T.SELECTION_SCHEMA, 'method_sha256': T.method_sha256(), 'split_sha256': cap.split_sha256,
                 'objective': m['1_selector']['objective'] + '; summed over the preregistered subsample only',
                 'subsample': {'decision': 'dec-1007-052055-aae6 option B', 'sample_file': str(a.sample),
                               'sample_sha256': S.sha(a.sample), 'n': smp['n'], 'rule': smp['rule'], 'salt': smp['salt']},
                 'scope': m['1_selector']['scope'],
                 'fold': {'rule': m['1_selector']['rule'], 'prefix_rows': T.PREFIX_ROWS,
                          'training': m['1_selector']['training'], 'validation': m['1_selector']['validation']},
                 'fit_min_strict': T.FROZEN_FIT_MIN,
                 'grid': {'sigma': T.sigmas(), 'shrinkage_k': T.shrinkage_ks(),
                          'removed_damping': m['2_grid']['removed_damping'],
                          'removed_shrinkage_k': list(T.REMOVED_SHRINKAGE_K)},
                 'final': {'reencode': m['1_selector']['final'], 'heldout_scored': False,
                           'scientific_selection_submitted': False},
                 'cells': out_cells, 'unit_chosen': unit_chosen,
                 'fit_only_proof': {'heldout_grams_used': 0},
                 'lowcount_policy': 'excluded from the choice only; encoded with the chosen sigma/k and kept in the bar',
                 'reduced_seconds': time.monotonic() - started, 'action_key': os.environ.get('PRISMABUILD_ACTION_KEY')}
    T.selection_binding(selection)  # the frozen consumer's own refusals, before save
    S.save(a.out, selection)
    print(json.dumps({'out': str(a.out), 'chosen': {T.cell_id(c['layer'], c['role']): c['chosen'] for c in out_cells}}), flush=True)


def regret(a):
    """Selector-only check on finished L40 units. FIT prefix validation SSE only; never HELD."""
    S.setup()
    cap = S.Captures(a.capture)
    smp = load_sample(a.sample, cap)
    by_cell = {(int(c['layer']), c['role']): c for c in smp['cells']}
    rng = random.Random(REGRET_SEED)
    report = []
    for (layer, role), rows in sorted(T.routed_cells().items()):
        if layer != a.layer:
            continue
        done = {}
        for r in rows:
            q = r['qname']
            if fit_count(cap, q) <= T.FROZEN_FIT_MIN:
                continue
            g = _load_grid(a.root, q, cap)
            if g is not None:
                done[q] = g
        names = sorted(done)
        full = T.objective_total(list(done.values()), set(names))
        require_finite_totals(full, T.cell_id(layer, role))
        ref = _argmin(full)
        def rel(choice):
            return full[choice] / full[ref] - 1.0
        pre = [q for q in by_cell[(layer, role)]['sample'] if q in done]
        pre_choice = _argmin(T.objective_total([done[q] for q in pre], set(pre))) if pre else None
        n = min(SAMPLE_N, len(names))
        agree, regrets = 0, []
        for _ in range(REGRET_RESAMPLES):
            sub = rng.sample(names, n)
            c = _argmin(T.objective_total([done[q] for q in sub], set(sub)))
            agree += c == ref
            regrets.append(rel(c))
        regrets.sort()
        report.append({'layer': layer, 'role': role, 'finished_eligible': len(names),
                       'eligible_count': by_cell[(layer, role)]['eligible_count'],
                       'reference_choice_over_finished': {'k': ref[0], 'sigma': ref[1]},
                       'preregistered_sample_finished': len(pre),
                       'preregistered_choice_over_finished_part': None if pre_choice is None else
                           {'k': pre_choice[0], 'sigma': pre_choice[1], 'relative_regret': rel(pre_choice)},
                       'random_subsets': {'n': n, 'resamples': REGRET_RESAMPLES, 'seed': REGRET_SEED,
                                          'agree_fraction': agree / REGRET_RESAMPLES,
                                          'relative_regret_mean': sum(regrets) / len(regrets),
                                          'relative_regret_p90': regrets[int(0.9 * len(regrets)) - 1],
                                          'relative_regret_max': regrets[-1]},
                       'second_best_gap': sorted(full.values())[1] / full[ref] - 1.0})
    S.save(a.out, {'schema': 'd44.subsample_regret_check.v1', 'layer': a.layer, 'cells': report,
                   'scope': 'selector-only; FIT prefix validation SSE of finished grid units; no HELD read, no bar statistic',
                   'regret_definition': 'summed raw validation SSE over finished eligible units at the subset choice, divided by the same sum at the finished-set choice, minus 1'})
    print(json.dumps(report), flush=True)


def encode_cost(a):
    """Final-encode cost: full-FIT conditioned encode, FIT role only, placeholder (sigma, k)."""
    torch, CD2, _, _ = S.setup()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    m, cached, _, _ = S.population()
    by = {r['qname']: r for r in m['rows']}
    cap = fit_only_captures(a.capture)  # never opens HELD data (review C1)
    names = [t['payload']['qname'] for t in S.load(a.batch)['tasks']]
    S.require((int(a.k), float(a.sigma)) in {(k, s) for k, s in T.grid()}, 'placeholder setting outside the frozen grid')
    enc = CD2.Encoder(S.FMT)
    pools, rows = {}, []
    setup_s = time.monotonic() - started
    for q in names:
        if a.device == 'cuda':
            torch.cuda.synchronize()
        t_unit = time.monotonic()
        r = by[q]
        cell = (int(r['layer']), T.role_for_qname(q))
        width = cached[q]['identity']['source']['shape'][1]
        d, fp = cap.get(q, 'fit', width)  # FIT role only; the held-out role is never opened
        w, _ = S.source(r, cached, a.device)
        if cell not in pools:
            pools[cell] = T.load_pool(pool_path(a.pool_root, *cell), expect_cell=cell)[0]
        pool = pools[cell]
        # The frozen stage1.encode population checks (review C2).
        S.require(pool['split_sha256'] == cap.split_sha256, f'{q}: pool split differs from the capture')
        S.require(pool['member_nfit'][q] == int(d['count']), f'{q}: fit count drifted against the frozen pool')
        H = T.rawcount_hessian(d['hessian'], int(d['count']), pool['fit']['gram_raw_sum'], int(pool['fit']['count']), int(a.k))
        kw = S.fixed_kwargs(enc, q, H.to(dtype=torch.float32).contiguous(), T.fit_provenance(cap, fp), cached, a.device,
                            ldlq_sigma=float(a.sigma))
        if a.dry_run:
            rows.append({'qname': q, 'dry_run': True, 'kwargs_built': True})
            continue
        if a.device == 'cuda':
            torch.cuda.synchronize()  # preparation kernels stay outside the encode interval (review C4)
        t_enc = time.monotonic()
        [(render, blob)] = enc.encode([w], [kw], 'NONE')
        if a.device == 'cuda':
            torch.cuda.synchronize()
        enc_s = time.monotonic() - t_enc
        S.require(len(blob) == cached[q]['blob_bytes'], f'{q}: fixed serialized unit length changed')
        S.require(render.shape == w.shape and torch.isfinite(render).all().item(), f'{q}: decoded candidate invalid')
        if a.device == 'cuda':
            torch.cuda.synchronize()
        rows.append({'qname': q, 'encode_seconds': enc_s, 'unit_seconds': time.monotonic() - t_unit})
        del render, blob, H, kw, w, d
    out = {'schema': 'd44.final_encode_cost.v2', 'scope': 'cost evidence only; placeholder setting; no blob or receipt kept; FIT-only capture reader (no HELD data or HELD token ids)',
           'split_identity': cap.split_identity,
           'sigma': float(a.sigma), 'k': int(a.k), 'device': a.device, 'dry_run': a.dry_run, 'setup_seconds': setup_s,
           'units': rows, 'elapsed_seconds': time.monotonic() - started, 'action_key': os.environ.get('PRISMABUILD_ACTION_KEY')}
    S.save(Path(a.root) / 'cost' / (Path(a.batch).stem + ('.dry' if a.dry_run else '') + '.json'), out)
    print(json.dumps({k: out[k] for k in ('setup_seconds', 'elapsed_seconds', 'dry_run')} | {'units': len(rows)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    needs = {'sample': ('out',), 'reduce': ('root', 'out', 'sample'), 'regret': ('root', 'out', 'sample'),
             'encode-cost': ('root',)}  # review F3: required paths refuse at parse time
    for name, req in needs.items():
        s = sub.add_parser(name)
        s.add_argument('--capture', type=Path, default=S.CAPTURE)
        s.add_argument('--pool-root', type=Path, required=True)
        for opt in ('root', 'out', 'sample'):
            s.add_argument('--' + opt, type=Path, required=opt in req)
        s.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    sub.choices['regret'].add_argument('--layer', type=int, default=40)
    sub.choices['encode-cost'].add_argument('--batch', type=Path, required=True)
    sub.choices['encode-cost'].add_argument('--sigma', type=float, required=True)
    sub.choices['encode-cost'].add_argument('--k', type=int, required=True)
    sub.choices['encode-cost'].add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    {'sample': sample, 'reduce': reduce, 'regret': regret, 'encode-cost': encode_cost}[a.cmd](a)


if __name__ == '__main__':
    main()
