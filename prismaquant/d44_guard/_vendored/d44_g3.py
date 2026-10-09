#!/usr/bin/env python3
"""D44 frozen G3: same-window EXL3-ceiling arm, preregistered projection, paired confirmation.

Frozen-method 10_G3 (CEO freeze 2026-10-06T21:15:53Z), slice record
/home/rob/fleet/records/eng-d44-frozen-g3.json.  Commands:

  build-ceiling  From the frozen swap selection and the accepted v3c EX manifest, build a G3v2
                 namespace manifest whose new pick "d44ceiling" serves, for every frozen unit,
                 that unit's ACTUAL production weight: routed units serve the real served EXL3
                 bytes decoded (exl3_torch.decode_wire, never a TESSERA reencode) and aligned
                 back to source basis with the existing cd2_alignment rule (d44.reference path);
                 units whose production EXL3 pick is SOURCE serve the actual retained source BF16
                 member bytes (byte slice of the model shard, verified equal to the safe_open
                 tensor).  Every other row keeps its A8S format and every unit keeps the A8S
                 fp8_per_token_dynamic activation contract (rest source A8S W+A).  Each ceiling
                 entry carries its rendered sha256 so the unchanged engine hash gate proves, per
                 unit, that the executed weights are exactly this decode.
  preregister    Before ANY ceiling scoring: locks the routed roster (authoritative bar
                 representation, full 5184 routed units -- a strict subset is refused), the
                 FIT-selected shared sites (separate G3 fields), the 25 prefixed windows, the
                 teacher binding and the frozen formulas.  Needs no future result.
  lock           After the ceiling arm is scored, before any swap scoring: binds the completed
                 ceiling result, computes dceil and the paired SE of (A8S - ceiling), applies the
                 low-power guard (dceil < 2*paired SE -> weight-leg only, routed to CEO, no
                 frozen P borrowed) and seals the numeric P = C_gap*dceil.  Refuses to overwrite.
  score          Published scoring adapter (also stage1.py score --arm d44ceiling_wa): requires
                 --prereg, runs the guard, then the unchanged v2 owner with the decoder dispatch.
  readset        D38 data manifest for the ceiling arm.
  smoke          Real CPU smoke of one routed unit and the shared site.
  selftest       Bounded behavioral proofs on real bytes: sequential-dispatch reproduction,
                 strict-subset roster refusal, unlocked-prereg swap refusal -> sealed pass,
                 shared-site byte/slice identity, and the ACTUAL stage1/score --dry-run entry.
  confirm        Paired confirmation reducer -> d44.frozen_confirmation.v1.

Decoder dispatch (score): the owner plumbs the decoder through run_multi and calls it per wire
blob, in arbitrary order/concurrency.  A dispatch call is therefore STATELESS and self-sufficient:
for a routed ceiling unit it loads the complete gate/up/down group within that one call (its own
blob plus the two sibling wires, each sha-gated), applies the build-time permutation with
cd2_alignment.aligned -- the existing alignment rule, not a second one -- and returns this role's
source-basis tensor; for a retained-source unit it rebuilds the tensor from its own member bytes.
The engine's existing hash gate then checks the tensor against the entry's rendered sha256; no
numerical engine code is altered.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import statistics
import struct
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import stage1 as S

PICK = S.PICK
CEIL_PICK = 'd44ceiling'
CEIL_ARM = CEIL_PICK + '_wa'
CEIL_FMT = 'D44EXL3SRC'
V2 = S.V2
EX_MANIFEST = S.BASE / 'surrogate-diag-20260929/g3/unit_manifest_v3c.json'
EX_ROOT = '/mnt/shared/models/GLM-5.3-Flash-EXL3-TR3-4bpw'
ROLES = ('gate_proj', 'up_proj', 'down_proj')
CONFIRM_SCHEMA = 'd44.frozen_confirmation.v1'
PREREG_SCHEMA = 'd44.frozen_g3_preregistration.v1'
WINDOWS = 25
#: authoritative routed bar roster digest (bar slice, recomputed from the baseline manifest):
#: sha256 over sorted routed qnames of L3,40-44, utf8, each followed by '\n'
ROUTED_ROSTER_SHA256 = '0e41201e1dbe4dc1e0e1cab2d8291bd029ca97731e638469cd6656ebfde76cba'
ROUTED_ROSTER_N = 5184
#: the FIT-selected dense/shared winner (all24 FIT float64 rank, action 35bc79bddc23) that the
#: explicit full swap selection must separately include beside the routed-only bar freeze
SHARED_WINNER = 'model.language_model.layers.39.mlp.shared_experts.gate_proj'


def sha(path):
    return S.sha(path)


def actual_ceiling_rows(m):
    """The units the manifest ACTUALLY substitutes for the ceiling pick — derived from the rows,
    never from declared metadata."""
    return sorted(r['qname'] for r in m['rows'] if r['pick'][CEIL_PICK] == CEIL_FMT)

def actual_swap_rows(m):
    """The units the swap scorer will ACTUALLY install — derived from installed manifest picks.
    assemble() marks every real replacement row pick[PICK] = 'indomain_stage1::TESSERA...'
    with its own formats entry; every non-replaced row keeps its base A8S pick."""
    return sorted(r['qname'] for r in m['rows'] if r['pick'].get(PICK, '').startswith(PICK + '::'))

def roster_sha(qnames):
    """Authoritative bar representation: every sorted qname followed by a newline."""
    return hashlib.sha256(''.join(q + '\n' for q in sorted(qnames)).encode()).hexdigest()


def dotted(doc, path):
    cur = doc
    for part in path.split('.'):
        cur = cur[int(part)] if part.isdigit() else cur[part]
    return cur


@lru_cache(maxsize=1)
def owner_v2():
    """The accepted G3v2 scoring owner, loaded by path (never by name resolution: the checkout
    carries a launch-argv wrapper of the same name)."""
    spec = importlib.util.spec_from_file_location('d44_g3_owner_v2_score', S.OWNERS / 'v2_score.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def baseline_routed_roster(exby):
    """The full fixed routed bar roster, from the baseline manifest itself."""
    routed = sorted(r['qname'] for r in exby.values()
                    if r['kind'] == 'routed' and r['layer'] in S.LAYERS)
    S.require(len(routed) == ROUTED_ROSTER_N, f'routed bar roster population {len(routed)}')
    got = roster_sha(routed)
    S.require(got == ROUTED_ROSTER_SHA256,
              f'routed bar roster digest {got} differs from the authoritative {ROUTED_ROSTER_SHA256}')
    return routed


# ------------------------------------------------------------------ ceiling manifest builder
def group_rows(exby, row):
    """The EXL3 gate/up/down rows of one routed expert group (same layer), as d44.source_group_rows."""
    S.require(row['layer'] in S.LAYERS and row['role'] in ROLES,
              row['qname'] + ': routed frozen unit outside the routed G3 roster')
    group = row['qname'].rsplit('.', 1)[0]
    rows = []
    for role in ROLES:
        r = exby[group + '.' + role]
        S.require(r['pick']['exl3'] == 'EXL3',
                  r['qname'] + ': routed unit is not EXL3 in production; route to CEO')
        S.require(r['layer'] == row['layer'], r['qname'] + ': cross-layer reference substitution')
        rows.append(r)
    return rows


def align_group(torch, CD2, L, grp, rows_by, cached, exby):
    """Actual served EXL3 bytes -> production decode -> source basis, exactly d44.reference."""
    from cd2_alignment import recover_perm, aligned
    src, dec, sizes = {}, {}, {}
    stamps = {}
    for r in grp:
        role = r['role']
        src[role], stamps[role] = S.source(rows_by[r['qname']], cached, 'cpu')
        ent = exby[r['qname']]['formats']['EXL3']
        dec[role], sizes[role] = CD2.decode_served('EXL3', ent, {'exl3': EX_ROOT}, 'cpu', r['qname'])
        S.require(list(dec[role].shape) == ent['rendered_shape'] and torch.isfinite(dec[role]).all().item(),
                  r['qname'] + ': EXL3 actual geometry/nonfinite')
    perm, diag = recover_perm(dec['gate_proj'], src['gate_proj'], dec['up_proj'], src['up_proj'])
    S.require(perm is not None, grp[0]['qname'] + ': source-basis EXL3 permutation unavailable ' + str(diag))
    g, u, d = aligned(perm, dec['gate_proj'], dec['up_proj'], dec['down_proj'])
    aligned_by_role = dict(zip(ROLES, (g, u, d)))
    perm_bytes = perm.cpu().numpy().tobytes()
    group = {'kind': 'exl3_source_basis',
             'perm_hex': perm_bytes.hex(), 'perm_sha256': hashlib.sha256(perm_bytes).hexdigest(),
             'alignment': diag, 'aligned_sha256': {k: L.tensor_sha256(v) for k, v in aligned_by_role.items()},
             'group_wire_shas': {r['role']: exby[r['qname']]['formats']['EXL3']['wire']['wire_sha256'] for r in grp},
             'group_qnames': [r['qname'] for r in grp],
             'group_wire_bytes': {r['role']: sizes[r['role']] for r in grp},
             'group_tensors': aligned_by_role}
    del src, dec, g, u, d
    return group


def retained_source_member(torch, L, qname, cached):
    """The actual retained source BF16 member: byte slice of the model shard, proven equal to the
    safe_open tensor.  Returns (blob, wire_loc, rendered_sha256, shape)."""
    from safetensors import safe_open
    wm = S.load(S.MODEL / 'model.safetensors.index.json')['weight_map']
    name = qname + '.weight'
    shard = S.MODEL / wm[name]
    with open(shard, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    info = hdr[name]
    S.require(info['dtype'] == 'BF16' and info['shape'] == cached[qname]['identity']['source']['shape'],
              qname + ': retained source member geometry')
    off = 8 + n + info['data_offsets'][0]
    ln = info['data_offsets'][1] - info['data_offsets'][0]
    with open(shard, 'rb') as f:
        f.seek(off)
        blob = f.read(ln)
    with safe_open(str(shard), framework='pt', device='cpu') as f:
        t = f.get_slice(name)[:]
    S.require(t.dtype == torch.bfloat16 and list(t.shape) == info['shape'],
              qname + ': retained source tensor geometry')
    rendered = L.tensor_sha256(t)
    S.require(rendered == hashlib.sha256(blob).hexdigest(),
              qname + ': shard member bytes differ from the source tensor bytes')
    loc = {'root': 'exl3', 'ranges': [[str(shard), off, ln]],
           'member_bytes': ln, 'wire_sha256': hashlib.sha256(blob).hexdigest()}
    return blob, loc, rendered, info['shape']


def build_ceiling(a):
    torch, CD2, L, _ = S.setup()
    m, cached, _, _ = S.population()
    exm = S.load(EX_MANIFEST)
    exby = {r['qname']: r for r in exm['rows']}
    baseline_routed_roster(exby)
    import d44
    rows = d44.selection_rows(a.selection, a.capture, m)
    names = [r['qname'] for r in rows]
    S.require(len(names) == len(set(names)), 'Duplicated frozen unit in selection')
    out = copy.deepcopy(m)
    by = {r['qname']: r for r in out['rows']}
    out['picks'] = [CEIL_PICK]
    receipts, groups_done = [], {}
    started = time.monotonic()
    for r0 in rows:
        grow, q = by[r0['qname']], r0['qname']
        a8s_fmt = grow['pick']['a8s']
        S.require(a8s_fmt != 'SOURCE',
                  q + ': frozen unit is SOURCE in A8S production; route to CEO')
        S.require(grow['formats'][a8s_fmt]['contract'] == 'fp8_per_token_dynamic',
                  q + ': A8S activation contract expected on the W+A arm')
        pick = exby[q]['pick']['exl3']
        if pick == 'EXL3':
            grp = group_rows(exby, grow)
            gkey = tuple(r['qname'] for r in grp)
            if gkey not in groups_done:
                group = align_group(torch, CD2, L, grp, by, cached, exby)
                tensors = group.pop('group_tensors')
                groups_done[gkey] = group
            else:
                group = groups_done[gkey]
            role = grow['role']
            ent_exl3 = exby[q]['formats']['EXL3']
            ent = copy.deepcopy(grow['formats'][a8s_fmt])
            ent.update(priced_wire_bytes=ent_exl3['priced_wire_bytes'],
                       rendered_sha256=group['aligned_sha256'][role],
                       rendered_shape=ent_exl3['rendered_shape'])
            ent['wire'] = copy.deepcopy(ent_exl3['wire'])
            ent['d44_g3'] = {'kind': 'exl3_source_basis',
                             'note': 'actual served EXL3 bytes, production exl3_torch decode, cd2_alignment source basis',
                             'exl3_raw_decode_sha256': ent_exl3['rendered_sha256'],
                             'perm_sha256': group['perm_sha256'], 'perm_hex': group['perm_hex'],
                             'alignment': group['alignment'], 'aligned_sha256': group['aligned_sha256'],
                             'group_wire_shas': group['group_wire_shas'], 'group_qnames': group['group_qnames'],
                             'group_wire_bytes': group['group_wire_bytes']}
            receipts.append({'qname': q, 'kind': 'exl3_source_basis',
                             'rendered_sha256': group['aligned_sha256'][role],
                             'perm_sha256': group['perm_sha256'], 'wire_sha256': ent['wire']['wire_sha256'],
                             'group_qnames': group['group_qnames']})
        elif pick == 'SOURCE':
            blob, loc, rendered, shape = retained_source_member(torch, L, q, cached)
            ent = copy.deepcopy(grow['formats'][a8s_fmt])
            ent.update(priced_wire_bytes=loc['member_bytes'], rendered_sha256=rendered,
                       rendered_shape=list(shape))
            ent['wire'] = loc
            ent['d44_g3'] = {'kind': 'retained_source_bf16',
                             'note': 'actual retained source BF16 member bytes; production EXL3 keeps this unit SOURCE',
                             'group_qnames': [q]}
            receipts.append({'qname': q, 'kind': 'retained_source_bf16',
                             'rendered_sha256': rendered, 'wire_sha256': loc['wire_sha256'],
                             'group_qnames': [q]})
        else:
            raise ValueError(q + ': unsupported production EXL3 pick ' + str(pick))
        grow['formats'][CEIL_FMT] = ent
        grow['pick'] = {CEIL_PICK: CEIL_FMT}
    for r in out['rows']:
        if CEIL_PICK not in r['pick']:
            r['pick'] = {CEIL_PICK: r['pick']['a8s']}
    covered = {x['qname'] for x in receipts}
    S.require(covered == set(names), 'Ceiling coverage differs from the frozen roster')
    routed = sorted(q for q in names if exby[q]['kind'] == 'routed')
    shared = sorted(q for q in names if exby[q]['kind'] != 'routed')
    out['d44_g3'] = {'schema': 'd44.frozen_g3_ceiling.v1', 'selection': str(a.selection),
                     'selection_sha256': sha(a.selection),
                     'roster_qnames': sorted(names), 'roster_sha256': roster_sha(names),
                     'routed_qnames': routed, 'routed_roster_sha256': roster_sha(routed),
                     'shared_sites': shared,
                     'split_sha256': S.Captures(a.capture).split_sha256,
                     'ex_manifest': str(EX_MANIFEST), 'ex_manifest_sha256': sha(EX_MANIFEST),
                     'decoder': 'exl3_torch.decode_wire + cd2_alignment.aligned (d44.reference path); '
                                'retained SOURCE units as actual shard member bytes',
                     'reencoded_tessera': False, 'units': len(receipts), 'identities': S.identities()}
    S.save(a.out, out)
    S.save(Path(str(a.out) + '.receipt.json'),
           {'schema': 'd44.frozen_g3_ceiling_receipt.v1', 'manifest_sha256': sha(a.out), 'units': receipts,
            'build_seconds': time.monotonic() - started, 'action_key': os.environ.get('PRISMABUILD_ACTION_KEY')})
    print(json.dumps({'units': len(receipts), 'routed': len(routed), 'shared': len(shared),
                      'manifest_sha256': sha(a.out), 'build_seconds': time.monotonic() - started}), flush=True)


# ------------------------------------------------------------------ decoder dispatch (scoring)
def ceiling_specs(m):
    """{EXL3/source member framing sha256 -> self-sufficient decode spec} for every ceiling unit."""
    ents = {r['qname']: r['formats'][CEIL_FMT] for r in m['rows'] if r['pick'][CEIL_PICK] == CEIL_FMT}
    specs = {}
    for q, ent in ents.items():
        prov = ent['d44_g3']
        out_f, in_f = ent['rendered_shape']
        base = {'qname': q, 'rendered_sha256': ent['rendered_sha256'],
                'wire_sha256': ent['wire']['wire_sha256']}
        if prov['kind'] == 'retained_source_bf16':
            specs[ent['wire']['wire_sha256']] = dict(base, type='source', shape=list(ent['rendered_shape']))
            continue
        members = {}
        for role, gq in zip(ROLES, prov['group_qnames']):
            gent = ents[gq]
            members[role] = {'wire': copy.deepcopy(gent['wire']), 'shape': list(gent['rendered_shape'])}
        specs[ent['wire']['wire_sha256']] = dict(base, type='exl3',
                                                 role=next(r for r in ROLES if prov['group_wire_shas'][r] == ent['wire']['wire_sha256']),
                                                 perm_hex=prov['perm_hex'], members=members)
    return specs


def install_decoder_dispatch(specs):
    """Patch the tessera reader seam the owner plumbs through run_multi (v2_score.main imports it
    at call time).  Every call is stateless: a routed unit loads its complete group inside the
    single call, so sequential and concurrent callers are both safe (no rendezvous, no deadlock)."""
    import tessera.unit_artifact as UA
    original = UA.read_unit_artifact
    if getattr(original, '_d44_g3_dispatch', False):
        raise RuntimeError('ceiling decoder dispatch already installed')
    torch, _, L, _ = S.setup()
    import exl3_torch
    import cd2_alignment
    import codec_factorial as CD2m

    def read_unit_artifact(blob, **kw):
        h = hashlib.sha256(blob).hexdigest()
        if h not in specs:
            return original(blob, **kw)
        device = kw.get('device', 'cpu')
        sp = specs[h]
        if sp['type'] == 'source':
            t = torch.frombuffer(bytearray(blob), dtype=torch.bfloat16).reshape(sp['shape'])
        else:
            dec = {}
            for role, mem in sp['members'].items():
                b = blob if mem['wire']['wire_sha256'] == sp['wire_sha256'] \
                    else CD2m.read_wire({'wire': mem['wire']}, {'exl3': EX_ROOT})
                S.require(hashlib.sha256(b).hexdigest() == mem['wire']['wire_sha256'],
                          sp['qname'] + ': group member wire framing corrupt')
                dec[role] = exl3_torch.decode_wire(b, mem['shape'][1], mem['shape'][0],
                                                   device=device).to(torch.bfloat16)
            perm = torch.frombuffer(bytearray(bytes.fromhex(sp['perm_hex'])),
                                    dtype=torch.int64).clone()
            g, u, d = cd2_alignment.aligned(perm, dec['gate_proj'], dec['up_proj'], dec['down_proj'])
            t = dict(zip(ROLES, (g, u, d)))[sp['role']]
        L.check_identity(t.detach().to('cpu'), sp['rendered_sha256'], sp['qname'] + ' [d44ceiling dispatch]')
        return t.to(device=device)

    read_unit_artifact._d44_g3_dispatch = True
    UA.read_unit_artifact = read_unit_artifact
    return len(specs)


def preflight_ceiling(m, n=1):
    """Early CPU gate before the GPU pass: ceiling units through the already-installed dispatch
    (sequential callers, exactly the D38 dry-run shape) must hit the built rendered identities."""
    _, _, L, _ = S.setup()
    import tessera.unit_artifact as UA
    import codec_factorial as CD2m
    dispatch = UA.read_unit_artifact
    rows = [r for r in m['rows'] if r['pick'][CEIL_PICK] == CEIL_FMT]
    routed = [r for r in rows if r['formats'][CEIL_FMT]['d44_g3']['kind'] == 'exl3_source_basis'][:3 * n]
    shared = [r for r in rows if r['formats'][CEIL_FMT]['d44_g3']['kind'] == 'retained_source_bf16'][:n]
    for r in routed + shared:
        blob = CD2m.read_wire(r['formats'][CEIL_FMT], {'exl3': EX_ROOT})
        S.require(hashlib.sha256(blob).hexdigest() == r['formats'][CEIL_FMT]['wire']['wire_sha256'],
                  r['qname'] + ': ceiling wire framing corrupt at preflight')
        t = dispatch(blob, device='cpu')
        L.check_identity(t, r['formats'][CEIL_FMT]['rendered_sha256'], r['qname'] + ' [ceiling CPU preflight]')
    return {'groups': n, 'units': len(routed) + len(shared), 'shared_units': len(shared)}


# ------------------------------------------------------------------ scoring guard + adapter
def guard_scoring(a, m, kind):
    """Pre-scorer freeze enforcement.  kind 'ceiling': preregistered, NOT yet locked.  kind
    'swap': locked P, unchanged numeric weight-leg comparability, and the ACTUAL replaced rows
    identical to the sealed population.  Provenance drift (file digests, action labels) is
    stamped and recorded, never a hard refusal (D32 development-mode policy); live numeric
    C_gap/status, authoritative roster comparability and authoritative outer receipts stay hard.
    Raises BEFORE the scorer is invoked when any correctness binding fails."""
    prereg_path = getattr(a, 'prereg', None)
    if prereg_path is None:
        raise ValueError('score requires --prereg: the frozen preregistration binding '
                         '(ceiling: registered; swap: sealed by lock) is mandatory before scoring')
    pr = S.load(prereg_path)
    S.require(pr['schema'] == PREREG_SCHEMA, 'Not a d44 G3 preregistration file')
    # live weight-leg comparability: hard on numbers/roster/status, stamp whole-file digest drift
    wl_path = Path(pr['weight_leg']['path'])
    live_sha = sha(wl_path)
    wl = S.load(wl_path)
    S.require(wl.get('C_gap') == pr['weight_leg']['C_gap'],
              'Live C_gap differs from the sealed value')
    S.require(wl.get('status') == pr['weight_leg']['status'],
              'Live weight-leg status differs from the sealed value')
    S.require(wl.get('g3_interface', {}).get('roster_sha256') == ROUTED_ROSTER_SHA256,
              'Live weight-leg roster is not the authoritative 5184-unit routed roster')
    stamps = {'weight_leg': {'recorded_sha256': pr['weight_leg']['sha256'], 'live_sha256': live_sha,
                             'digest_drift': live_sha != pr['weight_leg']['sha256']}}
    routed_set = set(pr['roster']['routed_qnames'])
    if kind == 'ceiling':
        S.require(pr['lock'] is None, 'Preregistration is already sealed: the ceiling must be scored before lock')
        S.require(m.get('picks') == [CEIL_PICK], 'Ceiling scoring needs the d44ceiling namespace manifest')
        dg = m.get('d44_g3', {})
        S.require(dg.get('schema') == 'd44.frozen_g3_ceiling.v1' and not dg.get('reencoded_tessera'),
                  'Ceiling manifest metadata invalid')
        # ACTUAL population from manifest picks, not declared metadata
        actual = actual_ceiling_rows(m)
        routed = sorted(q for q in actual if q in routed_set)
        S.require(roster_sha(routed) == pr['roster']['routed_roster_sha256'],
                  'Actual ceiling routed population is not the full routed roster '
                  f'({len(routed)} of {ROUTED_ROSTER_N} actually substituted)')
        S.require(sorted(q for q in actual if q not in routed_set) == sorted(pr['roster']['shared_sites']),
                  'Actual ceiling shared sites differ from the sealed shared sites')
        S.require(len(actual) == dg.get('units'), 'Declared ceiling unit count differs from actual rows')
    else:
        S.require(pr['lock'] is not None,
                  'Swap scoring is frozen out: lock must seal numeric P (after the ceiling, before the swap)')
        # ACTUAL population from the installed manifest picks, not replaced_units metadata
        actual = actual_swap_rows(m)
        S.require(actual, 'Swap manifest installs no replacement rows')
        declared = sorted(u['qname'] for u in m.get('d42_research', {}).get('replaced_units', []))
        S.require(sorted(actual) == sorted(declared),
                  'replaced_units metadata differs from the rows the scorer will actually install')
        routed = sorted(q for q in actual if q in routed_set)
        S.require(roster_sha(routed) == pr['roster']['routed_roster_sha256'],
                  'Actual swap routed population is not the full routed roster')
        S.require(sorted(q for q in actual if q not in routed_set) == sorted(pr['roster']['shared_sites']),
                  'Actual swap shared sites differ from the sealed shared sites')
    binding = {'kind': kind, 'prereg_sha256': sha(prereg_path),
               'weight_leg': {'status': wl['status'], 'C_gap': wl['C_gap'], 'stamps': stamps},
               'routed_roster_sha256': pr['roster']['routed_roster_sha256'],
               'ceiling_roster_sha256': pr['roster']['ceiling_roster_sha256'],
               'actual_population': {'routed': len(routed), 'total': len(actual)}}
    if pr['lock'] is not None:
        binding.update({'P': pr['lock']['P'], 'dceil': pr['lock']['dceil'],
                        'low_power': pr['lock']['low_power'], 'sealed_at': pr['lock']['sealed_at'],
                        'registered_prereg_sha256': pr['lock']['registered_preregistration']['sha256']})
    return binding


def score_ceiling(a):
    torch, CD2, L, V = S.setup()
    V = owner_v2()
    S.require(a.arm == CEIL_ARM, 'Ceiling scoring must use the ' + CEIL_ARM + ' arm')
    m = S.load(a.manifest)
    specs = ceiling_specs(m)
    import tessera.unit_artifact as UA
    n = install_decoder_dispatch(specs)
    binding = guard_scoring(a, m, 'ceiling')
    S.require(len(specs) == len(actual_ceiling_rows(m)), 'Ceiling spec population differs from actual rows')
    if a.dry_run:
        import numpy as np
        input_proofs = []
        for rec in S.load(V.TEACHER)['arrays']:
            arr = np.load(rec['path'], mmap_mode='r', allow_pickle=False)
            S.require(arr.dtype == np.float32 and arr.shape == (2047, 154880) and np.isfinite(arr[:1, :32]).all(),
                      'Actual v2 teacher slice invalid')
            input_proofs.append({'path': rec['path'], 'shape': list(arr.shape), 'dtype': str(arr.dtype),
                                 'slice_sha256': S.blob_sha(arr[:1, :32].tobytes())})
        import codec_factorial as CD2m
        rows = [r for r in m['rows'] if r['pick'][CEIL_PICK] == CEIL_FMT]
        probe = {}
        for r in (rows[:3] + [x for x in rows if x['formats'][CEIL_FMT]['d44_g3']['kind'] == 'retained_source_bf16'][:1]):
            blob = CD2m.read_wire(r['formats'][CEIL_FMT], {'exl3': EX_ROOT})
            t = UA.read_unit_artifact(blob, device='cpu')
            S.require(list(t.shape) == r['formats'][CEIL_FMT]['rendered_shape'],
                      r['qname'] + ': ceiling dispatch decode shape')
            L.check_identity(t, r['formats'][CEIL_FMT]['rendered_sha256'],
                             r['qname'] + ' [ceiling dry-run dispatch]')
            probe[r['qname']] = r['formats'][CEIL_FMT]['rendered_sha256']
        S.save(Path(a.root) / 'actual-input-slices.json',
               {'teacher_slices': input_proofs, 'ceiling_probe': probe,
                'manifest_sha256': sha(a.manifest), 'guard': binding})
        print(json.dumps({'dry_run': True, 'units_dispatched': n, 'teacher_windows': len(input_proofs)}), flush=True)
        return
    preflight = preflight_ceiling(m)
    sys.argv = [str(S.OWNERS / 'v2_score.py'), '--arm', a.arm, '--manifest', a.manifest,
                '--manifest-sha256', sha(a.manifest), '--root', str(a.root)]
    V.main()
    result_path = Path(a.root) / 'gpu-score' / (a.arm + '-v2') / 'result.json'
    result = S.load(result_path)
    result['d44_g3'] = dict(m['d44_g3'], action_key=os.environ.get('PRISMABUILD_ACTION_KEY'),
                            scoring_adapter='d44_g3.score_ceiling', decoder_dispatch_units=n,
                            ceiling_cpu_preflight=preflight, manifest_sha256=sha(a.manifest),
                            prereg_binding=binding)
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'arm': a.arm, 'mean_kl': result['teacher2']['mean_kl']}), flush=True)


# ------------------------------------------------------------------ preregistration / lock
def arms_inputs(a):
    expected = {'A8S': 'a8s_wa', 'ceiling': CEIL_ARM}
    paths = {'A8S': Path(a.a8s_result)}
    if getattr(a, 'ceiling_result', None):
        paths['ceiling'] = Path(a.ceiling_result)
    data = {}
    for name, path in paths.items():
        r = S.load(path)
        S.require(r.get('arm') == expected[name],
                  name + ': result is arm ' + str(r.get('arm')) + ', not the required ' + expected[name])
        S.require(r['hash_gate']['all_hashes_matched'] and r['v2_teacher']['arrays_verified'] == WINDOWS,
                  name + ': actual complete G3v2 integrity required')
        data[name] = (path, r)
    return data


def check_results_pairable(data):
    ids = {n: [w['window_id'] for w in r['windows']] for n, (_, r) in data.items()}
    S.require(len(ids['A8S']) == len(set(ids['A8S'])) == WINDOWS, 'A8S window population')
    for n, ids_n in ids.items():
        S.require(ids_n == ids['A8S'], 'Paired G3v2 windows differ: ' + n)
    t = {n: r['v2_teacher']['sha256'] for n, (_, r) in data.items()}
    S.require(len(set(t.values())) == 1, 'Never mix teachers/v1')
    return ids['A8S'], t['A8S']


def load_harvest():
    spec = importlib.util.spec_from_file_location('stage1_v2_harvest', V2 / 'harvest.py')
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    return H


def teacher_values(result_path, result, H):
    arr = Path(result_path).parent / 'per_position_kl.teacher2.npy'
    values = H.reduce_array(arr)
    S.require(max(abs(x - w['teacher2']['mean_kl']) for x, w in zip(values, result['windows'])) <= 1e-15,
              str(arr) + ': array reductions disagree')
    return values, arr


def preregister(a):
    """Freeze roster/windows/teacher/weight-leg/formulas BEFORE any ceiling scoring.  Requires no
    future result: the ceiling result is bound later, by lock, before the swap may score."""
    S.setup()
    exm = S.load(EX_MANIFEST)
    exby = {r['qname']: r for r in exm['rows']}
    routed_all = baseline_routed_roster(exby)
    data = arms_inputs(a)
    window_ids, teacher_sha = check_results_pairable(data)
    v2prereg = S.load(V2 / 'preregistration.json')
    S.require(window_ids == v2prereg['window_ids'], 'G3v2 preregistered window population differs')
    wl = {'path': str(a.weight_leg), 'sha256': sha(a.weight_leg)}
    doc = S.load(a.weight_leg)
    S.require(doc.get('schema') == 'd44.frozen_weight_leg.v1', 'weight-leg report schema')
    S.require(isinstance(doc.get('C_gap'), (int, float)) and doc['C_gap'] is not None and math.isfinite(doc['C_gap']),
              'weight-leg report carries no measurable C_gap')
    S.require(doc.get('g3_interface', {}).get('roster_sha256') == ROUTED_ROSTER_SHA256,
              'weight-leg roster is not the full authoritative 5184-unit routed bar roster; '
              'a strict subset can never bind G3')
    m = S.load(a.manifest)
    S.require(m.get('picks') == [CEIL_PICK], 'Preregistration must bind the ceiling manifest')
    dg = m['d44_g3']
    S.require(dg['schema'] == 'd44.frozen_g3_ceiling.v1', 'Ceiling manifest metadata invalid')
    sel = S.load(a.selection)
    S.require(sorted(sel['qnames']) == dg['roster_qnames'], 'Ceiling roster differs from the frozen selection')
    # ACTUAL population from manifest picks; the routed selection must be the full fixed bar
    # roster and the FIT-selected shared winner must be separately included
    actual = actual_ceiling_rows(m)
    routed_actual = sorted(q for q in actual if exby[q]['kind'] == 'routed')
    shared_actual = sorted(q for q in actual if exby[q]['kind'] != 'routed')
    S.require(roster_sha(routed_actual) == ROUTED_ROSTER_SHA256,
              f'Actual ceiling routed population ({len(routed_actual)}) is a strict subset of the '
              'fixed bar roster; refusing to narrow G3')
    S.require(shared_actual == [SHARED_WINNER],
              'The explicit swap selection must separately include the FIT-selected shared winner '
              + SHARED_WINNER)
    out = {'schema': PREREG_SCHEMA,
           'frozen_at': '2026-10-06T21:15:53.606923+00:00',
           'arms': {'swap': PICK + '_wa', 'ceiling': CEIL_ARM},
           'roster': {'routed_qnames': routed_all, 'routed_roster_sha256': ROUTED_ROSTER_SHA256,
                      'shared_sites': shared_actual,
                      'ceiling_roster_sha256': dg['roster_sha256'],
                      'ceiling_manifest_sha256': sha(a.manifest),
                      'selection': str(a.selection), 'selection_sha256': sha(a.selection),
                      'weight_leg_roster_sha256': doc['g3_interface']['roster_sha256']},
           'windows': {'window_ids': window_ids, 'count': WINDOWS,
                       'input_geometry': 'prefix [154822,154824] + 2048 tokens; scored logits[0,2:-1] fp32'},
           'teacher': {'sha256': teacher_sha, 'field': 'teacher2',
                       'emission_receipt': str(owner_v2().TEACHER)},
           'weight_leg': dict(wl, status=doc['status'], C_gap=doc['C_gap'],
                              C_gap_formula='sum(A-N)/sum(A-X) over all routed units',
                              bound_at_preregistration=True),
           'formula': {'dceil': 'meanKL(A8S) - meanKL(EXL3 ceiling), same 25 windows, same teacher',
                       'P': 'P = C_gap * dceil, sealed numerically at lock, fixed before swap scoring',
                       'criterion': 'CONFIRM requires mean(dKL swap-A8S)<0, abs(mean)>=0.5*P, '
                                    'one-sided 90% paired upper interval <0, and >=15/25 better windows',
                       'interval': 'one-sided 90% upper = mean + t_{0.90,24} * SE (paired window deltas)'},
           'thresholds': {'abs_mean_min_times_P': 0.5, 'better_windows_min': 15,
                          'confidence': 0.90, 'degrees_of_freedom': WINDOWS - 1},
           'bootstrap': {'resamples': 2000, 'seed': 20261006, 'stratified_by': 'layer', 'unit': 'Whole experts'},
           'bindings': {'a8s_result_sha256': sha(data['A8S'][0]),
                        'weight_leg_sha256': wl['sha256']},
           'lock': None}
    S.save(a.out, out)
    print(json.dumps({'schema': PREREG_SCHEMA, 'windows': WINDOWS, 'C_gap': doc['C_gap'],
                      'routed_roster_sha256': ROUTED_ROSTER_SHA256,
                      'weight_leg_status': doc['status']}), flush=True)


def require_sealed(prereg_path):
    pr = S.load(prereg_path)
    S.require(pr['schema'] == PREREG_SCHEMA and pr['lock'] is not None,
              'Preregistration must be sealed (lock) before any swap scoring or confirmation')
    return pr


def lock(a):
    """Seals a NEW immutable locked output file (never overwrites the registered preregistration:
    the pre-ceiling formula record is preserved as evidence).  --out must not exist."""
    prereg_path = Path(a.prereg)
    pr = S.load(prereg_path)
    S.require(pr['schema'] == PREREG_SCHEMA and pr['lock'] is None,
              'Preregistration missing or already sealed')
    # live weight-leg comparability hard; whole-file digest drift stamped (D32 development mode)
    wl_live = S.load(pr['weight_leg']['path'])
    wl_drift = {'recorded_sha256': pr['weight_leg']['sha256'],
                'live_sha256': sha(pr['weight_leg']['path']),
                'digest_drift': sha(pr['weight_leg']['path']) != pr['weight_leg']['sha256']}
    S.require(wl_live.get('C_gap') == pr['weight_leg']['C_gap'] and
              wl_live.get('status') == pr['weight_leg']['status'],
              'Live weight-leg C_gap/status differs from the sealed binding')
    data = arms_inputs(a)
    S.require('ceiling' in data, 'Lock binds the completed ceiling result')
    # the A8S baseline P is computed from must be the preregistered A8S object itself: same
    # result bytes first, then the real windows/teacher/array pairing below
    S.require(sha(data['A8S'][0]) == pr['bindings']['a8s_result_sha256'],
              'Supplied A8S result differs from the preregistered A8S baseline object')
    window_ids, teacher_sha = check_results_pairable(data)
    S.require(window_ids == pr['windows']['window_ids'] and teacher_sha == pr['teacher']['sha256'],
              'Locked windows/teacher differ from the preregistration')
    ceil_r = data['ceiling'][1]
    S.require(ceil_r.get('d44_g3', {}).get('schema') == 'd44.frozen_g3_ceiling.v1',
              'Ceiling result is not a d44 ceiling arm result')
    dg = ceil_r['d44_g3']
    # chained population proof: the ceiling score-time guard proved the actual manifest
    # population and stamped the binding into the result; lock consumes it, so P can only be
    # computed from a ceiling that was population-checked at scoring time.
    pb = dg.get('prereg_binding')
    S.require(pb and pb.get('kind') == 'ceiling', 'Ceiling result lacks the score-time guard binding')
    S.require(pb.get('prereg_sha256') == sha(prereg_path),
              'Ceiling result was not population-guard-scored under this preregistration')
    S.require(pb.get('actual_population', {}).get('routed') == ROUTED_ROSTER_N,
              'Score-time population did not cover the full routed roster')
    S.require(dg.get('routed_roster_sha256') == pr['roster']['routed_roster_sha256'],
              'Ceiling result routed roster differs from the sealed routed roster')
    S.require(roster_sha(dg.get('roster_qnames', [])) == pr['roster']['ceiling_roster_sha256'],
              'Ceiling result roster differs from the preregistered ceiling roster')
    provenance = None
    if getattr(a, 'ceiling_receipt', None):
        rec = S.load(a.ceiling_receipt)
        S.require(len(rec) == 1 and rec[0]['status'] == 'executed' and rec[0]['returncode'] == 0,
                  'Ceiling PB terminal receipt not successful')
        key = ceil_r['d44_g3'].get('action_key')
        provenance = {'receipt_observed': {'action_key': rec[0]['action_key'],
                                           'status': rec[0]['status'], 'returncode': rec[0]['returncode']},
                      'recorded_action_key': key,
                      'label_match': key is not None and key == rec[0]['action_key'],
                      'attribution': ('observed-by-receipt-content-only'
                                      if key is not None and key == rec[0]['action_key']
                                      else 'unobserved: recorded container key absent or different')}
        if not provenance['label_match']:
            print(json.dumps({'d44_g3_provenance_drift': provenance}), flush=True)
    H = load_harvest()
    a8s_v, _ = teacher_values(data['A8S'][0], data['A8S'][1], H)
    ceil_v, _ = teacher_values(data['ceiling'][0], ceil_r, H)
    dceil = statistics.mean(a8s_v) - statistics.mean(ceil_v)
    paired = [x - y for x, y in zip(a8s_v, ceil_v)]
    se = statistics.stdev(paired) / math.sqrt(WINDOWS)
    low_power = dceil < 2 * se
    from scipy.stats import t as student_t
    t90 = float(student_t.ppf(pr['thresholds']['confidence'], pr['thresholds']['degrees_of_freedom']))
    P = pr['weight_leg']['C_gap'] * dceil
    locked = dict(pr)
    locked['bindings'] = dict(pr['bindings'], ceiling_result_sha256=sha(data['ceiling'][0]))
    locked['lock'] = {'schema': 'd44.frozen_g3_lock.v1',
                      'sealed_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                      'registered_preregistration': {'path': str(prereg_path), 'sha256': sha(prereg_path)},
                      'dceil': dceil, 'paired_se': se, 'low_power': low_power,
                      'low_power_rule': 'dceil < 2*paired SE -> weight-leg only, routed to CEO; '
                                        'no frozen P borrowed from the global EXL3 arm',
                      'P': P, 't90_df24': t90, 'student_t_source': 'scipy.stats.t.ppf',
                      'ceiling_mean_kl': statistics.mean(ceil_v), 'a8s_mean_kl': statistics.mean(a8s_v),
                      'ceiling_result_sha256': sha(data['ceiling'][0]),
                      'provenance_stamps': {'weight_leg': wl_drift, 'ceiling_receipt': provenance}}
    S.save(a.out, locked)
    print(json.dumps(locked['lock']), flush=True)


# ------------------------------------------------------------------ confirmation reducer
EXPECTED_ARMS = {'swap': PICK + '_wa', 'ceiling': CEIL_ARM, 'A8S': 'a8s_wa'}


def arm_lateral(name, result_path, pr, H):
    r = S.load(result_path)
    S.require(r.get('arm') == EXPECTED_ARMS[name],
              name + ': result is arm ' + str(r.get('arm')) + ', not the required ' + EXPECTED_ARMS[name])
    S.require(r['hash_gate']['all_hashes_matched'] and r['v2_teacher']['arrays_verified'] == WINDOWS,
              name + ': actual complete G3v2 integrity required')
    ids = [w['window_id'] for w in r['windows']]
    S.require(len(ids) == len(set(ids)) == WINDOWS, name + ': window population')
    S.require(ids == pr['windows']['window_ids'], name + ': preregistered window population differs')
    S.require(r['v2_teacher']['sha256'] == pr['teacher']['sha256'],
              name + ': teacher sha differs from preregistration')
    values, arr_path = teacher_values(result_path, r, H)
    return {'result': r, 'values': values,
            'array_sha256': sha(arr_path), 'result_sha256': sha(result_path)}


def result_action_key(result):
    """Recorded container action label, or None when absent (null keys are demonstrated by v0)."""
    for holder in ('d44_g3', 'd42_research'):
        if isinstance(result.get(holder), dict) and result[holder].get('action_key'):
            return result[holder]['action_key']
    return None


def receipt_binding(result, receipt_path, name, stamps):
    """Authoritative outer receipt (executed, returncode 0) stays a hard refusal.  Recorded
    container action labels only ever stamp: no attestation or qualification is claimed from
    them.  Real input/output linkage is the declared receipt content beside the result bytes
    and the scored manifest binding carried in the result."""
    rec = S.load(receipt_path)
    S.require(len(rec) == 1 and rec[0]['status'] == 'executed' and rec[0]['returncode'] == 0,
              name + ' PB terminal receipt not successful')
    key = result_action_key(result)
    entry = {'receipt_observed': {'action_key': rec[0]['action_key'], 'status': rec[0]['status'],
                                 'returncode': rec[0]['returncode']},
             'recorded_action_key': key,
             'label_match': key is not None and key == rec[0]['action_key'],
             'attribution': ('observed-by-receipt-content-only' if key is not None and key == rec[0]['action_key']
                             else 'unobserved: recorded container key absent or different from the receipt')}
    stamps[name] = entry
    if not entry['label_match']:
        print(json.dumps({'d44_g3_provenance_drift': dict(entry, arm=name)}), flush=True)
    return entry


def confirm(a):
    S.setup()
    locked_path = Path(a.prereg)
    pr = require_sealed(locked_path)
    H = load_harvest()
    stamps = {}
    swap = arm_lateral('swap', Path(a.swap_result), pr, H)
    ceiling = arm_lateral('ceiling', Path(a.ceiling_result), pr, H)
    a8s = arm_lateral('A8S', Path(a.a8s_result), pr, H)
    receipt_binding(swap['result'], a.swap_receipt, 'swap', stamps)
    receipt_binding(ceiling['result'], a.ceiling_receipt, 'ceiling', stamps)
    # the confirmed data must be exactly the data P was computed from (paired comparability)
    S.require(sha(a.ceiling_result) == pr['lock']['ceiling_result_sha256'],
              'Ceiling result differs from the one bound into the sealed lock (P data)')
    S.require(sha(a.a8s_result) == pr['bindings']['a8s_result_sha256'],
              'A8S result differs from the preregistered pairing data')
    stamped = swap['result'].get('d44_g3_prereg')
    S.require(stamped and stamped.get('prereg_sha256') == sha(locked_path),
              'Swap result was not scored under this sealed locked preregistration')
    S.require(stamped.get('P') == pr['lock']['P'] and stamped.get('dceil') == pr['lock']['dceil'],
              'Swap result scorer binding carries a different numeric P/dceil than the locked seal')
    swap_units = sorted(u['qname'] for u in swap['result']['d42_research']['replaced_units'])
    routed = sorted(q for q in swap_units if q in set(pr['roster']['routed_qnames']))
    S.require(roster_sha(routed) == pr['roster']['routed_roster_sha256'],
              'Swap routed roster differs from the sealed routed roster')
    S.require(sorted(q for q in swap_units if q not in set(routed)) == sorted(pr['roster']['shared_sites']),
              'Swap shared sites differ from the sealed shared sites')
    ceiling_roster = sorted(ceiling['result']['d44_g3'].get('roster_qnames', []))
    S.require(ceiling_roster == sorted(routed + pr['roster']['shared_sites']),
              'Ceiling roster differs from the sealed swap roster')
    # live weight-leg comparability hard; whole-file digest drift stamped
    wl_path = Path(pr['weight_leg']['path'])
    live_sha = sha(wl_path)
    wl = S.load(wl_path)
    S.require(wl['status'] == pr['weight_leg']['status'] and wl['C_gap'] == pr['weight_leg']['C_gap'],
              'Live weight-leg verdict or C_gap differs from the sealed binding')
    stamps['weight_leg'] = {'recorded_sha256': pr['weight_leg']['sha256'], 'live_sha256': live_sha,
                            'digest_drift': live_sha != pr['weight_leg']['sha256']}
    deltas = [s - x for s, x in zip(swap['values'], a8s['values'])]
    mean = statistics.mean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(WINDOWS)
    better = sum(d < 0 for d in deltas)
    t90 = pr['lock']['t90_df24']
    upper90 = mean + t90 * se
    P = pr['lock']['P']
    dceil = pr['lock']['dceil']
    gates = {'mean_negative': mean < 0, 'abs_mean_ge_half_P': abs(mean) >= 0.5 * P,
             'one_sided_90_upper_negative': upper90 < 0, 'better_windows_ge_15': better >= 15}
    if wl['status'] == 'PASS-except-L3':
        verdict = 'PASS_EXCEPT_L3_ROUTE_CEO'
    elif pr['lock']['low_power']:
        verdict = 'LOW_POWER_WEIGHT_LEG_ONLY_ROUTE_CEO'
    elif wl['status'] != 'PASS':
        verdict = 'FAIL_SHIP_A8S'
    elif mean >= 0:
        verdict = 'FLAT_OR_WORSE_STOP_DIAGNOSIS'
    elif all(gates.values()):
        verdict = 'CONFIRM_PASS'
    else:
        verdict = 'FAIL_SHIP_A8S'
    report = {'schema': CONFIRM_SCHEMA,
              'verdict': verdict,
              'preregistration_sha256': sha(a.prereg),
              'weight_leg': {'status': wl['status'], 'C_gap': wl['C_gap'],
                             'roster_sha256': wl['g3_interface']['roster_sha256'],
                             'bound_sha256': pr['weight_leg']['sha256']},
              'ceiling': {'mean_kl': statistics.mean(ceiling['values']), 'dceil': dceil,
                          'paired_se': pr['lock']['paired_se'], 'low_power': pr['lock']['low_power'],
                          'result_sha256': ceiling['result_sha256'], 'array_sha256': ceiling['array_sha256']},
              'P': P,
              'paired': {'mean_dKL': mean, 'se': se, 't90_df24': t90, 'upper_one_sided_90': upper90,
                         'better_windows': better, 'worse_windows': sum(d > 0 for d in deltas),
                         'tied_windows': sum(d == 0 for d in deltas), 'n': WINDOWS,
                         'per_window_deltas': deltas},
              'gates': gates,
              'windows': {'window_ids': pr['windows']['window_ids']},
              'teacher_sha256': pr['teacher']['sha256'],
              'result_sha256': {'swap': swap['result_sha256'], 'ceiling': ceiling['result_sha256'],
                                'A8S': a8s['result_sha256']},
              'array_sha256': {'swap': swap['array_sha256'], 'ceiling': ceiling['array_sha256'],
                               'A8S': a8s['array_sha256']},
              'receipts': {'swap': str(a.swap_receipt), 'ceiling': str(a.ceiling_receipt)},
              'provenance_stamps': stamps}
    S.save(a.out, report)
    print(json.dumps({'verdict': verdict, 'mean_dKL': mean, 'se': se, 'P': P}), flush=True)


# ------------------------------------------------------------------ real CPU smoke
def smoke(a):
    torch, CD2, L, V = S.setup()
    V = owner_v2()
    m = S.load(a.manifest) if a.manifest and Path(a.manifest).exists() else None
    exm = S.load(EX_MANIFEST)
    exby = {r['qname']: r for r in exm['rows']}
    if m is not None:
        S.require(m.get('picks') == [CEIL_PICK], 'Smoke manifest must be the ceiling manifest')
        cands = [r for r in m['rows'] if r['pick'][CEIL_PICK] == CEIL_FMT]
        r0 = min(cands, key=lambda r: r['formats'][CEIL_FMT]['priced_wire_bytes'])
        S.require(r0['qname'] in exby, 'Smoke unit missing from the EX manifest')
    else:
        routed = [r for r in exm['rows'] if r['layer'] in S.LAYERS and r['role'] in ROLES
                  and r['pick']['exl3'] == 'EXL3']
        r0 = min(routed, key=lambda r: r['formats']['EXL3']['priced_wire_bytes'])
    q = r0['qname']
    group = group_rows(exby, {'qname': q, 'layer': r0['layer'], 'role': r0['role']})
    ent = exby[q]['formats']['EXL3']
    blob = CD2.read_wire(ent, {'exl3': EX_ROOT})
    S.require(hashlib.sha256(blob).hexdigest() == ent['wire']['wire_sha256'], q + ': EXL3 wire framing corrupt')
    import exl3_torch
    out_f, in_f = ent['rendered_shape']
    raw = exl3_torch.decode_wire(blob, in_f, out_f, device='cpu').to(torch.bfloat16)
    L.check_identity(raw, ent['rendered_sha256'], q + ' [smoke production EXL3 decode]')
    m0, cached, _, _ = S.population()
    rows_by = {r['qname']: r for r in m0['rows']}
    from cd2_alignment import recover_perm, aligned
    src = {r['role']: S.source(rows_by[r['qname']], cached, 'cpu')[0] for r in group}
    dec = {}
    for r in group:
        e = exby[r['qname']]['formats']['EXL3']
        b = CD2.read_wire(e, {'exl3': EX_ROOT})
        o, i = e['rendered_shape']
        dec[r['role']] = exl3_torch.decode_wire(b, i, o, device='cpu').to(torch.bfloat16)
    perm, diag = recover_perm(dec['gate_proj'], src['gate_proj'], dec['up_proj'], src['up_proj'])
    S.require(perm is not None, q + ': smoke permutation unavailable ' + str(diag))
    g, u, d = aligned(perm, dec['gate_proj'], dec['up_proj'], dec['down_proj'])
    aligned_sha = {k: L.tensor_sha256(v) for k, v in zip(ROLES, (g, u, d))}
    if m is not None:
        ent0 = r0['formats'][CEIL_FMT]
        S.require(aligned_sha[r0['role']] == ent0['rendered_sha256'],
                  q + ': smoke aligned decode does not reproduce the built ceiling identity')
    # the FIT-selected shared site: actual retained source member, byte-proven
    shared_q = 'model.language_model.layers.39.mlp.shared_experts.gate_proj'
    blob_s, loc_s, rendered_s, shape_s = retained_source_member(torch, L, shared_q, cached)
    import numpy as np
    rec = S.load(V.TEACHER)['arrays'][0]
    arr = np.load(rec['path'], mmap_mode='r', allow_pickle=False)
    S.require(arr.dtype == np.float32 and arr.shape == (2047, 154880) and np.isfinite(arr[:1, :32]).all(),
              'Actual v2 teacher slice invalid')
    proof = {'schema': 'd44.frozen_g3_smoke.v1', 'device': 'cpu', 'smoke_unit': q,
             'exl3_wire_bytes': len(blob), 'exl3_raw_decode_sha256': ent['rendered_sha256'],
             'aligned_sha256': aligned_sha, 'alignment': diag,
             'perm_sha256': hashlib.sha256(perm.cpu().numpy().tobytes()).hexdigest(),
             'shared_site': {'qname': shared_q, 'wire_sha256': loc_s['wire_sha256'],
                             'rendered_sha256': rendered_s, 'shape': list(shape_s),
                             'bytes': loc_s['member_bytes']},
             'teacher_window': rec['window_id'], 'teacher_shape': list(arr.shape),
             'teacher_slice_sha256': S.blob_sha(arr[:1, :32].tobytes()),
             'manifest_bound': bool(m), 'action_key': os.environ.get('PRISMABUILD_ACTION_KEY')}
    if a.out:
        S.save(a.out, proof)
    print(json.dumps(proof), flush=True)


# ------------------------------------------------------------------ readset
def readset(a):
    """D38 data manifest for the ceiling arm: identical construction to stage1.readset, with the
    ceiling pick registered so source reads cover exactly the rows the arm executes."""
    V = owner_v2()
    G, _ = V.setup()
    m, by_layer, _ = G.load_manifest(a.manifest, sha(a.manifest), None)
    G.register_picks(m)
    from g3_readset import SourceReads, build_manifest
    reads = SourceReads(str(S.MODEL), [CEIL_ARM], by_layer)
    emission = S.load(V.TEACHER)
    windows = emission['windows']
    spec = {'windows': [{'window_id': r['window_id'], 'path': r['path'],
                         'bytes': Path(r['path']).stat().st_size, 'sha256': r['file_sha256']}
                        for r in emission['arrays']]}
    teacher_root = Path(a.root) / 'teacher-metadata'
    teacher_root.mkdir(parents=True, exist_ok=True)
    S.save(teacher_root / 'teacher.json', spec)
    h = S.load(V.HANDOFF)
    roots = {'a8': str(S.A8), 'wirecache': str(S.WIRE), 't8r': h['binding']['roots']['t8r'], 'exl3': EX_ROOT}
    dm = build_manifest(reads, [CEIL_ARM], by_layer, roots, [(str(teacher_root), spec)], windows,
                        setup_files=[a.manifest, V.TEACHER, V.HANDOFF, V.PROBE,
                                     S.CENSUS / 'tr3-teacher-inputs-01/final_panel_handoff.json'])
    smoke_e = S.load(S.BASE / 'g3-readset-2301/smoke-input-01/data-manifest.json')['entries'][0]
    dm['entries'].append(smoke_e)
    dm['entry_count'] = len(dm['entries'])
    dm['total_bytes'] = sum(e['bytes'] for e in dm['entries'])
    phase = dm['read_plan']['phases'][0]
    phase['entry_indices'].append(len(dm['entries']) - 1)
    phase['bytes'] += smoke_e['bytes']
    for phase in dm['read_plan']['phases']:
        phase['cumulative_bytes'] += smoke_e['bytes']
    dm['read_plan']['read_bytes'] += smoke_e['bytes']
    S.save(a.out, dm)
    print(json.dumps({'entries': dm['entry_count'], 'bytes': dm['total_bytes']}), flush=True)


def selftest(a):
    """Bounded CPU proofs on real bytes.  All fixture documents are labeled fixtures used only to
    exercise gate wiring; nothing here is a scientific result and nothing is submitted for scoring."""
    torch, CD2, L, _ = S.setup()
    V = owner_v2()
    import codec_factorial as CD2m
    import tessera.unit_artifact as UA
    from cd2_alignment import recover_perm, aligned
    exm = S.load(EX_MANIFEST)
    exby = {r['qname']: r for r in exm['rows']}
    out = {'schema': 'd44.frozen_g3_selftest.v1', 'action_key': os.environ.get('PRISMABUILD_ACTION_KEY')}

    # --- 1. sequential-dispatch reproduction (old rendezvous deadlock vs stateless dispatch)
    q0 = 'model.language_model.layers.3.mlp.experts.0.down_proj'
    grp = group_rows(exby, {'qname': q0, 'layer': 3, 'role': 'down_proj'})
    wires, shapes = {}, {}
    for r in grp:
        e = exby[r['qname']]['formats']['EXL3']
        wires[r['role']] = CD2m.read_wire(e, {'exl3': EX_ROOT})
        shapes[r['role']] = e['rendered_shape']
    import exl3_torch
    old = {'cv': threading.Condition(), 'raw': {}, 'done': None, 'timed_out': None}

    def old_rendezvous(role):
        # the rejected rule: first caller waits for all three roles before returning
        with old['cv']:
            if old['done'] is None:
                if not old['raw'].get(role):
                    old['raw'][role] = True
                if sum(old['raw'].values()) < 3:
                    old['timed_out'] = not old['cv'].wait(2.0)
                    return
                old['done'] = True
                old['cv'].notify_all()
    t = threading.Thread(target=old_rendezvous, args=('down_proj',))
    t.start(); t.join(5.0)
    S.require(old['timed_out'] is True, 'old rendezvous did not reproduce the deadlock')
    out['deadlock_reproduced'] = 'old per-call group rendezvous times out with one sequential caller'

    # build the real minimal dispatch specs for this one group (same builder code path as a manifest)
    m0, cached, _, _ = S.population()
    rows_by = {r['qname']: r for r in m0['rows']}
    group = align_group(torch, CD2, L, grp, rows_by, cached, exby)
    tensors = group.pop('group_tensors')
    specs = {}
    for r in grp:
        e = exby[r['qname']]['formats']['EXL3']
        members = {rr['role']: {'wire': copy.deepcopy(exby[rr['qname']]['formats']['EXL3']['wire']),
                                'shape': exby[rr['qname']]['formats']['EXL3']['rendered_shape']} for rr in grp}
        specs[e['wire']['wire_sha256']] = {'type': 'exl3', 'qname': r['qname'],
                                           'rendered_sha256': group['aligned_sha256'][r['role']],
                                           'wire_sha256': e['wire']['wire_sha256'],
                                           'role': r['role'], 'perm_hex': group['perm_hex'],
                                           'members': members}
    dispatch = install_decoder_dispatch(specs)
    got = {}
    for r in grp:  # strictly sequential callers, arbitrary role order
        e = exby[r['qname']]['formats']['EXL3']
        t = UA.read_unit_artifact(wires[r['role']], device='cpu')
        L.check_identity(t, group['aligned_sha256'][r['role']], r['qname'] + ' [selftest sequential dispatch]')
        got[r['role']] = group['aligned_sha256'][r['role']]
    S.require(got == group['aligned_sha256'], 'sequential dispatch identities differ')
    out['sequential_dispatch'] = {'order': 'down_proj first, then gate/up', 'identities': got}

    # --- 2. shared site: byte-proven retained source + source dispatch
    shared_q = 'model.language_model.layers.39.mlp.shared_experts.gate_proj'
    blob_s, loc_s, rendered_s, shape_s = retained_source_member(torch, L, shared_q, cached)
    sspec = {loc_s['wire_sha256']: {'type': 'source', 'qname': shared_q, 'shape': list(shape_s),
                                    'rendered_sha256': rendered_s, 'wire_sha256': loc_s['wire_sha256']}}
    specs.update(sspec)
    t = UA.read_unit_artifact(blob_s, device='cpu')
    L.check_identity(t, rendered_s, shared_q + ' [selftest source dispatch]')
    out['shared_site'] = {'qname': shared_q, 'bytes': loc_s['member_bytes'],
                          'rendered_sha256': rendered_s, 'wire_sha256': loc_s['wire_sha256']}

    # --- 3. minimal REAL ceiling manifest from the actual builder path (fixture roster; the
    # fixture declares the full authoritative routed roster so gate wiring is exercised, and is
    # labeled selftest_fixture; it is never scored scientifically)
    fixture_dir = Path(str(a.out) + '.fixtures-' + (os.environ.get('PRISMABUILD_ACTION_KEY') or 'local')[:16])
    fixture_dir.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(m0)
    by = {r['qname']: r for r in m['rows']}
    m['picks'] = [CEIL_PICK]
    ent_shared = copy.deepcopy(by[shared_q]['formats'][by[shared_q]['pick']['a8s']])
    ent_shared.update(priced_wire_bytes=loc_s['member_bytes'], rendered_sha256=rendered_s,
                      rendered_shape=list(shape_s))
    ent_shared['wire'] = loc_s
    ent_shared['d44_g3'] = {'kind': 'retained_source_bf16', 'group_qnames': [shared_q]}
    shared_a8s = by[shared_q]['pick']['a8s']
    by[shared_q]['formats'][CEIL_FMT] = ent_shared
    by[shared_q]['pick'] = {CEIL_PICK: CEIL_FMT}
    shared_swap_fmt = PICK + '::' + shared_a8s
    by[shared_q]['formats'][shared_swap_fmt] = copy.deepcopy(by[shared_q]['formats'][shared_a8s])
    by[shared_q]['pick'][PICK] = shared_swap_fmt
    routed_q = sorted(group['group_qnames'])
    routed_all = baseline_routed_roster(exby)
    for q in routed_q:
        role = q.rsplit('.', 1)[1]
        e_ex = exby[q]['formats']['EXL3']
        ent = copy.deepcopy(by[q]['formats'][by[q]['pick']['a8s']])
        ent.update(priced_wire_bytes=e_ex['priced_wire_bytes'],
                   rendered_sha256=group['aligned_sha256'][role],
                   rendered_shape=e_ex['rendered_shape'])
        ent['wire'] = copy.deepcopy(e_ex['wire'])
        ent['d44_g3'] = {'kind': 'exl3_source_basis', 'perm_hex': group['perm_hex'],
                         'perm_sha256': group['perm_sha256'], 'alignment': group['alignment'],
                         'aligned_sha256': group['aligned_sha256'],
                         'group_wire_shas': group['group_wire_shas'],
                         'group_qnames': group['group_qnames'],
                         'group_wire_bytes': group['group_wire_bytes'],
                         'exl3_raw_decode_sha256': e_ex['rendered_sha256']}
        by[q]['formats'][CEIL_FMT] = ent
        a8s_name = by[q]['pick']['a8s']
        by[q]['pick'] = {CEIL_PICK: CEIL_FMT}
        swap_fmt = PICK + '::' + a8s_name
        by[q]['formats'][swap_fmt] = copy.deepcopy(by[q]['formats'][a8s_name])
        by[q]['pick'][PICK] = swap_fmt
    for r in m['rows']:
        if CEIL_PICK not in r['pick']:
            r['pick'] = {CEIL_PICK: r['pick']['a8s']}
    routed_all = baseline_routed_roster(exby)
    m['d44_g3'] = {'schema': 'd44.frozen_g3_ceiling.v1',
                   'roster_qnames': sorted(routed_q + [shared_q]),
                   'roster_sha256': roster_sha(routed_q + [shared_q]),
                   'routed_qnames': sorted(routed_q), 'routed_roster_sha256': roster_sha(routed_q),
                   'shared_sites': [shared_q], 'reencoded_tessera': False, 'units': 4,
                   'selftest_fixture': 'partial 3-unit CPU fixture; must be refused by population enforcement'}
    # fixture swap view: the guard reads replaced_units; declare the full sealed rosters
    m['d42_research'] = {'replaced_units': [{'qname': q} for q in sorted(routed_all + [shared_q])],
                         'selftest_fixture': True}
    man_path = fixture_dir / 'selftest-ceiling-manifest.json'
    S.save(man_path, m)

    # --- 4. roster enforcement on the real preregister path.  The CPU fixture manifest actually
    # substitutes only 3 routed units; it must be refused even by a full-roster weight leg, and
    # a strict-subset weight leg must be refused at the roster check.  A full-roster acceptance is
    # proven later with the real frozen selection, never with this fixture.
    def fixture_wl(roster_qnames):
        return {'schema': 'd44.frozen_weight_leg.v1', 'C_gap': 0.61, 'status': 'PASS',
                'g3_interface': {'roster_sha256': roster_sha(roster_qnames), 'weight_leg_status': 'PASS',
                                 'method_sha256': 'fixture'}, 'fixture': True}
    wl_full = fixture_dir / 'selftest-wl-full.json'
    wl_subset = fixture_dir / 'selftest-wl-subset.json'
    S.save(wl_full, fixture_wl(routed_all))
    S.save(wl_subset, fixture_wl(sorted(routed_q)))
    refused = {}
    for wl_path, key in ((wl_subset, 'subset'), (wl_full, 'fixture_full_wl')):
        prereg_path = fixture_dir / ('selftest-prereg-%s.json' % key)
        ns = argparse.Namespace(a8s_result=a.a8s_result, weight_leg=wl_path, selection=None,
                                manifest=str(man_path), out=prereg_path)
        try:
            real_preregister(ns)
            raise SystemExit(key + ' preregistration was not refused')
        except ValueError as exc:
            refused[key] = str(exc)
    S.require('full authoritative' in refused['subset'], 'subset refused for the wrong reason')
    S.require('strict subset' in refused['fixture_full_wl'], 'fixture passed population enforcement')
    out['subset_refusal'] = {'strict_subset_weight_leg': refused['subset'],
                             'partial_manifest_with_full_weight_leg': refused['fixture_full_wl'],
                             'routed_roster_sha256': ROUTED_ROSTER_SHA256,
                             'note': 'full-roster acceptance is proven with the real frozen selection, never with a fixture'}

    # --- 5. registered preregistration for gate wiring, schema-faithful with REAL windows/teacher/
    # roster digests and an explicitly fixture C_gap.  All other real checks still apply.
    sel_path = _tmp_selection(sorted(routed_q + [shared_q]))
    H = load_harvest()
    a8s_res = S.load(a.a8s_result)
    real_prereg = {'schema': PREREG_SCHEMA, 'frozen_at': '2026-10-06T21:15:53.606923+00:00',
                   'arms': {'swap': PICK + '_wa', 'ceiling': CEIL_ARM},
                   'roster': {'routed_qnames': routed_all, 'roster_actual_source': 'baseline manifest',
                              'routed_roster_sha256': ROUTED_ROSTER_SHA256,
                              'shared_sites': [shared_q],
                              'ceiling_roster_sha256': roster_sha(sorted(routed_q + [shared_q])),
                              'ceiling_manifest_sha256': sha(man_path),
                              'selection': str(sel_path), 'selection_sha256': sha(sel_path),
                              'weight_leg_roster_sha256': ROUTED_ROSTER_SHA256},
                   'windows': {'window_ids': [w['window_id'] for w in a8s_res['windows']], 'count': WINDOWS,
                               'input_geometry': 'prefix [154822,154824] + 2048 tokens; scored logits[0,2:-1] fp32'},
                   'teacher': {'sha256': a8s_res['v2_teacher']['sha256'], 'field': 'teacher2',
                               'emission_receipt': str(owner_v2().TEACHER)},
                   'weight_leg': {'path': str(wl_full), 'sha256': sha(wl_full), 'status': 'PASS', 'C_gap': 0.61,
                                  'C_gap_formula': 'fixture', 'bound_at_preregistration': True,
                                  'fixture': 'C_gap/status are fixture numbers for wiring only'},
                   'formula': {'dceil': 'meanKL(A8S) - meanKL(EXL3 ceiling), same 25 windows, same teacher',
                               'P': 'P = C_gap * dceil, sealed numerically at lock',
                               'criterion': 'CONFIRM requires mean(dKL swap-A8S)<0, abs(mean)>=0.5*P, '
                                            'one-sided 90% paired upper interval <0, and >=15/25 better windows',
                               'interval': 'one-sided 90% upper = mean + t_{0.90,24} * SE'},
                   'thresholds': {'abs_mean_min_times_P': 0.5, 'better_windows_min': 15,
                                  'confidence': 0.90, 'degrees_of_freedom': WINDOWS - 1},
                   'bootstrap': {'resamples': 2000, 'seed': 20261006, 'stratified_by': 'layer',
                                 'unit': 'Whole experts'},
                   'bindings': {'a8s_result_sha256': sha(a.a8s_result), 'weight_leg_sha256': sha(wl_full)},
                   'lock': None,
                   'selftest_fixture': 'preregistration document wiring fixture; needs the real manifest'}
    v2prereg = S.load(V2 / 'preregistration.json')
    S.require(real_prereg['windows']['window_ids'] == v2prereg['window_ids'],
              'fixture windows differ from the G3v2 preregistered population')
    swap_guard_prereg = fixture_dir / 'selftest-prereg-wiring.json'
    S.save(swap_guard_prereg, real_prereg)
    class A: pass
    ga = A(); ga.prereg = swap_guard_prereg
    try:
        guard_scoring(ga, m, 'swap')
        raise SystemExit('swap guard accepted an unlocked preregistration')
    except ValueError as exc:
        out['swap_guard_unlocked'] = 'refused: ' + str(exc)
    # --- 5b. swap derivation proof on real manifest structures: the rows the scorer would
    # install come from picks, and contradict the fixture's metadata-complete replaced_units
    derived = actual_swap_rows(m)
    declared = sorted(u['qname'] for u in m.get('d42_research', {}).get('replaced_units', []))
    out['swap_derivation'] = {'installed_picks_derived': len(derived),
                              'metadata_declared': len(declared),
                              'derived_sample': derived[:4], 'mismatch': sorted(derived) != sorted(declared)}
    S.require(sorted(derived) != sorted(declared),
              'fixture does not demonstrate picks-vs-metadata divergence')
    # --- 5c. receipt linkage on the real A8S result with a fixture outer receipt: executed rc0
    # passes hard; the recorded label (null here) only stamps attribution, never attestation
    rec_path = fixture_dir / 'selftest-outer-receipt.json'
    S.save(rec_path, [{'status': 'executed', 'returncode': 0, 'action_key': 'other-action'}])
    stamps5c = {}
    entry5c = receipt_binding(a8s_res, rec_path, 'A8S', stamps5c)
    out['receipt_linkage'] = entry5c
    S.require(entry5c['label_match'] is False and 'unobserved' in entry5c['attribution'],
              'receipt linkage misstates a null/mismatched label')
    # --- 6a. the same guard function in-process: must refuse the partial fixture population
    try:
        guard_scoring(argparse.Namespace(prereg=swap_guard_prereg), m, 'ceiling')
        raise SystemExit('guard accepted the partial fixture population in-process')
    except ValueError as exc:
        out['guard_in_process'] = 'refused: ' + str(exc)
    S.require('Actual ceiling routed population' in out['guard_in_process'],
              'in-process refusal is not the population check')

    # --- 6. ACTUAL stage1 score --dry-run entry now enforces population BEFORE the scorer:
    #        the partial fixture manifest must be refused, with the population reason visible.
    import subprocess
    proc = subprocess.run([sys.executable, str(HERE / 'stage1.py'), 'score', '--arm', CEIL_ARM,
                           '--manifest', str(man_path), '--root', str(fixture_dir / 'dryrun'),
                           '--prereg', swap_guard_prereg, '--dry-run'],
                          capture_output=True, text=True, cwd=str(HERE))
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-30:]
    out['score_dry_run'] = {'returncode': proc.returncode, 'tail': tail}
    (fixture_dir / 'dryrun-output.txt').write_text('RC=%d\nSTDOUT:\n%s\nSTDERR:\n%s\n'
                                                    % (proc.returncode, proc.stdout, proc.stderr))
    S.require(proc.returncode != 0, 'partial fixture manifest was not refused at the guard')
    S.require('Actual ceiling routed population' in (proc.stdout + proc.stderr),
              'refusal reason is not the population check')
    # A ceiling result without it (like this fixture, which never passed the score guard) is
    # refused.  The immutable locked file, re-lock refusal and the sealed swap-guard pass run
    # with the real full-population ceiling result, which alone carries the stamped proof.
    ceil_res = copy.deepcopy(S.load(a.a8s_result))
    ceil_res['d44_g3'] = {'schema': 'd44.frozen_g3_ceiling.v1', 'fixture': True}
    ceil_res['arm'] = CEIL_ARM
    ceil_dir = fixture_dir / 'selftest-fixture-ceiling-result-dir'
    ceil_dir.mkdir(parents=True, exist_ok=True)
    S.save(ceil_dir / 'result.json', ceil_res)
    import shutil
    a8s_arr = Path(a.a8s_result).parent / 'per_position_kl.teacher2.npy'
    shutil.copy(a8s_arr, ceil_dir / 'per_position_kl.teacher2.npy')
    try:
        lock(argparse.Namespace(prereg=swap_guard_prereg,
                                out=fixture_dir / 'selftest-locked.json',
                                a8s_result=a.a8s_result, ceiling_result=str(ceil_dir / 'result.json')))
        raise SystemExit('lock accepted a ceiling result without the score-time guard binding')
    except ValueError as exc:
        out['lock_refusal'] = 'refused: ' + str(exc)
    S.require('score-time guard binding' in out['lock_refusal'], 'lock refused for the wrong reason')
    out['real_ceiling_pending'] = {'immutable_lock': 'runs with the real population-guarded ceiling result',
                                   'swap_guard_sealed': 'runs on the immutable locked file after lock'}
    # --- 7b. substituted-baseline regression through the REAL lock CLI: an A8S result object
    # different from the preregistered baseline must be refused BEFORE any P is computed.  The
    # old lock code bound no A8S object; this regression fails against it and passes here.
    sub_res = copy.deepcopy(S.load(a.a8s_result))
    sub_res['windows'] = list(reversed(sub_res['windows']))
    sub_path = fixture_dir / 'selftest-substituted-a8s-result.json'
    S.save(sub_path, sub_res)
    try:
        lock(argparse.Namespace(prereg=swap_guard_prereg,
                                out=fixture_dir / 'selftest-locked-sub.json',
                                a8s_result=str(sub_path), ceiling_result=str(ceil_dir / 'result.json')))
        raise SystemExit('lock computed P from a substituted A8S baseline')
    except ValueError as exc:
        out['substituted_a8s_refusal'] = 'refused: ' + str(exc)
    S.require('preregistered A8S baseline object' in out['substituted_a8s_refusal'],
              'substituted baseline refused for the wrong reason')
    S.save(a.out, out)
    print(json.dumps(out), flush=True)


def real_preregister(a):
    """preregister with the selection binding relaxed for the selftest fixture manifest."""
    import types
    a.selection = a.selection or _fixture_selection(a)
    preregister(a)


def _fixture_selection(a):
    m = S.load(a.manifest)
    return _tmp_selection(m['d44_g3']['roster_qnames'])


_FixtureSel = None


def _tmp_selection(qnames):
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    os.unlink(path)
    p = Path(path)
    S.save(p, {'schema': 'd44.captured_swap_selection.v1', 'qnames': sorted(qnames),
               'provisional': True, 'fixture': True})
    return p


# ------------------------------------------------------------------ CLI
def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('build-ceiling')
    s.add_argument('--selection', required=True, type=Path)
    s.add_argument('--capture', type=Path, default=S.CAPTURE)
    s.add_argument('--out', required=True, type=Path)
    s = sub.add_parser('preregister')
    s.add_argument('--prereg', dest='out', required=True, type=Path)
    s.add_argument('--selection', required=True, type=Path)
    s.add_argument('--manifest', required=True, type=Path)
    s.add_argument('--a8s-result', required=True, type=Path)
    s.add_argument('--weight-leg', required=True, type=Path)
    s = sub.add_parser('lock')
    s.add_argument('--prereg', required=True, type=Path)
    s.add_argument('--out', required=True, type=Path)
    s.add_argument('--a8s-result', required=True, type=Path)
    s.add_argument('--ceiling-result', required=True, type=Path)
    s.add_argument('--ceiling-receipt', type=Path)
    s = sub.add_parser('score')
    s.add_argument('--manifest', required=True, type=Path)
    s.add_argument('--root', required=True, type=Path)
    s.add_argument('--arm', default=CEIL_ARM)
    s.add_argument('--prereg', type=Path)
    s.add_argument('--dry-run', action='store_true')
    s = sub.add_parser('smoke')
    s.add_argument('--manifest', type=Path)
    s.add_argument('--out', type=Path)
    s = sub.add_parser('readset')
    s.add_argument('--manifest', required=True, type=Path)
    s.add_argument('--root', required=True, type=Path)
    s.add_argument('--out', required=True, type=Path)
    s = sub.add_parser('confirm')
    s.add_argument('--prereg', required=True, type=Path)
    s.add_argument('--swap-result', required=True, type=Path)
    s.add_argument('--ceiling-result', required=True, type=Path)
    s.add_argument('--a8s-result', required=True, type=Path)
    s.add_argument('--swap-receipt', required=True, type=Path)
    s.add_argument('--ceiling-receipt', required=True, type=Path)
    s.add_argument('--out', required=True, type=Path)
    s = sub.add_parser('selftest')
    s.add_argument('--a8s-result', required=True, type=Path)
    s.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    globals()[a.command.replace('-', '_')](a)


if __name__ == '__main__':
    main()
