#!/usr/bin/env python3
"""D44 frozen routed weight-leg gate, using the original CD2 decoders.

For unit u: E_u(c) = trace((W_c-W_source) XtX_heldout (W_c-W_source)^T).
XtX is full, unnormalized, captured in float32; contractions accumulate in float64.
The 512-row prefixes are validity witnesses only, never an error estimator.

Frozen-method points 5-7+9 (CEO freeze 2026-10-06T21:15:53Z):
- Bar scope: routed gate/up/down units of the 288 experts in each of the six
  decision layers 3,40,41,42,43,44 -- exactly 5184 units, fixed from the
  actual baseline manifest. Dense/shared units are excluded from the bar and
  reported as the separate existing six-unit subset line (closure 4.813849...
  percent), all six early captured sites included (not ranking-selected), with the explicit disclosure.
- The frozen roster is whatever the baseline manifest declares, captured or
  not. A missing unit is UNMEASURED -- never PASS, never dropped from the
  roster, never silently excluded from the gate.
- Primary metric C_gap = sum(A-N)/sum(A-X) over ALL routed units. The
  A8S-mass-weighted C is a secondary readout, not a gate. The obsolete
  non-ship ratio field is gone. Per-layer f_l=(A_l-N_l)/(A_l-X_l).
- Weight leg: exactly 2000 whole-expert resamples (gate/up/down joint),
  stratified by layer, seed 20261006; raw SSE resampled and C_gap/f_l
  recomputed. PASS iff C_gap>=0.5, global bootstrap p10>=0.35, at least
  4/6 positive f_l and no clearly adverse layer (f_l<0 and layer p90<0).
  Only-L3-clearly-adverse with C_gap_without_L3>=0.5 reports
  PASS-except-L3, routed to CEO, not an ordinary PASS.

closure-unit is the same CPU/GPU entry via encode_launch.py. A provisional
selection permits individually checked encoding and closure on CPU or GPU, but
never G3 assembly/scoring; no HELD scoring or scientific selection happens
here before the parent approves the complete frozen-method implementation.
G3 scoring remains the existing v2_launch.py/v2_score.py, unchanged numerically.
"""
from __future__ import annotations
import argparse
import hashlib
import math
import os
import random
from pathlib import Path
import stage1 as S
import d44_weight_leg as W

EX_MANIFEST = S.BASE/'surrogate-diag-20260929/g3/unit_manifest_v3c.json'
FROZEN_METHOD_SOURCE = '/home/rob/fleet/ceo/exec/eng-ldlq-indomain/d44-frozen-method-20261006.json'
FROZEN_METHOD_SHA256 = '5fff936525afeb721d32b14a99be9125f2b97f40a2f71220e53527fe1105abd8'
ROUTED_UNIT_TOTAL = len(S.LAYERS)*288*3
DENSE_SHARED_LINE = {'line':'historical fixed-byte six-unit subset (separate; NOT in the routed bar)',
                     'closure_fraction':0.04813849171193591,
                     'closure_percent':4.813849171193591,
                     'selection_basis':'none; all six early captured sites were included; neither the all24 FIT nor the all24 HELD ranking selected this subset',
                     'disclosure':'No all24 held-out ranking was run for the six-unit subset; all six early captured sites were included',
                     'dense_shared_units_in_routed_bar':0}
EX_ROOT = Path('/mnt/shared/models/GLM-5.3-Flash-EXL3-TR3-4bpw')
PRIOR = S.BASE/'codec-decomp-20260930/analysis/codec_rd/cdiag_analysis.py'


def stem(q):
    return q.replace('.', '__')


def roster_sha256(rows):
    h = hashlib.sha256()
    for q in sorted(r['qname'] for r in rows):
        h.update(q.encode()); h.update(b'\n')
    return h.hexdigest()


def roster(captures, manifest):
    """Frozen routed roster (method point 7): every routed gate/up/down unit of
    the 288 experts in each decision layer, fixed from the actual baseline
    manifest. What was captured never changes the roster; a unit absent from
    the capture stays in the roster and is reported unmeasured."""
    rows = [r for r in manifest['rows'] if r['kind']=='routed' and int(r['layer']) in S.LAYERS]
    S.require(len(rows)==ROUTED_UNIT_TOTAL, f'Frozen routed roster must be exactly {ROUTED_UNIT_TOTAL} units, manifest declares {len(rows)}')
    by = {}
    for r in rows:
        S.require(r['qname'] not in by, 'Duplicate routed unit in baseline manifest')
        S.require(r['pick']['a8s']==S.FMT, r['qname']+': fixed R1024 recipe required')
        by[r['qname']] = r
    if captures is not None:
        routed_captured = {q for q in captures.units if by.get(q,{}).get('kind')=='routed'}
        S.require(routed_captured <= set(by), 'Captured routed unit outside frozen routed roster')
    return rows


def coverage(rows, manifest):
    """Routed-only freeze coverage. Dense/shared units are excluded from the
    bar (method point 5) and never appear here."""
    result = []
    expected = [r for r in manifest['rows'] if r['kind']=='routed' and int(r['layer']) in S.LAYERS]
    S.require(len(expected)==ROUTED_UNIT_TOTAL, 'Coverage baseline must be the full frozen routed roster')
    picked = {r['qname'] for r in rows}
    for layer in sorted(S.LAYERS):
        want = [r for r in expected if int(r['layer'])==layer]
        got = [r for r in want if r['qname'] in picked]
        experts = {r['qname'].rsplit('.',1)[0] for r in want}
        got_experts = {r['qname'].rsplit('.',1)[0] for r in got}
        roles = {role: sum(1 for r in got if r['qname'].endswith('.'+role)) for role in ('gate_proj','up_proj','down_proj')}
        missing = sorted(r['qname'] for r in want if r['qname'] not in picked)
        result.append({'layer':layer,'covered_units':len(got),'routed_units_in_layer':len(want),
                       'full':len(got)==len(want),
                       'status':'full routed layer' if len(got)==len(want) else 'partial routed layer; missing units unmeasured, never excluded',
                       'experts_total':len(experts),'experts_with_all_three_roles_measured':len(got_experts),
                       'roles':roles,'missing_qnames':missing})
    return result


def selection_rows(path, capture, manifest, *, assembling=False):
    sel = S.load(path)
    cap = S.Captures(capture)
    S.require(sel['split_sha256']==cap.split_sha256, 'Selection and actual split differ')
    names = sel['qnames']
    S.require(len(names)==len(set(names)), 'Duplicated frozen unit')
    by = {r['qname']:r for r in manifest['rows']}
    S.require(all(q in by and q in cap.units for q in names), 'Frozen actual input missing')
    # Later captures do not silently widen an already frozen arm. Every unit
    # captured at freeze time is present in verified_inputs; live role bytes
    # must still agree for the candidate and evaluation to be comparable.
    if not sel["provisional"]:
        proofs = {p["qname"]:p for p in sel["verified_inputs"]}
        S.require(set(proofs)==set(names), "Frozen capture verification coverage incomplete")
        for q in names:
            for role in ("fit", "heldout"):
                actual = cap.units[q][1][role]
                S.require(proofs[q][role]["sha256"]==actual["sha256"] and proofs[q][role]["count"]==actual["count"], q+": frozen/live role bytes or counts differ")
    if assembling:
        S.require(not sel['provisional'], 'A provisional CPU selection cannot become a G3 arm')
    return [by[q] for q in names]


def freeze(a):
    S.setup()
    m, cached, _, _ = S.population()
    cap = S.Captures(a.capture)
    rows = roster(cap,m)
    captured = set(cap.units)
    proofs = []
    for r in rows:
        q = r['qname']
        if q not in captured:
            continue
        width = cached[q]['identity']['source']['shape'][1]
        roles = {}
        for role in ('fit','heldout'):
            _, roles[role] = cap.get(q,role,width)
        proofs.append({'qname':q,**roles})
    S.save(a.out,{'schema':'d44.frozen_routed_roster.v1','qnames':[r['qname'] for r in rows],
           'provisional':a.provisional,'frozen_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
           'split_sha256':cap.split_sha256,'capture_manifest_files':[{'path':str(p),'sha256':S.sha(p)} for p in sorted(Path(a.capture).glob('layers/L*/manifest.json'))],
           'verified_inputs':proofs,'coverage':coverage(rows,m),
           'roster':{'experts_per_layer':288,'roles':['gate','up','down'],'layers':list(S.LAYERS),
                     'routed_unit_total':ROUTED_UNIT_TOTAL,'roster_sha256':roster_sha256(rows),
                     'source':'actual baseline manifest; captured-or-not never changes the roster; missing units unmeasured, never dropped'},
           'dense_shared':dict(DENSE_SHARED_LINE),
           'g3_swap_selection':{'separate_fields_from_routed_bar':True,
                                'may_include_units_outside_routed_bar':True,
                                'selection_basis':'FIT ranking; the FIT-ranking winner outside the routed bar may be selected',
                                'note':'freeze only records the selection contract; no G3 arm or scoring is implemented here'},
           'method':{'source':FROZEN_METHOD_SOURCE,'sha256':FROZEN_METHOD_SHA256,'frozen_at':'2026-10-06T21:15:53.606923+00:00'},
           'baseline_values':{'A8S':0.01319,'EXL3':0.01069,'paired_noise_about':0.0003},
           'identity_policy':'[DEV-MODE] recorded identity drift stamps; byte/shape/count correctness refuses'})


def source_group_rows(row, exby):
    if exby[row['qname']]['pick']['exl3']=='SOURCE':
        return [row]
    group = row['qname'].rsplit('.',1)[0]
    return [exby[group+'.'+role] for role in ('gate_proj','up_proj','down_proj')]


def reference(row, w, cached, exby, device):
    torch, CD2, _, _ = S.setup()
    er = exby[row['qname']]
    if er['pick']['exl3']=='SOURCE':
        return w, {'reference':'actual EXL3 served arm retains SOURCE for this unit','wire_bytes':0,'alignment':None}
    S.require(er['pick']['exl3']=='EXL3', 'Unsupported actual EXL3 reference pick')
    from cd2_alignment import recover_perm, aligned
    group = source_group_rows(row,exby)
    src, dec, sizes = {}, {}, {}
    for r in group:
        q, role = r['qname'], r['role']
        S.require(r['layer']==row['layer'], 'Cross-layer reference substitution')
        src[role] = w if q==row['qname'] else S.source(r,cached,device)[0]
        ent = r['formats']['EXL3']
        dec[role], sizes[role] = CD2.decode_served('EXL3',ent,{'exl3':str(EX_ROOT)},device,q)
        S.require(list(dec[role].shape)==ent['rendered_shape'] and torch.isfinite(dec[role]).all().item(), q+': EXL3 actual geometry/nonfinite')
    perm, diag = recover_perm(dec['gate_proj'],src['gate_proj'],dec['up_proj'],src['up_proj'])
    S.require(perm is not None, row['qname']+': source-basis EXL3 permutation unavailable '+str(diag))
    g,u,d = aligned(perm,dec['gate_proj'],dec['up_proj'],dec['down_proj'])
    tensor = dict(gate_proj=g,up_proj=u,down_proj=d)[row['role']]
    return tensor, {'reference':'same-layer actual EXL3 bytes, CD2 source-basis alignment',
                    'alignment':diag,'permutation_sha256':S.blob_sha(perm.cpu().numpy().tobytes()),'wire_bytes':sizes[row['role']],
                    'group_qnames':[r['qname'] for r in group]}


def replacement(row, cap, cached, root, device):
    torch, _, L, _ = S.setup()
    q = row['qname']
    recpath = Path(root)/'receipts'/(stem(q)+'.json')
    if not recpath.exists():
        return None, {'status':'unmeasured; actual replacement bytes not yet produced','receipt':str(recpath)}
    rec = S.load(recpath)
    path = Path(rec['blob_path'])
    S.require(rec['qname']==q and not rec['dry_run'], q+': real same-unit encoded receipt required')
    blob = path.read_bytes()
    S.require(len(blob)==rec['blob_bytes']==cached[q]['blob_bytes'] and S.blob_sha(blob)==rec['blob_sha256'],q+': fixed-byte candidate integrity')
    S.wire_facts(blob,cached[q])
    for role in ('fit','heldout'):
        e = cap.units[q][1][role]
        p = rec[role]
        S.require(p['role']==role and p['split_sha256']==cap.split_sha256 and p['sha256']==e['sha256'] and p['count']==e['count'],q+': actual split/candidate role mismatch')
    from tessera.unit_artifact import read_unit_artifact
    t = read_unit_artifact(blob,device=device).to(torch.bfloat16)
    S.require(list(t.shape)==rec['rendered_shape'] and L.tensor_sha256(t)==rec['rendered_sha256'],q+': own candidate decode integrity')
    return t, {'status':'measured actual replacement','receipt':str(recpath),'receipt_sha256':S.sha(recpath),'blob':str(path),'blob_sha256':rec['blob_sha256'],'bytes':len(blob)}


def output_sse(source, decoded, hessian, *, block=32):
    import torch
    S.require(source.shape==decoded.shape and source.ndim==2 and hessian.shape==(source.shape[1],source.shape[1]), 'Weight-leg geometry mismatch')
    h = hessian.to(device=source.device,dtype=torch.float64)
    sums = []
    for i in range(0,source.shape[0],block):
        delta = decoded[i:i+block].double()-source[i:i+block].double()
        sums.append(((delta @ h)*delta).sum().item())
    value = math.fsum(sums)
    S.require(math.isfinite(value) and value>=0, 'Nonfinite or negative measured output SSE')
    return value


def closure_unit(a):
    torch, _, _, _ = S.setup()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m,cached,_,_ = S.population()
    exby = {r['qname']:r for r in S.load(a.ex_manifest)['rows']}
    cap = S.Captures(a.capture)
    by = {r['qname']:r for r in roster(cap,m)}
    if a.selection:
        allowed = {r['qname'] for r in selection_rows(a.selection,a.capture,m)}
        S.require(set(a.qname)<=allowed,'Unit outside frozen roster')
    for q in a.qname:
        r = by[q]
        width = cached[q]['identity']['source']['shape'][1]
        fit, fp = cap.get(q,'fit',width)
        # FIT is a provenance dependency only; never used in the error expression.
        del fit
        held, hp = cap.get(q,'heldout',width)
        w, stamp = S.source(r,cached,a.device)
        old = S.served(r,a.device)
        ex, ref = reference(r,w,cached,exby,a.device)
        new, np = replacement(r,cap,cached,a.root,a.device)
        values = {'A8S':None,'new':None,'EXL3':None}
        if not a.dry_run:
            H = held['hessian']
            for name,t in (('A8S',old),('new',new),('EXL3',ex)):
                if t is not None:
                    values[name] = output_sse(w,t,H)
        report = {'schema':'d44.heldout_unit_error.v1','qname':q,'layer':r['layer'],'kind':r['kind'],
                  'dry_run':a.dry_run,'output_sse':values,'heldout_count':held['count'],
                  'mean_output_sse_per_actual_routed_row':{k:v/held['count'] if v is not None else None for k,v in values.items()},
                  'metric':'full unnormalized HELDOUT XtX, no FIT or prefix estimator; unweighted linear-output weight leg, not nonlinear MoE output or routing-weighted energy',
                  'fit':fp,'heldout':hp,'reference':ref,'replacement':np,'source':stamp,
                  'actual_source_shape':list(w.shape),'prefix_rows_not_used_for_error':held['inputs'].shape[0],
                  'ex_manifest_sha256':S.sha(a.ex_manifest),'identities':S.identities(),
                  'action_key':os.environ.get('PRISMABUILD_ACTION_KEY')}
        path = Path(a.reports)/(stem(q)+'.json')
        S.save(path,report)
        print(__import__('json').dumps({'qname':q,'output_sse':values,'replacement_status':np['status'],'report':str(path)}),flush=True)
        del held,w,old,ex,new


def aggregate(reports, rows, manifest):
    """Frozen routed weight-leg aggregation (method points 5-7+9).

    Schema d44.frozen_weight_leg.v1. Primary C_gap=sum(A-N)/sum(A-X) over ALL
    routed units; per-layer f_l=(A_l-N_l)/(A_l-X_l); A8S-mass-weighted C is a
    secondary readout, not a gate. Reports outside the roster are refused;
    roster units without reports stay unmeasured and force UNMEASURED -- they
    are never excluded. The obsolete non-ship ratio field is gone.
    """
    by = {r['qname']:r for r in reports}
    S.require(len(by)==len(reports),'Duplicate unit report')
    outside = sorted(set(by)-{r['qname'] for r in rows})
    S.require(not outside, 'Report outside frozen routed roster: %s'%outside)
    names = ('A8S','new','EXL3')
    layers = sorted({int(r['layer']) for r in rows})
    missing_units = []
    per_layer, layer_table = {}, []
    for layer in layers:
        lrows = [r for r in rows if int(r['layer'])==layer]
        experts = {}
        layer_missing = []
        for r in sorted(lrows,key=lambda r:r['qname']):
            rep = by.get(r['qname'])
            key = r['qname'].rsplit('.',1)[0]
            slot = experts.setdefault(key,{'A':[],'N':[],'X':[],'ok':True})
            if rep is None or any(rep['output_sse'][n] is None for n in names):
                slot['ok'] = False
                layer_missing.append(r['qname'])
                continue
            slot['A'].append(rep['output_sse']['A8S']); slot['N'].append(rep['output_sse']['new']); slot['X'].append(rep['output_sse']['EXL3'])
        triples = []
        for key in sorted(experts):
            s = experts[key]
            if not s['ok']:
                continue
            triples.append((math.fsum(s['A']),math.fsum(s['N']),math.fsum(s['X'])))
        missing_units += layer_missing
        tot = tuple(math.fsum(t[i] for t in triples) if triples else None for i in range(3))
        f, f_reason = W.ratio(*tot)
        per_layer[layer] = {'triples':triples,'missing_units':layer_missing}
        layer_table.append({'layer':layer,'units_in_layer':len(lrows),
                            'experts_in_layer':len({r['qname'].rsplit('.',1)[0] for r in lrows}),
                            'measured_experts':len(triples),
                            'missing_units':layer_missing,
                            'raw_sse':{'A8S':tot[0],'new':tot[1],'EXL3':tot[2]},
                            'f_l':f,'f_l_reason':f_reason})
    gate = W.weight_leg(per_layer)
    boot = gate['bootstrap']
    measured = [r for r in rows if r['qname'] in by]
    A_sum = math.fsum(by[r['qname']]['output_sse']['A8S'] for r in measured if by[r['qname']]['output_sse']['A8S'] is not None)
    total_mass = math.fsum(t['raw_sse']['A8S'] for t in layer_table if t['raw_sse']['A8S'] is not None)
    secondary = math.fsum(t['raw_sse']['A8S']/total_mass*t['f_l'] for t in layer_table
                          if t['raw_sse']['A8S'] is not None and t['f_l'] is not None) if total_mass>0 and gate['status']!='UNMEASURED' and all(t['f_l'] is not None for t in layer_table) else None
    result = {'schema':'d44.frozen_weight_leg.v1','status':gate['status'],'reasons':gate['reasons'],
              'C_gap':gate['C_gap'],'C_gap_reason':gate['C_gap_reason'],
              'C_gap_formula':'sum(A8S-new)/sum(A8S-EXL3) over ALL routed units; primary gate',
              'C_gap_without_L3':{'value':gate['C_gap_without_exception_layer'],'reason':gate['C_gap_without_exception_layer_reason'],
                                  'use':'PASS-except-L3 exception readout only'},
              'raw_totals':gate['raw_totals'] if gate['raw_totals'] else {'A8S':None,'new':None,'EXL3':None},
              'measured_units':len(measured),'missing_unit_count':len(missing_units),'missing_units':sorted(missing_units),
              'missing_unit_policy':'unmeasured, never PASS, never excluded from the roster or the gate',
              'layers':layer_table,'layer_f':gate['layer_f'],
              'clearly_adverse_layers':gate['clearly_adverse_layers'],
              'positive_layers':gate['positive_layers'],
              'bootstrap':{'resamples':boot['resamples'],'seed':boot['seed'],'stratified_by':boot['stratified_by'],
                           'unit':boot['unit'],'quantile':boot['quantile'],
                           'c_gap_defined_draws':boot['c_gap_defined_draws'],
                           'c_gap_undefined_draws':boot['c_gap_undefined_draws'],
                           'c_gap_undefined_reasons':boot['c_gap_undefined_reasons'],
                           'layer_f_undefined_draws':boot['layer_f_undefined_draws'],
                           'C_gap_p10':boot['C_gap_p10'],'C_gap_p50':boot['C_gap_p50'],'C_gap_p90':boot['C_gap_p90'],
                           'layer_f_p90':boot['layer_f_p90']} if boot else None,
              'secondary':{'error_mass_weighted_closure':secondary,
                           'note':'A8S-mass-weighted C; secondary readout, NOT a gate',
                           'formula':'sum_l (A_l/sum A) * f_l'},
              'roster':{'routed_unit_total':ROUTED_UNIT_TOTAL,'layers':list(S.LAYERS),'experts_per_layer':288,
                        'roles':['gate','up','down'],'roster_sha256':roster_sha256(rows),
                        'dense_shared':'excluded from the routed bar; separate HISTORICAL six-unit subset line (all six early captured sites included; neither all24 FIT nor all24 HELD ranking selected it), closure 4.813849171193591 percent'},
              'method':{'source':FROZEN_METHOD_SOURCE,'sha256':FROZEN_METHOD_SHA256},
              'g3_interface':{'weight_leg_status':gate['status'],'C_gap':gate['C_gap'],
                              'method_sha256':FROZEN_METHOD_SHA256,'roster_sha256':roster_sha256(rows)},
              'g3_confirmation':'unmeasured; unchanged campaign G3v2 arm required; flat or worse stops diagnosis',
              'baseline_values':{'A8S':0.01319,'EXL3':0.01069,'paired_noise_about':0.0003}}
    return result


def closure(a):
    S.setup()
    m,_,_,_ = S.population()
    cap = S.Captures(a.capture)
    rows = roster(cap,m)
    allowed = {r['qname'] for r in selection_rows(a.selection,a.capture,m)}
    report_paths = {r['qname']:Path(a.reports)/(stem(r['qname'])+'.json') for r in rows}
    rs = []
    for r in rows:
        path = report_paths[r['qname']]
        if not path.exists():
            continue
        rep = S.load(path)
        S.require(rep['qname'] in allowed, rep['qname']+': report outside the fixed frozen selection')
        S.require(rep['heldout']['split_sha256']==cap.split_sha256 and rep['heldout']['role']=='heldout','Mixed held-out scope')
        S.require(rep['heldout']['sha256']==cap.units[rep['qname']][1]['heldout']['sha256'],'Unit report refers to different actual held-out bytes')
        rs.append(rep)
    result = aggregate(rs,rows,m)
    result.update(selection_sha256=S.sha(a.selection),
                  unit_report_sha256={q:S.sha(p) for q,p in sorted(report_paths.items()) if p.exists()},
                  historical_comparison={'path':str(PRIOR),'sha256':S.sha(PRIOR),'reuse':'CD2 per-layer raw error sums and A8S error shares; new split metric is linear XtX rather than CD1 nonlinear routed-output energy'})
    S.save(a.out,result)
    print(__import__('json').dumps({'C_gap':result['C_gap'],'status':result['status'],
                                    'measured_units':result['measured_units'],'missing_unit_count':result['missing_unit_count']}),flush=True)


def data_readset(a):
    # Metadata-only construction: PB materializes/hash-checks entries lacking a digest.
    S.setup()
    m,cached,_,_ = S.population()
    cap = S.Captures(a.capture)
    by = {r['qname']:r for r in roster(cap,m)}
    exby = {r['qname']:r for r in S.load(a.ex_manifest)['rows']}
    wm = S.load(S.MODEL/'model.safetensors.index.json')['weight_map']
    entries = {}
    def add(path,offset=0,size=None,digest=None):
        path = Path(path)
        n = path.stat().st_size-offset if size is None else size
        entries[(str(path),offset,n)] = {'path':str(path),'offset':offset,'bytes':n,'sha256':digest}
    for p in (S.CACHE,S.MANIFEST,a.ex_manifest,Path(a.capture)/'split-manifest.json',cap.split['source_draw']['path'],S.MODEL/'model.safetensors.index.json',PRIOR):
        add(p)
    for p in sorted(Path(a.capture).glob('layers/L*/manifest.json')):
        add(p)
    for p in Path(cap.split['source_draw']['path']).parent.iterdir():
        if p.is_file():
            add(p)
    for p in S.OWNERS.glob('*.py'):
        add(p)
    for q in a.qname:
        r = by[q]
        for role in ('fit','heldout'):
            e = cap.units[q][1][role]
            add(Path(a.capture)/e['file'],size=e['bytes'],digest=e['sha256'])
        for sr in source_group_rows(r,exby):
            # SourceFDs can digest a whole consumed shard. Declare the WHOLE file,
            # not only the selected tensor interval. No read-budget understatement.
            add(S.MODEL/wm[sr['qname']+'.weight'])
            if exby[sr['qname']]['pick']['exl3']=='EXL3':
                ent = sr['formats']['EXL3']['wire']
                for sh,off,n in ent['ranges']:
                    add(EX_ROOT/sh,off,n)
        loc = r['formats'][S.FMT]['wire']
        off,n = S.setup()[2].member_location(loc)
        add(S.A8/loc['shard'],off,n,cached[q]['blob_sha256'])
        rec = Path(a.root)/'receipts'/(stem(q)+'.json')
        if rec.exists():
            add(rec)
            rnew = S.load(rec)
            add(rnew['blob_path'],size=rnew['blob_bytes'],digest=rnew['blob_sha256'])
    if a.selection:
        add(a.selection)
    data = list(entries.values())
    S.save(a.out,{'schema':'prismaquant.prismabuild.data_manifest.v1','mount_prefix':'/mnt/shared','entries':data,
                 'entry_count':len(data),'total_bytes':sum(e['bytes'] for e in data),
                 'produced_by':{'tool':'d44.data-readset; metadata only; whole source shards'},
                 'annotations':{'phases':[{'name':'actual-unit-inputs','bytes':sum(e['bytes'] for e in data),'cumulative_bytes':sum(e['bytes'] for e in data)}]}})


def fixtures(a):
    torch,CD2,_,_ = S.setup()
    from cd2_alignment import recover_perm,aligned
    X = torch.arange(48,dtype=torch.float64).reshape(12,4)/10
    WGT = torch.arange(20,dtype=torch.float64).reshape(5,4)/7
    old,new,ex = WGT+0.5,WGT+0.2,WGT+0.1
    H = (X.T@X).float()
    checks = {}
    for name,t in (('A8S',old),('new',new),('EXL3',ex)):
        actual = output_sse(WGT,t,H)
        direct = ((X@(t-WGT).T)**2).sum().item()
        S.require(math.isclose(actual,direct,rel_tol=1e-7), 'Full XtX/direct fixture mismatch')
        checks[name] = {'sse':actual,'direct_sse':direct}
    # Off-diagonal entries and rows beyond a two-row witness change the answer.
    S.require(not math.isclose(checks['A8S']['sse'],((X[:2]@(old-WGT).T)**2).sum().item()),'Prefix estimator slipped into metric')
    g = torch.eye(4)
    u = torch.eye(4)*2
    d = torch.arange(12,dtype=torch.float32).reshape(3,4)
    perm = torch.tensor([2,0,3,1])
    got,diag = recover_perm(g[perm],g,u[perm],u)
    S.require(torch.equal(got,perm) and all(torch.equal(x,y) for x,y in zip(aligned(got,g[perm],u[perm],d[:,perm]),(g,u,d))),'Original CD2 alignment fixture mismatch')
    rows = [{'qname':'u0','layer':3,'kind':'routed','pick':{'a8s':S.FMT}},{'qname':'u1','layer':40,'kind':'routed','pick':{'a8s':S.FMT}}]
    manifest = {'rows':rows+[{'qname':'u2','layer':40,'kind':'routed','pick':{'a8s':S.FMT}}]}
    def reports(vals):
        return [{'qname':r['qname'],'heldout_count':12,'output_sse':dict(zip(('A8S','new','EXL3'),v))} for r,v in zip(rows,vals)]
    measured = aggregate(reports([(10.,7.,4.),(30.,22.,10.)]),rows,manifest)
    S.require(math.isclose(measured['C_gap'],(10-7+30-22)/(10-4+30-10)),'Primary C_gap mismatch')
    S.require(math.isclose(measured['secondary']['error_mass_weighted_closure'],0.425),'Layer error mass weighting fixture mismatch')
    S.require(not any('not_ship_metric' in k for k in measured),'Obsolete non-ship ratio field still present')
    zero = aggregate(reports([(10.,7.,10.),(30.,22.,10.)]),rows,manifest)
    negative = aggregate(reports([(10.,7.,12.),(30.,22.,10.)]),rows,manifest)
    missing = aggregate(reports([(10.,None,4.),(30.,22.,10.)]),rows,manifest)
    partial = aggregate(reports([(10.,7.,4.)]),rows,manifest)
    S.require(all(r['secondary']['error_mass_weighted_closure'] is None for r in (zero,negative,missing,partial)),'Undefined/unmeasured fraction fabricated')
    S.require(zero['status']=='FAIL' and zero['layers'][0]['f_l'] is None and 'denominator' in zero['layers'][0]['f_l_reason'],'Zero gap denominator not an explicit FAIL')
    S.require(negative['status']=='FAIL' and negative['layers'][0]['f_l'] is None,'Negative gap denominator not an explicit FAIL')
    S.require(missing['status']=='UNMEASURED' and missing['missing_units']==['u0'],'Null unit SSE not explicitly unmeasured')
    S.require(partial['status']=='UNMEASURED' and set(partial['missing_units'])=={'u1'} and measured['status']=='FAIL','Missing roster unit not explicitly unmeasured')
    S.require(partial['C_gap_without_L3']['value'] is None and 'reason' in partial['C_gap_without_L3'] and partial['g3_interface']['weight_leg_status']=='UNMEASURED','UNMEASURED result must still carry the complete frozen schema incl. exception readout and G3 interface')
    _required = {'schema','status','reasons','C_gap','C_gap_reason','C_gap_without_L3','raw_totals','measured_units','missing_unit_count','missing_units','missing_unit_policy','layers','layer_f','clearly_adverse_layers','positive_layers','bootstrap','secondary','roster','method','g3_interface','g3_confirmation','baseline_values'}
    for r in (measured,zero,negative,missing,partial):
        S.require(_required <= set(r),'Frozen weight-leg schema incomplete: %s'%sorted(_required-set(r)))
    # Frozen weight-leg gate boundaries on six synthetic layers, 288 whole experts.
    def six(good,adverse=None,l3=None,zero=(),count=W.EXPERTS_PER_LAYER):
        out = {}
        for l in W.LAYERS:
            if adverse and l in adverse: out[l] = {'triples':[(1.,2.,0.5)]*count,'missing_units':[]}
            elif l3 is not None and l==3: out[l] = {'triples':[l3]*count,'missing_units':[]}
            elif l in zero: out[l] = {'triples':[(2.,2.,0.)]*count,'missing_units':[]}
            else: out[l] = {'triples':[good]*count,'missing_units':[]}
        return out
    def statuses(**kw):
        return W.weight_leg(six(**kw))['status']
    # PASS boundary: C_gap exactly 0.5, p10=0.5, all six positive.
    S.require(statuses(good=(2.,1.,0.))=='PASS','C_gap exactly 0.5 must PASS (boundary inclusive)')
    S.require(statuses(good=(2.,1.0002,0.))=='FAIL','C_gap just below 0.5 must FAIL')
    # p10 boundary: identical experts make every resample identical.
    p10case = W.weight_leg(six(good=(2.,1.3,0.)))
    S.require(math.isclose(p10case['bootstrap']['C_gap_p10'],0.35,rel_tol=1e-12) and p10case['status']=='FAIL','p10 exactly 0.35 with C_gap<0.5 must FAIL on C_gap')
    # Exactly four positive-f layers passes; three fails.
    S.require(statuses(good=(10.,1.,0.),l3=(2.,2.,0.),zero=(40,))=='PASS','Exactly 4 positive-f layers must PASS')
    S.require(statuses(good=(10.,1.,0.),l3=(2.,2.,0.),zero=(40,41))=='FAIL','Only 3 positive-f layers must FAIL')
    # Clearly adverse L3 only: f_l<0 and layer bootstrap p90<0.
    exc = W.weight_leg(six(good=(10.,1.,0.),l3=(1.,2.,0.5)))
    S.require(exc['status']=='PASS-except-L3' and exc['clearly_adverse_layers']==[3] and math.isclose(exc['C_gap_without_exception_layer'],(50-5)/(50-0)),'Only-L3 adverse must be PASS-except-L3')
    S.require(W.weight_leg(six(good=(10.,1.,0.),adverse=(3,40)))['status']=='FAIL','Two adverse layers must FAIL, not PASS-except-L3')
    S.require(W.weight_leg(six(good=(10.,1.,0.),l3=(1.,2.,0.5)),exception_c_gap_min=0.95)['status']=='FAIL','PASS-except-L3 requires C_gap_without_L3>=0.5')
    # Missing layer forces UNMEASURED even with everything else perfect.
    msix = six(good=(2.,1.,0.)); msix[40]['missing_units'] = ['layer.40.mlp.experts.7.down_proj']
    S.require(W.weight_leg(msix)['status']=='UNMEASURED','Missing unit must force UNMEASURED, never PASS')
    # Bootstrap: 2000 resamples, exact seed, stratified whole-expert raw-sum recomputation.
    b = W.weight_leg(six(good=(2.,1.,0.)))['bootstrap']
    S.require(b['resamples']==2000 and b['seed']==20261006 and len(b['c_gap_samples'])==2000,'Bootstrap metadata mismatch')
    flip = six(good=(2.,1.,0.)); flip[44]['triples'] = [(2.,1.,9.)]*W.EXPERTS_PER_LAYER
    rng = random.Random(20261006)
    RA,RN,RX = [],[],[]
    for l in W.LAYERS:
        ra,rn,rx = [],[],[]
        for _ in range(W.EXPERTS_PER_LAYER):
            ea,en,ex = flip[l]['triples'][rng.randrange(W.EXPERTS_PER_LAYER)]
            ra.append(ea); rn.append(en); rx.append(ex)
        RA.append(math.fsum(ra)); RN.append(math.fsum(rn)); RX.append(math.fsum(rx))
    bf = W.weight_leg(flip)['bootstrap']
    S.require(math.isclose(bf['c_gap_samples'][0],(math.fsum(RA)-math.fsum(RN))/(math.fsum(RA)-math.fsum(RX))),'First bootstrap resample not the documented whole-expert stratified raw-sum recomputation')
    S.require(b['c_gap_samples']==W.weight_leg(six(good=(2.,1.,0.)))['bootstrap']['c_gap_samples'],'Bootstrap not deterministic under fixed seed')
    S.require(W.weight_leg(flip)['bootstrap']['c_gap_samples']!=b['c_gap_samples'],'Role-unit change did not move the joint whole-expert resample')
    # Failing-before regression: draws with negative resampled denominators keep
    # their SIGNED ratios and block the gate; no draw is ever silently dropped.
    mix = {l:{'triples':[(10.,9.4,9.)]*(W.EXPERTS_PER_LAYER-1)+[(10.,10.,260.)],'missing_units':[]} for l in W.LAYERS}
    mb = W.weight_leg(mix)
    S.require(mb['bootstrap']['c_gap_defined_draws']+mb['bootstrap']['c_gap_undefined_draws']==2000,'Bootstrap must account for all 2000 draws')
    S.require(mb['bootstrap']['C_gap_p10']<0,'Negative-denominator bootstrap tail silently dropped')
    S.require(mb['status']=='FAIL','Gate qualified despite negative signed bootstrap tail')
    # Zero-denominator draws are counted explicitly, never discarded silently.
    zpool = [(1.,0.,1.),(1.,0.,0.)]
    zboot = W.bootstrap_experts({3:zpool})
    zrng = random.Random(20261006); zexpected = 0
    for _ in range(2000):
        picks = [zpool[zrng.randrange(2)] for _ in range(2)]
        if math.fsum(p[0] for p in picks)-math.fsum(p[2] for p in picks)==0: zexpected += 1
    S.require(zboot['c_gap_undefined_draws']==zexpected and zboot['c_gap_undefined_reasons'].get('zero gap denominator')==zexpected,'Undefined-denominator bootstrap draws not counted explicitly')
    S.require(zboot['c_gap_defined_draws']+zboot['c_gap_undefined_draws']==2000,'Draw accounting mismatch')
    # Real encoder seam on a cheap deterministic fixture, NOT replacement weights.
    _,cached,_,dense = S.population()
    q = 'd44.fixture'
    enc = CD2.Encoder(S.FMT)
    cw = torch.arange(16*128,dtype=torch.float32).reshape(16,128).sin().to(torch.bfloat16)/16
    ch = torch.eye(128,dtype=torch.float32)
    fixture_prov = {"fixture":"deterministic CPU only, fit role", "fit_tokens":128,
                    "text_sha256":S.blob_sha(b"D44 deterministic identity-row fixture"),
                    "fit_ids_sha256":S.blob_sha(torch.arange(128,dtype=torch.int32).numpy().tobytes())}
    kw = S.fixed_kwargs(enc,q,ch,fixture_prov, {q:cached[dense[0]['qname']]},'cpu')
    [(render,blob)] = enc.encode([cw],[kw],'NONE')
    S.wire_facts(blob,cached[dense[0]['qname']])
    S.require(render.shape==cw.shape and torch.isfinite(render).all().item(),'Actual R1024 fixture decode failed')
    S.save(a.out,{'schema':'d44.deterministic_fixture.v1','full_XtX_checks':checks,'alignment':diag,'weighted':measured,
                 'zero_gap':zero,'negative_gap':negative,'missing_replacement':missing,
                 'real_R1024_encoder_fixture':{'shape':list(cw.shape),'bytes':len(blob),'sha256':S.blob_sha(blob),'not_actual_replacement':True},
                 'action_key':os.environ.get('PRISMABUILD_ACTION_KEY')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    for name in ('freeze','closure-unit','closure','data-readset','fixtures'):
        s = sub.add_parser(name)
        s.add_argument('--capture',type=Path,default=S.CAPTURE)
        s.add_argument('--root',type=Path,default=S.OUTPUT)
        s.add_argument('--out',type=Path)
        s.add_argument('--selection',type=Path)
        s.add_argument('--ex-manifest',type=Path,default=EX_MANIFEST)
        s.add_argument('--reports',type=Path)
        s.add_argument('--qname',action='append')
        s.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        s.add_argument('--dry-run',action='store_true')
        s.add_argument('--provisional',action='store_true')
    a = p.parse_args()
    if a.command=='closure-unit':
        S.require(bool(a.qname) and a.reports is not None,'--qname and --reports required')
    else:
        S.require(a.out is not None,'--out required')
    if a.command=='closure':
        S.require(a.selection is not None and a.reports is not None,'Frozen selection and reports required')
    if a.command=='data-readset':
        S.require(bool(a.qname),'Exact qnames required for data read set')
    globals()[a.command.replace('-','_')](a)


if __name__=='__main__':
    main()
