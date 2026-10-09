#!/usr/bin/env python3
"""D42 fixed-recipe research arm. Execution belongs to admitted PB payloads.

Commands: metadata, rank, timings, plan, encode, assemble, score, readset, reduce.
D44 closure and frozen all-captured-unit selection live in d44.py.
No production encoder, dispatcher, rotation, allocator or serving pin is changed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from functools import lru_cache
import statistics
import sys
import time
STARTED = time.monotonic()

BASE = Path('/mnt/shared/tessera-measurements')
V2 = BASE / 'g3-v2-rebaseline-20261005'
OWNERS = V2 / 'source'
CENSUS = BASE / 'glm-canonical-census-20260908'
WIRE = CENSUS / 'activation-runtime-allocation-20260911/extension-r1024-02/workspace/merged/cache/wire'
CACHE = WIRE / 'cached_units.a8s-bf16menu-c3ceec52fc22.v1.json'
MANIFEST = BASE / 't8-prefix-probe-20261005/metadata/unit_manifest_prefix_probe.json'
A8 = Path('/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported')
MODEL = Path('/mnt/shared/models/GLM-5.3-Flash-BF16')
CAPTURE = BASE / 'eng-ldlq-indomain-20261006/research-capture-shared-01'
OUTPUT = BASE / 'eng-ldlq-indomain-20261006/stage1-fixed-01'
PYTHON = '/home/rob/venvs/pq-pbdc4803-tessera-b40c93cb/bin/python'
FMT = 'TESSERA_E4M3_K1_R1024'
PICK = 'indomain_stage1'
LAYERS = (3, 40, 41, 42, 43, 44)
HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def blob_sha(blob):
    return hashlib.sha256(blob).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


@lru_cache(maxsize=1)
def setup():
    os.environ.setdefault('PRISMAQUANT_DEV_MODE', '1')
    os.environ.setdefault('G3_PQ_ROOT', str(BASE / 'surrogate-diag-20260929/src/pq-7882eda3'))
    os.environ.setdefault('TESSERA_SRC', '/mnt/shared/tessera-pins/b40c93cb73745097e57a1ba4cf5b9eee166c759a/src')
    sys.path[:0] = [str(OWNERS), os.environ['G3_PQ_ROOT'], os.environ['G3_PQ_ROOT'] + '/tools', os.environ['TESSERA_SRC']]
    import torch
    torch.set_num_threads(1)
    import codec_factorial as CD2
    import g3_lib as L
    import v2_score as V
    return torch, CD2, L, V


@lru_cache(maxsize=1)
def identities():
    import tessera.export as tx
    import prismaquant.tessera_render as tr
    return {'harness': {'path': str(HERE / 'stage1.py'), 'sha256': sha(HERE / 'stage1.py')},
            'cd2': {'path': str(HERE / 'codec_factorial.py'), 'sha256': sha(HERE / 'codec_factorial.py')},
            'd44': {n: sha(HERE/n) for n in ('d44.py','cd2_alignment.py','source_file.py')},
            'numerical_owners': {n: sha(OWNERS / n) for n in ('v2_score.py', 'g3_offline_decoded_kl.py', 'g3_lib.py')},
            'encoder_files': {str(Path(m.__file__)): sha(m.__file__) for m in (tx, tr)},
            'cached_producer_manifest_sha256': sha(CACHE), 'base_manifest_sha256': sha(MANIFEST),
            'identity_policy': 'D32 stamps; not immutable-provider qualification', 'rotation': 'NONE'}


@lru_cache(maxsize=1)
def population():
    m, cached = load(MANIFEST), load(CACHE)['units']
    require(not m['problems'], 'Base manifest has incomplete/corrupt units')
    counts = {i: sum(int(r['layer']) == i for r in m['rows']) for i in range(45)}
    require(counts == {**{i: 3 for i in range(3)}, **{i: 867 for i in range(3, 45)}}, 'Actual body roster must be L0..44 (no MTP45)')
    dense = [r for r in m['rows'] if r['kind'] != 'routed' and r['pick']['a8s'] != 'SOURCE']
    require(len(dense) == 24, f'Rank ALL 24 quantized dense/shared units; got {len(dense)}')
    routed = [r for r in m['rows'] if r['kind'] == 'routed' and int(r['layer']) in LAYERS]
    require(len(routed) == len(LAYERS) * 288 * 3, 'Incomplete routed decision-layer body')
    for r in dense + routed:
        q, ent = r['qname'], r['formats'][r['pick']['a8s']]
        require(r['pick']['a8s'] == FMT, f'{q}: fixed R1024 format required')
        require(q in cached, f'{q}: cached producer contract absent')
        c = cached[q]
        require(c['blob_bytes'] == ent['priced_wire_bytes'] == ent['wire']['member_bytes'], f'{q}: baseline wire lengths disagree')
        recipe = c['identity']['recipe']
        require((recipe['grid'], recipe['q256'], recipe['body'], recipe['plane']) == ('E4M3', 1024, 'window', 'channel'), f'{q}: recipe drift')
        require(c['identity']['source']['shape'] == ent['rendered_shape'], f'{q}: producer/render geometry differs')
    return m, cached, routed, dense


def wire_facts(blob, contract):
    from tessera.unit_artifact import parse_unit_metadata
    from tessera.manifest import RotationState
    p = parse_unit_metadata(blob, device='cpu')
    f, want = p.role_facts(), contract['identity']['recipe']
    require((f['grid'],f['q256'],f['body'].lower(),f['plane'].lower(),f['span']) == (want['grid'],want['q256'],want['body'],want['plane'],want['span']), 'Actual unit body/rung/plane recipe changed')
    require(p.rotation == RotationState.NONE and p.manifest.window_bits == want['window_bits'], 'Actual rotation/window recipe changed')
    reach = p.manifest.reach
    require((reach.window_seed if reach else 0) == want['seed'], 'Actual window seed changed')
    require((reach.window_sigma if reach else None) == want['sigma'] and (reach.channel_sigma if reach else None) == want['channel_sigma'], 'Actual scale spread schedule changed')
    return dict(f, encoder_profile_id=p.manifest.encoder_profile_id.hex(), encoder_fixture_id=p.manifest.encoder_fixture_id.hex())


def served(row, device='cpu'):
    torch, CD2, L, _ = setup()
    c = population()[1][row['qname']]
    ent = row['formats'][row['pick']['a8s']]
    blob = CD2.read_wire(ent, {'a8': str(A8), 'wirecache': str(WIRE)})
    require(len(blob) == c['blob_bytes'] and blob_sha(blob) == c['blob_sha256'], row['qname'] + ': own A8S body bytes corrupt')
    wire_facts(blob,c)
    from tessera.unit_artifact import read_unit_artifact
    t = read_unit_artifact(blob, device=device).to(torch.bfloat16)
    require(list(t.shape) == ent['rendered_shape'], row['qname'] + ': served decode shape')
    L.check_identity(t, ent['rendered_sha256'], row['qname'] + ' served decode own-byte identity')
    return t


def source(row, cached, device='cpu', small=False):
    torch, _, L, _ = setup()
    from safetensors import safe_open
    q = row['qname']
    wm = load(MODEL / 'model.safetensors.index.json')['weight_map']
    name = q + '.weight'
    if small:
        with safe_open(str(MODEL / wm[name]), framework='pt', device='cpu') as f:
            view = f.get_slice(name)
            require(list(view.get_shape()) == cached[q]['identity']['source']['shape'], q + ': actual source shape')
            t = view[:1, :32]
    else:
        from source_file import tensor
        t = tensor(MODEL/wm[name],name)
        require(list(t.shape) == cached[q]['identity']['source']['shape'], q + ': actual source geometry')
    require(t.dtype == torch.bfloat16 and torch.isfinite(t).all().item(), q + ': actual source dtype/nonfinite')
    if not small:
        from tessera.cached_unit import digest_host_tensor
        # Recorded-vs-running source is a stamp, never a new D32 identity wall.
        stamp = {'raw_sha256': L.tensor_sha256(t), 'producer_digest': digest_host_tensor(t),
                 'recorded_raw_sha256': row['source_sha256'], 'recorded_producer_digest': cached[q]['identity']['source']['sha256']}
    else:
        stamp = {'small_actual_shape': list(t.shape), 'source_shard': wm[name]}
    return t.to(device), stamp


class Captures:
    """Actual same-pass fixed-coordinate split. No merged-H fallback."""
    def __init__(self, root):
        self.root = Path(root)
        from g3_residency import read_file
        split_raw = read_file(self.root/'split-manifest.json')
        self.split = json.loads(split_raw)
        s = self.split
        require(s['seed'] == 20261001 and s['sample_count'] == 512 and s['tokens_per_sample'] == 512, 'Incorrect approved draw')
        require(s['same_forward_pass'] is True and s['roles']['fit']['sample_range'] == [0, 384]
                and s['roles']['heldout']['sample_range'] == [384, 512], 'Split is not fixed source-coordinate same-pass disjoint')
        require(s['roles']['fit']['token_ids_sha256'] != s['roles']['heldout']['token_ids_sha256'], 'Separate role token identities required')
        require(bool(s['source_draw']) and all(s['roles'][r]['provenance'] for r in ('fit', 'heldout')), 'Original draw/role provenance missing')
        from prismaquant.calibration_data import load_calibration_input
        import torch
        draw = s['source_draw']
        ids, receipt = load_calibration_input(draw['path'], expected_sha256=draw['sha256'], n_samples=512, seqlen=512)
        require(draw['shape'] == [512,512] and receipt['provenance'] == draw['provenance'], 'Actual original draw provenance differs')
        require(draw['provenance']['seed'] == 20261001, 'Actual saved draw seed')
        # Canonical body identity is distinct from the JSON file's own bytes.
        # Rebuild data-bearing identity fields from the actual parsed draw.
        derived_body = copy.deepcopy({k:v for k,v in s.items() if k != 'split_sha256'})
        derived_body.update(seed=draw['provenance']['seed'],sample_count=ids.shape[0],tokens_per_sample=ids.shape[1])
        derived_body['source_draw'].update(provenance=receipt['provenance'],shape=list(ids.shape))
        if 'dtype' in derived_body['source_draw']:
            derived_body['source_draw']['dtype'] = str(ids.dtype)
        for role, bounds in (('fit',(0,384)), ('heldout',(384,512))):
            e = s['roles'][role]
            token_sha = blob_sha(ids[bounds[0]:bounds[1]].to(torch.int32).numpy().tobytes())
            require(e['token_ids_sha256'] == e['provenance']['fit_ids_sha256'] == token_sha, 'Role token hashes are not actual fixed-coordinate IDs')
            require(e['provenance']['fit_tokens'] == (bounds[1]-bounds[0])*512 and e['provenance']['text_sha256'] == draw['provenance']['text_sha256'], 'Role tokens/distribution differ')
            require(e['provenance']['hessian_role'] == ('fit' if role == 'fit' else 'held-out'), 'Mixed-H role provenance')
            derived = derived_body['roles'][role]
            derived.update(sample_range=list(bounds),token_ids_sha256=token_sha)
            if 'token_count' in derived:
                derived['token_count'] = (bounds[1]-bounds[0])*ids.shape[1]
            derived['provenance'].update(fit_ids_sha256=token_sha,fit_tokens=(bounds[1]-bounds[0])*ids.shape[1],text_sha256=receipt['provenance']['text_sha256'])
        from prismaquant.cost_stage_checkpoint import canonical_json_sha256
        self.split_sha256 = canonical_json_sha256(derived_body,where='actual fixed-coordinate split body')
        require(s['split_sha256'] == self.split_sha256,'Split body identity differs from actual derived role data')
        self.split_manifest_file_sha256 = blob_sha(split_raw)
        self.units = {}
        for path in sorted(self.root.glob('layers/L*/manifest.json')):
            m = load(path)
            require(m['split_sha256'] == self.split_sha256, 'Layer captures name different actual split body')
            if 'split_manifest_file_sha256' in m:
                require(m['split_manifest_file_sha256'] == self.split_manifest_file_sha256,'Layer-bound split JSON own-byte integrity')
            for q, roles in m['units'].items():
                require(q not in self.units, 'Duplicated captured unit')
                require(roles['fit']['count'] + roles['heldout']['count'] == m['full_counts'][q], q + ': split drops or duplicates original census rows')
                self.units[q] = (path.parent, roles)

    def get(self, q, role, width):
        import torch
        require(q in self.units, q + ': capture dependency missing')
        parent, roles = self.units[q]
        require(set(roles) == {'fit', 'heldout'}, q + ': actual role pair required')
        e = roles[role]
        path = self.root / e['file']
        from g3_residency import read_file
        import io
        raw = read_file(path)
        require(len(raw) == e['bytes'] and blob_sha(raw) == e['sha256'], q + ': capture own-file corruption')
        d = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        H, X, ids = d['hessian'], d['inputs'], d['prefix_sample_ids']
        require(X is not None and ids.dtype == torch.int64 and ids.ndim == 1, q + ': actual bounded prefix and int64 sample coordinates required')
        require(list(ids.shape) == e['prefix_sample_ids_shape'], q + ': prefix coordinate count differs')
        require(d['name'] == q and d['role'] == role and d['count'] == e['count'] and d['count'] > 0, q + ': actual routed count/role')
        require(H.dtype == X.dtype == torch.float32 and H.shape == (width, width) and X.ndim == 2 and X.shape[1] == width, q + ': capture dtype/shape')
        require(list(H.shape) == e['hessian_shape'] and list(X.shape) == e['inputs_shape'], q + ': manifest shapes')
        require(X.shape[0] == min(d['count'],512) and len(ids) == X.shape[0], q + ': actual bounded-prefix/routed counts')
        lo, hi = self.split['roles'][role]['sample_range']
        require(d['count'] <= (hi-lo)*512, q + ': routed count exceeds actual source coordinates')
        if '.experts.' not in q:
            require(d['count'] == (hi-lo)*512, q + ': dense/shared role must cover every source token')
        require(all(lo <= int(i) < hi for i in ids), q + ': prefix contains opposite-role source coordinates')
        require(torch.isfinite(H).all().item() and torch.isfinite(X).all().item() and math.isfinite(float(d['max_abs'])), q + ': nonfinite capture')
        require(torch.equal(H, H.T), q + ': H must be full unnormalized symmetric XtX')
        require(d['max_abs'] >= (X.abs().max().item() if X.numel() else 0), q + ': actual maximum bound')
        return d, {'path': str(path), 'sha256': e['sha256'], 'count': d['count'], 'role': role,
                   'split_sha256':self.split_sha256,'split_manifest_file_sha256':self.split_manifest_file_sha256, 'draw': self.split['source_draw'],
                   'token_ids_sha256': self.split['roles'][role]['token_ids_sha256']}


@lru_cache(maxsize=1)
def split_identity(root):
    return Captures(root).split_sha256


def fixed_kwargs(enc, q, H, prov, cached, device, ldlq_sigma=None):
    from tessera.manifest import RotationState
    settings = cached[q]['identity']['calibration']['settings']
    # Clone every producer setting explicitly through the existing ActivationSource owner.
    keys = ('ldlq_sigma', 'ldlq_block', 'refit_objective', 'refit_objective_trailing', 'refit_reach_floor', 'refit_gauss_seidel')
    require(set(settings) == set(keys) | {'hessian'}, q + ': unhandled cached producer setting')
    if ldlq_sigma is not None:
        # Frozen-selector override: sigma rides the EXISTING ldlq_sigma owner; every
        # other cached producer kwarg stays the cached one. No second damping is added.
        settings = {**settings, 'ldlq_sigma': float(ldlq_sigma)}
    actual = enc.tx.ActivationSource({q: H}, prov, **{k: settings[k] for k in keys})
    # Same ActivationSource.for_unit seam as cd2, with the producer's exact
    # settings supplied explicitly. Compute its LDL/refit inputs only once.
    return actual.for_unit(q,H.shape[0],device,scale_plane=enc.recipe.scale_plane,rotation=RotationState.NONE)


def metadata(a):
    torch, CD2, L, V = setup()
    m, cached, routed, dense = population()
    enc = CD2.Encoder(FMT)
    rows = [routed[0], dense[0]]
    proofs = []
    for r in rows:
        t = served(r)
        _, s = source(r, cached, small=True)
        proofs.append({'qname': r['qname'], 'actual_decoded_shape': list(t.shape), 'actual_decoded_dtype': str(t.dtype), 'bytes': cached[r['qname']]['blob_bytes'], 'source': s})
    save(a.out, {'schema': 'd42.metadata_decoder_preflight.v1', 'identities': identities(), 'actual_body_layers': list(LAYERS),
                 'routed_units': len(routed), 'rank_population': [r['qname'] for r in dense], 'proofs': proofs,
                 'H_dependency': str(a.capture), 'H_loaded': False, 'encoder_recipe': cached[rows[0]['qname']]['identity']['recipe']})


def rank(a):
    torch, _, _, _ = setup()
    from d44 import output_sse
    _, cached, _, dense = population()
    captures = Captures(a.capture)
    scores, preflight_reads = [], []
    for r in dense:
        w, stamp = source(r, cached, a.device)
        q = served(r, a.device)
        d, proof = captures.get(r['qname'], 'fit', w.shape[1])
        if a.dry_run:
            _, _, L, _ = setup()
            wm = load(MODEL/'model.safetensors.index.json')['weight_map']
            sh = MODEL/wm[r['qname']+'.weight']
            preflight_reads.append(data_entry(sh))
            preflight_reads.append({'path':proof['path'],'offset':0,'bytes':Path(proof['path']).stat().st_size,'sha256':proof['sha256']})
            loc = r['formats'][FMT]['wire']
            off,n = L.member_location(loc)
            preflight_reads.append({'path':str(A8/loc['shard']),'offset':off,'bytes':n,'sha256':cached[r['qname']]['blob_sha256']})
            scores.append({'qname':r['qname'],'source_shape':list(w.shape),'capture':proof})
            continue
        # Reuse the authoritative full-float64 weight-leg contraction.
        total = output_sse(w, q, d['hessian'])
        require(math.isfinite(total) and total >= 0, r['qname'] + ': invalid output SSE')
        scores.append({'qname': r['qname'], 'output_sse': total, 'mean_output_sse_per_row': total/d['count'], 'capture': proof, 'source': stamp})
    if a.dry_run:
        preflight_reads.extend(data_entry(p) for p in (CACHE,MANIFEST,Path(a.capture)/'split-manifest.json',captures.split['source_draw']['path'],MODEL/'model.safetensors.index.json'))
        preflight_reads.extend(data_entry(p) for p in sorted(Path(a.capture).glob('layers/L*/manifest.json')))
        unique = {(e['path'],e['offset']):e for e in preflight_reads}
        entries = list(unique.values())
        dm_path = Path(a.out).with_suffix('.data.json')
        save(dm_path,{'schema':'prismaquant.prismabuild.data_manifest.v1','mount_prefix':'/mnt/shared','entries':entries,'entry_count':len(entries),'total_bytes':sum(e['bytes'] for e in entries),'produced_by':{'tool':'stage1.rank-preflight'},'annotations':{}})
        save(a.out,{'schema':'d42.all24_rank_preflight.v1','dry_run':True,'actual_units':scores,'data_manifest':str(dm_path),'split_sha256':captures.split_sha256,'split_manifest_file_sha256':captures.split_manifest_file_sha256,'identities':identities()})
        return
    scores.sort(key=lambda r: (-r['output_sse'], r['qname']))
    save(a.out, {'schema': 'd44.all24_fit_ranking.v1', 'metric': 'sum over actual FIT rows ||X(Wq-Ws)^T||^2; full unnormalized XtX, float64 W-only leg',
                 'units': scores, 'selected_qname': scores[0]['qname'], 'identities': identities(), 'split_sha256':captures.split_sha256,'split_manifest_file_sha256':captures.split_manifest_file_sha256})


def selected(a):
    m, cached, routed, dense = population()
    if getattr(a, "selection", None):
        import d44
        return m, cached, d44.selection_rows(a.selection, a.capture, m, assembling=getattr(a, "command", "") == "assemble")
    ranking = load(a.ranking)
    require({r['qname'] for r in ranking['units']} == {r['qname'] for r in dense} and len(ranking['units']) == 24, 'Missing/duplicate all24 ranking')
    ordered = sorted(ranking['units'], key=lambda r: (-r['output_sse'], r['qname']))
    require(ranking['selected_qname'] == ordered[0]['qname'], 'Chosen site is not maximum actual weight-leg output error')
    require(ranking['split_sha256'] == split_identity(a.capture), 'Ranking uses a different actual split body')
    require(all(r['capture']['role'] == 'fit' and r['capture']['split_sha256'] == ranking['split_sha256'] for r in ordered), 'Ranking must use actual FIT captures')
    return m, cached, routed + [r for r in dense if r['qname'] == ranking['selected_qname']]


def data_entry(path, offset=0, size=None):
    path = Path(path)
    size = path.stat().st_size-offset if size is None else size
    h = hashlib.sha256()
    with path.open('rb') as f:
        f.seek(offset)
        left = size
        while left:
            b = f.read(min(left, 8 << 20))
            require(bool(b), 'Truncated declared PB input')
            h.update(b)
            left -= len(b)
    return {'path': str(path), 'offset': offset, 'bytes': size, 'sha256': h.hexdigest()}


def plan(a):
    _, _, L, _ = setup()
    _, cached, rows = selected(a)
    captures = Captures(a.capture)
    timings = load(a.timings)
    require(timings['observed'] is True and timings['device'] == a.device and timings['action_key'], 'Real same-device PB runtime receipts required, no invented estimates')
    tasks, residency = [], {}
    wm = load(MODEL / 'model.safetensors.index.json')['weight_map']
    condition_cells, plan_method_drift = None, None
    if getattr(a, 'condition_selection', None):
        import d44_training as T
        sel = load(a.condition_selection)
        plan_method_drift = T.selection_binding(sel)
        require(sel['split_sha256'] == captures.split_sha256, 'condition selection split differs from the capture')
        condition_cells = {(int(c['layer']), c['role']): c for c in sel['cells']}
        setup_pool_paths = {c['pool_fit']['path'] for c in sel['cells']}
        for p in [a.condition_selection, *sorted(setup_pool_paths)]:
            require(Path(p).exists(), f'{p}: condition input missing at plan time; build the selection/pools first')
        for p in sorted(setup_pool_paths):
            receipt_path = Path(str(p)).with_suffix('.receipt.json')
            require(receipt_path.exists(), f'{receipt_path}: pool receipt sidecar missing at plan time')
    setup_entries = [data_entry(p) for p in (CACHE,MANIFEST,a.selection or a.ranking,Path(a.capture)/'split-manifest.json',captures.split['source_draw']['path'],MODEL/'model.safetensors.index.json')]
    if condition_cells is not None:
        # Real read order: selection, then each cell pool with its own-digest
        # receipt sidecar — every actual consumed file is declared.
        setup_entries.append(data_entry(a.condition_selection))
        for p in sorted(setup_pool_paths):
            setup_entries.append(data_entry(p))
            receipt_path = Path(str(p)).with_suffix('.receipt.json')
            require(receipt_path.exists(), f'{receipt_path}: pool receipt sidecar missing at plan time')
            setup_entries.append(data_entry(receipt_path))
    setup_entries.extend(data_entry(path) for path in sorted(Path(a.capture).glob('layers/L*/manifest.json')))
    for r in rows:
        q = r['qname']
        require(q in captures.units,q+': actual FIT/HELDOUT files pending')
        roles = captures.units[q][1]
        geometry = 'x'.join(map(str,cached[q]['identity']['source']['shape']))
        t = timings['geometries'][geometry]
        require(t['unit_seconds'] > 0 and t['setup_seconds'] > 0,'Invalid measured whole-unit runtime')
        key = 'L' + str(r['layer'])
        residency[key] = {'key':key,'setup_seconds':max(t['setup_seconds'],residency.get(key,{}).get('setup_seconds',0)),'setup_evidence':timings['action_key']}
        reads = list(setup_entries)
        for role in ('fit','heldout'):
            e = roles[role]
            path = Path(a.capture)/e['file']
            require(path.stat().st_size == e['bytes'],q+': actual role file length')
            reads.append({'path':str(path),'offset':0,'bytes':e['bytes'],'sha256':e['sha256']})
        sh = wm[q+'.weight']
        reads.append(data_entry(MODEL/sh))
        loc = r['formats'][FMT]['wire']
        off,n = L.member_location(loc)
        reads.append({'path':str(A8/loc['shard']),'offset':off,'bytes':n,'sha256':cached[q]['blob_sha256']})
        tid = 'unit-' + hashlib.sha256(q.encode()).hexdigest()[:20]
        tasks.append({'id':tid,'output_id':tid,'payload':{'qname':q,'reads':reads},'residency_key':key,'estimated_seconds':t['unit_seconds'],'estimate_evidence':timings['action_key']})
    chosen_arg = ['--selection',str(a.selection)] if a.selection else ['--ranking',str(a.ranking)]
    cond_arg = ['--condition-selection',str(a.condition_selection)] if condition_cells is not None else []
    common = {'argv':[PYTHON if a.device == 'cpu' else 'python3','encode_launch.py','--batch','{pb.task_batch}','--capture',str(a.capture),*chosen_arg,*cond_arg,'--root',str(a.root),'--device',a.device],
              'cwd':str(a.source_cwd),'demand':{'cpu':1,'mem_gb':a.mem_gb,**({'gpu':1} if a.device=='cuda' else {})},
              'gpu_memory_gb':a.gpu_memory_gb if a.device=='cuda' else None,'data_manifest':None,
              'env':{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
                     **({'TMPDIR':'/tmp','PRISMAQUANT_TMPDIR':'/tmp'} if a.device=='cpu' else {})},
              'tags':['gb10'] if a.device=='cuda' else ['x86'],'timeout_s':1800}
    require(a.device != 'cuda' or a.gpu_memory_gb is not None,'Explicit GPU subset budget required')
    save(a.out,{'schema':'prismabuild.logical_request.v1','common':common,
         'roster':{'schema':'prismabuild.logical_task_roster.v1','tasks':tasks},
         'batch_policy':{'schema':'prismabuild.roster_batch_policy.v1','residencies':list(residency.values()),'max_setup_fraction':a.max_setup_fraction,'max_estimated_wall_seconds':1800},
         'task_data_manifest':{'schema':'prismabuild.task_data_manifest.v1','payload_field':'reads','mount_prefix':'/mnt/shared','residency_tier':None,'residency_ram':'auto','mover_readers':4,'mover_mem_gb':1}})
    if condition_cells is not None:
        # Metadata-only drift stamp beside the plan; the logical_request schema is untouched.
        save(Path(str(a.out) + '.method-drift.json'), {'selection': str(a.condition_selection), 'method_drift': plan_method_drift})


def encode(a):
    torch, CD2, L, _ = setup()
    _, cached, rows = selected(a)
    by = {r['qname']: r for r in rows}
    captures = Captures(a.capture)
    batch = load(a.batch) if a.batch else None
    names = [t['payload']['qname'] for t in batch['tasks']] if batch else [a.qname]
    require(len(names) == len(set(names)) and all(q in by for q in names), 'Encode batch outside exact chosen body roster')
    enc = CD2.Encoder(FMT)
    cond_cells, method_drift = None, None
    if getattr(a, 'condition_selection', None):
        import d44_training as T
        sel = load(a.condition_selection)
        # Hard numeric binding to the live frozen method; the whole-file method
        # digest is a metadata-only drift stamp, never a refusal.
        method_drift = T.selection_binding(sel)
        require(sel['split_sha256'] == captures.split_sha256, 'condition selection split differs from the capture')
        cond_cells = {(int(c['layer']), c['role']): c for c in sel['cells']}
    receipts = []
    start = time.monotonic()
    for q in names:
        unit_start = time.monotonic()
        r = by[q]
        width = cached[q]['identity']['source']['shape'][1]
        if cond_cells is not None and r['kind'] == 'routed':
            fit_entry = captures.units.get(q, (None, {}))[1].get('fit')
            if fit_entry is not None:
                ecount = int(fit_entry['count'])
                if ecount == 0:
                    require(False, q + ': zero-FIT routed expert cannot receive a conditioned encode; '
                                       'no FIT Hessian exists (method prerequisite; unmeasured, not invented)')
                require(ecount > 0, q + ': invalid negative FIT count (never zero)')
            # A missing fit entry stays on the existing path: captures.get below
            # refuses it by name as a capture dependency (unmeasured, not zero).
        d, fp = captures.get(q, 'fit', width)
        held, hp = captures.get(q, 'heldout', width)
        w, stamp = source(r, cached, a.device)
        original = served(r, a.device)
        prov = dict(captures.split['roles']['fit']['provenance'])
        prov.update(split_sha256=fp['split_sha256'],source_draw=fp['draw'],capture_role='fit',capture_file_sha256=fp['sha256'])
        conditioning = None
        if cond_cells is not None and r['kind'] == 'routed':
            # Frozen final encode: EVERY routed unit reencodes on the full-FIT
            # conditioned Hessian with its cell's chosen sigma/k, on the raw count
            # scale; smaller (low-count) experts are encoded with the same chosen
            # setting. Dense/shared units stay on the unconditioned production path.
            import d44_training as T
            cell = cond_cells.get((int(r['layer']), T.role_for_qname(q)))
            require(cell is not None, q + ': routed unit has no frozen condition cell')
            require(q in cell['unit_roster'], q + ': expert absent from the frozen cell roster')
            pool, pool_receipt = T.load_pool(cell['pool_fit']['path'], expect_cell=(int(r['layer']), T.role_for_qname(q)))
            # Numerical pool/count/cell/split binding to the selection's chosen
            # pool: the consumed bytes must be the selected pool, not another
            # valid same-shape pool. Pool bytes stay strict against their own
            # sidecar digest (load_pool); the pool's method stamp is drift only.
            bind = cell['pool_fit']
            require(pool_receipt['pool_sha256'] == bind['sha256'], q + ': consumed pool bytes differ from the selection pool binding')
            require(int(pool['fit']['count']) == int(bind['count']) and int(pool['fit']['members']) == int(bind['members']),
                    q + ': consumed pool counts/members differ from the selection pool binding')
            require(pool['split_sha256'] == captures.split_sha256, q + ': pool split differs from the capture')
            require(pool['member_nfit'][q] == int(d['count']), q + ': fit count drifted against the frozen pool')
            sigma, k = cell['chosen']['sigma'], cell['chosen']['k']
            H_enc = T.rawcount_hessian(d['hessian'], int(d['count']), pool['fit']['gram_raw_sum'], int(pool['fit']['count']), k)
            kw = fixed_kwargs(enc, q, H_enc.to(dtype=torch.float32).contiguous(), prov, cached, a.device, ldlq_sigma=sigma)
            conditioning = {'sigma': sigma, 'k': k, 'selection': str(a.condition_selection),
                            'selection_sha256': sha(a.condition_selection), 'method_drift': method_drift,
                            'pool_fit_sha256': bind['sha256'], 'pool_fit_count': int(pool['fit']['count']),
                            'nfit': int(d['count']),
                            'hessian_scale': 'raw count scale: H_fit + k * pool_fit_mean (k pseudo-rows at the pool mean)'}
            del H_enc, pool
        else:
            kw = fixed_kwargs(enc, q, d['hessian'], prov, cached, a.device)
        if a.dry_run:
            receipt = {'qname': q, 'dry_run': True, 'actual_source_shape': list(w.shape), 'actual_decoder_shape': list(original.shape), 'fit': fp, 'heldout': hp, 'source': stamp, 'conditioning': conditioning}
        else:
            t0 = time.monotonic()
            [(render, blob)] = enc.encode([w], [kw], 'NONE')
            facts = wire_facts(blob,cached[q])
            require(len(blob) == cached[q]['blob_bytes'], q + ': fixed serialized unit length changed')
            require(render.shape == w.shape and render.dtype == torch.bfloat16 and torch.isfinite(render).all().item(), q + ': decoded candidate invalid')
            from tessera.unit_artifact import read_unit_artifact
            own = read_unit_artifact(blob, device=a.device).to(torch.bfloat16)
            require(torch.equal(render, own), q + ': decoder roundtrip differs from own bytes')
            filename = q.replace('.', '__') + '.tessera'
            path = Path(a.root)/'units'/filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:
                f.write(blob); f.flush(); os.fsync(f.fileno())
            receipt = {'qname': q, 'dry_run': False, 'blob_path': str(path), 'blob_bytes': len(blob), 'blob_sha256': blob_sha(blob),
                       'rendered_sha256': L.tensor_sha256(render), 'rendered_shape': list(render.shape), 'rendered_dtype': str(render.dtype),
                       'contract': r['formats'][FMT]['contract'], 'fit': fp, 'heldout': hp, 'source': stamp,
                       'producer_contract': cached[q]['identity'], 'wire_facts':facts, 'encode_seconds': time.monotonic()-t0, 'unit_seconds':time.monotonic()-unit_start, 'device':a.device, 'identities': identities(), 'conditioning': conditioning}
            receipt.update(action_key=os.environ.get('PRISMABUILD_ACTION_KEY'),setup_seconds=unit_start-STARTED)
        save(Path(a.root)/'receipts'/(q.replace('.', '__')+('.dry.json' if a.dry_run else '.json')), receipt)
        receipts.append(receipt)
        require(time.monotonic()-start < 1800, 'Encode batch exceeded real 30-minute hard bound')
    print(json.dumps({'units': len(receipts), 'elapsed_seconds': time.monotonic()-start, 'dry_run': a.dry_run}), flush=True)


def assemble(a):
    torch, _, L, _ = setup()
    m, cached, rows = selected(a)
    selected_names = {r['qname'] for r in rows}
    total_old = total_new = 0
    out = copy.deepcopy(m)
    out['picks'] = [PICK]
    candidate_records = []
    for r in out['rows']:
        q, oldfmt = r['qname'], r['pick']['a8s']
        r['pick'] = {PICK: oldfmt}
        if oldfmt != 'SOURCE':
            total_old += r['formats'][oldfmt]['priced_wire_bytes']
        if q not in selected_names:
            if oldfmt != 'SOURCE':
                total_new += r['formats'][oldfmt]['priced_wire_bytes']
            continue
        rec = load(Path(a.root)/'receipts'/(q.replace('.', '__')+'.json'))
        path = Path(rec['blob_path'])
        require(not rec['dry_run'] and path.stat().st_size == rec['blob_bytes'] == cached[q]['blob_bytes'] and sha(path) == rec['blob_sha256'], q + ': actual candidate bytes incomplete/corrupt')
        from tessera.unit_artifact import read_unit_artifact
        t = read_unit_artifact(path.read_bytes(), device='cpu').to(torch.bfloat16)
        require(list(t.shape) == rec['rendered_shape'] and L.tensor_sha256(t) == rec['rendered_sha256'], q + ': candidate own-body decode integrity')
        require(rec['fit']['split_sha256'] == rec['heldout']['split_sha256'] == split_identity(a.capture) and rec['fit']['role'] == 'fit', q + ': FIT-only candidate comparability')
        fmt = PICK + '::' + FMT
        ent = copy.deepcopy(r['formats'][oldfmt])
        ent.update(priced_wire_bytes=rec['blob_bytes'], rendered_sha256=rec['rendered_sha256'], rendered_shape=rec['rendered_shape'])
        ent['wire'] = {'root':'wirecache', 'shard':str(path), 'offset':0, 'length':rec['blob_bytes'], 'member_bytes':rec['blob_bytes'], 'wire_sha256':rec['blob_sha256'], 'dtype':'U8', 'shape':[rec['blob_bytes']]}
        r['formats'][fmt] = ent
        r['pick'][PICK] = fmt
        total_new += rec['blob_bytes']
        candidate_records.append({'qname':q, 'receipt_sha256':sha(Path(a.root)/'receipts'/(q.replace('.', '__')+'.json')), 'wire_sha256':rec['blob_sha256']})
    require(len(candidate_records) == len(selected_names) and total_old == total_new, "Complete selected-unit coverage or exact total bytes failed")
    out['d42_research'] = {'identities':identities(), 'source_payload_identity':m['payload_sha256'], 'candidate_content_sha256':blob_sha(json.dumps(candidate_records,sort_keys=True).encode()),
                          'split_sha256':split_identity(a.capture),'split_manifest_file_sha256':sha(Path(a.capture)/'split-manifest.json'), 'ranking_sha256':sha(a.ranking) if not a.selection else None, 'selection':load(a.selection) if a.selection else None, 'replaced_units':candidate_records,
                          'unchanged_nonpicked':True, 'chosen_wire_bytes':total_new, 'baseline_chosen_wire_bytes':total_old, 'training_role':'fit', 'rotation':'NONE'}
    save(a.out, out)


def readset(a):
    _, _, _, V = setup()
    G, _ = V.setup()
    m, by_layer, _ = G.load_manifest(a.manifest, sha(a.manifest), None)
    G.register_picks(m)
    from g3_readset import SourceReads, build_manifest
    reads = SourceReads(str(MODEL), [PICK+'_wa'], by_layer)
    emission = load(V.TEACHER)
    windows = emission['windows']
    spec = {'windows':[{'window_id':r['window_id'], 'path':r['path'], 'bytes':Path(r['path']).stat().st_size, 'sha256':r['file_sha256']} for r in emission['arrays']]}
    # Existing readset requires teacher.json at the metadata root. Use the actual
    # emitted receipt as setup; replace that single metadata entry, no numerical change.
    teacher_root = Path(a.root)/'teacher-metadata'
    teacher_root.mkdir(parents=True, exist_ok=True)
    save(teacher_root/'teacher.json', spec)
    h = load(V.HANDOFF)
    roots = {'a8':str(A8), 'wirecache':str(WIRE), 't8r':h['binding']['roots']['t8r'], 'exl3':'/mnt/shared/models/GLM-5.3-Flash-EXL3-TR3-4bpw'}
    dm = build_manifest(reads,[PICK+'_wa'],by_layer,roots,[(str(teacher_root),spec)],windows,
                        setup_files=[a.manifest,V.TEACHER,V.HANDOFF,V.PROBE,CENSUS/'tr3-teacher-inputs-01/final_panel_handoff.json'])
    # Keep the existing D38 real staged-range proof in this entry's readset.
    smoke = load(BASE/'g3-readset-2301/smoke-input-01/data-manifest.json')['entries'][0]
    dm['entries'].append(smoke)
    dm['entry_count'] = len(dm['entries'])
    dm['total_bytes'] = sum(e['bytes'] for e in dm['entries'])
    # The D38 range must actually be in a scheduled residency phase.
    phase = dm['read_plan']['phases'][0]
    phase['entry_indices'].append(len(dm['entries'])-1)
    phase['bytes'] += smoke['bytes']
    for phase in dm['read_plan']['phases']:
        phase['cumulative_bytes'] += smoke['bytes']
    dm['read_plan']['read_bytes'] += smoke['bytes']
    save(a.out, dm)


def score(a):
    if a.arm == 'd44ceiling_wa':
        # D44 frozen G3 ceiling adapter: unchanged v2 owner, EXL3-decode dispatch installed.
        import d44_g3
        d44_g3.score_ceiling(a)
        return
    _, _, _, V = setup()
    require(a.arm == PICK+'_wa', 'Stage1 scoring must preserve A8S W+A activation contract')
    m = load(a.manifest)
    # D44 frozen-G3 guard: swap scoring is frozen out until lock seals numeric P; the guard runs
    # BEFORE any scorer invocation and its sealed binding is stamped into the result.
    import d44_g3
    prereg_binding = d44_g3.guard_scoring(a, m, 'swap')
    require(m['d42_research']['training_role'] == 'fit' and m['d42_research']['rotation'] == 'NONE', 'Invalid Stage1 candidate')
    selection = m['d42_research'].get('selection')
    require(not selection or not selection['provisional'], 'A provisional research selection cannot become a G3 scoring arm')
    # Execute the unchanged numerical owners, prefix adapter, emitted teacher and
    # SourceReads/read-skip implementation. Candidate manifest is the only input change.
    sys.argv = [str(OWNERS/'v2_score.py'), '--arm',a.arm,'--manifest',a.manifest,'--manifest-sha256',sha(a.manifest),'--root',str(a.root)] + (['--dry-run'] if a.dry_run else [])
    if a.dry_run:
        # D38 reads small actual input slices beyond the owner's one-layer proof.
        import numpy as np
        input_proofs = []
        for r in load(V.TEACHER)['arrays']:
            arr = np.load(r['path'], mmap_mode='r', allow_pickle=False)
            require(arr.dtype == np.float32 and arr.shape == (2047,154880) and np.isfinite(arr[:1,:32]).all(), 'Actual v2 teacher slice invalid')
            input_proofs.append({'path':r['path'], 'shape':list(arr.shape), 'dtype':str(arr.dtype), 'slice_sha256':blob_sha(arr[:1,:32].tobytes())})
        chosen = next(r for r in m['rows'] if r['qname'] == m['d42_research']['replaced_units'][0]['qname'])
        key = chosen['pick'][PICK]
        from tessera.unit_artifact import read_unit_artifact
        blob = Path(chosen['formats'][key]['wire']['shard']).read_bytes()
        require(blob_sha(blob) == chosen['formats'][key]['wire']['wire_sha256'], 'Actual candidate file corruption')
        read_unit_artifact(blob, device='cpu')
        save(Path(a.root)/'actual-input-slices.json', {'teacher_slices':input_proofs, 'candidate_bytes':len(blob), 'manifest_sha256':sha(a.manifest)})
    V.main()
    if not a.dry_run:
        result_path = Path(a.root)/'gpu-score'/(a.arm+'-v2')/'result.json'
        result = load(result_path)
        result['d42_research'] = dict(m['d42_research'],action_key=os.environ.get('PRISMABUILD_ACTION_KEY'),actual_source_identities=identities())
        result['d44_g3_prereg'] = prereg_binding
        result_path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')


def reduce(a):
    spec = importlib.util.spec_from_file_location('stage1_v2_harvest',V2/'harvest.py')
    H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    paths = {'candidate':Path(a.result), 'A8S':V2/'a8s-03/gpu-score/a8s_wa-v2/result.json', 'EXL3':V2/'exl3-03/gpu-score/exl3_w-v2/result.json'}
    data, values = {}, {}
    for name, path in paths.items():
        r = load(path)
        require(r['hash_gate']['all_hashes_matched'] and r['v2_teacher']['arrays_verified'] == 25, name+': actual complete G3v2 integrity required')
        ids = [w['window_id'] for w in r['windows']]
        require(len(ids) == len(set(ids)) == 25, name+': window population')
        data[name] = r
        values[name] = H.reduce_array(path.parent/'per_position_kl.teacher2.npy')
        require(max(abs(x-w['teacher2']['mean_kl']) for x,w in zip(values[name],r['windows'])) <= 1e-15, name+': array reductions disagree')
    c = data['candidate']
    require(c['arm'] == PICK+'_wa', 'Wrong candidate arm')
    for name in ('A8S','EXL3'):
        require([w['window_id'] for w in c['windows']] == [w['window_id'] for w in data[name]['windows']], 'Paired G3v2 windows differ')
        require(c['v2_teacher']['sha256'] == data[name]['v2_teacher']['sha256'], 'Never mix teachers/v1')
    receipt = load(a.receipt)
    require(len(receipt) == 1 and receipt[0]['status'] == 'executed' and receipt[0]['returncode'] == 0, 'Candidate PB terminal receipt not successful')
    require(c['d42_research']['action_key'] == receipt[0]['action_key'], 'A different action receipt cannot qualify this candidate result')
    prereg = load(V2/'preregistration.json')
    require([w['window_id'] for w in c['windows']] == prereg['window_ids'], 'G3v2 preregistered window population differs')
    stable = [i for i,w in enumerate(prereg['window_ids']) if w not in prereg['flip_window_ids']]
    report = {'schema':'d42.g3v2_stage1.paired.v1', 'candidate':H.stats(values['candidate']), 'paired':{}, 'teacher':c['v2_teacher'],
              'result_sha256':{n:sha(p) for n,p in paths.items()}, 'receipt':receipt[0]}
    for name in ('A8S','EXL3'):
        delta = [x-y for x,y in zip(values['candidate'],values[name])]
        report['paired'][name] = dict(H.stats(delta), better_windows=sum(d<0 for d in delta), worse_windows=sum(d>0 for d in delta), tied_windows=sum(d==0 for d in delta), per_window_deltas=delta, reference_mean=statistics.mean(values[name]))
        report['paired'][name]['stable_19'] = dict(H.stats([delta[i] for i in stable]),better_windows=sum(delta[i]<0 for i in stable))
        report['paired'][name]['reference_result_sha256'] = sha(paths[name])
    report['candidate']['stable_19'] = H.stats([values['candidate'][i] for i in stable])
    save(a.out, report)


def timings(a):
    endings = [e for path in a.receipt for e in load(path)]
    successful = {e['action_key'] for e in endings if e['status']=='executed' and e['returncode']==0}
    geometries, devices = {}, set()
    for path in a.unit_receipt:
        r = load(path)
        require(not r['dry_run'] and r['action_key'] in successful,'Timing evidence is not this actual successful PB unit')
        devices.add(r['device'])
        key = 'x'.join(map(str,r['rendered_shape']))
        t = geometries.setdefault(key,{'unit_seconds':0.0,'setup_seconds':0.0,'receipt_sha256':[]})
        t['unit_seconds'] = max(t['unit_seconds'],r['unit_seconds'])
        t['setup_seconds'] = max(t['setup_seconds'],r['setup_seconds'])
        t['receipt_sha256'].append(sha(path))
    require(len(devices)==1,'Do not mix CPU/GPU runtime estimates')
    save(a.out,{'observed':True,'device':devices.pop(),'action_key':','.join(sorted(successful)),'geometries':geometries,'terminal_receipts':endings})


def confirm(a):
    # D44 frozen G3 paired-confirmation reducer adapter (schema d44.frozen_confirmation.v1).
    import d44_g3
    d44_g3.confirm(a)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('metadata','rank','timings','plan','encode','assemble','readset','score','reduce','confirm'):
        s = sub.add_parser(command)
        s.add_argument('--capture',type=Path,default=CAPTURE)
        s.add_argument('--root',type=Path,default=OUTPUT)
        s.add_argument('--out',type=Path)
        s.add_argument('--ranking',type=Path,default=OUTPUT/'ranking.json')
        s.add_argument('--selection',type=Path)
        s.add_argument('--manifest',default=str(OUTPUT/'unit-manifest.json'))
        if command in ('encode','plan','rank'):
            s.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        if command == 'encode':
            s.add_argument('--batch')
            s.add_argument('--qname')
            s.add_argument('--dry-run',action='store_true')
            s.add_argument('--condition-selection',type=Path)
        if command == 'rank':
            s.add_argument('--dry-run',action='store_true')
        if command == 'plan':
            s.add_argument('--timings',required=True)
            s.add_argument('--mem-gb',type=int,required=True)
            s.add_argument('--gpu-memory-gb',type=int)
            s.add_argument('--max-setup-fraction',type=float,required=True)
            s.add_argument('--source-cwd',type=Path,default=HERE)
            s.add_argument('--condition-selection',type=Path)
        if command == 'score':
            s.add_argument('--arm',default=PICK+'_wa')
            s.add_argument('--prereg',type=Path)
            s.add_argument('--dry-run',action='store_true')
        if command == 'reduce':
            s.add_argument('--result',required=True)
            s.add_argument('--receipt',required=True)
        if command == 'confirm':
            s.add_argument('--prereg',required=True,type=Path)
            s.add_argument('--swap-result',required=True,type=Path)
            s.add_argument('--ceiling-result',required=True,type=Path)
            s.add_argument('--a8s-result',required=True,type=Path)
            s.add_argument('--swap-receipt',required=True,type=Path)
            s.add_argument('--ceiling-receipt',required=True,type=Path)
        if command == 'timings':
            s.add_argument('--receipt',action='append',required=True)
            s.add_argument('--unit-receipt',action='append',required=True)
    a = p.parse_args()
    if a.command not in ('encode','score'):
        require(a.out is not None,'--out required')
    if a.command == 'encode':
        require(bool(a.batch) != bool(a.qname),'Exactly one --batch or --qname required')
    globals()[a.command](a)


if __name__ == '__main__':
    main()
