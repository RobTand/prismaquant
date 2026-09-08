"""PB CPU-only narrow-band byte/work/admission analysis; never submits rows."""
from collections import Counter
import copy
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
ROOT = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908')
PREP = ROOT/'full-anchor-preparation-03'
OUT = ROOT/'first-release-rate-band-01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    from prismaquant.tessera_menu import expand_tessera_menu
    from prismaquant.tessera_campaign import round_one_rates
    from prismaquant.tessera_render import require_tessera_batch_encoder
    from tools.dispatch_tessera_campaign import _streamed_resource_plan
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only planning required')
    budget_path = ROOT/'exl3-first-artifact-01/root-common-surface-header-audit.json'
    if sha(budget_path) != '245bc10daacd7f0f2626b5d857d9ba1341ec89c9bab17cdb6fccf8927a611472':
        raise ValueError('accepted EXL3 common-surface budget changed')
    budget = json.loads(budget_path.read_text())['summary']
    census = json.loads((ROOT/'workspace/census.json').read_text())
    plan = json.loads((PREP/'workspace/plan.json').read_text())
    spec = json.loads((PREP/'anchor-spec.frozen.json').read_text())
    if len(census['unit_shapes']) != 36423 or len(plan['rows']) != 132:
        raise ValueError('full original roster required')
    shapes = Counter(tuple(shape) for shape in census['unit_shapes'].values())
    menu = {shape: expand_tessera_menu(shape, mode='readable', tp_degree=1,
                                      parallel_kind='none') for shape in shapes}
    snap = lambda rate, allowed: min(allowed, key=lambda r: (abs(r-rate), r))
    bands = []
    for band in [(768, 1024), (832, 1088), (896, 1152)]:
        families = sorted(set(r.family for entries in menu.values() for r in entries))
        per_shape, uniform = {}, {}
        for shape, count in shapes.items():
            per_shape[str(list(shape))] = {}
            for family in families:
                entries = [r for r in menu[shape] if r.family == family]
                allowed = sorted(r.body_rate_q256 for r in entries)
                selected = round_one_rates(allowed, band=band, anchors=3, snap=snap)
                anchors = [r for r in entries if r.body_rate_q256 in selected]
                per_shape[str(list(shape))][family] = [dict(q256=r.body_rate_q256,
                    bpp=r.bpp, bytes=r.memory_bytes) for r in anchors]
        for family in families:
            shared = set.intersection(*[{r.body_rate_q256 for r in entries
                if r.family == family and band[0] <= r.body_rate_q256 <= band[1]}
                for entries in menu.values()])
            points = []
            for rung in sorted(shared):
                total = sum(count*next(r.memory_bytes for r in menu[shape]
                    if r.family == family and r.body_rate_q256 == rung)
                    for shape, count in shapes.items())
                points.append((rung, total))
            affordable = [(r, b) for r, b in points if b <= budget['common']['tensor_payload_bytes']]
            uniform[family] = dict(legal_shared_rungs=len(shared),
                highest_uniform_within_common_budget=None if not affordable else
                    dict(q256=affordable[-1][0], bytes=affordable[-1][1],
                         bpp=float(Fraction(8*affordable[-1][1],budget['common']['parameters']))),
                endpoints=[] if not points else [dict(q256=r, bytes=b) for r,b in [points[0],points[-1]]])
        initial = sum(count*sum(len(v) for v in per_shape[str(list(shape))].values())
                      for shape, count in shapes.items())
        bands.append(dict(body_q256=list(band), initial_per_linear_anchors=initial,
                          per_shape=per_shape, uniform_byte_checks=uniform))
    print('PASS exact menu and common-budget metadata analysis', flush=True)
    require_tessera_batch_encoder()
    batches = []
    for width in [1, 2, 4, 8, 16, 24, 32]:
        estimates = []
        for row in plan['rows']:
            resource = copy.deepcopy(row['resources'])
            phase = resource['phases']['resident_anchors']
            for key in ['compatible_batch_weight_bytes', 'encoder_memo_bytes']:
                phase[key] *= width
            memory = max(sum(v.values()) for v in resource['phases'].values())
            estimates.append((memory, row['row_id']))
        maximum, widest = max(estimates)
        candidate_spec = copy.deepcopy(spec)
        argv = candidate_spec['campaign_argv']
        argv[argv.index('--anchor-batch-size')+1] = str(width)
        entry = next(row for row in plan['rows'] if row['row_id'] == widest)
        checked = _streamed_resource_plan(candidate_spec, census, entry['members'], selected_source=True)
        if checked['memory_bytes'] != maximum:
            raise ValueError('existing planner disagrees with exact linear batch term derivation')
        batches.append(dict(batch_size=width, maximum_memory_bytes=maximum,
            mem_gib=math.ceil(maximum/2**30), widest_row=widest,
            fits_104gib=maximum <= 104*2**30,
            fits_104gib_with_previous_4gib_observer=maximum+4*2**30 <= 104*2**30,
            independently_checked_widest_resource=checked,
            rows_fitting=sum(memory <= 104*2**30 for memory,_ in estimates)))
        print('PASS shared planner batch', width, math.ceil(maximum/2**30), flush=True)
    completed = []
    for row_id in ['row-0076','row-0087']:
        folder = PREP/'workspace/rows'/row_id
        manifest = folder/'cost.anchors.json'
        for path in sorted((folder/'cost.anchors.json.parts/units').glob('*.pkl')):
            envelope = pickle.loads(path.read_bytes())
            body = envelope['payload']
            if hashlib.sha256(body).hexdigest() != envelope['payload_sha256']:
                raise ValueError('completed anchor state digest differs')
            state = pickle.loads(body)
            for anchor in state.get('anchors', []):
                completed.append(dict(row=row_id, unit=envelope['qname'], anchor=anchor,
                                      record=state.get('wire_records',{}).get(anchor['format_name'])))
    report = dict(status='DERIVED_CPU_ONLY_NO_GPU_SUBMISSION', budget=budget,
        budget_audit_sha256=sha(budget_path), source_census_sha256=sha(ROOT/'workspace/census.json'),
        capture_sha256=plan['calibration_cache']['sha256'], groups=132, units=36423,
        shapes={str(list(k)):v for k,v in shapes.items()}, bands=bands, batches=batches,
        completed_anchor_records=completed, completed_anchor_count=len(completed),
        scope='Exact intrinsic wire-byte estimates through the existing menu accountant, baseline row resource bounds plus shared planner checks, and accepted partial journal records. No source weights, X/H payloads, model forward, wire render, new capture or GPU work.',
        limitations=['Uniform affordability is a byte feasibility witness, not per-Linear empirical allocation or a quality result.',
            'The current campaign CLI has no per-structure family allowlist; a readable narrow band still prices every admitted family. The downstream packed research allocation profile restricts routed choices to E4M3_K1/BF16.',
            'Rate-band changes require a new journal identity; existing seed intake can verify old wires. Out-of-band adopted anchors can widen subsequent grids and need explicit disposition before reuse.',
            'Timing values in completed journal records describe only those completed encodes; no throughput prediction or ETA is extrapolated to new band or batch widths.',
            'PB retains the existing 132 whole-group quanta and owns placement; compatible batching occurs only inside each action.'])
    OUT.mkdir(exist_ok=True)
    path = OUT/'derivation.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(output=str(path), sha256=sha(path), completed=len(completed)),sort_keys=True))


if __name__ == '__main__':
    main()
