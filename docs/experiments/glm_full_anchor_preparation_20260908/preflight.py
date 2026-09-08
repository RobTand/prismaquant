"""PB CPU-only source/menu/original-input preflight; never loads weights or X/H."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat

import torch
from safetensors.torch import load_file
from prismaquant.perturbed_x_cache import activation_cache_filename
from prismaquant.production_weight_cache import _production_cache_source_sha256
from prismaquant.tessera_campaign import _calibration_tokens
from prismaquant.tessera_menu import expand_tessera_menu, tessera_runtime_contract
from tessera.cached_unit import encoder_source_sha256
from tessera.hessian_capture import normalize_reference_load_policy, MAX_METADATA_BYTES
from tools.dispatch_tessera_campaign import load_spec, _streamed_resource_plan, partition_rows_by_fit

ROOT = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908')
OUT = ROOT/'full-anchor-preparation-01'
SOURCE = OUT/'producer-source-9d2314819'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not os.environ.get('PRISMAQUANT_TESSERA_DEV_PIN')
torch.set_num_threads(1)
cpath = ROOT/'workspace/census.json'
assert sha(cpath) == 'b63f7bf6c4320714b4ceb38fbd6996e032e0f0c9b82ac2a30a8337d846e358fd'
census = json.loads(cpath.read_text())
names = sorted(census['unit_shapes'])
assert len(names) == 36423 and len(census['anchor_groups']) == 132
assert sorted(n for group in census['anchor_groups'].values() for n in group) == names

def file_stat(name):
    path = ROOT/'workspace/calibration-cache/inputs'/activation_cache_filename(name)
    observed = path.lstat()
    assert stat.S_ISREG(observed.st_mode) and observed.st_size > 0, name
    k = census['unit_shapes'][name][1]
    return dict(name=name, path=str(path), bytes=observed.st_size,
                hessian_bytes=4*k*k, prefix_upper_bytes=4*k*512)

with ThreadPoolExecutor(max_workers=2) as pool:
    files = list(pool.map(file_stat, names))
largest = max(files, key=lambda r:r['bytes'])
policy = normalize_reference_load_policy(dict(schema='tessera.hessian_reference_load.v1',
    max_metadata_bytes=MAX_METADATA_BYTES, max_file_bytes=largest['bytes'],
    max_hessian_bytes=max(r['hessian_bytes'] for r in files)))
menus = {}
for shape in sorted(set(tuple(v) for v in census['unit_shapes'].values())):
    rows = expand_tessera_menu(shape, mode='readable', tp_degree=1, parallel_kind='none')
    groups = {}
    for family in sorted(set(r.family for r in rows)):
        selected = [r for r in rows if r.family == family]
        groups[family] = dict(count=len(selected), min_q256=min(r.body_rate_q256 for r in selected),
            max_q256=max(r.body_rate_q256 for r in selected), min_bpp=float(min(r.bits_per_param for r in selected)),
            max_bpp=float(max(r.bits_per_param for r in selected)), above_8_bpp=sum(r.bits_per_param > 8 for r in selected),
            route_statuses=sorted(set(r.admission.route_status for r in selected)))
    menus[str(list(shape))] = dict(count=len(rows), families=groups)

print('PASS original roster file geometry and current readable menus', flush=True)

artifact = ROOT/'exact-calibration-input-01/calibration_tokens.safetensors'
assert sha(artifact) == '9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329'
tokens, text = _calibration_tokens(census['model'],512,512,0)
ids = torch.cat(tokens).contiguous()
assert torch.equal(ids, load_file(str(artifact))['calibration_ids'])
fit_sha = hashlib.sha256(ids.to(torch.int32).numpy().tobytes()).hexdigest()
text_sha = hashlib.sha256(text.encode()).hexdigest()
assert fit_sha == census['fit_ids_sha256'] and text_sha == census['text_sha256']
print('PASS exact original calibration token/text equality', flush=True)
spec = load_spec(Path(__file__).with_name('anchor-spec.draft.json'))
resources = {f'row-{i:04d}': _streamed_resource_plan(spec,census,members,selected_source=True)
    for i,(_,members) in enumerate(sorted(census['anchor_groups'].items()))}
row_memory = {k:__import__('math').ceil(v['memory_bytes']/2**30) for k,v in resources.items()}
admitted, declined = partition_rows_by_fit(row_memory,1,spec['box_memory_gb'])
assert len(admitted) == 132 and not declined
assert max(v['memory_bytes'] for v in resources.values())/2**30 < 99
capture = ROOT/'workspace/calibration-cache/capture_manifest.json'
result = dict(schema='prismaquant.glm_full_anchor_preflight.v1', status='metadata_preflight_complete',
    scope='No model weights, X/H payload reads, GPU work, capture acceptance or native qualification.',
    census=dict(path=str(cpath), sha256=sha(cpath), bytes=cpath.stat().st_size, units=len(names), groups=132),
    source=dict(path=str(SOURCE), encoder_source_sha256=encoder_source_sha256(),
                prismaquant_package_sha256=_production_cache_source_sha256(),
                contract_sha256=sha(SOURCE/'src/tessera/serving/runtime_contract.json'),
                development_pin_enabled=False, development_contract_loaded=tessera_runtime_contract() is not None),
    menus=menus, admission=dict(rows=132,admitted=len(admitted),declined=declined,
        max_memory_gib=max(v['memory_bytes'] for v in resources.values())/2**30,
        max_pb_mem_gb=max(row_memory.values()),qualification=resources['row-0076']),
    file_geometry=dict(stat_count=len(files), largest=largest,
       size_histogram=dict(Counter(r['bytes'] for r in files)), max_hessian_bytes=policy['max_hessian_bytes'],
       max_prefix_bytes=max(r['prefix_upper_bytes'] for r in files)),
    hessian_reference_policy=policy,
    metadata_cap_status='Existing producer hard limit, provisional until final canonical/descriptor sizes are checked.',
    canonical_capture=dict(path=str(capture), exists=capture.is_file(), sha256=None,
        status='Not accepted or hashed here; root freezes completed original capture.'),
    tokens=dict(artifact_sha256=sha(artifact), exact_int64_equality=True, fit_ids_sha256=fit_sha,
        text_sha256=text_sha, helper_source_sha256=hashlib.sha256(inspect.getsource(_calibration_tokens).encode()).hexdigest()),
    runtime=dict(torch=torch.__version__, torch_cuda_build=torch.version.cuda,
                 affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
(OUT/'preflight.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2,sort_keys=True))
