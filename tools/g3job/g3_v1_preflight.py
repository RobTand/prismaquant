"""Preflight the actual old scorer argv and actual artifact decode; no forward or metric change."""
import ast
import json
import os
from pathlib import Path
import sys

import g3_offline_decoded_kl as g3
from tessera.unit_artifact import read_unit_artifact

cmd = json.loads(sys.argv[1])
qualify_then_score = sys.argv[2:] == ['--qualify-then-score']
original_arms = dict(g3.ARMS)
# Reuse the archived scorer's parser construction, including its actual defaults.
source = ast.parse(Path(g3.__file__).read_text())
main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
body = []
for n in main.body:
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in n.targets):
        break
    body.append(n)
ns = dict(vars(g3))
exec(compile(ast.Module(body=body, type_ignores=[]), g3.__file__, 'exec'), ns)
args = ns['p'].parse_args(cmd[2:])
m, by_layer, msha = g3.read_g3_manifest(args.manifest, args.manifest_sha256, None)
g3.register_picks(m)
arms = args.arms.split(',') if args.arms else [args.arm]
plan, totals = g3.plan_arms(arms, by_layer)
for arm in arms:
    for rows in by_layer.values():
        g3.hook_skip(arm, rows)
sys.path.insert(0, str(g3.PQ))
from experiments.glm_tr3_full_vocab import load_panel
from g3_prepared_source import validate_prepared_source
from tools.full_kl_teacher_payload import canonical_sha256
from tools.build_streamed_full_kl_teacher import _source_derivative_policy
panel, panel_inputs = load_panel(args.panel, arrays_root=args.arrays_root)
ids = [w['window_id'] for w in panel['windows']]
assert len(ids) == 25 and len(panel_inputs) == 25
teachers = g3.Teachers([('teacher04', args.teacher, args.teacher_sha256), ('teacher2', args.teacher2, args.teacher2_sha256)], ids)
identity, source_receipt = validate_prepared_source(
    Path(args.model).resolve(strict=True), args.source_preparation, args.source_preparation_sha256,
    stored_identity=teachers.specs[0][4]["source_model_identity"])
print("G3 stored source metadata", source_receipt["identity_sha256"], flush=True)
identity_sha = source_receipt["identity_sha256"] if g3.dev_mode_enabled() else canonical_sha256(identity)
g3.source_identity_check(teachers.specs[0][4], identity_sha, args.reference_binding,
                         args.reference_binding_sha256, identity, Path(args.model))
_source_derivative_policy(args)
for _, teacher_root, _, windows, _ in teachers.specs:
    for w in windows.values():
        assert (teacher_root / w['path']).stat().st_size == w['bytes']
key = next((g3.ARMS[arm][0] for arm in arms if g3.ARMS[arm][0] is not None), None)
reader = g3.WireReader(key, {}, {'t8r': args.t8r_export, 'a8': args.a8_export, 'wirecache': args.wire_cache, 'exl3': args.exl3_dir})
selected = {}
for r in m['rows']:
    if key is not None and r.get(key + '_format') != 'SOURCE':
        selected.setdefault((r['kind'], r['role'], r[key + '_format']), r)
checks = []
try:
    for r in selected.values():
        blob = reader._read(r)
        dec = read_unit_artifact(blob, device='cuda').to(g3.torch.bfloat16)
        got = g3.L.check_identity(dec, r[key + '_rendered_sha256'], r['qname'])
        assert list(dec.shape) == r[key + '_rendered_shape']
        checks.append({'qname': r['qname'], 'format': r[key + '_format'], 'wire': r[key + '_wire'], 'rendered_sha256': got})
        print('preflight actual artifact decoded', r['qname'], r[key + '_format'], got, flush=True)
        del blob, dec
finally:
    reader.close()
result = {'schema': 'campaign.t8_v1.real_invocation_preflight/1', 'argv': cmd, 'parsed_args': vars(args), 'manifest_sha256': msha, 'artifact_binding': m['artifact_binding'], 'window_ids': ids, 'teachers': [x[0] for x in teachers.specs], 'plan_totals': totals, 'decode_checks': checks, 'device': g3.torch.cuda.get_device_name(0), 'torch': g3.torch.__version__, 'cuda': g3.torch.version.cuda, 'container_content_sha256': os.environ.get('PRISMAQUANT_CONTAINER_CONTENT_SHA256'), 'instrument_source_sha256': g3.sha256_file(g3.__file__)}
result['prepared_source_receipt'] = source_receipt
path = Path('/out/invocation-preflight.json')
with path.open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps({'output': str(path), 'sha256': g3.sha256_file(path), 'decode_checks': len(checks)}), flush=True)
if qualify_then_score:
    teachers.close()
    g3.ARMS.clear()
    g3.ARMS.update(original_arms)
    sys.argv = cmd[1:]
    g3.main()
