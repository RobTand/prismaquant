"""Check that an assembled T4 overlay loads as a prepared completion, CPU-only.

Runs on the four bindings ``assemble_t4_overlay.py`` publishes
(``catalog-pair-inputs.json``):

1. ``joint_catalog_extension.verify_catalog_pair``: scientific identity,
   strict additivity, per-cell qualification and source/H adoption.
2. ``tessera_joint_aura.load_measured_anchor_input`` on the extended plan's
   inputs. Its ``candidate_overlay`` intake runs ``attach_candidate_overlay``
   against the original measured journal, which is the check the first catalog
   failed on every cell.
3. The model-free half of the check a cost run makes on a prepared completion
   (``tessera_joint_aura``, the ``prepared`` branch): schema and status, plan
   digest, the ordered candidate roster, cell count, render census, PWC
   inputs, verified-cell coverage, render paths, render origins and wire
   digests.

Out of scope: the source-model, implementation and projection-backend
identities, the render file hashes and ``_live_targets``. Those need the
streamed source model or a full read of every render. This tool refuses any
open under the plan's source model directory and declares no data manifest.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bound_json(binding):
    raw = Path(binding['path']).read_bytes()
    assert sha(raw) == binding['sha256'], binding
    return json.loads(raw)


def refuse_model_reads(model_root):
    """Refuse any Python-level open under the source model directory.

    Guards the stop rule for this check: it must read campaign files only,
    never a model shard. Opens from native code (a Rust safetensors reader) do
    not raise audit events; nothing in the checked path uses one.
    """
    root = os.path.realpath(model_root)

    def hook(event, hook_args):
        if event == 'open' and hook_args and isinstance(hook_args[0], (str, bytes, os.PathLike)):
            path = os.path.realpath(os.fsdecode(hook_args[0]))
            if path == root or path.startswith(root + os.sep):
                raise PermissionError(f'overlay check refuses to read the source model: {path}')

    sys.addaudithook(hook)


def check_completion(completion, *, plan_sha256, data, cache):
    """The model-free prepared-completion checks, in the cost run's order."""
    from prismaquant.tessera_joint_aura import PREPARED_SCHEMA, render_origin_census

    assert completion.get('schema') == PREPARED_SCHEMA, completion.get('schema')
    assert completion.get('status') == 'complete'
    assert completion.get('plan_sha256') == plan_sha256, 'prepared plan_sha256'
    assert completion.get('measured_cells') == len(data.cells), 'prepared measured_cells'
    census = render_origin_census(cell['render_origin'] for cell in data.cells.values())
    for key in ('render_origins', 'render_comparisons'):
        assert completion.get(key) == census[key], 'prepared ' + key
        assert cache.metadata.get(key) == census[key], 'prepared cache ' + key
    assert completion['formats_by_qname'] == {n: list(v) for n, v in data.formats_by_qname.items()}, \
        'prepared exact candidate roster'
    assert cache.metadata['inputs'] == data.inputs, 'prepared source bindings'
    assert set(cache.metadata['verified_cells']) == set(data.cells), 'prepared verified cell coverage'
    assert cache.weights == {pair: cell['render'] for pair, cell in data.cells.items()}, \
        'prepared render paths'
    for pair, cell in data.cells.items():
        verified = cache.metadata['verified_cells'][pair]
        assert verified['render_origin'] == cell['render_origin'], pair
        assert verified['wire_sha256'] == cell['record']['blob_sha256'], pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, help='catalog-pair-inputs.json')
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--out', required=True, help='result JSON; must not exist')
    args = parser.parse_args()
    out = Path(args.out)
    assert not out.exists(), out
    from prismaquant.joint_catalog_extension import ADDED_FORMAT, verify_catalog_pair
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant.tessera_joint_aura import load_measured_anchor_input

    started = time.monotonic()
    inputs = bound_json({'path': args.inputs, 'sha256': args.inputs_sha256})
    plan = bound_json(inputs['extended_plan'])
    refuse_model_reads(plan['model'])
    evidence = verify_catalog_pair(inputs)
    print(json.dumps({'step': 'verify_catalog_pair', 's': round(time.monotonic() - started, 1)}), flush=True)
    data = load_measured_anchor_input(plan['inputs'], verify_payloads=False, require_existing_renders=True,
                                      historical_encoder_reuse=plan.get('historical_encoder_reuse'))
    added = sum(1 for (_name, fmt) in data.cells if fmt == ADDED_FORMAT)
    print(json.dumps({'step': 'load_measured_anchor_input', 'cells': len(data.cells), 'added': added,
                      's': round(time.monotonic() - started, 1)}), flush=True)
    completion = bound_json(inputs['extended_prepared'])
    raw = Path(completion['production_cache']['path']).read_bytes()
    assert sha(raw) == completion['production_cache']['sha256'], 'qualified PWC'
    cache = pickle.loads(raw)
    del raw
    assert isinstance(cache, ProductionWeightCache)
    check_completion(completion, plan_sha256=inputs['extended_plan']['sha256'], data=data, cache=cache)
    result = {'status': 'passed', 'inputs': {'path': args.inputs, 'sha256': args.inputs_sha256},
              'evidence': evidence, 'loader_cells': len(data.cells), 'loader_units': len(data.formats_by_qname),
              'loader_added_cells': added, 'added_format': ADDED_FORMAT,
              'scope': 'verify_catalog_pair + load_measured_anchor_input(verify_payloads=False) + '
                       'model-free prepared-completion checks; opens under the source model refused',
              'seconds': round(time.monotonic() - started, 1)}
    raw = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({**result, 'result': {'path': str(out), 'sha256': sha(raw)}}), flush=True)


if __name__ == '__main__':
    main()
