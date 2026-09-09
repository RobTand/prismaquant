"""Profile a complete campaign group under both source snapshot policies.

Run prepare and run through PB. This calls the existing campaign and observer;
there is no calibration, encoder, cache, or placement implementation here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def prepare(spec_path, workspace, row_id, out):
    from tools.dispatch_tessera_campaign import load_spec, _streamed_resource_plan
    spec = load_spec(spec_path)
    workspace, out = Path(workspace), Path(out)
    original = json.loads((workspace/'plan.json').read_text())
    row = next(row for row in original['rows'] if row['row_id'] == row_id)
    if len(row['groups']) != 1 or len(row['members']) != 2:
        raise ValueError('this paired measurement requires one complete two-Linear group')
    census = json.loads(Path(original['census']).read_text())
    manifest = json.loads(Path(original['manifest']).read_text())
    index = next(i for i, r in enumerate(original['rows']) if r['row_id'] == row_id)
    template = manifest[index]['argv']
    offset = template.index('prismaquant.tessera_campaign')+1
    command = template[offset:]
    resources = {}
    for policy in ('whole-layer-v1', 'selected-tensors-v1'):
        selected = copy.deepcopy(spec)
        selected['campaign_argv'] += ['--source-snapshot-policy', policy]
        resources[policy] = _streamed_resource_plan(selected, census, row['members'], selected_source=True)
    _reference_identity, reference_states = priced_state(Path(row['dir']))
    reference_sha = hashlib.sha256(json.dumps(reference_states, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    # Preserve the parent recipe's runtime baseline; profiler export can hold
    # one bounded trace and its in-memory representation during finalization.
    memory = math.ceil((max(r['memory_bytes'] for r in resources.values())+
        original['process_baseline_bytes'])/1024**3)+4
    if memory > spec['box_memory_gb']:
        raise ValueError('paired source measurement exceeds the existing box budget')
    bindings = {str(path): digest(path) for path in (
        Path(spec_path), workspace/'plan.json', Path(original['manifest']),
        Path(original['census']), Path(row['units']),
        Path(command[command.index('--calibration-cache')+1]))}
    plan = dict(schema='prismaquant.selected_snapshot_scope_ab.v1',
        row_id=row_id, groups=row['groups'], members=row['members'], command=command,
        reference_semantics_sha256=reference_sha,
        container=spec['container'], environment=spec['env'], resources=resources,
        input_sha256=bindings, requested_memory_gib=memory, requested_cpus=spec['cpus'],
        order=['whole-layer-v1', 'selected-tensors-v1', 'selected-tensors-v1', 'whole-layer-v1'],
        measurement_scope='Complete group, fresh Python process and compiler caches per arm; shared host page cache.',
        out=str(out/'native'), expected_source_units=len(row['members']))
    out.mkdir(parents=True, exist_ok=True)
    write(out/'plan.json', plan)
    print(json.dumps(dict(plan=str(out/'plan.json'), sha256=digest(out/'plan.json'),
        memory_gib=memory, source_preparation_bytes={k:v['phases']['source_preparation']
        for k,v in resources.items()})), flush=True)


def priced_state(root):
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path
    path = root/'cost.anchors.json'
    manifest = json.loads(path.read_text())
    states = {}
    for unit in manifest['units']:
        name = unit['qname']
        state = _load_unit(unit_path(path.with_name(path.name+'.parts'), name),
            stage=manifest['stage'], qname=name, identity_sha256=manifest['identity_sha256'])
        if not state['anchors'] or state.get('unservable'):
            raise ValueError('paired group has empty or refused anchor state')
        anchors = sorted(({k:v for k,v in row.items() if k != 'seconds'}
                          for row in state['anchors']), key=lambda row: row['format_name'])
        wires = state['wire_records']
        for record in wires.values():
            actual = root/'cache/wire'/record['file']
            if (digest(actual) != record['blob_sha256'] or
                    actual.stat().st_size != record['blob_bytes']):
                raise ValueError('paired wire bytes differ from their journal')
        states[name] = dict(anchors=anchors, wires=wires)
    return manifest['identity_sha256'], states


def run(path, expected):
    if digest(path) != expected:
        raise ValueError('paired measurement plan changed')
    plan = json.loads(Path(path).read_text())
    if plan['schema'] != 'prismaquant.selected_snapshot_scope_ab.v1':
        raise ValueError('unknown paired source measurement schema')
    for name, sha in plan['input_sha256'].items():
        if digest(name) != sha:
            raise ValueError('paired measurement input changed: '+name)
    for name, value in plan['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('paired measurement environment changed: '+name)
    root = Path(plan['out'])
    root.mkdir(parents=True, exist_ok=False)
    rows, baseline = [], None
    for index, policy in enumerate(plan['order']):
        out = root/f'arm-{index:02d}'
        out.mkdir()
        command = list(plan['command'])
        for flag, value in {'--out': out/'cost.pkl', '--cache-dir': out/'cache',
                            '--checkpoint': out/'cost.anchors.json'}.items():
            command[command.index(flag)+1] = str(value)
        command += ['--source-snapshot-policy', policy]
        argv = [sys.executable, '-u', '-m', 'experiments.glm_full_capture_profile',
            '--evidence-out', str(out/'profile'), '--selected-anchors',
            '--anchor-profile-calls', '0,3', '--anchor-trace-max-bytes', str(256*1024**2),
            '--anchor-cuda-only', '--anchor-profile-seconds', '2', '--', *command]
        environment = dict(os.environ, TRITON_CACHE_DIR=str(out/'triton'),
            TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
        started = time.time()
        with (out/'command.log').open('x') as log:
            result = subprocess.run(argv, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record = dict(index=index, policy=policy, started_unix=started,
            finished_unix=time.time(), returncode=result.returncode, command=argv)
        rows.append(record)
        write(out/'exit.json', record)
        if result.returncode:
            raise RuntimeError(f'paired source arm {index} failed: {out}/command.log')
        if not (out/'cost.pkl').is_file() or (out/'cost.pkl').stat().st_size == 0:
            raise ValueError('paired campaign did not publish its cost output')
        current = priced_state(out)
        semantic_sha = hashlib.sha256(json.dumps(current[1], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if semantic_sha != plan['reference_semantics_sha256']:
            raise ValueError('paired anchors differ from the existing completed production group')
        if len(current[1]) != plan['expected_source_units']:
            raise ValueError('paired group lost selected units')
        if baseline is None:
            baseline = current
        elif current != baseline:
            raise ValueError('paired source identity, anchor quality/bpp, or wire differs')
        record.update(exact_parity=True, units=len(current[1]),
            anchors=sum(len(unit['anchors']) for unit in current[1].values()))
        write(out/'parity.json', record)
        print(json.dumps(record), flush=True)
    write(root/'result.json', dict(schema=plan['schema'], plan_sha256=expected,
        exact_parity=True, rows=rows, measurement_scope=plan['measurement_scope']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    for name in ('spec', 'workspace', 'row-id', 'out'):
        prep.add_argument('--'+name, required=True)
    native = sub.add_parser('run')
    native.add_argument('--plan', required=True)
    native.add_argument('--plan-sha256', required=True)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args.spec, args.workspace, args.row_id, args.out)
    else:
        run(args.plan, args.plan_sha256)


if __name__ == '__main__':
    main()
