#!/usr/bin/env python3
"""Publish a new Stage B metadata generation only after real capture/qualification gates.

Run this CPU metadata producer through PB. It never submits children. The emitted
launch argv is executed by the coordinator using the published PB dispatcher.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

from prismaquant.joint_catalog_extension import create_extension, require_extension
from prismaquant.joint_stageb_resources import verify_policy
from tools.regenerate_joint_quanta import _load_json, _publish, _pretty, main as regenerate


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def extend_parent(parent, additions, *, old_plan, new_plan, prepared, extension):
    """Preserve historical layer extents; complete actual candidate reads downstream."""
    result = copy.deepcopy(parent)
    annotations = result['annotations']
    if annotations['plan_sha256'] != old_plan['sha256']:
        raise ValueError('base parent does not name the original scientific plan')
    phases = annotations['phases']
    if phases[0]['name'] != 'head':
        raise ValueError('base parent needs its original first head phase')
    end = phases[0]['bytes']; count = 0; offset = 0
    while count < len(result['entries']) and offset < end:
        offset += result['entries'][count]['bytes']; count += 1
    if offset != end:
        raise ValueError('head phase does not end at an entry boundary')
    existing = {(r['path'], r['offset'], r['bytes']) for r in result['entries'][:count]}
    added = []
    for row in additions:
        key = (row['path'], row['offset'], row['bytes'])
        if key not in existing:
            added.append(row); existing.add(key)
    result['entries'][count:count] = added
    growth = sum(r['bytes'] for r in added)
    phases[0]['bytes'] += growth
    for phase in phases:
        phase['cumulative_bytes'] += growth
    result['entry_count'] = len(result['entries'])
    result['total_bytes'] += growth
    annotations.update(plan=new_plan['path'], plan_sha256=new_plan['sha256'],
        measured_cells=sum(f != 'BF16' for fs in prepared['formats_by_qname'].values() for f in fs),
        catalog_extension=extension, executable_prepared_inputs_required=True,
        parent_layer_extents='historical_only; actual candidates completed by executable prepared-input producer')
    # Counts/bytes below belong to the old walk, not the new control head.
    for key in ('counts', 'bytes', 'wire_dir', 'argv'):
        annotations.pop(key, None)
    result['produced_by'] = {'tool': 'tools.prepare_extended_joint_quanta',
        'plan': new_plan['path'], 'plan_sha256': new_plan['sha256'],
        'catalog_extension': extension}
    return result


def metadata_entries(inputs, plan, prepared, extension, receipt):
    """Add only control dependencies; executable readsets own tensor payloads."""
    from prismaquant.tessera_joint_allocation import _read_bound
    bindings = [*inputs.values(), extension, receipt, prepared['production_cache']]
    bindings += [plan['stage_b_resource_policy'], plan['served_activation_policy'], plan['inputs']['candidate_overlay']]
    documents = [json.loads(_read_bound(b, 'Stage B metadata closure')) for b in
                 (plan['stage_b_resource_policy'], plan['served_activation_policy'], plan['inputs']['candidate_overlay'])]
    resources, activation, catalog = documents
    bindings += list(resources['inputs'].values())
    bindings += [activation[k] for k in ('original_prepared', 'original_cache', 'census')]
    bindings += [catalog[k] for k in ('old_prepared', 'old_pwc', 'cost', 'reseal_proof')]
    proof = json.loads(_read_bound(catalog['reseal_proof'], 'encoder adoption proof'))
    paths = {b['path'] for b in bindings}
    paths.add(proof['fixture_id']['result'])
    paths.update(arm['result'] for arm in proof['arms'])
    # Capture checkpoint metadata is consumed beside its declared tensor entries.
    capture = json.loads(_read_bound(receipt, 'original completed capture'))
    directory = Path(capture['boundary_storage']['directory']).parent
    paths.update(str(directory/'checkpoints'/f"boundary-{c['boundary']:03d}"/'checkpoint.json')
                 for c in capture['checkpoints'])
    entries = []
    for path in sorted(paths):
        p = Path(path)
        if not p.is_absolute() or not str(p).startswith('/mnt/shared/') or not p.is_file():
            raise ValueError(f'control dependency is not a present shared file: {path}')
        entries.append({'path': str(p), 'offset': 0, 'bytes': p.stat().st_size, 'sha256': None})
    return entries


def prepare(args):
    inputs = _load_json(args.pair_inputs, digest=args.pair_inputs_sha256, where='catalog pair')
    plan = _load_json(Path(inputs['extended_plan']['path']), digest=inputs['extended_plan']['sha256'], where='extended plan')
    prepared = _load_json(Path(inputs['extended_prepared']['path']), digest=inputs['extended_prepared']['sha256'], where='extended preparation')
    receipt_binding = {'path': str(args.adjoint_receipt.resolve()), 'sha256': args.adjoint_receipt_sha256}
    receipt = _load_json(args.adjoint_receipt, digest=args.adjoint_receipt_sha256, where='complete original capture')
    parent = _load_json(args.parent_manifest, digest=args.parent_manifest_sha256, where='original parent manifest')
    derivation = _load_json(args.derivation, digest=args.derivation_sha256, where='original quantum derivation')
    spec = _load_json(args.spec, digest=args.spec_sha256, where='reviewed Stage B container spec')
    policy = verify_policy(plan['stage_b_resource_policy'])
    from tools.dispatch_joint_quanta import _container_wrap
    _container_wrap(args.spec, ['python3'], resource_policy=policy)
    if str(spec.get('env', {}).get('PRISMAQUANT_PROD_ACT_SCALES')) != '0':
        raise ValueError('Stage B spec must explicitly seal native static activation semantics')
    root = args.metadata_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    proof_path = root/'catalog-extension.json'
    if proof_path.exists():
        extension = bind(proof_path)
        require_extension(extension, receipt=receipt, plan_sha256=inputs['extended_plan']['sha256'],
                          prepared_sha256=inputs['extended_prepared']['sha256'])
    else:
        extension = create_extension(inputs=inputs, adjoint_capture=receipt_binding, output=proof_path)
    additions = metadata_entries(inputs, plan, prepared, extension, receipt_binding)
    updated = extend_parent(parent, additions, old_plan=inputs['original_plan'],
        new_plan=inputs['extended_plan'], prepared=prepared, extension=extension)
    updated['produced_by']['original_parent_manifest'] = {'path': str(args.parent_manifest), 'sha256': args.parent_manifest_sha256}
    parent_path = root/'parent.json.gz'
    _publish(parent_path, gzip.compress(_pretty(updated), mtime=0), where='extended control parent')
    partition_path = root/'partition.json'
    _publish(partition_path, _pretty(policy['derivation']), where='actual resource partition')
    derivation_path = root/'derivation-input.json'
    _publish(derivation_path, _pretty(derivation), where='original derivation')
    spec_path = root/'stage-b-spec.json'
    _publish(spec_path, _pretty(spec), where='reviewed Stage B spec')
    generator = ['--plan', inputs['extended_plan']['path'], '--plan-sha256', inputs['extended_plan']['sha256'],
        '--prepared', inputs['extended_prepared']['path'], '--prepared-sha256', inputs['extended_prepared']['sha256'],
        '--parent-manifest', str(parent_path), '--parent-manifest-sha256', bind(parent_path)['sha256'],
        '--derivation', str(derivation_path), '--partition', str(partition_path),
        '--output-root', plan['output_root'], '--metadata-root', str(root),
        '--adjoint-receipt', str(args.adjoint_receipt), '--catalog-extension', extension['path'],
        '--catalog-extension-sha256', extension['sha256'], '--executable-readsets',
        '--source-layers-prefix', 'model.language_model.layers.']
    if regenerate(generator) != 0:
        raise ValueError('generator refused; no launch package published')
    launch = ['python3', 'tools/dispatch_joint_quanta.py', '--records', str(root/'records'),
        '--output-root', plan['output_root'], '--adjoint-receipt', str(args.adjoint_receipt),
        '--plan', inputs['extended_plan']['path'], '--spec', str(spec_path), '--priority', '-10',
        '--state', str(root/'dispatch-state.json')]
    _publish(root/'launch.json', _pretty({'schema': 'prismaquant.extended_stage_b_launch.v1',
        'catalog_extension': extension, 'resource_policy': plan['stage_b_resource_policy'],
        'generator_argv': generator, 'coordinator_argv': launch,
        'requires': 'reviewed source checkout; all GPU work through published PB dispatcher'}), where='launch recipe')
    print(json.dumps({'status': 'metadata_ready', 'launch': str(root/'launch.json'), 'catalog_extension': extension}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pair-inputs', 'adjoint-receipt', 'parent-manifest', 'derivation', 'spec'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--metadata-root', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        prepare(args)
    except (ValueError, OSError) as exc:
        parser.exit(3, f'Stage B metadata refused: {exc}\n')


if __name__ == '__main__':
    main()
