#!/usr/bin/env python3
"""Publish a new Stage B metadata generation only after real capture/qualification gates.

Run this CPU metadata producer through PB. It never submits children. The emitted
launch argv is executed by the coordinator using the published PB dispatcher.

Band granularity (PQ #993): the Stage A proof is the completed receipt or any
set of sealed checkpoint bands. Records are emitted only for the layers whose
checkpoint has a proof. Re-running with more bands is additive and idempotent:
the catalog extension binds the Stage A run header (so the first band creates
it), the extended parent carries no Stage A proof, and every earlier record,
slice and readset republishes byte for byte.
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


def metadata_entries(inputs, plan, prepared, extension):
    """Add only control dependencies; executable readsets own tensor payloads.

    Nothing here depends on which Stage A proof is in hand (PQ #993), so
    the parent is one set of bytes for every band set: the extension binds
    the run header, and the completed receipt and the per-checkpoint
    ``checkpoint.json`` files are no longer listed. Nothing staged those
    head entries for a quantum: a quantum reads its own checkpoint's
    ``checkpoint.json`` directly from the adjoint space and checks it
    against the slice's sealed checkpoint record (``load_adjoint_checkpoint``);
    its executable readset stages that checkpoint's tensor entries.
    """
    from prismaquant.tessera_joint_allocation import _read_bound
    bindings = [*inputs.values(), extension, prepared['production_cache']]
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
    entries = []
    for path in sorted(paths):
        p = Path(path)
        if not p.is_absolute() or not str(p).startswith('/mnt/shared/') or not p.is_file():
            raise ValueError(f'control dependency is not a present shared file: {path}')
        entries.append({'path': str(p), 'offset': 0, 'bytes': p.stat().st_size, 'sha256': None})
    return entries


def check_stage_b_spec(spec_path, spec, policy):
    """Refuse a container spec every Stage B quantum row would refuse.

    The dispatcher's own wrapper runs here, with the grace set a Stage B
    quantum row declares: the head grace and the 900 s chunk grace. The spec's
    ``PRISMAQUANT_STAGED_RANGE_WAIT_S`` must sit below the smaller one, its
    host/device envelope must equal the resource policy's limits, and a
    declared cotangent workspace needs its identity mount. A spec that fails
    would publish metadata whose every quantum the dispatcher then refuses.
    """
    from tools.dispatch_joint_quanta import (
        CHUNK_PROGRESS_GRACE_S, HEAD_PROGRESS_GRACE_S, DispatchRefused, _container_wrap)
    try:
        _container_wrap(spec_path, ['python3'], resource_policy=policy,
                        progress=[('head', HEAD_PROGRESS_GRACE_S), ('chunk', CHUNK_PROGRESS_GRACE_S)])
    except DispatchRefused as exc:
        raise ValueError(f'Stage B spec refused by the quantum dispatcher: {exc}') from exc
    if str(spec.get('env', {}).get('PRISMAQUANT_PROD_ACT_SCALES')) != '0':
        raise ValueError('Stage B spec must explicitly seal native static activation semantics')


def prepare(args):
    inputs = _load_json(args.pair_inputs, digest=args.pair_inputs_sha256, where='catalog pair')
    plan = _load_json(Path(inputs['extended_plan']['path']), digest=inputs['extended_plan']['sha256'], where='extended plan')
    prepared = _load_json(Path(inputs['extended_prepared']['path']), digest=inputs['extended_prepared']['sha256'], where='extended preparation')
    if bool(args.adjoint_receipt) != bool(args.adjoint_receipt_sha256):
        raise ValueError('--adjoint-receipt and --adjoint-receipt-sha256 go together')
    proofs = ([(args.adjoint_receipt, args.adjoint_receipt_sha256)] if args.adjoint_receipt else [])
    if len(args.adjoint_band) != len(args.adjoint_band_sha256):
        raise ValueError('every --adjoint-band needs one --adjoint-band-sha256')
    proofs += list(zip(args.adjoint_band, args.adjoint_band_sha256))
    if not proofs:
        raise ValueError('Stage B needs sealed Stage A proof: the completed receipt or a checkpoint band')
    from prismaquant.joint_adjoint_slices import load_stage_a_receipt_like, stage_a_run_header
    first_binding = {'path': str(Path(proofs[0][0]).resolve()), 'sha256': proofs[0][1]}
    documents = [load_stage_a_receipt_like(path, digest) for path, digest in proofs]
    bands = sorted((doc['band']['boundary'] for doc in documents if doc['status'] == 'band'), reverse=True)
    parent = _load_json(args.parent_manifest, digest=args.parent_manifest_sha256, where='original parent manifest')
    derivation = _load_json(args.derivation, digest=args.derivation_sha256, where='original quantum derivation')
    spec = _load_json(args.spec, digest=args.spec_sha256, where='reviewed Stage B container spec')
    policy = verify_policy(plan['stage_b_resource_policy'])
    check_stage_b_spec(args.spec, spec, policy)
    root = args.metadata_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    proof_path = root/'catalog-extension.json'
    if proof_path.exists():
        extension = bind(proof_path)
        require_extension(extension, run_header=stage_a_run_header(documents[0]),
                          plan_sha256=inputs['extended_plan']['sha256'],
                          prepared_sha256=inputs['extended_prepared']['sha256'])
    else:
        extension = create_extension(inputs=inputs, adjoint_capture=first_binding, output=proof_path)
    additions = metadata_entries(inputs, plan, prepared, extension)
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
    proof_argv = []
    for (path, _), document in zip(proofs, documents):
        proof_argv += ['--adjoint-receipt' if document['status'] == 'complete' else '--adjoint-band',
                       str(path)]
    generator = ['--plan', inputs['extended_plan']['path'], '--plan-sha256', inputs['extended_plan']['sha256'],
        '--prepared', inputs['extended_prepared']['path'], '--prepared-sha256', inputs['extended_prepared']['sha256'],
        '--parent-manifest', str(parent_path), '--parent-manifest-sha256', bind(parent_path)['sha256'],
        '--derivation', str(derivation_path), '--partition', str(partition_path),
        '--output-root', plan['output_root'], '--metadata-root', str(root),
        *proof_argv, '--catalog-extension', extension['path'],
        '--catalog-extension-sha256', extension['sha256'], '--executable-readsets',
        '--source-layers-prefix', 'model.language_model.layers.',
        # PQ #1010: the head intake runs once, here; each quantum's head
        # phase declares its layer's sealed slice instead of re-walking.
        '--head-slices']
    if regenerate(generator) != 0:
        raise ValueError('generator refused; no launch package published')
    launch = ['python3', 'tools/dispatch_joint_quanta.py', '--records', str(root/'records'),
        '--output-root', plan['output_root'], *proof_argv,
        '--plan', inputs['extended_plan']['path'], '--spec', str(spec_path), '--priority', '-10',
        '--state', str(root/'dispatch-state.json')]
    # One launch recipe per proof set, so a later band set adds a recipe
    # instead of rewriting one (the completed receipt keeps launch.json).
    launch_name = ('launch.json' if args.adjoint_receipt else
                   'launch.bands-' + '-'.join(f'{b:03d}' for b in bands) + '.json')
    _publish(root/launch_name, _pretty({'schema': 'prismaquant.extended_stage_b_launch.v1',
        'catalog_extension': extension, 'resource_policy': plan['stage_b_resource_policy'],
        'generator_argv': generator, 'coordinator_argv': launch,
        'requires': 'reviewed source checkout; all GPU work through published PB dispatcher'}), where='launch recipe')
    print(json.dumps({'status': 'metadata_ready', 'launch': str(root/launch_name), 'catalog_extension': extension}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pair-inputs', 'parent-manifest', 'derivation', 'spec'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--adjoint-receipt', type=Path, default=None,
                        help='the completed Stage A receipt')
    parser.add_argument('--adjoint-receipt-sha256', default=None)
    parser.add_argument('--adjoint-band', type=Path, action='append', default=[],
                        help='a sealed checkpoint band (repeatable, PQ #993)')
    parser.add_argument('--adjoint-band-sha256', action='append', default=[])
    parser.add_argument('--metadata-root', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        prepare(args)
    except (ValueError, OSError) as exc:
        parser.exit(3, f'Stage B metadata refused: {exc}\n')


if __name__ == '__main__':
    main()
