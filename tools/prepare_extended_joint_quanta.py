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

from prismaquant.joint_catalog_extension import (
    create_extension, extension_campaign_identity, require_extension)
from prismaquant.joint_replay_regime import REPLAY_REGIME_ENV
from prismaquant.stage_b_prep_io import (
    PreparationPublicationRefused, PreparationReadRefused, bind_preparation_publication,
    bind_staged_reads, publish_files, read_input)
from tools.regenerate_joint_quanta import (_load_json, _pretty, _retained_budget_provenance,
    main as regenerate)


#: The checkpoint reader prefix the preparation passes to the generator
#: (``--source-layers-prefix``, PQ #900).
SOURCE_LAYERS_PREFIX = 'model.language_model.layers.'


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


#: What ``joint_cost_stage_a`` seals as ``run_identity.read_manifest_sha256``
#: when the dispatcher bound no read manifest.
UNBOUND_READ_MANIFEST_SHA256 = '0' * 64


def proofs_name_parent_as_read(proofs, *, parent_sha256, plan_sha256):
    """Whether every Stage A proof seals ``parent_sha256`` as what its run read (#1126).

    Stage A records the manifest the dispatcher gave it to read as
    ``run_identity.read_manifest_sha256`` and the plan it ran as
    ``run_identity.plan_sha256``. A run whose plan was re-derived from its
    parent's plan (R13) read a parent that names the older plan; that sealed
    identity, not the parent's own annotation, is what binds the parent to
    the run. The all-zero digest of an unbound run admits nothing.
    """
    if not proofs or parent_sha256 == UNBOUND_READ_MANIFEST_SHA256:
        return False
    for proof in proofs:
        identity = proof.get('run_identity')
        if (not isinstance(identity, dict)
                or identity.get('read_manifest_sha256') != parent_sha256
                or identity.get('plan_sha256') != plan_sha256):
            return False
    return True


def extend_parent(parent, additions, *, old_plan, new_plan, prepared, extension,
                  read_by_original_run=False):
    """Preserve historical layer extents; complete actual candidate reads downstream.

    The base parent names the original plan, or, with ``read_by_original_run``
    (:func:`proofs_name_parent_as_read`), is the manifest the original run's
    sealed identity says it read; the result then records that rule.
    """
    result = copy.deepcopy(parent)
    annotations = result['annotations']
    parent_plan_sha256 = annotations['plan_sha256']
    if parent_plan_sha256 != old_plan['sha256'] and not read_by_original_run:
        raise ValueError('base parent does not name the original scientific plan, '
                         'and no Stage A proof seals it as what the original run read')
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
    if parent_plan_sha256 != old_plan['sha256']:
        result['produced_by']['parent_admitted_by'] = {
            'rule': 'original_run_read_manifest', 'parent_plan_sha256': parent_plan_sha256}
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
    entries = []
    for path in sorted(control_paths(inputs, plan, prepared, extension)):
        p = Path(path)
        if not p.is_absolute() or not str(p).startswith('/mnt/shared/') or not p.is_file():
            raise ValueError(f'control dependency is not a present shared file: {path}')
        entries.append({'path': str(p), 'offset': 0, 'bytes': p.stat().st_size, 'sha256': None})
    return entries


def control_paths(inputs, plan, prepared, extension=None):
    """The control files the Stage B head closes over, as a set of paths.

    ``extension`` is None before the catalog extension exists: the
    preparation read set (PQ #1070) is built before this action writes it.
    """
    return set(control_digests(inputs, plan, prepared, extension))


def control_digests(inputs, plan, prepared, extension=None):
    """:func:`control_paths` with each file's bound SHA-256, or None when unbound.

    The preparation's data manifest declares a digest for every entry
    (PQ #1092); a bound digest is taken from its binding, not rehashed.
    """
    from prismaquant.tessera_joint_allocation import _read_bound
    bindings = [*inputs.values(), *([] if extension is None else [extension]),
                prepared['production_cache']]
    if extension is not None:
        # A v3 extension derives the original run's null scope from the
        # original plan and the frozen campaign identity (PQ #1126); every
        # consumer re-reads that identity file, so it is a control dependency.
        identity = extension_campaign_identity(
            json.loads(_read_bound(extension, 'Stage B catalog extension')))
        if identity is not None:
            bindings.append(identity)
    bindings += [plan['stage_b_resource_policy'], plan['served_activation_policy'], plan['inputs']['candidate_overlay']]
    documents = [json.loads(_read_bound(b, 'Stage B metadata closure')) for b in
                 (plan['stage_b_resource_policy'], plan['served_activation_policy'], plan['inputs']['candidate_overlay'])]
    resources, activation, catalog = documents
    bindings += list(resources['inputs'].values())
    bindings += [activation[k] for k in ('original_prepared', 'original_cache', 'census')]
    bindings += [catalog[k] for k in ('old_prepared', 'old_pwc', 'cost', 'reseal_proof')]
    proof = json.loads(_read_bound(catalog['reseal_proof'], 'encoder adoption proof'))
    digests = {}
    for binding in bindings:
        if digests.get(binding['path'], binding['sha256']) != binding['sha256']:
            raise ValueError(f"control file {binding['path']} is bound to two digests")
        digests[binding['path']] = binding['sha256']
    for path in (proof['fixture_id']['result'], *(arm['result'] for arm in proof['arms'])):
        digests.setdefault(path, None)
    return digests


def require_derived_budget(plan, *, plan_sha256):
    """The extended plan's retained budget must be its resource policy's (PQ #1022).

    Stage B metadata is only ever produced for a plan whose retained budget
    was derived from the roster it must admit. A plan that binds no policy
    carries an operator-declared budget -- the base GLM plan's declared a
    4 MiB candidate delta no matrix in the roster fits -- and refuses here,
    naming the fix, before any head intake or record is produced.
    """
    if (plan.get('execution') or {}).get('retained_operator_windows') is None:
        raise ValueError(f'extended plan {plan_sha256} seals no retained operator windows: refusing')
    if plan.get('stage_b_resource_policy') is None:
        raise ValueError(
            f'extended plan {plan_sha256} binds no stage_b_resource_policy, so its retained budget '
            'was not derived from the roster; derive it with '
            'prismaquant.joint_stageb_resources.derive_policy and bind it as stage_b_resource_policy')
    return _retained_budget_provenance(plan, plan_sha256=plan_sha256)


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
    stage_b_replay_mode(spec)


def stage_b_replay_mode(spec):
    """The replay mode the reviewed spec launches every quantum in (PQ #1011).

    ``spill`` when the spec declares the one-pass replay spill, else
    ``windowed``. The read plan is sealed for this mode, and the quantum
    refuses a launch in the other one.
    """
    from prismaquant.joint_replay_spill import stage_b_spill_config
    env = {key: str(value) for key, value in (spec.get('env') or {}).items()}
    try:
        declared = stage_b_spill_config(env)
    except ValueError as exc:
        raise ValueError(f'Stage B spec declares an incomplete spill: {exc}') from exc
    return 'windowed' if declared is None else 'spill'


def campaign_identity_binding(args):
    """The frozen campaign identity the command line binds, or None."""
    path = getattr(args, 'campaign_identity', None)
    digest = getattr(args, 'campaign_identity_sha256', None)
    if bool(path) != bool(digest):
        raise ValueError('--campaign-identity and --campaign-identity-sha256 go together')
    if path is None:
        return None
    return {'path': str(Path(path).resolve()), 'sha256': str(digest)}


def prepare(args):
    if getattr(args, 'allowed_tiers', None) is not None and args.data_manifest_sha256 is None:
        raise ValueError('--allowed-tiers needs --data-manifest-sha256')
    strict = []
    if getattr(args, 'data_manifest_sha256', None) is not None:
        # PQ #1092: every declared input off the stage, bound before the
        # first read; the files this action writes under the metadata root
        # are read where they are. The generator binds the same digest.
        from prismaquant.staged_tier_policy import DEFAULT_ALLOWED_TIERS
        tiers = args.allowed_tiers or DEFAULT_ALLOWED_TIERS
        bind_staged_reads(manifest_sha256=args.data_manifest_sha256,
                          allowed_tiers=tiers, own_outputs=[args.metadata_root])
        strict = ['--data-manifest-sha256', args.data_manifest_sha256,
                  '--allowed-tiers', tiers]
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
    # Off the stage only when strict: without the flag the call is as before.
    proof_read = ({'read': lambda p, sha: read_input(p, sha256=sha, where='Stage A proof')}
                  if strict else {})
    documents = [load_stage_a_receipt_like(path, digest, **proof_read)
                 for path, digest in proofs]
    bands = sorted((doc['band']['boundary'] for doc in documents if doc['status'] == 'band'), reverse=True)
    parent = _load_json(args.parent_manifest, digest=args.parent_manifest_sha256, where='original parent manifest')
    derivation = _load_json(args.derivation, digest=args.derivation_sha256, where='original quantum derivation')
    spec = _load_json(args.spec, digest=args.spec_sha256, where='reviewed Stage B container spec')
    policy = require_derived_budget(plan, plan_sha256=inputs['extended_plan']['sha256'])
    check_stage_b_spec(args.spec, spec, policy)
    replay_mode = stage_b_replay_mode(spec)
    root = args.metadata_root.resolve()
    # With --produced-output, inside the admitted PrismaBuild action, every
    # file below is a produced output committed at its origin (PQ #1070);
    # without it each is published directly, as before.
    publication = bind_preparation_publication(root, required=getattr(args, 'produced_output', False))
    root.mkdir(parents=True, exist_ok=True)
    proof_path = root/'catalog-extension.json'
    if proof_path.exists():
        extension = bind(proof_path)
        require_extension(extension, run_header=stage_a_run_header(documents[0]),
                          plan_sha256=inputs['extended_plan']['sha256'],
                          prepared_sha256=inputs['extended_prepared']['sha256'])
    else:
        # PQ #1126: a run that sealed campaign_scope null (R13) needs the
        # frozen campaign identity, so the extension derives the original
        # campaign's scope from the original plan; a run that sealed its
        # scope ignores the binding and creates the v2 bytes it always did.
        extension = create_extension(
            inputs=inputs, adjoint_capture=first_binding, output=proof_path,
            campaign_identity=campaign_identity_binding(args),
            publish=lambda path, raw: bool(publish_files(
                publication, 'extension', [(path, raw, 'catalog extension')])))
    additions = metadata_entries(inputs, plan, prepared, extension)
    read_by_original_run = proofs_name_parent_as_read(
        documents, parent_sha256=args.parent_manifest_sha256,
        plan_sha256=inputs['original_plan']['sha256'])
    updated = extend_parent(parent, additions, old_plan=inputs['original_plan'],
        new_plan=inputs['extended_plan'], prepared=prepared, extension=extension,
        read_by_original_run=read_by_original_run)
    updated['produced_by']['original_parent_manifest'] = {'path': str(args.parent_manifest), 'sha256': args.parent_manifest_sha256}
    parent_path = root/'parent.json.gz'
    partition_path = root/'partition.json'
    derivation_path = root/'derivation-input.json'
    spec_path = root/'stage-b-spec.json'
    publish_files(publication, 'inputs', [
        (parent_path, gzip.compress(_pretty(updated), mtime=0), 'extended control parent'),
        (partition_path, _pretty(policy['derivation']), 'actual resource partition'),
        (derivation_path, _pretty(derivation), 'original derivation'),
        (spec_path, _pretty(spec), 'reviewed Stage B spec')])
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
        '--source-layers-prefix', SOURCE_LAYERS_PREFIX,
        # PQ #1010: the head intake runs once, here; each quantum's head
        # phase declares its layer's sealed slice instead of re-walking.
        '--head-slices',
        # PQ #1011: the read plan is sealed for the spec's replay mode.
        '--replay-mode', replay_mode]
    regime = (spec.get('env') or {}).get(REPLAY_REGIME_ENV)
    if replay_mode == 'spill' and regime is not None:
        # The spill bound counts parts per capture group, so each layer's
        # seal takes the regime the spec launches every quantum in.
        generator += ['--replay-regime', str(regime)]
    if publication is not None:
        # The generator files its groups under this action's publication.
        generator.append('--produced-output')
    generator += strict
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
    publish_files(publication, 'launch', [(root/launch_name, _pretty({
        'schema': 'prismaquant.extended_stage_b_launch.v1',
        'catalog_extension': extension, 'resource_policy': plan['stage_b_resource_policy'],
        'generator_argv': generator, 'coordinator_argv': launch,
        'requires': 'reviewed source checkout; all GPU work through published PB dispatcher'}),
        'launch recipe')])
    print(json.dumps({'status': 'metadata_ready', 'launch': str(root/launch_name),
                      'catalog_extension': extension,
                      'produced_batches': [] if publication is None else publication.batches}))


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
    parser.add_argument('--campaign-identity', type=Path, default=None,
                        help='the frozen campaign identity file (PQ #1126); required when '
                             'the Stage A proofs seal campaign_scope null, so the catalog '
                             'extension derives the original campaign scope from the '
                             'original plan')
    parser.add_argument('--campaign-identity-sha256', default=None)
    parser.add_argument('--data-manifest-sha256', default=None,
                        help='read every declared input off the PrismaBuild stage, '
                             'digest-checked, and refuse rather than read the pool '
                             '(PQ #1092); the digest of the data manifest the action '
                             'was submitted with. Forwarded to the generator')
    parser.add_argument('--allowed-tiers', default=None,
                        help='with --data-manifest-sha256: the staged tiers a read '
                             'may come from (default ram,ssd)')
    parser.add_argument('--produced-output', action='store_true',
                        help='commit every file as a PrismaBuild produced output of this '
                             'admitted action (PQ #1070); the action must declare '
                             'the write-only template tools/stage_b_preparation_submission.py '
                             'writes, over --metadata-root')
    args = parser.parse_args(argv)
    try:
        prepare(args)
    except (ValueError, OSError, PreparationPublicationRefused, PreparationReadRefused) as exc:
        parser.exit(3, f'Stage B metadata refused: {exc}\n')


if __name__ == '__main__':
    main()
