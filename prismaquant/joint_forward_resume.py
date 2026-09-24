"""Explicit, contained forward recovery from PB-acknowledged exact entries.

This authority does not alter ``working_artifacts_reusable``. Old files retain
their original session and producer; a fresh action reads them as sealed inputs.
Only stateless profiles can reconstruct the per-batch forward state.

A recovery can itself be contained mid-forward. Its capsule then chains: the
top segment holds the boundaries that owner wrote, and ``imported`` pins the
capsule it resumed from. Every segment is re-verified against PB on load.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat

from .dev_mode import seal_check

SCHEMA = 'prismaquant.joint_forward_recovery.v1'


class ForwardRecoveryRefused(RuntimeError):
    pass


#: One PB record or proof document: an instance, commitments, export
#: manifest, receipt, export record, owner request or specification.
PROOF_READ_MAX_BYTES = 128 << 20
#: A capsule is assembled from bounded reads. Each group holds its export
#: manifest twice (parsed and raw), its receipt and its export record; the
#: rest is the specification and, for a chained segment, the owner's request.
_GROUP_READS = 4
_HEADER_READS = 2


def capsule_byte_limit(document):
    """The most a capsule declaring these groups can occupy."""
    groups = document.get('groups')
    if not isinstance(groups, list):
        raise ForwardRecoveryRefused('recovery capsule declares no groups')
    return (_HEADER_READS + _GROUP_READS * len(groups)) * PROOF_READ_MAX_BYTES


def _stat_fence(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read(path, expected=None):
    """Read one proof document; a pinned read verifies its digest first.

    An unpinned read is one PB record, bounded by ``PROOF_READ_MAX_BYTES``.
    A pinned read hashes the file in fixed chunks before loading it, so a
    wrong file is refused without being held in memory; a pinned capsule is
    then bounded by the groups it declares, not by one record's bound.
    """
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (
            expected is None and before.st_size > PROOF_READ_MAX_BYTES):
        raise ForwardRecoveryRefused('recovery proof is not a bounded regular file')
    if expected is not None:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                digest.update(chunk)
        if _stat_fence(path.lstat()) != _stat_fence(before):
            raise ForwardRecoveryRefused('recovery proof changed during read')
        if digest.hexdigest() != expected:
            raise ForwardRecoveryRefused('recovery proof SHA256 mismatch')
    raw = path.read_bytes()
    if _stat_fence(path.lstat()) != _stat_fence(before):
        raise ForwardRecoveryRefused('recovery proof changed during read')
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ForwardRecoveryRefused('recovery proof SHA256 mismatch')
    document = json.loads(raw)
    if expected is not None and len(raw) > (
            capsule_byte_limit(document)
            if isinstance(document, dict) and document.get('schema') == SCHEMA
            else PROOF_READ_MAX_BYTES):
        raise ForwardRecoveryRefused(
            'recovery proof is larger than its declared groups allow')
    return document, digest


def _sdk():
    from .staged_lease import sdk_submodule
    return {name: sdk_submodule(name) for name in
            ('pool', 'reader_lease', 'produced_output', 'produced_spool', 'core')}


def require_contained(queue, instance, sdk):
    owner = instance['owner_action_key']
    if sdk['pool']._read_json(queue.item_path(sdk['pool'].CLAIMED, owner)) is not None:
        raise ForwardRecoveryRefused('original forward owner is still claimed')
    ok, reason = sdk['reader_lease'].containment_certificate_ok(queue, {
        'action_key': owner, **instance['owner_attempt']})
    if not ok:
        raise ForwardRecoveryRefused('original forward owner is not contained: ' + reason)


def _checked_group(group, *, queue, instance, template, commitments, sdk):
    manifest, receipt, record = (group[k] for k in ('manifest', 'receipt', 'record'))
    if set(record) != {'export_key', 'manifest_sha256', 'batch_id', 'action'}:
        raise ForwardRecoveryRefused('export record has an invalid shape')
    raw = group['manifest_raw'].encode()
    if json.loads(raw) != manifest or hashlib.sha256(raw).hexdigest() != record['manifest_sha256']:
        raise ForwardRecoveryRefused('export manifest digest mismatch')
    batch = manifest['batch_id']
    if (manifest['owner'] != instance['owner_action_key'] or
            manifest['instance'] != instance or manifest['template'] != template or
            record['batch_id'] != batch):
        raise ForwardRecoveryRefused('export owner/attempt/template mismatch')
    action = sdk['core'].validate_action(record['action'])
    inputs = [entry for entry in action['inputs'] if entry['id'] == 'produced-spool-manifest']
    if (action['action_key'] != record['export_key'] or
            len(inputs) != 1 or inputs[0]['sha256'] != record['manifest_sha256'] or
            action['params']['produced_spool'] != {
                'owner': instance['owner_action_key'], 'batch_id': batch,
                'manifest_sha256': record['manifest_sha256']}):
        raise ForwardRecoveryRefused('export action does not seal this group')
    sdk['produced_spool']._check_receipt(receipt, manifest, record)
    try:
        _filed, descriptors = sdk['produced_output']._load_batch_record(
            queue.root, instance, template, commitments['batches'][batch], batch)
    except (KeyError, ValueError) as exc:
        raise ForwardRecoveryRefused('group has no complete immutable PB descriptor') from exc
    by_path = {item['path']: item for item in descriptors
               if item['artifact_class'] == 'payload'}
    if set(by_path) != {e['destination_path'] for e in manifest['entries']}:
        raise ForwardRecoveryRefused('export and committed descriptor coverage differs')
    for entry in manifest['entries']:
        item = by_path[entry['destination_path']]
        if item['sha256'] != entry['sha256'] or item['bytes'] != entry['bytes']:
            raise ForwardRecoveryRefused('export and committed descriptor bytes differ')
    return manifest['entries']


#: Keys of a bind identity and of a campaign identity that name a recorded
#: run identity: run seals (PQ #1147). Every other key names the data the
#: forward rows were computed from (calibration, probes, partition, roster)
#: and stays a wall in both modes.
_BIND_SEAL_KEYS = frozenset({'source_model', 'producer_source_sha256'})
_CAMPAIGN_SEAL_KEYS = frozenset({
    'plan_sha256', 'prepared_sha256', 'read_manifest_sha256', 'campaign_scope'})


def _differs_outside(recorded, running, seal_keys):
    """Whether two identities differ in a key that is not a run seal."""
    if not isinstance(recorded, dict) or not isinstance(running, dict):
        return recorded != running
    return any(recorded.get(key) != running.get(key)
               for key in set(recorded) | set(running) if key not in seal_keys)


def validate_forward_state(document, *, bind_identity, campaign_identity):
    from .cost_stage_checkpoint import canonical_json_sha256
    if document.get('schema') != SCHEMA:
        raise ForwardRecoveryRefused('unsupported forward recovery schema')
    old = document['original_bind_identity']
    new = dict(bind_identity)
    compatibility = document['implementation_compatibility']
    seal_check('forward recovery implementation', compatibility,
               {'original': old['producer_source_sha256'],
                'recovery': new['producer_source_sha256'],
                'scope': 'forward-identical-memory-only'},
               where='forward recovery capsule',
               refusal=ForwardRecoveryRefused(
                   'implementation compatibility is not explicitly bound'))
    new['producer_source_sha256'] = old['producer_source_sha256']
    campaign = document.get('published_campaign_identity', document['campaign_identity'])
    if old != new or campaign != campaign_identity:
        refusal = ForwardRecoveryRefused(
            'forward source/calibration/execution/campaign identity differs')
        if (_differs_outside(old, new, _BIND_SEAL_KEYS)
                or _differs_outside(campaign, campaign_identity, _CAMPAIGN_SEAL_KEYS)):
            raise refusal
        seal_check('forward source and campaign identity',
                   {'bind': old, 'campaign': campaign},
                   {'bind': new, 'campaign': campaign_identity},
                   where='forward recovery capsule', refusal=refusal)
    if document['session']['run_identity_sha256'] != canonical_json_sha256(
            old, where='exact boundary source'):
        raise ForwardRecoveryRefused('original forward session identity mismatch')
    frontier = document['frontier']
    count = document['n_batches']
    if (type(frontier) is not int or frontier < 1 or type(count) is not int or count < 1
            or count != old['execution_partition']['partition_count']):
        raise ForwardRecoveryRefused('forward coverage geometry differs')
    return frontier, count


def _records(document, entries):
    from .perturbed_x_cache import EXACT_ACTIVATION_SCHEMA
    shape, dtype = document['entry_shape'], document['entry_dtype']
    import math
    import torch
    if (not isinstance(shape, list) or len(shape) not in (3, 4) or
            any(type(n) is not int or n <= 0 for n in shape) or
            dtype not in ('torch.bfloat16', 'torch.float16', 'torch.float32')):
        raise ForwardRecoveryRefused('unsupported exact boundary geometry')
    width = torch.empty((), dtype=getattr(torch, dtype.split('.')[1])).element_size()
    tensor_bytes = math.prod(shape) * width
    first = document.get('first_boundary', 0)
    if type(first) is not int or not 0 <= first <= document['frontier']:
        raise ForwardRecoveryRefused('forward recovery segment range is invalid')
    result = {str(b): {} for b in range(first, document['frontier'] + 1)}
    for entry in entries:
        path = Path(entry['destination_path'])
        match = re.fullmatch(r'boundary-(\d+)-(\d+)-at-(\d+)\.pt', path.name)
        if match is None:
            raise ForwardRecoveryRefused('recovery group contains a non-boundary entry')
        batch, boundary, at = map(int, match.groups())
        if (boundary != at or str(boundary) not in result or
                not 0 <= batch < document['n_batches'] or batch in result[str(boundary)] or
                path.parent.parent.name != document['session']['generation']):
            raise ForwardRecoveryRefused('duplicate or foreign forward coordinate')
        identity = {'session': document['session'], 'slot': f'boundary-{batch}-{boundary}',
                    'kind': 'boundary', 'coordinates':
                    {'batch': batch, 'boundary': boundary, 'probe': None}}
        metadata = {'schema': EXACT_ACTIVATION_SCHEMA, 'identity': identity,
                    'shape': shape, 'dtype': dtype, 'tensor_bytes': tensor_bytes}
        result[str(boundary)][batch] = {'name': path.stem, 'path': str(path),
            'sha256': entry['sha256'], 'file_bytes': entry['bytes'],
            'tensor_bytes': tensor_bytes, 'shape': shape, 'dtype': dtype, 'metadata': metadata}
    if any(set(rows) != set(range(document['n_batches'])) for rows in result.values()):
        raise ForwardRecoveryRefused('incomplete whole-draw forward boundary coverage')
    return {boundary: [rows[i] for i in range(document['n_batches'])]
            for boundary, rows in result.items()}


MAX_CHAIN_SEGMENTS = 16


def _campaign(document):
    return document.get('published_campaign_identity', document['campaign_identity'])


def _prior(document):
    """The sha-pinned capsule this segment's owner resumed from, or None."""
    bound = document.get('imported')
    if bound is None:
        if document.get('first_boundary', 0) != 0:
            raise ForwardRecoveryRefused('a partial forward segment names no imported prefix')
        return None
    prior, _ = _read(bound['path'], bound['sha256'])
    first = document.get('first_boundary')
    if (prior.get('schema') != SCHEMA or type(first) is not int or
            first != prior['frontier'] + 1 or document['frontier'] < first):
        raise ForwardRecoveryRefused('forward recovery segment does not continue its import')
    for field in ('n_batches', 'entry_shape', 'entry_dtype', 'queue_root',
                  'source_campaign_record'):
        if prior.get(field) != document.get(field):
            raise ForwardRecoveryRefused('forward recovery segments differ in ' + field)
    # The imported capsule's recovery implementation is this owner's producer,
    # over the same science and campaign.
    validate_forward_state(prior, bind_identity=document['original_bind_identity'],
                           campaign_identity=_campaign(document))
    return prior


def chain_documents(document):
    """Top segment first; each imported capsule is read and linked once."""
    chain = [document]
    while (prior := _prior(chain[-1])) is not None:
        if len(chain) >= MAX_CHAIN_SEGMENTS:
            raise ForwardRecoveryRefused('forward recovery chain is too long')
        chain.append(prior)
    return chain


def _merge(document, parts):
    """Disjoint segment records that cover boundaries 0..frontier exactly."""
    records = {}
    for rows in parts:
        if set(rows) & set(records):
            raise ForwardRecoveryRefused('forward recovery segments overlap')
        records.update(rows)
    if set(records) != {str(b) for b in range(document['frontier'] + 1)}:
        raise ForwardRecoveryRefused('incomplete whole-draw forward boundary coverage')
    return records


def chain_records(document):
    """Every recovered coordinate of a sha-pinned chain; no PB reads."""
    return _merge(document, [
        _records(segment, [entry for group in segment['groups']
                           for entry in group['manifest']['entries']])
        for segment in chain_documents(document)])


#: Verified chains a Stage B owner attached in this process, keyed by the
#: top capsule's (path, sha256). The digest pins every byte of the chain, so
#: a later owner attaching the same capsule reuses the references instead of
#: re-reading and re-parsing every imported capsule.
_ATTACHED_CHAINS: dict = {}


def attached_chain(capsule):
    """The top segment's identity and every reference of a pinned chain."""
    key = (str(capsule['path']), str(capsule['sha256']))
    cached = _ATTACHED_CHAINS.get(key)
    if cached is None:
        from .joint_adjoint_checkpoints import reference_from_record
        document, _ = _read(capsule['path'], capsule['sha256'])
        if document.get('schema') != SCHEMA:
            raise ForwardRecoveryRefused('unsupported attached forward recovery schema')
        identity = {'session': document['session'], 'frontier': document['frontier'],
                    'owner': document['instance']['owner_action_key'],
                    'attempt': document['instance']['owner_attempt']}
        references = frozenset(reference_from_record(row)
                               for rows in chain_records(document).values() for row in rows)
        while len(_ATTACHED_CHAINS) >= MAX_CHAIN_SEGMENTS:
            _ATTACHED_CHAINS.pop(next(iter(_ATTACHED_CHAINS)))
        cached = _ATTACHED_CHAINS[key] = (identity, references)
    return cached


def _require_imported_by_owner(document, sdk):
    """The segment's owner sealed exactly this import into its own command."""
    action = sdk['core'].validate_action(document['owner_action'])
    bound = document['imported']
    argv = action['params']['command']
    flags = [i for i, arg in enumerate(argv) if arg == '--forward-recovery']
    if (action['action_key'] != document['instance']['owner_action_key'] or
            len(flags) != 1 or argv[flags[0] + 1:flags[0] + 4] != [
                bound['path'], '--forward-recovery-sha256', bound['sha256']]):
        raise ForwardRecoveryRefused('segment owner did not import this capsule')


def _verified_chain_records(document, sdk):
    """Containment, sealed exports and PB descriptors for every segment."""
    parts = []
    for segment in chain_documents(document):
        queue = sdk['pool'].PoolQueue(segment['queue_root'])
        instance = sdk['produced_output'].validate_instance(segment['instance'])
        template = sdk['produced_output'].validate_template(segment['template'])
        require_contained(queue, instance, sdk)
        directory = sdk['produced_output'].instance_dir(queue.root, instance)
        if _read(directory / 'instance.json')[0] != instance:
            raise ForwardRecoveryRefused('original PB instance changed')
        commitments = _read(directory / 'commitments.json')[0]
        if 'imported' in segment:
            _require_imported_by_owner(segment, sdk)
        entries = []
        for group in segment['groups']:
            entries.extend(_checked_group(group, queue=queue, instance=instance,
                template=template, commitments=commitments, sdk=sdk))
        parts.append(_records(segment, entries))
    return _merge(document, parts)


@dataclass(frozen=True)
class ForwardRecovery:
    frontier: int
    n_batches: int
    records: dict
    receipt_binding: dict

    def install(self, storage):
        from .joint_adjoint_checkpoints import reference_from_record
        storage.authorize_forward_inputs([
            reference_from_record(row) for rows in self.records.values() for row in rows])

    def batch_references(self, index):
        from .joint_adjoint_checkpoints import reference_from_record
        return [reference_from_record(self.records[str(b)][index])
                for b in range(self.frontier + 1)]


def require_stateless_profile(runner):
    from .model_profiles.base import ModelProfile
    for name in ('new_forward_pass_state', 'capture_forward_pass_state',
                 'isolated_layer_pass_state'):
        if getattr(type(runner.profile), name) is not getattr(ModelProfile, name):
            raise ForwardRecoveryRefused('forward recovery requires inherited stateless profile hooks')


def await_forward_inputs(references):
    """Wait for declared immutable inputs using PB's ordinary map readiness."""
    from .staged_tier_policy import policy_is_active
    if not references or not policy_is_active():
        return
    import time
    from .residency_map import RANGE_HIT, residency_resolver
    from .residency_shard_reader import await_staged_spans, staged_range_wait_s
    from .staged_lease import (LeaseRefused, stage_cover_is_published,
                               stage_covers_are_published)
    resolver = residency_resolver()
    if resolver is None:
        raise LeaseRefused('forward-recovery-input-map-absent', kind='availability')
    # One cover lookup per poll for the whole window (PQ #997), falling back
    # to one per entry when the batched answer cannot name the missing key.
    verdict = await_staged_spans(resolver,
        [(ref.path, 0, ref.file_bytes, ref.file_bytes) for ref in references],
        deadline=time.monotonic() + staged_range_wait_s(), published=stage_cover_is_published,
        published_batch=stage_covers_are_published)
    if verdict != RANGE_HIT:
        raise LeaseRefused('forward-recovery-input-' + verdict, kind='availability')


def require_published_campaign(document, *, bind_identity, campaign_identity):
    """A capsule that publishes its campaign must name the sealed source campaign.

    Also the check a Stage A chain seed makes on the capsule it borrows
    rows from (``stage_a_chain_seed``, PQ #1016).
    """
    if 'published_campaign_identity' not in document:
        return
    from .joint_forward_campaign import resolve_forward_campaign
    source = document['source_campaign_record']
    record = _read(source['path'], source['sha256'])[0]
    prepared = _read(record['campaign']['prepared_path'],
                     campaign_identity['prepared_sha256'])[0]
    resolved = resolve_forward_campaign(document,
        plan_sha256=campaign_identity['plan_sha256'],
        prepared_sha256=campaign_identity['prepared_sha256'],
        read_manifest_sha256=campaign_identity['read_manifest_sha256'],
        formats_by_qname=prepared['formats_by_qname'],
        calibration_shape=bind_identity['calibration_shape'])
    if resolved != campaign_identity:
        refusal = ForwardRecoveryRefused('recovery campaign is not the sealed source campaign')
        if _differs_outside(resolved, campaign_identity, _CAMPAIGN_SEAL_KEYS):
            raise refusal
        seal_check('recovery campaign', resolved, campaign_identity,
                   where='forward recovery capsule', refusal=refusal)


def load_forward_recovery(bound, *, bind_identity, campaign_identity, runner, storage):
    if bound is None:
        return None
    document, digest = _read(bound['path'], bound['sha256'])
    require_published_campaign(document, bind_identity=bind_identity,
                               campaign_identity=campaign_identity)
    frontier, count = validate_forward_state(document, bind_identity=bind_identity,
                                              campaign_identity=campaign_identity)
    require_stateless_profile(runner)
    if frontier > runner.num_layers:
        raise ForwardRecoveryRefused('forward frontier exceeds model layers')
    sdk = _sdk()
    instance = sdk['produced_output'].validate_instance(document['instance'])
    records = _verified_chain_records(document, sdk)
    binding = {'schema': SCHEMA, 'capsule': {'path': str(bound['path']), 'sha256': digest},
               'frontier': frontier, 'original_session': document['session'],
               'original_owner': instance['owner_action_key'],
               'original_attempt': instance['owner_attempt']}
    result = ForwardRecovery(frontier, count, records, binding)
    result.install(storage)
    return result


def chained_specification(imported, *, spool_directory, owner_request):
    """Derive the next segment from its imported capsule and its owner's spool.

Nothing derived here is trusted: the freezer and the loader re-verify every
field against the owner's sealed request, PB and the imported chain.
"""
    from .cost_stage_checkpoint import canonical_json_sha256
    prior, _ = _read(imported['path'], imported['sha256'])
    action = _read(owner_request)[0]
    manifests = [manifest for manifest in (_read(path)[0] for path in
                 sorted(Path(spool_directory).glob('*/manifest.json')))
                 if manifest['owner'] == action['action_key'] and
                 manifest['batch_id'].startswith('stagea-boundary-')]
    if not manifests:
        raise ForwardRecoveryRefused('owner spool holds no exact boundary groups')
    instance, template = manifests[0]['instance'], manifests[0]['template']
    directories = {Path(entry['destination_path']).parent.parent for manifest in manifests
                   for entry in manifest['entries']}
    if (len(directories) != 1 or any(manifest['instance'] != instance or
                                     manifest['template'] != template for manifest in manifests)):
        raise ForwardRecoveryRefused('owner spool spans more than one generation')
    directory = directories.pop()
    bind = {**prior['original_bind_identity'],
            'producer_source_sha256': prior['implementation_compatibility']['recovery']}
    session = {'generation': directory.name, 'run_identity_sha256':
               canonical_json_sha256(bind, where='chained forward recovery session')}
    if _read(directory / 'generation.json')[0].get('session') != session:
        raise ForwardRecoveryRefused('owner generation is not the imported continuation')
    document = {field: prior[field] for field in (
        'schema', 'queue_root', 'n_batches', 'entry_shape', 'entry_dtype',
        'source_campaign_record', 'published_campaign_identity') if field in prior}
    document.update(instance=instance, template=template, session=session,
                    original_bind_identity=bind, campaign_identity=_campaign(prior),
                    first_boundary=prior['frontier'] + 1, frontier=prior['frontier'] + 1,
                    imported=dict(imported), owner_action=action)
    return document


def build_forward_recovery(*, spool_directory, output, specification=None,
                           bind_current_implementation=False, frontier=None,
                           imported=None, owner_request=None):
    """Freeze complete ACK-backed groups after containment; metadata reads only.

The supplied specification is the operator-reviewed science/implementation
compatibility declaration. It never licenses changing the original files.
With ``imported``, the specification is derived instead, and this segment
continues the imported chain.
"""
    if (specification is None) == (imported is None):
        raise ForwardRecoveryRefused('pass a specification or an imported capsule')
    if imported is None:
        document = dict(specification)
    elif not bind_current_implementation:
        raise ForwardRecoveryRefused('a chained segment binds the current implementation')
    else:
        document = chained_specification(imported, spool_directory=spool_directory,
                                         owner_request=owner_request)
    if frontier is not None:
        if type(frontier) is not int or frontier < 1:
            raise ForwardRecoveryRefused('frontier must be an explicit positive integer')
        document['frontier'] = frontier
    if bind_current_implementation:
        from .aura_cost import _aura_source_sha256
        document['implementation_compatibility'] = {
            'original': document['original_bind_identity']['producer_source_sha256'],
            'recovery': _aura_source_sha256(), 'scope': 'forward-identical-memory-only'}
    prospective = {**document['original_bind_identity'], 'producer_source_sha256':
                   document['implementation_compatibility']['recovery']}
    validate_forward_state(document, bind_identity=prospective,
                           campaign_identity=_campaign(document))
    sdk = _sdk()
    first = document.get('first_boundary', 0)
    groups = []
    for path in sorted(Path(spool_directory).glob('*/manifest.json')):
        manifest, digest = _read(path)
        if manifest['owner'] != document['instance']['owner_action_key']:
            continue
        match = re.match(r'stagea-boundary-b(\d+)-g', manifest['batch_id'])
        if match is None or not first <= int(match[1]) <= document['frontier']:
            continue
        record = _read(path.parent / 'export.json')[0]
        if record['manifest_sha256'] != digest:
            raise ForwardRecoveryRefused('sealed local export manifest changed')
        groups.append({'manifest': manifest, 'manifest_raw': path.read_text(),
                       'record': record, 'receipt': _read(path.parent / 'receipt.json')[0]})
    document['groups'] = groups
    records = _verified_chain_records(document, sdk)
    raw = (json.dumps(document, sort_keys=True, separators=(',', ':')) + '\n').encode()
    with Path(output).open('xb') as handle:
        handle.write(raw)
        handle.flush()
        __import__('os').fsync(handle.fileno())
    import os
    directory_fd = os.open(Path(output).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {'path': str(output), 'sha256': hashlib.sha256(raw).hexdigest(),
            'groups': len(groups), 'segments': len(chain_documents(document)),
            'entries': sum(len(rows) for rows in records.values())}
