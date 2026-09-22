"""Explicit, contained forward recovery from PB-acknowledged exact entries.

This authority does not alter ``working_artifacts_reusable``. Old files retain
their original session and producer; a fresh action reads them as sealed inputs.
Only stateless profiles can reconstruct the per-batch forward state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat

SCHEMA = 'prismaquant.joint_forward_recovery.v1'


class ForwardRecoveryRefused(RuntimeError):
    pass


def _read(path, expected=None):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 128 << 20:
        raise ForwardRecoveryRefused('recovery proof is not a bounded regular file')
    raw = path.read_bytes()
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ForwardRecoveryRefused('recovery proof changed during read')
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ForwardRecoveryRefused('recovery proof SHA256 mismatch')
    return json.loads(raw), digest


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


def validate_forward_state(document, *, bind_identity, campaign_identity):
    from .cost_stage_checkpoint import canonical_json_sha256
    if document.get('schema') != SCHEMA:
        raise ForwardRecoveryRefused('unsupported forward recovery schema')
    old = document['original_bind_identity']
    new = dict(bind_identity)
    compatibility = document['implementation_compatibility']
    if compatibility != {'original': old['producer_source_sha256'],
                          'recovery': new['producer_source_sha256'],
                          'scope': 'forward-identical-memory-only'}:
        raise ForwardRecoveryRefused('implementation compatibility is not explicitly bound')
    new['producer_source_sha256'] = old['producer_source_sha256']
    if old != new or document.get('published_campaign_identity', document['campaign_identity']) != campaign_identity:
        raise ForwardRecoveryRefused('forward source/calibration/execution/campaign identity differs')
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
    result = {str(b): {} for b in range(document['frontier'] + 1)}
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
    from .staged_lease import LeaseRefused, stage_cover_is_published
    resolver = residency_resolver()
    if resolver is None:
        raise LeaseRefused('forward-recovery-input-map-absent', kind='availability')
    verdict = await_staged_spans(resolver,
        [(ref.path, 0, ref.file_bytes, ref.file_bytes) for ref in references],
        deadline=time.monotonic() + staged_range_wait_s(), published=stage_cover_is_published)
    if verdict != RANGE_HIT:
        raise LeaseRefused('forward-recovery-input-' + verdict, kind='availability')


def load_forward_recovery(bound, *, bind_identity, campaign_identity, runner, storage):
    if bound is None:
        return None
    document, digest = _read(bound['path'], bound['sha256'])
    if 'published_campaign_identity' in document:
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
            raise ForwardRecoveryRefused('recovery campaign is not the sealed source campaign')
    frontier, count = validate_forward_state(document, bind_identity=bind_identity,
                                              campaign_identity=campaign_identity)
    require_stateless_profile(runner)
    if frontier > runner.num_layers:
        raise ForwardRecoveryRefused('forward frontier exceeds model layers')
    sdk = _sdk()
    queue = sdk['pool'].PoolQueue(document['queue_root'])
    instance = sdk['produced_output'].validate_instance(document['instance'])
    template = sdk['produced_output'].validate_template(document['template'])
    require_contained(queue, instance, sdk)
    directory = sdk['produced_output'].instance_dir(queue.root, instance)
    if _read(directory / 'instance.json')[0] != instance:
        raise ForwardRecoveryRefused('original PB instance changed')
    commitments = _read(directory / 'commitments.json')[0]
    entries = []
    for group in document['groups']:
        entries.extend(_checked_group(group, queue=queue, instance=instance,
            template=template, commitments=commitments, sdk=sdk))
    records = _records(document, entries)
    binding = {'schema': SCHEMA, 'capsule': {'path': str(bound['path']), 'sha256': digest},
               'frontier': frontier, 'original_session': document['session'],
               'original_owner': instance['owner_action_key'],
               'original_attempt': instance['owner_attempt']}
    result = ForwardRecovery(frontier, count, records, binding)
    result.install(storage)
    return result


def build_forward_recovery(*, specification, spool_directory, output,
                           bind_current_implementation=False, frontier=None):
    """Freeze complete ACK-backed groups after containment; metadata reads only.

The supplied specification is the operator-reviewed science/implementation
compatibility declaration. It never licenses changing the original files.
"""
    document = dict(specification)
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
                           campaign_identity=document.get('published_campaign_identity', document['campaign_identity']))
    sdk = _sdk()
    groups = []
    for path in sorted(Path(spool_directory).glob('*/manifest.json')):
        manifest, digest = _read(path)
        if manifest['owner'] != document['instance']['owner_action_key']:
            continue
        match = re.match(r'stagea-boundary-b(\d+)-g', manifest['batch_id'])
        if match is None or int(match[1]) > document['frontier']:
            continue
        record = _read(path.parent / 'export.json')[0]
        if record['manifest_sha256'] != digest:
            raise ForwardRecoveryRefused('sealed local export manifest changed')
        groups.append({'manifest': manifest, 'manifest_raw': path.read_text(),
                       'record': record, 'receipt': _read(path.parent / 'receipt.json')[0]})
    document['groups'] = groups
    queue = sdk['pool'].PoolQueue(document['queue_root'])
    instance = sdk['produced_output'].validate_instance(document['instance'])
    require_contained(queue, instance, sdk)
    commitments = _read(sdk['produced_output'].instance_dir(queue.root, instance) /
                        'commitments.json')[0]
    entries = []
    for group in groups:
        entries.extend(_checked_group(group, queue=queue, instance=instance,
            template=document['template'], commitments=commitments, sdk=sdk))
    _records(document, entries)
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
            'groups': len(groups), 'entries': len(entries)}
