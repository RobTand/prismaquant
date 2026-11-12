"""Finite recovery chains of exact, selected native batch publications.

Successful results and explicitly declared failed checkpoints remain distinct.
Scientific/session/partition/campaign joins retain their independent gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .dev_mode import seal_check
from .digests import DIRECT_ASCII_STRICT, bytes_sha256hex
from .schemas import strict_json_loads
from .staged_manifest import read_document
from .stage_a_produced_output import (read_produced_selection, read_bound_document,
                                      publish_document, MAX_ENTRIES, MAX_PAYLOAD_BYTES)

SCHEMA = 'prismaquant.joint_forward_recovery.v2'
PROOF_READ_MAX_BYTES = 64 << 20
MAX_CHAIN_BYTES = 128 << 20
MAX_CHAIN_SEGMENTS = 16
_STATE_KEYS = frozenset(('original_bind_identity', 'campaign_identity',
    'implementation_compatibility', 'n_batches', 'entry_shape', 'entry_dtype'))
_OPTIONAL_STATE_KEYS = frozenset(('first_boundary', 'imported', 'source_campaign_record',
                                 'published_campaign_identity'))
_CAPSULE_KEYS = _STATE_KEYS | frozenset(('schema', 'selection', 'checkpoint_exit',
    'session', 'frontier', 'groups'))


class ForwardRecoveryRefused(RuntimeError):
    pass


def capsule_byte_limit(document):
    return PROOF_READ_MAX_BYTES


def _read(path, expected=None):
    raw, _identity = read_document(path, PROOF_READ_MAX_BYTES)
    sha = bytes_sha256hex(raw)
    if expected is not None and sha != expected:
        raise ForwardRecoveryRefused('recovery proof SHA256 mismatch')
    def invalid(key):
        return ForwardRecoveryRefused(f'invalid recovery JSON: {key}')
    return strict_json_loads(raw, duplicate=invalid, constant=invalid), sha

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
    if (type(frontier) is not int or not 1 <= frontier < MAX_ENTRIES
            or type(count) is not int or not 1 <= count <= MAX_ENTRIES
            or (frontier + 1) * count > MAX_ENTRIES
            or count != old['execution_partition']['partition_count']):
        raise ForwardRecoveryRefused('forward coverage geometry differs')
    return frontier, count


def _records(document, entries):
    from .exact_activation_contract import EXACT_ACTIVATION_SCHEMA
    shape, dtype = document['entry_shape'], document['entry_dtype']
    import math
    if (not isinstance(shape, list) or len(shape) not in (3, 4) or
            any(type(n) is not int or n <= 0 for n in shape) or
            dtype not in ('torch.bfloat16', 'torch.float16', 'torch.float32')):
        raise ForwardRecoveryRefused('unsupported exact boundary geometry')
    width = {'torch.bfloat16': 2, 'torch.float16': 2, 'torch.float32': 4}[dtype]
    tensor_bytes = math.prod(shape) * width
    frontier, count = document['frontier'], document['n_batches']
    if (type(frontier) is not int or not 1 <= frontier < MAX_ENTRIES
            or type(count) is not int or not 1 <= count <= MAX_ENTRIES
            or (frontier + 1) * count > MAX_ENTRIES):
        raise ForwardRecoveryRefused('finite whole-draw geometry required')
    first = document.get('first_boundary', 0)
    if type(first) is not int or not 0 <= first <= document['frontier']:
        raise ForwardRecoveryRefused('forward recovery segment range is invalid')
    result = {str(b): {} for b in range(first, document['frontier'] + 1)}
    for entry in entries:
        path = Path(entry['path'])
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


def _campaign(document):
    return document.get('published_campaign_identity', document['campaign_identity'])


def chain_documents(document):
    chain, seen, aggregate = [], set(), 0
    while True:
        keys = set(document)
        if (document.get('schema') != SCHEMA or not _CAPSULE_KEYS <= keys
                or keys - _CAPSULE_KEYS - _OPTIONAL_STATE_KEYS):
            raise ForwardRecoveryRefused('unsupported or malformed native recovery capsule')
        aggregate += len(DIRECT_ASCII_STRICT.encoded(document))
        if aggregate > MAX_CHAIN_BYTES or len(chain) >= MAX_CHAIN_SEGMENTS:
            raise ForwardRecoveryRefused('finite recovery chain limit exceeded')
        chain.append(document)
        bound = document.get('imported')
        if bound is None:
            if document.get('first_boundary', 0) != 0:
                raise ForwardRecoveryRefused('partial forward segment has no imported prefix')
            break
        if not isinstance(bound, dict) or set(bound) != {'path', 'sha256'}:
            raise ForwardRecoveryRefused('exact imported capsule binding required')
        key = (bound['path'], bound['sha256'])
        if key in seen:
            raise ForwardRecoveryRefused('cyclic recovery chain')
        seen.add(key)
        prior, _ = _read(bound['path'], bound['sha256'])
        if (document.get('first_boundary') != prior['frontier'] + 1
                or document['frontier'] < document['first_boundary']):
            raise ForwardRecoveryRefused('forward recovery segment does not continue import')
        for field in ('n_batches', 'entry_shape', 'entry_dtype', 'source_campaign_record'):
            if prior.get(field) != document.get(field):
                raise ForwardRecoveryRefused('forward recovery segments differ in ' + field)
        validate_forward_state(prior, bind_identity=document['original_bind_identity'],
                               campaign_identity=_campaign(document))
        document = prior
    return chain


def _merge(document, parts):
    records = {}
    for part in parts:
        if set(part).intersection(records):
            raise ForwardRecoveryRefused('overlapping forward segments')
        records.update(part)
    if set(records) != {str(b) for b in range(document['frontier'] + 1)}:
        raise ForwardRecoveryRefused('incomplete whole-draw forward boundary coverage')
    return records


def _verified_chain_records(document):
    parts, entries_seen, bytes_seen, all_entries = [], 0, 0, []
    for segment in chain_documents(document):
        verified = read_produced_selection(segment['selection'], checkpoint_exit=segment['checkpoint_exit'],
                                          verify_payloads=False)
        if verified.index['session'] != segment['session']:
            raise ForwardRecoveryRefused('selected publication session differs')
        state_binding = verified.declaration['inputs'].get('forward_state')
        if state_binding is None:
            raise ForwardRecoveryRefused('producer launch declares no scientific forward state')
        state = read_bound_document(state_binding)
        expected = {key: segment[key] for key in _STATE_KEYS | _OPTIONAL_STATE_KEYS if key in segment}
        if state != expected:
            raise ForwardRecoveryRefused('capsule scientific state differs from launch-bound input')
        if 'imported' in segment:
            argv, bound = verified.selected.launch['argv'], segment['imported']
            for flag, value in (('--forward-recovery', bound['path']),
                                ('--forward-recovery-sha256', bound['sha256'])):
                positions = [i for i, argument in enumerate(argv) if argument == flag]
                if len(positions) != 1 or argv[positions[0] + 1:positions[0] + 2] != [value]:
                    raise ForwardRecoveryRefused('segment owner did not import this exact capsule')
        bindings = segment['groups']
        if not isinstance(bindings, list) or not 0 < len(bindings) <= len(verified.groups):
            raise ForwardRecoveryRefused('finite selected group subset required')
        def coordinate(binding):
            if not isinstance(binding, dict) or set(binding) != {'path', 'sha256', 'bytes'}:
                raise ForwardRecoveryRefused('exact immutable group binding required')
            return binding['path'], binding['sha256'], binding['bytes']
        roster = {coordinate(binding): group for binding, group in
                  zip(verified.index['groups'], verified.groups)}
        used, entries = set(), []
        for binding in bindings:
            key = coordinate(binding)
            if key not in roster or key in used:
                raise ForwardRecoveryRefused('capsule names a foreign or duplicate group')
            used.add(key)
            entries.extend(roster[key]['entries'])
        entries_seen += len(entries)
        bytes_seen += sum(row['bytes'] for row in entries)
        if entries_seen > MAX_ENTRIES or bytes_seen > MAX_PAYLOAD_BYTES:
            raise ForwardRecoveryRefused('aggregate recovery payload exceeds bound')
        all_entries.extend(entries)
        parts.append(_records(segment, entries))
    from .native_results import verify_result_artifact
    from .stage_a_produced_output import MAX_FILE_BYTES
    for entry in all_entries:
        verify_result_artifact(entry, max_bytes=MAX_FILE_BYTES)
    return _merge(document, parts)


def chain_records(document):
    return _verified_chain_records(document)


def attached_chain(capsule):
    from .joint_adjoint_checkpoints import reference_from_record
    document, _ = _read(capsule['path'], capsule['sha256'])
    records = _verified_chain_records(document)
    identity = {'session': document['session'], 'frontier': document['frontier'],
                'producer': document['selection']['job_id']}
    references = frozenset(reference_from_record(row)
        for rows in records.values() for row in rows)
    return identity, references


@dataclass(frozen=True)
class ForwardRecovery:
    frontier: int
    n_batches: int
    records: dict
    receipt_binding: dict

    def install(self, storage):
        from .joint_adjoint_checkpoints import reference_from_record
        storage.authorize_forward_inputs([reference_from_record(row)
            for rows in self.records.values() for row in rows])

    def batch_references(self, index):
        from .joint_adjoint_checkpoints import reference_from_record
        return [reference_from_record(self.records[str(b)][index]) for b in range(self.frontier + 1)]


def require_stateless_profile(runner):
    from .model_profiles.base import ModelProfile
    for name in ('new_forward_pass_state', 'capture_forward_pass_state', 'isolated_layer_pass_state'):
        if getattr(type(runner.profile), name) is not getattr(ModelProfile, name):
            raise ForwardRecoveryRefused('forward recovery requires inherited stateless profile hooks')


def await_forward_inputs(references):
    """Require immutable declared coverage; native warm dependencies precede us."""
    from .staged_tier_policy import policy_is_active
    if not references or not policy_is_active():
        return
    from .residency_map import residency_resolver
    from .staged_lease import LeaseRefused
    resolver = residency_resolver()
    if resolver is None:
        raise LeaseRefused('forward-recovery-input-map-absent', kind='availability')
    for ref in references:
        if resolver.staged_read(ref.path, expected_sha256=ref.sha256) is None:
            raise LeaseRefused('forward-recovery-input-unpublished', kind='availability')


def require_published_campaign(document, *, bind_identity, campaign_identity):
    if 'published_campaign_identity' not in document:
        return
    from .joint_forward_campaign import resolve_forward_campaign
    source = document['source_campaign_record']
    record = _read(source['path'], source['sha256'])[0]
    prepared = _read(record['campaign']['prepared_path'], campaign_identity['prepared_sha256'])[0]
    resolved = resolve_forward_campaign(document,
        plan_sha256=campaign_identity['plan_sha256'], prepared_sha256=campaign_identity['prepared_sha256'],
        read_manifest_sha256=campaign_identity['read_manifest_sha256'],
        formats_by_qname=prepared['formats_by_qname'], calibration_shape=bind_identity['calibration_shape'])
    if resolved != campaign_identity:
        refusal = ForwardRecoveryRefused('recovery campaign is not the sealed source campaign')
        if _differs_outside(resolved, campaign_identity, _CAMPAIGN_SEAL_KEYS):
            raise refusal
        seal_check('recovery campaign', resolved, campaign_identity,
                   where='forward recovery capsule', refusal=refusal)


def load_forward_recovery(bound, *, bind_identity, campaign_identity, runner, storage):
    if bound is None:
        return None
    document, sha = _read(bound['path'], bound['sha256'])
    require_published_campaign(document, bind_identity=bind_identity, campaign_identity=campaign_identity)
    frontier, count = validate_forward_state(document, bind_identity=bind_identity, campaign_identity=campaign_identity)
    require_stateless_profile(runner)
    if frontier > runner.num_layers:
        raise ForwardRecoveryRefused('forward frontier exceeds model layers')
    records = _verified_chain_records(document)
    binding = {'schema': SCHEMA, 'capsule': {'path': str(bound['path']), 'sha256': sha},
        'frontier': frontier, 'original_session': document['session'],
        'producer': document['selection']['job_id'], 'checkpoint_exit': document['checkpoint_exit']}
    result = ForwardRecovery(frontier, count, records, binding)
    result.install(storage)
    return result


def build_forward_recovery(*, selection, output, specification, frontier=None,
                           imported=None, bind_current_implementation=False,
                           checkpoint_exit=None):
    state = dict(specification)
    if not _STATE_KEYS <= set(state) or set(state) - _STATE_KEYS - _OPTIONAL_STATE_KEYS:
        raise ForwardRecoveryRefused('finite exact scientific specification required')
    if imported is not None and state.get('imported') != imported:
        raise ForwardRecoveryRefused('import differs from explicit scientific specification')
    if bind_current_implementation:
        from .aura_provenance import producer_package_sha256
        if state['implementation_compatibility']['recovery'] != producer_package_sha256():
            raise ForwardRecoveryRefused('current implementation differs from explicit compatibility')
    verified = read_produced_selection(selection, checkpoint_exit=checkpoint_exit)
    first = state.get('first_boundary', 0)
    if type(frontier) is not int or not 1 <= frontier <= MAX_ENTRIES:
        raise ForwardRecoveryRefused('frontier must be an explicit bounded positive integer')
    groups = []
    for binding, group in zip(verified.index['groups'], verified.groups):
        if all((match := re.fullmatch(r'boundary-(\d+)-(\d+)-at-(\d+)\.pt', Path(row['path']).name))
               and first <= int(match[2]) <= frontier for row in group['entries']):
            groups.append(binding)
    document = {'schema': SCHEMA, **state, 'selection': selection,
        'checkpoint_exit': checkpoint_exit, 'session': verified.index['session'],
        'frontier': frontier, 'groups': groups}
    prospective = {**state['original_bind_identity'],
        'producer_source_sha256': state['implementation_compatibility']['recovery']}
    validate_forward_state(document, bind_identity=prospective, campaign_identity=_campaign(document))
    records = _verified_chain_records(document)
    binding = publish_document(Path(output), document, limit=PROOF_READ_MAX_BYTES)
    return {**binding, 'groups': len(groups), 'segments': len(chain_documents(document)),
            'entries': sum(len(rows) for rows in records.values())}
