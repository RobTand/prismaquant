"""Launch-bound application batches. Flux owns execution; IO owns bytes.

Group durability permits a producer's own bounded readback. Independent reuse
requires selected terminal native authority, including explicit failed-checkpoint
policy when applicable. There is no transport backend or persistent capacity ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading

from . import native_results
from .digests import DIRECT_ASCII_STRICT, bytes_sha256hex, text_sha256hex
from .file_identity import file_stat_signature
from .schemas import strict_json_loads
from .staged_manifest import (read_document, publish_staged_json, open_regular,
                              path as checked_path, digest, integer)

DECLARATION_SCHEMA = 'prismaquant.produced_declaration.v1'
BATCH_SCHEMA = 'prismaquant.produced_batch.v1'
INDEX_SCHEMA = 'prismaquant.produced_index.v1'
DESCRIPTOR_SCHEMA = 'prismaquant.produced_descriptor.v1'
BOUNDARY_TEMPLATE_ID = 'pq-stagea-boundary-entries-v1'
BOUNDARY_SLOT = 'boundary_entries'
HEADER_ENVELOPE_BYTES = 65536
MAX_DOCUMENT_BYTES = 16 << 20
MAX_GROUPS = 16384
MAX_GROUP_ENTRIES = 1024
MAX_ENTRIES = 65536
MAX_FILE_BYTES = 4 << 30
MAX_PAYLOAD_BYTES = 64 << 40
DECLARATION_FLAG = '--produced-declaration'
DECLARATION_SHA_FLAG = '--produced-declaration-sha256'
_DECLARATION_KEYS = frozenset(('schema', 'template_id', 'output_prefix', 'slot',
    'durable_maxima', 'group_size', 'max_entry_tensor_bytes', 'window_groups',
    'write_only', 'spool', 'max_groups', 'inputs', 'checkpoint_exit'))
_DESCRIPTOR_KEYS = frozenset(('schema', 'slot', 'artifact_class', 'path', 'bytes',
    'sha256', 'producer_generation', 'file_identity'))


class BoundaryProducedBindingError(RuntimeError):
    pass


class BoundaryProducedPrewriteRefused(RuntimeError):
    def __init__(self, *, batch_id, refusal):
        self.batch_id, self.refusal = batch_id, refusal
        super().__init__(f'prewrite refused before any byte for {batch_id}: {refusal}')


class BoundaryProducedPublicationFailed(RuntimeError):
    pass


class BoundaryMaterializationIncomplete(RuntimeError):
    pass


def _fields(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise BoundaryProducedBindingError(f'{label}: exact fields required')
    return value


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,256}', value) or value in ('.', '..'):
        raise BoundaryProducedBindingError('bounded bare identifier required')
    return value


def _limit(value, maximum, *, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value <= maximum:
        raise BoundaryProducedBindingError('finite integer limit required')
    return value


def _decode(raw):
    def invalid(key):
        return BoundaryProducedBindingError(f'invalid produced JSON: {key}')
    return strict_json_loads(raw, duplicate=invalid, constant=invalid)


def read_bound_document(binding, *, limit=MAX_DOCUMENT_BYTES):
    _fields(binding, ('path', 'sha256', 'bytes'), 'document binding')
    checked_path(binding['path']); digest(binding['sha256'])
    _limit(binding['bytes'], limit)
    raw, first = read_document(binding['path'], limit)
    if (len(raw) != binding['bytes'] or bytes_sha256hex(raw) != binding['sha256']
            or file_stat_signature(os.stat(binding['path'], follow_symlinks=False)) != first):
        raise BoundaryProducedBindingError('immutable document changed')
    return _decode(raw)


def publish_document(filename, document, *, limit=MAX_DOCUMENT_BYTES):
    raw = DIRECT_ASCII_STRICT.encoded(document) + b'\n'
    _limit(limit, 64 << 20)
    _limit(len(raw), limit)
    filename = Path(checked_path(str(filename)))
    # The shared publisher provides exclusive durable publication. Replays may
    # observe an identical document; no observation overwrites existing bytes.
    try:
        sha = publish_staged_json(filename, document)
    except FileExistsError:
        prior, _ = read_document(filename, limit)
        if prior != raw:
            raise BoundaryProducedPublicationFailed('immutable publication conflicts')
        sha = bytes_sha256hex(prior)
    return {'path': str(filename), 'sha256': sha, 'bytes': len(raw)}


def boundary_group_ceiling_bytes(*, group_size, max_entry_tensor_bytes,
                                 header_envelope_bytes=HEADER_ENVELOPE_BYTES):
    for value in (group_size, max_entry_tensor_bytes, header_envelope_bytes):
        _limit(value, MAX_PAYLOAD_BYTES)
    return group_size * (max_entry_tensor_bytes + header_envelope_bytes)


def validate_declaration(value):
    _fields(value, _DECLARATION_KEYS, 'declaration')
    if value['schema'] != DECLARATION_SCHEMA or type(value['write_only']) is not bool:
        raise BoundaryProducedBindingError('unsupported produced declaration')
    for field in ('template_id', 'slot'):
        _name(value[field])
    checked_path(value['output_prefix'])
    _fields(value['durable_maxima'], ('payload_max_bytes', 'checkpoint_max_bytes', 'temp_max_bytes'), 'maxima')
    for name, maximum in value['durable_maxima'].items():
        _limit(maximum, MAX_PAYLOAD_BYTES, zero=name != 'payload_max_bytes')
    _limit(value['group_size'], MAX_GROUP_ENTRIES)
    _limit(value['max_entry_tensor_bytes'], MAX_FILE_BYTES - HEADER_ENVELOPE_BYTES)
    _limit(value['max_groups'], MAX_GROUPS)
    _limit(value['window_groups'], MAX_GROUPS, zero=value['write_only'])
    if value['write_only'] and value['window_groups'] != 0:
        raise BoundaryProducedBindingError('write-only declaration has no read window')
    if not value['write_only'] and value['window_groups'] < 2:
        raise BoundaryProducedBindingError('readback needs at least two bounded groups')
    if value['spool'] is not None:
        _fields(value['spool'], ('root', 'max_bytes'), 'spool')
        checked_path(value['spool']['root'])
        _limit(value['spool']['max_bytes'], MAX_PAYLOAD_BYTES)
    inputs = value['inputs']
    if not isinstance(inputs, dict) or len(inputs) > 63 or 'declaration' in inputs:
        raise BoundaryProducedBindingError('finite distinct input roles required')
    total, paths = 0, set()
    for role, binding in inputs.items():
        _name(role)
        _fields(binding, ('path', 'sha256', 'bytes'), 'input')
        checked_path(binding['path']); digest(binding['sha256'])
        total += _limit(binding['bytes'], 1 << 30)
        if binding['path'] in paths:
            raise BoundaryProducedBindingError('duplicate input path')
        paths.add(binding['path'])
    _limit(total, 1 << 30, zero=True)
    if value['checkpoint_exit'] is not None:
        _limit(value['checkpoint_exit'], 124)
    return value


def build_boundary_template(*, output_prefix, artifact_max_bytes, group_size,
        max_entry_tensor_bytes, checkpoint_max_bytes=None, concurrent_groups=2,
        template_id=BOUNDARY_TEMPLATE_ID, write_only=False, spool=None,
        max_groups=MAX_GROUPS, inputs=None):
    return validate_declaration({'schema': DECLARATION_SCHEMA,
        'template_id': template_id, 'output_prefix': str(output_prefix),
        'slot': BOUNDARY_SLOT,
        'durable_maxima': {'payload_max_bytes': artifact_max_bytes,
            'checkpoint_max_bytes': artifact_max_bytes if checkpoint_max_bytes is None else checkpoint_max_bytes,
            'temp_max_bytes': artifact_max_bytes},
        'group_size': group_size, 'max_entry_tensor_bytes': max_entry_tensor_bytes,
        'window_groups': 0 if write_only else concurrent_groups, 'write_only': write_only,
        'spool': spool, 'max_groups': max_groups, 'inputs': {} if inputs is None else inputs,
        'checkpoint_exit': None})


def _launch_binding(producer, binding):
    argv = producer['argv']
    for flag, expected in ((DECLARATION_FLAG, binding['path']), (DECLARATION_SHA_FLAG, binding['sha256'])):
        positions = [i for i, arg in enumerate(argv) if arg == flag]
        if len(positions) != 1 or argv[positions[0] + 1:positions[0] + 2] != [expected]:
            raise BoundaryProducedBindingError('declaration is not bound by actual launch intent')


def current_declaration(*, producer=None):
    """Read the declaration named by this authenticated current native argv."""
    producer = native_results.execution_context() if producer is None else producer
    argv, values = producer['argv'], {}
    for flag in (DECLARATION_FLAG, DECLARATION_SHA_FLAG):
        positions = [i for i, argument in enumerate(argv) if argument == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            raise BoundaryProducedBindingError('current native launch has no exact declaration binding')
        values[flag] = argv[positions[0] + 1]
    raw, _ = read_document(values[DECLARATION_FLAG], MAX_DOCUMENT_BYTES)
    binding = {'path': values[DECLARATION_FLAG], 'sha256': values[DECLARATION_SHA_FLAG], 'bytes': len(raw)}
    return validate_declaration(read_bound_document(binding)), binding


def declaration_launch_arguments(filename):
    """Bind validated immutable application intent to a native argv."""
    raw, _ = read_document(str(filename), MAX_DOCUMENT_BYTES)
    validate_declaration(_decode(raw))
    return [DECLARATION_FLAG, str(checked_path(str(filename))),
            DECLARATION_SHA_FLAG, bytes_sha256hex(raw)]


def add_declaration_arguments(parser):
    parser.add_argument(DECLARATION_FLAG, type=Path)
    parser.add_argument(DECLARATION_SHA_FLAG)


def _session(value):
    _fields(value, ('generation', 'run_identity_sha256'), 'session')
    _name(value['generation']); digest(value['run_identity_sha256'])
    return value


def _file_identity(filename):
    fd = open_regular(str(filename))
    try:
        return list(file_stat_signature(os.fstat(fd)))
    finally:
        os.close(fd)


def validate_descriptor(row, declaration):
    _fields(row, _DESCRIPTOR_KEYS, 'descriptor')
    if (row['schema'] != DESCRIPTOR_SCHEMA or row['slot'] != declaration['slot']
            or row['artifact_class'] not in ('payload', 'checkpoint')):
        raise BoundaryProducedBindingError('foreign descriptor slot or class')
    _name(row['producer_generation']); checked_path(row['path']); digest(row['sha256'])
    _limit(row['bytes'], MAX_FILE_BYTES)
    if not Path(row['path']).is_relative_to(declaration['output_prefix']):
        raise BoundaryProducedBindingError('descriptor escapes declared output')
    identity = row['file_identity']
    if (not isinstance(identity, list) or len(identity) != 5
            or any(type(n) is not int or n < 0 for n in identity) or identity[2] != row['bytes']):
        raise BoundaryProducedBindingError('invalid descriptor file identity')
    return row


class BoundaryProducedPublication:
    DEFAULT_SLOT = BOUNDARY_SLOT
    REQUIRES_READBACK = False

    def __init__(self, *, declaration, binding, producer):
        self.declaration = validate_declaration(read_bound_document(binding))
        if self.declaration != declaration:
            raise BoundaryProducedBindingError('declaration differs from immutable launch input')
        self.binding = dict(binding)
        _launch_binding(producer, binding)
        if declaration['slot'] != self.DEFAULT_SLOT or (self.REQUIRES_READBACK and declaration['write_only']):
            raise BoundaryProducedBindingError('publication slot/readback contract differs')
        self.producer = producer
        self.session = None
        self._lock = threading.RLock()
        self._prewrites, self._groups, self._released = {}, {}, set()
        self._directory = Path(producer['record_dir']) / 'produced'
        self._directory.mkdir(exist_ok=True)
        (self._directory / 'batches').mkdir(exist_ok=True)

    @classmethod
    def bind_from_admitted_owner(cls, *, declaration_path=None, declaration_sha256=None):
        try:
            producer = native_results.execution_context()
            if declaration_path is None and declaration_sha256 is None:
                declaration, binding = current_declaration(producer=producer)
                return cls(declaration=declaration, binding=binding, producer=producer)
            raw, _ = read_document(declaration_path, MAX_DOCUMENT_BYTES)
            binding = {'path': str(declaration_path), 'sha256': declaration_sha256, 'bytes': len(raw)}
            declaration = read_bound_document(binding)
            return cls(declaration=declaration, binding=binding,
                       producer=producer)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise BoundaryProducedBindingError(f'cannot bind native produced owner: {exc}') from exc

    @property
    def output_prefix(self):
        return self.declaration['output_prefix']

    @property
    def write_only(self):
        return self.declaration['write_only']

    @property
    def generation(self):
        return self.producer['launch_sha256'][:16]

    def contains(self, filename):
        try:
            checked_path(str(filename))
            root = Path(self.output_prefix)
            target = Path(filename)
            return target.is_relative_to(root) and target.resolve().is_relative_to(root.resolve()) and all(
                not parent.is_symlink() for parent in (target, *target.parents))
        except (OSError, ValueError, TypeError):
            return False

    def bind_session(self, session):
        _session(session)
        if self.session is not None and self.session != session:
            raise BoundaryProducedBindingError('publication session changed')
        self.session = dict(session)

    def durable_maxima(self):
        return dict(self.declaration['durable_maxima'])

    @staticmethod
    def group_ceiling_bytes(*, entries, max_entry_tensor_bytes,
                            header_envelope_bytes=HEADER_ENVELOPE_BYTES):
        return boundary_group_ceiling_bytes(group_size=entries,
            max_entry_tensor_bytes=max_entry_tensor_bytes,
            header_envelope_bytes=header_envelope_bytes)

    def batch_id_for(self, *, boundary_index, group_index, probe_index=None, kind='boundary'):
        for number in (boundary_index, group_index):
            _limit(number, MAX_ENTRIES, zero=True)
        if probe_index is not None:
            _limit(probe_index, MAX_ENTRIES, zero=True)
        _name(kind)
        coordinate = [kind, boundary_index, group_index, probe_index, self.producer['launch_sha256']]
        readable = f'b{boundary_index}' + ('' if probe_index is None else f'p{probe_index}')
        return f'stagea-{kind}-{readable}-g{group_index}-{self.generation}-{DIRECT_ASCII_STRICT.sha256(coordinate)[:12]}'

    def require_prewrite(self, *, batch_id, payload_ceiling_bytes, paths,
                         temp_ceiling_bytes=0, checkpoint_ceiling_bytes=0):
        _name(batch_id)
        ceilings = {'payload_max_bytes': payload_ceiling_bytes,
                    'checkpoint_max_bytes': checkpoint_ceiling_bytes, 'temp_max_bytes': temp_ceiling_bytes}
        for value in ceilings.values():
            _limit(value, MAX_PAYLOAD_BYTES, zero=True)
        if (not isinstance(paths, list) or not 0 < len(paths) <= 2 * MAX_GROUP_ENTRIES
                or len(set(paths)) != len(paths) or any(not self.contains(p) for p in paths)):
            raise BoundaryProducedPrewriteRefused(batch_id=batch_id, refusal='invalid owned path roster')
        request = {'ceilings': ceilings, 'paths': list(paths)}
        with self._lock:
            prior = self._prewrites.get(batch_id)
            if prior is not None:
                if prior != request or batch_id in self._released:
                    raise BoundaryProducedPrewriteRefused(batch_id=batch_id, refusal='changed or released prewrite')
                return {'ok': True}
            if len(self._prewrites) >= self.declaration['max_groups']:
                raise BoundaryProducedPrewriteRefused(batch_id=batch_id, refusal='group count exceeds bound')
            for name, maximum in self.durable_maxima().items():
                held = sum(row['ceilings'][name] for key, row in self._prewrites.items() if key not in self._released)
                if held + ceilings[name] > maximum:
                    raise BoundaryProducedPrewriteRefused(batch_id=batch_id, refusal=f'{name} ceiling exceeded')
            for filename in paths:
                if os.path.lexists(filename):
                    raise BoundaryProducedPrewriteRefused(batch_id=batch_id, refusal='prewrite path already occupied')
            existing = {p for row in self._prewrites.values() for p in row['paths']}
            if existing.intersection(paths):
                raise BoundaryProducedPrewriteRefused(batch_id=batch_id, refusal='path already owned by another group')
            self._prewrites[batch_id] = request
        return {'ok': True}

    def abort_prewrite(self, *, batch_id):
        with self._lock:
            row = self._prewrites[batch_id]
            if any(os.path.lexists(p) for p in row['paths']):
                return {'ok': False, 'refusal': 'owned paths remain'}
            self._released.add(batch_id)
            return {'ok': True}

    def extend_prewrite(self, *, batch_id, checkpoint_ceiling_bytes, paths,
                        temp_ceiling_bytes):
        """Grow an active checkpoint reservation before any new file is written."""
        _limit(checkpoint_ceiling_bytes, MAX_PAYLOAD_BYTES)
        _limit(temp_ceiling_bytes, MAX_PAYLOAD_BYTES, zero=True)
        with self._lock:
            prior = self._prewrites[batch_id]
            if (batch_id in self._groups or batch_id in self._released
                    or prior['ceilings']['payload_max_bytes'] or
                    not isinstance(paths, list) or not paths or
                    len(prior['paths']) + len(paths) > 2 * MAX_GROUP_ENTRIES or
                    len(set(paths)) != len(paths) or
                    any(not self.contains(p) or os.path.lexists(p) for p in paths)):
                raise BoundaryProducedPrewriteRefused(batch_id=batch_id, refusal='invalid checkpoint extension')
            existing = {p for row in self._prewrites.values() for p in row['paths']}
            if existing.intersection(paths):
                raise BoundaryProducedPrewriteRefused(batch_id=batch_id, refusal='extension repeats owned paths')
            ceilings = dict(prior['ceilings'])
            if temp_ceiling_bytes < ceilings['temp_max_bytes']:
                raise BoundaryProducedPrewriteRefused(batch_id=batch_id, refusal='checkpoint temp bound shrank')
            ceilings['checkpoint_max_bytes'] += checkpoint_ceiling_bytes
            ceilings['temp_max_bytes'] = temp_ceiling_bytes
            for name, maximum in self.durable_maxima().items():
                held = sum(row['ceilings'][name] for key, row in self._prewrites.items()
                           if key not in self._released and key != batch_id)
                if held + ceilings[name] > maximum:
                    raise BoundaryProducedPrewriteRefused(batch_id=batch_id, refusal=f'{name} ceiling exceeded')
            self._prewrites[batch_id] = {'ceilings': ceilings, 'paths': prior['paths'] + paths}
        return {'ok': True}

    def descriptor_for(self, reference, *, producer_generation, artifact_class='payload'):
        row = {'schema': DESCRIPTOR_SCHEMA, 'slot': self.declaration['slot'],
            'artifact_class': artifact_class, 'path': reference.path,
            'bytes': reference.file_bytes, 'sha256': reference.sha256,
            'producer_generation': producer_generation, 'file_identity': _file_identity(reference.path)}
        return validate_descriptor(row, self.declaration)

    def manifest_digest_for(self, descriptors):
        return DIRECT_ASCII_STRICT.sha256(descriptors)

    def publish(self, *, batch_id, descriptors, lifetime='retain'):
        with self._lock:
            if self.session is None or lifetime not in ('retain', 'consumed'):
                raise BoundaryProducedPublicationFailed('bound session and lifetime required')
            _limit(len(descriptors), MAX_GROUP_ENTRIES)
            prewrite = self._prewrites[batch_id]
            paths, classes = set(), {'payload': 0, 'checkpoint': 0}
            for row in descriptors:
                validate_descriptor(row, self.declaration)
                if (row['producer_generation'] != batch_id or row['path'] in paths
                        or row['path'] not in prewrite['paths'] or not self.contains(row['path'])):
                    raise BoundaryProducedPublicationFailed('descriptor roster differs from prewrite')
                if _file_identity(row['path']) != row['file_identity']:
                    raise BoundaryProducedPublicationFailed('destination identity changed')
                native_results.verify_result_artifact(row, max_bytes=MAX_FILE_BYTES)
                paths.add(row['path']); classes[row['artifact_class']] += row['bytes']
            if any(classes[k] > prewrite['ceilings'][k + '_max_bytes'] for k in classes):
                raise BoundaryProducedPublicationFailed('actual bytes exceed prewrite ceiling')
            if any(os.path.lexists(p) for p in set(prewrite['paths']) - paths):
                raise BoundaryProducedPublicationFailed('omitted prewrite paths still exist')
            document = {'schema': BATCH_SCHEMA, 'batch_id': batch_id,
                'producer': self.producer, 'declaration': self.binding, 'session': self.session,
                'lifetime': lifetime, 'entries': descriptors}
            bound = publish_document(self._directory / 'batches' / f'{_name(batch_id)}.json', document)
            self._groups[batch_id] = bound
            return {'ok': True, 'ref': bound}

    commit_origin = publish

    def reclaim_origin(self, batch_id):
        return self.abort_prewrite(batch_id=batch_id)

    def unlink_origin(self, batch_id, reference):
        from .staged_manifest import unlink_owned
        with self._lock:
            group = read_bound_document(self._groups[batch_id])
            rows = [row for row in group['entries'] if row['path'] == reference.path]
            if (len(rows) != 1 or rows[0]['sha256'] != reference.sha256
                    or rows[0]['bytes'] != reference.file_bytes):
                raise BoundaryProducedPublicationFailed('unlink differs from published writer identity')
            return unlink_owned(reference.path, rows[0]['file_identity'])

    def origin_only_lifetimes(self):
        return {key: read_bound_document(binding)['lifetime'] for key, binding in self._groups.items()}

    def finalize(self, path, *, dependencies=(), checkpoint_exit=None):
        with self._lock:
            if self.session is None or not self._groups:
                raise BoundaryProducedPublicationFailed('cannot finalize empty or unbound publication')
            if checkpoint_exit is None:
                if set(self._prewrites) - self._released - set(self._groups):
                    raise BoundaryProducedPublicationFailed('incomplete prewrite cannot finalize successfully')
            elif (type(checkpoint_exit) is not int or
                  checkpoint_exit != self.declaration['checkpoint_exit']):
                raise BoundaryProducedPublicationFailed('checkpoint exit differs from immutable launch input')
            index = {'schema': INDEX_SCHEMA, 'producer': self.producer,
                'declaration': self.binding, 'session': self.session,
                'checkpoint_exit': checkpoint_exit,
                'groups': [self._groups[key] for key in sorted(self._groups) if key not in self._released]}
            index_path = self._directory / 'index.json'
            publish_document(index_path, index)
            inputs = {'declaration': self.binding, **self.declaration['inputs']}
            return native_results.publish_result(Path(path), producer_id=self.declaration['template_id'],
                inputs=inputs, artifacts={'produced_index': index_path}, dependencies=dependencies,
                max_artifact_bytes=1 << 30)


@dataclass(frozen=True)
class VerifiedProducedPublication:
    selected: native_results.SelectedResult
    index: dict
    declaration: dict
    groups: tuple[dict, ...]


def read_produced_selection(selection, *, checkpoint_exit=None, verify_payloads=True):
    selected = (native_results.read_selected_result(selection) if checkpoint_exit is None else
                native_results.read_selected_checkpoint(selection, payload_exit=checkpoint_exit))
    if set(selected.artifact_payloads) != {'produced_index'}:
        raise BoundaryProducedBindingError('selected result is not an exact produced index')
    index = _fields(_decode(selected.artifact_payloads['produced_index']),
        ('schema', 'producer', 'declaration', 'session', 'checkpoint_exit', 'groups'), 'index')
    declaration = validate_declaration(read_bound_document(index['declaration']))
    _launch_binding(selected.publication['producer'], index['declaration'])
    expected_inputs = {'declaration': index['declaration'], **declaration['inputs']}
    if (index['schema'] != INDEX_SCHEMA or index['producer'] != selected.publication['producer']
            or index['checkpoint_exit'] != checkpoint_exit
            or selected.publication['inputs'] != expected_inputs or
            selected.publication['producer_id'] != declaration['template_id']):
        raise BoundaryProducedBindingError('native index producer/input/declaration differs')
    if checkpoint_exit is not None and declaration['checkpoint_exit'] != checkpoint_exit:
        raise BoundaryProducedBindingError('failed checkpoint not explicitly declared by launch')
    _session(index['session'])
    bindings = index['groups']
    if not isinstance(bindings, list):
        raise BoundaryProducedBindingError('finite group bindings required')
    _limit(len(bindings), declaration['max_groups'])
    groups, batches, paths, total, count = [], set(), set(), 0, 0
    # Validate the entire finite metadata roster before reading any payload.
    for binding in bindings:
        group = _fields(read_bound_document(binding),
            ('schema', 'batch_id', 'producer', 'declaration', 'session', 'lifetime', 'entries'), 'batch')
        _name(group['batch_id'])
        if (group['schema'] != BATCH_SCHEMA or group['batch_id'] in batches
                or group['producer'] != index['producer'] or group['declaration'] != index['declaration']
                or group['session'] != index['session'] or group['lifetime'] not in ('retain', 'consumed')
                or not isinstance(group['entries'], list)):
            raise BoundaryProducedBindingError('batch identity/roster differs')
        _limit(len(group['entries']), MAX_GROUP_ENTRIES)
        batches.add(group['batch_id'])
        for row in group['entries']:
            validate_descriptor(row, declaration)
            if row['path'] in paths or row['producer_generation'] != group['batch_id']:
                raise BoundaryProducedBindingError('duplicate or foreign descriptor')
            paths.add(row['path']); total += row['bytes']; count += 1
        groups.append(group)
    _limit(count, MAX_ENTRIES); _limit(total, MAX_PAYLOAD_BYTES)
    if verify_payloads:
        for group in groups:
            for row in group['entries']:
                native_results.verify_result_artifact(row, max_bytes=MAX_FILE_BYTES)
    return VerifiedProducedPublication(selected, index, declaration, tuple(groups))
