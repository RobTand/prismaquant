"""Immutable staged extents. Publication is offline; launch never hashes sources."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

SCHEMA = 'prismaquant.staged_input_manifest.v1'
PATH_ENV = 'PRISMAQUANT_STAGED_MANIFEST'
SHA_ENV = 'PRISMAQUANT_STAGED_MANIFEST_SHA256'
READ_ENV = 'PRISMAQUANT_READ_CONTRACT_SHA256'
MAX_BYTES = 16 << 20
MAX_OBJECTS = 1024


class ManifestRefused(ValueError):
    pass


def digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ManifestRefused('immutable identity requires lowercase SHA256')
    return value


def path(value):
    if not isinstance(value, str) or not value.startswith('/') or '\0' in value or os.path.normpath(value) != value:
        raise ManifestRefused('path must be normalized and absolute')
    return value


def integer(value, *, positive=False):
    if type(value) is not int or not (1 if positive else 0) <= value < (1 << 63):
        raise ManifestRefused('invalid bounded byte offset/length')
    return value


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def open_regular(filename):
    """Refuse symlinks in every component, including a raced final component."""
    filename = Path(path(os.fspath(filename)))
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in filename.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(filename.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ManifestRefused('staged input must be a regular file')
        return fd
    finally:
        os.close(directory)


def read_document(filename, limit):
    fd = open_regular(filename)
    try:
        before = signature(os.fstat(fd))
        if before[2] > limit:
            raise ManifestRefused('document exceeds bound')
        raw = os.pread(fd, limit + 1, 0)
        if len(raw) != before[2] or signature(os.fstat(fd)) != before:
            raise ManifestRefused('document changed while reading')
        return raw, before
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Extent:
    source_path: str
    source_bytes: int
    source_sha256: str
    source_offset: int
    length: int
    object_path: str
    object_bytes: int
    object_sha256: str
    object_offset: int
    sha256: str

    def entry(self, object_path=None):
        return {'declared_path': self.source_path, 'source_bytes': self.source_bytes,
                'source_sha256': self.source_sha256, 'offset': self.source_offset,
                'bytes': self.length, 'stage_path': object_path or self.object_path,
                'object_bytes': self.object_bytes, 'object_sha256': self.object_sha256,
                'object_offset': self.object_offset, 'sha256': self.sha256}


@dataclass(frozen=True)
class StagedManifest:
    filename: str
    sha256: str
    read_contract_sha256: str
    phase_id: str
    producer: dict
    lifetime: dict
    extents: tuple[Extent, ...]

    def members(self):
        return [{'key': key, 'bytes': size} for key, size in sorted({
            e.object_sha256: e.object_bytes for e in self.extents}.items())]


def load_manifest(filename, sha256):
    raw, _ = read_document(filename, MAX_BYTES)
    if hashlib.sha256(raw).hexdigest() != digest(sha256):
        raise ManifestRefused('staged manifest digest mismatch')
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != {'schema', 'read_contract_sha256', 'phase_id', 'producer', 'lifetime', 'entries'} or data['schema'] != SCHEMA:
        raise ManifestRefused('unsupported staged manifest schema/fields')
    digest(data['read_contract_sha256'])
    if not isinstance(data['phase_id'], str) or not data['phase_id'] or len(data['phase_id']) > 256:
        raise ManifestRefused('phase identity required')
    producer = data['producer']
    if not isinstance(producer, dict) or set(producer) != {'source_commit', 'receipt_sha256'} or not re.fullmatch('[0-9a-f]{40}', str(producer['source_commit'])):
        raise ManifestRefused('producer source and receipt identity required')
    digest(producer['receipt_sha256'])
    life = data['lifetime']
    if not isinstance(life, dict):
        raise ManifestRefused('lifetime required')
    if life.get('mode') == 'host_local_cache':
        if set(life) != {'mode', 'root', 'owner_host'} or life['owner_host'] not in {'dl380g10', 'sparky', 'sparklina'}:
            raise ManifestRefused('host-local owner/root required')
        path(life['root'])
    elif life.get('mode') == 'retained_shared_phase':
        if (set(life) != {'mode', 'namespace', 'reader_scopes', 'retention'} or
                life['retention'] != 'until_all_readers_stopped_or_fenced' or
                not isinstance(life['reader_scopes'], list) or not life['reader_scopes'] or
                any(not isinstance(s, str) or not s for s in life['reader_scopes']) or
                len(set(life['reader_scopes'])) != len(life['reader_scopes'])):
            raise ManifestRefused('shared phase must retain explicit reader scopes until fenced')
        path(life['namespace'])
    else:
        raise ManifestRefused('unsupported lifetime')
    if not isinstance(data['entries'], list) or not data['entries']:
        raise ManifestRefused('nonempty extents required')
    extents, sources, objects, object_paths = [], {}, {}, {}
    for row in data['entries']:
        if not isinstance(row, dict) or set(row) != {'source', 'object', 'source_offset', 'object_offset', 'length', 'sha256'}:
            raise ManifestRefused('invalid extent fields')
        for name in ('source', 'object'):
            obj = row[name]
            if not isinstance(obj, dict) or set(obj) != {'path', 'bytes', 'sha256'}:
                raise ManifestRefused('invalid source/object identity')
            path(obj['path']); integer(obj['bytes'], positive=True); digest(obj['sha256'])
        src, obj = row['source'], row['object']
        start, offset, length = integer(row['source_offset']), integer(row['object_offset']), integer(row['length'], positive=True)
        if start + length > src['bytes'] or offset + length > obj['bytes']:
            raise ManifestRefused('extent exceeds source/object bounds')
        source_id, object_id = (src['bytes'], src['sha256']), (obj['bytes'], obj['sha256'])
        if sources.setdefault(src['path'], source_id) != source_id or object_paths.setdefault(obj['path'], object_id) != object_id:
            raise ManifestRefused('conflicting identity')
        if objects.setdefault(obj['sha256'], obj['bytes']) != obj['bytes']:
            raise ManifestRefused('conflicting object length')
        if life['mode'] == 'retained_shared_phase' and not Path(obj['path']).is_relative_to(life['namespace']):
            raise ManifestRefused('object outside retained phase')
        extents.append(Extent(src['path'], src['bytes'], src['sha256'], start, length,
                              obj['path'], obj['bytes'], obj['sha256'], offset, digest(row['sha256'])))
    if len(objects) > MAX_OBJECTS:
        raise ManifestRefused('manifest exceeds 1024 unique objects')
    extents.sort(key=lambda e: (e.source_path, e.source_offset))
    for a, b in zip(extents, extents[1:]):
        if a.source_path == b.source_path and a.source_offset + a.length > b.source_offset:
            raise ManifestRefused('overlapping source extents')
    return StagedManifest(str(filename), sha256, data['read_contract_sha256'], data['phase_id'], producer, life, tuple(extents))


def publish_json(filename, data):
    filename = Path(filename)
    raw = (json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.publish-', dir=filename.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, filename)
        directory = os.open(filename.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)
    return hashlib.sha256(raw).hexdigest()


def materialize(entries, *, destination, manifest_path, read_contract_sha256, phase_id, producer, lifetime):
    """Copy declared extents synchronously through the existing IO primitive.

    Source identities are supplied by the authenticated offline producer, not
    recomputed by every launch. Object/extent digests are computed while copying.
    Caller owns admission and the explicitly provisioned publication directory.
    """
    from .io_engine import copy_extent
    destination = Path(destination)
    rows = []
    for row in entries:
        src = row['source']
        fd = open_regular(src['path'])
        temp_fd, temporary = tempfile.mkstemp(prefix='.extent-', dir=destination)
        try:
            before = signature(os.fstat(fd))
            if before[2] != src['bytes']:
                raise ManifestRefused('source length changed')
            start, size = integer(row['source_offset']), integer(row['length'], positive=True)
            if start + size > before[2]:
                raise ManifestRefused('source extent out of bounds')
            actual = copy_extent(fd, temp_fd, start, size)
            if signature(os.fstat(fd)) != before:
                raise ManifestRefused('source changed during materialization')
            if row.get('sha256') is not None and row['sha256'] != actual:
                raise ManifestRefused('source extent digest mismatch')
            os.fchmod(temp_fd, 0o444); os.fsync(temp_fd)
            target = destination / actual
            os.link(temporary, target)
            directory = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            rows.append({'source': src, 'source_offset': start, 'length': size,
                         'object_offset': 0, 'sha256': actual,
                         'object': {'path': str(target), 'bytes': size, 'sha256': actual}})
        finally:
            os.close(fd); os.close(temp_fd); os.unlink(temporary)
    data = {'schema': SCHEMA, 'read_contract_sha256': read_contract_sha256,
            'phase_id': phase_id, 'producer': producer, 'lifetime': lifetime, 'entries': rows}
    # Validate before publication, without a temporary launch or PB context.
    fd, temporary = tempfile.mkstemp(prefix='.manifest-', dir=Path(manifest_path).parent)
    os.close(fd); os.unlink(temporary)
    try:
        sha = publish_json(temporary, data)
        load_manifest(temporary, sha)
        os.link(temporary, manifest_path)
        directory = os.open(Path(manifest_path).parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return sha
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
