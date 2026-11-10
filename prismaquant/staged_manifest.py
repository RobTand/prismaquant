"""Immutable staged extents. Publication is offline; launch never hashes sources."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType

from .file_identity import file_stat_signature as signature
from .io_buffers import open_nofollow_directory as _open_directory
from .digests import bytes_sha256hex, file_digest_sha256hex, DIRECT_ASCII_LAX

from .staged_tier_policy import PATH_ENV, SHA_ENV, READ_ENV

SCHEMA = 'prismaquant.staged_input_manifest.v1'
MAX_BYTES = 16 << 20
MAX_OBJECTS = 1024


class ManifestRefused(ValueError):
    pass


def decode_document(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ManifestRefused('duplicate document fields')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


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


def open_directory(filename, *, create=False, identity=None):
    """Open a canonical directory without following any symlink component."""
    filename = path(os.fspath(filename))
    try:
        return _open_directory(filename, create=create, identity=identity)
    except ValueError as exc:
        raise ManifestRefused(str(exc)) from exc


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


def unlink_owned(filename, identity, *, missing_ok=False, directory_identity=None):
    """Remove only the recorded regular file through its canonical parent."""
    filename = Path(path(os.fspath(filename)))
    directory = open_directory(str(filename.parent), identity=directory_identity)
    try:
        try:
            observed = os.stat(filename.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        if not stat.S_ISREG(observed.st_mode) or tuple(identity) != signature(observed):
            raise ManifestRefused('owned unlink file identity changed')
        os.unlink(filename.name, dir_fd=directory)
        return True
    finally:
        os.close(directory)


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


def load_staged_manifest(filename, sha256):
    raw, _ = read_document(filename, MAX_BYTES)
    if bytes_sha256hex(raw) != digest(sha256):
        raise ManifestRefused('staged manifest digest mismatch')
    data = decode_document(raw)
    if not isinstance(data, dict) or set(data) != {'schema', 'read_contract_sha256', 'phase_id', 'producer', 'lifetime', 'entries'} or data['schema'] != SCHEMA:
        raise ManifestRefused('unsupported staged manifest schema/fields')
    digest(data['read_contract_sha256'])
    if not isinstance(data['phase_id'], str) or not data['phase_id'] or len(data['phase_id']) > 256:
        raise ManifestRefused('phase identity required')
    producer = data['producer']
    if (not isinstance(producer, dict) or set(producer) != {'source_commit', 'receipt_sha256'} or
            not isinstance(producer['source_commit'], str) or
            not re.fullmatch('[0-9a-f]{40}', producer['source_commit'])):
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
    if 'reader_scopes' in life:
        life = {**life, 'reader_scopes': tuple(life['reader_scopes'])}
    return StagedManifest(str(filename), sha256, data['read_contract_sha256'], data['phase_id'],
                          MappingProxyType(producer), MappingProxyType(life), tuple(extents))


def publish_staged_json(filename, data):
    filename = Path(filename)
    raw = DIRECT_ASCII_LAX.encoded(data) + b'\n'
    import secrets
    directory = open_directory(str(filename.parent))
    temporary = None
    try:
        candidate = '.publish-' + secrets.token_hex(16)
        fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        temporary = candidate
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, filename.name, src_dir_fd=directory, dst_dir_fd=directory,
                follow_symlinks=False)
        os.unlink(temporary, dir_fd=directory)
        temporary = None
        os.fsync(directory)
        observed = open_regular(str(filename))
        try:
            if signature(os.fstat(observed)) != signature(os.stat(
                    filename.name, dir_fd=directory, follow_symlinks=False)):
                raise ManifestRefused('publication directory changed')
        finally:
            os.close(observed)
    finally:
        if temporary is not None:
            os.unlink(temporary, dir_fd=directory)
        os.close(directory)
    return bytes_sha256hex(raw)


def materialize(entries, *, destination, manifest_path, read_contract_sha256, phase_id, producer, lifetime):
    """Copy declared extents synchronously through the existing IO primitive.

    Source identities are supplied by the authenticated offline producer, not
    recomputed by every launch. Object/extent digests are computed while copying.
    Caller owns admission and the explicitly provisioned publication directory.
    """
    from .io_buffers import copy_extent
    destination = Path(destination)
    rows = []
    for row in entries:
        src = row['source']
        path(src['path']); integer(src['bytes'], positive=True); digest(src['sha256'])
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
            if start == 0 and size == src['bytes'] and actual != src['sha256']:
                raise ManifestRefused('source digest mismatch')
            os.fchmod(temp_fd, 0o444); os.fsync(temp_fd)
            target = destination / actual
            try:
                os.link(temporary, target)
            except FileExistsError:
                existing = open_regular(str(target))
                try:
                    first = signature(os.fstat(existing))
                    with os.fdopen(os.dup(existing), 'rb') as stream:
                        existing_digest = file_digest_sha256hex(stream)
                    if first[2] != size or existing_digest != actual or signature(os.fstat(existing)) != first:
                        raise ManifestRefused('existing publication conflicts with immutable object')
                finally:
                    os.close(existing)
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
        sha = publish_staged_json(temporary, data)
        load_staged_manifest(temporary, sha)
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
