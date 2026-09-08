"""Read-only authenticated source descriptors for the bounded GLM graph gate.

This experiment adapter binds the existing loader's reads to authenticated
objects. It does not load tensors, cache weights, or replace model code.
"""
from __future__ import annotations

import builtins
from contextlib import ExitStack, contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import threading
import time
from unittest.mock import patch


BLOCK_BYTES = 8 * 1024**2
MAX_HEADER_BYTES = 16 * 1024**2
METADATA_NAMES = ('config.json', 'model.safetensors.index.json')


def stat_identity(value):
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def checked_json(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f'input content SHA256 mismatch: {path}')
    return json.loads(raw)


def derive_source_roster(model, weight_map, profile, *, last_layer=5):
    """Use the loader's profile mapping; retain fixed tensors and 0..5 only."""
    root = Path(model).resolve()
    prefix = 'model.language_model.layers.'
    keys = {}
    body_layers = set()
    for checkpoint_key, shard in weight_map.items():
        live = profile.checkpoint_to_live_name(checkpoint_key, multimodal=True)
        if live is None:
            continue
        if live.startswith(prefix):
            layer = int(live[len(prefix):].split('.')[0])
            if layer > last_layer:
                continue
            body_layers.add(layer)
        path = (root / shard).resolve()
        if path.parent != root or path.suffix != '.safetensors':
            raise ValueError('source index shard escapes the original model directory')
        keys.setdefault(str(path), []).append(checkpoint_key)
    if body_layers != set(range(last_layer + 1)):
        raise ValueError('source index does not cover every original prefix/lookahead layer')
    return {path: sorted(value) for path, value in sorted(keys.items())}


class AuthenticatedSourceInputs:
    """Own O_RDONLY FDs from content hashing through all source consumption.

    CPU preflight authenticates only config/index, validates payload roster
    against the pinned lead manifest, and does not hash original payloads.
    Native preflight hashes each selected payload once in this FD lifetime.
    Path/fstat comparisons detect replacement or modification after hashing;
    they are freshness checks, not substitutes for content authentication.
    """
    def __init__(self, model, manifest, *, block_bytes=BLOCK_BYTES):
        self.root = Path(model).resolve()
        if not 1 <= block_bytes <= BLOCK_BYTES:
            raise ValueError('source hash buffer must be bounded')
        self.block_bytes = block_bytes
        self.expected = {}
        for item in manifest:
            path = str(Path(item['path']).resolve())
            if path in self.expected or Path(path).parent != self.root:
                raise ValueError('source manifest contains duplicate or out-of-root paths')
            if not isinstance(item.get('sha256'), str) or len(item['sha256']) != 64:
                raise ValueError('source manifest lacks a content digest')
            self.expected[path] = item
        self.fds, self.identities, self.authenticated = {}, {}, {}
        self.payload_keys = {}
        self.reads, self.metadata_reads, self.violations = [], {}, []
        self._lock = threading.Lock()
        self._open = os.open
        self._builtin_open, self._io_open = builtins.open, io.open
        self.started = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.require_unchanged()
        finally:
            for fd in self.fds.values():
                os.close(fd)
            self.fds.clear()

    def _identity(self, path, fd):
        descriptor = os.fstat(fd)
        if not stat.S_ISREG(descriptor.st_mode):
            raise ValueError('source descriptor is not a regular file')
        value = stat_identity(descriptor)
        if stat_identity(os.stat(path)) != value:
            raise RuntimeError('source pathname no longer identifies its held descriptor')
        return value

    def require_unchanged(self, path=None):
        for name in ([path] if path is not None else self.fds):
            if self._identity(name, self.fds[name]) != self.identities[name]:
                raise RuntimeError(f'authenticated source changed during consumption: {name}')
        if self.violations:
            raise RuntimeError(f'unauthenticated source read was attempted: {self.violations[0]}')

    def _refuse(self, reason):
        with self._lock:
            self.violations.append(reason)
        raise RuntimeError(reason)

    def authenticate(self, path):
        path = str(Path(path).resolve())
        if path in self.authenticated:
            self.require_unchanged(path)
            return self.authenticated[path]
        if path not in self.expected:
            raise ValueError(f'consumed source file is absent from the pinned content manifest: {path}')
        fd = self._open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            before = self._identity(path, fd)
            if before['bytes'] != self.expected[path]['bytes']:
                raise ValueError('source length differs from expected content manifest')
            digest = hashlib.sha256()
            offset = 0
            while offset < before['bytes']:
                data = os.pread(fd, min(self.block_bytes, before['bytes'] - offset), offset)
                if not data:
                    raise RuntimeError('source ended during descriptor content authentication')
                digest.update(data)
                if hasattr(os, 'posix_fadvise'):
                    os.posix_fadvise(fd, offset, len(data), os.POSIX_FADV_DONTNEED)
                offset += len(data)
                del data
            if self._identity(path, fd) != before:
                raise RuntimeError('source changed during descriptor content authentication')
            actual = digest.hexdigest()
            if actual != self.expected[path]['sha256']:
                raise ValueError(f'source content SHA256 mismatch: {path}')
            self.fds[path], self.identities[path] = fd, before
            self.authenticated[path] = dict(path=path, expected_sha256=self.expected[path]['sha256'],
                actual_sha256=actual, bytes_hashed=offset, descriptor_identity=before,
                maximum_hash_buffer_bytes=self.block_bytes)
            return self.authenticated[path]
        except BaseException:
            os.close(fd)
            raise

    def read_metadata(self, name):
        if name not in METADATA_NAMES:
            raise ValueError('unsupported original source metadata')
        path = str(self.root / name)
        self.authenticate(path)
        size = self.identities[path]['bytes']
        if size > MAX_HEADER_BYTES:
            raise ValueError('source metadata exceeds the bounded input cap')
        raw = os.pread(self.fds[path], size, 0)
        self.require_unchanged(path)
        if len(raw) != size:
            raise RuntimeError('short authenticated metadata read')
        return json.loads(raw)

    def bind_roster(self, roster):
        if self.payload_keys:
            raise RuntimeError('source payload roster is already bound')
        for path, keys in roster.items():
            if path not in self.expected or not keys or len(keys) != len(set(keys)):
                raise ValueError('derived source roster is not covered by the pinned manifest')
            current = stat_identity(os.stat(path))
            if current['bytes'] != self.expected[path]['bytes']:
                raise ValueError('derived source file length differs from expected manifest')
        self.payload_keys = {path: set(keys) for path, keys in roster.items()}

    def authenticate_payloads(self):
        if not self.payload_keys:
            raise RuntimeError('source payload authentication needs its derived roster')
        for path in sorted(self.payload_keys):
            self.authenticate(path)

    def report(self):
        return dict(schema='prismaquant.glm_prefix_source_descriptors.v1',
            authentication='SHA256 content then held O_RDONLY descriptor consumption; stat checks freshness only',
            authenticated=list(self.authenticated.values()),
            selected_payload_files=[dict(path=p, bytes=self.expected[p]['bytes'],
                expected_sha256=self.expected[p]['sha256'], keys=sorted(keys))
                for p, keys in sorted(self.payload_keys.items())],
            selected_payload_bytes=sum(self.expected[p]['bytes'] for p in self.payload_keys),
            unselected_manifest_files=sorted(set(self.expected) - set(self.payload_keys)
                - {str(self.root / name) for name in METADATA_NAMES}),
            payload_reader_calls=list(self.reads), metadata_only_reads=list(self.metadata_reads.values()),
            violations=list(self.violations), descriptors_open=len(self.fds))

    def _path(self, path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return None
        resolved = Path(os.fsdecode(path)).resolve()
        return str(resolved) if resolved.parent == self.root else None

    def _bound_path(self, path):
        if path not in self.authenticated:
            self._refuse(f'source payload read before content authentication: {path}')
        self.require_unchanged(path)
        return f'/proc/self/fd/{self.fds[path]}'

    def open(self, original, file, mode='r', *args, **kwargs):
        path = self._path(file)
        if path is None:
            return original(file, mode, *args, **kwargs)
        if mode not in ('r', 'rt', 'rb'):
            self._refuse('source adapter permits only read-only file opens')
        # Small metadata reads and Python payload readers both use held FDs.
        with self._lock:
            self.reads.append(dict(path=path, reader='python_open_authenticated_fd'))
        return original(self._bound_path(path), mode, *args, **kwargs)

    def os_open(self, file, flags, mode=0o777, *, dir_fd=None):
        path = self._path(file)
        if path is None:
            return self._open(file, flags, mode, dir_fd=dir_fd)
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            self._refuse('source adapter refused a writable descriptor')
        with self._lock:
            self.reads.append(dict(path=path, reader='os_open_authenticated_fd'))
        return self._open(self._bound_path(path), flags, mode)

    def safe_open(self, original, file, *args, metadata_only=False, **kwargs):
        path = self._path(file)
        if path is None:
            self._refuse(f'source safetensors reader escaped the original model: {file}')
        if path not in self.authenticated:
            if not metadata_only or path not in self.all_indexed_shards:
                self._refuse(f'source safetensors file is not authenticated: {path}')
            return HeaderOnlyFile(self, path)
        fdpath = self._bound_path(path)
        return BoundSafeOpen(self, path, original(fdpath, *args, **kwargs), metadata_only)

    @contextmanager
    def reader_binding(self, *, all_indexed_shards):
        """Cover safe_open, ordinary file readers, and os.open/pread users."""
        from prismaquant import layer_streaming, streaming_model
        self.all_indexed_shards = set(all_indexed_shards)
        self.require_unchanged()
        for path in self.payload_keys:
            if path not in self.authenticated:
                raise RuntimeError('native reader binding requires authenticated payload contents')
        layer_open, metadata_open = layer_streaming.safe_open, streaming_model.safe_open
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(layer_streaming, 'safe_open',
                    lambda file, *a, **kw: self.safe_open(layer_open, file, *a, **kw)))
                stack.enter_context(patch.object(streaming_model, 'safe_open',
                    lambda file, *a, **kw: self.safe_open(metadata_open, file, *a, metadata_only=True, **kw)))
                stack.enter_context(patch.object(builtins, 'open',
                    lambda file, mode='r', *a, **kw: self.open(self._builtin_open, file, mode, *a, **kw)))
                stack.enter_context(patch.object(io, 'open',
                    lambda file, mode='r', *a, **kw: self.open(self._io_open, file, mode, *a, **kw)))
                stack.enter_context(patch.object(os, 'open', self.os_open))
                yield
        finally:
            self.require_unchanged()


class MetadataSlice:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype

    def get_shape(self):
        return list(self.shape)

    def get_dtype(self):
        return self.dtype


class HeaderOnlyFile:
    """Preserve the existing all-layer size census without opening payloads."""
    def __init__(self, owner, path):
        self.owner, self.path = owner, path

    def __enter__(self):
        fd = self.owner._open(self.path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            before = self.owner._identity(self.path, fd)
            size_raw = os.pread(fd, 8, 0)
            size = int.from_bytes(size_raw, 'little')
            if len(size_raw) != 8 or not 0 < size <= min(MAX_HEADER_BYTES, before['bytes'] - 8):
                raise ValueError('source metadata-only safetensors header exceeds cap')
            raw = os.pread(fd, size, 8)
            if len(raw) != size or self.owner._identity(self.path, fd) != before:
                raise RuntimeError('source header changed during metadata-only inspection')
            self.header = json.loads(raw)
            item = dict(path=self.path, header_sha256=hashlib.sha256(size_raw + raw).hexdigest(),
                bytes_read=size+8, payload_bytes_read=0, descriptor_identity=before,
                scope='observed header identity only; original payload not consumed/authenticated')
            with self.owner._lock:
                previous = self.owner.metadata_reads.setdefault(self.path, item)
            if previous != item:
                raise RuntimeError('source header identity changed between metadata reads')
            return self
        finally:
            os.close(fd)

    def __exit__(self, *_args):
        self.header = None

    def get_slice(self, name):
        item = self.header[name]
        return MetadataSlice(item['shape'], item['dtype'])

    def get_tensor(self, name):
        self.owner._refuse(f'payload requested from an unauthenticated metadata-only shard: {name}')


class BoundSafeOpen:
    def __init__(self, owner, path, context, metadata_only):
        self.owner, self.path, self.context = owner, path, context
        self.metadata_only = metadata_only

    def __enter__(self):
        self.owner.require_unchanged(self.path)
        self.file = self.context.__enter__()
        return self

    def __exit__(self, *args):
        try:
            return self.context.__exit__(*args)
        finally:
            self.file = None
            self.owner.require_unchanged(self.path)

    def get_tensor(self, name):
        if self.metadata_only or name not in self.owner.payload_keys.get(self.path, ()):
            self.owner._refuse(f'payload key is outside the authenticated consumed roster: {name}')
        self.owner.require_unchanged(self.path)
        value = self.file.get_tensor(name)
        with self.owner._lock:
            self.owner.reads.append(dict(path=self.path, key=name, reader='safe_open_authenticated_fd'))
        return value

    def get_slice(self, name):
        value = self.file.get_slice(name)
        return MetadataSlice(value.get_shape(), value.get_dtype())

    def keys(self):
        return self.file.keys()

    def metadata(self):
        return self.file.metadata()
