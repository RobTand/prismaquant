"""One immutable staged-input resolver for every existing IO consumer."""
from __future__ import annotations

import bisect
import json
import os
from pathlib import Path
import socket
import threading

from .staged_manifest import (SCHEMA, PATH_ENV as ENV_VAR, SHA_ENV, READ_ENV,
    ManifestRefused, load_staged_manifest, read_document, open_regular, signature, decode_document)
from .staged_tier_policy import active_policy

RANGE_HIT, RANGE_UNDECLARED = 'hit', 'undeclared'


class StagedReadRefused(RuntimeError):
    pass


def residency_map_key(path, offset=0):
    return f'{offset}:{path}'


class ResidencyResolver:
    def __init__(self, map_path, *, manifest_sha256, read_contract_sha256=None):
        self.manifest = load_staged_manifest(map_path, manifest_sha256)
        self._by_source = {}
        self._entries = {}
        for extent in self.manifest.extents:
            self._by_source.setdefault(extent.source_path, []).append(extent)
            self._entries[(extent.source_path, extent.source_offset)] = extent
        self._starts = {name: [e.source_offset for e in rows] for name, rows in self._by_source.items()}
        self._owner = os.getpid()
        self._lock = threading.RLock()
        self._fds = {}
        self._pins = []
        self._closed = False
        self._windows = 0
        self._stats = dict(bytes_from_stage=0, bytes_from_ram=0, bytes_from_pool=0,
                           hits=0, range_hits=0, misses=0, range_misses=0)
        self._fallbacks, self._serving = [], []
        self._paths = {}
        self._map_signature = None
        if read_contract_sha256 is not None:
            self.bind_manifest_sha256(read_contract_sha256)
        try:
            self._bind_lifetime()
        except BaseException:
            for fd in self._pins:
                os.close(fd)
            raise

    def _require_owner(self):
        if os.getpid() != self._owner:
            raise RuntimeError('staged resolver operations from a forked child are refused')
        if self._closed:
            raise RuntimeError('staged resolver is closed')

    def _bind_lifetime(self):
        life = self.manifest.lifetime
        if life['mode'] == 'host_local_cache':
            if socket.gethostname().split('.')[0] != life['owner_host']:
                raise ManifestRefused('host-local objects require their owner host')
            filename = os.environ.get('FLUXCACHE_MAP_FILE')
            if not filename:
                raise ManifestRefused('host-local input requires canonical run-set inherited map/pins')
            raw, self._map_signature = read_document(filename, 256 << 10)
            mapping = decode_document(raw)
            required = {m['key'] for m in self.manifest.members()}
            if not isinstance(mapping, dict) or set(mapping) != required:
                raise ManifestRefused('inherited map must cover the exact bound object set')
            root = Path(life['root'])
            for key, filename in mapping.items():
                if filename != str(root / 'objects' / key):
                    raise ManifestRefused('inherited object outside cache root')
                # The external helper owns locks. Check inherited shared pins
                # within its same-owner host-local contract, without flocking.
                pin = root / 'locks' / key
                covered = False
                for name in os.listdir('/proc/self/fd'):
                    try:
                        fd = int(name)
                        if os.get_inheritable(fd) and Path(os.readlink(f'/proc/self/fd/{fd}')) == pin:
                            info = Path(f'/proc/self/fdinfo/{fd}').read_text()
                            if 'FLOCK' in info and 'READ' in info:
                                covered = True
                                self._pins.append(os.dup(fd))
                                break
                    except (OSError, ValueError):
                        continue
                if not covered:
                    raise ManifestRefused('inherited run-set pin missing')
            self._paths = mapping
        else:
            # No retirement API exists here. Unknown scopes and native clean
            # retain the entire shared phase until its owner fences readers.
            self._paths = {e.object_sha256: e.object_path for e in self.manifest.extents}

    def bind_manifest_sha256(self, digest):
        if digest != self.manifest.read_contract_sha256:
            raise ManifestRefused('scientific read contract differs from staged manifest')

    def source_size(self, declared):
        rows = self._by_source.get(os.fspath(declared))
        if not rows:
            raise StagedReadRefused('source is not declared')
        return rows[0].source_bytes

    def _answer(self, extent):
        entry = extent.entry(self._paths[extent.object_sha256])
        entry['_resolver'] = self
        return entry

    def staged_range_outcome(self, declared, start, end, *, declared_size=None):
        self._require_owner()
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError('read span must be nonempty and bounded')
        name = os.fspath(declared)
        rows = self._by_source.get(name, ())
        index = bisect.bisect_right(self._starts.get(name, ()), start) - 1
        if index >= 0:
            e = rows[index]
            if declared_size is not None and declared_size != e.source_bytes:
                raise StagedReadRefused('source length differs from manifest')
            if e.source_offset <= start and end <= e.source_offset + e.length:
                self._stats['range_hits'] += 1
                return self._answer(e), RANGE_HIT
        self._stats['range_misses'] += 1
        return None, RANGE_UNDECLARED

    def staged_range(self, declared, start, end, *, declared_size=None):
        return self.staged_range_outcome(declared, start, end, declared_size=declared_size)[0]

    def staged_read(self, declared, *, offset=0, expected_sha256=None):
        if not self.stages(declared):
            return None
        size = self.source_size(declared)
        entry = self.staged_range(declared, offset, size)
        if entry is None or offset != 0 or entry['offset'] != 0 or entry['bytes'] != size:
            return None
        if expected_sha256 is not None and expected_sha256 != entry['source_sha256']:
            raise StagedReadRefused('source digest differs from requested identity')
        if entry['sha256'] != entry['source_sha256']:
            raise StagedReadRefused('whole source/extent digests conflict')
        return entry

    def stages(self, declared):
        return os.fspath(declared) in self._by_source

    def declared_readset(self):
        result = {}
        for e in self.manifest.extents:
            result.setdefault(e.source_path, []).append((e.source_offset, e.source_offset + e.length))
        return result

    def open_entry(self, entry):
        self._require_owner()
        key = entry['object_sha256']
        # Only resolver-issued entries can open through this owner.
        extent = self._entries.get((entry.get('declared_path'), entry.get('offset')))
        if entry.get('_resolver') is not self or extent is None or self._answer(extent) != entry:
            raise StagedReadRefused('entry differs from immutable manifest')
        with self._lock:
            if key not in self._fds:
                fd = open_regular(self._paths[key])
                first = signature(os.fstat(fd))
                if first[2] != entry['object_bytes']:
                    os.close(fd)
                    raise StagedReadRefused('object length differs from manifest')
                self._fds[key] = (fd, first)
            # dup() would share the witness's file position across readers.
            # Each consumer needs its own position as well as its own lifetime.
            fd = open_regular(self._paths[key])
            try:
                self.check_entry(entry, fd)
                return fd
            except BaseException:
                os.close(fd)
                raise

    def check_entry(self, entry, fd):
        self._require_owner()
        first = self._fds[entry['object_sha256']][1]
        if signature(os.fstat(fd)) != first:
            raise StagedReadRefused('object changed during reader lifetime')
        probe = open_regular(self._paths[entry['object_sha256']])
        try:
            if signature(os.fstat(probe)) != first:
                raise StagedReadRefused('object path changed during reader lifetime')
        finally:
            os.close(probe)

    def close(self):
        self._require_owner()
        with self._lock:
            self._require_owner()
            if self._windows:
                raise RuntimeError('cannot close staged resolver with live reader windows; join readers first')
            for fd, _ in self._fds.values():
                os.close(fd)
            for fd in self._pins:
                os.close(fd)
            self._pins.clear()
            self._fds.clear()
            self._closed = True

    def retain_window(self):
        self._require_owner()
        with self._lock:
            self._require_owner()
            self._windows += 1

    def release_window(self):
        self._require_owner()
        with self._lock:
            self._windows -= 1

    def record_fallback(self, declared, reason):
        self._fallbacks.append({'path': str(declared), 'reason': reason})

    def record_serving_tier(self, declared, tier, **kwargs):
        self._serving.append({'path': str(declared), 'tier': tier, **kwargs})

    def record_stage_read(self, declared, nbytes):
        self._stats['bytes_from_stage'] += nbytes

    record_stage_range_read = record_stage_read

    def record_pool_read(self, declared, nbytes):
        self._stats['bytes_from_pool'] += nbytes

    def report(self):
        return {**self._stats, 'manifest_sha256': self.manifest.sha256,
                'read_contract_sha256': self.manifest.read_contract_sha256,
                'fallbacks': list(self._fallbacks), 'serving_tiers': list(self._serving)}


_RESOLVER = None
_RESOLVER_LOCK = threading.RLock()
_PROCESS_PID = os.getpid()


def residency_resolver():
    global _RESOLVER
    if os.getpid() != _PROCESS_PID:
        raise RuntimeError('global staged resolver from a forked child is refused')
    with _RESOLVER_LOCK:
        if _RESOLVER is None:
            filename = os.environ.get(ENV_VAR)
            if filename:
                # Native launch activates the ssd policy before any resolver
                # exists. A manifest without it is a partial configuration.
                if active_policy() is None:
                    raise ManifestRefused('a staged manifest requires the active staged-tier policy')
                _RESOLVER = ResidencyResolver(filename, manifest_sha256=os.environ.get(SHA_ENV),
                                              read_contract_sha256=os.environ.get(READ_ENV))
        return _RESOLVER


def bind_residency_manifest(digest):
    resolver = residency_resolver()
    if resolver is None:
        raise ManifestRefused('explicit staged manifest path and digest are required')
    resolver.bind_manifest_sha256(digest)


def residency_report():
    resolver = residency_resolver()
    return None if resolver is None else resolver.report()


def reset_residency_resolver_for_tests():
    global _RESOLVER
    if os.getpid() != _PROCESS_PID:
        raise RuntimeError('global staged resolver reset from a forked child is refused')
    with _RESOLVER_LOCK:
        if _RESOLVER is not None and not _RESOLVER._closed:
            _RESOLVER.close()
        _RESOLVER = None
