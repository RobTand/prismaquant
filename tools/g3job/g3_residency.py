"""G3's reads of immutable staged ranges, translated through the G3 container mounts.

The bound staged manifest (``prismaquant.staged_launch``) is the only authority: the
existing resolver answers each declared range and a ``LeaseWindow`` holds the
phase's descriptors until the phase finishes. A range the manifest does not declare
reads its declared origin. A declared range that fails an integrity check raises and
never reads another copy.
"""
import json
import os
from pathlib import Path
import threading
import time
from g3_pq_policy.digests import bytes_sha256hex
from g3_pq_policy.staged_manifest import PATH_ENV, SHA_ENV, MAX_BYTES, load_staged_manifest, read_document

_LOCK = threading.Lock()
_READER = None
PLAN_SCHEMA = 'prismaquant.prismabuild.data_manifest.v2'


def resolve_g3_origin(path, mounts=None):
    mounts = json.loads(os.environ.get('G3_HOST_MOUNTS', '{}')) if mounts is None else mounts
    path = os.path.abspath(path)
    for target in sorted(mounts, key=len, reverse=True):
        if path == target or path.startswith(target.rstrip('/') + '/'):
            return mounts[target].rstrip('/') + path[len(target):]
    return path


def read_fd(fd, offset, size):
    out = bytearray(size)
    view, done = memoryview(out), 0
    while done < size:
        n = os.preadv(fd, [view[done:]], offset + done)
        if n <= 0:
            raise IOError(f'truncated staged range at {offset + done}, need {size - done} bytes')
        done += n
    return out


class StagedReader:
    def __init__(self):
        from g3_pq_policy import residency_map as maps
        from g3_pq_policy import staged_lease as lease
        self.resolver = maps.residency_resolver()
        if self.resolver is None:
            raise RuntimeError('G3 staged reader requires an explicit staged manifest')
        self.maps, self.lease = maps, lease
        self.stats = {"staged_bytes": 0, "staged_reads": 0, "read_s": 0.0, "tiers": {}, "phase_acquires": 0}
        self.lock = threading.RLock()
        self.windows = {}
        self.closed_phases = set()
        self.phase_keys, self.key_phases, self.extents = {}, {}, {}
        if os.environ.get("G3_READ_PLAN"):
            plan = json.loads(Path(os.environ["G3_READ_PLAN"]).read_bytes())
            prefix = os.environ.get("G3_PHASE_PREFIX", "")
            for phase in plan["read_plan"]["phases"]:
                if prefix and not phase["name"].startswith(prefix):
                    continue
                name = phase["name"][len(prefix):]
                keys = []
                for i in phase["entry_indices"]:
                    row = plan["entries"][i]
                    key = self.maps.residency_map_key(row["path"], row["offset"])
                    self.extents[key] = (row["path"], row["offset"], row["bytes"])
                    keys.append(key)
                self.phase_keys[name] = keys
                for key in keys:
                    self.key_phases.setdefault(key, name)

    def _declared(self, key):
        path, offset, size = self.extents[key]
        entry, verdict = self.resolver.staged_range_outcome(path, offset, offset + size)
        if verdict != self.maps.RANGE_HIT or entry["offset"] != offset or entry["bytes"] != size:
            return None
        return path, entry

    def _window(self, key, declared, entry, phase):
        for window in self.windows.get(phase, []):
            if key in window["entries"]:
                return window
        pinned = {k for window in self.windows.get(phase, []) for k in window["entries"]}
        items, keys = [(declared, entry)], [key]
        for other in self.phase_keys.get(phase, []):
            if other == key or other in pinned or other not in self.extents:
                continue
            found = self._declared(other)
            if found is not None:
                items.append(found)
                keys.append(other)
        lease, keys = self.lease.acquire_entries_window(self.resolver, items)
        lease.__enter__()
        window = {"lease": lease, "entries": dict(zip(keys, (e for _, e in items))), "active": 0}
        self.windows.setdefault(phase, []).append(window)
        self.stats["phase_acquires"] += 1
        return window

    def finish_phase(self, phase):
        with self.lock:
            windows = self.windows.get(phase, [])
            if any(window["active"] for window in windows):
                raise RuntimeError(f"{phase}: staged phase still has open descriptors")
            self.closed_phases.add(phase)
            for window in list(windows):
                window["lease"].__exit__(None, None, None)
                windows.remove(window)

    def close(self):
        for phase in list(self.windows):
            self.finish_phase(phase)

    def read(self, path, offset, size):
        declared = resolve_g3_origin(path)
        key = self.maps.residency_map_key(declared, offset)
        with self.lock:
            # An undeclared range reads its declared origin and never raises; a declared
            # range never reads another copy.
            entry, verdict = self.resolver.staged_range_outcome(declared, offset, offset + size)
            if entry is None:
                return None
            if entry["offset"] != offset or entry["bytes"] != size:
                raise RuntimeError(f"{key}: staged range differs from requested {size} bytes at {offset}")
            self.extents.setdefault(key, (declared, offset, size))
            phase = self.key_phases.get(key, "setup" if self.phase_keys else "smoke")
            if phase in self.closed_phases:
                raise RuntimeError(f"{key}: read from completed phase {phase}")
            window = self._window(key, declared, entry, phase)
            entry = window["entries"][key]
            window["active"] += 1
        start = time.perf_counter()
        fd = None
        try:
            fd, serving = window["lease"].open(key)
            raw = read_fd(fd, entry["object_offset"], size)
            if bytes_sha256hex(raw) != entry["sha256"]:
                raise RuntimeError(f"{key}: staged bytes differ from their own digest")
            with self.lock:
                self.stats["staged_bytes"] += size
                self.stats["staged_reads"] += 1
                self.stats["read_s"] += time.perf_counter() - start
                tier_stats = self.stats["tiers"].setdefault("stage", {"reads": 0, "bytes": 0})
                tier_stats["reads"] += 1
                tier_stats["bytes"] += size
            return raw
        finally:
            if fd is not None:
                window["lease"].close_fd(fd)
            with self.lock:
                window["active"] -= 1


def staged_range(path, offset, size):
    global _READER
    if not os.environ.get(PATH_ENV):
        return None
    with _LOCK:
        if _READER is None:
            _READER = StagedReader()
    return _READER.read(path, offset, size)


def read_g3_input(path, size=None):
    size = os.stat(path).st_size if size is None else size
    raw = staged_range(str(path), 0, size)
    if raw is not None:
        return raw
    with open(path, 'rb') as f:
        raw = f.read()
    if len(raw) != size:
        raise IOError(f'{path}: truncated file, expected {size} bytes')
    return raw


def receipt():
    return {'staged_manifest': os.environ.get(PATH_ENV), 'manifest_sha256': os.environ.get(SHA_ENV),
            **(_READER.stats if _READER is not None else {'staged_bytes': 0, 'staged_reads': 0})}


def finish_phase(phase):
    if _READER is not None:
        _READER.finish_phase(phase)


def close_reader():
    if _READER is not None:
        _READER.close()


def launch_read_plan(path, sha256):
    """The application read plan, by explicit path and digest, equal to the bound manifest's extents."""
    import gzip
    from g3_pq_policy import residency_map as maps
    from g3_pq_policy.staged_lease import ReadsetUnbound
    raw, _ = read_document(path, MAX_BYTES)
    if bytes_sha256hex(raw) != sha256:
        raise ReadsetUnbound(f'read plan {path} does not hash to its sha256')
    plan = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
    if (not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA
            or not isinstance(plan.get('entries'), list) or not plan['entries']
            or not isinstance(plan.get('read_plan', {}).get('phases'), list)):
        raise ReadsetUnbound('the G3 read plan schema or phases are unsupported')
    resolver = maps.residency_resolver()
    if resolver is None:
        raise ReadsetUnbound('G3 read plan requires an explicit staged manifest')
    declared = {(e.source_path, e.source_offset, e.length) for e in resolver.manifest.extents}
    wanted = {(row['path'], row['offset'], row['bytes']) for row in plan['entries']}
    if wanted != declared:
        raise ReadsetUnbound('the staged manifest does not equal the read plan extents')
    return plan


def container_contract():
    """Readonly mounts for the bound manifest and read contract, and the container refusal.

    The adapter forwards the five native names and binds the phase namespace readonly.
    A host-local manifest cannot cross the container boundary.
    """
    named = os.environ.get(PATH_ENV)
    if not named:
        return [], {}
    manifest = load_staged_manifest(named, os.environ[SHA_ENV])
    if manifest.lifetime['mode'] != 'retained_shared_phase':
        raise RuntimeError('a host_local_cache staged manifest cannot cross the container boundary; '
                           'the descriptor transfer is an unbuilt unit 4D obligation')
    seen, mounts = set(), []
    for filename in (named, os.environ['PRISMAQUANT_READ_CONTRACT_PATH']):
        directory = str(Path(filename).parent)
        if directory not in seen:
            seen.add(directory)
            mounts.append({"source": directory, "target": directory, "readonly": True})
    return mounts, {}
