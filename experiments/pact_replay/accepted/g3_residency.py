"""PB's launch-bound staged ranges, translated through G3 container mounts.

Only the admitted helper tree owns map validation, pinning and release. A map
miss reads the declared origin as the published contract specifies; a named
range never falls back after a pin/integrity failure. Pins outlive descriptors.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

_LOCK = threading.Lock()
_READER = None


def host_path(path, mounts=None):
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
        helper = os.environ['PRISMABUILD_READER_HELPER_ROOT']
        sys.path.insert(0, str(Path(helper) / 'src'))
        from prismabuild import reader_lease, residency_map, pool
        self.lease, self.maps = reader_lease, residency_map
        context = reader_lease.injected_context(env=os.environ)
        if not context['ok']:
            raise RuntimeError(f'PB staged reader context: {context["refusal"]}')
        self.ctx = context['ctx']
        self.queue = pool.PoolQueue(self.ctx['queue_root'])
        self.root = Path(self.ctx['queue_root']) / 'residency'
        self.stats = {"staged_bytes": 0, "staged_reads": 0, "read_s": 0.0, "tiers": {}}
        self.lock = threading.Lock()

    def read(self, path, offset, size):
        mapping = self.maps.read_map(self.ctx['map_path'])
        key = self.maps.residency_map_key(host_path(path), offset)
        entry = mapping['entries'].get(key)
        if entry is None:
            return None
        if entry['bytes'] != size:
            raise RuntimeError(f'{key}: staged range length differs from requested {size}')
        # Stage is the published qualified policy. A RAM admission/read remains
        # an independent PB qualification; no application-side RAM overlay.
        tier, epoch = mapping['tier_id'], ''
        covers = self.lease.covers_for_keys(self.root, self.ctx['action_key'], [key], tier_id=tier,
                                          manifest_sha256=mapping['manifest_sha256'], epoch=epoch)
        if not covers['ok']:
            raise RuntimeError(f'{key}: PB covering material: {covers["refusal"]}')
        acq = self.lease.acquire_for(self.ctx, tier_id=tier, epoch=epoch, covers=covers['covers'],
                                    expected={key: {'bytes': size, 'sha256': entry['sha256']}},
                                    span={'start_bytes': 0, 'end_bytes': size}, acquire_token=uuid.uuid4().hex,
                                    residency_root=self.root)
        if not acq['ok']:
            raise RuntimeError(f'{key}: PB reader lease: {acq["refusal"]}')
        start = time.perf_counter()
        fd = None
        try:
            fd, serving = self.lease.open_pinned(self.queue, acq['pin'], acq['ref_id'], key, residency_root=self.root)
            raw = read_fd(fd, 0, size)
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise RuntimeError(f'{key}: staged bytes differ from their own digest')
            with self.lock:
                self.stats['staged_bytes'] += size
                self.stats['staged_reads'] += 1
                self.stats['read_s'] += time.perf_counter() - start
                tier_stats = self.stats["tiers"].setdefault(serving["tier_id"], {"reads": 0, "bytes": 0})
                tier_stats["reads"] += 1
                tier_stats["bytes"] += size
            return raw
        finally:
            if fd is not None:
                os.close(fd)
            released = self.lease.release(self.queue, acq['pin_id'], acq['ref_id'],
                                          consumer_action_key=self.ctx['action_key'], residency_root=self.root)
            if not released:
                raise RuntimeError(f'{key}: PB reader release failed: {released}')


def staged_range(path, offset, size):
    global _READER
    if not os.environ.get('PRISMABUILD_RESIDENCY_MAP'):
        return None
    with _LOCK:
        if _READER is None:
            _READER = StagedReader()
    return _READER.read(path, offset, size)


def read_file(path, size=None):
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
    return {'map': os.environ.get('PRISMABUILD_RESIDENCY_MAP'), 'stage_only': True,
            **(_READER.stats if _READER is not None else {'staged_bytes': 0, 'staged_reads': 0})}


def container_contract():
    """Exact paths and launch context forwarded by g3_launch (queue is writable)."""
    if not os.environ.get('PRISMABUILD_RESIDENCY_MAP'):
        return [], {}
    names = ['PRISMABUILD_RESIDENCY_MAP', 'PRISMABUILD_ACTION_KEY', 'PRISMABUILD_ACTION_NONCE',
             'PRISMABUILD_ACTION_SCOPE', 'PRISMABUILD_QUEUE_ROOT', 'PRISMABUILD_READER_HELPER_ROOT']
    env = {name: os.environ[name] for name in names}
    map_path = Path(env["PRISMABUILD_RESIDENCY_MAP"])
    # The published map lives inside queue/residency. A readonly bind there
    # would mask queue/residency/leases even when queue itself is writable.
    # Bind the directory at a separate readonly view so atomic map updates
    # remain visible and the SDK writes pins through the original queue.
    map_target = "/g3-pb-residency-map"
    env["PRISMABUILD_RESIDENCY_MAP"] = str(Path(map_target) / map_path.name)
    mounts = [{"source": str(map_path.parent), "target": map_target, "readonly": True}]
    for path, readonly in [(env["PRISMABUILD_READER_HELPER_ROOT"], True),
                           (env["PRISMABUILD_QUEUE_ROOT"], False),
                           ("/stage/prewarm", True), ("/ram/prewarm", True)]:
        mounts.append({"source": path, "target": path, "readonly": readonly})
    return mounts, env