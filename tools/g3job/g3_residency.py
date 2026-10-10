"""PB's launch-bound staged ranges, translated through G3 container mounts.

Only the admitted helper tree owns map validation, pinning and release. A map
miss reads the declared origin. A named range never uses an origin fallback.
Each phase holds batched tier leases until all its descriptors close.
An unavailable RAM cover can select published stage covers. Integrity failures refuse.
"""
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid
from g3_pq_policy.digests import bytes_sha256hex
from g3_pq_policy.staged_lease import _classify

_LOCK = threading.Lock()
_READER = None


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
        helper = os.environ['PRISMABUILD_READER_HELPER_ROOT']
        sys.path.insert(0, str(Path(helper) / 'src'))
        from prismabuild import client
        self.lease, self.maps = client, client
        context = client.injected_context(env=os.environ)
        if not context['ok']:
            raise RuntimeError(f'PB staged reader context: {context["refusal"]}')
        self.ctx = context['ctx']
        self.queue = client.PoolQueue(self.ctx['queue_root'])
        self.root = Path(self.ctx['queue_root']) / 'residency'
        self.stats = {"staged_bytes": 0, "staged_reads": 0, "read_s": 0.0, "tiers": {},
                      "map_parses": 0, "phase_acquires": 0}
        self.lock = threading.RLock()
        self.mapping = None
        self.map_identity = None
        self.windows = {}
        self.closed_phases = set()
        self.phase_keys, self.key_phases = {}, {}
        if os.environ.get("G3_READ_PLAN"):
            plan = json.loads(Path(os.environ["G3_READ_PLAN"]).read_bytes())
            prefix = os.environ.get("G3_PHASE_PREFIX", "")
            for phase in plan["read_plan"]["phases"]:
                if prefix and not phase["name"].startswith(prefix):
                    continue
                name = phase["name"][len(prefix):]
                keys = [self.maps.residency_map_key(plan["entries"][i]["path"], plan["entries"][i]["offset"])
                        for i in phase["entry_indices"]]
                self.phase_keys[name] = keys
                for key in keys:
                    self.key_phases.setdefault(key, name)

    def _map(self):
        stat = os.stat(self.ctx["map_path"])
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if identity != self.map_identity:
            # RAM overlays can change without a new fragment-count generation.
            # Cache the atomic file identity, not only mapping["generation"].
            self.mapping = self.maps.read_residency_map(self.ctx["map_path"])
            self.map_identity = identity
            self.stats["map_parses"] = self.stats.get("map_parses", 0) + 1
        return self.mapping

    def _window(self, mapping, key, phase):
        for window in self.windows.get(phase, []):
            if key in window["entries"]:
                return window
        entry = mapping["entries"][key]
        tier = mapping["ram_tier_id"] if "ram_path" in entry else mapping["tier_id"]
        epoch = mapping["ram_epoch"] if "ram_path" in entry else ""
        pinned = {k for window in self.windows.get(phase, []) for k in window["entries"]}
        keys = self.phase_keys.get(phase, mapping["entries"])
        selected = {k: mapping["entries"][k] for k in keys if k in mapping["entries"] and k not in pinned
                    and ("ram_path" in mapping["entries"][k]) == ("ram_path" in entry)}
        covers = self.lease.covers_for_keys(self.root, self.ctx["action_key"], list(selected), tier_id=tier,
                                          manifest_sha256=mapping["manifest_sha256"], epoch=epoch)
        if not covers["ok"] and "ram_path" in entry and _classify(covers["refusal"]) == "availability":
            # PB permits unavailable RAM covers to select the same bytes on stage.
            # Integrity and unknown refusals never select another copy.
            tier, epoch = mapping["tier_id"], ""
            covers = self.lease.covers_for_keys(self.root, self.ctx["action_key"], list(selected), tier_id=tier,
                                              manifest_sha256=mapping["manifest_sha256"], epoch=epoch)
        if not covers["ok"]:
            raise RuntimeError(f'{phase}: PB covering material: {covers["refusal"]}')
        acq = self.lease.acquire_for(
            self.ctx, tier_id=tier, epoch=epoch, covers=covers["covers"],
            expected={k: {"bytes": e["bytes"], "sha256": e["sha256"]} for k, e in selected.items()},
            span={"start_bytes": 0, "end_bytes": sum(e["bytes"] for e in selected.values())},
            acquire_token=uuid.uuid4().hex, residency_root=self.root)
        if not acq["ok"]:
            raise RuntimeError(f'{phase}: PB reader lease: {acq["refusal"]}')
        window = {"acq": acq, "entries": selected, "active": 0}
        self.windows.setdefault(phase, []).append(window)
        self.stats["phase_acquires"] = self.stats.get("phase_acquires", 0) + 1
        return window

    def finish_phase(self, phase):
        with self.lock:
            windows = self.windows.get(phase, [])
            if any(window["active"] for window in windows):
                raise RuntimeError(f"{phase}: PB phase still has open descriptors")
            self.closed_phases.add(phase)
            for window in list(windows):
                acq = window["acq"]
                released = self.lease.release(self.queue, acq["pin_id"], acq["ref_id"],
                                              consumer_action_key=self.ctx["action_key"], residency_root=self.root)
                if not released:
                    raise RuntimeError(f"{phase}: PB reader release failed: {released}")
                windows.remove(window)

    def close(self):
        for phase in list(self.windows):
            self.finish_phase(phase)

    def read(self, path, offset, size):
        key = self.maps.residency_map_key(resolve_g3_origin(path), offset)
        with self.lock:
            # A map miss reads the declared origin, so it never raises:
            # the default phase below only labels mapped keys with no
            # tracked phase, and unmapped keys return before this check.
            mapping = self._map()
            entry = mapping["entries"].get(key)
            if entry is None:
                return None
            phase = self.key_phases.get(key, "setup" if self.phase_keys else "smoke")
            if phase in self.closed_phases:
                raise RuntimeError(f"{key}: read from completed phase {phase}")
            if entry["bytes"] != size:
                raise RuntimeError(f"{key}: staged range length differs from requested {size}")
            window = self._window(mapping, key, phase)
            entry = window["entries"][key]
            acq = window["acq"]
            window["active"] += 1
        start = time.perf_counter()
        fd = None
        try:
            fd, serving = self.lease.open_pinned(self.queue, acq['pin'], acq['ref_id'], key, residency_root=self.root)
            raw = read_fd(fd, 0, size)
            if bytes_sha256hex(raw) != entry['sha256']:
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
            with self.lock:
                window["active"] -= 1


def staged_range(path, offset, size):
    global _READER
    if not os.environ.get('PRISMABUILD_RESIDENCY_MAP'):
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
    return {'map': os.environ.get('PRISMABUILD_RESIDENCY_MAP'),
            **(_READER.stats if _READER is not None else {'staged_bytes': 0, 'staged_reads': 0})}


def finish_phase(phase):
    if _READER is not None:
        _READER.finish_phase(phase)


def close_reader():
    if _READER is not None:
        _READER.close()


def launch_read_plan():
    """Read the claimed manifest through the public, admitted PB client."""
    from g3_pq_policy.staged_lease import load_claimed_manifest
    helper = os.environ["PRISMABUILD_READER_HELPER_ROOT"]
    sys.path.insert(0, str(Path(helper) / "src"))
    from prismabuild import client
    context = client.injected_context(env=os.environ)
    if not context["ok"]:
        raise RuntimeError(f'PB staged reader context: {context["refusal"]}')
    return load_claimed_manifest(client, context["ctx"])


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
