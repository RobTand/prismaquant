"""Measure container compilation-cache bytes across a Stage A/B row (PQ #2463).

The cache ROOT/MAX pair (PQ #1820, ``PRISMAQUANT_CONTAINER_CACHE_ROOT`` and
``PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES``) declares an operator-chosen
reservation. No party has measured cache bytes on a real row. This module
records the peak first, so the ceiling derives from evidence, not a guess.

Byte convention: allocated bytes count 512 times ``st_blocks`` for each
unique device/inode pair, directories included. Apparent bytes sum
``st_size``. Allocated bytes bound the disk the cache holds. Apparent bytes
are the file lengths a reader sees. The derivation reads the allocated
peak only.

Method: a sampler thread walks the cache root every ``interval_s`` seconds
and keeps the largest observed sample. The recorded number is the maximum
observed sample, not the final directory size and not an asserted exact peak
between samples. A scan that fails or an event overflow invalidates the
evidence: the receipt records the error and the caller must reject it.

Ceiling formula: ``C = G * ceil((P + H) / G)`` with ``G = 1073741824``. ``P``
is the first accepted observed peak. ``H`` is headroom the operator declares
before the repeat run, justified from observed cache growth, scan gaps, and
compiler concurrency. The workload scope binds the ceiling: model, runtime,
concurrency, and initial cache state. A repeat run under the same scope must
fit ``C``. A peak above ``C`` rejects the ceiling.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

#: GiB in bytes: the ceiling granularity PrismaBuild charges.
GIB_BYTES = 1073741824
#: Default sampler interval in seconds (250 ms per the plan).
SAMPLE_INTERVAL_S = 0.25
#: Receipt schema for a cache-peak measurement.
SCHEMA = "prismaquant.container_cache_peak.v1"
#: Receipt schema for a derived cache ceiling.
CEILING_SCHEMA = "prismaquant.container_cache_ceiling.v1"


def _allocation_bytes(path: Path) -> int:
    """Allocated bytes of one path: 512 times its block count."""
    return int(path.stat().st_blocks) * 512


def scan_cache_bytes(root) -> dict:
    """Scan ``root`` and return allocated and apparent byte totals.

    Each unique ``(st_dev, st_ino)`` pair counts once, so hard links never
    count twice. Directories count: their entries hold bytes on the disk.
    Symlinks count as themselves (``lstat``), never their target: a cache
    root that escapes through a link is a refusal elsewhere, not bytes here.
    """
    root = Path(root)
    allocated = 0
    apparent = 0
    files = 0
    dirs = 0
    seen: set[tuple[int, int]] = set()
    errors: list[str] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            st = current.lstat()
        except OSError as exc:
            errors.append(f"{current}: {exc}")
            continue
        key = (int(st.st_dev), int(st.st_ino))
        if key in seen:
            continue
        seen.add(key)
        allocated += int(st.st_blocks) * 512
        if current.is_symlink():
            try:
                apparent += len(os.readlink(current).encode())
            except OSError as exc:
                errors.append(f"{current}: {exc}")
            continue
        if current.is_dir():
            dirs += 1
            try:
                stack.extend(current / name for name in sorted(os.listdir(current)))
            except OSError as exc:
                errors.append(f"{current}: {exc}")
            continue
        files += 1
        apparent += int(st.st_size)
    return {"allocated_bytes": allocated, "apparent_bytes": apparent,
            "files": files, "dirs": dirs, "errors": list(errors)}


def describe_initial_state(root) -> dict:
    """Inventory ``root`` before a row runs: totals plus per-subdirectory bytes."""
    root = Path(root)
    state = {"exists": root.exists(), "totals": scan_cache_bytes(root), "subdirs": {}}
    if not root.is_dir() or root.is_symlink():
        return state
    try:
        names = sorted(os.listdir(root))
    except OSError as exc:
        state["totals"]["errors"].append(f"{root}: {exc}")
        return state
    for name in names:
        child = root / name
        try:
            if child.is_symlink() or not child.is_dir():
                continue
        except OSError as exc:
            state["totals"]["errors"].append(f"{child}: {exc}")
            continue
        state["subdirs"][name] = scan_cache_bytes(child)
    return state


class CachePeakSampler:
    """Sample a cache root at a fixed interval and keep the observed peak.

    The peak is the largest observed sample, never the final size. ``errors``
    collects every failed scan and every gap longer than twice the interval.
    Any entry invalidates the evidence: a failed scan may miss bytes, and a
    gap may miss the peak. ``incomplete_scan`` marks a run whose sampler
    stopped early, such as a crashed row.
    """

    def __init__(self, root, *, interval_s=SAMPLE_INTERVAL_S):
        self.root = Path(root)
        self.interval_s = float(interval_s)
        if self.interval_s <= 0:
            raise ValueError("sample interval must be positive")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="container-cache-peak",
                                        daemon=True)
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.peak_allocated_bytes = 0
        self.peak_sample_index: int | None = None

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop()

    def _take(self):
        started = time.time()
        try:
            totals = scan_cache_bytes(self.root)
        except OSError as exc:
            with self._lock:
                self.errors.append(f"scan failed: {exc}")
            return
        finished = time.time()
        with self._lock:
            index = len(self.samples)
            self.samples.append({"index": index, "unix": started,
                                 "scan_duration_s": finished - started, **totals})
            if totals["errors"]:
                self.errors.extend(f"sample {index}: {item}"
                                   for item in totals["errors"])
            if totals["allocated_bytes"] > self.peak_allocated_bytes:
                self.peak_allocated_bytes = totals["allocated_bytes"]
                self.peak_sample_index = index

    def _run(self):
        while not self._stop.is_set():
            self._take()
            self._stop.wait(self.interval_s)

    def stop(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join()
        # A closing scan after the row ends: the last periodic sample can
        # predate cleanup, which would report a stale final size. The peak
        # is a monotonic maximum, so one more sample only corrects it.
        self._take()

    def result(self, *, incomplete_scan=False) -> dict:
        """The sampler receipt: peak, samples, gaps, and errors."""
        with self._lock:
            samples = list(self.samples)
            errors = list(self.errors)
            peak = self.peak_allocated_bytes
            peak_index = self.peak_sample_index
        gaps: list[dict] = []
        for first, second in zip(samples, samples[1:]):
            gap = second["unix"] - first["unix"]
            if gap > 2 * self.interval_s:
                gaps.append({"after_index": first["index"], "gap_s": gap})
                errors.append(f"gap of {gap:.3f}s after sample {first['index']}")
        final_allocated = samples[-1]["allocated_bytes"] if samples else 0
        return {"sample_interval_s": self.interval_s, "samples": samples,
                "sample_count": len(samples), "gaps": gaps,
                "peak_allocated_bytes": peak, "peak_sample_index": peak_index,
                "final_allocated_bytes": final_allocated,
                "errors": errors, "incomplete_scan": bool(incomplete_scan),
                "valid": not errors and not incomplete_scan and bool(samples)}


def derive_cache_ceiling(first_peak_bytes, *, headroom_bytes) -> dict:
    """Derive the declared cache ceiling ``C = G * ceil((P + H) / G)``.

    ``P`` is the first accepted observed peak. ``H`` is headroom the
    operator declares before the repeat run. ``G`` is one GiB. Both inputs
    must be non-negative integers; the ceiling is at least one GiB.
    """
    peak = int(first_peak_bytes)
    headroom = int(headroom_bytes)
    if peak < 0 or headroom < 0:
        raise ValueError("peak and headroom must be non-negative")
    if int(first_peak_bytes) != peak or int(headroom_bytes) != headroom:
        raise ValueError("peak and headroom must be whole bytes")
    ceiling = max(1, math.ceil((peak + headroom) / GIB_BYTES)) * GIB_BYTES
    return {"peak_bytes": peak, "headroom_bytes": headroom,
            "gib_bytes": GIB_BYTES, "ceiling_bytes": ceiling,
            "formula": "C = G * ceil((P + H) / G)"}


def write_receipt(path, receipt: dict) -> str:
    """Write ``receipt`` as canonical JSON and return its SHA-256 hex."""
    from .cost_stage_checkpoint import atomic_write_bytes
    from .digests import bytes_sha256hex

    raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False,
                      default=str) + "\n").encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(Path(path), raw)
    return bytes_sha256hex(raw)


def measure_around(run, root, *, interval_s=SAMPLE_INTERVAL_S) -> dict:
    """Run ``run()`` under the sampler and return the sampler result.

    Takes the initial inventory first. The sampler starts before ``run``
    and stops after it returns or raises. A raise marks the scan
    incomplete: the peak may miss the crash's bytes.
    """
    initial = describe_initial_state(root)
    sampler = CachePeakSampler(root, interval_s=interval_s)
    failure = None
    sampler._thread.start()
    try:
        run()
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sampler.stop()
    result = sampler.result(incomplete_scan=failure is not None)
    return {"schema": SCHEMA, "initial_state": initial, "measurement": result,
            "failure": failure}
