"""Measure container compilation-cache bytes across a Stage A/B row (PQ #2463).

The cache ROOT/MAX pair (PQ #1820, ``PRISMAQUANT_CONTAINER_CACHE_ROOT`` and
``PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES``) declares an operator-chosen
reservation. The retained PQ #2463 rows measure a workload-specific peak.
This module records samples so the ceiling derives from evidence.

Byte convention: allocated bytes count 512 times ``st_blocks`` for each
unique device/inode pair, directories included. Apparent bytes sum
``st_size``. Allocated bytes bound the disk the cache holds. Apparent bytes
are the file lengths a reader sees. The derivation reads the allocated
peak only.

Method: a sampler child walks the cache root every ``interval_s`` seconds
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
import time
from pathlib import Path

from .digests import DIRECT_ASCII_SPACED_LAX

#: GiB in bytes: the ceiling granularity PrismaBuild charges.
GIB_BYTES = 1073741824
#: Default sampler interval in seconds (250 ms per the plan).
SAMPLE_INTERVAL_S = 0.25
#: Receipt schema for a cache-peak measurement.
SCHEMA = "prismaquant.container_cache_peak.v1"
#: Receipt schema for a derived cache ceiling.
CEILING_SCHEMA = "prismaquant.container_cache_ceiling.v1"


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


def _sample_loop(root: str, interval_s: float, out_path: str,
                 control) -> None:
    """Confirm the initial sample and scan through the requested row end."""
    index = 0
    phase = "initial"
    try:
        while True:
            started = time.time()
            totals = scan_cache_bytes(root)
            line = {"index": index, "unix": started, "phase": phase,
                    "scan_duration_s": time.time() - started, **totals}
            with open(out_path, "a") as handle:
                handle.write(DIRECT_ASCII_SPACED_LAX.text(line) + "\n")
            if index == 0:
                # Confirm only after the complete sample reaches the log.
                control.send(not totals["errors"])
            if phase == "final":
                return
            index += 1
            phase = ("final" if control.poll(interval_s)
                     and control.recv() == "stop" else "periodic")
    finally:
        control.close()


class CachePeakSampler:
    """Sample a cache root at a fixed interval and keep the observed peak.

    The peak is the largest observed sample, never the final size. ``errors``
    collects every failed scan and every gap longer than twice the interval.
    Any entry invalidates the evidence: a failed scan may miss bytes, and a
    gap may miss the peak. ``incomplete_scan`` marks missing lifecycle coverage
    or a failed row, even when a final inventory exists.

    Sampling runs in a forked child process with its own GIL: the row
    under measurement compiles and holds this process's GIL for longer
    than the sample interval, which starved the earlier sampler thread
    past the gap limit. The child appends one JSON line per scan to a
    file outside the measured root. Entry waits for the first complete sample.
    The child writes a final sample after the parent requests the row end.
    The parent requires a successful child exit and both coverage boundaries.

    Enter the sampler before starting threads or CUDA work: ``fork``
    carries only the calling thread, and a child forked after CUDA
    initialization must never touch the driver (this child only reads
    the filesystem).
    """

    def __init__(self, root, *, interval_s=SAMPLE_INTERVAL_S):
        self.root = Path(root)
        self.interval_s = float(interval_s)
        if not math.isfinite(self.interval_s) or self.interval_s <= 0:
            raise ValueError("sample interval must be finite and positive")
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.peak_allocated_bytes = 0
        self.peak_sample_index: int | None = None
        self._tmpdir: object = None
        self._process: object = None
        self._out_path: object = None
        self._control: object = None
        self._started_unix: float | None = None
        self._finished_unix: float | None = None
        self._child_exitcode: int | None = None
        self._complete = False
        self._row_failed = False

    def __enter__(self):
        import multiprocessing
        import tempfile
        from multiprocessing.connection import wait
        from .io_engine import ENGINE

        if self._tmpdir is not None or self._started_unix is not None:
            raise RuntimeError("sampler cannot start twice")
        tmpdir = tempfile.mkdtemp(prefix="container-cache-peak-")
        self._tmpdir = tmpdir
        self._out_path = os.path.join(tmpdir, "samples.jsonl")
        Path(self._out_path).write_text("")
        context = multiprocessing.get_context("fork")
        receiver, sender = context.Pipe(duplex=True)
        self._control = receiver
        try:
            self._process = ENGINE.start_scan_process(
                _sample_loop,
                args=(str(self.root), self.interval_s, self._out_path, sender),
                name="container-cache-peak")
            sender.close()
            available = wait([receiver, self._process.sentinel], timeout=30)
            try:
                confirmed = receiver in available and receiver.recv() is True
            except EOFError:
                confirmed = False
            if not confirmed or not self._process.is_alive():
                raise RuntimeError("sampler did not confirm the first complete scan")
            self._started_unix = time.time()
        except BaseException:
            self.errors.append("sampler startup failed")
            self.stop()
            raise
        finally:
            sender.close()
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        self._row_failed = exc_type is not None
        self.stop()

    def _collect(self):
        """Read the child's samples and fold them into the observed peak."""
        import json as _json

        try:
            raw = Path(str(self._out_path)).read_text()
        except OSError as exc:
            self.errors.append(f"sample log unreadable: {exc}")
            return
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                sample = _json.loads(line)
            except ValueError as exc:
                self.errors.append(f"sample line undecodable: {exc}")
                continue
            index = len(self.samples)
            if sample.get("index") != index:
                self.errors.append(f"sample sequence is incomplete at index {index}")
            self.samples.append(sample)
            for item in sample.get("errors", []):
                self.errors.append(f"sample {index}: {item}")
            if int(sample.get("allocated_bytes", 0)) > self.peak_allocated_bytes:
                self.peak_allocated_bytes = int(sample["allocated_bytes"])
                self.peak_sample_index = index

    def stop(self):
        if self._tmpdir is None:
            return
        self._finished_unix = time.time()
        if self._process is not None:
            if not self._process.is_alive():
                self.errors.append("sampler child exited before the row ended")
            else:
                try:
                    self._control.send("stop")
                except OSError as exc:
                    self.errors.append(f"sampler stop request failed: {exc}")
            self._process.join(timeout=30)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=30)
                self.errors.append("sampler child did not exit; terminated")
            self._child_exitcode = self._process.exitcode
            if self._child_exitcode != 0:
                self.errors.append(f"sampler child exit code: {self._child_exitcode}")
            self._process.close()
            self._process = None
        self._control.close()
        self._collect()
        if self._started_unix is not None and len(self.samples) >= 2:
            first, last = self.samples[0], self.samples[-1]
            self._complete = (
                first.get("phase") == "initial"
                and first["unix"] + first["scan_duration_s"] <= self._started_unix
                and last.get("phase") == "final"
                and last["unix"] >= self._finished_unix
                and self._child_exitcode == 0)
        if not self._complete:
            self.errors.append("sampler did not cover the complete row")
        import shutil

        shutil.rmtree(str(self._tmpdir), ignore_errors=True)
        self._tmpdir = None

    def result(self, *, incomplete_scan=False) -> dict:
        """The sampler receipt: peak, samples, gaps, and errors."""
        samples = list(self.samples)
        errors = list(self.errors)
        peak = self.peak_allocated_bytes
        peak_index = self.peak_sample_index
        incomplete_scan = bool(incomplete_scan or self._row_failed or not self._complete)
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
                "row_started_unix": self._started_unix,
                "row_finished_unix": self._finished_unix,
                "child_exitcode": self._child_exitcode,
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




def measure_around(run, root, *, interval_s=SAMPLE_INTERVAL_S) -> dict:
    """Run ``run()`` under the sampler and return the sampler result.

    Takes the initial inventory first. The sampler starts before ``run``
    and stops after it returns or raises. A raise marks the scan
    incomplete: the peak may miss the crash's bytes.
    """
    initial = describe_initial_state(root)
    sampler = CachePeakSampler(root, interval_s=interval_s)
    failure = None
    sampler.__enter__()
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
