"""Per-phase IO accounting that any long-running stage can use.

A span is a named stretch of wall time in one process. When it closes, it
prints one machine-readable line and keeps the same record in a list. The
record holds the wall time and the change in every ``/proc/self/io`` counter,
plus the change in whatever other counters the caller registered: the
residency map's ``bytes_from_ram``/``bytes_from_stage``/``bytes_from_pool``,
or the GPU power sampler's watts. Spans nest. Each record names the span that
was open around it, so a reader can rebuild the tree from the lines alone.

Stage B uses it per window, per (window, probe) replay, per probe capture,
and around the head, checkpoint load, own-source install and records out.
Stage A's split runs can open the same spans around their own phases.
Nothing here reads or writes the workload's data, so wrapping a phase does
not change its IO path.

**What ``/proc/self/io`` counts.** The whole thread group, so reads made by
prefetch threads are counted. Child processes are not counted: the
``nvidia-smi`` power sampler is a child, and so is anything a stage runs
through ``subprocess``. PrismaQuant's readers run in threads, not child
processes. The cgroup-wide sum of ``/proc/PID/io`` over ``cgroup.procs`` is
the cross-check for anything that runs as a child.

* ``rchar``/``wchar`` are bytes through ``read``/``write`` and friends,
  including page-cache hits. Reads through ``mmap`` are not counted.
* ``read_bytes``/``write_bytes`` are bytes the kernel fetched from or sent to
  a storage layer for this process. Spill read-back that stays in the page
  cache moves ``rchar`` and not ``read_bytes``.
* Neither says where the bytes came from. The residency counters say which
  tier served a declared read, and the host's ``/proc/diskstats`` and
  ``/proc/self/mountstats`` say which device or NFS mount carried them.
  Match spans to those series by ``start_unix``/``end_unix``.

Reading ``/proc/self/io`` itself adds one ``read`` syscall and a few hundred
bytes of ``rchar`` to each reading.

**Rate lines.** :class:`ReadRateReporter` prints a rate and an ETA line
every ``every_entries`` entries or ``every_s`` seconds while a long read
runs. It reports nothing to PrismaBuild. Reads into a disposable scratch are
not durable work, and PB #480 counts only durable work as progress.

**Instrumentation never raises into the workload.** A counter that cannot
be read is recorded as ``None`` with the reason, and a failure to build or
print a record is printed and dropped. A workload exception passes through
the span unchanged. The span records it as the outcome.

**Telemetry readers and the sampler thread.** This module is the one home
of the process and host readers (``/proc/self/io``, ``/proc/meminfo``,
``/proc/self/status`` and ``nvidia-smi`` GPU power) and of the one sampler
thread, :class:`PeriodicSampler`, which every periodic sampler is built on
(PQ #1299). It imports nothing from PrismaQuant, so any stage can use it.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

IO_SPAN_SCHEMA = "prismaquant.io_span.v1"
READ_RATE_SCHEMA = "prismaquant.read_rate.v1"
#: The prefix of every span line. A reader greps for it and parses the rest
#: of the line as JSON.
IO_SPAN_MARKER = "[io-span]"
#: The prefix of every rate line.
READ_RATE_MARKER = "[read-rate]"
#: ``/proc/<pid>/io`` fields, in the kernel's order.
PROC_IO_FIELDS = ("rchar", "wchar", "syscr", "syscw", "read_bytes",
                  "write_bytes", "cancelled_write_bytes")
#: GB10's GPU power envelope, as the Stage B counters record it.
GB10_POWER_ENVELOPE_W = 140.0
#: Span record keys a caller's attributes may not replace.
_RESERVED = frozenset({
    "schema", "scope", "span", "parent", "start_unix", "end_unix", "wall_s",
    "outcome", "error", "proc_io", "proc_io_error", "sources",
    "source_errors"})


def read_proc_io(path: str | Path = "/proc/self/io") -> dict[str, int]:
    """Every counter in ``/proc/self/io`` (or ``path``), as integers.

    Raises ``OSError`` when the file cannot be read, which is what the
    existing callers that stamp ``io_before``/``io_after`` expect.
    """
    values = {}
    for line in Path(path).read_text().splitlines():
        key, value = line.split(":", 1)
        values[key.strip()] = int(value)
    return values


def _numeric(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def counter_delta(later: Mapping | None, earlier: Mapping | None) -> dict | None:
    """The change in each numeric key present in both readings."""
    if later is None or earlier is None:
        return None
    return {key: later[key] - earlier[key] for key in later
            if key in earlier and _numeric(later[key]) and _numeric(earlier[key])}


def _error_text(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"
    return text if len(text) <= 400 else text[:397] + "..."


class GpuPowerSpanSource:
    """Watts over a span, from a running 1 Hz power sampler.

    ``sampler`` is anything with a ``samples`` list of watts and an
    ``interval_s`` (:class:`GpuPowerSampler`). A span
    shorter than the interval can see no sample, and then reports none.
    """

    def __init__(self, sampler, *, envelope_w: float = GB10_POWER_ENVELOPE_W):
        self._sampler = sampler
        self.envelope_w = float(envelope_w)

    def snapshot(self) -> dict:
        samples = list(self._sampler.samples)
        return {"samples": len(samples), "watt_sum": float(sum(samples))}

    def summarize(self, delta: Mapping) -> dict:
        count = int(delta.get("samples", 0))
        if count <= 0:
            return {"samples": 0, "envelope_w": self.envelope_w}
        interval = float(getattr(self._sampler, "interval_s", 1.0))
        mean = float(delta["watt_sum"]) / count
        return {"samples": count, "joules": float(delta["watt_sum"]) * interval,
                "mean_w": mean, "envelope_w": self.envelope_w,
                "envelope_fraction": mean / self.envelope_w}


class IoSpan:
    """One open span. Close it with :meth:`IoSpanLog.close`."""

    __slots__ = ("name", "attrs", "parent", "start_unix", "start_clock",
                 "proc_io", "proc_io_error", "sources", "source_errors",
                 "closed")

    def __init__(self, name: str, attrs: dict, parent: str | None):
        self.name = name
        self.attrs = attrs
        self.parent = parent
        self.start_unix = 0.0
        self.start_clock = 0.0
        self.proc_io: dict | None = None
        self.proc_io_error: str | None = None
        self.sources: dict = {}
        self.source_errors: dict = {}
        self.closed = False


class IoSpanLog:
    """Open and close named spans, print one line per span, keep the records.

    ``scope`` names the run (a quantum id, a Stage A band), so lines from
    several processes in one log can be told apart. ``sources`` maps a name
    to either a callable returning a flat mapping of numbers, or an object
    with ``snapshot()`` and optionally ``summarize(delta)``. A source that
    returns ``None`` records ``None`` for that span.
    """

    def __init__(self, *, scope: str, log: Callable[[str], Any] | None = None,
                 sources: Mapping[str, Any] | None = None,
                 proc_io: Callable[[], Mapping] = read_proc_io,
                 clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time):
        self.scope = str(scope)
        self._log = log or (lambda line: print(line, flush=True))
        self._sources = dict(sources or {})
        self._proc_io = proc_io
        self._clock = clock
        self._wall = wall
        self._open: list[IoSpan] = []
        self.records: list[dict] = []

    def add_source(self, name: str, source) -> None:
        """Register another counter source; it applies to spans opened later."""
        self._sources[str(name)] = source

    # -- readings -----------------------------------------------------------

    def _read_proc_io(self):
        try:
            return dict(self._proc_io()), None
        except Exception as exc:  # noqa: BLE001 -- recorded, never raised
            return None, _error_text(exc)

    def _read_sources(self):
        values, errors = {}, {}
        for name, source in self._sources.items():
            try:
                snapshot = (source.snapshot() if hasattr(source, "snapshot")
                            else source())
                values[name] = None if snapshot is None else dict(snapshot)
            except Exception as exc:  # noqa: BLE001 -- recorded, never raised
                values[name] = None
                errors[name] = _error_text(exc)
        return values, errors

    # -- spans --------------------------------------------------------------

    def open(self, name: str, **attrs) -> IoSpan:
        """Open a span now. Attributes go into the record as top-level keys."""
        clash = _RESERVED & set(attrs)
        if clash:
            raise ValueError(f"span attributes may not use the reserved keys "
                             f"{sorted(clash)}")
        span = IoSpan(str(name), dict(attrs),
                      self._open[-1].name if self._open else None)
        span.start_unix = self._wall()
        span.start_clock = self._clock()
        span.proc_io, span.proc_io_error = self._read_proc_io()
        span.sources, span.source_errors = self._read_sources()
        self._open.append(span)
        return span

    def close(self, span: IoSpan, *, error: BaseException | None = None,
              outcome: str | None = None) -> dict | None:
        """Close ``span``, print its line and return its record.

        ``error`` records a workload exception; ``outcome`` overrides the
        outcome name. Closing a span twice is a no-op that returns ``None``.
        """
        if span.closed:
            return None
        span.closed = True
        try:
            self._open.remove(span)
        except ValueError:
            pass
        try:
            record = self._record(span, error=error, outcome=outcome)
        except Exception as exc:  # noqa: BLE001 -- instrumentation only
            self._emit_failure(span.name, exc)
            return None
        self.records.append(record)
        try:
            self._log(f"{IO_SPAN_MARKER} "
                      + json.dumps(record, sort_keys=True, allow_nan=False,
                                   default=str))
        except Exception as exc:  # noqa: BLE001 -- instrumentation only
            self._emit_failure(span.name, exc)
        return record

    def _emit_failure(self, name, exc):
        try:
            self._log(f"{IO_SPAN_MARKER} accounting for span {name!r} failed: "
                      f"{_error_text(exc)}")
        except Exception:  # noqa: BLE001 -- nothing left to report through
            pass

    def _record(self, span, *, error, outcome):
        end_clock = self._clock()
        end_unix = self._wall()
        proc_io, proc_io_error = self._read_proc_io()
        sources, source_errors = self._read_sources()
        record = {
            "schema": IO_SPAN_SCHEMA,
            "scope": self.scope,
            "span": span.name,
            "parent": span.parent,
            "start_unix": span.start_unix,
            "end_unix": end_unix,
            "wall_s": end_clock - span.start_clock,
            "outcome": outcome or ("ok" if error is None else "error"),
            "proc_io": counter_delta(proc_io, span.proc_io),
        }
        record.update(span.attrs)
        if error is not None:
            record["error"] = _error_text(error)
        if proc_io_error or span.proc_io_error:
            record["proc_io_error"] = proc_io_error or span.proc_io_error
        if self._sources or span.sources:
            deltas = {}
            for name, source in self._sources.items():
                delta = counter_delta(sources.get(name), span.sources.get(name))
                if delta is not None and hasattr(source, "summarize"):
                    delta = source.summarize(delta)
                deltas[name] = delta
            record["sources"] = deltas
        errors = {**span.source_errors, **source_errors}
        if errors:
            record["source_errors"] = errors
        return record

    @contextmanager
    def span(self, name: str, **attrs):
        """``with log.span("checkpoint-load"):`` -- open, run, close."""
        handle = self.open(name, **attrs)
        try:
            yield handle
        except BaseException as error:
            self.close(handle, error=error)
            raise
        self.close(handle)

    def close_open(self, *, error: BaseException | None = None,
                   outcome: str = "interrupted") -> list[dict]:
        """Close every span still open, innermost first, and return the records.

        A run that fails between a window's open and close callbacks leaves
        that span open. Its failure path calls this so the record still
        lands, with ``outcome`` set to ``interrupted``.
        """
        closed = []
        for span in reversed(list(self._open)):
            record = self.close(span, error=error, outcome=outcome)
            if record is not None:
                closed.append(record)
        return closed

    @property
    def open_names(self) -> list[str]:
        return [span.name for span in self._open]


class ReadRateReporter:
    """A rate and ETA line every ``every_entries`` entries or ``every_s`` seconds.

    The caller reports each entry it has read with :meth:`entry` and ends
    with :meth:`done`. A line prints when either threshold has passed since
    the last line, so the cadence holds for any entry size. The reporter has
    no thread of its own: while one entry is outstanding it prints nothing,
    and the gap between lines is then the measurement.

    Nothing is reported to PrismaBuild. See the module docstring.
    """

    def __init__(self, label: str, *, total_entries: int, total_bytes: int,
                 every_entries: int = 64, every_s: float = 30.0,
                 log: Callable[[str], Any] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 proc_io: Callable[[], Mapping] = read_proc_io):
        if int(every_entries) <= 0 or not float(every_s) > 0:
            raise ValueError("rate line thresholds must be positive")
        self.label = str(label)
        self.total_entries = int(total_entries)
        self.total_bytes = int(total_bytes)
        self.every_entries = int(every_entries)
        self.every_s = float(every_s)
        self._log = log or (lambda line: print(line, flush=True))
        self._clock = clock
        self._proc_io = proc_io
        self.entries = 0
        self.bytes = 0
        self._started = clock()
        self._proc_start = self._read_proc_io()
        self._last_clock = self._started
        self._last_entries = 0
        self._last_bytes = 0
        self._done = False
        self.lines = 0

    def _read_proc_io(self):
        try:
            return dict(self._proc_io())
        except Exception:  # noqa: BLE001 -- recorded as None
            return None

    def entry(self, nbytes: int) -> None:
        self.entries += 1
        self.bytes += int(nbytes)
        now = self._clock()
        if (self.entries - self._last_entries >= self.every_entries
                or now - self._last_clock >= self.every_s):
            self._emit(now, final=False)

    def done(self) -> None:
        if self._done:
            return
        self._done = True
        self._emit(self._clock(), final=True)

    def _emit(self, now, *, final):
        try:
            elapsed = now - self._started
            interval = now - self._last_clock
            interval_bytes = self.bytes - self._last_bytes
            interval_rate = interval_bytes / interval if interval > 0 else None
            mean_rate = self.bytes / elapsed if elapsed > 0 else None
            rate = interval_rate if interval_rate else mean_rate
            remaining = max(0, self.total_bytes - self.bytes)
            proc = counter_delta(self._read_proc_io(), self._proc_start)
            record = {
                "schema": READ_RATE_SCHEMA,
                "label": self.label,
                "final": bool(final),
                "entries": self.entries,
                "entries_total": self.total_entries,
                "bytes": self.bytes,
                "bytes_total": self.total_bytes,
                "elapsed_s": elapsed,
                "interval_s": interval,
                "interval_mb_s": (None if interval_rate is None
                                  else interval_rate / 1e6),
                "mean_mb_s": None if mean_rate is None else mean_rate / 1e6,
                "eta_s": (0.0 if remaining == 0 else
                          remaining / rate if rate else None),
                "proc_io": proc,
            }
            self._log(f"{READ_RATE_MARKER} "
                      + json.dumps(record, sort_keys=True, allow_nan=False))
            self.lines += 1
        except Exception as exc:  # noqa: BLE001 -- instrumentation only
            try:
                self._log(f"{READ_RATE_MARKER} {self.label}: rate line failed: "
                          f"{_error_text(exc)}")
            except Exception:  # noqa: BLE001
                pass
        self._last_clock = now
        self._last_entries = self.entries
        self._last_bytes = self.bytes


#: The residency map's per-tier counters (PQ #736), as ``residency_report``
#: returns them. Their change over a span says which tier served the declared
#: reads made in it.
RESIDENCY_TIER_KEYS = ("bytes_from_ram", "bytes_from_stage", "bytes_from_pool",
                       "fallback_count", "ram_fallback_count")


def residency_tier_bytes() -> dict | None:
    """The residency map's tier counters now, or ``None`` with no map bound."""
    from .residency_map import residency_report

    report = residency_report()
    if report is None:
        return None
    return {key: int(report.get(key) or 0) for key in RESIDENCY_TIER_KEYS}


def stage_span_log(scope: str, *, power_sampler=None,
                   log: Callable[[str], Any] | None = None) -> IoSpanLog:
    """The span log a campaign stage opens: process IO, residency tiers, watts.

    ``power_sampler`` is the stage's running ``GpuPowerSampler``; without
    one the spans carry no power. Stage B's quantum uses this, and Stage A's
    split runs can use it unchanged.
    """
    sources: dict[str, Any] = {"residency": residency_tier_bytes}
    if power_sampler is not None:
        sources["gpu_power"] = GpuPowerSpanSource(power_sampler)
    return IoSpanLog(scope=scope, sources=sources, log=log)


def failure_outcome(error: BaseException, *, open_spans=()) -> dict:
    """The ``outcome`` block a failure path stamps into its counters.

    ``open_spans`` names the spans that were still open when the error
    arrived: the phase the run failed in.
    """
    return {"status": "failed", "error_type": type(error).__name__,
            "error": _error_text(error), "open_spans": list(open_spans)}


# -- host and process readers (PQ #1299) -----------------------------------


def _read_kb_table(path) -> dict[str, int]:
    """``Key: value [kB]`` lines as integers, with ``kB`` values in bytes.

    ``/proc/meminfo`` and ``/proc/<pid>/status`` share this format. A line
    whose first value is not an integer (``Name``, ``State``,
    ``Cpus_allowed_list``) is skipped; a unitless count is kept as is.
    Raises ``OSError`` when the file cannot be read.
    """
    values = {}
    for line in Path(path).read_text().splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        values[key.strip()] = value * 1024 if fields[1:2] == ["kB"] else value
    return values


def read_meminfo(path: str | Path = "/proc/meminfo") -> dict[str, int]:
    """Every ``/proc/meminfo`` field; the ``kB`` ones in bytes."""
    return _read_kb_table(path)


def read_proc_status(path: str | Path = "/proc/self/status") -> dict[str, int]:
    """The integer fields of ``/proc/self/status``; the ``kB`` ones in bytes."""
    return _read_kb_table(path)


def mem_available_bytes(path: str | Path = "/proc/meminfo") -> int:
    """The host's ``MemAvailable`` in bytes.

    On GB10 unified memory this is the box-level free memory that host and
    device allocations both draw from. Raises ``OSError`` when the file
    cannot be read and ``RuntimeError`` when it has no ``MemAvailable``.
    """
    value = read_meminfo(path).get("MemAvailable")
    if value is None:
        raise RuntimeError("/proc/meminfo has no MemAvailable")
    return value


def read_mountstats(path: str | Path = "/proc/self/mountstats") -> dict[str, dict]:
    """Per mount point, the NFS client's ``bytes`` row and every per-op row.

    ``{mount: {"bytes": [...] or None, "ops": {OP: [...]}}}``, every value
    an integer in the kernel's column order. ``bytes`` columns 0 and 4 are
    the client's and the server's read bytes; an op's columns are ops,
    transmissions, timeouts, bytes sent, bytes received, and queue, RTT and
    execute milliseconds, with an error count on newer kernels. A mount that
    is not NFS has neither. Raises ``OSError`` when the file cannot be read.
    """
    out: dict[str, dict] = {}
    row, per_op = None, False
    for line in Path(path).read_text().splitlines():
        if line.startswith("device "):
            parts = line.split()
            row = (out.setdefault(parts[parts.index("on") + 1], {"bytes": None, "ops": {}})
                   if "on" in parts[:-1] else None)
            per_op = False
            continue
        if row is None:
            continue
        text = line.strip()
        if text.startswith("bytes:"):
            row["bytes"] = [int(value) for value in text.split()[1:]]
        elif text.startswith("per-op statistics"):
            per_op = True
        elif per_op:
            name, sep, rest = text.partition(":")
            fields = rest.split()
            if sep and fields and all(field.isdigit() for field in fields):
                row["ops"][name] = [int(field) for field in fields]
    return out


def nfs_read_bytes(path: str | Path = "/proc/self/mountstats") -> dict[str, tuple[int, int]]:
    """``{mount: (client_read_bytes, server_read_bytes)}`` for every NFS mount."""
    return {mount: (row["bytes"][0], row["bytes"][4])
            for mount, row in read_mountstats(path).items() if row["bytes"]}


# -- the sampler thread (PQ #1299) -----------------------------------------


class PeriodicSampler:
    """The one sampler thread: call ``tick`` every ``interval_s`` seconds.

    ``tick`` takes and keeps one reading. It returns ``False`` to end the
    thread; an exception it raises also ends it, so a tick catches what it
    means to survive. ``tick_first`` takes the first reading at once;
    otherwise the thread waits one interval first. The thread is a daemon,
    so a sampler nobody stops never holds the process open.
    """

    def __init__(self, tick: Callable[[], Any], *, interval_s: float, name: str,
                 tick_first: bool = True):
        self.interval_s = float(interval_s)
        self._tick = tick
        self._tick_first = bool(tick_first)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def start(self) -> "PeriodicSampler":
        self._thread.start()
        return self

    def request_stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread.ident is not None:
            self._thread.join(timeout)

    def stop(self, timeout: float | None = None) -> None:
        self.request_stop()
        self.join(timeout)

    def __enter__(self) -> "PeriodicSampler":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    def _run(self) -> None:
        if not self._tick_first and self._stop.wait(self.interval_s):
            return
        while not self._stop.is_set():
            if self._tick() is False:
                return
            self._stop.wait(self.interval_s)


class MemAvailableFloor:
    """The box's lowest ``MemAvailable`` over a window, sampled on a thread.

    On GB10 unified memory, device allocations are host memory, so
    ``MemAvailable`` is the box-level reading of what a phase takes. The
    window is read once on entry and once on exit as well as every
    ``interval_s``. ``minimum`` is ``{"bytes", "unix"}`` of the lowest
    reading and ``first`` the entry reading.
    """

    def __init__(self, interval_s: float, *, name: str = "mem-available-floor"):
        self.interval_s = float(interval_s)
        self._lock = threading.Lock()
        self.first = None
        self.minimum = None
        self.samples = 0
        self._sampler = PeriodicSampler(self._tick, interval_s=interval_s, name=name)

    def _sample(self) -> None:
        value, now = mem_available_bytes(), time.time()
        with self._lock:
            self.samples += 1
            if self.first is None:
                self.first = {"bytes": value, "unix": now}
            if self.minimum is None or value < self.minimum["bytes"]:
                self.minimum = {"bytes": value, "unix": now}

    def _tick(self) -> None:
        try:
            self._sample()
        except (OSError, RuntimeError):
            pass

    def __enter__(self) -> "MemAvailableFloor":
        self._sample()
        self._sampler.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._sampler.stop()
        self._sample()


class GpuPowerSampler:
    """``nvidia-smi --query-gpu=power.draw`` sampling in-process, 1 Hz by default.

    ``nvidia_smi.gpu_utilization`` is non-diagnostic on GB10 (AGENTS.md
    principle 13), so the counters carry joules, watts and the kernel-active
    ratio instead. A missing or failing sampler is recorded, never silent and
    never zero. ``times`` holds each sample's host ``time.time()``, so
    :meth:`watts_between` can read the watts over any span of a run.
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = float(interval_s)
        self.samples: list[float] = []
        self.times: list[float] = []
        self.error: str | None = None
        self._process = None
        self._sampler = None

    def start(self) -> "GpuPowerSampler":
        import subprocess

        # ``-l`` takes whole seconds; a shorter interval needs ``-lms``.
        loop = (["-l", str(int(self.interval_s))] if self.interval_s == int(self.interval_s)
                else ["-lms", str(round(self.interval_s * 1000))])
        try:
            self._process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits", *loop],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except (OSError, ValueError) as exc:
            self.error = f"sampler launch failed: {exc}"
            return self
        # nvidia-smi paces the readings, so the thread blocks on the next
        # line rather than waiting an interval of its own.
        self._sampler = PeriodicSampler(self._read_line, interval_s=0, name="gpu-power")
        self._sampler.start()
        return self

    def _read_line(self) -> bool:
        try:
            line = self._process.stdout.readline()
        except (OSError, ValueError) as exc:
            if not self._sampler.stopping:
                self.error = f"sampler read failed: {exc}"
            return False
        if not line or self._sampler.stopping:
            return False
        value = line.strip().split(",")[0].strip()
        try:
            watts = float(value)
        except ValueError:
            return True
        self.times.append(time.time())
        self.samples.append(watts)
        return True

    def watts_between(self, start: float, end: float) -> list[float]:
        """The watts sampled at host times ``start <= t <= end``."""
        return [watts for when, watts in zip(self.times, self.samples)
                if start <= when <= end]

    def stop(self) -> dict:
        if self._sampler is not None:
            self._sampler.request_stop()
        try:
            if self._process is not None:
                self._process.terminate()
                self._process.wait(timeout=5)
        except Exception:  # noqa: BLE001 - teardown best effort, sample list stands
            pass
        if self._sampler is not None:
            self._sampler.join(timeout=2)
        watts = sorted(self.samples)
        block = {
            "sample_count": len(watts),
            "interval_s": self.interval_s,
            "gpu_joules": sum(watts) * self.interval_s if watts else None,
            "gpu_power_w_p50": watts[len(watts) // 2] if watts else None,
            "gpu_power_w_p95": watts[max(0, int(0.95 * len(watts)) - 1)] if watts else None,
            "gpu_power_w_max": watts[-1] if watts else None,
        }
        if self.error:
            block["sampler_error"] = self.error
        return block


def drop_page_cache(paths, *, missing_ok: bool = False) -> int:
    """Advise the kernel to drop each whole file's cached pages; return how
    many files were advised.

    The one owner (PQ #1302) of the measurement step benches run before a
    cold read, so the next read is a read. A path that cannot be opened
    raises, or with ``missing_ok`` is skipped.
    """
    done = 0
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            if not missing_ok:
                raise
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            done += 1
        finally:
            os.close(fd)
    return done


__all__ = [
    "GB10_POWER_ENVELOPE_W", "GpuPowerSampler", "GpuPowerSpanSource",
    "IO_SPAN_MARKER", "IO_SPAN_SCHEMA", "IoSpan", "IoSpanLog",
    "MemAvailableFloor", "PROC_IO_FIELDS", "PeriodicSampler",
    "READ_RATE_MARKER", "READ_RATE_SCHEMA", "RESIDENCY_TIER_KEYS",
    "ReadRateReporter", "counter_delta", "drop_page_cache", "failure_outcome",
    "mem_available_bytes", "nfs_read_bytes", "read_meminfo", "read_mountstats",
    "read_proc_io", "read_proc_status", "residency_tier_bytes", "stage_span_log",
]
