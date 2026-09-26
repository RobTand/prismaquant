"""A bounded ``torch.profiler`` timeline of one Stage B capture pass (PQ #1269).

Dev-only and off by default. :data:`PROFILE_ENV` names a directory; without
it :func:`pass_profile_request` returns ``None`` and the quantum runs exactly
as before: no profiler, no timing, no extra pass, and the same records.

With it set, the quantum records, for each probe the request names:

* ``capture``: probe ``p``'s spill capture pass. Every capture group's wall
  (a host clock, start and end) and the pass's own start and end are written,
  so the groups and the gaps between them account for the whole pass.
  ``torch.profiler`` (CPU and CUDA activities, ``record_shapes``, and
  ``with_stack`` when asked) records ``active`` groups after ``wait`` skipped
  and ``warmup`` warm groups.
* ``windowed``: before that capture, a shadow of the windowed replay's pass:
  the same backward over single stored batches, with a throwaway statistics
  lease on window zero's units (the lease a windowed replay's pass feeds) and
  no spill hooks. It is not final, so it stores nothing into the cotangent
  plane and forks the cotangent owners, as a windowed replay's earlier
  windows do. It stops after ``shadow_batches`` stored batches, and the lease
  is discarded unread. Its batches are profiled with the same schedule.

* ``render``: one retained render window of the spill replay (PQ #1348).
  Its units are the window's probes: each spans the probe's spill replay,
  its operator records and its projections, up to the probe's consumer. The
  commit after the last probe is outside every unit. The unit records carry
  the spill reader's counter deltas (reader wait, bytes and read calls), so a
  host gap in the trace can be named. The window's per-window kernel-time
  profiler is off while this session owns the window's profiler.

Each capture or shadow session writes
``<dir>/<quantum>-p<probe>-<kind>.trace.json.gz`` (Chrome trace),
``.key_averages.txt`` and ``.timing.json``; a render session writes
``<dir>/<quantum>-w<window>-render.*``. The pass's result bytes do not
change, but the shadow adds a pass's worth of time to its probe and a trace
export stalls its window, so a profiled row is a measurement, never a
campaign row.

``PRISMAQUANT_STAGE_B_PASS_PROFILE_SPEC`` is comma-separated ``key=value``:

* ``capture=1+2:stack`` -- the probes whose capture pass is profiled, joined
  by ``+``, each optionally ``:stack`` for ``with_stack`` (default ``1``);
* ``windowed=1`` -- the probe whose capture is preceded by the shadow
  windowed pass, or ``none`` (default ``1``);
* ``shadow_batches=48`` -- stored batches the shadow runs (default 48);
* ``wait=8,warmup=1,active=2`` -- the capture schedule, in capture groups;
* ``shadow_wait=16,shadow_warmup=2,shadow_active=8`` -- the shadow's, in
  stored batches;
* ``render=5`` -- the retained window whose probes are profiled, optionally
  ``:stack``, or ``none`` (default ``none``);
* ``render_wait=1,render_warmup=1,render_active=1`` -- the render schedule,
  in probes.

A render-only row names ``capture=,windowed=none``, or the defaults also
profile probe 1's capture and run its shadow.
"""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

PROFILE_ENV = "PRISMAQUANT_STAGE_B_PASS_PROFILE"
SPEC_ENV = "PRISMAQUANT_STAGE_B_PASS_PROFILE_SPEC"
SCHEMA = "prismaquant.stage_b_pass_profile.v1"

_DEFAULTS = {
    "capture": "1",
    "windowed": "1",
    "shadow_batches": "48",
    "wait": "8",
    "warmup": "1",
    "active": "2",
    "shadow_wait": "16",
    "shadow_warmup": "2",
    "shadow_active": "8",
    "render": "none",
    "render_wait": "1",
    "render_warmup": "1",
    "render_active": "1",
}

RENDER_OWNS_KERNEL_PROFILER = (
    "the render pass profile (PQ #1348) owns this window's torch.profiler")


class ShadowPassDone(Exception):
    """Ends the bounded shadow pass; caught by the quantum's shadow driver."""


@dataclass(frozen=True)
class PassProfileRequest:
    out_dir: Path
    capture_probes: dict  # probe -> with_stack
    windowed_probe: int | None
    shadow_batches: int
    wait: int
    warmup: int
    active: int
    shadow_wait: int
    shadow_warmup: int
    shadow_active: int
    render_window: int | None = None
    render_stack: bool = False
    render_wait: int = 1
    render_warmup: int = 1
    render_active: int = 1

    def profiles_capture(self, probe: int) -> bool:
        return int(probe) in self.capture_probes

    def profiles_render(self, window: int) -> bool:
        return self.render_window is not None and int(window) == self.render_window

    def session(self, *, kind: str, probe: int | None = None, identity: dict,
                stop_after=None, window: int | None = None, counters=None):
        if kind == "capture":
            schedule = (self.wait, self.warmup, self.active)
            stack = bool(self.capture_probes.get(int(probe), False))
        elif kind == "windowed":
            schedule = (self.shadow_wait, self.shadow_warmup, self.shadow_active)
            stack = False
        elif kind == "render":
            if window is None:
                raise ValueError("a render pass profile names its window")
            schedule = (self.render_wait, self.render_warmup, self.render_active)
            stack = self.render_stack
        else:
            raise ValueError(f"unknown pass profile kind {kind!r}")
        return PassProfileSession(
            self.out_dir, kind=kind, probe=None if probe is None else int(probe),
            schedule=schedule, with_stack=stack, identity=identity,
            stop_after=stop_after, window=None if window is None else int(window),
            counters=counters)


def _spec_int(spec, key, minimum=0):
    value = int(spec[key])
    if value < minimum:
        raise ValueError(f"{SPEC_ENV}: {key}={value} is below {minimum}")
    return value


def pass_profile_request(environ=None) -> PassProfileRequest | None:
    """The request the environment makes, or ``None`` (the default: off)."""
    environ = os.environ if environ is None else environ
    raw_dir = environ.get(PROFILE_ENV)
    if not raw_dir:
        return None
    spec = dict(_DEFAULTS)
    for item in filter(None, (environ.get(SPEC_ENV) or "").split(",")):
        key, sep, value = item.partition("=")
        if not sep or key.strip() not in _DEFAULTS:
            # A capture probe list is itself comma-free; anything else is a typo.
            raise ValueError(f"{SPEC_ENV}: unknown or malformed item {item!r}")
        spec[key.strip()] = value.strip()
    captures = {}
    for token in filter(None, spec["capture"].split("+")):
        probe, _, flag = token.partition(":")
        if flag not in ("", "stack"):
            raise ValueError(f"{SPEC_ENV}: capture flag {flag!r} is not 'stack'")
        captures[int(probe)] = flag == "stack"
    windowed = None if spec["windowed"] in ("", "none") else int(spec["windowed"])
    render_window, render_stack = None, False
    if spec["render"] not in ("", "none"):
        window, _, flag = spec["render"].partition(":")
        if flag not in ("", "stack"):
            raise ValueError(f"{SPEC_ENV}: render flag {flag!r} is not 'stack'")
        render_window, render_stack = int(window), flag == "stack"
        if render_window == 0 and (captures or windowed is not None):
            # Window zero's slot holds the spill captures (PQ #1172), and
            # two torch.profiler sessions cannot nest.
            raise ValueError(f"{SPEC_ENV}: render=0 profiles the window that "
                             "holds the captures; name capture=,windowed=none")
    return PassProfileRequest(
        out_dir=Path(raw_dir), capture_probes=captures, windowed_probe=windowed,
        shadow_batches=_spec_int(spec, "shadow_batches", 1),
        wait=_spec_int(spec, "wait"), warmup=_spec_int(spec, "warmup"),
        active=_spec_int(spec, "active", 1),
        shadow_wait=_spec_int(spec, "shadow_wait"),
        shadow_warmup=_spec_int(spec, "shadow_warmup"),
        shadow_active=_spec_int(spec, "shadow_active", 1),
        render_window=render_window, render_stack=render_stack,
        render_wait=_spec_int(spec, "render_wait"),
        render_warmup=_spec_int(spec, "render_warmup"),
        render_active=_spec_int(spec, "render_active", 1))


@dataclass
class PassProfileSession:
    """One profiled pass: unit walls for all units, a trace for a few."""

    out_dir: Path
    kind: str
    probe: int | None
    schedule: tuple
    with_stack: bool
    identity: dict
    stop_after: int | None = None
    window: int | None = None
    # A callable returning numeric counters; each unit records their deltas.
    counters: object = None
    units: list = field(default_factory=list)
    _unit_counters: dict | None = None
    _profiler: object = None
    _unit_start: float | None = None
    _pass_start: float | None = None
    _pass_end: float | None = None
    _trace_paths: list = field(default_factory=list)
    _errors: list = field(default_factory=list)
    export_s: float = 0.0
    _annotation: object = None

    @property
    def stem(self) -> str:
        quantum = str(self.identity.get("quantum_id", "quantum"))
        if self.window is not None:
            return f"{quantum}-w{self.window}-{self.kind}"
        return f"{quantum}-p{self.probe}-{self.kind}"

    def __enter__(self):
        import torch
        from torch.profiler import ProfilerActivity, profile, schedule

        self.out_dir.mkdir(parents=True, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        wait, warmup, active = self.schedule
        self._profiler = profile(
            activities=activities,
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=self._trace_ready,
            record_shapes=True, with_stack=self.with_stack, profile_memory=False)
        self._profiler.__enter__()
        self._pass_start = time.time()
        self._pass_start_perf = time.perf_counter()
        return self

    def _trace_ready(self, prof):
        started = time.perf_counter()
        try:
            self._export(prof)
        finally:
            self.export_s += time.perf_counter() - started

    def _export(self, prof):
        base = self.out_dir / self.stem
        try:
            trace = base.with_suffix(".trace.json.gz")
            prof.export_chrome_trace(str(trace))
            self._trace_paths.append(str(trace))
        except Exception as exc:  # a lost trace is recorded, never fatal
            self._errors.append(f"export_chrome_trace: {exc!r}")
        try:
            table = prof.key_averages().table(sort_by="self_cuda_time_total",
                                              row_limit=80, max_name_column_width=90)
            (base.with_suffix(".key_averages.txt")).write_text(table)
            if self.with_stack:
                stacked = prof.key_averages(group_by_stack_n=6).table(
                    sort_by="self_cpu_time_total", row_limit=60,
                    max_name_column_width=90)
                (base.with_suffix(".key_averages_stack.txt")).write_text(stacked)
        except Exception as exc:
            self._errors.append(f"key_averages: {exc!r}")

    def unit_begin(self) -> None:
        # A CPU annotation per unit, so the trace separates the units from
        # the gaps between them (the prefetch and staging waits).
        from torch.profiler import record_function

        self._annotation = record_function(f"pq_stage_b_{self.kind}_unit")
        self._annotation.__enter__()
        self._unit_counters = dict(self.counters()) if self.counters is not None else None
        self._unit_start = time.perf_counter()

    def span(self, label: str):
        """A CPU annotation inside a unit, so the trace splits the unit."""
        from torch.profiler import record_function

        return record_function(f"pq_stage_b_{self.kind}_{label}")

    def unit_end(self) -> None:
        end = time.perf_counter()
        if self._annotation is not None:
            self._annotation.__exit__(None, None, None)
            self._annotation = None
        start = self._unit_start if self._unit_start is not None else end
        index = len(self.units)
        wait, warmup, active = self.schedule
        state = ("wait" if index < wait else "warmup" if index < wait + warmup
                 else "active" if index < wait + warmup + active else "after")
        unit = {"index": index, "start_s": start - self._pass_start_perf,
                "end_s": end - self._pass_start_perf,
                "wall_s": end - start, "profiler": state}
        if self._unit_counters is not None:
            now = dict(self.counters())
            unit["counter_deltas"] = {key: now[key] - before
                                      for key, before in self._unit_counters.items()
                                      if key in now}
        self.units.append(unit)
        self._unit_start = None
        self._unit_counters = None
        self._profiler.step()
        if self.stop_after is not None and len(self.units) >= self.stop_after:
            raise ShadowPassDone()

    def __exit__(self, exc_type, exc, tb):
        end_perf = time.perf_counter()
        self._pass_end = time.time()
        if self._annotation is not None:  # a unit that raised
            self._annotation.__exit__(None, None, None)
            self._annotation = None
        try:
            self._profiler.__exit__(None, None, None)
        except Exception as error:
            self._errors.append(f"profiler exit: {error!r}")
        walls = [u["wall_s"] for u in self.units]
        steady = [u["wall_s"] for u in self.units if u["profiler"] in ("wait", "after")]
        profiled = [u["wall_s"] for u in self.units if u["profiler"] == "active"]
        gaps = [b["start_s"] - a["end_s"] for a, b in zip(self.units, self.units[1:])]
        pass_wall = end_perf - self._pass_start_perf
        record = {
            "schema": SCHEMA, "kind": self.kind, "probe": self.probe,
            "window": self.window,
            "identity": self.identity, "host": socket.gethostname(),
            "schedule": dict(zip(("wait", "warmup", "active"), self.schedule)),
            "with_stack": self.with_stack,
            "pass_start_unix": self._pass_start, "pass_end_unix": self._pass_end,
            "pass_wall_s": pass_wall,
            "units_count": len(self.units),
            "units_wall_sum_s": sum(walls),
            "gaps_sum_s": sum(gaps),
            "trace_export_s": self.export_s,
            "before_first_unit_s": self.units[0]["start_s"] if self.units else None,
            "after_last_unit_s": (pass_wall - self.units[-1]["end_s"]) if self.units else None,
            "unprofiled_unit_wall_mean_s": (sum(steady) / len(steady)) if steady else None,
            "profiled_unit_wall_mean_s": (sum(profiled) / len(profiled)) if profiled else None,
            "stopped_early": exc_type is ShadowPassDone,
            "ended_by": None if exc_type is None else exc_type.__name__,
            "traces": self._trace_paths, "errors": self._errors,
            "units": self.units,
        }
        (self.out_dir / f"{self.stem}.timing.json").write_text(
            json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
        return exc_type is ShadowPassDone
