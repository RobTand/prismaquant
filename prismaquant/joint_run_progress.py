"""Local work counters and log cadence for joint AURA runs.

Writers report entries after publication. Read transitions name the phase
being observed; they grant no execution, residency or success authority.
Native walltime and authenticated results own those separate boundaries.
"""
from __future__ import annotations

import json
import math
import os
import time

DEFAULT_INTERVAL_S = 60.0
INTERVAL_ENV = "PRISMAQUANT_JOINT_PROGRESS_INTERVAL_S"
HEAD_PHASE = "head"


def layer_phase_name(layer: int) -> str:
    """The application read plan's local phase name for one layer."""
    return f"layer-{int(layer)}"


def log_progress(units: int, phase: str, unit: str | None = None,
                 *, log=print) -> None:
    """Log an observed cumulative count without publishing execution evidence."""
    log("application progress: " + json.dumps({
        "phase": str(phase), "units_completed": int(units), "unit": unit,
    }, sort_keys=True))


def interval_seconds(environ=None) -> float:
    """The finite positive local reporting interval requested by this run."""
    raw = (os.environ if environ is None else environ).get(INTERVAL_ENV)
    if raw is None or raw == "":
        return DEFAULT_INTERVAL_S
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{INTERVAL_ENV}={raw!r} is not a number of seconds") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{INTERVAL_ENV}={raw!r} must be finite positive seconds")
    return value


_RESIDENCY_KEYS = ("hits", "misses", "range_hits", "range_misses", "fallback_count")


def _residency_counters(resolver):
    if resolver is None:
        return None
    report = resolver.report()
    return {key: int(report.get(key) or 0) for key in _RESIDENCY_KEYS}


class JointRunProgress:
    """Log cumulative durable work and read counters on one injected clock."""

    def __init__(self, *, layers, partitions, base_units=0, log=print,
                 interval_s=None, clock=time.monotonic, phases=None,
                 commit=None, resolver=None):
        self.layers = int(layers)
        self.partitions = int(partitions)
        self.base_units = int(base_units)
        self._log = log
        self._clock = clock
        self._interval_s = (interval_seconds() if interval_s is None
                            else float(interval_s))
        if not math.isfinite(self._interval_s) or self._interval_s <= 0:
            raise ValueError("interval_s must be finite positive seconds")
        self._phases = ((HEAD_PHASE, *(layer_phase_name(i) for i in range(self.layers)))
                        if phases is None else (tuple(phases) or None))
        self._commit = commit
        self._resolver_override = resolver
        self._entries = 0
        self._priced = 0
        self._partition = None
        self._kind = None
        self._layer = None
        self._layer_entries = 0
        self._phase = None
        self._phase_index = -1
        self._undeclared = set()
        self._last_emit = self._clock()
        self._last_entries = 0
        self._last_residency = None
        self._committed_units = None
        self._committed_phase = None
        self.lines = 0
        self.commits = 0
        self.suppressed = 0

    @property
    def units(self) -> int:
        return self.base_units + self._entries + self._priced

    @property
    def phase(self) -> str | None:
        return self._phase

    def entry(self, *, layer, partition, kind="boundary") -> None:
        """Observe one entry only after its writer has published the bytes."""
        self._entries += 1
        layer = int(layer)
        if layer != self._layer:
            self._layer = layer
            self._layer_entries = 0
        self._layer_entries += 1
        self._partition = int(partition)
        self._kind = str(kind)
        if kind == "boundary":
            self.enter(layer_phase_name(layer))
        self.flush()

    def priced_units(self, count) -> None:
        """Fold in durable journal rows without regressing the observed count."""
        count = int(count)
        if count > self._priced:
            self._priced = count

    def enter(self, name) -> None:
        """Name an explicit local phase without moving backward in its roster."""
        if self._phases is None:
            return
        try:
            index = self._phases.index(str(name))
        except ValueError:
            if name not in self._undeclared:
                self._undeclared.add(str(name))
                where = (f"units commit under {self._phase!r}" if self._phase
                         else "nothing commits until a declared phase is entered")
                self._log(f"progress: phase {name!r} is not one this action "
                          f"declared; {where}")
            return
        if index > self._phase_index:
            self._phase_index = index
            self._phase = self._phases[index]

    def flush(self, *, force=False) -> bool:
        """Emit existing work/read diagnostics at the requested local cadence."""
        now = self._clock()
        elapsed = now - self._last_emit
        if not force and elapsed < self._interval_s:
            return False
        written = self._entries - self._last_entries
        rate = written / elapsed if elapsed > 0 else 0.0
        residency = _residency_counters(self._resolve())
        parts = [
            f"{self._phase or 'unreported'}:",
            f"layers {0 if self._layer is None else self._layer + 1}/{self.layers}",
            f"partitions {self._layer_entries}/{self.partitions}",
            f"entries {self._entries} (+{written}, {rate:.2f}/s)",
            f"units {self.units}",
        ]
        if residency is not None:
            before = self._last_residency or {key: 0 for key in _RESIDENCY_KEYS}
            parts.append("residency " + " ".join(
                f"{key}+{residency[key] - before[key]}" for key in _RESIDENCY_KEYS))
            self._last_residency = residency
        else:
            parts.append("residency unset")
        self._log("joint run progress: " + " ".join(parts))
        self.lines += 1
        self._last_emit = now
        self._last_entries = self._entries
        self._report()
        return True

    def _report(self) -> None:
        """Notify an explicitly supplied local observer once per advancement."""
        if self._phase is None:
            return
        advanced = (self._committed_units is None
                    or self.units > self._committed_units
                    or self._phase != self._committed_phase)
        if not advanced:
            self.suppressed += 1
            return
        if self._commit is not None and self._commit(self._phase, self.units):
            self.commits += 1
        self._committed_units = self.units
        self._committed_phase = self._phase

    def _resolve(self):
        if self._resolver_override is not None:
            return self._resolver_override
        from .residency_map import residency_resolver
        return residency_resolver()


__all__ = ["DEFAULT_INTERVAL_S", "HEAD_PHASE", "INTERVAL_ENV",
           "JointRunProgress", "interval_seconds", "layer_phase_name",
           "log_progress"]
