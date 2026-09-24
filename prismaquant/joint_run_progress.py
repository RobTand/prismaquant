"""Say what the joint AURA run is doing, on a clock, to the log and to the fleet.

The run's two longest stretches printed one line between them. The head walk
resolved 512 anchor roster entries in 4 h 14 min and said nothing; the boundary
capture then wrote 23 040 durable entry files in 137.4 min and printed one line
at each end (action ``ad8803aa``, claimed 12:10:14, capture 16:24:17--18:41:42,
2026-09-18). For those six and a half hours a wedge and a healthy run were the
same observation, and PrismaBuild's residency window -- which publishes the next
phase's movers and releases the finished ones **on the consumer's accepted
progress** -- had nothing to advance on, so 19 of 46 phases staged, 1 egressed,
and the 744 GB stage reached 0 B available (PB #632).

Nothing here is measured that the loop did not already know. The entry count,
the layer and the partition are in hand at every step; this module only puts
them where a reader and the fleet can see them.

**Two outputs, one clock.**

* a log line, so a human reading the attempt log can tell a working run from a
  wedged one without counting files by hand;
* a PrismaBuild progress record, so the stall allowance is workload-based and
  the residency window advances.

Both fire on a **time** interval rather than per entry, so the cadence is a
property of the reader and not of the model: a 45-layer run and a 92-layer run
report at the same rate. The default interval comes from the standing rule that
a phase silent for more than two minutes is a defect; half of that leaves room
for one missed tick.

**What a unit is.** One durable exact-boundary entry file, counted by
``StreamedBoundaryArtifacts`` as it publishes it -- the file is on disk, named
and length-checked, before the counter moves. PB #480 requires exactly that:
cumulative units are reported only after durable work, never on entering a
loop. The count is cumulative across the whole run and across phases, so the
head walk's committed roster entries are the base the capture continues from.

**What a phase is, and why this module must not invent one.** The window
advances on a phase *name*, matched against the read plan the submission
sealed (``residency_plan.remaining``: a name the plan does not carry reads as
the beginning). The joint read set names a run's layer phases
``layer-<L>`` -- :func:`layer_phase_name`, which the manifest builder and this
reporter share so the two cannot drift -- and PrismaBuild hands the action the
list it sealed in ``PRISMABUILD_ACTION_PROGRESS_PHASES``. A name that is not in
that list is logged and **not** committed: the worker would refuse it, and an
action reporting only what the worker refuses is indistinguishable from one
reporting nothing at all.

**Backwards is not a phase.** The capture walks layers upward, the reverse pass
walks them down, and the read plan is written in the forward order. The
reporter therefore never names a phase earlier than the highest it has already
named. PrismaBuild's linear policy would clamp it anyway; doing it here keeps
the log line and the record telling the same story.

**Unset is inert.** With no progress channel in the environment the commit is a
no-op and the behaviour is byte-identical to this module's absence, which is
the ordinary case for every run submitted without the transport.
"""
from __future__ import annotations

import json
import os
import time

#: Rob's standing rule is that a phase silent for more than two minutes is a
#: defect. Half of it, so one missed tick is still inside the rule.
DEFAULT_INTERVAL_S = 60.0

#: How often a line is emitted, in seconds. A non-positive or unreadable value
#: is refused rather than silently replaced: an operator who asked for a
#: cadence and got another one is worse off than one who got an error.
INTERVAL_ENV = "PRISMAQUANT_JOINT_PROGRESS_INTERVAL_S"

#: The phases the submission sealed, as PrismaBuild's own contract exports
#: them. Read as data -- PrismaQuant never imports the fleet runtime.
PHASES_ENV = "PRISMABUILD_ACTION_PROGRESS_PHASES"

#: The phase a joint pass reads its head under, in the joint read set and in
#: ``joint_prewarm_phases``. Spelled here too so this module needs no import
#: from the prepare-side phase table to recognise it.
HEAD_PHASE = "head"


def layer_phase_name(layer: int) -> str:
    """The joint read set's phase name for one layer of a ``run``.

    One definition, several readers: the joint pass, export and AQUA read sets
    all write their layer phases with it and this reporter reports under it.
    Two spellings of the same convention is how a consumer ends up committing a
    name the window cannot match, which reads as no progress at all. Only the
    joint run reports today; the other two declare no progress phases, and the
    shared name is what keeps that a choice rather than a divergence.
    """
    return f"layer-{int(layer)}"


def declared_phases(environ=None) -> tuple[str, ...] | None:
    """The phase names this launch may report under, or ``None`` if unknown.

    Unknown means no contract, a launcher that does not publish the list, or a
    container that did not forward it. All three leave the reporter unable to
    check a name, and it then commits nothing rather than guessing: an
    undeclared phase is refused by the worker, and a run whose every report is
    refused looks exactly like a run that reports nothing.
    """
    raw = (os.environ if environ is None else environ).get(PHASES_ENV) or ""
    if not raw:
        return None
    try:
        names = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(names, list) or not names:
        return None
    if not all(isinstance(name, str) and name for name in names):
        return None
    return tuple(names)


def interval_seconds(environ=None) -> float:
    """The reporting cadence this run was asked for."""
    raw = (os.environ if environ is None else environ).get(INTERVAL_ENV)
    if raw is None or raw == "":
        return DEFAULT_INTERVAL_S
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{INTERVAL_ENV}={raw!r} is not a number of seconds") from exc
    if not value > 0:
        raise ValueError(f"{INTERVAL_ENV}={raw!r} must be a positive number of seconds")
    return value


_RESIDENCY_KEYS = ("hits", "misses", "range_hits", "range_misses", "fallback_count")


def _residency_counters(resolver):
    """The resolver's read counters, or ``None`` when no map was named.

    The reason nobody could say whether ``ad8803aa`` read the stage or the pool
    is that the run said nothing about its own reads. These are the counters
    ``residency_map.ResidencyResolver.report`` already keeps (PQ #736); the
    line prints what changed since the last one, which is what distinguishes a
    stage that is being read from a stage that is merely full.
    """
    if resolver is None:
        return None
    report = resolver.report()
    return {key: int(report.get(key) or 0) for key in _RESIDENCY_KEYS}


class JointRunProgress:
    """One clock, one log line, one committed count.

    ``layers`` and ``partitions`` are the totals the line reports against.
    ``base_units`` is what earlier stages of this same run already committed --
    the head walk's roster entries -- because the count PrismaBuild accepts is
    cumulative across phases and a restart at zero is a regression, not
    advancement.
    """

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
        # ``phases=None`` reads the launch's own declaration; an explicit
        # empty list is "this launch declared none", which is the same inert
        # state as no channel at all.
        self._phases = declared_phases() if phases is None else (tuple(phases) or None)
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

    # -- what the run tells it ------------------------------------------------

    @property
    def units(self) -> int:
        """Cumulative durable units: entries written plus what came before."""
        return self.base_units + self._entries + self._priced

    @property
    def phase(self) -> str | None:
        """The declared phase this reporter is committing under, if any."""
        return self._phase

    def entry(self, *, layer, partition, kind="boundary") -> None:
        """One durable exact-boundary entry file has landed.

        Called from the writer, after the file is published, so the counter
        never runs ahead of the bytes it stands for.
        """
        self._entries += 1
        layer = int(layer)
        if layer != self._layer:
            self._layer = layer
            self._layer_entries = 0
        self._layer_entries += 1
        self._partition = int(partition)
        self._kind = str(kind)
        if kind == "boundary":
            # Cotangent writes run the layers backwards and would otherwise
            # name a phase the plan has already released.
            self.enter(layer_phase_name(layer))
        self.flush()

    def priced_units(self, count) -> None:
        """Durable units the cost stage has journalled, beside the entries."""
        count = int(count)
        if count > self._priced:
            self._priced = count

    def enter(self, name) -> None:
        """Name the declared phase this run is reading now, never an earlier one."""
        if self._phases is None:
            return
        try:
            index = self._phases.index(str(name))
        except ValueError:
            if name not in self._undeclared:
                self._undeclared.add(str(name))
                # enter() keeps the phase already entered, and the count
                # commits under it; say so (PQ #1187).
                where = (f"units commit under {self._phase!r}" if self._phase
                         else "nothing commits until a declared phase is entered")
                self._log(f"progress: phase {name!r} is not one this action "
                          f"declared; {where}")
            return
        if index > self._phase_index:
            self._phase_index = index
            self._phase = self._phases[index]

    # -- what it tells the log and the fleet ----------------------------------

    def flush(self, *, force=False) -> bool:
        """Emit a line and commit the count, if the interval has elapsed."""
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
        """Commit, but only what the worker would accept as advancement.

        ``ProgressWatch`` accepts a record that passes its highest count or
        enters a later phase, and counts everything else as a rejection on the
        receipt. Sending a record we already know is a replay would put a
        rejection count on an action that is working perfectly, so the same
        rule is applied here and the line is printed either way: a line with no
        new entries is the diagnostic -- it says the run is alive and nothing
        landed.
        """
        if self._phase is None:
            return
        advanced = (self._committed_units is None
                    or self.units > self._committed_units
                    or self._phase != self._committed_phase)
        if not advanced:
            self.suppressed += 1
            return
        if self._committer()(self._phase, self.units):
            self.commits += 1
        self._committed_units = self.units
        self._committed_phase = self._phase

    # -- wiring ---------------------------------------------------------------

    def _resolve(self):
        if self._resolver_override is not None:
            return self._resolver_override
        from .residency_map import residency_resolver
        return residency_resolver()

    def _committer(self):
        if self._commit is not None:
            return self._commit
        from .prismabuild_progress import report
        return lambda phase, units: report(phase, units, unit="boundary_entries")


__all__ = ["DEFAULT_INTERVAL_S", "HEAD_PHASE", "INTERVAL_ENV", "PHASES_ENV",
           "JointRunProgress", "declared_phases", "interval_seconds",
           "layer_phase_name"]
