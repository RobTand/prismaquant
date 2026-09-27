"""Report the exporter child's durable shard commits to PrismaBuild's watchdog.

The uniform-arm body export runs the producer's ``export_tessera_serving.py``
as a child process from a pinned producer source PrismaQuant must not modify,
so the exporter itself cannot write a progress record.  Attempt 3 of the A8
v39 body export (PB ``d008b42a9c74…``, 2026-09-27) was killed at 3316 s with
``termination_reason: no_progress`` while it was 30 s from finishing: it had
declared an ``export`` phase but never reported a unit, so the phase grace --
sized for the whole run -- acted as a wall clock and discarded ~55 min of
steady work (#1516).

This module is the parent's half of the fix.  The producer's exporter
publishes every serving shard through :func:`save_serving_shard`
(``experiments/export_tessera_serving.py``): the payload is written to a
dot-prefixed ``.name.….partial`` staging file in the destination directory
and moved onto its final name with one ``Path.replace`` (producer #366,
"publish complete, host-readable main/twin bytes in one rename").  A file
that appears under a final ``model-*-of-*.safetensors`` name is therefore
complete and durable by construction, and the ``.partial`` staging names
never match the pattern.  So the watcher counts **final names on first
appearance** -- no quiescence heuristic, no second observation -- and reports
one progress unit per shard.  What the watcher certifies is that the shard
file was published, not what it contains: the export receipt's manifest
asserts remain the gate on that.

Follows the discipline of :mod:`prismaquant.joint_run_progress`: the phase
name must be one the submission declared (``PRISMABUILD_ACTION_PROGRESS_
PHASES``) or it is logged and not committed; the count is cumulative and
monotone; with no progress channel in the environment every report is a
no-op and the watcher changes nothing.  A phase declared with a stall
allowance should be sized **per shard** -- :func:`stall_allowance` derives
one from the measured rate -- never as one allowance for the whole run,
which is the failure mode #1516 filed.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .joint_run_progress import declared_phases
from .prismabuild_progress import report as _report

__all__ = [
    "DEFAULT_POLL_SECONDS",
    "SHARD_PATTERN",
    "ShardCommitProgress",
    "declared_phases",
    "stall_allowance",
]

#: Default poll cadence.  A shard is committed on first appearance, so this
#: only bounds how far behind the reports lag the renames (one interval), and
#: a stall allowance of a few minutes spans many missed polls.
DEFAULT_POLL_SECONDS = 30.0

#: The shard file name pattern the exporter publishes by atomic rename
#: (``model-00001-of-00120.safetensors`` and siblings).  The dot-prefixed
#: ``.partial`` staging files never match a ``model-*`` glob.
SHARD_PATTERN = "model-*-of-*.safetensors"


def _default_log(message):
    """Print a diagnostic so it survives a killed parent (#1516).

    Under the pool transport the action's stdout is a pipe, and a buffered
    line is discarded when the watchdog SIGKILLs the process: the phase
    refusal that explained the r6 kill never reached the attempt log.  The
    module's own log always flushes; a caller-supplied ``log`` must too.
    """

    print(message, flush=True)


def stall_allowance(seconds_per_shard: float, *, minimum: float = 300.0,
                    multiplier: float = 10.0) -> float:
    """A per-shard stall allowance from the measured rate (#1516).

    The A8 export measured 3,241 s for 120 shards -- 27 s per shard -- so a
    stall of ten shards is under five minutes while a genuine wedged writer
    trips it long before the run-length grace would have.

    ``minimum`` is literal because PQ's :mod:`prismabuild_progress` exposes no
    poll-cadence constant to derive it from: the worker samples the progress
    channel on a private timer (observed ~30 s in the A4 wrappers' read-only
    observer logs), and the 300 s floor is ten such samples of slack so a
    fast exporter cannot arm a grace tighter than the watchdog's own rhythm.
    ``multiplier`` paces the *mean* per-shard time; replacing it with a
    distribution-derived figure (p99 or max) is deferred until the r6
    py-spy/Netdata profile yields a measured per-shard spread to derive from.
    """

    if seconds_per_shard <= 0:
        raise ValueError("seconds_per_shard must be positive")
    return max(minimum, multiplier * seconds_per_shard)


class ShardCommitProgress:
    """Watch one export output directory and report published shards.

    ``poll`` is idempotent and cheap enough to call from any loop; ``watch``
    drives it around a child process until the child exits, then performs one
    final scan.  Counting never regresses: a shard that was committed stays
    committed even if a filesystem hiccup hides it briefly.
    """

    def __init__(self, out_dir, *, phase, pattern=SHARD_PATTERN,
                 expected=None, poll_seconds=DEFAULT_POLL_SECONDS,
                 environ=None, log=None):
        self.out_dir = Path(out_dir)
        self.phase = str(phase)
        self.pattern = str(pattern)
        self.expected = None if expected is None else int(expected)
        self.poll_seconds = float(poll_seconds)
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.environ = os.environ if environ is None else environ
        self._log = log if log is not None else _default_log
        self._phases = declared_phases(self.environ)
        self._committed: set[str] = set()
        self._phase_refused = False
        if self._phases is not None and self.phase not in self._phases:
            self._emit(f"shard progress: phase {self.phase!r} is not declared; "
                       "no units will be committed to the progress channel")
            self._phase_refused = True

    def _emit(self, message):
        # Flushing is the point: a refusal nobody can read is a stall nobody
        # can diagnose (#1516).
        self._log(message)

    def _scan(self):
        published = set()
        for path in self.out_dir.glob(self.pattern):
            if path.is_symlink() or not path.is_file():
                continue
            if path.stat().st_size <= 0:
                continue
            published.add(path.name)
        return published

    def poll(self) -> int:
        """Observe the directory once; report shards published since last time.

        A shard commits on its first appearance under a final name: the
        exporter's ``Path.replace`` makes the final name complete and durable
        by construction, so there is nothing to wait for.  Returns the
        cumulative committed count.
        """

        newly = self._scan() - self._committed
        self._committed |= newly
        if newly and not self._phase_refused:
            _report(self.phase, len(self._committed),
                    unit=sorted(newly)[-1])
        return len(self._committed)

    def watch(self, child, *, poll_seconds: float | None = None) -> int:
        """Poll around ``child`` (a :class:`subprocess.Popen`) until it exits.

        The loop reports as it goes and one final scan after exit picks up
        shards renamed between the last poll and the child's exit.  The
        return is the cumulative committed count, not the child's status.
        """

        step = self.poll_seconds if poll_seconds is None else poll_seconds
        while child.poll() is None:
            self.poll()
            time.sleep(step)
        return self.poll()

    @property
    def committed(self) -> tuple[str, ...]:
        """Committed shard names in sort order."""

        return tuple(sorted(self._committed))

    def expected_complete(self) -> bool:
        """Whether every expected shard has committed."""

        if self.expected is None:
            raise ValueError("expected shard count was not declared")
        return len(self._committed) >= self.expected
