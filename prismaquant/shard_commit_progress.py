"""Report the exporter child's durable shard commits to PrismaBuild's watchdog.

The uniform-arm body export runs the producer's ``export_tessera_serving.py``
as a child process from a pinned producer source PrismaQuant must not modify,
so the exporter itself cannot write a progress record.  Attempt 3 of the A8
v39 body export (PB ``d008b42a9c74…``, 2026-09-27) was killed at 3316 s with
``termination_reason: no_progress`` while it was 30 s from finishing: it had
declared an ``export`` phase but never reported a unit, so the phase grace --
sized for the whole run -- acted as a wall clock and discarded ~55 min of
steady work (#1516).

This module is the parent's half of the fix.  It watches the child's output
directory and reports **one progress unit per durably committed shard**: a
shard file that has been observed twice, at least one interval apart, with
identical non-zero size and mtime, is committed.  A file still growing is
never counted, so the counter cannot run ahead of the work it stands for --
the same rule ``prismabuild_progress`` states for journal shards.  What the
watcher certifies is *quiescence* (written and closed), not content: the
export receipt's manifest asserts remain the gate on what the shards contain.

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

import json
import os
import time
from pathlib import Path

from .prismabuild_progress import report as _report

#: Default observation interval.  Two observations an interval apart are the
#: quiescence proof, so a shard commits at most one interval after its last
#: write, and a stall allowance of a few minutes spans many missed polls.
DEFAULT_INTERVAL_SECONDS = 30.0

#: The shard file name pattern the exporter writes (``model-00001-of-00120
#: .safetensors`` and siblings), including the manifest the wrapper asserts
#: on separately, which is excluded because it is not a progress unit.
SHARD_PATTERN = "model-*-of-*.safetensors"


def declared_phases(environ=None) -> tuple[str, ...] | None:
    """The phases this action's submission sealed, or ``None`` if unset.

    Same reading as :func:`prismaquant.joint_run_progress.declared_phases`:
    the worker publishes the list as a JSON array
    (``prismabuild.pool`` mints ``json.dumps([phase.name, ...])``), so a
    bare comma list is unknown rather than a guess.  A8 r6 (PB action
    ``0a8bdf67``, 2026-09-27) was killed by a comma parse that read
    ``'["export"]'`` as one bogus phase name, refused the real one, and
    turned a declared stall contract into a wall clock (#1516).
    """

    environ = os.environ if environ is None else environ
    raw = environ.get("PRISMABUILD_ACTION_PROGRESS_PHASES", "")
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
    trips it long before the run-length grace would have.  The floor keeps a
    fast exporter from arming a grace tighter than PB's own poll cadence.
    """

    if seconds_per_shard <= 0:
        raise ValueError("seconds_per_shard must be positive")
    return max(minimum, multiplier * seconds_per_shard)


class ShardCommitProgress:
    """Watch one export output directory and report committed shards.

    ``poll`` is idempotent and cheap enough to call from any loop; ``watch``
    drives it around a child process until the child exits, then commits the
    remainder in a single scan -- after the writer is gone every remaining
    pattern file is closed by definition, so the quiescence rule collapses to
    existence and non-zero size.  Counting never regresses: a shard that was
    committed stays committed even if a filesystem hiccup hides it briefly.
    """

    def __init__(self, out_dir, *, phase, pattern=SHARD_PATTERN,
                 expected=None, interval_seconds=DEFAULT_INTERVAL_SECONDS,
                 environ=None, log=None):
        self.out_dir = Path(out_dir)
        self.phase = str(phase)
        self.pattern = str(pattern)
        self.expected = None if expected is None else int(expected)
        self.interval_seconds = float(interval_seconds)
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.environ = os.environ if environ is None else environ
        self._log = log if log is not None else _default_log
        self._phases = declared_phases(self.environ)
        self._committed: dict[str, tuple[int, int]] = {}
        self._pending: dict[str, tuple[int, int]] = {}
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
        observed = {}
        for path in self.out_dir.glob(self.pattern):
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
            if stat.st_size <= 0:
                continue
            observed[path.name] = (stat.st_size, stat.st_mtime_ns)
        return observed

    def poll(self, *, child_alive: bool = True) -> int:
        """Observe the directory once; report if new shards committed.

        Returns the cumulative committed count.  While ``child_alive`` is
        true a shard commits only on a second identical observation; with the
        child gone, existence and non-zero size commit immediately.
        """

        observed = self._scan()
        newly = []
        for name, fingerprint in sorted(observed.items()):
            if name in self._committed:
                continue
            if not child_alive or self._pending.get(name) == fingerprint:
                self._committed[name] = fingerprint
                newly.append(name)
            else:
                self._pending[name] = fingerprint
        for name in newly:
            self._pending.pop(name, None)
        if newly and not self._phase_refused:
            _report(self.phase, len(self._committed),
                    unit=newly[-1])
        return len(self._committed)

    def watch(self, child, *, poll_seconds: float | None = None) -> int:
        """Poll around ``child`` (a :class:`subprocess.Popen`) until it exits.

        The loop reports as it goes and one final scan after exit commits the
        last shards, so the watcher never outlives the child by an interval.
        The return is the cumulative committed count, not the child's status.
        """

        step = self.interval_seconds if poll_seconds is None else poll_seconds
        while child.poll() is None:
            self.poll(child_alive=True)
            time.sleep(step)
        return self.poll(child_alive=False)

    @property
    def committed(self) -> tuple[str, ...]:
        """Committed shard names in sort order."""

        return tuple(sorted(self._committed))

    def expected_complete(self) -> bool:
        """Whether every expected shard has committed."""

        if self.expected is None:
            raise ValueError("expected shard count was not provided")
        return len(self._committed) >= self.expected
