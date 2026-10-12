"""Supervise replay silence outside the numerical process."""
from __future__ import annotations

import ctypes
import json
import math
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import re
import signal
import time

DEFAULT_STALL_SECONDS = 1800


def require_stall_seconds(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("The replay needs a positive finite stall allowance")
    return value


class StallWatch:
    """Only completed work resets the monotonic silence clock."""

    def __init__(self, allowance, *, clock=None):
        self.allowance = require_stall_seconds(allowance)
        self.clock = clock or time.monotonic
        self.last_time = self.clock()
        self.last = {"layer": None, "sequence": None, "stream": None, "units_completed": 0}

    def check(self, now=None):
        now = self.clock() if now is None else now
        quiet = now - self.last_time
        if quiet >= self.allowance:
            raise TimeoutError(
                f"PACT replay stalled for {quiet:.3f}s (allowance {self.allowance:g}s); "
                f"last layer={self.last['layer']} sequence={self.last['sequence']} "
                f"stream={self.last['stream']}")
        return self.allowance - quiet

    def advance(self, record):
        if record["units_completed"] <= self.last["units_completed"]:
            return False
        self.check(record["monotonic_seconds"])
        self.last_time = record["monotonic_seconds"]
        self.last = record
        return True


class ReplayProgress:
    """Append every completed step and reuse the PrismaBuild wire writer."""

    def __init__(self, history, allowance, *, notify=lambda record: None, clock=None):
        self.watch = StallWatch(allowance, clock=clock)
        self.notify = notify
        self.history = Path(history)
        self.history.parent.mkdir(parents=True, exist_ok=True)
        self.last_key = None
        self._append({"event": "start", "reported_unix": time.time(),
                      "monotonic_seconds": self.watch.last_time, "pid": os.getpid()})

    def _append(self, record):
        with self.history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def check(self, label=None):
        self.watch.check()

    def __call__(self, phase, unit):
        from prismaquant.prismabuild_progress import report, RECORD_SCHEMA

        if (phase, unit) == self.last_key:
            return
        now = self.watch.clock()
        self.watch.check(now)
        match = re.fullmatch(r"layer-(\d+)-stream-(.+)", phase)
        layer = int(match[1]) if match else self.watch.last["layer"]
        stream = match[2] if match else self.watch.last["stream"]
        sample = re.fullmatch(r"complete (?:band sequence|teacher score) (\d+)", unit)
        sequence = int(sample[1]) if sample else self.watch.last["sequence"]
        count = self.watch.last["units_completed"] + 1
        description = f"layer={layer} sequence={sequence} stream={stream}; {unit}"
        record = {"schema": RECORD_SCHEMA, "phase": phase, "units_completed": count,
                  "unit": description, "reported_unix": time.time(),
                  "monotonic_seconds": now, "layer": layer,
                  "sequence": sequence, "stream": stream}
        self._append(record)
        report(phase, count, unit=description)
        self.watch.advance(record)
        self.last_key = (phase, unit)
        self.notify(record)
        print("PACT progress: " + json.dumps(record, sort_keys=True), flush=True)


def _set_parent_death_signal():
    """Ask the kernel to SIGKILL this worker when its parent dies.

    The worker leaves the parent process group with setsid, so the group
    kill in supervise never reaches it after a parent SIGKILL. The death
    signal fires even inside a native call that runs no Python code.
    """
    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl PR_SET_PDEATHSIG failed")


def _worker(work, sender, receiver):
    receiver.close()
    parent = os.getppid()
    os.setsid()
    _set_parent_death_signal()
    if os.getppid() != parent:
        # The parent died between fork and prctl, so the signal never
        # armed. Exit now instead of running orphaned past the watch.
        raise SystemExit(128 + signal.SIGKILL)
    try:
        work(sender.send)
        sender.send({"event": "complete", "monotonic_seconds": time.monotonic()})
    finally:
        sender.close()


def supervise(work, allowance):
    """Stop a blocked Python or native operation after proven silence.

    Fork before the command imports Torch or creates numerical workers.
    The parent waits on the progress pipe and the process sentinel.
    No numerical computation or scientific output moves to the parent.
    """
    watch = StallWatch(allowance)
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(work, sender, receiver))
    process.start()
    sender.close()
    previous = signal.getsignal(signal.SIGTERM)

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    try:
        pipe_open = True
        completed = False
        while True:
            remaining = max(0.0, watch.allowance - (watch.clock() - watch.last_time))
            ready = wait([receiver, process.sentinel] if pipe_open else [process.sentinel],
                         timeout=remaining)
            if pipe_open and receiver in ready:
                try:
                    record = receiver.recv()
                    if record.get("event") == "complete":
                        watch.check(record["monotonic_seconds"])
                        completed = True
                    else:
                        watch.advance(record)
                except EOFError:
                    pipe_open = False
                continue
            if process.sentinel in ready:
                process.join()
                if process.exitcode:
                    raise SystemExit(process.exitcode if process.exitcode > 0 else 128 - process.exitcode)
                if not completed:
                    raise RuntimeError("The replay worker exited without a completion record")
                return
            # Drain queued progress before a silence verdict. Parent delay
            # cannot erase work that the numerical process completed.
            watch.check()
    finally:
        signal.signal(signal.SIGTERM, previous)
        # Kill the isolated group even if the direct worker already exited.
        # This also stops native calls that cannot process Python signals.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.is_alive():
                process.kill()
        process.join()
        receiver.close()
        process.close()
