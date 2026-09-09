"""Bounded, ordered publication of artifacts the campaign has already made.

The Tessera campaign encodes a batch of units on the GPU, scores each one, and
then writes two files per unit: the rendered weight shard through
:func:`prismaquant.production_weight_cache._store_rendered_weight_entry`, and
the ``.tessera`` wire blob beside it.  Both writes happen on the thread that
just finished the encode, so the next batch's encode does not start until the
previous batch's bytes are on disk.  On the R1088 endpoint a sixteen-unit arm
spends about 1.13 to 1.16 s inside sixteen ``torch.save`` calls and about
0.40 s inside sixteen ``Path.write_bytes`` calls, against roughly 54 us of
fsync: that is serialisation and copying, not device latency, and none of it
needs the GPU.

This module hands those two writes to one bounded writer thread so they can run
while the next batch encodes.  What it deliberately does **not** do:

* **It is not a second cache.**  A job calls the same
  ``_store_rendered_weight_entry`` and the same tmp-plus-``os.replace`` wire
  write the synchronous path calls, with the same arguments.  There is one
  cache mechanism and this is not another one.
* **It does not change what durable means.**  ``_store_rendered_weight_entry``
  is invoked exactly as the campaign invokes it today, so a published file is
  one that has been through ``os.replace``, and an fsync happens only where
  ``release_completed_anchor_file_pages`` is configured.  Publication here
  means *the same thing it means now*, on another thread.
* **It does not decide when a receipt is written.**  The publisher only reports
  which jobs are finished.  The caller is what turns a finished job into a
  journal row, and that is the ordering rule the campaign has to keep: an
  anchor may not reach the checkpoint before its own bytes have.

Ownership and bounds:

* The caller stages the CPU bytes -- the device-to-host copy stays on the
  thread that owns the device work -- and then transfers them.  A submitted job
  owns its tensor and its blob until it is published, and nothing else may
  write to them.
* Every job is charged ``tensor.nbytes + len(blob)`` and :meth:`submit` blocks
  once the outstanding charge would exceed the budget, so staging cannot grow
  without limit while the writer falls behind.  A single artifact larger than
  the whole budget still publishes: it waits for an empty queue rather than
  waiting for room that can never appear.
* One writer thread and a FIFO queue, so jobs run in submission order and
  completions are reported in that order.  Recovery stays deterministic.

Failure is closed.  When a job raises, the writer keeps the exception, discards
everything queued behind it *without writing any of it*, and every later
:meth:`submit` and :meth:`drain` raises :class:`PublicationError` from the
original.  Jobs that finished before the failure are real files, so
:meth:`completed` still reports them and the caller may journal them; that is
what makes the resume after a failed action skip work it has actually done
rather than redo it or, worse, trust a row whose bytes never landed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Hashable

SCHEMA = "prismaquant.tessera_publication.v1"

__all__ = [
    "SCHEMA",
    "PublicationError",
    "PublicationJob",
    "BoundedPublisher",
]


class PublicationError(RuntimeError):
    """A staged artifact did not reach the disk it was promised to."""


@dataclass(frozen=True)
class PublicationJob:
    """One unit's files, staged and ready to be written.

    ``key`` names the job to the caller; the campaign uses
    ``(qname, format_name)``.  ``charged_bytes`` is what the staged bytes cost
    while they wait.  ``publish`` performs the writes and must be safe to run
    on a thread other than the one that built it.
    """

    key: Hashable
    charged_bytes: int
    publish: Callable[[], None]

    def __post_init__(self) -> None:
        if int(self.charged_bytes) < 0:
            raise ValueError("a publication job cannot be charged negative bytes")
        if not callable(self.publish):
            raise TypeError("a publication job needs a callable to publish with")


class BoundedPublisher:
    """One writer thread, a byte budget, and order-preserving completions."""

    def __init__(self, *, budget_bytes: int, name: str = "tessera-publication"):
        budget = int(budget_bytes)
        if budget <= 0:
            raise ValueError(
                "a publisher needs a positive byte budget; the synchronous "
                "path is no publisher at all, not a publisher of size zero")
        self._budget = budget
        self._cond = threading.Condition()
        self._queued: deque[PublicationJob] = deque()
        self._done: deque[Hashable] = deque()
        self._charged = 0
        self._outstanding = 0
        self._failure: BaseException | None = None
        self._closing = False
        self._published = 0
        self._peak_charged_bytes = 0
        self._submit_blocked_seconds = 0.0
        self._publish_seconds = 0.0
        # Daemon so an unhandled exception on the main thread cannot leave the
        # interpreter waiting on a writer nobody is going to drain.  The
        # ordered shutdown is :meth:`close`, which callers run from a finally.
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # -- caller side --------------------------------------------------------

    def submit(self, job: PublicationJob) -> None:
        """Hand over one unit's staged bytes, blocking while over budget."""
        charge = int(job.charged_bytes)
        with self._cond:
            self._raise_failure()
            if self._closing:
                raise PublicationError("publisher is closed; nothing more can be staged")
            waited = time.monotonic()
            # ``self._charged`` in the guard is what admits an artifact bigger
            # than the entire budget: it waits for an empty queue, and then
            # goes, instead of waiting for room that cannot appear.
            while self._charged and self._charged + charge > self._budget:
                self._cond.wait()
                self._raise_failure()
            self._submit_blocked_seconds += time.monotonic() - waited
            self._queued.append(job)
            self._charged += charge
            self._outstanding += 1
            if self._charged > self._peak_charged_bytes:
                self._peak_charged_bytes = self._charged
            self._cond.notify_all()

    def completed(self) -> list:
        """Take the keys published since the last call, in publication order.

        This does not raise on a writer failure.  The keys it returns are files
        that exist, and a caller that journals them is recording work that was
        really done; the failure is raised by the next :meth:`submit` or
        :meth:`drain`, and callers that must notice it without either can read
        :attr:`failure`.
        """
        with self._cond:
            out = list(self._done)
            self._done.clear()
            return out

    def drain(self) -> list:
        """Wait for every submitted job, then take the completed keys."""
        with self._cond:
            while self._outstanding and self._failure is None:
                self._cond.wait()
            # Raise before taking the keys, so a caller that hits the failure
            # can still collect what did land on its way out.
            self._raise_failure()
            out = list(self._done)
            self._done.clear()
            return out

    def close(self) -> None:
        """Stop the writer.  Does not wait for the queue; drain first."""
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        self._thread.join()

    def __enter__(self) -> "BoundedPublisher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def failure(self) -> BaseException | None:
        with self._cond:
            return self._failure

    @property
    def outstanding(self) -> int:
        with self._cond:
            return self._outstanding

    def stats(self) -> dict:
        """What the run charged and how long the encode thread waited."""
        with self._cond:
            return {
                "schema": SCHEMA,
                "budget_bytes": int(self._budget),
                "published": int(self._published),
                "peak_charged_bytes": int(self._peak_charged_bytes),
                "submit_blocked_seconds": float(self._submit_blocked_seconds),
                "publish_seconds": float(self._publish_seconds),
                "failed": self._failure is not None,
            }

    # -- writer side --------------------------------------------------------

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise PublicationError(
                "a staged artifact was not published; nothing queued behind "
                "the failure was written") from self._failure

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queued and not self._closing:
                    self._cond.wait()
                if not self._queued:
                    return
                job = self._queued.popleft()
            started = time.monotonic()
            try:
                job.publish()
            except BaseException as exc:  # noqa: BLE001 - recorded and re-raised
                with self._cond:
                    if self._failure is None:
                        self._failure = exc
                    # Fail closed. Whatever was staged behind this job is
                    # dropped unwritten, and its charge with it, so a caller
                    # blocked on the budget wakes into the failure instead of
                    # waiting on a writer that has stopped.
                    self._queued.clear()
                    self._charged = 0
                    self._outstanding = 0
                    self._cond.notify_all()
                return
            elapsed = time.monotonic() - started
            with self._cond:
                self._done.append(job.key)
                self._charged -= int(job.charged_bytes)
                self._outstanding -= 1
                self._published += 1
                self._publish_seconds += elapsed
                self._cond.notify_all()
            # Release the staged tensor and blob now rather than at the next
            # iteration's pop: the budget says how much may be resident, and a
            # job held one loop longer than its charge is a job outside it.
            del job
