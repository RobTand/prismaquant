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

The campaign's default backend hands those writes to one bounded writer
thread. An explicit ``submit_task=ENGINE.submit`` backend instead runs finite
FIFO drains on the existing shared IO engine; it creates no additional pool
or writer thread. Stage B's CPU slice exposes this backend without yet changing
retained-window execution or production defaults. What this module does **not** do:

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
* **The budget is reserved before the bytes are made, not after.**  The caller
  calls :meth:`reserve` with the size it is about to stage, and that call is
  what blocks; only then does it allocate.  Charging at submit time would have
  left the producer holding one more artifact than the budget allows, because
  the copy it is about to hand over already exists by then.  An artifact
  larger than the whole budget is refused rather than admitted as a special
  case: admitting it would mean the declared bound is not the bound, and the
  answer to a render that does not fit is a bigger budget, chosen by whoever
  is accounting for the memory.
* One active writer and a FIFO queue, so jobs and completions keep submission
  order. A shared-backend publisher also bounds reserved, queued, running and
  uncollected completion slots. Consumers collect completions before staging
  another batch. Each shared dispatch publishes at most ``max_jobs`` jobs;
  a completion callback rearms remaining work behind already queued IO.
  An idle drain returns its worker. Recovery stays deterministic.

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
from concurrent.futures import CancelledError, Future
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


def _clear_failure_tracebacks(error: BaseException) -> None:
    """Keep failure types/messages without retaining staged bytes in frames."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        pending.extend(link for link in (current.__cause__, current.__context__)
                       if link is not None)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


class BoundedPublisher:
    """Bounded FIFO publication with consumer-owned durable acknowledgements.

    By default, retain the campaign's existing writer thread. ``submit_task``
    selects a shared execution backend (``ENGINE.submit`` for Stage B). It
    requires ``max_jobs`` and runs finite drains, never idle condition waits
    on shared workers. The job bound includes uncollected completions, so
    consumers must collect acknowledgements before requesting more slots.
    """

    def __init__(self, *, budget_bytes: int, name: str = "tessera-publication",
                 submit_task: Callable[[Callable[[], None]], Future] | None = None,
                 max_jobs: int | None = None):
        budget = int(budget_bytes)
        if budget <= 0:
            raise ValueError(
                "a publisher needs a positive byte budget; the synchronous "
                "path is no publisher at all, not a publisher of size zero")
        if submit_task is not None and not callable(submit_task):
            raise TypeError("publisher submit_task must be callable")
        if max_jobs is not None and (type(max_jobs) is not int or max_jobs <= 0):
            raise ValueError("publisher max_jobs must be a positive integer")
        if submit_task is not None and max_jobs is None:
            raise ValueError("a shared-backend publisher requires max_jobs")
        self._budget = budget
        self._submit_task = submit_task
        self._max_jobs = max_jobs
        self._reserved_jobs = 0
        self._task_active = False
        self._tasks: set[Future] = set()
        self._cond = threading.Condition()
        self._queued: deque[PublicationJob] = deque()
        self._done: deque[Hashable] = deque()
        self._charged = 0
        self._reserved = 0
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
        self._thread = None
        if submit_task is None:
            self._thread = threading.Thread(target=self._run, name=name, daemon=True)
            self._thread.start()

    # -- caller side --------------------------------------------------------

    def reserve(self, nbytes: int, *, jobs: int = 1) -> None:
        """Take room for bytes that do not exist yet, blocking until it fits.

        A single-job reservation is followed by one :meth:`submit` or
        :meth:`release`. A batch reserves ``jobs`` slots before encoding,
        submits those jobs, then releases unused bytes with ``jobs=0``.
        Consume acknowledgements before requesting slots they still hold;
        an uncollected completion deliberately retains its metadata credit.
        A single consumer uses :meth:`drain` before reserving past a full job
        ceiling, including running slots: calling ``completed()`` before they
        finish does not let that consumer collect them while blocked here.
        """
        charge = int(nbytes)
        if charge < 0:
            raise ValueError("cannot reserve negative bytes")
        if type(jobs) is not int or jobs <= 0:
            raise ValueError("a reservation needs a positive job count")
        if self._max_jobs is not None and jobs > self._max_jobs:
            raise PublicationError("one reservation exceeds the publisher job limit")
        if charge > self._budget:
            raise PublicationError(
                f"one artifact of {charge} bytes does not fit a publication "
                f"budget of {self._budget}; raise the caller's byte ceiling "
                "or publish synchronously. Admitting it would mean the "
                "declared bound is not the bound.")
        with self._cond:
            self._raise_failure()
            if self._closing:
                raise PublicationError("publisher is closed; nothing more can be staged")
            waited = time.monotonic()
            while (self._charged + charge > self._budget
                   or (self._max_jobs is not None
                       and self._reserved_jobs + self._outstanding
                       + len(self._done) + jobs > self._max_jobs)):
                self._cond.wait()
                self._raise_failure()
                if self._closing:
                    raise PublicationError("publisher is closed; nothing more can be staged")
            self._submit_blocked_seconds += time.monotonic() - waited
            self._charged += charge
            self._reserved += charge
            if self._max_jobs is not None:
                self._reserved_jobs += jobs
            if self._charged > self._peak_charged_bytes:
                self._peak_charged_bytes = self._charged

    def release(self, nbytes: int, *, jobs: int = 1) -> None:
        """Return unstaged credit; ``jobs=0`` returns a batch's leftover bytes."""
        charge = int(nbytes)
        if charge < 0 or type(jobs) is not int or jobs < 0:
            raise ValueError("cannot release negative bytes or job counts")
        with self._cond:
            if self._failure is not None:
                return  # failure already cancelled every reservation
            if charge > self._reserved or (self._max_jobs is not None
                                           and jobs > self._reserved_jobs):
                raise PublicationError("release exceeds the unstaged reservation")
            self._charged -= charge
            self._reserved -= charge
            if self._max_jobs is not None:
                self._reserved_jobs -= jobs
            self._cond.notify_all()

    def submit(self, job: PublicationJob) -> None:
        """Hand over one unit's staged bytes against an existing reservation."""
        charge = int(job.charged_bytes)
        with self._cond:
            self._raise_failure()
            if self._closing:
                raise PublicationError("publisher is closed; nothing more can be staged")
            if charge > self._reserved:
                raise PublicationError(
                    f"{charge} bytes were submitted against {self._reserved} "
                    "reserved; every charged job reserves before it stages")
            if self._max_jobs is not None and self._reserved_jobs < 1:
                raise PublicationError("every bounded job reserves its slot before staging")
            self._reserved -= charge
            if self._max_jobs is not None:
                self._reserved_jobs -= 1
            self._queued.append(job)
            self._outstanding += 1
            if self._submit_task is not None and not self._task_active:
                self._dispatch_locked()
                self._raise_failure()
            self._cond.notify_all()

    def completed(self) -> list:
        """Take the keys published since the last call, in publication order.

        This does not raise on a writer failure.  The keys it returns are files
        that exist, and a caller that journals them is recording work that was
        really done; the failure is raised by the next :meth:`reserve`,
        :meth:`submit` or :meth:`drain`, and callers that must notice it without either can read
        :attr:`failure`.
        """
        with self._cond:
            out = list(self._done)
            self._done.clear()
            self._cond.notify_all()
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
            self._cond.notify_all()
            return out

    def cancel_pending(self) -> list:
        """Stop admission/drop queued tail without releasing a running charge.

        The caller retains its own primary exception. An internal cancellation
        marker wakes blocked producers without retaining that caller's graph.
        Close/join before collecting the genuinely published prefix; drain
        raises rather than representing a cancelled submission as success.
        """
        with self._cond:
            self._closing = True
            keys = [job.key for job in self._queued]
            charge = sum(int(job.charged_bytes) for job in self._queued)
            count = len(self._queued)
            # Drop ownership before notifying reusable credit. A running job
            # is outside this deque and retains its charge until retirement.
            self._queued.clear()
            self._charged -= charge + self._reserved
            self._outstanding -= count
            self._reserved = self._reserved_jobs = 0
            if self._failure is None:
                self._failure = CancelledError("publication admission cancelled")
            self._cond.notify_all()
            return keys

    def close(self) -> None:
        """Publish what is queued, then join owned execution tasks.

        The writer keeps taking jobs until the queue is empty, so a close on
        the way out of a failed run still lands the bytes that were already
        computed.  It reports nothing and raises nothing: use :meth:`drain`
        when the completions matter, and :meth:`close` when what matters is
        that the thread is gone and nothing was abandoned half written.  After
        a writer failure the queued tail is cancelled, but close still waits
        for its owning task to retire.
        """
        with self._cond:
            self._closing = True
            self._cond.notify_all()
            if self._thread is None:
                # A completion callback may rearm another bounded drain. Wait
                # for the entire owned chain, not a snapshot of its first task.
                while self._tasks:
                    self._cond.wait()
        if self._thread is not None:
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
            result = {
                "schema": SCHEMA,
                "budget_bytes": int(self._budget),
                "published": int(self._published),
                "peak_charged_bytes": int(self._peak_charged_bytes),
                "submit_blocked_seconds": float(self._submit_blocked_seconds),
                "publish_seconds": float(self._publish_seconds),
                "failed": self._failure is not None,
            }
            if self._submit_task is not None:
                result.update(backend="shared", max_jobs=self._max_jobs,
                              charged_bytes=self._charged,
                              reserved_jobs=self._reserved_jobs,
                              held_jobs=self._reserved_jobs + self._outstanding
                              + len(self._done))
            return result

    # -- writer side --------------------------------------------------------

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise PublicationError(
                "a staged artifact was not published; nothing queued behind "
                "the failure was written") from self._failure

    def _fail_locked(self, error: BaseException) -> None:
        """Cancel queued ownership before returning its staging credit."""
        if self._failure is None:
            _clear_failure_tracebacks(error)
            self._failure = error
        self._queued.clear()
        self._charged = self._reserved = self._outstanding = self._reserved_jobs = 0
        self._task_active = False
        self._cond.notify_all()

    def _dispatch_locked(self) -> None:
        """Dispatch one finite drain; the caller holds the condition lock."""
        assert self._submit_task is not None
        self._task_active = True
        try:
            task = self._submit_task(self._run)
            if not isinstance(task, Future):
                raise TypeError("publisher submit_task must return a Future")
            self._tasks.add(task)
            task.add_done_callback(self._task_done)
        except BaseException as exc:
            self._fail_locked(exc)

    def _task_done(self, task: Future) -> None:
        """Retire/rearm behind queued IO and surface dispatch failure."""
        error = (CancelledError("publication task was cancelled")
                 if task.cancelled() else task.exception())
        with self._cond:
            self._tasks.discard(task)
            self._task_active = False
            if error is not None:
                self._fail_locked(error)
            elif self._queued and self._failure is None:
                self._dispatch_locked()
            self._cond.notify_all()

    def _run(self) -> None:
        processed = 0
        limit = self._max_jobs if self._submit_task is not None else None
        while limit is None or processed < limit:
            with self._cond:
                while (not self._queued and not self._closing
                       and self._submit_task is None):
                    self._cond.wait()
                if not self._queued:
                    # Shared ownership retires only in _task_done, which also
                    # handles a producer enqueueing before Future completion.
                    if self._submit_task is None:
                        self._task_active = False
                    self._cond.notify_all()
                    return
                job = self._queued.popleft()
            started = time.monotonic()
            try:
                job.publish()
            except BaseException as exc:  # noqa: BLE001 - recorded and re-raised
                _clear_failure_tracebacks(exc)
                del job
                with self._cond:
                    self._fail_locked(exc)
                return
            elapsed = time.monotonic() - started
            key, charge = job.key, int(job.charged_bytes)
            # Drop staged ownership BEFORE the producer can reuse its credit.
            del job
            with self._cond:
                self._done.append(key)
                self._charged -= charge
                self._outstanding -= 1
                self._published += 1
                self._publish_seconds += elapsed
                self._cond.notify_all()
            processed += 1
