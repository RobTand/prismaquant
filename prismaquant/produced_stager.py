"""One background thread for a bound owner's PrismaBuild calls (#895).

``StreamedBoundaryArtifacts`` stages its own boundary entries through
PrismaBuild. Every one of those calls used to run on the thread that drives
the GPU, and at production group size (64 entries of 16 MiB) that left the
GPU busy 18 percent of the time. This module is the other thread.

It is a serial executor with two lanes and nothing else. It does not know
what a group is; the owner hands it closures and keeps every decision.

* **Lane 1** holds the work whose order matters to the owner: a read that the
  compute thread is waiting on (``URGENT``) and the retirement asks of a
  window that just closed (``ORDERED``). First in, first out.
* **Lane 2** holds ``OPTIONAL`` work nobody is waiting for: publishing a group
  when its last entry lands, staging a plane ahead, claiming the next group's
  prewrite, reclaiming a durable charge. First in, first out, and only when
  lane 1 is empty. It is bounded: a full lane blocks the submitter, which is
  the backpressure that keeps the owner from queueing work faster than
  PrismaBuild takes it.

One task runs at a time, because PrismaBuild serializes one owner's mutating
calls on the output prefix's ownership lock anyway (``posix_lock.held`` takes
a per-path ``threading.RLock`` before the file lock). A second thread would
wait on that lock and gain nothing. The cost is stated rather than hidden: an
urgent task waits for the task in flight to finish, and ``urgent_delay_s``
measures it.

That cost is only acceptable while the task in flight is working. An
optional task that finds it must wait for something outside this owner (a
local export that has not landed, say) returns ``Requeue(delay_s)`` instead
of sleeping: it leaves the lane, goes back to the end of lane 2, and is not
taken again for ``delay_s``. Lane 1 runs in the meantime (PQ #989). A
deferred task is still queued for ``pending``, ``drain`` and
``wait_keys_idle``, and ``close`` drops it like any other queued optional
task.
"""
from __future__ import annotations

from collections import deque
import threading
import time

URGENT = "urgent"
ORDERED = "ordered"
OPTIONAL = "optional"


class StagerClosed(RuntimeError):
    """The stager takes no more work; the caller runs the step itself."""


class Requeue:
    """An optional task's answer: run me again in ``delay_s``, off the lane.

    Returned, not raised, by the call of an ``OPTIONAL`` task that has to
    wait on something it does not own. The task keeps its identity, its
    keys and its waiter; nothing it counted is released, so a task returns
    this only before it has taken anything it would have to give back.
    """

    __slots__ = ("delay_s",)

    def __init__(self, delay_s):
        delay_s = float(delay_s)
        if not delay_s > 0.0:
            raise ValueError("a requeue delay must be positive")
        self.delay_s = delay_s


class StagerTask:
    """One submitted closure, with its outcome once it has run."""

    __slots__ = ("call", "kind", "label", "keys", "on_drop", "keep_on_close",
                 "waited", "submitted", "started", "finished", "result",
                 "error", "dropped", "not_before", "requeues", "busy_s",
                 "_done")

    def __init__(self, call, *, kind, label, keys, on_drop, keep_on_close,
                 waited):
        self.call = call
        self.kind = kind
        self.label = label
        self.keys = frozenset(keys)
        self.on_drop = on_drop
        self.keep_on_close = keep_on_close
        self.waited = waited
        self.submitted = time.monotonic()
        self.started = None
        self.finished = None
        self.result = None
        self.error = None
        self.dropped = False
        #: A requeued task is not taken before this instant.
        self.not_before = None
        #: How many times the task gave the lane back (``Requeue``).
        self.requeues = 0
        #: Time spent running, over every run; the deferrals are not in it.
        self.busy_s = 0.0
        self._done = threading.Event()

    def wait(self, timeout=None):
        """The task's result, or its own exception re-raised as it was."""

        if not self._done.wait(timeout):
            raise TimeoutError(
                f"stager task {self.label!r} did not finish within {timeout}s")
        if self.dropped:
            raise StagerClosed(
                f"stager task {self.label!r} was dropped at close")
        if self.error is not None:
            raise self.error
        return self.result


class ProducedStager:
    """A two-lane serial executor on one daemon thread."""

    def __init__(self, *, name, capacity, poll=None, poll_s=2.0,
                 on_done=None, on_error=None):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("stager capacity must be a positive int")
        self._cond = threading.Condition()
        self._lane1: deque = deque()
        self._lane2: deque = deque()
        self._inflight = None
        self._closing = False
        self._dead = None
        self._stranded = ()
        self._stranded_error = None
        self._capacity = capacity
        self._poll = poll
        self._poll_s = float(poll_s)
        self._next_poll = time.monotonic() + self._poll_s
        self._on_done = on_done
        self._on_error = on_error
        self._late_drop_errors = []
        self.started = time.monotonic()
        self.queue_peak = 0
        self.requeues = 0
        self._thread = threading.Thread(target=self._run, name=name,
                                        daemon=True)
        self._thread.start()

    # -- submitting --------------------------------------------------------

    @property
    def ident(self):
        return self._thread.ident

    def alive(self):
        return self._thread.is_alive()

    def death(self):
        """The exception that ended the worker, or None.

        A worker that dies takes the whole lane with it: the tasks it
        strands are dropped here and the owner has to hear about it, or a
        durable charge nobody gives back is lost in silence (PQ #959).
        The labels of the stranded tasks and any failure from their drop
        callbacks are attached to this exception as notes.
        """

        with self._cond:
            return self._dead

    def stranded(self):
        """The labels of the tasks the worker's death dropped, in order."""

        with self._cond:
            return self._stranded

    def submit(self, call, *, kind, label, keys=(), on_drop=None,
               keep_on_close=False, waited=False):
        """Queue one closure. Blocks only while the optional lane is full."""

        task = StagerTask(call, kind=kind, label=label, keys=keys,
                          on_drop=on_drop, keep_on_close=keep_on_close,
                          waited=waited)
        with self._cond:
            while True:
                if self._closing or self._dead is not None or (
                        not self._thread.is_alive()):
                    raise StagerClosed("the produced-output stager is closed")
                if kind != OPTIONAL or len(self._lane2) < self._capacity:
                    break
                self._cond.wait(1.0)
            (self._lane2 if kind == OPTIONAL else self._lane1).append(task)
            self.queue_peak = max(self.queue_peak,
                                  len(self._lane1) + len(self._lane2))
            self._cond.notify_all()
        return task

    def pending(self):
        with self._cond:
            return (len(self._lane1) + len(self._lane2)
                    + (0 if self._inflight is None else 1))

    def drain(self, timeout=None):
        """Wait until nothing is queued or running. True when that held."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._lane1 or self._lane2 or self._inflight is not None:
                if not self._thread.is_alive():
                    return False
                remaining = (None if deadline is None
                             else deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(1.0 if remaining is None
                                else min(remaining, 1.0))
        return True

    def wait_keys_idle(self, keys, *, labels=None, timeout=None):
        """Wait until no queued or running task names one of ``keys``."""

        keys = frozenset(keys)
        deadline = None if timeout is None else time.monotonic() + timeout

        def busy():
            tasks = list(self._lane1) + list(self._lane2)
            if self._inflight is not None:
                tasks.append(self._inflight)
            return any(task.keys & keys
                       and (labels is None or task.label in labels)
                       for task in tasks)

        with self._cond:
            while busy():
                if not self._thread.is_alive():
                    return False
                remaining = (None if deadline is None
                             else deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(1.0 if remaining is None
                                else min(remaining, 1.0))
        return True

    def close(self, *, timeout):
        """Stop taking work, drop what nobody needs, finish the rest, join.

        Optional tasks that have not started are dropped, except those
        marked ``keep_on_close``: a publication nobody will read is not
        worth a mover, but a durable charge still has to be given back.
        Returns True when the thread ended inside ``timeout``.
        """

        deadline = time.monotonic() + timeout
        with self._cond:
            self._closing = True
            kept, dropped = deque(), []
            for task in self._lane2:
                (kept if task.keep_on_close else dropped).append(task)
            self._lane2 = kept
            self._cond.notify_all()
        errors = []
        for task in dropped:
            task.dropped = True
            task.finished = time.monotonic()
            try:
                if task.on_drop is not None:
                    task.on_drop()
            except BaseException as exc:                  # noqa: BLE001
                errors.append(exc)
            finally:
                task._done.set()
        self._thread.join(max(0.0, deadline - time.monotonic()))
        with self._cond:
            # A task that asked to be requeued after close began is dropped
            # on the worker thread; its callback's failure is raised here.
            errors.extend(self._late_drop_errors)
            self._late_drop_errors = []
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"another stager drop callback failed: {error!r}")
            raise errors[0]
        return not self._thread.is_alive()

    # -- the thread --------------------------------------------------------

    def _take(self):
        """The next task, a poll when one is due, or None to stop."""

        with self._cond:
            while True:
                if self._lane1:
                    task = self._lane1.popleft()
                    break
                now = time.monotonic()
                if (self._poll is not None and not self._closing
                        and now >= self._next_poll):
                    self._next_poll = now + self._poll_s
                    task = StagerTask(self._poll, kind=ORDERED, label="poll",
                                      keys=(), on_drop=None,
                                      keep_on_close=False, waited=False)
                    break
                task = self._take_optional(now)
                if task is not None:
                    break
                if self._closing and not self._lane2:
                    return None
                timeout = None
                if self._poll is not None and not self._closing:
                    timeout = max(min(self._next_poll - now, self._poll_s),
                                  0.01)
                wake = min((queued.not_before for queued in self._lane2
                            if queued.not_before is not None), default=None)
                if wake is not None:
                    until = max(wake - now, 0.001)
                    timeout = until if timeout is None else min(timeout,
                                                                until)
                self._cond.wait(timeout)
            self._inflight = task
            self._cond.notify_all()
            return task

    def _take_optional(self, now):
        """The first lane-2 task not deferred past ``now``, or None."""

        for index, task in enumerate(self._lane2):
            if task.not_before is None or task.not_before <= now:
                del self._lane2[index]
                return task
        return None

    def _requeue(self, task):
        """Put a task that answered ``Requeue`` back on lane 2.

        True when the task was requeued or dropped here, so the caller does
        not finish it. The move from in-flight to queued happens under one
        hold of the condition: nothing that waits on the lanes ever sees
        the task in neither place. After ``close`` began, a task close would
        have dropped is dropped the same way instead.
        """

        answer = task.result
        if not isinstance(answer, Requeue):
            return False
        task.result = None
        if task.kind != OPTIONAL:
            task.error = TypeError(
                f"stager task {task.label!r} asked to be requeued from the "
                f"{task.kind} lane; only an optional task may wait off the "
                "lane")
            return False
        task.requeues += 1
        with self._cond:
            self._inflight = None
            self.requeues += 1
            if not self._closing or task.keep_on_close:
                task.not_before = task.finished + answer.delay_s
                self._lane2.append(task)
                self._cond.notify_all()
                return True
            self._cond.notify_all()
        task.dropped = True
        try:
            if task.on_drop is not None:
                task.on_drop()
        except BaseException as exc:                      # noqa: BLE001
            with self._cond:
                self._late_drop_errors.append(exc)
        finally:
            task._done.set()
        return True

    def _run(self):
        try:
            while True:
                task = self._take()
                if task is None:
                    return
                task.started = time.monotonic()
                try:
                    task.result = task.call()
                except BaseException as exc:                # noqa: BLE001
                    task.error = exc
                task.finished = time.monotonic()
                task.busy_s += max(task.finished - task.started, 0.0)
                if self._requeue(task):
                    continue
                try:
                    if task.error is not None and not task.waited and (
                            self._on_error is not None):
                        self._on_error(task, task.error)
                    if self._on_done is not None:
                        self._on_done(task)
                finally:
                    with self._cond:
                        self._inflight = None
                        self._cond.notify_all()
                    task._done.set()
        except BaseException as exc:                        # noqa: BLE001
            with self._cond:
                self._dead = exc
            raise
        finally:
            with self._cond:
                orphans = list(self._lane1) + list(self._lane2)
                self._lane1.clear()
                self._lane2.clear()
                self._inflight = None
                self._stranded = tuple(task.label for task in orphans)
                self._cond.notify_all()
            # A stranded task gets what ``close`` gives a dropped one: its
            # own drop callback, so the bookkeeping a durable charge or a
            # refusal count depends on still happens, and its waiter is
            # released. ``keep_on_close`` is no exemption here -- close
            # keeps those tasks because the worker can still RUN them, and
            # a dead worker cannot (PQ #959).
            errors = []
            for task in orphans:
                task.dropped = True
                task.finished = time.monotonic()
                try:
                    if task.on_drop is not None:
                        task.on_drop()
                except BaseException as exc:              # noqa: BLE001
                    errors.append(exc)
                finally:
                    task._done.set()
            self._note_stranded(errors)

    def _note_stranded(self, errors):
        """Preserve the first drop failure where the owner can read it.

        This runs on a thread that is ending, so there is nowhere to
        raise: the first failure is kept in ``_stranded_error`` with the
        rest attached to it as notes -- the shape ``close`` uses for the
        error it raises -- and the death itself gets a note naming what
        it stranded. The owner reads both when it stops the stager.
        """

        if errors:
            for error in errors[1:]:
                errors[0].add_note(
                    f"another stager drop callback failed: {error!r}")
        with self._cond:
            if errors and self._stranded_error is None:
                self._stranded_error = errors[0]
            dead, stranded = self._dead, self._stranded
        if dead is None:
            return
        if stranded:
            dead.add_note(
                f"the stager worker died holding {len(stranded)} queued "
                f"task(s), dropped here: {', '.join(stranded)}")
        if errors:
            dead.add_note(
                f"a stager drop callback failed: {errors[0]!r}")

    def stranded_error(self):
        """The first drop-callback failure from the death path, or None."""

        with self._cond:
            return self._stranded_error


class OwnerLock:
    """The owner's bookkeeping lock, which a thread gives up around a call.

    ``held`` nests on the thread that holds it. ``yielded`` releases it for
    the length of a PrismaBuild call and takes it back, so bookkeeping is
    never locked across a call that can take seconds: decide under the lock,
    call without it, commit under it again. On a thread that does not hold
    the lock, ``yielded`` does nothing.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None

    def held(self):
        return _Held(self)

    def yielded(self):
        return _Yielded(self)


class _Held:
    __slots__ = ("_outer", "_took")

    def __init__(self, outer):
        self._outer = outer
        self._took = False

    def __enter__(self):
        outer = self._outer
        me = threading.get_ident()
        if outer._owner != me:
            outer._lock.acquire()
            outer._owner = me
            self._took = True
        return self

    def __exit__(self, *_exc):
        if self._took:
            self._outer._owner = None
            self._outer._lock.release()
        return False


class _Yielded:
    __slots__ = ("_outer", "_gave")

    def __init__(self, outer):
        self._outer = outer
        self._gave = False

    def __enter__(self):
        outer = self._outer
        if outer._owner == threading.get_ident():
            outer._owner = None
            outer._lock.release()
            self._gave = True
        return self

    def __exit__(self, *_exc):
        if self._gave:
            self._outer._lock.acquire()
            self._outer._owner = threading.get_ident()
        return False


__all__ = ["URGENT", "ORDERED", "OPTIONAL", "StagerClosed", "Requeue",
           "StagerTask", "ProducedStager", "OwnerLock"]
