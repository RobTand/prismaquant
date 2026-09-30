"""Consumer-owned durable frontier over the existing bounded IO publisher.

This optional host path changes scheduling, not unit encoding or measurement
identity. IO workers receive only a destination and an owned builtin snapshot;
serialization, hashing and atomic publication run on the existing IO engine.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from copy import deepcopy
from threading import Lock
import sys

from . import aura_cost
from .io_engine import ENGINE
from .tessera_publication import BoundedPublisher, PublicationError, PublicationJob

SETTING = "checkpoint_publication_budget_bytes"
# Two pickle frame buffers, traversal/memo overhead and small encoder objects.
# Input graph objects are conservatively counted too, even when already resident.
ENCODER_FIXED_BYTES = 1 << 20
MAX_DEPTH = 64


def publication_budget(value: object) -> int:
    if value is None:
        return 0
    if type(value) is not int or value < 0:
        raise ValueError(f"{SETTING} must be a nonnegative integer")
    return value


def publication_geometry(budget: int, windows: Sequence[Mapping]) -> tuple[int, int]:
    jobs = 2 * max((len(window["names"]) for window in windows), default=1)
    jobs = max(2, jobs)
    slot = budget // jobs
    if budget and slot <= ENCODER_FIXED_BYTES:
        raise ValueError(f"{SETTING} is too small for {jobs} bounded staging slots")
    return jobs, slot


def snapshot_bound(state: object, *, limit: int) -> int:
    """Conservative bound for builtin tensor-free pickle graphs, not a profile.

    Per unique object, 32 times its Python size covers UTF-8 conversion,
    framing, payload/envelope serialization copies and producer state copies.
    Another 1024 bytes covers traversal and both pickle memo tables, including
    growth slack. The fixed allowance bounds depth-limited traversal/frames.
    Unsupported reducer objects are refused instead of invoking unbounded
    custom serialization. The capped encoder is an independent final guard.
    """
    seen: set[int] = set()
    charged = ENCODER_FIXED_BYTES

    def visit(value: object, depth: int) -> None:
        nonlocal charged
        if depth > MAX_DEPTH:
            raise ValueError("checkpoint graph exceeds bounded staging depth")
        kind = type(value)
        if kind not in (dict, list, tuple, str, bytes, int, float, bool, type(None)):
            raise ValueError("checkpoint staging requires a builtin tensor-free graph")
        if id(value) in seen:
            return
        charged += 32 * sys.getsizeof(value) + 1024
        if charged > limit:
            raise ValueError("checkpoint graph exceeds staging bytes bound")
        seen.add(id(value))
        if isinstance(value, dict):
            for key, item in value.items():
                visit(key, depth + 1)
                visit(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, depth + 1)

    visit(state, 0)
    return charged


def check_construction(*, references: object, rows: int, probes: int,
                       seed_base: int, limit: int) -> None:
    """Preflight the quantum's bounded builtin snapshot before it is built.

    Rows add three probe vectors and a zero-format component dictionary per
    probe, plus fixed row/state dictionaries. Count their objects/memo entries
    conservatively; resident identity/component graphs also cover canonical
    identity hashing and serializer copies via snapshot_bound's multiplier.
    Huge seed integers are charged per generated probe ID, not only once.
    """
    extra = rows * (65536 + probes * (16384 + 32 * sys.getsizeof(seed_base)))
    if extra + ENCODER_FIXED_BYTES >= limit:
        raise ValueError("checkpoint construction exceeds staging bytes bound")
    snapshot_bound(references, limit=limit - extra)


class _EncodedBytes:
    """Small shared accounting, never a reference to consumer state or progress."""

    def __init__(self):
        self._lock = Lock()
        self._value = 0

    def add(self, size: int) -> None:
        with self._lock:
            self._value += size

    def value(self) -> int:
        with self._lock:
            return self._value


def _publish_snapshot(path, *, name, identity_sha256, state, max_bytes, encoded_bytes):
    encoded = aura_cost._encode_aura_unit_checkpoint(
        qname=name, identity_sha256=identity_sha256, state=state,
        max_bytes=max_bytes)
    encoded_bytes.add(len(encoded))
    aura_cost.atomic_write_bytes(path, encoded)


class CheckpointPublicationLedger:
    """Bounded measured/submitted/durable state; all methods run on consumer."""

    def __init__(self, *, checkpoint_root, identity_sha256: str,
                 windows: Sequence[Mapping], completed: set[str],
                 acknowledge: Callable[[], None], window_done: Callable[[int], None],
                 budget_bytes: int):
        self._jobs, self._slot = publication_geometry(budget_bytes, windows)
        self._publisher = BoundedPublisher(
            budget_bytes=budget_bytes, max_jobs=self._jobs,
            submit_task=ENGINE.submit, name="stage-b-unit-checkpoint")
        self._root = checkpoint_root
        self._identity = identity_sha256
        self._completed = completed
        self._acknowledge = acknowledge
        self._window_done = window_done
        self._submitted: set[str] = set()
        self._windows: deque[tuple[int, tuple[str, ...]]] = deque()
        self._next_window = 0
        self._closed = False
        self._submitted_count = self._acknowledged_count = 0
        self._encoded_bytes = _EncodedBytes()
        self._peak_windows = 0

    def _accept(self, keys) -> None:
        for name in keys:
            if name not in self._submitted:
                raise PublicationError(f"unstaged checkpoint acknowledgement: {name}")
            self._submitted.remove(name)
            self._completed.add(name)
            self._acknowledged_count += 1
        if keys:
            self._acknowledge()
        while self._windows and all(name in self._completed
                                    for name in self._windows[0][1]):
            index, _names = self._windows.popleft()
            self._window_done(index)

    def poll(self) -> None:
        self._accept(self._publisher.completed())
        if self._publisher.failure is not None:
            raise PublicationError("retained checkpoint publication failed") from self._publisher.failure

    def start_window(self, index: int, names: Sequence[str]) -> None:
        self.poll()
        if index != self._next_window:
            raise PublicationError("checkpoint windows must be registered in sealed order")
        # Includes skipped/resumed windows: metadata cannot grow behind a
        # held predecessor even when the later window submits no new units.
        if len(self._windows) == 2:
            self.flush()
        self._windows.append((index, tuple(names)))
        self._next_window += 1
        self._peak_windows = max(self._peak_windows, len(self._windows))
        self._accept([])

    def _freeze(self, name: str, state_factory: Callable[[int], Mapping]) -> PublicationJob:
        state = state_factory(self._slot)
        snapshot_bound(state, limit=self._slot)
        # Builtin validation above prevents custom reducers. deepcopy preserves
        # aliases/cycles and therefore the existing pickle byte representation,
        # without lending mutable rows or identity graphs to an IO worker.
        # The reservation includes this copy, encoder buffers and memo tables.
        owned = deepcopy(state)
        return PublicationJob(
            name, self._slot,
            partial(_publish_snapshot,
                    aura_cost._aura_unit_checkpoint_path(self._root, name),
                    name=name, identity_sha256=self._identity, state=owned,
                    max_bytes=self._slot // 4, encoded_bytes=self._encoded_bytes))

    def submit(self, name: str, state_factory: Callable[[int], Mapping]) -> bool:
        self.poll()
        if name in self._completed or name in self._submitted:
            return False
        # Completion frees byte ownership but not its acknowledgement slot.
        # The single consumer drains/reaps before reserving a full ceiling.
        if len(self._submitted) == self._jobs:
            self.flush()
        self._publisher.reserve(self._slot)
        try:
            self._publisher.submit(self._freeze(name, state_factory))
        except BaseException:
            self._publisher.release(self._slot)
            raise
        self._submitted.add(name)
        self._submitted_count += 1
        return True

    def flush(self) -> None:
        try:
            keys = self._publisher.drain()
        except BaseException:
            self._accept(self._publisher.completed())
            raise
        self._accept(keys)
        self.poll()

    def cancel(self) -> None:
        """Stop the queued tail, join running write, retain only real prefix."""
        self._publisher.cancel_pending()
        self._publisher.close()
        self._accept(self._publisher.completed())
        self._submitted.clear()
        self._windows.clear()
        self._closed = True

    def close(self) -> None:
        self._publisher.close()
        self._closed = True

    def stats(self) -> dict:
        return {**self._publisher.stats(),
                "schema": "prismaquant.stage_b_checkpoint_publication.v1",
                "slot_bytes": self._slot,
                "windows_pending_peak": self._peak_windows,
                "submitted_units": self._submitted_count,
                "acknowledged_units": self._acknowledged_count,
                "encoded_bytes": self._encoded_bytes.value(),
                "pending_units": len(self._submitted), "closed": self._closed}
