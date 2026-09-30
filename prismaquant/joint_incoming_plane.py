"""Ordered verified incoming-plane consumption, shared by checkpoint and handoff.

This is an adapter over the existing exact-entry stream, not another reader,
cache, pool or tensor residency mechanism.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import time

import torch


class IncomingPlaneStream:
    """One probe's consecutive verified rows; layout queries perform no reads."""

    def __init__(self, probe, entries, stream, *, coordinate_of,
                 label="checkpoint", refused: type[Exception] = RuntimeError):
        self.probe = int(probe)
        self._entries = list(entries)
        self._stream = stream
        self._coordinate_of = coordinate_of
        self._next = 0
        self._layouts = {}
        for batch, entry in enumerate(self._entries):
            dtype = getattr(torch, str(entry["dtype"]).removeprefix("torch."), None)
            if not isinstance(dtype, torch.dtype):
                raise refused(f"{label} entry {entry['name']!r} has no torch dtype")
            self._layouts[(self.probe, batch)] = (tuple(entry["shape"]), dtype)
        self.telemetry = {"probe": self.probe, "entries": 0,
                          "tensor_bytes": 0, "wait_s": 0.0}

    def layout(self, key):
        try:
            return self._layouts[tuple(key)]
        except KeyError:
            raise RuntimeError(
                f"probe {self.probe}'s incoming stream has no row {key!r}") from None

    def take(self, keys):
        rows = []
        for key in keys:
            key = tuple(key)
            if self._next >= len(self._entries) or key != (self.probe, self._next):
                raise RuntimeError(
                    f"probe {self.probe}'s incoming stream is read in capture "
                    f"order: row {key!r} was asked for, the next is "
                    f"{(self.probe, self._next)!r}")
            started = time.monotonic()
            entry, tensor = next(self._stream)
            self.telemetry["wait_s"] += time.monotonic() - started
            if entry is not self._entries[self._next] or \
                    tuple(self._coordinate_of(entry))[:2] != key:
                raise RuntimeError(
                    f"probe {self.probe}'s incoming stream yielded "
                    f"{entry.get('name')!r} for row {key!r}")
            self._next += 1
            self.telemetry["entries"] += 1
            self.telemetry["tensor_bytes"] += int(entry["tensor_bytes"])
            rows.append(tensor)
            tensor = None
        return rows

    def materialize(self, plane):
        while self._next < len(self._entries):
            key = (self.probe, self._next)
            (plane[key],) = self.take([key])

    @property
    def exhausted(self):
        return self._next == len(self._entries)


@contextmanager
def open_incoming_stream(probe, entries, *, session, max_resident_bytes,
                         residency_check, stream_factory) -> Iterator[IncomingPlaneStream]:
    """Join/release exact reads on every exit; require clean-pass exhaustion."""
    from .joint_adjoint_checkpoints import stream_exact_entry_tensors

    generator = stream_exact_entry_tensors(
        entries, expected_session=session, max_resident_bytes=int(max_resident_bytes),
        residency_check=residency_check, deadline_per_window=True)
    primary = None
    try:
        stream = stream_factory(probe, entries, generator)
        try:
            yield stream
        except BaseException as error:
            primary = error
            raise
    finally:
        try:
            generator.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note(f"incoming stream cleanup failed: {type(cleanup).__name__}: {cleanup}")
    if not stream.exhausted:
        raise RuntimeError(
            f"probe {int(probe)}'s pass left "
            f"{len(entries) - stream.telemetry['entries']} incoming rows unread")
