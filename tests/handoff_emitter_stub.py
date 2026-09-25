"""A handoff emitter double for quanta that emit (PQ #1251).

Since PQ #1251 the quantum core starts its emitter's writer with
``stream(...)`` before its passes, reports each slot a final pass stores
through ``mark_final``, and calls ``finish(cotangent_owners)`` in the
tail, under the ``handoff-out`` span. :class:`StubHandoffEmitter` keeps
that protocol without writing anything: ``finish`` calls ``on_finish``
with the keyword arguments the serial ``emit`` took, where the core used
to call ``emit``. It also refuses what the real stream refuses: a key
outside the plane, a key stored twice by a final pass, and a finish
before every key is final.
"""
from __future__ import annotations


class StubHandoffEmitter:
    """``capture_batch``, ``published`` and ``export_report`` as the real one has."""

    def __init__(self, *, capture_batch, on_finish, published=None):
        self.capture_batch = capture_batch
        self.published = {} if published is None else published
        self.export_report = None
        self._on_finish = on_finish
        self.streams = []

    def stream(self, *, grad_plane, n_probes, n_batches, kda_capture_kernel):
        if self.streams:
            raise RuntimeError("a quantum emits its handoff once")
        stream = _StubStream(self, grad_plane=grad_plane, n_probes=int(n_probes),
                             n_batches=int(n_batches),
                             kda_capture_kernel=kda_capture_kernel)
        self.streams.append(stream)
        return stream

    def emit(self, *, grad_plane, cotangent_owners, n_probes, n_batches,
             kda_capture_kernel):
        with self.stream(grad_plane=grad_plane, n_probes=n_probes,
                         n_batches=n_batches,
                         kda_capture_kernel=kda_capture_kernel) as stream:
            stream.mark_final([(probe, batch) for probe in range(int(n_probes))
                               for batch in range(int(n_batches))])
            return stream.finish(cotangent_owners)


class _StubStream:
    def __init__(self, emitter, *, grad_plane, n_probes, n_batches,
                 kda_capture_kernel):
        self._emitter = emitter
        self._kwargs = {"grad_plane": grad_plane, "n_probes": n_probes,
                        "n_batches": n_batches,
                        "kda_capture_kernel": kda_capture_kernel}
        self._expected = {(probe, batch) for probe in range(n_probes)
                          for batch in range(n_batches)}
        self.final = []
        self.tee_slots = 0
        self.finished = False
        self.telemetry = {"stub": True}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def attach_tee(self, slots):
        self.tee_slots = len(slots)

    def mark_final(self, keys, rows=None):
        for key in keys:
            key = (int(key[0]), int(key[1]))
            if key not in self._expected:
                raise RuntimeError(f"handoff key {key!r} is outside the plane")
            if key in self.final:
                raise RuntimeError(f"a final pass stored handoff key {key!r} twice")
            self.final.append(key)

    def finish(self, cotangent_owners):
        missing = sorted(self._expected - set(self.final))
        if missing:
            raise RuntimeError(f"{len(missing)} handoff key(s) were never stored "
                               f"by a final pass, first {missing[0]!r}")
        self.finished = True
        self._emitter._on_finish(cotangent_owners=cotangent_owners, **self._kwargs)
        return self._emitter.published
