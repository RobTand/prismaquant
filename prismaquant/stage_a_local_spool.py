"""Precommit local entries handed to PrismaBuild's durable export owner.

This adapter serializes nothing and moves nothing. It remembers exact writer
references until PB acknowledges immutable canonical origins. Only that ack may
advance durable entry progress or release the PB-owned local reservation.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
import threading
import time

ROOT_ENV = "PRISMABUILD_PRODUCED_SPOOL_ROOT"
MAX_ENV = "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"


class BoundarySpoolRefused(RuntimeError):
    pass


class BoundaryOutputSpool:
    def __init__(self, backend, *, capacity_deferred, timeout_s):
        self.backend = backend
        self.capacity_deferred = capacity_deferred
        self.timeout_s = float(timeout_s)
        self._lock = threading.RLock()
        self._groups = {}
        self._pending = set()
        self._durable_progress = []

    @classmethod
    def from_publication(cls, publication, *, timeout_s):
        env = getattr(publication, "env", {})
        root = env.get(ROOT_ENV)
        if not root:
            if env.get(MAX_ENV):
                raise BoundarySpoolRefused(f"{MAX_ENV} requires {ROOT_ENV}")
            return None
        from .staged_lease import sdk_submodule
        module = sdk_submodule("produced_spool")
        if getattr(module, "API_VERSION", None) != 1:
            raise BoundarySpoolRefused("the sealed PB helper lacks produced_spool API v1")
        backend = module.ProducedSpool(
            publication.queue, publication.instance, publication.template,
            cas_root=publication.cas_root, root=root,
            max_bytes=int(env.get(MAX_ENV, str(32 << 30))))
        return cls(backend, capacity_deferred=module.SpoolCapacityDeferred,
                   timeout_s=timeout_s)

    def reserve(self, batch_id, ceiling_bytes):
        deadline = time.monotonic() + self.timeout_s
        while True:
            with self._lock:
                if batch_id in self._groups:
                    group = self._groups[batch_id]
                    if group["ceiling_bytes"] != ceiling_bytes or group["released"]:
                        raise BoundarySpoolRefused("local reservation replay changed or was released")
                    return group["directory"]
                try:
                    directory = self.backend.reserve_group(batch_id, ceiling_bytes=ceiling_bytes)
                except self.capacity_deferred:
                    self._poll_all_locked()
                else:
                    self._groups[batch_id] = dict(
                        directory=Path(directory), references=[], submitted=False,
                        durable=False, released=False,
                        ceiling_bytes=int(ceiling_bytes))
                    return Path(directory)
            if time.monotonic() >= deadline:
                raise TimeoutError("local output spool stayed full inside its bounded wait")
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def directory(self, batch_id):
        with self._lock:
            return self._groups[batch_id]["directory"]

    def record(self, batch_id, local_reference, canonical_directory):
        canonical = dataclasses.replace(
            local_reference,
            path=str(Path(canonical_directory) / Path(local_reference.path).name))
        with self._lock:
            group = self._groups[batch_id]
            if group["submitted"]:
                raise BoundarySpoolRefused("cannot append to a submitted output group")
            if any(ref.name == local_reference.name for ref, _ in group["references"]):
                raise BoundarySpoolRefused("local output group repeats an entry")
            group["references"].append((local_reference, canonical))
        return canonical

    def submit(self, batch_id):
        with self._lock:
            group = self._groups[batch_id]
            if group["submitted"]:
                return
            entries = [dict(source_path=local.path, destination_path=canonical.path,
                            bytes=local.file_bytes, sha256=local.sha256)
                       for local, canonical in group["references"]]
            if not entries:
                raise BoundarySpoolRefused("cannot export an empty output group")
            answer = self.backend.submit_group(batch_id, entries=entries)
            if not answer.get("ok"):
                raise BoundarySpoolRefused(f"PB refused local export: {answer!r}")
            group["submitted"] = True
            self._pending.add(batch_id)

    def _poll_locked(self, batch_id, group):
        if not group["submitted"]:
            return False
        if not group["durable"]:
            answer = self.backend.poll_group(batch_id)
            if not answer.get("ok"):
                raise BoundarySpoolRefused(f"PB local export failed: {answer!r}")
            if answer.get("complete") is not True:
                return False
            # Only the backend's verified durable receipt establishes this.
            group["durable"] = True
            self._durable_progress.extend(ref for _, ref in group["references"])
        if not group["released"]:
            answer = self.backend.release_group(batch_id)
            if not answer.get("ok"):
                raise BoundarySpoolRefused(f"PB retained local export debt: {answer!r}")
            group["released"] = True
            group["references"].clear()
            self._pending.discard(batch_id)
        return True

    def _poll_all_locked(self):
        # Only bounded in-flight exports are examined at a write boundary;
        # completed historical groups do not multiply polling work.
        for batch_id in tuple(self._pending):
            self._poll_locked(batch_id, self._groups[batch_id])

    def await_group(self, batch_id, *, deadline=None):
        deadline = time.monotonic() + self.timeout_s if deadline is None else deadline
        while True:
            with self._lock:
                group = self._groups[batch_id]
                if not group["submitted"]:
                    raise BoundarySpoolRefused("incomplete local group has no export submission")
                if self._poll_locked(batch_id, group):
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"local output group {batch_id!r} export has not landed")
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def durable_entries(self):
        """Drain newly acknowledged references on the compute thread only."""
        with self._lock:
            self._poll_all_locked()
            committed, self._durable_progress = self._durable_progress, []
            return committed

    def drain(self):
        deadline = time.monotonic() + self.timeout_s
        with self._lock:
            keys = list(self._groups)
        for batch_id in keys:
            self.await_group(batch_id, deadline=deadline)

    def pending(self, batch_id):
        with self._lock:
            group = self._groups.get(batch_id)
            return group is not None and not group["released"]

    def report(self):
        with self._lock:
            return dict(schema="prismaquant.stage_a_local_spool.v1",
                        groups=len(self._groups),
                        durable_groups=sum(g["durable"] for g in self._groups.values()),
                        pending_groups=sum(not g["released"] for g in self._groups.values()),
                        retained_ceiling_bytes=sum(g["ceiling_bytes"]
                            for g in self._groups.values() if not g["released"]))
