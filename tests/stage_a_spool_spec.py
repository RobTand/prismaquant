"""A campaign container spec fixture that declares the produced output spool.

The Stage A dispatch refuses a spec without the spool (PQ #1012), so every
fixture spec a Stage A row seals declares one: a root on the executing box's
own disk, its byte bound, the paced-export opt-in, and the writable identity
bind the container requires. The paths are text; no test writes under them.
"""
from __future__ import annotations

import copy

SPOOL_ROOT = "/home/rob/pb-spool/fixture"
SPOOL_MAX_BYTES = str(32 << 30)
SPOOL_ENV = {
    "PRISMABUILD_PRODUCED_SPOOL_ROOT": SPOOL_ROOT,
    "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES": SPOOL_MAX_BYTES,
    "PRISMABUILD_PRODUCED_SPOOL_PACED_EXPORT": "1",
}
SPOOL_MOUNT = {"source": SPOOL_ROOT, "target": SPOOL_ROOT, "readonly": False}


def with_spool(spec: dict) -> dict:
    """``spec`` plus the spool's environment and identity bind."""

    spec = copy.deepcopy(spec)
    spec.setdefault("env", {}).update(SPOOL_ENV)
    mounts = spec.setdefault("container", {}).setdefault("mounts", [])
    mounts.append(dict(SPOOL_MOUNT))
    return spec
