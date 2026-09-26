"""A campaign container spec fixture that declares the produced output spool.

The Stage A dispatch refuses a spec without the spool (PQ #1012), so every
fixture spec a Stage A row seals declares one: a root on the executing box's
own disk, its byte bound, the paced-export opt-in, and the writable identity
bind the container requires. The paths are text; no test writes under them.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

SPOOL_ROOT = "/home/rob/pb-spool/fixture"
SPOOL_MAX_BYTES = str(32 << 30)
SPOOL_ENV = {
    "PRISMABUILD_PRODUCED_SPOOL_ROOT": SPOOL_ROOT,
    "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES": SPOOL_MAX_BYTES,
    "PRISMABUILD_PRODUCED_SPOOL_PACED_EXPORT": "1",
}
SPOOL_MOUNT = {"source": SPOOL_ROOT, "target": SPOOL_ROOT, "readonly": False}

#: The fixture plan's plane geometry. The Stage A row seals its spool bound
#: as two cotangent planes derived from the plan and the model config
#: (PQ #1110): 2 probes x 8 one-row entries of 16 x 64 bf16 values, each
#: reserved with the writer's 64 KiB envelope, two planes.
FIXTURE_HIDDEN = 64
FIXTURE_EXECUTION = {"n_probes": 2, "n_calib_samples": 8, "calib_seqlen": 16,
                     "probe_microbatch": 1,
                     "boundary_storage": {"prefetch_batches": 4}}
SPOOL_WINDOW_BYTES = 2 * 2 * 8 * (16 * FIXTURE_HIDDEN * 2 + 65536)
#: The spool environment the Stage A row seals: the spec's, with the byte
#: bound replaced by the plan's two-plane window, and the host window opt-in
#: that has placement charge it (PQ #1120).
STAGE_A_SPOOL_ENV = {**SPOOL_ENV,
                     "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES": str(SPOOL_WINDOW_BYTES),
                     "PRISMABUILD_PRODUCED_SPOOL_HOST_WINDOW": "1"}
#: The spool environment a quantum row seals: the spec's, with the host
#: window opt-in, so placement charges its bound too (PQ #1364).
QUANTUM_SPOOL_ENV = {**SPOOL_ENV, "PRISMABUILD_PRODUCED_SPOOL_HOST_WINDOW": "1"}


def stage_a_plan(root, **fields) -> dict:
    """A fixture plan that states the Stage A plane geometry.

    Writes a model config under ``root`` (hidden size and dtype, the two
    fields the dispatcher reads) and returns the plan's ``model`` and
    ``execution`` fields with ``fields`` merged on top.
    """

    model = Path(root) / "fixture-model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(json.dumps(
        {"hidden_size": FIXTURE_HIDDEN, "dtype": "bfloat16"}))
    return {"model": str(model), "execution": copy.deepcopy(FIXTURE_EXECUTION),
            **fields}


def with_spool(spec: dict) -> dict:
    """``spec`` plus the spool's environment and identity bind."""

    spec = copy.deepcopy(spec)
    spec.setdefault("env", {}).update(SPOOL_ENV)
    mounts = spec.setdefault("container", {}).setdefault("mounts", [])
    mounts.append(dict(SPOOL_MOUNT))
    return spec
