"""Fixtures for a Stage A row's produced output spool.

The spool's root and byte bound live in the launch-bound produced declaration
(``declaration['spool']``), never in environment. A fixture spec carries only the
writable identity bind the container needs; ``write_declaration`` writes the
declaration the shipped Stage A row requires. The paths are text; no test writes
under them.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

SPOOL_ROOT = "/home/rob/pb-spool/fixture"
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


def write_declaration(directory) -> Path:
    """A valid launch-bound Stage A produced declaration bounding the spool at the fixture window."""
    from prismaquant.stage_a_produced_output import build_boundary_template
    path = Path(directory) / "stage-a-declaration.json"
    path.write_text(json.dumps(build_boundary_template(
        output_prefix="/out/root", artifact_max_bytes=1 << 20, group_size=1,
        max_entry_tensor_bytes=1 << 16,
        spool={"root": SPOOL_ROOT, "max_bytes": SPOOL_WINDOW_BYTES})))
    return path


def declared_stage_a_argv(builder, manifest, campaign, **kwargs):
    """Call a shipped ``stage_a_argv`` with the launch-bound declaration the Stage A row requires."""
    kwargs.setdefault("produced_declaration", write_declaration(Path(campaign["plan_path"]).parent))
    return builder(manifest, campaign, **kwargs)


def with_spool(spec: dict) -> dict:
    """``spec`` plus the spool's writable identity bind."""

    spec = copy.deepcopy(spec)
    mounts = spec.setdefault("container", {}).setdefault("mounts", [])
    mounts.append(dict(SPOOL_MOUNT))
    return spec
