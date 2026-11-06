"""Shared synthetic inputs for tessera expert projection tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

from prismaquant import tessera_expert_projection as tep


STACK = "model.layers.2.feed_forward.experts"
SHARD = "model.safetensors"


def _unit(expert: int, role: str, rows: int, cols: int, **overrides) -> dict:
    """A record shaped like ``export_tessera_serving.plan_expert_stack`` writes."""
    projection = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}[role]
    group = "w2" if role == "w2" else "w13"
    tensor = f"{STACK}.{expert}.{role}.weight"
    record = {
        "tensor": tensor, "wire": f"{STACK}.{expert}.{role}.wire",
        "source_tensor": tensor, "source_layout": tep.SOURCE_LAYOUT_UNPACKED,
        "source_slice": {"expert": expert, "selector": "whole", "transpose": False},
        "expert": expert, "projection": projection, "group": group,
        "rows": rows, "cols": cols,
    }
    record.update(overrides)
    return record


def _projection(experts=(0, 1), *, stacks=(STACK,), n=8, k=4) -> dict:
    tensors = {}
    stack_entries = {}
    for stack in stacks:
        units = []
        for expert in experts:
            for role, (rows, cols) in (("w1", (n, k)), ("w3", (n, k)), ("w2", (k, n))):
                unit = _unit(expert, role, rows, cols)
                unit["tensor"] = unit["source_tensor"] = f"{stack}.{expert}.{role}.weight"
                units.append(unit)
                tensors[unit["tensor"]] = SHARD
        stack_entries[stack] = {
            "source_layout": tep.SOURCE_LAYOUT_UNPACKED, "grid": "E4M3", "q256": 1024,
            "experts": len(experts), "units": units,
        }
    return {
        "schema": tep.PROJECTION_SCHEMA,
        "stacks": stack_entries,
        "source": {
            "config_sha256": "c" * 64, "auxiliary_sha256": {},
            "files": {SHARD: "f" * 64}, "tensors": tensors,
        },
    }


def _declared(experts=(0, 1), *, stacks=(STACK,), n=8, k=4) -> dict:
    return {
        stack: {
            f"{stack}.{expert}.{role}": shape
            for expert in experts
            for role, shape in (("w1", (n, k)), ("w3", (n, k)), ("w2", (k, n)))
        }
        for stack in stacks
    }


def _record(tmp_path: Path, name: str, unit: dict, *, q256=1024, grid="E4M3") -> dict:
    blob = hashlib.sha256(name.encode()).digest() * 3
    file = name.replace(".", "__") + "__TESSERA_E4M3_K1_R1024.tessera"
    (tmp_path / file).write_bytes(blob)
    return {
        "file": file, "blob_sha256": hashlib.sha256(blob).hexdigest(),
        "blob_bytes": len(blob),
        "identity": {
            "schema": "tessera.cached_unit_inputs.v1", "unit": name,
            "recipe": {"grid": grid, "q256": q256},
            "projection": {key: unit[key] for key in tep.UNIT_IDENTITY_KEYS},
        },
    }
