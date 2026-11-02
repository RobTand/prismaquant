"""Header and profile planning for a Tessera serving export."""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any


class TesseraExportLaneError(RuntimeError):
    """The Tessera export lane refuses this run.  Always actionable."""


def export_setup(model_path: str | Path, plan: Mapping[str, Any]) -> dict:
    """Derive export work and census inputs without runtime qualification.

    This reads source headers and profile metadata. The exporter still owns
    construction, native-route, byte-integrity and serving checks.
    """
    from .export_partition import whole_layer_partitions
    from .model_profiles import detect_profile
    from .name_projection import NameProjection
    from .tessera_plan_writer import tessera_surface
    from tessera import serving_parts
    from tessera.grammar import require_column_groups
    from .tessera_render import TESSERA_HALF

    source = Path(model_path).resolve()
    profile = detect_profile(str(source))
    if profile.structure_spec() is None:
        raise TesseraExportLaneError("export setup requires a known model profile")
    config = json.loads((source / "config.json").read_text())
    architectures = list(profile.declared_architectures())
    if not architectures:
        raise TesseraExportLaneError("export setup requires config architectures for the construction census")
    inventory = serving_parts.source_inventory(source)
    surface = tessera_surface()
    surface.validate_serving_plan(dict(plan))
    _shards, dense, packed, routed = surface.quantizable(source)
    stacks = surface.expert_stacks(routed)
    packed_stacks = surface.packed_expert_stacks(packed)
    if set(stacks) & set(packed_stacks):
        raise TesseraExportLaneError("expert stack appears in both source layouts")
    entries = {name: value for name, value in plan.items() if name != "schema"}
    stack_entries = {}
    for name, value in entries.items():
        if name in stacks or name in packed_stacks:
            if not isinstance(value, dict):
                raise TesseraExportLaneError(f"{name}: a stack plan requires its explicit source layout")
            stack_entries[name] = value
        elif name not in dense and name not in routed:
            raise TesseraExportLaneError(f"{name}: unsupported module kind or absent source tensor")
        elif isinstance(value, dict):
            if name in routed:
                raise TesseraExportLaneError(f"{name}: routed units require a stack plan")
            grid = surface.grid_for_name(value["grid"])
            if surface.MOE_ROUTER.match(name):
                raise TesseraExportLaneError(f"{name}: unsupported module kind for quantization: router")
            try:
                require_column_groups(dense[name][1], TESSERA_HALF)
                from .tessera_menu import tessera_shape_legal
                from .tessera_formats import tessera_family
                family = tessera_family(grid.name.removesuffix(f"x{grid.arity}"), grid.arity)
                legal, reason = tessera_shape_legal(family, value["q256"], dense[name])
            except (ValueError, surface.TesseraError) as exc:
                raise TesseraExportLaneError(f"{name}: unsupported shape {dense[name]}: {exc}") from exc
            if not legal:
                raise TesseraExportLaneError(f"{name}: unsupported shape {dense[name]}: {reason}")
    # The producer derives expert counts, orientations and canonical slices.
    projection = surface.project_expert_plan({**dense, **packed, **routed}, config, stack_entries)
    names = NameProjection(profile)
    units = []
    for kind, shapes in (("dense", dense), ("routed_moe", routed), ("packed_routed_moe", packed)):
        for name, shape in sorted(shapes.items()):
            units.append({"tensor": name, "shape": list(shape), "kind": kind,
                          "source_shard": inventory[name],
                          "recipe_unit": names.recipe_unit(names.checkpoint_to_live(name).target)})
    partitions = whole_layer_partitions(inventory, serving_parts)
    from .lane_spec import lane_spec_for_container
    lane = lane_spec_for_container("tessera")
    return {"schema": "prismaquant.tessera_export_setup.v1", "scope": "headers_and_profile_only",
            "profile": profile.name, "wired_architectures": sorted(lane.wired_architectures),
            "profile_wired": lane.wires(profile.name), "source_tensors": inventory,
            "units": units, "expert_projection": projection,
            "partition_count": len(partitions), "partitions": partitions,
            "construction_census": {"status": "not_run", "repo_env": "TESSERA_REPO",
                "architectures": architectures, "model": str(source),
                "argv": ["python3", "tools/tessera_construction_census.py", str(source),
                         "construction-census.json", "--device", "meta"]},
            "runtime_qualification": "not_run"}
