#!/usr/bin/env python
"""Write the Tessera exporter's plan from a PrismaQuant allocation (prismaquant #1587).

PrismaQuant's allocator owns the decision -- which body Linear takes which
``TESSERA_<BASE>_K<arity>_R<rung>`` -- so PrismaQuant writes the exporter's
plan document itself, in the schema Tessera publishes
(``tessera.serving_plan.v1``, RobTand/tessera#687): per tensor
``{"grid": ..., "q256": ...}`` or the string ``"BF16"``, per ``<moe>.experts``
stack ``{"grid", "q256", "source_layout"}``.  Until #1587 the arm ran
Tessera's ``experiments/plan_from_layer_config.py --prismaquant``, a script
Tessera calls unsupported and which imported ``prismaquant.tessera_formats``
-- the dependency direction Rob ruled out (RobTand/tessera#599).  The
translation now lives here, on the PrismaQuant side of the boundary, and the
arm calls Tessera for exactly one thing: the supported exporter entry point
``python -m tessera.export_serving``.

What stays Tessera's is the *classification*: what a body Linear is, which
source leaves vLLM fuses into one module, what one fused scheme is, and how a
routed stack projects onto source units.  Those are imported from the Tessera
**package** (``tessera.export_serving``, ``tessera.serving_plan``,
``tessera.control``, ``tessera.serving_parts``) -- never from ``experiments/``
-- so there is one statement of each rule and a plan-time check cannot drift
from the export-time check.  A pin whose package does not carry the exporter
(pre-#687, e.g. 38e96012) is refused by name rather than guessed around.

What stays PrismaQuant's is the *accounting*: the sidecar beside the plan
carries each unit's charged wire bits computed with PrismaQuant's own
``tessera_formats.artifact_bpp`` -- in-tree, with no tree to point at -- so
the bytes served remain the bytes the allocator priced.

The planning semantics are the ones the Tessera translator established:

**Non-Tessera quantised choices are refused.**  A plan that mixed one in
(NVFP4, FP8_DYNAMIC, ...) would be dropped silently by the exporter's
``unknown`` check or exported as something the allocator did not choose; those
are counted and refused by name.  BF16 is not a refusal -- it is a plain BF16
module.

**The fused-group invariant is checked before the encode.**  vLLM builds ONE
method per fused module, so ``q/k/v`` must agree on family/grid/body/plane,
and so must ``gate/up``.  The key is ``tessera.serving_plan.module_scheme_key``,
imported, not restated.  A disagreement is refused up front with the members
and their rungs; ``--allow-fused-disagreement`` writes the plan that will
SERVE -- every member of a disagreeing group ``"BF16"``, the demotion recorded
(``fused_disagreements[].planned_as``, ``totals.demoted_to_bf16_params``).
The RATE is not part of the invariant (#37): members are decoded from their
own manifests, so a per-member (mink) allocation plans and encodes member by
member.

A whole-GROUP option name (``TESSERA_E4M3_K1_G3``) is refused by name: it
names a family and an option index, not the rung each member took.
``expand_fused_sibling_assignment`` is supposed to have expanded it before the
assignment was written.

**Coverage is explicit.**  ``--cover as-allocated`` plans exactly the units
the allocation names and names every other body Linear ``"BF16"``
explicitly -- an unnamed tensor takes the exporter's ``--grid`` default, which
is a 4-bit rung, not a passthrough.  ``--cover broadcast-by-role`` applies a
single-layer allocation's per-role assignment at every depth, is refused
unless the allocation is single-layer with matching shapes, and is stamped as
the EXTRAPOLATION it is.

The sidecar ``<out>.provenance.json`` (schema ``prismaquant.tessera_plan.v1``)
carries the source path, the allocation's ``__prismaquant__`` block, the
coverage decision, the per-unit shape/rung/charged-bits table, the uniform
control arm this plan has to beat, and the selection requirement a mixed-rung
plan triggers.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from .digests import indent2_json_file_bytes

#: ``TESSERA_<BASE>_K<arity>_R<rung>`` -- the allocator's format spelling.
FORMAT = re.compile(r"^TESSERA_(?P<base>[A-Z0-9]+)_K(?P<arity>\d+)_R(?P<rung>\d+)$")
FAMILY = re.compile(r"^TESSERA_(?P<base>[A-Z0-9]+)_K(?P<arity>\d+)$")
#: ``TESSERA_<BASE>_K<arity>_G<n>`` -- a whole fused GROUP at one family with a
#: rung per member, the option PrismaQuant's group knapsack builds.  Not a
#: rung and no rate it could stand for, so it is refused by name.
GROUP = re.compile(r"^TESSERA_(?P<base>[A-Z0-9]+)_K(?P<arity>\d+)_G(?P<option>\d+)$")

#: What the allocator may pick that is not a Tessera wire and is still fine.
BF16_CHOICES = {"BF16", "bfloat16", "bf16"}

#: The plan document's schema, published by the Tessera package
#: (``tessera.serving_plan.v1``, RobTand/tessera#687).
PLAN_SCHEMA = "tessera.serving_plan.v1"


class PlanError(SystemExit):
    pass


def tessera_surface() -> SimpleNamespace:
    """Bind the classification names from the Tessera **package**.

    Every model-shape question in this file -- what a body Linear is, which
    leaves fuse into one vLLM module, what one scheme is, how a routed stack
    projects -- is answered by Tessera's own code, imported here as one
    surface so the plan-time check and the export-time check share one
    statement of each rule.  The names live in the package only since
    RobTand/tessera#687; on an older pin (38e96012) the import fails and the
    writer refuses by name instead of guessing around the missing exporter.
    """
    try:
        from tessera.control import (
            control_block, grid_for_name, selection_requirement,
            uniform_control, units_from_plan)
        from tessera.errors import TesseraError
        from tessera.export_serving import (
            MOE_ROUTER, MOE_SOURCE_UNPACKED, body_layer, expert_stacks,
            fused_module, packed_expert_stacks,
            project_expert_plan, quantizable)
        from tessera.serving_parts import (
            SOURCE_ROSTER_FIELDS, source_roster_identity)
        from tessera.serving_plan import (
            family_for, module_scheme_key)
    except ImportError as exc:
        raise RuntimeError(
            "the Tessera pin packages no tessera.export_serving/"
            "tessera.serving_plan (RobTand/tessera#687): PrismaQuant writes "
            "the plan against the supported exporter only -- bump the "
            f"Tessera pin. (import failed: {exc})") from exc
    return SimpleNamespace(
        MOE_ROUTER=MOE_ROUTER, MOE_SOURCE_UNPACKED=MOE_SOURCE_UNPACKED,
        TesseraError=TesseraError,
        control_block=control_block, grid_for_name=grid_for_name,
        selection_requirement=selection_requirement,
        uniform_control=uniform_control, units_from_plan=units_from_plan,
        body_layer=body_layer, expert_stacks=expert_stacks,
        family_for=family_for, fused_module=fused_module,
        module_scheme_key=module_scheme_key,
        packed_expert_stacks=packed_expert_stacks,
        project_expert_plan=project_expert_plan, quantizable=quantizable,
        SOURCE_ROSTER_FIELDS=SOURCE_ROSTER_FIELDS,
        source_roster_identity=source_roster_identity)


def grid_of(family: str) -> str:
    """``TESSERA_E4M3_K1 -> "E4M3"``, ``TESSERA_E2M1_K2 -> "E2M1x2"``."""
    match = FAMILY.fullmatch(family)
    if not match:
        raise PlanError(f"not a Tessera family name: {family!r}")
    arity = int(match.group("arity"))
    return match.group("base") + ("" if arity == 1 else f"x{arity}")


def parse_entry(qname: str, entry) -> tuple:
    """``(kind, payload)``: ``("tessera", (grid, rung, family))``, ``("bf16", None)`` or ``("other", label)``."""
    if isinstance(entry, str):
        if entry in BF16_CHOICES:
            return ("bf16", None)
        if not entry.startswith("TESSERA_"):
            return ("other", entry)
        entry = {"tessera_format": entry}
    if not isinstance(entry, dict):
        raise PlanError(f"{qname}: unreadable layer_config entry {entry!r}")
    fmt = entry.get("tessera_format")
    if fmt:
        if GROUP.fullmatch(fmt):
            raise PlanError(
                f"{qname}: {fmt!r} is a whole-GROUP option (one family, a rung per member), "
                "not a rung, and there is no single rate it could stand for -- the members "
                "have different shapes and different sensitivities, which is the entire "
                "reason the option exists.  PrismaQuant expands it to its members' own rungs "
                "in expand_fused_sibling_assignment before writing an assignment; a plan that "
                "still carries the group name means that expansion did not run.  Re-export "
                "the layer_config from an allocator that expands it.")
        match = FORMAT.fullmatch(fmt)
        if not match:
            raise PlanError(f"{qname}: {fmt!r} is not the TESSERA_<BASE>_K<arity>_R<rung> spelling")
        family = f"TESSERA_{match.group('base')}_K{match.group('arity')}"
        declared = entry.get("tessera_family")
        if declared and declared != family:
            raise PlanError(f"{qname}: tessera_family {declared!r} disagrees with tessera_format {fmt!r}")
        rung = int(match.group("rung"))
        body_q256 = entry.get("tessera_body_rate_q256")
        if body_q256 is not None and int(body_q256) != rung:
            raise PlanError(f"{qname}: tessera_body_rate_q256 {body_q256} disagrees with {fmt!r}")
        return ("tessera", (grid_of(family), rung, family))
    label = entry.get("data_type") or entry.get("format") or entry.get("bits")
    if str(label) in BF16_CHOICES:
        return ("bf16", None)
    return ("other", str(label))


def refuse_non_tessera_choices(other: dict) -> None:
    """One home for the refusal of a quantised choice no Tessera wire serves."""
    if not other:
        return
    counts = collections.Counter(other.values())
    sample = sorted(other)[:5]
    raise PlanError(
        f"{len(other)} unit(s) carry a non-Tessera QUANTISED choice "
        f"({dict(counts)}); the Tessera plugin serves TESSERA_* wires only, so one "
        f"checkpoint cannot hold these and a Tessera wire at the same time.  Units, "
        f"first five: {sample}.  BF16 is not in this count -- a BF16 choice is a plain "
        f"BF16 module and is planned as one.")


def read_carried_projection(config: dict):
    """The carried expert projection, schema-checked, or ``None`` without one."""
    carried = (config.get("__prismaquant__") or {}).get("tessera_expert_projection")
    if carried is None:
        return None
    if not isinstance(carried, dict) or carried.get("schema") != "prismaquant.tessera_expert_projection.v1":
        raise PlanError("unreadable carried Tessera expert projection")
    producer = carried.get("producer", {})
    request = carried.get("request")
    if not isinstance(request, dict) or producer.get("schema") != "tessera.expert_projection.v1":
        raise PlanError("carried projection lacks the producer request/answer")
    return carried


def refuse_before_source(config: dict, surface, research_input=None) -> None:
    """Every refusal the layer_config and explicit inputs decide on their own.

    Runs before the planner opens the source checkpoint, so a plan that cannot
    be written fails in seconds, not after a pass over the source.
    """
    read_carried_projection(config)
    choices = {qname: parse_entry(qname, entry) for qname, entry in config.items()
               if not qname.startswith("__")}
    refuse_non_tessera_choices({qname: payload for qname, (kind, payload) in choices.items()
                                if kind == "other"})
    # Router BF16 is a disposition, not an exporter override: GateLinear never
    # asks a quantization method to load it and the exporter refuses that key.
    for qname, (kind, _payload) in sorted(choices.items()):
        if surface.MOE_ROUTER.fullmatch(qname + ".weight") and kind != "bf16":
            raise PlanError(f"{qname}: an immutable MoE router must remain BF16")
    if research_input is None:
        return
    schemes = [{"family": surface.family_for(surface.grid_for_name(payload[0])),
                "grid": payload[0]}
               for kind, payload in choices.values() if kind == "tessera"]
    if schemes and not any(research_input.config.applies_to(scheme) for scheme in schemes):
        raise PlanError("research_selected_moe names no routed target it serves "
                        "(TESSERA_FP8/E4M3 or TESSERA_BF16/BF16): no allocation entry takes "
                        "either, so no planned expert stack can")


def model_plan_context(model: Path, config: dict, surface, *, research_selected: bool = False):
    """Use the exporter's classification and its carried projection, never guess slices."""
    _shards, dense, packed, routed = surface.quantizable(model)
    stacks = surface.expert_stacks(routed)
    packed_stacks = surface.packed_expert_stacks(packed)
    if set(stacks) & set(packed_stacks):
        raise PlanError("expert stack appears in both packed and unpacked source layouts")
    shapes = {**dense, **routed}
    if not shapes:
        raise PlanError(f"no 2-D body weight tensors in {model}")
    members = {stack: [name for expert in experts.values() for name, _shape in expert.values()]
               for stack, experts in stacks.items()}
    layouts = {stack: surface.MOE_SOURCE_UNPACKED for stack in stacks}
    carried = read_carried_projection(config)
    if carried is not None:
        producer, request = carried["producer"], carried["request"]
        # Bind the projection to this checkpoint's config and tensor roster,
        # not its payload digests: every output below is recomputed from
        # headers and config.json, and the exporter's partition stamps, their
        # merge and the cached-unit intake bind the bytes they read.
        source = producer.get("source")
        if (not isinstance(source, dict)
                or {field: source.get(field) for field in surface.SOURCE_ROSTER_FIELDS}
                != surface.source_roster_identity(model)):
            raise PlanError("carried expert projection source identity disagrees with this checkpoint")
        current = surface.project_expert_plan({**dense, **packed, **routed},
                    json.loads((Path(model) / "config.json").read_text()), request,
                    research_selected=research_selected)
        if current["stacks"] != producer.get("stacks"):
            raise PlanError("carried expert projection disagrees with the producer's current source projection")
        bindings = carried.get("stacks")
        if not isinstance(bindings, dict) or set(bindings) != set(current["stacks"]):
            raise PlanError("carried projection stack binding does not cover its producer answer")
        for stack, projected in current["stacks"].items():
            expected = {unit["tensor"][:-len(".weight")]:
                        {key: value for key, value in unit.items() if key != "wire"}
                        for unit in projected["units"]}
            if bindings[stack] != expected:
                raise PlanError(f"{stack}: carried unit binding disagrees with the producer projection")
            members[stack] = [unit["tensor"] for unit in projected["units"]]
            layouts[stack] = projected["source_layout"]
            shapes.update({unit["tensor"]: (unit["rows"], unit["cols"])
                           for unit in projected["units"]})
    # An unallocated packed stack is an explicit passthrough. Quantized logical
    # units require the producer projection above to put them in the shape table.
    for stack in packed_stacks:
        members.setdefault(stack, [])
    return shapes, members, layouts


def stack_plan(plan: dict, members: dict, layouts: dict) -> dict:
    """Replace logical leaves only when the whole stack has one exact choice."""
    result = dict(plan)
    for stack, tensors in sorted(members.items()):
        choices = [result.pop(tensor) for tensor in tensors]
        choice = choices[0] if choices else "BF16"
        if any(value != choice for value in choices):
            raise PlanError(f"{stack}: the producer serves the whole stack at one exact rung; "
                            "mixed Tessera/BF16 choices or differing rungs cannot be exported")
        result[stack] = (dict(choice, source_layout=layouts[stack])
                         if isinstance(choice, dict) else choice)
    return result


def role_of(qname: str) -> str:
    return qname.rsplit(".", 1)[-1]


def layer_of(qname: str, surface=None) -> int:
    body_layer = surface.body_layer if surface is not None else None
    tensor = qname + ".weight"
    return body_layer(tensor) if body_layer is not None else int(tensor.split(".")[2])


def fused_key(qname: str, surface):
    """``(fused module qname, ordered member qnames)`` or ``None``.

    Delegated to the exporter's ``fused_module`` -- the exporter owns the
    roster of source leaves vLLM merges into one module (q/k/v, every
    non-routed gate/up including ``mlp.shared_experts``, LFM's dense
    ``feed_forward.w1/w3`` -> ``w13``), and a restatement here is how this
    converter once checked the fused invariant on two of those groups and
    skipped the rest (tessera#211).  One rule, one home.
    """
    fused = surface.fused_module(qname + ".weight")
    if fused is None:
        return None
    module, members = fused
    return module, tuple(member[: -len(".weight")] for member in members)


def charged_bits(family: str, rung: int, shape) -> Fraction:
    """PrismaQuant's own charged wire bits for this unit.

    The allocator's byte budget is spent in this currency, so it is the number
    an export must reproduce.  Computed in-tree with
    ``prismaquant.tessera_formats.artifact_bpp`` -- the #1587 inversion of the
    old ``--prismaquant`` tree pointer: two accountings of one wire is the
    drift this sidecar exists to catch, and PrismaQuant owns the accounting.
    """
    from .tessera_formats import artifact_bpp
    rows, cols = shape
    return Fraction(artifact_bpp(family, rung, shape=(rows, cols))) * rows * cols


def uniform_control_block(plan: dict, shapes: dict, surface, *, rule: str = "nearest"):
    """The byte-matched uniform arm this plan has to beat, as a record.

    An allocation over a rate axis is a *claim* -- that choosing rungs beats
    spending the same bytes at one rung -- and on 2026-09-02 that claim was
    false by 2.00x while every other check in the pipeline passed.  So the
    sidecar carries the arm that tests it, priced but not served, next to the
    bpp (tessera#3, principle 12).  It records rather than refuses: a plan
    whose control cannot be byte-matched is still a plan; what it must never
    do is stay silent about it, so the reason lands in the block.
    """
    try:
        units = surface.units_from_plan(plan, shapes)
        control = surface.uniform_control(units, rule=rule, assert_match=False)
    except surface.TesseraError as exc:
        return {"schema": "tessera.uniform_control.v1", "built": False,
                "refusal": str(exc)}
    block = surface.control_block(control)
    block["built"] = True
    return block


def build_serving_plan(config: dict, shapes: dict, *, cover: str, allow_disagreement: bool,
          surface, control_rule: str = "nearest", with_control: bool = True):
    meta = config.get("__prismaquant__")
    assignment = {k: v for k, v in config.items() if not k.startswith("__")}
    if not assignment:
        raise PlanError("layer_config names no units")

    tessera, bf16, other = {}, [], {}
    for qname, entry in sorted(assignment.items()):
        kind, payload = parse_entry(qname, entry)
        if kind == "tessera":
            tessera[qname] = payload
        elif kind == "bf16":
            bf16.append(qname)
        else:
            other[qname] = payload
    refuse_non_tessera_choices(other)

    priced_layers = sorted({layer_of(q, surface) for q in tessera}
                           | {layer_of(q, surface) for q in bf16})
    all_layers = sorted({layer_of(t[: -len(".weight")], surface) for t in shapes})

    plan, units, broadcast_from = {}, [], None
    if cover == "as-allocated":
        chosen = dict(tessera)
        for qname in bf16:
            plan[qname + ".weight"] = "BF16"
    elif cover == "broadcast-by-role":
        if len(priced_layers) != 1:
            raise PlanError(
                f"--cover broadcast-by-role needs a single-layer allocation to broadcast; this "
                f"one names layers {priced_layers}.  Broadcasting a multi-layer allocation would "
                f"have to invent a rule for which layer's rate wins.")
        broadcast_from = priced_layers[0]
        by_role = {role_of(q): payload for q, payload in tessera.items()}
        bf16_roles = {role_of(q) for q in bf16}
        chosen = {}
        for tensor, shape in shapes.items():
            qname = tensor[: -len(".weight")]
            role = role_of(qname)
            source = f"model.layers.{broadcast_from}.{qname.split('.', 3)[3]}.weight"
            if role in bf16_roles and role not in by_role:
                plan[tensor] = "BF16"
                continue
            if role not in by_role:
                plan[tensor] = "BF16"        # unpriced role: say BF16, do not assume it
                continue
            if source in shapes and shapes[source] != shape:
                raise PlanError(
                    f"{qname} is {shape} but the priced {source} is {shapes[source]}; a rung is "
                    f"a rate on a SHAPE (the CHANNEL plane amortises over rows), so broadcasting "
                    f"it onto a different shape would be a different rate.  Use "
                    f"--cover as-allocated.")
            chosen[qname] = by_role[role]
    else:                                                       # pragma: no cover - argparse
        raise PlanError(f"unknown coverage mode {cover!r}")

    # A tensor the plan does not name is NOT a BF16 module: the exporter falls
    # back to its own --grid/--q256 default, which is E2M1x2 q256=896 unless a
    # caller happens to override it.  Name every remaining body Linear BF16
    # explicitly and let the plan, not an exporter default, be the record of
    # what the allocation said.
    for tensor in shapes:
        if tensor not in plan and tensor[: -len(".weight")] not in chosen:
            plan[tensor] = "BF16"

    # The fused invariant, checked before the encode rather than after it.
    groups, disagreements = {}, []
    for qname in chosen:
        key = fused_key(qname, surface)
        if key is None:
            continue
        module, members = key
        if module in groups:
            continue
        groups[module] = members
        present = [m for m in members if m in chosen]
        # NOT ``{(grid, q256)}`` (#37): what one vLLM method is built from is
        # the family/grid/body/plane its route decodes with, keyed by the
        # exporter's own rule so the plan-time check and the export-time check
        # cannot describe two different sets.  A rate disagreement is not a
        # disagreement -- the roles are decoded from their own manifests and
        # the scheme carries a per-role ``q256`` list.
        schemes = {surface.module_scheme_key(surface.grid_for_name(chosen[m][0]), chosen[m][1])
                   for m in present}
        if len(present) != len(members) or len(schemes) != 1:
            disagreements.append({
                "module": module,
                "members": {m: (f"{chosen[m][2]}_R{chosen[m][1]}" if m in chosen else "ABSENT")
                            for m in members},
            })
    if disagreements and not allow_disagreement:
        raise PlanError(
            f"{len(disagreements)} fused module(s) do not share one scheme "
            f"(family, grid, body, scale plane): "
            f"{json.dumps(disagreements[:3], indent=2)}\n"
            f"vLLM builds ONE quantization method per fused module, so the exporter would pass "
            f"the whole group through as BF16.  That is a finding about the allocation -- it "
            f"chose an assignment this serving path cannot express.  Differing RUNGS are not "
            f"this refusal (#37): they are written as a per-role q256 list and served.  Re-run "
            f"with --allow-fused-disagreement to write the plan anyway and let the exporter's "
            f"own passthrough handle it.")

    # The override writes a plan, and what it must write is the plan that will
    # SERVE.  The exporter's answer to a disagreeing group is to drop every
    # member and pass the module through, so a plan that still names Tessera
    # rungs for those members describes an encode that will not happen, and
    # the sidecar -- whose whole job is "the bytes served must be the bytes
    # priced" -- reports a rate three to four times below the one the
    # checkpoint will carry.
    demoted_params = 0
    for entry in disagreements:
        entry["planned_as"] = "BF16"
        entry["reason"] = (
            "vLLM builds one quantization method per fused module; the "
            "exporter passes the whole group through at source precision"
        )
        members = groups[entry["module"]]
        entry["demoted_params"] = {}
        for member in members:
            # Only a member the allocation PRICED as Tessera is demoted; a
            # sibling the allocation itself chose BF16 (which is what made the
            # group disagree) is planned exactly as chosen and counts nothing.
            was_chosen = chosen.pop(member, None) is not None
            tensor = member + ".weight"
            if tensor in shapes:
                plan[tensor] = "BF16"
                if was_chosen:
                    rows, columns = shapes[tensor]
                    entry["demoted_params"][member] = rows * columns
                    demoted_params += rows * columns

    total_params, total_charged = 0, Fraction(0)
    for qname, (grid, rung, family) in sorted(chosen.items()):
        tensor = qname + ".weight"
        if tensor not in shapes:
            raise PlanError(f"{qname} is not a 2-D body Linear in the model: no tensor {tensor}")
        rows, cols = shapes[tensor]
        plan[tensor] = {"grid": grid, "q256": rung}
        bits = charged_bits(family, rung, (rows, cols))
        total_params += rows * cols
        total_charged += bits
        units.append({
            "tensor": tensor, "qname": qname, "role": role_of(qname),
            "layer": layer_of(qname, surface),
            "family": family, "grid": grid, "q256": rung, "rows": rows, "columns": cols,
            "params": rows * cols,
            "prismaquant_charged_bits": float(bits),
            "prismaquant_charged_bits_exact": [bits.numerator, bits.denominator],
            "prismaquant_charged_bpp": float(bits / (rows * cols)),
        })

    provenance = {
        "schema": "prismaquant.tessera_plan.v1",
        "plan_schema": PLAN_SCHEMA,
        "source_layer_config": None,                             # filled by main
        "prismaquant_meta": meta,
        "coverage": {
            "mode": cover,
            "extrapolated": cover == "broadcast-by-role",
            "broadcast_from_layer": broadcast_from,
            "allocation_units": len(tessera) + len(bf16),
            "allocation_layers": priced_layers,
            "model_body_linears": len(shapes),
            "model_layers": len(all_layers),
            "planned_tessera_units": len(chosen),
            "planned_bf16_units": sum(1 for v in plan.values() if v == "BF16"),
            "unplanned_body_linears": len(shapes) - len(plan),
            "note": ("every body Linear the allocation did not name is planned as BF16 "
                     "explicitly, because an unnamed tensor takes the exporter's --grid default, "
                     "not a passthrough" if cover == "as-allocated" else
                     "the allocation's per-ROLE assignment applied at every depth; the allocator "
                     "priced only the layer(s) above and nothing here says the same rate is right "
                     "at another depth"),
        },
        "fused_disagreements": disagreements,
        "fused_disagreement_policy": (
            "none" if not disagreements else
            "demoted_to_bf16_by_--allow-fused-disagreement"
        ),
        "totals": {
            "tessera_units": len(chosen),
            "quantized_params": total_params,
            "demoted_to_bf16_params": demoted_params,
            "prismaquant_charged_bits": float(total_charged) if total_charged else None,
            "prismaquant_charged_bpp": (float(total_charged / total_params)
                                        if total_charged and total_params else None),
        },
        "units": units,
    }
    if with_control:
        provenance["uniform_control"] = uniform_control_block(
            plan, shapes, surface, rule=control_rule)
    # The menu's selection requirement (tessera#2): a plan at more than one
    # (grid, rung) embodies a rung selection the surrogate made, and it ships
    # only validated-surrogate-selected.  Stamped, never refused.
    provenance["selection"] = surface.selection_requirement(
        surface.units_from_plan(plan, shapes))
    return plan, provenance


def plan_from_assignment(config: dict, shapes: dict, members: dict, layouts: dict, *,
                         model, cover: str, allow_disagreement: bool, surface,
                         control_rule: str = "nearest", with_control: bool = True,
                         research_input=None):
    """The whole translation: router filter, absence check, plan, stack plan."""
    if cover != "as-allocated" and members:
        raise PlanError("broadcast-by-role cannot extrapolate routed expert stacks; use as-allocated")
    # The refusal for a router the allocation quantised is in
    # ``refuse_before_source``; this set is the filter it leaves behind.
    routers = {name for name in shapes if surface.MOE_ROUTER.fullmatch(name)}
    allocation = {name: entry for name, entry in config.items()
                  if name + ".weight" not in routers}
    shapes = {name: shape for name, shape in shapes.items() if name not in routers}
    for name in allocation:
        if not name.startswith("__") and name + ".weight" not in shapes:
            raise PlanError(f"{name}: allocation unit is absent from the producer's logical body projection")
    plan, provenance = build_serving_plan(allocation, shapes, cover=cover,
                             allow_disagreement=allow_disagreement,
                             surface=surface, control_rule=control_rule,
                             with_control=with_control)
    logical_plan = plan
    plan = stack_plan(logical_plan, members, layouts)
    # Only quantized expert stacks need model-config geometry. Dense planning
    # remains a header-only operation, including its selection warning.
    selected_stacks = {stack: plan[stack] for stack in members
                       if isinstance(plan[stack], dict)}
    if research_input is not None:
        research_input.config.require_targets({
            stack: {"structure": "routed_moe",
                    "family": surface.family_for(surface.grid_for_name(choice["grid"])),
                    "grid": choice["grid"]}
            for stack, choice in selected_stacks.items()}, "resident")
        provenance["research_selected_moe"] = research_input.record()
    if selected_stacks:
        _shards, dense, packed, routed = surface.quantizable(model)
        surface.project_expert_plan({**dense, **packed, **routed},
                                    json.loads((Path(model) / "config.json").read_text()),
                                    selected_stacks,
                                    research_selected=research_input is not None)
    provenance["expert_stacks"] = {stack: {"units": stack_members, "planned_as": plan[stack]}
                                   for stack, stack_members in members.items()}
    provenance["immutable_bf16_routers"] = sorted(routers)
    return plan, provenance


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layer_config", type=Path, help="PrismaQuant layer_config.json")
    ap.add_argument("model", type=Path, help="the source checkpoint, read for tensor names and shapes")
    ap.add_argument("out", type=Path, help="the exporter's --plan-json")
    ap.add_argument("--cover", choices=("as-allocated", "broadcast-by-role"), default="as-allocated",
                    help="as-allocated: plan exactly what the allocation names (default).  "
                         "broadcast-by-role: apply its per-role assignment at every depth "
                         "(an EXTRAPOLATION, stamped as one in the sidecar)")
    ap.add_argument("--allow-fused-disagreement", action="store_true",
                    help="write the plan even when a fused module's members disagree on "
                         "family/grid/body/plane (differing RUNGS are not a disagreement).  The whole "
                         "group is then planned BF16 -- which is what the exporter does with it -- "
                         "and the demotion is recorded in the sidecar")
    ap.add_argument("--write-uniform-plan", type=Path, default=None,
                    help="also write the byte-matched UNIFORM control plan here -- the arm "
                         "the candidate has to beat at the same bytes (tessera#3).  The byte "
                         "match is ASSERTED before it is written")
    ap.add_argument("--control-rule", choices=("nearest", "no_larger"), default="nearest",
                    help="nearest: minimise |candidate - control| bytes (default).  no_larger: "
                         "the heaviest rung that does not outweigh the candidate")
    ap.add_argument("--no-uniform-control", action="store_true",
                    help="do not price the uniform control into the sidecar")
    ap.add_argument("--research-selected-moe-json", type=Path, default=None,
                    help="explicit research selected expert execution input; permits its "
                         "reader-legal BF16 expert planning without claiming a production cell")
    args = ap.parse_args(argv)

    surface = tessera_surface()
    config = json.loads(args.layer_config.read_text())
    research_input = None
    if args.research_selected_moe_json is not None:
        from tessera.moe_execution import ResearchSelectedMoeInput
        research_input = ResearchSelectedMoeInput.read(args.research_selected_moe_json)
    refuse_before_source(config, surface, research_input)
    shapes, stack_members, layouts = model_plan_context(
        args.model, config, surface, research_selected=research_input is not None)
    plan, provenance = plan_from_assignment(
        config, shapes, stack_members, layouts, model=args.model, cover=args.cover,
        allow_disagreement=args.allow_fused_disagreement, surface=surface,
        control_rule=args.control_rule, with_control=not args.no_uniform_control,
        research_input=research_input)
    logical_plan = {tensor: value for tensor, value in plan.items()
                    if tensor not in stack_members}
    logical_plan.update({tensor: value for tensor, value in plan.items()
                         if tensor in stack_members})
    provenance["source_layer_config"] = str(args.layer_config.resolve())
    provenance["model"] = str(args.model.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Plan bytes go through the digest owner's spelling (PQ #1508): the
    # acceptance compares parsed plans, so the owner's trailing LF is safe.
    args.out.write_bytes(indent2_json_file_bytes(plan))
    sidecar = args.out.with_suffix(args.out.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2))

    cov = provenance["coverage"]
    print(f"{args.layer_config}")
    print(f"  allocation: {cov['allocation_units']} unit(s) on layer(s) {cov['allocation_layers']}"
          f" of {cov['model_body_linears']} body Linears over {cov['model_layers']} layers")
    print(f"  coverage {cov['mode']}"
          + ("  [EXTRAPOLATED from layer "
             f"{cov['broadcast_from_layer']}]" if cov["extrapolated"] else ""))
    print(f"  planned: {cov['planned_tessera_units']} Tessera, {cov['planned_bf16_units']} BF16, "
          f"{cov['unplanned_body_linears']} unnamed (must be 0: an unnamed tensor takes the "
          f"exporter's --grid default, which is a 4-bit rung, not BF16)")
    by_rung = collections.Counter((u["family"], u["q256"]) for u in provenance["units"])
    for (family, rung), n in sorted(by_rung.items()):
        print(f"    {family}_R{rung}: {n}")
    demoted = provenance["totals"]["demoted_to_bf16_params"]
    if demoted:
        modules = len(provenance["fused_disagreements"])
        print(f"  DEMOTED: {modules} fused module(s) whose members do not share one scheme "
              f"are planned BF16 ({demoted} params, "
              f"{100.0 * demoted / max(demoted + provenance['totals']['quantized_params'], 1):.1f}% "
              f"of the allocated body).  This plan is NOT the allocation that was chosen; "
              f"--allow-fused-disagreement asked for the exporter's passthrough and this is it.")
    totals = provenance["totals"]
    if totals["prismaquant_charged_bpp"] is not None:
        print(f"  PrismaQuant charges {totals['prismaquant_charged_bits']:.0f} bits over "
              f"{totals['quantized_params']} params = "
              f"{totals['prismaquant_charged_bpp']:.6f} bpp")
    block = provenance.get("uniform_control")
    if block and block.get("built"):
        control = block["control"]
        match = control["match"]
        print(f"  uniform control: {control['grid']} R{control['q256']} at "
              f"{match['control_bpp']:.6f} bpp against the candidate's "
              f"{match['candidate_bpp']:.6f} "
              f"({match['relative_slack_ppm']:.1f} ppm, the {match['fatter_arm']} arm is "
              f"fatter){'' if match['byte_matched'] else '  NOT BYTE-MATCHED'}")
        print("    unserved: this is the arm the candidate has to beat at the same bytes")
    elif block:
        print(f"  uniform control: NOT BUILT -- {block['refusal']}")
    selection = provenance.get("selection")
    if selection is not None and selection["requires_validation"]:
        pairs = ", ".join(f"{grid} R{q256}" for grid, q256 in selection["distinct_rungs"])
        print(f"  SELECTION WARNING: this plan sits at {len(selection['distinct_rungs'])} "
              f"distinct (grid, rung) pairs ({pairs}), so it embodies a rung selection "
              f"no served KL has validated.  The menu requires "
              f"{selection['mode_required']} selection: serve the byte-matched uniform "
              f"control before this plan ships.")
    if args.write_uniform_plan is not None:
        units = surface.units_from_plan(logical_plan, shapes)
        control = surface.uniform_control(units, rule=args.control_rule)
        args.write_uniform_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_uniform_plan.write_bytes(
            indent2_json_file_bytes(
                stack_plan(control.plan, stack_members, layouts)))
        print(f"  -> {args.write_uniform_plan}  (uniform {control.grid} "
              f"R{control.q256}, the control arm)")
    print(f"  -> {args.out}\n  -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
