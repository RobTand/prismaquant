"""The serving plan is a PrismaQuant decision document (prismaquant #1587).

PrismaQuant's allocator chose the rates, so PrismaQuant writes the exporter's
plan itself, in the schema Tessera publishes (``tessera.serving_plan.v1``,
RobTand/tessera#687).  The model classification and the fused-scheme rule stay
Tessera's: the writer imports them from the Tessera package (never from
``experiments/``), so there is one statement of what a body Linear and a fused
scheme are.  The charged-bits sidecar is computed with PrismaQuant's own
``artifact_bpp`` -- in-tree now, with no ``--prismaquant`` tree to pass.

These tests drive the planning logic through a synthetic Tessera surface: the
same names the real package exports (``quantizable``, ``expert_stacks``,
``fused_module``, ``module_scheme_key`` ...), faked in-process so the decision
logic is testable on any pin.  The real-package import itself is covered by a
gated test that runs wherever ``tessera.export_serving`` is importable, and the
end-to-end export is covered by ``test_tessera_plan_bytes_equal.py``.
"""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import tessera_plan_writer as writer

# -- the synthetic Tessera package surface ------------------------------------

QKV = ("model.layers.0.self_attn.q_proj.weight",
       "model.layers.0.self_attn.k_proj.weight",
       "model.layers.0.self_attn.v_proj.weight")
GATE_UP = ("model.layers.0.mlp.gate_proj.weight",
           "model.layers.0.mlp.up_proj.weight")
DENSE = {
    QKV[0]: (32, 32), QKV[1]: (8, 32), QKV[2]: (8, 32),
    "model.layers.0.self_attn.o_proj.weight": (32, 32),
    GATE_UP[0]: (64, 32), GATE_UP[1]: (64, 32),
    "model.layers.0.mlp.down_proj.weight": (32, 64),
    "model.layers.1.self_attn.o_proj.weight": (32, 32),
    "model.layers.1.mlp.down_proj.weight": (32, 64),
}
FUSED = {
    "model.layers.0.self_attn": QKV,
    "model.layers.0.mlp": GATE_UP,
}
STACK = "model.layers.2.mlp.experts"
STACK_TENSORS = {
    f"{STACK}.expert_0.{role}.weight": shape
    for role, shape in (("w1", (16, 32)), ("w3", (16, 32)), ("w2", (32, 16)))
}
ROUTER = "model.layers.2.mlp.gate"
ROSTER_IDENTITY = {"config_sha256": "c" * 64, "files": [], "tensors": {}}


def _surface(*, dense=None, routed=None, stacks=None, scheme=None):
    """A fake with the exact names the writer imports from the package."""
    dense = dict(DENSE) if dense is None else dense
    routed = dict(routed or {})
    stacks = stacks if stacks is not None else {}

    def fused_module(tensor):
        for module, members in FUSED.items():
            if tensor in members:
                return module, members
        return None

    def quantizable(model):
        return ([], dict(dense), {}, dict(routed))

    def expert_stacks(routed_shapes):
        return {stack: {0: {name: shape for name, shape in tensors.items()}}
                for stack, tensors in stacks.items()}

    return SimpleNamespace(
        MOE_ROUTER=re.compile(r"\.mlp\.gate\.weight$"),
        MOE_SOURCE_UNPACKED="unpacked_per_expert",
        TesseraError=type("TesseraError", (Exception,), {}),
        quantizable=quantizable,
        expert_stacks=expert_stacks,
        packed_expert_stacks=lambda packed: {},
        project_expert_plan=lambda *a, **k: {"stacks": {}},
        fused_module=fused_module,
        body_layer=lambda tensor: int(tensor.split(".")[2]),
        family_for=lambda grid: {"E4M3": "TESSERA_E4M3",
                                 "E2M1x2": "TESSERA_NVFP4"}[grid],
        grid_for_name=lambda name: SimpleNamespace(name=name),
        module_scheme_key=lambda grid, q256, structure="dense": (
            scheme(grid) if scheme else (
                {"E4M3": "TESSERA_E4M3", "E2M1x2": "TESSERA_NVFP4"}[grid.name],
                grid.name, "WINDOW", "LUT")),
        units_from_plan=lambda plan, shapes: [
            SimpleNamespace(grid=value["grid"], q256=value["q256"])
            for value in plan.values() if isinstance(value, dict)],
        uniform_control=lambda units, rule="nearest", assert_match=False:
            SimpleNamespace(grid="E4M3", q256=1024,
                            plan={"model.layers.0.mlp.down_proj.weight":
                                  {"grid": "E4M3", "q256": 1024}}),
        control_block=lambda control: {
            "schema": "tessera.uniform_control.v1",
            "control": {"grid": control.grid, "q256": control.q256,
                        "match": {"byte_matched": True, "control_bpp": 4.0,
                                  "candidate_bpp": 4.0, "relative_slack_ppm": 0.0,
                                  "fatter_arm": "neither"}}},
        selection_requirement=lambda units: {
            "requires_validation": len({(u.grid, u.q256) for u in units}) > 1,
            "distinct_rungs": sorted({(u.grid, u.q256) for u in units}),
            "mode_required": "validated-surrogate"},
        SOURCE_ROSTER_FIELDS=("config_sha256", "files", "tensors"),
        source_roster_identity=lambda model: ROSTER_IDENTITY,
    )


def _unit(tensor):
    return tensor[: -len(".weight")]


def _context(surface, config, tmp_path, *, research_selected=False):
    return writer.model_plan_context(
        tmp_path, config, surface, research_selected=research_selected)


# -- planning: coverage, BF16 completion, per-role rungs ----------------------

def test_as_allocated_plans_exactly_the_allocation_and_completes_with_bf16(
        tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {
        "model.layers.0.self_attn.q_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        "model.layers.0.self_attn.k_proj": {"tessera_format": "TESSERA_E4M3_K1_R896"},
        "model.layers.0.self_attn.v_proj": {"tessera_format": "TESSERA_E4M3_K1_R768"},
        "model.layers.0.mlp.gate_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        "model.layers.0.mlp.up_proj": {"tessera_format": "TESSERA_E4M3_K1_R896"},
        "model.layers.1.mlp.down_proj": "BF16",
    }
    shapes, members, layouts = _context(surface, config, tmp_path)
    plan, provenance = writer.build(config, shapes, cover="as-allocated",
                                    allow_disagreement=False,
                                    control_rule="nearest", with_control=True,
                                    surface=surface)
    assert plan["model.layers.0.self_attn.q_proj.weight"] == {"grid": "E4M3", "q256": 1024}
    assert plan["model.layers.0.self_attn.k_proj.weight"] == {"grid": "E4M3", "q256": 896}
    assert plan["model.layers.0.self_attn.v_proj.weight"] == {"grid": "E4M3", "q256": 768}
    assert plan["model.layers.0.mlp.gate_proj.weight"] == {"grid": "E4M3", "q256": 1024}
    assert plan["model.layers.0.mlp.up_proj.weight"] == {"grid": "E4M3", "q256": 896}
    # differing rungs inside one fused module are NOT a disagreement (#37)
    assert provenance["fused_disagreements"] == []
    # every body Linear the allocation did not name is explicit BF16
    assert plan["model.layers.0.self_attn.o_proj.weight"] == "BF16"
    assert plan["model.layers.0.mlp.down_proj.weight"] == "BF16"
    assert plan["model.layers.1.self_attn.o_proj.weight"] == "BF16"
    assert plan["model.layers.1.mlp.down_proj.weight"] == "BF16"
    coverage = provenance["coverage"]
    assert coverage["mode"] == "as-allocated"
    assert coverage["planned_tessera_units"] == 5
    assert coverage["planned_bf16_units"] == 4
    assert coverage["unplanned_body_linears"] == 0


def test_fused_scheme_disagreement_refuses(tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {
        "model.layers.0.self_attn.q_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        "model.layers.0.self_attn.k_proj": {"tessera_format": "TESSERA_E2M1_K2_R896"},
        "model.layers.0.self_attn.v_proj": {"tessera_format": "TESSERA_E4M3_K1_R768"},
    }
    shapes, _members, _layouts = _context(surface, config, tmp_path)
    with pytest.raises(SystemExit, match="do not share one scheme"):
        writer.build(config, shapes, cover="as-allocated", allow_disagreement=False,
                     control_rule="nearest", with_control=True, surface=surface)


def test_allow_fused_disagreement_demotes_the_whole_group_to_bf16(
        tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {
        "model.layers.0.self_attn.q_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        "model.layers.0.self_attn.k_proj": {"tessera_format": "TESSERA_E2M1_K2_R896"},
        "model.layers.0.self_attn.v_proj": {"tessera_format": "TESSERA_E4M3_K1_R768"},
        "model.layers.0.mlp.gate_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        "model.layers.0.mlp.up_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
    }
    shapes, _m, _l = _context(surface, config, tmp_path)
    plan, provenance = writer.build(config, shapes, cover="as-allocated",
                                    allow_disagreement=True,
                                    control_rule="nearest", with_control=True,
                                    surface=surface)
    for member in QKV:
        assert plan[member] == "BF16"
    disagreement = provenance["fused_disagreements"][0]
    assert disagreement["module"] == "model.layers.0.self_attn"
    assert disagreement["planned_as"] == "BF16"
    assert provenance["totals"]["demoted_to_bf16_params"] == sum(
        DENSE[m][0] * DENSE[m][1] for m in QKV)
    # the agreeing fused module is untouched
    assert plan["model.layers.0.mlp.gate_proj.weight"] == {"grid": "E4M3", "q256": 1024}


def test_partially_allocated_fused_module_is_a_disagreement(tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {"model.layers.0.self_attn.q_proj":
              {"tessera_format": "TESSERA_E4M3_K1_R1024"}}
    shapes, _m, _l = _context(surface, config, tmp_path)
    with pytest.raises(SystemExit, match="do not share one scheme"):
        writer.build(config, shapes, cover="as-allocated", allow_disagreement=False,
                     control_rule="nearest", with_control=True, surface=surface)


# -- refusals the inputs decide on their own ----------------------------------

def test_group_option_spelling_is_refused_by_name(tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {"model.layers.0.mlp.down_proj":
              {"tessera_format": "TESSERA_E4M3_K1_G3"}}
    with pytest.raises(SystemExit, match="whole-GROUP option"):
        writer.parse_entry("model.layers.0.mlp.down_proj",
                           config["model.layers.0.mlp.down_proj"])


def test_non_tessera_quantized_choice_is_refused(tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {"model.layers.0.mlp.down_proj": "AWQ"}
    with pytest.raises(SystemExit, match="non-Tessera QUANTISED choice"):
        writer.refuse_before_source(config, surface)


def test_immutable_router_must_remain_bf16(tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {ROUTER: {"tessera_format": "TESSERA_E4M3_K1_R1024"}}
    with pytest.raises(SystemExit, match="must remain BF16"):
        writer.refuse_before_source(config, surface)


def test_allocation_unit_absent_from_the_model_refuses(tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {"model.layers.9.mlp.up_proj":
              {"tessera_format": "TESSERA_E4M3_K1_R1024"}}
    shapes, members, layouts = _context(surface, config, tmp_path)
    with pytest.raises(SystemExit,
                       match="absent from the producer's logical body projection"):
        writer.plan_from_assignment(config, shapes, members, layouts,
                                    model=tmp_path, cover="as-allocated",
                                    allow_disagreement=False,
                                    control_rule="nearest", with_control=True,
                                    surface=surface)


# -- broadcast-by-role --------------------------------------------------------

def test_broadcast_by_role_extrapolates_and_stamps(tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {"model.layers.0.mlp.down_proj":
              {"tessera_format": "TESSERA_E4M3_K1_R1024"}}
    shapes, _m, _l = _context(surface, config, tmp_path)
    plan, provenance = writer.build(config, shapes, cover="broadcast-by-role",
                                    allow_disagreement=False,
                                    control_rule="nearest", with_control=True,
                                    surface=surface)
    assert plan["model.layers.0.mlp.down_proj.weight"] == {"grid": "E4M3", "q256": 1024}
    assert plan["model.layers.1.mlp.down_proj.weight"] == {"grid": "E4M3", "q256": 1024}
    assert provenance["coverage"]["extrapolated"] is True
    assert provenance["coverage"]["broadcast_from_layer"] == 0


def test_broadcast_by_role_needs_a_single_layer(tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {
        "model.layers.0.mlp.down_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        "model.layers.1.mlp.down_proj": {"tessera_format": "TESSERA_E4M3_K1_R896"},
    }
    with pytest.raises(SystemExit, match="single-layer allocation"):
        writer.build(config, dict(DENSE),
                     cover="broadcast-by-role", allow_disagreement=False,
                     control_rule="nearest", with_control=True, surface=surface)


# -- routed expert stacks -----------------------------------------------------

def _stack_surface():
    return _surface(
        dense={k: v for k, v in DENSE.items()},
        routed={name: shape for name, shape in STACK_TENSORS.items()},
        stacks={STACK: STACK_TENSORS})


def test_a_routed_stack_is_planned_at_one_exact_rung(tmp_path, monkeypatch):
    surface = _stack_surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    fmt = {"tessera_format": "TESSERA_E4M3_K1_R1024"}
    config = {
        "model.layers.0.self_attn.q_proj": fmt,
        "model.layers.0.self_attn.k_proj": fmt,
        "model.layers.0.self_attn.v_proj": fmt,
        **{f"{STACK}.expert_0.{role}": fmt
           for role in ("w1", "w3", "w2")},
    }
    shapes, members, layouts = _context(surface, config, tmp_path)
    plan, provenance = writer.plan_from_assignment(
        config, shapes, members, layouts, model=tmp_path, cover="as-allocated",
        allow_disagreement=False, control_rule="nearest", with_control=True,
        surface=surface)
    assert plan[STACK] == {"grid": "E4M3", "q256": 1024,
                           "source_layout": "unpacked_per_expert"}
    for tensor in STACK_TENSORS:
        assert tensor not in plan
    assert provenance["expert_stacks"][STACK]["planned_as"] == {
        "grid": "E4M3", "q256": 1024, "source_layout": "unpacked_per_expert"}


def test_mixed_choices_inside_a_stack_refuse(tmp_path, monkeypatch):
    surface = _stack_surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {
        "model.layers.0.self_attn.q_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        "model.layers.0.self_attn.k_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        "model.layers.0.self_attn.v_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        f"{STACK}.expert_0.w1": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        f"{STACK}.expert_0.w3": {"tessera_format": "TESSERA_E4M3_K1_R896"},
        f"{STACK}.expert_0.w2": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
    }
    shapes, members, layouts = _context(surface, config, tmp_path)
    with pytest.raises(SystemExit, match="one exact rung"):
        writer.plan_from_assignment(config, shapes, members, layouts,
                                    model=tmp_path, cover="as-allocated",
                                    allow_disagreement=False,
                                    control_rule="nearest", with_control=True,
                                    surface=surface)


def test_carried_projection_supplies_the_stack_membership(tmp_path, monkeypatch):
    surface = _stack_surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    packed_layout = "out_first_chunked"
    producer_stacks = {
        STACK: {"source_layout": packed_layout,
                "units": [{"tensor": name, "rows": shape[0], "cols": shape[1]}
                          for name, shape in STACK_TENSORS.items()]}}
    surface.project_expert_plan = lambda *a, **k: {"stacks": producer_stacks}
    fmt = {"tessera_format": "TESSERA_E4M3_K1_R1024"}
    config = {
        "model.layers.0.self_attn.q_proj": fmt,
        "model.layers.0.self_attn.k_proj": fmt,
        "model.layers.0.self_attn.v_proj": fmt,
        **{f"{STACK}.expert_0.{role}": fmt for role in ("w1", "w3", "w2")},
        "__prismaquant__": {
            "tessera_expert_projection": {
                "schema": "prismaquant.tessera_expert_projection.v1",
                "request": {"stacks": [STACK]},
                "producer": {
                    "schema": "tessera.expert_projection.v1",
                    "source": dict(ROSTER_IDENTITY),
                    "stacks": producer_stacks},
                "stacks": {
                    STACK: {
                        _unit(name): {"tensor": name,
                                      "rows": shape[0], "cols": shape[1]}
                        for name, shape in STACK_TENSORS.items()}}},
        },
    }
    shapes, members, layouts = _context(surface, config, tmp_path)
    assert layouts[STACK] == packed_layout
    plan, provenance = writer.plan_from_assignment(
        config, shapes, members, layouts, model=tmp_path, cover="as-allocated",
        allow_disagreement=False, control_rule="nearest", with_control=True,
        surface=surface)
    assert plan[STACK]["source_layout"] == packed_layout


def test_carried_projection_disagreement_with_the_producer_refuses(
        tmp_path, monkeypatch):
    surface = _stack_surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    producer_stacks = {
        STACK: {"source_layout": "unpacked_per_expert",
                "units": [{"tensor": name, "rows": shape[0], "cols": shape[1]}
                          for name, shape in STACK_TENSORS.items()]}}
    surface.project_expert_plan = lambda *a, **k: {"stacks": producer_stacks}
    fmt = {"tessera_format": "TESSERA_E4M3_K1_R1024"}
    config = {
        "model.layers.0.self_attn.q_proj": fmt,
        "model.layers.0.self_attn.k_proj": fmt,
        "model.layers.0.self_attn.v_proj": fmt,
        f"{STACK}.expert_0.w1": fmt,
        "__prismaquant__": {
            "tessera_expert_projection": {
                "schema": "prismaquant.tessera_expert_projection.v1",
                "request": {"stacks": [STACK]},
                "producer": {
                    "schema": "tessera.expert_projection.v1",
                    "source": {"config_sha256": "c" * 64},
                    "stacks": {"OTHER": {"units": []}}},
                "stacks": {}},
        },
    }
    with pytest.raises(SystemExit,
                       match="current source projection|stack binding"):
        writer.model_plan_context(tmp_path, config, surface)


# -- the charged-bits sidecar is PrismaQuant's own accounting -----------------

def test_the_sidecar_carries_prismaquant_wire_accounting(tmp_path, monkeypatch):
    from prismaquant.tessera_formats import artifact_bpp
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    config = {
        "model.layers.0.self_attn.q_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        "model.layers.0.self_attn.k_proj": {"tessera_format": "TESSERA_E4M3_K1_R896"},
        "model.layers.0.self_attn.v_proj": {"tessera_format": "TESSERA_E4M3_K1_R768"},
        "model.layers.0.mlp.gate_proj": {"tessera_format": "TESSERA_E4M3_K1_R1024"},
        "model.layers.0.mlp.up_proj": {"tessera_format": "TESSERA_E4M3_K1_R896"},
    }
    shapes, _m, _l = _context(surface, config, tmp_path)
    plan, provenance = writer.build(config, shapes, cover="as-allocated",
                                    allow_disagreement=False,
                                    control_rule="nearest", with_control=True,
                                    surface=surface)
    units = {u["qname"]: u for u in provenance["units"]}
    expected = Fraction(artifact_bpp("TESSERA_E4M3", 1024, shape=(32, 32))) * 32 * 32
    assert units["model.layers.0.self_attn.q_proj"]["prismaquant_charged_bits_exact"] == \
        [expected.numerator, expected.denominator]
    assert units["model.layers.0.self_attn.q_proj"]["prismaquant_charged_bpp"] == \
        float(expected / (32 * 32))
    assert provenance["totals"]["prismaquant_charged_bpp"] is not None


# -- the plan document and its provenance -------------------------------------

def test_the_plan_and_provenance_schemas_are_named(tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    fmt = {"tessera_format": "TESSERA_E4M3_K1_R1024"}
    config = {f"model.layers.0.self_attn.{role}_proj": fmt
              for role in ("q", "k", "v")}
    config.update({f"model.layers.0.mlp.{role}_proj": fmt
                   for role in ("gate", "up")})
    shapes, _m, _l = _context(surface, config, tmp_path)
    plan, provenance = writer.build(config, shapes, cover="as-allocated",
                                    allow_disagreement=False,
                                    control_rule="nearest", with_control=True,
                                    surface=surface)
    assert provenance["schema"] == "prismaquant.tessera_plan.v1"
    assert provenance["plan_schema"] == "tessera.serving_plan.v1"
    assert provenance["uniform_control"]["built"] is True
    assert provenance["selection"]["requires_validation"] is True


def test_main_writes_the_plan_and_the_sidecar(tmp_path, monkeypatch):
    surface = _surface()
    monkeypatch.setattr(writer, "tessera_surface", lambda: surface)
    (tmp_path / "model").mkdir()
    assignment = tmp_path / "layer_config.json"
    fmt = {"tessera_format": "TESSERA_E4M3_K1_R1024"}
    assignment.write_text(json.dumps({
        **{f"model.layers.0.self_attn.{role}_proj": fmt for role in ("q", "k", "v")},
        **{f"model.layers.0.mlp.{role}_proj": fmt for role in ("gate", "up")},
        "model.layers.1.mlp.down_proj": "BF16"}))
    out = tmp_path / "plan.json"
    rc = writer.main(["--model", str(tmp_path / "model"),
                      str(assignment), str(out)])
    assert rc == 0
    plan = json.loads(out.read_text())
    assert plan["model.layers.0.self_attn.q_proj.weight"] == {"grid": "E4M3", "q256": 1024}
    sidecar = json.loads(out.with_suffix(".json.provenance.json").read_text())
    assert sidecar["schema"] == "prismaquant.tessera_plan.v1"


# -- the Tessera surface itself ------------------------------------------------

def test_the_surface_imports_from_the_package_not_experiments():
    """``tessera_surface`` binds the package names; old pins refuse by name.

    On the 38e96012 pin the package does not carry ``tessera.export_serving``
    yet, so the writer must refuse with the pin requirement named
    (RobTand/tessera#687).  On a pin that packages it, the surface must bind
    the package's own objects.
    """
    try:
        module = __import__("tessera.export_serving")
        serving_plan = __import__("tessera.serving_plan")
    except ImportError:
        with pytest.raises(RuntimeError, match="tessera#687"):
            writer.tessera_surface()
        return
    surface = writer.tessera_surface()
    assert surface.quantizable is module.quantizable
    assert surface.module_scheme_key is serving_plan.module_scheme_key
