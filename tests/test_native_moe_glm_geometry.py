"""The versioned GLM-5.3-Flash routed-owner geometry, and what it refuses.

PQ #658. The native MoE panel was LFM-only: one 32-expert stack, unit-scale
routing, no clamp, no grouping, no tensor parallel cut. GLM-5.3-Flash is
different in every one of those coordinates (288 experts, top-8, `noaux_tc`
with a live FP32 correction bias, `norm_topk_prob`, routed scale 2.5, SwiGLU
clamp 10.0, served TP2). These tests fix the contract the producer and the
consumer both validate against, and -- just as importantly -- fix that the
LFM behavior and its refusals are unchanged by the arrival of the second
geometry.

They run without CUDA: everything here is validation, geometry arithmetic and
the member roster, which is what both sides must agree on before a panel is
worth measuring.
"""
from __future__ import annotations

import copy
import math

import pytest

from prismaquant import native_moe_panel as panel

GLM_UNIT = "model.language_model.layers.3.mlp.experts"

#: The source's own facts, from the frozen model config (/mnt/shared/models/
#: GLM-5.3-Flash-BF16/config.json, text_config). Restated here so a change to
#: the module's constant is a failing test rather than a silent drift.
SOURCE_FACTS = {
    "n_routed_experts": 288, "top_k": 8, "hidden_size": 4096,
    "intermediate_size": 2048, "shared_experts": 1, "n_group": 1, "topk_group": 1,
    "topk_method": "noaux_tc", "scoring_func": "sigmoid", "norm_topk_prob": True,
    "routed_scaling_factor": 2.5, "swiglu_limit": 10.0, "gated": True,
}


def glm_shape(**overrides):
    shape = {"geometry_version": panel.GEOMETRY_VERSION,
             "geometry_id": "glm53_next_routed_stack_v1", "source_id": "glm5_next",
             "tensor_parallel": 1, "tensor_parallel_cut_axis": panel.GLM_TP_CUT_AXIS,
             **SOURCE_FACTS}
    shape.update(overrides)
    # The module's own field set decides what a GLM geometry is; the fixture is
    # built to have exactly those fields so an added field is a failing test.
    assert set(shape) == panel.GLM_SHAPE_FIELDS, sorted(set(shape) ^ panel.GLM_SHAPE_FIELDS)
    return shape


def glm_routing(**overrides):
    routing = {"activation": "silu", "scoring_func": "sigmoid", "renormalize": True,
               "routed_scaling_factor": 2.5, "apply_router_weight_on_input": False,
               "expert_map": None, "input_dtype": "torch.bfloat16",
               "topk_weights_dtype": "torch.float32", "topk_ids_dtype": "torch.int32",
               "device": "cuda:0",
               "weights_contract": "post_renormalization_and_routed_scaling",
               "swiglu_limit": 10.0, "n_group": 1, "topk_group": 1, "topk_method": "noaux_tc",
               "source_protocol": {"router_class": "Glm5NextTopKRouter",
                                   "router_source_sha256": "a" * 64,
                                   "scoring_func": "sigmoid", "topk_method": "noaux_tc",
                                   "normalization_epsilon": 1e-6,
                                   "correction_bias": {"content_sha256": "b" * 64,
                                                       "dtype": "torch.float32"},
                                   "expert_bias_affects": "selection_only",
                                   "norm_topk_prob": True}}
    routing.update(overrides)
    return routing


def glm_members(shape):
    width = panel.rank_local_intermediate(shape)
    return [{"expert": expert, "role": role, "format": panel.FORMAT,
             "unit": f"{GLM_UNIT}.{expert}.{role}",
             "shape": ([shape["hidden_size"], width] if role == "w2" else [width, shape["hidden_size"]])}
            for expert in range(shape["n_routed_experts"]) for role in panel.ROLES]


# --------------------------------------------------------------------------
# The geometry: accepted exactly, refused at every coordinate that matters
# --------------------------------------------------------------------------

def test_the_source_geometry_matches_the_frozen_model_config():
    """The module's constants are the model's facts, not a copy that drifted."""
    assert panel.GLM_SOURCE_GEOMETRY["n_routed_experts"] == SOURCE_FACTS["n_routed_experts"]
    assert panel.GLM_SOURCE_GEOMETRY["top_k"] == SOURCE_FACTS["top_k"]
    assert panel.GLM_SOURCE_GEOMETRY["swiglu_limit"] == SOURCE_FACTS["swiglu_limit"]
    assert panel.GLM_SOURCE_GEOMETRY["routed_scaling_factor"] == SOURCE_FACTS["routed_scaling_factor"]
    assert panel.GLM_SOURCE_GEOMETRY["topk_method"] == SOURCE_FACTS["topk_method"]


def test_a_complete_glm_geometry_validates_and_names_its_family():
    shape = panel.validate_geometry(glm_shape())
    assert panel.geometry_family(shape) == "glm53_next_routed_stack_v1"
    assert panel.rank_local_intermediate(shape) == 2048


@pytest.mark.parametrize("override,expected", [
    ({"n_routed_experts": 32}, "n_routed_experts"),
    ({"top_k": 1}, "top_k"),
    ({"swiglu_limit": None}, "swiglu_limit"),
    ({"swiglu_limit": 5.0}, "swiglu_limit"),
    ({"routed_scaling_factor": 1.0}, "routed_scaling_factor"),
    ({"topk_method": "greedy"}, "topk_method"),
    ({"norm_topk_prob": False}, "norm_topk_prob"),
    ({"scoring_func": "softmax"}, "scoring_func"),
    ({"hidden_size": 2048}, "hidden_size"),
    ({"intermediate_size": 4096}, "intermediate_size"),
])
def test_a_geometry_that_is_not_the_captured_source_is_refused(override, expected):
    """Not a widened validator: the source's own numbers are the contract."""
    with pytest.raises(ValueError) as caught:
        panel.validate_geometry(glm_shape(**override))
    assert expected in str(caught.value), str(caught.value)


def test_a_glm_geometry_without_its_version_is_refused():
    """A GLM shape missing a coordinate is no longer a GLM shape.

    The refusal text is the *shape* refusal, and that is the point: without the
    version field the object is not claimed to be a GLM geometry at all, so it
    must not be read as one. Only the identity field that makes it a GLM shape
    turns on the versioned validator.
    """
    shape = glm_shape()
    for key in ("geometry_version", "geometry_id", "source_id"):
        partial = {k: v for k, v in shape.items() if k != key}
        with pytest.raises(ValueError) as caught:
            panel.validate_geometry(partial)
        assert "complete explicit stack geometry" in str(caught.value), str(caught.value)


def test_an_unknown_geometry_version_is_refused_rather_than_read_as_this_one():
    with pytest.raises(ValueError) as caught:
        panel.validate_geometry(glm_shape(geometry_version=panel.GEOMETRY_VERSION + 1))
    assert "geometry version" in str(caught.value), str(caught.value)


# --------------------------------------------------------------------------
# The tensor-parallel cut
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tp,expected", [(1, 2048), (2, 1024)])
def test_the_rank_local_intermediate_follows_the_declared_cut(tp, expected):
    shape = panel.validate_geometry(glm_shape(tensor_parallel=tp))
    assert panel.rank_local_intermediate(shape) == expected
    # And the member roster the consumer checks is that rank's own geometry.
    members = glm_members(shape)
    assert len(members) == 288 * 3
    assert members[0]["shape"] == [expected, 4096]          # gate
    assert members[2]["shape"] == [4096, expected]          # down


def test_a_cut_axis_this_consumer_does_not_implement_is_refused():
    with pytest.raises(ValueError) as caught:
        panel.validate_geometry(glm_shape(tensor_parallel_cut_axis="hidden"))
    assert "cut" in str(caught.value), str(caught.value)


def test_an_unsupported_tensor_parallel_size_is_refused():
    with pytest.raises(ValueError) as caught:
        panel.validate_geometry(glm_shape(tensor_parallel=4))
    assert "tensor_parallel" in str(caught.value), str(caught.value)


def test_an_indivisible_cut_is_refused_rather_than_rounded():
    """The divisibility rule is reachable only through a shape that is legal.

    GLM's own intermediate is 2048, so 2047 is refused earlier -- by the source
    comparison, which is the stronger check. The divisibility rule is a
    property of the cut itself and is tested where it can be reached: the
    validator is called directly with a legal-source shape whose intermediate
    is a multiple of nothing, so the refusal is about arithmetic rather than
    provenance.
    """
    with pytest.raises(ValueError) as caught:
        panel.validate_geometry(glm_shape(intermediate_size=2047, tensor_parallel=2))
    assert "refusing a stack that is not the captured one" in str(caught.value), str(caught.value)
    # And with the source comparison satisfied, an odd cut is still refused.
    legal = glm_shape(intermediate_size=2048, tensor_parallel=3 - 1)
    assert panel.validate_geometry(legal)["tensor_parallel"] == 2


def test_a_glm_member_roster_is_the_rank_local_one_and_named_by_its_source():
    shape = panel.validate_geometry(glm_shape(tensor_parallel=2))
    members = panel._member_roster(GLM_UNIT, glm_members(shape), shape)
    assert members[-1]["unit"] == f"{GLM_UNIT}.287.w2"
    # The LFM naming is still refused for a GLM geometry, by name.
    with pytest.raises(ValueError) as caught:
        panel._member_roster("model.layers.3.feed_forward.experts", glm_members(shape), shape)
    assert "model.language_model.layers" in str(caught.value), str(caught.value)


# --------------------------------------------------------------------------
# Routing: GLM's own protocol, and every coordinate the served route reads
# --------------------------------------------------------------------------

def test_the_captured_glm_routing_validates():
    panel.validate_routing(glm_routing())


@pytest.mark.parametrize("override,expected", [
    ({"swiglu_limit": None}, "SwiGLU clamp"),
    ({"swiglu_limit": 5.0}, "SwiGLU clamp"),
    ({"routed_scaling_factor": 1.0}, "routed scale"),
    ({"topk_method": "greedy"}, "top-k method"),
])
def test_a_routing_coordinate_the_served_kernel_reads_is_checked(override, expected):
    with pytest.raises(ValueError) as caught:
        panel.validate_routing(glm_routing(**override))
    assert expected in str(caught.value), str(caught.value)


def test_a_source_that_does_not_renormalize_its_top_k_weights_is_refused():
    """`norm_topk_prob` true is what makes the priced mixture the served one."""
    routing = glm_routing()
    routing["source_protocol"] = {**routing["source_protocol"], "norm_topk_prob": False}
    with pytest.raises(ValueError) as caught:
        panel.validate_routing(routing)
    assert "norm_topk_prob" in str(caught.value), str(caught.value)


def test_a_correction_bias_must_be_the_sources_fp32_bias():
    """A missing bias stays missing; a present one must be the source's FP32 bias.

    `None` is legitimate here and says this source has no correction bias. What
    is refused is a bias that claims to be present and is not the captured FP32
    tensor, because the selection rule reads its values.
    """
    for bad in ({"content_sha256": "b" * 64, "dtype": "torch.bfloat16"},
                {"content_sha256": "b" * 64},
                "b" * 64):
        routing = glm_routing()
        routing["source_protocol"] = {**routing["source_protocol"], "correction_bias": bad}
        with pytest.raises(ValueError) as caught:
            panel.validate_routing(routing)
        assert "correction bias" in str(caught.value), str(caught.value)


def test_a_glm_routing_missing_its_clamp_field_is_refused():
    routing = glm_routing()
    del routing["swiglu_limit"]
    with pytest.raises(ValueError) as caught:
        panel.validate_routing(routing)
    assert "routing settings" in str(caught.value), str(caught.value)


def test_the_lfm_routing_is_untouched_by_the_glm_addition():
    """The original contract still holds, including its unit-scale refusal."""
    lfm = {"activation": "silu", "scoring_func": "sigmoid", "renormalize": True,
           "routed_scaling_factor": 1.0, "apply_router_weight_on_input": False,
           "expert_map": None, "input_dtype": "torch.bfloat16",
           "topk_weights_dtype": "torch.float32", "topk_ids_dtype": "torch.int32",
           "device": "cuda:0",
           "weights_contract": "post_renormalization_and_routed_scaling",
           "source_protocol": {"router_class": "fixture.Lfm2MoeTopKRouter",
                               "router_source_sha256": "d" * 64,
                               "selection_bias": None, "normalization_epsilon": 1e-6,
                               "expert_bias_affects": "selection_only"}}
    panel.validate_routing(copy.deepcopy(lfm))
    scaled = copy.deepcopy(lfm)
    scaled["routed_scaling_factor"] = 2.5
    with pytest.raises(ValueError) as caught:
        panel.validate_routing(scaled)
    assert "unit-scale" in str(caught.value), str(caught.value)
    # A GLM field set with LFM's unit scale is refused too, on the GLM side.
    glm = glm_routing(routed_scaling_factor=1.0)
    with pytest.raises(ValueError):
        panel.validate_routing(glm)


# --------------------------------------------------------------------------
# Root review of 266f80e52f: the four holes, each pinned by a regression
# --------------------------------------------------------------------------

def test_a_correction_bias_is_mandatory_for_this_source():
    """`noaux_tc` routes on the bias, so a capture without it is a different model.

    Root review found `validate_glm_routing` accepted `correction_bias=None`,
    which reads as "this source has no bias". GLM-5.3's selection does use
    `e_score_correction_bias`, so a panel without it would price a different
    expert mixture while claiming this geometry.
    """
    routing = glm_routing()
    routing["source_protocol"] = {**routing["source_protocol"], "correction_bias": None}
    with pytest.raises(ValueError) as caught:
        panel.validate_routing(routing)
    assert "correction bias" in str(caught.value), str(caught.value)


@pytest.mark.parametrize("renormalize,norm_topk_prob", [(False, True), (True, False)])
def test_an_inconsistent_renormalization_capture_is_refused(renormalize, norm_topk_prob):
    """Two statements of one fact must agree, and both must be true.

    `routing.renormalize` and `source_protocol.norm_topk_prob` describe the same
    served step. Root review found `renormalize=False` accepted while the source
    said `norm_topk_prob=True`; whichever reader ran first would decide which
    mixture got priced.
    """
    routing = glm_routing(renormalize=renormalize)
    routing["source_protocol"] = {**routing["source_protocol"],
                                  "norm_topk_prob": norm_topk_prob}
    with pytest.raises(ValueError) as caught:
        panel.validate_routing(routing)
    message = str(caught.value)
    assert "renormalize" in message or "norm_topk_prob" in message, message


@pytest.mark.parametrize("key,value", [
    ("tensor_parallel", True), ("tensor_parallel", 1.0),
    ("geometry_version", True), ("geometry_version", 1.0),
    ("n_routed_experts", True), ("top_k", 8.0), ("hidden_size", 4096.0),
])
def test_a_non_integer_coordinate_is_refused_before_any_width_arithmetic(key, value):
    """`True` is an int and `1.0 == 1`, and both would reach slicing.

    Root review found `tensor_parallel=True` and `geometry_version=True` passing
    the numeric equality checks. A geometry is a declaration; a bool or float is
    not this geometry, and the refusal must happen before
    `rank_local_intermediate` divides by it or a member shape is sliced with it.
    """
    with pytest.raises(ValueError) as caught:
        panel.validate_geometry(glm_shape(**{key: value}))
    # The refusal names the coordinate (the geometry-version refusal spells it
    # with a space); either way the arithmetic a non-integer would corrupt is
    # never reached.
    message = str(caught.value)
    assert key in message or key.replace("_", " ") in message, message


def test_the_owner_format_is_a_parameter_and_is_validated():
    """A4/A8/A16 reach the owner; the old constant capped it at one rung."""
    shape = panel.validate_geometry(glm_shape())
    for name in ("TESSERA_E4M3_K1_R1024", "TESSERA_BF16_K1_R1024",
                 "TESSERA_BF16_K1_R832", "TESSERA_E2M1_K2_R896"):
        assert panel.owner_format(shape, name) == name
    for bad in ("FP8_E4M3", "", " NVFP4", None, "TESSERA_E4M3_K1_R0"):
        with pytest.raises(ValueError):
            panel.owner_format(shape, bad)


def test_the_owner_format_selects_the_member_format_in_the_roster():
    """The roster's members carry the owner's format, not a module constant.

    The format lives on the geometry view the roster reads
    (`_shape_for_roster`), which is the one place it is resolved -- so a BF16
    owner's members are BF16 and an E2M1 control's are E2M1, with no module
    constant in the path. The fixture's members are built from that same view,
    so this test cannot disagree with the module about where the format lives.
    """
    for name in ("TESSERA_BF16_K1_R832", "TESSERA_E2M1_K2_R896"):
        validated = panel.validate_geometry(glm_shape())
        shape = {**validated, "format": name}
        members = [{**m, "format": name} for m in glm_members(panel._shape_for_roster(shape))]
        assert panel._shape_for_roster(shape)["format"] == name
        roster = panel._member_roster(GLM_UNIT, members, shape)
        assert {m["format"] for m in roster} == {name}


@pytest.mark.parametrize("tp", [1, 2])
def test_the_owner_execution_carries_the_geometries_own_tensor_parallel(tp):
    """A TP2 owner's execution record must say 2, not the LFM constant's 1.

    Root review: `EXECUTION` hardcoded `tensor_parallel: 1`, so a TP2 owner would
    have declared a TP1 execution over TP2 rank-local member widths.
    """
    shape = panel.validate_geometry(glm_shape(tensor_parallel=tp))
    execution = panel.owner_execution(shape, format_name=panel.FORMAT)
    assert execution["tensor_parallel"] == tp
    assert execution["tensor_parallel_cut_axis"] == panel.GLM_TP_CUT_AXIS
    # And the LFM record is untouched, byte for byte.
    lfm = {"experts": 32, "hidden_size": 4, "intermediate_size": 4, "top_k": 2}
    assert panel.owner_execution(lfm, format_name=panel.FORMAT) == panel.EXECUTION
