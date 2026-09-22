"""Whole routed-stack preparation and native evidence, without serving imports.

A stack measurement binds every original source/PWC/wire member and the actual
captured routed invocation. It never adds leaf medians or creates a new group
cost. Native loading, fused execution and resource collection belong to the
separate Tessera producer; fixed/full-model resources remain unknown here.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import re

from .joint_aura import identity_sha256, validate_joint_aura_entry
from .measured_runtime_prices import RuntimeBinding
from .native_operator_panel import (PHASES, _bytes, _equal, _number, _sha,
                                    operator_route_identity)
from .tessera_formats import parse_tessera_format_name

INPUT_SCHEMA = "prismaquant.native_moe_inputs.v1"
PANEL_SCHEMA = "tessera.native_moe_panel.v1"
#: One rank's own cut record: the window, not a quality identity.
RANK_RENDER_SCHEMA = "prismaquant.native_moe_rank_render.v1"
#: The LFM stack this panel was built for, in the shape its validator reads.
LFM_STACK = "lfm2_moe"
LFM_EXPERTS = 32
FORMAT = "TESSERA_E4M3_K1_R1024"
ROLES = ("w1", "w3", "w2")
EXECUTION = {"owner_kind": "complete_routed_moe", "mode": "resident",
             "execution_mode": "eager", "tensor_parallel": 1, "expert_parallel": 1,
             "include_router": False, "topk_selection": "external", "shared_experts": False,
             "monolithic": False, "bias": False}
ROUTING_FIELDS = {"activation", "scoring_func", "renormalize", "routed_scaling_factor",
                  "apply_router_weight_on_input", "expert_map", "input_dtype",
                  "topk_weights_dtype", "topk_ids_dtype", "device", "weights_contract", "source_protocol"}
#: The GLM stack's routing is a strict superset of the LFM field set: it adds
#: the SwiGLU clamp, the expert grouping and the source's own top-k method, and
#: the served route reads all three, so they are required rather than optional.
GLM_ROUTING_FIELDS = ROUTING_FIELDS | {"swiglu_limit", "n_group", "topk_group", "topk_method"}
#: GLM's router protocol differs from LFM's in what it *is*, not only in what
#: it is called: `noaux_tc` selection carries a live FP32 correction bias, and
#: `norm_topk_prob` decides whether the selected weights are renormalized.
GLM_SOURCE_PROTOCOL_FIELDS = {"router_class", "router_source_sha256", "scoring_func",
                              "topk_method", "normalization_epsilon", "correction_bias",
                              "expert_bias_affects", "norm_topk_prob"}

# --------------------------------------------------------------------------
# Versioned native routed-owner geometries.
#
# v1 was LFM-only: one complete 32-expert sigmoid/SiLU unit-scale stack named
# `model.layers.N.feed_forward.experts`, with no clamp, no grouping, no bias and
# no tensor parallelism.  GLM-5.3-Flash is a different source protocol in every
# one of those coordinates: 288 routed experts, top-8 selection, `noaux_tc`
# scoring with a live FP32 correction bias, `norm_topk_prob` renormalization, a
# routed scale of 2.5, a SwiGLU clamp at 10.0 and a served TP2 cut.  A geometry
# is therefore an explicit, versioned object that both sides validate against,
# not a widened constant: the LFM behavior and its refusals are preserved
# exactly, and GLM arrives as its own id with its own facts.
# --------------------------------------------------------------------------
GEOMETRY_VERSION = 1
#: The GLM-5.3-Flash routed-stack identity, read from the model config.
GLM_SOURCE_GEOMETRY = {
    "geometry_id": "glm53_next_routed_stack_v1",
    "source_id": "glm5_next",
    "n_routed_experts": 288,
    "top_k": 8,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "swiglu_limit": 10.0,
    "n_group": 1,
    "topk_group": 1,
    "gated": True,
    "shared_experts": 1,
    "intermediate_size": 2048,
    "hidden_size": 4096,
}
#: The served tensor-parallel cuts.  `nvfp4_moe_route.py:374-381` divides the
#: intermediate axis, so a rank-local stack carries `intermediate/tp` rows for
#: gate/up and columns for down.  Why a TP=1 native owner may price TP=2 is a
#: property of the CUT, not an assumption, and it is stated in the field.
GLM_TP_CUT_AXIS = "intermediate"
SUPPORTED_TP_SIZES = (1, 2)
_LFM_UNIT = re.compile(r"model\.layers\.[0-9]+\.feed_forward\.experts")
_GLM_UNIT = re.compile(r"model\.language_model\.layers\.[0-9]+\.mlp\.experts")
LFM_SHAPE_FIELDS = {"experts", "hidden_size", "intermediate_size", "top_k"}
GLM_SHAPE_FIELDS = {"geometry_version", "geometry_id", "source_id", "n_routed_experts",
                    "top_k", "hidden_size", "intermediate_size", "shared_experts", "n_group",
                    "topk_group", "topk_method", "scoring_func", "norm_topk_prob",
                    "routed_scaling_factor", "swiglu_limit", "gated", "tensor_parallel",
                    "tensor_parallel_cut_axis"}
#: The producer's canonical wire spelling for each role this consumer prices,
#: pinned from Tessera's ``MOE_SHARD_PROJECTIONS``. The serving package is not
#: importable from here (AGENTS.md principle 5), so the table is restated and
#: CHECKED: a member whose name is neither this spelling nor the role spelling
#: is refused, and the producer keeps the source spelling because its wire
#: record's own ``identity.unit`` is checked against it -- renaming a member on
#: this side would make that record unverifiable against the bytes it names.
ROLE_PROJECTIONS = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}


def owner_format(shape, format_name):
    """The one format a whole routed owner holds, validated as Tessera-shaped.

    A whole owner is ONE format for all of its members -- that is the serving
    constraint and the producer asserts it per member -- but WHICH format is a
    parameter, not a constant. It was `FORMAT` (E4M3 K1 R1024) everywhere, which
    silently capped the owner at one rung of one family and is exactly the kind
    of wiring that makes an A4/A8/A16 experiment unreachable while the geometry
    validator looks complete.
    """
    parsed = parse_tessera_format_name(format_name)
    if parsed is None:
        raise ValueError(f"native MoE owner format {format_name!r} is not a Tessera format")
    if format_name != format_name.strip():
        raise ValueError("native MoE owner format must be trimmed")
    return format_name


def owner_execution(shape, *, format_name):
    """The execution record, with the geometry's own tensor-parallel fact.

    `EXECUTION` is the LFM record and stays byte-identical for LFM. A GLM owner
    states its rank's own TP, because the priced member shapes are the rank-local
    ones: a TP2 owner's execution record says 2, so a receipt cannot claim a TP1
    execution over TP2 member widths.
    """
    if geometry_family(shape) != "glm53_next_routed_stack_v1":
        return dict(EXECUTION)
    return {**EXECUTION, "tensor_parallel": shape["tensor_parallel"],
            "tensor_parallel_cut_axis": shape["tensor_parallel_cut_axis"]}


def _shape_for_roster(shape):
    """The member roster's own view: expert count, this rank's width, one format.

    The view is what the roster reads, and it normalizes the two geometries'
    own spellings -- ``experts`` for LFM, ``n_routed_experts`` for GLM -- into
    the single ``experts`` the roster walks, so no caller has to know which
    geometry it holds to count members. The format comes from the caller's
    shape (a prepared owner's) and defaults to this module's constant only when
    none was declared. ``rank_local_intermediate`` is the width THIS rank's
    cut of a member carries; the member record's own geometry is the container,
    which is ``intermediate_size`` and stays untouched here.
    """
    glm = geometry_family(shape) == "glm53_next_routed_stack_v1"
    width = member_intermediate_width(shape)
    return {**shape, "experts": shape["n_routed_experts"] if glm else shape["experts"],
            "rank_local_intermediate": width,
            "format": shape.get("format", FORMAT)}


def _at_or_above_one(value, name):
    """A positive integer, where `True` and `1.0` are not integers.

    `type(value) is not int` is what makes that true: `isinstance(True, int)`
    and `1.0 == 1` both hold, so a geometry carrying `tensor_parallel=True` or
    `intermediate_size=2047.0` would otherwise reach the width arithmetic and
    the slice bounds as a bool or a float. A geometry is a declaration, and a
    declaration that is not an integer is not this geometry.
    """
    if type(value) is not int or value < 1:
        raise ValueError(f"{name}: a positive integer is required")
    return value


def validate_geometry(shape):
    """The complete explicit stack geometry, in the versioned shape.

    The LFM shape is unchanged, field for field, and keeps its refusal text:
    a stack that is not one complete 32-expert unit is still refused by name.
    GLM's geometry is a *different* shape, admitted only when it states its
    version and its own source facts, so widening this validator cannot make an
    LFM panel stop being an LFM panel.
    """
    if (isinstance(shape, dict)
            and set(shape) - {"tensor_parallel_rank"} == GLM_SHAPE_FIELDS
            and shape.get("geometry_id") is not None):
        return validate_glm_geometry(shape)
    if not isinstance(shape, dict) or set(shape) != LFM_SHAPE_FIELDS:
        raise ValueError("native MoE requires complete explicit stack geometry")
    if any(type(value) is not int or value < 1 for value in shape.values()):
        raise ValueError("native MoE geometry must be positive integers")
    if shape["experts"] != LFM_EXPERTS or not 1 <= shape["top_k"] <= shape["experts"]:
        raise ValueError("native MoE panel supports one complete 32-expert stack")
    return shape


def validate_glm_geometry(shape):
    """GLM-5.3-Flash's routed stack, with every coordinate stated rather than read.

    Every field is compared against the frozen source geometry above, so a
    panel cannot quietly price `top_k=1` or a different routed scale and still
    claim to be the GLM owner.  `tensor_parallel` is accepted at 1 or 2 and the
    cut axis is fixed: the rank-local intermediate is
    `intermediate_size // tensor_parallel`, which is what the serving route
    actually reads (`nvfp4_moe_route.py:374-381`). Execution rank is optional
    here, validated when supplied, and required when selecting a TP2 window.
    """
    if set(shape) - {"tensor_parallel_rank"} != GLM_SHAPE_FIELDS:
        raise ValueError("GLM routed owner requires exactly the versioned geometry fields")
    if (type(shape["geometry_version"]) is not int
            or type(GEOMETRY_VERSION) is not int
            or shape["geometry_version"] != GEOMETRY_VERSION):
        raise ValueError(
            f"GLM routed owner geometry version {shape['geometry_version']!r} is not "
            f"this consumer's {GEOMETRY_VERSION}")
    for key in ("tensor_parallel",):
        _at_or_above_one(shape[key], key)
    for key, expected in (GLM_SOURCE_GEOMETRY.items()):
        if shape[key] != expected:
            raise ValueError(
                f"GLM routed owner {key} is {shape[key]!r}, and this source is "
                f"{expected!r}; refusing a stack that is not the captured one")
    for key in ("hidden_size", "intermediate_size", "n_routed_experts", "top_k",
                "shared_experts", "n_group", "topk_group"):
        _at_or_above_one(shape[key], key)
    if shape["top_k"] > shape["n_routed_experts"]:
        raise ValueError("GLM routed owner top_k exceeds its expert count")
    if type(shape["norm_topk_prob"]) is not bool or type(shape["gated"]) is not bool:
        raise ValueError("GLM routed owner boolean coordinates must be booleans")
    if (not isinstance(shape["swiglu_limit"], (int, float))
            or not math.isfinite(shape["swiglu_limit"]) or shape["swiglu_limit"] <= 0):
        raise ValueError("GLM routed owner needs its finite positive SwiGLU clamp")
    if (not isinstance(shape["routed_scaling_factor"], (int, float))
            or not math.isfinite(shape["routed_scaling_factor"])
            or shape["routed_scaling_factor"] <= 0):
        raise ValueError("GLM routed owner needs its finite positive routed scale")
    if shape["tensor_parallel"] not in SUPPORTED_TP_SIZES:
        raise ValueError(
            f"GLM routed owner tensor_parallel {shape['tensor_parallel']!r} is outside "
            f"the supported cuts {SUPPORTED_TP_SIZES}")
    if "tensor_parallel_rank" in shape:
        _execution_rank(shape)
    if shape["tensor_parallel_cut_axis"] != GLM_TP_CUT_AXIS:
        raise ValueError(
            "GLM routed owner declares a tensor-parallel cut this consumer does not "
            f"implement: {shape['tensor_parallel_cut_axis']!r}, not {GLM_TP_CUT_AXIS!r}")
    if shape["intermediate_size"] % shape["tensor_parallel"]:
        raise ValueError("GLM routed owner intermediate is not divisible by its TP cut")
    return shape


def _execution_rank(shape):
    """Validate the execution coordinate without changing canonical geometry."""
    if "tensor_parallel_rank" not in shape:
        if shape["tensor_parallel"] == 1:
            return 0
        raise ValueError("GLM TP2 slicing requires an explicit tensor_parallel_rank")
    rank = shape["tensor_parallel_rank"]
    if type(rank) is not int or rank < 0:
        raise ValueError("GLM routed owner tensor_parallel_rank is not a rank index")
    if rank >= shape["tensor_parallel"]:
        raise ValueError(
            f"GLM routed owner tensor_parallel_rank {rank!r} is not inside a world of "
            f"{shape['tensor_parallel']!r}")
    return rank


def rank_local_intermediate(shape):
    """The intermediate width one rank of this owner materializes."""
    return shape["intermediate_size"] // shape["tensor_parallel"]


def member_intermediate_width(shape):
    """The intermediate width THIS rank's cut of the owner carries.

    One home for the cut arithmetic: the roster's own view, the render check
    and the gate/up pack all read it, so none of them can disagree about how
    wide this rank's members are. An LFM owner has no cut and its member width
    is the declared one.
    """
    if geometry_family(shape) == "glm53_next_routed_stack_v1":
        return rank_local_intermediate(shape)
    return shape["intermediate_size"]


def container_member_shape(shape, role):
    """The WHOLE container one expert role frames: the module's own geometry.

    A Tessera checkpoint holds one whole unit per role whatever world serves it
    -- the artifact is tensor-parallel agnostic -- so this is the geometry the
    wire's own identity names and the geometry every rank's cut is taken from.
    The rank-local reading is :func:`rank_local_member_shape`.
    """
    if role == "w2":
        return [shape["hidden_size"], shape["intermediate_size"]]
    return [shape["intermediate_size"], shape["hidden_size"]]


def rank_local_member_shape(shape, role):
    """This rank's own cut of one member role, in the container's coordinates."""
    width = member_intermediate_width(shape)
    if role == "w2":
        return [shape["hidden_size"], width]
    return [width, shape["hidden_size"]]


def member_window(shape, role):
    """``(rows, cols, axis)``: this rank's own window of the whole container.

    Rows for the column-parallel gate/up members, columns for the row-parallel
    down member -- ``tensor_parallel_cut_axis`` applied to one member, which is
    the range the loader makes at this world size. ``axis`` is the wire
    cutter's own spelling (``"row"``/``"column"``), and a world of one holds
    the container whole.
    """
    rows, cols = container_member_shape(shape, role)
    if geometry_family(shape) != "glm53_next_routed_stack_v1":
        return (0, rows), (0, cols), ("column" if role == "w2" else "row")
    world, rank = shape["tensor_parallel"], _execution_rank(shape)
    width = shape["intermediate_size"] // world
    lo, hi = rank * width, (rank + 1) * width
    if role == "w2":
        return (0, rows), (lo, hi), "column"
    return (lo, hi), (0, cols), "row"


def decoded_rank_member(blob, shape, role, *, device, where):
    """The original wire's own decode, cut to THIS rank's window.

    The artifact holds the WHOLE module, so what this rank must hold is the cut
    the loader itself makes. Above a world of one that cut goes through the
    wire format's own cutter -- ``tessera.layout.can_shard``/``slice_unit``,
    the producer-neutral primitive the loader calls -- never through an import
    of the serving route (AGENTS.md principle 5) and never by slicing encoded
    planes here. A row cut of a trellis unit carries the state its first
    surviving row starts from; the cutter computes that state, this module
    does not. Comparing the whole container against a rank-local render would
    compare two widths and pass only at a world of one.
    """
    import torch
    from tessera import unit_artifact
    if geometry_family(shape) != "glm53_next_routed_stack_v1":
        decoded = unit_artifact.read_unit_artifact(blob, device=str(device))
        return decoded.to(torch.bfloat16)
    rows, cols, axis = member_window(shape, role)
    world = shape["tensor_parallel"]
    if world == 1:
        decoded = unit_artifact.read_unit_artifact(blob, device=str(device))
        return decoded.to(torch.bfloat16)
    from tessera.layout import can_shard, slice_unit
    parsed = unit_artifact.parse_unit_artifact(blob, str(device))
    if can_shard(parsed, world, axis) is not True:
        raise ValueError(
            f"{where}: this rank's {axis} cut of a {world}-way world is not a cut this "
            "wire format admits")
    cut = slice_unit(parsed, rows=rows, cols=cols)
    # The forests are the parent's: ALPHABET and DESCENDANT are whole-unit and
    # travel across a cut untouched, and the cut keeps the parent's rates per
    # surviving column, so the shard decodes against the table it was written
    # from -- bit for bit the parent's own window.
    return unit_artifact.reconstruct_unit(cut, parsed.forests, parsed.code).to(torch.bfloat16)


#: Parsed qualified quality preparations by ``(binding sha256, PWC sha256)``.
#: ``prepare_moe_inputs`` and ``freeze_moe_panel`` each qualify the same
#: panel, so without this the whole production cache is read, hashed and
#: unpickled twice per panel (P3 #682). The bytes are still authenticated by
#: :func:`tessera_joint_allocation._read_bound` on the miss path; a hit
#: additionally requires the PWC file's stat fence to be unchanged, and a
#: drifted fence re-reads and re-verifies. The memoized objects are read,
#: never mutated, by the qualifier below.
_QUALIFIED_QUALITY_MEMO = {}


def _memoized_qualified_cache(completion, pickle, ProductionWeightCache):
    """Return the bound preparation's ``ProductionWeightCache``, unpickled once."""
    from pathlib import Path

    from .tessera_joint_allocation import _bound_stat_fence, _read_bound

    record = completion["production_cache"]
    try:
        fence = _bound_stat_fence(Path(record["path"]))
    except (KeyError, TypeError, OSError):
        fence = None
    key = (completion.get("plan_sha256"), record.get("sha256")
           if isinstance(record, dict) else None)
    if fence is not None:
        hit = _QUALIFIED_QUALITY_MEMO.get(key)
        if hit is not None and hit[0] == fence:
            return hit[1]
    cache = pickle.loads(_read_bound(record, "native qualified PWC"))
    if isinstance(cache, ProductionWeightCache) and fence is not None:
        _QUALIFIED_QUALITY_MEMO[key] = (fence, cache)
    return cache


def _qualified_quality_members(binding, *, members, source_model, calibration):
    """The historical full-container render proof, read from its own bound PWC.

    A rank holds a CUT of the module's render, and a quality identity may not
    be taken on a cut: what the campaign qualified is the WHOLE container, and
    that is what the joint quality row must name. This reads that proof out of
    the ORIGINAL preparation the caller binds -- the prepared completion and
    the ``ProductionWeightCache`` it names, each authenticated by its own
    digest through :func:`tessera_joint_allocation._read_bound` -- rather than
    from anything this producer wrote about itself. Nothing is rendered, no
    cache is created, and the encoder identity is joined through the producer's
    own canonical-JSON grammar rather than a second spelling of one hash.
    """
    import pickle
    from .cost_stage_checkpoint import canonical_json_sha256
    from .production_weight_cache import ProductionWeightCache
    from .tessera_joint_allocation import _read_bound
    from .tessera_joint_aura import (HISTORICAL_WIRE_VALIDATION, PREPARED_SCHEMA,
                                     RENDER_COMPARISON_BY_ORIGIN)

    completion_raw = _read_bound(binding, "native full-quality preparation")
    completion = json.loads(completion_raw)
    _equal(completion.get("schema"), PREPARED_SCHEMA, "quality prepared schema")
    _equal(completion.get("status"), "complete", "quality prepared completion")
    _equal(completion["source_model_identity"], source_model, "quality source model")
    for field in ("calibration_sha256", "shape", "dtype"):
        _equal(completion["calibration_input"][field], calibration[field],
               f"quality calibration {field}")
    cache = _memoized_qualified_cache(completion, pickle, ProductionWeightCache)
    if not isinstance(cache, ProductionWeightCache):
        raise ValueError("native quality preparation requires the actual ProductionWeightCache")
    metadata = cache.metadata or {}
    _equal(metadata.get("schema"), PREPARED_SCHEMA, "quality PWC schema")
    for field in ("plan_sha256", "source_model_identity", "source_execution",
                  "implementation_sha256", "reader_identity", "projection_backend"):
        _equal(metadata.get(field), completion[field], f"quality PWC {field}")
    verified = metadata.get("verified_cells")
    if not isinstance(verified, dict) or "inputs" not in metadata:
        raise ValueError("native quality preparation lacks its qualified render receipts")
    result = {}
    for member in members:
        name, fmt = member["unit"], member["format"]
        if fmt not in completion["formats_by_qname"].get(name, []):
            raise ValueError(f"{name}: absent from the qualified candidate roster")
        if (name, fmt) not in verified:
            raise ValueError(f"{name}: absent from the qualified render receipts")
        proof = verified[name, fmt]
        for field in ("source_weight", "activation"):
            _equal(proof[field], member[field], f"{name} qualified {field}")
        # The qualified render is the CONTAINER the member record names, which
        # is the one geometry a quality identity may be taken on.
        _equal(proof["rendered_weight"]["shape"], member["shape"], f"{name} full quality shape")
        _equal(proof["source_weight"]["shape"], member["shape"], f"{name} full source shape")
        _equal(proof["wire_sha256"], member["wire"]["blob_sha256"], f"{name} qualified wire")
        identity = member["wire"]["record"]["identity"]
        _equal(proof["encoding_identity_sha256"],
               canonical_json_sha256(identity, where=f"{name} original encoding"),
               f"{name} qualified encoder")
        _equal(identity["unit"], name, f"{name} encoder unit")
        if proof["render_origin"] not in RENDER_COMPARISON_BY_ORIGIN:
            raise ValueError(f"{name}: qualified render carries no closed-vocabulary origin")
        _equal(proof["render_comparison"], RENDER_COMPARISON_BY_ORIGIN[proof["render_origin"]],
               f"{name} qualified render comparison")
        _sha(proof["render_file_sha256"], f"{name} qualified render file")
        result[name] = dict(proof)
    return result, {"prepared": binding, "plan_sha256": completion["plan_sha256"],
                    "wire_validation": HISTORICAL_WIRE_VALIDATION,
                    "inputs": metadata["inputs"]}


def _rank_render_proof(member, shape, quality):
    """Bind THIS rank's measured render to the window its qualified wire declares.

    The proof is the cut's own record and never a quality identity: it names
    the window of the whole container this rank holds, the qualified full
    render the cut was taken from, and the encoder identity of the wire both
    sides share. :func:`member_window` is the one home for that arithmetic, so
    a window this record claims and a window the loader reads cannot differ.
    """
    rows, cols, axis = member_window(shape, member["role"])
    _equal(member["rendered_weight"]["shape"], rank_local_member_shape(shape, member["role"]),
           f"{member['unit']} observed rank render shape")
    return {"schema": RANK_RENDER_SCHEMA, "unit": member["unit"],
            "format": member["format"], "role": member["role"],
            "rank": _execution_rank(shape), "world_size": shape["tensor_parallel"],
            "rows": list(rows), "cols": list(cols), "axis": axis,
            "wire_sha256": member["wire"]["blob_sha256"],
            "encoding_identity_sha256": quality["encoding_identity_sha256"],
            "qualified_render_sha256": identity_sha256(quality["rendered_weight"]),
            "rendered_weight": member["rendered_weight"]}


def validate_routing(routing):
    if not isinstance(routing, dict) or set(routing) not in (ROUTING_FIELDS, GLM_ROUTING_FIELDS):
        raise ValueError("native MoE requires exact captured routing settings")
    if set(routing) == GLM_ROUTING_FIELDS:
        return validate_glm_routing(routing)
    if (routing["activation"] != "silu" or routing["scoring_func"] != "sigmoid"
            or type(routing["renormalize"]) is not bool
            or routing["apply_router_weight_on_input"] is not False
            or routing["expert_map"] is not None
            or routing["input_dtype"] != "torch.bfloat16"
            or routing["topk_weights_dtype"] not in ("torch.bfloat16", "torch.float32")
            or routing["topk_ids_dtype"] not in ("torch.int32", "torch.int64")
            or routing["device"] != "cuda:0"
            or routing["weights_contract"] != "post_renormalization_and_routed_scaling"
            or type(routing["routed_scaling_factor"]) not in (int, float)
            or routing["routed_scaling_factor"] != 1):
        raise ValueError("native MoE reference requires the captured sigmoid/SiLU output-weighted unit-scale route")
    source = routing["source_protocol"]
    if (not isinstance(source, dict) or set(source) != {"router_class", "router_source_sha256",
            "selection_bias", "normalization_epsilon", "expert_bias_affects"}
            or not isinstance(source["router_class"], str) or not source["router_class"]
            or source["normalization_epsilon"] != 1e-6
            or source["expert_bias_affects"] != "selection_only"):
        raise ValueError("native MoE requires the actual source router protocol")
    _sha(source["router_source_sha256"], "router source")
    if source["selection_bias"] is not None:
        _sha(source["selection_bias"]["content_sha256"], "source selection bias")
    # These are already the weights passed INTO the actual experts forward.
    # No normalization or scaling is applied to them a second time here.
    # BF16 source normalization does not imply an exact sum of one.
    return routing


def validate_glm_routing(routing):
    """GLM-5.3-Flash's captured routing, with its own facts checked.

    Three of these are the reason a widened LFM validator would have been
    wrong rather than merely permissive.  The source route is `noaux_tc`, not
    LFM's plain sigmoid-and-bias selection, and it carries a live FP32
    correction bias that changes *which* experts run.  `norm_topk_prob` is what
    makes the top-k weights a normalized distribution before the routed scale
    is applied, so a panel that dropped it would price a different mixture.
    The SwiGLU clamp at 10.0 and the routed scale of 2.5 are read by the served
    kernels (`nvfp4_moe_route.py:700,780` pass `swiglu_limit` down as
    `gemm1_clamp_limit`), so a reference without them is not this owner.

    The reference is the producer's; the shared contract here is only what the
    two sides must agree on for a receipt to mean anything.
    """
    if (routing["activation"] != "silu" or routing["scoring_func"] != "sigmoid"
            or type(routing["renormalize"]) is not bool
            or routing["apply_router_weight_on_input"] is not False
            or routing["expert_map"] is not None
            or routing["input_dtype"] != "torch.bfloat16"
            or routing["topk_weights_dtype"] not in ("torch.bfloat16", "torch.float32")
            or routing["topk_ids_dtype"] not in ("torch.int32", "torch.int64")
            or routing["device"] != "cuda:0"
            or routing["weights_contract"] != "post_renormalization_and_routed_scaling"):
        raise ValueError("GLM routed owner is outside its captured route protocol")
    if routing["topk_method"] != GLM_SOURCE_GEOMETRY["topk_method"]:
        raise ValueError(
            f"GLM routed owner top-k method is {routing['topk_method']!r}, and this "
            f"source is {GLM_SOURCE_GEOMETRY['topk_method']!r}")
    if routing["n_group"] != GLM_SOURCE_GEOMETRY["n_group"] or routing["topk_group"] != GLM_SOURCE_GEOMETRY["topk_group"]:
        raise ValueError("GLM routed owner group selection differs from the captured source")
    if (not isinstance(routing["swiglu_limit"], (int, float))
            or not math.isfinite(routing["swiglu_limit"])
            or routing["swiglu_limit"] != GLM_SOURCE_GEOMETRY["swiglu_limit"]):
        raise ValueError(
            f"GLM routed owner SwiGLU clamp is {routing['swiglu_limit']!r}, and this "
            f"source is {GLM_SOURCE_GEOMETRY['swiglu_limit']!r}")
    if (not isinstance(routing["routed_scaling_factor"], (int, float))
            or routing["routed_scaling_factor"] != GLM_SOURCE_GEOMETRY["routed_scaling_factor"]):
        raise ValueError(
            f"GLM routed owner routed scale is {routing['routed_scaling_factor']!r}, and "
            f"this source is {GLM_SOURCE_GEOMETRY['routed_scaling_factor']!r}")
    source = routing["source_protocol"]
    if (not isinstance(source, dict) or set(source) != GLM_SOURCE_PROTOCOL_FIELDS
            or not isinstance(source["router_class"], str) or not source["router_class"]
            or source["normalization_epsilon"] != 1e-20
            or source["expert_bias_affects"] != "selection_only"):
        raise ValueError("GLM routed owner requires the actual GLM source router protocol")
    if source["scoring_func"] != "sigmoid" or source["topk_method"] != GLM_SOURCE_GEOMETRY["topk_method"]:
        raise ValueError("GLM routed owner source protocol names a different selection rule")
    if source["norm_topk_prob"] is not True:
        raise ValueError(
            "GLM routed owner requires norm_topk_prob true: the served route applies "
            "normalized top-k weights before the routed scale, and a panel without it "
            "prices a different mixture")
    _sha(source["router_source_sha256"], "router source")
    # The correction bias is MANDATORY for this source, not optional. GLM-5.3
    # routes with `noaux_tc`: `e_score_correction_bias` takes part in which
    # experts are selected, so a panel without it does not price this model's
    # mixture even when every other coordinate matches. `None` here would read
    # as "this source has no bias", which is a different source.
    correction = source["correction_bias"]
    if not isinstance(correction, dict) or set(correction) != {"content_sha256", "dtype"}:
        raise ValueError(
            "GLM routed owner requires its live FP32 correction bias, with its "
            "content digest and dtype; None is a different source's protocol")
    _sha(correction["content_sha256"], "source correction bias")
    if correction["dtype"] != "torch.float32":
        raise ValueError("GLM routed owner correction bias is the source's FP32 bias, not a cast")
    # `renormalize` and the source protocol's `norm_topk_prob` are the same
    # fact stated twice, and both are read: the served route normalizes the
    # selected weights before applying the routed scale. An inconsistent
    # capture -- one of them true and the other false -- is refused rather than
    # resolved in whichever direction the reader happens to check first.
    if routing["renormalize"] is not True:
        raise ValueError(
            "GLM routed owner requires renormalize true: the served route applies "
            "normalized top-k weights before the routed scale")
    if routing["renormalize"] is not source["norm_topk_prob"]:
        raise ValueError(
            "GLM routed owner capture is internally inconsistent: renormalize "
            f"{routing['renormalize']!r} against source norm_topk_prob "
            f"{source['norm_topk_prob']!r}")
    return routing


def geometry_family(shape):
    """Which versioned geometry this shape is, by the fields it carries.

    Read through :func:`geometry_only`, because a prepared owner's shape also
    carries its format and its rank-local width and those are not geometry
    coordinates: a shape that has been through `_shape_for_roster` is still the
    same geometry.
    """
    if _declares_glm_fields(shape):
        return "glm53_next_routed_stack_v1"
    return "lfm2_moe_routed_stack_v1"


def _member_unit_names(unit, expert, role):
    """Every spelling of one member this consumer admits, the source one first.

    The source checkpoint spells the expert projections ``gate_proj``/
    ``up_proj``/``down_proj``; this consumer's role vocabulary spells the same
    three ``w1``/``w3``/``w2``. Both name one member, and the producer keeps the
    source spelling because its wire record's own ``identity.unit`` is checked
    against the member's unit -- so both are admitted here rather than one being
    renamed into the other.
    """
    canonical = f"{unit}.{expert}.{role}"
    projection = ROLE_PROJECTIONS.get(role)
    return (canonical,) if projection is None else (canonical, f"{unit}.{expert}.{projection}")


def _expect_member_roster(unit, members, shape, *, count, pattern, where):
    expected = [(expert, role, f"{unit}.{expert}.{role}")
                for expert in range(count) for role in ROLES]
    if not isinstance(members, list) or len(members) != len(expected):
        raise ValueError(
            f"native MoE panel requires all {len(expected)} explicitly ordered source members")
    for member, (expert, role, name) in zip(members, expected):
        if (type(member["expert"]) is not int or member["expert"] != expert
                or member["role"] != role
                or member["unit"] not in _member_unit_names(unit, expert, role)
                or member["format"] != shape["format"]):
            raise ValueError("native MoE expert-role member ordering/format differs")
        # The member's own shape is the CONTAINER: a Tessera checkpoint holds
        # one whole unit per role whatever world serves it, so the module's
        # geometry is what the wire identity names. This rank's own cut is the
        # render, and a roster that claimed it here would describe the shard as
        # if it were the module.
        _equal(member["shape"], container_member_shape(shape, role), f"{name} shape")
    return members


#: The keys a prepared owner adds to its geometry, beyond the contract fields.
#: `experts` is an ALIAS of the geometry's own expert count -- LFM spells it
#: `experts`, GLM spells it `n_routed_experts`, and the roster walks one key.
#: It is listed here because it is derived, not declared: treating it as a
#: contract field made the derived view misclassify as the other geometry
#: (root review of 266f80e52f/9e1f2634).
_OWNER_VIEW_KEYS = ("format", "rank_local_intermediate", "experts")


def geometry_only(shape):
    """Declared geometry and optional execution rank, without derived fields.

    A shape that has been through :func:`_shape_for_roster` is still the same
    geometry, which is the property this function exists to preserve: dropping
    the derived keys is what makes the view round-trip. `experts` is dropped
    only when the geometry declares its count under the other spelling, so an
    LFM shape keeps the `experts` it actually declared.
    """
    if not isinstance(shape, dict):
        return {}
    glm = _declares_glm_fields(shape)
    dropped = {"format", "rank_local_intermediate"}
    if glm:
        dropped.add("experts")
    return {key: value for key, value in shape.items() if key not in dropped}


def _declares_glm_fields(shape):
    """True when the declared fields are GLM's set, ignoring derived keys.

    Read without calling :func:`geometry_family`, which needs this answer; the
    two are deliberately distinct so the predicate has no cycle.
    """
    if not isinstance(shape, dict):
        return False
    declared = {key for key in shape
                if key not in ("format", "rank_local_intermediate", "experts",
                               "tensor_parallel_rank")}
    return declared == set(GLM_SHAPE_FIELDS)


def _member_roster(unit, members, shape):
    """This owner's ordered members, in the geometry's own naming and shape.

    Both geometries carry the WHOLE container each member frames: an LFM unit's
    cut is trivial so its container is its declared width, and a GLM unit's
    container is the module's, which is what the wire identity names. The rank
    local cut is the render and the runtime binding, not this record. The two
    rosters share this function, not a shape constant.
    """
    validate_geometry(geometry_only(shape))
    if not isinstance(unit, str):
        raise ValueError("native MoE panel needs an exact routed stack name")
    if geometry_family(shape) == "glm53_next_routed_stack_v1":
        if _GLM_UNIT.fullmatch(unit) is None:
            raise ValueError(
                "GLM routed owner needs its source stack name, "
                "model.language_model.layers.<N>.mlp.experts")
        count = shape["n_routed_experts"]
    else:
        if _LFM_UNIT.fullmatch(unit) is None:
            raise ValueError("native MoE panel needs an exact LFM routed stack name")
        count = shape["experts"]
    return _expect_member_roster(unit, members, _shape_for_roster(shape), count=count,
                                 pattern=None, where="native MoE members")


def _validate_prefix_capture(capture, source_model_identity=None):
    """A fresh source prefix is explicit evidence, never a full-load stamp."""
    from .streaming_model import validate_streaming_prefix_initialization_contract
    contract=validate_streaming_prefix_initialization_contract(capture['model_load_contract'])
    if (capture['unit']!='model.language_model.layers.3.mlp.experts'
            or geometry_family(capture['shape'])!='glm53_next_routed_stack_v1'
            or contract['total_model_layers']!=45 or contract['observed_layers']!=[0,1,2,3]
            or contract['dtype']!='torch.bfloat16'):
        raise ValueError('fresh routing prefix scope differs from GLM source layers0..3')
    if (contract['layers_prefix']!='model.language_model.layers.' or
            not contract['model_class'].endswith('.Glm5NextForConditionalGeneration') or
            not {'lm_head.weight','model.language_model.embed_tokens.weight',
                 'model.language_model.norm.weight'} <= set(contract['head_state_names'])):
        raise ValueError('fresh prefix lacks the actual GLM head/model initialization coverage')
    replay=capture.get('replay',{})
    if (replay.get('schema')!='prismaquant.glm_routing_boundary_replay.v1'
            or replay.get('layer')!=3 or replay.get('sample')!=0
            or replay.get('stop')!='before_original_packed_experts_forward'):
        raise ValueError('fresh routing prefix lacks its exact original replay coordinates')
    provenance=capture.get('source_acquisition',capture)
    reuse=provenance.get('source_cache_reuse',{})
    if (provenance.get('dev_uncertified') is not True
            or provenance.get('dev_mode',{}).get('PRISMAQUANT_DEV_MODE')!='1'
            or reuse.get('complete_checkpoint') is not True
            or reuse.get('validator')!='validate_cached_streamed_model_identity'):
        raise ValueError('fresh prefix requires explicit complete-cache DEV source authority')
    _sha(reuse.get('content_sha256'),'prefix cached complete source')
    binding=reuse.get('binding',{})
    _sha(binding.get('sha256'),'prefix cache binding')
    if not isinstance(binding.get('path'),str) or not binding['path']:
        raise ValueError('prefix source cache binding lacks its path')
    if source_model_identity is not None:
        _equal(reuse['content_sha256'],source_model_identity['content_sha256'],'prefix complete source identity')
        _equal(capture['producer_source']['tensors'],source_model_identity['checkpoint_weight_map'],'prefix checkpoint source map')
        mapping={name:{'tensor':key,'file':Path(source_model_identity['checkpoint_weight_map'][key]).name}
                 for name,key in source_model_identity['weight_map'].items()}
        _equal(contract['source_map_sha256'],identity_sha256(mapping),'prefix live source map')
    return contract


def _calibration_and_capture(calibration, capture, *, unit, shape, routing):
    if calibration.get("schema") != "prismaquant.calibration_input.v1":
        raise ValueError("native MoE needs the exact calibration receipt")
    _sha(calibration["calibration_sha256"], "calibration")
    if capture.get("schema") != "prismaquant.routed_boundary_capture.v1":
        raise ValueError("native MoE needs an actual routed-boundary capture")
    _equal(geometry_only(capture["shape"]), geometry_only(shape), "capture source geometry")
    for key, expected in (("unit", unit), ("routing", routing),
                          ("calibration_sha256", calibration["calibration_sha256"]),
                          ("calibration_shape", calibration["shape"]),
                          ("calibration_dtype", calibration["dtype"])):
        _equal(capture[key], expected, f"capture {key}")
    from .tessera_expert_projection import _require_source_identity
    source = _require_source_identity(capture["producer_source"])
    _sha(source["config_sha256"], "capture source config")
    for digest in (*source["files"].values(), *source["auxiliary_sha256"].values()):
        _sha(digest, "capture source file")
    if not source["files"] or not source["tensors"] or not isinstance(capture["runtime_config"], dict):
        raise ValueError("native MoE capture needs its full producer source and actual runtime config")
    _sha(capture["capture_source_sha256"], "capture producer")
    from . import validate_pretrained_initialization_contract
    if isinstance(capture['model_load_contract'],dict) and capture['model_load_contract'].get('schema')=='prismaquant.streaming_prefix_initialization.v1':
        _validate_prefix_capture(capture)
    else:
        validate_pretrained_initialization_contract(capture["model_load_contract"])
    if capture["attention_implementation"] != "eager":
        raise ValueError("native MoE capture must record the actual eager source backend")
    runtime = capture["capture_runtime"]
    if (not isinstance(runtime, dict) or set(runtime) != {"torch", "cuda", "transformers"}
            or any(not isinstance(value, str) or not value for value in runtime.values())
            or runtime["transformers"] != capture["model_load_contract"]["transformers_version"]):
        raise ValueError("native MoE capture runtime does not match its canonical model load")


def _no_preclip():
    from .memory_management import env_truthy
    if env_truthy("PRISMAQUANT_PROD_ACT_SCALES", default=True):
        raise ValueError("native MoE protocol requires explicit PRISMAQUANT_PROD_ACT_SCALES=0")


def _validate_phase_tensors(x, ids, weights, shape, *, cuda):
    import torch
    if not all(isinstance(value, torch.Tensor) for value in (x, ids, weights)):
        raise ValueError("native MoE requires actual input, top-k IDs and weights tensors")
    if (x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != shape["hidden_size"]
            or x.dtype != torch.bfloat16):
        raise ValueError("native MoE input must be a nonempty 2-D BF16 hidden-state batch")
    if (ids.shape != (x.shape[0], shape["top_k"]) or weights.shape != ids.shape
            or ids.dtype not in (torch.int32, torch.int64)
            or weights.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("native MoE top-k requires captured integer IDs and BF16/FP32 weights at matching geometry")
    if len({value.device for value in (x, ids, weights)}) != 1 or (cuda and x.device.type != "cuda"):
        raise ValueError("native MoE routed invocation must be resident on one CUDA device")
    if (not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any()) or bool((ids < 0).any()) or bool((ids >= _shape_for_roster(shape)["experts"]).any())):
        raise ValueError("native MoE routed invocation has nonfinite values or invalid assignments")
    if shape["top_k"] > 1 and bool((ids.sort(dim=-1).values.diff(dim=-1) == 0).any()):
        raise ValueError("native MoE capture repeats an expert within a token's top-k")


def packed_reference(experts_module, x, ids, weights, gate_up_weight, down_weight, *, spec, cache,
                     input_unit=None, intermediate_unit=None):
    """Use the shared packed-expert operation at both existing QDQ boundaries.

    Packing below is transient preparation, not a second residency/cache path.
    The source module supplies its actual activation operation. Real preparation
    requires CUDA; this arithmetic helper can also exercise small CPU fixtures.
    """
    from .measure_quant_cost import _packed_experts_forward_with_weights
    from .perturbed_x_cache import _activation_qdq
    _no_preclip()
    return _packed_experts_forward_with_weights(
        experts_module, x, ids, weights, gate_up_weight, down_weight,
        input_quantize=lambda value: _activation_qdq(value, spec, cache.activation_max_abs or {}, input_unit),
        intermediate_quantize=lambda value: _activation_qdq(value, spec, cache.activation_max_abs or {}, intermediate_unit))


def prepare_moe_inputs(cache, source_weights, phase_tensors, *, unit, members, shape, routing,
                       calibration_receipt, routing_capture, experts_module, profile,
                       wire_blobs, wire_records, encoding_identities, numerics,
                       max_resident_bytes, max_temporary_bytes, runtime_image, serving_config_sha256, probe_request,
                       format_name=FORMAT, probe_calibration_receipt=None, probe_scope=None,
                       quality_prepared=None, quality_source_model=None, retain_source_tensors=True):
    """Prepare one complete routed reference from existing resident PWC data.

    Member records explicitly declare the expert/role order. Source tensors and
    producer encoding identities are independent inputs; they must not be copied
    from an unverified wire record. The caller pins their source artifact bytes.
    """
    import torch
    from tessera.cached_unit import verify_cached_unit, tensor_identity as producer_tensor_identity
    from . import format_registry as fr
    from .joint_aura import activation_identity, prefetch_joint_cache
    from .production_weight_cache import ProductionWeightCache, _cb_cache_tensor_identity
    _no_preclip()
    format_name = owner_format(shape, format_name)
    shape = {**shape, "format": format_name}
    execution = owner_execution(shape, format_name=format_name)
    rank_local = shape.get("tensor_parallel", 1) > 1
    if rank_local and (quality_prepared is None or quality_source_model is None):
        raise ValueError("native rank-local preparation requires independently bound "
                         "full-quality preparation")
    _member_roster(unit, members, shape)
    validate_routing(routing)
    _calibration_and_capture(calibration_receipt, routing_capture, unit=unit, shape=shape, routing=routing)
    _probe_calibration({"calibration": calibration_receipt, "probe_calibration": probe_calibration_receipt,
                        "probe_scope": probe_scope})
    if set(numerics) != {"atol", "rtol"}:
        raise ValueError("native MoE needs predeclared numerical tolerances")
    for name, value in numerics.items():
        _number(value, name)
    if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", runtime_image) is None:
        raise ValueError("native MoE requires an immutable image RepoDigest")
    _sha(serving_config_sha256, "serving configuration")
    if not isinstance(cache, ProductionWeightCache):
        raise TypeError("native MoE requires the actual ProductionWeightCache")
    expected_profile = ("lfm2_moe" if geometry_family(shape) == "lfm2_moe_routed_stack_v1"
                        else "glm5_next")
    expected_experts = shape["experts"] if "experts" in shape else shape["n_routed_experts"]
    if (profile.name != expected_profile
            or type(experts_module).__name__ not in profile.packed_expert_module_class_names()
            or getattr(experts_module, "num_experts", None) != expected_experts):
        raise ValueError(
            f"native MoE reference requires the source profile's actual packed "
            f"{expected_profile} experts module")
    names = [member["unit"] for member in members]
    for values in (source_weights, wire_blobs, wire_records, encoding_identities):
        if set(values) != set(names):
            raise ValueError("native MoE source/wire/encoding coverage must equal all declared members")
    if set(phase_tensors) != set(PHASES) or set(routing_capture["phases"]) != set(PHASES):
        raise ValueError("native MoE requires captured prefill and decode invocations")
    for phase in PHASES:
        observed = phase_tensors[phase]
        if set(observed) != {"input", "topk_ids", "topk_weights", "source_topk_ids", "source_topk_weights"}:
            raise ValueError("native MoE phase tensor set differs")
        _validate_phase_tensors(observed["input"], observed["topk_ids"], observed["topk_weights"], shape, cuda=True)
        for key in ("input", "topk_ids", "topk_weights"):
            tensor = observed[key]
            _equal(_cb_cache_tensor_identity(tensor), routing_capture["phases"][phase][key], f"captured {phase} {key}")
            _equal(str(tensor.dtype), routing[{"input": "input_dtype", "topk_ids": "topk_ids_dtype",
                                              "topk_weights": "topk_weights_dtype"}[key]], f"{phase} supplied dtype")
            _equal(str(tensor.device), routing["device"], f"{phase} supplied device")
        validate_transport(observed, routing_capture["phases"][phase]["transport"])
    device = phase_tensors["prefill"]["input"].device
    if phase_tensors["decode"]["input"].device != device:
        raise ValueError("native MoE phases must share the resident reference device")
    packed_bytes = sum(math.prod(container_member_shape(shape, member['role'])) * 2 for member in members)
    # One gate/up+down pack plus the largest transient decoded member. Input,
    # output and GEMM activations are covered separately by PB's action budget.
    required = packed_bytes + max(math.prod(container_member_shape(shape, member['role'])) * 2 for member in members)
    if required > _bytes(max_temporary_bytes, "temporary packing budget"):
        raise ValueError("native MoE reference pack exceeds its explicit temporary budget")
    prefetch = prefetch_joint_cache(cache, names, {name: [format_name] for name in names},
                                   max_resident_bytes=max_resident_bytes)
    spec = fr.get_format(format_name)
    tensors, actual_members, rendered = {}, [], {}
    for member in members:
        name = member["unit"]
        role = member["role"]
        # PWC prefetch materializes disk shards on CPU. Use the same explicit
        # device transfer as the dense native reference; preserve stored dtype.
        source, full_render = source_weights[name], cache.get(name, format_name).to(device=device)
        if list(full_render.shape) != container_member_shape(shape, role):
            raise ValueError(f"{name}: PWC must retain the full quality render")
        if rank_local:
            rows, cols, _axis = member_window(shape, role)
            render = full_render[rows[0]:rows[1], cols[0]:cols[1]].contiguous()
        else:
            render = full_render
        # The source is the WHOLE container the wire's own identity names; the
        # render is THIS rank's cut of it. The two are the same tensor only at a
        # world of one, so comparing them as one shape would refuse every real
        # TP2 panel and accept only the TP1 spelling that hides the cut.
        if ((source.device.type not in ("cpu", "cuda") or
             source.device.type == "cuda" and source.device != device)
                or render.device != device or source.dtype != torch.bfloat16
                or render.dtype != torch.bfloat16
                or list(source.shape) != container_member_shape(shape, role)
                or list(render.shape) != rank_local_member_shape(shape, role)
                or not bool(torch.isfinite(source).all())
                or not bool(torch.isfinite(render).all())):
            raise ValueError(
                f"native MoE source/PWC tensor is not the declared resident BF16 member: {name} "
                f"(source {list(source.shape)} of container {container_member_shape(shape, role)}, "
                f"render {list(render.shape)} of this rank's "
                f"{rank_local_member_shape(shape, role)})")
        _equal(encoding_identities[name]["source"], producer_tensor_identity(source), f"{name} encoder source")
        _equal(encoding_identities[name]["unit"], name, f"{name} encoder unit")
        verify_cached_unit(wire_blobs[name], wire_records[name], encoding_identities[name])
        decoded = decoded_rank_member(wire_blobs[name], shape, role, device=device,
                                      where=f"{name} wire/PWC")
        _equal(_cb_cache_tensor_identity(decoded), _cb_cache_tensor_identity(render), f"{name} wire/PWC")
        del decoded
        activation = activation_identity(spec, cache.activation_max_abs or {}, name)
        if activation["clip_enabled"]:
            raise ValueError("native MoE reference refuses activation preclip")
        if spec.static_activation_contract is not None and activation["input_global_scale"] is None:
            raise ValueError("native A4 reference requires executed static scales")
        actual_members.append({**member, "source_weight": _cb_cache_tensor_identity(source),
            "rendered_weight": _cb_cache_tensor_identity(render), "activation": activation,
            "wire": {"blob_sha256": hashlib.sha256(wire_blobs[name]).hexdigest(),
                     "blob_bytes": len(wire_blobs[name]), "record": wire_records[name]}})
        if rank_local:
            actual_members[-1]["quality_rendered_weight"] = _cb_cache_tensor_identity(full_render)
        if retain_source_tensors:
            tensors["source_weight/" + name] = source
        tensors["rendered_weight/" + name] = render
        rendered[name] = render
    if spec.static_activation_contract is not None:
        for stage_roles in (("w1", "w3"), ("w2",)):
            scales = {m["activation"]["input_global_scale"] for m in actual_members if m["role"] in stage_roles}
            if len(scales) != 1:
                raise ValueError("native A4 references require one executed group scale per stage")
    quality_context = None
    if rank_local:
        # The historical render was qualified under the campaign's own full
        # calibration draw, not under the bounded first-sequence probe screen,
        # so the parent receipt is what the qualification is bound to.
        quality_members, quality_context = _qualified_quality_members(
            quality_prepared, members=actual_members, source_model=quality_source_model,
            calibration=calibration_receipt)
        for member in actual_members:
            quality = quality_members[member["unit"]]
            _equal(member["quality_rendered_weight"], quality["rendered_weight"],
                   f"{member['unit']} actual full PWC render")
            member["rank_render_proof"] = _rank_render_proof(member, shape, quality)
    # Reuse the format/profile's declared gate/up roles. This pack is discarded
    # before the producer's native preparation and never persisted as a cache.
    roster = _shape_for_roster(shape)
    # THIS rank's own width and THIS roster's own member names: the pack holds
    # the rank-local experts the render loop just checked, and a member spelled
    # in the source's projection vocabulary must be found by the name it
    # declares rather than by a name this module would have invented.
    width = member_intermediate_width(shape)
    by_role = {(member["expert"], member["role"]): member["unit"] for member in members}
    with torch.inference_mode():
        gate_up = torch.empty(roster["experts"], 2 * width, shape["hidden_size"],
                              dtype=torch.bfloat16, device=device)
        for expert in range(roster["experts"]):
            for role_index, role in enumerate(ROLES[:2]):
                gate_up[expert, role_index * width:(role_index + 1) * width].copy_(
                    rendered[by_role[(expert, role)]])
        down = torch.stack([rendered[by_role[(expert, "w2")]] for expert in range(roster["experts"])])
        phases = {}
        from .perturbed_x_cache import _activation_qdq
        for phase in PHASES:
            values = {key: phase_tensors[phase][key] for key in ("input", "topk_ids", "topk_weights")}
            output = packed_reference(experts_module, values["input"], values["topk_ids"], values["topk_weights"],
                                      gate_up, down, spec=spec, cache=cache,
                                      input_unit=by_role[(0, "w1")], intermediate_unit=by_role[(0, "w2")])
            qdq = _activation_qdq(values["input"], spec, cache.activation_max_abs or {}, by_role[(0, "w1")])
            for key, value in {**values, "reference_qdq": qdq, "reference_output": output}.items():
                tensors[f"{phase}.{key}"] = value
            phases[phase] = {"m": values["input"].shape[0], "transport": routing_capture["phases"][phase]["transport"], **{key: _cb_cache_tensor_identity(tensors[f"{phase}.{key}"])
                for key in ("input", "topk_ids", "topk_weights", "reference_qdq", "reference_output")}}
    del gate_up, down
    reference_file = Path(inspect.getfile(type(experts_module)))
    return {"schema": INPUT_SCHEMA, "unit": unit, "format": format_name, "shape": dict(shape),
            **({"source_acquisition":routing_capture["source_acquisition"]} if "source_acquisition" in routing_capture else {}),
            **({"quality_preparation": quality_context} if quality_context is not None else {}),
            "members": actual_members, "profile_role_order": list(ROLES), "routing": dict(routing), "execution": execution,
            "calibration": calibration_receipt, "probe_calibration": probe_calibration_receipt,
            "probe_scope": probe_scope, "routing_capture": routing_capture,
            "routing_capture_sha256": identity_sha256(routing_capture),
            "numerics": dict(numerics), "runtime_image": runtime_image, "serving_config_sha256": serving_config_sha256, "probe_request": probe_request,
            "reference": {"operation": "prismaquant.measure_quant_cost._packed_experts_forward_with_weights",
                          "module_class": f"{type(experts_module).__module__}.{type(experts_module).__qualname__}",
                          "module_source_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
                          "profile": profile.name, "temporary_pack_bytes": packed_bytes,
                          "activation_preclip": False, "format": format_name},
            "prefetch": prefetch, "phases": phases}, tensors


def validate_transport(values, transport):
    """Check actual raw-to-supplied tensors before their original store is left.

    Source IDs and BF16 router weights are retained by the shared capture. A
    runtime-friendly integer cast or FP32 promotion must preserve every value;
    normalization, reordering and rounded narrowing are not transport.
    """
    import torch
    from .production_weight_cache import _cb_cache_tensor_identity
    if not isinstance(transport, dict) or set(transport) != {"topk_ids", "topk_weights"}:
        raise ValueError("native MoE needs both explicit routed transport records")
    for name in ("topk_ids", "topk_weights"):
        raw, supplied, record = values["source_" + name], values[name], transport[name]
        if (not isinstance(raw, torch.Tensor) or not isinstance(record, dict)
                or set(record) != {"source", "supplied", "operation"}):
            raise ValueError("native MoE routed transport lacks original tensors or exact records")
        _equal(_cb_cache_tensor_identity(raw), record["source"], f"raw {name}")
        _equal(_cb_cache_tensor_identity(supplied), record["supplied"], f"supplied {name}")
        expected_operation = "identity" if raw.dtype == supplied.dtype else "lossless_dtype_conversion"
        _equal(record["operation"], expected_operation, f"{name} transport operation")
        if raw.shape != supplied.shape or raw.device != supplied.device:
            raise ValueError("native MoE routed transport changes shape/device")
        if name == "topk_ids":
            if raw.dtype not in (torch.int32, torch.int64) or supplied.dtype not in (torch.int32, torch.int64):
                raise ValueError("native MoE routed IDs must stay integers")
        elif raw.dtype not in (torch.bfloat16, torch.float32) or supplied.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("native MoE routing weights must preserve BF16/FP32 values")
        if (not torch.equal(raw.to(supplied.dtype), supplied)
                or not torch.equal(supplied.to(raw.dtype), raw)):
            raise ValueError("native MoE routed transport changes actual IDs or weights")


def _transport_identity(phase):
    transport = phase["transport"]
    if not isinstance(transport, dict) or set(transport) != {"topk_ids", "topk_weights"}:
        raise ValueError("native MoE phase requires captured routing transport provenance")
    for name, record in transport.items():
        if not isinstance(record, dict) or set(record) != {"source", "supplied", "operation"}:
            raise ValueError("native MoE routed transport record is incomplete")
        _equal(record["supplied"], phase[name], f"{name} supplied capture identity")
        source, supplied = record["source"], record["supplied"]
        _sha(source["content_sha256"], f"{name} source tensor")
        _equal(source["shape"], supplied["shape"], f"{name} transport shape")
        expected = "identity" if source["dtype"] == supplied["dtype"] else "lossless_dtype_conversion"
        _equal(record["operation"], expected, f"{name} transport operation")
        if expected == "identity":
            _equal(source, supplied, f"{name} unchanged transport")


def _workspace_identity(workspace):
    if (workspace.get("schema") != "tessera.native_moe_workspace.v1"
            or workspace.get("owner") != "vllm.WorkspaceManager"
            or type(workspace.get("num_ubatches")) is not int or workspace["num_ubatches"] != 1
            or type(workspace.get("num_lanes")) is not int or workspace["num_lanes"] != 1
            or workspace.get("locked") is not True or not isinstance(workspace.get("slots"), list)):
        raise ValueError("native MoE workspace is not a frozen single-lane runtime allocation")
    _bytes(workspace["resident_bytes"], "workspace resident")
    seen = set()
    for slot in workspace["slots"]:
        index = slot["index"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("native MoE workspace slot identity repeats")
        seen.add(index)
        if "allocation" in slot:
            if set(slot) != {"index", "allocation"} or slot["allocation"] is not None:
                raise ValueError("native MoE empty workspace slot is not an unallocated slot")
            continue
        for name in ("storage_bytes", "logical_bytes", "storage_offset"):
            _bytes(slot[name], f"workspace {name}")
        if (slot["device"] != "cuda:0" or not isinstance(slot["dtype"], str)
                or not isinstance(slot["shape"], list) or not isinstance(slot["stride"], list)
                or len(slot["shape"]) != len(slot["stride"])
                or any(type(value) is not int or value < 0 for value in slot["shape"] + slot["stride"])):
            raise ValueError("native MoE workspace geometry differs from the supported device")
    return workspace


def _native_member_identity(member):
    return {**({"input_global_scale": member["activation"]["input_global_scale"]}
               if member["activation"].get("input_global_scale") is not None else {}), **{key: member[key] for key in ("unit", "expert", "role", "format", "shape", "source_weight", "rendered_weight")},
            "wire_sha256": member["wire"]["blob_sha256"],
            "wire_record_sha256": identity_sha256(member["wire"]["record"])}


def _source_execution(value, *, unit):
    if (not isinstance(value, dict) or set(value) != {"schema", "modules"}
            or value["schema"] != "prismaquant.joint_aura.source_execution.v1"
            or not isinstance(value["modules"], dict) or not value["modules"]):
        raise ValueError("native MoE requires explicit source execution identity")
    for name, selectors in value["modules"].items():
        if (not isinstance(name, str) or not isinstance(selectors, dict) or not selectors
                or not set(selectors) <= {"attention", "experts"}):
            raise ValueError("native MoE source execution selectors are malformed")
    for name in ("", unit):
        selectors = value["modules"].get(name, {})
        if selectors.get("attention") != "eager" or not isinstance(selectors.get("experts"), str) or not selectors["experts"]:
            raise ValueError("native MoE source execution lacks resolved root/target backends")
    json.dumps(value, allow_nan=False)
    return value


def _qualified_source_execution(inputs, probe, path, expected_sha256):
    """Verify a fresh source replay of a retained boundary, never rewrite it."""
    import struct
    raw = Path(path).read_bytes()
    _equal(hashlib.sha256(raw).hexdigest(), _sha(expected_sha256, "source qualification"), "source qualification file")
    result = json.loads(raw)
    if (result.get("schema") != "prismaquant.packed_joint_screen.v1" or result.get("mode") != "source"
            or result.get("passed") is not True):
        raise ValueError("source execution qualification requires a successful independent source replay")
    proof = result["retained_boundary_qualification"]
    if proof.get("schema") != "prismaquant.packed_source_boundary_qualification.v1":
        raise ValueError("source execution qualification schema unsupported")
    _equal(proof["unit_qname"], inputs["unit"], "qualified source unit")
    _equal(proof["artifact_sha256"], inputs["source_capture"]["routing_boundary_sha256"], "qualified original boundary file")
    metadata, capture = proof["boundary_metadata"], inputs["routing_capture"]
    for key in ("unit", "shape", "profile_role_order"):
        _equal(metadata[key], inputs[key], f"qualified {key}")
    for key in ("calibration_sha256", "calibration_shape", "calibration_dtype", "producer_source",
                "runtime_config", "capture_source_sha256", "model_load_contract", "attention_implementation", "capture_runtime"):
        _equal(metadata[key], capture[key], f"qualified original {key}")
    for key in ("torch", "cuda", "transformers"):
        _equal(proof["runtime"][key], capture["capture_runtime"][key], f"qualified actual runtime {key}")
    _equal(proof["source_model_identity"], probe["source_model"], "qualified source model")
    subset, parent, calibrated = proof["calibration_subset"], inputs["calibration"], _probe_calibration(inputs)
    for key, expected in (("artifact_sha256", parent["artifact_sha256"]), ("full_shape", parent["shape"]),
            ("row", 0), ("shape", [1, parent["shape"][1]]), ("dtype", parent["dtype"]),
            ("subset_artifact_sha256", calibrated["artifact_sha256"]), ("sha256", calibrated["calibration_sha256"])):
        _equal(subset[key], expected, f"qualified calibration {key}")
    prefill = inputs["phases"]["prefill"]
    count = prefill["m"]
    expected_tensors = {"inputs": prefill["input"],
        "top_k_index": prefill["transport"]["topk_ids"]["source"],
        "top_k_weights": prefill["transport"]["topk_weights"]["source"],
        "expert_bias": inputs["routing"]["source_protocol"]["selection_bias"],
        "coordinates": {"shape": [count, 2], "dtype": "torch.int64",
            "content_sha256": hashlib.sha256(b"".join(struct.pack("<qq", 0, row) for row in range(count))).hexdigest()}}
    comparisons = proof["tensor_comparisons"]
    _equal(sorted(comparisons), sorted(expected_tensors), "qualified boundary tensor roster")
    for name, expected in expected_tensors.items():
        compared = comparisons[name]
        if compared["equal"] is not True:
            raise ValueError(f"qualified boundary {name} is not bit-exact")
        for key in ("shape", "dtype"):
            _equal(compared[key], expected[key], f"qualified boundary {name} {key}")
            _equal(metadata["tensors"][name][key], expected[key], f"qualified original {name} {key}")
        for key in ("actual_sha256", "captured_sha256"):
            _equal(compared[key], expected["content_sha256"], f"qualified boundary {name} {key}")
        _equal(metadata["tensors"][name]["content_sha256"], expected["content_sha256"], f"qualified original {name} bytes")
    execution = _source_execution(proof["source_execution_identity"], unit=inputs["unit"])
    _equal(execution, proof["streamed_source_execution_identity"], "qualified reference/streamed source execution")
    return execution


def validate_compact_native_route(inputs, route):
    """Check preflight route facts without importing the serving runtime.

    Tessera checks the actual launch against its own versioned table. This
    consumer checks that the frozen declaration has the format's same owner
    and activation contract, and explicitly refuses stock materialisation.
    """
    fmt = inputs["format"]
    families = (("TESSERA_E2M1_", "TESSERA_NVFP4", "e2m1_group16_ue4m3_static"),
                ("TESSERA_E4M3_", "TESSERA_FP8", "fp8_per_token_dynamic"),
                ("TESSERA_BF16_", "TESSERA_BF16", "bf16_unquantized"))
    matches = [(family, contract) for prefix, family, contract in families if fmt.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("whole native owner format has no declared family")
    family, contract = matches[0]
    if (route.get("kind") != "moe" or route.get("policy") != f"{family}:resident"
            or route.get("contract") != contract):
        raise ValueError("native MoE route family/residency/activation differs")
    if (not isinstance(route.get("decoder"), str) or not route["decoder"].startswith("native_")
            or not isinstance(route.get("symbol"), str) or not route["symbol"]):
        raise ValueError("packed native MoE receipt requires an explicit native launch")
    return family


def freeze_moe_panel(inputs, preflight, cost_rows, *, cost_sha256,
                     source_execution_qualification_path=None, source_execution_qualification_sha256=None):
    """Bind all aligned member rows to one actual whole-stack native operator.

    RuntimeBinding is the existing cost/runtime join. No local or simultaneous
    group cost is computed by this boundary, and no leaf timing is summed.
    """
    from .joint_aura import _validated_assignment
    _sha(cost_sha256, "cost payload")
    if inputs.get("schema") != INPUT_SCHEMA:
        raise ValueError("native MoE input schema unsupported")
    if (preflight.get("schema") != "tessera.native_moe_preflight.v1"
            or preflight.get("status") != "untimed_preparation"):
        raise ValueError("native MoE panel requires untimed producer preparation")
    members = _member_roster(inputs["unit"], inputs["members"], inputs["shape"])
    validate_routing(inputs["routing"])
    _equal(inputs["execution"], owner_execution(inputs["shape"], format_name=inputs["format"]),
           "input execution")
    _equal(inputs["profile_role_order"], list(ROLES), "source profile role order")
    _calibration_and_capture(inputs["calibration"], inputs["routing_capture"], unit=inputs["unit"],
                             shape=inputs["shape"], routing=inputs["routing"])
    _equal(inputs["routing_capture_sha256"], identity_sha256(inputs["routing_capture"]), "routing capture digest")
    rows = _validated_assignment(cost_rows, "additive")  # validation only; no scalar summary
    if set(rows) != {member["unit"] for member in members}:
        raise ValueError("native MoE runtime binding must cover exactly all 96 members")
    first = rows[members[0]["unit"]]
    probe, request = first["probe_identity"], inputs["probe_request"]
    from .native_execution_binding import require_reference_quantizer
    reference_quantizer = require_reference_quantizer(inputs, members[0]["activation"], probe)
    if (source_execution_qualification_path is None) != (source_execution_qualification_sha256 is None):
        raise ValueError("source execution qualification requires paired file and digest")
    captured_execution = inputs["routing_capture"].get("source_execution")
    if source_execution_qualification_path is not None:
        qualified = _qualified_source_execution(inputs, probe, source_execution_qualification_path,
                                                source_execution_qualification_sha256)
        if captured_execution is not None:
            _equal(captured_execution, qualified, "captured/qualified source execution")
        captured_execution = qualified
    execution = _source_execution(captured_execution, unit=inputs["unit"])
    _equal(execution, _source_execution(probe.get("source_execution"), unit=inputs["unit"]),
           "capture/probe source execution")
    for key in ("n_probes", "seed_base", "token_scope", "temperature", "distribution", "normalization"):
        _equal(probe[key], request[key], f"predeclared probe {key}")
    _equal(probe["source_model"]["source"], request["source_model"], "probe source model")
    _equal({Path(item["path"]).name: item["sha256"] for item in probe["source_model"]["shards"]},
           request["source_shards"], "probe source checkpoint bytes")
    from .cost_streaming import canonical_streamed_model_semantic_config
    source = inputs["routing_capture"]["producer_source"]
    _equal(source["files"], request["source_shards"], "capture/probe full source files")
    _equal(source["config_sha256"], request["source_config_sha256"], "captured source config bytes")
    _equal(source["auxiliary_sha256"], request["source_auxiliary_sha256"], "captured source auxiliary files")
    # Streamed identity v1 seals logical -> original tensor names, separately
    # from its full shard hashes. Compare that actual shared contract, including
    # every original name; do not demand a nonexistent filename-map field.
    _equal(sorted(source["tensors"]), sorted(set(probe["source_model"]["weight_map"].values())),
           "capture/probe original checkpoint tensor names")
    if "checkpoint_weight_map" in probe["source_model"]:
        _equal(source["tensors"], probe["source_model"]["checkpoint_weight_map"], "capture/probe original checkpoint tensor map")
    _equal(canonical_streamed_model_semantic_config(inputs["routing_capture"]["runtime_config"], where="captured config"),
           canonical_streamed_model_semantic_config(probe["source_model"]["config"], where="probe config"),
           "capture/probe runtime config")
    probe_calibration = _probe_calibration(inputs)
    for field, key in (("calibration_sha256", "calibration_sha256"), ("calibration_shape", "shape"), ("calibration_dtype", "dtype")):
        _equal(probe[field], probe_calibration[key], f"joint {field}")
    quality_members = None
    if inputs["shape"].get("tensor_parallel", 1) > 1:
        if "quality_preparation" not in inputs:
            raise ValueError("native rank-local panel requires independently bound "
                             "full-quality preparation")
        for member in members:
            if not {"quality_rendered_weight", "rank_render_proof"} <= set(member):
                raise ValueError(
                    f"{member['unit']}: a rank-local member declares no full quality "
                    "render and no rank render proof")
        # The panel's own calibration receipt, not the bounded first-sequence
        # probe screen: the historical render was qualified under the campaign
        # draw, and a subset would refuse every real preparation.
        quality_members, quality_context = _qualified_quality_members(
            inputs["quality_preparation"]["prepared"], members=members,
            source_model=probe["source_model"], calibration=inputs["calibration"])
        _equal(quality_context, inputs["quality_preparation"], "quality preparation context")
    for member in members:
        joint = rows[member["unit"]]["joint_operator_identity"]
        quality_render = member["rendered_weight"]
        if quality_members is not None:
            quality = quality_members[member["unit"]]
            quality_render = quality["rendered_weight"]
            _equal(member["quality_rendered_weight"], quality_render,
                   f"{member['unit']} full quality render")
            _equal(member["rank_render_proof"], _rank_render_proof(member, inputs["shape"], quality),
                   f"{member['unit']} rank render proof")
        for key, expected in (("qname", member["unit"]), ("format", member["format"]),
                              ("source_weight", member["source_weight"]), ("rendered_weight", quality_render),
                              ("activation", member["activation"])):
            _equal(joint[key], expected, f"{member['unit']} joint {key}")
        if joint["activation"].get("clip_enabled") is not False:
            raise ValueError("native MoE refuses clipped member joint rows")
    operator = preflight["operator"]
    _equal(operator["members"], [_native_member_identity(member) for member in members], "native member roster")
    for key, expected in (("shape", inputs["shape"]), ("routing", inputs["routing"]),
                          ("profile_role_order", inputs["profile_role_order"]),
                          ("routing_capture_sha256", inputs["routing_capture_sha256"])):
        _equal(operator[key], expected, f"native {key}")
    _equal(preflight["runtime_sha256"], identity_sha256(preflight["runtime"]), "runtime digest")
    _equal(preflight["native_tensors_sha256"], identity_sha256(operator["native_tensors"]), "native tensors")
    _equal(preflight["scheme_sha256"], identity_sha256(operator["scheme"]), "native scheme")
    _equal(operator["config_sha256"], identity_sha256(operator["config"]), "native MoE config")
    _equal(preflight["runtime"]["execution"], inputs["execution"], "native execution")
    _equal(preflight["runtime"]["image"], inputs["runtime_image"], "native image")
    _equal(operator["serving_config_sha256"], _sha(inputs["serving_config_sha256"], "serving configuration"), "native serving config")
    _workspace_identity(preflight["workspace"])
    _equal(preflight["workspace_sha256"], identity_sha256(preflight["workspace"]), "workspace digest")
    route = operator["declared_route"]
    if route.get("decoder") == "torch_materialize_stock":
        # Retain historical FP8 receipt intake. New packed execution preparation
        # explicitly refuses this route in validate_compact_native_route.
        for key, expected in (("kind", "moe"), ("policy", "TESSERA_FP8:resident"),
                              ("contract", "fp8_per_token_dynamic")):
            _equal(route[key], expected, f"native route {key}")
        if not isinstance(route["symbol"], str) or re.fullmatch(r"vllm\.fused_moe\.modular_kernel:.+", route["symbol"]) is None:
            raise ValueError("native MoE route must name the actual modular backend")
    else:
        validate_compact_native_route(inputs, route)
    phases = {}
    for phase in PHASES:
        expected = inputs["phases"][phase]
        _transport_identity(expected)
        _equal(operator["phases"][phase]["transport"], expected["transport"], f"native {phase} transport")
        for key in ("input", "topk_ids", "topk_weights"):
            _equal(expected[key], inputs["routing_capture"]["phases"][phase][key], f"{phase} capture {key}")
        phases[phase] = {**expected, "expected_route": route}
    # The member record's own geometry is the CONTAINER the wire identity
    # names; the runtime binding carries THIS rank's cut of it. The two are
    # different fields and agree only at a world of one, so the binding is
    # derived from the frozen cut rather than copied off the member -- and the
    # derivation is checked against this module's own rank-local reading, so
    # one home decides the arithmetic.
    from .measured_runtime_prices import rank_local_member_shapes
    member_shapes = rank_local_member_shapes(
        {member["unit"]: member["shape"] for member in members},
        tensor_parallel=inputs["shape"].get("tensor_parallel", 1),
        where=f"{inputs['unit']} runtime binding")
    for member in members:
        _equal(list(member_shapes[member["unit"]]),
               rank_local_member_shape(inputs["shape"], member["role"]),
               f"{member['unit']} rank-local binding")
    binding = RuntimeBinding(
        {member["unit"]: member["format"] for member in members},
        {name: row["joint_operator_identity_sha256"] for name, row in rows.items()},
        member_shapes,
        operator_route_identity(route))
    return json.loads(json.dumps({"schema": PANEL_SCHEMA, "unit": inputs["unit"],
        **({"source_acquisition":inputs["source_acquisition"]} if "source_acquisition" in inputs else {}),
        **({"reference_served_quantizer": reference_quantizer} if reference_quantizer is not None else {}),
        **({"quality_preparation": inputs["quality_preparation"]} if quality_members is not None else {}),
        "format": inputs["format"],
        "shape": inputs["shape"], "members": members, "profile_role_order": list(ROLES),
        "routing": inputs["routing"], "routing_capture_sha256": inputs["routing_capture_sha256"],
        "source_sha256": probe["source_model"]["content_sha256"], "calibration_sha256": probe["calibration_sha256"],
        "probe_scope": inputs.get("probe_scope"),
        "source_execution": execution,
        "source_execution_qualification_sha256": source_execution_qualification_sha256,
        "cost_sha256": cost_sha256, "serving_config_sha256": inputs["serving_config_sha256"], "probe_identity_sha256": first["probe_identity_sha256"],
        "runtime_binding": binding.as_dict(), "execution": inputs["execution"], "runtime": preflight["runtime"],
        "native_tensors_sha256": preflight["native_tensors_sha256"], "scheme_sha256": preflight["scheme_sha256"],
        "config_sha256": operator["config_sha256"], "workspace": preflight["workspace"],
        "workspace_sha256": preflight["workspace_sha256"], "numerics": inputs["numerics"], "phases": phases}, allow_nan=False))


def captured_moe_boundary(module, args, kwargs, coordinates, *, unit, source_model,
                          calibration_receipt, producer_source, prefill_rows=512):
    """Return actual first-sequence routed tensors for the existing PAC writer.

    The generic packed collector supplies the original forward arguments and
    calibration coordinates. No router is recomputed here. Returning None on
    later samples lets the collector retain only this bounded native input.
    """
    import torch
    from . import pretrained_initialization_contract
    from .model_profiles import profile_from_model
    from .production_weight_cache import _cb_cache_tensor_identity
    from .tessera_expert_projection import _require_source_identity
    if (type(prefill_rows) is not int or prefill_rows < 1
            or calibration_receipt.get("schema") != "prismaquant.calibration_input.v1"
            or calibration_receipt["shape"][1] < prefill_rows):
        raise ValueError("native MoE boundary requires an exact calibration and bounded first sequence")
    if (not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 2
            or coordinates.shape[1] != 2 or coordinates.dtype != torch.int64):
        raise ValueError("native MoE collector must provide original int64 sample/token coordinates")
    selected = (coordinates[:, 0] == 0) & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < prefill_rows)
    if not bool(selected.any()):
        return None
    positions = coordinates[selected]
    if (positions.shape[0] != prefill_rows
            or not torch.equal(positions[:, 1], torch.arange(prefill_rows, device=coordinates.device))):
        raise ValueError("native MoE first sequence must arrive completely and in original token order")
    model_load_contract = pretrained_initialization_contract(source_model)
    if getattr(source_model.config, "_attn_implementation", None) != "eager":
        raise ValueError("native MoE capture requires the actual eager source attention backend")
    profile = profile_from_model(source_model)
    if (profile.name != "lfm2_moe" or type(module).__name__ not in profile.packed_expert_module_class_names()
            or source_model.get_submodule(unit) is not module):
        raise ValueError("native MoE capture target is not the source model's declared LFM experts")
    # Use the class's declared signature: a hook wrapper may expose *args, but
    # that is not permission to guess positional meanings in a generic collector.
    bound = inspect.signature(type(module).forward).bind(module, *args, **kwargs)
    required = ("hidden_states", "top_k_index", "top_k_weights")
    if any(name not in bound.arguments for name in required):
        raise ValueError("source packed forward does not expose the required routed arguments")
    x, ids, weights = (bound.arguments[name] for name in required)
    if any(not isinstance(value, torch.Tensor) or value.shape[0] != coordinates.shape[0]
           for value in (x, ids, weights)):
        raise ValueError("captured router tensors are not aligned with original calibration coordinates")
    # The outer calibration loop owns CPU coordinates. Move its exact selector
    # losslessly for CUDA indexing; retain the original coordinate values/store.
    selected_on_device = selected.to(device=x.device)
    x, ids, weights = (value[selected_on_device].contiguous() for value in (x, ids, weights))
    shape = {"experts": int(module.num_experts), "hidden_size": int(x.shape[1]),
             "intermediate_size": int(module.down_proj.shape[-1]), "top_k": int(ids.shape[1])}
    validate_geometry(shape)
    _validate_phase_tensors(x, ids, weights, shape, cuda=True)
    parent = source_model.get_submodule(unit.rsplit(".", 1)[0])
    router = parent.gate
    bias = getattr(parent, "expert_bias", None)
    if (type(router).__name__ != "Lfm2MoeTopKRouter" or bias is None or bias.dtype != torch.float32
            or list(bias.shape) != [shape["experts"]] or bias.device != x.device
            or not bool(torch.isfinite(bias).all())):
        raise ValueError("native MoE capture requires the actual FP32-biased LFM router")
    source = _require_source_identity(producer_source)
    routing = {"activation": "silu", "scoring_func": "sigmoid", "renormalize": router.norm_topk_prob,
        "routed_scaling_factor": router.routed_scaling_factor, "apply_router_weight_on_input": False,
        "expert_map": None, "input_dtype": str(x.dtype), "topk_weights_dtype": str(weights.dtype),
        "topk_ids_dtype": str(ids.dtype), "device": str(x.device),
        "weights_contract": "post_renormalization_and_routed_scaling",
        "source_protocol": {"router_class": f"{type(router).__module__}.{type(router).__qualname__}",
            "router_source_sha256": hashlib.sha256(Path(inspect.getfile(type(router))).read_bytes()).hexdigest(),
            "selection_bias": _cb_cache_tensor_identity(bias), "normalization_epsilon": 1e-6,
            "expert_bias_affects": "selection_only"}}
    validate_routing(routing)
    tensors = {"inputs": x, "top_k_index": ids, "top_k_weights": weights,
               "coordinates": positions.contiguous(), "expert_bias": bias}
    from .joint_aura import source_execution_identity
    return {"tensors": tensors, "metadata": {
        "schema": "prismaquant.native_moe_raw_boundary.v1", "unit": unit, "shape": shape,
        "routing": routing, "profile_role_order": list(ROLES),
        "scope": "first calibration sequence; decode uses its first row, not autoregressive generation",
        "calibration_sha256": calibration_receipt["calibration_sha256"],
        "calibration_shape": calibration_receipt["shape"], "calibration_dtype": calibration_receipt["dtype"],
        "producer_source": source, "runtime_config": source_model.config.to_dict(),
        "source_execution": source_execution_identity(source_model),
        "model_load_contract": model_load_contract,
        "attention_implementation": source_model.config._attn_implementation,
        "capture_runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                            "transformers": __import__("transformers").__version__},
        "capture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tensors": {name: _cb_cache_tensor_identity(value) for name, value in tensors.items()}}}


def consume_moe_receipt(path, *, expected_sha256, expected_panel, memory_trace_path=None):
    """Retain whole-stack evidence and distinct persistent runtime workspace.

    WorkspaceManager storage is shared persistent runtime state, not per-apply
    scratch or a complete measured model-fixed price. No cross-row composition
    rule for it is invented here; the observation remains inadmissible as a
    complete allocator runtime table.
    """
    from .native_operator_panel import (native_operator_measurement, native_operator_scratch,
                                        validate_native_numerics)
    raw = Path(path).read_bytes()
    _equal(hashlib.sha256(raw).hexdigest(), _sha(expected_sha256, "receipt"), "receipt file")
    receipt = json.loads(raw)
    if receipt.get("schema") == "prismaquant.native_moe_late_binding.v1":
        from .native_moe_execution_binding import resolve_execution_binding
        receipt = resolve_execution_binding(receipt, expected_panel)
    if (receipt.get("schema") != "tessera.native_moe_operator_receipt.v1"
            or receipt.get("status") != "timing_admissible"):
        raise ValueError("native MoE receipt has no admitted whole-apply observation")
    if expected_panel.get("schema") != PANEL_SCHEMA:
        raise ValueError("native MoE receipt requires its independently frozen panel")
    _equal(receipt["panel"], expected_panel, "receipt panel")
    _equal(receipt["panel_sha256"], identity_sha256(expected_panel), "receipt panel digest")
    _equal(receipt["runtime"], expected_panel["runtime"], "receipt runtime")
    _equal(receipt["runtime_sha256"], identity_sha256(expected_panel["runtime"]), "receipt runtime digest")
    # The producer publishes workspace identity in its complete frozen panel,
    # with the independently observed bytes/digest in resources below. There
    # are no duplicate top-level workspace fields in the v1 receipt.
    workspace = receipt["panel"]["workspace"]
    _workspace_identity(workspace)
    _equal(identity_sha256(workspace), expected_panel["workspace_sha256"], "receipt workspace digest")
    binding = RuntimeBinding.from_dict(expected_panel["runtime_binding"])
    members = _member_roster(expected_panel["unit"], expected_panel["members"], expected_panel["shape"])
    _equal(dict(binding.member_formats), {member["unit"]: member["format"] for member in members}, "runtime member coverage")
    # The binding carries THIS rank's cut of each member, not the container the
    # member record and the wire identity frame. Restating it from the frozen
    # geometry is what keeps a panel from publishing a container shape where the
    # runtime loads a cut -- and the canonical combination of every rank's cut
    # is the container, which the roster above has already checked.
    _equal({unit: tuple(shape) for unit, shape in binding.member_shapes.items()},
           {member["unit"]: tuple(rank_local_member_shape(expected_panel["shape"], member["role"]))
            for member in members},
           "runtime member shapes")
    operator = receipt["operator"]
    _equal(operator["members"], [_native_member_identity(member) for member in members], "receipt native members")
    for key in ("shape", "routing", "profile_role_order", "routing_capture_sha256", "serving_config_sha256"):
        _equal(operator[key], expected_panel[key], f"receipt {key}")
    _equal(identity_sha256(operator["native_tensors"]), expected_panel["native_tensors_sha256"], "receipt native tensors")
    _equal(identity_sha256(operator["scheme"]), expected_panel["scheme_sha256"], "receipt scheme")
    _equal(identity_sha256(operator["config"]), expected_panel["config_sha256"], "receipt MoE config")
    _equal(operator["config_sha256"], expected_panel["config_sha256"], "receipt MoE config digest")
    resources = receipt["resources"]
    complete = resources.get("status") == "complete_operator_bound"
    workspace_bytes = _bytes(resources["workspace_resident_bytes"], "runtime workspace resident")
    _equal(workspace_bytes, expected_panel["workspace"]["resident_bytes"], "workspace accounting")
    _equal(resources["workspace_sha256"], expected_panel["workspace_sha256"], "resource workspace")
    if complete:
        if memory_trace_path is None:
            raise ValueError("complete native MoE resource bound requires its actual memory trace")
        trace = json.loads(Path(memory_trace_path).read_text())
        _equal(identity_sha256(trace), resources["trace_sha256"], "memory trace")
        _equal(trace["capture"]["collector_library_sha256"],
               expected_panel["runtime"]["resource_collector"]["library_sha256"], "resource collector")
    observations = {}
    for phase in PHASES:
        observed, expected = receipt["phases"][phase], expected_panel["phases"][phase]
        for name in ("input", "topk_ids", "topk_weights", "reference_qdq", "reference_output", "transport"):
            _equal(observed[name], expected[name], f"{phase} {name}")
        route = observed["route"]
        _equal({key: route[key] for key in expected["expected_route"]}, expected["expected_route"], f"{phase} route")
        geometry = expected_panel["shape"]
        shape = f"M{expected['m']}:N{2 * geometry['intermediate_size']}:K{geometry['hidden_size']}"
        if route.get("state") != "served" or route.get("reason") is not None or route.get("shape") != shape:
            raise ValueError(f"{phase}: native MoE route state/shape differs")
        validate_native_numerics(observed["numerics"], expected_panel["numerics"],
                                 phase=phase, kind="numerics")
        # Exact unconditionally: this panel's reference_qdq is _activation_qdq
        # of the member spec, which is the identity for a spec that does not
        # quantise and the shared oracle for one that does, so zero is
        # reachable either way.  One mechanism, not a weaker copy of it -- the
        # dense panel reaches the same gate through its own single member.
        validate_native_numerics(observed["qdq_numerics"], expected_panel["numerics"],
                                 phase=phase, kind="qdq_numerics", exact=True)
        measurement = native_operator_measurement(observed["measurement"], path=path, expected_sha256=expected_sha256)
        bound = resources["phases"][phase].get("bound")
        scratch = native_operator_scratch(bound, phase=phase) if complete else None
        observations[phase] = {"measurement": measurement.as_dict(), "median_ms": measurement.median_ms,
                              "peak_scratch_bytes": scratch, "resource_bound": bound,
                              "input_bytes": expected["input"]["logical_bytes"],
                              "output_bytes": expected["reference_output"]["logical_bytes"]}
    return {"schema": "prismaquant.native_moe_observation.v1", "status": "operator_evidence",
            "unit": expected_panel["unit"], "format": expected_panel["format"],
            "runtime_binding": binding.as_dict(),
            "panel_sha256": identity_sha256(expected_panel), "receipt_sha256": expected_sha256,
            "cost_sha256": expected_panel["cost_sha256"], "probe_scope": expected_panel.get("probe_scope"),
            "phases": observations,
            "serialized_unit_bytes": sum(_bytes(member["wire"]["blob_bytes"], "member wire") for member in members),
            "resident_bytes": _bytes(resources["resident_bytes"], "native layer resident"),
            "workspace_resident_bytes": workspace_bytes, "workspace_sha256": expected_panel["workspace_sha256"],
            "full_model_resources": None, "runtime_table_admissible": False,
            "unknown": ["fixed_and_full_model_resources", "cross_operator_workspace_composition"]
                       + ([] if complete else ["native_operator_scratch"])}


def routed_boundary_inputs(payload, *, calibration_receipt, capture_manifest, device, source_model_identity=None):
    """Transport an independently hashed PAC boundary into the native protocol.

    Storage may be on CPU, but metadata describes the original CUDA invocation.
    Only lossless int32/FP32 transport and the declared first-row decode subset
    are introduced. Canonical capture provenance and every original tensor are
    checked before forming either phase.
    """
    import copy
    import torch
    from .production_weight_cache import _cb_cache_tensor_identity as tensor_identity
    metadata = payload.get("boundary_metadata") or {}
    if (payload.get("source") != "routed_boundary_capture"
            or metadata.get("schema") != "prismaquant.native_moe_raw_boundary.v1"
            or capture_manifest.get("schema") != "prismaquant.tessera_calibration_cache.v2"
            or capture_manifest.get("status") != "complete"):
        raise ValueError("native MoE requires an original PAC boundary and canonical capture v2")
    shape, unit = metadata["shape"], metadata["unit"]
    validate_geometry(shape)
    validate_routing(metadata["routing"])
    _equal(metadata["profile_role_order"], list(ROLES), "captured profile role order")
    identity = capture_manifest["identity"]
    prefix=isinstance(metadata['model_load_contract'],dict) and metadata['model_load_contract'].get('schema')=='prismaquant.streaming_prefix_initialization.v1'
    if prefix:
        if source_model_identity is None:
            raise ValueError('fresh prefix intake requires independently bound original source identity')
        _validate_prefix_capture(metadata,source_model_identity)
    for key in (("attention_implementation", "capture_runtime") if prefix else
                ("model_load_contract", "attention_implementation", "capture_runtime")):
        _equal(metadata[key], identity[key], f"boundary/capture {key}")
    original = {}
    for key in ("inputs", "top_k_index", "top_k_weights", "coordinates", "expert_bias"):
        value = payload[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"native MoE PAC lacks original {key} tensor")
        _equal(tensor_identity(value), metadata["tensors"][key], f"original boundary {key}")
        original[key] = value
    x, ids, weights = (original[key] for key in ("inputs", "top_k_index", "top_k_weights"))
    _validate_phase_tensors(x, ids, weights, shape, cuda=False)
    coords = original["coordinates"]
    if (coords.dtype != torch.int64 or list(coords.shape) != [x.shape[0], 2]
            or x.shape[0] != calibration_receipt["shape"][1]
            or not torch.equal(coords[:, 0], torch.zeros(x.shape[0], dtype=torch.int64, device=coords.device))
            or not torch.equal(coords[:, 1], torch.arange(x.shape[0], dtype=torch.int64, device=coords.device))):
        raise ValueError("native MoE PAC must retain the complete ordered first calibration sequence")
    for key, value in (("input_dtype", x), ("topk_ids_dtype", ids), ("topk_weights_dtype", weights)):
        _equal(str(value.dtype), metadata["routing"][key], f"original {key}")
    bias = original["expert_bias"]
    if bias.dtype != torch.float32 or list(bias.shape) != [_shape_for_roster(shape)["experts"]] or not bool(torch.isfinite(bias).all()):
        raise ValueError("native MoE original router bias must remain finite FP32")
    protocol = metadata["routing"]["source_protocol"]
    if geometry_family(shape) == "glm53_next_routed_stack_v1":
        _equal({key: tensor_identity(bias)[key] for key in ("dtype", "content_sha256")},
               protocol["correction_bias"], "original GLM correction bias")
    else:
        _equal(tensor_identity(bias), protocol["selection_bias"], "original router bias")
    routing = copy.deepcopy(metadata["routing"])
    routing.update(topk_ids_dtype="torch.int32", topk_weights_dtype="torch.float32")
    phases, values = {}, {}
    for phase, count in (("prefill", x.shape[0]), ("decode", 1)):
        raw_ids, raw_weights = ids[:count].to(device), weights[:count].to(device)
        values[phase] = {"input": x[:count].to(device), "source_topk_ids": raw_ids,
            "source_topk_weights": raw_weights, "topk_ids": raw_ids.to(torch.int32),
            "topk_weights": raw_weights.to(torch.float32)}
        transport = {key: {"source": tensor_identity(values[phase]["source_" + key]),
            "supplied": tensor_identity(values[phase][key]),
            "operation": "identity" if values[phase]["source_" + key].dtype == values[phase][key].dtype
                         else "lossless_dtype_conversion"} for key in ("topk_ids", "topk_weights")}
        validate_transport(values[phase], transport)
        phases[phase] = {"m": count, "transport": transport,
            **{key: tensor_identity(values[phase][key]) for key in ("input", "topk_ids", "topk_weights")}}
    capture = {key: copy.deepcopy(metadata[key]) for key in ("unit", "shape", "calibration_sha256",
        "calibration_shape", "calibration_dtype", "producer_source", "runtime_config", "capture_source_sha256",
        "model_load_contract", "attention_implementation", "capture_runtime", "scope")}
    capture.update(schema="prismaquant.routed_boundary_capture.v1", routing=routing, phases=phases)
    if 'replay' in metadata:capture['replay']=copy.deepcopy(metadata['replay'])
    if "source_execution" in metadata:
        capture["source_execution"] = copy.deepcopy(metadata["source_execution"])
    if metadata.get("dev_uncertified") is not None:
        if metadata["dev_uncertified"] is not True:
            raise ValueError("source acquisition must retain its uncertified DEV declaration")
        capture["source_acquisition"]={key:copy.deepcopy(metadata[key]) for key in
            ("dev_uncertified","dev_mode","source_cache_reuse","capture_device_envelope") if key in metadata}
    _calibration_and_capture(calibration_receipt, capture, unit=unit, shape=shape, routing=routing)
    source = capture["producer_source"]
    for name, digest in {**source["files"], **source["auxiliary_sha256"], "config.json": source["config_sha256"]}.items():
        # Auxiliary files outside the ordinary capture glob still remain sealed
        # by the canonical census producer; files present in both must agree.
        if name in identity["source_files"]:
            _equal(digest, identity["source_files"][name], f"boundary/capture source {name}")
    return capture, values, bias.to(device)


def _probe_calibration(inputs):
    """Keep a bounded joint screen distinct from full render/capture provenance."""
    parent = inputs["calibration"]
    subset, scope = inputs.get("probe_calibration"), inputs.get("probe_scope")
    if subset is None and scope is None:
        return parent
    if (not isinstance(subset, dict) or subset.get("schema") != "prismaquant.calibration_input.v1"
            or not isinstance(scope, dict) or set(scope) != {"schema", "parent_calibration_sha256",
                "subset_calibration_sha256", "sample_indices", "scope"}
            or scope["schema"] != "prismaquant.native_probe_subset.v1"
            or scope["sample_indices"] != [0] or scope["scope"] != "first_sequence_integration_screen"
            or subset.get("shape") != [1, parent["shape"][1]] or subset.get("dtype") != parent["dtype"]):
        raise ValueError("native MoE subset probe needs its explicit first-sequence screen binding")
    _equal(scope["parent_calibration_sha256"], parent["calibration_sha256"], "probe parent calibration")
    _equal(scope["subset_calibration_sha256"], _sha(subset["calibration_sha256"], "probe subset"), "probe subset calibration")
    return subset


def verified_probe_subset(parent_tokens, subset_tokens, *, parent_calibration, subset_calibration):
    """Verify actual exact IDs before a first-sequence screen is authorized.

    Full Hessian/reference provenance remains in the original capture. The
    subset is separate joint currency, never relabeled as the full draw.
    """
    import torch
    from .production_weight_cache import _cb_cache_tensor_identity
    for tokens, receipt, label in ((parent_tokens, parent_calibration, "parent"),
                                   (subset_tokens, subset_calibration, "subset")):
        if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.int64
                or receipt.get("schema") != "prismaquant.calibration_input.v1"):
            raise ValueError("native MoE probe subset requires exact int64 calibration tensors")
        actual = _cb_cache_tensor_identity(tokens)
        for key, expected in (("shape", actual["shape"]), ("dtype", actual["dtype"]),
                              ("calibration_sha256", actual["content_sha256"])):
            _equal(receipt[key], expected, f"actual {label} {key}")
    if not torch.equal(parent_tokens[:1].cpu(), subset_tokens.cpu()):
        raise ValueError("native MoE probe subset differs from actual first calibration sequence")
    scope = {"schema": "prismaquant.native_probe_subset.v1",
        "parent_calibration_sha256": parent_calibration["calibration_sha256"],
        "subset_calibration_sha256": subset_calibration["calibration_sha256"],
        "sample_indices": [0], "scope": "first_sequence_integration_screen"}
    _probe_calibration({"calibration": parent_calibration, "probe_calibration": subset_calibration, "probe_scope": scope})
    return scope
