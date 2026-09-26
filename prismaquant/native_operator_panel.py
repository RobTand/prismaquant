"""Frozen PWC inputs and a narrow native operator receipt bridge.

No serving runtime is imported here. Tessera owns native preparation/execution;
PQ supplies the actual production render, shared activation QDQ and joint-cost
identity. An operator observation never invents fixed/full-model resources or
becomes a complete measured-runtime allocation table by itself.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Mapping

from .joint_aura import identity_sha256, validate_joint_aura_entry
from .measured_runtime_prices import OperatorMeasurement
from .digests import DIRECT_ASCII_STRICT

INPUT_SCHEMA = "prismaquant.native_dense_inputs.v1"
PANEL_SCHEMA = "tessera.native_dense_panel.v1"
EXECUTION = {"owner_kind": "single_dense", "mode": "resident",
             "execution_mode": "eager", "tensor_parallel": 1, "bias": False}
PHASES = ("prefill", "decode")


def operator_route_identity(route):
    """The route class a table binding names, not the GEMM symbol alone.

    `TESSERA_FP8` and `TESSERA_NVFP4` both execute `torch._scaled_mm`, on
    differently packed operands under different activation contracts. A
    binding that carried only the symbol made the two indistinguishable in the
    one field a downstream consumer compares, so two route classes read as one.

    The identity is therefore the declared route itself -- every coordinate the
    producer declared and `consume_native_receipt` admitted the observed route
    against -- spelled by the same canonical `json.dumps` `identity_sha256`
    hashes with, so no second canonical form of a route exists in this tree and
    key order is not part of the answer. A route with no named symbol is
    refused rather than given an identity: a route nobody can execute is not a
    class.
    """
    if not isinstance(route, Mapping) or not isinstance(route.get("symbol"), str) or not route["symbol"].strip():
        raise ValueError("a declared route must name the symbol it executes")
    return DIRECT_ASCII_STRICT.text(route)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name}: lowercase SHA256 required")
    return value


def _equal(actual, expected, name):
    if identity_sha256(actual) != identity_sha256(expected):
        raise ValueError(f"native panel {name} differs from independently frozen input")


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: finite nonnegative number required")
    return value


def _bytes(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: nonnegative integer bytes required")
    return value


#: bf16's unit roundoff and fp32's.  The only constants in the derived bound,
#: and both come from the numerical precision of a dtype, which is the one
#: source principle 2 admits.
_U_BF16 = 2.0 ** -8
_U_FP32 = 2.0 ** -24


def derive_gemm_numerics(magnitude: float, *, k: int) -> tuple[dict, dict]:
    """The GEMM-output tolerance, derived from the dtypes and the operands.

    Both sides compute the same dot products and differ only in how they round
    on the way.  The reference rounds the dequantised activation and the
    rendered weight to bf16, accumulates in fp32 and rounds the result to
    bf16; the native kernel consumes the codes and scales and does the same
    accumulation in a different order.  Per output element ``j``::

        |native - ref|  <=  (4 u_bf16 + 2 K u_fp32) * sum_i |qx_i| |w_ij|

    -- ``2 u_bf16`` for the two operand roundings, ``u_bf16`` for each side's
    final rounding, ``gamma_K ~ K u_fp32`` per side for the fp32 accumulation
    (the per-block scale products fold into that term), and terms of order
    ``u^2`` dropped.

    Two properties worth naming.  First, ``4 u_bf16`` is ``2^-6`` -- exactly
    the constant this replaces.  The old default had the right MAGNITUDE and
    the wrong QUANTITY: it multiplied ``|expected|`` instead of
    ``sum_i |qx_i||w_ij|``, which are the same thing only when nothing cancels,
    and they differ by however much the dot product cancels -- which is
    per unit, and per row count.  That is the whole mechanism by which one
    unit passed and another failed on a constant neither of them derived.
    Second, this is the SCALAR projection of a per-element bound, because the
    receipt harness reads one ``atol``/``rtol`` pair: taking the maximum over
    ``j`` makes it correct and loose, loosest where the worst row cancels
    hardest.  A per-element receipt would tighten it and is filed rather than
    approximated here.

    ``rtol`` is zero: the whole bound is carried by ``atol``, because the
    quantity it is proportional to is not ``|expected|``.
    """
    _number(magnitude, "operand magnitude")
    if type(k) is not int or k < 1:
        raise ValueError("derived GEMM tolerance needs the contraction length")
    coefficient = 4.0 * _U_BF16 + 2.0 * k * _U_FP32
    atol = coefficient * float(magnitude)
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("derived GEMM tolerance is not a finite bound")
    return ({"atol": atol, "rtol": 0.0}, {
        "schema": "prismaquant.native_gemm_tolerance.v1",
        "bound": "(4*u_bf16 + 2*K*u_fp32) * max_j sum_i |qx_i||w_ij|",
        "u_bf16": _U_BF16, "u_fp32": _U_FP32, "k": int(k),
        "coefficient": coefficient, "operand_magnitude": float(magnitude),
        "rtol_is_zero_because": "the bound is proportional to the operand "
                                "magnitude sum, not to |expected|",
        "scope": "gemm_output_only",
    })


def native_platform(device=None) -> str:
    """The platform key a published attestation is addressed by.

    ``sm_<major><minor>``, the same spelling
    ``native_receipt_table`` and ``tessera_route_receipt`` already build a
    serving context from, so an attestation is looked up under the name the
    route receipt is priced under. No CUDA device is no platform: this is
    called on the producer's GPU path and a host with no device has no
    activation to price.
    """
    import torch

    if not torch.cuda.is_available():
        raise ValueError(
            "native panel: no CUDA device, so there is no platform to look a "
            "quantiser attestation up under")
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def require_attested_activation_oracle(activation, *, platform, table=None,
                                       executing_image=""):
    """Refuse to freeze a ``reference_qdq`` this producer cannot attest.

    ``reference_qdq`` is not a runtime artifact: it is PrismaQuant's own
    re-implementation of the runtime's activation quantiser, and a panel that
    freezes it is about to price an activation residual against a rounding
    rule nobody published.  Principle 14 reads that as refused, so the
    attestation is taken BEFORE the reference exists, at the one place the
    reference is made.

    A unit whose format does not quantise its input has no such rule and
    freezes nothing to attest; it returns ``None`` and is unaffected.

    ``executing_image`` is the registry digest the panel will run in, and it
    SELECTS the attestation when the contract publishes more than one for the
    platform -- one per serving image since Tessera contract v33.  Left empty
    against a single-table platform, nothing changes: the stamp carries that
    table's image and :func:`require_panel_execution_scope` still refuses the
    frozen panel when the two disagree.  Left empty against a platform that
    publishes several, the reader refuses rather than picking one.
    """
    if not activation.get("quantizes_input"):
        return None
    if not platform:
        raise ValueError(
            "native panel: an activation attestation is addressed by platform "
            "and none was supplied; a table taken on another device attests "
            "another kernel")
    contract = (activation.get("static_contract") or {}).get("execution")
    if not contract:
        # A quantiser whose scale is DERIVED FROM x: both sides compute the
        # same function of the same tensor, which is why every fp8 cell
        # measured on 2026-09-13 agreed at exactly 0.0.  That is evidence the
        # oracle is right, not an attestation that it is the same function,
        # and the difference is recorded here rather than left to be inferred
        # from the stamp's absence.  Its table is a different shape -- there is
        # no G to publish against -- so inventing one here would be the guess
        # this mechanism exists to refuse; filed instead.  The exact gate in
        # validate_native_numerics still applies, and still bites.
        return {
            "schema": "prismaquant.activation_quantizer_attestation.v1",
            "activation_contract": None,
            "quantizer": activation.get("quantizer"),
            "status": "unattested_dynamic_scale",
            "attests": None,
            # No rounding rule is attested here, so there is no claim to
            # scope to an image -- but the absence is written down rather
            # than left to be inferred from a missing key (#715).
            "generated": None,
            "generated_absent_because": "no static table addresses a scale "
                                        "derived from x, so there is no "
                                        "attested table and no image it was "
                                        "generated on",
            "does_not_attest": ["activation_to_code_rounding"],
            "why": "the scale is derived from x, so no static table addresses "
                   "it; see RobTand/prismaquant#567",
        }
    from .tessera_runtime_contract import require_activation_quantizer_attested

    return require_activation_quantizer_attested(
        contract, platform=platform, table=table,
        executing_image=executing_image or "")


def prepare_native_inputs(cache, source_weight, activation_rows, *, unit, format_name,
                          calibration_receipt, wire_blob, wire_record, encoding_identity,
                          prefill_rows, decode_rows, max_resident_bytes,
                          activation_quantizers=None, runtime_image=""):
    """Prepare independent references from existing resident PWC/activation data.

    ``encoding_identity`` must be derived by the producer from the actual
    source/Hessian/calibration settings, not copied from the wire record. The
    caller pins that preparation input artifact before executing this function.
    Artifact transport and tensor hashing occur outside any timed native apply.

    **The numerical tolerance is derived here, not passed in.**  It used to be
    an argument whose only value in practice was a CLI default of ``0.015625``
    -- 2^-6, a dtype-shaped number wearing a derivation it did not have
    (principle 2), applied to ``|expected|`` and a flat floor rather than to
    the quantity the error is actually proportional to.  That is why one
    unit's verdict came out a function of its calibrated ``max_abs`` and of how
    many rows were sampled.  See :func:`derive_gemm_numerics` for what replaces
    it and :func:`validate_native_numerics` for the activation gate, which is
    now exact and has no tolerance at all.

    ``activation_quantizers`` is a seam for tests only: left ``None``, the
    attestation is read from the installed runtime's packaged contract.

    ``runtime_image`` is the image this panel will be measured in, the same
    reference the driver writes into ``inputs["runtime_image"]`` and
    ``consume_native_receipt`` later compares the receipt against. It is
    passed here because the contract may publish one quantiser attestation
    per serving image (Tessera contract v33), and which one covers this
    measurement is decided by the image, never by list order
    (RobTand/prismaquant#926).
    """
    import torch
    from tessera.cached_unit import verify_cached_unit
    from tessera.unit_artifact import read_unit_artifact
    from . import format_registry as fr
    from .joint_aura import activation_identity, prefetch_joint_cache
    from .perturbed_x_cache import _activation_qdq
    from .production_weight_cache import ProductionWeightCache, _cb_cache_tensor_identity

    if not isinstance(cache, ProductionWeightCache):
        raise TypeError("native panel requires the actual ProductionWeightCache")
    for value, name in ((source_weight, "source"), (activation_rows, "activation rows")):
        if value.ndim != 2 or value.dtype != torch.bfloat16 or value.device.type != "cuda":
            raise ValueError(f"native panel {name} requires resident 2-D CUDA BF16")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"native panel {name} is nonfinite")
    if activation_rows.shape[1] != source_weight.shape[1]:
        raise ValueError("native panel activation/source width differs")
    for rows in (prefill_rows, decode_rows):
        if type(rows) is not int or rows < 1 or rows > activation_rows.shape[0]:
            raise ValueError("native panel phase rows exceed retained calibration activations")
    if calibration_receipt.get("schema") != "prismaquant.calibration_input.v1":
        raise ValueError("native panel requires an exact calibration input receipt")
    _sha(calibration_receipt["calibration_sha256"], "calibration")
    prefetch = prefetch_joint_cache(cache, [unit], {unit: [format_name]},
                                   max_resident_bytes=max_resident_bytes)
    rendered = cache.get(unit, format_name).to(device=source_weight.device)
    if rendered.shape != source_weight.shape or rendered.dtype != torch.bfloat16:
        raise ValueError("native panel PWC source/render dtype or shape differs")
    verify_cached_unit(wire_blob, wire_record, encoding_identity)
    decoded = read_unit_artifact(wire_blob, device=str(rendered.device)).to(rendered.dtype)
    _equal(_cb_cache_tensor_identity(decoded), _cb_cache_tensor_identity(rendered), "wire/PWC decode")
    del decoded
    spec = fr.get_format(format_name)
    activation = activation_identity(spec, cache.activation_max_abs or {}, unit)
    if activation["clip_enabled"]:
        raise ValueError("native operator does not implement PQ's optional activation preclip")
    attestation = require_attested_activation_oracle(
        activation, platform=native_platform(source_weight.device),
        table=activation_quantizers, executing_image=runtime_image or "")
    tensors = {"source_weight": source_weight, "rendered_weight": rendered}
    phases = {}
    magnitude = 0.0
    with torch.inference_mode():
        for phase, count in (("prefill", prefill_rows), ("decode", decode_rows)):
            x = activation_rows[:count].contiguous()
            qx = (_activation_qdq(x, spec, cache.activation_max_abs or {}, unit)
                  if spec.act_quant_changes_input else x)
            output = torch.nn.functional.linear(qx, rendered)
            # The bound below is proportional to sum_i |qx_i||w_ij|, so it is
            # read off the operands of the apply it bounds, once per phase, and
            # the looser of the two covers both -- the panel carries one
            # tolerance and the harness applies it to both phases.
            magnitude = max(magnitude, float(torch.nn.functional.linear(
                qx.abs().float(), rendered.abs().float()).max()))
            for name, value in (("input", x), ("reference_qdq", qx), ("reference_output", output)):
                tensors[f"{phase}.{name}"] = value
            phases[phase] = {"m": count, **{name: _cb_cache_tensor_identity(tensors[f"{phase}.{name}"])
                for name in ("input", "reference_qdq", "reference_output")}}
    numerics, derivation = derive_gemm_numerics(magnitude, k=source_weight.shape[1])
    return {
        "schema": INPUT_SCHEMA, "unit": unit, "format": format_name,
        "shape": list(source_weight.shape), "source_weight": _cb_cache_tensor_identity(source_weight),
        "rendered_weight": _cb_cache_tensor_identity(rendered), "activation": activation,
        "calibration": calibration_receipt, "activation_rows": _cb_cache_tensor_identity(activation_rows),
        "wire": {"blob_sha256": hashlib.sha256(wire_blob).hexdigest(), "blob_bytes": len(wire_blob),
                 "record": wire_record}, "numerics": dict(numerics),
        "numerics_derivation": derivation,
        "activation_quantizer_attestation": attestation,
        "execution": dict(EXECUTION),
        "phases": phases, "prefetch": prefetch,
    }, tensors


def freeze_native_panel(inputs, preflight, cost_row, *, cost_sha256):
    """Join independently frozen PWC references to untimed native facts and cost.

    A legacy MSE row is refused; only actual validated joint AURA rows can bind
    this panel. Native facts are declarations for subsequent measurement, not
    substituted numerical references or evidence of successful execution.
    """
    _sha(cost_sha256, "cost payload")
    if inputs.get("schema") != INPUT_SCHEMA:
        raise ValueError("native panel input schema unsupported")
    if (preflight.get("schema") != "tessera.native_dense_preflight.v1"
            or preflight.get("status") != "untimed_preparation"):
        raise ValueError("native panel requires untimed producer preparation")
    if not validate_joint_aura_entry(cost_row):
        raise ValueError("native panel requires an actual joint AURA cost row")
    joint = cost_row["joint_operator_identity"]
    operator = preflight["operator"]
    probe = cost_row["probe_identity"]
    from .native_execution_binding import require_reference_quantizer
    reference_quantizer = require_reference_quantizer(inputs, joint["activation"], probe)
    request = inputs["probe_request"]
    for key in ("n_probes", "seed_base", "token_scope", "temperature", "distribution", "normalization"):
        _equal(probe[key], request[key], f"predeclared probe {key}")
    _equal(probe["source_model"]["source"], request["source_model"], "probe source model")
    _equal({Path(item["path"]).name: item["sha256"] for item in probe["source_model"]["shards"]},
           request["source_shards"], "probe source checkpoint bytes")
    for key, expected in (("qname", inputs["unit"]), ("format", inputs["format"]),
                          ("source_weight", inputs["source_weight"]),
                          ("rendered_weight", inputs["rendered_weight"]),
                          ("activation", inputs["activation"])):
        _equal(joint[key], expected, f"joint {key}")
    _equal(probe["calibration_sha256"], inputs["calibration"]["calibration_sha256"], "joint calibration")
    _equal(probe["calibration_shape"], inputs["calibration"]["shape"], "joint calibration shape")
    _equal(probe["calibration_dtype"], inputs["calibration"]["dtype"], "joint calibration dtype")
    for key in ("source_weight", "rendered_weight"):
        _equal(operator[key], inputs[key], f"native {key}")
    _equal(operator["input_global_scale"], joint["activation"]["input_global_scale"], "native input scale")
    _equal(operator["clip_enabled"], False, "native clip")
    _equal(operator["wire_sha256"], inputs["wire"]["blob_sha256"], "native wire")
    _equal(operator["wire_record_sha256"], identity_sha256(inputs["wire"]["record"]), "native wire record")
    _equal(preflight["runtime_sha256"], identity_sha256(preflight["runtime"]), "runtime digest")
    _equal(preflight["native_tensors_sha256"], identity_sha256(operator["native_tensors"]), "native tensors")
    _equal(preflight["scheme_sha256"], identity_sha256(operator["scheme"]), "native scheme")
    _equal(preflight["runtime"]["execution"], EXECUTION, "native execution")
    _equal(preflight["runtime"]["image"], inputs["runtime_image"], "native image reference")
    route = operator["declared_route"]
    _equal(route["contract"], operator["activation_contract"], "activation route")
    panel = {
        "schema": PANEL_SCHEMA,
        **({"reference_served_quantizer": reference_quantizer} if reference_quantizer is not None else {}),
        "unit": inputs["unit"], "format": inputs["format"],
        "shape": inputs["shape"], "source_sha256": probe["source_model"]["content_sha256"],
        "calibration_sha256": probe["calibration_sha256"], "cost_sha256": cost_sha256,
        "probe_identity_sha256": cost_row["probe_identity_sha256"],
        "joint_operator_identity_sha256": cost_row["joint_operator_identity_sha256"],
        "joint_operator_identity": joint, "wire": inputs["wire"], "execution": dict(EXECUTION),
        "runtime": preflight["runtime"], "native_tensors_sha256": preflight["native_tensors_sha256"],
        "scheme_sha256": preflight["scheme_sha256"], "numerics": inputs["numerics"],
        "numerics_derivation": inputs["numerics_derivation"],
        "activation_quantizer_attestation": inputs["activation_quantizer_attestation"],
        "phases": {phase: {**inputs["phases"][phase], "expected_route": route} for phase in PHASES},
    }
    return json.loads(json.dumps(panel, allow_nan=False))


def consume_native_receipt(path, *, expected_sha256, expected_panel, memory_trace_path=None):
    """Validate an exact receipt and retain unknown full-model resource fields.

    Returns warmed operator evidence only. The existing runtime table still
    needs independently measured fixed work/KV resources and whole-unit/fused
    coverage; this bridge does not fabricate that table or its missing prices.
    """
    raw = Path(path).read_bytes()
    _equal(hashlib.sha256(raw).hexdigest(), _sha(expected_sha256, "receipt"), "receipt file")
    receipt = json.loads(raw)
    if receipt.get("schema") == "prismaquant.native_dense_late_binding.v1":
        from .native_execution_binding import resolve_execution_binding
        receipt = resolve_execution_binding(receipt, expected_panel)
    if receipt.get("schema") != "tessera.native_dense_operator_receipt.v1" or receipt.get("status") != "timing_admissible":
        raise ValueError("native receipt has no admitted numerical/timing observation")
    _equal(receipt["panel"], expected_panel, "receipt panel")
    _equal(receipt["panel_sha256"], identity_sha256(expected_panel), "receipt panel digest")
    _equal(receipt["runtime"], expected_panel["runtime"], "receipt runtime")
    _equal(receipt["runtime_sha256"], identity_sha256(expected_panel["runtime"]), "receipt runtime digest")
    operator = receipt["operator"]
    joint = expected_panel["joint_operator_identity"]
    for key in ("source_weight", "rendered_weight"):
        _equal(operator[key], joint[key], f"receipt {key}")
    _equal(operator["input_global_scale"], joint["activation"]["input_global_scale"], "receipt activation scale")
    _equal(operator["clip_enabled"], False, "receipt activation clip")
    _equal(operator["wire_sha256"], expected_panel["wire"]["blob_sha256"], "receipt wire")
    _equal(operator["wire_record_sha256"], identity_sha256(expected_panel["wire"]["record"]), "receipt wire record")
    _equal(identity_sha256(operator["native_tensors"]), expected_panel["native_tensors_sha256"], "receipt native tensors")
    _equal(identity_sha256(operator["scheme"]), expected_panel["scheme_sha256"], "receipt scheme")
    resources = receipt["resources"]
    complete = resources.get("status") == "complete_operator_bound"
    if complete:
        if memory_trace_path is None:
            raise ValueError("complete native resource bound requires its actual memory trace")
        trace = json.loads(Path(memory_trace_path).read_text())
        _equal(identity_sha256(trace), resources["trace_sha256"], "memory trace")
        _equal(trace["capture"]["collector_library_sha256"],
               expected_panel["runtime"]["resource_collector"]["library_sha256"], "resource collector")
    quantizes_input = bool(
        expected_panel["joint_operator_identity"]["activation"]["quantizes_input"])
    require_panel_activation_attestation(expected_panel, quantizes_input)
    activation_scope = require_panel_execution_scope(
        expected_panel, executing_image=(expected_panel.get("runtime") or {}).get("image"))
    observations = {}
    for phase in PHASES:
        observed, expected = receipt["phases"][phase], expected_panel["phases"][phase]
        for name in ("input", "reference_qdq", "reference_output"):
            _equal(observed[name], expected[name], f"{phase} {name}")
        route = observed["route"]
        _equal({key: route[key] for key in expected["expected_route"]}, expected["expected_route"], f"{phase} route")
        if (route.get("state") != "served" or route.get("reason") is not None
                or route.get("shape") != f"M{expected['m']}:N{expected_panel['shape'][0]}:K{expected_panel['shape'][1]}"):
            raise ValueError(f"{phase}: native route state/shape differs")
        validate_native_numerics(observed["numerics"], expected_panel["numerics"],
                                 phase=phase, kind="numerics")
        validate_native_numerics(observed["qdq_numerics"], expected_panel["numerics"],
                                 phase=phase, kind="qdq_numerics",
                                 exact=quantizes_input)
        measurement = native_operator_measurement(observed["measurement"], path=path, expected_sha256=expected_sha256)
        bound = resources["phases"][phase].get("bound")
        scratch = None
        if complete:
            scratch = native_operator_scratch(bound, phase=phase)
        # #578: the observed schedule rides the observation where a gate can
        # read it instead of living in prose. A price measured on an
        # older-architecture schedule is not a native-hardware price, so the
        # observation says so in its own unknown list; promoting that to a
        # refusal is a measured-delta decision for Rob, not this gate.
        schedule = observed_kernel_schedule(route)
        arch = kernel_arch_status(schedule, platform=route.get("platform"))
        unknowns = ["fixed_and_full_model_resources"] + ([] if complete else ["native_operator_scratch"])
        if arch == "older_arch":
            unknowns.append("native_kernel_schedule")
        observations[phase] = {"measurement": measurement.as_dict(), "median_ms": measurement.median_ms,
                               "peak_scratch_bytes": scratch, "resource_bound": bound,
                               "input_bytes": expected["input"]["logical_bytes"],
                               "output_bytes": expected["reference_output"]["logical_bytes"],
                               "kernel_schedule": schedule, "kernel_arch": arch}
    return {"schema": "prismaquant.native_dense_observation.v1", "status": "operator_evidence",
            "panel_sha256": identity_sha256(expected_panel), "receipt_sha256": expected_sha256,
            "unit": expected_panel["unit"], "format": expected_panel["format"],
            "cost_sha256": expected_panel["cost_sha256"], "phases": observations,
            "activation_scope": activation_scope,
            "serialized_unit_bytes": expected_panel["wire"]["blob_bytes"],
            "resident_bytes": _bytes(resources["resident_bytes"], "resident"),
            "full_model_resources": None, "runtime_table_admissible": False,
            "unknown": unknowns}


def require_panel_activation_attestation(panel, quantizes_input):
    """A panel that priced an activation residual carries what attested it.

    The freeze-time refusal and this one are the same mechanism at the two
    points an unattested oracle could get in: a panel produced before the
    attestation existed carries ``None`` here and is refused rather than
    admitted on the strength of having been frozen earlier.
    """
    stamp = panel.get("activation_quantizer_attestation")
    if not quantizes_input:
        if stamp is not None:
            raise ValueError(
                "native panel: a format that does not quantise its input "
                "carries a quantiser attestation; the panel and the cost row "
                "disagree about what was priced")
        return
    if not isinstance(stamp, dict) or stamp.get(
            "schema") != "prismaquant.activation_quantizer_attestation.v1":
        raise ValueError(
            "native panel: this panel priced an activation residual through "
            "PrismaQuant's own activation quantiser and carries no attestation "
            "that the runtime executes the same rounding rule. Re-freeze it "
            "against a Tessera contract that publishes activation_quantizers "
            "(RobTand/prismaquant#567); an unattested oracle is refused, not "
            "assumed (principle 14).")


_IMAGE_REFERENCE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")


def _image_digest(reference, side):
    """The content digest of one image reference, or a named refusal.

    Only a digest reference says which bytes ran: a tag moves.  Both sides of
    this comparison publish one -- the contract's ``generated.image`` and the
    receipt runtime's image -- so anything else is ``not verified`` rather
    than a comparison made on a name.
    """
    if not isinstance(reference, str) or not _IMAGE_REFERENCE.match(reference):
        raise ValueError(
            f"native panel: the {side} image is {reference!r}, which is not a "
            "digest reference (repository@sha256:<64 hex>). A tag names a "
            "moving target, so there is nothing here to compare and the "
            "panel's activation scope is NOT VERIFIED "
            "(RobTand/prismaquant#715).")
    return reference.split("@sha256:", 1)[1]


def require_panel_execution_scope(panel, *, executing_image):
    """The attested rounding rule and the executing image are one claim.

    The quantiser table is generated by RUNNING the runtime's kernel, and the
    contract says in which image
    (``activation_quantizers.platforms[p].generated.image``).  A cell measured
    in a different image executed a different build of the same operator, and
    on 2026-09-18 that is exactly what happened: seven ``TESSERA_E2M1_K2``
    cells ran in ``eugr/spark-vllm@sha256:0afec8d4…`` (vLLM
    ``0.28.1rc1.dev397+gfd4a15126``) carrying an attestation taken in
    ``vllm/vllm-openai@sha256:61fc8a89…`` (vLLM ``0.28.0``), and nothing
    refused them.  Principle 14's corollary: a capability claim inherits the
    scope of the artifact it was measured on.

    The witness for the executing side is ``panel["runtime"]["image"]``.  That
    is the image the operator was measured in, ``consume_native_receipt`` has
    already bound it to ``receipt["runtime"]``, and it is a registry digest --
    the same namespace the contract publishes.  ``served_quantizer
    .image_content_sha256`` (#712) is NOT the witness: it is a digest of the
    producer image's contents, in a different namespace from a registry
    digest, and it lives in the run's ``identity.json`` rather than in the
    panel this function is handed.

    Scope applies to a stamp that attests a rounding rule.  A dynamic-scale
    stamp (``activation_contract: null``) attests none -- both sides compute
    the same function of ``x`` and the stamp already records that it attests
    nothing -- so there is no claim to scope and this returns ``None``.  A
    stamp that DOES attest and carries no ``generated`` block is not verified,
    never a pass.  The comparison is identity: no tolerance, no allow-list, no
    judgement about whether two builds are close enough.

    What is compared is the image digest alone.  ``vllm`` and ``torch`` live
    inside the image, so the digest already pins them.  ``driver``, ``device``
    and ``compute_capability`` do NOT: they are host facts outside the image,
    the digest pins nothing about them, and this fixture is the proof -- the
    cell ran on driver ``595.91.07`` while the table was generated on
    ``595.84``.  Whether that difference can move the rounding rule is not
    established, so both sides of all three are RECORDED and neither is
    compared.  Starting to compare them is a decision with evidence behind it,
    not a default; the record is what makes it reversible.
    """
    stamp = panel.get("activation_quantizer_attestation")
    if not isinstance(stamp, dict) or stamp.get("activation_contract") is None:
        return None
    generated = stamp.get("generated")
    if not isinstance(generated, dict):
        raise ValueError(
            "native panel: this panel's activation attestation for "
            f"{stamp.get('activation_contract')!r} carries no `generated` "
            "scope, so the contract bytes it was read from do not say which "
            "image ran the kernel that produced the table. An attestation "
            "with no scope is NOT VERIFIED, not a pass: re-freeze against a "
            "contract that publishes activation_quantizers.platforms[...]"
            ".generated (RobTand/prismaquant#715). "
            f"Reason recorded by the producer: "
            f"{stamp.get('generated_absent_because')!r}")
    attested = _image_digest(generated.get("image"), "attested `generated.image`")
    executed = _image_digest(executing_image, "executing panel runtime")
    if attested != executed:
        raise ValueError(
            "native panel: the quantiser attestation was generated in a "
            "different image than the one this cell executed in, so it does "
            "not cover this measurement (RobTand/prismaquant#715).\n"
            f"  attested generated.image: sha256:{attested} "
            f"(vllm {generated.get('vllm')!r}, torch {generated.get('torch')!r}, "
            f"driver {generated.get('driver')!r})\n"
            f"  executing runtime image:  sha256:{executed} ({executing_image})\n"
            "Two builds of one operator are two objects (#567). Re-measure "
            "the cell in the attested image, or publish a table generated in "
            "the executing one; there is no tolerance and no allow-list here.")
    gpu = (panel.get("runtime") or {}).get("gpu") or {}
    return {"schema": "prismaquant.panel_execution_scope.v2",
            "activation_contract": stamp.get("activation_contract"),
            "attested_image": generated.get("image"),
            "executing_image": executing_image,
            "attested_vllm": generated.get("vllm"),
            "attested_torch": generated.get("torch"),
            "attested_driver": generated.get("driver"),
            "attested_device": generated.get("device"),
            "attested_compute_capability": generated.get("compute_capability"),
            # The executing side of the three HOST facts, as the panel recorded
            # them, verbatim: absent is None, never a placeholder that would
            # read as a reading nobody took.
            "executing_driver": gpu.get("driver_version"),
            "executing_device": gpu.get("name"),
            "executing_compute_capability": gpu.get("capability"),
            "compared": "image_content_digest",
            "not_compared": {
                "pinned_by_the_image_digest": ["vllm", "torch"],
                "host_facts_outside_the_image": ["driver", "device",
                                                 "compute_capability"]},
            "why_not_compared": {
                "pinned_by_the_image_digest":
                    "vllm and torch are inside the image, so the digest "
                    "already pins the bytes those version strings describe",
                "host_facts_outside_the_image":
                    "the driver, the device and its compute capability are "
                    "host facts outside the image, so the digest pins nothing "
                    "about them; by ruling (Rob, 2026-09-18) a driver change "
                    "never invalidates an attestation -- drivers move all "
                    "the time and nothing is revalidated for one -- so both "
                    "sides are recorded for the card and never compared"}}


_KERNEL_ARCH = re.compile(r"(?:sm_?|cutlass_?|cutlass3x_sm)(\d{2,3})", re.IGNORECASE)


def observed_kernel_schedule(route):
    """The executed kernel schedule a native route observation names, if any.

    RobTand/prismaquant#578: the BF16 route's sm80 CUTLASS schedule on sm_121
    lived only in a measurement doc, read by no gate, because the receipt
    route carries no schedule key and the consumer had no vocabulary for one.
    This is the vocabulary: the producer's ``kernel_schedule`` string when the
    observation names one, else ``None``. A present-but-empty schedule is a
    defect, not an absence.
    """
    schedule = (route or {}).get("kernel_schedule")
    if schedule is None:
        return None
    if not isinstance(schedule, str) or not schedule.strip():
        raise ValueError("native route kernel_schedule must be a nonempty string when present")
    return schedule


def kernel_arch_status(schedule, *, platform):
    """Whether an observed schedule is native to the platform it ran on.

    ``unknown`` is the honest default: no schedule observed, no platform
    declared, or a schedule that names no ``sm`` generation. ``older_arch`` is
    a schedule from an older generation than the platform (an sm80 schedule on
    sm_121); generations share their tens digit's family, so an sm_120-family
    schedule on sm_121 is ``native``, exactly as the fp4 sweep measured.
    """
    if schedule is None or not isinstance(platform, str):
        return "unknown"
    match = _KERNEL_ARCH.search(schedule)
    here = _KERNEL_ARCH.search(platform)
    if match is None or here is None:
        return "unknown"
    if int(match.group(1)) // 10 < int(here.group(1)) // 10:
        return "older_arch"
    return "native"


def validate_native_numerics(error, numerics, *, phase, kind, exact=False):
    """The shared frozen numerical gate for dense and whole routed operators.

    ``exact`` is the activation representation's gate and it has no tolerance.
    Given the same ``x``, the same ``G`` and a bit-identical stored scale, the
    two sides compute the same integer code; the honest per-element bound is
    zero.  The comparison is reported on dequantised bf16, but with a fixed
    scale the dequantisation is injective over the sixteen codes -- adjacent
    E2M1 magnitudes differ by at least 33% relative and bf16's spacing is
    2^-7, so distinct codes stay distinct through the cast -- which makes
    exact value agreement exact CODE agreement, up to the two zero codes that
    dequantise to the same number and that no GEMM can tell apart.

    So a nonzero ``max_abs_error`` here is a wrong activation, not a rounding
    epsilon, and there is no value of ``atol`` that separates the two.  On
    2026-09-13 the tolerance let ``v_proj`` decode through at 0.0078125 while
    ``o_proj`` (0.0957) and ``v_proj`` prefill (0.1436) failed -- three code
    flips, one verdict apart, decided by a constant derived from nothing.
    """
    if error.get("status") != "passed" or error.get("finite") is not True:
        raise ValueError(f"{phase}: refused native {kind} comparison")
    _equal({key: error[key] for key in ("atol", "rtol")}, numerics, f"{phase} tolerance")
    if _number(error["max_normalized_error"], "normalized numerical error") > 1:
        raise ValueError(f"{phase}: numerical error exceeds frozen tolerance")
    if exact and _number(error.get("max_abs_error"), f"{kind} absolute error") != 0.0:
        raise ValueError(
            f"{phase}: the runtime's {kind} differs from PrismaQuant's by "
            f"{error['max_abs_error']!r}. With a bit-identical stored scale "
            "that is at least one E2M1 code flipped, not a rounding "
            "difference: the priced activation residual and the executed one "
            "are different objects (RobTand/prismaquant#567, principle 8). "
            "Fix the oracle against the runtime's published quantiser table; "
            "widening a tolerance here admits a wrong activation.")


def native_operator_measurement(timing, *, path, expected_sha256):
    """Keep exact repeated whole-apply observations through the existing type."""
    if timing.get("sample_unit") != "single_apply":
        raise ValueError("native timing is not repeated individual operator invocations")
    return OperatorMeasurement.from_dict({key: timing[key] for key in
        ("method", "samples_ms", "warmup_iterations")} | {
            "receipt_path": str(path), "receipt_sha256": expected_sha256})


def native_operator_scratch(bound, *, phase):
    """Read the producer's conservative two-domain bound; retain its scope."""
    if (not isinstance(bound, dict) or bound.get("status") != "complete_operator_bound"
            or bound.get("full_model_fixed_resources_complete") is not False
            or bound.get("composition") != "sum_of_independent_peaks_including_output"):
        raise ValueError(f"{phase}: native resource scope/bound mismatch")
    native = _bytes(bound["external_native_peak_bytes"], "external native peak")
    torch_peak = _bytes(bound["torch_peak_increment_bytes"], "torch peak")
    scratch = _bytes(bound["peak_scratch_bytes"], "scratch")
    if scratch != native + torch_peak:
        raise ValueError(f"{phase}: resource peaks do not compose to the declared bound")
    return scratch
