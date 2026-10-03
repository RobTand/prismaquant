"""Explicit geometry for one fresh Stage A row/probe (PQ #2010/#2008).

The full calibration tensor remains the input and identity. Only execution is
narrowed; Fisher noise keeps its global row and full-draw normalization. This
module owns no model, source, cache, autograd graph or qualification override.
Original CUDA material remains refused by the existing source owner.
"""
from __future__ import annotations

from pathlib import Path

from .cost_stage_checkpoint import publish_new_bytes
from .dev_mode import dev_mode_enabled
from .digests import DIRECT_ASCII_SPACED_STRICT, bytes_sha256hex, indent2_json_file_bytes
from .schemas import Contract, strict_json_loads
from .stage_inputs import read_bound

SPEC_SCHEMA = "prismaquant.stage_a.selected_row_diagnostic.v1"
RECEIPT_SCHEMA = "prismaquant.stage_a.selected_row_diagnostic_receipt.v1"
MARKER_NAME = "selected-row-diagnostic.json"
RECEIPT_NAME = "selected-row-diagnostic-receipt.json"
_FIELDS = frozenset({"schema", "calibration_shape", "calibration_dtype",
                     "calibration_tensor_sha256", "selected_global_row", "probe_seed",
                     "global_token_count", "vocab_size", "through"})


class SelectedRowDiagnosticRefused(ValueError):
    """The diagnostic's explicit coordinates or full draw do not agree."""


_contract = Contract(SelectedRowDiagnosticRefused, "selected-row diagnostic: ")


def normalize_diagnostic_spec(document) -> dict:
    """Closed, model-independent geometry; the enclosing plan binds authority."""
    _contract.require(dev_mode_enabled(), "selected-row diagnostic requires dev mode")
    _contract.exact_mapping(document, keys=_FIELDS, where="spec")
    _contract.require(document["schema"] == SPEC_SCHEMA, "diagnostic schema mismatch")
    shape = document["calibration_shape"]
    _contract.require(type(shape) is list and len(shape) == 2,
                      "calibration_shape must be [rows, sequence_length]")
    rows, tokens = [_contract.integer(v, where="calibration_shape", minimum=1) for v in shape]
    _contract.string(document["calibration_dtype"], where="calibration_dtype")
    _contract.sha256(document["calibration_tensor_sha256"], where="calibration_tensor_sha256")
    _contract.integer(document["selected_global_row"], where="selected_global_row",
                      minimum=0, maximum=rows - 1)
    for field in ("probe_seed", "through"):
        _contract.integer(document[field], where=field, minimum=0)
    _contract.integer(document["vocab_size"], where="vocab_size", minimum=1)
    _contract.integer(document["global_token_count"], where="global_token_count", minimum=1)
    _contract.require(document["global_token_count"] == rows * tokens,
                      "global_token_count must equal the full all-logit draw")
    return {**document, "calibration_shape": [rows, tokens]}


def load_diagnostic_spec(path, expected_sha256) -> dict:
    """Decode the same bound bytes that were authenticated; reject ambiguity."""
    _contract.sha256(expected_sha256, where="spec sha256")
    try:
        raw = read_bound({"path": str(path), "sha256": expected_sha256}, "selected-row diagnostic")
        return normalize_diagnostic_spec(strict_json_loads(
            raw, duplicate=lambda key: SelectedRowDiagnosticRefused(f"diagnostic duplicate key {key}"),
            constant=lambda key: SelectedRowDiagnosticRefused(f"diagnostic nonfinite constant {key}")))
    except (OSError, ValueError) as exc:
        raise SelectedRowDiagnosticRefused(f"selected-row diagnostic spec refused: {exc}") from exc


def bind_diagnostic_draw(spec, calib_ids, *, execution, num_layers) -> dict:
    """Validate before output creation; never treat the selected row as a draw."""
    from .stage_a_chain_seed import tensor_payload_sha256

    spec = normalize_diagnostic_spec(spec)
    _contract.require(list(calib_ids.shape) == spec["calibration_shape"],
                      "diagnostic full calibration shape differs")
    _contract.require(str(calib_ids.dtype) == spec["calibration_dtype"],
                      "diagnostic calibration dtype differs")
    _contract.require(tensor_payload_sha256(calib_ids) == spec["calibration_tensor_sha256"],
                      "diagnostic full calibration tensor differs")
    _contract.require(type(execution["n_probes"]) is int and execution["n_probes"] == 1,
                      "diagnostic needs exactly one probe")
    _contract.require(type(execution["seed_base"]) is int
                      and execution["seed_base"] == spec["probe_seed"], "diagnostic seed differs")
    _contract.require(type(execution.get("probe_microbatch")) is int
                      and execution["probe_microbatch"] == 1,
                      "diagnostic requires explicit probe_microbatch=1")
    _contract.require(execution.get("token_scope", "all") == "all"
                      and execution.get("temperature", 1.0) == 1.0,
                      "diagnostic requires all logits at temperature 1")
    _contract.require(spec["through"] < num_layers, "diagnostic through is outside the chain")
    return spec


def require_diagnostic_original_owner(owner, *, device, model=None) -> None:
    """Use the existing authority and device gate; never fabricate a provider."""
    from .tessera_calibration_cache import CaptureSourceAuthentication

    _contract.require(isinstance(owner, CaptureSourceAuthentication)
                      and owner.is_qualified_original_material,
                      "diagnostic requires the existing qualified original material owner")
    if model is not None:
        _contract.require(Path(model).resolve() == owner.root.resolve(),
                          "diagnostic model differs from the original owner's root")
    owner.require_material_device(device)


def diagnostic_marker_path(space) -> Path:
    return Path(space) / MARKER_NAME


def write_diagnostic_marker(space, spec) -> None:
    _contract.require(publish_new_bytes(diagnostic_marker_path(space), indent2_json_file_bytes(spec)),
                      "diagnostic root already holds a marker")


def write_diagnostic_receipt(space, receipt) -> dict:
    _contract.require(receipt.get("schema") == RECEIPT_SCHEMA
                      and diagnostic_marker_path(space).is_file(), "diagnostic receipt needs its marker")
    path = Path(space) / RECEIPT_NAME
    raw = indent2_json_file_bytes(receipt)
    _contract.require(publish_new_bytes(path, raw), "diagnostic receipt already exists")
    return {"path": str(path), "sha256": bytes_sha256hex(raw)}


SESSION_PREPARATION_SCHEMA = "prismaquant.original_diagnostic_session_preparation.v1"
SESSION_PREPARATION_NAME = "original-diagnostic-session-preparation.json"
SESSION_PREPARATION_OWNER = "original-diagnostic-prep"




def load_original_diagnostic_context(base_plan_input, static_authority_input):
    """Bind the render-free controls; this grants no source or CUDA admission."""
    from .cost_stage_checkpoint import canonical_json_sha256
    from .joint_adjoint_checkpoints import adjoint_space, boundary_entry_directory
    from .source_generation import (
        _control, _resources, normalize_original_diagnostic_base_plan,
        normalize_original_diagnostic_execution,
        normalize_original_diagnostic_preparation,
        normalize_original_source_static_authority,
        original_diagnostic_session_identity,
    )

    _, static = _control(static_authority_input, "original diagnostic static authority")
    static = normalize_original_source_static_authority(static)
    _, base = _control(base_plan_input, "original diagnostic base plan")
    base = normalize_original_diagnostic_base_plan(base)
    _, prepared = _control(base["prepared"], "original diagnostic preparation")
    prepared = normalize_original_diagnostic_preparation(prepared)
    static_sha256 = canonical_json_sha256(static, where="original static authority")
    _contract.require(base["static_authority_sha256"] == static_sha256,
                      "diagnostic base static authority differs")
    _contract.require(base["read_manifest"] == static["readset"],
                      "diagnostic static source manifest differs")
    _contract.require(prepared["resources"] == static["resources"],
                      "diagnostic prepared resource binding differs")
    for key in ("source_model_identity", "source_execution", "calibration"):
        _contract.require(prepared[key] == static[key], f"diagnostic prepared {key} differs")
    _contract.require(prepared["source_model_identity"]["source"] == base["model"],
                      "diagnostic source root differs")
    _, document = _control(base["execution"], "original diagnostic execution")
    document = normalize_original_diagnostic_execution(document)
    policy = document["boundary_storage"]
    _contract.require(policy["directory"] == str(boundary_entry_directory(adjoint_space(base["output_root"]))),
                      "original diagnostic boundary directory differs from its issued root")
    _, resources = _control(static["resources"], "original diagnostic resources")
    resources = _resources(resources)
    _contract.require(policy["max_artifact_bytes"] <= resources["artifact_bytes"],
                      "diagnostic artifact policy exceeds its bound envelope")
    _contract.require(policy["max_resident_bytes"] + policy["max_auxiliary_bytes"]
                      <= resources["cpu_bytes"],
                      "diagnostic resident and auxiliary policy exceeds CPU envelope")
    identity = original_diagnostic_session_identity(
        base_plan=base, base_plan_sha256=base_plan_input["sha256"], prepared=prepared,
        execution_sha256=base["execution"]["sha256"])
    return {"base_plan": base, "prepared": prepared, "static_authority": static,
            "execution": document, "resources": resources, "session_identity": identity,
            "base_plan_input": dict(base_plan_input),
            "static_authority_input": dict(static_authority_input)}


def load_original_diagnostic_issued_context(binding, *, base_plan_input, authority):
    """Join the issuer's bound static input to this independently selected authority."""
    from .source_generation import (
        _control, ORIGINAL_STATIC_AUTHORITY_KEYS, normalize_original_source_static_authority,
    )

    _, issued = _control(binding, "original diagnostic issued context")
    _contract.require(isinstance(issued, dict) and isinstance(issued.get("static_authority"), dict),
                      "diagnostic preparation lacks its independently bound static input")
    context = load_original_diagnostic_context(base_plan_input, issued["static_authority"])
    expected = normalize_original_source_static_authority(
        {key: authority[key] for key in ORIGINAL_STATIC_AUTHORITY_KEYS})
    _contract.require(context["static_authority"] == expected,
                      "issued diagnostic static tuple differs from selected full authority")
    receipt = read_original_diagnostic_session_preparation(
        binding, context=context, expected_session=authority["session"])
    return {**context, "session_preparation_input": dict(binding), "session_preparation": receipt}


def read_original_diagnostic_session_preparation(binding, *, context, expected_session):
    """Check this issued pending generation, never a foreign or resumed capture."""
    from .cost_stage_checkpoint import canonical_json_sha256
    from .cost_streaming import StreamedBoundaryArtifacts
    from .source_generation import _control, _session

    _, receipt = _control(binding, "original diagnostic session preparation")
    _contract.exact_mapping(receipt, keys={
        "schema", "status", "source_computation", "cuda_computation", "capture_complete",
        "source_admitted", "base_plan", "static_authority", "prepared", "execution",
        "session_identity", "session", "boundary_policy", "owner"},
        where="original diagnostic session preparation")
    _contract.require(receipt["schema"] == SESSION_PREPARATION_SCHEMA
                      and receipt["status"] == "session_prepared",
                      "diagnostic requires its actual control-preparation receipt")
    for key in ("source_computation", "cuda_computation", "capture_complete", "source_admitted"):
        _contract.require(receipt[key] is False, f"session preparation cannot claim {key}")
    base = context["base_plan"]
    for key, expected in (("base_plan", context["base_plan_input"]),
                          ("static_authority", context["static_authority_input"]),
                          ("prepared", base["prepared"]), ("execution", base["execution"]),
                          ("session_identity", context["session_identity"])):
        _contract.require(receipt[key] == expected, f"issued diagnostic {key} differs")
    session = _session(receipt["session"])
    _contract.require(session == _session(expected_session), "issued diagnostic session differs")
    _contract.require(session["run_identity_sha256"] == canonical_json_sha256(
        context["session_identity"], where="exact boundary source"),
        "issued diagnostic session identity differs")
    policy = context["execution"]["boundary_storage"]
    expected_policy = {key: value for key, value in policy.items() if key != "directory"}
    _contract.require(receipt["boundary_policy"] == expected_policy,
                      "issued diagnostic policy differs")
    directory = Path(policy["directory"]) / session["generation"]
    expected_owner = {"label": SESSION_PREPARATION_OWNER,
                      "path": str(directory / "owners" / (SESSION_PREPARATION_OWNER + ".json"))}
    _contract.require(receipt["owner"] == expected_owner, "issued diagnostic owner differs")
    owner_fields = {
        "phase": "original_diagnostic_session_preparation",
        "source_computation": False, "cuda_computation": False,
        "capture_complete": False, "source_admitted": False,
        "session_identity": context["session_identity"],
        "base_plan": context["base_plan_input"],
        "static_authority": context["static_authority_input"],
    }
    StreamedBoundaryArtifacts(policy).inspect_published_session(
        session, identity=context["session_identity"], pending=True,
        owner_label=SESSION_PREPARATION_OWNER, owner_fields=owner_fields)
    return receipt


def prepare_original_diagnostic_session(base_plan_input, static_authority_input):
    """Issue an actual pending artifact session, never a source/capture receipt."""
    from .aura_cost import _aura_source_sha256
    from .calibration_data import load_calibration_input
    from .cost_streaming import StreamedBoundaryArtifacts
    from .joint_adjoint_checkpoints import adjoint_space

    context = load_original_diagnostic_context(base_plan_input, static_authority_input)
    base, prepared = context["base_plan"], context["prepared"]
    _contract.require(prepared["implementation_sha256"] == _aura_source_sha256(),
                      "diagnostic preparation implementation differs from current capture code")
    ids, calibration = load_calibration_input(base["calibration_input"]["path"],
        expected_sha256=base["calibration_input"]["sha256"], n_samples=512, seqlen=512)
    _contract.require(calibration == prepared["calibration"],
                      "diagnostic decoded full calibration differs from prepared context")
    del ids
    space = adjoint_space(base["output_root"])
    receipt_path = space / SESSION_PREPARATION_NAME
    _contract.require(not space.exists(), "diagnostic session root already exists")
    policy = context["execution"]["boundary_storage"]
    with StreamedBoundaryArtifacts(policy) as storage:
        storage.bind(context["session_identity"], n_probes=1, published=True,
                     owner_label=SESSION_PREPARATION_OWNER)
        storage.stamp_owner(phase="original_diagnostic_session_preparation",
            source_computation=False, cuda_computation=False, capture_complete=False,
            source_admitted=False, session_identity=context["session_identity"],
            base_plan=base_plan_input, static_authority=static_authority_input)
        receipt = {
            "schema": SESSION_PREPARATION_SCHEMA, "status": "session_prepared",
            "source_computation": False, "cuda_computation": False,
            "capture_complete": False, "source_admitted": False,
            "base_plan": dict(base_plan_input), "static_authority": dict(static_authority_input),
            "prepared": base["prepared"], "execution": base["execution"],
            "session_identity": context["session_identity"], "session": dict(storage.session),
            "boundary_policy": storage.identity,
            "owner": {"label": SESSION_PREPARATION_OWNER, "path": str(storage.status_path())},
        }
        raw = indent2_json_file_bytes(receipt)
        _contract.require(publish_new_bytes(receipt_path, raw),
                          "diagnostic session preparation receipt already exists")
    return {"path": str(receipt_path), "sha256": bytes_sha256hex(raw), "receipt": receipt}


def main(argv=None):
    """The explicit PB CPU metadata issuance command; it never runs a model."""
    import argparse
    from .stage_b_prep_io import bind_staged_reads

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-original-session", action="store_true", required=True)
    parser.add_argument("--base-plan", required=True)
    parser.add_argument("--base-plan-sha256", required=True)
    parser.add_argument("--static-authority", required=True)
    parser.add_argument("--static-authority-sha256", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--allowed-tiers", choices=("ram", "ssd", "ram,ssd"), required=True)
    args = parser.parse_args(argv)
    bind_staged_reads(manifest_sha256=args.data_manifest_sha256,
                      allowed_tiers=args.allowed_tiers)
    result = prepare_original_diagnostic_session(
        {"path": args.base_plan, "sha256": args.base_plan_sha256},
        {"path": args.static_authority, "sha256": args.static_authority_sha256})
    print(DIRECT_ASCII_SPACED_STRICT.text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
