"""Seal one Stage A checkpoint band from sealed sources (PQ #993).

Stage A seals a strided checkpoint (``checkpoints/boundary-b/checkpoint.json``)
hours before it writes its end-of-run receipt. A layer quantum reads only its
*slice* of Stage A (``joint_adjoint_checkpoints.stage_a_slice``): the run
header, one checkpoint and the forward boundary entries of its own chain. The
band for checkpoint ``b`` carries exactly what the layers served by ``b``
read, under the receipt's own field names, so ``stage_a_slice(band, L)``
equals ``stage_a_slice(receipt, L)`` byte for byte.

This tool is a read-only CPU action. It reads:

* the checkpoint's own manifest, verified against its self-seal and its
  entry files' sizes (the check ``load_adjoint_checkpoint`` makes against a
  receipt record);
* the run header from sealed sources only: the run's sealed PB request
  (plan, prepared and read-manifest digests, output root, stride, chain
  regime and the environment its Stage A process runs under), the plan and
  prepared documents under those digests, the boundary generation's
  ``generation.json``, and either the forward-recovery capsule the request
  binds or the run's sealed chain state (``chain-state.json``, PQ #1122);
* the band's forward boundary entries from the sources the receipt itself
  uses: the capsule chain for recovered boundaries, the generation's own
  entry files for boundaries the run wrote;
* the run's sealed chain resume records (``stage_a_chain_resume``): a band
  of a checkpoint sealed below a resume that switched implementations
  carries that declaration under ``resume_compatibility``, as the receipt
  does.

It never writes under the run's output root: the band goes to a path the
caller names outside it.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path

from .cost_stage_checkpoint import canonical_json_sha256
from .joint_adjoint_checkpoints import (
    ADJOINT_BAND_SCHEMA,
    ADJOINT_CAPTURE_ENTRY_POINT,
    ADJOINT_CHECKPOINT_SCHEMAS,
    CHAIN_REGIME_KEY,
    AdjointSliceRefused,
    ChainRegimeRefused,
    _CHECKPOINT_NAME,
    adjoint_slice_sha256,
    adjoint_space,
    band_layers,
    boundary_entry_directory,
    chain_regime_identity,
    checkpoint_cotangent_plane,
    checkpoint_directory,
    checkpoint_is_referenced,
    checkpoint_seal_sha256,
    derive_checkpoint_boundaries,
    exact_entry_record,
    normalize_chain_regime,
    reference_from_record,
    stage_a_run_header,
    stage_a_run_header_sha256,
    stage_a_slice,
    validate_band_receipt,
    write_band_receipt,
)
from .matmul_arithmetic import (
    BF16_REDUCTION_FIELD,
    MatmulArithmeticRefused,
    bf16_reduction_from_environment,
)
from .stage_a_chain_resume import (
    RESUME_COMPATIBILITY_KEY,
    ChainResumeRefused,
    chain_state_path,
    load_chain_state,
    resume_declarations,
)
from .stage_a_chain_seed import seed_marker_path, seed_receipt_path
from .digests import file_sha256hex

BAND_TOOL_ENTRY_POINT = "prismaquant.joint_adjoint_band"
BAND_RESULT_SCHEMA = "prismaquant.joint_adjoint_band.result.v1"
STAGE_A_MODULES = ("prismaquant.joint_adjoint_capture", "prismaquant.joint_cost_stage_a")
_ZERO_DIGEST = "0" * 64
_ABSENT = object()


class BandRefused(RuntimeError):
    """The sealed sources do not reconstruct one checkpoint band."""


_sha256_file = partial(file_sha256hex, block_size=1 << 20)


def _json_file(path, sha256=None, *, where: str):
    raw = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise BandRefused(f"{where} digest mismatch at {path}")
    return json.loads(raw)


# --------------------------------------------------------------------------
# The sealed checkpoint and the boundary generation
# --------------------------------------------------------------------------


def read_sealed_checkpoint(space: Path, boundary: int) -> tuple[dict, dict]:
    """The durable ``checkpoint.json`` for ``boundary``, self-seal verified.

    Returns the manifest record (the object Stage A's receipt lists under
    ``checkpoints``) and its file binding. Every entry must sit in the
    checkpoint's own ``entries`` directory with the size it declares; entry
    bytes stay the quantum reader's to verify.
    """
    directory = checkpoint_directory(space, boundary)
    path = directory / "checkpoint.json"
    raw = path.read_bytes()
    record = json.loads(raw)
    if (not isinstance(record, dict) or record.get("schema") not in ADJOINT_CHECKPOINT_SCHEMAS
            or record.get("boundary") != int(boundary)):
        raise BandRefused(f"{path} is not the sealed checkpoint of boundary {boundary}")
    if set(record) != {"schema", "boundary", "session", "activation_entries",
                       "shared_state_entries", "cotangent_sha256"}:
        raise BandRefused(f"{path} carries fields a sealed checkpoint does not")
    if record["cotangent_sha256"] != checkpoint_seal_sha256(record):
        raise BandRefused(f"{path} does not seal its own manifest")
    if record["session"].get("kind") != "adjoint_checkpoint":
        raise BandRefused(f"{path} is not an adjoint checkpoint session")
    expected_parent = directory / "entries"
    try:
        # PQ #1036: a referenced checkpoint's cotangent rows are the owner's
        # own entries at this boundary, in this generation's directory; the
        # helper holds them to that identity. Its shared states, like every
        # entry of a copied checkpoint, sit in the checkpoint's own entries.
        checkpoint_cotangent_plane(record)
    except ValueError as exc:
        raise BandRefused(f"{path}: {exc}") from exc
    own = (record["shared_state_entries"] if checkpoint_is_referenced(record)
           else record["activation_entries"] + record["shared_state_entries"])
    for entry in own:
        if Path(entry["path"]).parent != expected_parent:
            raise BandRefused(f"checkpoint entry {entry['name']} escaped {expected_parent}")
    for entry in record["activation_entries"] + record["shared_state_entries"]:
        entry_path = Path(entry["path"])
        try:
            size = entry_path.stat().st_size
        except OSError as exc:
            raise BandRefused(f"checkpoint entry {entry_path} is not durable") from exc
        if size != entry["file_bytes"]:
            raise BandRefused(f"checkpoint entry {entry_path} size differs from its manifest")
    return record, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def checkpoint_batches(record: dict) -> int:
    """The calibration partition count a checkpoint's cotangent plane covers."""
    try:
        coordinates = set(checkpoint_cotangent_plane(record))
    except ValueError as exc:
        raise BandRefused(str(exc)) from exc
    batches = sorted({batch for _, batch in coordinates})
    probes = sorted({probe for probe, _ in coordinates})
    if (not batches or batches != list(range(len(batches)))
            or len(coordinates) != len(batches) * len(probes)):
        raise BandRefused("checkpoint cotangent plane is not a whole probe x batch grid")
    return len(batches)


def run_tail_boundary(space: Path, session: dict) -> int:
    """The run's layer count: the highest checkpoint its generation sealed.

    Stage A serializes the tail checkpoint (boundary ``num_layers``) first and
    every later checkpoint below it, so once any checkpoint of a generation
    is durable its tail is too. Checkpoints of another generation (a stale
    run's) do not count; renamed-aside copies do not match the name.
    """
    marks = []
    root = Path(space) / "checkpoints"
    for path in root.iterdir():
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        manifest = path / "checkpoint.json"
        if match is None or not manifest.is_file():
            continue
        try:
            stored = json.loads(manifest.read_bytes())
        except ValueError:
            continue
        if stored.get("session") == session:
            marks.append(int(match[1]))
    if not marks:
        raise BandRefused(f"no sealed checkpoint of generation {session.get('generation')}")
    return max(marks)


def read_generation(space: Path, session: dict) -> tuple[dict, dict]:
    """``(session, policy)`` from the boundary generation's own status file."""
    path = boundary_entry_directory(space) / str(session["generation"]) / "generation.json"
    document = _json_file(path, where="boundary generation")
    expected = {"generation": session["generation"],
                "run_identity_sha256": session["run_identity_sha256"]}
    if document.get("session") != expected:
        raise BandRefused(f"{path} names another boundary session than the checkpoint")
    policy = document.get("policy")
    if not isinstance(policy, dict):
        raise BandRefused(f"{path} carries no boundary storage policy")
    return expected, policy


def _exact_metadata(path: Path) -> dict:
    """The writer's embedded exact metadata, without reading tensor bytes."""
    import torch
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    metadata = payload.get("exact") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise BandRefused(f"{path} carries no exact entry metadata")
    return metadata


def own_boundary_entry(directory: Path, session: dict, *, batch: int, boundary: int) -> dict:
    """The receipt record of a forward boundary entry this generation wrote."""
    from .perturbed_x_cache import EXACT_ACTIVATION_SCHEMA, activation_cache_filename
    name = f"boundary-{batch}-{boundary}-at-{boundary}"
    path = Path(directory) / str(session["generation"]) / "entries" / activation_cache_filename(name)
    metadata = _exact_metadata(path)
    identity = {"session": session, "slot": f"boundary-{batch}-{boundary}", "kind": "boundary",
                "coordinates": {"batch": batch, "boundary": boundary, "probe": None}}
    if (metadata.get("schema") != EXACT_ACTIVATION_SCHEMA or metadata.get("identity") != identity):
        raise BandRefused(f"{path} is not boundary {boundary} batch {batch} of this generation")
    record = {
        "name": name, "path": str(path), "sha256": _sha256_file(path),
        "tensor_bytes": int(metadata["tensor_bytes"]), "file_bytes": path.stat().st_size,
        "shape": [int(dim) for dim in metadata["shape"]], "dtype": str(metadata["dtype"]),
        "metadata": {key: metadata[key] for key in
                     ("schema", "identity", "shape", "dtype", "tensor_bytes")},
    }
    return record


# --------------------------------------------------------------------------
# The forward-recovery capsule
# --------------------------------------------------------------------------


def capsule_recovery(forward_recovery: dict | None, *, plan_sha256, prepared_sha256,
                     read_manifest_sha256):
    """``(bind_identity, receipt_binding, campaign, records)`` of a capsule.

    The bind identity is the capsule's original science with the recovering
    run's implementation, exactly what ``joint_forward_resume`` checks the
    recovering run's own identity against; the receipt binding is the block
    Stage A writes under ``boundary_storage.forward_recovery``.
    """
    if forward_recovery is None:
        return None
    from .joint_forward_resume import SCHEMA, _campaign, _read, chain_records
    path = str(forward_recovery["path"])
    document, digest = _read(path, forward_recovery["sha256"])
    if document.get("schema") != SCHEMA:
        raise BandRefused("the forward-recovery capsule has a foreign schema")
    compatibility = document["implementation_compatibility"]
    bind_identity = {**document["original_bind_identity"],
                     "producer_source_sha256": compatibility["recovery"]}
    campaign = _campaign(document)
    for field, expected in (("plan_sha256", plan_sha256), ("prepared_sha256", prepared_sha256),
                            ("read_manifest_sha256", read_manifest_sha256)):
        if campaign.get(field) != expected:
            raise BandRefused(f"the forward-recovery capsule answers for another {field}")
    receipt_binding = {
        "schema": SCHEMA, "capsule": {"path": path, "sha256": digest},
        "frontier": document["frontier"], "original_session": document["session"],
        "original_owner": document["instance"]["owner_action_key"],
        "original_attempt": document["instance"]["owner_attempt"]}
    records = chain_records(document)
    for rows in records.values():
        for row in rows:
            if exact_entry_record(reference_from_record(row)) != row:
                raise BandRefused("a recovered boundary record does not round-trip")
    return bind_identity, receipt_binding, campaign, records


# --------------------------------------------------------------------------
# The band
# --------------------------------------------------------------------------


def refuse_seed_space(space) -> None:
    """Refuse a Stage A seed run's space (RobTand/prismaquant#1016).

    A seed is a measurement that borrows another run's checkpoint; its
    checkpoints never feed Stage B.
    """
    held = [path for path in (seed_marker_path(space), seed_receipt_path(space))
            if path.exists()]
    if held:
        raise BandRefused(
            f"{held[0]} marks a Stage A seed run: a seed is a measurement, not a "
            "campaign run, and its checkpoints feed no band")


def build_band_receipt(*, output_root, boundary: int, plan_sha256: str, prepared_sha256: str,
                       read_manifest_sha256: str | None, stride_value: int,
                       stride_source, unit_roster_sha256: str | None = None,
                       campaign_scope=None, bind_identity: dict | None = None,
                       forward_recovery: dict | None = None,
                       sources: dict | None = None,
                       chain_regime: dict | None = None,
                       bf16_reduction: bool = True,
                       sealed_run_identity: dict | None = None) -> dict:
    """Reconstruct the band of checkpoint ``boundary`` from sealed sources.

    Without a forward-recovery capsule the caller supplies the run's bind
    identity with the unit roster and scope: from the run's sealed chain
    state when it has one (``sealed_chain_state``), else rebuilt by
    ``stage_a_bind_identity`` from the sealed plan, prepared completion and
    calibration input. With a capsule, all three come from it. Either way
    the bind identity must hash to the generation's ``run_identity_sha256``:
    a header that does not answer for the run that sealed the checkpoint
    refuses.

    ``chain_regime`` is the run's sealed ``--chain-batch-size`` and
    ``--chain-probe-fusion`` (RobTand/prismaquant#997); ``None`` is the
    default regime, which stamps nothing. ``bf16_reduction`` is the bf16
    reduced-precision reduction flag the run's environment pinned
    (PQ #1028): ``False`` stamps it, as Stage A's own run identity does.

    ``sealed_run_identity`` is the run header the run's chain state sealed
    (PQ #1122). The band's header must equal it field for field; a field
    that differs refuses, named.
    """
    boundary = int(boundary)
    try:
        regime_identity = chain_regime_identity(chain_regime or {})
    except ChainRegimeRefused as exc:
        raise BandRefused(str(exc)) from exc
    output_root = Path(output_root)
    space = adjoint_space(output_root)
    refuse_seed_space(space)
    from .stage_a_retirement import refuse_retired_space
    refuse_retired_space(space, BandRefused, what="its checkpoints feed no band")
    checkpoint, checkpoint_binding = read_sealed_checkpoint(space, boundary)
    marker = checkpoint["session"]
    session, policy = read_generation(space, {
        "generation": marker["generation"], "run_identity_sha256": marker["run_identity_sha256"]})
    read_manifest = read_manifest_sha256 or _ZERO_DIGEST
    recovery = capsule_recovery(forward_recovery, plan_sha256=plan_sha256,
                                prepared_sha256=prepared_sha256,
                                read_manifest_sha256=read_manifest)
    recovered: dict = {}
    receipt_binding = None
    if recovery is not None:
        capsule_bind, receipt_binding, campaign, recovered = recovery
        if bind_identity is not None and bind_identity != capsule_bind:
            raise BandRefused("the supplied bind identity differs from the capsule's")
        bind_identity = capsule_bind
        for field, supplied in (("unit_roster_sha256", unit_roster_sha256),
                                ("campaign_scope", campaign_scope)):
            if supplied is not None and supplied != campaign[field]:
                raise BandRefused(f"the supplied {field} differs from the capsule's")
        unit_roster_sha256 = campaign["unit_roster_sha256"]
        campaign_scope = campaign["campaign_scope"]
    if bind_identity is None or unit_roster_sha256 is None:
        raise BandRefused("a run without a forward-recovery capsule needs its sealed "
                          "bind identity and unit roster")
    if canonical_json_sha256(bind_identity, where="exact boundary source") != \
            session["run_identity_sha256"]:
        raise BandRefused("the reconstructed run identity does not hash to the "
                          "boundary generation's run_identity_sha256")

    num_layers = run_tail_boundary(space, marker)
    boundaries = derive_checkpoint_boundaries(num_layers, int(stride_value))
    if boundary not in boundaries:
        raise BandRefused(f"boundary {boundary} is not a stride-{stride_value} checkpoint "
                          f"of the {num_layers}-layer run {list(boundaries)}")
    layers = band_layers(boundaries, boundary)
    n_batches = checkpoint_batches(checkpoint)
    directory = boundary_entry_directory(space)
    entries = {}
    for k in sorted(layers):
        rows = recovered.get(str(k))
        if rows is not None:
            if len(rows) != n_batches:
                raise BandRefused(f"recovered boundary {k} covers {len(rows)} batches, "
                                  f"not {n_batches}")
            entries[str(k)] = rows
        else:
            entries[str(k)] = [own_boundary_entry(directory, session, batch=batch, boundary=k)
                               for batch in range(n_batches)]

    run_identity = {
        "plan_sha256": str(plan_sha256),
        "prepared_sha256": str(prepared_sha256),
        "read_manifest_sha256": str(read_manifest),
        "implementation_sha256": str(bind_identity["producer_source_sha256"]),
        "unit_roster_sha256": str(unit_roster_sha256),
        "campaign_scope": campaign_scope,
        "n_probes": int(bind_identity["n_probes"]),
        "seed_base": int(bind_identity["seed_base"]),
        "calibration_shape": list(bind_identity["calibration_shape"]),
        "calibration_sha256": bind_identity["calibration_sha256"],
        **({CHAIN_REGIME_KEY: regime_identity} if regime_identity is not None else {}),
        # Absent at PyTorch's default, as in Stage A's run identity (PQ #1028).
        **({} if bf16_reduction else {BF16_REDUCTION_FIELD: False}),
    }
    if sealed_run_identity is not None:
        differing = sorted(
            key for key in set(run_identity) | set(sealed_run_identity)
            if run_identity.get(key, _ABSENT) != sealed_run_identity.get(key, _ABSENT))
        if differing:
            raise BandRefused(
                "the band's run identity differs from the one the run's chain state "
                f"sealed in {', '.join(differing)}")

    band = {
        "schema": ADJOINT_BAND_SCHEMA,
        "entry_point": ADJOINT_CAPTURE_ENTRY_POINT,
        "status": "band",
        "band": {"boundary": boundary, "layers": list(layers)},
        "run_identity": run_identity,
        "stride": {"value": int(stride_value), "source": stride_source,
                   "boundaries": [int(b) for b in boundaries],
                   "max_chain_layers": int(stride_value) - 1},
        "boundary_storage": {
            "session": session,
            "policy": policy,
            "directory": str(directory),
            **({"forward_recovery": receipt_binding} if receipt_binding is not None else {}),
        },
        "boundary_entries": entries,
        "checkpoints": [checkpoint],
        "sources": {"tool": BAND_TOOL_ENTRY_POINT, "checkpoint": checkpoint_binding,
                    "generation": str(directory / session["generation"] / "generation.json"),
                    **(sources or {})},
    }
    # A checkpoint sealed below a chain resume that switched implementations
    # carries that declaration, outside the run header and every slice
    # (RobTand/prismaquant#1001). A band of any other checkpoint is unchanged.
    try:
        declarations = resume_declarations(space, session, below=boundary)
    except ChainResumeRefused as exc:
        raise BandRefused(str(exc)) from exc
    if declarations:
        band[RESUME_COMPATIBILITY_KEY] = declarations
    validate_band_receipt(band)
    return band


def stage_a_bind_identity(config: dict, prepared: dict) -> dict:
    """A non-recovered run's bind identity from its sealed plan and preparation.

    The same fields ``run_adjoint_capture_core`` binds: the prepared
    completion's source model and implementation, the plan's calibration
    input (loaded under its sealed digest) and probe execution.
    """
    from .calibration_data import load_calibration_input
    from .cost_streaming import validate_streamed_model_identity

    execution = config["execution"]
    ids, _ = load_calibration_input(
        config["calibration_input"]["path"],
        expected_sha256=config["calibration_input"]["sha256"],
        n_samples=execution["n_calib_samples"], seqlen=execution["calib_seqlen"])
    probe_microbatch = int(execution.get("probe_microbatch", 0))
    batch_rows = min(probe_microbatch or len(ids), len(ids))
    partition = None
    if probe_microbatch:
        partition = {
            "schema": "prismaquant.aura.streamed_microbatch.v1",
            "requested_rows": probe_microbatch, "effective_rows": batch_rows,
            "partition_count": len(range(0, len(ids), batch_rows)),
            "row_order": "contiguous_complete_sequences",
            "gradient_diagnostics": "sum_output_operators_fp32_before_norm"}
    return {
        "source_model": validate_streamed_model_identity(
            prepared["source_model_identity"], where="adjoint band"),
        "producer_source_sha256": prepared["implementation_sha256"],
        "calibration_sha256": hashlib.sha256(
            ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "calibration_shape": list(ids.shape),
        "calibration_dtype": str(ids.dtype),
        "n_probes": int(execution["n_probes"]), "seed_base": int(execution["seed_base"]),
        "token_scope": "all", "temperature": 1.0,
        "execution_partition": partition,
        "campaign_stage": "joint_adjoint_capture",
    }


# --------------------------------------------------------------------------
# The sealed request
# --------------------------------------------------------------------------


def stage_a_argv(command) -> dict:
    """The Stage A flags a sealed PB request carries, container wrapper aside."""
    starts = [index for index in range(len(command) - 1)
              if command[index] == "-m" and command[index + 1] in STAGE_A_MODULES]
    if len(starts) != 1:
        raise BandRefused("the sealed request runs no single Stage A entry point")
    tail = list(command[starts[0] + 2:])
    if "--chain-seed" in tail:
        raise BandRefused("the sealed request is a Stage A seed run: a seed is a "
                          "measurement, not a campaign run, and feeds no band")
    flags = {}
    for name in ("--plan", "--plan-sha256", "--prepared", "--prepared-sha256",
                 "--output-root", "--stride", "--read-manifest-sha256",
                 "--forward-recovery", "--forward-recovery-sha256",
                 "--chain-batch-size", "--chain-probe-fusion"):
        positions = [index for index, word in enumerate(tail) if word == name]
        if len(positions) > 1:
            raise BandRefused(f"the sealed request repeats {name}")
        if positions:
            flags[name] = tail[positions[0] + 1]
    for required in ("--plan", "--plan-sha256", "--prepared", "--prepared-sha256",
                     "--output-root"):
        if required not in flags:
            raise BandRefused(f"the sealed request carries no {required}")
    if ("--forward-recovery" in flags) != ("--forward-recovery-sha256" in flags):
        raise BandRefused("the sealed request pairs no forward-recovery digest")
    return flags


def request_chain_regime(flags) -> dict:
    """The chain regime a sealed request's flags run (RobTand/prismaquant#997).

    Absent flags are the Stage A CLI's defaults: batch size 1, fusion off.
    """
    size = flags.get("--chain-batch-size", "1")
    fusion = flags.get("--chain-probe-fusion", "off")
    if not size.isdecimal() or fusion not in ("on", "off"):
        raise BandRefused(
            f"the sealed request carries a malformed chain regime: "
            f"--chain-batch-size {size!r} --chain-probe-fusion {fusion!r}")
    try:
        return normalize_chain_regime(int(size), fusion == "on")
    except ChainRegimeRefused as exc:
        raise BandRefused(str(exc)) from exc


def request_environment(request) -> dict:
    """The environment a sealed request's Stage A process runs under.

    Behind the campaign container wrapper that is the container spec's
    ``env``: the wrapper forwards it into the container and never sets the
    Stage A settings itself (``tools/tessera_campaign_container.py``). A bare
    Stage A command runs under the request's own environment.
    """
    command = request["params"]["command"]
    start = next(index for index in range(len(command) - 1)
                 if command[index] == "-m" and command[index + 1] in STAGE_A_MODULES)
    wrapper = list(command[:start])
    if "--spec" in wrapper:
        try:
            spec = json.loads(wrapper[wrapper.index("--spec") + 1])
        except (IndexError, ValueError) as exc:
            raise BandRefused("the sealed request's container spec is not JSON") from exc
        env = spec.get("env") if isinstance(spec, dict) else None
        return dict(env) if isinstance(env, dict) else {}
    variables = (request.get("environment") or {}).get("variables") or {}
    return {**variables, **(request["params"].get("env") or {})}


def request_bf16_reduction(request) -> bool:
    """The bf16 reduction flag a sealed request pins (PQ #1028)."""
    try:
        return bf16_reduction_from_environment(request_environment(request))
    except MatmulArithmeticRefused as exc:
        raise BandRefused(f"the sealed request's environment: {exc}") from exc


def sealed_chain_state(space, *, plan_sha256: str, prepared_sha256: str,
                       read_manifest_sha256: str) -> tuple[dict | None, dict | None]:
    """The run's sealed chain state and its source, or ``(None, None)``.

    Stage A writes ``chain-state.json`` once, after its tail checkpoint, with
    the run identity and the bind identity it bound (PQ #1001): for a dev-mode
    run that is the *running* producer, which may differ from the prepared
    completion's (PQ #1122). The state must seal itself and name the plan,
    preparation and read manifest the request names; the caller then checks
    its bind identity against the boundary generation.
    """
    path = chain_state_path(space)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, None
    digest = hashlib.sha256(raw).hexdigest()
    try:
        state = load_chain_state(space, digest)
    except ChainResumeRefused as exc:
        raise BandRefused(str(exc)) from exc
    header = state["run_identity"]
    for field, requested in (("plan_sha256", plan_sha256),
                             ("prepared_sha256", prepared_sha256),
                             ("read_manifest_sha256", read_manifest_sha256)):
        if header.get(field) != requested:
            raise BandRefused(
                f"the run's chain state {path} seals {field} {header.get(field)}, "
                f"not the request's {requested}")
    return state, {"path": str(path), "sha256": digest}


def band_from_request(request_path, *, boundary: int, request_sha256: str | None = None) -> dict:
    """The band of ``boundary`` for the Stage A run a sealed PB request names."""
    from .joint_cost_stage_a import resolve_stride

    request_path = Path(request_path)
    raw = request_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if request_sha256 is not None and digest != request_sha256:
        raise BandRefused(f"sealed request digest mismatch at {request_path}")
    request = json.loads(raw)
    flags = stage_a_argv(request["params"]["command"])
    plan_sha256, prepared_sha256 = flags["--plan-sha256"], flags["--prepared-sha256"]
    config = _json_file(flags["--plan"], plan_sha256, where="Stage A plan")
    prepared = _json_file(flags["--prepared"], prepared_sha256, where="Stage A preparation")
    if Path(config["output_root"]).resolve() != Path(flags["--output-root"]).resolve():
        raise BandRefused("the plan output_root is not the request's --output-root")
    stride_value, stride_source = resolve_stride(
        config, int(flags["--stride"]) if "--stride" in flags else None)
    output_root = Path(flags["--output-root"])
    read_manifest = flags.get("--read-manifest-sha256")
    forward_recovery = None
    bind_identity = roster = state = state_source = None
    scope = config.get("campaign_scope")
    if "--forward-recovery" in flags:
        forward_recovery = {"path": flags["--forward-recovery"],
                            "sha256": flags["--forward-recovery-sha256"]}
        scope = None
    else:
        # The run's own sealed record of what it bound comes first (PQ #1122).
        # A seed or retired space refuses before anything in it is read.
        space = adjoint_space(output_root)
        refuse_seed_space(space)
        from .stage_a_retirement import refuse_retired_space
        refuse_retired_space(space, BandRefused, what="its checkpoints feed no band")
        state, state_source = sealed_chain_state(
            space, plan_sha256=plan_sha256, prepared_sha256=prepared_sha256,
            read_manifest_sha256=read_manifest or _ZERO_DIGEST)
    if state is not None:
        bind_identity = state["bind_identity"]
        roster = state["run_identity"]["unit_roster_sha256"]
        scope = state["run_identity"]["campaign_scope"]
    elif forward_recovery is None:
        # A run that sealed no chain state: rebuild the bind identity from the
        # plan and the prepared completion, as before.
        bind_identity = stage_a_bind_identity(config, prepared)
        roster = hashlib.sha256("".join(
            f"{name}\n" for name in sorted(prepared["formats_by_qname"])).encode()).hexdigest()
    return build_band_receipt(
        output_root=output_root, boundary=boundary,
        plan_sha256=plan_sha256, prepared_sha256=prepared_sha256,
        read_manifest_sha256=read_manifest,
        stride_value=stride_value, stride_source=stride_source,
        unit_roster_sha256=roster, campaign_scope=scope, bind_identity=bind_identity,
        forward_recovery=forward_recovery, chain_regime=request_chain_regime(flags),
        bf16_reduction=request_bf16_reduction(request),
        sealed_run_identity=None if state is None else state["run_identity"],
        sources={"request": {"path": str(request_path), "sha256": digest,
                             "action_key": request.get("action_key")},
                 "plan": {"path": flags["--plan"], "sha256": plan_sha256},
                 "prepared": {"path": flags["--prepared"], "sha256": prepared_sha256},
                 **({"chain_state": state_source} if state_source is not None else {})})


def band_summary(band: dict, *, path: Path, file_sha256: str) -> dict:
    """What the action reports: band digests and every served layer's slice digest."""
    boundary = band["band"]["boundary"]
    return {
        "schema": BAND_RESULT_SCHEMA, "boundary": boundary,
        "band": {"path": str(path), "sha256": file_sha256,
                 "canonical_sha256": canonical_json_sha256(band, where="band")},
        "run_header_sha256": stage_a_run_header_sha256(stage_a_run_header(band)),
        "layers": band["band"]["layers"],
        "slices": {str(layer): adjoint_slice_sha256(stage_a_slice(band, layer))
                   for layer in band["band"]["layers"]},
    }


def _refuse_inside(output: Path, output_root: Path) -> None:
    root = Path(os.path.realpath(output_root))
    target = Path(os.path.realpath(output.parent)) / output.name
    if target == root or root in target.parents:
        raise BandRefused(f"a band is never written under the run's output root {root}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--request", type=Path, required=True,
                        help="the Stage A run's sealed PB request (CAS requests/<key>.json)")
    parser.add_argument("--request-sha256", default=None)
    parser.add_argument("--boundary", type=int, required=True,
                        help="the sealed stride checkpoint to band")
    parser.add_argument("--output", type=Path, required=True,
                        help="the band path; never under the run's output root")
    args = parser.parse_args(argv)
    try:
        request = json.loads(args.request.read_bytes())
        output_root = Path(stage_a_argv(request["params"]["command"])["--output-root"])
        _refuse_inside(args.output, output_root)
        band = band_from_request(args.request, boundary=args.boundary,
                                 request_sha256=args.request_sha256)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        file_sha256 = write_band_receipt(args.output, band)
    except (BandRefused, AdjointSliceRefused, OSError, KeyError, ValueError) as exc:
        print(f"joint_adjoint_band: refused: {type(exc).__name__}: {exc}", flush=True)
        return 3
    print(json.dumps(band_summary(band, path=args.output, file_sha256=file_sha256),
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
