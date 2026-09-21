"""Stage A of the distributed joint-AURA cost campaign: the adjoint capture.

``python3 -m prismaquant.joint_cost_stage_a`` -- the campaign's one
sequential consumer (contract ``docs/design/distributed_campaign_2026-09-19.md``
§2.2 stage A, §3.3): the single run's forward boundary capture and tail
cotangents, then a render-free cotangent chain 44->0 that publishes a strided
checkpoint every S layers and nothing else. No renders are read, no cost rows
are produced, no operator windows replay. Its receipt
(``<output_root>/layer-quanta/adjoint/adjoint-capture.json``, sealed
atomically, first writer wins) is what gates the dispatcher's quantum
publication; quanta bind it by digest and refuse a moved or mismatched one.

The plan block's entry name ``prismaquant.joint_adjoint_capture`` (§5.1) is
this module's lane alias; both spellings execute here.

The plan's ``source_prefetch`` budget is the capture's IO contract, and a
frozen plan can still be wrong about it (#819: the sealed single-worker pin
starves the GPU). ``--prefetch-override`` / ``PRISMAQUANT_STAGE_A_PREFETCH_OVERRIDE``
name an explicit override document validated by the plan's own field grammar;
it replaces the budget for one run only and stamps the deviation into the
run's provenance (``results.json``, ``counters.json``) -- the IO-side #809
seam: explicit input beside the sealed plan, recorded, never silent.

The plan's ``boundary_storage.max_artifact_bytes`` is the capture's durable
contract, and a frozen plan can still be wrong about it (#882: the sealed
416 GiB ceiling cannot hold the ~584 GiB retained peak).
``--artifact-budget-bytes`` / ``PRISMAQUANT_STAGE_A_ARTIFACT_BUDGET_BYTES``
name an explicit byte count validated as a strict positive integer; it
replaces the ceiling for one run only (through the core's existing
``boundary_artifact_bytes``) and stamps the deviation into the run's
provenance (``results.json``, ``counters.json``, the adjoint receipt) --
the durable-side #809 seam. An under-budget invocation refuses in an early
preflight from live geometry, before any expensive forward; the runtime
reserve / write / commit guards stay authoritative for serialized bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import time
from pathlib import Path

import torch

from .cost_stage_checkpoint import atomic_write_bytes, canonical_json_sha256
from .joint_adjoint_checkpoints import (
    ADJOINT_CAPTURE_ENTRY_POINT,
    ADJOINT_RECEIPT_SCHEMA,
    DEFAULT_STRIDE,
    QUANTUM_COUNTERS_SCHEMA,
    GpuPowerSampler,
    KernelTimeProfiler,
    adjoint_space,
    boundary_entry_directory,
    chain_layers_for,
    derive_checkpoint_boundaries,
    dev_mode_stamp,
    exact_entry_record,
    render_free_layer_roll,
    require_dev_mode,
    write_adjoint_checkpoint,
    write_adjoint_receipt,
)
from .joint_layer_quanta import (
    ADJOINT_TAIL_PHASE,
    adjoint_chain_phase_name,
    adjoint_forward_phase_name,
)


def stage_a_forward_observer(progress):
    """Map the runner's source-phase callbacks onto read-plan phases.

    The visitor reports ``source_loading`` (prefetch/speculation starts)
    and ``capture_forward`` (installed, computing) per layer, ascending;
    both name the layer being read, so both map to that layer's forward
    phase. Anything else is ignored: never invent a phase, never advance
    on an unknown stage, and with no progress channel observe nothing.
    """
    def observe(stage, layer, _auxiliary_bytes):
        if progress is None:
            return
        if stage in ("source_loading", "capture_forward"):
            progress.enter(adjoint_forward_phase_name(int(layer)))
            progress.flush(force=True)
    return observe

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_IDENTITY_REFUSED = 3

#: The durable-side explicit-input seam (the #809 pattern, second budget):
#: a frozen plan's sealed ``boundary_storage.max_artifact_bytes`` stays the
#: campaign's identity, and one run may replace it with an explicitly
#: recorded byte count -- stamped into the run's result, counters and
#: receipt provenance, never silent, never a default. The PB channel is the
#: CLI flag (the container launcher forwards no ambient action environment
#: into the payload); the environment variable serves direct invocations,
#: and two explicit sources that disagree refuse. Units are bytes, always.
ARTIFACT_BUDGET_ENV = "PRISMAQUANT_STAGE_A_ARTIFACT_BUDGET_BYTES"
ARTIFACT_BUDGET_STAMP_SCHEMA = (
    "prismaquant.joint_adjoint_capture.artifact_budget.v1")
#: The exact-entry writer's small PyTorch zip header envelope
#: (``cost_streaming.StreamedBoundaryArtifacts.write``: ``nbytes + 65536``),
#: mirrored here so the preflight bounds the same ceiling the runtime
#: reserve enforces. The runtime guard stays authoritative for serialized
#: bytes and unpredicted overhead; this is only the preflight's bound.
ARTIFACT_FILE_HEADER_BYTES = 65536

#: The IO-side explicit-input seam (the #809 pattern): a frozen plan's sealed
#: ``source_prefetch`` budget stays the campaign's identity, and one run may
#: replace it with an explicitly recorded override document -- stamped into
#: the run's result and counter provenance, never silent, never a default.
#: The PB channel is the CLI flag (the container launcher forwards no ambient
#: action environment into the payload); the environment variable serves
#: direct invocations, and two explicit sources that disagree refuse.
PREFETCH_OVERRIDE_ENV = "PRISMAQUANT_STAGE_A_PREFETCH_OVERRIDE"
PREFETCH_OVERRIDE_INPUT_SCHEMA = (
    "prismaquant.joint_adjoint_capture.prefetch_override_input.v1")
PREFETCH_OVERRIDE_STAMP_SCHEMA = (
    "prismaquant.joint_adjoint_capture.prefetch_override.v1")


class AdjointIdentityRefused(RuntimeError):
    """A plan/prepared digest or binding mismatch, before anything runs."""


def shared_adjoint_snapshot(cotangents) -> dict:
    """Borrowed whole-plane shared-adjoint snapshot, no bulk CPU copies.

    Maps ``(probe, batch)`` to each live owner's
    :meth:`SharedStateCotangents.borrowed_state_dict`. Values borrow live
    accumulator storages on the production CPU-contiguous path, so the
    whole-plane snapshot adds no new tensor backing while the watched
    originals stay live. Any owner needing CPU pinning/contiguity refuses
    BEFORE copying -- the caller must take the owner-budgeted fallback
    (:func:`shared_adjoint_copy_plan` + hold, then ``state_dict`` copies
    inside that hold). The caller must drop the snapshot before the next
    harvest and must not mutate owners while ``write_adjoint_checkpoint``
    serializes it. This is the actual Stage-A checkpoint snapshot boundary.
    """
    return {(probe, batch): cotangents[probe][batch].borrowed_state_dict()
            for probe in range(len(cotangents))
            for batch in range(len(cotangents[probe]))}


def shared_adjoint_copy_plan(cotangents) -> tuple[bool, int]:
    """Whether the snapshot needs an owner-budgeted CPU copy, and how many bytes.

    No allocation: sums each owner's :meth:`snapshot_copy_plan` without
    materializing any tensor. Returns ``(needs_copy, copy_bytes)`` where
    ``copy_bytes`` covers whole-owner copies for owners needing pinning
    (``state_dict`` copies whole owners, so the hold covers whole owners,
    not just exceptional tensors). Meta/non-strided refuse (TypeError)
    like the checkpoint estimator. Stage A holds ``copy_bytes`` across the
    synchronous snapshot + write lifetime before building any mapping.
    """
    needs = False
    total = 0
    for row in cotangents:
        for owner in row:
            owner_needs, owner_bytes = owner.snapshot_copy_plan()
            needs = needs or owner_needs
            total += owner_bytes
    return (bool(needs), int(total))


def write_checkpoint_with_snapshot(storage, space, *, boundary, session, plane,
                                   cotangents, shared_pass) -> dict:
    """Snapshot + hold + writer lifetime the actual Stage-A caller uses.

    Zero-copy borrowed snapshot when every accumulator is already CPU
    contiguous; otherwise hold the precomputed exceptional-owner copy bytes
    BEFORE any ``state_dict`` materialization, copy only exceptional owners
    inside that hold while borrowing contiguous owners, keep the hold
    across the synchronous ``write_adjoint_checkpoint``, and release copies
    before releasing the hold (snapshot cleared inside the hold). Returns
    the checkpoint record. Meta/non-strided/quiescence fail closed before
    any hold or copy. ``serialize_checkpoint`` and the budget regressions
    call this one operation -- no test-only execution path.
    """
    from .joint_adjoint_checkpoints import write_adjoint_checkpoint

    needs_copy, copy_bytes = shared_adjoint_copy_plan(cotangents)
    if not needs_copy:
        snapshot = shared_adjoint_snapshot(cotangents)
        try:
            return write_adjoint_checkpoint(
                space, boundary=boundary, session=session,
                cotangents=plane, shared_adjoint=snapshot,
                shared_pass=shared_pass, owner=storage)
        finally:
            snapshot.clear()
    with storage.hold_transient_metadata(copy_bytes, "shared-adjoint CPU snapshot"):
        snapshot = {}
        try:
            for probe in range(len(cotangents)):
                for batch in range(len(cotangents[probe])):
                    owner = cotangents[probe][batch]
                    owner_needs, _ = owner.snapshot_copy_plan()
                    snapshot[(probe, batch)] = (
                        owner.state_dict() if owner_needs else owner.borrowed_state_dict())
            return write_adjoint_checkpoint(
                space, boundary=boundary, session=session,
                cotangents=plane, shared_adjoint=snapshot,
                shared_pass=shared_pass, owner=storage)
        finally:
            snapshot.clear()


def resolve_stride(config, cli_stride) -> tuple[int, str]:
    """S comes from the plan's ``distributed_campaign`` block when present
    (§3.4: a plan knob, not a CLI guess); a CLI value may supply it only for
    a plan that does not declare the block, and a disagreement refuses."""
    plan_block = (config.get("distributed_campaign") or {})
    plan_stride = plan_block.get("cotangent_checkpoint_stride")
    if plan_stride is not None:
        if cli_stride is not None and int(cli_stride) != int(plan_stride):
            raise AdjointIdentityRefused(
                f"--stride {cli_stride} disagrees with the plan's sealed "
                f"stride {plan_stride}")
        return int(plan_stride), "plan"
    return int(cli_stride if cli_stride is not None else DEFAULT_STRIDE), "cli"


def load_prefetch_override(path) -> dict:
    """Load and grammar-check one prefetch-override document.

    The ``source_prefetch`` block passes the plan's own completeness check
    (:func:`prismaquant.tessera_joint_aura._source_prefetch`): the same six
    fields, the same positivity/finite/lookahead rules, prefetched residency
    still required. The document additionally carries a non-empty ``reason``
    -- an override without a recorded reason is silent by construction, and
    the deviation stamp quotes it verbatim.
    """
    from .tessera_joint_aura import _source_prefetch

    path = Path(path)
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"prefetch override {path}: unreadable JSON: {exc}") from exc
    if (not isinstance(document, dict)
            or set(document) != {"schema", "reason", "source_prefetch"}):
        raise ValueError(f"prefetch override {path}: exactly schema, reason "
                         "and source_prefetch required")
    if document["schema"] != PREFETCH_OVERRIDE_INPUT_SCHEMA:
        raise ValueError(f"prefetch override {path}: schema is not "
                         f"{PREFETCH_OVERRIDE_INPUT_SCHEMA!r}")
    reason = document["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"prefetch override {path}: a non-empty reason is "
                         "required -- an unexplained override is a silent one")
    return {"reason": reason, "source_prefetch": _source_prefetch(document),
            "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def resolve_prefetch_override(config, cli_path=None, environ=None) -> dict:
    """Resolve the run's source-prefetch budget beside the sealed plan.

    The plan's block is validated exactly as before and remains the
    campaign's identity -- nothing here rewrites plan bytes or the digests
    that bind them. An explicit override (the CLI flag, or
    ``PREFETCH_OVERRIDE_ENV`` for a direct invocation) replaces the budget
    **for this run only**, and the return carries the deviation stamp both
    provenance files quote verbatim. Two explicit sources that disagree
    refuse (the #809 rule); neither source present is the plan's budget
    verbatim; there is no default override and no silent one.
    """
    from .tessera_joint_aura import _source_prefetch

    environ = os.environ if environ is None else environ
    plan_budget = _source_prefetch(config)
    cli_path = Path(cli_path) if cli_path is not None else None
    env_raw = str(environ.get(PREFETCH_OVERRIDE_ENV, "") or "").strip()
    env_path = Path(env_raw) if env_raw else None
    if cli_path is not None and env_path is not None and cli_path != env_path:
        raise AdjointIdentityRefused(
            f"--prefetch-override {cli_path} disagrees with "
            f"{PREFETCH_OVERRIDE_ENV}={env_path}")
    source, path = (("cli", cli_path) if cli_path is not None
                    else ("env", env_path) if env_path is not None
                    else (None, None))
    if path is None:
        return {"run_used": plan_budget, "override": None}
    override = load_prefetch_override(path)
    stamp = {
        "schema": PREFETCH_OVERRIDE_STAMP_SCHEMA,
        "source": source,
        "path": override["path"],
        "sha256": override["sha256"],
        "reason": override["reason"],
        "plan_sealed": plan_budget,
        "run_used": override["source_prefetch"],
    }
    return {"run_used": override["source_prefetch"], "override": stamp}


def _parse_artifact_budget_bytes(value, *, where: str) -> int:
    """Strict positive-integer byte count; bools, floats and non-decimals refuse.

    Units are bytes, always. A bool is an int subclass but never a byte
    budget; a float string (``"1e9"``, ``"640.0"``) is never an exact count.
    Strings must be ASCII decimal: ``str.isdigit`` also accepts non-ASCII
    digits (superscripts, fullwidth) that ``int()`` then refuses with an
    untyped error, so the ASCII check runs first and every invalid input
    carries the named refusal.
    """
    if isinstance(value, bool):
        raise AdjointIdentityRefused(
            f"{where} artifact budget must be a positive integer byte count, "
            f"got bool {value!r}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or any(ch not in "0123456789" for ch in text):
            raise AdjointIdentityRefused(
                f"{where} artifact budget must be a positive integer byte "
                f"count (ASCII decimal bytes), got {value!r}")
        parsed = int(text, 10)
    else:
        raise AdjointIdentityRefused(
            f"{where} artifact budget must be a positive integer byte count "
            f"(bytes), got {type(value).__name__} {value!r}")
    if parsed <= 0:
        raise AdjointIdentityRefused(
            f"{where} artifact budget must be a positive integer byte count "
            f"(bytes), got {parsed!r}")
    return int(parsed)


def _plan_sealed_artifact_bytes(config) -> int:
    """The sealed plan's durable ceiling, validated strictly, never rewritten."""
    try:
        block = config["execution"]["boundary_storage"]
        sealed = block["max_artifact_bytes"]
    except (KeyError, TypeError) as exc:
        raise AdjointIdentityRefused(
            "plan execution.boundary_storage.max_artifact_bytes is missing: "
            f"{exc}") from exc
    if type(sealed) is not int or sealed <= 0:
        raise AdjointIdentityRefused(
            "plan execution.boundary_storage.max_artifact_bytes must be a "
            f"positive integer byte count (bytes), got {sealed!r}")
    return int(sealed)


def resolve_artifact_budget_override(config, cli_value=None, environ=None) -> dict:
    """Resolve the run's durable artifact ceiling beside the sealed plan.

    The plan's ``max_artifact_bytes`` is validated exactly as before and
    remains the campaign's identity -- nothing here rewrites plan bytes or
    the digests that bind them. An explicit override (the CLI flag, or
    ``ARTIFACT_BUDGET_ENV`` for a direct invocation) replaces the ceiling
    **for this run only**, and the return carries the deviation stamp the
    run's result, counters and receipt quote verbatim. Two explicit sources
    that disagree refuse (the #809 rule); neither source present is the
    plan's budget verbatim; there is no default override and no silent one.
    """
    environ = os.environ if environ is None else environ
    plan_sealed = _plan_sealed_artifact_bytes(config)
    cli_parsed = (None if cli_value is None
                  else _parse_artifact_budget_bytes(cli_value, where="--artifact-budget-bytes"))
    env_raw = str(environ.get(ARTIFACT_BUDGET_ENV, "") or "").strip()
    env_parsed = (None if not env_raw
                  else _parse_artifact_budget_bytes(env_raw, where=ARTIFACT_BUDGET_ENV))
    if cli_parsed is not None and env_parsed is not None and cli_parsed != env_parsed:
        raise AdjointIdentityRefused(
            f"--artifact-budget-bytes {cli_parsed} disagrees with "
            f"{ARTIFACT_BUDGET_ENV}={env_parsed}")
    source, run_used = ((("cli", cli_parsed) if cli_parsed is not None
                         else ("env", env_parsed) if env_parsed is not None
                         else (None, plan_sealed)))
    if source is None:
        return {"run_used": int(plan_sealed), "override": None}
    stamp = {
        "schema": ARTIFACT_BUDGET_STAMP_SCHEMA,
        "source": source,
        "unit": "bytes",
        "plan_sealed_bytes": int(plan_sealed),
        "run_used_bytes": int(run_used),
    }
    return {"run_used": int(run_used), "override": stamp}


def estimate_stage_a_artifact_demand(
    *, n_probes: int, n_full_batches: int, remainder_rows: int,
    per_full_tensor_nbytes: int, per_remainder_tensor_nbytes: int | None = None,
    num_layers: int, stride: int,
    header_bytes: int = ARTIFACT_FILE_HEADER_BYTES,
    shared_per_checkpoint_bytes: int = 0,
    manifest_per_checkpoint_bytes: int = 0,
) -> dict:
    """Bound Stage A durable demand from geometry alone; allocates nothing.

    Counts files the actual caller retains, never a second execution path:
    ``num_layers`` retained INPUT boundary groups (46 written, the tail
    retires the final boundary, so ``num_layers`` stay live), one live
    cotangent plane set (``n_probes`` probe planes x batches entries -- four
    probe planes of 512 entries each on the 4-probe production panel; roll
    replaces the previous), and one checkpoint copy of that plane set per
    strided boundary (``derive_checkpoint_boundaries``).

    Batches are counted exactly: ``n_full_batches`` full batches plus one
    remainder batch of ``remainder_rows`` rows when nonzero. A remainder
    batch's tensor is smaller than a full one, so ``ceil(n_rows /
    batch_rows)`` full-size tensors would be an UPPER estimate -- the hard
    floor below sums full groups plus the true remainder geometry
    (``_stage_a_per_tensor_nbytes`` derives both through the profile's own
    expansion on ``meta`` tensors).

    Returns ``lower_bound_bytes`` -- the hard floor of true raw tensor
    bytes, before headers / shared / manifest -- which the preflight
    enforces as a mandatory early refusal; and ``planning_estimate_bytes``
    -- an explicitly named planning allowance, NOT a proven upper bound
    (per-file ``header_bytes`` envelope, plus the caller-supplied
    per-checkpoint shared and manifest allowances). The shared allowance
    reuses the plan's ``max_auxiliary_bytes`` as a stated assumption:
    deduplicated live storages can serialize into separate retained files,
    so it does not prove a ceiling. The runtime reserve / write / commit
    guards stay authoritative for serialized bytes and unpredicted
    overhead. Both are ints.
    """
    for name, value in (("n_probes", n_probes),
                        ("per_full_tensor_nbytes", per_full_tensor_nbytes),
                        ("num_layers", num_layers), ("stride", stride)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"stage A artifact demand needs positive {name}")
    for name, value in (("n_full_batches", n_full_batches),
                        ("remainder_rows", remainder_rows)):
        if type(value) is not int or value < 0:
            raise ValueError(f"stage A artifact demand needs nonnegative {name}")
    if n_full_batches == 0 and remainder_rows == 0:
        raise ValueError("stage A artifact demand needs at least one batch")
    if remainder_rows > 0 and (type(per_remainder_tensor_nbytes) is not int
                               or per_remainder_tensor_nbytes <= 0):
        raise ValueError("stage A artifact demand needs a positive "
                         "per_remainder_tensor_nbytes for a remainder batch")
    for name, value in (("header_bytes", header_bytes),
                        ("shared_per_checkpoint_bytes", shared_per_checkpoint_bytes),
                        ("manifest_per_checkpoint_bytes",
                         manifest_per_checkpoint_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError(f"stage A artifact demand needs nonnegative {name}")
    boundaries = derive_checkpoint_boundaries(int(num_layers), int(stride))
    n_checkpoints = len(boundaries)
    n_retained_groups = int(num_layers)
    has_remainder = int(remainder_rows) > 0
    n_batches_total = int(n_full_batches) + (1 if has_remainder else 0)
    per_full = int(per_full_tensor_nbytes)
    per_rem = int(per_remainder_tensor_nbytes) if has_remainder else 0
    # True raw sum per boundary group / probe plane set: full groups plus
    # the remainder batch at its own smaller size.
    per_group_raw = int(n_full_batches) * per_full + per_rem
    lower = ((n_retained_groups + int(n_probes)
              + n_checkpoints * int(n_probes)) * per_group_raw)
    full_envelope = per_full + int(header_bytes)
    rem_envelope = (per_rem + int(header_bytes)) if has_remainder else 0
    unit_sets = n_retained_groups + int(n_probes) + n_checkpoints * int(n_probes)
    planning = (unit_sets * int(n_full_batches) * full_envelope
                + (unit_sets * rem_envelope if has_remainder else 0)
                + n_checkpoints * int(shared_per_checkpoint_bytes)
                + n_checkpoints * int(manifest_per_checkpoint_bytes)
                + max(full_envelope, rem_envelope))
    return {
        "boundaries": [int(b) for b in boundaries],
        "n_checkpoints": int(n_checkpoints),
        "n_retained_boundary_groups": int(n_retained_groups),
        "n_full_batches": int(n_full_batches),
        "remainder_rows": int(remainder_rows),
        "n_batches_total": int(n_batches_total),
        "n_probes": int(n_probes),
        "per_full_tensor_nbytes": per_full,
        "per_remainder_tensor_nbytes": (per_rem if has_remainder else 0),
        "per_full_file_envelope_bytes": int(full_envelope),
        "lower_bound_bytes": int(lower),
        "planning_estimate_bytes": int(planning),
        "shared_per_checkpoint_bytes": int(shared_per_checkpoint_bytes),
        "manifest_per_checkpoint_bytes": int(manifest_per_checkpoint_bytes),
    }


def _stage_a_per_tensor_nbytes(runner, *, batch_rows: int, seqlen: int) -> int:
    """Per-boundary tensor bytes from live geometry; allocates no bulk bytes.

    Reads the embedding width off the live model (``base_model.embed_tokens``),
    expands it through the runner's own profile (GLM mHC ``hc_mult`` streams,
    passthrough otherwise) on a ``meta`` tensor, and counts
    ``numel * dtype itemsize``. A ``meta`` tensor owns no storage, so no
    buffer is allocated merely to calc size. Refuses fail-closed when the
    geometry cannot be derived rather than guessing.
    """
    import torch

    if type(batch_rows) is not int or batch_rows <= 0:
        raise AdjointIdentityRefused(
            f"stage A preflight needs a positive batch row count, got {batch_rows!r}")
    if type(seqlen) is not int or seqlen <= 0:
        raise AdjointIdentityRefused(
            f"stage A preflight needs a positive sequence length, got {seqlen!r}")
    try:
        embed = runner.base_model.embed_tokens
        hidden_width = int(embed.weight.shape[1])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise AdjointIdentityRefused(
            "stage A preflight cannot derive the hidden width from the live "
            f"model's base_model.embed_tokens: {exc}") from exc
    if hidden_width <= 0:
        raise AdjointIdentityRefused(
            f"stage A preflight saw a non-positive hidden width {hidden_width!r}")
    try:
        dtype_itemsize = int(torch.empty((), dtype=runner.dtype).element_size())
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise AdjointIdentityRefused(
            "stage A preflight cannot derive the execution dtype itemsize: "
            f"{exc}") from exc
    try:
        hidden_meta = torch.empty(
            (int(batch_rows), int(seqlen), int(hidden_width)),
            dtype=runner.dtype, device="meta")
        expanded = runner.profile.expand_hidden_for_layers(
            hidden_meta, runner.base_model)
        per_tensor = int(expanded.numel()) * int(dtype_itemsize)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise AdjointIdentityRefused(
            "stage A preflight cannot expand the hidden geometry through "
            f"the runner's own profile: {exc}") from exc
    if per_tensor <= 0:
        raise AdjointIdentityRefused(
            f"stage A preflight derived a non-positive tensor size {per_tensor!r}")
    return int(per_tensor)


def preflight_stage_a_artifact_budget(
    *, declared_bytes: int, demand: dict, plan_sealed_bytes: int | None = None,
) -> dict:
    """Refuse an under-budget Stage A invocation before any expensive forward.

    ``demand`` is :func:`estimate_stage_a_artifact_demand`: the hard
    ``lower_bound_bytes`` floor (true raw tensor bytes, full groups plus
    true remainder geometry) is the mandatory early-refusal threshold;
    ``planning_estimate_bytes`` is an explicitly named planning allowance,
    NOT a proven upper bound -- it assumes the per-file header envelope
    plus the stated per-checkpoint shared (plan auxiliary bound, which
    deduplicated live storages can exceed in retained files) and manifest
    allowances. ``declared_bytes`` is the run's ceiling (the override's
    ``run_used`` or the plan's sealed value). A below-floor ceiling refuses
    with the concrete required / declared values and the named remedy --
    never a silent raise. The runtime reserve / write / commit guards stay
    authoritative for serialized bytes and unpredicted overhead; this
    catches geometry the plan never derived, like the 416 GiB plan against
    the ~584 GiB floor.
    """
    if type(declared_bytes) is bool or type(declared_bytes) is not int:
        raise AdjointIdentityRefused(
            "stage A preflight needs an integer declared artifact budget "
            f"(bytes), got {declared_bytes!r}")
    floor = demand.get("lower_bound_bytes")
    planning = demand.get("planning_estimate_bytes")
    if type(floor) is not int or floor <= 0:
        raise AdjointIdentityRefused(
            "stage A preflight demand carries no positive lower_bound_bytes")
    if type(planning) is not int or planning < floor:
        raise AdjointIdentityRefused(
            "stage A preflight demand carries no planning_estimate_bytes "
            "at or above its floor")
    if int(declared_bytes) >= int(floor):
        return {"declared_bytes": int(declared_bytes),
                "required_floor_bytes": int(floor),
                "planning_estimate_bytes": int(planning)}
    sealed_note = (f" (plan sealed {int(plan_sealed_bytes)} bytes)"
                   if plan_sealed_bytes is not None else "")
    raise AdjointIdentityRefused(
        "stage A artifact budget below this invocation's hard geometry "
        f"floor: need >= {int(floor)} bytes raw "
        f"({int(floor) / 1024 ** 3:.2f} GiB true tensor bytes before "
        "headers/shared/manifest); planning allowance "
        f"{int(planning)} bytes "
        f"({int(planning) / 1024 ** 3:.2f} GiB, assuming per-file header "
        f"{ARTIFACT_FILE_HEADER_BYTES} B plus "
        f"{demand.get('shared_per_checkpoint_bytes')} B shared and "
        f"{demand.get('manifest_per_checkpoint_bytes')} B manifest per "
        f"checkpoint -- not a proven ceiling); declared "
        f"{int(declared_bytes)} bytes "
        f"({int(declared_bytes) / 1024 ** 3:.2f} GiB){sealed_note} for "
        f"{demand.get('n_retained_boundary_groups')} retained boundary "
        f"groups x {demand.get('n_batches_total')} batches "
        f"({demand.get('n_full_batches')} full + "
        f"{'one remainder batch' if demand.get('remainder_rows') else 'no remainder'}) + "
        f"{demand.get('n_probes')} probe planes live + "
        f"{demand.get('n_checkpoints')} checkpoints at boundaries "
        f"{demand.get('boundaries')}. Remedy: re-invoke with "
        f"--artifact-budget-bytes {int(planning)} (or "
        f"{ARTIFACT_BUDGET_ENV}={int(planning)}); the sealed plan is "
        "unchanged and the override is stamped into the run provenance.")


def _run_artifact_preflight(runner, calib_ids, execution, stride_value,
                            space, artifact) -> dict:
    """Early durable-budget preflight from already-available geometry.

    Runs after the runner + calibration load and before any expensive
    forward / layer traversal. Uses the live runner + calibration shapes
    (no bulk buffers: embedding width off the live model, stream expansion
    through the runner's own profile on ``meta`` tensors -- full batches
    and the remainder batch each at their own true size), the writer's own
    file-header envelope, the plan's auxiliary bound as a stated shared
    planning assumption, and the checkpoint manifest estimator on the real
    synthetic file plan. The hard floor refuses fail-closed with the
    concrete required / declared values and the named remedy; an
    underivable geometry or manifest estimate refuses named rather than
    guessing. The runtime reserve / write / commit guards stay
    authoritative for serialized bytes and unpredicted overhead.
    """
    _n_probes = int(execution["n_probes"])
    _probe_microbatch = int(execution.get("probe_microbatch", 0))
    _n_rows = int(calib_ids.shape[0])
    _seqlen = int(calib_ids.shape[1])
    _batch_rows = min(_probe_microbatch or _n_rows, _n_rows)
    _n_full = _n_rows // _batch_rows
    _rem_rows = _n_rows % _batch_rows
    _per_full = _stage_a_per_tensor_nbytes(
        runner, batch_rows=int(_batch_rows), seqlen=int(_seqlen))
    _per_rem = (_stage_a_per_tensor_nbytes(
        runner, batch_rows=int(_rem_rows), seqlen=int(_seqlen))
        if _rem_rows else None)
    _n_batches_total = _n_full + (1 if _rem_rows else 0)
    try:
        _aux_bound = int(execution["boundary_storage"]["max_auxiliary_bytes"])
    except (KeyError, TypeError) as exc:
        raise AdjointIdentityRefused(
            "stage A preflight needs the plan's boundary_storage."
            f"max_auxiliary_bytes shared planning assumption: {exc}") from exc
    if type(_aux_bound) is bool or not isinstance(_aux_bound, int) or _aux_bound < 0:
        raise AdjointIdentityRefused(
            "stage A preflight needs a nonnegative plan "
            f"max_auxiliary_bytes, got {_aux_bound!r}")
    try:
        from .joint_adjoint_checkpoints import (
            _checkpoint_manifest_envelope_bytes,
            checkpoint_directory,
        )
    except ImportError as exc:
        raise AdjointIdentityRefused(
            "stage A preflight cannot reuse the checkpoint manifest "
            f"estimator: {exc}") from exc
    try:
        _manifest_probe_boundary = derive_checkpoint_boundaries(
            int(runner.num_layers), int(stride_value))[0]
        _manifest_dir = checkpoint_directory(space, int(_manifest_probe_boundary))
        _dummy_session = {"generation": "0" * 32, "kind": "adjoint_checkpoint",
                          "run_identity_sha256": "0" * 64}
        _batch_sizes = [_batch_rows] * _n_full + ([_rem_rows] if _rem_rows else [])
        _act_plan = [{
            "probe_index": p, "batch_index": b,
            "slot": f"cotangent-{p}-{b}",
            "name": f"cotangent-{p}-{b}",
            "path": str(_manifest_dir / "entries" / f"cotangent-{p}-{b}.pt"),
            "tensor_bytes": (int(_per_full) if _rows == _batch_rows
                             else int(_per_rem)),
            "file_envelope": ((int(_per_full) if _rows == _batch_rows
                               else int(_per_rem))
                              + int(ARTIFACT_FILE_HEADER_BYTES)),
            "shape": [int(_rows), int(_seqlen)],
            "dtype": str(runner.dtype),
        } for p in range(_n_probes) for b, _rows in enumerate(_batch_sizes)]
        _shared_names = (
            [f"shared-adjoint-{p}-{b}" for p in range(_n_probes)
             for b in range(_n_batches_total)]
            + [f"shared-pass-{b}" for b in range(_n_batches_total)])
        _n_shared = len(_shared_names)
        _per_shared_envelope = max(
            int(ARTIFACT_FILE_HEADER_BYTES),
            int(_aux_bound) // max(1, _n_shared) if _aux_bound else int(
                ARTIFACT_FILE_HEADER_BYTES))
        _shared_plan = [{
            "name": name,
            "path": str(_manifest_dir / "entries" / f"{name}.pkl"),
            "file_envelope": int(_per_shared_envelope),
        } for name in _shared_names]
        _manifest_bytes = int(_checkpoint_manifest_envelope_bytes(
            boundary=int(_manifest_probe_boundary), session=_dummy_session,
            activation_plan=_act_plan, shared_plan=_shared_plan))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise AdjointIdentityRefused(
            "stage A preflight cannot bound the checkpoint manifest from "
            f"this invocation's geometry: {exc}") from exc
    _demand = estimate_stage_a_artifact_demand(
        n_probes=int(_n_probes), n_full_batches=int(_n_full),
        remainder_rows=int(_rem_rows),
        per_full_tensor_nbytes=int(_per_full),
        per_remainder_tensor_nbytes=_per_rem,
        num_layers=int(runner.num_layers), stride=int(stride_value),
        header_bytes=int(ARTIFACT_FILE_HEADER_BYTES),
        shared_per_checkpoint_bytes=int(_aux_bound),
        manifest_per_checkpoint_bytes=int(_manifest_bytes))
    _preflight = preflight_stage_a_artifact_budget(
        declared_bytes=int(artifact["run_used"]), demand=_demand,
        plan_sealed_bytes=int(artifact["override"]["plan_sealed_bytes"]
                              if artifact["override"] is not None
                              else int(artifact["run_used"])))
    print(f"joint_cost_stage_a: artifact preflight: declared "
          f"{_preflight['declared_bytes']} bytes "
          f"({_preflight['declared_bytes'] / 1024 ** 3:.2f} GiB) >= floor "
          f"{_preflight['required_floor_bytes']} bytes "
          f"({_preflight['required_floor_bytes'] / 1024 ** 3:.2f} GiB raw; "
          f"planning allowance {_preflight['planning_estimate_bytes']} bytes); "
          f"full-tensor {_per_full} bytes x {_n_full} batches"
          f"{f' + remainder {_per_rem} bytes x 1 batch' if _rem_rows else ''}, "
          f"boundaries {_demand['boundaries']}", flush=True)
    return {
        "declared_bytes": _preflight["declared_bytes"],
        "required_floor_bytes": _preflight["required_floor_bytes"],
        "planning_estimate_bytes": _preflight["planning_estimate_bytes"],
        "demand": {k: _demand[k] for k in (
            "boundaries", "n_checkpoints", "n_retained_boundary_groups",
            "n_full_batches", "remainder_rows", "n_batches_total",
            "n_probes", "per_full_tensor_nbytes",
            "per_remainder_tensor_nbytes",
            "shared_per_checkpoint_bytes",
            "manifest_per_checkpoint_bytes")},
    }


def run_adjoint_capture_core(
    runner, calib_ids, *, execution, output_root, stride,
    source_model_identity, unit_roster_sha256, plan_sha256, prepared_sha256,
    read_manifest_sha256, implementation_sha256, campaign_scope=None,
    boundary_artifact_bytes=None, artifact_budget_stamp=None,
    min_free_gib=0.0, progress=None, produced_output=None,
) -> dict:
    """Forward boundaries, tail cotangents, strided render-free chain.

    Returns the receipt dict (the caller seals it). The forward capture and
    the tail cotangents are the single run's own calls; the chain is
    :func:`render_free_layer_roll` -- the completed-layer leg of
    ``compute_aura_cost_streamed``'s reverse walk -- run for every layer,
    with boundary entries **retained** (every quantum reads them later) and a
    checkpoint serialized at each strided boundary.

    ``boundary_artifact_bytes`` (optional, bytes) replaces the sealed
    ``max_artifact_bytes`` for this invocation only; ``artifact_budget_stamp``
    (optional, the :func:`resolve_artifact_budget_override` deviation stamp)
    is carried into the receipt provenance verbatim. Neither rewrites the
    sealed plan.

    ``produced_output`` (optional) is a bound
    ``stage_a_produced_output.BoundaryProducedPublication`` -- this
    action's own PrismaBuild produced-output instance. With it, the
    boundary entries this capture writes are declared, staged and read
    back through PrismaBuild instead of being unreadable to their own
    writer: an admitted action's outputs are not in the run's sealed input
    map, so the strict allowed-tier reader refuses them, and there is no
    own-session exemption. Without it the capture behaves exactly as
    before and its entries resolve through the ordinary input map.

    The publication's geometry is derived HERE, from the numbers this
    invocation actually runs under, not from a planning figure: the
    publication group is the storage policy's own read window
    (``prefetch_batches``), the per-entry tensor ceiling is the boundary
    tensor this panel produces, and the durable origin class maximum is
    the EFFECTIVE ``max_artifact_bytes`` -- the plan's sealed value or
    this run's override, whichever is in force.
    """
    from .cost_streaming import (
        StreamedBoundaryArtifacts,
        normalize_boundary_storage,
        prefetched_boundary_batches,
        validate_streamed_model_identity,
    )
    from .kl_fisher import ROW_PROBE_LAYOUT, fisher_probe_scalar
    from .sensitivity_probe import SharedStateCotangents, kv_cotangent_path_enabled

    n_probes = int(execution["n_probes"])
    seed_base = int(execution["seed_base"])
    token_scope = "all"
    temperature = 1.0
    probe_microbatch = int(execution.get("probe_microbatch", 0))
    num_layers = runner.num_layers
    boundaries = derive_checkpoint_boundaries(num_layers, stride)
    space = adjoint_space(output_root)
    space.mkdir(parents=True, exist_ok=True)

    storage_policy = dict(normalize_boundary_storage(execution["boundary_storage"]))
    storage_policy["directory"] = str(boundary_entry_directory(space))
    if boundary_artifact_bytes is not None:
        if type(boundary_artifact_bytes) is bool or not isinstance(
                boundary_artifact_bytes, int) or boundary_artifact_bytes <= 0:
            raise AdjointIdentityRefused(
                "stage A boundary_artifact_bytes must be a positive integer "
                f"byte count (bytes), got {boundary_artifact_bytes!r}")
        storage_policy["max_artifact_bytes"] = int(boundary_artifact_bytes)
    if artifact_budget_stamp is not None and not isinstance(
            artifact_budget_stamp, dict):
        raise AdjointIdentityRefused(
            "stage A artifact_budget_stamp must be a mapping or None")
    storage = StreamedBoundaryArtifacts(storage_policy)

    batch_rows = min(probe_microbatch or len(calib_ids), len(calib_ids))
    row_offsets = list(range(0, len(calib_ids), batch_rows))
    probe_layout = None
    execution_partition = None
    if probe_microbatch:
        sequence_length = int(calib_ids.shape[1])
        probe_layout = {
            "schema": ROW_PROBE_LAYOUT,
            "global_rows": len(calib_ids),
            "sequence_length": sequence_length,
            "selected_tokens_per_row": sequence_length,
            "vocab_size": int(runner._head().weight.shape[0]),
            "token_scope": token_scope,
            "global_token_count": len(calib_ids) * sequence_length,
        }
        execution_partition = {
            "schema": "prismaquant.aura.streamed_microbatch.v1",
            "requested_rows": probe_microbatch,
            "effective_rows": batch_rows,
            "partition_count": len(row_offsets),
            "row_order": "contiguous_complete_sequences",
            "gradient_diagnostics": "sum_output_operators_fp32_before_norm",
        }

    bind_identity = {
        "source_model": validate_streamed_model_identity(
            source_model_identity, where="adjoint capture"),
        "producer_source_sha256": implementation_sha256,
        "calibration_sha256": hashlib.sha256(
            calib_ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "calibration_shape": list(calib_ids.shape),
        "calibration_dtype": str(calib_ids.dtype),
        "n_probes": n_probes, "seed_base": seed_base,
        "token_scope": token_scope, "temperature": temperature,
        "execution_partition": execution_partition,
        "campaign_stage": "joint_adjoint_capture",
    }

    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)

    operator_windows = execution.get("operator_windows")
    checkpoints: list[dict] = []
    chain_telemetry: list[dict] = []
    started = time.time()
    capture_started = None
    tail_started = None
    chain_started = None
    chain_backwards = 0

    def log(message: str) -> None:
        print(f"joint_cost_stage_a: {message}", flush=True)

    with storage:
        storage.bind(bind_identity, n_probes=n_probes, published=True)
        if produced_output is not None:
            # After bind, because the entry directory this owner chose is
            # what must sit inside the publication's bound output prefix;
            # the binding refuses an own-generation path outside it here
            # rather than at the first descriptor.
            storage.bind_produced_output(
                produced_output,
                group_size=int(storage_policy["prefetch_batches"]),
                n_batches=len(calib_ids),
                max_entry_tensor_bytes=_stage_a_per_tensor_nbytes(
                    runner, batch_rows=batch_rows,
                    seqlen=int(calib_ids.shape[1])))
            log("boundary capture: produced-output binding "
                f"{produced_output.instance['owner_action_key']} "
                f"prefix={produced_output.output_prefix} "
                f"group={int(storage_policy['prefetch_batches'])}")
        if progress is not None:
            # The initial boundary loop reports under the declared head
            # phase until the first forward observer fires. Entering moves
            # no units: the head walk's committed total is already the
            # reporter's base, and only published entries advance it.
            from .joint_run_progress import HEAD_PHASE
            progress.enter(HEAD_PHASE)
            storage.watch_progress(progress)
        log(f"boundary capture: calib {tuple(calib_ids.shape)} in "
            f"{len(row_offsets)} partition(s) across {num_layers} layers ...")
        capture_started = time.time()
        batches = runner.capture_layer_major_boundaries(
            [calib_ids[offset:offset + batch_rows] for offset in row_offsets],
            storage=storage,
            source_phase=(stage_a_forward_observer(progress)
                          if progress is not None else None))
        log(f"boundary capture done in {(time.time() - capture_started) / 60:.1f} min; "
            f"starting {n_probes}-probe tail cotangents")

        device, dtype = runner.device, runner.dtype
        cotangents = [[SharedStateCotangents(enabled=kv_cotangent_path_enabled())
                       for _ in batches] for _ in range(n_probes)]
        grad_outs = [[] for _ in range(n_probes)]
        tail_plane: dict[tuple[int, int], torch.Tensor] = {}
        storage.watch_auxiliary(batches, cotangents)
        storage.check_auxiliary(batches, cotangents=cotangents)
        tail_started = time.time()
        with prefetched_boundary_batches(storage, batches, num_layers) as tail_batches:
            for batch_index, batch, tail_cpu, _unused in tail_batches:
                try:
                    for probe_index in range(n_probes):
                        tail = tail_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
                        logits = runner.tail_logits(batch, tail)
                        if probe_layout is not None and list(logits.shape) != [
                                len(batch.input_ids), int(calib_ids.shape[1]),
                                probe_layout["vocab_size"]]:
                            raise RuntimeError(
                                "adjoint capture tail differs from bound probe geometry")
                        probe = fisher_probe_scalar(
                            logits, seed=seed_base + probe_index,
                            token_scope=token_scope, temperature=temperature,
                            distribution="rademacher",
                            **({"token_count_override": probe_layout["global_token_count"],
                                "global_row_offset": row_offsets[batch_index]}
                               if probe_layout is not None else {}),
                        )
                        probe.backward()
                        if tail.grad is None:
                            raise RuntimeError(
                                "adjoint capture tail produced no cotangent")
                        tail_plane[(probe_index, batch_index)] = tail.grad.detach().to("cpu")
                        grad_outs[probe_index].append(storage.write(
                            tail.grad, batch_index=batch_index,
                            boundary_index=num_layers, probe_index=probe_index))
                        del logits, probe, tail
                    storage.retire(batch.activations_cpu[-1])
                    batch.activations_cpu[-1] = torch.empty(0)
                finally:
                    tail_cpu = logits = probe = tail = None
        log(f"tail cotangents done in {(time.time() - tail_started) / 60:.1f} min; "
            f"publishing the tail checkpoint at boundary {num_layers}")

        def serialize_checkpoint(boundary: int, plane) -> None:
            shared_pass = {batch: batches[batch].shared_pass_state
                           for batch in range(len(batches))}
            record = write_checkpoint_with_snapshot(
                storage, space, boundary=boundary,
                session={"generation": storage.session["generation"],
                         "kind": "adjoint_checkpoint",
                         "run_identity_sha256": storage.session["run_identity_sha256"]},
                plane=plane, cotangents=cotangents, shared_pass=shared_pass)
            checkpoints.append(record)
            log(f"checkpoint published at boundary {boundary} "
                f"({len(plane)} cotangent entries)")

        # The tail set is the first checkpoint: layer num_layers-1's quantum
        # chains nothing (§3.1).
        serialize_checkpoint(num_layers, tail_plane)
        tail_plane.clear()
        if progress is not None:
            # The tail checkpoint is durable work landed while the read plan
            # stays on forward-last: count it without leaving the phase the
            # tier still holds. There is deliberately no tail progress phase
            # (a name the sealed plan does not carry would reset its
            # tracking); the phase name survives only in the log line below.
            progress.entry(layer=num_layers, partition=0,
                           kind="tail_checkpoint")
            progress.flush(force=True)
            log(f"{ADJOINT_TAIL_PHASE} checkpoint published at boundary "
                f"{num_layers}; read plan stays on "
                f"{adjoint_forward_phase_name(num_layers - 1)}")

        chain_started = time.time()
        if num_layers > 0:
            # The chain's first layer reads a plane the forward pass wrote
            # and retired long ago; every later layer's plane is asked for
            # by the roll before it (``render_free_layer_roll``). Asking
            # here starts its movers before the first window needs them.
            storage.stage_produced_boundary_ahead(num_layers - 1)
        for layer in reversed(range(num_layers)):
            if progress is not None:
                progress.enter(adjoint_chain_phase_name(layer))
                progress.flush(force=True)
            layer_started = time.time()
            try:
                runner.context.install(
                    layer,
                    require_prefetched=runner.require_prefetched_residency,
                    **({"prefetch_following": False}
                       if operator_windows is not None else {}),
                )
                if operator_windows is None:
                    runner.schedule_reverse_prefetch(layer)
                else:
                    successors = range(
                        max(0, layer - runner.prefetch_lookahead), layer)
                    for successor in reversed(successors):
                        runner.context.schedule_prefetch(successor)
                    settle = getattr(runner.context, "settle_prefetched_layers", None)
                    if callable(settle):
                        settle(successors)
                    elif torch.device(runner.device).type == "cuda":
                        raise RuntimeError(
                            "render-free chain requires source prefetch settlement")
                stash: dict[tuple[int, int], torch.Tensor] = {}

                def roll(tensor, batch_index, probe_index):
                    # Layer 0 is the walk's last roll: no read follows it,
                    # and its entries are retired right after the loop.
                    grad_outs[probe_index][batch_index] = storage.write(
                        tensor, batch_index=batch_index, boundary_index=layer,
                        probe_index=probe_index,
                        previous=grad_outs[probe_index][batch_index],
                        **({} if layer > 0 else {"read_back": False}))
                    stash[(probe_index, batch_index)] = tensor

                chain_backwards += render_free_layer_roll(
                    runner, storage=storage, batches=batches, layer=layer,
                    cotangents=cotangents, n_probes=n_probes,
                    incoming_entries=grad_outs, incoming_tensor=None,
                    roll=roll, min_free_gib=min_free_gib)
                if layer in boundaries:
                    serialize_checkpoint(layer, stash)
                chain_telemetry.append({
                    "layer": layer, "wall_s": time.time() - layer_started,
                    "checkpoint": layer in boundaries,
                })
            finally:
                runner.context.unload(layer)
        log(f"render-free chain done in {(time.time() - chain_started) / 60:.1f} min "
            f"({chain_backwards} backwards)")

        # The walk ends holding boundary-0 cotangent entries -- the last roll.
        # Every strided checkpoint above them is serialized and sealed, so
        # these are deliberately retired rather than left as dead weight the
        # published generation would otherwise keep forever.
        for probe in range(n_probes):
            for batch in range(len(batches)):
                reference = grad_outs[probe][batch]
                if reference is not None:
                    storage.retire(reference)
                    grad_outs[probe][batch] = None

        boundary_entries: dict[str, list[dict]] = {}
        for boundary in range(num_layers):
            boundary_entries[str(boundary)] = [
                exact_entry_record(batch.activations_cpu[boundary])
                for batch in batches]
        retention = storage.receipt()

    receipt = {
        "schema": ADJOINT_RECEIPT_SCHEMA,
        "entry_point": ADJOINT_CAPTURE_ENTRY_POINT,
        "status": "complete",
        "run_identity": {
            "plan_sha256": str(plan_sha256),
            "prepared_sha256": str(prepared_sha256),
            "read_manifest_sha256": str(read_manifest_sha256),
            "implementation_sha256": str(implementation_sha256),
            "unit_roster_sha256": str(unit_roster_sha256),
            "campaign_scope": campaign_scope,
            "n_probes": n_probes,
            "seed_base": seed_base,
            "calibration_shape": list(calib_ids.shape),
            "calibration_sha256": bind_identity["calibration_sha256"],
        },
        "stride": {"value": int(stride), "source": None,  # filled by caller
                   "boundaries": [int(b) for b in boundaries],
                   "max_chain_layers": int(stride) - 1},
        "boundary_storage": {
            "session": storage.session,
            "policy": storage.identity,
            "directory": str(boundary_entry_directory(space)),
        },
        "boundary_entries": boundary_entries,
        "checkpoints": checkpoints,
        "retention": retention,
        "artifact_budget_override": artifact_budget_stamp,
        "telemetry": {
            "started_unix": started,
            "wall_s": time.time() - started,
            "capture_wall_s": (time.time() - capture_started
                               if capture_started else None),
            "tail_wall_s": ((tail_started and chain_started
                             and chain_started - tail_started) or None),
            "chain_wall_s": (time.time() - chain_started
                             if chain_started else None),
            "chain_backwards": chain_backwards,
            "chain_layers": chain_telemetry,
        },
        "dev_mode": dev_mode_stamp(),
    }
    return receipt


def _io_counters() -> dict:
    values = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value)
    return values


def bind_stage_a_produced_output(*, artifact_max_bytes, tier=None,
                                 queue_root=None, env=None,
                                 command_extra=()):
    """This action's own PrismaBuild produced-output publication, or None.

    The boundary entries Stage A writes are its OWN outputs, so they are
    not in the run's sealed input map and the strict allowed-tier reader
    refuses them (``staged-not-serving``) with no own-session exemption.
    Binding here is what lets the same action read them back, through
    PrismaBuild, without touching the input resolver.

    The template is NOT built here. It is sealed pre-submit as an input of
    this action's own request, and ``bind_from_admitted_owner`` reads the
    declaration the submission made -- an operator dictionary assembled at
    runtime would be a second source for a number admission already fixed.
    What this DOES check is that the two agree: the declared durable
    payload maximum must equal the EFFECTIVE artifact max this invocation
    runs under (the plan's sealed ``max_artifact_bytes`` or this run's
    override). A capture admitted against one budget and running under
    another is producer/consumer drift and must refuse, not adapt.

    ``tier`` is optional and normally omitted: the payload carries tier
    CLASSES, not a tier id, so the tier is the declaration's own -- derived
    when the template permits exactly one, refused when it permits several
    (that choice belongs to whoever declared the template).

    ``None`` is returned in exactly ONE case: this process carries no
    PrismaBuild launch context at all (a legacy or local invocation), so
    there is no owner to bind and the capture behaves as it always did.
    An ADMITTED action that fails to bind REFUSES, carrying the original
    reason -- a missing SDK, a bad claim, a template mismatch. The
    alternative is the shape that costs the most here: the binding is lost
    silently, 512 boundary files are written again, and the failure
    surfaces only when the first read asks for bytes nobody declared.
    """

    from .stage_a_produced_output import (
        BoundaryProducedBindingError, BoundaryProducedPublication)

    source = dict(os.environ) if env is None else dict(env)
    admitted = bool(source.get("PRISMABUILD_ACTION_KEY"))
    if not admitted:
        if queue_root is not None:
            raise AdjointIdentityRefused(
                "stage A was given a PrismaBuild queue root but carries no "
                "PRISMABUILD_ACTION_KEY: a produced-output owner is an "
                "ADMITTED action, and binding one from a half-present "
                "launch context is refused rather than guessed")
        return None
    try:
        publication = BoundaryProducedPublication.bind_from_admitted_owner(
            queue_root=queue_root, tier=tier, env=source,
            command_extra=tuple(command_extra))
    except BoundaryProducedBindingError as exc:
        raise AdjointIdentityRefused(
            "stage A is an admitted PrismaBuild action "
            f"({source['PRISMABUILD_ACTION_KEY'][:12]}) and cannot bind its "
            f"own produced output: {exc}") from exc
    declared = int(publication.durable_maxima().get("payload_max_bytes", 0))
    if declared != int(artifact_max_bytes):
        raise AdjointIdentityRefused(
            "stage A produced-output template declares a durable payload "
            f"maximum of {declared} bytes, but this invocation runs under "
            f"{int(artifact_max_bytes)}: the admitted budget and the "
            "effective budget must be the same number (re-declare the "
            "template, or drop the run override) -- refusing rather than "
            "writing origin bytes against a budget nobody admitted")
    return publication


def run_adjoint_capture(
    config, *, plan_sha256, prepared, output_root, stride=None,
    read_manifest_sha256=None, data_manifest_sha256=None, resume=False,
    prefetch_override=None, artifact_budget_bytes=None,
) -> dict:
    """Load the head phase and run the adjoint capture (one PB action)."""
    from .aura_cost import _aura_source_sha256
    from .calibration_data import load_calibration_input
    from .cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
    from .glm_capture_compatibility import require_capture_compatibility
    from .gpu_guard import require_cuda_hot_path
    from .joint_projection_backend import executing_image, prewarm_projection_backend
    from .joint_run_progress import JointRunProgress
    from .model_profiles import detect_profile
    from .residency_map import bind_residency_manifest, residency_report
    from .tessera_joint_aura import (
        ACTIVATION_SCALE_ENV,
        _bound,
        _preflight_run_prepared,
        _same,
        _seed_source_identity_cache,
        load_measured_anchor_input,
    )
    from .tessera_reader import load_declared_reader

    require_cuda_hot_path("joint_cost_stage_a", "cuda")
    from .autoscale import require_bounded_capture_environment

    execution = config["execution"]
    if (config.get("qualification_window") is not None
            or execution.get("retained_operator_windows") is not None):
        require_bounded_capture_environment(os.environ)
    os.environ[ACTIVATION_SCALE_ENV] = execution["production_act_scales"]
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    if Path(config["output_root"]).resolve() != Path(output_root).resolve():
        raise AdjointIdentityRefused(
            f"plan output_root {config['output_root']} is not --output-root "
            f"{output_root}")
    _bound(prepared, "prepared anchors")
    stride_value, stride_source = resolve_stride(config, stride)
    prefetch = resolve_prefetch_override(config, prefetch_override)
    artifact = resolve_artifact_budget_override(
        config, artifact_budget_bytes)

    space = adjoint_space(output_root)
    space.mkdir(parents=True, exist_ok=True)
    bind_residency_manifest(data_manifest_sha256)
    result = {
        "schema": "prismaquant.joint_adjoint_capture.execution.v1",
        "command": "adjoint-capture",
        "plan_sha256": plan_sha256,
        "prepared_sha256": prepared["sha256"],
        "stride": {"value": stride_value, "source": stride_source},
        "prefetch_override": prefetch["override"],
        "artifact_budget_override": artifact["override"],
        "env": {"host": socket.gethostname(), "started_epoch": time.time(),
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "affinity": sorted(os.sched_getaffinity(0))},
        "dev_mode": dev_mode_stamp(),
        "phases": [], "passed": False,
    }
    result["env"]["container_content_sha256"] = executing_image()
    if prefetch["override"] is not None:
        stamp = prefetch["override"]
        print("joint_cost_stage_a: prefetch override: plan sealed "
              f"{stamp['plan_sealed']} replaced by {stamp['run_used']} "
              f"(source {stamp['source']}, reason: {stamp['reason']})",
              flush=True)
    if artifact["override"] is not None:
        astamp = artifact["override"]
        print("joint_cost_stage_a: artifact budget override: plan sealed "
              f"{astamp['plan_sealed_bytes']} bytes replaced by "
              f"{astamp['run_used_bytes']} bytes (source {astamp['source']}); "
              "sealed plan unchanged",
              flush=True)

    sampler = GpuPowerSampler().start()
    kernel = KernelTimeProfiler()
    runner = None
    started, before_io = time.time(), _io_counters()
    try:
        reader = load_declared_reader(config.get("reader"))
        reader_identity = None if reader is None else reader.identity
        implementation = _aura_source_sha256()
        projection_backend = prewarm_projection_backend(
            execution.get("projection_backend"), device="cuda")
        result["projection_backend"] = projection_backend.identity
        _preflight_run_prepared(prepared, plan_sha256=plan_sha256,
                                implementation_sha256=implementation,
                                reader_identity=reader_identity,
                                projection_backend=projection_backend.identity)
        data = load_measured_anchor_input(
            config["inputs"], reader=reader, synthesis_device="cuda",
            progress_phase="head",
            head_checkpoint=space / "head-walk", head_resume=resume,
            require_existing_renders=True, verify_payloads=False,
            historical_encoder_reuse=config.get("historical_encoder_reuse"))
        _same(config["model"], data.census["model"], "requested source model")
        _same(data.census["attention_implementation"], "eager",
              "qualified source attention")
        ids, calibration = load_calibration_input(
            config["calibration_input"]["path"],
            expected_sha256=config["calibration_input"]["sha256"],
            n_samples=execution["n_calib_samples"],
            seqlen=execution["calib_seqlen"])
        original_draw = data.payload["provenance"]["hessian"]["calibration_identity"]
        for name in ("fit_ids_sha256", "text_sha256", "nsamples", "seqlen", "seed"):
            _same(calibration["provenance"].get(name), original_draw.get(name),
                  f"original full draw {name}")
        result["calibration_input"] = calibration

        completion = json.loads(_bound(prepared, "prepared anchors").read_text())
        _same(completion.get("plan_sha256"), plan_sha256, "prepared plan")

        identity_cache_path = _seed_source_identity_cache(config, space / "run")
        # The single-run path threads the plan's derivative binding and its
        # source-prefetch budget into the model build (tessera_joint_aura's
        # execute()); stage A builds the same model and threads the same two
        # plan fields -- the corrected GLM runtime refuses to load unbound
        # (the d4578e5e6af4 failure), and the prefetch budget is the plan's
        # own answer to #737's single-worker pin. An explicitly recorded
        # override (#819) replaces the budget for this run only; the plan's
        # block is what the deviation stamp names as sealed.
        runner = build_streamed_causal_lm(
            config["model"], device=torch.device("cuda"), dtype=torch.bfloat16,
            offload_folder=str(space / "run" / "offload"),
            profile=detect_profile(config["model"]), attn_implementation="eager",
            source_authentication=None,
            source_derivative=execution.get("source_derivative"),
            **prefetch["run_used"])
        require_capture_compatibility(config.get("source_capture_compatibility"),
                                      capture=config["canonical_capture"],
                                      model=runner.model)
        source = build_streamed_model_identity(runner, config["model"],
                                               identity_cache_path=identity_cache_path)
        _same(completion.get("source_model_identity"), source,
              "prepared source identity")
        result.update(source_model_identity=source, units=len(data.formats_by_qname),
                      measured_cells=len(data.cells))
        roster_digest = hashlib.sha256("".join(
            f"{name}\n" for name in sorted(data.formats_by_qname)).encode()).hexdigest()

        result["artifact_preflight"] = _run_artifact_preflight(
            runner, ids, execution, stride_value, space, artifact)

        kernel.__enter__()
        progress = JointRunProgress(
            layers=runner.num_layers, partitions=1,
            base_units=data.progress_committed, log=lambda message: print(
                f"joint_cost_stage_a: {message}", flush=True))
        publication = bind_stage_a_produced_output(
            artifact_max_bytes=int(artifact["run_used"]))
        receipt = run_adjoint_capture_core(
            runner, ids.to(runner.device), execution=execution,
            output_root=output_root, stride=stride_value,
            source_model_identity=source, unit_roster_sha256=roster_digest,
            plan_sha256=plan_sha256, prepared_sha256=prepared["sha256"],
            read_manifest_sha256=(read_manifest_sha256
                                  or "0" * 64),
            implementation_sha256=implementation,
            campaign_scope=config.get("campaign_scope"),
            boundary_artifact_bytes=int(artifact["run_used"]),
            artifact_budget_stamp=artifact["override"],
            min_free_gib=config.get("min_free_gib", 0.0), progress=progress,
            produced_output=publication)
        receipt["stride"]["source"] = stride_source
        torch.cuda.synchronize()
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if result["peak_gpu_bytes"] > config["max_gpu_bytes"]:
            raise RuntimeError("observed GPU allocation exceeds declared budget")
        write_adjoint_receipt(space, receipt)
        result["adjoint_receipt"] = {
            "path": str(space / "adjoint-capture.json"),
            "sha256": hashlib.sha256(
                (space / "adjoint-capture.json").read_bytes()).hexdigest(),
        }
        result["checkpoints"] = [
            {"boundary": entry["boundary"],
             "cotangent_sha256": entry["cotangent_sha256"]}
            for entry in receipt["checkpoints"]]
        result["passed"] = True
    finally:
        kernel.__exit__(None, None, None)
        if runner is not None:
            completed, runner = runner, None
            completed.shutdown()
        gpu = sampler.stop()
        result["env"]["finished_epoch"] = time.time()
        result["phases"].append({"phase": "adjoint-capture", "start_epoch": started,
                                 "end_epoch": result["env"]["finished_epoch"]})
        result["io_before"], result["io_after"] = before_io, _io_counters()
        result["gpu"] = gpu
        result["kernel_active_s"] = kernel.kernel_active_s
        result["kernel_profiler_error"] = kernel.error
        residency = residency_report()
        if residency is not None:
            result["residency"] = residency

    counters = {
        "schema": QUANTUM_COUNTERS_SCHEMA.replace(
            "joint_layer_quantum", "joint_adjoint_capture"),
        "stage": "adjoint_capture",
        "wall_s": result["phases"][0]["end_epoch"] - started,
        "kernel_active_s": kernel.kernel_active_s,
        "gpu_joules": gpu.get("gpu_joules"),
        "gpu_power_w_p50": gpu.get("gpu_power_w_p50"),
        "gpu_power_w_p95": gpu.get("gpu_power_w_p95"),
        "gpu_power_w_max": gpu.get("gpu_power_w_max"),
        "gpu_sampler_samples": gpu.get("sample_count"),
        "gpu_power_envelope_w": 140.0,
        "kernel_active_ratio": (
            kernel.kernel_active_s / (result["phases"][0]["end_epoch"] - started)
            if not kernel.error else None),
        "phases": [{"name": "adjoint-capture",
                    "bytes_from_ram": (residency or {}).get("bytes_from_ram"),
                    "bytes_from_stage": (residency or {}).get("bytes_from_stage"),
                    "bytes_from_pool": (residency or {}).get("bytes_from_pool")}],
        "stride": {"value": stride_value, "source": stride_source},
        "prefetch_override": prefetch["override"],
        "artifact_budget_override": artifact["override"],
        "artifact_preflight": result.get("artifact_preflight"),
    }
    atomic_write_bytes(space / "counters.json",
                       (json.dumps(counters, sort_keys=True, indent=2,
                                   allow_nan=False) + "\n").encode())
    atomic_write_bytes(space / "results.json",
                       (json.dumps(result, sort_keys=True, indent=2,
                                   allow_nan=False) + "\n").encode())
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run stage A of the distributed joint-AURA cost campaign: "
                    "forward boundaries, tail cotangents and the strided "
                    "render-free cotangent chain (contract §2.2 stage A).")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=None,
                        help="cotangent checkpoint stride when the plan does "
                             "not declare distributed_campaign (S is a plan "
                             "knob; a disagreement with the plan refuses)")
    parser.add_argument("--read-manifest-sha256", default=None,
                        help="digest of the parent run's sealed data manifest")
    parser.add_argument("--data-manifest-sha256", default=None)
    from .staged_tier_policy import DEFAULT_ALLOWED_TIERS
    parser.add_argument("--allowed-tiers", default=DEFAULT_ALLOWED_TIERS,
                        help="sealed staged-tier declaration for GPU-consumed "
                             "bulk inputs: comma subset of {ram,ssd}, RAM "
                             "first (default %(default)s). Pool/HDD bulk "
                             "opens refuse under this declaration.")
    parser.add_argument("--prefetch-override", type=Path, default=None,
                        help="explicit source_prefetch override document "
                             "(schema %s): replaces the plan's sealed "
                             "budget for this run only, validated by the "
                             "same field grammar, with the deviation "
                             "stamped into results.json and counters.json"
                             % PREFETCH_OVERRIDE_INPUT_SCHEMA)
    parser.add_argument("--artifact-budget-bytes", default=None,
                        help="explicit durable artifact ceiling in bytes "
                             "(positive integer): replaces the plan's sealed "
                             "boundary_storage.max_artifact_bytes for this "
                             "run only, with the deviation stamped into "
                             "results.json, counters.json and the adjoint "
                             "receipt; the sealed plan is unchanged")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        require_dev_mode("joint_cost_stage_a")
        from .tessera_joint_aura import _load_plan

        config = _load_plan(args.plan, args.plan_sha256)
        from .staged_tier_policy import activate_staged_tier_policy
        try:
            allowed = activate_staged_tier_policy(args.allowed_tiers)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"[STAGED-TIER] bulk inputs serve from {','.join(sorted(allowed))}; "
              f"pool/HDD bulk opens refuse", flush=True)
        result = run_adjoint_capture(
            config, plan_sha256=args.plan_sha256,
            prepared={"path": str(args.prepared), "sha256": args.prepared_sha256},
            output_root=args.output_root, stride=args.stride,
            read_manifest_sha256=args.read_manifest_sha256,
            data_manifest_sha256=args.data_manifest_sha256, resume=args.resume,
            prefetch_override=args.prefetch_override,
            artifact_budget_bytes=args.artifact_budget_bytes)
    except AdjointIdentityRefused as exc:
        print(f"adjoint_identity_refused: {exc}", flush=True)
        return EXIT_IDENTITY_REFUSED
    print(json.dumps({key: result[key] for key in (
        "command", "passed", "stride", "checkpoints")}))
    return EXIT_OK if result["passed"] else EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
