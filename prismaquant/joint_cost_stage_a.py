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
from contextlib import contextmanager
from pathlib import Path

import torch

from .cost_stage_checkpoint import atomic_write_bytes, canonical_json_sha256
from .joint_adjoint_checkpoints import (
    ADJOINT_CAPTURE_ENTRY_POINT,
    ADJOINT_RECEIPT_SCHEMA,
    CHAIN_REGIME_KEY,
    DEFAULT_STRIDE,
    QUANTUM_COUNTERS_SCHEMA,
    ChainRegimeRefused,
    GpuPowerSampler,
    KernelTimeProfiler,
    adjoint_space,
    boundary_entry_directory,
    chain_layers_for,
    chain_regime_identity,
    checkpoint_cotangent_plane,
    derive_checkpoint_boundaries,
    dev_mode_stamp,
    exact_entry_record,
    occupied_checkpoint_directories,
    open_adjoint_checkpoint,
    render_free_layer_roll,
    normalize_chain_regime,
    require_dev_mode,
    write_adjoint_checkpoint,
    write_adjoint_receipt,
)
from .joint_layer_quanta import (
    ADJOINT_TAIL_PHASE,
    adjoint_chain_phase_name,
    adjoint_forward_phase_name,
)
from .produced_output_spool import plane_partitions, sealed_spool_root
from .source_read_plan import chain_opening_window, chain_prefetch_window


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
#: Opt in to a whole-capture ``torch.profiler`` session. Off by default: the
#: session's scope is the whole capture, which nothing bounds, and Kineto holds
#: every CUDA record in host memory until the stop (PQ #899). ``1`` restores
#: the measurement for a capture small enough to afford it.
KERNEL_PROFILE_ENV = "PRISMAQUANT_STAGE_A_KERNEL_PROFILE"
KERNEL_PROFILE_NOT_MEASURED = (
    "not measured: a whole-capture torch.profiler session holds every CUDA "
    "record in host memory until its stop (PQ #899); "
    f"{KERNEL_PROFILE_ENV}=1 opts in")
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
    harvest and must not mutate owners while the checkpoint writer
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


def write_checkpoint_with_snapshot(storage, space, *, boundary, session, cotangents,
                                   shared_pass, plane=None, attempt=None,
                                   batch_offset=0) -> dict:
    """Snapshot + hold + writer lifetime the actual Stage-A caller uses.

    Zero-copy borrowed snapshot when every accumulator is already CPU
    contiguous; otherwise hold the precomputed exceptional-owner copy bytes
    BEFORE any ``state_dict`` materialization, copy only exceptional owners
    inside that hold while borrowing contiguous owners, keep the hold
    across the synchronous write, and release copies before releasing the
    hold (snapshot cleared inside the hold). Returns the checkpoint record.
    Meta/non-strided/quiescence fail closed before any hold or copy.
    ``seal_checkpoint`` and the budget regressions call this one
    operation -- no test-only execution path.

    Exactly one of ``plane`` and ``attempt``. With ``attempt`` (Stage A,
    RobTand/prismaquant#1002) the cotangents are already written, as the
    roll produced them, and this writes the shared states and seals; a
    failure retains the attempt. With ``plane`` it is
    ``write_adjoint_checkpoint``'s read-back writer.

    ``batch_offset`` (a chain split quantum, PQ #738) is the global index of
    ``cotangents``' first batch: the snapshot is keyed by global batch, as
    the checkpoint names every entry.
    """
    from .joint_adjoint_checkpoints import write_adjoint_checkpoint

    if (plane is None) == (attempt is None):
        raise RuntimeError("a checkpoint write takes exactly one of a plane and an attempt")

    def write(snapshot):
        if attempt is None:
            return write_adjoint_checkpoint(
                space, boundary=boundary, session=session,
                cotangents=plane, shared_adjoint=snapshot,
                shared_pass=shared_pass, owner=storage)
        if attempt.boundary != int(boundary):
            raise RuntimeError(
                f"checkpoint attempt {attempt.boundary} cannot seal boundary {boundary}")
        try:
            attempt.write_shared_states(snapshot, shared_pass)
            return attempt.seal()
        except BaseException:
            attempt.abandon()
            raise

    needs_copy, copy_bytes = shared_adjoint_copy_plan(cotangents)
    if not needs_copy:
        snapshot = shared_adjoint_snapshot(cotangents)
        if batch_offset:
            snapshot = {(probe, batch_offset + batch): state
                        for (probe, batch), state in snapshot.items()}
        try:
            return write(snapshot)
        finally:
            snapshot.clear()
    with storage.hold_transient_metadata(copy_bytes, "shared-adjoint CPU snapshot"):
        snapshot = {}
        try:
            for probe in range(len(cotangents)):
                for batch in range(len(cotangents[probe])):
                    owner = cotangents[probe][batch]
                    owner_needs, _ = owner.snapshot_copy_plan()
                    snapshot[(probe, batch_offset + batch)] = (
                        owner.state_dict() if owner_needs else owner.borrowed_state_dict())
            return write(snapshot)
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


def _require_chain_regime_fits(regime, storage_policy, *, n_probes, entry_bytes):
    """Refuse a chain regime the sealed boundary policy cannot run.

    A batch group never spans two read windows, so the batch size divides
    ``prefetch_batches``; a fused window holds each batch's boundary entry
    and every probe's incoming entry inside ``max_resident_bytes``
    (``fused_window_size``). ``render_free_layer_roll`` refuses the same
    things; this asks before the capture instead of after it.
    ``entry_bytes`` is called only for a fused regime.
    """
    from .cost_streaming import fused_window_size

    window = int(storage_policy["prefetch_batches"])
    if window % int(regime["batch_size"]):
        raise ChainRegimeRefused(
            f"chain batch size {regime['batch_size']} does not divide the "
            f"sealed read window of {window} batches")
    if regime["probe_fusion"]:
        fused_window_size(
            prefetch_batches=window,
            max_resident_bytes=storage_policy["max_resident_bytes"],
            per_batch_bytes=(1 + int(n_probes)) * int(entry_bytes()),
            batch_size=regime["batch_size"], write_bytes=int(entry_bytes()))


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
    readback_allowance_bytes: int | None = None,
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

    ``readback_allowance_bytes`` is set for an owner that reads its groups
    back from a local spool (:func:`stage_a_owner_reads_back_locally`,
    PQ #1120). Such an owner never commits its groups, so each group's
    durable charge stays at its prewrite ceiling, every entry at the bound
    entry size plus the 64 KiB envelope, for the group's whole life. For it
    the allowance, not the floor, is the gate: it is the planning allowance
    with every entry at the full batch's size (:func:`_run_artifact_preflight`),
    and a budget below it refuses here, naming the floor and the allowance,
    where it would otherwise pass and fail with
    ``prewrite-exceeds-payload-maxima`` hours into the forward.
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
    if readback_allowance_bytes is not None and (
            type(readback_allowance_bytes) is not int
            or readback_allowance_bytes < planning):
        raise AdjointIdentityRefused(
            "stage A preflight read-back allowance must be an integer at or "
            f"above the planning allowance {int(planning)}, got "
            f"{readback_allowance_bytes!r}")
    sealed_note = (f" (plan sealed {int(plan_sealed_bytes)} bytes)"
                   if plan_sealed_bytes is not None else "")
    if int(declared_bytes) >= int(floor) and (
            readback_allowance_bytes is None
            or int(declared_bytes) >= int(readback_allowance_bytes)):
        return {"declared_bytes": int(declared_bytes),
                "required_floor_bytes": int(floor),
                "planning_estimate_bytes": int(planning),
                "readback_allowance_bytes": (
                    None if readback_allowance_bytes is None
                    else int(readback_allowance_bytes))}
    if int(declared_bytes) >= int(floor):
        allowance = int(readback_allowance_bytes)
        raise AdjointIdentityRefused(
            "stage A artifact budget below the read-back allowance: this "
            "owner reads its groups back from its local output spool and "
            "never commits them, so PrismaBuild keeps each group charged at "
            "its prewrite ceiling, every entry at the bound entry size plus "
            f"{ARTIFACT_FILE_HEADER_BYTES} B (PQ #1110, #1120). Need >= "
            f"{allowance} bytes ({allowance / 1024 ** 3:.2f} GiB, the "
            "planning allowance with every entry at the full batch's size); "
            f"the hard floor is {int(floor)} bytes "
            f"({int(floor) / 1024 ** 3:.2f} GiB raw tensor bytes) and the "
            f"planning allowance {int(planning)} bytes; declared "
            f"{int(declared_bytes)} bytes "
            f"({int(declared_bytes) / 1024 ** 3:.2f} GiB){sealed_note}. "
            f"Remedy: re-invoke with --artifact-budget-bytes {allowance} (or "
            f"{ARTIFACT_BUDGET_ENV}={allowance}); the sealed plan is "
            "unchanged and the override is stamped into the run provenance.")
    remedy = int(planning if readback_allowance_bytes is None
                 else readback_allowance_bytes)
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
        f"--artifact-budget-bytes {remedy} (or "
        f"{ARTIFACT_BUDGET_ENV}={remedy}); the sealed plan is "
        "unchanged and the override is stamped into the run provenance.")


def stage_a_owner_reads_back_locally(environ=None) -> bool:
    """Whether this launch's Stage A owner reads back from a local spool.

    Stage A binds its produced-output owner from the launch environment
    (:func:`bind_stage_a_produced_output`): an admitted action, one with a
    ``PRISMABUILD_ACTION_KEY``, binds one, and its local output spool is the
    root that environment seals (``produced_output_spool.sealed_spool_root``,
    which ``ProducedOutputSpool.from_publication`` reads). The owner always
    reads its groups back: the capture passes no ``origin_lifetime``, which
    a write-only template requires
    (``StreamedBoundaryArtifacts.bind_produced_output``). Such an owner reads
    the groups it wrote from its own box and never commits them, so each
    group stays charged at its prewrite ceiling (PQ #1110, #1120).
    ``environ`` defaults to ``os.environ``, the environment the owner binds
    from.
    """
    source = os.environ if environ is None else environ
    if not source.get("PRISMABUILD_ACTION_KEY"):
        return False
    return sealed_spool_root(source) is not None


def _run_artifact_preflight(runner, calib_ids, execution, stride_value,
                            space, artifact, *, environ=None) -> dict:
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

    An owner that reads back from a local spool
    (:func:`stage_a_owner_reads_back_locally`, read from ``environ``) is
    gated on its read-back allowance, not the floor (PQ #1120): the planning
    allowance with a partial last batch priced as a full one, because
    PrismaBuild reserves every entry of a group at the bound entry size, the
    full batch's (``boundary_group_ceiling_bytes``).
    """
    _n_probes = int(execution["n_probes"])
    _probe_microbatch = int(execution.get("probe_microbatch", 0))
    _n_rows = int(calib_ids.shape[0])
    _seqlen = int(calib_ids.shape[1])
    _batch_rows, _ = plane_partitions(n_rows=_n_rows,
                                      probe_microbatch=_probe_microbatch)
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
    def _estimate(per_remainder):
        return estimate_stage_a_artifact_demand(
            n_probes=int(_n_probes), n_full_batches=int(_n_full),
            remainder_rows=int(_rem_rows),
            per_full_tensor_nbytes=int(_per_full),
            per_remainder_tensor_nbytes=per_remainder,
            num_layers=int(runner.num_layers), stride=int(stride_value),
            header_bytes=int(ARTIFACT_FILE_HEADER_BYTES),
            shared_per_checkpoint_bytes=int(_aux_bound),
            manifest_per_checkpoint_bytes=int(_manifest_bytes))

    _demand = _estimate(_per_rem)
    _readback_allowance = None
    if stage_a_owner_reads_back_locally(environ):
        # Every entry of a group is reserved at the bound entry size, the
        # full batch's, and never committed down to its actual bytes.
        _readback_allowance = int(_estimate(
            int(_per_full) if _rem_rows else None)["planning_estimate_bytes"])
    _preflight = preflight_stage_a_artifact_budget(
        declared_bytes=int(artifact["run_used"]), demand=_demand,
        plan_sealed_bytes=int(artifact["override"]["plan_sealed_bytes"]
                              if artifact["override"] is not None
                              else int(artifact["run_used"])),
        readback_allowance_bytes=_readback_allowance)
    _gate = ("floor" if _readback_allowance is None
             else f"read-back allowance {_readback_allowance} bytes and floor")
    print(f"joint_cost_stage_a: artifact preflight: declared "
          f"{_preflight['declared_bytes']} bytes "
          f"({_preflight['declared_bytes'] / 1024 ** 3:.2f} GiB) >= {_gate} "
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
        "readback_allowance_bytes": _preflight["readback_allowance_bytes"],
        "demand": {k: _demand[k] for k in (
            "boundaries", "n_checkpoints", "n_retained_boundary_groups",
            "n_full_batches", "remainder_rows", "n_batches_total",
            "n_probes", "per_full_tensor_nbytes",
            "per_remainder_tensor_nbytes",
            "shared_per_checkpoint_bytes",
            "manifest_per_checkpoint_bytes")},
    }


def _resumed_chain_inputs(plan, *, recovery, n_batches, n_probes, samples=None):
    """What a resumed chain reads, checked before anything is removed (PQ #1001).

    Returns ``(boundary_rows, own_boundaries, checkpoint_plane)``: every
    forward boundary reference by layer, the ones this generation wrote
    (the rest are the capsule's, already installed), and the sealed
    checkpoint's activation cotangent references by ``(probe, batch)``.

    ``samples`` (a chain split quantum, PQ #738) is the ``(start, stop)`` of
    global batches this owner rolls. The sealed checkpoint is still a whole
    plane; the owner borrows only its range's boundaries and cotangents,
    keyed by global batch.
    """
    from .joint_adjoint_checkpoints import reference_from_record

    document = plan.document
    num_layers = int(document["num_layers"])
    table = document["boundary_entries"]
    recovered = {} if recovery is None else recovery.records
    rows = {}
    for boundary in range(num_layers):
        column = table[str(boundary)]
        if [row["name"] for row in column] != [
                f"boundary-{batch}-{boundary}-at-{boundary}" for batch in range(n_batches)]:
            raise AdjointIdentityRefused(
                f"chain resume refused: the chain state's boundary {boundary} is not "
                f"{n_batches} batches in order")
        if str(boundary) in recovered and recovered[str(boundary)] != column:
            raise AdjointIdentityRefused(
                f"chain resume refused: the chain state's boundary {boundary} is not the "
                "capsule's")
        rows[boundary] = [reference_from_record(row) for row in column]
    own = [reference for boundary in range(num_layers) if str(boundary) not in recovered
           for reference in rows[boundary]]
    record = plan.checkpoints[-1]
    if record["boundary"] != plan.boundary:
        raise AdjointIdentityRefused(
            "chain resume refused: the resume checkpoint is not the lowest sealed one")
    try:
        rows_by_key = checkpoint_cotangent_plane(record)
    except ValueError as exc:
        raise AdjointIdentityRefused(f"chain resume refused: {exc}") from exc
    plane = {key: reference_from_record(row) for key, row in rows_by_key.items()}
    if set(plane) != {(probe, batch) for probe in range(n_probes)
                      for batch in range(n_batches)}:
        raise AdjointIdentityRefused(
            f"chain resume refused: checkpoint {plan.boundary} is not a whole "
            f"{n_probes} x {n_batches} cotangent plane")
    if samples is not None:
        start, stop = samples
        own = [reference for boundary in range(num_layers) if str(boundary) not in recovered
               for reference in rows[boundary][start:stop]]
        plane = {(probe, batch): reference for (probe, batch), reference in plane.items()
                 if start <= batch < stop}
    return rows, own, plane


def _restore_resumed_chain(runner, storage, space, plan, *, inputs, partitions, n_probes,
                           batch_offset=0):
    """The chain's state at the sealed checkpoint a resume starts from (PQ #1001).

    Rebuilt the way a layer quantum rebuilds its chain from a checkpoint
    (``joint_cost_quantum._rebuild_batches`` and ``load_state_dict``), so a
    resumed chain runs the arithmetic a quantum's chain is already tested to
    run bitwise. The one difference: the checkpoint's activation cotangents
    stay references, read one bounded window at a time by the first roll,
    instead of a plane loaded into memory.

    Returns ``(batches, cotangents, grad_outs)`` as the uninterrupted run
    holds them right after sealing checkpoint ``plan.boundary``. With
    ``batch_offset`` (a split quantum, PQ #738) they are the uninterrupted
    run's for the ``len(partitions)`` batches from that global index.
    """
    from .joint_adjoint_checkpoints import load_checkpoint_shared_states

    rows, own, plane = inputs
    num_layers = int(plan.document["num_layers"])
    storage.authorize_resume_inputs(own, list(plane.values()), boundary=plan.boundary)
    storage.adopt_committed_checkpoints(plan.checkpoints, plan.checkpoint_directories)
    shared_adjoint, shared_pass = load_checkpoint_shared_states(
        space, plan.checkpoints[-1],
        shared_state_max_bytes=storage.config["max_auxiliary_bytes"])
    return _chain_at_checkpoint(
        runner, storage, rows=rows, plane=plane, shared_adjoint=shared_adjoint,
        shared_pass=shared_pass, partitions=partitions, n_probes=n_probes,
        num_layers=num_layers, batch_offset=batch_offset)


def _restore_seeded_chain(runner, storage, plan, *, partitions, n_probes, num_layers):
    """The chain's state at another run's sealed checkpoint (PQ #1016).

    The seed's fresh owner borrows the capsule rows ``plan.through .. b - 1``
    (:meth:`authorize_forward_inputs`) and checkpoint ``b``'s cotangent plane
    under that checkpoint's own session (:meth:`authorize_seed_checkpoint`),
    then rebuilds the chain exactly as a chain resume does. Boundaries the
    seed never reads hold ``None``: the chain stops at ``plan.through``.
    """
    from .joint_adjoint_checkpoints import load_checkpoint_shared_states, reference_from_record

    rows = {boundary: [reference_from_record(row) for row in column]
            for boundary, column in plan.rows.items()}
    plane = {key: reference_from_record(row) for key, row in plan.plane.items()}
    storage.authorize_forward_inputs(
        [reference for column in rows.values() for reference in column])
    storage.authorize_seed_checkpoint(list(plane.values()), boundary=plan.boundary,
                                      session=plan.record["session"])
    shared_adjoint, shared_pass = load_checkpoint_shared_states(
        plan.source_space, plan.record,
        shared_state_max_bytes=storage.config["max_auxiliary_bytes"])
    return _chain_at_checkpoint(
        runner, storage, rows=rows, plane=plane, shared_adjoint=shared_adjoint,
        shared_pass=shared_pass, partitions=partitions, n_probes=n_probes,
        num_layers=num_layers)


def _chain_at_checkpoint(runner, storage, *, rows, plane, shared_adjoint, shared_pass,
                         partitions, n_probes, num_layers, batch_offset=0):
    """``(batches, cotangents, grad_outs)`` right after a checkpoint is sealed.

    ``rows`` maps each forward boundary the chain reads to its references by
    batch; ``plane`` maps ``(probe, batch)`` to the checkpoint's cotangent
    reference. Both, and the shared states, are keyed by global batch; the
    lists returned hold ``len(partitions)`` batches from ``batch_offset``
    (0 but for a split quantum, PQ #738), in order.
    """
    from .joint_cost_quantum import _rebuild_batches
    from .sensitivity_probe import SharedStateCotangents, kv_cotangent_path_enabled

    n_batches = len(partitions)
    batches = _rebuild_batches(
        runner, partitions=partitions,
        shared_pass={batch - batch_offset: state for batch, state in shared_pass.items()
                     if batch_offset <= batch < batch_offset + n_batches})
    for batch_index, batch in enumerate(batches):
        batch.activations_cpu = [rows[boundary][batch_offset + batch_index]
                                 if boundary in rows else None
                                 for boundary in range(num_layers)] + [torch.empty(0)]
    cotangents = [[SharedStateCotangents(enabled=kv_cotangent_path_enabled())
                   for _ in batches] for _ in range(n_probes)]
    for (probe, batch), state in shared_adjoint.items():
        if batch_offset <= batch < batch_offset + n_batches:
            cotangents[probe][batch - batch_offset].load_state_dict(state)
    grad_outs = [[plane[(probe, batch_offset + batch)] for batch in range(n_batches)]
                 for probe in range(n_probes)]
    storage.watch_auxiliary(batches, cotangents)
    storage.check_auxiliary(batches, cotangents=cotangents)
    return batches, cotangents, grad_outs


def run_adjoint_capture_core(
    runner, calib_ids, *, execution, output_root, stride,
    source_model_identity, unit_roster_sha256, plan_sha256, prepared_sha256,
    read_manifest_sha256, implementation_sha256, campaign_scope=None,
    boundary_artifact_bytes=None, artifact_budget_stamp=None,
    min_free_gib=0.0, progress=None, produced_output=None, forward_recovery=None,
    chain_batch_size=1, chain_probe_fusion=False, chain_resume=None,
    arithmetic_extra=None, chain_seed=None, chain_split=None,
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

    ``chain_batch_size`` and ``chain_probe_fusion`` are the chain regime
    (RobTand/prismaquant#997, ``render_free_layer_roll``). A non-default
    regime is stamped into the receipt's ``run_identity`` under
    ``chain_regime``, so every band and slice carries it and a quantum
    rebuilds its chain with the same batch size. The default stamps nothing.

    ``chain_resume`` (optional) relaunches this same run from its lowest
    sealed checkpoint (RobTand/prismaquant#1001, ``stage_a_chain_resume``):
    ``{"chain_state_sha256", "declaration", "resume_from"}``. The run's own
    header, session and forward boundaries are adopted from the sealed chain
    state a fresh run writes after its tail checkpoint; the forward capture
    and the tail are not run again. ``implementation_sha256`` is always the
    running implementation: under a resume the header keeps the one the
    chain state sealed, and a different running implementation needs the
    declaration. ``arithmetic_extra`` adds the entry point's fields (the
    container image, the projection backend) to the arithmetic stamp the
    chain state seals.

    ``chain_seed`` (optional, dev mode) is a normalized
    ``stage_a_chain_seed`` spec (RobTand/prismaquant#1016): a measurement
    run in a scratch root that borrows another run's sealed checkpoint and
    its capsule's forward rows under pinned digests, rolls the chain down to
    the spec's ``through`` boundary and returns a seed receipt instead of an
    adjoint receipt. It runs no forward pass and no tail, writes no chain
    state, and refuses beside ``chain_resume`` or ``forward_recovery``.

    ``chain_split`` (with ``chain_resume``, PQ #738) runs one part of a
    resumed chain split by sample (``stage_a_chain_split``). A **prep**
    (``{role: prep, through, ranges}``) seals the round's resume record and
    removes the named ranges' rolling entries, and rolls nothing. A
    **quantum** (``{role: quantum, through, samples, digest_layer}``)
    rebinds the run's session as one owner of the global batches
    ``samples``, rolls them from the lowest sealed checkpoint to
    ``through``, seals a partial checkpoint of its range at every stride
    checkpoint on the way, and returns a quantum receipt. The join publishes
    each whole checkpoint from the partials.
    """
    from .cost_streaming import (
        StreamedBoundaryArtifacts,
        normalize_boundary_storage,
        prefetched_boundary_batches,
        validate_streamed_model_identity,
    )
    from .kl_fisher import ROW_PROBE_LAYOUT, fisher_probe_scalar
    from .matmul_arithmetic import bf16_reduction_stamp
    from .sensitivity_probe import SharedStateCotangents, kv_cotangent_path_enabled
    from .stage_a_chain_resume import (
        RESUME_COMPATIBILITY_KEY,
        ChainResumeRefused,
        apply_chain_resume,
        build_chain_state,
        chain_arithmetic_stamp,
        chain_state_path,
        load_chain_state,
        plan_chain_resume,
        producer_binding,
        resume_declarations,
        resume_directory,
        write_chain_state,
    )
    from .stage_a_chain_seed import (
        SEED_RECEIPT_SCHEMA,
        ChainSeedRefused,
        compare_seed_plane,
        plan_chain_seed,
        preflight_seed_root,
        tensor_payload_sha256,
        write_seed_marker,
    )

    split = None
    if chain_split is not None:
        from .stage_a_chain_split import ChainSplitRefused, normalize_chain_split
        try:
            split = normalize_chain_split(chain_split)
        except ChainSplitRefused as exc:
            raise AdjointIdentityRefused(f"chain split refused: {exc}") from exc
        if chain_resume is None or chain_seed is not None:
            raise AdjointIdentityRefused(
                "chain split refused: a split continues a run's own sealed chain, so it "
                "is a chain resume and never a seed")
    quantum = split is not None and split["role"] == "quantum"
    if chain_seed is not None:
        if chain_resume is not None or forward_recovery is not None:
            raise AdjointIdentityRefused(
                "chain seed refused: a seed binds a fresh scratch run; it is neither a "
                "chain resume nor a forward recovery")
        # Before the first mkdir: a scratch root inside the source run's root
        # must refuse without creating a directory there (PQ #1016).
        try:
            preflight_seed_root(output_root, chain_seed)
        except ChainSeedRefused as exc:
            raise AdjointIdentityRefused(f"chain seed refused: {exc}") from exc
    chain_regime = normalize_chain_regime(chain_batch_size, chain_probe_fusion)
    regime_identity = chain_regime_identity(chain_regime)
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

    # One entry per batch in each probe's plane; the dispatcher seals the
    # spool window from the same partition (PQ #1121).
    batch_rows, row_offsets = plane_partitions(
        n_rows=len(calib_ids), probe_microbatch=probe_microbatch)
    # The regime must fit the sealed read window before anything is
    # captured: a refusal at the first chain layer would come after the
    # whole forward capture (RobTand/prismaquant#997).
    _require_chain_regime_fits(
        chain_regime, storage_policy, n_probes=n_probes,
        entry_bytes=lambda: _stage_a_per_tensor_nbytes(
            runner, batch_rows=batch_rows, seqlen=int(calib_ids.shape[1])))
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

    # Under a chain resume the run header keeps the implementation the chain
    # state sealed; the running one is only compared (PQ #1001).
    chain_state = None
    if chain_resume is not None:
        try:
            chain_state = load_chain_state(space, chain_resume["chain_state_sha256"])
        except ChainResumeRefused as exc:
            raise AdjointIdentityRefused(f"chain resume refused: {exc}") from exc
    else:
        # A fresh run writes the chain state once, after its tail, and seals
        # its receipt over the resume records it finds: another run's must
        # refuse now, not after the forward pass (PQ #1001).
        stale = [path for path in (chain_state_path(space), resume_directory(space))
                 if path.exists()]
        if stale:
            raise AdjointIdentityRefused(
                "stage A output root already holds another run's chain state or "
                f"resume records ({', '.join(str(path) for path in stale)}): relaunch "
                "that run with --resume-chain-state-sha256, or rename them aside")
    header_implementation = (implementation_sha256 if chain_state is None
                             else chain_state["run_identity"]["implementation_sha256"])
    bind_identity = {
        "source_model": validate_streamed_model_identity(
            source_model_identity, where="adjoint capture"),
        "producer_source_sha256": header_implementation,
        "calibration_sha256": hashlib.sha256(
            calib_ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "calibration_shape": list(calib_ids.shape),
        "calibration_dtype": str(calib_ids.dtype),
        "n_probes": n_probes, "seed_base": seed_base,
        "token_scope": token_scope, "temperature": temperature,
        "execution_partition": execution_partition,
        "campaign_stage": "joint_adjoint_capture",
    }
    run_identity = {
        "plan_sha256": str(plan_sha256),
        "prepared_sha256": str(prepared_sha256),
        "read_manifest_sha256": str(read_manifest_sha256),
        "implementation_sha256": str(header_implementation),
        "unit_roster_sha256": str(unit_roster_sha256),
        "campaign_scope": campaign_scope,
        "n_probes": n_probes,
        "seed_base": seed_base,
        "calibration_shape": list(calib_ids.shape),
        "calibration_sha256": bind_identity["calibration_sha256"],
        **({CHAIN_REGIME_KEY: regime_identity}
           if regime_identity is not None else {}),
        # Absent at PyTorch's default, so a default run's identity keeps its
        # bytes (PQ #1028).
        **bf16_reduction_stamp(),
    }
    stride_block = {"value": int(stride), "boundaries": [int(b) for b in boundaries],
                    "max_chain_layers": int(stride) - 1}
    arithmetic = chain_arithmetic_stamp(runner, arithmetic_extra)
    resume_plan = None
    if chain_state is not None:
        from .cost_stage_checkpoint import canonical_json
        try:
            resume_plan = plan_chain_resume(
                space, chain_state, recomputed=canonical_json({
                    "run_identity": run_identity, "stride": stride_block,
                    "bind_identity": bind_identity, "arithmetic": arithmetic,
                    "n_batches": len(row_offsets), "num_layers": num_layers,
                    "artifact_budget_override": artifact_budget_stamp,
                    "boundary_policy": storage.identity,
                    "boundary_directory": str(boundary_entry_directory(space)),
                }, where="Stage A chain resume"),
                running_implementation_sha256=implementation_sha256,
                declaration=chain_resume.get("declaration"),
                resume_from=chain_resume.get("resume_from"))
        except ChainResumeRefused as exc:
            raise AdjointIdentityRefused(f"chain resume refused: {exc}") from exc
    split_marks = ()
    batch_offset = 0
    samples = None
    label = None
    if split is not None:
        from .stage_a_chain_split import (
            ChainSplitRefused, check_ranges, partial_directory, quantum_label,
            split_boundaries)
        group_size = int(storage_policy["prefetch_batches"])
        try:
            split_marks = split_boundaries(boundaries, resume_plan.boundary,
                                           split["through"])
            if quantum:
                samples = tuple(split["samples"])
                check_ranges([samples], n_batches=len(row_offsets), group_size=group_size,
                             where="the quantum's")
                if split["digest_layer"] is not None and not (
                        split["through"] <= split["digest_layer"] < resume_plan.boundary):
                    raise ChainSplitRefused(
                        f"the quantum rolls layers {resume_plan.boundary - 1} down to "
                        f"{split['through']}; it never writes layer {split['digest_layer']}")
        except ChainSplitRefused as exc:
            raise AdjointIdentityRefused(f"chain split refused: {exc}") from exc
        if quantum:
            batch_offset = samples[0]
            label = quantum_label(resume_plan.boundary, split["through"], *samples)
        else:
            return _split_prep(space, resume_plan, split, boundaries=boundaries,
                               n_batches=len(row_offsets), group_size=group_size)
    seed_plan = None
    if chain_seed is not None:
        try:
            seed_plan = plan_chain_seed(
                space, output_root, chain_seed, bind_identity=bind_identity,
                campaign_identity={
                    "plan_sha256": plan_sha256, "prepared_sha256": prepared_sha256,
                    "read_manifest_sha256": read_manifest_sha256,
                    "unit_roster_sha256": unit_roster_sha256,
                    "campaign_scope": campaign_scope},
                running_implementation_sha256=implementation_sha256,
                n_batches=len(row_offsets), n_probes=n_probes, num_layers=num_layers)
        except ChainSeedRefused as exc:
            raise AdjointIdentityRefused(f"chain seed refused: {exc}") from exc

    def boundary_storage_block(recovery):
        return {
            "session": storage.session,
            "policy": storage.identity,
            "directory": str(boundary_entry_directory(space)),
            **({"forward_recovery": recovery.receipt_binding}
               if recovery is not None else {}),
        }

    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)

    operator_windows = execution.get("operator_windows")
    checkpoints: list[dict] = []
    chain_telemetry: list[dict] = []
    started = time.time()
    # Read, never set, here: ``matmul_arithmetic.pin_matmul_arithmetic``
    # sets it at the entry point, and the run identity carries it when it is
    # off (PQ #1028). A seed also records the value its chain ran under
    # (PQ #1038).
    bf16_reduction = bool(
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    capture_started = None
    tail_started = None
    chain_started = None
    chain_backwards = 0

    def log(message: str) -> None:
        print(f"joint_cost_stage_a: {message}", flush=True)

    resume_record = None
    if seed_plan is not None:
        # Before the bind creates anything: this root is a seed's for good,
        # and the band tool refuses it (RobTand/prismaquant#1016).
        write_seed_marker(space, seed_plan, run_identity=run_identity)
    # A split quantum keeps its own record: several quanta fail independently.
    failure_records = (None if label is None
                       else _split_quantum_path(space, label, "produced-output.failures.jsonl"))
    with _failed_produced_output_record(space, storage, path=failure_records), storage:
        if resume_plan is None:
            storage.bind(bind_identity, n_probes=n_probes, published=True)
        else:
            # A split quantum is one owner of the run's generation among
            # several: its status is its own file (PQ #738).
            storage.rebind(resume_plan.session, identity=bind_identity,
                           n_probes=n_probes, owner_label=label)
        if produced_output is not None:
            # After bind, because the entry directory this owner chose is
            # what must sit inside the publication's bound output prefix;
            # the binding refuses an own-generation path outside it here
            # rather than at the first descriptor.
            storage.bind_produced_output(
                produced_output,
                group_size=int(storage_policy["prefetch_batches"]),
                # The entries a plane holds, one per batch, not the rows: a
                # microbatched plan writes fewer (PQ #1121).
                n_batches=len(row_offsets),
                max_entry_tensor_bytes=_stage_a_per_tensor_nbytes(
                    runner, batch_rows=batch_rows,
                    seqlen=int(calib_ids.shape[1])),
                # The fused roll reads every probe's incoming group beside
                # the boundary group (RobTand/prismaquant#997).
                **({"read_order": "sample_major"}
                   if chain_regime["probe_fusion"] else {}),
                # A split quantum writes only its own range's groups (PQ #738).
                **({"batch_range": samples} if quantum else {}))
            log("boundary capture: produced-output binding "
                f"{produced_output.instance['owner_action_key']} "
                f"prefix={produced_output.output_prefix} "
                f"group={int(storage_policy['prefetch_batches'])}")
        if quantum:
            # What a later prep reads before it removes this range's rolling
            # entries: the PrismaBuild owner that writes them (PQ #738).
            storage.stamp_owner(
                producer=producer_binding(produced_output),
                split={"from": resume_plan.boundary, "through": split["through"],
                       "samples": list(samples)})
        if progress is not None:
            # The initial boundary loop reports under the declared head
            # phase until the first forward observer fires. Entering moves
            # no units: the head walk's committed total is already the
            # reporter's base, and only published entries advance it.
            from .joint_run_progress import HEAD_PHASE
            progress.enter(HEAD_PHASE)
            storage.watch_progress(progress)
        from .joint_forward_resume import load_forward_recovery
        recovery = load_forward_recovery(forward_recovery, bind_identity=bind_identity,
            campaign_identity={"plan_sha256": plan_sha256, "prepared_sha256": prepared_sha256,
                "read_manifest_sha256": read_manifest_sha256,
                "unit_roster_sha256": unit_roster_sha256, "campaign_scope": campaign_scope},
            runner=runner, storage=storage)
        if recovery is not None:
            log(f"verified forward recovery through boundary {recovery.frontier}, "
                f"{recovery.n_batches} complete calibration partitions")
        if resume_plan is not None and (
                (None if recovery is None else recovery.receipt_binding)
                != resume_plan.document["boundary_storage"].get("forward_recovery")):
            raise AdjointIdentityRefused(
                "chain resume refused: the relaunch binds another forward-recovery "
                "capsule than the run its chain state seals")

        def open_checkpoint(boundary: int):
            # Reserved before the pass that produces the plane, which writes
            # each cotangent into the checkpoint as it rolls
            # (RobTand/prismaquant#1002). Nothing is read back, so the
            # checkpoint stages no reads and the read-ahead the pass asked
            # for its successor survives it. Each cotangent is the gradient
            # of its batch's input boundary cast to the runner dtype, so the
            # boundary entry's shape and that dtype are its exact spec.
            # Keyed by global batch: a split quantum's partial names its
            # range's entries as the whole checkpoint does (PQ #738).
            itemsize = torch.empty((), dtype=runner.dtype).element_size()
            specs = {}
            for batch in range(len(batches)):
                shape = [int(dim) for dim in batches[batch].activations_cpu[boundary].shape]
                nbytes = itemsize
                for dim in shape:
                    nbytes *= dim
                for probe in range(n_probes):
                    specs[probe, batch_offset + batch] = (nbytes, shape, str(runner.dtype))
            return open_adjoint_checkpoint(
                space, boundary=boundary, session=checkpoint_session(),
                specs=specs,
                shared_adjoint_keys=[(probe, batch_offset + batch)
                                     for probe in range(len(cotangents))
                                     for batch in range(len(cotangents[probe]))],
                shared_pass_keys=range(batch_offset, batch_offset + len(batches)),
                owner=storage,
                # PQ #1036: the checkpoint names the owner's entries at this
                # boundary instead of writing a second copy of the plane.
                referenced=True,
                **({"directory": partial_directory(space, boundary, *samples)}
                   if quantum else {}))

        def checkpoint_session():
            return {"generation": storage.session["generation"],
                    "kind": "adjoint_checkpoint",
                    "run_identity_sha256": storage.session["run_identity_sha256"]}

        def seal_checkpoint(attempt) -> None:
            shared_pass = {batch_offset + batch: batches[batch].shared_pass_state
                           for batch in range(len(batches))}
            record = write_checkpoint_with_snapshot(
                storage, space, boundary=attempt.boundary,
                session=checkpoint_session(), attempt=attempt,
                cotangents=cotangents, shared_pass=shared_pass,
                **({"batch_offset": batch_offset} if quantum else {}))
            checkpoints.append(record)
            log(f"checkpoint published at boundary {attempt.boundary} "
                f"({attempt.written_activations} cotangent entries)")

        if resume_plan is not None:
            # Every check first; the interrupted attempt's working entries
            # are removed only once the relaunch is known to continue it.
            inputs = _resumed_chain_inputs(resume_plan, recovery=recovery,
                                           n_batches=len(row_offsets), n_probes=n_probes,
                                           samples=samples)
            offsets = (row_offsets if samples is None
                       else row_offsets[samples[0]:samples[1]])
            if not quantum:
                resume_record = apply_chain_resume(
                    space, resume_plan, producer=producer_binding(produced_output))
            batches, cotangents, grad_outs = _restore_resumed_chain(
                runner, storage, space, resume_plan, inputs=inputs,
                partitions=[calib_ids[offset:offset + batch_rows] for offset in offsets],
                n_probes=n_probes, batch_offset=batch_offset)
            chain_top = resume_plan.boundary
            if quantum:
                # The prep ran the round's resume record once (PQ #738).
                log(f"chain split quantum {label}: batches {samples[0]}:{samples[1]} "
                    f"below sealed checkpoint {chain_top} down to {split['through']}")
            else:
                checkpoints.extend(resume_plan.checkpoints)
                log(f"chain resume {resume_record['index']}: continuing below sealed "
                    f"checkpoint {chain_top}; removed "
                    f"{resume_record['removed_rolling_entries']} rolling entries, set "
                    f"aside {len(resume_record['partial_checkpoints_set_aside'])} partial "
                    "checkpoint directories")
        elif seed_plan is not None:
            batches, cotangents, grad_outs = _restore_seeded_chain(
                runner, storage, seed_plan,
                partitions=[calib_ids[offset:offset + batch_rows]
                            for offset in row_offsets],
                n_probes=n_probes, num_layers=num_layers)
            chain_top = seed_plan.boundary
            log(f"chain seed: continuing checkpoint {chain_top} of "
                f"{seed_plan.source_root} down to boundary {seed_plan.through} "
                f"(sealed by implementation {seed_plan.sealed_by[:12]})")
        else:
            chain_top = num_layers
        # A seed stops at its ``through`` boundary, and so does a split
        # quantum (PQ #738); every other run rolls to 0.
        chain_bottom = (seed_plan.through if seed_plan is not None
                        else split["through"] if quantum else 0)
        # A seed also seals its last plane, stride boundary or not: the plane
        # at ``through`` is its result, and the end of the walk retires the
        # rolling entries. The stride block and run identity keep the plan's
        # boundaries (RobTand/prismaquant#997). A split quantum seals a
        # partial at every stride checkpoint of its round.
        checkpoint_layers = (set(split_marks) if quantum
                             else set(boundaries) if seed_plan is None
                             else set(boundaries) | {seed_plan.through})
        # A quantum's payload digests at one named layer, as it writes them:
        # the bitwise check against another run's plane at that layer.
        layer_digests = ({} if quantum and split["digest_layer"] is not None
                         else None)
        # The seed's rolled plane at its compare boundary, hashed as written.
        plane_digests = ({} if seed_plan is not None and seed_plan.compare is not None
                         else None)
        if resume_plan is None and seed_plan is None:
            log(f"boundary capture: calib {tuple(calib_ids.shape)} in "
                f"{len(row_offsets)} partition(s) across {num_layers} layers ...")
            capture_started = time.time()
            batches = runner.capture_layer_major_boundaries(
                [calib_ids[offset:offset + batch_rows] for offset in row_offsets],
                storage=storage,
                **({"forward_recovery": recovery} if recovery is not None else {}),
                source_phase=(stage_a_forward_observer(progress)
                              if progress is not None else None))
            log(f"boundary capture done in {(time.time() - capture_started) / 60:.1f} min; "
                f"starting {n_probes}-probe tail cotangents")

            device, dtype = runner.device, runner.dtype
            cotangents = [[SharedStateCotangents(enabled=kv_cotangent_path_enabled())
                           for _ in batches] for _ in range(n_probes)]
            grad_outs = [[] for _ in range(n_probes)]
            storage.watch_auxiliary(batches, cotangents)
            storage.check_auxiliary(batches, cotangents=cotangents)
            tail_started = time.time()
            # The tail set is the first checkpoint: layer num_layers-1's quantum
            # chains nothing (§3.1). It is written as the tail produces it.
            tail_checkpoint = open_checkpoint(num_layers)
            try:
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
                                grad_outs[probe_index].append(storage.write(
                                    tail.grad, batch_index=batch_index,
                                    boundary_index=num_layers, probe_index=probe_index))
                                tail_checkpoint.reference_activation(
                                    probe_index, batch_index,
                                    grad_outs[probe_index][-1])
                                del logits, probe, tail
                            storage.retire(batch.activations_cpu[-1])
                            batch.activations_cpu[-1] = torch.empty(0)
                        finally:
                            tail_cpu = logits = probe = tail = None
                log(f"tail cotangents done in {(time.time() - tail_started) / 60:.1f} min; "
                    f"sealing the tail checkpoint at boundary {num_layers}")
                seal_checkpoint(tail_checkpoint)
            except BaseException:
                tail_checkpoint.abandon()
                raise
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
            # The chain state a relaunch resumes from (PQ #1001): written
            # once, after the tail checkpoint is sealed, so its existence
            # implies the tail checkpoint's.
            write_chain_state(space, build_chain_state(
                run_identity=run_identity, stride=stride_block,
                boundary_storage=boundary_storage_block(recovery),
                bind_identity=bind_identity, arithmetic=arithmetic,
                boundary_entries={
                    str(boundary): [exact_entry_record(batch.activations_cpu[boundary])
                                    for batch in batches]
                    for boundary in range(num_layers)},
                n_batches=len(batches), num_layers=num_layers,
                artifact_budget_override=artifact_budget_stamp,
                tail_checkpoint=checkpoints[0],
                producer=producer_binding(produced_output)))

        chain_started = time.time()
        # The chain's install order, top down. Every prefetch below stays
        # inside it: a walk that ends above layer 0 (a seed, or a chain
        # through a stride boundary) declares no source below
        # ``chain_bottom``, so the strict reader refuses such a read and the
        # settlement re-raises it (PQ #1100). The next process to need the
        # layer below reads it itself.
        chain_order = list(range(chain_top - 1, chain_bottom - 1, -1))
        if chain_top > chain_bottom:
            # The chain's first layer reads a plane the forward pass wrote
            # and retired long ago; every later layer's plane is asked for
            # by the roll before it (``render_free_layer_roll``). Asking
            # here starts its movers before the first window needs them.
            storage.stage_produced_boundary_ahead(chain_top - 1)
            # The same holds for the chain's source layers: each install
            # refuses a layer that is neither resident nor in flight, and each
            # layer asks only for the layers below it. The first
            # ``lookahead`` layers are asked for here, nearest first, the
            # reverse twin of the forward pass's opening prefetches. A seed or
            # a resume runs no forward pass, so nothing else asks for them
            # (RobTand/prismaquant#997). After a fresh walk the top layers are
            # usually still resident, and ``schedule_prefetch`` owns a
            # resident layer until its install instead of reading it again
            # (RobTand/prismaquant#1124), so nothing is read twice.
            for layer in chain_opening_window(chain_order, runner.prefetch_lookahead):
                runner.context.schedule_prefetch(layer)
        for position, layer in enumerate(chain_order):
            if progress is not None:
                progress.enter(adjoint_chain_phase_name(layer))
                progress.flush(force=True)
            layer_started = time.time()
            attempt = None
            try:
                runner.context.install(
                    layer,
                    require_prefetched=runner.require_prefetched_residency,
                    **({"prefetch_following": False}
                       if operator_windows is not None else {}),
                )
                successors = chain_prefetch_window(
                    chain_order, position, runner.prefetch_lookahead)
                if operator_windows is None:
                    # The roll's one read ``lookahead`` below, inside the walk.
                    if len(successors) == runner.prefetch_lookahead:
                        runner.schedule_reverse_prefetch(layer)
                else:
                    for successor in successors:
                        runner.context.schedule_prefetch(successor)
                    settle = getattr(runner.context, "settle_prefetched_layers", None)
                    if callable(settle):
                        # Still in the chain's admitted source-loading
                        # window, before any backward workspace is opened.
                        settle(successors, retry_availability=True)
                    elif torch.device(runner.device).type == "cuda":
                        raise RuntimeError(
                            "render-free chain requires source prefetch settlement")
                summary = getattr(runner.context, "prefetch_summary", None)
                if callable(summary):
                    # The loader's counters at every chain boundary, once
                    # the successors settled and before the roll
                    # (RobTand/prismaquant#1124).
                    log(f"chain layer {layer} settled: {summary()}")
                # A checkpoint layer writes its plane into the checkpoint as
                # it rolls (RobTand/prismaquant#1002): opened here, after the
                # layer's sources settled and before any cotangent exists.
                attempt = (open_checkpoint(layer) if layer in checkpoint_layers
                           else None)

                def roll(tensor, batch_index, probe_index):
                    # ``batch_index`` is the roll's own position; the entry,
                    # its checkpoint row and every digest name the global
                    # batch (a split quantum starts at ``batch_offset``).
                    batch = batch_offset + batch_index
                    if plane_digests is not None and layer == chain_bottom:
                        plane_digests[probe_index, batch] = (
                            tensor_payload_sha256(tensor))
                    if layer_digests is not None and layer == split["digest_layer"]:
                        layer_digests[probe_index, batch] = (
                            tensor_payload_sha256(tensor))
                    # The walk's last roll: no read follows it. Its entries
                    # are retired right after the loop, or kept by the
                    # checkpoint that names them.
                    grad_outs[probe_index][batch_index] = storage.write(
                        tensor, batch_index=batch, boundary_index=layer,
                        probe_index=probe_index,
                        previous=grad_outs[probe_index][batch_index],
                        **({} if layer > chain_bottom else {"read_back": False}))
                    if attempt is not None:
                        attempt.reference_activation(
                            probe_index, batch,
                            grad_outs[probe_index][batch_index])

                chain_backwards += render_free_layer_roll(
                    runner, storage=storage, batches=batches, layer=layer,
                    cotangents=cotangents, n_probes=n_probes,
                    incoming_entries=grad_outs, incoming_tensor=None,
                    roll=roll, min_free_gib=min_free_gib,
                    # The next layer's first window reads probe 0's rolled
                    # entries; ``grad_outs`` holds them once probe 0 ran.
                    # A fused next window reads every probe's.
                    then=((layer - 1, grad_outs if chain_regime["probe_fusion"]
                           else grad_outs[0]) if layer > chain_bottom else None),
                    batch_size=chain_regime["batch_size"],
                    probe_fusion=chain_regime["probe_fusion"])
                seal_s = None
                if attempt is not None:
                    sealed = time.time()
                    seal_checkpoint(attempt)
                    seal_s = time.time() - sealed
                if layer_digests is not None and layer == split["digest_layer"]:
                    # Published the moment the layer is rolled, not with
                    # the receipt four layers later: a digest mismatch is
                    # the round's early stop (PQ #738).
                    _write_split_digests(space, label, samples,
                                         _split_digest_block(split, layer_digests))
                chain_telemetry.append({
                    "layer": layer, "wall_s": time.time() - layer_started,
                    "checkpoint": layer in checkpoint_layers,
                    "checkpoint_write_s": (None if attempt is None
                                           else attempt.write_seconds),
                    "checkpoint_seal_s": seal_s,
                    "checkpoint_reference_wait_s": (
                        None if attempt is None else attempt.reference_wait_seconds),
                })
                attempt = None
            except BaseException:
                # Files may exist: retain the attempt's envelope, never
                # release it. The owner's close disposes the directory; only
                # a killed run leaves it, for a chain resume to set aside.
                if attempt is not None:
                    attempt.abandon()
                raise
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

        storage.settle_local_output()
        boundary_entries: dict[str, list[dict]] = {}
        if seed_plan is None and not quantum:
            for boundary in range(num_layers):
                boundary_entries[str(boundary)] = [
                    exact_entry_record(batch.activations_cpu[boundary])
                    for batch in batches]
        retention = storage.receipt()

    receipt_stride = {"value": int(stride), "source": None,  # filled by caller
                      "boundaries": [int(b) for b in boundaries],
                      "max_chain_layers": int(stride) - 1}
    if quantum:
        from .stage_a_chain_split import QUANTUM_RECEIPT_SCHEMA
        # One part of a split chain: its partials are the join's inputs,
        # and it is never a band source itself (PQ #738).
        receipt = {
            "schema": QUANTUM_RECEIPT_SCHEMA,
            "entry_point": ADJOINT_CAPTURE_ENTRY_POINT,
            "status": "complete",
            "run_identity": run_identity,
            "stride": receipt_stride,
            "boundary_storage": boundary_storage_block(recovery),
            "split": {"label": label, "from": resume_plan.boundary,
                      "through": split["through"], "boundaries": list(split_marks),
                      "samples": list(samples)},
            "implementation_sha256": str(implementation_sha256),
            "partials": [{"boundary": record["boundary"],
                          "directory": str(partial_directory(
                              space, record["boundary"], *samples)),
                          "cotangent_sha256": record["cotangent_sha256"],
                          "cotangents": len(record["activation_entries"])}
                         for record in checkpoints],
            "digests": (None if layer_digests is None
                        else _split_digest_block(split, layer_digests)),
            "retention": retention,
            "artifact_budget_override": artifact_budget_stamp,
            "telemetry": {
                "started_unix": started,
                "wall_s": time.time() - started,
                "chain_wall_s": (time.time() - chain_started
                                 if chain_started else None),
                "chain_backwards": chain_backwards,
                "chain_layers": chain_telemetry,
            },
            "dev_mode": dev_mode_stamp(),
        }
        receipt["telemetry"].update(_produced_output_block(storage))
        return receipt
    if seed_plan is not None:
        # A seed is a measurement, never a campaign run: its receipt has its
        # own schema and file, and the band tool refuses its space
        # (RobTand/prismaquant#1016).
        receipt = {
            "schema": SEED_RECEIPT_SCHEMA,
            "entry_point": ADJOINT_CAPTURE_ENTRY_POINT,
            "status": "complete",
            "bandable": False,
            "run_identity": run_identity,
            "stride": receipt_stride,
            "boundary_storage": boundary_storage_block(None),
            "seed": seed_plan.binding,
            # Always spelled here, both values. The run identity carries the
            # flag only when it is off (PQ #1028), so a default seed binds
            # as it did before.
            "matmul_reduction": {
                "allow_bf16_reduced_precision_reduction": bf16_reduction},
            "checkpoints": checkpoints,
            "plane_comparison": compare_seed_plane(
                seed_plan, plane_digests,
                max_resident_bytes=int(storage_policy["max_resident_bytes"])),
            "retention": retention,
            "artifact_budget_override": artifact_budget_stamp,
            "telemetry": {
                "started_unix": started,
                "wall_s": time.time() - started,
                "chain_wall_s": (time.time() - chain_started
                                 if chain_started else None),
                "chain_backwards": chain_backwards,
                "chain_layers": chain_telemetry,
            },
            "dev_mode": dev_mode_stamp(),
        }
        receipt["telemetry"].update(_produced_output_block(storage))
        return receipt

    # Every implementation declaration of this run's resumes, outside the
    # header digest (PQ #1001). A run never resumed, or resumed only under
    # the implementation that sealed its checkpoints, carries none.
    declarations = resume_declarations(space, storage.session)
    receipt = {
        "schema": ADJOINT_RECEIPT_SCHEMA,
        "entry_point": ADJOINT_CAPTURE_ENTRY_POINT,
        "status": "complete",
        "run_identity": run_identity,
        "stride": receipt_stride,
        "boundary_storage": boundary_storage_block(recovery),
        **({RESUME_COMPATIBILITY_KEY: declarations} if declarations else {}),
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
            **({"chain_resume": {key: resume_record[key] for key in (
                "index", "switch_checkpoint", "implementation_sha256",
                "removed_rolling_entries", "partial_checkpoints_set_aside")}}
               if resume_record is not None else {}),
        },
        "dev_mode": dev_mode_stamp(),
    }
    # After the owner closed: ``retention`` above was taken inside the
    # ``with`` block, so it cannot see the settle, and a stage copy
    # PrismaBuild would not retire must be on the artifact, not only on
    # stdout.
    receipt["telemetry"].update(_produced_output_block(storage))
    return receipt


def _split_digest_block(split, layer_digests) -> dict:
    """A quantum's payload digests at its digest layer, by ``"probe-batch"``."""
    return {"layer": split["digest_layer"],
            "payload_sha256": {f"{probe}-{batch}": digest for (probe, batch), digest
                               in sorted(layer_digests.items())}}


def _write_split_digests(space, label, samples, block) -> Path:
    """Write a quantum's digest block beside its receipt, before the receipt."""
    from .stage_a_chain_split import SPLIT_DIGESTS_SCHEMA

    path = _split_quantum_path(space, label, "digests.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema": SPLIT_DIGESTS_SCHEMA, "label": label,
                "samples": list(samples), **block}
    atomic_write_bytes(path, (json.dumps(document, sort_keys=True, indent=2)
                              + "\n").encode())
    return path


def _split_quantum_path(space, label, suffix) -> Path:
    """A split quantum's own file beside its receipt (PQ #738)."""
    from .stage_a_chain_split import quantum_directory

    return quantum_directory(space) / f"{label}.{suffix}"


def _split_prep(space, plan, split, *, boundaries, n_batches, group_size) -> dict:
    """A split round's prep (PQ #738): the resume record, run once; no roll.

    ``plan_chain_resume`` has already checked the relaunch. This scopes its
    leftovers to the ranges the prep launches, removes them, sets aside
    those ranges' partials and owner status files, and seals the round's
    resume record with the split stamped in it.
    """
    import dataclasses

    from .stage_a_chain_resume import apply_chain_resume
    from .stage_a_chain_split import (
        PREP_RECEIPT_SCHEMA, ChainSplitRefused, apply_split_prep, prepare_split_prep)

    started = time.time()
    storage = plan.document["boundary_storage"]
    generation = Path(storage["directory"]) / str(storage["session"]["generation"])
    try:
        prep = prepare_split_prep(space, plan, split, boundaries=boundaries,
                                  generation_directory=generation, n_batches=n_batches,
                                  group_size=group_size)
    except ChainSplitRefused as exc:
        raise AdjointIdentityRefused(f"chain split refused: {exc}") from exc
    record = apply_chain_resume(space, dataclasses.replace(plan, leftovers=prep["leftovers"]),
                                producer=None, split=prep["stamp"])
    moved = apply_split_prep(prep, index=record["index"])
    print(f"joint_cost_stage_a: chain split prep {record['index']}: rounds "
          f"{plan.boundary} -> {split['through']} over {len(split['ranges'])} ranges; "
          f"removed {record['removed_rolling_entries']} rolling entries, set aside "
          f"{len(moved['set_aside'])} partials and owner files", flush=True)
    return {"schema": PREP_RECEIPT_SCHEMA, "status": "complete", "resume": record,
            "boundaries": prep["boundaries"], "ranges": split["ranges"],
            "set_aside": moved["set_aside"],
            "telemetry": {"started_unix": started, "wall_s": time.time() - started},
            "dev_mode": dev_mode_stamp()}


def _produced_output_block(storage) -> dict:
    """The owner's closing staging facts, as a receipt block that seals.

    This runs after the last layer, so it may only add to the receipt. The
    receipt is sealed with ``json.dumps(allow_nan=False)``: a reason
    PrismaBuild worded with something JSON cannot carry travels as its repr,
    and a report that cannot be taken or carried is recorded as its own
    error. A finished capture is never failed by its own telemetry. An
    unbound owner adds nothing, so its receipt is unchanged.
    """

    produced = getattr(storage, "produced_output_report", None)
    if not callable(produced):
        return {}
    try:
        report = produced()
        if report is None:
            return {}
        return {"produced_output": json.loads(json.dumps(
            report, sort_keys=True, default=repr, allow_nan=False))}
    except Exception as exc:
        return {"produced_output": {"report_error": repr(exc)[:400]}}


#: One JSON line per failed capture attempt in the adjoint space: the owner's
#: staging records, which a finished capture seals into its receipt instead.
FAILED_PRODUCED_OUTPUT_RECORDS = "produced-output.failures.jsonl"


@contextmanager
def _failed_produced_output_record(space, storage, *, path=None):
    """Keep the owner's staging records when the capture fails (PQ #1110).

    A finished capture seals them into its receipt
    (:func:`_produced_output_block`). A failed one writes no receipt, so the
    refusals, waits and evictions of its local output spool -- the export a
    barrier refused on and its state among them -- would otherwise survive
    only in the log. Runs after the owner has closed, and appends one line
    per attempt, so a later attempt never overwrites an earlier one's. A
    record that cannot be written is said once and never replaces the
    failure that is propagating.
    """

    try:
        yield
    except BaseException as error:
        try:
            block = _produced_output_block(storage)
            if block:
                line = json.dumps({
                    "schema": "prismaquant.stage_a.produced_output_failure.v1",
                    "unix": time.time(),
                    "error": f"{type(error).__name__}: {error}"[:2000],
                    **block}, sort_keys=True, default=repr, allow_nan=False)
                target = (Path(space) / FAILED_PRODUCED_OUTPUT_RECORDS if path is None
                          else Path(path))
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "a") as out:
                    out.write(line + "\n")
        except Exception as record_error:            # noqa: BLE001
            print("joint_cost_stage_a: the failed capture's produced-output "
                  f"record could not be written: {record_error!r}", flush=True)
        raise


def _stage_a_kernel_profiler() -> KernelTimeProfiler:
    """The capture's kernel-time profiler: not measured unless asked for.

    The GPU power sampler beside it is bounded and stays on, so the receipt
    still says how loaded the device was (principle 15). What is given up by
    default is the kernel-time sum, and the block says so rather than
    reporting a zero.
    """
    if os.environ.get(KERNEL_PROFILE_ENV) == "1":
        return KernelTimeProfiler()
    return KernelTimeProfiler(not_measured=KERNEL_PROFILE_NOT_MEASURED)


def _io_counters() -> dict:
    from .io_spans import read_proc_io

    return read_proc_io()


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


def _stage_a_device_envelope(config, *, environ=None):
    """Apply an explicitly requested allocator ceiling before GPU preparation.

    The general cache-budget variable did not enforce Stage A's allocator.
    Keep legacy calls without it unchanged; an explicit value must be finite
    and positive and can only tighten the scientific plan's post-hoc ceiling.
    CUDA runtime/native allocations remain outside Torch's allocator and are
    bounded by PrismaBuild's GPU and aggregate action backstops.
    """
    import math
    from .memory_management import enforce_device_envelope

    environ = os.environ if environ is None else environ
    raw = environ.get("PRISMAQUANT_MAX_GPU_MEM_GB")
    if raw is None or not str(raw).strip():
        return None
    try:
        gib = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage A device envelope must be positive finite GiB") from exc
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError("Stage A device envelope must be positive finite GiB")
    requested = int(gib * 1024 ** 3)
    limit = config.get("max_gpu_bytes", requested)
    if type(limit) is not int or limit <= 0:
        raise ValueError("Stage A plan device envelope must be positive integer bytes")
    applied = min(requested, limit)
    record = enforce_device_envelope("cuda", applied, where="Stage A device envelope")
    return {**record, "source": "PRISMAQUANT_MAX_GPU_MEM_GB",
            "requested_bytes": requested, "plan_max_gpu_bytes": limit}


def run_adjoint_capture(
    config, *, plan_sha256, prepared, output_root, stride=None,
    read_manifest_sha256=None, data_manifest_sha256=None, resume=False,
    prefetch_override=None, artifact_budget_bytes=None, forward_recovery=None,
    chain_batch_size=1, chain_probe_fusion=False, chain_resume=None,
    chain_seed=None, head_walk=False, chain_split=None,
) -> dict:
    """Load the head phase and run the adjoint capture (one PB action).

    The head is the prepared completion's (``stage_a_head``, PQ #1051): the
    roster, the counts and the progress base come from the completion the
    ``prepared`` binding pins, and Stage A reads none of the anchor catalog.
    ``head_walk=True`` is the verification arm: it walks the catalog as
    Stage A did before #1051 (``load_measured_anchor_input``), runs the
    walk's checks, and requires the walk's roster and cell count to equal the
    completion's. The two arms' receipts differ only in ``head.walked``.

    ``chain_resume`` relaunches the run from its lowest sealed checkpoint
    (``run_adjoint_capture_core``, PQ #1001). With ``head_walk``, it implies
    the walk's own ``resume``: the relaunch re-verifies the head journal it
    banked.

    ``chain_seed`` (a normalized ``stage_a_chain_seed`` spec, PQ #1016) runs
    the plan's own run's sealed checkpoint on into ``output_root``, a scratch
    root that is not the plan's: the plan's ``output_root`` must be the
    source run's root instead. The seed receipt is written to
    ``seed-receipt.json``, never ``adjoint-capture.json``.

    ``chain_split`` (with ``chain_resume``, PQ #738) runs one row of a split
    round. A prep's receipt goes to ``split/preps/resume-NNN.json``; a
    quantum's to ``split/quanta/<label>.json``; and each writes its results
    and counters beside it, never over the run's own.
    """
    from .aura_cost import _aura_source_sha256
    from .calibration_data import load_calibration_input
    from .cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
    from .glm_capture_compatibility import require_capture_compatibility
    from .gpu_guard import require_cuda_hot_path
    from .joint_projection_backend import executing_image, prewarm_projection_backend
    from .joint_run_progress import JointRunProgress
    from .model_profiles import detect_profile
    from .residency_map import bind_residency_manifest, residency_report
    from .stage_a_head import prepared_head, stage_a_roster, walked_head
    from .tessera_joint_aura import (
        ACTIVATION_SCALE_ENV,
        _bound,
        _pb_commit,
        _preflight_run_prepared,
        _same,
        _seed_source_identity_cache,
        load_measured_anchor_input,
    )
    from .tessera_reader import load_declared_reader

    require_cuda_hot_path("joint_cost_stage_a", "cuda")
    # A malformed bf16 reduction setting refuses before anything is pinned,
    # read or loaded (PQ #1028).
    from .matmul_arithmetic import (
        MatmulArithmeticRefused, bf16_reduction_from_environment, pin_matmul_arithmetic)
    try:
        bf16_reduction_from_environment(os.environ)
    except MatmulArithmeticRefused as exc:
        raise AdjointIdentityRefused(str(exc)) from exc
    from .autoscale import require_bounded_capture_environment

    execution = config["execution"]
    if (config.get("qualification_window") is not None
            or execution.get("retained_operator_windows") is not None):
        require_bounded_capture_environment(os.environ)
    os.environ[ACTIVATION_SCALE_ENV] = execution["production_act_scales"]
    torch.set_num_threads(1)
    pin_matmul_arithmetic()

    if chain_seed is None and (
            Path(config["output_root"]).resolve() != Path(output_root).resolve()):
        raise AdjointIdentityRefused(
            f"plan output_root {config['output_root']} is not --output-root "
            f"{output_root}")
    if chain_seed is not None:
        # A seed measures the plan's own run from a scratch root: the plan
        # names the source run's root, and --output-root is somewhere else
        # (RobTand/prismaquant#1016). Both are checked before any GPU work.
        from .stage_a_chain_seed import ChainSeedRefused, preflight_seed_root
        if chain_resume is not None or forward_recovery is not None:
            raise AdjointIdentityRefused(
                "chain seed refused: a seed is neither a chain resume nor a forward "
                "recovery")
        try:
            seed_source = preflight_seed_root(output_root, chain_seed)
        except ChainSeedRefused as exc:
            raise AdjointIdentityRefused(f"chain seed refused: {exc}") from exc
        if Path(config["output_root"]).resolve() != seed_source.resolve():
            raise AdjointIdentityRefused(
                f"chain seed refused: the seed checkpoint is not under the plan's "
                f"output_root {config['output_root']}")
    try:
        regime_stamp = chain_regime_identity(
            normalize_chain_regime(chain_batch_size, chain_probe_fusion))
    except ChainRegimeRefused as exc:
        raise AdjointIdentityRefused(str(exc)) from exc
    occupied = occupied_checkpoint_directories(adjoint_space(output_root))
    if occupied and chain_resume is None:
        # A checkpoint attempt creates its boundary with exist_ok=False,
        # so an occupied path fails the run only when the adjoint sweep
        # reaches it -- after the head intake and the forward pass. Refuse
        # here instead, before any GPU work.
        raise AdjointIdentityRefused(
            "stage A writes checkpoint paths that already exist: "
            + ", ".join(str(path) for path in occupied)
            + "; a checkpoint directory is never reused, so move a stale "
            "one aside (rename it) before submitting")
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
        **({"chain_regime": regime_stamp} if regime_stamp is not None else {}),
        **({"chain_resume": dict(chain_resume)} if chain_resume is not None else {}),
        **({"chain_seed": dict(chain_seed)} if chain_seed is not None else {}),
        **({"chain_split": dict(chain_split)} if chain_split is not None else {}),
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
    kernel = _stage_a_kernel_profiler()
    runner = None
    started, before_io = time.time(), _io_counters()
    try:
        result["device_envelope"] = _stage_a_device_envelope(config)
        if result["device_envelope"] is not None:
            print("joint_cost_stage_a: enforced Torch allocator envelope "
                  f"{result['device_envelope']['device_envelope_bytes']} bytes "
                  "before backend/model allocation; native CUDA allocations "
                  "remain covered by PB action limits", flush=True)
        reader = load_declared_reader(config.get("reader"))
        reader_identity = None if reader is None else reader.identity
        implementation = _aura_source_sha256()
        projection_backend = prewarm_projection_backend(
            execution.get("projection_backend"), device="cuda")
        result["projection_backend"] = projection_backend.identity
        completion = _preflight_run_prepared(
            prepared, plan_sha256=plan_sha256, implementation_sha256=implementation,
            reader_identity=reader_identity, projection_backend=projection_backend.identity)
        # A wall in dev mode too, where the preflight only records a plan
        # mismatch: the completion's roster is this plan's only because the
        # prepare ran under it (PQ #1051).
        _same(completion.get("plan_sha256"), plan_sha256, "prepared plan")
        data = None
        if head_walk:
            data = load_measured_anchor_input(
                config["inputs"], reader=reader, synthesis_device="cuda",
                progress_phase="head",
                head_checkpoint=space / "head-walk",
                head_resume=bool(resume or chain_resume is not None),
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
        if data is not None:
            original_draw = data.payload["provenance"]["hessian"]["calibration_identity"]
            for name in ("fit_ids_sha256", "text_sha256", "nsamples", "seqlen", "seed"):
                _same(calibration["provenance"].get(name), original_draw.get(name),
                      f"original full draw {name}")
            head = walked_head(data, completion, prepared=prepared,
                               calibration=calibration)
        else:
            head = prepared_head(completion, prepared=prepared, calibration=calibration)
            # The completion attests the units the prepare's walk verified:
            # reported once, at the cumulative count the walk would have
            # reached, as the Stage B head slice does (PQ #1010).
            from .joint_run_progress import HEAD_PHASE
            _pb_commit(head.progress_units, HEAD_PHASE)
        result["calibration_input"] = calibration
        result["head"] = head.record

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
        result.update(source_model_identity=source, units=head.units,
                      measured_cells=head.measured_cells)
        # A seed's capsule names the campaign the way a recovery capsule does.
        roster_digest, capture_scope = stage_a_roster(
            head.formats_by_qname,
            capsule=forward_recovery if chain_seed is None else chain_seed["capsule"],
            plan_sha256=plan_sha256, prepared_sha256=prepared["sha256"],
            read_manifest_sha256=read_manifest_sha256 or "0" * 64,
            calibration_shape=list(ids.shape),
            campaign_scope=config.get("campaign_scope"))

        result["artifact_preflight"] = _run_artifact_preflight(
            runner, ids, execution, stride_value, space, artifact)

        kernel.__enter__()
        progress = JointRunProgress(
            layers=runner.num_layers, partitions=1,
            base_units=head.progress_units, log=lambda message: print(
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
            campaign_scope=capture_scope,
            boundary_artifact_bytes=int(artifact["run_used"]),
            artifact_budget_stamp=artifact["override"],
            min_free_gib=config.get("min_free_gib", 0.0), progress=progress,
            produced_output=publication, forward_recovery=forward_recovery,
            chain_batch_size=chain_batch_size, chain_probe_fusion=chain_probe_fusion,
            chain_resume=chain_resume, chain_seed=chain_seed, chain_split=chain_split,
            arithmetic_extra={
                "container_content_sha256": result["env"]["container_content_sha256"],
                "projection_backend": projection_backend.identity})
        if "stride" in receipt:
            receipt["stride"]["source"] = stride_source
        receipt["device_envelope"] = result["device_envelope"]
        torch.cuda.synchronize()
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if result["peak_gpu_bytes"] > config["max_gpu_bytes"]:
            raise RuntimeError("observed GPU allocation exceeds declared budget")
        # Outside the run identity and the chain state: which arm read the
        # head is a fact about this launch, not about the science.
        receipt["head"] = head.record
        split_files = None
        if chain_split is not None:
            split_files = _write_split_receipt(space, receipt)
            result["split_receipt"] = split_files["receipt"]
        elif chain_seed is not None:
            from .stage_a_chain_seed import write_seed_receipt
            result["seed_receipt"] = write_seed_receipt(space, receipt)
            result["plane_comparison"] = (
                None if receipt["plane_comparison"] is None
                else {key: receipt["plane_comparison"][key]
                      for key in ("boundary", "equal", "different", "bitwise_equal")})
        else:
            write_adjoint_receipt(space, receipt)
            result["adjoint_receipt"] = {
                "path": str(space / "adjoint-capture.json"),
                "sha256": hashlib.sha256(
                    (space / "adjoint-capture.json").read_bytes()).hexdigest(),
            }
        result["checkpoints"] = [
            {"boundary": entry["boundary"],
             "cotangent_sha256": entry["cotangent_sha256"]}
            for entry in receipt.get("checkpoints", receipt.get("partials", []))]
        result["passed"] = True
    except BaseException as error:
        # Said before any teardown. On 2026-09-21 the teardown below took the
        # box down and the kernel killed this process before its traceback
        # printed: PrismaBuild recorded exit 137 and no reason (PQ #899).
        print(f"joint_cost_stage_a: capture failed: {type(error).__name__}: "
              f"{error}", flush=True)
        raise
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
        result["kernel_active_s"] = kernel.block()["kernel_active_s"]
        result["kernel_profiler_error"] = kernel.error
        residency = residency_report()
        if residency is not None:
            result["residency"] = residency

    counters = {
        "schema": QUANTUM_COUNTERS_SCHEMA.replace(
            "joint_layer_quantum", "joint_adjoint_capture"),
        "stage": "adjoint_capture",
        "wall_s": result["phases"][0]["end_epoch"] - started,
        "kernel_active_s": kernel.block()["kernel_active_s"],
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
        "device_envelope": result.get("device_envelope"),
    }
    # A split row's own files: several rows of one run finish independently.
    counters_path, results_path = (
        (space / "counters.json", space / "results.json") if split_files is None
        else (Path(split_files["counters"]), Path(split_files["results"])))
    atomic_write_bytes(counters_path,
                       (json.dumps(counters, sort_keys=True, indent=2,
                                   allow_nan=False) + "\n").encode())
    atomic_write_bytes(results_path,
                       (json.dumps(result, sort_keys=True, indent=2,
                                   allow_nan=False) + "\n").encode())
    return result


def _write_split_receipt(space, receipt) -> dict:
    """Write a split row's receipt; return it and its results and counters paths."""
    from .stage_a_chain_split import PREP_RECEIPT_SCHEMA, quantum_directory, split_root

    if receipt["schema"] == PREP_RECEIPT_SCHEMA:
        stem = split_root(space) / "preps" / f"resume-{receipt['resume']['index']:03d}"
    else:
        stem = quantum_directory(space) / receipt["split"]["label"]
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)
               + "\n").encode()
    path = stem.with_name(stem.name + ".json")
    atomic_write_bytes(path, payload)
    return {"receipt": {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()},
            "results": str(stem.with_name(stem.name + ".results.json")),
            "counters": str(stem.with_name(stem.name + ".counters.json"))}


def _chain_resume_argument(args):
    """The core's ``chain_resume`` from the CLI flags, or ``None``."""
    if args.resume_chain_state_sha256 is None:
        return None
    from .stage_a_chain_resume import ChainResumeRefused, parse_declaration
    try:
        declaration = parse_declaration(args.resume_implementation_compatibility)
    except ChainResumeRefused as exc:
        raise AdjointIdentityRefused(str(exc)) from exc
    return {"chain_state_sha256": args.resume_chain_state_sha256,
            "declaration": declaration, "resume_from": args.resume_from_checkpoint}


def _chain_split_argument(args, parser):
    """The core's ``chain_split`` from the split flags, or ``None`` (PQ #738)."""
    if args.chain_split_prep is None and args.chain_split_quantum is None:
        if args.chain_split_ranges is not None or args.chain_split_digest_layer is not None:
            parser.error("--chain-split-ranges and --chain-split-digest-layer belong to "
                         "a split prep or quantum")
        return None
    if args.chain_split_prep is not None and args.chain_split_quantum is not None:
        parser.error("a row is a split prep or a split quantum, not both")
    if args.resume_chain_state_sha256 is None or args.resume_from_checkpoint is None:
        parser.error("a split row continues a run's sealed chain: it needs "
                     "--resume-chain-state-sha256 and --resume-from-checkpoint")
    from .stage_a_chain_split import ChainSplitRefused, parse_ranges
    try:
        if args.chain_split_prep is not None:
            if args.chain_split_ranges is None or args.chain_split_digest_layer is not None:
                parser.error("a split prep takes --chain-split-ranges and no digest layer")
            return {"role": "prep", "through": args.chain_split_prep,
                    "ranges": parse_ranges(args.chain_split_ranges)}
        if args.chain_split_ranges is not None:
            parser.error("a split quantum names its own samples in --chain-split-quantum")
        pieces = args.chain_split_quantum.split(":")
        if len(pieces) != 3 or not all(piece.isdigit() for piece in pieces):
            parser.error("--chain-split-quantum is THROUGH:START:STOP")
        through, start, stop = (int(piece) for piece in pieces)
        return {"role": "quantum", "through": through, "samples": [start, stop],
                "digest_layer": args.chain_split_digest_layer}
    except ChainSplitRefused as exc:
        parser.error(str(exc))


def _chain_seed_argument(args):
    """The core's ``chain_seed`` from ``--chain-seed``, or ``None``."""
    if args.chain_seed is None:
        return None
    from .stage_a_chain_seed import ChainSeedRefused, load_seed_spec
    try:
        return load_seed_spec(args.chain_seed, args.chain_seed_sha256)
    except ChainSeedRefused as exc:
        raise AdjointIdentityRefused(f"chain seed refused: {exc}") from exc


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
    parser.add_argument("--resume", action="store_true",
                        help="accepted from the sealed Stage A row; Stage A takes "
                             "its head from the prepared completion and has no "
                             "head walk to resume (PQ #1051)")
    parser.add_argument("--forward-recovery", type=Path)
    parser.add_argument("--forward-recovery-sha256")
    parser.add_argument("--chain-batch-size", type=int, default=1,
                        help="calibration samples per render-free chain "
                             "forward and backward (default %(default)s). "
                             "B > 1 changes rounding and is stamped into the "
                             "receipt's run identity (RobTand/prismaquant#997)")
    parser.add_argument("--chain-probe-fusion", choices=("on", "off"), default="off",
                        help="one chain forward per sample group and one "
                             "backward per probe (default %(default)s); "
                             "bitwise-neutral at a fixed batch size, stamped "
                             "into the run identity")
    parser.add_argument("--resume-chain-state-sha256", default=None,
                        help="relaunch this run from its lowest sealed checkpoint: "
                             "the digest of <output-root>/layer-quanta/adjoint/"
                             "chain-state.json, which the run wrote after its tail "
                             "checkpoint (RobTand/prismaquant#1001). The relaunch "
                             "adopts the run's header and boundary session and "
                             "refuses on any regime, plan, preparation, capsule "
                             "or arithmetic difference")
    parser.add_argument("--resume-from-checkpoint", type=int, default=None,
                        help="the checkpoint boundary the relaunch expects to "
                             "resume from; refuses unless it is the run's lowest "
                             "sealed checkpoint")
    parser.add_argument("--resume-implementation-compatibility", default=None,
                        metavar="FROM:TO",
                        help="dev mode only: declare that the relaunch continues "
                             "a chain sealed by implementation FROM under "
                             "implementation TO. Recorded in the receipt and in "
                             "every band sealed below the switch; never inferred")
    parser.add_argument("--chain-seed", type=Path, default=None,
                        help="dev mode only: a prismaquant.stage_a.chain_seed.v1 "
                             "spec. Continues the plan's own run's sealed "
                             "checkpoint, sealed by another implementation, into "
                             "--output-root, a scratch root, for measurement; "
                             "writes a seed receipt the band tool refuses "
                             "(RobTand/prismaquant#1016)")
    parser.add_argument("--chain-seed-sha256", default=None)
    parser.add_argument("--chain-split-prep", type=int, default=None, metavar="THROUGH",
                        help="a split round's prep row (PQ #738): seal the round's "
                             "resume record and remove the rolling entries of "
                             "--chain-split-ranges; rolls nothing")
    parser.add_argument("--chain-split-ranges", default=None, metavar="S:E,...",
                        help="the sample ranges a split prep launches")
    parser.add_argument("--chain-split-quantum", default=None,
                        metavar="THROUGH:START:STOP",
                        help="one split quantum (PQ #738): roll global samples "
                             "START..STOP-1 from the lowest sealed checkpoint down "
                             "to THROUGH, sealing a partial checkpoint of the range "
                             "at every stride checkpoint on the way")
    parser.add_argument("--chain-split-digest-layer", type=int, default=None,
                        help="a split quantum also records each rolled payload's "
                             "sha256 at this layer in its receipt")
    args = parser.parse_args(argv)
    chain_split = _chain_split_argument(args, parser)
    if bool(args.chain_seed) != bool(args.chain_seed_sha256):
        parser.error("--chain-seed and --chain-seed-sha256 must be paired")
    if args.chain_seed is not None and (
            args.resume_chain_state_sha256 is not None or args.forward_recovery):
        parser.error("--chain-seed binds a fresh scratch run: it takes no "
                     "--resume-chain-state-sha256 or --forward-recovery")
    if args.resume_chain_state_sha256 is None and (
            args.resume_from_checkpoint is not None
            or args.resume_implementation_compatibility is not None):
        parser.error("--resume-from-checkpoint and --resume-implementation-compatibility "
                     "need --resume-chain-state-sha256")
    if bool(args.forward_recovery) != bool(args.forward_recovery_sha256):
        parser.error("--forward-recovery and --forward-recovery-sha256 must be paired")
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
            artifact_budget_bytes=args.artifact_budget_bytes,
            forward_recovery=({"path": str(args.forward_recovery),
                               "sha256": args.forward_recovery_sha256}
                              if args.forward_recovery else None),
            chain_batch_size=args.chain_batch_size,
            chain_probe_fusion=args.chain_probe_fusion == "on",
            chain_resume=_chain_resume_argument(args),
            chain_seed=_chain_seed_argument(args), chain_split=chain_split)
    except AdjointIdentityRefused as exc:
        print(f"adjoint_identity_refused: {exc}", flush=True)
        return EXIT_IDENTITY_REFUSED
    print(json.dumps({key: result[key] for key in (
        "command", "passed", "stride", "checkpoints", "plane_comparison",
        "split_receipt") if key in result}))
    return EXIT_OK if result["passed"] else EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
