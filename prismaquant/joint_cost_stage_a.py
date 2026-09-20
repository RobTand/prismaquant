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


def run_adjoint_capture_core(
    runner, calib_ids, *, execution, output_root, stride,
    source_model_identity, unit_roster_sha256, plan_sha256, prepared_sha256,
    read_manifest_sha256, implementation_sha256, campaign_scope=None,
    boundary_artifact_bytes=None, min_free_gib=0.0, progress=None,
) -> dict:
    """Forward boundaries, tail cotangents, strided render-free chain.

    Returns the receipt dict (the caller seals it). The forward capture and
    the tail cotangents are the single run's own calls; the chain is
    :func:`render_free_layer_roll` -- the completed-layer leg of
    ``compute_aura_cost_streamed``'s reverse walk -- run for every layer,
    with boundary entries **retained** (every quantum reads them later) and a
    checkpoint serialized at each strided boundary.
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
        storage_policy["max_artifact_bytes"] = int(boundary_artifact_bytes)
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
        if progress is not None:
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
            shared_adjoint = {
                (probe, batch): cotangents[probe][batch].state_dict()
                for probe in range(n_probes) for batch in range(len(batches))}
            shared_pass = {batch: batches[batch].shared_pass_state
                           for batch in range(len(batches))}
            record = write_adjoint_checkpoint(
                space, boundary=boundary,
                session={"generation": storage.session["generation"],
                         "kind": "adjoint_checkpoint",
                         "run_identity_sha256": storage.session["run_identity_sha256"]},
                cotangents=plane, shared_adjoint=shared_adjoint,
                shared_pass=shared_pass)
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
                    grad_outs[probe_index][batch_index] = storage.write(
                        tensor, batch_index=batch_index, boundary_index=layer,
                        probe_index=probe_index,
                        previous=grad_outs[probe_index][batch_index])
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


def run_adjoint_capture(
    config, *, plan_sha256, prepared, output_root, stride=None,
    read_manifest_sha256=None, data_manifest_sha256=None, resume=False,
    prefetch_override=None,
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

        kernel.__enter__()
        progress = JointRunProgress(
            layers=runner.num_layers, partitions=1,
            base_units=data.progress_committed, log=lambda message: print(
                f"joint_cost_stage_a: {message}", flush=True))
        receipt = run_adjoint_capture_core(
            runner, ids.to(runner.device), execution=execution,
            output_root=output_root, stride=stride_value,
            source_model_identity=source, unit_roster_sha256=roster_digest,
            plan_sha256=plan_sha256, prepared_sha256=prepared["sha256"],
            read_manifest_sha256=(read_manifest_sha256
                                  or "0" * 64),
            implementation_sha256=implementation,
            campaign_scope=config.get("campaign_scope"),
            min_free_gib=config.get("min_free_gib", 0.0), progress=progress)
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
            prefetch_override=args.prefetch_override)
    except AdjointIdentityRefused as exc:
        print(f"adjoint_identity_refused: {exc}", flush=True)
        return EXIT_IDENTITY_REFUSED
    print(json.dumps({key: result[key] for key in (
        "command", "passed", "stride", "checkpoints")}))
    return EXIT_OK if result["passed"] else EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
