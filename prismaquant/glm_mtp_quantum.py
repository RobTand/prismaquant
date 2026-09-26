"""Price GLM-5.3's MTP layer on the MTP head's self-KL (PQ #1353, M5 of #1271).

The M4 preparation qualifies every ``(unit, rung)`` render of the MTP layer
into a ``ProductionWeightCache``, over the ``mtp`` source scope. This module
turns those renders into joint-AURA rows, using the body's arithmetic with
the MTP objective as the seed:

* **Forward.** Each calibration sequence's target post-final-norm hidden
  state ``h`` (M1's published final-hidden entries, hash-verified per entry)
  and the next-token embeddings go through the MTP layer, then the target
  ``lm_head`` (``glm_mtp.mtp_logits``). ``h`` is a graph root, so the backward
  stops at the MTP layer; pricing layer-45 weights needs no body gradient.
* **Seed.** ``glm_mtp.mtp_probe_scalar``: the body's Rademacher probe on the
  draft logits, every row, normalized by ``n * (T - 1)``. So
  ``0.5 * mean_k <G_k, dW>^2`` is the second-order KL of the draft
  distribution per MTP row.
* **Projection.** ``JointOperatorStatisticsLease`` observes the priced Linears
  (packed routed experts and the shared expert) and projects each render's
  ``dW`` through ``joint_statistics_replay.observe_and_project_windows``, the
  same statistics and projection arithmetic as a body layer.
* **Rows.** ``joint_aura.make_joint_aura_entry`` on a probe identity carrying
  ``glm_mtp.mtp_objective_identity``, so a body join refuses them.

The output is a ``prismaquant.glm_mtp_cost.v1`` payload, the input of the
allocator's MTP path (``glm_mtp_selection``). BF16 passthrough is not a priced
row; the selection adds it where the source is BF16.

``tessera_joint_aura run`` calls :func:`compute_mtp_cost` for a plan whose
``source_scope`` is ``mtp``, after the same prepared-cache checks as a body
run.
"""
from __future__ import annotations

import hashlib
import time
from typing import Callable, Mapping

import torch

from . import glm_mtp

SCHEMA = "prismaquant.glm_mtp_cost.v1"
SCOPE = "mtp"


def priced_formats(formats) -> tuple[str, ...]:
    """The rungs the MTP layer prices: every rung but zero-cost passthrough."""
    from .aura_cost import _ZERO_COST_FORMATS

    return tuple(fmt for fmt in formats if fmt not in _ZERO_COST_FORMATS)


class ResidentHiddenStates:
    """``read(index)`` over the final-hidden entries for several passes.

    The published entries are read through a single-pass verified stream
    (``glm_mtp_capture.final_hidden_stream``). Each probe, and each statistics
    window of a probe, replays every sequence, so the first pass keeps each
    entry on the CPU as it arrives and later passes read those copies. That
    repeated-pass dependency is why the whole column ends up resident (about
    4 MiB per 512-token sequence for GLM-5.3-Flash). The first pass still
    streams: work starts on sequence 0 while later entries are being read.
    """

    def __init__(self, read_once: Callable[[int], torch.Tensor], n_sequences: int):
        self._read_once = read_once
        self._n = int(n_sequences)
        self._held: list[torch.Tensor] = []

    def __call__(self, index: int) -> torch.Tensor:
        if index < len(self._held):
            return self._held[index]
        if index != len(self._held) or index >= self._n:
            raise RuntimeError(
                f"final-hidden entries are read in order on the first pass; asked for "
                f"{index}, next is {len(self._held)}")
        value = self._read_once(index).detach().to("cpu")
        self._held.append(value)
        return value

    def require_complete(self):
        if len(self._held) != self._n:
            raise RuntimeError(f"read {len(self._held)} of {self._n} final-hidden entries")


def mtp_probe_identity(*, source_model, calibration_ids, n_probes, seed_base, arithmetic,
                       source_execution, mtp_layer, vocab_size) -> dict:
    """The joint probe identity of an MTP row.

    The body's v2 fields (``joint_cost_quantum``) with the MTP geometry: each
    sequence contributes ``T - 1`` rows, the probe is normalized by
    ``n * (T - 1)``, and ``glm_mtp.mtp_objective_identity`` names the seed.
    """
    from .aura_cost import _aura_source_sha256
    from .kl_fisher import ROW_PROBE_LAYOUT

    n, length = (int(size) for size in calibration_ids.shape)
    probe = {
        "schema": "prismaquant.joint_aura.probes.v2",
        "source_model": source_model,
        "calibration_sha256": hashlib.sha256(
            calibration_ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "calibration_shape": [n, length],
        "calibration_dtype": str(calibration_ids.dtype),
        "n_probes": int(n_probes), "seed_base": int(seed_base),
        "token_scope": "all", "temperature": 1.0,
        "distribution": "rademacher", "normalization": "global_kl_fisher",
        "producer_source_sha256": _aura_source_sha256(),
        "source_execution": source_execution,
        "arithmetic": arithmetic,
        "noise_layout": {
            "schema": ROW_PROBE_LAYOUT, "global_rows": n, "sequence_length": length,
            "selected_tokens_per_row": length - 1, "vocab_size": int(vocab_size),
            "token_scope": "all",
            "global_token_count": glm_mtp.mtp_global_token_count(n, length),
        },
    }
    return glm_mtp.with_mtp_objective(probe, glm_mtp.mtp_objective_identity(
        mtp_layer=int(mtp_layer), sequence_length=length, n_sequences=n))


def mtp_priced_modules(model, profile) -> dict:
    """``{qname: module}`` for the priced MTP units, as the lease observes them.

    The routed experts are the profile's packed-expert views, built from the
    installed parameters (the lease fingerprints their storage); the shared
    expert's three Linears are its ``nn.Linear`` modules.
    """
    from .routed_experts import profile_declared_packed_expert_projections

    units = glm_mtp.mtp_priced_units(model, profile)
    layer = model.layer
    prefix = glm_mtp.mtp_checkpoint_prefix(layer.layer_idx)
    modules = {member.qname: member
               for member in profile_declared_packed_expert_projections(model, profile)}
    for name, module in layer.mlp.shared_experts.named_children():
        qname = f"{prefix}mlp.shared_experts.{name}"
        if qname in units:
            modules[qname] = module
    if set(modules) != set(units):
        raise RuntimeError("MTP priced modules differ from the priced units")
    return dict(sorted(modules.items()))


def mtp_backward(model, embed_tokens, lm_head, calibration_ids, hidden_states, *,
                 seed: int, device, min_free_gib=None, free_gib=None, report=None):
    """One probe's forward and backward over every calibration sequence.

    Sequence ``i`` is its own global row block (``global_row_offset=i``), so
    the probe noise does not depend on how rows are batched. ``h`` is a graph
    root that requires grad: with frozen weights and a detached ``h`` no
    output would require grad and no statistics observer would fire.
    """
    n = int(calibration_ids.shape[0])
    dtype = model.layer.eh_proj.weight.dtype
    for index in range(n):
        if min_free_gib is not None and free_gib is not None and free_gib() < min_free_gib:
            raise RuntimeError("MTP probe crossed the free memory floor")
        ids = calibration_ids[index:index + 1].to(device)
        hidden = hidden_states(index).to(device=device, dtype=dtype).detach().requires_grad_(True)
        logits = glm_mtp.mtp_logits(model.layer, embed_tokens, lm_head, ids, hidden)
        scalar = glm_mtp.mtp_probe_scalar(logits, seed=int(seed), global_row_offset=index,
                                          n_sequences=n)
        scalar.backward()
        del ids, hidden, logits, scalar
        if report is not None:
            report(index + 1)


def _group_units(census, names):
    """The census's anchor groups over the priced units: each takes one rung."""
    groups = {group: sorted(members) for group, members in census["anchor_groups"].items()}
    covered = [name for members in groups.values() for name in members]
    if sorted(covered) != sorted(names) or len(covered) != len(set(covered)):
        raise RuntimeError("MTP census anchor groups do not partition the priced units")
    return groups


def compute_mtp_cost(model, embed_tokens, lm_head, calibration_ids, hidden_states, *,
                     production_cache, formats_by_qname, wire_bytes, census, profile,
                     n_probes, seed_base, source_model, projection_backend,
                     operator_windows, device, guard=None, min_free_gib=None,
                     free_gib=None, progress=None) -> dict:
    """The MTP layer's ``glm_mtp_cost.v1`` payload from qualified renders.

    ``model`` is the installed :class:`~prismaquant.glm_mtp.MtpCheckpointModel`
    and ``hidden_states(i)`` returns sequence ``i``'s ``[1, T, H]`` hidden
    state (:class:`ResidentHiddenStates`). ``production_cache`` is the M4
    cache: its ``verified_cells`` must name, for every priced ``(unit, rung)``,
    the source and render tensors and the activation contract this pass
    prices, or the pass refuses. ``wire_bytes[unit][rung]`` are the priced
    rungs' serialized bytes from the merged campaign table.
    """
    from . import format_registry as fr
    from .glm_mtp_capture import sequence_progress
    from .joint_aura import (activation_identity, identity_sha256, make_joint_aura_entry,
                             source_execution_identity,
                             validated_probe_identity)
    from .joint_served_activation import joint_activation_maxima
    from .joint_statistics_replay import (observe_and_project_windows,
                                          statistics_arithmetic_identity)
    from .production_weight_cache import _cb_cache_tensor_identity

    if getattr(production_cache, "_joint_served_activation", None) is not None:
        raise RuntimeError("MTP pricing has no served activation policy; its menu is "
                           "weight rungs with their own activation contracts")
    modules = mtp_priced_modules(model, profile)
    if set(formats_by_qname) != set(modules):
        outside = sorted(set(formats_by_qname) - set(modules))[:4]
        missing = sorted(set(modules) - set(formats_by_qname))[:4]
        raise RuntimeError(f"MTP roster differs from the priced units: outside {outside}, "
                           f"unpriced {missing}")
    rungs = {name: priced_formats(formats_by_qname[name]) for name in modules}
    if any(not formats for formats in rungs.values()):
        raise RuntimeError("an MTP unit has no priced rung")
    verified = production_cache.metadata["verified_cells"]
    missing = sorted((name, fmt) for name, formats in rungs.items() for fmt in formats
                     if (name, fmt) not in verified)
    if missing:
        raise RuntimeError(f"MTP renders were not qualified: {missing[:4]}")
    for name, formats in rungs.items():
        if set(wire_bytes.get(name, {})) < set(formats):
            raise RuntimeError(f"MTP unit {name}: the campaign table has no wire bytes for "
                               f"{sorted(set(formats) - set(wire_bytes.get(name, {})))}")
    groups = _group_units(census, list(modules))

    maxima = joint_activation_maxima(production_cache)
    specs = {name: {fmt: fr.get_format(fmt) for fmt in rungs[name]} for name in modules}
    dtype = model.layer.eh_proj.weight.dtype
    arithmetic = statistics_arithmetic_identity(dtype, projection_backend)
    arithmetic["operator_windows"] = dict(operator_windows)
    probe_identity = mtp_probe_identity(
        source_model=source_model, calibration_ids=calibration_ids, n_probes=n_probes,
        seed_base=seed_base, arithmetic=arithmetic,
        source_execution=source_execution_identity(model),
        mtp_layer=model.layer.layer_idx, vocab_size=int(lm_head.weight.shape[0]))
    validated = validated_probe_identity(probe_identity)
    probe_sha256 = identity_sha256(validated)

    # The installed sources and the activation contracts depend on nothing a
    # probe computes, so they are checked against the preparation before the
    # first forward, not after a full backward pass.
    sources, activations, operators = {}, {}, {}
    for name, module in modules.items():
        sources[name] = _cb_cache_tensor_identity(module.weight.detach())
        for fmt in rungs[name]:
            cell = verified[(name, fmt)]
            if sources[name] != cell["source_weight"]:
                raise RuntimeError(f"MTP source weight differs from the prepared one for {name}")
            activations[(name, fmt)] = activation_identity(specs[name][fmt], maxima, name)
            if activations[(name, fmt)] != cell["activation"]:
                raise RuntimeError(f"MTP activation contract differs from the prepared one for "
                                   f"{name}@{fmt}")

    def record_operator(name, fmt, source, rendered):
        # Hashed once per pair: every later load of the same render is checked
        # against the prepared file SHA-256 by the cache itself.
        if (name, fmt) in operators:
            return
        rendered_identity = _cb_cache_tensor_identity(rendered)
        if rendered_identity != verified[(name, fmt)]["rendered_weight"]:
            raise RuntimeError(f"MTP render differs from the prepared one for {name}@{fmt}")
        activation = activations[(name, fmt)]
        operators[(name, fmt)] = {
            "schema": "prismaquant.joint_aura.operator.v2",
            "qname": name, "format": fmt,
            "source_weight": sources[name], "rendered_weight": rendered_identity,
            "activation": activation, "arithmetic": arithmetic,
            "probe_identity_sha256": probe_sha256,
        }

    n_sequences = int(calibration_ids.shape[0])
    components = {(name, fmt): [] for name in modules for fmt in rungs[name]}
    receipts, timings = [], []
    measured = {name: module for name, module in modules.items()}
    source_seal = None
    for probe_index in range(int(n_probes)):
        started = time.monotonic()
        passes = [0]

        def backward(*, final, lease, _probe=probe_index):
            del final, lease
            passes[0] += 1
            mtp_backward(model, embed_tokens, lm_head, calibration_ids, hidden_states,
                         seed=int(seed_base) + _probe, device=device,
                         min_free_gib=min_free_gib, free_gib=free_gib,
                         report=sequence_progress(f"[mtp-quantum] probe {_probe + 1}/{n_probes} "
                                                  f"pass {passes[0]}:", n_sequences))

        terms, diagnostics, receipt = observe_and_project_windows(
            measured, specs, production_cache, operator_windows, backward=backward,
            record_operator=record_operator, collect_col_energy=False,
            backend=projection_backend, guard=guard, source_fingerprints=source_seal)
        if set(terms) != set(components):
            raise RuntimeError("MTP probe did not project every priced render")
        # A routed expert no calibration row reaches has no gradient and
        # prices zero; the shared expert sees every row, so silence there is
        # a broken observer, not a measurement.
        silent = sorted(name for name, row in diagnostics.items() if not row["observed_calls"])
        dead = [name for name in silent if not hasattr(modules[name], "expert_id")]
        if dead:
            raise RuntimeError(f"MTP units every row reaches observed nothing: {dead[:4]}")
        for key, value in terms.items():
            components[key].append(value)
        hidden_states_complete = getattr(hidden_states, "require_complete", None)
        if hidden_states_complete is not None:
            hidden_states_complete()
        receipts.append(dict(probe_index=probe_index, passes=passes[0], **receipt))
        timings.append({"probe_index": probe_index, "seconds": time.monotonic() - started,
                        "passes": passes[0],
                        "unobserved_units": len(silent)})
        print(f"[mtp-quantum] probe {probe_index + 1}/{n_probes} projected "
              f"{len(terms)} renders in {timings[-1]['seconds']:.0f} s over "
              f"{passes[0]} pass(es)", flush=True)
        if progress is not None:
            progress(probe_index + 1)

    costs = {}
    for name in modules:
        costs[name] = {}
        for fmt in rungs[name]:
            # make_joint_aura_entry validated this row against ``validated``,
            # whose source model was checked once, above. The published row
            # carries the ordinary ``probe_identity``, and the check after this
            # loop re-proves that it still hashes to the validated digest.
            # Validating each row again after the swap re-hashed the whole
            # streamed source identity once per row: about 4 GPU-idle minutes
            # for GLM-5.3's 1734 rows (PQ #1396). The body producer
            # (joint_cost_quantum) makes the same swap without it.
            row = make_joint_aura_entry(operator_identity=operators[(name, fmt)],
                                        probe_identity=validated,
                                        signed_components=components[(name, fmt)])
            row["probe_identity"] = probe_identity
            costs[name][fmt] = row
    if identity_sha256(probe_identity) != probe_sha256:
        raise RuntimeError("MTP probe identity changed during measurement")
    return {
        "schema": SCHEMA,
        "mtp_layer": int(model.layer.layer_idx),
        "costs": costs,
        "wire_bytes": {name: {fmt: int(wire_bytes[name][fmt]) for fmt in rungs[name]}
                       for name in modules},
        "params": {name: int(module.weight.numel()) for name, module in modules.items()},
        "source_dtype": {name: str(module.weight.dtype).removeprefix("torch.")
                         for name, module in modules.items()},
        "groups": groups,
        "provenance": {
            "probe_identity_sha256": probe_sha256,
            "operator_window_receipts": receipts,
            "probe_timings": timings,
            "observed_last_probe": {name: {"tokens": row["observed_tokens"], "calls": row["observed_calls"]}
                         for name, row in diagnostics.items()},
        },
    }


def run_mtp_scope(runner, *, config, data, production_cache, calibration_ids, calibration,
                  source_model, projection_backend, operator_windows, device_bytes,
                  read_ahead_bytes=256 << 20, progress=None) -> dict:
    """``tessera_joint_aura run`` over the ``mtp`` source scope.

    ``runner`` is the scoped source runner, whose model is the MTP layer
    under its checkpoint name. Installs that layer, reads the embedding and
    the target ``lm_head`` through the capture's source owner (each shard
    authenticated before a tensor from it is used), and streams M1's
    final-hidden entries named by the MTP census. Returns the
    ``glm_mtp_cost.v1`` payload of :func:`compute_mtp_cost`.
    """
    from contextlib import closing

    from . import glm_mtp_capture as cap
    from .aura_cost import _free_gib
    from .joint_statistics_replay import operator_window_guard

    if config.get("joint_eval") is not None:
        raise RuntimeError("MTP pricing runs on the full draw, not an evaluation panel")
    if operator_windows is None:
        raise RuntimeError("MTP pricing needs the plan's operator windows")
    extension = data.census.get("mtp_extension") or {}
    final_ref = extension.get("final_hidden")
    if not final_ref:
        raise RuntimeError("the priced census is not an MTP census; it names no final-hidden entries")
    layer = int(extension["layer"])
    if tuple(runner.source_layers) != (layer,):
        raise RuntimeError(f"the source runner holds layers {tuple(runner.source_layers)}, "
                           f"not the MTP layer {layer}")
    final, _ = cap.read_bound_json(final_ref["path"], final_ref["sha256"])
    if final["inputs"]["calibration_sha256"] != calibration["calibration_sha256"]:
        raise RuntimeError("the final hidden states were computed on another calibration draw")
    if tuple(calibration_ids.shape)[0] != len(final["records"]):
        raise RuntimeError("the final-hidden entries do not cover the calibration draw")

    runner.context.schedule_prefetch(layer)
    runner.context.install(layer, require_prefetched=runner.require_prefetched_residency,
                           prefetch_following=False)
    model = runner.model
    if getattr(model, "layer", None) is None or int(model.layer.layer_idx) != layer:
        raise RuntimeError("the scoped source model is not the MTP layer")

    from .tessera_joint_aura import _prepare_source_owner

    owner = _prepare_source_owner(config, data)
    try:
        embed_weight, _ = glm_mtp.read_checkpoint_tensor(
            config["model"], "model.language_model.embed_tokens.weight", source_authentication=owner)
        head_weight, _ = glm_mtp.read_checkpoint_tensor(
            config["model"], "lm_head.weight", source_authentication=owner)
        authentication = owner.receipt()
    finally:
        owner.close()
    dtype = model.layer.eh_proj.weight.dtype
    device = runner.device
    embed = torch.nn.Embedding.from_pretrained(embed_weight.to(device=device, dtype=dtype),
                                               freeze=True)
    lm_head = torch.nn.Linear(head_weight.shape[1], head_weight.shape[0], bias=False,
                              device="meta")
    lm_head.weight = torch.nn.Parameter(head_weight.to(device=device, dtype=dtype),
                                        requires_grad=False)
    del embed_weight, head_weight

    wire = {name: {fmt: row["wire_bytes"] for fmt, row in rows.items() if "wire_bytes" in row}
            for name, rows in data.payload["costs"].items()}
    guard = operator_window_guard(device, device_bytes=device_bytes)
    read, stream = cap.final_hidden_stream(final, int(calibration_ids.shape[0]),
                                           read_ahead_bytes=int(read_ahead_bytes))
    with closing(stream):
        payload = compute_mtp_cost(
            model, embed, lm_head, calibration_ids,
            ResidentHiddenStates(read, int(calibration_ids.shape[0])),
            production_cache=production_cache, formats_by_qname=data.formats_by_qname,
            wire_bytes=wire, census=data.census, profile=runner.profile,
            n_probes=config["execution"]["n_probes"], seed_base=config["execution"]["seed_base"],
            source_model=source_model, projection_backend=projection_backend,
            operator_windows=operator_windows, device=device, guard=guard,
            min_free_gib=config["min_free_gib"], free_gib=_free_gib, progress=progress)
    payload["provenance"].update(
        final_hidden=dict(final_ref), head_source_authentication=authentication,
        head_tensors={"embed_tokens": "model.language_model.embed_tokens.weight",
                      "lm_head": "lm_head.weight"})
    return payload


__all__ = ["SCHEMA", "SCOPE", "ResidentHiddenStates", "compute_mtp_cost", "mtp_backward",
           "mtp_priced_modules", "mtp_probe_identity", "priced_formats", "run_mtp_scope"]
