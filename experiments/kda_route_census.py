"""Top-8 routing census for the KDA capture kernel's chain rolls (PQ #1214, E1).

A Stage B row in KDA kernel mode rolls its chain through a KDA layer with the
admitted kernel (``kda_gram_v1``) in place of the verified Torch fallback. The
kernel computes the same attention with its own rounding, so the layer's MoE
router sees a slightly different input and may select different experts for
some tokens. A different selection changes the backward graph, not only its
rounding. This census counts those selections.

For one layer, it runs the layer's forward on the exact Stage A input
boundaries of the row, in the fused chain roll's own groups (``batch_size``
consecutive stored batches, from the slice's chain regime). Each group runs
four forwards, in the order K1, F1, K2, F2: K runs inside the admitted
kernel's dispatch, F runs the fallback. Every forward stops at the packed
experts' call, after the router has chosen, so no expert runs.

* K1 against F1 is the measurement: which tokens' top-8 expert sets differ,
  by how many experts, how far each token was from its selection boundary,
  and how much the attention output, the MoE input and the router logits
  moved (plane magnitude, differing elements, maximum and relative
  Frobenius differences).
* K2 and F2 repeat K1 and F1 on the same bytes. They must be bitwise equal
  to them; otherwise the census records an instrument failure rather than a
  measurement.
* On a layer without KDA attention, the dispatch has nothing to replace, so
  K1 must equal F1 bitwise. That makes such a layer a null check of the
  instrument, not a measurement of the kernel.

Scope: forwards only. The census says whether the kernel changes the chain
roll's routing on identical inputs. It does not measure the rolled
cotangent; the row A/B does.

Subcommands:

* ``manifest`` builds the census's PrismaBuild data manifest (v2) for one
  layer from the row's executable readset: the head entries the census
  reads, the layer's source phase and its input boundaries.
* ``spec`` derives the census container spec from a campaign spec.
* ``run`` runs the census (GPU, inside the campaign container).
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time

import torch

SCHEMA = "prismaquant.kda_route_census.v1"
MANIFEST_SCHEMA = "prismaquant.prismabuild.data_manifest.v2"
TOOL = "experiments/kda_route_census.py"
ENTRY_POINT = "experiments.kda_route_census"
#: One group's forwards, in order. K runs in the kernel's dispatch, F does not.
RUN_ORDER = ("K1", "F1", "K2", "F2")
#: The kernel's counts for one forward of a KDA layer, stopped before any
#: backward: one call, two Gram passes forward (glm_kda_capture_kernel).
KERNEL_FORWARD = {"calls": 1, "gram_forward": 2, "gram_backward": 0}
NO_KERNEL = {"calls": 0, "gram_forward": 0, "gram_backward": 0}
#: What each forward captures, in this order.
CAPTURED = ("attention_output", "moe_input", "router_logits", "top_k_index",
            "top_k_weights")
#: The planes the census compares elementwise (E4).
PLANES = ("attention_output", "moe_input", "router_logits")
#: The per-token arrays a window shard holds.
TOKEN_ARRAYS = ("fallback_experts", "kernel_experts", "changed", "fallback_margin",
                "kernel_margin", "choice_delta", "logit_delta", "moe_input_rel",
                "weight_rel", "reassigned")
#: The GLM classes the hooks attach to; ``LayerCapture`` refuses others.
GLM_CLASSES = {"router": "Glm5NextTextTopkRouter", "experts": "Glm5NextTextExperts"}
#: The campaign spec entries the census does not use: the Stage B replay,
#: spill and produced-output spool (``derive_census_spec``).
SPEC_ENV_DROPPED = (
    "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES", "PRISMABUILD_PRODUCED_SPOOL_ROOT",
    "PRISMAQUANT_STAGE_B_REPLAY_REGIME", "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES",
    "PRISMAQUANT_STAGE_B_SPILL_ROOT", "PRISMAQUANT_STAGE_B_KDA_KERNEL")
COTANGENT_ROOT_ENV = "PRISMAQUANT_STAGE_B_COTANGENT_ROOT"
COTANGENT_MAX_ENV = "PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES"


class CensusRefused(RuntimeError):
    """An input, a count or a capture the census cannot measure with."""


class StopAtExperts(BaseException):
    """Raised by the experts pre-hook: the forward stops before the experts run.

    A ``BaseException`` so that no ``except Exception`` in the model code
    swallows it, as ``glm_routing_replay`` stops its replay.
    """


# ---------------------------------------------------------------------------
# Pure comparisons (CPU-testable)
# ---------------------------------------------------------------------------


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Equal shape, dtype and bytes; ``-0.0`` differs from ``0.0`` here."""
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.is_floating_point():
        width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[left.element_size()]
        return torch.equal(left.contiguous().view(width), right.contiguous().view(width))
    return torch.equal(left, right)


def differing_elements(left: torch.Tensor, right: torch.Tensor) -> int:
    """How many elements differ in their bytes."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise CensusRefused("differing_elements compares tensors of one shape and dtype")
    if left.is_floating_point():
        width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[left.element_size()]
        return int((left.contiguous().view(width) != right.contiguous().view(width)).sum())
    return int((left != right).sum())


def sorted_routes(top_k_index: torch.Tensor) -> torch.Tensor:
    """Each token's experts in ascending order; refuses a repeated expert."""
    if top_k_index.ndim != 2:
        raise CensusRefused("top-k indices are [tokens, k]")
    ordered = torch.sort(top_k_index.long(), dim=-1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise CensusRefused("a token selects one expert twice")
    return ordered


def aligned_weights(top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
    """The routing weights in ascending expert order, so two runs align by expert."""
    order = torch.sort(top_k_index.long(), dim=-1, stable=True).indices
    return torch.gather(top_k_weights.float(), -1, order)


def changed_experts(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Per token, how many of ``other``'s experts ``reference`` does not select.

    Both are ``[tokens, k]`` sets of distinct experts, so the count is the
    same in either direction.
    """
    present = (other.long().unsqueeze(-1) == reference.long().unsqueeze(-2)).any(-1)
    return (~present).sum(-1)


def selection_scores(router_logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """The router's selection scores, computed as the GLM router computes them."""
    scores = router_logits.float().sigmoid()
    return scores + bias


def recompute_selection(router_logits, bias, *, n_group, topk_group, top_k):
    """The router's noaux_tc choice again, with each token's margins.

    Returns ``(selected, expert_margin, group_margin)``: the selected
    experts ``[tokens, top_k]`` in descending score order, the gap between
    the ``top_k``-th and the next eligible selection score (``inf`` when no
    expert is left), and the gap between the ``topk_group``-th and the next
    group score (``None`` when every group is eligible).
    """
    choice = selection_scores(router_logits, bias)
    tokens, experts = choice.shape
    if experts % n_group:
        raise CensusRefused(f"{experts} experts do not split into {n_group} groups")
    per_group = experts // n_group
    group_scores = choice.view(tokens, n_group, per_group).topk(2, dim=-1).values.sum(-1)
    group_margin = None
    if topk_group < n_group:
        ranked_groups = group_scores.sort(dim=-1, descending=True).values
        group_margin = ranked_groups[:, topk_group - 1] - ranked_groups[:, topk_group]
    group_index = group_scores.topk(topk_group, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_index, 1).bool()
    expert_mask = group_mask.unsqueeze(-1).expand(tokens, n_group, per_group).reshape(
        tokens, experts)
    eligible = choice.masked_fill(~expert_mask, float("-inf"))
    ranked = eligible.sort(dim=-1, descending=True)
    selected = ranked.indices[:, :top_k]
    if top_k < experts:
        expert_margin = ranked.values[:, top_k - 1] - ranked.values[:, top_k]
    else:
        expert_margin = torch.full((tokens,), float("inf"), device=choice.device)
    return selected, expert_margin, group_margin


class PlaneStats:
    """E4's plane comparison, accumulated over groups in FP64.

    ``reference`` is the fallback's plane and ``other`` the kernel's.
    """

    FIELDS = ("elements", "sum_abs_reference", "differing", "max_abs", "sum_sq_diff",
              "sum_sq_reference")

    def __init__(self):
        self.elements = 0
        self.sum_abs_reference = 0.0
        self.differing = 0
        self.max_abs = 0.0
        self.sum_sq_diff = 0.0
        self.sum_sq_reference = 0.0

    def add(self, reference: torch.Tensor, other: torch.Tensor) -> None:
        self.differing += differing_elements(reference, other)
        ref = reference.double()
        diff = other.double() - ref
        self.elements += ref.numel()
        self.sum_abs_reference += float(ref.abs().sum())
        if diff.numel():
            self.max_abs = max(self.max_abs, float(diff.abs().max()))
        self.sum_sq_diff += float((diff * diff).sum())
        self.sum_sq_reference += float((ref * ref).sum())

    def partial(self) -> dict:
        return {field: getattr(self, field) for field in self.FIELDS}

    def merge(self, partial: dict) -> None:
        self.elements += int(partial["elements"])
        self.sum_abs_reference += float(partial["sum_abs_reference"])
        self.differing += int(partial["differing"])
        self.max_abs = max(self.max_abs, float(partial["max_abs"]))
        self.sum_sq_diff += float(partial["sum_sq_diff"])
        self.sum_sq_reference += float(partial["sum_sq_reference"])

    def as_dict(self) -> dict:
        elements = self.elements
        return {
            "elements": elements,
            "mean_abs_reference": self.sum_abs_reference / elements if elements else None,
            "differing_elements": self.differing,
            "differing_fraction": self.differing / elements if elements else None,
            "max_abs_diff": self.max_abs,
            "rel_fro": (math.sqrt(self.sum_sq_diff / self.sum_sq_reference)
                        if self.sum_sq_reference else None),
        }


def compare_modes(kernel: dict, fallback: dict, *, bias: torch.Tensor, routing: dict) -> dict:
    """Per-token comparison of one group's K1 (kernel) and F1 (fallback) captures.

    Returns CPU tensors, one value per token (``[tokens]``) or per token and
    slot (``[tokens, k]``), plus each mode's count of tokens whose recomputed
    selection is not the router's own (a tie at the boundary, or an
    instrument fault).
    """
    top_k = int(routing["top_k"])
    fallback_experts = sorted_routes(fallback["top_k_index"])
    kernel_experts = sorted_routes(kernel["top_k_index"])
    if fallback_experts.shape[1] != top_k or kernel_experts.shape[1] != top_k:
        raise CensusRefused(f"the router selected {fallback_experts.shape[1]} experts, not {top_k}")
    changed = changed_experts(fallback_experts, kernel_experts)
    flipped = changed > 0
    recompute_mismatches = {}
    margins = {}
    for mode, capture, experts in (("F1", fallback, fallback_experts),
                                   ("K1", kernel, kernel_experts)):
        selected, margin, _ = recompute_selection(
            capture["router_logits"], bias, n_group=int(routing["n_group"]),
            topk_group=int(routing["topk_group"]), top_k=top_k)
        recompute_mismatches[mode] = int((changed_experts(experts, selected) > 0).sum())
        margins[mode] = margin
    choice_delta = (selection_scores(kernel["router_logits"], bias)
                    - selection_scores(fallback["router_logits"], bias)).abs().amax(-1)
    logit_delta = (kernel["router_logits"].float()
                   - fallback["router_logits"].float()).abs().amax(-1)
    moe_kernel = kernel["moe_input"].float()
    moe_fallback = fallback["moe_input"].float()
    tiny = torch.finfo(torch.float32).tiny
    moe_input_rel = ((moe_kernel - moe_fallback).norm(dim=-1)
                     / moe_fallback.norm(dim=-1).clamp_min(tiny))
    weights_kernel = aligned_weights(kernel["top_k_index"], kernel["top_k_weights"])
    weights_fallback = aligned_weights(fallback["top_k_index"], fallback["top_k_weights"])
    weight_rel = ((weights_kernel - weights_fallback).abs()
                  / weights_fallback.abs().clamp_min(tiny)).amax(-1)
    weight_rel = weight_rel.masked_fill(flipped, float("nan"))
    scale = float(routing["routed_scaling_factor"])
    absent = ~(fallback_experts.unsqueeze(-1) == kernel_experts.unsqueeze(-2)).any(-1)
    reassigned = (weights_fallback / scale * absent).sum(-1)
    arrays = {
        "fallback_experts": fallback_experts.to(torch.int16),
        "kernel_experts": kernel_experts.to(torch.int16),
        "changed": changed.to(torch.uint8),
        "fallback_margin": margins["F1"].float(),
        "kernel_margin": margins["K1"].float(),
        "choice_delta": choice_delta.float(),
        "logit_delta": logit_delta.float(),
        "moe_input_rel": moe_input_rel.float(),
        "weight_rel": weight_rel.float(),
        "reassigned": reassigned.float(),
    }
    return {"arrays": {name: value.detach().cpu().contiguous() for name, value in arrays.items()},
            "recompute_mismatches": recompute_mismatches}


def _quantiles(values: torch.Tensor) -> dict | None:
    values = values[torch.isfinite(values)].double()
    if not values.numel():
        return None
    ordered = values.sort().values
    def at(q):
        return float(ordered[min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1)])
    return {"count": int(ordered.numel()), "min": float(ordered[0]), "p50": at(0.5),
            "p99": at(0.99), "max": float(ordered[-1]), "mean": float(ordered.mean())}


def summarize(arrays: dict, *, sequence_length: int, routing: dict) -> dict:
    """The census's per-layer answer from every token's arrays, in batch order."""
    changed = arrays["changed"].long()
    tokens = int(changed.numel())
    if tokens % sequence_length:
        raise CensusRefused(f"{tokens} tokens are not whole {sequence_length}-token sequences")
    flipped = changed > 0
    per_sequence = flipped.view(-1, sequence_length).sum(-1)
    histogram = torch.bincount(changed, minlength=int(routing["top_k"]) + 1)
    margin = arrays["fallback_margin"].double()
    choice = arrays["choice_delta"].double()
    exact_bound = int(routing["n_group"]) == int(routing["topk_group"])
    beyond = int((flipped & (margin > 2 * choice)).sum()) if exact_bound else None
    return {
        "tokens": tokens,
        "sequences": tokens // sequence_length,
        "flipped_tokens": int(flipped.sum()),
        "flipped_fraction": float(flipped.sum()) / tokens if tokens else None,
        "changed_expert_slots": int(changed.sum()),
        "changed_histogram": [int(value) for value in histogram],
        "sequences_with_a_flip": int((per_sequence > 0).sum()),
        "max_flips_in_a_sequence": int(per_sequence.max()) if tokens else 0,
        "flips_per_sequence": [int(value) for value in per_sequence],
        # A flip needs its fallback margin within twice the largest score
        # move (exact when every group is eligible); a count above zero is
        # an instrument fault.
        "flips_beyond_margin_bound": beyond,
        "fallback_margin_all": _quantiles(margin),
        "fallback_margin_flipped": _quantiles(margin[flipped]),
        "choice_delta_all": _quantiles(choice),
        "choice_delta_flipped": _quantiles(choice[flipped]),
        "logit_delta_all": _quantiles(arrays["logit_delta"]),
        "moe_input_rel_all": _quantiles(arrays["moe_input_rel"]),
        "weight_rel_unflipped": _quantiles(arrays["weight_rel"][~flipped]),
        "reassigned_flipped": _quantiles(arrays["reassigned"][flipped]),
        "reassigned_mass_total": float(arrays["reassigned"].double().sum()),
    }


# ---------------------------------------------------------------------------
# Hooks and the per-group driver
# ---------------------------------------------------------------------------


class LayerCapture:
    """Hooks on one decoder layer: attention output, router logits, expert inputs.

    The experts pre-hook records the arguments the packed experts would get
    and raises :class:`StopAtExperts`. Everything captured is detached, so a
    stopped forward keeps no autograd graph alive.
    """

    def __init__(self, layer_module, *, classes=GLM_CLASSES):
        self.attention = layer_module.self_attn
        self.router = layer_module.mlp.gate
        self.experts = layer_module.mlp.experts
        if classes is not None:
            found = {"router": type(self.router).__name__,
                     "experts": type(self.experts).__name__}
            if found != dict(classes):
                raise CensusRefused(f"the layer's MoE is {found}, not {dict(classes)}")
        self._handles = []
        self._taken = {}

    def __enter__(self):
        self._handles = [
            self.attention.register_forward_hook(self._attention_hook),
            self.router.register_forward_hook(self._router_hook),
            self.experts.register_forward_pre_hook(self._experts_hook, with_kwargs=True),
        ]
        return self

    def __exit__(self, *exc_info):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False

    def _store(self, name, value):
        if name in self._taken:
            raise CensusRefused(f"one forward produced {name} twice")
        if not isinstance(value, torch.Tensor):
            raise CensusRefused(f"{name} is not a tensor")
        self._taken[name] = value.detach()

    def _attention_hook(self, module, args, output):
        self._store("attention_output", output[0] if isinstance(output, tuple) else output)

    def _router_hook(self, module, args, output):
        if not isinstance(output, tuple) or len(output) != 3:
            raise CensusRefused("the router returned something other than "
                                "(logits, weights, indices)")
        self._store("router_logits", output[0])

    def _experts_hook(self, module, args, kwargs):
        bound = inspect.signature(type(module).forward).bind(module, *args, **kwargs)
        self._store("moe_input", bound.arguments["hidden_states"])
        self._store("top_k_index", bound.arguments["top_k_index"])
        self._store("top_k_weights", bound.arguments["top_k_weights"])
        raise StopAtExperts()

    def take(self) -> dict:
        taken, self._taken = self._taken, {}
        missing = [name for name in CAPTURED if name not in taken]
        if missing:
            raise CensusRefused(f"the forward captured no {missing}")
        return taken


def run_group(forward, capture: LayerCapture, staged: torch.Tensor, *, dispatch, counts,
              kda_layer: bool, fallback_active, rng_state=None, rng_fence=None,
              timer=None) -> dict:
    """One group's four forwards, in :data:`RUN_ORDER`; returns each mode's capture.

    ``forward(leaf)`` runs the layer on ``leaf``; ``dispatch`` is the
    kernel's context (``AdmittedKdaKernel._dispatch``); ``counts()`` reads
    the kernel's counters; ``fallback_active()`` says whether the module
    global is the fallback. Each forward gets its own leaf copy of
    ``staged`` with ``requires_grad``, as the fused chain roll stages it.
    A kernel forward of a KDA layer must add exactly :data:`KERNEL_FORWARD`
    to the counters; every other forward must add nothing.
    """
    captures = {}
    timings = {}
    deltas = {}
    for mode in RUN_ORDER:
        kernel = mode.startswith("K")
        if not kernel and not fallback_active():
            raise CensusRefused(f"{mode}: the KDA dispatch is not the fallback")
        leaf = staged.clone().requires_grad_(True)
        saved = rng_state() if rng_state is not None else None
        before = counts()
        started = timer() if timer is not None else None
        reached = False
        with (dispatch if kernel else nullcontext()):
            try:
                forward(leaf)
            except StopAtExperts:
                reached = True
        if timer is not None:
            timings[mode] = timer(started)
        after = counts()
        if rng_fence is not None:
            rng_fence(saved)
        if not reached:
            raise CensusRefused(f"{mode}: the forward finished without reaching the packed experts")
        delta = {key: after[key] - before[key] for key in after}
        expected = KERNEL_FORWARD if (kernel and kda_layer) else NO_KERNEL
        if delta != expected:
            raise CensusRefused(f"{mode}: the kernel counters moved by {delta}, "
                                f"not {expected} (KDA layer: {kda_layer})")
        deltas[mode] = delta
        captures[mode] = capture.take()
        leaf = None
    return {"captures": captures, "timings_ms": timings, "count_deltas": deltas}


def group_checks(captures: dict) -> dict:
    """Bitwise repeat and null checks for one group, per captured name."""
    return {
        "kernel_repeat": {name: bitwise_equal(captures["K1"][name], captures["K2"][name])
                          for name in CAPTURED},
        "fallback_repeat": {name: bitwise_equal(captures["F1"][name], captures["F2"][name])
                            for name in CAPTURED},
        "kernel_equals_fallback": {name: bitwise_equal(captures["K1"][name],
                                                       captures["F1"][name])
                                   for name in CAPTURED},
    }


# ---------------------------------------------------------------------------
# The data manifest and the container spec
# ---------------------------------------------------------------------------


def phase_names(layer: int) -> tuple[str, str, str]:
    return ("head", f"layer-{int(layer):03d}-source", f"layer-{int(layer):03d}-bound")


def boundary_records(adjoint_slice: dict, layer: int) -> list[dict]:
    """The slice's input boundaries of ``layer``, in batch order."""
    entries = adjoint_slice["boundary_entries"].get(str(int(layer)))
    if not entries:
        raise CensusRefused(f"the slice holds no boundaries of layer {layer}")
    by_name = {entry["name"]: entry for entry in entries}
    records = []
    for batch in range(len(entries)):
        name = f"boundary-{batch}-{int(layer)}-at-{int(layer)}"
        if name not in by_name:
            raise CensusRefused(f"the slice has no boundary entry {name}")
        records.append(by_name[name])
    return records


def build_census_manifest(readset: dict, *, readset_path: str, readset_sha256: str,
                          adjoint_slice: dict, slice_sha256: str, record: dict,
                          record_sha256: str, plan: dict, plan_sha256: str,
                          layer: int) -> dict:
    """The census's data manifest v2: three phases, each from the row's own reads.

    * ``head``: the entries of the row readset's ``head`` phase that the
      census reads through the staged readers, the calibration tokens and
      the model's head shards. The census reads its small JSON inputs
      directly.
    * ``layer-LLL-source``: the row readset's source phase of ``layer``,
      ``chain-LLL-source`` for a chain layer, ``own-LLL-source`` for the
      row's own layer.
    * ``layer-LLL-bound``: the layer's input boundaries, one entry per
      stored batch, each the row readset's entry for the same file with the
      slice's size and digest.
    """
    layer = int(layer)
    entries = readset["entries"]
    phases = {phase["name"]: phase for phase in readset["read_plan"]["phases"]}
    chain = [int(value) for value in record["adjoint"]["chain_layers"]]
    if layer == int(record["layer"]):
        source_phase = f"own-{layer:03d}-source"
    elif layer in chain:
        source_phase = f"chain-{layer:03d}-source"
    else:
        raise CensusRefused(f"layer {layer} is neither row {record['layer']}'s layer nor "
                            f"in its chain {chain}")
    for name in ("head", source_phase):
        if name not in phases:
            raise CensusRefused(f"the row readset has no {name} phase")
    model_dir = str(plan["model"]).rstrip("/") + "/"
    calibration = plan["calibration_input"]
    head = []
    for index in phases["head"]["entry_indices"]:
        entry = entries[index]
        if entry["path"] == calibration["path"]:
            if entry["sha256"] != calibration["sha256"] or entry["offset"] != 0:
                raise CensusRefused("the readset's calibration entry is not the plan's")
            head.append(index)
        elif entry["path"].startswith(model_dir):
            head.append(index)
    if not any(entries[index]["path"] == calibration["path"] for index in head):
        raise CensusRefused("the row readset's head phase does not read the calibration tokens")
    by_file = {}
    for index, entry in enumerate(entries):
        if entry["offset"] == 0:
            by_file.setdefault(entry["path"], index)
    bound = []
    for boundary in boundary_records(adjoint_slice, layer):
        index = by_file.get(boundary["path"])
        if index is None:
            raise CensusRefused(f"the row readset does not read {boundary['path']}")
        entry = entries[index]
        if entry["bytes"] != boundary["file_bytes"] or entry["sha256"] not in (
                None, boundary["sha256"]):
            raise CensusRefused(f"the row readset's entry for {boundary['name']} is not "
                                "the slice's")
        bound.append(index)
    new_entries, position = [], {}
    phase_indices = []
    for indices in (head, phases[source_phase]["entry_indices"], bound):
        mine = []
        for index in indices:
            entry = entries[index]
            key = (entry["path"], entry["offset"])
            if key not in position:
                position[key] = len(new_entries)
                new_entries.append({"path": entry["path"], "offset": entry["offset"],
                                    "bytes": entry["bytes"], "sha256": entry["sha256"]})
            if position[key] in mine:
                raise CensusRefused(f"a phase reads {key} twice")
            mine.append(position[key])
        phase_indices.append(mine)
    names = phase_names(layer)
    read_phases, cumulative = [], 0
    for name, indices in zip(names, phase_indices):
        size = sum(new_entries[index]["bytes"] for index in indices)
        cumulative += size
        read_phases.append({"name": name, "entry_indices": indices, "bytes": size,
                            "cumulative_bytes": cumulative})
    return {
        "schema": MANIFEST_SCHEMA,
        "produced_by": {"tool": TOOL, "entry_point": ENTRY_POINT,
                        "source_readset": str(readset_path),
                        "source_readset_sha256": readset_sha256},
        "annotations": {
            "census_layer": layer, "quantum_id": record["quantum_id"],
            "record_sha256": record_sha256, "slice_sha256": slice_sha256,
            "plan_sha256": plan_sha256,
            "phase_sources": {
                names[0]: "row readset phase head: the calibration tokens and model shard entries",
                names[1]: f"row readset phase {source_phase}",
                names[2]: f"slice boundaries boundary-*-{layer}-at-{layer}, row readset entries",
            },
        },
        "mount_prefix": readset["mount_prefix"],
        "entries": new_entries,
        "entry_count": len(new_entries),
        "total_bytes": sum(entry["bytes"] for entry in new_entries),
        "read_plan": {"phases": read_phases, "read_bytes": cumulative},
    }


def derive_census_spec(spec: dict, *, cotangent_max_bytes: int) -> tuple[dict, list[str]]:
    """The census container spec from a campaign spec, and what changed.

    Drops the Stage B replay, spill, produced-spool and kernel-mode settings
    and their mounts: the census writes none of them and admits its kernel
    itself. Keeps the cotangent scratch root, with ``cotangent_max_bytes``,
    because the launcher puts the container's write caches (Triton,
    inductor, temp) under it; the census writes no cotangent.
    """
    derived = json.loads(json.dumps(spec))
    env = derived["env"]
    changes = []
    dropped_roots = set()
    for name in SPEC_ENV_DROPPED:
        if name in env:
            if name.endswith("_ROOT"):
                dropped_roots.add(env[name])
            changes.append(f"env {name} removed")
            del env[name]
    if COTANGENT_ROOT_ENV not in env:
        raise CensusRefused("the campaign spec declares no cotangent scratch root for the caches")
    changes.append(f"env {COTANGENT_MAX_ENV} {env.get(COTANGENT_MAX_ENV)} -> {cotangent_max_bytes}")
    env[COTANGENT_MAX_ENV] = str(int(cotangent_max_bytes))
    mounts = []
    for mount in derived["container"]["mounts"]:
        if mount["source"] in dropped_roots:
            changes.append(f"mount {mount['source']} removed")
            continue
        mounts.append(mount)
    derived["container"]["mounts"] = mounts
    return derived, changes


# ---------------------------------------------------------------------------
# The GPU run
# ---------------------------------------------------------------------------


def _read_bound(path, sha256, label) -> bytes:
    raw = Path(path).read_bytes()
    if sha256_bytes(raw) != sha256:
        raise CensusRefused(f"{label} {path} does not hash to {sha256}")
    return raw


def _atomic_write(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()


def _clean(value):
    """JSON-safe: non-finite floats become strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


class Progress:
    """Cumulative durable units under the declared phases (PrismaBuild #480)."""

    def __init__(self, names):
        from prismaquant.joint_run_progress import declared_phases

        declared = declared_phases()
        if declared is not None and tuple(declared) != tuple(names):
            raise CensusRefused(f"the action declares phases {list(declared)}, "
                                f"the census reports {list(names)}")
        self.declared = declared
        self.units = 0
        self.phase = None

    def enter(self, name: str) -> None:
        self.phase = name
        self._report()

    def commit(self, units: int) -> None:
        self.units = int(units)
        self._report()

    def _report(self) -> None:
        from prismaquant.prismabuild_progress import report

        report(self.phase, self.units, unit="census_files")


def run_census(args) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from prismaquant.cost_streaming import build_streamed_model_identity
    from prismaquant.dev_mode import seal_check
    from prismaquant.glm_kda_capture_kernel import admit_kda_capture_kernel, layer_runs_kda
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.joint_adjoint_checkpoints import (
        GpuPowerSampler, _chain_group_batch, _chain_rng_fence, _chain_rng_state,
        _require_per_sample_state, _stage_to_device, read_exact_entry_tensors)
    from prismaquant.joint_adjoint_slices import chain_regime_of, load_adjoint_slice
    from prismaquant.joint_aura import source_execution_identity
    from prismaquant.joint_cost_quantum import (
        _install_with_settlement, _rebuild_batches, build_quantum_source_runner,
        require_slice_bf16_reduction)
    from prismaquant.joint_projection_backend import executing_image
    from prismaquant.kernels import kda_chunk
    from prismaquant.matmul_arithmetic import pin_matmul_arithmetic
    from prismaquant.residency_map import bind_residency_manifest, residency_report
    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    from prismaquant.tessera_joint_aura import _load_plan
    from prismaquant.calibration_data import load_calibration_input
    from safetensors.torch import save_file

    started_wall = time.time()
    layer = int(args.layer)
    names = phase_names(layer)
    progress = Progress(names)
    progress.enter(names[0])
    require_cuda_hot_path("kda_route_census", "cuda")

    # ---- inputs, each bound by digest --------------------------------------
    record_raw = _read_bound(args.record, args.record_sha256, "record")
    record = json.loads(record_raw)
    adjoint_slice = load_adjoint_slice(
        args.adjoint_slice, args.adjoint_slice_sha256, layer=int(record["layer"]),
        checkpoint_boundary=int(record["adjoint"]["checkpoint_boundary"]))
    if record["adjoint"]["slice_sha256"] != args.adjoint_slice_sha256:
        raise CensusRefused("the record binds another Stage A slice")
    config = _load_plan(args.plan, args.plan_sha256, defer_pool_reads=True)
    prepared = json.loads(_read_bound(args.prepared, args.prepared_sha256, "prepared"))
    chain = [int(value) for value in record["adjoint"]["chain_layers"]]
    if layer != int(record["layer"]) and layer not in chain:
        raise CensusRefused(f"layer {layer} is not row {record['layer']}'s layer or chain {chain}")
    output_root = Path(args.output_root)
    # The census writes only under its own root: never beside an input, and
    # never inside a row's or a plan's output space.
    for owned in (config["output_root"], record["output_space"]["root"],
                  Path(args.record).parent, Path(args.adjoint_slice).parent,
                  Path(args.plan).parent, Path(args.prepared).parent,
                  adjoint_slice["boundary_storage"]["directory"]):
        owned = Path(owned).resolve()
        if output_root.resolve() == owned or owned in output_root.resolve().parents:
            raise CensusRefused(f"the census output root {output_root} is inside {owned}")
    if output_root.exists() and any(output_root.iterdir()):
        raise CensusRefused(f"the census output root {output_root} is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    shards = output_root / "shards"
    shards.mkdir()
    execution = config["execution"]
    if int(execution.get("probe_microbatch", 0)) != 1:
        raise CensusRefused("the census maps one stored batch to one sequence "
                            "(probe_microbatch 1)")
    regime = chain_regime_of(adjoint_slice["run_identity"])
    group_size = int(regime["batch_size"])
    boundaries = boundary_records(adjoint_slice, layer)
    n_batches = len(boundaries)
    if n_batches % group_size:
        raise CensusRefused(f"{n_batches} batches do not split into groups of {group_size}")
    groups_total = n_batches // group_size
    groups = groups_total if args.groups is None else min(int(args.groups), groups_total)
    window_groups = int(args.window_groups)
    session = adjoint_slice["boundary_storage"]["session"]

    # ---- arithmetic and tiers, as the quantum pins them ---------------------
    torch.set_num_threads(1)
    allow_bf16 = pin_matmul_arithmetic()
    require_slice_bf16_reduction(adjoint_slice, allow_bf16, where="kda route census")
    if args.allowed_tiers is not None:
        allowed = activate_staged_tier_policy(args.allowed_tiers)
        print(f"[STAGED-TIER] bulk inputs serve from {','.join(sorted(allowed))}", flush=True)
    bind_residency_manifest(args.data_manifest_sha256)

    power = GpuPowerSampler().start()
    ids, calibration = load_calibration_input(
        config["calibration_input"]["path"],
        expected_sha256=config["calibration_input"]["sha256"],
        n_samples=int(execution["n_calib_samples"]), seqlen=int(execution["calib_seqlen"]))
    calibration_sha256 = sha256_bytes(ids.detach().cpu().contiguous().numpy().tobytes())
    if calibration_sha256 != adjoint_slice["run_identity"]["calibration_sha256"]:
        raise CensusRefused("the calibration tokens are not the ones Stage A captured")
    if int(ids.shape[0]) != n_batches:
        raise CensusRefused(f"{ids.shape[0]} calibration rows for {n_batches} boundaries")
    identity_binding = config["source_identity_cache"]
    identity_bytes = _read_bound(identity_binding["path"], identity_binding["sha256"],
                                 "source identity cache")
    head_tensors = ((record.get("executable_readset") or {}).get("head_source")
                    or {}).get("tensors")
    runner = build_quantum_source_runner(
        config, offload_folder=output_root / "run" / "offload",
        sealed_head_tensors=head_tensors)
    source = build_streamed_model_identity(runner, config["model"],
                                           identity_cache_bytes=identity_bytes)
    source_matched = seal_check(
        "prepared source identity", prepared.get("source_model_identity"), source,
        where="kda route census: the prepared source versus the running one",
        refusal=lambda: CensusRefused("prepared source identity: identity mismatch"))
    execution_identity = source_execution_identity(runner.model)
    execution_matched = execution_identity == prepared.get("source_execution")
    admitted = admit_kda_capture_kernel(args.kernel, runner.model, device=runner.device)
    dispatch = admitted._dispatch
    modeling = dispatch._modeling
    fallback = dispatch._fallback
    batches = _rebuild_batches(runner, partitions=[ids[row:row + 1] for row in range(n_batches)],
                               shared_pass={})
    header = {
        "schema": SCHEMA, "tool_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "git_commit": os.environ.get("PRISMAQUANT_IDENTITY_GIT_COMMIT"),
        "layer": layer, "quantum_id": record["quantum_id"],
        "record": {"path": str(args.record), "sha256": args.record_sha256},
        "plan": {"path": str(args.plan), "sha256": args.plan_sha256},
        "prepared": {"path": str(args.prepared), "sha256": args.prepared_sha256},
        "adjoint_slice": {"path": str(args.adjoint_slice), "sha256": args.adjoint_slice_sha256},
        "data_manifest_sha256": args.data_manifest_sha256,
        "allowed_tiers": args.allowed_tiers,
        "chain_regime": dict(regime), "group_size": group_size, "groups": groups,
        "groups_total": groups_total, "window_groups": window_groups,
        "run_order": list(RUN_ORDER), "calibration_sha256": calibration_sha256,
        "kernel": {"name": admitted.identity["name"],
                   "identity_sha256": admitted.identity_sha256,
                   "qualification_sha256": admitted.qualification_sha256,
                   "qualification_matched": admitted.qualification_matched,
                   "identity": admitted.identity},
        "source_identity_matched": bool(source_matched),
        "source_execution_matched": bool(execution_matched),
        "container_content_sha256": executing_image(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "allow_bf16_reduced_precision_reduction": bool(allow_bf16),
    }

    # ---- the layer's source --------------------------------------------------
    progress.enter(names[1])
    source_started = time.time()
    # The quantum's own install, for a walk of one layer: nothing after it
    # is prefetched, because the readset declares no other layer.
    _install_with_settlement(runner, layer, operator_windows=None, order=(layer,),
                             settle_successors=False)
    source_s = time.time() - source_started
    _require_per_sample_state(runner, batches, layer, [], range(n_batches),
                              where="kda route census")
    layer_module = runner.layers[layer]
    kda_layer = layer_runs_kda(layer_module)
    declared_kda = runner.model.config.get_text_config().layer_types[layer] == "linear_attention"
    if kda_layer != declared_kda:
        raise CensusRefused(f"layer {layer}: KDA by class {kda_layer}, by config {declared_kda}")
    router = layer_module.mlp.gate
    bias = router.e_score_correction_bias.detach().float()
    routing = {"n_group": int(router.num_group), "topk_group": int(router.topk_group),
               "top_k": int(router.top_k), "routed_scaling_factor": float(router.routed_scaling_factor),
               "norm_topk_prob": bool(router.norm_topk_prob), "experts": int(router.num_experts)}
    header["routing"] = routing
    header["kda_layer"] = kda_layer

    # ---- the groups ----------------------------------------------------------
    progress.enter(names[2])
    device, dtype = runner.device, runner.dtype
    plane_stats = {name: PlaneStats() for name in PLANES}
    token_arrays = {name: [] for name in TOKEN_ARRAYS}
    group_rows = []
    journal = {"header": header, "windows": []}
    windows = [list(range(start, min(start + window_groups, groups)))
               for start in range(0, groups, window_groups)]

    def read_window(window):
        members = [batch for group in window
                   for batch in range(group * group_size, (group + 1) * group_size)]
        records = [boundaries[batch] for batch in members]
        tensors = read_exact_entry_tensors(records, expected_session=session)
        return {batch: tensors[record_["name"]] for batch, record_ in zip(members, records)}

    def timer(started=None):
        if started is None:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            return start
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        end.synchronize()
        return float(started.elapsed_time(end))

    torch.cuda.reset_peak_memory_stats(device)
    loop_power = GpuPowerSampler().start()
    loop_started = time.time()
    units = 0
    with LayerCapture(layer_module) as capture, ThreadPoolExecutor(max_workers=1) as reader:
        pending = reader.submit(read_window, windows[0]) if windows else None
        for position, window in enumerate(windows):
            window_started = time.time()
            read_wait = time.time()
            window_tensors = pending.result()
            read_wait = time.time() - read_wait
            pending = (reader.submit(read_window, windows[position + 1])
                       if position + 1 < len(windows) else None)
            window_arrays = {name: [] for name in TOKEN_ARRAYS}
            window_planes = {name: PlaneStats() for name in PLANES}
            window_rows = []
            for group in window:
                members = list(range(group * group_size, (group + 1) * group_size))
                staged = _stage_to_device([window_tensors.pop(batch) for batch in members],
                                          device=device, dtype=dtype).detach()
                batch = _chain_group_batch(runner, batches, members, {})
                result = run_group(
                    lambda leaf: runner.isolated_layer(batch, layer, leaf, pass_state={}),
                    capture, staged, dispatch=dispatch, counts=kda_chunk.counts,
                    kda_layer=kda_layer,
                    fallback_active=lambda: modeling.chunk_kimi_delta_attention is fallback,
                    rng_state=lambda: _chain_rng_state(device),
                    rng_fence=lambda saved: _chain_rng_fence(saved, device), timer=timer)
                captures = result["captures"]
                checks = group_checks(captures)
                compared = compare_modes(captures["K1"], captures["F1"], bias=bias,
                                         routing=routing)
                for name in PLANES:
                    window_planes[name].add(captures["F1"][name], captures["K1"][name])
                for name in TOKEN_ARRAYS:
                    window_arrays[name].append(compared["arrays"][name])
                window_rows.append({
                    "group": group, "batches": members, "checks": checks,
                    "timings_ms": result["timings_ms"],
                    "count_deltas": result["count_deltas"],
                    "recompute_mismatches": compared["recompute_mismatches"],
                    "flipped_tokens": int((compared["arrays"]["changed"] > 0).sum())})
                staged = captures = result = None
            tensors = {name: torch.cat(window_arrays[name]) for name in TOKEN_ARRAYS}
            partials = {name: window_planes[name].partial() for name in PLANES}
            shard = shards / f"window-{position:03d}.safetensors"
            temporary = shard.with_name(f".{shard.name}.{os.getpid()}.tmp")
            save_file(tensors, str(temporary), metadata={
                "schema": SCHEMA, "window": str(position),
                "groups": json.dumps(window), "rows": json.dumps(_clean(window_rows)),
                "planes": json.dumps(partials)})
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, shard)
            raw = shard.read_bytes()
            journal["windows"].append({
                "window": position, "groups": [window[0], window[-1] + 1],
                "shard": str(shard.relative_to(output_root)), "sha256": sha256_bytes(raw),
                "bytes": len(raw), "wall_s": time.time() - window_started,
                "read_wait_s": read_wait})
            _atomic_write(output_root / "journal.json", _json_bytes(_clean(journal)))
            units += 1
            progress.commit(units)
            for name in TOKEN_ARRAYS:
                token_arrays[name].append(tensors[name])
            for name in PLANES:
                plane_stats[name].merge(partials[name])
            group_rows.extend(window_rows)
            print(f"kda route census: layer {layer} window {position + 1}/{len(windows)} "
                  f"groups {window[0]}..{window[-1]} flipped "
                  f"{sum(row['flipped_tokens'] for row in window_rows)} "
                  f"wall {time.time() - window_started:.1f}s", flush=True)
    loop_s = time.time() - loop_started
    loop_block = loop_power.stop()
    peak = {"max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(device))}
    runner.context.unload(layer)

    # ---- the answer ------------------------------------------------------------
    arrays = {name: torch.cat(parts) if parts else torch.empty(0)
              for name, parts in token_arrays.items()}
    sequence_length = int(execution["calib_seqlen"])
    failures = []
    repeat = {}
    for check in ("kernel_repeat", "fallback_repeat", "kernel_equals_fallback"):
        repeat[check] = {name: sum(1 for row in group_rows if not row["checks"][check][name])
                         for name in CAPTURED}
    if any(repeat["kernel_repeat"].values()) or any(repeat["fallback_repeat"].values()):
        failures.append("a repeated forward was not bitwise equal to the first")
    if not kda_layer and any(repeat["kernel_equals_fallback"].values()):
        failures.append("a layer without KDA attention differs between the kernel's "
                        "dispatch and the fallback")
    mismatches = {mode: sum(row["recompute_mismatches"][mode] for row in group_rows)
                  for mode in ("F1", "K1")}
    summary = summarize(arrays, sequence_length=sequence_length, routing=routing) if group_rows else None
    if summary is not None and summary["flips_beyond_margin_bound"]:
        failures.append("a flip lies outside its margin bound")
    timings = {mode: sum(row["timings_ms"][mode] for row in group_rows) for mode in RUN_ORDER}
    census = {
        **header,
        "host": socket.gethostname(),
        "groups_measured": len(group_rows),
        "instrument": {"ok": not failures, "failures": failures,
                       "groups_not_bitwise": repeat,
                       "recompute_mismatches": mismatches},
        "routing_census": summary,
        "planes": {name: plane_stats[name].as_dict() for name in PLANES},
        "timing": {"forward_ms_by_mode": timings, "loop_s": loop_s, "source_install_s": source_s,
                   "wall_s": time.time() - started_wall},
        "gpu_power_loop": loop_block,
        "gpu_power_run": power.stop(),
        "gpu_memory_peak": peak,
        # The process's own host peak, next to its GPU peak: on unified memory the
        # two together are what a later census reserves (Linux reports KiB).
        "host_memory_peak": {"ru_maxrss_bytes":
                             int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024},
        "residency": residency_report(),
        "journal": {"path": "journal.json",
                    "sha256": sha256_bytes((output_root / "journal.json").read_bytes())
                    if journal["windows"] else None},
    }
    _atomic_write(output_root / "census.json", _json_bytes(_clean(census)))
    units += 1
    progress.commit(units)
    print(json.dumps(_clean({"layer": layer, "kda_layer": kda_layer,
                             "instrument_ok": not failures,
                             "flipped_tokens": (summary or {}).get("flipped_tokens"),
                             "tokens": (summary or {}).get("tokens")})), flush=True)
    return 0 if not failures else 5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _manifest_command(args) -> int:
    import gzip

    readset_raw = _read_bound(args.readset, args.readset_sha256, "row readset")
    readset = json.loads(gzip.decompress(readset_raw) if readset_raw[:2] == b"\x1f\x8b"
                         else readset_raw)
    adjoint_slice = json.loads(_read_bound(args.adjoint_slice, args.adjoint_slice_sha256, "slice"))
    record = json.loads(_read_bound(args.record, args.record_sha256, "record"))
    plan = json.loads(_read_bound(args.plan, args.plan_sha256, "plan"))
    manifest = build_census_manifest(
        readset, readset_path=args.readset, readset_sha256=args.readset_sha256,
        adjoint_slice=adjoint_slice, slice_sha256=args.adjoint_slice_sha256, record=record,
        record_sha256=args.record_sha256, plan=plan, plan_sha256=args.plan_sha256,
        layer=args.layer)
    out = Path(args.out)
    if out.exists():
        raise CensusRefused(f"{out} exists")
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _atomic_write(out, raw)
    print(json.dumps({"manifest": str(out), "sha256": sha256_bytes(raw),
                      "phases": [{key: phase[key] for key in ("name", "bytes")}
                                 | {"entries": len(phase["entry_indices"])}
                                 for phase in manifest["read_plan"]["phases"]],
                      "total_bytes": manifest["total_bytes"]}, indent=1))
    return 0


def _spec_command(args) -> int:
    spec = json.loads(_read_bound(args.spec, args.spec_sha256, "campaign spec"))
    derived, changes = derive_census_spec(spec, cotangent_max_bytes=args.cotangent_max_bytes)
    out = Path(args.out)
    if out.exists():
        raise CensusRefused(f"{out} exists")
    raw = (json.dumps(derived, indent=1, sort_keys=True) + "\n").encode()
    _atomic_write(out, raw)
    print(json.dumps({"spec": str(out), "sha256": sha256_bytes(raw), "changes": changes},
                     indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=ENTRY_POINT, description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def bound(sub, name, help_text):
        sub.add_argument(f"--{name}", required=True, help=help_text)
        sub.add_argument(f"--{name}-sha256", required=True)

    manifest = commands.add_parser("manifest", help="build one layer's census data manifest")
    bound(manifest, "readset", "the row's executable readset (data manifest v2, gzip or plain)")
    bound(manifest, "adjoint-slice", "the row's Stage A slice")
    bound(manifest, "record", "the row's quantum record")
    bound(manifest, "plan", "the plan whose model and calibration the census reads")
    manifest.add_argument("--layer", type=int, required=True)
    manifest.add_argument("--out", required=True)

    spec = commands.add_parser("spec", help="derive the census container spec")
    bound(spec, "spec", "the campaign container spec")
    spec.add_argument("--cotangent-max-bytes", type=int, required=True)
    spec.add_argument("--out", required=True)

    run = commands.add_parser("run", help="run the census on one layer (GPU)")
    bound(run, "plan", "the plan (model, calibration, source prefetch, derivative)")
    bound(run, "prepared", "the prepared record (source identity)")
    bound(run, "adjoint-slice", "the row's Stage A slice (input boundaries)")
    bound(run, "record", "the row's quantum record (layer, chain, head source)")
    run.add_argument("--layer", type=int, required=True)
    run.add_argument("--kernel", default="kda_gram_v1")
    run.add_argument("--output-root", required=True)
    run.add_argument("--data-manifest-sha256", required=True)
    run.add_argument("--allowed-tiers", default="ram,ssd")
    run.add_argument("--groups", type=int, default=None,
                     help="measure only the first N groups (a smoke run)")
    run.add_argument("--window-groups", type=int, default=16)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "manifest":
            return _manifest_command(args)
        if args.command == "spec":
            return _spec_command(args)
        return run_census(args)
    except CensusRefused as exc:
        print(f"KDA ROUTE CENSUS REFUSED: {exc}", flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
