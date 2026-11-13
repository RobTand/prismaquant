#!/usr/bin/env python3
"""sensitivity_probe.py — model staging and model-structure helpers.

Text-only and multimodal staging of a checkpoint, MoE expert and router
discovery with the router tracker, grouped-Linear dispatch, the packed-expert
slice index and the cross-layer K/V cotangent accumulator that the AURA cost
stages use. The probe Fisher that gave this module its name is removed
(#3448).
"""
from __future__ import annotations

from . import probe_inventory

import json
import atexit
import os
import shutil
from collections import defaultdict
from pathlib import Path
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


_STAGED_TEMP_DIRS: list[Path] = []


def _prismaquant_temp_parent() -> Path:
    root = (
        os.environ.get("PRISMAQUANT_TMPDIR")
        or os.environ.get("TMPDIR")
    )
    parent = Path(root) if root else Path.cwd() / ".prismaquant_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _mk_stage_dir(prefix: str) -> Path:
    staged = Path(tempfile.mkdtemp(
        prefix=prefix,
        dir=str(_prismaquant_temp_parent()),
    ))
    _STAGED_TEMP_DIRS.append(staged)
    return staged


def _cleanup_stage_dirs() -> None:
    for path in reversed(_STAGED_TEMP_DIRS):
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_stage_dirs)


# ---------------------------------------------------------------------------
# Text-only staging
# ---------------------------------------------------------------------------
#
# One rule, one home (issue #210): "which config keys make a checkpoint
# text-only when no profile claims it" and the staging steps that act on
# them live ONLY in `_stage_text_only_impl` below. `stage_text_only`
# (this module) and `perturbed_x_cache.stage_text_only_under_work_root`
# are both thin wrappers around it, differing only in where the staged
# directory is created:
#   - `stage_text_only`: under `_prismaquant_temp_parent()` (TMPDIR-ish),
#     registered in `_STAGED_TEMP_DIRS` for `atexit` cleanup.
#   - `stage_text_only_under_work_root`: under an explicit `work_root`
#     the caller owns, never under /tmp, with no `atexit` registration.
# Before this merge the two call sites carried separately-typed copies of
# the same hardcoded fallback strip-key list (7 keys, textually identical
# in both files). Keep both public names and their exact signatures and
# behavior so no caller moves.
def _read_text_stage_config(path: Path) -> dict:
    """Read the input config separately from text-only transformation rules."""
    from .staged_whole_file import read_source_metadata_text

    return json.loads(read_source_metadata_text(path, label="text-only bootstrap config"))


def text_only_stage_config(config: dict, *, profile) -> dict | None:
    """Derive existing text-only staging rules without IO or input mutation.

    None preserves the source configuration; a dictionary is exactly what the
    filesystem staging writer previously emitted. Profile selection stays with
    the caller, including its existing dead-override refusal.
    """
    import copy

    cfg = copy.deepcopy(config)
    strip_keys = (list(profile.stage_text_only_strip_keys())
                  if profile is not None
                  else ["vision_config", "audio_config", "speech_config",
                        "image_token_id", "video_token_id",
                        "vision_start_token_id", "vision_end_token_id"])
    # Run staging when either (a) the model is multimodal (has a
    # config section that must be stripped for text-only load),
    # (b) any of the profile-declared strip_keys is present, or
    # (c) the config declares `num_local_experts` — we need to
    # alias it as `num_experts` for transformers 5.x's FP8 MoE
    # integration, which does `getattr(cfg, "num_local_experts",
    # cfg.num_experts)` with the fallback arg eagerly evaluated and
    # raises on the attribute access when only the long-form name
    # is defined.
    needs_num_experts_alias = (
        "num_local_experts" in cfg and "num_experts" not in cfg)
    if not any(k in cfg for k in
               ("vision_config", "text_config", "audio_config", "speech_config")) \
            and not any(k in cfg for k in strip_keys) \
            and not needs_num_experts_alias:
        return None
    promote_inner_mt = (profile.stage_text_only_promote_inner_model_type()
                        if profile is not None else False)

    for k in strip_keys:
        cfg.pop(k, None)

    # Transformers 5.x's `integrations/finegrained_fp8.py:__init__`
    # does `getattr(config, "num_local_experts", config.num_experts)` —
    # the fallback is evaluated eagerly and raises on MoE configs that
    # only declare `num_local_experts` (e.g. MiniMax M2). Alias both
    # names so the attribute access always succeeds.
    if "num_local_experts" in cfg and "num_experts" not in cfg:
        cfg["num_experts"] = cfg["num_local_experts"]

    if "text_config" in cfg:
        tc = cfg.pop("text_config")
        for k, v in tc.items():
            if k == "model_type":
                # Some families (Gemma 4) want the inner model_type
                # (e.g. gemma4_text) to take over so AutoConfig
                # resolves to the text-specific config class. Others
                # (Qwen 3.5 MoE) want the outer model_type to stay
                # because the ForCausalLM class is annotated with the
                # multimodal-umbrella config. Profile decides.
                if promote_inner_mt:
                    cfg[k] = v
                continue
            # text_config OVERRIDES top-level for the text-only staged
            # model. Multimodal models (Gemma 4, some Qwen variants)
            # carry their multimodal-combined hidden_size / num_layers /
            # head_dim at the top level and the text-specific values in
            # text_config. Loading weights needs the text_config values
            # since that's what the text checkpoint tensors were trained
            # with. If we didn't override, the loader sees mismatched
            # shapes ([2304] top-level vs [2816] actual weights) and
            # either raises or silently zero-inits the mismatched params.
            cfg[k] = v
    archs = cfg.get("architectures", [])
    if archs:
        cfg["architectures"] = [
            a.replace("ForConditionalGeneration", "ForCausalLM") for a in archs
        ]

    return cfg


def _stage_text_only_impl(
    model_path: str,
    *,
    staging_root: str | Path | None,
) -> str:
    src = Path(model_path)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return str(src)
    cfg = _read_text_stage_config(cfg_path)

    # Profile-driven: ask the registered ModelProfile which config keys
    # to strip and whether to promote `text_config.model_type`.
    from .model_profiles import DeadVendoredOverrideError, detect_profile
    try:
        profile = detect_profile(str(src))
    except DeadVendoredOverrideError:
        # The hardcoded default strip-key list below is for a checkpoint no
        # profile claims. On a dead override it stages the model with a
        # different config than the profile declares -- and every probe
        # statistic (sensitivity_probe) or cached activation row
        # (perturbed_x_cache) gathered afterwards describes that wrong
        # staging (#202).
        raise
    except Exception:
        profile = None
    rewritten = text_only_stage_config(cfg, profile=profile)
    if rewritten is None:
        return str(src)
    cfg = rewritten

    if staging_root is None:
        staged = _mk_stage_dir("prismaquant_stage_")
    else:
        root = Path(staging_root)
        root.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(prefix="prismaquant_stage_", dir=str(root)))
    skip = {"config.json", "preprocessor_config.json",
            "video_preprocessor_config.json", "processor_config.json"}
    for p in src.iterdir():
        if p.name in skip:
            continue
        (staged / p.name).symlink_to(p.resolve())
    with open(staged / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return str(staged)


def stage_text_only(model_path: str) -> str:
    return _stage_text_only_impl(model_path, staging_root=None)


# ---------------------------------------------------------------------------
# Multimodal staging (mirror of stage_text_only)
# ---------------------------------------------------------------------------
def stage_multimodal(model_path: str) -> str:
    """Stage a multimodal model PRESERVING `vision_config` (and `audio_config`
    where present). This is the mirror of `stage_text_only`, used by the
    multimodal streaming skeleton and the export: the loader must see the
    complete multimodal config so the visual tower materializes.

    Unlike `stage_text_only` we don't strip multimodal keys, don't promote
    `text_config.model_type`, and don't rewrite architectures. We symlink
    every source file (including preprocessor_config.json,
    video_preprocessor_config.json, processor_config.json) into the
    staged dir verbatim.

    When the source has no multimodal config keys, this is effectively a
    no-op and we return the source path unchanged — matching
    `stage_text_only`'s fast path.
    """
    src = Path(model_path)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return str(src)
    with open(cfg_path) as f:
        cfg = json.load(f)
    # If this checkpoint has no multimodal bits, nothing to do — return
    # the source directly so the caller doesn't pay for a symlink tree
    # on pure-text checkpoints.
    if not any(k in cfg for k in ("vision_config", "audio_config",
                                  "speech_config")):
        return str(src)

    staged = _mk_stage_dir("prismaquant_mm_stage_")
    for p in src.iterdir():
        if p.name == "config.json":
            continue
        (staged / p.name).symlink_to(p.resolve())
    # Write the source config through verbatim. We intentionally do NOT
    # strip vision_config/audio_config, do NOT rewrite architectures,
    # and do NOT promote text_config.model_type — the multimodal loader
    # needs the full config as-is.
    with open(staged / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return str(staged)


# ---------------------------------------------------------------------------
# Grouped-BMM Linear (`wo_a` shape)
#
# A declared grouped module (spec `probe.grouped_module_class_names`) is an
# nn.Linear subclass whose forward consumes its `[G*R, D]` weight plane as
# `W[g, r, d]`: `y[..., g, r] = sum_d x[..., g, d] * W[g, r, d]`
# (view + bmm in `DeepseekV4GroupedLinear.forward`). The dense accumulator's
# flatten-to-2D cannot represent this consumption: its chunk_h comes out
# [R, D] against an [G*R, D] weight (the `chunk_h * w.pow(2)` broadcast that
# motivated the original probe skip), and pooling groups into the token axis
# destroys the per-(g,r) channel marginals. The functions below are the ONE
# grouped accumulation mechanism; both backends fold its outputs into the
# same stats keys the dense rows use.
#
# TP note (walker campaign addendum, 2026-08-22): identity, dispositions,
# and every field below are properties of the WHOLE logical tensor. A future
# Tensor-Parallel shard of `wo_a` would cut the stored plane's ROW axis
# (axis 0 = G*R; the profile's `base_model_tp_plan` declares `rowwise`),
# so a shard boundary must not straddle a group (a cut inside g's R rows
# would make any future group-aware format illegal at that TP degree).
# Nothing here bakes rank/shard identity into keys or sizes; byte totals
# are TOTAL-of-logical, per-device splits are a serving-runtime concern.


def grouped_linear_groups(mod, profile=None) -> int | None:
    """Group count `G` for a profile-declared grouped-BMM Linear, else None.

    Dispatch is EXPLICIT: only classes the profile declares under
    ``probe.grouped_module_class_names`` are grouped — never a shape
    heuristic. A declared class whose instance lacks a usable ``n_groups``
    fails fast: silently falling through to the dense accumulator would
    produce the inflated, mis-shaped numbers the old probe skip existed to
    prevent.
    """
    import torch.nn as _nn
    if not isinstance(mod, _nn.Linear) or profile is None:
        return None
    try:
        declared = tuple(profile.probe_grouped_module_class_names())
    except AttributeError:
        declared = ()
    if type(mod).__name__ not in declared:
        return None
    n_groups = getattr(mod, "n_groups", None)
    if n_groups is None:
        raise ValueError(
            f"{type(mod).__name__} is declared under "
            "probe.grouped_module_class_names but carries no n_groups "
            "attribute; refusing to fall through to the dense Fisher "
            "accumulator (its numbers would be wrong-shaped)")
    g = int(n_groups)
    out_features = int(mod.out_features)
    if g <= 0 or out_features % g != 0:
        raise ValueError(
            f"{type(mod).__name__}: n_groups={g} does not divide "
            f"out_features={out_features}")
    return g


def _packed_expert_slice_index(weight: torch.Tensor,
                               param: torch.Tensor) -> int | None:
    """Expert index e such that `weight` is `param[e]` — a dim-0 select
    view of the packed [E, M, N] parameter. Returns None when `weight`
    is not such a slice (the caller then falls back to the plain linear
    and `_PackedWeightGradFallback` catches the weight gradient)."""
    if weight.dim() != 2 or param.dim() != 3:
        return None
    if weight.device.type == "meta" or param.device.type == "meta":
        return None
    if tuple(weight.shape) != tuple(param.shape[1:]):
        return None
    if tuple(weight.stride()) != tuple(param.stride()[1:]):
        return None
    stride0 = param.stride(0)
    if stride0 <= 0:
        return None
    try:
        if (weight.untyped_storage().data_ptr()
                != param.untyped_storage().data_ptr()):
            return None
    except Exception:
        return None
    off = weight.storage_offset() - param.storage_offset()
    if off < 0 or off % stride0 != 0:
        return None
    e = off // stride0
    if e >= param.shape[0]:
        return None
    return int(e)


def discover_moe_structure(
    model: nn.Module,
    profile=None,
) -> dict[str, tuple[str, str]]:
    """Return {expert_linear_qname: (router_qname, expert_id_str)}.

    Walk the module tree.  For any module that has a child attribute named
    `experts` or `block_sparse_moe_experts` that is a ModuleList, find a
    sibling Linear in the same parent whose out_features equals len(experts).
    That Linear is the router.
    """
    if profile is None:
        from .model_profiles import DeadVendoredOverrideError, profile_from_model
        try:
            profile = profile_from_model(model)
        except DeadVendoredOverrideError:
            # The profile names the packed-expert projections this walk looks
            # for. `None` narrows the candidate set silently, so MoE experts
            # go undiscovered and simply never get probed (#202).
            raise
        except Exception:
            profile = None
    projection_candidates = probe_inventory.packed_expert_projection_candidate_names(profile)

    def _router_matches_num_experts(child: nn.Module, num_experts: int) -> bool:
        if isinstance(child, nn.Linear) and child.out_features == num_experts:
            return True
        weight = getattr(child, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
            return int(weight.shape[0]) == num_experts
        return False

    expert_info: dict[str, tuple[str, str]] = {}
    for parent_qname, parent in model.named_modules():
        candidates = []
        for attr in ("experts", "block_sparse_moe_experts",
                     "moe_experts", "expert_layer"):
            experts_container = getattr(parent, attr, None)
            if experts_container is None or not isinstance(experts_container, nn.Module):
                continue
            # Two possible layouts:
            #   A) experts_container IS the list (nn.ModuleList / nn.Sequential /
            #      AutoRound's SequentialQwen3_5MoeExperts which subclasses ModuleList)
            #   B) experts_container is a plain nn.Module with numbered children
            #      (e.g. Qwen3_5MoeExperts after in-place unfuse: children are
            #      named "0", "1", ..., each holding per-expert Linears).
            #
            # Both layouts are detected by looking at child names that are
            # consecutive integer strings starting from 0.
            child_dict = dict(experts_container.named_children())
            numeric_keys = sorted(
                [k for k in child_dict if k.isdigit()],
                key=int,
            )
            if numeric_keys:
                # Require the numeric children to be 0..N-1 (no gaps)
                if [int(k) for k in numeric_keys] != list(range(len(numeric_keys))):
                    continue
                if not all(isinstance(child_dict[k], nn.Module) for k in numeric_keys):
                    continue
                candidates.append((attr, experts_container, "nested", numeric_keys))
                continue

            # Linear-loop layout after MoE unfuse: experts container itself
            # remains a module, but its packed projections become ModuleLists:
            #   experts.gate_up_proj.<expert_idx>
            #   experts.down_proj.<expert_idx>
            projection_lists = {}
            for proj_name in projection_candidates:
                proj = getattr(experts_container, proj_name, None)
                if proj is None or not isinstance(proj, nn.Module):
                    continue
                proj_children = dict(proj.named_children())
                proj_numeric = sorted([k for k in proj_children if k.isdigit()], key=int)
                if not proj_numeric:
                    continue
                if [int(k) for k in proj_numeric] != list(range(len(proj_numeric))):
                    continue
                if not all(isinstance(proj_children[k], nn.Module) for k in proj_numeric):
                    continue
                projection_lists[proj_name] = proj_numeric
            if projection_lists:
                # Require a consistent expert count across projections.
                expert_lists = list(projection_lists.values())
                if all(v == expert_lists[0] for v in expert_lists[1:]):
                    candidates.append((attr, experts_container, "linear_loop", expert_lists[0]))
        if not candidates:
            continue
        attr_name, experts_container, layout, numeric_keys = candidates[0]
        num_experts = len(numeric_keys)

        # Find sibling Linear (or any module whose output feature dim
        # equals num_experts) that acts as the router.
        router_qname = None
        for child_name, child in parent.named_children():
            if child is experts_container:
                continue
            if _router_matches_num_experts(child, num_experts):
                router_qname = (f"{parent_qname}.{child_name}"
                                if parent_qname else child_name)
                break
        if router_qname is None:
            continue

        experts_root = (f"{parent_qname}.{attr_name}"
                        if parent_qname else attr_name)
        if layout == "nested":
            for eid_str in numeric_keys:
                expert_mod = child_dict[eid_str]
                for sub_name, sub_mod in expert_mod.named_modules():
                    if not isinstance(sub_mod, nn.Linear) or sub_name == "":
                        continue
                    leaf = f"{experts_root}.{eid_str}.{sub_name}"
                    expert_info[leaf] = (router_qname, eid_str)
        else:
            for proj_name in projection_candidates:
                proj = getattr(experts_container, proj_name, None)
                if proj is None or not isinstance(proj, nn.Module):
                    continue
                proj_children = dict(proj.named_children())
                for eid_str in numeric_keys:
                    sub_mod = proj_children.get(eid_str)
                    if not isinstance(sub_mod, nn.Linear):
                        continue
                    leaf = f"{experts_root}.{proj_name}.{eid_str}"
                    expert_info[leaf] = (router_qname, eid_str)

    return expert_info


def discover_moe_routers(
    model: nn.Module,
    profile=None,
) -> dict[str, int]:
    """Return router module qnames and expert counts for all MoE blocks.

    ``discover_moe_structure`` intentionally describes *per-expert Linear*
    layouts.  Packed Qwen3.5/3.6 experts instead keep their projections as
    3-D parameters, so that function has no Linear leaves to return even
    though the sibling router is fully observable.  Coverage accounting only
    needs the router module itself; discover it independently from either a
    numbered expert container or a profile-declared packed 3-D parameter.
    """
    if profile is None:
        from .model_profiles import DeadVendoredOverrideError, profile_from_model
        try:
            profile = profile_from_model(model)
        except DeadVendoredOverrideError:
            # As in `discover_moe_structure` (#202): the profile declares the
            # packed 3-D parameters this discovery keys on, so `None` silently
            # under-reports routers and the coverage accounting built on them.
            raise
        except Exception:
            profile = None
    packed_names = probe_inventory.packed_expert_param_name_set(profile)
    routers: dict[str, int] = {}

    def _matches(child: nn.Module, num_experts: int) -> bool:
        if isinstance(child, nn.Linear) and child.out_features == num_experts:
            return True
        weight = getattr(child, "weight", None)
        return bool(
            isinstance(weight, torch.Tensor)
            and weight.ndim >= 1
            and int(weight.shape[0]) == num_experts
        )

    for parent_qname, parent in model.named_modules():
        for attr in ("experts", "block_sparse_moe_experts",
                     "moe_experts", "expert_layer"):
            experts = getattr(parent, attr, None)
            if not isinstance(experts, nn.Module):
                continue
            numeric = [name for name, _child in experts.named_children()
                       if name.isdigit()]
            num_experts = len(numeric)
            if not num_experts:
                for name, param in experts.named_parameters(recurse=False):
                    if (name in packed_names and isinstance(param, torch.Tensor)
                            and param.ndim == 3 and int(param.shape[0]) > 0):
                        num_experts = int(param.shape[0])
                        break
            if not num_experts:
                continue
            for child_name, child in parent.named_children():
                if child is experts:
                    continue
                if _matches(child, num_experts):
                    qname = (f"{parent_qname}.{child_name}"
                             if parent_qname else child_name)
                    routers[qname] = num_experts
                    break
    return routers


def read_top_k(model: nn.Module, default: int = 2) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return default
    for attr in ("num_experts_per_tok", "moe_top_k", "num_active_experts"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        for attr in ("num_experts_per_tok", "moe_top_k"):
            v = getattr(text_cfg, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    return default


# ---------------------------------------------------------------------------
# The KV-cotangent path: Fisher cotangents for cross-layer SHARED state
# ---------------------------------------------------------------------------
def kv_cotangent_path_enabled() -> bool:
    """Whether the reverse sweep routes shared-state cotangents back to the
    producing layer (default: yes).

    `PRISMAQUANT_KV_COTANGENT=0` restores the pre-fix severed-cotangent
    behavior for an A/B. One definition, read by every reverse sweep."""
    raw = os.environ.get("PRISMAQUANT_KV_COTANGENT")
    if raw is None:
        return True
    return raw not in ("0", "", "false", "False", "FALSE", "no", "NO")


class SharedStateCotangents:
    """Route the Fisher cotangent of cross-layer SHARED forward state back to
    the layer that PRODUCED it.

    **The measurement gap.** Some architectures share activations across layers
    within one forward pass: Gemma4's KV-sharing layers reuse the K/V computed
    by an earlier *storing* layer, threaded through the ``shared_kv_states``
    dict that ``ModelProfile.new_forward_pass_state`` declares. The Fisher
    probe's phase-3 forwards each layer in ISOLATION and hands a sharing layer
    the phase-1 *capture* of that K/V, which is necessarily detached (it is a
    CPU snapshot taken from a ``torch.no_grad()`` pass). A sharing layer's
    backward therefore stops dead at the borrowed K/V, and ``h_trace`` for the
    storing layer's ``k_proj``/``v_proj`` counts only its OWN layer's gradient
    — it misses the gradient flowing through every layer that consumes its
    K/V. The under-count lands precisely on the layers that feed other layers,
    i.e. exactly the ones the allocator should be most careful with. It also
    truncates the chained ``grad_out`` handed to every layer BELOW the storing
    layer, since that input gradient is missing the same paths.

    **The fix, in one reverse pass.** ``graft`` replaces each borrowed tensor
    with a grad-enabled LEAF clone. After that consumer's backward, ``harvest``
    reads ``leaf.grad`` — exactly the cotangent this consumer contributes to
    the producer's shared tensor — and sums it per (kwarg, source key,
    position); gradients from several consumers of one source add, which is
    what the sum is. When the sweep later reaches the producing layer,
    ``produced_roots`` returns the tensors that layer wrote into the container
    paired with those accumulated gradients, so the caller can drive one
    ``torch.autograd.backward([out, *roots], [grad_out, *grads])``. Autograd
    accumulates both contributions at the producer's K/V node before its
    ``grad_fn`` runs, so each Linear's full-backward hook still fires ONCE,
    with the total — the same gy an end-to-end backward would have delivered.

    **Why one pass suffices.** A consumer always comes AFTER its producer in
    the forward order (Gemma4 derives ``kv_shared_layer_index`` from the layers
    strictly BEFORE the sharing point), and phase-3 sweeps in REVERSE, so every
    consumer has been harvested by the time its producer is forwarded.

    Bookkeeping rules that keep this honest:

    - A borrowed leaf is never seeded as a root. It has no ``grad_fn`` (and its
      identity is tracked), so a source whose cotangent is still accumulating
      cannot be fed back into a later consumer and double-counted.
    - A seeded key is POPPED. Two layers writing one key would otherwise each
      collect the same consumers' cotangent; the sweep-end ``pending_keys``
      diagnostic reports anything never consumed.
    - Accumulation is promoted to at least fp32, so many bf16 consumer
      cotangents don't lose the small ones; the seed is cast back to the
      producer tensor's dtype.
    - ``graft`` of an empty/absent pass state returns the caller's own object
      and records nothing, so architectures that declare no shared state take
      byte-for-byte the same path they did before this class existed.
    """

    def __init__(self, *, enabled: bool = True):
        self.enabled = bool(enabled)
        # (kwarg, source key, position) -> summed cotangent (>= fp32).
        self._acc: dict[tuple, torch.Tensor] = {}
        # Per-layer: grafted leaves awaiting harvest, and the containers this
        # layer may have written into.
        self._live: list[tuple[tuple, torch.Tensor]] = []
        self._live_ids: set[int] = set()
        self._containers: list[tuple[str, dict]] = []
        # Diagnostics (reported at sweep end; never used to make a decision).
        self.n_grafted = 0
        self.n_harvested = 0
        self.n_seeded = 0
        self.n_no_grad = 0
        self.nondifferentiable: list[str] = []

    # -- consumer side ----------------------------------------------------
    def graft(self, pass_state):
        """Return `pass_state` with every borrowed tensor replaced by a
        grad-enabled leaf clone, recording the leaves for `harvest` and the
        containers for `produced_roots`. Hand the RESULT to the layer call."""
        if not self.enabled or not pass_state:
            return pass_state
        grafted: dict = {}
        tracked = False
        for kwarg, container in pass_state.items():
            if not isinstance(container, dict):
                # Not a keyed shared-state container; pass through untouched
                # rather than guess at its structure.
                grafted[kwarg] = container
                continue
            fresh = {
                key: self._graft_value(kwarg, key, value)
                for key, value in container.items()
            }
            grafted[kwarg] = fresh
            self._containers.append((kwarg, fresh))
            tracked = True
        return grafted if tracked else pass_state

    def _graft_value(self, kwarg, key, value):
        if isinstance(value, torch.Tensor):
            return self._leaf(kwarg, key, None, value)
        if isinstance(value, (tuple, list)):
            out = [
                self._leaf(kwarg, key, pos, item)
                if isinstance(item, torch.Tensor) else item
                for pos, item in enumerate(value)
            ]
            return tuple(out) if isinstance(value, tuple) else out
        return value

    def _leaf(self, kwarg, key, pos, tensor: torch.Tensor) -> torch.Tensor:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            # Integer/bool shared state carries no cotangent at all — the
            # producer's Fisher genuinely cannot be recovered through it.
            self.nondifferentiable.append(f"{kwarg}[{key!r}][{pos}]:{tensor.dtype}")
            return tensor
        try:
            slot = (kwarg, key, pos)
            hash(slot)
        except TypeError:
            self.nondifferentiable.append(f"{kwarg}[{key!r}]:unhashable-key")
            return tensor
        leaf = tensor.detach().clone().requires_grad_(True)
        self._live.append((slot, leaf))
        self._live_ids.add(id(leaf))
        self.n_grafted += 1
        return leaf

    def harvest(self) -> int:
        """Fold this layer's leaf gradients into the per-source accumulator and
        end the layer. Call AFTER the backward; returns the number folded."""
        folded = 0
        for slot, leaf in self._live:
            grad = leaf.grad
            if grad is None:
                # The layer never used the borrowed tensor on a path that
                # reaches its output (or the cotangent was structurally zero).
                self.n_no_grad += 1
                continue
            acc_dtype = torch.promote_types(grad.dtype, torch.float32)
            contrib = grad.detach().to(acc_dtype)
            prev = self._acc.get(slot)
            if prev is None:
                self._acc[slot] = contrib.clone()
            else:
                prev.add_(contrib)
            leaf.grad = None
            folded += 1
        self.n_harvested += folded
        self._live.clear()
        self._live_ids.clear()
        self._containers.clear()
        return folded

    # -- producer side ----------------------------------------------------
    def produced_roots(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """`(tensors, grads)` — extra backward roots for the layer just
        forwarded: the shared tensors IT produced, paired with the cotangent
        its consumers accumulated. Call after the forward, before the
        backward. Empty when this layer produced nothing consumers wanted."""
        if not self.enabled or not self._acc:
            return [], []
        order: list[int] = []
        by_id: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for kwarg, container in self._containers:
            for key, value in list(container.items()):
                for pos, tensor in self._iter_positions(value):
                    slot = (kwarg, key, pos)
                    grad = self._acc.get(slot)
                    if grad is None:
                        continue
                    if id(tensor) in self._live_ids or tensor.grad_fn is None:
                        # A tensor this layer BORROWED, not produced: it is a
                        # leaf clone with no graph. Seeding it would inject
                        # other consumers' cotangent into this layer's harvest.
                        continue
                    self._acc.pop(slot)
                    seed = grad.to(device=tensor.device, dtype=tensor.dtype)
                    if id(tensor) in by_id:
                        # Same object written at two positions (Gemma4's
                        # `value_states = key_states` k_eq_v layers): one root,
                        # summed grad — duplicate roots would be legal but this
                        # is unambiguous.
                        prev_t, prev_g = by_id[id(tensor)]
                        by_id[id(tensor)] = (prev_t, prev_g + seed)
                    else:
                        by_id[id(tensor)] = (tensor, seed)
                        order.append(id(tensor))
                    self.n_seeded += 1
        roots = [by_id[i][0] for i in order]
        grads = [by_id[i][1] for i in order]
        return roots, grads

    @staticmethod
    def _iter_positions(value):
        if isinstance(value, torch.Tensor):
            yield None, value
        elif isinstance(value, (tuple, list)):
            for pos, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    yield pos, item

    # -- serialization ----------------------------------------------------
    def snapshot_copy_plan(self) -> tuple[bool, int]:
        """Whether this owner needs an owner-budgeted CPU copy, and how many bytes.

        No allocation: inspects live accumulators only. Returns
        ``(needs_copy, copy_bytes)`` where ``needs_copy`` is True when any
        accumulator is not already CPU contiguous strided, and ``copy_bytes``
        sums ``numel * element_size`` over this owner's accumulators when a
        copy is needed (0 otherwise). Meta or non-strided tensors refuse
        (TypeError) like the checkpoint estimator -- they are unaccountable,
        never silently copied. Quiescence refuses exactly like ``state_dict``.
        """
        if self._live or self._containers or self._live_ids:
            raise RuntimeError("shared cotangent serialization requires a quiescent owner")
        needs = False
        total = 0
        for tensor in self._acc.values():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    "exact boundary checkpoint cannot account opaque shared state "
                    f"{type(tensor).__name__}")
            if tensor.is_meta or tensor.layout != torch.strided:
                raise TypeError(
                    "exact boundary checkpoint cannot account this state tensor")
            if not (tensor.device.type == "cpu" and tensor.is_contiguous()):
                needs = True
            total += tensor.numel() * tensor.element_size()
        return (needs, int(total) if needs else 0)

    def borrowed_state_dict(self) -> dict:
        """Borrowed quiescent view, zero-copy only when proven safe.

        Same portable mapping as :meth:`state_dict` when every accumulator
        is already CPU contiguous strided: borrowed by detached reference
        (shared backing, no new allocation). Any accumulator needing CPU
        pinning or contiguity -- CUDA/non-CPU device, non-contiguous layout,
        or any other non-pinned storage -- refuses BEFORE copying: the
        caller must take the owner-budgeted fallback (precomputed
        ``snapshot_copy_plan`` bytes held across the synchronous snapshot +
        write lifetime, then :meth:`state_dict` copies inside that hold).
        The caller must drop the view before the next graft/harvest and must
        not mutate owners while the view serializes. Quiescence refuses
        exactly like ``state_dict``.
        """
        if self._live or self._containers or self._live_ids:
            raise RuntimeError("shared cotangent serialization requires a quiescent owner")
        rows = []
        for slot, tensor in self._acc.items():
            if not (isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
                    and not tensor.is_meta and tensor.device.type == "cpu"
                    and tensor.is_contiguous()):
                raise RuntimeError(
                    "shared cotangent borrowed view needs an owner-budgeted "
                    "CPU snapshot: refused before copying")
            rows.append({
                "slot": [slot[0], slot[1], slot[2]],
                "tensor": tensor.detach(),
            })
        return {
            "enabled": bool(self.enabled),
            "accumulators": rows,
            "counters": {name: int(getattr(self, name))
                         for name in ("n_grafted", "n_harvested", "n_seeded", "n_no_grad")},
            "nondifferentiable": list(self.nondifferentiable),
        }

    def is_empty(self) -> bool:
        """No accumulated cotangent and no live graft.

        With an empty isolated pass state, ``graft`` returns it unchanged,
        ``produced_roots`` is ``([], [])`` and ``harvest`` folds nothing: the
        owner changes no backward. The batched and fused render-free chain
        (RobTand/prismaquant#997) runs only while every owner it would skip
        is empty.
        """
        return not (self._acc or self._live or self._containers or self._live_ids)

    def state_dict(self) -> dict:
        """The completed shared-pass adjoint state, CPU-pinned and picklable.

        The distributed cost campaign's strided adjoint checkpoints serialize
        this beside the activation cotangent (contract
        ``distributed_campaign_2026-09-19.md`` §2.2 stage A): only the
        completed accumulator crosses a serialization boundary -- a live
        graft belongs to the current autograd graph, so serializing mid-layer
        refuses exactly like ``resident_tensors`` does.
        """
        if self._live or self._containers or self._live_ids:
            raise RuntimeError("shared cotangent serialization requires a quiescent owner")
        return {
            "enabled": bool(self.enabled),
            "accumulators": [
                {
                    "slot": [slot[0], slot[1], slot[2]],
                    "tensor": tensor.detach().to(
                        device="cpu", copy=True, memory_format=torch.contiguous_format),
                }
                for slot, tensor in self._acc.items()
            ],
            "counters": {name: int(getattr(self, name))
                         for name in ("n_grafted", "n_harvested", "n_seeded", "n_no_grad")},
            "nondifferentiable": list(self.nondifferentiable),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore ``state_dict`` output into this (fresh) owner."""
        if self._live or self._containers or self._live_ids or self._acc:
            raise RuntimeError("shared cotangent restore requires an empty owner")
        if not isinstance(state, dict) or not {"enabled", "accumulators", "counters"} <= set(state):
            raise ValueError("shared cotangent state dict is not one")
        self.enabled = bool(state["enabled"])
        for row in state["accumulators"]:
            slot = (row["slot"][0], row["slot"][1], row["slot"][2])
            if slot in self._acc:
                raise ValueError("shared cotangent state dict repeats a slot")
            tensor = row["tensor"]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("shared cotangent state dict slot is not a tensor")
            self._acc[slot] = tensor.detach().clone(memory_format=torch.contiguous_format)
        for name, value in state["counters"].items():
            setattr(self, name, int(value))
        self.nondifferentiable = [str(item) for item in state.get("nondifferentiable", [])]

    # -- diagnostics ------------------------------------------------------
    def fork_for_replay(self, *, max_resident_bytes: int) -> "SharedStateCotangents":
        """Fork completed adjoints for one disposable target replay.

        The budget covers the fork's new compact tensor storages only; the
        caller still owns and charges this original accumulator. ``graft``,
        ``produced_roots`` and ``harvest`` may mutate the fork freely. Only a
        final committed replay should use the original owner. No live graph,
        mutable accumulator storage or diagnostic list crosses the fork.
        """
        if self._live or self._containers or self._live_ids:
            raise RuntimeError("shared cotangent replay requires a quiescent owner")
        if type(max_resident_bytes) is not int or max_resident_bytes < 0:
            raise ValueError("shared cotangent replay requires a nonnegative tensor byte cap")
        needed = sum(t.numel() * t.element_size() for t in self._acc.values())
        if needed > max_resident_bytes:
            raise RuntimeError("shared cotangent replay exceeds its tensor residency budget")
        fork = type(self)(enabled=self.enabled)
        try:
            for slot, tensor in self._acc.items():
                fork._acc[slot] = tensor.detach().clone(memory_format=torch.contiguous_format)
            for name in ("n_grafted", "n_harvested", "n_seeded", "n_no_grad"):
                setattr(fork, name, getattr(self, name))
            fork.nondifferentiable = list(self.nondifferentiable)
            return fork
        except BaseException:
            fork.release_resident_state()
            raise

    def resident_tensors(self) -> tuple[torch.Tensor, ...]:
        """Expose retained shared adjoints for the streamed owner's byte cap.

        Call at layer boundaries after harvest; a live graft belongs to the
        current autograd graph, so treating it as completed state must refuse.
        """
        if self._live or self._containers:
            raise RuntimeError("shared cotangent residency queried before harvest")
        return tuple(self._acc.values())

    def release_resident_state(self) -> None:
        """Release adjoints/graph references when their entire sweep closes."""
        self._acc.clear()
        self._live.clear()
        self._live_ids.clear()
        self._containers.clear()

    def pending_keys(self) -> list[tuple]:
        """Accumulated cotangents no producer ever claimed. Non-empty means the
        sweep never forwarded the producing layer (or it stopped writing the
        key), so that producer's Fisher is still under-counted."""
        return sorted(self._acc, key=repr)

    def summary(self) -> str:
        pending = self.pending_keys()
        parts = [
            f"enabled={self.enabled}",
            f"grafted={self.n_grafted}",
            f"harvested={self.n_harvested}",
            f"seeded={self.n_seeded}",
        ]
        if self.n_no_grad:
            parts.append(f"no_grad={self.n_no_grad}")
        if self.nondifferentiable:
            parts.append(f"nondifferentiable={len(self.nondifferentiable)}")
        if pending:
            parts.append(f"UNCLAIMED={[repr(k) for k in pending]}")
        return "kv_cotangent[" + " ".join(parts) + "]"


# ---------------------------------------------------------------------------
# Router tracker: per-(router, expert) activation probability
# ---------------------------------------------------------------------------
class RouterTracker:
    def __init__(self, model: nn.Module, routers: list[str], top_k: int):
        self.top_k = top_k
        self.counts_t: dict[str, torch.Tensor] = {}
        self.active_counts_t: dict[str, torch.Tensor] = {}
        self.total_tokens: dict[str, int] = defaultdict(int)
        self._handles = []
        for rq in routers:
            try:
                mod = model.get_submodule(rq)
            except AttributeError:
                continue
            n_experts = None
            if isinstance(mod, nn.Linear):
                n_experts = mod.out_features
            else:
                weight = getattr(mod, "weight", None)
                if isinstance(weight, torch.Tensor) and weight.ndim >= 1:
                    n_experts = int(weight.shape[0])
            if not isinstance(n_experts, int) or n_experts <= 0:
                continue
            self.counts_t[rq] = torch.zeros(n_experts, dtype=torch.float64)
            self.active_counts_t[rq] = torch.zeros(n_experts, dtype=torch.int64)
            self._handles.append(mod.register_forward_hook(self._make_hook(rq)))

    def _make_hook(self, router_qname: str):
        def hook(module, inp, out):
            scores = out if isinstance(out, torch.Tensor) else out[0]
            flat = scores.detach().reshape(-1, scores.size(-1))
            k = min(self.top_k, flat.size(-1))
            topk_v, topk_i = flat.topk(k, dim=-1)
            probs = F.softmax(topk_v, dim=-1)
            weighted = torch.bincount(
                topk_i.reshape(-1),
                weights=probs.reshape(-1).to(torch.float64),
                minlength=int(scores.size(-1)),
            )
            active = torch.bincount(
                topk_i.reshape(-1),
                minlength=int(scores.size(-1)),
            )
            self.total_tokens[router_qname] += flat.size(0)
            self.counts_t[router_qname].add_(weighted.cpu())
            self.active_counts_t[router_qname].add_(active.cpu().to(torch.int64))
        return hook

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def prob(self, router_qname: str, eid: str) -> float:
        total = self.total_tokens.get(router_qname, 0)
        if total == 0:
            return 0.0
        counts = self.counts_t.get(router_qname)
        if counts is None:
            return 0.0
        idx = int(eid)
        if idx < 0 or idx >= counts.numel():
            return 0.0
        return float(counts[idx].item()) / total

    @property
    def counts(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for router, counts in self.counts_t.items():
            nz = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
            out[router] = {
                str(int(i)): float(counts[int(i)].item())
                for i in nz.tolist()
            }
        return out

    @property
    def active_counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for router, counts in self.active_counts_t.items():
            nz = torch.nonzero(counts > 0, as_tuple=False).reshape(-1)
            out[router] = {
                str(int(i)): int(counts[int(i)].item())
                for i in nz.tolist()
            }
        return out

    @property
    def route_stats(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        active = self.active_counts
        for router, counts in self.counts.items():
            total = int(self.total_tokens.get(router, 0))
            denom = max(total, 1)
            out[router] = {
                "total_tokens": total,
                "mass": counts,
                "active_count": active.get(router, {}),
                "prob": {
                    eid: float(mass) / denom
                    for eid, mass in counts.items()
                },
            }
        return out
