#!/usr/bin/env python3
"""sensitivity_probe.py — per-Linear empirical Fisher diagonal trace.

What this measures
------------------
For each tracked Linear with weight W, this script estimates the
per-token empirical Fisher diagonal trace

    H_trace = Σ_w E_token[(∂L/∂W_w)²]

where L is the per-token negative log-likelihood (same loss a language
model is trained against) and the expectation is over a calibration
corpus. This quantity is used by allocator.py as the sensitivity score
in the closed-form predicted Δloss

    Δloss ≈ 0.5 · H_trace · MSE_W                           (eq. 3 in
                                                             allocator.py)

Naming. The literature uses several names for E[(∂L/∂W)²]: "empirical
Fisher", "Gauss-Newton diagonal", "gradient-squared". This is NOT the
true Hessian diagonal — for that you would need a vHv-style probe
(Hutchinson). The empirical Fisher equals the Hessian only at a true
loss minimum, which a calibration corpus does not in general satisfy.
For ranking layers and predicting first-order quantization sensitivity,
empirical Fisher is the standard HAWQ-V1 choice and works well.

How the per-token estimator stays unbiased
------------------------------------------
HuggingFace's `out.loss` is `mean(CE)` over the T tokens in a batch, so
its gradient is `(1/T) · Σ_t ∂CE_t/∂W`. Squaring that under-estimates
per-token Fisher by a factor of T (under the standard assumption of
independent per-token gradients). To avoid that, we reconstruct CE with
`reduction="sum"` for the backward pass; the gradient then aggregates
per-token gradients without the 1/T factor, and dividing the
accumulated `||grad_W||²_F` by total tokens recovers the per-token
Fisher trace estimator directly.

Other features:
  - route-aware MoE scaling (discover routers by walking module tree;
    divide each expert's H_trace by observed routing probability so
    sparse experts' Fisher is comparable to dense layers')
  - per-token importance weighting (harder tokens count more); this
    reweights the loss but preserves the per-token-Fisher units when
    used with sum reduction
  - activation snapshot cache for measure_quant_cost.py

Memory:
  - params requires_grad_(False)   → no gradient tensor storage
  - gradient checkpointing on      → activations are recomputed during backward
  - backward hooks reduce grad_w to a scalar inline and drop it
Result: peak ≈ model weights + one-block activation, fits in 128 GB for 35 B.

Model-agnostic:
  - Router discovered via module walk (any Linear whose out_features equals a
    sibling ModuleList named experts, gates, etc.)
  - Top-k read from model.config (num_experts_per_tok)
  - Dense models just skip RouterTracker
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Text-only staging
# ---------------------------------------------------------------------------
def stage_text_only(model_path: str) -> str:
    src = Path(model_path)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return str(src)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if not any(k in cfg for k in
               ("vision_config", "text_config", "audio_config", "speech_config")):
        return str(src)

    # Profile-driven: ask the registered ModelProfile which config keys
    # to strip and whether to promote `text_config.model_type`.
    try:
        from .model_profiles import detect_profile
        profile = detect_profile(str(src))
    except Exception:
        profile = None
    strip_keys = (list(profile.stage_text_only_strip_keys())
                  if profile is not None
                  else ["vision_config", "audio_config", "speech_config",
                        "image_token_id", "video_token_id",
                        "vision_start_token_id", "vision_end_token_id"])
    promote_inner_mt = (profile.stage_text_only_promote_inner_model_type()
                        if profile is not None else False)

    import tempfile
    for k in strip_keys:
        cfg.pop(k, None)
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

    staged = Path(tempfile.mkdtemp(prefix="prismquant_stage_"))
    skip = {"config.json", "preprocessor_config.json",
            "video_preprocessor_config.json", "processor_config.json"}
    for p in src.iterdir():
        if p.name in skip:
            continue
        (staged / p.name).symlink_to(p.resolve())
    with open(staged / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return str(staged)


class _GradNormCapture(torch.autograd.Function):
    """Identity in forward; in backward, accumulates per-expert
    per-output-channel squared-gradient into `channel_accumulator` and
    the scalar Frobenius-norm sum into `scalar_accumulator`, then
    returns None for the weight gradient — which tells autograd to NOT
    accumulate to the leaf parameter's .grad.

    Used to capture the per-token empirical Fisher diagonal trace and
    per-channel diagonal of a packed expert tensor (e.g. Qwen3.6's
    `gate_up_proj` of shape `[E, 2*I, H]`) without ever storing a
    full-size .grad on the leaf.

    Storage: scalar is a float. Channel-diag is a [E, M] tensor — 256
    experts × 1024 out rows × 4 B ≈ 1 MB per packed param, well below
    the 2 GB full-resolution per-weight alternative that would be
    infeasible at 35B scale.

    Why return None? With 40 MoE layers × 2 packed params × ~5 GB of
    bf16 grads = 400 GB if .grad were retained per leaf. By returning
    None we tell autograd "this input doesn't need a stored gradient";
    .grad stays None on the leaf and only the transient grad_output
    (one per backward node, freed in topological order) is alive at
    any one time.
    """

    @staticmethod
    def forward(ctx, weight, name, scalar_accumulator, channel_accumulator):
        ctx.name = name
        ctx.scalar_acc = scalar_accumulator
        ctx.channel_acc = channel_accumulator
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        g = grad_output.detach()
        # Scalar Frobenius-norm squared — streamed to avoid materializing
        # the full squared tensor for very large packed params.
        flat = g.reshape(-1)
        chunk = 1_000_000
        total = 0.0
        for i in range(0, flat.numel(), chunk):
            total += float(flat[i:i + chunk].float().pow(2).sum().item())
        ctx.scalar_acc[ctx.name] = ctx.scalar_acc.get(ctx.name, 0.0) + total
        # Per-expert per-output-channel diagonal: reduce along the
        # in-feature axis (the last dim). For a [E, M, N] packed param,
        # result is [E, M]. Accumulated across backward samples.
        if g.dim() == 3:
            per_ch = g.float().pow(2).sum(dim=-1)      # [E, M]
        elif g.dim() == 2:
            per_ch = g.float().pow(2).sum(dim=-1, keepdim=False)  # [out]
        else:
            per_ch = None
        if per_ch is not None:
            cur = ctx.channel_acc.get(ctx.name)
            if cur is None:
                ctx.channel_acc[ctx.name] = per_ch.cpu()
            else:
                cur.add_(per_ch.cpu())
        return None, None, None, None


_PACKED_EXPERT_PARAM_NAMES = {
    "gate_up_proj", "down_proj",            # Qwen3.5 / 3.6 packed MoE
    "w1", "w2", "w3",                       # Mixtral-style legacy
    "gate_proj", "up_proj",                 # Some HF layouts
}


def _is_packed_experts_module(module: nn.Module) -> bool:
    """A module qualifies as a packed-experts container iff (a) its
    class name contains "Experts" (case-insensitive), and (b) it owns
    at least one 3D nn.Parameter whose attribute name is in
    `_PACKED_EXPERT_PARAM_NAMES`.

    The class-name check excludes other modules that happen to own 3D
    parameters — most importantly Conv1d in linear-attention paths,
    whose `weight` is shape `[out, in, kernel]`. The param-name check
    is a second safety net against unusual modules with unrelated 3D
    state.
    """
    cls_name = type(module).__name__.lower()
    if "expert" not in cls_name:
        return False
    for n, p in module.named_parameters(recurse=False):
        if (isinstance(p, nn.Parameter)
                and p.dim() == 3
                and n in _PACKED_EXPERT_PARAM_NAMES):
            return True
    return False


def _packed_experts_param_names(module: nn.Module) -> list[str]:
    """Return the attribute names of all 3D packed parameters on
    `module`, restricted to the known MoE expert names. Order is stable
    across Python runs."""
    names = []
    for n, p in module.named_parameters(recurse=False):
        if (isinstance(p, nn.Parameter)
                and p.dim() == 3
                and n in _PACKED_EXPERT_PARAM_NAMES):
            names.append(n)
    return sorted(names)


_PRISMQUANT_PATCH_SENTINEL = "_prismquant_packed_expert_patch"
_PRISMQUANT_CHANNEL_SENTINEL = "_prismquant_packed_expert_channel_patch"


def install_packed_expert_hooks(
    model: nn.Module,
    accumulator: dict,
    channel_accumulator: dict | None = None,
) -> dict[str, dict]:
    """Patch every packed-experts module's forward so its 3D parameters
    route through `_GradNormCapture` before each use.

    `accumulator` collects the scalar Frobenius-norm squared (matches
    the nn.Linear `h_trace_raw`). `channel_accumulator` collects the
    per-expert per-output-channel diagonal as [E, M] CPU tensors (for
    the full per-weight Fisher cost model). The channel accumulator is
    optional for backward compatibility; when None, packed experts
    contribute only their scalar trace and the allocator falls back to
    the scalar proxy for those entries.

    Returns a metadata dict keyed by `<module_qname>.<param_name>` with
    the same shape/role information stored for nn.Linear modules in
    `FisherAccumulator`. The probe inserts these into its main `stats`
    dict so the allocator can treat them uniformly.

    Idempotent across calls. If a module has already been patched (by
    a prior call within the same Python process), we re-bind both the
    scalar and channel accumulator references to the new dicts rather
    than wrapping the patch again. This is essential for the incremental
    probe path, which constructs a fresh FisherAccumulator per shard
    against a single loaded model.

    Activation snapshotting for measure_quant_cost is handled by
    `FisherAccumulator` directly (forward hook on the experts module).
    """
    meta: dict[str, dict] = {}
    for qname, module in model.named_modules():
        if not _is_packed_experts_module(module):
            continue
        param_names = _packed_experts_param_names(module)
        if not param_names:
            continue

        # Idempotent re-bind path. The sentinel holds a reference to the
        # mutable accumulator dict that patched_forward writes to. We
        # rebind it (clear contents and adopt the new dict's identity by
        # swapping references) — but the simpler primitive is to update
        # the closure's *target dict identity* via attribute, since
        # patched_forward's closure already binds the original dict by
        # reference. Easiest: store the live accumulator on the module
        # and have patched_forward read it indirectly each call.
        if hasattr(module, _PRISMQUANT_PATCH_SENTINEL):
            # Update the live accumulator binding for this module's patch.
            setattr(module, _PRISMQUANT_PATCH_SENTINEL, accumulator)
            setattr(module, _PRISMQUANT_CHANNEL_SENTINEL, channel_accumulator)
            # Still report metadata so callers can refresh their stats dict.
            for pn in param_names:
                p_existing = module._parameters.get(pn)
                if p_existing is None:
                    continue
                shape = tuple(p_existing.shape)
                full_name = f"{qname}.{pn}" if qname else pn
                meta[full_name] = {
                    "h_trace_raw": 0.0,
                    "h_w2_sum_raw": 0.0,
                    "w_max_abs": float(p_existing.detach().abs().max().item()),
                    "w_norm_sq": float(p_existing.detach().pow(2).sum().item()),
                    "n_params": int(p_existing.numel()),
                    "in_features": int(shape[2]),
                    "out_features": int(shape[1]),
                    "num_experts": int(shape[0]),
                    "n_tokens_seen": 0,
                    "route_prob": None,
                    "router_path": None,
                    "expert_id": None,
                    "_packed_experts_module": qname,
                    "_packed_param": pn,
                }
            continue

        # Enable grad on packed params so autograd computes their gradient
        # through our identity wrapper.
        for pn in param_names:
            getattr(module, pn).requires_grad_(True)

        for pn in param_names:
            p: nn.Parameter = getattr(module, pn)
            full_name = f"{qname}.{pn}" if qname else pn
            shape = tuple(p.shape)
            # Convention: shape[0] = num_experts; the per-expert matrix is
            # the trailing two dims. Use (out_features, in_features) =
            # (shape[1], shape[2]) to match nn.Linear's convention; the
            # allocator's predicted_dloss only needs n_params correct.
            num_experts = int(shape[0])
            out_features = int(shape[1])
            in_features = int(shape[2])
            n_params = int(p.numel())
            meta[full_name] = {
                "h_trace_raw": 0.0,
                "h_w2_sum_raw": 0.0,  # not measured for packed; kept for schema
                "w_max_abs": float(p.detach().abs().max().item()),
                "w_norm_sq": float(p.detach().pow(2).sum().item()),
                "n_params": n_params,
                "in_features": in_features,
                "out_features": out_features,
                "num_experts": num_experts,
                "n_tokens_seen": 0,
                "route_prob": None,  # rolled into per-expert sensitivity by sum
                "router_path": None,
                "expert_id": None,
                "_packed_experts_module": qname,
                "_packed_param": pn,
            }

        # Patch forward to wrap each packed param with _GradNormCapture.
        # The original forward uses self.<pn>; we shadow those attributes
        # with the wrapped tensors for the duration of the call. nn.Module
        # __getattribute__ checks _parameters before __dict__, so we have
        # to temporarily move the param out of _parameters and shadow via
        # __dict__ to make the wrapped tensor visible to the original
        # forward.
        original_forward = module.forward
        ns = list(param_names)
        full_names = [f"{qname}.{pn}" if qname else pn for pn in ns]
        mod_ref = module

        # Store the live accumulators as attributes so subsequent calls
        # to install_packed_expert_hooks can re-bind them (per-shard) by
        # just updating these attributes. patched_forward reads them
        # indirectly each invocation via getattr.
        setattr(mod_ref, _PRISMQUANT_PATCH_SENTINEL, accumulator)
        setattr(mod_ref, _PRISMQUANT_CHANNEL_SENTINEL, channel_accumulator)

        def patched_forward(*args, _ns=ns, _full=full_names, _orig=original_forward,
                            _mod=mod_ref, **kwargs):
            acc = getattr(_mod, _PRISMQUANT_PATCH_SENTINEL, None)
            ch_acc = getattr(_mod, _PRISMQUANT_CHANNEL_SENTINEL, None)
            if acc is None:
                # Should not happen, but degrade gracefully.
                return _orig(*args, **kwargs)
            if ch_acc is None:
                # Provide a scratch dict we can throw away if the caller
                # didn't opt in to per-channel accumulation, so the
                # autograd Function signature stays uniform.
                ch_acc = {}
            saved_params = {}
            wrapped = {}
            for pn, fn in zip(_ns, _full):
                saved_params[pn] = _mod._parameters.pop(pn)
                wrapped[pn] = _GradNormCapture.apply(
                    saved_params[pn], fn, acc, ch_acc)
                _mod.__dict__[pn] = wrapped[pn]
            try:
                return _orig(*args, **kwargs)
            finally:
                for pn in _ns:
                    _mod.__dict__.pop(pn, None)
                    _mod._parameters[pn] = saved_params[pn]

        module.forward = patched_forward

    return meta


def resolve_execution_device(model: nn.Module, requested_device: str) -> torch.device:
    """Choose the device used for input ids / embeddings during probing.

    When `device_map="auto"` is used for model load, the model can be sharded
    across CPU and GPU. In that case we want to feed tokens to the device that
    owns the input embedding weights rather than assuming a single global
    `cuda`/`cpu` target.
    """
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device(requested_device)


# ---------------------------------------------------------------------------
# Model-agnostic MoE discovery
# ---------------------------------------------------------------------------
def discover_moe_structure(model: nn.Module) -> dict[str, tuple[str, str]]:
    """Return {expert_linear_qname: (router_qname, expert_id_str)}.

    Walk the module tree.  For any module that has a child attribute named
    `experts` or `block_sparse_moe_experts` that is a ModuleList, find a
    sibling Linear in the same parent whose out_features equals len(experts).
    That Linear is the router.
    """
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
            for proj_name in ("gate_up_proj", "down_proj", "w1", "w2", "w3"):
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
            for proj_name in ("gate_up_proj", "down_proj", "w1", "w2", "w3"):
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
# Router tracker: per-(router, expert) activation probability
# ---------------------------------------------------------------------------
class RouterTracker:
    def __init__(self, model: nn.Module, routers: list[str], top_k: int):
        self.top_k = top_k
        self.counts_t: dict[str, torch.Tensor] = {}
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
            self.total_tokens[router_qname] += flat.size(0)
            self.counts_t[router_qname].add_(weighted.cpu())
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


# ---------------------------------------------------------------------------
# Fisher accumulator with activation snapshot cache
# ---------------------------------------------------------------------------
class FisherAccumulator:
    def __init__(self, model: nn.Module, tracked: list[str],
                 expert_info: dict[str, tuple[str, str]],
                 act_cache_dir: Path | None = None,
                 input_rows: int = 256,
                 hook_packed_experts: bool = True,
                 h_detail_dir: Path | None = None):
        self.stats: dict[str, dict] = {}
        self._saved_inputs: dict[str, torch.Tensor] = {}
        self._fwd_handles, self._bwd_handles = [], []
        self.tracked = set(tracked)
        self.expert_info = expert_info
        self.cache_dir = act_cache_dir
        self.h_detail_dir = Path(h_detail_dir) if h_detail_dir else None
        self.input_rows = input_rows
        self._input_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._rows_got: dict[str, int] = defaultdict(int)
        # Packed expert grad-norm accumulator: written by _GradNormCapture
        # during backward, read in finalize().
        self._packed_grad_acc: dict[str, float] = {}
        # Per-(experts module qname) sample count (one per backward),
        # populated by the experts forward hook below.
        self._packed_sample_count: dict[str, int] = defaultdict(int)
        # Per-experts-module activation snapshots, captured live so
        # measure_quant_cost can read packed expert inputs.
        self._packed_act_snaps: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._packed_act_rows: dict[str, int] = defaultdict(int)

        # Per-layer accumulator for full per-weight Fisher diagonal.
        # Keyed by Linear qname -> CPU fp64 tensor of shape [out, in]
        # matching the weight shape. Accumulated by the backward hook;
        # written to `probe_detail/<sanitized_name>.pt` in finalize()
        # so the probe pickle stays small while downstream consumers
        # can lazy-load the full H per layer.
        self._h_full: dict[str, torch.Tensor] = {}
        # Same idea for packed experts, but reduced to per-expert
        # per-output-channel: [E, M] instead of [E, M, N]. Full
        # per-weight for 80 packed tensors at 35B scale is 160+ GB;
        # per-channel is 160 MB total — still a vector form.
        self._h_packed_channel: dict[str, torch.Tensor] = {}
        for name, mod in model.named_modules():
            if name not in self.tracked or not isinstance(mod, nn.Linear):
                continue
            w = mod.weight
            router_qname, eid = expert_info.get(name, (None, None))
            self.stats[name] = {
                "h_trace_raw": 0.0,
                "h_w2_sum_raw": 0.0,
                "w_max_abs": float(w.detach().abs().max().item()),
                "w_norm_sq": float(w.detach().pow(2).sum().item()),
                "n_params": int(w.numel()),
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "n_tokens_seen": 0,
                "route_prob": None,
                "router_path": router_qname,
                "expert_id": eid,
            }
            # GPU fp32 accumulator: transferred to CPU fp64 at finalize.
            self._h_full[name] = torch.zeros(
                mod.out_features, mod.in_features,
                dtype=torch.float32, device=w.device,
            )
            self._fwd_handles.append(
                mod.register_forward_hook(self._make_fwd(name)))
            self._bwd_handles.append(
                mod.register_full_backward_hook(self._make_bwd(name, mod)))

        if hook_packed_experts:
            packed_meta = install_packed_expert_hooks(
                model,
                accumulator=self._packed_grad_acc,
                channel_accumulator=self._h_packed_channel,
            )
            for full_name, meta in packed_meta.items():
                # Filter against the tracked set when tracked is a regex
                # match; here we accept any packed param under a tracked
                # parent (the regex from run_probe_pass already filters
                # by layer).
                experts_qname = meta.pop("_packed_experts_module")
                meta.pop("_packed_param", None)
                # Heuristic: include packed entry if any of its conjugate
                # "in this same parent layer" Linears are tracked. This
                # makes shard regexes (`model.layers.X.`) work cleanly.
                parent_layer = ".".join(experts_qname.split(".")[:3])  # e.g. model.layers.7
                if any(t.startswith(parent_layer + ".") for t in self.tracked):
                    self.stats[full_name] = meta
                    # Register a forward hook on the experts module to
                    # bump the per-backward sample count (used to keep
                    # n_tokens_seen aligned with the Linear path's
                    # accounting).
                    try:
                        experts_mod = model.get_submodule(experts_qname)
                    except AttributeError:
                        continue

                    def _exp_fwd(_mod, inp, _out, _qn=experts_qname,
                                 _full=full_name, _x_acc=self._packed_act_snaps,
                                 _r=self._packed_act_rows):
                        x = inp[0] if isinstance(inp, tuple) else inp
                        if isinstance(x, torch.Tensor):
                            self._packed_sample_count[_full] += int(
                                x.detach().reshape(-1, x.size(-1)).size(0))
                            if act_cache_dir is not None:
                                need = self.input_rows - _r[_qn]
                                if need > 0:
                                    flat = x.detach().reshape(-1, x.size(-1))
                                    if flat.size(0) > need:
                                        idx = torch.randperm(flat.size(0),
                                                             device=flat.device)[:need]
                                        flat = flat.index_select(0, idx)
                                    _x_acc[_qn].append(flat.to("cpu"))
                                    _r[_qn] += flat.size(0)

                    self._fwd_handles.append(
                        experts_mod.register_forward_hook(_exp_fwd))

    def _make_fwd(self, name: str):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            self._saved_inputs[name] = x.detach()
            if self.cache_dir is not None:
                need = self.input_rows - self._rows_got[name]
                if need > 0:
                    flat = x.detach().reshape(-1, x.size(-1))
                    if flat.size(0) > need:
                        idx = torch.randperm(flat.size(0), device=flat.device)[:need]
                        flat = flat.index_select(0, idx)
                    self._input_snaps[name].append(flat.to("cpu"))
                    self._rows_got[name] += flat.size(0)
        return hook

    def _make_bwd(self, name: str, mod_ref: nn.Linear):
        def hook(module, grad_input, grad_output):
            gy = grad_output[0]
            x = self._saved_inputs.pop(name, None)
            if x is None or gy is None:
                return
            gy2 = gy.reshape(-1, gy.size(-1))
            x2 = x.reshape(-1, x.size(-1))
            grad_w = gy2.t() @ x2
            grad_w_sq = grad_w.pow(2)
            # Full per-weight Fisher accumulation: required for the
            # `predicted_dloss = 0.5 · <H_full, MSE_W_full>` cost model
            # that replaces the scalar `h_trace · mse_scalar` proxy.
            acc = self._h_full.get(name)
            if acc is not None:
                acc.add_(grad_w_sq.float())
            self.stats[name]["h_trace_raw"] += float(grad_w_sq.sum().item())
            w = mod_ref.weight.detach()
            self.stats[name]["h_w2_sum_raw"] += float(
                (grad_w_sq * w.pow(2)).sum().item())
            self.stats[name]["n_tokens_seen"] += x2.size(0)
        return hook

    def finalize(self, tracker: RouterTracker | None):
        # Flush packed-expert grad-norm accumulator into stats h_trace_raw.
        # The packed accumulator key matches the stats key by construction
        # (full param name `<experts_qname>.<param_name>`).
        for full_name, raw in self._packed_grad_acc.items():
            if full_name in self.stats:
                self.stats[full_name]["h_trace_raw"] += float(raw)
                self.stats[full_name]["n_tokens_seen"] = int(
                    self._packed_sample_count.get(full_name, 0))

        if tracker is not None:
            for name, s in self.stats.items():
                if s["router_path"]:
                    s["route_prob"] = tracker.prob(
                        s["router_path"], s["expert_id"])

        for s in self.stats.values():
            tokens = max(s["n_tokens_seen"], 1)
            if s["route_prob"] is not None and s["route_prob"] > 0:
                s["h_trace"] = (s["h_trace_raw"] / tokens) / s["route_prob"]
                s["h_w2_sum"] = (s["h_w2_sum_raw"] / tokens) / s["route_prob"]
            else:
                s["h_trace"] = s["h_trace_raw"] / tokens
                s["h_w2_sum"] = s["h_w2_sum_raw"] / tokens

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for name, snaps in self._input_snaps.items():
                if not snaps:
                    continue
                X = torch.cat(snaps, dim=0).to(torch.bfloat16).contiguous()
                fname = re.sub(r"[^A-Za-z0-9_-]", "__", name) + ".pt"
                torch.save({"inputs": X, "name": name},
                           self.cache_dir / fname)
            # Also write packed-experts module input snapshots. We key
            # these by the experts module qname (not the parameter name);
            # measure_quant_cost looks for the same input regardless of
            # which packed parameter is being measured.
            for experts_qname, snaps in self._packed_act_snaps.items():
                if not snaps:
                    continue
                X = torch.cat(snaps, dim=0).to(torch.bfloat16).contiguous()
                fname = re.sub(r"[^A-Za-z0-9_-]", "__", experts_qname) + ".pt"
                torch.save({"inputs": X, "name": experts_qname},
                           self.cache_dir / fname)

        # Write per-layer Fisher H detail files (the full per-weight
        # diagonal for Linears, per-expert per-output-channel for packed
        # experts). The probe pickle stores scalar summaries + the
        # detail-file path; measure_quant_cost loads them lazily so the
        # full allocator cost model can run without bloating probe.pkl.
        if self.h_detail_dir is not None:
            self.h_detail_dir.mkdir(parents=True, exist_ok=True)
            sub = re.compile(r"[^A-Za-z0-9_-]")
            for name, acc in self._h_full.items():
                if name not in self.stats:
                    continue
                tokens = max(self.stats[name]["n_tokens_seen"], 1)
                rp = self.stats[name].get("route_prob")
                # Apply the same normalization as the scalar trace.
                if rp is not None and rp > 0:
                    h = acc.to(torch.float32).cpu() / (tokens * rp)
                else:
                    h = acc.to(torch.float32).cpu() / tokens
                fname = sub.sub("__", name) + ".pt"
                torch.save({"h_diag": h, "name": name, "kind": "linear",
                            "shape": list(h.shape)},
                           self.h_detail_dir / fname)
                self.stats[name]["h_detail_path"] = fname
            for full_name, ch in self._h_packed_channel.items():
                if full_name not in self.stats:
                    continue
                tokens = max(self.stats[full_name]["n_tokens_seen"], 1)
                # Packed experts don't carry a router_path — routing is
                # baked into the Fisher signal via how often each expert
                # was selected. Normalize by token count only.
                h = ch.to(torch.float32) / tokens
                fname = sub.sub("__", full_name) + ".pt"
                torch.save({"h_diag": h, "name": full_name, "kind": "packed",
                            "shape": list(h.shape)},
                           self.h_detail_dir / fname)
                self.stats[full_name]["h_detail_path"] = fname

    def remove_hooks(self):
        for h in self._fwd_handles + self._bwd_handles:
            h.remove()
        self._fwd_handles.clear()
        self._bwd_handles.clear()


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
def load_calibration(tokenizer, source: str, n_samples: int,
                     seqlen: int) -> torch.Tensor:
    """Load calibration from a HuggingFace dataset id, a local .jsonl, or
    a local .txt file. JSONL rows can have either {"text": ...} or
    {"messages": [...]} for chat-style data.
    """
    import os
    from datasets import load_dataset

    texts: list[str] = []
    if source.endswith(".jsonl") and os.path.exists(source):
        with open(source) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "messages" in obj:
                    try:
                        texts.append(tokenizer.apply_chat_template(
                            obj["messages"], tokenize=False))
                    except Exception:
                        continue
                elif "text" in obj:
                    texts.append(obj["text"])
    elif source.endswith(".txt") and os.path.exists(source):
        with open(source) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif source == "ultrachat_200k":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split="train_sft", streaming=True)
        for row in ds:
            msgs = row.get("messages", [])
            if not msgs:
                continue
            try:
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
            except Exception:
                # Tokenizer has no chat template (base / pt models).
                # Join plain message contents — good enough as calibration.
                parts = []
                for m in msgs:
                    if isinstance(m, dict):
                        c = m.get("content")
                        if isinstance(c, str) and c:
                            parts.append(c)
                if parts:
                    texts.append("\n\n".join(parts))
            if len(texts) >= n_samples * 8:
                break
    else:
        # Generic HF dataset loader. Handles three common schemas:
        #   1. {"text": "..."} — raw text corpora (pile, wikitext, etc.)
        #   2. {"messages": [...]} — chat-format SFT (ultrachat, tulu-3, etc.)
        #   3. anything else — falls back to first string column
        # Streaming when possible so we don't download the full dataset for
        # just 32 samples.
        try:
            ds = load_dataset(source, split="train", streaming=True)
            stream = True
        except Exception:
            ds = load_dataset(source, split="train")
            stream = False

        # Probe one row to detect schema
        iterator = iter(ds) if stream else ds
        first = next(iterator) if stream else (ds[0] if len(ds) else {})
        schema = None
        if "messages" in first:
            schema = "messages"
        elif "text" in first:
            schema = "text"
        else:
            # pick first string-valued column
            for k, v in first.items():
                if isinstance(v, str):
                    schema = k
                    break
        if schema is None:
            raise ValueError(f"Could not find text or messages field in {source}")
        print(f"[probe] {source} schema: {schema}", flush=True)

        # Re-iterate (we consumed the first row)
        if stream:
            ds = load_dataset(source, split="train", streaming=True)
            iterator = iter(ds)
        else:
            iterator = iter(ds)

        for row in iterator:
            if schema == "messages":
                msgs = row.get("messages") or row.get("conversations") or []
                if not msgs:
                    continue
                try:
                    texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
                except Exception:
                    # Tokenizer has no chat template (base / pt models)
                    # or the template rejects this message shape.
                    # Fall back to concatenating the message contents —
                    # good enough as calibration text, decouples the
                    # probe from the tokenizer's chat-template state.
                    parts = []
                    for m in msgs:
                        if isinstance(m, dict):
                            c = m.get("content")
                            if isinstance(c, str) and c:
                                parts.append(c)
                    if parts:
                        texts.append("\n\n".join(parts))
            else:
                v = row.get(schema)
                if isinstance(v, str) and v.strip():
                    texts.append(v)
            if len(texts) >= n_samples * 8:
                break

    # Two-pass sampling:
    #   1) first pass picks any sample already >= seqlen tokens
    #   2) fallback packs multiple short samples together (separated by
    #      EOS) to reach seqlen. This makes SFT/chat datasets with short
    #      turns (tulu-3, glaive) usable without lowering seqlen.
    random.seed(42)
    samples = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids
        if ids.size(1) < seqlen:
            continue
        start = random.randint(0, ids.size(1) - seqlen)
        samples.append(ids[0, start:start + seqlen])
        if len(samples) >= n_samples:
            break

    if len(samples) < n_samples:
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        # Pack short samples by concatenating with EOS separator
        buf: list[int] = []
        for t in texts:
            ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids[0].tolist()
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= seqlen and len(samples) < n_samples:
                samples.append(torch.tensor(buf[:seqlen], dtype=torch.long))
                buf = buf[seqlen:]
            if len(samples) >= n_samples:
                break

    if len(samples) < n_samples:
        print(f"[probe] warning: only got {len(samples)}/{n_samples} samples "
              f"(even with packing). Consider wider corpus.",
              flush=True)
    return torch.stack(samples[:n_samples], dim=0)


def per_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1), reduction="none")
    return ce.view(shift_labels.size())


def load_probe_model_and_tokenizer(model_path: str,
                                   requested_device: str,
                                   dtype: torch.dtype,
                                   device_map: str | None = None,
                                   gradient_checkpointing: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    staged = stage_text_only(model_path)
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
    load_device_map = device_map if device_map is not None else requested_device

    model = AutoModelForCausalLM.from_pretrained(
        staged, torch_dtype=dtype, device_map=load_device_map,
        low_cpu_mem_usage=False, trust_remote_code=True,
    )
    model.eval()

    # Packed MoE experts (e.g. Qwen3.5/3.6's 3D `gate_up_proj` /
    # `down_proj`) are sensed natively by FisherAccumulator via
    # install_packed_expert_hooks. No unfuse step needed; auto_round is
    # not a probe-time dependency.

    exec_device = resolve_execution_device(model, requested_device)
    print(f"[probe] execution device: {exec_device} "
          f"(load device_map={load_device_map})", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    return staged, tokenizer, model, exec_device, load_device_map


def run_probe_pass(model: nn.Module,
                   tokenizer,
                   calib: torch.Tensor,
                   model_name: str,
                   dataset_name: str,
                   seqlen: int,
                   dtype_name: str,
                   requested_device: str,
                   load_device_map,
                   exec_device: torch.device,
                   linear_include: str,
                   linear_exclude: str,
                   importance_weighting: bool,
                   activation_cache_dir: str | None,
                   output_path: str,
                   h_detail_dir: str | None = None):
    inc = re.compile(linear_include)
    exc = re.compile(linear_exclude)
    tracked = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear)
               and inc.search(n) and not exc.search(n)]
    print(f"[probe] tracking {len(tracked)} Linear layers", flush=True)

    expert_info_all = discover_moe_structure(model)
    expert_info = {k: v for k, v in expert_info_all.items() if k in tracked}
    top_k = read_top_k(model, default=2)
    routers = sorted({r for r, _ in expert_info.values()})
    print(f"[probe] MoE: {len(expert_info)} expert linears, "
          f"{len(routers)} routers, top_k={top_k}", flush=True)
    if len(expert_info) == 0:
        diag_count = 0
        for pname, pmod in model.named_modules():
            for attr in ("experts", "block_sparse_moe_experts",
                         "moe_experts", "expert_layer"):
                child = getattr(pmod, attr, None)
                if child is None or not isinstance(child, nn.Module):
                    continue
                kids = list(child.named_children())
                numkids = [k for k, _ in kids if k.isdigit()]
                print(f"[probe/diag] parent={pname!r} attr={attr!r} "
                      f"container_cls={type(child).__name__} "
                      f"n_children={len(kids)} n_numeric_children={len(numkids)}"
                      f" first_children={[k for k,_ in kids[:5]]}",
                      flush=True)
                diag_count += 1
                if diag_count >= 3:
                    break
            if diag_count >= 3:
                break

    tracker = RouterTracker(model, routers, top_k) if routers else None
    cache_dir = Path(activation_cache_dir) if activation_cache_dir else None
    detail_dir = Path(h_detail_dir) if h_detail_dir else None
    acc = FisherAccumulator(model, tracked, expert_info, cache_dir,
                            h_detail_dir=detail_dir)

    print(f"[probe] calibration shape: {calib.shape}", flush=True)

    model.train()
    t_fwd = t_bwd = 0.0
    for i in range(calib.size(0)):
        ids = calib[i:i+1].to(exec_device)
        t0 = time.time()
        with torch.no_grad():
            embed = model.get_input_embeddings()(ids)
        embed.requires_grad_(True)
        out = model(inputs_embeds=embed, labels=ids)
        logits = out.logits
        t_fwd += time.time() - t0

        t0 = time.time()
        # Use sum-reduction CE so per-token gradients aggregate without
        # the 1/T factor that mean-reduction introduces. The accumulated
        # ||grad_W||²_F divided by total tokens then gives the per-token
        # empirical Fisher diagonal trace under the standard assumption
        # of independence across token positions.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        lp = F.log_softmax(
            shift_logits.reshape(-1, shift_logits.size(-1)), dim=-1)
        gather = -lp.gather(1, shift_labels.reshape(-1, 1)).squeeze(1)
        if importance_weighting:
            with torch.no_grad():
                tok = per_token_ce(logits.detach(), ids).reshape(-1)
                mean = float(tok.mean().item())
            # Importance weights renormalized to mean ~1 so the per-token
            # Fisher units are preserved (the weights only redistribute
            # contributions across token positions, not change the total).
            w = (tok / max(mean, 1e-6)).clamp(0.25, 4.0)
            loss = (gather * w).sum()
        else:
            loss = gather.sum()
        loss.backward()
        t_bwd += time.time() - t0

        if (i + 1) % 4 == 0 or i == 0:
            n_tok = max(int(gather.numel()), 1)
            mean_loss = float(loss.detach().item()) / n_tok
            print(f"[probe] sample {i+1}/{calib.size(0)} "
                  f"loss={mean_loss:.3f} "
                  f"fwd_avg={t_fwd/(i+1):.2f}s bwd_avg={t_bwd/(i+1):.2f}s",
                  flush=True)

        del out, loss, ids, embed, logits
        acc._saved_inputs.clear()

    acc.finalize(tracker)
    acc.remove_hooks()
    if tracker is not None:
        tracker.remove_hooks()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "stats": acc.stats,
            "router_counts": dict(tracker.counts) if tracker else {},
            "router_totals": dict(tracker.total_tokens) if tracker else {},
            "expert_info": expert_info,
            "meta": {
                "model": model_name,
                "dataset": dataset_name,
                "nsamples": calib.size(0),
                "seqlen": seqlen,
                "dtype": dtype_name,
                "device_map": str(load_device_map),
                "execution_device": str(exec_device),
                "top_k": top_k,
                "importance_weighting": importance_weighting,
                "activation_cache_dir": str(cache_dir) if cache_dir else None,
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
            },
        }, f)
    print(f"[probe] wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Measure per-Linear sensitivity (Fisher trace) with "
                    "route-aware MoE weighting and per-token importance.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="ultrachat_200k",
                    help="HF dataset name, or path to .jsonl/.txt")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None,
                    help="HF from_pretrained device_map. Defaults to --device. "
                         "Use 'auto' to allow CPU/GPU model sharding while still "
                         "running the probe on the embedding device.")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output", required=True,
                    help="Pickle with per-Linear stats")
    ap.add_argument("--activation-cache-dir", default=None,
                    help="Save per-Linear input activation snapshots here "
                         "(for measure_quant_cost.py)")
    ap.add_argument("--linear-include", default=".*")
    ap.add_argument("--linear-exclude",
                    # Routers stay BF16 — quantizing per-token routing
                    # decisions is high-risk for negligible memory gain.
                    # `lm_head` is intentionally NOT excluded; the
                    # allocator can pick a sensible format for it.
                    default=r"(?:mlp\.gate$|mlp\..*gate$|"
                            r"\.router(?:$|\.)|"
                            r"block_sparse_moe\.gate$)")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false",
                    dest="gradient_checkpointing")
    ap.add_argument("--importance-weighting", action="store_true", default=True)
    ap.add_argument("--no-importance-weighting", action="store_false",
                    dest="importance_weighting")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    print(f"[probe] loading {args.model}", flush=True)
    t0 = time.time()
    _, tokenizer, model, exec_device, load_device_map = load_probe_model_and_tokenizer(
        args.model,
        requested_device=args.device,
        dtype=dtype,
        device_map=args.device_map,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    print(f"[probe] loaded in {time.time()-t0:.1f}s", flush=True)

    calib = load_calibration(tokenizer, args.dataset, args.nsamples, args.seqlen)
    run_probe_pass(
        model=model,
        tokenizer=tokenizer,
        calib=calib,
        model_name=args.model,
        dataset_name=args.dataset,
        seqlen=args.seqlen,
        dtype_name=args.dtype,
        requested_device=args.device,
        load_device_map=load_device_map,
        exec_device=exec_device,
        linear_include=args.linear_include,
        linear_exclude=args.linear_exclude,
        importance_weighting=args.importance_weighting,
        activation_cache_dir=args.activation_cache_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
