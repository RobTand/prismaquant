"""Router-weighted expert activation pruning — observer.

Implements the REAP saliency score (Cerebras 2510.13999, Eq. 9) as a
pure-observer class: forward hooks on the router + each expert module
accumulate per-(router, expert) saliency during calibration, without
modifying model weights or the forward path.

    S_j = mean_{t in tokens_active(j)} [ g_j(t) · ||f_j(t)||_2 ]

where
    g_j(t)     = router softmax value for expert j on token t
    f_j(t)     = expert j's output (BEFORE routing-weight multiply)
                 for token t
    ||·||_2    = L2 norm along the hidden dim
    active(j)  = the subset of tokens where j appears in the top-k
                 routing selection

The score measures the average output magnitude an expert adds to a
layer, weighted by how strongly the router selected it. High-saliency
experts handle either many tokens OR a few tokens with large-norm
outputs (niche experts) — both patterns are correctly protected.
Low-saliency experts get pruned first by the allocator.

### Supported model layouts

This observer works for MoE layouts where `discover_moe_structure`
returns a nested tree — i.e. where each expert is a Python module
(`mlp.experts.{eid}`) with its own forward method. Qwen3.5 / 3.6,
Mixtral, GPT-OSS, and the transformers reference Deepseek-V3 layout
are all nested. The packed-3D-tensor layouts (MiniMax pre-unfuse,
certain Kimi checkpoints) are not supported and will silently skip
those layers — the caller should unfuse first via the existing
`_set_minimax_fast_moe` utility or equivalent profile hook.

### Thread safety / reentrancy

The observer is single-threaded. Each router forward populates a
transient `_last_topk_*` buffer consumed by the same-layer expert
hooks that fire next. If the caller invokes forward passes in
overlapping threads, the buffers will race. This matches the rest of
the prismaquant probe machinery (single-threaded streaming).

### Storage

Accumulators live on CPU (`float64` for numeric stability on long
calibration runs), so a 256-expert × 64-layer model costs ~128 KB
of RAM — negligible. GPU memory is untouched.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["ExpertSaliencyTracker", "saliency_from_moe_structure"]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class ExpertSaliencyTracker:
    """Install forward hooks that accumulate REAP saliency per expert.

    Lifecycle:
        tracker = ExpertSaliencyTracker(model, routers_and_experts, top_k)
        # run forward passes (probe phase 2 / phase 3)
        saliency = tracker.saliency()       # {router_qname: {eid: float}}
        raw = tracker.raw_stats()           # sum_g_norm + count per expert
        tracker.remove_hooks()

    Idempotent `remove_hooks()` — safe to call multiple times.
    """

    def __init__(
        self,
        model: nn.Module,
        routers_and_experts: list[tuple[str, str, list[int]]],
        top_k: int,
        softmax_dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
          routers_and_experts: list of
              ``(router_qname, experts_parent_qname, expert_ids)``
              triples. For each entry we hook the router at
              ``model.get_submodule(router_qname)`` and each expert at
              ``model.get_submodule(f"{experts_parent_qname}.{eid}")``.
              The caller typically derives this list from
              ``sensitivity_probe.discover_moe_structure``; see the
              ``saliency_from_moe_structure`` helper below.
          top_k: number of experts selected per token by the router.
              Usually read from the model config via
              ``sensitivity_probe.read_top_k``.
          softmax_dtype: dtype in which to compute the router softmax.
              bf16 routers can overflow their own softmax on pathological
              logit magnitudes; fp32 is the safe default.
        """
        self.top_k = int(top_k)
        self.softmax_dtype = softmax_dtype

        self._handles: list = []
        # Per-(router, expert) accumulators live on CPU in fp64.
        # `sum_g_norm` / `count` feed the mean-contribution saliency
        # (REAP's published formula). `max_g_norm` feeds an alternative
        # niche-protecting saliency that prefers experts with a few
        # large-magnitude contributions over many small ones — useful
        # when the allocator wants to avoid dropping rare-but-critical
        # experts (e.g., a "closing-code-block" expert that fires on
        # <1% of tokens but contributes an outsized norm when it does).
        # Both reductions are derived from the same per-token-per-active
        # contribution sample so they cost the same single forward pass.
        self.sum_g_norm: dict[str, torch.Tensor] = {}
        self.count: dict[str, torch.Tensor] = {}
        self.max_g_norm: dict[str, torch.Tensor] = {}
        # Transient per-forward cache. Populated by the router hook,
        # consumed by each expert's hook that fires afterward in the
        # same layer's forward.
        self._last_topk_probs: dict[str, torch.Tensor] = {}
        self._last_topk_ids: dict[str, torch.Tensor] = {}
        # Record which (router, expert) we actually installed so the
        # saliency() output covers exactly those.
        self._registered: dict[str, set[int]] = defaultdict(set)

        for router_qname, experts_parent_qname, expert_ids in routers_and_experts:
            try:
                router_mod = model.get_submodule(router_qname)
            except AttributeError:
                continue
            num_experts = self._infer_num_experts(router_mod)
            if num_experts is None:
                continue

            # Hook the router. Even if no experts attach successfully we
            # don't keep the router hook — add handles lazily.
            router_handle = router_mod.register_forward_hook(
                self._make_router_hook(router_qname)
            )
            added_any_expert = False

            if router_qname not in self.sum_g_norm:
                self.sum_g_norm[router_qname] = torch.zeros(num_experts, dtype=torch.float64)
                self.count[router_qname] = torch.zeros(num_experts, dtype=torch.int64)
                self.max_g_norm[router_qname] = torch.zeros(num_experts, dtype=torch.float64)

            for eid in expert_ids:
                expert_qname = (
                    f"{experts_parent_qname}.{eid}" if experts_parent_qname else str(eid)
                )
                try:
                    expert_mod = model.get_submodule(expert_qname)
                except AttributeError:
                    continue
                h = expert_mod.register_forward_hook(
                    self._make_expert_hook(router_qname, int(eid))
                )
                self._handles.append(h)
                self._registered[router_qname].add(int(eid))
                added_any_expert = True

            if added_any_expert:
                self._handles.append(router_handle)
            else:
                router_handle.remove()
                # Drop the accumulator for a router we couldn't attach
                # to; otherwise saliency() would report phantom zeros.
                self.sum_g_norm.pop(router_qname, None)
                self.count.pop(router_qname, None)

    # -- hook factories -----------------------------------------------------

    @staticmethod
    def _infer_num_experts(router_mod: nn.Module) -> int | None:
        if isinstance(router_mod, nn.Linear):
            return int(router_mod.out_features)
        w = getattr(router_mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim >= 1:
            return int(w.shape[0])
        return None

    def _make_router_hook(self, router_qname: str) -> Callable:
        softmax_dtype = self.softmax_dtype
        top_k = self.top_k

        def hook(_module, _inp, out):
            scores = out if isinstance(out, torch.Tensor) else out[0]
            if not isinstance(scores, torch.Tensor):
                return
            flat = scores.detach().reshape(-1, scores.size(-1))
            k = min(top_k, int(flat.size(-1)))
            topk_v, topk_i = flat.topk(k, dim=-1)
            probs = F.softmax(topk_v.to(softmax_dtype), dim=-1)
            # Store on CPU. Expert hook fires later in the same
            # forward and reads these; moving to CPU early lets us
            # free the GPU-side tensor promptly.
            self._last_topk_probs[router_qname] = probs.detach().cpu()
            self._last_topk_ids[router_qname] = topk_i.detach().cpu().to(torch.int64)

        return hook

    def _make_expert_hook(self, router_qname: str, expert_idx: int) -> Callable:
        def hook(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                expert_out = out
            else:
                expert_out = out[0]
            if not isinstance(expert_out, torch.Tensor):
                return
            # ||f_j(t)||_2 along the hidden dim — expert_out is
            # [num_tokens_routed_to_this_expert, hidden_size] in the
            # HF-reference MoE dispatch. Compute the norm in fp64 on
            # the source device so the result is numerically identical
            # to a pure-fp64 reference; the norm tensor is tiny
            # (num_tokens scalars) so cost is negligible.
            norms = expert_out.detach().to(torch.float64).norm(dim=-1).cpu()

            tk_ids = self._last_topk_ids.get(router_qname)
            tk_probs = self._last_topk_probs.get(router_qname)
            if tk_ids is None or tk_probs is None:
                # Router hook didn't run before this expert — unusual;
                # skip rather than corrupt stats.
                return

            # For each token in tk_ids, find whether expert_idx appears
            # in its top-k. When it does, `gate_vals_per_token` holds
            # the softmax probability assigned to this expert; when it
            # doesn't, the entry is 0 (a masked sum).
            mask = (tk_ids == expert_idx)                       # [tokens, k]
            gate_vals_per_token = (
                tk_probs.to(torch.float64) * mask.to(torch.float64)
            ).sum(dim=-1)                                       # [tokens]
            active = gate_vals_per_token > 0
            active_gates = gate_vals_per_token[active]          # [n_active]

            # Match expert-output norms to active tokens. Two dispatch
            # layouts exist in the wild:
            #   (a) HF reference: each expert receives ONLY its active
            #       tokens' inputs. len(norms) == n_active.
            #   (b) Broadcast: expert sees every token, masked
            #       internally. len(norms) == len(tk_ids).
            # We handle both and safely skip shape mismatches.
            n_active = int(active_gates.numel())
            n_all = int(tk_ids.size(0))
            if norms.numel() == n_active:
                use_norms = norms
            elif norms.numel() == n_all:
                use_norms = norms[active]
            else:
                return

            contribution = active_gates * use_norms
            acc_sum = self.sum_g_norm.get(router_qname)
            acc_count = self.count.get(router_qname)
            acc_max = self.max_g_norm.get(router_qname)
            if acc_sum is None or acc_count is None or acc_max is None:
                return
            acc_sum[expert_idx] += float(contribution.sum().item())
            acc_count[expert_idx] += n_active
            # Running max — updated against the per-token-per-active-expert
            # contribution distribution, not a batch aggregate. Captures
            # rare-but-strong activations that the mean dilutes.
            batch_max = float(contribution.max().item())
            if batch_max > float(acc_max[expert_idx].item()):
                acc_max[expert_idx] = batch_max

        return hook

    # -- public interface ---------------------------------------------------

    def remove_hooks(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def saliency(
        self,
        reduction: str = "mean",
    ) -> dict[str, dict[int, float]]:
        """Return S_j per (router, expert_id).

        Args:
          reduction:
            - ``"mean"`` (REAP's published formula): average of
              `g_j · ||f_j||_2` over tokens that activated expert j. Best
              when an expert's importance correlates with typical
              contribution magnitude.
            - ``"max"``: running max over the same contributions. Niche
              experts that contribute a large norm on a few tokens get a
              high score even if their mean is small; good when the
              allocator wants to protect specialization.
            - ``"max_mean_geomean"``: geometric mean of max and mean —
              a mid-point that punishes "tiny-mean but tiny-max" experts
              harder than mean alone while still weighting frequency.

        Experts that never activated during calibration get a saliency
        of 0.0 — the allocator can treat them as "safe to drop" directly.
        """
        if reduction not in ("mean", "max", "max_mean_geomean"):
            raise ValueError(f"unknown reduction {reduction!r}")
        out: dict[str, dict[int, float]] = {}
        for qname, s in self.sum_g_norm.items():
            c = self.count[qname]
            m = self.max_g_norm[qname]
            slc: dict[int, float] = {}
            for e in range(s.numel()):
                c_e = int(c[e].item())
                mean_val = float(s[e].item()) / c_e if c_e > 0 else 0.0
                max_val = float(m[e].item())
                if reduction == "mean":
                    slc[int(e)] = mean_val
                elif reduction == "max":
                    slc[int(e)] = max_val
                else:  # max_mean_geomean
                    slc[int(e)] = float((mean_val * max_val) ** 0.5)
            out[qname] = slc
        return out

    def raw_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return raw accumulators. Useful for merging across shards
        OR for downstream saliency recomputation (e.g. Fisher-weighted
        saliency in the allocator, which needs `sum_g_norm` + `count`
        + `max_g_norm` separately so it can blend with weight
        sensitivity Fisher traces at cost-evaluation time)."""
        return {
            qname: {
                "sum_g_norm": self.sum_g_norm[qname].clone(),
                "count": self.count[qname].clone(),
                "max_g_norm": self.max_g_norm[qname].clone(),
            }
            for qname in self.sum_g_norm
        }

    def registered_experts(self) -> dict[str, list[int]]:
        """Return the (router_qname -> sorted expert_ids) map that was
        actually installed. Consumers (e.g. the allocator) use this to
        know which experts are measured vs treated as zero-saliency."""
        return {
            qname: sorted(eids) for qname, eids in self._registered.items()
        }


# ---------------------------------------------------------------------------
# Helper: build the (router, experts_parent, ids) list from prismaquant's
# existing MoE discovery output
# ---------------------------------------------------------------------------
def saliency_from_moe_structure(
    expert_info: dict[str, tuple[str, str]],
) -> list[tuple[str, str, list[int]]]:
    """Translate `discover_moe_structure`'s output to the tracker's init format.

    ``expert_info`` has shape ``{expert_linear_qname: (router_qname, eid)}``
    where expert_linear_qname is the qname of a Linear inside an expert —
    e.g. ``model.layers.3.mlp.experts.0.gate_proj``. We strip the
    trailing ``.{eid}.*`` segment to recover the experts-parent qname
    (``model.layers.3.mlp.experts``) and group unique expert ids per router.

    Returns an ordered list of ``(router_qname, experts_parent, [eid, ...])``
    ready to pass to ``ExpertSaliencyTracker(__init__, ..., routers_and_experts=)``.
    """
    by_router: dict[tuple[str, str], set[int]] = defaultdict(set)
    # We only want the .eid. segment that immediately follows an
    # "experts" container — not any stray numeric segment earlier
    # in the path (e.g. `model.layers.0.mlp.experts.0.gate_proj` has
    # two "0"s; we want the second).
    _EXPERTS_CONTAINERS = (
        "experts",
        "block_sparse_moe_experts",
        "moe_experts",
        "expert_layer",
    )
    for leaf_qname, (router_qname, eid_str) in expert_info.items():
        try:
            eid = int(eid_str)
        except (TypeError, ValueError):
            continue
        parts = leaf_qname.split(".")
        experts_parent = None
        for i in range(1, len(parts)):
            if parts[i] != eid_str:
                continue
            # Walk backwards looking for the "experts" marker. The eid
            # immediately follows an experts-container segment in the
            # nested layout; the projection-list layout
            # (experts.gate_up_proj.<eid>) places a projection name
            # between "experts" and the eid, but those don't have a
            # per-eid hookable module — skip those.
            if parts[i - 1] in _EXPERTS_CONTAINERS:
                experts_parent = ".".join(parts[:i])
                break
        if experts_parent is None:
            # Nothing matched — skip. (Packed layouts without a per-eid
            # hookable submodule.)
            continue
        by_router[(router_qname, experts_parent)].add(eid)

    return [
        (router_qname, experts_parent, sorted(eids))
        for (router_qname, experts_parent), eids in by_router.items()
    ]
