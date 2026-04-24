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

Two layouts, both handled via the same `ExpertSaliencyTracker`:

1. **Nested** (per-expert `nn.Module`): `mlp.experts.{eid}` is a module
   with its own forward method. Mixtral, GPT-OSS, and the HF-reference
   Deepseek-V3 implementations are nested. Each expert module gets a
   forward hook; the router's hook stashes top-k probs/ids for the
   same-layer expert hooks to consume.

2. **Packed-3D** (batched expert tensors): `mlp.experts.gate_up_proj`
   is a `[num_experts, 2*inter, hidden]` nn.Parameter — there's no
   per-expert nn.Module to hook. Qwen3.5 / Qwen3.6 MoE use this
   layout in the HF reference code. We instance-level monkey-patch the
   packed-experts module's forward to replicate its per-expert compute
   while accumulating S_j inside the same loop; the patch is reversed
   in `remove_hooks()` and is zero-overhead (shares the real forward's
   matmuls, not a duplicate pass).

Callers drive layout selection by populating either the
`routers_and_experts` list (layout 1) or the `packed_moe_blocks` list
(layout 2). Both can coexist on the same tracker instance.

### Thread safety / reentrancy

The observer is single-threaded. Each router forward populates a
transient `_last_topk_*` buffer consumed by the same-layer expert
hooks that fire next. If the caller invokes forward passes in
overlapping threads, the buffers will race. This matches the rest of
the prismaquant probe machinery (single-threaded streaming).

### Storage

Accumulators live on the hook's firing device (`float64` for numeric
stability on long calibration runs). On unified-memory systems (GB10)
this avoids duplicating tensors across CPU/GPU pools — both are the
same physical RAM but torch's allocator treats them as separate, so
`.cpu()` inside a hook would double-allocate. We harvest to CPU once
in `saliency()`. A 256-expert × 64-layer model costs ~128 KB total —
negligible on either device.
"""
from __future__ import annotations

import re
import types
from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "ExpertSaliencyTracker",
    "saliency_from_moe_structure",
    "saliency_from_packed_moe",
]


# Attribute names commonly used on packed-experts modules for the
# gate/up fusion and the down projection. Detection is additive — a
# module is "packed" when any of these is a 3D nn.Parameter.
_PACKED_GATE_UP_NAMES = ("gate_up_proj",)
_PACKED_DOWN_NAMES = ("down_proj",)
_PACKED_ALL_NAMES = _PACKED_GATE_UP_NAMES + _PACKED_DOWN_NAMES + ("w1", "w2", "w3")


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
        packed_moe_blocks: list[dict] | None = None,
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
          packed_moe_blocks: list of dicts from
              ``saliency_from_packed_moe`` for MoE blocks whose experts
              are stored as packed 3D tensors (no per-expert module to
              hook). Each entry holds ``{"router_qname",
              "experts_qname", "num_experts"}``. For these blocks we
              install an instance-level monkey-patch on the experts
              module's forward that replicates its own per-expert loop
              while emitting S_j contributions along the way — the
              patch is reverted by ``remove_hooks()``. Currently
              specialized to the Qwen3.5/3.6 ``Qwen3_5MoeExperts``
              compute (gate_up_proj + act + down_proj), which is the
              only packed-3D layout in the supported model set.
        """
        self.top_k = int(top_k)
        self.softmax_dtype = softmax_dtype

        self._handles: list = []
        # Instance-level monkey-patched experts modules (packed-3D path).
        # Populated by `_install_packed_experts_patch`; reverted by
        # `remove_hooks`.
        self._patched_packed_modules: list[nn.Module] = []
        # Per-(router, expert) accumulators in fp64. Lazily allocated on
        # the hook's firing device (first time the router's forward runs
        # after materialization). `sum_g_norm` / `count` feed the mean-
        # contribution saliency (REAP's published formula); `max_g_norm`
        # feeds the niche-protecting saliency that prefers experts with
        # a few large-magnitude contributions. Device-local storage
        # avoids CPU↔GPU duplication on unified-memory systems and
        # lets the hot-path accumulation stay sync-free (no .item()
        # per expert per forward).
        # Shape: [num_experts] per router. Storage cost is trivial.
        self._num_experts_by_router: dict[str, int] = {}
        self.sum_g_norm: dict[str, torch.Tensor] = {}
        self.count: dict[str, torch.Tensor] = {}
        self.max_g_norm: dict[str, torch.Tensor] = {}
        # `sum_g_norm_sq[router][j]` accumulates Σ_t g_j(t) · ||f_j(t)||²
        # over active tokens — REAP's "dropout loss" per expert, before
        # normalization. Divided by total tokens seen in `saliency(
        # reduction="reap_dropout")` to produce Δ L_j in units directly
        # comparable to the allocator's weight-MSE·Fisher Δloss terms.
        self.sum_g_norm_sq: dict[str, torch.Tensor] = {}
        # `total_tokens_by_router[router]` is the batch-size × seq-len
        # count of tokens across all calibration forwards on this
        # router — the normalizer for the REAP dropout formula. Not
        # per-expert; every token contributes to the router's total
        # regardless of which top-k set included a given expert.
        self.total_tokens_by_router: dict[str, torch.Tensor] = {}
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

            # Record expected num_experts so saliency() can return
            # dead routers (never fired) with shape-correct zeros.
            # Actual allocation is deferred to the first hook fire via
            # `_ensure_accumulators`, on the hook's device.
            self._num_experts_by_router[router_qname] = num_experts

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
                # Drop the expected-count entry for a router we couldn't
                # attach to; otherwise saliency() would report phantom
                # zeros for a router that never fires.
                self._num_experts_by_router.pop(router_qname, None)

        # Packed-3D path: for each block, monkey-patch the packed
        # experts module's forward to accumulate S_j alongside its own
        # per-expert compute. No duplicate matmuls, no forward hooks
        # (the patch replaces the method entirely).
        for entry in packed_moe_blocks or []:
            self._install_packed_experts_patch(model, entry)

    def _install_packed_experts_patch(
        self,
        model: nn.Module,
        entry: dict,
    ) -> bool:
        """Swap `experts_mod.forward` for a wrapper that replicates its
        own per-expert compute and writes S_j into the tracker. Returns
        True on success, False if the module layout isn't recognized
        or is already patched.
        """
        router_qname = entry.get("router_qname")
        experts_qname = entry.get("experts_qname")
        num_experts = entry.get("num_experts")
        if not (router_qname and experts_qname and num_experts):
            return False
        try:
            experts_mod = model.get_submodule(experts_qname)
        except AttributeError:
            return False
        # Require the Qwen3_5MoeExperts-shaped API: `gate_up_proj`,
        # `down_proj`, and `act_fn` attributes. Other packed layouts
        # (Mixtral w1/w2/w3, MiniMax) can be added via sibling
        # implementations if they enter the supported set.
        gate_up = getattr(experts_mod, "gate_up_proj", None)
        down = getattr(experts_mod, "down_proj", None)
        act_fn = getattr(experts_mod, "act_fn", None)
        if not (isinstance(gate_up, nn.Parameter) and gate_up.dim() == 3):
            return False
        if not (isinstance(down, nn.Parameter) and down.dim() == 3):
            return False
        if act_fn is None:
            return False
        if hasattr(experts_mod, "_pq_saliency_patched"):
            return False  # idempotent
        experts_mod._pq_saliency_patched = True
        experts_mod._pq_saliency_tracker = self
        experts_mod._pq_saliency_router = router_qname
        experts_mod._pq_saliency_original_forward = experts_mod.forward
        experts_mod.forward = types.MethodType(
            _qwen3_5_moe_experts_saliency_forward, experts_mod,
        )
        self._patched_packed_modules.append(experts_mod)
        self._num_experts_by_router[router_qname] = int(num_experts)
        self._registered[router_qname].update(range(int(num_experts)))
        return True

    # -- hook factories -----------------------------------------------------

    @staticmethod
    def _infer_num_experts(router_mod: nn.Module) -> int | None:
        if isinstance(router_mod, nn.Linear):
            return int(router_mod.out_features)
        w = getattr(router_mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim >= 1:
            return int(w.shape[0])
        return None

    def _ensure_accumulators(self, router_qname: str, device: torch.device) -> None:
        """Lazily allocate per-router accumulators on the hook's firing
        device. No-op once allocated; safe to call every hook.

        On unified-memory systems (GB10) we put them on the same device
        as the forward so the hot-path accumulations stay on-device
        without .cpu() transfers or .item() syncs.
        """
        if router_qname in self.sum_g_norm:
            return
        num = self._num_experts_by_router.get(router_qname)
        if num is None:
            return
        self.sum_g_norm[router_qname] = torch.zeros(
            num, dtype=torch.float64, device=device,
        )
        self.count[router_qname] = torch.zeros(
            num, dtype=torch.int64, device=device,
        )
        self.max_g_norm[router_qname] = torch.zeros(
            num, dtype=torch.float64, device=device,
        )
        self.sum_g_norm_sq[router_qname] = torch.zeros(
            num, dtype=torch.float64, device=device,
        )
        self.total_tokens_by_router[router_qname] = torch.zeros(
            (), dtype=torch.int64, device=device,
        )

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
            # Stay on the forward's device. Expert hooks that fire later
            # in the same layer read these directly without a GPU→CPU
            # hop. topk_i is already int64 on the device; no cast needed.
            self._last_topk_probs[router_qname] = probs.detach()
            self._last_topk_ids[router_qname] = topk_i.detach()
            # Accumulate the router's total-tokens-seen for REAP's
            # dropout-loss normalizer (Δ L_j = Σ g·||f||² / T_cal). The
            # router fires exactly once per forward, so this counts
            # each token exactly once regardless of top-k.
            self._ensure_accumulators(router_qname, flat.device)
            acc_total = self.total_tokens_by_router.get(router_qname)
            if acc_total is not None:
                acc_total += flat.size(0)

        return hook

    def _make_expert_hook(self, router_qname: str, expert_idx: int) -> Callable:
        def hook(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                expert_out = out
            else:
                expert_out = out[0]
            if not isinstance(expert_out, torch.Tensor):
                return

            tk_ids = self._last_topk_ids.get(router_qname)
            tk_probs = self._last_topk_probs.get(router_qname)
            if tk_ids is None or tk_probs is None:
                # Router hook didn't run before this expert — unusual;
                # skip rather than corrupt stats.
                return

            device = expert_out.device
            self._ensure_accumulators(router_qname, device)
            acc_sum = self.sum_g_norm.get(router_qname)
            acc_count = self.count.get(router_qname)
            acc_max = self.max_g_norm.get(router_qname)
            acc_sum_sq = self.sum_g_norm_sq.get(router_qname)
            acc_total = self.total_tokens_by_router.get(router_qname)
            if (acc_sum is None or acc_count is None or acc_max is None
                    or acc_sum_sq is None or acc_total is None):
                return

            # ||f_j(t)||_2 along the hidden dim — expert_out is
            # [num_tokens_routed_to_this_expert, hidden_size] in the
            # HF-reference MoE dispatch. fp64 norm stays on-device;
            # tensor is tiny (num_tokens scalars).
            norms = expert_out.detach().to(torch.float64).norm(dim=-1)

            # For each token in tk_ids, gate_vals[t] is the softmax
            # probability this expert received on token t (0 when the
            # expert wasn't in t's top-k).
            mask = (tk_ids == expert_idx)                       # [tokens, k]
            gate_vals = (
                tk_probs.to(torch.float64) * mask.to(torch.float64)
            ).sum(dim=-1)                                       # [tokens]
            active = gate_vals > 0
            n_all = int(tk_ids.shape[0])  # python int, no sync

            # Match expert-output norms to active tokens. Two dispatch
            # layouts exist in the wild:
            #   (a) HF reference: each expert receives ONLY its active
            #       tokens' inputs. norms.shape[0] == n_active.
            #   (b) Broadcast: expert sees every token, masked
            #       internally. norms.shape[0] == n_all.
            # Both layouts' norms.shape[0] is a static shape attr — no
            # sync. The n_active path still needs one sync for the
            # boolean-select; the n_all path stays fully sync-free.
            nshape0 = int(norms.shape[0])
            if nshape0 == n_all:
                # Broadcast layout: keep full-shape, mask via `where`
                # so max/sum reductions don't see stale negative data.
                zero = torch.zeros((), dtype=torch.float64, device=device)
                contribution_full = torch.where(active, gate_vals * norms, zero)
                contrib_sum = contribution_full.sum()
                # active positions contain a >=0 product; inactive
                # positions are exactly 0. Max over the full tensor
                # equals max over active positions provided any active
                # contribution is >= 0 (which they are — norm and prob
                # are both non-negative).
                contrib_max = contribution_full.max()
                # REAP dropout loss contribution per token: g·||f||²
                # (not (g·||f||)²). Zero on inactive tokens by the
                # gate mask.
                contribution_sq = torch.where(
                    active, gate_vals * norms.pow(2), zero,
                )
                contrib_sum_sq = contribution_sq.sum()
                n_active_inc = active.to(torch.int64).sum()
            else:
                # Assume HF layout (expert saw only active tokens). The
                # boolean select below produces a data-dependent shape
                # that forces one sync; unavoidable for this layout.
                active_gates = gate_vals[active]
                if active_gates.numel() != nshape0:
                    # Layout mismatch (neither broadcast nor HF) — skip
                    # rather than corrupt stats.
                    return
                contribution = active_gates * norms
                contrib_sum = contribution.sum()
                contrib_max = (
                    contribution.max() if contribution.numel() > 0
                    else torch.zeros((), dtype=torch.float64, device=device)
                )
                contrib_sum_sq = (active_gates * norms.pow(2)).sum()
                n_active_inc = torch.tensor(nshape0, dtype=torch.int64, device=device)

            # In-place accumulate on device — no sync, no host round-trip.
            acc_sum[expert_idx] += contrib_sum
            acc_count[expert_idx] += n_active_inc
            acc_max[expert_idx] = torch.maximum(acc_max[expert_idx], contrib_max)
            acc_sum_sq[expert_idx] += contrib_sum_sq
            # Token-total accumulation happens in the router hook, not
            # here — router fires exactly once per forward regardless
            # of which experts are active.

        return hook

    # -- public interface ---------------------------------------------------

    def remove_hooks(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        # Revert packed-3D monkey-patches.
        for mod in self._patched_packed_modules:
            orig = getattr(mod, "_pq_saliency_original_forward", None)
            if orig is not None:
                try:
                    mod.forward = orig
                except Exception:
                    pass
            for attr in (
                "_pq_saliency_patched",
                "_pq_saliency_tracker",
                "_pq_saliency_router",
                "_pq_saliency_original_forward",
            ):
                if hasattr(mod, attr):
                    try:
                        delattr(mod, attr)
                    except Exception:
                        pass
        self._patched_packed_modules.clear()

    def saliency(
        self,
        reduction: str = "mean",
    ) -> dict[str, dict[int, float]]:
        """Return S_j per (router, expert_id).

        Args:
          reduction:
            - ``"mean"``: average of ``g_j · ||f_j||_2`` over tokens
              that activated expert j — a ranking signal, not a Δloss.
              Best when an expert's importance correlates with typical
              contribution magnitude.
            - ``"max"``: running max over the same contributions. Niche
              experts that contribute a large norm on a few tokens get a
              high score even if their mean is small; good when the
              allocator wants to protect specialization.
            - ``"max_mean_geomean"``: geometric mean of max and mean —
              a mid-point that punishes "tiny-mean but tiny-max" experts
              harder than mean alone while still weighting frequency.
            - ``"reap_dropout"`` (REAP paper Eq. 11): the direct
              per-expert DROPOUT LOSS
              ``Δ L_j ≈ (1/T_cal) Σ_t g_j(t) · ||f_j(t)||²``. Units
              match the allocator's other Δloss terms (weight-MSE ·
              Fisher), so no scalar recalibration is needed — the
              allocator consumes this directly as prune-cost.

        Experts that never activated during calibration get a saliency
        of 0.0 — the allocator can treat them as "safe to drop" directly.
        """
        if reduction not in ("mean", "max", "max_mean_geomean", "reap_dropout"):
            raise ValueError(f"unknown reduction {reduction!r}")
        out: dict[str, dict[int, float]] = {}
        # Cover dead routers (no hook fires → no accumulators allocated)
        # with their expected-count zeros so the allocator gets a
        # shape-complete saliency dict.
        qnames = set(self._num_experts_by_router) | set(self.sum_g_norm)
        for qname in qnames:
            s = self.sum_g_norm.get(qname)
            c = self.count.get(qname)
            m = self.max_g_norm.get(qname)
            num_experts = self._num_experts_by_router.get(
                qname, s.numel() if s is not None else 0,
            )
            # Single bulk sync per router — harvest accumulators to CPU
            # once and do the fp64 Python extraction from CPU tensors.
            # Compared to the previous N-experts × 3 `.item()` calls,
            # this collapses to one device→host transfer per router.
            sq = self.sum_g_norm_sq.get(qname)
            tt = self.total_tokens_by_router.get(qname)
            if s is None:
                s_cpu = torch.zeros(num_experts, dtype=torch.float64)
                c_cpu = torch.zeros(num_experts, dtype=torch.int64)
                m_cpu = torch.zeros(num_experts, dtype=torch.float64)
                sq_cpu = torch.zeros(num_experts, dtype=torch.float64)
                tt_val = 0
            else:
                s_cpu = s.detach().to("cpu")
                c_cpu = c.detach().to("cpu")
                m_cpu = m.detach().to("cpu")
                sq_cpu = sq.detach().to("cpu") if sq is not None else torch.zeros(
                    num_experts, dtype=torch.float64)
                tt_val = int(tt.detach().to("cpu").item()) if tt is not None else 0
            slc: dict[int, float] = {}
            for e in range(num_experts):
                c_e = int(c_cpu[e].item())
                mean_val = float(s_cpu[e].item()) / c_e if c_e > 0 else 0.0
                max_val = float(m_cpu[e].item())
                if reduction == "mean":
                    slc[int(e)] = mean_val
                elif reduction == "max":
                    slc[int(e)] = max_val
                elif reduction == "max_mean_geomean":
                    slc[int(e)] = float((mean_val * max_val) ** 0.5)
                else:  # reap_dropout
                    slc[int(e)] = (
                        float(sq_cpu[e].item()) / tt_val if tt_val > 0 else 0.0
                    )
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
# Packed-3D patched forward (Qwen3_5MoeExperts shape)
# ---------------------------------------------------------------------------
def _qwen3_5_moe_experts_saliency_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Instance-level replacement for ``Qwen3_5MoeExperts.forward``.

    Bit-identical output semantics to the upstream implementation; the
    only addition is REAP saliency accumulation inside the per-expert
    loop. ``self._pq_saliency_tracker`` and ``self._pq_saliency_router``
    are set by ``ExpertSaliencyTracker._install_packed_experts_patch``.
    """
    tracker = getattr(self, "_pq_saliency_tracker", None)
    router_qname = getattr(self, "_pq_saliency_router", None)
    if tracker is not None and router_qname is not None:
        tracker._ensure_accumulators(router_qname, hidden_states.device)
        acc_sum = tracker.sum_g_norm.get(router_qname)
        acc_count = tracker.count.get(router_qname)
        acc_max = tracker.max_g_norm.get(router_qname)
        acc_sum_sq = tracker.sum_g_norm_sq.get(router_qname)
        acc_total = tracker.total_tokens_by_router.get(router_qname)
        # For the packed path we hook the experts module directly — no
        # router hook fires for us. Increment the token total here,
        # once per forward.
        if acc_total is not None:
            acc_total += hidden_states.shape[0]
    else:
        acc_sum = acc_count = acc_max = acc_sum_sq = None

    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        # One sync per forward (amortized across all experts in this
        # block) — convert the "which experts got hit" tensor to a
        # Python list once, not once per iteration.
        expert_hit_ints = (
            torch.greater(expert_mask.sum(dim=(-1, -2)), 0)
            .nonzero().flatten().tolist()
        )

    for e_int in expert_hit_ints:
        if e_int == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[e_int])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, self.gate_up_proj[e_int]).chunk(2, dim=-1)
        inter = self.act_fn(gate) * up
        expert_out = F.linear(inter, self.down_proj[e_int])

        # REAP saliency: f_j BEFORE routing-weight multiply, weighted
        # by the top-k prob that routed each token to this expert.
        if acc_sum is not None:
            gate_vals = top_k_weights[token_idx, top_k_pos].to(torch.float64)
            norms = expert_out.to(torch.float64).norm(dim=-1)
            contribution = gate_vals * norms                   # g·||f||
            contribution_sq = gate_vals * norms.pow(2)         # g·||f||² (REAP)
            acc_sum[e_int] += contribution.sum()
            acc_count[e_int] += norms.numel()
            if norms.numel() > 0:
                acc_max[e_int] = torch.maximum(acc_max[e_int], contribution.max())
            acc_sum_sq[e_int] += contribution_sq.sum()

        # Finish the layer's own compute.
        routed = expert_out * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, routed.to(final_hidden_states.dtype),
        )

    return final_hidden_states


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


# ---------------------------------------------------------------------------
# Helper: discover packed-3D MoE blocks (Qwen3.5/3.6 style)
# ---------------------------------------------------------------------------
def saliency_from_packed_moe(model: nn.Module) -> list[dict]:
    """Walk ``model`` and return one entry per packed-3D MoE block —
    blocks where the experts container holds ``gate_up_proj`` and
    ``down_proj`` as 3D ``nn.Parameter`` instead of a ``ModuleList`` of
    per-expert modules.

    Returns a list of dicts with keys ``router_qname``,
    ``experts_qname``, and ``num_experts`` ready to pass to
    ``ExpertSaliencyTracker(__init__, ..., packed_moe_blocks=...)``.

    Dense models and nested-MoE models return an empty list.
    """
    results: list[dict] = []
    for parent_qname, parent in model.named_modules():
        # Find a sibling pair: (router, experts-container).
        # Typical Qwen3.5 layout: parent has `gate` (TopKRouter) and
        # `experts` (packed). Parent is the SparseMoeBlock.
        gate = getattr(parent, "gate", None)
        experts = getattr(parent, "experts", None)
        if gate is None or experts is None:
            continue
        if not isinstance(experts, nn.Module):
            continue
        # Packed-3D detection: at least one 3D nn.Parameter matching
        # one of the known packed-param names.
        has_packed = False
        for pn in _PACKED_ALL_NAMES:
            p = getattr(experts, pn, None)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                has_packed = True
                break
        if not has_packed:
            continue
        # Infer num_experts from the router. Qwen3_5MoeTopKRouter stores
        # its weight directly as a Parameter of shape [num_experts, D];
        # nn.Linear routers expose out_features.
        if isinstance(gate, nn.Linear):
            num_experts = int(gate.out_features)
        else:
            w = getattr(gate, "weight", None)
            if not isinstance(w, torch.Tensor) or w.ndim < 1:
                continue
            num_experts = int(w.shape[0])
        router_qname = f"{parent_qname}.gate" if parent_qname else "gate"
        experts_qname = f"{parent_qname}.experts" if parent_qname else "experts"
        results.append({
            "router_qname": router_qname,
            "experts_qname": experts_qname,
            "num_experts": num_experts,
        })
    return results
