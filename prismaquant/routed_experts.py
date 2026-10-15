"""Profile-owned routed-expert discovery, routing and explicit replay.

Tensor rank is deliberately absent from the classification contract.  A
target is routed-expert state only when the resolved ``ModelProfile`` assigns
its qname to a packed-expert serving-format group.  That profile accessor
accepts both physical representations used by PrismaQuant: a packed parameter
(``...experts.gate_up_proj``) and an unpacked per-expert ``nn.Linear``
(``...experts.7.gate_proj``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def enumerate_packed_experts(model: nn.Module, target_names: set[str],
                              profile=None,
                              ) -> list[tuple[str, nn.Parameter, str, nn.Module]]:
    """Find every 3D nn.Parameter that lives directly under a module
    named like an MoE experts container. Uses the same class-name +
    param-name filters as probe_inventory.is_packed_experts_module
    so we never accidentally treat e.g. a Conv1d weight as a packed
    expert tensor.

    Returns [(canonical_name, packed_param, module_qname, module), ...]
    where canonical_name is `<module_qname>.<param_name>` to match the
    probe's stat keys. Only entries appearing in `target_names` are
    returned.
    """
    from prismaquant.probe_inventory import (
        is_packed_experts_module as _is_packed_experts_module,
        packed_experts_param_names as _packed_experts_param_names,
    )
    out = []
    for qname, mod in model.named_modules():
        if not _is_packed_experts_module(mod, profile):
            continue
        for pn in _packed_experts_param_names(mod, profile):
            p = getattr(mod, pn)
            full = f"{qname}.{pn}" if qname else pn
            if full in target_names:
                out.append((full, p, qname, mod))
    return out


def packed_experts_parent_module(model: nn.Module, experts_qname: str) -> nn.Module | None:
    if not experts_qname:
        return None
    if experts_qname.endswith(".experts"):
        parent_qname = experts_qname[: -len(".experts")]
    elif ".experts." in experts_qname:
        parent_qname = experts_qname.rsplit(".experts.", 1)[0]
    elif "." in experts_qname:
        parent_qname = experts_qname.rsplit(".", 1)[0]
    else:
        return None
    try:
        return model.get_submodule(parent_qname)
    except AttributeError:
        return None


def packed_experts_router(parent_module: nn.Module | None) -> nn.Module | None:
    if parent_module is None:
        return None
    for attr in ("gate", "router"):
        router = getattr(parent_module, attr, None)
        if isinstance(router, nn.Module):
            return router
    return None


def packed_router_topk(
    router: nn.Module,
    hidden_states: torch.Tensor,
    e_score_correction_bias: torch.Tensor | None = None,
    *,
    expert_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (top_k_index, top_k_weights) for a Qwen/DeepSeek-style router.

    Some routers (HYV3TopKRouter) take the parent MoE block's
    e_score_correction_bias buffer as a required positional. Newer LFM routers
    similarly take the parent's expert_bias. Callers supply the original
    buffer so capture and replay follow the model's real token assignments.
    """
    import inspect
    try:
        fwd_params = inspect.signature(router.forward).parameters
    except (TypeError, ValueError):
        fwd_params = {}
    if "e_score_correction_bias" in fwd_params:
        if e_score_correction_bias is None:
            raise ValueError(
                f"{type(router).__name__}.forward requires "
                "e_score_correction_bias; the parent module must supply it")
        out = router(
            hidden_states,
            e_score_correction_bias.to(hidden_states.device),
        )
    elif "expert_bias" in fwd_params:
        if expert_bias is None and getattr(router, "use_expert_bias", True):
            raise ValueError(
                f"{type(router).__name__}.forward requires expert_bias; "
                "the parent module must supply it")
        out = router(hidden_states, expert_bias=(
            expert_bias.to(hidden_states.device) if expert_bias is not None else None))
    else:
        out = router(hidden_states)
    if isinstance(out, (tuple, list)):
        if len(out) >= 3:
            second = out[1]
            third = out[2]
            if not isinstance(second, torch.Tensor) or not isinstance(third, torch.Tensor):
                raise ValueError("router tuple entries 1 and 2 must be tensors")
            if second.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                indices = second
                weights = third
            elif third.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                weights = second
                indices = third
            else:
                raise ValueError("router tuple does not expose integer top-k indices")
            if weights.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("router top-k weights must be floating point")
            return indices.to(device=hidden_states.device), weights.to(
                device=hidden_states.device,
            )
        if len(out) >= 2:
            first, second = out[0], out[1]
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                if second.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                    return second.to(device=hidden_states.device), first.to(
                        device=hidden_states.device,
                    )
                if first.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                    return first.to(device=hidden_states.device), second.to(
                        device=hidden_states.device,
                    )
                raise ValueError("router tuple does not expose integer top-k indices")
    if not isinstance(out, torch.Tensor):
        raise ValueError(f"unsupported router output type {type(out)!r}")
    top_k = int(getattr(router, "top_k", 0) or 0)
    if top_k <= 0:
        raise ValueError("router returned logits only and exposes no top_k")
    probs = torch.softmax(out.float(), dim=-1)
    weights, indices = torch.topk(probs, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return indices.to(device=hidden_states.device), weights.to(device=hidden_states.device)


def packed_experts_forward_with_weights(
    experts_mod: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    input_quantize=None,
    intermediate_quantize=None,
) -> torch.Tensor:
    """Replay the packed-experts forward with explicit packed weights.

    This mirrors the Qwen3.5/DeepSeek-style fused expert module: one
    packed gate/up tensor, one packed down tensor, and router-provided
    token-to-expert assignments. Quantization hooks are placed at the
    same Linear boundaries as the dense measurement path.
    """
    num_experts = int(getattr(experts_mod, "num_experts", gate_up_weight.size(0)))
    act_fn = getattr(experts_mod, "act_fn", None)
    apply_gate = getattr(experts_mod, "_apply_gate", None)
    if act_fn is None and not callable(apply_gate):
        raise ValueError("packed experts module exposes no activation operation")
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index.to(torch.long), num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_t in expert_hit:
        expert_idx = int(expert_idx_t[0].item())
        if expert_idx >= num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        if input_quantize is not None:
            current_state = input_quantize(current_state)
        gate_up = F.linear(current_state, gate_up_weight[expert_idx])
        apply_gate = getattr(experts_mod, "_apply_gate", None)
        if callable(apply_gate):
            current_hidden_states = apply_gate(gate_up)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = act_fn(gate) * up
        if intermediate_quantize is not None:
            current_hidden_states = intermediate_quantize(current_hidden_states)
        current_hidden_states = F.linear(
            current_hidden_states,
            down_weight[expert_idx],
        )
        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0,
            token_idx,
            current_hidden_states.to(final_hidden_states.dtype),
        )
    return final_hidden_states


def derive_per_expert_activations(
    experts_mod: nn.Module,
    X: torch.Tensor,
    parent_mod: nn.Module | None,
    *,
    capture_down: bool = True,
    max_rows_per_expert: int | None = None,
    subsample_seed: int = 1234,
) -> dict:
    """Single source of truth for per-expert GPTQ activations.

    Routes the module-level expert input ``X`` ([*, hidden]) through the MoE
    block's own router and collects, per expert ``e``, exactly the tensors the
    per-expert GPTQ Hessian needs — identical to the routed forward in
    ``packed_experts_forward_with_weights``, but COLLECTING activations instead
    of producing the output. Shared by the cost path and the export render path
    so the routing + SwiGLU derivation lives in ONE place (no duplication).

    Returns a dict of length-E lists:
      - ``gate_up``: each [n_e, hidden]  — the routed input to ``gate_up_proj``.
      - ``down``:    each [n_e, inter]   — the post-SwiGLU input to ``down_proj``
                     (``_apply_gate(gate_up)`` when present, else ``act_fn(gate)*up``).
                     Empty list when ``capture_down=False``.
      - ``gate_weights``: each [n_e]     — the router weight per routed token.
      - ``row_counts``: list[int]        — routed tokens/expert BEFORE subsample
                     (use for the fail-on-insufficient-routed-rows gate).

    Subsampling (``max_rows_per_expert``) is deterministic (fixed ``subsample_seed``)
    so the render is reproducible. ``None`` keeps every routed row. Raises (never
    silently degrades) if the router / act_fn cannot be resolved.
    """
    gate_up_w = getattr(experts_mod, "gate_up_proj", None)
    if gate_up_w is None:
        raise ValueError("packed experts module lacks gate_up_proj")
    num_experts = int(getattr(experts_mod, "num_experts", gate_up_w.size(0)))
    act_fn = getattr(experts_mod, "act_fn", None)
    apply_gate = getattr(experts_mod, "_apply_gate", None)
    router = packed_experts_router(parent_mod)
    if router is None:
        raise ValueError("no router found for packed-experts module")
    Xf = X.reshape(-1, X.size(-1))
    dev, dt, hidden = Xf.device, Xf.dtype, Xf.size(-1)
    inter = gate_up_w.size(1) // 2
    with torch.no_grad():
        route_fn = getattr(parent_mod, "route_tokens_to_experts", None)
        if callable(route_fn):
            top_k_index, top_k_weights = route_fn(router(Xf))
        else:
            top_k_index, top_k_weights = packed_router_topk(
                router, Xf,
                e_score_correction_bias=getattr(
                    parent_mod, "e_score_correction_bias", None),
                expert_bias=getattr(parent_mod, "expert_bias", None),
            )
        top_k_index = top_k_index.to(torch.long)
        order, counts = routed_pair_order(top_k_index, num_experts)
    n_tokens = int(top_k_index.size(0))
    # ``order`` lists the routed (top-k slot, token) pairs grouped by expert,
    # each group in the row-major order ``torch.where(one_hot(...)[e])``
    # yields. Subsampling permutes and truncates a group exactly as the
    # per-expert loop did: same generator, same seed, same row order.
    kept = list(counts)
    if max_rows_per_expert is not None and any(n > max_rows_per_expert for n in counts):
        pieces = []
        start = 0
        for e, n in enumerate(counts):
            group = order[start:start + n]
            start += n
            if n > max_rows_per_expert:
                gen = torch.Generator(device=dev).manual_seed(subsample_seed + e)
                keep = torch.randperm(n, device=dev, generator=gen)[:max_rows_per_expert]
                group = group[keep]
                kept[e] = int(group.size(0))
            pieces.append(group)
        order = torch.cat(pieces)
    token_idx = order % n_tokens if n_tokens else order
    top_k_pos = order // n_tokens if n_tokens else order
    # One gather for every expert's rows and one for their router weights;
    # the per-expert outputs are views split from them, with no host sync.
    gate_up_list = list(Xf[token_idx].split(kept))
    gw_list = list(top_k_weights[token_idx, top_k_pos].split(kept))
    down_list: list[torch.Tensor] = []
    for e, n in enumerate(counts):
        if n == 0:
            # The empties keep the dtype the per-expert loop gave them.
            gate_up_list[e] = torch.empty(0, hidden, device=dev, dtype=dt)
            gw_list[e] = torch.empty(0, device=dev, dtype=dt)
            if capture_down:
                down_list.append(torch.empty(0, inter, device=dev, dtype=dt))
            continue
        if capture_down:
            with torch.no_grad():
                gate_up = F.linear(gate_up_list[e], gate_up_w[e])
                if callable(apply_gate):
                    di = apply_gate(gate_up)
                elif act_fn is not None:
                    g, u = gate_up.chunk(2, dim=-1)
                    di = act_fn(g) * u
                else:
                    raise ValueError(
                        "packed experts module exposes neither _apply_gate nor act_fn")
            down_list.append(di)
    return {"gate_up": gate_up_list, "down": down_list,
            "gate_weights": gw_list, "row_counts": counts}


def routed_pair_order(
    top_k_index: torch.Tensor, num_experts: int,
) -> tuple[torch.Tensor, list[int]]:
    """Group the routed (top-k slot, token) pairs by expert in one pass.

    ``top_k_index`` is ``[tokens, top_k]``. Returns ``(order, counts)``:
    ``order`` holds flat pair indices ``slot * tokens + token``, stably
    sorted by expert, so expert ``e``'s group is in the row-major (slot,
    token) order ``torch.where(F.one_hot(top_k_index).permute(2, 1, 0)[e])``
    yields. ``counts[e]`` is the group size. The only host sync is the one
    ``.tolist()`` that reads the counts; the per-expert ``torch.where`` this
    replaces synced once per expert (PQ #1931).
    """
    pair_expert = top_k_index.t().reshape(-1)
    invalid = ((pair_expert < 0) | (pair_expert >= num_experts)).sum().view(1)
    order = torch.argsort(pair_expert, stable=True)
    counts_t = torch.zeros(num_experts, dtype=torch.long, device=pair_expert.device)
    counts_t.index_add_(0, pair_expert.clamp(0, num_experts - 1),
                        torch.ones_like(pair_expert))
    host = torch.cat([counts_t, invalid]).tolist()
    if host[-1]:
        raise ValueError(
            f"router returned {host[-1]} expert indices outside [0, {num_experts})")
    return order, host[:-1]



def resolve_routed_expert_profile(model: nn.Module, profile=None):
    """Resolve one profile for every routed-expert decision in a call."""
    if profile is None:
        from prismaquant.model_profiles import profile_from_model

        profile = profile_from_model(model)
    if profile is None:
        raise RuntimeError(
            "routed-expert classification could not resolve a ModelProfile"
        )
    return profile


def _profile_call(profile, accessor: str, *args,
                  purpose: str = "classify", subject: str = "membership"):
    """Call one routed-expert profile accessor, refusing by name.

    The one owner (PQ #1302); ``expert_empirical_cost`` names its own purpose
    ("render") and subject ("layout") in the refusal text.
    """
    method = getattr(profile, accessor, None)
    if not callable(method):
        raise RuntimeError(
            f"profile {type(profile).__name__} cannot {purpose} routed experts: "
            f"missing callable {accessor}()"
        )
    try:
        return method(*args)
    except Exception as exc:
        raise RuntimeError(
            f"profile {type(profile).__name__} could not determine routed-"
            f"expert {subject} via {accessor}()"
        ) from exc


def _projection_names(profile) -> frozenset[str]:
    """Every expert projection spelling the profile declares.

    The three accessors name different namespaces: live unpacked modules,
    packed-checkpoint projections, and vLLM's canonical FusedMoE scheme
    projections.  Their union is the only accepted vocabulary.
    """
    def declared_names(raw, *, where: str) -> tuple[str, ...]:
        if isinstance(raw, (str, bytes)):
            raise RuntimeError(
                f"profile {type(profile).__name__} returned malformed "
                f"{where}: expected a collection of projection names"
            )
        try:
            values = tuple(raw)
        except TypeError as exc:
            raise RuntimeError(
                f"profile {type(profile).__name__} returned malformed "
                f"{where}: expected a collection of projection names"
            ) from exc
        if any(not isinstance(name, str) or not name for name in values):
            raise RuntimeError(
                f"profile {type(profile).__name__} returned malformed "
                f"{where}: {values!r}"
            )
        return values

    packed = _profile_call(profile, "packed_expert_param_names")
    unpacked = _profile_call(profile, "unpacked_expert_projection_names")
    try:
        packed_names = declared_names(
            packed, where="packed_expert_param_names()"
        )
        names = set(packed_names)
        names.update(declared_names(
            unpacked, where="unpacked_expert_projection_names()"
        ))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"profile {type(profile).__name__} returned malformed routed-"
            "expert projection declarations"
        ) from exc
    for packed_name in packed_names:
        for accessor in (
            "packed_expert_projection_names",
            "vllm_fused_moe_scheme_projection_names",
        ):
            declared = _profile_call(profile, accessor, packed_name)
            names.update(declared_names(
                declared, where=f"{accessor}({packed_name!r})"
            ))
    if not names or any(not name or "." in name for name in names):
        raise RuntimeError(
            f"profile {type(profile).__name__} returned an invalid routed-"
            f"expert projection vocabulary: {sorted(names)!r}"
        )
    return frozenset(names)


@dataclass(frozen=True)
class RoutedExpertMatch:
    """One qname classified by a profile, independent of tensor rank."""

    qname: str
    profile_qname: str
    group_key: str
    projection_name: str
    regex_declared: bool


@dataclass(frozen=True)
class UnpackedExpertLinear:
    """A profile-declared routed expert represented by one ``nn.Linear``."""

    qname: str
    module: nn.Linear
    group_key: str
    unit_qname: str
    expert_id: int
    projection_name: str


class ProfileRoutedExpertClassifier:
    """Strict adapter around the existing ``ModelProfile`` accessors."""

    def __init__(self, profile):
        self.profile = profile
        self.projection_names = _projection_names(profile)
        raw_regex = _profile_call(profile, "per_expert_moe_regex")
        self.per_expert_pattern: re.Pattern[str] | None = None
        if raw_regex is not None:
            if not isinstance(raw_regex, str) or not raw_regex:
                raise RuntimeError(
                    f"profile {type(profile).__name__} returned a malformed "
                    "per_expert_moe_regex()"
                )
            body = raw_regex.removeprefix("re:")
            try:
                self.per_expert_pattern = re.compile(body)
            except re.error as exc:
                raise RuntimeError(
                    f"profile {type(profile).__name__} returned an invalid "
                    "per_expert_moe_regex()"
                ) from exc

    def _candidate_names(self, qname: str) -> tuple[str, ...]:
        recipe = _profile_call(
            self.profile, "live_to_recipe_name", str(qname)
        )
        if not isinstance(recipe, str) or not recipe:
            raise RuntimeError(
                f"profile {type(self.profile).__name__} returned malformed "
                f"live_to_recipe_name({qname!r})={recipe!r}"
            )
        vllm = _profile_call(
            self.profile, "to_vllm_internal_name", recipe
        )
        if not isinstance(vllm, str) or not vllm:
            raise RuntimeError(
                f"profile {type(self.profile).__name__} returned malformed "
                f"to_vllm_internal_name({recipe!r})={vllm!r}"
            )
        return tuple(dict.fromkeys((str(qname), recipe, vllm)))

    def classify(self, qname: str) -> RoutedExpertMatch | None:
        candidates = self._candidate_names(qname)
        grouped: list[tuple[str, str]] = []
        regex_declared = False
        for candidate in candidates:
            if (
                self.per_expert_pattern is not None
                and self.per_expert_pattern.fullmatch(candidate) is not None
            ):
                regex_declared = True
            group = _profile_call(
                self.profile, "packed_expert_format_group", candidate
            )
            if group is None:
                continue
            if not isinstance(group, str) or not group:
                raise RuntimeError(
                    f"profile {type(self.profile).__name__} returned malformed "
                    f"packed_expert_format_group({candidate!r})={group!r}"
                )
            grouped.append((candidate, group))

        if not grouped:
            if regex_declared:
                raise RuntimeError(
                    f"profile {type(self.profile).__name__} declares {qname!r} "
                    "as a per-expert Linear but cannot assign it a routed-"
                    "expert serving-format group"
                )
            return None

        group_keys = {group for _candidate, group in grouped}
        if len(group_keys) != 1:
            # `grouped` holds groups derived from the THREE spellings of ONE
            # Linear (live / recipe / vLLM-internal), and
            # `packed_expert_format_group` builds its key as
            # f"{parent}::__packed_format__:{projections}" from whichever
            # spelling it was handed. So on any model whose namespaces differ
            # by a prefix -- a multimodal wrapper contributes
            # `model.language_model.` on the checkpoint side and
            # `language_model.model.` on the vLLM side, against a bare
            # `model.` recipe name -- the raw keys ALWAYS differ and this gate
            # fires on a naming artifact rather than a declaration conflict.
            # Compare what actually carries meaning instead: the projection
            # grouping, and the structural position from `layers.` onward.
            # Those still catch a profile that groups one Linear two different
            # ways, or that points two spellings at different layers.
            def _meaning(key: str) -> tuple[str, str]:
                parent, _, grouping = key.partition("::")
                idx = parent.find("layers.")
                return (parent[idx:] if idx != -1 else parent), grouping

            if len({_meaning(group) for _candidate, group in grouped}) != 1:
                raise RuntimeError(
                    f"profile {type(self.profile).__name__} assigns conflicting "
                    f"routed-expert groups to {qname!r}: {sorted(group_keys)!r}"
                )
        profile_qname = grouped[0][0]
        projection = profile_qname.rsplit(".", 1)[-1]
        if projection not in self.projection_names:
            raise RuntimeError(
                f"profile {type(self.profile).__name__} grouped {qname!r} as "
                f"a routed expert but projection {projection!r} is absent "
                "from its expert projection accessors"
            )
        return RoutedExpertMatch(
            qname=str(qname),
            profile_qname=profile_qname,
            # Deterministic: the first spelling that produced a group, not an
            # arbitrary element of a set. Every member of one experts module
            # walks the same candidate order, so siblings agree and union-find
            # still fuses them.
            group_key=grouped[0][1],
            projection_name=projection,
            regex_declared=regex_declared,
        )


def profile_declared_unpacked_expert_linears(
    model: nn.Module,
    profile=None,
) -> list[UnpackedExpertLinear]:
    """Return every routed expert represented by a per-expert Linear.

    Numeric expert-id parsing happens only *after* the profile classified the
    qname.  An unfamiliar representation is therefore a declaration error,
    never a name heuristic or a quiet empty result.
    """
    profile = resolve_routed_expert_profile(model, profile)
    classifier = ProfileRoutedExpertClassifier(profile)
    out: list[UnpackedExpertLinear] = []
    for qname, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        match = classifier.classify(qname)
        if match is None:
            continue
        parts = str(qname).split(".")
        if len(parts) < 3 or not parts[-2].isdigit():
            raise RuntimeError(
                f"profile {type(profile).__name__} classified Linear "
                f"{qname!r} as routed expert state but cannot determine its "
                "expert id from the declared per-expert representation"
            )
        if (
            classifier.per_expert_pattern is not None
            and not match.regex_declared
        ):
            raise RuntimeError(
                f"profile {type(profile).__name__} grouped Linear {qname!r} "
                "as routed expert state but its per_expert_moe_regex() does "
                "not recognize any mapped qname"
            )
        out.append(UnpackedExpertLinear(
            qname=str(qname),
            module=module,
            group_key=match.group_key,
            unit_qname=".".join(parts[:-2]),
            expert_id=int(parts[-2]),
            projection_name=parts[-1],
        ))
    return sorted(out, key=lambda member: member.qname)


def packed_activation_input_kind(param_name: str) -> str:
    """Input names owned by the existing packed activation derivation."""
    kinds = {"gate_up_proj": "gate_up", "down_proj": "down"}
    if param_name not in kinds:
        raise RuntimeError(f"packed activation derivation does not support {param_name!r}")
    return kinds[param_name]


def declared_shared_capture_groups(unit_shapes, profile):
    """Plan input ownership from profile declarations, for checked admission.

    This is an expectation, not evidence of the live representation. A
    collector using the resulting memory bound must compare its actual groups
    with these groups before forwarding. In particular, declared experts that
    are ordinary unpacked Linears cannot claim packed gate/up sharing.
    """
    classifier = ProfileRoutedExpertClassifier(profile)
    groups = {}
    for name, shape in sorted(unit_shapes.items()):
        if (not isinstance(name, str) or not name or len(shape) != 2 or
                any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError('capture input grouping requires named positive 2-D unit shapes')
        match = classifier.classify(name)
        key = ('dense', name)
        if match is not None:
            parts = name.rsplit('.', 2)
            if len(parts) != 3 or not parts[1].isdigit() or not match.regex_declared:
                raise RuntimeError(f'capture unit has no declared per-expert representation: {name}')
            parent = profile.packed_expert_parent_for_projection(match.projection_name)
            key = ('packed', parts[0], int(parts[1]), packed_activation_input_kind(parent))
        groups.setdefault(key, []).append(name)
    for members in groups.values():
        if len({unit_shapes[name][1] for name in members}) != 1:
            raise RuntimeError(f'shared capture inputs have different widths: {members}')
    return {members[0]: members for members in sorted(groups.values())}


@dataclass(frozen=True)
class PackedExpertProjection:
    """One profile-declared 2-D view of a live packed expert parameter.

    ``qname`` is the virtual per-expert module spelling in the live namespace,
    not a Tessera serving-group name. The producer still owns the projection
    from source units to its executed groups and wire containers.
    """

    qname: str
    packed_qname: str
    module_qname: str
    module: nn.Module
    param_name: str
    expert_id: int
    projection_name: str
    weight: torch.Tensor

    @property
    def parameter(self) -> nn.Parameter:
        """The current packed leaf; streaming may replace it between leases."""
        value = getattr(self.module, self.param_name, None)
        if not isinstance(value, nn.Parameter) or value.ndim != 3:
            raise RuntimeError(f"packed projection {self.qname} has no live packed leaf")
        return value

    @property
    def output_slice(self) -> slice:
        """This projection's rows inside its expert's packed Linear output."""
        parent = self.parameter
        base = self.weight._base if self.weight._is_view() else self.weight
        if (base is not parent or self.weight.ndim != 2
                or self.weight.stride() != parent.stride()[1:]
                or self.weight.shape[1] != parent.shape[2]
                or parent.stride(1) <= 0
                or not 0 <= self.expert_id < parent.shape[0]):
            raise RuntimeError(f"packed projection {self.qname} is a stale or invalid source view")
        offset = (self.weight.storage_offset() - parent.storage_offset()
                  - self.expert_id * parent.stride(0))
        rows, remainder = divmod(offset, parent.stride(1))
        stop = rows + self.weight.shape[0]
        if remainder or rows < 0 or stop > parent.shape[1]:
            raise RuntimeError(f"packed projection {self.qname} has invalid source row geometry")
        return slice(rows, stop)

    def gradient_view(self, gradient: torch.Tensor) -> torch.Tensor:
        if gradient.shape != self.parameter.shape:
            raise RuntimeError(f"packed projection gradient shape differs for {self.qname}")
        return gradient[self.expert_id, self.output_slice, :]


def refresh_packed_expert_projections(members, profile) -> list[PackedExpertProjection]:
    """Rebind existing logical views after a streamed install or unload.

    The profile/export split is evaluated once per physical parameter. No
    checkpoint tensor or rendered tensor is copied or retained separately.
    """
    from .export_native_compressed import _split_packed_expert_tensor

    splits = {}
    refreshed = []
    for member in members:
        key = (id(member.module), member.param_name)
        if key not in splits:
            splits[key] = dict(_split_packed_expert_tensor(
                member.parameter, member.param_name, profile))
        try:
            weight = splits[key][member.projection_name][member.expert_id]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"packed projection topology changed for {member.qname}") from exc
        if weight.shape != member.weight.shape:
            raise RuntimeError(f"packed projection shape changed for {member.qname}")
        refreshed.append(replace(member, weight=weight))
    return refreshed


def profile_declared_packed_expert_projections(
    model: nn.Module, profile=None,
) -> list[PackedExpertProjection]:
    """Split declared packed tensors using the native exporter's exact views.

    This shares source topology, not an encoder or a serving grammar. Views
    retain the live weight device/dtype and do not allocate a second cache.
    """
    from .export_native_compressed import _split_packed_expert_tensor

    profile = resolve_routed_expert_profile(model, profile)
    targets = set(profile_declared_routed_expert_targets(model, profile))
    result = []
    for packed_qname, packed, module_qname, module in enumerate_packed_experts(
        model, targets, profile,
    ):
        param_name = packed_qname.rsplit(".", 1)[-1]
        for projection, weight in _split_packed_expert_tensor(packed, param_name, profile):
            for expert_id in range(int(weight.shape[0])):
                result.append(PackedExpertProjection(
                    qname=f"{module_qname}.{expert_id}.{projection}",
                    packed_qname=packed_qname, module_qname=module_qname,
                    module=module, param_name=param_name,
                    expert_id=expert_id, projection_name=projection,
                    weight=weight[expert_id],
                ))
    names = [member.qname for member in result]
    if len(set(names)) != len(names):
        raise RuntimeError("profile declares duplicate packed expert projection qnames")
    return sorted(result, key=lambda member: member.qname)


def profile_declared_routed_expert_targets(
    model: nn.Module,
    profile=None,
) -> list[str]:
    """All profile-declared routed targets, packed or unpacked."""
    profile = resolve_routed_expert_profile(model, profile)
    classifier = ProfileRoutedExpertClassifier(profile)
    out: set[str] = set()
    for qname, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if classifier.classify(qname) is not None:
                out.add(str(qname))
            continue
        for attr, _param in module.named_parameters(recurse=False):
            full = f"{qname}.{attr}" if qname else str(attr)
            if classifier.classify(full) is not None:
                out.add(full)
    return sorted(out)


__all__ = [
    "enumerate_packed_experts",
    "packed_experts_parent_module",
    "packed_experts_router",
    "packed_router_topk",
    "packed_experts_forward_with_weights",
    "derive_per_expert_activations",
    "routed_pair_order",
    "ProfileRoutedExpertClassifier",
    "RoutedExpertMatch",
    "UnpackedExpertLinear",
    "PackedExpertProjection",
    "profile_declared_packed_expert_projections",
    "refresh_packed_expert_projections",
    "profile_declared_routed_expert_targets",
    "profile_declared_unpacked_expert_linears",
    "resolve_routed_expert_profile",
]


def packed_expert_col_weights(col_weights, members_by_target, profile):
    """Per-expert imatrix vectors -> the ``(E, 1, in)`` stack entry each packed
    target needs, returned as a NEW mapping (the per-expert entries survive for
    the render identity, which is keyed by them).

    It lived in the codebook exporter until 2026-09-02 (as
    ``export_nvfp4_cb_streaming._packed_expert_col_weights``) and moved here
    when that lane was retired: the rule is about how a PACKED EXPERT stack
    pools its members' imatrix vectors, which is a routed-expert fact and not
    a property of any one container.

    A FUSED parent (``gate_up_proj`` = gate then up) has ONE input, so its two
    projections' vectors are two samples of the same per-column mean-square and
    are pooled by averaging. They are not identical in practice only because
    the probe caches each Linear's inputs separately under a row limit
    (DSv4-Flash layer 0: max |gate-up| ~ 0.3 against a vector norm ~ 4.6).
    Weighting one projection by the other's sample would be the actual error;
    per-row vectors cannot be expressed here at all, since the pack broadcasts
    ``(E, 1, in)`` across the whole stack."""
    out = dict(col_weights)
    for packed_qname, member_qnames in members_by_target.items():
        if packed_qname in out:
            continue
        projections = tuple(dict.fromkeys(
            projection for projection, _expert_id in member_qnames
        ))
        experts = sorted({e for _p, e in member_qnames})
        rows = []
        for e in experts:
            vecs = []
            for proj in projections:
                q = member_qnames[(proj, e)]
                if q not in col_weights:
                    raise ValueError(
                        f"{packed_qname}: CB stack member {q!r} has no "
                        "col_weights entry (no silent RTN)")
                vecs.append(torch.as_tensor(col_weights[q])
                            .reshape(-1).to(torch.float32))
            widths = {int(v.numel()) for v in vecs}
            if len(widths) != 1:
                raise ValueError(
                    f"{packed_qname}: expert {e} imatrix widths disagree "
                    f"across the fused projections {projections}: {widths}")
            rows.append(torch.stack(vecs).mean(dim=0) if len(vecs) > 1
                        else vecs[0])
        out[packed_qname] = torch.stack(rows).unsqueeze(1).contiguous()
    return out
