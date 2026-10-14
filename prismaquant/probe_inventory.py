"""Profile-declared probe unit recognition and inventory policy."""
from __future__ import annotations

import torch.nn as nn


def packed_expert_param_name_set(profile=None) -> set[str]:
    if profile is None:
        try:
            from .model_profiles import DefaultProfile
            profile = DefaultProfile()
        except Exception:
            profile = None
    if profile is not None:
        try:
            return set(profile.packed_expert_param_names())
        except Exception:
            pass
    return set()

def packed_expert_parent_for_projection(profile, projection_name: str) -> str | None:
    if profile is None:
        try:
            from .model_profiles import DefaultProfile
            profile = DefaultProfile()
        except Exception:
            profile = None
    if profile is not None:
        try:
            return profile.packed_expert_parent_for_projection(projection_name)
        except Exception:
            pass
    return None

def is_packed_experts_module(module: nn.Module, profile=None) -> bool:
    """A module qualifies as a packed-experts container iff (a) its
    class name contains "Experts" (case-insensitive), and (b) it owns
    at least one profile-declared 3D packed expert parameter.

    The class-name check excludes other modules that happen to own 3D
    parameters — most importantly Conv1d in linear-attention paths,
    whose `weight` is shape `[out, in, kernel]`. The param-name check
    is a second safety net against unusual modules with unrelated 3D
    state.
    """
    names = packed_expert_param_name_set(profile)
    cls_name = type(module).__name__.lower()
    if "expert" not in cls_name:
        return False
    for n, p in module.named_parameters(recurse=False):
        if (isinstance(p, nn.Parameter)
                and p.dim() == 3
                and n in names):
            return True
    return False

def packed_experts_param_names(module: nn.Module, profile=None) -> list[str]:
    """Return the attribute names of all 3D packed parameters on
    `module`, restricted to the known MoE expert names. Order is stable
    across Python runs."""
    names_allowed = packed_expert_param_name_set(profile)
    names = []
    for n, p in module.named_parameters(recurse=False):
        if (isinstance(p, nn.Parameter)
                and p.dim() == 3
                and n in names_allowed):
            names.append(n)
    return sorted(names)

def packed_expert_projection_candidate_names(profile=None) -> tuple[str, ...]:
    names = set(packed_expert_param_name_set(profile))
    if profile is not None:
        for name in list(names):
            try:
                names.update(profile.packed_expert_projection_names(name))
            except Exception:
                pass
    return tuple(sorted(names))


# Router/gate Linears carry routing logits, not quantizable weights.
BASE_LINEAR_EXCLUDE = (
    r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|block_sparse_moe\.gate$)"
)


