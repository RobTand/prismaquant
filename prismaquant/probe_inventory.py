"""Profile-declared probe unit recognition and inventory policy."""
from __future__ import annotations

from dataclasses import dataclass
import re

import torch.nn as nn


def packed_expert_param_name_set(profile=None) -> set[str]:
    if profile is None:
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    query = getattr(profile, "packed_expert_param_names", None)
    return set(query()) if query is not None else set()


def packed_expert_parent_for_projection(profile, projection_name: str) -> str | None:
    if profile is None:
        from .model_profiles import DefaultProfile
        profile = DefaultProfile()
    query = getattr(profile, "packed_expert_parent_for_projection", None)
    return query(projection_name) if query is not None else None


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
    names = packed_expert_param_name_set(profile)
    query = getattr(profile, "packed_expert_projection_names", None)
    if query is not None:
        for name in tuple(names):
            names.update(query(name))
    return tuple(sorted(names))


# Router/gate Linears carry routing logits, not quantizable weights.
BASE_LINEAR_EXCLUDE = (
    r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|block_sparse_moe\.gate$)"
)



@dataclass(frozen=True)
class ProbeUnit:
    module_name: str
    kind: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class ExcludedProbeLinear:
    module_name: str
    reason_code: str


@dataclass(frozen=True)
class ProbeInventory:
    priced_units: tuple[ProbeUnit, ...]
    excluded_linears: tuple[ExcludedProbeLinear, ...]

    @property
    def linear_names(self) -> tuple[str, ...]:
        return tuple(unit.module_name for unit in self.priced_units
                     if unit.kind in ('linear', 'grouped_linear'))


def linear_exclusion_for_profile(profile) -> str:
    query = getattr(profile, 'probe_linear_exclude_extra', None)
    extra = str(query() or '') if query is not None else ''
    return f'(?:{BASE_LINEAR_EXCLUDE}|{extra})' if extra else BASE_LINEAR_EXCLUDE


def build_probe_inventory(model: nn.Module, profile, *, linear_exclude=None) -> ProbeInventory:
    exclusion = re.compile(linear_exclusion_for_profile(profile)
                           if linear_exclude is None else linear_exclude)
    grouped_query = getattr(profile, 'probe_grouped_module_class_names', None)
    grouped = set(grouped_query()) if grouped_query is not None else set()
    priced, excluded = [], []
    for name, module in model.named_modules():
        if profile.should_probe_linear(name, module):
            if exclusion.search(name):
                excluded.append(ExcludedProbeLinear(name, 'probe_linear_excluded'))
            else:
                kind = 'grouped_linear' if type(module).__name__ in grouped else 'linear'
                priced.append(ProbeUnit(name, kind, (f'{name}.weight' if name else 'weight',)))
        elif is_packed_experts_module(module, profile):
            priced.append(ProbeUnit(name, 'packed_experts', tuple(
                f'{name}.{parameter}' if name else parameter
                for parameter in packed_experts_param_names(module, profile))))
        elif isinstance(module, nn.Linear):
            excluded.append(ExcludedProbeLinear(name, 'owner_not_probe_priceable'))
    return ProbeInventory(tuple(priced), tuple(excluded))
