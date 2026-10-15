"""The admitted run's assignment expansion and exact tensor accounting."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from . import format_registry as fr
from .allocator_candidates import (
    expand_fused_sibling_assignment, expand_packed_group_assignment,
    serialized_candidate_payload,
)
from .allocator_solver import _shape_from_stats, promote_serving_units
from .footprint import NVFP4_WEIGHT_ONLY_STATS_KEY, nvfp4_global_sidecar_bytes

if TYPE_CHECKING:
    from .model_profiles import ModelProfile


@dataclass(frozen=True)
class AggregationPolicy:
    packed: bool
    fused: bool


@dataclass(frozen=True)
class AssignmentExpansion:
    before_promotion: dict[str, str]
    assignment: dict[str, str]


class AssignmentPayload(TypedDict):
    bits_total: float
    quantizable_params: int
    bits_per_param: float
    missing_stats_names: list[str]


class AssignmentSpace:
    """Existing assignment policy bound to one admitted candidate/stat view.

    The run retains custody of the input maps. Per-name lookup skips non-dict
    entries; complete scope views retain literal overwrite precedence.
    Whole-artifact and runtime prices remain separate contracts.
    """

    def __init__(
        self, *, stats: dict, fixed_stats: dict, accounting_stats: dict,
        fixed_formats: Mapping[str, str], aggregation: AggregationPolicy,
        profile: ModelProfile | None, format_rank: dict[str, int],
        legal_formats: dict[str, set[str]],
    ) -> None:
        self._stats = stats
        self._fixed_stats = fixed_stats
        self._accounting_stats = accounting_stats
        self._fixed_formats = fixed_formats
        self._aggregation = aggregation
        self._profile = profile
        self._format_rank = format_rank
        self._legal_formats = legal_formats

    def scope_stats(self) -> dict:
        return {**self._accounting_stats, **self._fixed_stats, **self._stats}

    def unit_members(self, unit: str, fmt: str) -> dict[str, str]:
        """Members of one priced option, before fixed choices or promotion."""
        return expand_fused_sibling_assignment(
            expand_packed_group_assignment({unit: fmt}, self._stats), self._stats)

    def stats_for(self, name: str) -> dict | None:
        entry = self._stats.get(name)
        if isinstance(entry, dict):
            return entry
        fixed_entry = self._fixed_stats.get(name)
        if isinstance(fixed_entry, dict):
            return fixed_entry
        original_entry = self._accounting_stats.get(name)
        if isinstance(original_entry, dict):
            return original_entry
        return None

    def expand(
        self,
        assignment: dict[str, str],
        *,
        include_auxiliary: bool = True,
    ) -> AssignmentExpansion:
        """Expand DP super-items into the per-Linear seed-assignment shape.

        The allocator can solve over packed-serving-group and fused-sibling
        super-items. The KL probe's seed path wants ordinary module qnames;
        it already handles legality, pinning, and fused coherence, but giving
        it expanded names preserves the intended frontier point instead of
        making the super-item markers look like unknown entries.
        """
        expanded = dict(assignment)
        if self._aggregation.packed:
            expanded = expand_packed_group_assignment(expanded, self._stats)
        if self._aggregation.fused:
            expanded = expand_fused_sibling_assignment(expanded, self._stats)
        if include_auxiliary:
            expanded.update(self._fixed_formats)
        return AssignmentExpansion(
            before_promotion=expanded,
            assignment=promote_serving_units(
                expanded,
                self._format_rank,
                profile=self._profile,
                legal_formats=self._legal_formats,
            ),
        )

    def payload(
        self,
        assignment: Mapping[str, str],
        *,
        require_all_stats: bool,
    ) -> AssignmentPayload:
        """Exact assignment-scope tensor payload.

        Candidate memory remains the additive DP proposal cost.  This function
        is the exact filter/reporting path: NVFP4 global scale tensors are
        included.
        """
        total = 0.0
        params = 0
        missing: list[str] = []
        for name, fmt in assignment.items():
            entry = self.stats_for(name)
            if not isinstance(entry, dict):
                missing.append(name)
                continue
            params += int(entry.get("n_params", 0) or 0)
            # Super-items (packed groups, fused siblings) carry exact
            # per-format byte sums; their stats entries have no single
            # (out, in) shape, so the shape fallback is only for plain rows.
            memory_map = entry.get("_memory_bytes_by_format")
            if isinstance(memory_map, dict) and fmt in memory_map:
                total += 8.0 * memory_map[fmt]
                if fmt == "NVFP4":
                    total += 8.0 * nvfp4_global_sidecar_bytes(
                        name,
                        _shape_from_stats(entry),
                        weight_only=bool(
                            entry.get(NVFP4_WEIGHT_ONLY_STATS_KEY, False)
                        ),
                    )
                continue
            shape = _shape_from_stats(entry)
            payload_bytes, _identity, _sidecar_identity = serialized_candidate_payload(
                fr.get_format(fmt),
                shape,
                qname=name,
            )
            total += 8.0 * payload_bytes
            if fmt == "NVFP4":
                total += 8.0 * nvfp4_global_sidecar_bytes(
                    name,
                    shape,
                    weight_only=bool(
                        entry.get(NVFP4_WEIGHT_ONLY_STATS_KEY, False)
                    ),
                )
        if missing and require_all_stats:
            raise AssertionError(
                "exact assignment payload has no shape/stats for "
                f"{len(missing)} tensor(s): {sorted(missing)[:12]}"
            )
        return {
            "bits_total": float(total),
            "quantizable_params": int(params),
            "bits_per_param": float(total) / max(params, 1),
            "missing_stats_names": sorted(missing),
        }

    def bits_total(self, assignment: dict[str, str]) -> float:
        return float(self.payload(
            assignment,
            require_all_stats=True,
        )["bits_total"])
