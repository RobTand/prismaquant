"""Build score-specific menus for the production scalar solver."""
from __future__ import annotations

import math
from copy import deepcopy

from . import format_registry as fr
from .allocator import SOLVER_CONTRACT, solve_scalar
from .allocator_candidates import (
    aggregate_fused_siblings,
    aggregate_packed_serving_groups,
    build_candidates,
    reduce_continuous_menu,
    require_candidate_coverage,
    serving_groups_by_key,
)
from .allocator_solver import compute_assignment_predicted_dloss, legal_formats_from_candidates
from .assignment_space import AggregationPolicy, AssignmentSpace
from .model_profiles import DefaultProfile
from .priced_cell import CostPricingPolicy, resolve_cost_row


class ProductionScalarProblem:
    """Bind one population, format order and charge scope for both score arms.

    Supply JSON data for per-Linear stats and costs before menu reduction.
    Declare format names in promotion rank order. Use scalar predicted loss
    rows for options that change scores. Extra charges count in the body cap.
    Declare each extra sidecar once, with the options that use it.
    """

    solver_contract = SOLVER_CONTRACT

    def __init__(self, stats, costs, formats, *, currency, profile=None,
                 bit_precision=0.0001, fixed_bytes=0, shared_sidecars=None,
                 source_manifest=None, target_profile=None, context_by_unit=None):
        sidecars = shared_sidecars or {}
        if not currency or not stats:
            raise ValueError('Declare the scalar population and currency.')
        if not math.isfinite(bit_precision) or bit_precision <= 0:
            raise ValueError('Use a positive finite bit precision.')
        if any(type(size) is not int or size < 0
               for size in (fixed_bytes, *(size for size, _ in sidecars.values()))):
            raise ValueError('Use nonnegative integer byte charges.')
        self.currency = currency
        self._stats = deepcopy(stats)
        self._costs = deepcopy(costs)
        self._specs = {fmt: fr.get_format(fmt) for fmt in formats}
        self._rank = {fmt: i for i, fmt in enumerate(self._specs)}
        self._profile = DefaultProfile() if profile is None else profile
        self._precision = bit_precision
        self._fixed_bytes = fixed_bytes
        self._sidecars = {key: (size, tuple(sorted(set(users))))
                          for key, (size, users) in sidecars.items()}
        self._policy = CostPricingPolicy.from_environment()
        self._admission = dict(source_manifest=deepcopy(source_manifest), target_profile=target_profile,
                               context_by_unit=None if context_by_unit is None else dict(context_by_unit))
        _, _, candidates = self._build(self._costs)
        self.legal_formats = legal_formats_from_candidates(candidates)
        self._groups = serving_groups_by_key(self._stats, self._profile)
        self._deferred = frozenset(n for members in self._groups.values() for n in members)
        self._candidate_bytes = {n: {c.fmt: int(c.memory_bytes) for c in menu}
                                 for n, menu in candidates.items()}
        if any(fmt not in self.legal_formats.get(n, ())
               for _, users in self._sidecars.values() for n, fmt in users):
            raise ValueError('A sidecar user is not an admitted option.')

    def _build(self, costs):
        stats = deepcopy(self._stats)
        build = build_candidates(
            stats, costs, list(self._specs.values()), bit_precision=self._precision,
            defer_menu_reduction=stats.keys(), pricing_policy=self._policy, **self._admission,
        )
        require_candidate_coverage(stats, build.candidates)
        return stats, build.cost_rows, build.candidates

    def budget_scope(self):
        return dict(scope='expanded_body_and_declared_charges', fixed_bytes=self._fixed_bytes,
                    shared_sidecars=self._sidecars)

    def evidence_inputs(self):
        return dict(costs=self._costs, stats=self._stats, units=list(self._stats),
                    formats=list(self._specs), candidate_bytes=self._candidate_bytes,
                    groups=[(kind, key, sorted(members))
                            for (kind, key), members in sorted(self._groups.items())],
                    profile=type(self._profile).__qualname__,
                    source_manifest=self._admission['source_manifest'],
                    target_profile=self._admission['target_profile'],
                    serving_context={n: ctx.as_dict() for n, ctx in
                                     (self._admission['context_by_unit'] or {}).items()},
                    bit_precision=self._precision, ucb_z=self._policy.ucb_z)

    def solve(self, byte_budget, scores=None):
        solve, _ = self.prepare(scores)
        return solve(byte_budget)

    def prepare(self, scores=None):
        """Build new frontiers and retain prices for expanded selections."""
        costs = {n: dict(menu) for n, menu in self._costs.items()}
        for name, menu in (scores or {}).items():
            for fmt, loss in menu.items():
                if fmt not in self.legal_formats.get(name, ()):
                    raise ValueError('Change scores only for admitted options.')
                row, key = resolve_cost_row(costs[name], fmt)
                if row is None or set(row) != {'predicted_dloss'}:
                    raise ValueError('Change scores only for scalar cost rows.')
                if not math.isfinite(loss) or loss < 0:
                    raise ValueError('Use finite nonnegative loss scores.')
                costs[name][key] = {'predicted_dloss': loss}
        stats, costs, candidates = self._build(costs)
        if legal_formats_from_candidates(candidates) != self.legal_formats:
            raise ValueError('The score change changed the admitted menus.')
        linear_candidates = candidates
        candidates = reduce_continuous_menu(
            candidates, stats, bit_precision=self._precision, defer_reduction=self._deferred)
        options = dict(profile=self._profile, pricing_policy=self._policy)
        specs = list(self._specs.values())
        grouped, costs, candidates = aggregate_packed_serving_groups(stats, costs, specs, candidates, **options)
        grouped, costs, candidates = aggregate_fused_siblings(grouped, costs, specs, candidates, **options)
        candidates = reduce_continuous_menu(candidates, grouped, bit_precision=self._precision)
        require_candidate_coverage(grouped, candidates)
        space = AssignmentSpace(
            stats=grouped, accounting_stats=stats, fixed_stats={}, fixed_formats={},
            aggregation=AggregationPolicy(True, True), profile=self._profile,
            format_rank=self._rank, legal_formats=self.legal_formats,
        )
        params = sum(grouped[n]['n_params'] for n in candidates)

        def solve(byte_budget):
            return solve_scalar(
                grouped, candidates, byte_budget * 8 / params, self._specs, self._rank,
                self._precision, space=space, profile=self._profile, tolerance=0,
                diagnostics={}, byte_budget=byte_budget, fixed_bytes=self._fixed_bytes,
                shared_sidecars=self._sidecars.values(),
            )

        def loss(assignment):
            return compute_assignment_predicted_dloss(assignment, linear_candidates)

        return solve, loss
