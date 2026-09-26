"""Opt-in, research-only atomic proposals from a diagnostic joint AURA panel.

An uninvoked expert has UNKNOWN full-population price.  Its contribution to
the observed panel is zero only while embedded in its entire serving stack;
neither it nor that zero is an independent solver option.  This module does
not produce an exportable cost table or authorize an export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

from . import format_registry as fr
from .allocator_candidates import (
    aggregate_fused_siblings, aggregate_packed_serving_groups, build_candidates,
    expand_fused_sibling_assignment,
    expand_packed_group_assignment, fused_sibling_group_members,
    packed_serving_group_members, serialized_candidate_payload,
)
from .allocator_solver import solve_with_promotion
from .cluster_campaign import _atomic_write_new_bytes
from .joint_aura import assignment_probe_summary
from .tessera_formats import (family_q256_bounds, get_tessera_family,
                              parse_tessera_format_name, realisable_rungs)

SCHEMA = 'prismaquant.tessera_sampled_stack_proposal.v1'
PRIMARY_FAMILIES = frozenset({'TESSERA_E4M3_K1', 'TESSERA_BF16_K1'})


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _domain(families, costs, stats):
    """Full producer-legal grammar, independent of source and runtime masks."""
    _require(isinstance(families, (list, tuple)) and len(families) == len(set(families))
             and families, 'explicit unique full legal family roster required')
    _require(PRIMARY_FAMILIES <= set(families),
             'full legal family roster must retain primary E4M3_K1 and BF16_K1')
    shapes = {name: (int(s['out_features']), int(s['in_features']))
              for name, s in stats.items()}
    result = {}
    for family_name in families:
        family = get_tessera_family(family_name)
        _require(family.name == family_name, f'{family_name}: invalid producer family name')
        lo, hi = family_q256_bounds(family)
        # realisable_rungs is the producer's exact q256 resolution.  Its
        # promise holds for complete 256-column superblocks, regardless of
        # source-size caps or whether a runtime serves the resulting wire.
        rates = realisable_rungs(family)
        _require((rates.start, rates.stop - 1, rates.step) == (lo, hi, 1),
                 f'{family_name}: producer rate grammar differs from bounds')
        by_shape = {}
        for shape in sorted(set(shapes.values())):
            for rate in (lo, hi):
                fr.get_format(family.format_name(rate)).memory_bytes_for_shape(shape)
            by_shape[f'{shape[0]}x{shape[1]}'] = {
                'ranges_inclusive_q256': [[lo, hi]], 'count': len(rates),
                'price_outside_measured': None,
                'scope': 'producer_legal_not_source_ceiling_or_runtime_admission'}
        result[family_name] = {'producer_grammar_q256': [lo, hi],
            'producer_legal_by_shape': by_shape,
            'measured_rungs_by_shape': {f'{shape[0]}x{shape[1]}': sorted({
                int(parsed[1]) for name, rows in costs.items() if shapes[name] == shape
                for fmt in rows if (parsed := parse_tessera_format_name(fmt)) is not None
                and parsed[0].name == family_name}) for shape in sorted(set(shapes.values()))}}
    for name, rows in costs.items():
        for fmt in rows:
            parsed = parse_tessera_format_name(fmt)
            if parsed is None:
                continue
            _require(parsed[0].name in result, f'{name}/{fmt}: measured family missing from legal inventory')
            shape = shapes[name]
            ranges = result[parsed[0].name]['producer_legal_by_shape'][f'{shape[0]}x{shape[1]}']['ranges_inclusive_q256']
            _require(any(start <= parsed[1] <= stop for start, stop in ranges),
                     f'{name}/{fmt}: measured rung is outside producer/shape legal inventory')
    return result


def _stack_roster(payload, profile):
    from .tessera_expert_projection import PROJECTION_KEY, carried_units
    projection = payload['provenance'].get(PROJECTION_KEY)
    _require(projection is not None, 'sampled proposal needs original complete producer projection')
    _source, units, stack_of = carried_units(projection)
    _require(set(units) <= set(payload['costs']), 'producer stack member is absent from pilot')
    group_members = {}
    for name in payload['costs']:
        key = profile.packed_expert_format_group(name)
        if name in units:
            _require(key is not None, f'{name}: projected expert has no atomic profile group')
            group_members.setdefault(key, set()).add(name)
        else:
            _require(key is None, f'{name}: routed expert is missing from producer projection')
    stack_members = {stack: set(members) for stack, members in projection['stacks'].items()}
    _require({frozenset(members) for members in group_members.values()} ==
             {frozenset(members) for members in stack_members.values()},
             'profile atomic groups disagree with complete producer stack roster')
    _require(set(stack_of) == set(units), 'producer projected roster is incomplete')
    return units, stack_members


def propose_bound_payload(payload, *, profile, mutable_budget_bytes, immutable_bytes,
                          reserve_bytes, full_legal_families, target_profile='research',
                          bit_precision=0.01, max_exact_attempts=16,
                          serving_target=None):
    """Solve complete atomic stacks at sampled-panel prices, then exact-filter.

    The payload must be returned by ``bind_allocation_payload`` in the narrow
    ``sampled_joint_panel`` scope.  All byte inputs are integer artifact facts,
    not bpp estimates or assumed head sizes.
    """
    from .cost_currency import require_sampled_joint_run_currency
    from .tessera_joint_eval_panel import observation_status
    handoff = payload['provenance'].get('tessera_joint_allocation', {})
    _require(handoff.get('status') == 'research_sampled_joint_panel'
             and handoff.get('export_authority') is False,
             'sampled proposal needs authenticated research-only binding')
    currency = require_sampled_joint_run_currency(payload)
    for key, value in [('mutable_budget_bytes', mutable_budget_bytes),
                       ('immutable_bytes', immutable_bytes), ('reserve_bytes', reserve_bytes)]:
        _require(type(value) is int and value >= 0, f'{key}: exact nonnegative integer required')
    _require(type(bit_precision) in (float, int) and bit_precision > 0,
             'positive solver bit precision required')
    _require(type(max_exact_attempts) is int and max_exact_attempts > 0,
             'positive exact retry bound required')
    stats, costs = payload['stats'], payload['costs']
    _require(isinstance(full_legal_families, (list, tuple)) and
             PRIMARY_FAMILIES <= set(full_legal_families),
             'sampled proposal must retain both full legal primary families')
    from .tessera_serving_scope import ServingTarget
    original_scope = payload['provenance'].get('tessera_serving_scope')
    if serving_target is not None:
        _require(isinstance(serving_target, ServingTarget),
                 'validation export target must be explicit ServingTarget')
    units, stacks = _stack_roster(payload, profile)
    unknown = set()
    for name, s in stats.items():
        state = observation_status(s['joint_eval_observations'])
        if state == 'unknown_unobserved':
            _require(name in units, f'{name}: unobserved dense/independent unit has no panel price')
            unknown.add(name)
        for fmt, row in costs[name].items():
            _require(row['joint_eval_status'] == state, f'{name}/{fmt}: observation drift')
            _require(row['joint_operator_identity']['source_weight']['dtype'] == 'torch.bfloat16',
                     f'{name}: BF16 source footprint has not been authenticated')
    for stack, members in stacks.items():
        _require(members - unknown, f'{stack}: entire stack unobserved; no measured panel option')
        common = set.intersection(*(set(costs[name]) for name in members))
        _require(common, f'{stack}: no common measured candidate across complete stack')
        # Each expert's w1/w3/w2 consumes the SAME selected input route.  If
        # their observed per-probe token/call census differs, the panel is not
        # one sampled population and its grouped price cannot be interpreted.
        by_expert = {}
        for name in members:
            expert = int(units[name]['expert'])
            counts = stats[name]['joint_eval_observations']['per_probe']
            signature = tuple((item['tokens'], item['calls']) for item in counts)
            prior = by_expert.setdefault(expert, signature)
            _require(prior == signature,
                     f'{stack}: inconsistent per-probe route counts for expert {expert}')
    legal = _domain(full_legal_families, costs, stats)
    specs = [fr.get_format(fmt) for fmt in sorted({fmt for rows in costs.values() for fmt in rows})]
    mask = []
    # Group members keep whole menus until aggregation prices the group: a rung
    # pruned from one member drops out of the group's name intersection.
    deferred = (packed_serving_group_members(stats, profile)
                | fused_sibling_group_members(stats, profile))
    raw = build_candidates(stats, costs, specs,
                           source_manifest={name: 'bf16' for name in stats},
                           target_profile=target_profile, mask_records=mask,
                           tessera_menu_mode='research',
                           defer_menu_reduction=deferred)
    _require(set(raw) == set(stats) and all(raw.values()), 'some pilot unit has no legal measured candidate')
    grouped_stats, grouped_costs, grouped = aggregate_packed_serving_groups(
        stats, costs, specs, raw, profile)
    del grouped_costs
    for stack, members in stacks.items():
        matching = [name for name, row in grouped_stats.items()
                    if set(row.get('_packed_group_members', [])) == members]
        _require(len(matching) == 1 and matching[0] in grouped,
                 f'{stack}: entire projected stack was not aggregated atomically')
        options = {c.fmt for c in grouped[matching[0]]}
        _require(options <= set.intersection(*(set(costs[name]) for name in members)),
                 f'{stack}: group gained an unmeasured candidate')
    _require(all(name not in grouped for name in unknown),
             'unobserved member escaped as an independent allocator choice')
    # The fused aggregator's licensed per-member Tessera composites remain the
    # one existing path for dense gate/up and QKV groups.
    final_stats, _, final_candidates = aggregate_fused_siblings(
        grouped_stats, payload['costs'], specs, grouped, profile)
    _require(set(final_stats) == set(final_candidates) and all(final_candidates.values()),
             'fused aggregation lost a measured decision unit')
    total_params = sum(int(s['n_params']) for s in final_stats.values())
    _require(total_params > 0, 'no mutable quantizable parameters')
    target_bpp = 8.0 * mutable_budget_bytes / total_params
    rank = {spec.name: index for index, spec in enumerate(specs)}
    formats = {spec.name: spec for spec in specs}
    trace = []
    target = target_bpp
    selected = None
    for attempt in range(max_exact_attempts):
        solver_diag = {}
        assignment, solver_bpp = solve_with_promotion(
            final_stats, final_candidates, target, formats, rank, bit_precision,
            overshoot_tolerance=0.0, profile=profile, diagnostics=solver_diag)
        if assignment is None:
            trace.append({'attempt': attempt, 'solver': solver_diag, 'feasible': False})
            break
        expanded = expand_packed_group_assignment(
            expand_fused_sibling_assignment(assignment, final_stats), final_stats)
        _require(set(expanded) == set(stats), 'expanded proposal dropped or gained a unit')
        for stack, members in stacks.items():
            _require(len({expanded[name] for name in members}) == 1,
                     f'{stack}: stack assignment is not one exact rung')
        # The selected menu contains BF16 and Tessera only.  Ask the existing
        # serialization owner for every expanded tensor; no float bpp can
        # grant a byte of budget tolerance.
        _require(all(fmt == 'BF16' or parse_tessera_format_name(fmt) is not None
                     for fmt in expanded.values()), 'sampled proposal has an unsupported sidecar')
        exact = sum(serialized_candidate_payload(formats[fmt],
                    (stats[name]['out_features'], stats[name]['in_features']),
                    qname=name)[0]
                    for name, fmt in expanded.items())
        trace.append({'attempt': attempt, 'solver_bpp': solver_bpp,
                      'exact_mutable_bytes': exact, 'feasible': exact <= mutable_budget_bytes})
        if exact <= mutable_budget_bytes:
            selected = assignment, expanded, exact
            break
        target -= max((8.0 * (exact - mutable_budget_bytes) / total_params), bit_precision)
        if target <= 0:
            break
    _require(selected is not None, 'no exact-byte-feasible sampled panel proposal')
    assignment, expanded, exact = selected
    chosen = {name: costs[name][fmt] for name, fmt in expanded.items()}
    summary = assignment_probe_summary(chosen, objective='additive')
    _require(summary['probe_identity_sha256'] == currency['probe_identity_sha256'],
             'selected assignment probe identity changed')
    observations = {name: {'status': stats[name]['joint_eval_status'],
                           'counts': stats[name]['joint_eval_observations'],
                           'full_population_price': None,
                           'observed_panel_contribution': (0.0 if name in unknown
                               else float(chosen[name]['predicted_dloss']))}
                    for name in sorted(stats)}
    # A deterministic next request asks for missing neighboring rates at the
    # selected stack rung, without fabricating a price for any missing cell.
    followup = {}
    for stack, members in sorted(stacks.items()):
        fmt = expanded[next(iter(members))]
        parsed = parse_tessera_format_name(fmt)
        if parsed is None:
            continue
        family, rate = parsed
        lo, hi = family_q256_bounds(family)
        missing = [family.format_name(r) for r in (rate-1, rate+1)
                   if lo <= r <= hi and any(family.format_name(r) not in costs[n] for n in members)]
        if missing:
            followup[stack] = {'common_panel_members': sorted(members),
                               'requested_rungs': missing,
                               'estimated_price': None, 'purpose': 'adaptive_measurement_needed'}
    return {'schema': SCHEMA, 'status': 'research_proposal', 'export_authority': False,
            'production_export_authority': False,
            'research_validation_permitted': True,
            'validation_export_eligible': None,
            'validation_export_integrity_status': 'pending_separate_research_byte_exporter',
            'required_integrity_gates': [
                'authenticated_original_source_and_H_preparation',
                'selected_measured_wire_current_bytes_and_encoder_identity',
                'expanded_assignment_exact_mutable_and_fixed_bytes',
                'Tessera_research_byte_exporter_closed_source_cache_reader_runtime',
                'priced_H_and_activation_scales',
                'selected_cached_unit_bundle_current_byte_verification',
                'same_research_selected_moe_json_to_Tessera_plan_and_export',
            ],
            'official_native_export_gate': {
                'scope': 'production_promotion',
                'research_marker': 'ordinary_preflight_refused_pending_validation',
                'native_cell_qualification': 'not_evaluated_by_proposal; record separately',
                'waiver': False},
            'pilot': payload['provenance']['joint_eval'],
            'original_joint_plan_sha256': handoff['plan_sha256'],
            'original_prepared': handoff['prepared'],
            'price_source': 'sampled_joint_panel',
            'target_profile': target_profile,
            'serving_target': None if serving_target is None else serving_target.as_dict(),
            'measurement_serving_scope': original_scope,
            'measurement_to_validation_target_equivalence': 'pending_runtime_preflight',
            'objective': 'additive_sum_of_observed_panel_member_quadratic_prices',
            'uncertainty_scope': 'probe_sampling_conditional_on_fixed_'
                f"{payload['provenance']['joint_eval']['selection']['size']}_window_panel",
            'sampled_panel_summary': summary,
            'unobserved_member_policy': 'null_full_population_price; zero_panel_contribution_only_inside_complete_atomic_stack',
            'observations': observations, 'atomic_stacks': {k: sorted(v) for k, v in sorted(stacks.items())},
            'solver_assignment': assignment, 'expanded_assignment': expanded,
            'selected_assignment_sha256': selected_assignment_sha256(expanded),
            'exact_bytes': {'mutable_payload': exact, 'immutable': immutable_bytes,
                            'reserve': reserve_bytes, 'total': exact + immutable_bytes + reserve_bytes,
                            'mutable_budget': mutable_budget_bytes,
                            'total_budget': mutable_budget_bytes + immutable_bytes + reserve_bytes},
            'full_legal_rate_inventory': legal, 'measured_candidates_only': True,
            'full_domain_optimality_claimed': False, 'full_population_cost_claimed': False,
            'candidate_masks': mask, 'exact_filter_trace': trace,
            'candidate_acquisition_request_example': followup,
            'adaptive_next_request': {
                'status': 'not_derived_from_existing_rate_hull',
                'price_for_unmeasured_rungs': None,
                'full_domain_controller_claimed': False},
            'independent_selected_assignment_validation': {
                'required_for_production_promotion': True, 'status': 'pending',
                'retained_teacher_panel_positions': None,
                'fit_overlap_status': 'unverified',
                'retained_panel_label': 'retained_validation_panel_not_heldout_until_overlap_audit',
                'validation_helper_schema': 'prismaquant.sampled_proposal_validation.v1',
                'validation_helper_required_bindings': [
                    'selected_assignment_sha256', 'incumbent_assignment_sha256',
                    'metric_schema', 'metric_support', 'metric_identity_artifact_sha256',
                    'heldout_training_overlap_audit_sha256',
                    'sequence_independence_declared', 'resampling_scope',
                    'resampling_limitations'],
                'independent_sequence_metric_status': 'not_evaluated'}}


def _read_bound(bound, label):
    _require(isinstance(bound, dict) and set(bound) == {'path', 'sha256'},
             f'{label}: path and SHA256 binding required')
    raw = Path(bound['path']).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == bound['sha256'],
             f'{label}: input bytes changed')
    return raw


def selected_assignment_sha256(assignment):
    """Bind the expanded member assignment, not its optimizer super-item IDs."""
    _require(isinstance(assignment, dict) and assignment and
             all(isinstance(k, str) and isinstance(v, str) for k, v in assignment.items()),
             'selected research assignment must be complete qname/format pairs')
    return hashlib.sha256(json.dumps(assignment, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def bind_pilot_from_inputs(*, joint_binding, plan_binding):
    """Authenticate real completed pilot, original 512 draw, preparation and wires."""
    from .production_weight_cache import ProductionWeightCache
    from .tessera_joint_aura import SCHEMA as PLAN_SCHEMA, load_measured_anchor_input
    from .tessera_joint_allocation import bind_allocation_payload

    pilot = pickle.loads(_read_bound(joint_binding, 'pilot joint costs'))
    plan = json.loads(_read_bound(plan_binding, 'pilot plan'))
    _require(plan.get('schema') == PLAN_SCHEMA and plan.get('joint_eval') is not None,
             'pilot plan must carry diagnostic joint evaluation')
    _require(pilot['provenance']['tessera_joint_anchors']['plan_sha256'] == plan_binding['sha256']
             and pilot['provenance']['joint_eval'] == plan['joint_eval'],
             'pilot payload and plan identity disagree')
    prepared_binding = pilot['provenance']['tessera_joint_anchors']['prepared']
    prepared = json.loads(_read_bound(prepared_binding, 'prepared pilot completion'))
    _require(prepared['calibration_input']['artifact_sha256'] == plan['calibration_input']['sha256'],
             'original calibration artifact identity changed')
    from .calibration_data import load_calibration_input
    from .tessera_joint_eval_panel import select_panel
    ids, calibration = load_calibration_input(plan['calibration_input']['path'],
        expected_sha256=plan['calibration_input']['sha256'],
        n_samples=plan['execution']['n_calib_samples'],
        seqlen=plan['execution']['calib_seqlen'])
    _selected_ids, panel = select_panel(ids, calibration, plan['joint_eval'])
    _require(panel == pilot['provenance']['joint_eval'],
             'pilot evaluation token bytes differ from original calibration')
    cache = pickle.loads(_read_bound(prepared['production_cache'], 'prepared cache'))
    _require(isinstance(cache, ProductionWeightCache), 'prepared cache owner differs')
    data = load_measured_anchor_input(plan['inputs'], verify_payloads=False,
                                      require_existing_renders=True)
    _require(cache.weights == {pair: cell['render'] for pair, cell in data.cells.items()},
             'pilot prepared render paths differ from original anchors')
    bound = bind_allocation_payload(pilot, data, prepared, cache.metadata,
        plan_sha256=plan_binding['sha256'], prepared_binding=prepared_binding,
        scope='sampled_joint_panel')
    return bound, plan


def propose_from_bound_inputs(*, joint_binding, plan_binding, output_path,
                              mutable_budget_bytes, immutable_bytes, reserve_bytes,
                              full_legal_families, target_profile='research', bit_precision=0.01,
                              serving_target=None):
    """Emit a research proposal; no ordinary cost table or export authority."""
    from .model_profiles.registry import detect_profile
    from .tessera_serving_scope import ServingTarget
    _require(isinstance(serving_target, ServingTarget),
             'explicit validation ServingTarget required before reading large pilot inputs')
    bound, plan = bind_pilot_from_inputs(joint_binding=joint_binding,
                                          plan_binding=plan_binding)
    proposal = propose_bound_payload(bound, profile=detect_profile(plan['model']),
        mutable_budget_bytes=mutable_budget_bytes, immutable_bytes=immutable_bytes,
        reserve_bytes=reserve_bytes, full_legal_families=full_legal_families,
        target_profile=target_profile, bit_precision=bit_precision,
        serving_target=serving_target)
    proposal['input_bindings'] = {'pilot_joint_cost': joint_binding, 'pilot_plan': plan_binding,
                                  'prepared': bound['provenance']['tessera_joint_allocation']['prepared']}
    target = Path(output_path)
    _require(not target.exists(), 'proposal destination already exists')
    layer_target = target.with_suffix(target.suffix + '.validation-layer-config.json')
    _require(not layer_target.exists(), 'validation layer config destination already exists')
    raw = (json.dumps(proposal, indent=2, sort_keys=True) + '\n').encode()
    config = validation_layer_config(bound, proposal,
                                    proposal_sha256=hashlib.sha256(raw).hexdigest(),
                                    profile=detect_profile(plan['model']),
                                    serving_target=serving_target)
    _atomic_write_new_bytes(target, raw)
    _atomic_write_new_bytes(layer_target,
        (json.dumps(config, indent=2, sort_keys=True) + '\n').encode())
    return proposal


def validation_layer_config(bound_payload, proposal, *, proposal_sha256,
                            profile, serving_target=None):
    """Native layer config with the same metadata gates as the normal allocator.

    This does not make the research selection shippable: its marker demands an
    explicit proposal on the validation-export preflight and selected cache.
    """
    from .layer_config import LAYER_CONFIG_META_KEY
    from .tessera_expert_projection import allocation_expert_projection_block
    from .joint_catalog_extension import hessian_references
    from .tessera_menu import (assert_uniform_hessian_identity, priced_static_scales,
                               project_hessian_identity)
    from .tessera_serving_scope import (ServingTarget, context_by_unit_from_stats,
                                        scope_provenance)
    _require(len(proposal_sha256) == 64 and all(c in '0123456789abcdef' for c in proposal_sha256),
             'research proposal SHA-256 required')
    assignment = proposal['expanded_assignment']
    require_research_proposal_assignment(proposal, assignment)
    _require(isinstance(serving_target, ServingTarget),
             'validation layer config needs a separate explicit ServingTarget')
    contexts = context_by_unit_from_stats(serving_target, bound_payload['stats'], profile)
    config = {name: fr.get_format(fmt).autoround_config()
              for name, fmt in sorted(assignment.items())}
    config[LAYER_CONFIG_META_KEY] = {
        'schema': 'prismaquant.layer_config_meta.v1',
        'target_profile': proposal['target_profile'],
        'tessera_serving_scope': scope_provenance(serving_target, contexts),
        'tessera_hessian': project_hessian_identity(assert_uniform_hessian_identity(
            bound_payload['costs'], references=lambda: hessian_references(bound_payload)),
            assignment),
        'tessera_activation_static_scales': priced_static_scales(
            {name: fmt for name, fmt in assignment.items() if fmt.startswith('TESSERA_')},
            bound_payload['costs'],
            policy=(bound_payload['provenance']
                    .get('activation_static_scales', {}).get('policy'))),
        **allocation_expert_projection_block(bound_payload, assignment),
        'sampled_joint_proposal': {
            'schema': 'prismaquant.tessera_sampled_validation_export_binding.v1',
            'proposal_sha256': proposal_sha256,
            'selected_assignment_sha256': proposal['selected_assignment_sha256']},
    }
    return config


def require_research_proposal_assignment(proposal, assignment, *, pilot_joint_binding=None):
    """Bind a candidate export/selected-cache request to the reviewed proposal.

    Existing current-byte, Hessian, scale, encoder, route and runtime gates run
    after this admission; this verifier never reports their eligibility.
    """
    from .layer_config import canonicalize_assignment
    _require(isinstance(proposal, dict) and proposal.get('schema') == SCHEMA
             and proposal.get('status') == 'research_proposal'
             and proposal.get('production_export_authority') is False
             and proposal.get('validation_export_eligible') is None
             and proposal.get('research_validation_permitted') is True,
             'research export needs a pending-integrity sampled proposal')
    _require(proposal.get('pilot', {}).get('status') == 'diagnostic_pilot',
             'research proposal lost diagnostic pilot identity')
    expected = proposal.get('expanded_assignment')
    actual = canonicalize_assignment(assignment)
    _require(actual == expected and selected_assignment_sha256(actual) ==
             proposal.get('selected_assignment_sha256'),
             'research export assignment differs from expanded sampled proposal')
    if pilot_joint_binding is not None:
        _require(proposal.get('input_bindings', {}).get('pilot_joint_cost') == pilot_joint_binding,
                 'research selected cache pilot cost binding differs')
    return {'schema': 'prismaquant.tessera_sampled_validation_export_binding.v1',
            'proposal_schema': SCHEMA,
            'selected_assignment_sha256': proposal['selected_assignment_sha256'],
            'validation_export_eligible': None,
            'production_export_authority': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-cost', required=True)
    parser.add_argument('--pilot-cost-sha256', required=True)
    parser.add_argument('--pilot-plan', required=True)
    parser.add_argument('--pilot-plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mutable-budget-bytes', required=True, type=int)
    parser.add_argument('--immutable-bytes', required=True, type=int)
    parser.add_argument('--reserve-bytes', required=True, type=int)
    parser.add_argument('--full-legal-family', action='append', required=True)
    parser.add_argument('--target-profile', default='research')
    from .tessera_serving_scope import add_serving_scope_arguments, serving_target_from_args
    add_serving_scope_arguments(parser)
    args = parser.parse_args(argv)
    result = propose_from_bound_inputs(
        joint_binding={'path': args.pilot_cost, 'sha256': args.pilot_cost_sha256},
        plan_binding={'path': args.pilot_plan, 'sha256': args.pilot_plan_sha256},
        output_path=args.output, mutable_budget_bytes=args.mutable_budget_bytes,
        immutable_bytes=args.immutable_bytes, reserve_bytes=args.reserve_bytes,
        full_legal_families=args.full_legal_family, target_profile=args.target_profile,
        serving_target=serving_target_from_args(args))
    print(json.dumps({'schema': result['schema'], 'status': result['status'],
                      'output': args.output, 'exact_bytes': result['exact_bytes']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
