"""Write the selected allocation's priced activation values without recalibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def write_selected_scales(assignment_binding, output_dir):
    from .tessera_joint_allocation import _read_bound
    from .layer_config import canonicalize_assignment, layer_config_metadata
    from .schemas import validate_layer_config_payload
    from .tessera_expert_projection import (
        POPULATION_KEY, PROJECTION_KEY, carried_units, expand_stack_decision_assignment)
    from .tessera_formats import (parse_tessera_format_name, tessera_wire_recipe,
                                  tessera_serving_route, route_static_activation_contract)
    from .nvfp4_activation_contract import is_routed_expert_projection_name, resolve_input_global_scale_policy
    from .tessera_export_lane import PRICED_STATIC_SCALES_SCHEMA, _require_routed_scale_grouping_declaration
    from .tessera_campaign import write_export_inputs
    from .cost_stage_checkpoint import publish_new_bytes

    # Parse only the authenticated bytes; never reopen a mutable allocation to
    # independently recover values or projection ownership after verification.
    payload = json.loads(_read_bound(assignment_binding, 'selected scale allocation'))
    validate_layer_config_payload(payload, assignment_binding['path'])
    selected, metadata = canonicalize_assignment(payload), layer_config_metadata(payload)
    population = metadata.get(POPULATION_KEY)
    if isinstance(population, dict) and population.get('stack_decisions'):
        _, units, stack_of = carried_units(metadata.get(PROJECTION_KEY))
        selected, _ = expand_stack_decision_assignment(selected, population, units=units, stack_of=stack_of)
    block = metadata.get('tessera_activation_static_scales')
    if not isinstance(block, dict) or block.get('schema') != PRICED_STATIC_SCALES_SCHEMA:
        raise ValueError('selected allocation has no priced static scale block')
    scales = {}
    for name, fmt in sorted(selected.items()):
        parsed = parse_tessera_format_name(fmt)
        if parsed is None:
            continue
        family, rung = parsed
        route = tessera_serving_route(family, tessera_wire_recipe(family, rung), rung)
        if route_static_activation_contract(route) is None:
            continue
        value = block.get('units', {}).get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name}: selected allocation lacks a finite positive priced static scale')
        scales[name] = value
    if not scales:
        raise ValueError('selected allocation has no static activation scales to write')
    if block.get('input_global_scale_policy') is None:
        raise ValueError('selected allocation lacks its priced scale formula')
    formula = resolve_input_global_scale_policy(block['input_global_scale_policy'])
    routed = [name for name in scales if is_routed_expert_projection_name(name)]
    report = {}
    if routed:
        _require_routed_scale_grouping_declaration(block.get('activation_scale_grouping'),
            routed_units=routed, report=report, served_activation_policy=block.get('served_activation_policy'),
            priced_units=scales, selected=selected, scale_formula=formula)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    _, path, _ = write_export_inputs(out, hessians=None, hessian_rows={}, hessian_identity={},
        static_scales=scales, static_scale_policy=formula)
    result = {'schema': 'prismaquant.selected_priced_scales.v1', 'assignment': dict(assignment_binding),
        'input_scales': {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
        'units': len(scales), 'input_global_scale_policy': formula,
        'served_activation_policy': block.get('served_activation_policy'),
        'activation_scale_grouping_declaration': block.get('activation_scale_grouping'),
        'serving_qualified': False, **report}
    if not publish_new_bytes(out/'receipt.json', (json.dumps(result, sort_keys=True, indent=2)+'\n').encode()):
        raise ValueError('selected scale receipt already exists')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', required=True)
    parser.add_argument('--assignment-sha256', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args(argv)
    result = write_selected_scales({'path': args.assignment, 'sha256': args.assignment_sha256}, args.out_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
