#!/usr/bin/env python3
"""Make bounded full-domain acquisition requests from existing measured rows.

Run through PrismaBuild. This command is a scalar-output-MSE research adapter;
it neither needs a Fisher probe nor fabricates one. It produces acquisition
requests, not an allocation or a replacement for joint AURA. The source tensor
inventory is the metadata-only header inventory from the campaign coordinator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cost_currency import require_run_currency
from prismaquant.tessera_allocator import build_tessera_allocator_candidate
from prismaquant.tessera_full_domain_acquisition import (
    adaptive_acquisition_from_records, require_measured_recipe_binding,
)
from prismaquant.cost_stage_checkpoint import unit_path, _load_unit
from tessera.cached_unit import encoder_source_sha256
from prismaquant.tessera_legal_domain import live_pins, tessera_source_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--costs', type=Path, required=True)
    parser.add_argument('--source-tensors', type=Path, required=True)
    parser.add_argument('--anchor-parts', type=Path, required=True)
    parser.add_argument('--unit', action='append', required=True)
    parser.add_argument('--family', action='append', required=True)
    parser.add_argument('--max-new-points', type=int, required=True)
    parser.add_argument('--alpha-loss-per-byte', type=float)
    parser.add_argument('--boundary-policy', choices=('seed', 'defer'), default='seed',
                        help='Defer unknown boundary work while retaining the full legal domain.')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if len(set(args.unit)) != len(args.unit) or len(set(args.family)) != len(args.family):
        parser.error('unit/family selections must be distinct')
    if args.out.exists():
        parser.error('refusing to overwrite prior acquisition output')
    raw = args.costs.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    costs = pickle.loads(raw)
    del raw
    currency = require_run_currency(costs)
    if currency['expected_currency'] != 'render-score':
        parser.error('this adapter requires measured scalar render-score costs')
    source_raw = args.source_tensors.read_bytes()
    source = {r['name']: r for r in json.loads(source_raw)}
    reports = []
    active_encoder_sha256 = encoder_source_sha256()
    journal_bindings = {}
    for unit in args.unit:
        if unit not in costs['costs'] or unit + '.weight' not in source:
            parser.error(f'unit missing from cost/source intersection: {unit}')
        shape = source[unit + '.weight']['shape']
        part = unit_path(args.anchor_parts, unit)
        part_raw = part.read_bytes()
        envelope = pickle.loads(part_raw)
        state = _load_unit(part, stage='Tessera campaign', qname=unit,
                           identity_sha256=envelope['identity_sha256'])
        if part.read_bytes() != part_raw:
            parser.error(f'anchor journal changed during read: {part}')
        anchors = {r['format_name']: r for r in state['anchors']}
        journal_bindings[unit] = {
            'path': str(part), 'sha256': hashlib.sha256(part_raw).hexdigest(),
            'identity_sha256': envelope['identity_sha256'],
        }
        unit_source_identity = None
        for family in args.family:
            measured = []
            wire_bindings = {}
            for key, row in sorted(costs['costs'][unit].items()):
                if row.get('tessera_family') != family or row.get('output_mse_measured') is not True:
                    continue
                if row.get('cost_source') != 'tessera_campaign_measured':
                    parser.error(f'unrecognized measured provenance: {unit}/{key}')
                anchor = anchors.get(key, {})
                for target, original in (('output_mse', 'dloss'),
                        ('tessera_family', 'family'), ('tessera_body_rate_q256', 'body_rate_q256'),
                        ('activation_contract', 'activation_contract'),
                        ('activation_quantized', 'activation_quantized'), ('wire_bytes', 'wire_bytes')):
                    if row.get(target) != anchor.get(original):
                        parser.error(f'cost/journal {target} mismatch: {unit}/{key}')
                record = state['wire_records'].get(key, {})
                measured_source = record.get('identity', {}).get('source')
                if unit_source_identity is None:
                    unit_source_identity = measured_source
                elif measured_source != unit_source_identity:
                    parser.error(f'measured anchors mix source tensor identities: {unit}')
                wire_bindings[key] = require_measured_recipe_binding(
                    unit, shape, row, record, encoder_source_sha256=active_encoder_sha256)
                measured.append(build_tessera_allocator_candidate(
                    unit, shape, family=family, body_rate_q256=row['tessera_body_rate_q256'],
                    layout='tight', schedule=None, alphabets=None,
                    predicted_dloss=row['output_mse'],
                    # The table has no per-anchor repeated-sample uncertainty.
                    # This zero satisfies the pre-existing point-estimate API;
                    # the report explicitly withholds an uncertainty bound.
                    predicted_dloss_stderr=0.0,
                    target_profile='research',
                ))
            if not measured:
                parser.error(f'no measured anchor for {unit}/{family}')
            report = adaptive_acquisition_from_records(
                family, measured, max_new_points=args.max_new_points,
                alpha_loss_per_byte=args.alpha_loss_per_byte, boundary_policy=args.boundary_policy,
            )
            report['measured_wire_record_sha256'] = wire_bindings
            report['currency'] = 'output_mse_under_route_activation_contract'
            report['statistical_uncertainty'] = None
            report['uncertainty_scope'] = 'not_observed; RD refinement uses point estimates only'
            reports.append(report)
    result = {
        'schema': 'prismaquant.tessera_full_domain_campaign_acquisition.v1',
        'cost_path': str(args.costs), 'cost_sha256': digest,
        'source_tensor_inventory_sha256': hashlib.sha256(source_raw).hexdigest(),
        'cost_currency': currency, 'reports': reports,
        'journal_bindings': journal_bindings,
        'active_encoder_source_sha256': active_encoder_sha256,
        'measurement_verification': 'recorded anchor/recipe/source metadata; tensor and wire bodies not reread',
        'domain_pins': live_pins().as_dict(), 'producer_source_state': tessera_source_state(),
        'atomic_serving_group_expansion_required': True,
        'total_requested_quality_measurements': sum(len(r['proposed_q256']) for r in reports),
        'allocator_payload': False, 'production_qualified': False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'out': str(args.out), 'reports': len(reports),
                      'requested_quality_measurements': result['total_requested_quality_measurements']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
