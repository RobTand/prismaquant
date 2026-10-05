#!/usr/bin/env python3
"""Publish the exact selected MTP priced receipts as an original v1 child.

This is a closed metadata handoff to Tessera's cached-unit intake. It reruns
the existing M6→M4→M3 bound-cost join, checks every selected producer unit and
wire location, and leaves the historical wire bytes and identities untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cost_stage_checkpoint import publish_new_bytes

from prismaquant.glm_mtp_selection import (WIRE_BINDING_SCHEMA,
                                           backfill_mtp_selection_wires)
from prismaquant.layer_config import (canonicalize_assignment,
                                      layer_config_metadata,
                                      validate_layer_config_payload)
from prismaquant.tessera_expert_projection import (cached_units_manifest,
    carried_units, check_expert_wire_receipt, locate_expert_wire)
from prismaquant.tessera_formats import parse_tessera_format_name
from tessera.cached_unit import CACHE_SCHEMA, CachedUnitBundle


def selected_mtp_child(config: dict, *, cost_path: str, output: Path) -> dict:
    validate_layer_config_payload(config, 'MTP selected assignment')
    selection = layer_config_metadata(config).get('mtp_selection')
    if not isinstance(selection, dict) or selection.get('cost_path') != cost_path:
        raise ValueError('MTP child needs the allocation\'s exact bound cost path')
    if selection.get('mtp_expert_wire_binding_schema') != WIRE_BINDING_SCHEMA:
        raise ValueError('MTP child needs exact bound priced-wire metadata')
    expected = backfill_mtp_selection_wires(config, cost_path)['__prismaquant__']['mtp_selection']
    fields = ('mtp_joint_cost_sha256', 'mtp_expert_projection', 'mtp_expert_wires',
              'mtp_expert_wire_roots', 'mtp_expert_source_bindings',
              'mtp_expert_wire_binding_schema')
    if any(selection.get(field) != expected[field] for field in fields):
        raise ValueError('MTP child selection differs from bound M6→M4→M3 price')
    source, units, _stacks = carried_units(selection['mtp_expert_projection'])
    receipts = selection['mtp_expert_wires']
    roots = selection['mtp_expert_wire_roots']
    if not isinstance(receipts, dict) or not receipts or set(receipts) != set(roots):
        raise ValueError('MTP child selected receipt/root rosters differ')
    if set(receipts) != set(units):
        raise ValueError('MTP child does not cover the complete selected expert projection')
    paths = {str(Path(root).resolve()) for root in roots.values()}
    if len(paths) != 1 or paths != {str(output.parent.resolve())}:
        raise ValueError('MTP v1 child requires one original wire directory')
    assignment = canonicalize_assignment(config)
    records = {}
    for name, record in sorted(receipts.items()):
        fmt = assignment.get(name)
        parsed = parse_tessera_format_name(fmt) if isinstance(fmt, str) else None
        if parsed is None:
            raise ValueError(f'{name}: selected MTP expert has no Tessera rung')
        family, q256 = parsed
        checked = check_expert_wire_receipt(
            record, name=name, unit=units[name], q256=int(q256),
            grid=family.payload_grid().name)
        locate_expert_wire(checked, name=name, wire_dir=output.parent)
        records[name] = checked
    seals = {record['identity']['encoder_source_sha256'] for record in records.values()}
    if len(seals) != 1:
        raise ValueError('MTP v1 child mixes historical producer packages')
    manifest = cached_units_manifest(source, records, schema=CACHE_SCHEMA)
    CachedUnitBundle(manifest, output.parent, set(records), source)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', required=True)
    parser.add_argument('--assignment-sha256', required=True)
    parser.add_argument('--mtp-joint-cost', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'MTP child output exists: {output}')
    raw = Path(args.assignment).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.assignment_sha256:
        raise ValueError('MTP selected assignment SHA-256 differs')
    manifest = selected_mtp_child(
        json.loads(raw), cost_path=args.mtp_joint_cost, output=output)
    encoded = (json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                          allow_nan=False) + '\n').encode()
    if not publish_new_bytes(output, encoded):
        raise FileExistsError(f'MTP child output exists: {output}')
    print(json.dumps({'schema': 'prismaquant.mtp_cached_child_handoff.v1',
                      'manifest': str(output.resolve()),
                      'manifest_sha256': hashlib.sha256(encoded).hexdigest(),
                      'assignment_sha256': args.assignment_sha256,
                      'encoder_source_sha256': next(iter({
                          record['identity']['encoder_source_sha256']
                          for record in manifest['units'].values()})),
                      'units': len(manifest['units']),
                      'root': str(output.parent.resolve()),
                      'export_qualified': False, 'serving_qualified': False},
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
