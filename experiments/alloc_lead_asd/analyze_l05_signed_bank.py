"""Inspect accepted L05 signed rows without claiming a corrected model price.

Run through PrismaBuild with a bound staged readset. This diagnostic uses the
existing signed-probe arithmetic owner; it does not fabricate cost identities.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import torch

from prismaquant.cost_stage_checkpoint import atomic_write_bytes
from prismaquant.digests import bytes_sha256hex
from prismaquant.joint_aura import signed_probe_quadratic_summary
from prismaquant.residency_map import bind_residency_manifest, residency_report
from prismaquant.staged_tier_policy import activate_staged_tier_policy
from prismaquant.staged_whole_file import read_staged_whole_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('rows', 'binding', 'local_control'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
        p.add_argument('--' + name.replace('_', '-') + '-sha256', required=True)
    p.add_argument('--data-manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('output must be unused')
    torch.set_num_threads(1)
    activate_staged_tier_policy('ram,ssd')
    bind_residency_manifest(args.data_manifest_sha256)
    inputs, raw = {}, {}
    for name in ('rows', 'binding', 'local_control'):
        path, digest = getattr(args, name), getattr(args, name + '_sha256')
        raw[name] = read_staged_whole_file(path, digest, label='accepted L05 ' + name)
        inputs[name] = dict(path=str(path), sha256=digest, bytes=len(raw[name]))
    binding, control = json.loads(raw['binding']), json.loads(raw['local_control'])
    if (binding['publisher_revision'] != 'a6c167b62691b2bac901344b65cb651a70f53e43'
            or binding['layer'] != 5 or binding['sequences'] != [0, 1, 2, 3]
            or binding['sequence_length'] != 512 or binding['n_probes'] != 4
            or not binding['cotangents_are_banked_fixed_inputs']
            or not control['same_primal_operator_gate_passed']
            or control['resident_reference_gate_passed']
            or control['rows']['sha256'] != args.rows_sha256):
        raise ValueError('not the accepted fixed-cotangent L05 control')
    bank = torch.load(io.BytesIO(raw['rows']), map_location='cpu', weights_only=True)
    declared = {key.removesuffix('.weight') for key in binding['layer_keys']
                if key.endswith('.weight') and
                ('.mlp.experts.' in key or '.mlp.shared_experts.' in key)}
    if len(declared) != 867 or set(bank['selected_units']) != declared:
        raise ValueError('selected unit roster differs from the bound source keys')
    observed = {(r['unit'], r['probe']): r for r in bank['rows']}
    expected = {(unit, probe) for unit in declared for probe in range(4)}
    if len(observed) != len(bank['rows']) or set(observed) != expected:
        raise ValueError('incomplete or duplicate unit/probe coverage')
    columns, sequence_columns = {}, {sequence: {} for sequence in range(4)}
    for unit in sorted(declared):
        columns[unit] = []
        for sequence in range(4):
            sequence_columns[sequence][unit] = []
        for probe in range(4):
            row = observed[unit, probe]
            coords, signed = row['coordinates'], row['signed_rows']
            if (coords.dtype != torch.int64 or coords.ndim != 2 or coords.shape[1] != 3
                    or signed.dtype != torch.float64 or signed.shape != (len(coords),)
                    or not torch.isfinite(signed).all()
                    or not math.isfinite(row['fixed_g_dot'])):
                raise ValueError('malformed/nonfinite signed rows')
            columns[unit].append(row['fixed_g_dot'])
            for sequence in range(4):
                values = signed[coords[:, 0] == sequence]
                sequence_columns[sequence][unit].append(math.fsum(values.tolist()))
    for probe in range(4):
        for role in ('gate_proj', 'up_proj', 'down_proj'):
            for routed in (False, True):
                coordinates = [tuple(c) for (unit, k), row in observed.items()
                    if k == probe and unit.endswith(role) and ('.experts.' in unit) == routed
                    for c in row['coordinates'].tolist()]
                expected_coords = {(s, t, slot) for s in range(4) for t in range(512)
                                   for slot in (range(8) if routed else (-1,))}
                if len(coordinates) != len(expected_coords) or set(coordinates) != expected_coords:
                    raise ValueError('coordinate coverage differs')
    groups = {'all': sorted(declared),
              'routed': sorted(u for u in declared if '.experts.' in u),
              'shared': sorted(u for u in declared if '.shared_experts.' in u)}
    for role in ('gate_proj', 'up_proj', 'down_proj'):
        groups['routed_' + role] = [u for u in groups['routed'] if u.endswith(role)]
    def summarize(source, names):
        additive = signed_probe_quadratic_summary([source[u] for u in names])
        joint = signed_probe_quadratic_summary([source[u] for u in names], objective='joint_quadratic')
        return dict(units=len(names), additive=additive, joint_quadratic=joint,
                    joint_over_additive=joint['mean'] / additive['mean'] if additive['mean'] else None,
                    joint_minus_additive_per_probe=[a-b for a,b in zip(joint['per_probe'], additive['per_probe'])])
    result = dict(scope='Accepted original L05 n4/T512/K4 local fixed-cotangent signed-row decomposition',
        origin_action='1d8fb8ddab4bc4ec774c56ff545977475e1bad83d502ff3ed2a6b24bab18b51b',
        input_sha256=inputs, source_publisher=binding['publisher_revision'],
        historical_global_token_count=binding['global_token_count'], normalization_changed=False,
        groups={label:summarize(columns,names) for label,names in groups.items()},
        by_sequence={str(s):{label:summarize(values,names) for label,names in groups.items()}
                     for s,values in sequence_columns.items()},
        uncertainty_scope='probe sampling conditional on these four sequences and supplied banked cotangents',
        limitations=['not corrected full-model/global cotangent qualification',
                     'not full calibration or first25-window price', 'not a factor7 explanation',
                     'not standalone A KL or whole-network quality', 'not allocator price admission'],
        residency=residency_report())
    payload=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_write_bytes(args.output,payload)
    print(json.dumps(dict(report_sha256=bytes_sha256hex(payload),scope=result['scope'],
        groups=result['groups'],normalization_changed=False,limitations=result['limitations']),allow_nan=False))


if __name__ == '__main__':
    main()

