"""Audit banked tiny GLM rows, signed components and actual profiler backends.

Run through PrismaBuild on CPU. This never evaluates the original GLM model
or promotes a tiny mechanism control to an actual-model correctness result.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch


def file_identity(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def component_difference(reference, candidate):
    assert reference.keys() == candidate.keys(), "signed component key coverage differs"
    keys = sorted(reference)
    delta = [float(candidate[key]) - float(reference[key]) for key in keys]
    rms = math.sqrt(sum(float(reference[key]) ** 2 for key in keys) / len(keys))
    return {"components": len(keys), "max_abs": max(map(abs, delta)),
            "max_abs_over_reference_rms": max(map(abs, delta)) / max(rms, 1e-30),
            "exact": all(value == 0 for value in delta)}


def grouped_rows(records):
    groups = {}
    for row in records:
        key = row['unit'], int(row['probe'])
        assert key not in groups, f"duplicate tiny unit/probe record: {key}"
        groups[key] = row
    return groups


def row_audit(tensors):
    direct, captured = (grouped_rows(tensors[leg]['rows']) for leg in ('direct', 'capture'))
    assert direct.keys() == captured.keys() and len(direct) == 36
    fields = Counter()
    routed_coverage = {}
    dense_coverage = {}
    for key, original in direct.items():
        replay = captured[key]
        coordinates = original['coordinates']
        assert torch.equal(coordinates, replay['coordinates']), f"coordinate order differs: {key}"
        assert len(coordinates) == len({tuple(row) for row in coordinates.tolist()}), f"duplicate row: {key}"
        for name in ('x', 'dx', 'g'):
            assert original[name].shape == replay[name].shape, f"{key}: {name} shape differs"
            assert original[name].dtype == replay[name].dtype, f"{key}: {name} dtype differs"
            fields[name + '_exact'] += int(torch.equal(original[name], replay[name]))
        unit, probe = key
        role = unit.rsplit('.', 1)[-1]
        if '.experts.' in unit:
            coverage_key = (probe, role)
            values = routed_coverage.setdefault(coverage_key, [])
            values.extend(tuple(row) for row in coordinates.tolist())
        else:
            assert torch.equal(coordinates, torch.tensor(
                [[s, p, -1] for s in range(4) for p in range(32)])), f"dense row coverage differs: {key}"
            dense_coverage[str(key)] = len(coordinates)
    expected = {(s, p, slot) for s in range(4) for p in range(32) for slot in range(2)}
    assert len(routed_coverage) == 6
    for key, values in routed_coverage.items():
        assert len(values) == len(expected) and set(values) == expected, f"routed coverage differs: {key}"
    return {'records': len(direct), 'coordinate_order_exact': True,
            'routed_token_slot_coverage_exact': True, 'routed_pairs_per_probe_role': len(expected),
            'dense_records': len(dense_coverage), 'operand_exact_counts': dict(fields)}


def backend_audit(path):
    with gzip.open(path, 'rt') as handle:
        trace = json.load(handle)
    names = Counter(event.get('name', '') for event in trace['traceEvents'])
    kernels = {name: count for name, count in names.items()
               if any(marker in name.lower() for marker in (
                   'flash', 'attention', 'grouped', 'conv1d', 'causal_conv', 'bmm', 'mm'))}
    return {'identity': file_identity(path), 'backend_events': kernels,
            'scope': 'event identities/counts only; no timing or throughput comparison'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp32-root', type=Path, required=True)
    parser.add_argument('--bf16-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'schema': 'prismaquant.research.tiny_glm_audit.v1', 'legs': {},
              'scope': 'synthetic n4/T32/K2 mechanism control; actual GLM L05 remains open'}
    for dtype, root, result_name in (
            ('float32', args.fp32_root, 'result.partial.json'),
            ('bfloat16', args.bf16_root, 'result.json')):
        document = json.loads((root / result_name).read_text())
        leg = document['legs'][dtype]
        tensor_path = root / f'{dtype}.pt'
        tensors = torch.load(tensor_path, map_location='cpu', weights_only=True)
        audit = {'inputs': [file_identity(root / result_name), file_identity(tensor_path)],
                 'rows': row_audit(tensors),
                 'cotangent_differences': leg['cotangent_differences'],
                 'direct_vs_capture': component_difference(leg['direct_components'], leg['capture_components']),
                 'fixed_g_vs_operator': component_difference(leg['capture_components'], leg['operator_components']),
                 'trace': backend_audit(root / f'{dtype}.trace.json.gz'),
                 'direct_reuse': leg['direct_reuse']}
        if dtype == 'bfloat16':
            audit['operator_vs_spill'] = component_difference(leg['operator_components'], leg['spill']['components'])
            assert audit['operator_vs_spill']['exact'], 'same captured operands changed through spill'
            audit['permutation_refused'] = leg['spill']['permutation_refused']
            assert 'checksum mismatch' in audit['permutation_refused']
        result['legs'][dtype] = audit
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: exact coordinate coverage, contraction comparisons and BF16 permutation refusal audited')


if __name__ == '__main__':
    main()
