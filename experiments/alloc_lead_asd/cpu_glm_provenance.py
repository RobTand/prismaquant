"""Audit existing GLM L05 identities without loading model/capture tensors.

Run as an admitted CPU action. This establishes metadata consistency, never
that a stored cotangent equals an independently computed full-model gradient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path


BASE = Path('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913')
PATHS = {
    'cost': BASE / 'r13-stageb-20260923/a4/overlay/layer-quanta/layer-005/cost.pkl',
    'slice': BASE / 'r13-stageb-20260923/meta-045-fae344d/adjoint-slices/layer-005.json',
    'handoff': BASE / ('r13-stageb-20260923/a4/overlay/layer-quanta/layer-006/'
                       'handoff/31f5259905fe4c35b91e678809f967f2/handoff.json'),
    'plan': BASE / 'ws-sb4-1151/policy/overlay-plan.v8-chain.json',
}
FMT = 'TESSERA_E4M3_K1_R1024'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit_entries(records, *, boundary, kind, batches, probes=None):
    seen, sessions, shapes = set(), {}, Counter()
    for record in records:
        identity = record['metadata']['identity']
        coords = identity['coordinates']
        key = (int(coords['batch']), coords['probe'])
        assert key not in seen, f'duplicate {kind} coordinate {key}'
        seen.add(key)
        assert identity['kind'] == kind and int(coords['boundary']) == boundary
        assert record['shape'] == record['metadata']['shape']
        assert record['dtype'] == record['metadata']['dtype']
        sessions[digest(identity['session'])] = identity['session']
        shapes[(tuple(record['shape']), record['dtype'])] += 1
    expected = {(b, p) for b in range(batches)
                for p in (range(probes) if probes is not None else (None,))}
    assert seen == expected, f'{kind} coordinate coverage differs'
    assert len(sessions) == 1, f'{kind} mixes sessions'
    return {'records': len(records), 'coordinate_coverage_exact': True,
            'session': next(iter(sessions.values())),
            'tensor_layouts': [{'shape': list(shape), 'dtype': dtype, 'count': count}
                               for (shape, dtype), count in shapes.items()],
            'payload_tensor_bytes_total': sum(r['tensor_bytes'] for r in records),
            'first': records[0], 'last': records[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = {key: path.read_bytes() for key, path in PATHS.items()}
    cost = pickle.loads(raw['cost'])
    docs = {key: json.loads(value) for key, value in raw.items() if key != 'cost'}
    rows = [(name, formats[FMT]) for name, formats in cost['costs'].items() if FMT in formats]
    assert rows, 'no priced format rows'
    probes, activations, weights = {}, {}, Counter()
    totals = [0.0] * 4
    ranked = []
    for name, row in rows:
        probe, operator = row['probe_identity'], row['joint_operator_identity']
        probes[digest(probe)] = probe
        activations[digest(operator['activation'])] = operator['activation']
        weights[(tuple(operator['source_weight']['shape']), operator['source_weight']['dtype'])] += 1
        components = row['signed_components_per_probe']
        assert len(components) == 4
        values = [float(c['activation']) for c in components]
        totals = [a + b for a, b in zip(totals, values)]
        # Same additive A-side statistic used by the historical concentration scan.
        activation_price = 0.5 * sum(v * v for v in values) / len(values)
        ranked.append({'unit': name, 'activation_price': activation_price,
                       'activation_components': values,
                       'operator_identity_sha256': row['joint_operator_identity_sha256'],
                       'source_weight': operator['source_weight'],
                       'row_keys': sorted(row)})
    assert len(probes) == 1, 'L05 format rows disagree on probe identity'
    probe = next(iter(probes.values()))
    compact_probe = {key: value for key, value in probe.items() if key != 'source_model'}
    compact_probe['source_model_digest'] = digest(probe['source_model'])
    slice_doc, handoff, plan = docs['slice'], docs['handoff'], docs['plan']
    assert probe['calibration_sha256'] == slice_doc['run_identity']['calibration_sha256']
    assert probe['calibration_shape'] == slice_doc['run_identity']['calibration_shape']
    assert probe['n_probes'] == slice_doc['run_identity']['n_probes'] == handoff['n_probes']
    assert probe['seed_base'] == slice_doc['run_identity']['seed_base']
    summary = {
        'schema': 'prismaquant.research.glm_l05_metadata_audit.v1',
        'scope': 'existing metadata and priced rows; no model/capture tensors loaded',
        'not_established': 'independent full-model cotangent/activation row correctness',
        'inputs': {key: {'path': str(PATHS[key]), 'sha256': hashlib.sha256(value).hexdigest(),
                         'bytes': len(value)} for key, value in raw.items()},
        'cost_top_keys': sorted(cost), 'format': FMT, 'priced_units': len(rows),
        'probe_identity': compact_probe, 'probe_digest': next(iter(probes)),
        'activation_identities': list(activations.values()),
        'source_weight_layouts': [{'shape': list(shape), 'dtype': dtype, 'count': count}
                                  for (shape, dtype), count in weights.items()],
        'top_a_rows': sorted(ranked, key=lambda row: -row['activation_price'])[:10],
        'a_price_sum': sum(row['activation_price'] for row in ranked),
        'a_components_sum': totals,
        'slice_run_identity': slice_doc['run_identity'],
        'handoff_metadata': {key: value for key, value in handoff.items()
                             if key not in ('activation_entries', 'owner_states')},
        'boundaries': audit_entries(slice_doc['boundary_entries']['5'], boundary=5,
                                    kind='boundary', batches=handoff['n_batches']),
        'cotangents': audit_entries(handoff['activation_entries'], boundary=6,
                                    kind='cotangent', batches=handoff['n_batches'], probes=4),
        'plan_execution': plan['execution'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(f'PASS: {len(rows)} L05 rows, one probe identity, exact 512 x 4 coordinate coverage')
    print(f'Additive A price {summary["a_price_sum"]:.12g}; result {args.output}')


if __name__ == '__main__':
    main()
