"""Reconstruct the banked GLM calibration comparison, without a model forward.

Submit this bounded CPU audit through PrismaBuild. It proves saved operand
identities and score semantics, not the calibration variance or estimator bug.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load as load_safetensors


ROOT = Path('/mnt/shared/tessera-measurements')
RUNS = ROOT / 'surrogate-diag-20260929/g3/runs'
ANALYSIS = ROOT / 'codec-decomp-20260930/analysis/t16diag'
CALIBRATION = ROOT / 'glm-canonical-census-20260908/exact-calibration-input-01/calibration_tokens.safetensors'


def read(path, identities):
    raw = path.read_bytes()
    identities.append({'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    identities = []
    metadata = json.loads(read(args.metadata, identities))
    probe = metadata['probe_identity']
    raw_calibration = read(CALIBRATION, identities)
    ids = load_safetensors(raw_calibration)['calibration_ids']
    assert list(ids.shape) == probe['calibration_shape'] == [512, 512]
    assert hashlib.sha256(ids.tobytes()).hexdigest() == probe['calibration_sha256']
    documents, values = {}, {}
    for arm in ('null', 'a8_routed_only_w', 'a8_routed_only_wa'):
        root = RUNS / f'{arm}-cal1a'
        document = documents[arm] = json.loads(read(root / 'result.json', identities))
        read(root / 'per_position_kl.self.npy', identities)
        values[arm] = np.load(root / 'per_position_kl.self.npy', allow_pickle=False)
        assert values[arm].shape == (25, 2047) and np.isfinite(values[arm]).all()
        panel = document['calibration_panel']
        assert panel['calibration_input']['artifact_sha256'] == hashlib.sha256(raw_calibration).hexdigest()
        assert panel['calibration_input']['calibration_sha256'] == probe['calibration_sha256']
        assert panel['context_length'] == 2048
        for window, scored in zip(panel['windows'], document['windows']):
            rows = window['calibration_rows']
            assert window['window_id'] == scored['window_id']
            assert hashlib.sha256(ids[rows].reshape(-1).astype(np.int32).tobytes()).hexdigest() == window['tokens_int32_sha256']
    null = documents['null']
    assert (values['null'] == 0).all() and null['self']['windows_bitwise_equal'] == 25
    for document in documents.values():
        assert document['calibration_panel'] == null['calibration_panel']
        assert document['source_model_identity_sha256'] == null['source_model_identity_sha256']
        assert document['source_initialization_contract_sha256'] == null['source_initialization_contract_sha256']
        assert document['self_teacher'] == null['self_teacher']
        assert document['producer_identity_equals_teacher04'] and document['source_execution_equals_teacher04']
    aside = json.loads(read(ANALYSIS / 'aside_split.json', identities))
    rows = [row for row in aside['units']
            if row['kind'] == 'routed' and row['format'] == 'TESSERA_E4M3_K1_R1024']
    totals = {key: sum(float(row[key]) for row in rows) for key in ('w', 'a', 'cross', 'cost')}
    totals['within_unit_predicted_WA_minus_W'] = totals['cost'] - totals['w']
    bands = {}
    for name, (lo, hi) in {'same_context_0_510': (0, 511),
                           'cross_row_context_511_2046': (511, 2047),
                           'all_0_2046': (0, 2047)}.items():
        w = values['a8_routed_only_w'][:, lo:hi].mean(axis=1)
        wa = values['a8_routed_only_wa'][:, lo:hi].mean(axis=1)
        difference = wa - w
        bands[name] = {'positions_per_window': hi - lo,
                       'w_mean': float(w.mean()), 'wa_mean': float(wa.mean()),
                       'wa_minus_w_mean': float(difference.mean()),
                       'wa_minus_w_standard_error': float(difference.std(ddof=1) / math.sqrt(len(difference))),
                       'paired_window_increments': difference.tolist(),
                       'scope': 'saved finite decoded-weight arms scored against source self-teacher; not standalone source-A KL'}
    same_rows = [window['calibration_rows'][0] for window in null['calibration_panel']['windows']]
    assert same_rows == list(range(0, 100, 4))
    dry_path = ROOT / 'alloc-lead-asd/glm-replay/L05.dry-inputs.json'
    dry = json.loads(read(dry_path, identities))
    result = {'schema': 'prismaquant.research.glm_comparison_audit.v1',
              'inputs': identities, 'null_bitwise_windows': 25,
              'same_context_original_calibration_rows': same_rows,
              'same_context_row_count': len(same_rows),
              'same_context_output_positions': [0, 510],
              'priced_probe_identity': {key: probe[key] for key in (
                  'calibration_sha256', 'calibration_shape', 'token_scope', 'normalization',
                  'noise_layout', 'n_probes', 'seed_base', 'temperature', 'distribution')},
              'routed_r1024_group_count': len(rows), 'routed_r1024_aggregate': totals,
              'bands': bands,
              'saved_cost_row_keys': metadata['top_a_rows'][0]['row_keys'],
              'retained_replay_files': sorted(path.name for path in dry_path.parent.iterdir()),
              'replay_dry_input_failures': dry.get('failures'),
              'exact_subset_price_derivable': False,
              'reason': 'Saved per-unit components are sums over512 complete sequences. '
                        'No per-sequence components survive in the inspected replay directory. '
                        'The selected25 sequences and output positions0..510 cannot be recovered '
                        'from draw sums. Removing activation position511 would also retain the '
                        'output-position511 Fisher contribution in earlier activation cotangents.',
              'not_established': ['subset variance bound', 'actual GLM row correctness',
                                  'cause of the historical factor-seven A/W ratio',
                                  'matched standalone A or joint-network Fisher versus actual KL']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: banked calibration tokens/window bindings, source self-teacher, score bands and aggregate semantics audited')
    print(f'Matched-context source rows: {len(same_rows)}/512; exact subset prediction cannot be derived')


if __name__ == '__main__':
    main()
