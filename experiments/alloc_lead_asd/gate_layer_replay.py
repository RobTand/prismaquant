"""Gate full-draw replay slices against recorded Stage B unit totals (pq2572).

Reads every slice ``rows.pt`` for one layer, sums fixed-cotangent signed
rows per (unit, probe) across the tiled sequence ranges, and compares each
target unit against the layer quantum's recorded ``cost.pkl`` activation
component at 1e-3 relative. It also aggregates the banked per-row facts
for the target experts into per-sequence block facts. It changes no price
and names no cause; the report reads its gate record.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
from pathlib import Path

import torch

GATE_REL_TOL = 1e-3
GATE_ABS_FLOOR = 1e-12


def load_rows(path, digest):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != digest:
        raise RuntimeError(f'slice rows bytes differ from the indexed digest: {path}')
    bank = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slices-index', type=Path, required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--units', required=True)
    parser.add_argument('--cost', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    index = json.loads(args.slices_index.read_text())
    slices = [row for row in index['slices'] if row['layer'] == args.layer]
    assert slices
    ranges = sorted(tuple(row['sequences']) for row in slices)
    flat = [sequence for group in ranges for sequence in group]
    assert flat == list(range(512)), 'slice ranges do not tile batches 0-511 exactly once'
    units = args.units.split(',')
    assert units
    cost_raw = args.cost.read_bytes()

    quantum = pickle.loads(cost_raw)
    assert quantum.get('n_probes') == 4
    per_unit, per_sequence, facts = {}, {}, []
    for entry in slices:
        group = Path(entry['directory'])
        control = json.loads((group / 'local-control.json').read_text())
        assert control['same_primal_operator_gate_passed'] and control['coordinate_coverage_exact']
        assert control['max_relative_to_operator_rms'] <= 1e-4
        assert json.loads((group / 'result.json').read_text())['passed']
        binding = json.loads((group / 'binding.json').read_text())
        assert binding['layer'] == args.layer and binding['sequences'] == entry['sequences']
        bank = load_rows(group / 'rows.pt', control['rows']['sha256'])
        assert sorted(bank.get('fact_units', [])) == sorted(binding.get('fact_units', []))
        for row in bank['rows']:
            key = (row['unit'], row['probe'])
            if row['unit'] not in units:
                continue
            total = per_unit.setdefault(key, 0.0)
            values = row['signed_rows'].to(torch.float64)
            assert torch.isfinite(values).all()
            per_unit[key] = total + float(values.sum())
            coords = row['coordinates']
            for pos in range(len(values)):
                sequence = int(coords[pos][0])
                assert sequence in entry['sequences']
                seq_key = (row['unit'], row['probe'], sequence)
                per_sequence[seq_key] = per_sequence.get(seq_key, 0.0) + float(values[pos])
        for fact in bank.get('facts', []):
            assert fact['unit'] in units and fact['sequence'] in entry['sequences']
            for name in ('route_weight', 'input_amax', 'cotangent_norm', 'along_output', 'signed_row'):
                assert math.isfinite(fact[name])
            facts.append(fact)
    gate = []
    for unit in units:
        cell = quantum['costs'][unit]['TESSERA_E4M3_K1_R1024']
        assert cell['probe_ids'] == [7000, 7001, 7002, 7003]
        for probe in range(4):
            expected = cell['signed_components_per_probe'][probe]['activation']
            replayed = per_unit[(unit, probe)]
            denom = max(abs(expected), GATE_ABS_FLOOR)
            residual = abs(replayed - expected)
            relative = residual / denom
            gate.append(dict(unit=unit, probe=probe, replayed=replayed, stage_b=expected,
                             residual=residual, relative=relative, passed=bool(relative <= GATE_REL_TOL)))
    blocks = {}
    for unit in units:
        series = sorted(((sequence, per_sequence[(unit, probe, sequence)])
                         for probe in range(4) for sequence in range(512)
                         if (unit, probe, sequence) in per_sequence),
                        key=lambda row: row[1], reverse=True)
        values = [value for _, value in series]
        total = math.fsum(values)
        concentrated = sorted(values, reverse=True)
        blocks[unit] = dict(total=total,
            top1_sequence=int(series[0][0]), top1_value=float(series[0][1]),
            top1_share=float(series[0][1] / total) if total else None,
            top8_share=float(sum(concentrated[:8]) / total) if total else None,
            top32_share=float(sum(concentrated[:32]) / total) if total else None,
            nonzero_sequences=len([value for value in values if value != 0.0]))
    fact_summary = {}
    for unit in units:
        rows = [fact for fact in facts if fact['unit'] == unit]
        amax = sorted(fact['input_amax'] for fact in rows)
        route = sorted(fact['route_weight'] for fact in rows)
        along = [fact['along_output'] for fact in rows]
        fact_summary[unit] = dict(rows=len(rows), probes=sorted({fact['probe'] for fact in rows}),
            sequences=sorted({fact['sequence'] for fact in rows}),
            input_amax_max=float(amax[-1]) if amax else None,
            input_amax_p99=float(amax[math.ceil(0.99 * len(amax)) - 1]) if amax else None,
            input_amax_median=float(amax[len(amax) // 2]) if amax else None,
            route_weight_min=float(route[0]) if route else None,
            route_weight_median=float(route[len(route) // 2]) if route else None,
            along_output_mean=float(math.fsum(along) / len(along)) if along else None)
    result = dict(scope=f'pq2572 layer-{args.layer} replay gate against the recorded Stage B quantum',
        gate_rel_tol=GATE_REL_TOL, gate_abs_floor=GATE_ABS_FLOOR,
        slices=len(slices), units=units, gate=gate,
        all_passed=all(row['passed'] for row in gate),
        per_sequence_blocks=blocks, fact_summary=fact_summary,
        cost_sha256=hashlib.sha256(cost_raw).hexdigest(),
        cost_path=str(args.cost))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(layer=args.layer, slices=len(slices), units=units,
        all_passed=result['all_passed'],
        worst_relative=max(row['relative'] for row in gate),
        facts=len(facts)), indent=2))


if __name__ == '__main__':
    main()
