"""Paired uncertainty on banked historical L0/L1/L2 rung samples.

This consumes the immutable PB identity dump, whose nested identities were
truncated by its producer. It cannot admit current or historical cost rows.
No original weights or full catalog are read, and no allocator menu is changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from prismaquant.cost_stage_checkpoint import atomic_write_bytes
from prismaquant.digests import bytes_sha256hex
from prismaquant.joint_aura import paired_squared_probe_summary
from prismaquant.residency_map import bind_residency_manifest, residency_report
from prismaquant.staged_tier_policy import activate_staged_tier_policy
from prismaquant.staged_whole_file import read_staged_whole_file

ORIGIN_ACTION = 'dbd593f0c895b37d559a89e28516a1901a5bdd560e496db86815b829c589200e'
PAYLOAD_SHA256 = '270673121a107f98c4eb10f5e4cf317cb6f7b36afb3621bf2aac1a4f374c61ae'
RATES = (832, 960, 1024, 1088)
FAMILIES = ('TESSERA_E4M3_K1', 'TESSERA_BF16_K1')
UNITS = tuple(f'model.language_model.layers.{layer}.mlp.down_proj' for layer in range(3))


def inspect_bank(bank):
    if set(bank['units']) != set(UNITS):
        raise ValueError('banked unit coverage differs')
    result = {}
    for unit in UNITS:
        formats = bank['units'][unit]
        families = {}
        for family in FAMILIES:
            rows = {rate: formats[f'{family}_R{rate}'] for rate in RATES}
            reference = rows[RATES[0]]
            for rate, row in rows.items():
                operator = row['joint_operator_identity']
                if (operator['qname'] != unit or operator['format'] != f'{family}_R{rate}'
                        or row['probe_ids'] != [7000, 7001, 7002, 7003]
                        or operator['probe_identity_sha256'] != row['probe_identity_sha256']
                        or any(row[key] != reference[key] for key in
                               ('probe_identity', 'probe_identity_sha256', 'probe_ids', 'hessian_identity'))
                        or any(operator[key] != reference['joint_operator_identity'][key] for key in
                               ('source_weight', 'activation', 'arithmetic'))):
                    raise ValueError('published within-family identity fields differ')
                squared = row['x2_per_probe']
                signed = row['signed_per_probe']
                if (len(squared) != 4 or len(signed) != 4 or any(
                        not math.isfinite(x) or not math.isclose(x*x, y, rel_tol=1e-12, abs_tol=1e-30)
                        for x, y in zip(signed, squared))):
                    raise ValueError('banked signed/squared sample coverage differs')
                moments = paired_squared_probe_summary(squared, [0.] * 4)
                if any(not math.isclose(row[field], moments[expected], rel_tol=1e-12, abs_tol=1e-30)
                       for field, expected in (('predicted_dloss', 'mean_difference'),
                                               ('predicted_dloss_stderr', 'paired_standard_error'))):
                    raise ValueError('published row moments differ from raw samples')
            comparisons = []
            for index, higher in enumerate(RATES):
                for lower in RATES[:index]:
                    a, b = rows[higher], rows[lower]
                    paired = paired_squared_probe_summary(a['x2_per_probe'], b['x2_per_probe'])
                    independent = math.hypot(a['predicted_dloss_stderr'], b['predicted_dloss_stderr'])
                    paired_se = paired['paired_standard_error']
                    comparisons.append(dict(higher_rate=higher, lower_rate=lower, **paired,
                        independence_assumption_standard_error=independent,
                        descriptive_paired_signal_to_se=paired['mean_difference']/paired_se if paired_se else None,
                        descriptive_independent_signal_to_se=paired['mean_difference']/independent if independent else None,
                        positive_on_every_supplied_probe=all(x > 0 for x in paired['difference_per_probe'])))
            families[family] = dict(published_identity_fields_equal=True,
                source_weight=reference['joint_operator_identity']['source_weight'],
                probe_ids=reference['probe_ids'], probe_identity_sha256=reference['probe_identity_sha256'],
                comparisons=comparisons)
        result[unit] = families
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--data-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be unused')
    activate_staged_tier_policy('ram,ssd')
    bind_residency_manifest(args.data_manifest_sha256)
    raw = read_staged_whole_file(args.payload, PAYLOAD_SHA256, label='historical PB rung identity dump')
    prefix, suffix = b'=====L1ID-BEGIN=====\n', b'\n=====L1ID-END=====\n'
    if not raw.startswith(prefix) or not raw.endswith(suffix):
        raise ValueError('historical dump framing differs')
    units = inspect_bank(json.loads(raw[len(prefix):-len(suffix)]))
    report = dict(scope='Historical three-unit/four-probe rung screen; no cost admission',
        origin_action=ORIGIN_ACTION, payload_sha256=PAYLOAD_SHA256, payload_bytes=len(raw),
        units=units, uncertainty_scope='probe sampling conditional on historical fixed calibration',
        limitations=['nested identities are truncated; equality concerns published fields only',
            'no original source, current a6, global cotangent or cost-row qualification',
            'four probe samples do not establish a noise cause or rate-distortion expectation',
            'does not cover the other three old >2sigma outliers or all menu drops',
            'no allocator change, production price, served quality or factor7 claim'],
        residency=residency_report())
    payload = (json.dumps(report, indent=2, allow_nan=False)+'\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(args.output, payload)
    print(json.dumps(dict(report_sha256=bytes_sha256hex(payload), scope=report['scope'],
        units=units, limitations=report['limitations']), allow_nan=False))


if __name__ == '__main__':
    main()
