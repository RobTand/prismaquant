"""CPU follow-up: physical routed-family scope over the accepted byte analysis."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
ROOT = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908')
OUT = ROOT/'first-release-rate-band-01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from prismaquant.tessera_campaign import _checkpoint_identity_api
    from prismaquant.production_weight_cache import _production_cache_source_sha256
    from prismaquant.tessera_formats import get_tessera_family
    from prismaquant.tessera_footprint import tessera_exact_bits_for_shape
    prior_path = OUT/'derivation.json'
    assert sha(prior_path) == 'bc2edd132b4646063f10bb5f507a260a227081ec2443cac4da6618fbbd5cfed6'
    prior = json.loads(prior_path.read_text())
    budget_path = ROOT/'exl3-first-artifact-01/root-common-surface-header-audit.json'
    assert sha(budget_path) == prior['budget_audit_sha256']
    budget = json.loads(budget_path.read_text())
    structural = Counter((u['kind'], tuple(u['shape'])) for u in budget['units'])
    assert sum(n for (kind, _), n in structural.items() if kind == 'routed') == 36288
    assert sum(n for (kind, _), n in structural.items() if kind == 'dense') == 135
    family = get_tessera_family('TESSERA_E4M3_K1')
    dense_families = ['EXL3_BF16', 'TESSERA_E4M3_K1', 'TESSERA_BF16_K1', 'TESSERA_E2M1_K2']
    witnesses = {}
    for dense in dense_families:
        points = []
        for q in range(832, 1089):
            total = 0
            for (kind, shape), count in structural.items():
                if kind == 'dense' and dense == 'EXL3_BF16':
                    bits = 16*shape[0]*shape[1]
                else:
                    selected = family if kind == 'routed' else get_tessera_family(dense)
                    rung = 896 if selected.name == 'TESSERA_E2M1_K2' else q
                    bits = tessera_exact_bits_for_shape(selected, rung, shape)
                assert bits.denominator == 1 and int(bits) % 8 == 0
                total += count*int(bits)//8
            points.append(dict(q256=q, bytes=total,
                bpp=float(Fraction(8*total, prior['budget']['common']['parameters']))))
        affordable = [p for p in points if p['bytes'] <= prior['budget']['common']['tensor_payload_bytes']]
        witnesses[dense] = dict(highest_uniform_routed_affordable=affordable[-1],
                               endpoints=[points[0], points[-1]])
    journal = ROOT/'full-anchor-preparation-03/workspace/rows/row-0087/cost.anchors.json'
    identity = json.loads(journal.read_text())['identity']
    current = dict(encoder_source_sha256=_checkpoint_identity_api().encoder_source_sha256(),
                   prismaquant_source_sha256=_production_cache_source_sha256())
    assert all(identity[k] == v for k, v in current.items()), 'frozen pricing source moved'
    result = dict(schema='prismaquant.glm_first_release_physical_family_analysis.v1',
        input_sha256=sha(prior_path), source=current,
        structure_counts=[dict(kind=k, shape=s, count=n) for (k,s),n in sorted(structural.items())],
        initial_anchors=dict(routed=36288*2, dense=135*5, total=36288*2+135*5),
        selected_band=[832,1088], witnesses=witnesses,
        limits=['Routed physical builder supports E4M3 only; uniform BF16/E2M1 results in prior derivation are abstract byte arithmetic, not routed candidates.',
                'EXL3_BF16 dense witness retains exactly the baseline dense payload charge; it does not propose a Tessera BF16 wire.',
                'Uniform points are feasibility witnesses only; each Linear still needs measured quality/rate allocation within its serving stack.',
                'Intrinsic bytes include every accountant plane; outer wire envelopes and final exported files still need an artifact-level byte audit.',
                'This does not attest a serving release or launch GPU work.'])
    path = OUT/'physical-family-analysis.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(output=str(path), sha256=sha(path), result=result), sort_keys=True))


if __name__ == '__main__':
    main()
