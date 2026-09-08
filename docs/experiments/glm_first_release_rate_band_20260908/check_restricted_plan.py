"""PB CPU metadata validation of the actual restricted menu over the full census."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
ROOT = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908')
OUT = ROOT/'first-release-rate-band-01'
POLICY = dict(schema='prismaquant.tessera_campaign_family_restriction.v1',
    dense=['TESSERA_BF16_K1', 'TESSERA_E2M1_K2', 'TESSERA_E4M3_K1'],
    routed_moe=['TESSERA_E4M3_K1'])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    from prismaquant.model_profiles import detect_profile
    from prismaquant.tessera_campaign import expand_menus_for_targets, round_one_rates
    from prismaquant.tessera_expert_projection import bind_expert_projection
    from prismaquant.tessera_serving_scope import unit_structure_from_stats
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    before_path = OUT/'derivation.json'
    assert sha(before_path) == 'bc2edd132b4646063f10bb5f507a260a227081ec2443cac4da6618fbbd5cfed6'
    before = json.loads(before_path.read_text())
    census_path = ROOT/'workspace/census.json'
    assert sha(census_path) == before['source_census_sha256']
    census = json.loads(census_path.read_text())
    budget_path = ROOT/'exl3-first-artifact-01/root-common-surface-header-audit.json'
    assert sha(budget_path) == before['budget_audit_sha256']
    budget = json.loads(budget_path.read_text())
    projection = census['expert_projection']
    declared = {stack: {name: (row['rows'], row['cols']) for name, row in units.items()}
                for stack, units in projection['stacks'].items()}
    bind_expert_projection(projection['producer'], declared=declared)
    topology = {name: dict(router_path=None, expert_id=None) for name in census['dense_targets']}
    for stack, units in declared.items():
        count = projection['producer']['stacks'][stack]['experts']
        for name in units:
            assert name not in topology
            topology[name] = dict(_packed_experts_module=stack, num_experts=count)
    assert set(topology) == set(census['unit_shapes']) and len(topology) == 36423
    profile = detect_profile(census['model'])
    structures = {name: unit_structure_from_stats(name, row, profile) for name, row in topology.items()}
    for unit in budget['units']:
        expected = 'routed_moe' if unit['kind'] == 'routed' else 'dense'
        assert structures[unit['name']] == expected
        assert census['unit_shapes'][unit['name']] == unit['shape']
    weights = {name: SimpleNamespace(shape=shape) for name, shape in census['unit_shapes'].items()}
    menus = expand_menus_for_targets(weights, list(weights), mode='readable', tp_degree=1,
        parallel_kind='none', family_restriction=POLICY, structure_by_unit=structures)
    frequency = Counter(id(rows) for rows in menus.values())
    distinct = {id(rows): rows for rows in menus.values()}
    schedules = {}
    for key, rows in distinct.items():
        by_family = {}
        for entry in rows:
            by_family.setdefault(entry.family, []).append(entry.body_rate_q256)
        schedules[key] = {family: round_one_rates(sorted(rates), band=(832, 1088), anchors=3,
            snap=lambda rate, allowed: min(allowed, key=lambda r: (abs(r-rate), r)))
            for family, rates in by_family.items()}
    for name, rows in menus.items():
        assert set(schedules[id(rows)]) == set(POLICY[structures[name]])
    initial = sum(frequency[key]*sum(len(rates) for rates in families.values())
                  for key, families in schedules.items())
    assert initial == 73251
    witness = sum(frequency[key]*next(r.memory_bytes for r in rows
        if r.format_name == 'TESSERA_E4M3_K1_R1036') for key, rows in distinct.items())
    assert witness == 155546148864 <= before['budget']['common']['tensor_payload_bytes']
    batch = next(b for b in before['batches'] if b['batch_size'] == 8)
    assert batch['rows_fitting'] == 132 and batch['fits_104gib_with_previous_4gib_observer']
    report = dict(schema='prismaquant.glm_restricted_plan_check.v1',
        status='CPU_CHECKED_NO_GPU_SUBMISSION', policy=POLICY, band=[832,1088], batch_width=8,
        groups=len(census['anchor_groups']), units=len(menus), structures=dict(Counter(structures.values())),
        menu_cache_entries=len(distinct), initial_per_linear_anchors=initial,
        common_budget=before['budget']['common'], all_e4m3_q1036_intrinsic_bytes=witness,
        baseline_batch_resource_check=batch, capture_sha256=before['capture_sha256'],
        census_sha256=sha(census_path), common_budget_audit_sha256=sha(budget_path),
        reviewed_campaign_arguments=['--family-restriction', json.dumps(POLICY, sort_keys=True),
            '--rate-band','832,1088','--anchor-batch-size','8'],
        limits=['No sampling or row placement changes; all existing whole groups remain PB-owned quanta.',
            'The source seal for new prices must be issued after integration with the separately owned observer change.',
            'The witness is intrinsic byte feasibility only; final per-Linear allocation, outer-envelope audit, native fit and served quality remain unmeasured.'])
    assert report['groups'] == 132
    path = OUT/'restricted-plan-check.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(output=str(path), sha256=sha(path), initial_anchors=initial,
        units=len(menus), structure_counts=report['structures'], menu_cache_entries=len(distinct))))


if __name__ == '__main__':
    main()
