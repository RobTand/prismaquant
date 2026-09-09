"""Opt-in routed allocation policy, exercised before any stack aggregation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from prismaquant import allocator_candidates as ac
from prismaquant import format_registry as fr
from prismaquant import serving_profiles as sp
from prismaquant import tessera_menu as tm
from test_tessera_candidate_context import _Context, _fixture_specs

PROFILE = 'glm_packed_research_sm121'
ROUTED = 'model.language_model.layers.4.mlp.experts.17.gate_proj'
STACK = 'model.language_model.layers.4.mlp.experts.gate_up_proj'
DENSE = 'model.language_model.layers.0.mlp.down_proj'
E4M3 = ['TESSERA_E4M3_K1_R256', 'TESSERA_E4M3_K1_R1281', 'TESSERA_E4M3_K1_R2048']
OTHER = ['TESSERA_E2M1_K2_R896', 'TESSERA_BF16_K1_R1024', 'NVFP4', 'FP8_E4M3']


@pytest.fixture(autouse=True)
def readable_menu(monkeypatch):
    monkeypatch.setenv(tm.MENU_MODE_ENV, tm.MENU_READABLE)


def test_profile_is_explicit_research_and_inherits_only_existing_gates():
    profile = sp.load_serving_profile(PROFILE)
    base = sp.load_serving_profile('tessera_research_sm121')
    assert profile.emulation_only and profile.export_lane is None
    assert profile.target_platform == base.target_platform == 'sm_121'
    assert profile.tensor_parallel == base.tensor_parallel
    assert profile.tensor_parallel.world_size == 1
    assert profile.shape_rules == base.shape_rules
    assert not profile.supports_per_role_expert_schemes
    assert not base.format_rules
    assert sp.load_serving_profile(None).id == 'research'
    with pytest.raises(ValueError, match='emulation|export|lane'):
        sp.require_profile_export_lane(PROFILE, "tessera")


@pytest.mark.parametrize('qname', [ROUTED, STACK, 'model.language_model.layers.4.mlp.experts',
                                   'mlp.experts.0.down_proj'])
@pytest.mark.parametrize('fmt', E4M3 + ['BF16'])
def test_routed_names_accept_legal_family_rungs_and_bf16(qname, fmt):
    assert sp.check_serving_format(PROFILE, qname, fmt).legal


@pytest.mark.parametrize('fmt', OTHER + ['UNKNOWN', 'TESSERA_E4M3_K1_R255',
    'TESSERA_E4M3_K1_R2049', 'TESSERA_E4M3_K0_R512', 'TESSERA_BOGUS_K1_R512',
    'TESSERA_E4M3_K1_R*', 'TESSERA_E4M3_K1_G0', 'TESSERA_E4M3_K1_R512_suffix'])
def test_routed_names_refuse_other_families_unknown_names_and_invalid_rungs(fmt):
    decision = sp.check_serving_format(PROFILE, ROUTED, fmt)
    assert not decision.legal
    assert decision.rule == 'glm_routed_packed_research_families'


@pytest.mark.parametrize('values', ['TESSERA_E4M3_K1', None, {}, [None],
    ['TESSERA_BOGUS_K1'], ['TESSERA_E4M3_K0'], ['E4M3_K1'],
    ['TESSERA_E4M3_K1_R512'], ['TESSERA_E4M3_K01'], ['TESSERA_E4M3_K1*']])
def test_family_configuration_refuses_noncanonical_or_malformed_values(values):
    with pytest.raises(ValueError):
        sp.ServingFormatRule.from_dict(dict(id='bad', allow_tessera_families=values))


def test_family_union_retains_exact_denials_and_does_not_enable_wildcards():
    rule = sp.ServingFormatRule.from_dict(dict(id='union',
        allow_formats=['BF16'], allow_tessera_families=['TESSERA_E4M3_K1'],
        deny_formats=[E4M3[1]]))
    assert rule.check(ROUTED, 'BF16').legal
    assert rule.check(ROUTED, E4M3[0]).legal
    assert not rule.check(ROUTED, E4M3[1]).legal
    exact = sp.ServingFormatRule.from_dict(dict(id='exact', allow_formats=['TESSERA_E4M3_K1_R*']))
    assert not exact.check(ROUTED, E4M3[0]).legal


def candidate_table(names, specs, *, shape=(2048, 4096), packed=False):
    stats = {name: dict(h_trace=1., n_params=shape[0]*shape[1]*(2 if packed else 1),
        in_features=shape[1], out_features=shape[0], num_experts=2 if packed else 0)
        for name in names}
    costs = {name: {spec.name: dict(output_mse=0. if spec.name == 'BF16' else 1e-4,
        weight_mse=0. if spec.name == 'BF16' else 1e-4, output_mse_measured=True)
        for spec in specs} for name in names}
    return stats, costs


@pytest.mark.parametrize('fmt', E4M3 + OTHER + ['UNKNOWN'])
@pytest.mark.parametrize('qname,packed', [(ROUTED, False), (STACK, True),
    ('model.language_model.layers.4.mlp.experts', True)])
def test_actual_candidate_filter_masks_before_routed_grouping(fmt, qname, packed):
    spec = replace(fr.get_format('BF16'), name=fmt) if fmt == 'UNKNOWN' else fr.get_format(fmt)
    specs = [spec, fr.get_format('BF16')]
    stats, costs = candidate_table([qname], specs, packed=packed)
    masks = []
    result = ac.build_candidates(stats, costs, specs, target_profile=PROFILE,
        source_manifest={qname: 'bf16'}, mask_records=masks)
    expected = {'BF16', fmt} if fmt in E4M3 else {'BF16'}
    assert {candidate.fmt for candidate in result[qname]} == expected
    if fmt not in E4M3:
        assert [(m['format'], m['reason']) for m in masks] == [(fmt, 'profile_mismatch')]


@pytest.mark.parametrize('qname', [DENSE, 'model.layers.0.mlp.shared_experts.gate_proj',
    'model.layers.0.mlp.experts_extra.down_proj'])
@pytest.mark.parametrize('fmt', ['TESSERA_E2M1_K2_R896', 'TESSERA_BF16_K1_R1024', E4M3[1]])
def test_dense_readable_candidates_remain_available(qname, fmt):
    specs = [fr.get_format(fmt), fr.get_format('BF16')]
    stats, costs = candidate_table([qname], specs)
    result = ac.build_candidates(stats, costs, specs, target_profile=PROFILE,
        source_manifest={qname: 'bf16'})
    assert {candidate.fmt for candidate in result[qname]} == {fmt, 'BF16'}


def test_family_allowance_does_not_bypass_shape_refusal():
    from prismaquant.tessera_formats import TesseraFormatError
    specs = [fr.get_format(E4M3[-1]), fr.get_format('BF16')]
    stats, costs = candidate_table([ROUTED], specs, shape=(2048, 4095))
    # At TP=1 an unsliced window reaches the footprint's superblock check.
    # Preserve that existing hard refusal, rather than assuming a TP mask.
    with pytest.raises(TesseraFormatError, match='multiple of the 256-column'):
        ac.build_candidates(stats, costs, specs, target_profile=PROFILE,
            source_manifest={ROUTED: 'bf16'})


def test_family_allowance_does_not_bypass_source_ceiling():
    decision = ac.check_format_applicability((2048, 4096), E4M3[-1], qname=ROUTED,
        source_kind='fp8', target_profile=PROFILE)
    assert not decision.legal
    assert decision.reason == ac.SOURCE_BPP_EXCEEDED_REASON


def test_actual_candidate_filter_keeps_per_unit_context_gate(monkeypatch):
    fmt = E4M3[1]
    specs = _fixture_specs(monkeypatch, fmt, 'BF16')
    names = [ROUTED, STACK]
    stats, costs = candidate_table(names, specs)
    accepted = _Context(structure='routed_moe')
    refused = replace(accepted, runtime_image='unqualified/image')
    calls = []
    def admission(_fmt, *, serving_context=None):
        calls.append(serving_context)
        return SimpleNamespace(requires_serving_context=True,
            admits=lambda _mode: serving_context == accepted,
            detail='controlled per-unit context refusal')
    monkeypatch.setattr(tm, 'route_admission', admission)
    # Only the attestation answer is controlled. The real profile, shape,
    # source-footprint and candidate filtering paths all run.
    monkeypatch.setattr(ac, 'serving_lane_route', lambda *_a, **_k: None)
    masks = []
    result = ac.build_candidates(stats, costs, specs, target_profile=PROFILE,
        source_manifest=dict.fromkeys(names, 'bf16'),
        context_by_unit={ROUTED: accepted, STACK: refused}, mask_records=masks)
    assert {c.fmt for c in result[ROUTED]} == {fmt, 'BF16'}
    assert {c.fmt for c in result[STACK]} == {'BF16'}
    assert calls == [accepted, refused]
    assert [(m['qname'], m['reason']) for m in masks] == [(STACK, 'tessera_serving_context')]


def test_routed_allow_list_equals_the_pinned_contracts_routed_moe_families():
    """The allow-list must be the contract's answer, not a copy of it.

    ``allow_tessera_families`` is typed into the spec file and read by
    ``ServingFormatRule.check`` as a gate input, so it is a producer-side
    field stating which families the serving runtime executes for routed
    experts. Principle 14 says such a field is derived from the runtime's own
    machine-readable table or refused. Deriving the whole list is the larger
    change; this test is the smaller one that makes the copy honest, by
    asserting it still equals the projection it was copied from.

    What it catches is drift in the direction nothing else does. If Tessera
    publishes a ``routed_moe`` cell for a second family, the runtime executes
    that rung and this profile keeps refusing it, silently and forever: no
    export gate fires, because over-refusal is not a refusal anyone sees. The
    mirror case is already contained, but by accident rather than design -- a
    withdrawn cell leaves the profile admitting a family whose
    ``route_admission`` then resolves ``unattested``, so export still fails
    closed.

    No skip when the pinned contract is unavailable, which is the same stance
    the rest of the ``test_tessera_*`` family already takes. ``load_tessera_
    contract`` returns ``None`` both when Tessera is absent and when the
    installed Tessera is not the pinned commit, because it fails closed rather
    than reading whatever is installed. A test that skipped on either would
    pass quietly in exactly the environment where the drift it watches for
    would go unseen. ``tests/test_ci_tessera_install.py`` and
    ``tests/test_tessera_serving_pin.py`` are the gates for the install itself.
    """
    from prismaquant import lane_eligibility as le
    from prismaquant import tessera_runtime_contract as trc

    contract = trc.load_tessera_contract()
    assert contract is not None, (
        "no pinned Tessera contract: it is either not installed or not the "
        f"pinned commit ({trc.TESSERA_DEV_PIN_COMMIT[:12]}), "
        "and the loader fails closed on both. This test derives from that "
        "contract rather than restating it; tests/test_ci_tessera_install.py "
        "and tests/test_tessera_serving_pin.py gate the install itself")
    profile = sp.load_serving_profile(PROFILE)
    published = {
        cell.family for cell in contract.cells
        if cell.structure == le.STRUCTURE_ROUTED_MOE
        and cell.platform == profile.target_platform
    }
    rule, = [r for r in profile.format_rules
             if r.id == 'glm_routed_packed_research_families']
    assert set(rule.allow_tessera_families) == published, (
        f"{PROFILE} allows {sorted(rule.allow_tessera_families)} for routed "
        f"experts, but the pinned contract "
        f"({contract.commit[:12]}, {contract.lane_schema}) publishes "
        f"{sorted(published)} as its routed_moe families on "
        f"{profile.target_platform}")
    # Not vacuous in either direction: the projection must actually select,
    # and it must exclude a family the contract publishes only as dense.
    assert published
    assert {cell.family for cell in contract.cells} - published
