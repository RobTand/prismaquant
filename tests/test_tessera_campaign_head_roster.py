"""Explicit head lifts use the existing scoped roster; defaults stay pinned."""
from types import SimpleNamespace

import pytest

from prismaquant import tessera_campaign as tc
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from test_tessera_campaign_pinned_roster import _args, _census, _names


@pytest.mark.parametrize('only', [False, True])
def test_explicit_glm_head_lift_is_recorded_as_a_scoped_unit(only):
    profile = Glm5NextProfile()
    names = _names()
    roster = tc.campaign_roster(names['all'], profile, allow_pinned='lm_head',
                                pinned_roster_only=only)
    assert roster.lifted == ('lm_head',)
    assert set(roster.dense) == ({'lm_head'} if only else
                                set(names['dense'] + names['shared'] + ['lm_head']))
    assert all('embed' not in name for name in roster.dense)
    assert profile.is_pinned_name('lm_head'), 'the profile itself must not be unpinned'


@pytest.mark.parametrize('token', ['head', 'lm_head', 'language_model.lm_head'])
def test_head_alias_lift_matches_the_existing_cache_and_allocator_policy(token):
    profile = SimpleNamespace(
        lm_head_name=lambda: 'head', probe_linear_exclude_extra=lambda: None,
        is_pinned_name=lambda name: name in ('head', 'router'))
    roster = tc.campaign_roster(['body', 'head', 'embed_tokens', 'router'], profile,
                                allow_pinned=token, pinned_roster_only=True)
    assert roster.dense == roster.lifted == ('head',)
    assert roster.pinned == ('router',)


def test_head_census_binds_the_lift_and_refuses_body_scope(tmp_path):
    profile = Glm5NextProfile()
    roster = tc.campaign_roster(_names()['all'], profile, allow_pinned='lm_head',
                                pinned_roster_only=True)
    args = _args(allow_pinned='lm_head', pinned_roster_only=True)
    path, payload = _census(tmp_path, args, roster)
    assert payload['pinned_roster']['lifted'] == ['lm_head']
    tc.load_calibration_census(path, args=args)
    with pytest.raises(RuntimeError, match='must be the same scope'):
        tc.load_calibration_census(path, args=_args())


def test_an_explicit_head_token_still_refuses_if_the_head_is_absent():
    with pytest.raises(ValueError, match='lift no profile-pinned Linear'):
        tc.campaign_roster(['model.layers.0.mlp.down_proj'], Glm5NextProfile(),
                           allow_pinned='lm_head', pinned_roster_only=True)


def test_unset_head_alias_roster_remains_the_previous_roster():
    profile = SimpleNamespace(
        lm_head_name=lambda: 'head', probe_linear_exclude_extra=lambda: None,
        is_pinned_name=lambda name: name in ('head', 'router'))
    roster = tc.campaign_roster(['body', 'lm_head', 'head', 'embed_tokens', 'router'], profile)
    assert roster.dense == ('body',)
    assert roster.pinned == ('head', 'router')
    assert roster.lifted == ()


def test_current_producer_body_plan_still_refuses_a_quantized_head(tmp_path):
    import json
    import torch
    from safetensors.torch import save_file
    from prismaquant import tessera_plan_writer as writer

    # A real source-header census through the pinned producer, not a fabricated
    # positive dense lane or a manual body/head classification in PrismaQuant.
    (tmp_path / 'config.json').write_text(json.dumps({
        'architectures': ['Glm5NextForConditionalGeneration'], 'model_type': 'glm5_next'}))
    save_file({'lm_head.weight': torch.ones(32, 32),
               'model.language_model.layers.0.mlp.down_proj.weight': torch.ones(32, 32)},
              tmp_path / 'model.safetensors')
    surface = writer.tessera_surface()
    _, shapes, _, _ = surface.quantizable(tmp_path)
    assert 'lm_head.weight' not in shapes
    with pytest.raises(writer.PlanError, match='absent from the producer.*body projection'):
        writer.plan_from_assignment(
            {'lm_head': {'tessera_format': 'TESSERA_E4M3_K1_R1024'}},
            shapes, {}, {}, model=tmp_path, cover='as-allocated',
            allow_disagreement=False, surface=surface, with_control=False)
