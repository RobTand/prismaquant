"""GLM MLA's absorbed kv_b projection prices BF16 activations only (#2096)."""
from types import SimpleNamespace

import pytest

from prismaquant import tessera_campaign as campaign, tessera_menu as menu
from prismaquant.model_profiles.glm5_next import Glm5NextProfile


KV = "model.language_model.layers.3.self_attn.kv_b_proj"
Q = "model.language_model.layers.3.self_attn.q_b_proj"
FAMILIES = ("TESSERA_BF16_K1", "TESSERA_E4M3_K1", "TESSERA_E2M1_K2")


@pytest.fixture
def expanded(monkeypatch):
    calls = []

    def expand(shape, **kwargs):
        families = tuple(f.name for f in kwargs.get("families", menu.menu_families()))
        calls.append(families)
        return [SimpleNamespace(family=f, route_status="unattested") for f in families]

    from prismaquant.tessera_formats import get_tessera_family
    monkeypatch.setattr(menu, "menu_families", lambda: tuple(map(get_tessera_family, FAMILIES)))
    monkeypatch.setattr(menu, "expand_tessera_menu", expand)
    return calls


def test_absorbed_projection_keeps_t16_while_equal_shape_q_retains_full_menu(expanded):
    names = [KV, Q, "model.language_model.layers.7.self_attn.kv_b_proj"]
    weights = {name: SimpleNamespace(shape=(512, 256)) for name in names}
    rows = campaign.expand_menus_for_targets(weights, names, mode="readable",
        tp_degree=1, parallel_kind="none", profile=Glm5NextProfile())
    assert [r.family for r in rows[KV]] == ["TESSERA_BF16_K1"]
    assert [r.family for r in rows[Q]] == list(FAMILIES)
    assert rows[KV] is rows[names[2]] and rows[KV] is not rows[Q]
    assert len(expanded) == 2
    assert all(r.route_status == "unattested" for values in rows.values() for r in values)


@pytest.mark.parametrize("dense,expected", [
    (["TESSERA_BF16_K1", "TESSERA_E4M3_K1"], ["TESSERA_BF16_K1"]),
    (["TESSERA_E4M3_K1"], []),
])
def test_structural_restriction_intersects_unit_policy_without_reopening(expanded, dense, expected):
    policy = {"schema": campaign.FAMILY_RESTRICTION_SCHEMA, "dense": dense,
              "routed_moe": ["TESSERA_E4M3_K1"]}
    rows = campaign.expand_menus_for_targets({KV: SimpleNamespace(shape=(512, 256))}, [KV],
        mode="readable", tp_degree=1, parallel_kind="none", profile=Glm5NextProfile(),
        family_restriction=policy, structure_by_unit={KV: "dense"})
    assert [r.family for r in rows[KV]] == expected


def test_absent_profile_preserves_existing_menu(expanded):
    rows = campaign.expand_menus_for_targets({KV: SimpleNamespace(shape=(512, 256))}, [KV],
        mode="readable", tp_degree=1, parallel_kind="none")
    assert [r.family for r in rows[KV]] == list(FAMILIES)


@pytest.mark.parametrize("name", [KV, "model.layers.3.self_attn.kv_b_proj",
    "language_model.model.layers.3.self_attn.kv_b_proj",
    "model.layers.45.self_attn.kv_b_proj.weight", "self_attn.kv_b_proj"])
def test_unit_policy_recognizes_checkpoint_live_and_served_spellings(name):
    profile = Glm5NextProfile()
    assert profile.tessera_pricing_families(name) == ("TESSERA_BF16_K1",)
    assert profile.is_pinned_name(name)


@pytest.mark.parametrize("name", [Q, "model.layers.3.mlp.kv_b_proj",
    "model.layers.3.self_attn.kv_b_proj_extra", "lm_head"])
def test_unit_policy_does_not_constrain_other_linears(name):
    assert Glm5NextProfile().tessera_pricing_families(name) is None


@pytest.mark.parametrize("family", FAMILIES)
def test_seed_obeys_profile_arithmetic_without_global_restriction(family):
    fmt = f"{family}_R896"
    state = {"anchors": [{"qname": KV, "family": family,
        "format_name": fmt, "body_rate_q256": 896}], "wire_records": {fmt: {}}}
    kwargs = dict(family_restriction=None, structure_by_unit=None, profile=Glm5NextProfile())
    if family == "TESSERA_BF16_K1":
        campaign.require_seed_family_scope(KV, state, **kwargs)
    else:
        with pytest.raises(RuntimeError, match="unit family restriction"):
            campaign.require_seed_family_scope(KV, state, **kwargs)


def test_streamed_capture_uses_the_profile_unit_menu(expanded, monkeypatch):
    import torch
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.kv_b_proj = torch.nn.Linear(256, 512, device="meta", bias=False)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([layer])
    name = KV.replace("layers.3", "layers.0")
    runner = SimpleNamespace(model=model, layer_index_for_qname=lambda name: 0)
    profile = Glm5NextProfile()
    expand = campaign.expand_menus_for_targets

    class MenuPlanned(Exception):
        pass

    def stop_after_menu(weights, targets, **kwargs):
        rows = expand(weights, targets, **kwargs)
        assert [r.family for r in rows[name]] == ["TESSERA_BF16_K1"]
        raise MenuPlanned

    monkeypatch.setattr(campaign, "expand_menus_for_targets", stop_after_menu)
    with pytest.raises(MenuPlanned):
        campaign._run_streamed_calibration(SimpleNamespace(tp_degree=1), runner, profile,
            mode="readable", population=campaign.ExpertPopulation(members=(), declared={},
                packed_in_scope={}, omitted_outside_layer_stride={}),
            dense_targets=[name], expert_targets=[], scope_groups={name: [name]},
            tokens=[], corpus_text="", census=None, context_by_unit=None,
            attention_implementation="eager", capture_runtime={})
