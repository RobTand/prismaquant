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
