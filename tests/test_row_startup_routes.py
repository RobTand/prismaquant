"""Row-local route lookups retain every member gate (PQ #1741)."""
from collections import Counter

from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_formats as formats


def test_contract_refusals_are_queried_once_per_rung_and_structure(monkeypatch):
    calls = Counter()

    def refusal(family, rung, *, structure):
        calls[family, rung, structure] += 1
        return f"{structure} refused" if structure in ("dense", None) else None

    monkeypatch.setattr(formats, "tessera_served_route_refusal", refusal)
    members = ["d2", "p2", "unspecified", "d1", "p1"]
    result = campaign._served_route_refusals(
        "TESSERA_E4M3_K1", [1152, 768], members,
        encode_structure={"d1": "dense", "d2": "dense",
                          "p1": "routed_moe", "p2": "routed_moe"},
        projected_units={"p1": {}, "p2": {}})
    assert result == {str(rung): {
        "reason": "None refused", "members": ["unspecified"],
        "other_reasons": {"dense refused": ["d1", "d2"]}}
        for rung in (768, 1152)}
    assert calls == Counter({("TESSERA_E4M3_K1", rung, structure): 1
                             for rung in (768, 1152)
                             for structure in (None, "dense", "routed_moe")})


def test_unprojected_members_keep_their_gate_with_one_recipe_comparison(monkeypatch):
    calls = Counter()
    monkeypatch.setattr(formats, "tessera_served_route_refusal", lambda *a, **k: None)

    def served(family, rung, *, structure, refuse_unattested):
        assert structure == "routed_moe" and refuse_unattested is False
        calls["served", rung] += 1
        return "routed" if rung == 640 else "research"

    def research(family, rung):
        calls["research", rung] += 1
        return "research"

    monkeypatch.setattr(formats, "tessera_served_wire_recipe", served)
    monkeypatch.setattr(formats, "tessera_wire_recipe", research)
    members = ["u2", "projected", "u1"]
    result = campaign._served_route_refusals(
        "TESSERA_E2M1_K2", [896, 640], members,
        encode_structure=dict.fromkeys(members, "routed_moe"),
        projected_units={"projected": {}})
    assert result == {"640": {
        "reason": "TESSERA_E2M1_K2_R640: a routed unit with no producer projection is "
                  "adopted by Tessera's export intake on the dense receipt, which "
                  "cannot stamp the routed served wire",
        "members": ["u1", "u2"]}}
    assert calls == Counter({(kind, rung): 1 for kind in ("served", "research")
                             for rung in (640, 896)})


def test_a_row_shares_contract_answers_across_distinct_groups(monkeypatch):
    calls = Counter()

    def refusal(family, rung, *, structure):
        calls[family, rung, structure] += 1
        return f"{family} {rung} {structure} refused"

    monkeypatch.setattr(formats, "tessera_served_route_refusal", refusal)
    memo = {}
    for member in ("group1", "group2"):
        for family in ("TESSERA_E2M1_K2", "TESSERA_E4M3_K1"):
            for structure in ("dense", "routed_moe"):
                plan = campaign._served_route_refusals(
                    family, [768], [member], encode_structure={member: structure},
                    projected_units={}, route_cache=memo)
                assert plan == {"768": {"reason": f"{family} 768 {structure} refused",
                                        "members": [member]}}
    assert len(calls) == 4 and all(count == 1 for count in calls.values())


def test_refusal_memo_does_not_survive_a_planning_invocation(monkeypatch):
    reason = ["first"]
    calls = []

    def refusal(family, rung, *, structure):
        calls.append((family, rung, structure))
        return reason[0]

    monkeypatch.setattr(formats, "tessera_served_route_refusal", refusal)
    kwargs = dict(encode_structure={"u": "dense"}, projected_units={})
    assert campaign._served_route_refusals("TESSERA_E4M3_K1", [768], ["u"], **kwargs) == {
        "768": {"reason": "first", "members": ["u"]}}
    reason[0] = "second"
    assert campaign._served_route_refusals("TESSERA_E4M3_K1", [768], ["u"], **kwargs) == {
        "768": {"reason": "second", "members": ["u"]}}
    assert len(calls) == 2


def test_full_group_planner_reuses_facts_but_preserves_member_and_invocation_gates(monkeypatch):
    family = "TESSERA_E2M1_K2"
    groups = {"a": ["u1", "u2"], "b": ["u3", "u4"]}
    rates = {name: {family: {640, 896}} for members in groups.values() for name in members}
    contracts, recipes = Counter(), Counter()
    reason: list[str | None] = [None]

    def refusal(_family, rung, *, structure):
        assert _family == family and structure == "routed_moe"
        contracts[rung] += 1
        return reason[0]

    def served(_family, rung, **kwargs):
        recipes["served", rung] += 1
        return "tcq" if rung == 640 else "research"

    def research(_family, rung):
        recipes["research", rung] += 1
        return "research"

    monkeypatch.setattr(formats, "tessera_served_route_refusal", refusal)
    monkeypatch.setattr(formats, "tessera_served_wire_recipe", served)
    monkeypatch.setattr(formats, "tessera_wire_recipe", research)
    structures = dict.fromkeys(rates, "routed_moe")
    projected = {"u1": {}, "u3": {}}
    grids, refused = campaign.anchor_group_rate_grids(groups, rates,
        encode_structure=structures, projected_units=projected)
    assert grids == {key: {family: [896]} for key in groups}
    assert refused["a"][family]["640"]["members"] == ["u2"]
    assert refused["b"][family]["640"]["members"] == ["u4"]
    reason[0] = "new contract"
    grids, refused = campaign.anchor_group_rate_grids(groups, rates,
        encode_structure=structures, projected_units=projected)
    assert grids == {key: {} for key in groups}
    for key, members in groups.items():
        assert refused[key][family] == {str(rung): {"reason": "new contract", "members": members}
                                         for rung in (640, 896)}
    assert contracts == Counter({640: 2, 896: 2})
    assert recipes == Counter({(kind, rung): 1 for kind in ("served", "research")
                               for rung in (640, 896)})


def test_full_group_planner_keeps_the_explicit_row_memo(monkeypatch):
    family = "TESSERA_E4M3_K1"
    calls = []

    def refusal(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(formats, "tessera_served_route_refusal", refusal)
    memo = {}
    campaign._served_route_refusals(family, [768], ["first"],
        encode_structure={"first": "dense"}, projected_units={}, route_cache=memo)
    grids, refused = campaign.anchor_group_rate_grids({"other": ["second"]},
        {"second": {family: {768}}}, encode_structure={"second": "dense"},
        projected_units={}, route_cache=memo)
    assert grids == {"other": {family: [768]}} and refused == {}
    assert len(calls) == 1
