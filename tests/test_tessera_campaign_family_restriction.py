"""An explicit pricing restriction cannot become routed support or seed leakage."""
import copy
import json
from types import SimpleNamespace

import pytest

from prismaquant import tessera_campaign as campaign, tessera_menu as menu


POLICY = {
    "schema": "prismaquant.tessera_campaign_family_restriction.v1",
    "dense": ["TESSERA_BF16_K1", "TESSERA_E4M3_K1"],
    "routed_moe": ["TESSERA_E4M3_K1"],
}


def test_same_shape_dense_and_routed_get_different_explicit_family_menus(monkeypatch):
    calls = []

    def expand(shape, **kwargs):
        families = tuple(spec.name for spec in kwargs.get("families", menu.menu_families()))
        calls.append(families)
        return [SimpleNamespace(family=f, route_status="unattested") for f in families]

    monkeypatch.setattr(menu, "expand_tessera_menu", expand)
    weights = {name: SimpleNamespace(shape=(32, 256)) for name in ("a", "b", "c")}
    result = campaign.expand_menus_for_targets(
        weights, list(weights), mode="readable", tp_degree=1, parallel_kind="none",
        family_restriction=POLICY,
        structure_by_unit={"a": "dense", "b": "routed_moe", "c": "dense"})
    assert [r.family for r in result["b"]] == ["TESSERA_E4M3_K1"]
    assert [r.family for r in result["a"]] == POLICY["dense"]
    assert result["a"] is result["c"] and result["a"] is not result["b"]
    assert len(calls) == 2
    assert all(r.route_status == "unattested" for rows in result.values() for r in rows)


def test_restricted_seed_refuses_incompatible_routed_family():
    state = {"anchors": [{"qname": "expert", "family": "TESSERA_BF16_K1",
        "format_name": "TESSERA_BF16_K1_R896", "body_rate_q256": 896}],
        "wire_records": {"TESSERA_BF16_K1_R896": {}}}
    with pytest.raises(RuntimeError, match="family restriction"):
        campaign.require_seed_family_scope("expert", state,
            family_restriction=POLICY, structure_by_unit={"expert": "routed_moe"},
            rate_band=(832, 1088))


@pytest.mark.parametrize("change", [
    {"schema": "wrong"}, {"extra": []}, {"dense": []}, {"routed_moe": "TESSERA_E4M3_K1"},
    {"dense": ["BF16"]}, {"dense": ["TESSERA_FP8"]}, {"dense": [True]},
    {"dense": ["TESSERA_E4M3_K1", "TESSERA_E4M3_K1"]},
])
def test_policy_refuses_malformed_or_noncanonical_declarations(change):
    with pytest.raises(ValueError, match="family restriction"):
        campaign.parse_family_restriction({**POLICY, **change})


def test_policy_is_canonical_and_rejects_repeated_json_fields():
    policy = copy.deepcopy(POLICY)
    policy["dense"].reverse()
    assert campaign.parse_family_restriction(json.dumps(policy)) == POLICY
    assert campaign.parse_family_restriction(None) is None
    with pytest.raises(ValueError, match="repeats field"):
        campaign.parse_family_restriction('{"dense":[],"dense":[]}')


@pytest.mark.parametrize("structures", [None, {}, {"a": "dense"},
    {"a": "dense", "b": "unknown"}, {"a": "dense", "b": "routed_moe", "extra": "dense"}])
def test_restriction_requires_complete_closed_structure_map(structures):
    weights = {name: SimpleNamespace(shape=(32, 256)) for name in ("a", "b")}
    with pytest.raises(ValueError, match="authoritative structure"):
        campaign.expand_menus_for_targets(weights, list(weights), mode="readable",
            tp_degree=1, parallel_kind="none", family_restriction=POLICY,
            structure_by_unit=structures)


def test_restriction_refuses_structure_conflicting_with_serving_context():
    context = SimpleNamespace(structure="dense", key=lambda: ("dense",))
    with pytest.raises(ValueError, match="conflicts with serving context"):
        campaign.expand_menus_for_targets({"a": SimpleNamespace(shape=(32, 256))}, ["a"],
            mode="readable", tp_degree=1, parallel_kind="none", family_restriction=POLICY,
            structure_by_unit={"a": "routed_moe"}, context_by_unit={"a": context})


def test_default_still_uses_the_unrestricted_shared_menu(monkeypatch):
    calls = []
    monkeypatch.setattr(menu, "expand_tessera_menu", lambda shape, **kw: calls.append(kw) or ["all"])
    weights = {name: SimpleNamespace(shape=(32, 256)) for name in ("a", "b")}
    result = campaign.expand_menus_for_targets(weights, list(weights), mode="readable",
        tp_degree=1, parallel_kind="none")
    assert len(calls) == 1 and "families" not in calls[0]
    assert result["a"] is result["b"]


def _seed_state(*, family="TESSERA_E4M3_K1", rate=896):
    fmt = f"{family}_R{rate}"
    return {"anchors": [{"qname": "expert", "family": family,
        "format_name": fmt, "body_rate_q256": rate}],
        "wire_records": {fmt: {"file": "anchor.wire"}}}


@pytest.mark.parametrize("rate", [768, 1152])
def test_restricted_seed_refuses_a_compatible_family_outside_new_band(rate):
    with pytest.raises(RuntimeError, match="outside restricted rate band"):
        campaign.require_seed_family_scope("expert", _seed_state(rate=rate),
            family_restriction=POLICY, structure_by_unit={"expert": "routed_moe"},
            rate_band=(832, 1088))


def test_restricted_seed_keeps_compatible_in_band_candidates():
    campaign.require_seed_family_scope("expert", _seed_state(), family_restriction=POLICY,
        structure_by_unit={"expert": "routed_moe"}, rate_band=(832, 1088))
    state = _seed_state()
    state["anchors"][0]["body_rate_q256"] = True
    with pytest.raises(RuntimeError, match="format/identity disagree"):
        campaign.require_seed_family_scope("expert", state, family_restriction=POLICY,
            structure_by_unit={"expert": "routed_moe"})


def test_seed_scope_refuses_before_linking_any_wire_from_the_unit(tmp_path):
    from prismaquant.cost_stage_checkpoint import write_unit
    seed = tmp_path / "seed"
    (seed / "cache/wire").mkdir(parents=True)
    (seed / "cache/wire/anchor.wire").write_bytes(b"historical wire")
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    seed_inputs = {'currency': 'output_mse', 'calibration': {},
        'input_global_scale_policy': 'fixture',
        'units': {'expert': {'scoring_rows': {'sha256': 'rows'}, 'input_global_scale': 1.0}}}
    seed_sha = canonical_json_sha256(seed_inputs, where='seed fixture')
    parts = seed / "cost.anchors.json.parts"
    parts.mkdir()
    write_unit(parts, stage="Tessera campaign", qname="expert", identity_sha256=seed_sha,
               state=_seed_state(family="TESSERA_BF16_K1"))
    manifest = seed / "cost.anchors.json"
    manifest.write_text(json.dumps({"identity_sha256": seed_sha, "identity": seed_inputs}))
    output = tmp_path / "wire"
    output.mkdir()
    adopted = []
    with pytest.raises(RuntimeError, match="family restriction"):
        campaign._adopt_seed_checkpoint(manifest, None, targets=["expert"], wire_dir=output,
            adopt=lambda *args, **kw: adopted.append(args), admits=lambda *args: True,
            identity_sha256="new", expected_identity=seed_inputs, validate_state=lambda name, state:
                campaign.require_seed_family_scope(name, state, family_restriction=POLICY,
                    structure_by_unit={"expert": "routed_moe"}, rate_band=(832, 1088)))
    assert not adopted and list(output.iterdir()) == []


def _identity(monkeypatch, *, policy=POLICY, structure="dense"):
    from prismaquant import production_weight_cache
    monkeypatch.setattr(campaign, "_checkpoint_identity_api", lambda: SimpleNamespace(
        tensor_identity=lambda value: {"fixture": True}, encoder_source_sha256=lambda: "encoder"))
    monkeypatch.setattr(production_weight_cache, "_production_cache_source_sha256", lambda: "pq")
    monkeypatch.setattr(campaign.th, "encoder_recipe", lambda: {"fixture": True})
    return campaign._campaign_checkpoint_identity(weights={"unit": object()}, acts={}, hessians={},
        menus={"unit": [SimpleNamespace(format_name="TESSERA_E4M3_K1_R896")]},
        args=SimpleNamespace(menu_mode="readable", family_restriction=policy),
        calibration_identity={}, serving_scope=None, static_scales={}, static_scale_policy="fixture",
        structure_by_unit={"unit": structure})


def test_checkpoint_identity_binds_policy_and_structure_even_for_identical_rungs(monkeypatch, tmp_path):
    from prismaquant.cost_stage_checkpoint import prepare_journal
    identity = _identity(monkeypatch)
    assert identity["family_restriction"] == {"policy": POLICY, "structure_by_unit": {"unit": "dense"}}
    parts = tmp_path / "journal.parts"
    prepare_journal(parts, manifest_path=tmp_path / "journal.json", stage="test", resume=True,
                    identity=identity, qnames=["unit"])
    changed = _identity(monkeypatch, structure="routed_moe")
    with pytest.raises(RuntimeError, match="identity"):
        prepare_journal(parts, manifest_path=tmp_path / "journal.json", stage="test", resume=True,
                        identity=changed, qnames=["unit"])
    narrower = {**POLICY, "dense": ["TESSERA_E4M3_K1"]}
    assert _identity(monkeypatch, policy=narrower) != identity
    off = _identity(monkeypatch, policy=None)
    assert "family_restriction" not in off and "family_restriction" not in off["settings"]


def _restricted_payloads():
    from test_tessera_campaign_fanout import _payloads
    payloads = _payloads()
    for payload in payloads.values():
        name = next(iter(payload["costs"]))
        payload["provenance"]["family_restriction"] = {
            "policy": copy.deepcopy(POLICY), "structure_by_unit": {name: "dense"}}
    return payloads


def test_fanout_merge_preserves_the_complete_structure_restriction():
    from tools.dispatch_tessera_campaign import merge_payloads
    merged = merge_payloads(_restricted_payloads(), census={"counts": {"a": 16384, "b": 16384}},
                            capture_sha256="merged-digest")
    assert merged["provenance"]["family_restriction"] == {
        "policy": POLICY, "structure_by_unit": {"a": "dense", "b": "dense"}}


def test_fanout_merge_unions_a_dense_row_and_a_routed_row():
    """One campaign's rows never carry the same ``structure_by_unit``.

    The map is keyed by the row's OWN selected units, so a dense row and a
    routed row differ in it by construction.  Equality is required on
    ``policy`` alone and the maps are unioned, the way ``units`` is unioned; a
    whole-restriction equality check would make any census that mixes dense
    and routed rows unmergeable (RobTand/prismaquant#487).
    """
    from tools.dispatch_tessera_campaign import merge_payloads
    payloads = _restricted_payloads()
    payloads["row-0001"]["provenance"]["family_restriction"]["structure_by_unit"] = {
        "b": "routed_moe"}
    merged = merge_payloads(payloads, census={"counts": {"a": 16384, "b": 16384}},
                            capture_sha256="merged-digest")
    assert merged["provenance"]["family_restriction"] == {
        "policy": POLICY, "structure_by_unit": {"a": "dense", "b": "routed_moe"}}


def test_fanout_merge_refuses_one_unit_claimed_by_two_rows():
    """The union is not a silent overwrite: a repeated unit is a refusal."""
    from tools.dispatch_tessera_campaign import merge_payloads, MergeRefused
    payloads = _restricted_payloads()
    payloads["row-0001"]["provenance"]["family_restriction"]["structure_by_unit"] = {
        "a": "dense", "b": "routed_moe"}
    with pytest.raises(MergeRefused, match="family restriction"):
        merge_payloads(payloads, census={"counts": {"a": 16384, "b": 16384}},
                       capture_sha256="merged-digest")


@pytest.mark.parametrize("mutation", ["missing_policy", "different_policy", "missing_structure", "unknown_structure"])
def test_fanout_merge_refuses_incompatible_or_incomplete_restrictions(mutation):
    from tools.dispatch_tessera_campaign import merge_payloads, MergeRefused
    payloads = _restricted_payloads()
    prov = payloads["row-0001"]["provenance"]
    if mutation == "missing_policy":
        prov.pop("family_restriction")
    elif mutation == "different_policy":
        prov["family_restriction"]["policy"]["dense"] = ["TESSERA_E4M3_K1"]
    else:
        prov["family_restriction"]["structure_by_unit"] = {} if mutation == "missing_structure" else {"b": "unknown"}
    with pytest.raises(MergeRefused, match="family_restriction|family restriction"):
        merge_payloads(payloads, census={"counts": {"a": 16384, "b": 16384}}, capture_sha256="merged-digest")


def test_main_uses_projected_membership_and_persists_restriction(monkeypatch, tmp_path):
    import pickle
    from test_tessera_campaign_packed import _bridge_main_fixture, _expert_unit_names
    owner, argv, _model, encoded = _bridge_main_fixture(monkeypatch, tmp_path)
    previous = owner.expand_menus_for_targets
    seen = []

    def expanded(weights, targets, **kwargs):
        seen.append((kwargs["family_restriction"], kwargs["structure_by_unit"]))
        return previous(weights, targets, **kwargs)

    monkeypatch.setattr(owner, "expand_menus_for_targets", expanded)
    assert owner.main([*argv, "--family-restriction", json.dumps(POLICY)]) == 0
    assert seen[0][0] == POLICY
    structures = seen[0][1]
    assert structures["model.layers.0.attention"] == "dense"
    assert all(structures[name] == "routed_moe" for name in _expert_unit_names())
    assert set(structures) == {name for name, _fmt, _shape in encoded}
    with (tmp_path / "cost.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["provenance"]["family_restriction"] == {
        "policy": POLICY, "structure_by_unit": structures}
    journal = json.loads((tmp_path / "cost.anchors.json").read_text())
    assert journal["identity"]["family_restriction"] == payload["provenance"]["family_restriction"]


def test_main_refuses_ambiguous_topology_before_calibration(monkeypatch, tmp_path):
    from test_tessera_campaign_packed import _bridge_main_fixture
    from prismaquant import sensitivity_probe
    owner, argv, _model, encoded = _bridge_main_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(sensitivity_probe, "discover_moe_structure",
        lambda *args, **kw: {"model.layers.0.attention": ("router", None)})
    monkeypatch.setattr(owner, "_calibration_tokens", lambda *args:
        pytest.fail("ambiguous topology reached calibration"))
    with pytest.raises(ValueError, match="conflicting router_path/expert_id topology"):
        owner.main([*argv, "--family-restriction", json.dumps(POLICY)])
    assert not encoded
