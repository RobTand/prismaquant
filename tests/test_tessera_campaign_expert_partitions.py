"""Fixed-rate partitions keep a full group contract, not a new sample."""
import json
from typing import cast

import pytest

from prismaquant import tessera_campaign_selection as campaign_selection
from prismaquant import tessera_campaign_table as campaign_table
from prismaquant import tessera_campaign as campaign

V3 = "prismaquant.tessera_campaign_units.v3"
PARTITION = "prismaquant.tessera_campaign_expert_partition.v1"
GROUP = "s:layers.0.experts"


def roster():
    records = {f"expert{expert}.{role}": {"expert": expert, "projection": role,
                                        "rows": 8, "cols": 8}
               for expert in range(4) for role in ("gate", "up", "down")}
    return records, {name: "layers.0.experts" for name in records}


def selection(index=0):
    records, _ = roster()
    experts = list(range(index * 2, index * 2 + 2))
    return {"schema": V3, "groups": [{"key": GROUP, "members": sorted(records),
        "partition": {"schema": PARTITION, "experts_per_row": 2,
                      "index": index, "count": 2, "rate_q256": 768,
                      "members": sorted(n for n, r in records.items() if r["expert"] in experts)}}]}


def validate(record, **overrides):
    records, stacks = roster()
    kwargs = dict(unit_records=records, stack_of=stacks, rate_band="768,768",
                  max_rounds=1, seeded=False)
    kwargs.update(overrides)
    return campaign_table.expert_partition_members(record, **kwargs)


def test_versioned_partition_loads_and_prices_only_complete_expert_chunk(tmp_path):
    record = selection()
    path = tmp_path / "units.json"
    path.write_text(json.dumps(record))
    loaded = campaign_selection.load_unit_selection(path)
    campaign_table.select_anchor_groups(loaded, {GROUP: sorted(roster()[0])}, where="test")
    assert validate(loaded) == {GROUP: record["groups"][0]["partition"]["members"]}
    priced, audit, pi = campaign_selection.selection_priced_units(loaded)
    assert priced == set(record["groups"][0]["partition"]["members"])
    assert not audit and not pi


@pytest.mark.parametrize("overrides", [dict(rate_band="768,896"), dict(rate_band=None),
                                     dict(max_rounds=2), dict(seeded=True)])
def test_adaptive_or_seeded_partition_refuses(overrides):
    with pytest.raises(RuntimeError, match="partition"):
        validate(selection(), **overrides)


@pytest.mark.parametrize("mutation", ["role_gap", "overlap", "wrong_count", "wrong_rate", "sample"])
def test_partition_must_be_the_declared_expert_complete_chunk(mutation):
    record = selection()
    part = record["groups"][0]["partition"]
    if mutation == "role_gap":
        part["members"].pop()
    elif mutation == "overlap":
        part["members"].append(part["members"][0])
    elif mutation == "wrong_count":
        part["count"] = 3
    elif mutation == "wrong_rate":
        part["rate_q256"] = 896
    else:
        record["groups"][0]["sampled"] = part["members"]
    with pytest.raises(RuntimeError, match="partition"):
        validate(record)


def test_legacy_schema_cannot_hide_partition_metadata(tmp_path):
    record = selection()
    record["schema"] = campaign_selection.UNITS_SCHEMA
    path = tmp_path / "units.json"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="partition"):
        campaign_selection.load_unit_selection(path)


def test_partition_subsets_are_disjoint_and_exactly_cover_full_roster():
    left = set(validate(selection(0))[GROUP])
    right = set(validate(selection(1))[GROUP])
    assert not left & right
    assert left | right == set(roster()[0])


def test_shared_menu_grid_uses_full_group_not_local_partition(monkeypatch):
    owner = getattr(campaign, "anchor_group_rate_grids", None)
    assert callable(owner), "group rate/menu intersection is not shared"
    monkeypatch.setattr(campaign, "_served_route_refusals", lambda *a, **k: {})
    rates = {"a": {"e4m3": {768, 896}, "bf16": {1024}},
             "b": {"e4m3": {768}}}
    result = owner({GROUP: ["a", "b"]}, rates,
                   encode_structure=None, projected_units={})
    grids, refused = cast(tuple[dict, dict], result)
    assert grids == {GROUP: {"e4m3": [768]}}
    assert not refused
    local, _ = cast(tuple[dict, dict], owner({GROUP: ["a"]}, rates,
                    encode_structure=None, projected_units={}))
    assert local != grids  # The local menu would incorrectly widen this contract.


def test_partition_is_bound_to_real_carried_producer_records():
    from prismaquant import tessera_expert_projection as projection
    from test_tessera_expert_projection import _declared, _projection, STACK

    producer = _projection(experts=(0, 1, 2, 3), n=8, k=8)
    bound = projection.bind_expert_projection(
        producer, declared=_declared(experts=(0, 1, 2, 3), n=8, k=8))
    carried = projection.carried_projection(producer, bound,
        request=projection.stack_plan_request({STACK: ("E4M3", 1024)}), tool="/repo/tool.py")
    _, records, stacks = projection.carried_units(carried)
    names = sorted(records)
    key = "s:" + STACK
    priced = sorted(name for name, unit in records.items() if unit["expert"] < 2)
    record = {"schema": V3, "groups": [{"key": key, "members": names,
        "partition": {"schema": PARTITION, "experts_per_row": 2, "index": 0,
                      "count": 2, "rate_q256": 768, "members": priced}}]}
    assert campaign_table.expert_partition_members(record, unit_records=records,
        stack_of=stacks, rate_band="768,768", max_rounds=1) == {key: priced}
    edited = json.loads(json.dumps(carried))
    edited["stacks"][STACK][names[0]]["rows"] *= 2
    with pytest.raises(projection.ExpertProjectionError, match="disagrees"):
        projection.carried_units(edited)


def test_partition_refuses_heterogeneous_producer_geometry():
    records, stacks = roster()
    records["expert3.gate"]["rows"] = 16
    with pytest.raises(RuntimeError, match="geometry"):
        campaign_table.expert_partition_members(selection(), unit_records=records,
            stack_of=stacks, rate_band="768,768", max_rounds=1)


def _runtime_menu_scope(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from test_tessera_campaign_scope import NAMES, _request, _resolve

    base = _resolve(_request(tmp_path))
    context = SimpleNamespace(key=lambda: ("routed_moe",), structure="routed_moe")
    return replace(base, dense_targets=[NAMES[0]],
        partition_menu_targets=list(NAMES),
        census={"unit_shapes": {NAMES[0]: [8, 8], NAMES[1]: [16, 8]}},
        context_by_unit=dict.fromkeys(NAMES, context),
        structure_by_unit=dict.fromkeys(NAMES, "routed_moe")), context


def test_actual_runtime_menu_operation_keeps_full_roster_and_narrows_receipts(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import torch
    from prismaquant import tessera_menu
    from prismaquant.model_profiles import DefaultProfile

    scope, context = _runtime_menu_scope(tmp_path)
    names = scope.partition_menu_targets
    seen, planned = [], []

    def menu(shape, **kwargs):
        seen.append((shape, kwargs["serving_context"]))
        return [SimpleNamespace(format_name="TESSERA_E4M3_K1_R768")]

    actual_expand = campaign.expand_menus_for_targets
    def observe_geometry(weights, *args, **kwargs):
        planned.append({name: (tuple(weight.shape), weight.device.type)
                        for name, weight in weights.items()})
        return actual_expand(weights, *args, **kwargs)

    monkeypatch.setattr(tessera_menu, "expand_tessera_menu", menu)
    monkeypatch.setattr(campaign, "expand_menus_for_targets", observe_geometry)
    resolved = campaign.resolve_campaign_menu(
        {names[0]: torch.empty((8, 8), device='meta')}, scope, DefaultProfile(),
        mode="research", tp_degree=1, family_restriction=None, model_path='fixture',
        cache_dir=tmp_path/'cache', selected_source=False, streaming_head=False,
        source_authentication=None, selected_guard=None)
    assert resolved.partition is not None
    assert set(resolved.partition.full_menus) == set(names)
    assert set(resolved.menus) == {names[0]}
    assert set(resolved.context_by_unit) == set(resolved.structure_by_unit) == {names[0]}
    assert resolved.encode_structure == {names[0]: "routed_moe"}
    assert seen == [((8, 8), context), ((16, 8), context)]
    assert planned == [{names[0]: ((8, 8), 'meta'), names[1]: ((16, 8), 'meta')}]
    assert not (tmp_path/'cache').exists()


def test_runtime_menu_refuses_projected_units_declared_dense(monkeypatch, tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from prismaquant import tessera_menu
    from prismaquant.model_profiles import DefaultProfile

    scope, _context = _runtime_menu_scope(tmp_path)
    name = scope.dense_targets[0]
    scope = replace(scope, partition_menu_targets=None,
        context_by_unit={name: scope.context_by_unit[name]},
        structure_by_unit={name: 'routed_moe'}, encode_structure={name: 'dense'},
        population=replace(scope.population, declared={'stack': {name: [8, 8]}}))
    monkeypatch.setattr(tessera_menu, 'expand_tessera_menu', lambda *_args, **_kwargs: [
        SimpleNamespace(format_name='TESSERA_E4M3_K1_R768')])
    # The producer-bound projection is a controlled external fact; the real
    # operation must refuse its contradictory declared encoding structure.
    monkeypatch.setattr(campaign, '_project_expert_population', lambda *_args, **_kwargs:
        ({'stacks': {'stack': {}}}, {name: {'projection': 'gate_proj'}}))
    with pytest.raises(ValueError, match='projected expert units declared non-routed'):
        campaign.resolve_campaign_menu(
            {name: torch.empty((8, 8), device='meta')}, scope, DefaultProfile(),
            mode='research', tp_degree=1, family_restriction=None, model_path='fixture',
            cache_dir=tmp_path/'cache', selected_source=False, streaming_head=False,
            source_authentication=None, selected_guard=None)
    assert not (tmp_path/'cache').exists()


def test_menu_operation_has_all_actual_global_bindings():
    from tools.verify_campaign_operation_globals import unbound_global_reads

    assert unbound_global_reads(campaign.resolve_campaign_menu) == ()
