"""Packed-group members reach aggregation with their whole menus.

``reduce_continuous_menu`` drops a rung only when another rung is no larger in
bytes and no larger in cost. That is exact for a unit the DP chooses on its
own. A packed MoE member is not such a unit: ``aggregate_packed_serving_groups``
offers the group only the format NAMES legal for every member, then prices each
name by summing member bytes and costs. A rung dominated for one member can be
on the group's frontier, and pruning it per member removes it from the
intersection before the group is ever priced.

GLM-5.3 Flash showed the failure at full scale. On the 515-name menu, BF16 R1024
dominated E4M3 R1088 for 33,258 routed members but not for the rest, so every
one of the 42 expert stacks was left with E4M3 R832 alone. The allocator shipped
3.276 bpp with 30.4 GB of the byte budget unused, and it did worse on Δloss than
the 257-name E4M3 menu it contains.
"""
from __future__ import annotations

from types import SimpleNamespace

from prismaquant import allocator_candidates as ac
from prismaquant import format_registry as fr
from prismaquant.allocator_solver import Candidate

_LOW, _MID, _HIGH = (
    "TESSERA_E4M3_K1_R832", "TESSERA_BF16_K1_R1024", "TESSERA_E4M3_K1_R1088")
_BYTES = {_LOW: 832, _MID: 1024, _HIGH: 1088}
_N_PARAMS = 800
_MEMBERS = ("layer.mlp.experts.0.down_proj", "layer.mlp.experts.1.down_proj")
_DENSE = "layer.self_attn.o_proj"

# Member 0: the MID rung dominates HIGH (fewer bytes, lower cost).
# Member 1: LOW dominates MID (fewer bytes, equal cost).
# Group sums: LOW 20 @1664, MID 11 @2048, HIGH 3 @2176, so all three are on
# the group's frontier.
_LOSS = {
    _MEMBERS[0]: {_LOW: 10.0, _MID: 1.0, _HIGH: 2.0},
    _MEMBERS[1]: {_LOW: 10.0, _MID: 10.0, _HIGH: 1.0},
    _DENSE: {_LOW: 5.0, _MID: 1.0, _HIGH: 2.0},
}


def _profile():
    def group(name):
        return "layer.mlp.experts::packed" if ".experts." in name else None
    return SimpleNamespace(packed_expert_format_group=group)


def _inputs():
    stats = {
        name: {"n_params": _N_PARAMS, "h_trace": 1.0,
               "in_features": 20, "out_features": 40}
        for name in _LOSS
    }
    costs = {name: {fmt: {"predicted_dloss": loss}
                    for fmt, loss in row.items()}
             for name, row in _LOSS.items()}
    candidates = {
        name: [Candidate(fmt, 8.0 * _BYTES[fmt] / _N_PARAMS, _BYTES[fmt], loss)
               for fmt, loss in row.items()]
        for name, row in _LOSS.items()
    }
    specs = [fr.get_format(fmt) for fmt in (_LOW, _MID, _HIGH)]
    return stats, costs, candidates, specs


def _group_menu(candidates):
    (unit,) = [n for n in candidates if ac._PACKED_GROUP_MARKER in n]
    return {c.fmt for c in candidates[unit]}


def test_per_member_pruning_empties_the_group_intersection():
    stats, costs, candidates, specs = _inputs()
    reduced = ac.reduce_continuous_menu(candidates, stats)
    assert {c.fmt for c in reduced[_MEMBERS[0]]} == {_LOW, _MID}
    assert {c.fmt for c in reduced[_MEMBERS[1]]} == {_LOW, _HIGH}
    _, _, grouped = ac.aggregate_packed_serving_groups(
        stats, costs, specs, reduced, _profile())
    # The group loses both of its upgrades before anything prices it.
    assert _group_menu(grouped) == {_LOW}


def test_deferred_members_keep_the_group_frontier():
    stats, costs, candidates, specs = _inputs()
    deferred = ac.packed_serving_group_members(candidates, _profile())
    assert deferred == frozenset(_MEMBERS)
    reduced = ac.reduce_continuous_menu(
        candidates, stats, defer_reduction=deferred)
    for member in _MEMBERS:
        assert reduced[member] == candidates[member]
    # Units outside a packed group are still reduced as before.
    assert {c.fmt for c in reduced[_DENSE]} == {_LOW, _MID}
    grouped_stats, _, grouped = ac.aggregate_packed_serving_groups(
        stats, costs, specs, reduced, _profile())
    final = ac.reduce_continuous_menu(grouped, grouped_stats)
    assert _group_menu(final) == {_LOW, _MID, _HIGH}
    (unit,) = [n for n in final if ac._PACKED_GROUP_MARKER in n]
    priced = {c.fmt: (c.memory_bytes, c.predicted_dloss) for c in final[unit]}
    assert priced == {_LOW: (1664, 20.0), _MID: (2048, 11.0),
                      _HIGH: (2176, 3.0)}


def test_singletons_and_ungrouped_rows_are_not_deferred():
    lone = {"other.mlp.experts.0.down_proj": [], _DENSE: []}

    def group(name):
        return "other::packed" if ".experts." in name else None

    profile = SimpleNamespace(packed_expert_format_group=group)
    assert ac.packed_serving_group_members(lone, profile) == frozenset()
    assert ac.packed_serving_group_members(lone, None) == frozenset()


def test_build_candidates_forwards_the_deferral(monkeypatch):
    seen = {}

    def fake_reduce(out, stats, **kwargs):
        seen.update(kwargs)
        return out

    monkeypatch.setattr(ac, "reduce_continuous_menu", fake_reduce)
    stats, costs, _, specs = _inputs()
    ac.build_candidates(stats, costs, specs,
                        defer_menu_reduction=frozenset(_MEMBERS))
    assert seen["defer_reduction"] == frozenset(_MEMBERS)
