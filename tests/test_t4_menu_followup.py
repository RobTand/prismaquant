"""T-4 E2M1 menu tests through the canonical consumer path.

Synthetic tables prove source integration only. They qualify no speed,
weight error, served loss or serving frontier. The T-4 whole-bit rungs
come from the existing registry owner. The classes come from the real
producer. The verdicts come from the existing consumer owners.

Only the approved q896 menu admits. Every synthetic T-4 rung waits with
``performance_admission_not_established``, including the whole-bit cap
R768. The controls show the fixture refuses where it must.
"""
from __future__ import annotations

import copy
import json

from pathlib import Path
from types import MappingProxyType

from prismaquant.rung_allowability import RungAllowability

V3FIX = Path(__file__).parent / "fixtures" / "rung_allowability_v3"
T4 = "TESSERA_E2M1_K2"
DENSE_SCOPE = {"kernel_kind": "dense", "rows": 4, "columns": 8}


def _needs_producer(monkeypatch):
    from prismaquant import rung_allowability
    from _rung_allowability_producer import ExternalProducer
    producer = ExternalProducer()
    monkeypatch.setattr(rung_allowability, "_producer_api", lambda: producer)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "research")
    return producer


def _t4_spec():
    from prismaquant.tessera_formats import get_tessera_family
    return get_tessera_family(T4)


def _t4_whole_rungs():
    spec = _t4_spec()
    low, high = spec.mathematical_q256_bounds
    whole = [rung for rung in range(low, high + 1) if rung % 256 == 0]
    assert whole, f"{T4} has no whole-bit rung in {low}..{high}"
    return whole


def _t4_cap_rung():
    return max(_t4_whole_rungs())


def _t4_owner(monkeypatch, rung):
    """A synthetic measured T-4 table through the real producer.

    The shape copies the T-8 whole-bit fixture mechanics. The rate,
    arity, payload width and run widths follow the registry family and
    the producer class rule. The served wire per structure comes from
    the existing wire owner. The class identities come from the real
    producer. Nothing here invents admission.
    """
    from prismaquant.tessera_formats import tessera_served_wire_recipe
    producer = _needs_producer(monkeypatch)
    spec = _t4_spec()
    width, remainder = divmod(rung * spec.arity, 256)
    assert remainder == 0, f"R{rung} is not a pure {T4} class"
    table = copy.deepcopy(json.loads((V3FIX / "v3-whole.json").read_text()))
    table["format"] = T4
    table["scope"]["rung_min"] = rung
    table["scope"]["rung_max"] = rung
    table["scope"]["required_cells"] = [
        {"M": 8, "cell_id": "dense:t4:M8", "kernel_kind": "dense",
         "shape_id": "t4s0"},
        {"M": 8, "cell_id": "routed:t4:M8", "kernel_kind": "routed",
         "shape_id": "t4s1"},
    ]
    table["scope"]["shapes"] = [
        {"columns": 8, "kernel_kind": "dense", "rows": 4, "shape_id": "t4s0"},
        {"columns": 8, "kernel_kind": "routed", "rows": 4,
         "shape_id": "t4s1"},
    ]
    (row,) = table["rungs"]
    row["rung"] = rung
    for measurement in row["measurements"]:
        cell = measurement["cell_id"].split(":")[0]
        measurement["cell_id"] = f"{cell}:t4:M8"
        measurement["shape_id"] = "t4s0" if cell == "dense" else "t4s1"
        structure = "dense" if cell == "dense" else "routed_moe"
        recipe = tessera_served_wire_recipe(
            spec, rung, structure=structure, refuse_unattested=False)
        measurement["geometry"]["recipe"] = recipe.to_config()
        measurement["geometry"]["decode_width"].update(
            {"arity": spec.arity, "value_bits": 4, "run_widths": [width]})
        measurement["geometry"]["bits_per_256_weight_tile"] = {
            "denominator": 1, "numerator": rung}
    table["geometry_classes"] = producer.measured_geometry_classes(table)
    assert table["geometry_classes"], "producer derived no T-4 class"
    return RungAllowability(
        T4, MappingProxyType(dict(table["kernel_build"])),
        table["table_version"], "synthetic T-4 mechanics only", table,
        producer, MappingProxyType({rung: True}), frozenset({rung}))


def test_t4_whole_bit_cap_still_waits(monkeypatch):
    rung = _t4_cap_rung()
    owner = _t4_owner(monkeypatch, rung)
    assert not owner.allows(rung), owner.refusal(rung)
    assert not owner.allows(rung, scope=dict(DENSE_SCOPE)), owner.refusal(
        rung, scope=dict(DENSE_SCOPE))


def test_t4_lane_seam_refuses_unapproved_menu(monkeypatch):
    from prismaquant import allocator_candidates as candidates
    rung = _t4_cap_rung()
    owner = _t4_owner(monkeypatch, rung)
    admission = candidates.candidate_rung_admission(
        f"{T4}_R{rung}", target_profile="research",
        rung_allowability={T4: owner},
        allowability_scope=dict(DENSE_SCOPE))
    assert admission is not None
    assert not admission.measured_allowable, owner.refusal(
        rung, scope=dict(DENSE_SCOPE))
    assert not admission.admits("research"), admission.detail


def test_t4_unresolved_scope_still_waits(monkeypatch):
    from prismaquant import allocator_candidates as candidates
    rung = _t4_cap_rung()
    owner = _t4_owner(monkeypatch, rung)
    admission = candidates.candidate_rung_admission(
        f"{T4}_R{rung}", target_profile="research",
        rung_allowability={T4: owner},
        allowability_scope={"kernel_kind": "dense", "rows": 4})
    assert admission is not None
    assert not admission.admits("research")


def test_t4_wrong_activation_still_waits(monkeypatch):
    rung = _t4_cap_rung()
    owner = _t4_owner(monkeypatch, rung)
    scope = dict(DENSE_SCOPE, activation_contract="wrong")
    assert owner.refusal(rung, scope=scope) != ""


def test_t4_exact_bytes_are_integer_accountant_values():
    from fractions import Fraction
    from prismaquant.tessera_footprint import tessera_exact_bits_for_shape
    from prismaquant.tessera_formats import tessera_served_wire_recipe
    spec = _t4_spec()
    rung = _t4_cap_rung()
    recipe = tessera_served_wire_recipe(
        spec, rung, structure="dense", refuse_unattested=False)
    bits = tessera_exact_bits_for_shape(spec, rung, (4, 8), recipe=recipe)
    assert isinstance(bits, Fraction) and bits.denominator == 1
    assert int(bits) > 0
