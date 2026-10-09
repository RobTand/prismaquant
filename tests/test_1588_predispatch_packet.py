"""The #1588 pre-dispatch packet states verifiable facts.

The packet names the requested rungs, their legal state, their cell state,
the pins, and the estimate inputs. This test re-derives each claim through
the repo APIs. It fails when the packet is absent or stale.
"""
from __future__ import annotations

import json
from pathlib import Path

PACKET_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/measurements/1588-predispatch-packet-2026-10-09.json"
)

CLASS_STRUCTURE = {"routed": "routed_moe", "dense": "dense"}


def _packet() -> dict:
    return json.loads(PACKET_PATH.read_text(encoding="utf-8"))


def test_packet_pins_match_the_repo_pin_file():
    from prismaquant import tessera_legal_domain as domain

    packet = _packet()
    live = domain.live_pins().as_dict()
    assert packet["pins"] == live


def test_packet_rungs_are_producer_legal_without_holes():
    from prismaquant import tessera_legal_domain as domain

    packet = _packet()
    shapes = domain.GLM53_LINEAR_SHAPES
    seen: dict[str, set[int]] = {}
    for row in packet["requested_rows"]:
        legal, holes = domain.legal_rates(row["family"], shapes)
        assert row["producer_legal"] is True
        assert row["rate"] in set(legal)
        assert holes.get(row["rate"], ()) == ()
        seen.setdefault(row["family"], set()).add(row["rate"])
    assert {f: sorted(r) for f, r in seen.items()} == packet["unique_rates"]


def test_packet_cell_state_matches_the_pinned_contract():
    from prismaquant import tessera_legal_domain as domain

    packet = _packet()
    triples = domain.native_qualification_set()
    structures = set(domain.contract_structures())
    assert {"dense", "routed_moe"} <= structures
    for row in packet["requested_rows"]:
        want = sorted(
            t[2] for t in triples if (t[0], t[1]) == (row["family"], row["rate"])
        )
        assert row["qualified_structures"] == want, (
            f'{row["family"]}_R{row["rate"]} ({row["row_class"]})'
        )
        need = CLASS_STRUCTURE[row["row_class"]]
        assert row["cell_for_requested_class"] is (need in want)


def test_packet_estimate_inputs_are_consistent():
    packet = _packet()
    est = packet["estimate"]
    routed = sum(1 for r in packet["requested_rows"] if r["row_class"] == "routed")
    dense = sum(1 for r in packet["requested_rows"] if r["row_class"] == "dense")
    assert (routed, dense) == (7, 7)
    assert est["shape_time_rows"]["count"] == (routed * 1 + dense * 4) * 4
    assert est["shape_time_rows"]["count"] == 140
    assert est["cost_rows_gpu_h"]["encode_low"] == 145
    assert est["cost_rows_gpu_h"]["encode_high"] == 190
    assert est["cost_rows_gpu_h"]["joint_receipt"] == 0
    assert len(packet["source_commit"]) == 40
