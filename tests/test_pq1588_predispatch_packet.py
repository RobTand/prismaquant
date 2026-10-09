"""The #1588 pre-dispatch packet states verifiable facts (PQ #2558).

The packet names the 14 requested rungs, their legal state, their cell
state, the pins, the input digests, the namespace plan, and the hour
math. This test re-derives each claim through the repo APIs. It fails
when the packet is absent or stale: a moved pin, a changed input
digest, or edited hour math goes red here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from prismaquant import schemas

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = REPO_ROOT / "docs/results/pq1588_predispatch_2026-10-09.json"
DOC_PATH = REPO_ROOT / "docs/results/pq1588_predispatch_2026-10-09.md"

CLASS_STRUCTURE = {"routed": "routed_moe", "dense": "dense"}
SHA_RE = re.compile(r"[0-9a-f]{40}")
ROW_RE = re.compile(
    r"^\|\s*(TESSERA_\S+)\s*\|\s*(\d+)\s*\|\s*(routed|dense)\s*\|",
    re.MULTILINE,
)


def _packet() -> dict:
    return json.loads(PACKET_PATH.read_text(encoding="utf-8"))


def test_packet_shape_and_hour_math_validate():
    schemas.validate_pq1588_predispatch_packet(
        _packet(), path=str(PACKET_PATH)
    )


def test_packet_source_sha_comes_from_provenance():
    packet = _packet()
    provenance = packet["provenance"]
    sha = provenance["source_commit"]
    assert SHA_RE.fullmatch(sha) is not None
    assert packet["source_commit"] == sha
    assert packet["short_id"] == sha[:8]
    assert sha[:8] in packet["namespace"]["proposed_root"]


def test_packet_pins_match_the_repo_pin_file():
    from prismaquant import tessera_legal_domain as domain

    packet = _packet()
    live = domain.live_pins().as_dict()
    assert packet["pins"] == live


def test_packet_input_digests_match_live_sources():
    from prismaquant import tessera_legal_domain as domain

    packet = _packet()
    state = domain.tessera_source_state()
    assert packet["input_digests"]["tessera_export_sha256"] == state["export_sha256"]
    assert packet["input_digests"]["tessera_grammar_sha256"] == state["grammar_sha256"]
    assert (
        packet["input_digests"]["installed_contract_sha256"]
        == domain.live_pins().as_dict()["producer_installed_contract_sha256"]
    )


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
    assert sum(len(r) for r in seen.values()) == 11
    assert all(row["priced_in_joint_pkl"] is False for row in packet["requested_rows"])


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
    assert packet["namespace"]["created"] is False


def test_doc_table_equals_packet_set():
    packet = _packet()
    text = DOC_PATH.read_text(encoding="utf-8")
    doc_rows = {
        (family, int(rate), cls) for family, rate, cls in ROW_RE.findall(text)
    }
    packet_rows = {
        (row["family"], row["rate"], row["row_class"])
        for row in packet["requested_rows"]
    }
    assert len(doc_rows) == 14
    assert doc_rows == packet_rows
    assert "11" in text and "no price row" in text
