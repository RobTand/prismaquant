"""PQ #2471 scope probe: the #2459 authorization names live admission cells.

This module checks no prose. It reads the live v2 pin and the packaged
contract through the production admission path. It asserts the eight
cells the #2459 authorization names admit at their stated scope, and
that excluded scopes refuse. The dated prose record in
``docs/results/`` carries the serve evidence; static review reads it.
"""
from __future__ import annotations

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as tr

IMAGE_5BE13705 = (
    "localhost/prismaquant/spark-vllm-nccl230@sha256:"
    "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a"
)

AUTHORIZED = (
    ("TESSERA_BF16_K1", "dense", "batch", (832, 880, 960, 1024, 1088)),
    ("TESSERA_BF16_K1", "dense", "decode", (832, 880, 960, 1024, 1088)),
    ("TESSERA_BF16_K1", "routed_moe", "batch", (1024,)),
    ("TESSERA_BF16_K1", "routed_moe", "decode", (1024,)),
    ("TESSERA_E4M3_K1", "dense", "batch", (832, 960, 1024, 1088)),
    ("TESSERA_E4M3_K1", "dense", "decode", (832, 960, 1024, 1088)),
    ("TESSERA_E4M3_K1", "routed_moe", "batch", (896, 928, 1024, 1088)),
    ("TESSERA_E4M3_K1", "routed_moe", "decode", (896, 928, 1024, 1088)),
)


def _table():
    return lane.load_eligibility_table(contract_path=tr.tessera_serving_contract_path())


def _context(family, structure, regime):
    return lane.ServingContext(
        platform="sm_121", structure=structure, residency="resident",
        runtime_image=IMAGE_5BE13705, execution_mode="eager")


@pytest.mark.parametrize("family,structure,regime,rungs", AUTHORIZED)
def test_the_authorized_scope_admits_at_each_stated_rung(
        family, structure, regime, rungs):
    table = _table()
    cells = [c for c in table.cells
             if c.family == family and c.structure == structure
             and c.regime == regime and c.runtime_image == IMAGE_5BE13705]
    assert len(cells) == 1, (family, structure, regime)
    cell = cells[0]
    assert tuple(cell.rungs_q256) == rungs
    assert cell.qualification == "device_qualified"
    assert cell.requires_plugin == "tessera"
    assert lane.cell_evidence_admits(cell)[0]
    context = _context(family, structure, regime)
    assert lane.cell_matches_serving_context(cell, context)
    for rung in rungs:
        assert cell.covers_rate(rung), (cell.id, rung)
    assert not cell.covers_rate(1), cell.id


def test_the_authorized_image_serves_tp2_within_the_closed_world_ceiling():
    import json

    contract = json.loads(tr.tessera_serving_contract_path().read_text())
    tp = contract["tensor_parallel"]
    assert tp["semantics"] == "closed_world"
    worlds = {u["unit"]: u["max_world_size"] for u in tp["units"]}
    for family, _, _, _ in AUTHORIZED:
        assert worlds[family] == 2, family
    receipts = {r["id"]: r for r in tp["world_size_receipts"]}
    assert receipts["glm53_a4_stub_tp2_sm121"]["world_size"] == 2


def test_excluded_scopes_refuse():
    table = _table()
    vanilla = "vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
    assert not any(c.runtime_image == IMAGE_5BE13705
                   and c.family == "TESSERA_E2M1_K2" for c in table.cells)
    e2m1 = [c for c in table.cells if c.family == "TESSERA_E2M1_K2"]
    assert e2m1
    assert all(c.runtime_image != IMAGE_5BE13705 for c in e2m1)
    streamed = lane.ServingContext(
        platform="sm_121", structure="dense", residency="streamed",
        runtime_image=IMAGE_5BE13705, execution_mode="eager")
    cell = next(c for c in table.cells if c.runtime_image == IMAGE_5BE13705)
    assert not lane.cell_matches_serving_context(cell, streamed)
    compiled = lane.ServingContext(
        platform="sm_121", structure="dense", residency="resident",
        runtime_image=IMAGE_5BE13705, execution_mode="compiled")
    assert not lane.cell_matches_serving_context(cell, compiled)
    vanilla_cells = [c for c in table.cells if c.runtime_image == vanilla]
    assert vanilla_cells
    assert all(c.runtime_image != IMAGE_5BE13705 for c in vanilla_cells)
