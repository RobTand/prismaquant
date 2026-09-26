"""Contract v38 reuses two cell ids for a different claim; admission follows the claim.

Tessera contract v38 (tessera#604) withdrew the routed ``TESSERA_E4M3_K1``
cells ``tessera_e4m3_k1_routed_moe_sm121_{decode,batch}_resident``, which
attested q1024 on ``eugr/spark-vllm@sha256:0afec8d4…`` through the
materialising modular kernel, and minted new cells under the SAME ids for q896
on ``spark-vllm-nccl230@sha256:f8dbe1a0…`` through the compact window MoE
adapter.  Cell ids are derived from ``family_structure_platform_regime
[_residency]``; neither the image nor the rung is part of the id, so a reader
that keyed anything on the id alone would carry the old claim onto the new
cell.

These tests pin that nothing here does:

* the route resolver admits routed E4M3 at q896 under the new image's scope and
  names the reused id;
* routed E4M3 at q1024 is unattested under the new image, even though the
  ``TESSERA_E4M3_K1`` family row now lists 1024 among its attested rungs (that
  1024 is the dense cells');
* routed E4M3 at q1024 is unattested under the old image, whose claim is gone;
* the reviewed-answer drift check, which keys cells by id, reports a same-id
  cell whose scope moved as a changed row, not as unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import struct

import pytest
from importlib.resources import as_file

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as tr
from prismaquant import tessera_runtime_contract as contract
from prismaquant.tessera_serving_runtime_pin import (
    TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256,
)

NEW_IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
             "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5")
OLD_IMAGE = ("eugr/spark-vllm@sha256:"
             "0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c")
REUSED = {"decode": "tessera_e4m3_k1_routed_moe_sm121_decode_resident",
          "batch": "tessera_e4m3_k1_routed_moe_sm121_batch_resident"}


def _packaged_bytes() -> bytes:
    with as_file(tr.tessera_serving_contract_path()) as path:
        return path.read_bytes()


def _table():
    with as_file(tr.tessera_serving_contract_path()) as path:
        return lane.load_eligibility_table(contract_path=path)


def _routed(rung: int, family: str = "TESSERA_E4M3_K1"):
    return lane.UnitStructuralFacts(
        qname="model.layers.3.mlp.experts",
        format_name=f"{family}_R{rung}", payload_family=family,
        k=None, n_sub=None, structure="routed_moe", role_split=False,
        in_features=6144, out_features=2048, rate_q256=rung)


def _resolve(facts, image):
    return lane.resolve_unit_route(
        facts, _table(), platform="sm_121", residency="resident",
        runtime_image=image, execution_mode="eager")


def test_the_installed_contract_is_the_v38_pin():
    raw = _packaged_bytes()
    assert hashlib.sha256(raw).hexdigest() == TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256
    assert json.loads(raw)["contract_version"] == 38


def test_the_reused_ids_now_attest_q896_on_the_new_image():
    route = _resolve(_routed(896), NEW_IMAGE)
    assert route.route_status == lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG, route.as_dict()
    assert {r.regime: r.cell_id for r in route.regimes} == REUSED
    assert route.requires_serve_flags == ("TESSERA_SERVE_MODE=resident",)


@pytest.mark.parametrize("image", [NEW_IMAGE, OLD_IMAGE], ids=["new_image", "old_image"])
def test_routed_q1024_is_attested_under_neither_image(image):
    """The id survived; the q1024 claim did not, under any scope."""
    route = _resolve(_routed(1024), image)
    assert route.route_status == lane.ROUTE_STATUS_UNATTESTED, route.as_dict()
    assert not [r.cell_id for r in route.regimes if r.cell_id]


def test_the_family_row_lists_1024_for_the_dense_cells_not_the_routed_ones():
    """The family's attested rungs are a union over structures; the routed
    question is answered by the routed cells, which carry 896 only."""
    table = _table()
    routed = [c for c in table.cells
              if c.family == "TESSERA_E4M3_K1" and c.structure == "routed_moe"]
    assert sorted(c.id for c in routed) == sorted(REUSED.values())
    assert {tuple(c.rungs_q256) for c in routed} == {(896,)}
    dense_1024 = [c.id for c in table.cells if c.family == "TESSERA_E4M3_K1"
                  and c.structure == "dense" and 1024 in c.rungs_q256]
    assert dense_1024, "the family row's 1024 must come from a dense cell"
    rows = json.loads(_packaged_bytes())["formats"]
    e4m3 = next(r for r in rows if r.get("family") == "TESSERA_E4M3_K1") \
        if isinstance(rows, list) else rows["TESSERA_E4M3_K1"]
    assert 1024 in e4m3["attested_rungs_q256"]


def test_answer_drift_reports_a_reused_id_whose_scope_moved():
    """Mutation: rewind the reused cells to their v34 scope in a copy of the
    installed answer, and the id-keyed drift must name each as changed."""
    raw = _packaged_bytes()
    with as_file(tr.tessera_serving_contract_path()) as path:
        installed = contract.contract_answer(contract._load_at(
            str(path), hashlib.sha256(raw).hexdigest(), contract.TESSERA_DEV_PIN_COMMIT))
    assert contract._answer_drift(installed, installed) == []
    reviewed = copy.deepcopy(installed)
    rewound = 0
    for row in reviewed["cells"]:
        if row[0] in REUSED.values():
            row[5] = [1024]
            row[13] = {"execution_modes": ["eager"], "image": OLD_IMAGE}
            rewound += 1
    assert rewound == 2
    drift = contract._answer_drift(reviewed, installed)
    for cell_id in REUSED.values():
        named = [line for line in drift if f"cells[{cell_id}]" in line]
        assert named and "reviewed" in named[0] and "installed" in named[0], drift


# ---------------------------------------------------------------------------
# GLM layer 43's pick, through the export gate, on the packaged v38 table
# ---------------------------------------------------------------------------
LAYER43_EXPERT = "model.layers.43.mlp.experts.0.down_proj"


@dataclass(frozen=True)
class _Target:
    platform: str = "sm_121"
    runtime_image: str = NEW_IMAGE
    execution_mode: str = "eager"
    residency: str = "resident"

    def as_dict(self):
        return asdict(self)


def _layer43_case(tmp_path, fmt):
    """A one-expert MoE checkpoint header and an allocation that picks ``fmt``.

    Shaped like GLM's routed expert ``down_proj`` (6144 in, 2048 out) at
    layer 43. The export gate reads headers only, so no weight is written.
    """
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "architectures": ["Qwen3MoeForCausalLM"],
        "num_experts": 2, "num_hidden_layers": 46,
    }))
    header = json.dumps({LAYER43_EXPERT + ".weight": {
        "dtype": "BF16", "shape": [2048, 6144],
        "data_offsets": [0, 2048 * 6144 * 2]}}).encode()
    (model / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    assignment = tmp_path / "layer_config.json"
    assignment.write_text(json.dumps({
        LAYER43_EXPERT: {"data_type": "tessera", "bits": 4, "tessera_format": fmt},
        "__prismaquant__": {"tessera_serving_scope": {
            "target": _Target().as_dict(),
            "by_unit": {LAYER43_EXPERT: {**_Target().as_dict(),
                                         "structure": "routed_moe"}}}},
    }))
    return model, assignment


def test_layer43s_routed_e4m3_r896_pick_passes_the_export_scope_gate(tmp_path):
    """The real gate, the real packaged table, no contract substitution.

    ``require_assignment_scope`` resolves the unit on the reused cells; each
    of them admits q896 through ``cell_lane_admits``, whose only gated lane
    (``window_gemv``) these cells do not launch through.
    """
    from prismaquant import tessera_export_lane as export

    model, assignment = _layer43_case(tmp_path, "TESSERA_E4M3_K1_R896")
    report = export.require_assignment_scope(model, assignment, target=_Target())
    route = report["by_unit"][LAYER43_EXPERT]
    assert route["route_status"] == lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG, route
    assert {row["cell_id"] for row in route["regime_routes"]} == set(REUSED.values())
    table = _table()
    for cell_id in REUSED.values():
        cell = next(c for c in table.cells if c.id == cell_id)
        assert lane.lane_claim_for_cell(cell, table.lanes) is None
        assert lane.cell_lane_admits(cell, 896, table.lanes) == (True, "")


def test_layer43_at_the_withdrawn_q1024_is_refused_by_the_export_scope_gate(tmp_path):
    """And the gate is reading the rung: the old routed claim does not pass."""
    from prismaquant import tessera_export_lane as export

    model, assignment = _layer43_case(tmp_path, "TESSERA_E4M3_K1_R1024")
    with pytest.raises(export.TesseraExportLaneError, match="unattested"):
        export.require_assignment_scope(model, assignment, target=_Target())
