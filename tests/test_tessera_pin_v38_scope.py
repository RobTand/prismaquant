"""Admission follows the v39 scope, not the cell ids inherited from v38.

The GLM image now carries all six family/structure combinations; routed
E4M3 includes q1024 again on that image, not on the withdrawn v34 image.
Vanilla vLLM retains its dense E4M3 q1024 cells. These are packaged route
claims, not new served-quality measurements. Keep this regression's filename
as the v38 origin of the same-id/scope-drift check.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
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
VANILLA_IMAGE = ("vllm/vllm-openai@sha256:"
                 "61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14")
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


def test_the_installed_contract_is_the_v42_pin():
    # v40 (Tessera #675) adds only the producer_interface block, v41 optional
    # serving-code fields no cell stamps, and v42 the fused routed launches;
    # the admission scopes this module pins are v39's and do not move.
    raw = _packaged_bytes()
    assert hashlib.sha256(raw).hexdigest() == TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256
    assert json.loads(raw)["contract_version"] == 42


@pytest.mark.parametrize("rung", [832, 864, 896, 928, 944, 960, 1024, 1088])
def test_the_reused_ids_attest_the_v39_routed_rungs_on_the_glm_image(rung):
    route = _resolve(_routed(rung), NEW_IMAGE)
    assert route.route_status == lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG, route.as_dict()
    assert {r.regime: r.cell_id for r in route.regimes} == REUSED
    assert route.requires_serve_flags == ("TESSERA_SERVE_MODE=resident",)


@pytest.mark.parametrize("image,rung", [(OLD_IMAGE, 1024), (NEW_IMAGE, 800)])
def test_a_withdrawn_image_or_unattested_rung_still_refuses(image, rung):
    route = _resolve(_routed(rung), image)
    assert route.route_status == lane.ROUTE_STATUS_UNATTESTED, route.as_dict()
    assert not [r.cell_id for r in route.regimes if r.cell_id]


@pytest.mark.parametrize("residency", ["resident", "streamed"])
def test_vanilla_retains_dense_e4m3_q1024(residency):
    facts = replace(_routed(1024), structure="dense", qname="model.layers.3.mlp.down_proj")
    route = lane.resolve_unit_route(
        facts, _table(), platform="sm_121", residency=residency,
        runtime_image=VANILLA_IMAGE, execution_mode="eager")
    assert route.route_status == lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG, route.as_dict()
    assert {r.cell_id for r in route.regimes} == {
        "tessera_e4m3_k1_dense_sm121_decode", "tessera_e4m3_k1_dense_sm121_batch"}


def test_glm_image_carries_all_six_family_structure_combinations():
    """Pin the v39 admission decision; derive each cell's rates from its bytes."""
    cells = json.loads(_packaged_bytes())["lane_eligibility"]["cells"]
    glm = [c for c in cells if c["runtime"]["image"] == NEW_IMAGE]
    assert {(c["family"], c["structure"], c["regime"]) for c in glm} == {
        (family, structure, regime)
        for family in ("TESSERA_E4M3_K1", "TESSERA_BF16_K1", "TESSERA_E2M1_K2")
        for structure in ("dense", "routed_moe") for regime in ("decode", "batch")}
    for cell in glm:
        for rung in cell["rungs_q256"]:
            facts = replace(_routed(rung, cell["family"]), structure=cell["structure"])
            route = _resolve(facts, NEW_IMAGE)
            assert route.route_status == lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG, route.as_dict()
            assert cell["id"] in {r.cell_id for r in route.regimes}
    for cell in cells:
        if cell not in glm:
            assert cell["runtime"]["image"] == VANILLA_IMAGE
            assert (cell["family"], cell["structure"], cell["rungs_q256"]) == (
                "TESSERA_E4M3_K1", "dense", [1024])


def test_answer_drift_reports_a_reused_id_whose_scope_moved():
    """Mutation: rewind the reused cells to their v38 scope in a copy of the
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
            row[5] = [896]
            rewound += 1
    assert rewound == 2
    drift = contract._answer_drift(reviewed, installed)
    for cell_id in REUSED.values():
        named = [line for line in drift if f"cells[{cell_id}]" in line]
        assert named and "reviewed" in named[0] and "installed" in named[0], drift


# ---------------------------------------------------------------------------
# GLM layer 43's pick, through the export gate, on the packaged v39 table
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


@pytest.mark.parametrize("rung", [896, 1024])
def test_layer43s_routed_e4m3_pick_passes_the_export_scope_gate(tmp_path, rung):
    """The real gate, the real packaged table, no contract substitution.

    ``require_assignment_scope`` resolves the unit on the reused cells; each
    of them admits q896 through ``cell_lane_admits``.  Since contract v42 the
    cells also launch through the fused routed lane, whose predicate reads
    rate-4 columns only: at q1024 the route records the fused pair beside the
    compact one, and at q896 the lane refuses the plan and the route records
    the compact pair alone -- the stack the dispatch keeps on the compact
    adapter (PQ #1274).
    """
    from prismaquant import tessera_export_lane as export

    model, assignment = _layer43_case(tmp_path, f"TESSERA_E4M3_K1_R{rung}")
    report = export.require_assignment_scope(model, assignment, target=_Target())
    route = report["by_unit"][LAYER43_EXPERT]
    assert route["route_status"] == lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG, route
    assert {row["cell_id"] for row in route["regime_routes"]} == set(REUSED.values())
    table = _table()
    compact = ("tessera.native_window_moe.NativeWindowMoE.__call__",
               "native_window_moe_compact")
    fused = ("tessera.routed_fused.FusedRoutedWindowMoE.__call__",
             "native_routed_fused_window")
    want = [compact, fused] if rung == 1024 else [compact]
    for row in route["regime_routes"]:
        assert sorted((pair["symbol"], pair["decoder"])
                      for pair in row["executes"]) == sorted(want), row
    for cell_id in REUSED.values():
        cell = next(c for c in table.cells if c.id == cell_id)
        claim = lane.lane_claim_for_cell(cell, table.lanes)
        assert claim is not None and claim.extension == "tessera_routed_fused_e4m3"
        assert lane.cell_lane_admits(cell, rung, table.lanes) == (True, "")
        admits, _why, launches = lane.cell_rung_launches(cell, rung, table.lanes)
        assert admits and sorted(launches) == sorted(want)


def test_layer43_at_unattested_q800_is_refused_by_the_export_scope_gate(tmp_path):
    """The broader v39 claim is still bounded by its exact rung set."""
    from prismaquant import tessera_export_lane as export

    model, assignment = _layer43_case(tmp_path, "TESSERA_E4M3_K1_R800")
    with pytest.raises(export.TesseraExportLaneError, match="unattested"):
        export.require_assignment_scope(model, assignment, target=_Target())
