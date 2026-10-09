"""Authorization scope for PQ #2459 GPU qualification (PQ #2471).

The coordinator record
`docs/results/pq2471_serve_evidence_authorization_2026-10-09.md` names the
GLM T-8 serve evidence and the exact cell scope #2459 may qualify. These
tests bind that record to its required contents: the dated serve run and
its receipts, the authorized cells with their rung and TP bounds, the
excluded scopes, and the execution limits. Pin identities below are
literal historical evidence, not a gate against future runtime
identities: a later pin move must not fail this record.
"""
from __future__ import annotations

from pathlib import Path

RECORD = Path(__file__).resolve().parent.parent / "docs" / "results" / \
    "pq2471_serve_evidence_authorization_2026-10-09.md"

AUTHORIZED_CELLS = (
    "tessera_bf16_k1_dense_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
    "tessera_bf16_k1_dense_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
    "tessera_bf16_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
    "tessera_bf16_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
    "tessera_e4m3_k1_dense_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
    "tessera_e4m3_k1_dense_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
    "tessera_e4m3_k1_routed_moe_sm121_batch_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
    "tessera_e4m3_k1_routed_moe_sm121_decode_resident_runtime_1bd4e9052e00b217fe52b40a9b9a9e2445b6cdd504bef051d8410d3fbc72a96e",
)

EXCLUDED_CELLS = (
    "tessera_bf16_k1_dense_sm121_batch_resident",
    "tessera_e2m1_k2_dense_sm121_batch_resident",
    "tessera_e2m1_k2_routed_moe_sm121_batch_resident",
    "tessera_e4m3_k1_dense_sm121_batch",
    "tessera_e4m3_k1_dense_sm121_decode",
)


def test_the_authorization_record_exists():
    assert RECORD.is_file(), \
        "PQ #2471 record missing: no coordinator authorization for #2459"


def test_the_record_carries_the_historical_pin_identities():
    text = RECORD.read_text()
    assert "fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb" in text
    assert "ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e" in text


def test_the_record_binds_the_serve_run_to_its_date_and_receipt():
    text = RECORD.read_text()
    for fact in (
        "u4-R1-20261001T0058Z",
        "2026-10-01",
        "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a",
        "5fbf983a927983c15973e67ae0b9745802322af9d2c7c454be708172b1051d70",
        "0.027885896312391557",
        "tessera#774",
        "glm53-a8-bf16menu-20260930",
        "bb088715",
    ):
        assert fact in text, f"record omits serve fact {fact}"


def test_the_record_names_every_authorized_cell_with_bounds():
    text = RECORD.read_text()
    for cell_id in AUTHORIZED_CELLS:
        assert cell_id in text, f"record omits authorized cell {cell_id}"
    for bound in ("TP 1, 2", "q256", "batch", "decode"):
        assert bound in text, f"record omits cell bound {bound}"


def test_the_record_names_the_excluded_scopes_and_cells():
    text = RECORD.read_text()
    for scope in (
        "gfx1151",
        "gfx1201",
        "compiled",
        "streamed",
        "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5",
        "61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14",
        "TESSERA_E2M1_K2",
        "tessera_route_trace_union",
        "u4-A8-20260928T1809Z",
    ):
        assert scope in text, f"record omits excluded scope {scope}"
    for cell_id in EXCLUDED_CELLS:
        assert cell_id in text, f"record omits excluded cell {cell_id}"


def test_the_record_states_the_execution_limits():
    text = RECORD.read_text()
    for limit in ("priority 0", "30 minutes", "D38", "PrismaBuild"):
        assert limit in text, f"record omits execution limit {limit}"


def test_the_record_disclaims_qualification():
    text = RECORD.read_text()
    assert "does not qualify" in text
