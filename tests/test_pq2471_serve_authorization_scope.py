"""Authorization scope for PQ #2459 GPU qualification (PQ #2471).

The coordinator record
`docs/results/pq2471_serve_evidence_authorization_2026-10-09.md` names the
GLM T-8 serve evidence and the exact cell-to-runtime scope #2459 may
qualify. These tests bind that record to the live reviewed answer: the
record must exist, must carry the live pin identity, and its cell map must
equal the contract's own admission table. A scope that drifts from the
reviewed answer fails closed here, not on a GPU box.
"""
from __future__ import annotations

from importlib.resources import as_file
from pathlib import Path

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as tr
from prismaquant import tessera_runtime_contract as contract
from prismaquant import tessera_serving_runtime_pin as pin

RECORD = Path(__file__).resolve().parent.parent / "docs" / "results" / \
    "pq2471_serve_evidence_authorization_2026-10-09.md"


def _table():
    with as_file(tr.tessera_serving_contract_path()) as path:
        return lane.load_eligibility_table(contract_path=path)


def test_the_authorization_record_exists():
    assert RECORD.is_file(), \
        "PQ #2471 record missing: no coordinator authorization for #2459"


def test_the_record_carries_the_live_pin_identity():
    text = RECORD.read_text()
    assert pin.TESSERA_SERVING_RUNTIME_PINNED_COMMIT in text
    assert pin.TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256 in text
    assert contract.TESSERA_DEV_PIN_COMMIT in text


def test_the_record_names_every_live_cell_scope():
    table = _table()
    text = RECORD.read_text()
    cells = sorted(cell.id for cell in table.cells)
    assert len(cells) == 22, f"live answer moved: {len(cells)} cells"
    for cell_id in cells:
        assert cell_id in text, f"record omits live cell {cell_id}"


def test_the_record_maps_each_cell_to_its_live_image():
    table = _table()
    text = RECORD.read_text()
    images = {cell.runtime_image for cell in table.cells}
    assert len(images) == 3, f"live images moved: {sorted(images)}"
    for image in images:
        assert image in text, f"record omits live image {image}"


def test_the_record_names_the_excluded_scopes():
    text = RECORD.read_text()
    for scope in ("gfx1151", "gfx1201", "compiled", "streamed"):
        assert scope in text, f"record omits excluded scope {scope}"


def test_the_record_states_the_execution_limits():
    text = RECORD.read_text()
    for limit in ("priority 0", "30 minutes", "D38", "PrismaBuild"):
        assert limit in text, f"record omits execution limit {limit}"


def test_the_record_disclaims_qualification():
    text = RECORD.read_text()
    assert "does not qualify" in text
