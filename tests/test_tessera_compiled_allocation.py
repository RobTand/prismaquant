"""#2777: offline mode parity and full-regime refusal, never qualification.

The v5 populations below are synthetic. They exercise the existing explicit
metadata reader and allocator without publishing cells or changing either pin.
"""
import hashlib
import json

import pytest

from prismaquant import tessera_menu as menu
from prismaquant import tessera_runtime_contract as runtime
from prismaquant.lane_eligibility import ServingContext
from test_tessera_scope_endpoints import _v5_contract


LOW = "TESSERA_E4M3_K1_R768"
HIGH = "TESSERA_E4M3_K1_R1024"


def _explicit_population(tmp_path, monkeypatch, *, compiled_regimes):
    payload = _v5_contract(monkeypatch)
    for cell in payload["lane_eligibility"]["cells"]:
        cell["runtime"]["execution_modes"] = ["eager"]
        if cell["regime"] in compiled_regimes:
            cell["runtime"]["execution_modes"].append("compiled")
        if cell["family"] == "TESSERA_E4M3_K1":
            cell["rungs_q256"] = [768, 1024]
    path = tmp_path / "synthetic-contract.json"
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    parsed = runtime.read_tessera_contract(str(path), digest, "synthetic-publisher")
    monkeypatch.setattr(menu, "tessera_runtime_contract", lambda: parsed)
    return parsed, path, digest


def test_pinned_publisher_has_no_complete_compiled_population():
    from prismaquant import tessera_serving_runtime_pin as pins

    pin = pins.load_tessera_serving_runtime_pin()
    parsed = runtime.read_tessera_contract(
        str(runtime.contract_path()), pin.contract_sha256, pin.commit)
    assert parsed.cells
    assert set(parsed.regimes) == {"decode", "batch"}
    eager_populations = 0
    for cell in parsed.cells:
        for rate in cell.rungs_q256:
            for residency in cell.residency_modes:
                values = dict(platform=cell.platform, structure=cell.structure,
                              residency=residency, runtime_image=cell.runtime_image)
                eager_populations += bool(parsed.native_cells(
                    cell.family, rate, serving_context=ServingContext(
                        **values, execution_mode="eager")))
                assert parsed.native_cells(
                    cell.family, rate, serving_context=ServingContext(
                        **values, execution_mode="compiled")) == ()
    assert eager_populations > 0


def test_explicit_publisher_reader_refuses_replaced_bytes(tmp_path, monkeypatch):
    parsed, path, digest = _explicit_population(
        tmp_path, monkeypatch, compiled_regimes={"batch"})
    assert parsed.sha256 == digest
    payload = json.loads(path.read_bytes())
    payload["lane_eligibility"]["cells"][0]["runtime"]["execution_modes"].append("compiled")
    path.write_text(json.dumps(payload))
    with pytest.raises(runtime.TesseraContractError, match="declared digest"):
        runtime.read_tessera_contract(str(path), digest, "synthetic-publisher")
