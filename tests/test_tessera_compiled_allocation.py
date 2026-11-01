"""#2777: offline mode parity and full-regime refusal, never qualification.

The v5 populations below are synthetic. They exercise the existing explicit
metadata reader and allocator without publishing cells or changing either pin.
"""
from copy import deepcopy
import hashlib
import json
import pickle
import sys

import pytest

from prismaquant import allocator
from prismaquant import tessera_menu as menu
from prismaquant import tessera_runtime_contract as runtime
from prismaquant.layer_config import load_assignment
from prismaquant.lane_eligibility import ServingContext, native_rung_verdict
from prismaquant.tessera_formats import parse_tessera_format_name
from test_tessera_scope_endpoints import (
    DENSE, EXPERT, SHARED, IMAGE, _allocator_inputs, _cli_scope, _v5_contract,
)


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


@pytest.mark.parametrize("compiled_regimes", [set(), {"batch"}])
def test_compiled_allocation_requires_decode_and_batch(tmp_path, monkeypatch, compiled_regimes):
    _, path, _ = _explicit_population(
        tmp_path, monkeypatch, compiled_regimes=compiled_regimes)
    snapshot = runtime.read_runtime_snapshot(path)
    for structure in ("dense", "routed_moe"):
        context = ServingContext(platform="sm_121", structure=structure,
                                 residency="resident", runtime_image=IMAGE,
                                 execution_mode="compiled")
        verdict = native_rung_verdict(
            HIGH, snapshot.eligibility, snapshot.published_formats, serving_context=context)
        assert not verdict.admitted
        assert verdict.kind == "regime_population"
    argv = _allocator_inputs(tmp_path, HIGH)
    scope = _cli_scope()
    scope[scope.index("--tessera-execution-mode") + 1] = "compiled"
    monkeypatch.setattr(sys, "argv", ["allocator", *argv, *scope])
    with pytest.raises(SystemExit, match="producer-eligible.*reader-only/unsupported"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


def test_synthetic_complete_population_preserves_allocation_and_actual_modes(tmp_path, monkeypatch):
    _explicit_population(tmp_path, monkeypatch, compiled_regimes={"decode", "batch"})
    argv = _allocator_inputs(tmp_path, HIGH)
    argv[argv.index("--formats") + 1] = LOW + "," + HIGH
    cost_path = tmp_path / "cost.pkl"
    cost = pickle.loads(cost_path.read_bytes())
    cost["formats"] = [LOW, HIGH]
    for name, rows in cost["costs"].items():
        rows[LOW] = deepcopy(rows[HIGH])
        rows[LOW]["output_mse"] = 1e-4 if name == EXPERT else 8e-4
    cost_path.write_bytes(pickle.dumps(cost))
    original_build = allocator.build_candidates
    results = {}
    for mode in ("eager", "compiled"):
        observed = {}
        priced = {}

        def build(*args, **kwargs):
            built = original_build(*args, **kwargs)
            observed.update({name: {candidate.fmt: candidate.memory_bytes
                                    for candidate in candidates}
                             for name, candidates in built.candidates.items()})
            return built

        monkeypatch.setattr(allocator, "build_candidates", build)
        scope = _cli_scope()
        scope[scope.index("--tessera-execution-mode") + 1] = mode
        output = tmp_path / (mode + ".json")
        mode_argv = list(argv)
        mode_argv[mode_argv.index("--layer-config") + 1] = str(output)
        monkeypatch.setattr(sys, "argv", ["allocator", *mode_argv,
                            "--no-fused-aggregation", "--no-packed-aggregation", *scope])
        allocator.main()
        assignment = load_assignment(output)
        assert assignment == {DENSE: HIGH, EXPERT: LOW, SHARED: HIGH}
        metadata = json.loads(output.read_text())["__prismaquant__"]
        target = metadata["tessera_serving_scope"]["target"]
        assert target["execution_mode"] == mode
        assert target["runtime_image"] == IMAGE
        provenance = metadata["serving_lane_provenance"]["by_unit"]
        assert set(provenance) == set(assignment)
        for name, fmt in assignment.items():
            assert provenance[name]["format"] == fmt
            assert provenance[name]["serving_context"]["execution_mode"] == mode
            assert metadata["tessera_serving_scope"]["by_unit"][name]["execution_mode"] == mode
            family, rate = parse_tessera_format_name(fmt)
            priced[name] = (family.name, rate, observed[name][fmt])
        assert metadata["body_assignment_payload_bits_total"] == 8 * sum(
            price[2] for price in priced.values())
        results[mode] = dict(priced)
    assert results["eager"] == results["compiled"]
