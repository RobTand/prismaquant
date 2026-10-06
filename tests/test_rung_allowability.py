"""Synthetic D41 tables exercise admission mechanics, never measured qualification."""
from __future__ import annotations

import json
import shutil
import os
from pathlib import Path

import pytest

from prismaquant import tessera_lane, tessera_menu

FIXTURE = Path(__file__).parent / "fixtures" / "rung_allowability"
FAMILY = "TESSERA_E4M3_K1"
BUILD = json.loads((FIXTURE / FAMILY / "fixture-t8" / "v0001.json").read_text())["kernel_build"]


@pytest.fixture
def publication(tmp_path, monkeypatch):
    root = tmp_path / "publication"
    shutil.copytree(FIXTURE, root)
    if os.environ.get("TESSERA_RUNG_ALLOWABILITY_MODULE"):
        from prismaquant import rung_allowability
        from _rung_allowability_producer import ExternalProducer
        assert os.environ.get("TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256"), "pin canonical producer bytes"
        producer = ExternalProducer()
        monkeypatch.setattr(rung_allowability, "_producer_api", lambda: producer)
    return root




def _formats():
    from prismaquant.lane_eligibility import load_published_formats
    from prismaquant.tessera_runtime_contract import contract_path
    return load_published_formats(contract_path=contract_path())



def _load(root, **kwargs):
    from prismaquant.rung_allowability import load_rung_allowability
    return load_rung_allowability(root, format_entry=_formats()[FAMILY],
                                 expected_kernel_build=kwargs.pop("build", BUILD), **kwargs)


def _mutate(root, change):
    path = root / FAMILY / "fixture-t8" / "v0001.json"
    table = json.loads(path.read_text())
    change(table)
    path.write_text(json.dumps(table))


@pytest.mark.parametrize("rung,allowed,reason", [
    (1024, True, ""),
    (1025, False, "anomaly"),
    (1026, False, "missing_measurement"),
    (1027, False, "unlisted"),
])
def test_fixture_exclusions_join_existing_producer_admission(publication, rung, allowed, reason):
    table = _load(publication)
    admission = tessera_lane.rung_admission(f"{FAMILY}_R{rung}", allowability={FAMILY: table},
                                           require_allowability=True)
    assert admission.admits(tessera_menu.MENU_RESEARCH) is allowed
    assert reason in admission.detail
    if rung == 1026:
        assert table.refusal(rung) == "missing_measurement"
        assert not hasattr(table, "seal")


def test_production_hook_refuses_without_table():
    with pytest.raises(tessera_menu.TesseraMenuError, match="allowability.*required"):
        tessera_lane.rung_admission(f"{FAMILY}_R1024", require_allowability=True)


def test_production_candidate_builder_requires_table_even_in_research_menu(monkeypatch):
    from prismaquant import allocator_candidates as candidates, format_registry as registry
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "research")
    monkeypatch.setattr(candidates, "check_stats_format_applicability",
                        lambda *a, **k: candidates.FormatApplicability(True))
    spec = registry.get_format(f"{FAMILY}_R1024")
    stats = {"unit": {"in_features": 256, "out_features": 256, "n_params": 65536}}
    costs = {"unit": {spec.name: {"predicted_dloss": 0.1}}}
    with pytest.raises(tessera_menu.TesseraMenuError, match="allowability.*required"):
        candidates.build_candidates(stats, costs, [spec], target_profile="vllm_packed_moe")


def test_index_selects_current_version_not_directory_order(publication):
    path = publication / FAMILY / "fixture-t8" / "v0001.json"
    table = json.loads(path.read_text())
    table["table_version"] = 2
    table["rungs"][0]["measurement_status"] = "pending"
    table["rungs"][0]["supported"] = None
    (path.parent / "v0002.json").write_text(json.dumps(table))
    assert _load(publication).allows(1024)
    index = publication / "index.json"
    payload = json.loads(index.read_text())
    payload["formats"][FAMILY]["kernel_builds"][BUILD["id"]]["current_version"] = 2
    payload["formats"][FAMILY]["kernel_builds"][BUILD["id"]]["versions"]["2"] = {
        "path": f"{FAMILY}/fixture-t8/v0002.json", "table_schema": table["schema"],
        "table_status": table["table_status"]}
    index.write_text(json.dumps(payload))
    assert not _load(publication).allows(1024)


@pytest.mark.parametrize("change,match", [
    (lambda t: t.pop("table_version"), "version|schema key"),
    (lambda t: t.update(table_version=2), "version"),
    (lambda t: t["kernel_build"].update(architecture="other"), "kernel_build"),
    (lambda t: t["scope"].update(grid_step_q256=64), "grid|step"),
    (lambda t: t["rungs"].append(t["rungs"][0]), "duplicate"),
    (lambda t: t["rungs"][0]["measurements"][0].update(kernel_time_us=None), "measured|missing"),
    (lambda t: t["rungs"][0]["measurements"][0]["geometry"].update(alignment=None), "missing"),
    (lambda t: t["rungs"][0].update(quality={}), "quality"),
    (lambda t: t["rungs"][0].update(excluded=True, dominating_rung=960), "adjacent|step"),
])
def test_malformed_or_stale_publication_refuses(publication, change, match):
    _mutate(publication, change)
    with pytest.raises(ValueError, match=match):
        _load(publication)


def test_unreadable_current_table_refuses(publication):
    (publication / FAMILY / "fixture-t8" / "v0001.json").unlink()
    with pytest.raises(ValueError, match="cannot read"):
        _load(publication)


def test_observations_do_not_manufacture_exclusion(publication):
    _mutate(publication, lambda t: t["rungs"][0].update(
        observations=[{"kind": "slow"}, {"kind": "missing_census"}]))
    assert _load(publication).allows(1024)


def test_table_cannot_bypass_existing_run_table_rule(publication):
    from prismaquant.rung_allowability import load_rung_allowability
    row = _formats()[FAMILY]
    row["allowable_rungs"]["excluded_q256"] = [1024]
    assert not load_rung_allowability(publication, format_entry=row,
                                     expected_kernel_build=BUILD).allows(1024)


def test_metadata_only_reader_keeps_the_v56_serving_pin(publication):
    from prismaquant import tessera_runtime_contract as runtime
    from prismaquant import tessera_serving_runtime_pin as pin
    from prismaquant import rung_allowability
    before = runtime.contract_path().read_bytes()
    table = _load(publication)
    assert table.allows(1024)
    assert runtime.contract_path().read_bytes() == before
    assert json.loads(before)["contract_version"] == 56
    assert pin.load_tessera_serving_runtime_pin().commit == runtime.TESSERA_DEV_PIN_COMMIT
    producer = rung_allowability._producer_api()
    if hasattr(producer, "evidence"):
        assert producer.evidence["forbidden_imports"] == []
        print("D41 metadata-only producer:", json.dumps(producer.evidence, sort_keys=True))


def test_allocator_cli_consumes_fixture_and_excludes_cheaper_unmeasured_rows(
        publication, tmp_path, monkeypatch):
    import pickle
    from prismaquant import allocator
    from test_tessera_scope_endpoints import _allocator_inputs, _cli_scope, _v5_contract
    _v5_contract(monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "research")
    argv = _allocator_inputs(tmp_path, f"{FAMILY}_R1024")
    cost_path = Path(argv[argv.index("--costs") + 1])
    payload = pickle.loads(cost_path.read_bytes())
    names = [f"{FAMILY}_R{rung}" for rung in (1024, 1025, 1026, 1027)]
    for rows in payload["costs"].values():
        for name in names[1:]:
            rows[name] = {**rows[names[0]], "weight_mse": 0.0, "output_mse": 0.0}
    payload["formats"] = names
    cost_path.write_bytes(pickle.dumps(payload))
    argv[argv.index("--formats") + 1] = ",".join(names)
    builds = tmp_path / "observed-builds.json"
    builds.write_text(json.dumps({FAMILY: BUILD}))
    allocator.main([*argv, *_cli_scope(), "--no-fused-aggregation", "--no-packed-aggregation",
                    "--tessera-rung-allowability-root", str(publication),
                    "--tessera-rung-kernel-builds", str(builds)])
    layer = json.loads((tmp_path / "layer.json").read_text())
    from prismaquant.layer_config import load_assignment
    assert set(load_assignment(tmp_path / "layer.json").values()) == {names[0]}
    assert layer["__prismaquant__"]["tessera_menu"]["rung_allowability"][FAMILY]["table_version"] == 1


def test_mtp_menu_uses_the_same_measured_rung_input(publication, monkeypatch):
    from prismaquant import allocator, format_registry as registry
    table = _load(publication)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "research")
    monkeypatch.setattr(registry, "format_is_producer_eligible", lambda *a, **k: True)
    eligible = allocator._mtp_rung_attestation(
        None, None, rung_allowability={FAMILY: table}, target_profile="research")
    assert eligible("mtp.unit", f"{FAMILY}_R1024")
    assert not eligible("mtp.unit", f"{FAMILY}_R1025")
    assert not eligible("mtp.unit", f"{FAMILY}_R1026")
    assert not eligible("mtp.unit", f"{FAMILY}_R1027")


def test_build_diagnostics_do_not_become_new_identity_seals(publication):
    _mutate(publication, lambda t: t["kernel_build"].update(
        source_commit="another producer stamp", metadata={"note": "diagnostic only"}))
    assert _load(publication).allows(1024)


def _drift_on_second_index_read(publication, monkeypatch, change):
    from prismaquant import rung_allowability as reader
    original = reader.read_allowability_json
    reads = 0

    def read(path):
        nonlocal reads
        path = Path(path)
        if path == publication / "index.json":
            reads += 1
            if reads == 2:
                index = original(path)
                change(index)
                path.write_text(json.dumps(index))
        return original(path)

    monkeypatch.setattr(reader, "read_allowability_json", read)


def test_publication_drift_stamps_but_unavailable_evidence_refuses_in_both_modes(
        publication, monkeypatch, capsys):
    original_index = (publication / "index.json").read_text()
    change = lambda index: index["formats"].update(OTHER={"kernel_builds": {}})
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    _drift_on_second_index_read(publication, monkeypatch, change)
    assert _load(publication).allows(1024)
    captured = capsys.readouterr()
    assert "[DEV-MODE]" in captured.out + captured.err
    assert "D41 publication identity" in captured.out + captured.err
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    (publication / "index.json").write_text(original_index)
    _drift_on_second_index_read(publication, monkeypatch, change)
    with pytest.raises(ValueError, match="D41 publication identity"):
        _load(publication)
    (publication / "index.json").unlink()
    for mode in ("0", "1"):
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
        with pytest.raises(ValueError, match="cannot read"):
            _load(publication)
    (publication / "index.json").write_text(original_index)
    (publication / FAMILY / "fixture-t8" / "v0001.json").unlink()
    for mode in ("0", "1"):
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
        with pytest.raises(ValueError, match="cannot read"):
            _load(publication)


def test_current_version_drift_keeps_the_stored_measured_selection(
        publication, monkeypatch, capsys):
    path = publication / FAMILY / "fixture-t8" / "v0001.json"
    newer = json.loads(path.read_text())
    newer["table_version"] = 2
    newer["rungs"][0].update(measurement_status="pending", supported=None,
                             measurements=[], quality={})
    (path.parent / "v0002.json").write_text(json.dumps(newer))

    def publish(index):
        build = index["formats"][FAMILY]["kernel_builds"][BUILD["id"]]
        build["current_version"] = 2
        build["versions"]["2"] = {"path": f"{FAMILY}/fixture-t8/v0002.json",
                                  "table_schema": newer["schema"], "table_status": "partial"}

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    _drift_on_second_index_read(publication, monkeypatch, publish)
    stored = _load(publication)
    assert stored.table_version == 1
    assert stored.allows(1024)
    captured = capsys.readouterr()
    assert "[DEV-MODE]" in captured.out + captured.err
    assert "D41 publication identity" in captured.out + captured.err


def test_widened_current_scope_never_borrows_stored_narrow_measurements(
        publication, monkeypatch):
    path = publication / FAMILY / "fixture-t8" / "v0001.json"
    newer = json.loads(path.read_text())
    newer["table_version"] = 2
    newer["scope"]["required_cells"].append(
        {"cell_id": "dense-16", "kernel_kind": "dense", "shape_id": "fixture-shape", "M": 16})
    for row in newer["rungs"]:
        row.update(measurement_status="pending", supported=None, anomaly_flags=[],
                   measurements=[], quality={})
    (path.parent / "v0002.json").write_text(json.dumps(newer))
    original_index = (publication / "index.json").read_text()

    def publish(index):
        build = index["formats"][FAMILY]["kernel_builds"][BUILD["id"]]
        build["current_version"] = 2
        build["versions"]["2"] = {"path": f"{FAMILY}/fixture-t8/v0002.json",
                                  "table_schema": newer["schema"], "table_status": "partial"}

    for mode in ("1", "0"):
        (publication / "index.json").write_text(original_index)
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", mode)
        _drift_on_second_index_read(publication, monkeypatch, publish)
        with pytest.raises(ValueError, match="scope"):
            _load(publication)
