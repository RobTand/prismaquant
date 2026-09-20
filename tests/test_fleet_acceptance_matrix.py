"""The matrix is mechanical: every requirement names evidence that exists."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
MATRIX = json.loads((HERE / "fleet_acceptance_matrix.json").read_text())

LEDGER_IDS = {
    "SC-01", "SC-02", "SC-03",
    "ID-01", "ID-02", "ID-03", "ID-04", "ID-05", "ID-06", "ID-07",
    "ID-08", "ID-09",
    "RNG-01", "RNG-02", "RNG-03", "RNG-04",
    "SM-01", "SM-02", "SM-03",
    "INV-01", "INV-02", "INV-03", "INV-04", "INV-05", "INV-06", "INV-07",
    "INV-08", "INV-09", "INV-10", "INV-11", "INV-12",
    "TIER-01", "TIER-02", "TIER-03", "TIER-04",
    "PRG-01", "PRG-02", "PRG-03", "PRG-04", "PRG-05",
    "LIVE-01",
    "DEV-01", "DEV-02", "DEV-03", "DEV-04", "DEV-05",
    "SAFE-01", "SAFE-02", "SAFE-03",
    "ACC-01", "ACC-02", "ACC-03", "ACC-04", "ACC-05", "ACC-06",
    "VER-01",
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(HERE))
    return module


def test_every_requirement_names_evidence():
    """SC-03/VER-01: all 56 ledger IDs present, vocabulary closed, no blanks."""
    assert MATRIX["schema"] == "prismaquant.fleet_acceptance.matrix.v1"
    rows = {row["id"]: row for row in MATRIX["requirements"]}
    assert set(rows) == LEDGER_IDS, set(rows) ^ LEDGER_IDS
    for rid, row in rows.items():
        assert row["status"].split(":")[0] in (
            "qualified-now", "nonqualified", "out-of-scope"), (rid, row)
        assert row["evidence"], rid
        if row["status"].startswith("nonqualified"):
            assert len(row["status"]) > len("nonqualified:"), rid
        if row["status"].startswith("out-of-scope"):
            assert len(row["status"]) > len("out-of-scope:"), rid


def test_qualified_scenarios_exist_as_tests():
    """A qualified-now row must name runnable tests or runner scenarios."""
    runner = _load_module(HERE / "fleet_acceptance_runner.py",
                          "fleet_matrix_runner")
    for row in MATRIX["requirements"]:
        if not row["status"] == "qualified-now":
            continue
        for scenario in row["scenario"].split("+"):
            scenario = scenario.strip()
            if scenario.startswith("runner:"):
                name = scenario[len("runner:"):]
                assert name in runner.SCENARIOS, (row["id"], name)
            else:
                filename, _, test = scenario.partition("::")
                assert test, (row["id"], scenario)
                module = _load_module(HERE / filename,
                                      "fleet_matrix_" + filename[:-3])
                assert callable(getattr(module, test, None)), (
                    row["id"], scenario)


def test_harness_adds_no_parallel_paths():
    """SC-02: structural scan -- no dispatcher/cache/scheduler surface.

    Calling the real movers (``stage_move.move``) is compliance, not a
    second path; this scan forbids inventing dispatch, cache, or
    scheduler machinery beside PB's.
    """
    forbidden = ("dispatch_tessera_campaign", "pbcampaign", "Cache(",
                 "Scheduler(", "ThreadPoolExecutor(")
    for path in (HERE / "fleet_acceptance_pins.py",
                 HERE / "fleet_acceptance_runner.py",
                 Path(__file__).resolve().parents[1] / "tools"
                 / "fleet_qualification_driver.py"):
        if not path.is_file():
            continue
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (path.name, token)


def test_explicit_gates_name_owners():
    """Out-of-harness scopes stay explicit with owning lanes."""
    gates = MATRIX["explicit_gates"]
    assert len(gates) >= 4
    for gate in gates:
        assert gate["scope"] and gate["owner"], gate
