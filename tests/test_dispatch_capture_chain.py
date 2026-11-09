"""The capture chain dispatcher (PQ #1885).

The rows a sealed chain submits are real ``tessera_campaign`` rows, which
need a model and a GPU; here a fake submitter prints the detach line,
the test writes each row's PrismaBuild ending, and the chain fixture of
``test_capture_layer_chain`` does what the row would have done. So every
output the dispatcher checks before submitting a successor is one the
chain module itself wrote.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dispatch_capture_chain as dispatch  # noqa: E402
from dispatch_capture_chain import ChainDispatchRefused  # noqa: E402

from prismaquant import capture_layer_chain as chain  # noqa: E402
from test_capture_layer_chain import _boundary_policy, _Chain  # noqa: E402

TIMEOUT = 3600
BOOKEND = 900
REFUSED = (ChainDispatchRefused, chain.CaptureChainRefused, ValueError)


from native_dispatch_fixture import FakeNative


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    fixture = _Chain(tmp_path, root=tmp_path / "calibration-cache", prepare=False)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model": str(fixture.source), "campaign_argv": ["--streaming"],
                                "cwd": str(tmp_path), "python": "python3", "env": {}}))
    # The quantum's demand is the monolithic capture row's, from the existing
    # planner; what this test checks is that every row carries one.
    monkeypatch.setattr(dispatch, "row_memory_gb", lambda spec, members, census: 7)
    document = dispatch.seal(spec, tmp_path, ranges="0:1,1:2",
                             boundary_storage=_boundary_policy(fixture.boundaries),
                             timeout_s=TIMEOUT, bookend_timeout_s=BOOKEND)
    return fixture, spec, document, dispatch.round_directory(tmp_path), FakeNative(monkeypatch)


def test_the_sealed_rows_run_in_chain_order_at_priority_minus_ten(sealed):
    fixture, _spec, document, round_dir, _fleet = sealed
    assert [entry["name"] for entry in document["rows"]] == [
        "prep", "capture-000-001", "capture-001-002", "join"]
    assert document["capture_root"] == str(fixture.root)
    for entry in document["rows"]:
        row = entry["row"]
        argv = row["argv"]
        assert row["priority"] == -10
        assert "progress_phases" not in row
        assert argv[argv.index("--capture-chain") + 1] == entry["kind"]
        assert argv[argv.index("--capture-calibration-out") + 1] == str(fixture.root)
        if entry["kind"] == "quantum":
            assert row["timeout_s"] == TIMEOUT and row["demand"]["gpu"] == 1
            assert argv[argv.index("--capture-layer-range") + 1] == "{}:{}".format(*entry["layers"])
        else:
            assert row["timeout_s"] == BOOKEND and row["demand"]["gpu"] == 0
    prep_argv = document["rows"][0]["row"]["argv"]
    assert prep_argv[prep_argv.index("--capture-chain-ranges") + 1] == "0:1,1:2"
    assert json.loads((round_dir / "round.json").read_text()) == document


def test_a_gate_critical_chain_seals_every_row_in_the_band_it_is_given(tmp_path, monkeypatch):
    """PQ #1918: a chain that feeds a gate is campaign work and runs at 0, not -10."""
    fixture = _Chain(tmp_path, root=tmp_path / "calibration-cache", prepare=False)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model": str(fixture.source), "campaign_argv": ["--streaming"],
                                "cwd": str(tmp_path), "python": "python3", "env": {}}))
    monkeypatch.setattr(dispatch, "row_memory_gb", lambda spec, members, census: 7)
    assert dispatch.main(["seal", "--spec", str(spec), "--workspace", str(tmp_path),
                          "--ranges", "0:1,1:2", "--boundary-storage",
                          json.dumps(_boundary_policy(fixture.boundaries)),
                          "--timeout-s", str(TIMEOUT), "--bookend-timeout-s", str(BOOKEND),
                          "--priority", "0"]) == 0
    document = dispatch.load_round(dispatch.round_directory(tmp_path))
    assert [entry["row"]["priority"] for entry in document["rows"]] == [0, 0, 0, 0]


def test_a_quantum_declares_its_progress_phase_only_when_asked(tmp_path, monkeypatch):
    fixture = _Chain(tmp_path, root=tmp_path / "calibration-cache", prepare=False)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model": str(fixture.source), "campaign_argv": ["--streaming"],
                                "cwd": str(tmp_path), "python": "python3", "env": {}}))
    monkeypatch.setattr(dispatch, "row_memory_gb", lambda spec, members, census: 7)
    document = dispatch.seal(spec, tmp_path, ranges="0:1,1:2",
                             boundary_storage=_boundary_policy(fixture.boundaries),
                             timeout_s=TIMEOUT, bookend_timeout_s=BOOKEND,
                             progress_phases=((dispatch.QUANTUM_PROGRESS_PHASE, 600),))
    phases = {entry["name"]: entry["row"].get("progress_phases") for entry in document["rows"]}
    assert phases == {"prep": None, "capture-000-001": ["capture=600"],
                      "capture-001-002": ["capture=600"], "join": None}


@pytest.mark.parametrize("change, message", [
    (dict(ranges="0:1,2:3"), "uncovered"),
    (dict(ranges="0:2,1:3"), "overlaps"),
    (dict(timeout_s=None), "--timeout-s"),
    (dict(bookend_timeout_s=0), "--bookend-timeout-s"),
    (dict(argv=[]), "--streaming"),
])
def test_seal_refuses(tmp_path, monkeypatch, change, message):
    fixture = _Chain(tmp_path, root=tmp_path / "calibration-cache", prepare=False)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"model": str(fixture.source),
                                "campaign_argv": change.pop("argv", ["--streaming"]),
                                "cwd": str(tmp_path), "python": "python3", "env": {}}))
    monkeypatch.setattr(dispatch, "row_memory_gb", lambda spec, members, census: 7)
    options = dict(ranges="0:1,1:2", boundary_storage=_boundary_policy(fixture.boundaries),
                   timeout_s=TIMEOUT, bookend_timeout_s=BOOKEND)
    options.update(change)
    with pytest.raises(REFUSED, match=message):
        dispatch.seal(spec, tmp_path, **options)
    assert not (dispatch.round_directory(tmp_path) / "round.json").exists()



def test_a_chain_is_sealed_once(sealed):
    fixture, spec, _document, _round_dir, _fleet = sealed
    with pytest.raises(ChainDispatchRefused, match="sealed once"):
        dispatch.seal(spec, spec.parent, ranges="0:1,1:2",
                      boundary_storage=_boundary_policy(fixture.boundaries),
                      timeout_s=TIMEOUT, bookend_timeout_s=BOOKEND)


def test_each_row_is_submitted_only_after_its_predecessor_ended_and_left_its_outputs(sealed):
    fixture, _spec, _document, round_dir, fleet = sealed

    def submit(**options):
        return [result["name"] for result in dispatch.submit(round_dir, **options)]

    assert submit() == ["prep"]
    assert submit() == ["prep"]
    assert len(fleet.submitted) == 1
    fleet.end(round_dir, "prep")
    # PrismaBuild says exit 0, but the prep left no record: not ready.
    with pytest.raises(ChainDispatchRefused, match="no readable capture chain prep"):
        submit()
    fixture.prepare()
    assert submit() == ["capture-000-001"]
    fleet.end(round_dir, "capture-000-001")
    # Exit 0 without a complete owner and a fragment is not a finished quantum.
    with pytest.raises(ChainDispatchRefused, match="capture layers 0:1"):
        submit()
    fixture.quantum((0, 1))
    assert submit() == ["capture-001-002"]
    fleet.end(round_dir, "capture-001-002", returncode=1)
    with pytest.raises(ChainDispatchRefused, match="failed"):
        submit()
    with pytest.raises(ChainDispatchRefused, match="cannot retry a producer|cannot retry"):
        submit(retry="capture-000-001")
    assert submit(retry="capture-001-002") == ["capture-001-002"]
    assert (round_dir / "submissions" / "capture-001-002.attempt-1.json").is_file()
    fleet.end(round_dir, "capture-001-002")
    fixture.quantum((1, 2))
    assert submit() == ["join"]
    fleet.end(round_dir, "join")
    document = dispatch.load_round(round_dir)
    with pytest.raises(ChainDispatchRefused, match="no readable receipt"):
        dispatch.require_outputs(document, document["rows"][-1])
    chain.join(fixture.root, census_path=fixture.census)
    dispatch.require_outputs(document, document["rows"][-1])
    assert submit() == []
    assert len(fleet.submitted) == 5


def test_wait_walks_the_chain_and_stops_at_a_row_without_its_outputs(sealed, monkeypatch):
    fixture, _spec, _document, round_dir, fleet = sealed
    original = fleet.collect
    waited = []
    def waiting(state_dir, *, wait=False):
        if wait:
            job_id = fleet.states[str(state_dir)].job_id
            fleet.finish(job_id)
            waited.append(job_id)
            if len(waited) == 1:
                fixture.prepare()
        return original(state_dir, wait=wait)
    from prismaquant import tessera_campaign_rows
    monkeypatch.setattr(tessera_campaign_rows, "collect_rows", waiting)
    with pytest.raises(ChainDispatchRefused, match="capture layers 0:1"):
        dispatch.submit(round_dir, wait=True)
    assert len(waited) == 2
    assert len(fleet.submitted) == 2


def test_a_round_refuses_a_changed_census(sealed):
    fixture, _spec, _document, round_dir, fleet = sealed
    fixture.census.write_text(fixture.census.read_text() + " ")
    with pytest.raises(ChainDispatchRefused, match="census changed"):
        dispatch.submit(round_dir)
