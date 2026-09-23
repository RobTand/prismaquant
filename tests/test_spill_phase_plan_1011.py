"""The executable read plan is sealed for its replay mode (PQ #1011).

Under the one-pass spill (#994/#1003) a quantum reads the own boundary run
once per probe, during that probe's capture inside the first window's retained
lifetime; every window replay reads nothing. A plan sealed per (window, probe)
over-declared the run W times. A spill-sealed plan stages it once per probe in
``spill-p{probe}`` phases after ``render-00``; the windowed plan is unchanged.
The runtime half (phases reported equal the sealed plan, and the quantum
refuses a launch in the other mode) is proved on the real quantum in
``tests/test_stageb_one_pass_spill.py::test_spill_keeps_the_executable_readset_phase_order``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant import joint_layer_quanta as jl  # noqa: E402
from test_quantum_executable_readset import (  # noqa: E402
    CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _bound_inputs,
)
from test_stageb_prepared_render_inputs import (  # noqa: E402
    _bind_prepared, _dispatch_prepared, _prepared_inputs, _render_files,
)


def test_spill_phase_names():
    assert jl.executable_spill_phase_name(3) == "spill-p3"
    with pytest.raises(ValueError):
        jl.executable_spill_phase_name(-1)
    head = ["head", "checkpoint-load", "chain-003-source", "chain-003-bound",
            "own-002-source"]
    assert list(jl.quantum_executable_phase_names(
        [3], 2, n_probes=2, replay_windows=3, render_phases=True,
        replay_mode="spill")) == head + [
        "render-00", "spill-p0", "spill-p1", "render-01", "render-02"]
    assert list(jl.quantum_executable_phase_names(
        [3], 2, n_probes=2, replay_windows=3, replay_mode="spill")) == head + [
        "spill-p0", "spill-p1"]
    assert jl.quantum_executable_phase_names(
        [3], 2, n_probes=2, replay_windows=3, render_phases=True) == \
        jl.quantum_executable_phase_names(
            [3], 2, n_probes=2, replay_windows=3, render_phases=True,
            replay_mode="windowed")
    with pytest.raises(ValueError, match="replay mode"):
        jl.quantum_executable_phase_names([3], 2, n_probes=2, replay_windows=1,
                                          replay_mode="stream")


def _inputs(tmp_path, *, prepared=True):
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    extra = ({"prepared_inputs": _prepared_inputs(record, _render_files(tmp_path))}
             if prepared else {})
    return record, receipt, parent, kwargs, extra


def _build(inputs, *, replay_mode=None):
    record, receipt, parent, _kwargs, extra = inputs
    return jl.build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ), replay_mode=replay_mode, **extra)


@pytest.mark.parametrize("prepared", [True, False])
def test_spill_plan_reads_the_own_boundary_once_per_probe(tmp_path, prepared):
    inputs = _inputs(tmp_path, prepared=prepared)
    record = inputs[0]
    spill = _build(inputs, replay_mode="spill")
    windowed = _build(inputs)
    windows = len(record["windows"])
    assert windows >= 2
    names = [phase["name"] for phase in spill["read_plan"]["phases"]]
    assert names == list(jl.quantum_executable_phase_names(
        [3], 2, n_probes=N_PROBES, replay_windows=windows,
        render_phases=prepared, replay_mode="spill"))
    assert spill["annotations"]["replay_mode"] == "spill"
    by_name = {phase["name"]: phase for phase in windowed["read_plan"]["phases"]}
    own_run = by_name["replay-00-p0"]["entry_indices"]
    run_bytes = by_name["replay-00-p0"]["bytes"]
    spill_phases = [phase for phase in spill["read_plan"]["phases"]
                    if phase["name"].startswith("spill-")]
    assert len(spill_phases) == N_PROBES
    assert all(phase["entry_indices"] == own_run for phase in spill_phases)
    # Every own-boundary entry is counted once per probe, not per window.
    for index in own_run:
        assert sum(index in phase["entry_indices"]
                   for phase in spill["read_plan"]["phases"]) == N_PROBES
    assert (windowed["read_plan"]["read_bytes"] - spill["read_plan"]["read_bytes"]
            == (windows - 1) * N_PROBES * run_bytes)
    # Same staged bytes, fewer repeated reads.
    assert spill["entries"] == windowed["entries"]
    assert spill["read_plan"]["phases"][-1]["cumulative_bytes"] == \
        spill["read_plan"]["read_bytes"]


def test_windowed_plan_is_unchanged(tmp_path):
    inputs = _inputs(tmp_path, prepared=False)
    kwargs = inputs[3]
    default = _build(inputs)
    explicit = _build(inputs, replay_mode="windowed")
    assert "replay_mode" not in default["annotations"]
    assert default == explicit == kwargs["manifest"]
    assert jl.seal_manifest_bytes(default) == jl.seal_manifest_bytes(kwargs["manifest"])


def test_binder_carries_the_mode_and_refuses_the_other(tmp_path):
    bound, receipt, parent, manifest, _files, prepared = _bind_prepared(
        tmp_path, str(tmp_path / "run"), replay_mode="spill")
    assert bound["executable_readset"]["replay_mode"] == "spill"
    assert bound["executable_readset"]["phases"] == [
        phase["name"] for phase in manifest["read_plan"]["phases"]]
    (tmp_path / "w").mkdir()
    windowed, *_ = _bind_prepared(tmp_path / "w", str(tmp_path / "w" / "run"))
    assert "replay_mode" not in windowed["executable_readset"]
    # The spill manifest does not derive from a windowed binding.
    record = {key: value for key, value in bound.items()
              if key not in ("executable_readset", "identity_sha256")}
    record["identity_sha256"] = jl.canonical_sha256(
        record, where=f"quantum record {record['quantum_id']}")
    wire = jl.seal_manifest_bytes(manifest)
    with pytest.raises(ValueError, match="do not originate from the bound inputs"):
        jl.bind_quantum_executable(
            record, receipt, parent, manifest=manifest,
            manifest_path=bound["executable_readset"]["manifest_path"],
            manifest_sha256=hashlib.sha256(wire).hexdigest(),
            output_root=str(tmp_path / "run"), strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite=dict(RENDER_PREREQ), prepared_inputs=prepared)


def test_spill_manifest_passes_pb_phase_planner(tmp_path):
    from test_quantum_executable_readset import _pb
    core, tiers, _plans = _pb()
    manifest = _build(_inputs(tmp_path), replay_mode="spill")
    prefix = str(tmp_path)
    for entry in manifest["entries"]:
        if entry["path"].startswith("/fixture"):
            entry["path"] = "/mnt/shared/fixture" + entry["path"][len("/fixture"):]
        elif entry["path"].startswith(prefix):
            entry["path"] = "/mnt/shared/fixture-renders" + entry["path"][len(prefix):]
    ranges = tiers.manifest_phase_ranges(core.validate_data_manifest(manifest))
    assert [item["name"] for item in ranges] == [
        phase["name"] for phase in manifest["read_plan"]["phases"]]
    assert ranges[-1]["end_bytes"] == manifest["read_plan"]["read_bytes"]


def test_dispatch_main_accepts_a_spill_sealed_row(tmp_path, monkeypatch, capsys):
    (dispatch, bound, _receipt, _manifest, records, receipt_path,
     out) = _dispatch_prepared(tmp_path, monkeypatch, replay_mode="spill")
    gateway = dispatch.FakeGateway()
    code = dispatch.main(
        ["--records", str(records), "--output-root", str(out),
         "--adjoint-receipt", str(receipt_path)], _gateway=gateway,
        # No checkpoint behind the fixture; coverage is tested on its own
        # (test_readset_coverage_1095).
        _coverage=lambda rows: [])
    assert code == 0, capsys.readouterr().err
    argv = gateway.submitted[0]["argv"]
    declared = [argv[i + 1].split("=", 1)[0]
                for i, word in enumerate(argv) if word == "--progress-phase"]
    assert declared.index("render-00") + 1 == declared.index("spill-p0")
    assert declared[-1] == "render-01"
    assert not any(name.startswith("replay-") for name in declared)


def test_dispatch_refuses_a_row_whose_mode_disagrees_with_its_manifest(
        tmp_path, monkeypatch, capsys):
    (dispatch, bound, _receipt, _manifest, records, receipt_path,
     out) = _dispatch_prepared(tmp_path, monkeypatch, replay_mode="spill")
    tampered = json.loads(json.dumps(bound))
    tampered["executable_readset"].pop("replay_mode")
    body = {key: value for key, value in tampered.items() if key != "identity_sha256"}
    tampered["identity_sha256"] = jl.canonical_sha256(
        body, where=f"quantum record {tampered['quantum_id']}")
    (records / "layer-002.json").write_text(json.dumps(tampered))
    gateway = dispatch.FakeGateway()
    code = dispatch.main(
        ["--records", str(records), "--output-root", str(out),
         "--adjoint-receipt", str(receipt_path)], _gateway=gateway)
    assert code != 0 and not gateway.submitted
    assert "replay mode" in capsys.readouterr().err
