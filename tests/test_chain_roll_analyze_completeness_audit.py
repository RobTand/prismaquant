"""Real analyze CLI census regression; no workload, timing or fleet evidence.

All files are test-owned JSON. Required duration/memory fields are synthetic
serialization inputs, never measurements. No loader, profiler, Netdata, child
process or dispatcher is substituted or invoked; analyze runs without those
optional inputs. The real parser, analyzer, output file and JSON decoding run.
"""
from __future__ import annotations

import json

import pytest

from tools import chain_roll_bench as bench


def _inputs(tmp_path, *, include_head, head_digest="a" * 64):
    runs = [
        {"label": "base-r0", "arm": "base", "repeat": 0, "returncode": 0},
        {"label": "head-r0", "arm": "head", "repeat": 0, "returncode": 0},
    ]
    (tmp_path / "drive.json").write_text(json.dumps({
        "arms": {"base": "synthetic-base", "head": "synthetic-head"},
        "runs": runs,
    }))
    for run in runs if include_head else runs[:1]:
        (tmp_path / (run["label"] + ".json")).write_text(json.dumps({
            "label": run["label"], "arm": run["arm"],
            "roll_s": 1.0, "roll_started_unix": 10.0,
            "roll_ended_unix": 11.0, "memory": {},
            "plane_sha256": "a" * 64 if run["arm"] == "base" else head_digest,
            "fixture_scope": "synthetic report input; no measured workload",
        }))


def _analyze(tmp_path):
    assert bench.main(["analyze", "--out", str(tmp_path)]) == 0
    return json.loads((tmp_path / "report.json").read_text())


def test_analyze_cannot_report_cross_arm_equality_with_a_missing_declared_result(tmp_path):
    _inputs(tmp_path, include_head=False)
    try:
        report = _analyze(tmp_path)
    except (SystemExit, RuntimeError, ValueError) as error:
        assert any(word in str(error).lower() for word in ("missing", "incomplete")), str(error)
    else:
        assert report.get("payloads_identical_across_arms_and_repeats") is not True, (
            "analyze reported cross-arm equality although drive.json declares "
            "head-r0 and its result file is absent", report
        )


@pytest.mark.parametrize("equal", [True, False], ids=["complete_equal", "complete_unequal"])
def test_analyze_preserves_complete_result_controls(tmp_path, equal):
    _inputs(tmp_path, include_head=True, head_digest=("a" if equal else "b") * 64)
    report = _analyze(tmp_path)
    assert set(report["arms"]) == {"base", "head"}
    assert all(arm["repeats"] == 1 for arm in report["arms"].values())
    assert report["payloads_identical_across_arms_and_repeats"] is equal


@pytest.mark.parametrize("defect", [
    "failed_run", "foreign_label", "foreign_arm", "duplicate_label",
    "duplicate_coordinate", "empty_arms", "empty_runs", "unrepresented_arm",
    "undeclared_arm", "invalid_repeat",
])
def test_incomplete_census_refuses_before_profiles_or_report(tmp_path, monkeypatch, defect):
    _inputs(tmp_path, include_head=True)
    drive_path = tmp_path / "drive.json"
    drive = json.loads(drive_path.read_text())
    if defect == "failed_run":
        drive["runs"][1]["returncode"] = 1
    elif defect in ("foreign_label", "foreign_arm"):
        child = tmp_path / "head-r0.json"
        record = json.loads(child.read_text())
        record["label" if defect == "foreign_label" else "arm"] = "foreign"
        child.write_text(json.dumps(record))
    elif defect == "duplicate_label":
        drive["runs"].append({**drive["runs"][0], "repeat": 1})
    elif defect == "duplicate_coordinate":
        drive["runs"].append({**drive["runs"][0], "label": "duplicate"})
    elif defect == "empty_arms":
        drive["arms"] = {}
    elif defect == "empty_runs":
        drive["runs"] = []
    elif defect == "unrepresented_arm":
        drive["arms"]["third"] = "synthetic-third"
    elif defect == "undeclared_arm":
        drive["runs"][1]["arm"] = "foreign"
    elif defect == "invalid_repeat":
        drive["runs"][1]["repeat"] = -1
    drive_path.write_text(json.dumps(drive))
    (tmp_path / "base-r0.raw").write_text("must not be parsed")

    def unused(*args, **kwargs):
        pytest.fail("incomplete census reached profiling or Netdata")

    monkeypatch.setattr(bench, "_profile_shares", unused)
    monkeypatch.setattr(bench, "_netdata", unused)
    with pytest.raises(SystemExit, match="incomplete chain-roll census"):
        bench.main(["analyze", "--out", str(tmp_path), "--netdata", "http://unused"])
    assert not (tmp_path / "report.json").exists()


def test_complete_shuffled_two_repeat_census_preserves_all_results(tmp_path):
    _inputs(tmp_path, include_head=True)
    drive_path = tmp_path / "drive.json"
    drive = json.loads(drive_path.read_text())
    extra = []
    for run in drive["runs"]:
        repeated = {**run, "label": run["arm"] + "-r1", "repeat": 1}
        record = json.loads((tmp_path / (run["label"] + ".json")).read_text())
        record["label"] = repeated["label"]
        (tmp_path / (repeated["label"] + ".json")).write_text(json.dumps(record))
        extra.append(repeated)
    drive["runs"] = [extra[1], drive["runs"][0], extra[0], drive["runs"][1]]
    drive_path.write_text(json.dumps(drive))
    report = _analyze(tmp_path)
    assert all(row["repeats"] == 2 for row in report["arms"].values())
    assert all(row["roll_s"] == [1.0, 1.0] for row in report["arms"].values())
    assert report["payloads_identical_across_arms_and_repeats"] is True
