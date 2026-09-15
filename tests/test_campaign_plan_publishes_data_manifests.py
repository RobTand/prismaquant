"""Every row ``plan`` publishes already names the bytes the fleet warms it from.

A pbcampaign row without ``data_manifest`` is invisible to PrismaBuild's
prewarm loop and starts against cold spindles: 306.9 MB/s cold against
3298.7 MB/s warm for the same row (2026-09-11). Until this was fixed only
``submit`` attached the manifest, and only to the whole ``manifest.json``.
Every partial release -- ``manifest.remaining-130.json``, the
``remaining-19-*`` sets, the e2m1 gate and rest sets -- was a filtered copy of
``manifest.json`` handed to ``pbcampaign.py`` directly. None of those rows
carried a manifest, and nothing said so.

So the manifest is attached where the rows are born. A subset copied from what
``plan`` published inherits it, ``submit`` re-derives the same bytes at the
same paths, and ``check --manifest`` refuses a row that would still reach the
fleet cold. The manifest is sealed into the action key, so its bytes may not
depend on anything the plan's environment decides: a plan runs as a
PrismaBuild action, where ``HEAD`` is pbrun's snapshot commit rather than the
submitter's, and inside a container, where git refuses a checkout it calls
dubious.

Nothing here touches the shared mount: each test treats its own ``tmp_path``
as the mount the manifest declares.
"""
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch  # noqa: E402

from experiments import glm_data_manifests  # noqa: E402
from test_tessera_campaign_fanout import (  # noqa: E402
    _partition_workspace, _plan_args)


@pytest.fixture
def shared_mount(tmp_path, monkeypatch):
    monkeypatch.setattr(glm_data_manifests, "SHARED_MOUNT", str(tmp_path))
    return tmp_path


@pytest.fixture
def planned(shared_mount):
    """Four rows, planned the way the census plans them, with no box budget."""
    spec, workspace = _partition_workspace(shared_mount, box_memory_gb=None)
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    return spec, workspace


def _row_id(row):
    argv = row["argv"]
    return Path(argv[argv.index("--units") + 1]).stem


def _published(workspace):
    """Every byte ``plan`` publishes, by path."""
    paths = [workspace / "manifest.json", workspace / "plan.json",
             *sorted((workspace / "units").glob("row-*.json")),
             *sorted((workspace / "data-manifests").glob("*.json"))]
    return {path: path.read_bytes() for path in paths if path.is_file()}


def _pre_fix_rows(tmp_path):
    """``manifest.json`` as ``plan`` wrote it before it attached manifests.

    Written out rather than rebuilt with ``_row``: the claim is that ``argv``
    and ``demand`` -- the row's action identity -- are the bytes an unfixed
    planner published, and rebuilding them with the code under test would only
    compare that code with itself.
    """
    workspace = tmp_path / "campaign"
    rows = []
    for index, member in enumerate(["a", "b", "c", "wide"]):
        row_id = f"row-{index:04d}"
        row_dir = workspace / "rows" / row_id
        rows.append({
            "argv": ["python3", "-u", "-m", "prismaquant.tessera_campaign",
                     "--model", str(tmp_path / "model"),
                     "--out", str(row_dir / "cost.pkl"),
                     "--cache-dir", str(row_dir / "cache"),
                     "--checkpoint", str(row_dir / "cost.anchors.json"),
                     "--units", str(workspace / "units" / f"{row_id}.json"),
                     "--calibration-census", str(workspace / "census.json")],
            "cwd": str(tmp_path),
            "demand": {"gpu": 1, "cpu": 4,
                       "mem_gb": 68 if member == "wide" else 5},
            "env": {},
            "tags": ["gb10"],
            "retry_safe": True,
            "progress_phases": ["startup=3600", "pricing=900", "finalize=1800"],
            "timeout_s": 300,
        })
    return rows


def test_every_planned_row_names_a_data_manifest_prismabuild_accepts(planned):
    _, workspace = planned
    rows = json.loads((workspace / "manifest.json").read_text())
    assert [_row_id(row) for row in rows] == [
        "row-0000", "row-0001", "row-0002", "row-0003"]
    for row in rows:
        row_id = _row_id(row)
        assert row.get("data_manifest"), (
            f"{row_id} was published without a data manifest")
        # The only key added, and added last, so the row's own fields keep the
        # layout they had.
        assert list(row)[-1] == "data_manifest"
        path = Path(row["data_manifest"])
        assert path == workspace / "data-manifests" / f"{row_id}.data-manifest.json"
        blob = path.read_bytes()
        assert glm_data_manifests.check_manifest_bytes(blob, where=row_id) == blob
        manifest = glm_data_manifests.check_manifest(json.loads(blob), where=row_id)
        assert manifest["annotations"]["row_id"] == row_id
        assert manifest["total_bytes"] > 0


def test_argv_and_demand_are_byte_identical_to_the_pre_fix_plan(planned, shared_mount):
    _, workspace = planned
    rows = json.loads((workspace / "manifest.json").read_text())
    identity = [{key: value for key, value in row.items() if key != "data_manifest"}
                for row in rows]
    assert json.dumps(identity, indent=2) == json.dumps(
        _pre_fix_rows(shared_mount), indent=2)


def test_the_manifest_bytes_do_not_depend_on_what_git_says(shared_mount, monkeypatch):
    """A plan in a container, or under pbrun, must publish the same bytes.

    Inside a PrismaBuild action ``HEAD`` is pbrun's snapshot commit, and inside
    a container git exits 128 on a checkout owned by another uid. Either one
    moving a byte of the manifest moves the action key of every row, so the
    row a host-side ``submit`` re-derives would not be the row the plan
    published.
    """
    real_run = glm_data_manifests.subprocess.run

    def git_answers(returncode, stdout):
        def run(command, *args, **kwargs):
            if command and command[0] == "git":
                return types.SimpleNamespace(
                    returncode=returncode, stdout=stdout,
                    stderr="" if returncode == 0 else
                    "fatal: detected dubious ownership in repository")
            return real_run(command, *args, **kwargs)
        return run

    spec, workspace = _partition_workspace(shared_mount, box_memory_gb=None)
    monkeypatch.setattr(glm_data_manifests.subprocess, "run",
                        git_answers(0, "a" * 40 + "\n"))
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    before = _published(workspace)
    assert any(path.parent.name == "data-manifests" for path in before), (
        "plan published no data manifest")

    monkeypatch.setattr(glm_data_manifests.subprocess, "run",
                        git_answers(128, ""))
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    assert _published(workspace) == before


def test_a_subset_filtered_from_the_manifest_keeps_its_data_manifest(planned):
    """The operator's partial release, done the way it was done on 09-15."""
    _, workspace = planned
    rows = json.loads((workspace / "manifest.json").read_text())
    subset = [row for row in rows if _row_id(row) in {"row-0001", "row-0003"}]
    gate = workspace / "manifest.gate.json"
    gate.write_text(json.dumps(subset, indent=2) + "\n")

    for row in json.loads(gate.read_text()):
        assert row.get("data_manifest"), f"{_row_id(row)} lost its manifest"
        assert Path(row["data_manifest"]).is_file()
    assert dispatch.main(["check", "--workspace", str(workspace),
                          "--manifest", str(gate)]) == 0


def test_check_refuses_a_subset_that_would_reach_the_fleet_cold(planned):
    """The silent bypass, made loud where a subset is checked."""
    _, workspace = planned
    rows = json.loads((workspace / "manifest.json").read_text())

    stripped = [{key: value for key, value in row.items() if key != "data_manifest"}
                for row in rows[:2]]
    remaining = workspace / "manifest.remaining.json"
    remaining.write_text(json.dumps(stripped, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="data manifest") as refusal:
        dispatch.main(["check", "--workspace", str(workspace),
                       "--manifest", str(remaining)])
    # Every such row is named at once, not one per run.
    assert "row-0000" in str(refusal.value) and "row-0001" in str(refusal.value)

    # A manifest that belongs to another row is not this row's read set.
    swapped = [{**rows[0], "data_manifest": rows[1]["data_manifest"]}]
    remaining.write_text(json.dumps(swapped, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="row-0001"):
        dispatch.main(["check", "--workspace", str(workspace),
                       "--manifest", str(remaining)])


def test_submit_re_derives_the_planned_bytes_and_refuses_drift(planned, monkeypatch):
    _, workspace = planned
    calls = []
    monkeypatch.setattr(dispatch, "_pbcampaign",
                        lambda manifest, *, wait_s, receipts=None:
                        calls.append(manifest) or 0)
    before = _published(workspace)
    args = types.SimpleNamespace(workspace=workspace, wait_s=1)

    assert dispatch.cmd_submit(args) == 0
    submitted = json.loads(Path(calls[0]).read_text())
    assert submitted == json.loads((workspace / "manifest.json").read_text())
    assert _published(workspace) == before

    # A published manifest whose bytes no longer match the row's read set is
    # one a subset may already have sealed into an action key. Submitting the
    # row again under different bytes would run it twice, so it is refused.
    victim = Path(submitted[0]["data_manifest"])
    victim.write_bytes(victim.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="re-plan"):
        dispatch.cmd_submit(args)
    assert len(calls) == 1
    assert victim.read_bytes() == before[victim] + b"\n"


def test_a_row_whose_read_set_cannot_be_derived_refuses_the_plan(
        tmp_path, monkeypatch):
    """Refused before a byte is published, on a first plan and on a re-plan."""
    spec, workspace = _partition_workspace(tmp_path, box_memory_gb=None)
    monkeypatch.setattr(glm_data_manifests, "SHARED_MOUNT",
                        str(tmp_path / "elsewhere"))
    with pytest.raises(SystemExit, match="outside the shared mount"):
        dispatch.cmd_plan(_plan_args(spec, workspace))
    assert sorted(path.name for path in workspace.iterdir()) == ["census.json"]

    monkeypatch.setattr(glm_data_manifests, "SHARED_MOUNT", str(tmp_path))
    assert dispatch.cmd_plan(_plan_args(spec, workspace)) == 0
    before = _published(workspace)
    # The model's index goes missing: no row reads anything any more.
    (tmp_path / "model" / "model.safetensors.index.json").unlink()
    with pytest.raises(SystemExit, match="non-empty"):
        dispatch.cmd_plan(_plan_args(spec, workspace, groups_per_row=2))
    assert _published(workspace) == before
