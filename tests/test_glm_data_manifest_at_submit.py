"""Every submitted row carries the byte list the fleet needs to warm it.

A campaign row that reaches PrismaBuild without a data manifest is invisible
to the fleet's prewarm loop, and the row then starts against cold spindles:
26 MB/s over 64 GB on sparky (row-0074, 2026-09-12), about 40 minutes of idle
GPU.  Only the producer knows a row's read set, so ``submit`` derives one for
every row and refuses a row whose read set it cannot derive.

Two properties are load-bearing and neither is visible from the dispatcher
alone:

* the seed bytes come from the row's own ``--seed-wire-dir`` argv.  Reading
  the plan instead is what made every manifest of ``extension-r1024-02``
  declare ``seeds: 0`` while the row read 9.4-19 GB of wire at 41 MB/s
  (row-0065, 2026-09-12);
* the campaign argv is byte-identical with and without the manifest.  The
  manifest is a ``pbrun`` input; the campaign's own checkpoint identity is the
  argv, and a manifest that perturbed it would re-key work already committed.

Nothing here touches the shared mount: the campaign, its captures, its shard
and its seed wire directory are all built under ``tmp_path``.
"""

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch  # noqa: E402

from experiments import glm_data_manifests  # noqa: E402
from experiments.glm_arc_prewarm import RECORD_SIZE  # noqa: E402

MEMBER = "model.layers.0.mlp.experts.0.down_proj"
TENSOR_BYTES = 4 * RECORD_SIZE
CAPTURE_BYTES = 2 * RECORD_SIZE

#: A seed wire directory the way a row's ``--seed-wire-dir`` finds it: blobs
#: at the top, blobs one level down, and one entry that is not a regular file.
SEED_WIRE = {
    "unit-0000.tessera": 3 << 20,
    "unit-0001.tessera": 5 << 20,
    "parts/unit-0002.tessera": 7 << 20,
}


def _workspace(tmp_path):
    """A one-row campaign on disk, plus the seed wire its argv will name."""

    model = tmp_path / "model"
    model.mkdir()
    header = json.dumps({
        MEMBER + ".weight": {"dtype": "F8_E4M3", "shape": [TENSOR_BYTES],
                             "data_offsets": [0, TENSOR_BYTES]},
    }).encode()
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * TENSOR_BYTES)
    (model / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {MEMBER + ".weight": shard.name}}))

    workspace = tmp_path / "workspace"
    captures = workspace / "calibration-cache"
    (captures / "inputs").mkdir(parents=True)
    (captures / "inputs" / "member.pt").write_bytes(b"\0" * CAPTURE_BYTES)
    capture_manifest = captures / "capture_manifest.json"
    capture_manifest.write_text(json.dumps(
        {"entries": {MEMBER: {"path": "inputs/member.pt"}}}))

    units = workspace / "units" / "row-0000.json"
    units.parent.mkdir(parents=True)
    units.write_text(json.dumps({"groups": [{"members": [MEMBER]}]}))

    (workspace / "plan.json").write_text(json.dumps({
        "model": str(model),
        "calibration_cache": {"path": str(capture_manifest)},
        "rows": [{"row_id": "row-0000", "units": str(units),
                  "groups": ["expert-group-0"]}],
    }))

    seed_dir = tmp_path / "other-campaign" / "rows" / "row-0000" / "cache" / "wire"
    for name, size in SEED_WIRE.items():
        blob = seed_dir / name
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"\0" * size)
    # A dangling symlink beside the blobs: the prewarm reader opens every
    # entry O_NOFOLLOW, so a manifest that declared one would only ever record
    # an error against it.
    (seed_dir / "unit-0003.tessera").symlink_to(seed_dir / "absent.tessera")

    return workspace, units, seed_dir


def _row(units, seed_dir, *, extra=()):
    """The pbcampaign row ``plan`` writes, in the shape ``submit`` reads."""

    return {
        "argv": ["python3", "-u", "-m", "prismaquant.tessera_campaign",
                 "--units", str(units),
                 "--seed-wire-dir", str(seed_dir),
                 *extra],
        "cwd": "/home/rob/prismaquant",
        "demand": {"gpu": 1, "cpu": 4, "mem_gb": 104},
        "env": {},
        "tags": ["gb10"],
        "retry_safe": True,
    }


@pytest.fixture
def shared_mount(tmp_path, monkeypatch):
    """Treat the fixture's root as the shared mount the manifest declares."""

    monkeypatch.setattr(glm_data_manifests, "SHARED_MOUNT", str(tmp_path))
    return tmp_path


def test_seed_entries_come_from_the_rows_seed_wire_dir_argv(
    tmp_path, shared_mount,
):
    workspace, units, seed_dir = _workspace(tmp_path)
    campaign = glm_data_manifests.Campaign(str(workspace))
    row = _row(units, seed_dir)

    manifest = glm_data_manifests.build_manifest(
        campaign, "row-0000", {"tool": "test"}, row["argv"])

    declared = {entry["path"]: entry["bytes"] for entry in manifest["entries"]}
    for name, size in SEED_WIRE.items():
        assert declared.get(str(seed_dir / name)) == size, name
    assert manifest["annotations"]["counts"]["seeds"] == len(SEED_WIRE)
    assert manifest["annotations"]["bytes"]["seeds"] == sum(SEED_WIRE.values())
    assert manifest["annotations"]["seed_wire_dir"] == str(seed_dir)
    # The symlink is named in the directory and absent from the manifest.
    assert str(seed_dir / "unit-0003.tessera") not in declared

    # And this is the defect: the plan names no seed for this row, so a
    # manifest built without the argv declares none of those bytes.
    blind = glm_data_manifests.build_manifest(
        campaign, "row-0000", {"tool": "test"})
    assert blind["annotations"]["counts"]["seeds"] == 0
    assert blind["total_bytes"] < manifest["total_bytes"]
    assert manifest["total_bytes"] - blind["total_bytes"] == sum(SEED_WIRE.values())


def test_the_seed_wire_is_declared_last_so_a_short_warm_keeps_the_captures(
    tmp_path, shared_mount,
):
    """Entry order is the row's own read order, and the reader honours it.

    ``prewarm_loop.Reader`` walks ``entries`` in order and stops when ARC
    headroom runs out, so a warm cut short must lose the bytes the row reads
    last -- its seed wire -- rather than the captures it opens first.
    """
    workspace, units, seed_dir = _workspace(tmp_path)
    campaign = glm_data_manifests.Campaign(str(workspace))

    manifest = glm_data_manifests.build_manifest(
        campaign, "row-0000", {"tool": "test"}, _row(units, seed_dir)["argv"])

    seeds = {str(seed_dir / name) for name in SEED_WIRE}
    paths = [entry["path"] for entry in manifest["entries"]]
    first_seed = min(paths.index(path) for path in paths if path in seeds)
    assert all(path in seeds for path in paths[first_seed:])

    # ``annotations.phases`` names that boundary in bytes, so a consumer that
    # warms a prefix does not have to re-derive it from the paths.
    phases = manifest["annotations"]["phases"]
    assert [phase["name"] for phase in phases] == [
        "captures", "weight_extents", "seeds"]
    assert phases[-1]["cumulative_bytes"] == manifest["total_bytes"]
    assert sum(entry["bytes"] for entry in manifest["entries"][:first_seed]) == (
        phases[1]["cumulative_bytes"])
    assert phases[-1]["bytes"] == sum(SEED_WIRE.values())


def test_attaching_a_manifest_leaves_the_campaign_argv_byte_identical(
    tmp_path, shared_mount,
):
    workspace, units, seed_dir = _workspace(tmp_path)
    row = _row(units, seed_dir)
    before = json.dumps(row, sort_keys=True)

    attached = dispatch.attach_data_manifests(workspace, [row])

    assert len(attached) == 1
    assert json.dumps(row, sort_keys=True) == before, "the planned row was mutated"
    assert attached[0]["argv"] == row["argv"]
    assert json.dumps(attached[0]["argv"]) == json.dumps(row["argv"])
    assert set(attached[0]) - set(row) == {"data_manifest"}
    assert all(attached[0][field] == row[field] for field in row)
    assert Path(attached[0]["data_manifest"]).is_file()


def test_the_attached_manifest_is_what_prismabuild_accepts(
    tmp_path, shared_mount,
):
    workspace, units, seed_dir = _workspace(tmp_path)

    attached = dispatch.attach_data_manifests(
        workspace, [_row(units, seed_dir)])

    written = json.loads(Path(attached[0]["data_manifest"]).read_text())
    assert set(written) == set(glm_data_manifests.MANIFEST_KEYS)
    assert written["schema"] == "prismaquant.prismabuild.data_manifest.v1"
    for entry in written["entries"]:
        assert set(entry) == set(glm_data_manifests.ENTRY_KEYS)
    # The producer's own restatement of the contract, applied to the bytes on
    # disk rather than to the object that was written.
    glm_data_manifests.check_manifest(written, where="row-0000")


def test_two_submissions_of_one_campaign_write_identical_manifest_bytes(
    tmp_path, shared_mount,
):
    """The manifest is sealed into the action key, so it must not drift.

    ``pbrun`` ingests the manifest as a content-addressed input.  A hostname
    or a clock reading in ``produced_by`` would give the same row a new action
    key on every submit, and a finished row would stop being a cache hit --
    the opposite of what re-running ``submit`` is for.
    """
    workspace, units, seed_dir = _workspace(tmp_path)
    row = _row(units, seed_dir)

    first = dispatch.attach_data_manifests(workspace, [row])
    before = Path(first[0]["data_manifest"]).read_bytes()
    second = dispatch.attach_data_manifests(workspace, [row])
    after = Path(second[0]["data_manifest"]).read_bytes()

    assert first[0]["data_manifest"] == second[0]["data_manifest"]
    assert before == after
    assert b"unix" not in before and b"host" not in before


def test_a_row_whose_read_set_cannot_be_derived_is_refused(
    tmp_path, shared_mount,
):
    workspace, units, seed_dir = _workspace(tmp_path)

    nameless = _row(units, seed_dir)
    nameless["argv"] = [item for item in nameless["argv"]
                        if "units" not in str(item)]
    with pytest.raises(RuntimeError, match="no single units"):
        dispatch.attach_data_manifests(workspace, [nameless])

    ambiguous = _row(units, seed_dir,
                     extra=["--resume-units", str(units.parent / "row-0007.json")])
    with pytest.raises(RuntimeError, match="no single units"):
        dispatch.attach_data_manifests(workspace, [ambiguous])

    unplanned = _row(units.parent / "row-0009.json", seed_dir)
    with pytest.raises(RuntimeError, match="not a row of"):
        dispatch.attach_data_manifests(workspace, [unplanned])


def test_every_submitted_row_carries_a_manifest(
    tmp_path, shared_mount, monkeypatch,
):
    workspace, units, seed_dir = _workspace(tmp_path)
    rows = [_row(units, seed_dir)]
    (workspace / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    calls = {}

    def _record(manifest, *, wait_s, receipts=None):
        calls["manifest"] = manifest
        return 0

    # ``monkeypatch`` restores the real helper; deleting the attribute instead
    # would leave the module without it for every later test in the process.
    monkeypatch.setattr(dispatch, "_pbcampaign", _record)
    # ``submit`` runs two independent gates: the demand check this campaign's
    # rows were planned under (#522) and the read-set gate under test here.
    # This fixture is a plan written by hand, with no spec to re-derive a
    # demand from, so the demand gate is stood down and the two are exercised
    # together end to end by
    # ``tests/test_tessera_campaign_fanout.py::test_submit_hands_the_fleet_the_admissible_rows_only``.
    monkeypatch.setattr(dispatch, "_checked_manifest",
                        lambda args, *, manifest: [])
    assert dispatch.cmd_submit(
        type("Args", (), {"workspace": str(workspace), "wait_s": 1})) == 0

    submitted = json.loads(Path(calls["manifest"]).read_text())
    assert submitted, "nothing was submitted"
    assert all(row.get("data_manifest") for row in submitted)
    # What ``plan`` wrote is not what was submitted, and is left alone.
    assert Path(calls["manifest"]).name == dispatch.SUBMITTED_MANIFEST
    assert json.loads((workspace / "manifest.json").read_text()) == rows


def test_a_seed_wire_directory_reached_through_a_symlink_is_still_declared(
    tmp_path, shared_mount,
):
    """A row may name a link; the bytes behind it are the bytes it reads."""

    workspace, units, seed_dir = _workspace(tmp_path)
    link = tmp_path / "seed-wire-link"
    link.symlink_to(seed_dir)
    campaign = glm_data_manifests.Campaign(str(workspace))

    manifest = glm_data_manifests.build_manifest(
        campaign, "row-0000", {"tool": "test"}, _row(units, link)["argv"])

    assert manifest["annotations"]["counts"]["seeds"] == len(SEED_WIRE)
    assert manifest["annotations"]["bytes"]["seeds"] == sum(SEED_WIRE.values())
    declared = {entry["path"] for entry in manifest["entries"]}
    assert str(link / "unit-0000.tessera") in declared


def test_a_row_whose_named_seed_wire_yields_nothing_is_refused(
    tmp_path, shared_mount,
):
    """Zero seeds against a named directory is the defect, not an empty row.

    This is the exact shape row-0065 shipped: a manifest declaring
    ``seeds: 0`` for a row that went on to read 9.4-19 GB of wire at 41 MB/s.
    Whatever the cause -- a missing directory, a permission error, a renamed
    campaign -- the submit path refuses it instead of warming nothing.
    """
    workspace, units, seed_dir = _workspace(tmp_path)
    # A checkpoint the row also names, and that exists: the row's seed total
    # is then non-zero even with an empty wire directory, so the gate has to
    # look at the wire directory's own files.
    checkpoint = tmp_path / "seed-checkpoint.safetensors"
    checkpoint.write_bytes(b"\0" * (1 << 20))
    named = ["--seed-checkpoint", str(checkpoint)]

    absent = _row(units, tmp_path / "never-written", extra=named)
    with pytest.raises(SystemExit, match="no readable file was found"):
        dispatch.attach_data_manifests(workspace, [absent])

    empty = tmp_path / "empty-wire"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no readable file was found"):
        dispatch.attach_data_manifests(workspace, [_row(units, empty, extra=named)])

    # The checkpoint on its own is still declared when the wire directory has
    # something in it, so the gate refuses the missing wire, not the pairing.
    ok = dispatch.attach_data_manifests(
        workspace, [_row(units, seed_dir, extra=named)])
    manifest = json.loads(Path(ok[0]["data_manifest"]).read_text())
    assert manifest["annotations"]["counts"]["seeds"] == len(SEED_WIRE) + 1


def test_the_producer_restates_the_limits_prismabuild_enforces(
    tmp_path, shared_mount, monkeypatch,
):
    """The ceilings live with the producer, not only in the fleet.

    ``validate_data_manifest`` refuses more than
    ``DATA_MANIFEST_MAX_ENTRIES`` entries and ``load_data_manifest`` refuses a
    file over ``DATA_MANIFEST_MAX_BYTES``, both before anything is warmed.
    """
    assert glm_data_manifests.MAX_ENTRIES == 1_000_000
    assert glm_data_manifests.MAX_MANIFEST_BYTES == 64 * 1024 * 1024

    workspace, units, seed_dir = _workspace(tmp_path)
    campaign = glm_data_manifests.Campaign(str(workspace))
    manifest = glm_data_manifests.build_manifest(
        campaign, "row-0000", {"tool": "test"}, _row(units, seed_dir)["argv"])

    # The real ceilings are far above any GLM row, so the refusal is exercised
    # by lowering them rather than by building a million entries.
    monkeypatch.setattr(glm_data_manifests, "MAX_ENTRIES", 1)
    with pytest.raises(SystemExit, match="entries exceed 1"):
        glm_data_manifests.check_manifest(manifest)
    with pytest.raises(SystemExit, match="over the"):
        glm_data_manifests.check_manifest_bytes(
            b"x" * (glm_data_manifests.MAX_MANIFEST_BYTES + 1))
    assert glm_data_manifests.check_manifest_bytes(b"{}") == b"{}"
