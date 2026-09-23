"""The Stage B preparation's declared reads and its write path (PQ #1070).

``tools/prepare_extended_joint_quanta.py`` and ``tools/regenerate_joint_quanta.py``
run as one PrismaBuild action with a data manifest of what they read and a
write-only produced-output template over the metadata root. These tests hold
the PrismaBuild-free half:

* the read manifest ``tools/stage_b_preparation_submission.py`` writes for a
  generator run names every file that run opens for reading (an audit hook
  records the opens, so a read the manifest misses is a failure, not a guess);
* the manifest's pieces: the parent's head phase, one entry per
  ``(path, offset)``, the safetensors header ranges, the v2 shape PrismaBuild
  validates;
* the template's shape and ceiling;
* ``publish_files`` without a publication writes exactly as the tools did
  before, and ``--produced-output`` is the only way to bind one.

``test_stage_b_prep_produced_1070`` drives the same generator run under a
real admitted owner and checks what PrismaBuild committed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant import joint_layer_quanta as jl  # noqa: E402
from prismaquant import stage_b_prep_io as io  # noqa: E402
from test_stageb_readset_source_coverage import PREFIX, _cli_fixture  # noqa: E402
from test_quantum_executable_readset import _exec_argv  # noqa: E402


# -- the generator-run fixture ----------------------------------------------

def prep_fixture(tmp_path):
    """``_cli_fixture`` with every parent entry a present file under ``tmp_path``.

    The fixture's head entry names ``/fixture/model/shard-h.pt``, which no
    manifest over ``tmp_path`` can declare; it becomes a real 100-byte file.
    Returns ``(campaign, receipt_path, spans)``.
    """
    campaign, receipt_path, spans = _cli_fixture(tmp_path)
    head = tmp_path / "model" / "head.bin"
    head.write_bytes(b"\x01" * 100)
    parent = json.loads(campaign["parent_path"].read_text())
    assert parent["entries"][0]["bytes"] == 100
    parent["entries"][0]["path"] = str(head)
    campaign["parent_path"].write_text(json.dumps(parent))
    campaign["parent_sha"] = hashlib.sha256(
        campaign["parent_path"].read_bytes()).hexdigest()
    return campaign, receipt_path, spans


def generator_args(tmp_path, campaign, receipt_path, metadata_root):
    """The generator's argv as the preparation runs it, minus the head walk.

    ``--head-slices`` needs the plan's real head inputs, which this fixture
    does not carry; the head walk's reads are the parent's head phase, which
    the manifest declares whole (``head_phase_entries``).
    """
    return _exec_argv(tmp_path, campaign) + [
        "--metadata-root", str(metadata_root),
        "--adjoint-receipt", str(receipt_path),
        "--executable-readsets", "--source-layers-prefix", PREFIX]


def submission_args(tmp_path, campaign, receipt_path, metadata_root, out, *,
                    tier="prismabuild-stage:test"):
    return ["regenerate",
            "--plan", str(campaign["plan_path"]), "--plan-sha256", campaign["plan_sha"],
            "--prepared", str(campaign["prepared_path"]),
            "--prepared-sha256", campaign["prepared_sha"],
            "--parent-manifest", str(campaign["parent_path"]),
            "--parent-manifest-sha256", campaign["parent_sha"],
            "--derivation", str(campaign["derivation"]),
            "--partition", str(campaign["partition"]),
            "--adjoint-receipt", str(receipt_path),
            "--executable-readsets", "--source-layers-prefix", PREFIX,
            "--metadata-root", str(metadata_root), "--tier", tier,
            "--out", str(out), "--mount-prefix", str(tmp_path)]


def write_submission(tmp_path, campaign, receipt_path, metadata_root, **kwargs):
    """Run the submission tool; ``(submission, manifest, template)``."""
    import stage_b_preparation_submission as sub

    out = tmp_path / "submission"
    assert sub.main(submission_args(tmp_path, campaign, receipt_path,
                                    metadata_root, out, **kwargs)) == 0
    submission = json.loads((out / "submission.json").read_text())
    manifest = json.loads(gzip.decompress((out / "read-manifest.json.gz").read_bytes()))
    template = json.loads((out / "template.json").read_text())
    return submission, manifest, template


# -- read coverage, measured -------------------------------------------------

_RECORDING: list | None = None


def _audit(event, args):
    if _RECORDING is None or event != "open":
        return
    path, mode, flags = (tuple(args) + (None, None, None))[:3]
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    if isinstance(mode, str):
        reads = "r" in mode or "+" in mode
    else:
        access = (flags or 0) & os.O_ACCMODE
        reads = access in (os.O_RDONLY, os.O_RDWR)
    if reads:
        _RECORDING.append(os.fsdecode(path))


sys.addaudithook(_audit)


def _recorded_reads(call):
    global _RECORDING
    _RECORDING = []
    try:
        result = call()
    finally:
        seen, _RECORDING = _RECORDING, None
    return result, seen


def test_the_manifest_declares_every_file_the_generator_opens(tmp_path):
    import regenerate_joint_quanta as regen

    campaign, receipt_path, _spans = prep_fixture(tmp_path)
    metadata = tmp_path / "metadata-generation"
    submission, manifest, _template = write_submission(
        tmp_path, campaign, receipt_path, metadata)
    declared = {entry["path"] for entry in manifest["entries"]}
    assert submission["reads"] == {"entries": manifest["entry_count"],
                                   "bytes": manifest["total_bytes"]}

    code, opened = _recorded_reads(lambda: regen.main(
        generator_args(tmp_path, campaign, receipt_path, metadata)))
    assert code == 0
    assert (metadata / "records" / "records.json").is_file()
    inputs = set()
    for raw in opened:
        path = Path(os.path.normpath(os.path.abspath(raw)))
        if not path.is_relative_to(tmp_path) or path.is_dir():
            continue
        # The run's own outputs, read back to check what it just published.
        if path.is_relative_to(metadata):
            continue
        inputs.add(str(path))
    assert inputs, "the audit hook recorded no input read"
    undeclared = sorted(inputs - declared)
    assert not undeclared, f"read but not in the data manifest: {undeclared}"


def test_the_manifest_carries_the_header_ranges_the_generator_reads(tmp_path):
    campaign, receipt_path, _spans = prep_fixture(tmp_path)
    _submission, manifest, _template = write_submission(
        tmp_path, campaign, receipt_path, tmp_path / "metadata-generation")
    plan = json.loads(campaign["plan_path"].read_text())
    reads = jl.layer_source_header_reads(plan["model"], 4,
                                         checkpoint_layers_prefix=PREFIX)
    by_key = {(e["path"], e["offset"]): e["bytes"] for e in manifest["entries"]}
    for path, offset, size in reads:
        # A whole-file entry at the same offset covers the range too.
        assert by_key.get((str(path), offset), -1) >= size, path


# -- the manifest's pieces ---------------------------------------------------

def _parent(entries, head_bytes):
    return {"entries": entries, "annotations": {"phases": [
        {"name": "head", "bytes": head_bytes},
        {"name": "layer-0", "bytes": 10}]}}


def test_head_phase_entries_are_the_head_and_only_the_head():
    entries = [{"path": "/mnt/shared/a", "offset": 0, "bytes": 4, "sha256": "x"},
               {"path": "/mnt/shared/b", "offset": 0, "bytes": 6, "sha256": None},
               {"path": "/mnt/shared/c", "offset": 0, "bytes": 10, "sha256": None}]
    head = io.head_phase_entries(_parent(entries, 10))
    assert [e["path"] for e in head] == ["/mnt/shared/a", "/mnt/shared/b"]
    assert head[0]["sha256"] == "x"
    with pytest.raises(ValueError, match="entry boundary"):
        io.head_phase_entries(_parent(entries, 7))
    with pytest.raises(ValueError, match="no leading head phase"):
        io.head_phase_entries({"entries": entries, "annotations": {"phases": []}})


def test_read_entries_keep_one_row_per_path_and_offset(tmp_path):
    whole = tmp_path / "shard.safetensors"
    whole.write_bytes(b"\x00" * 64)
    other = tmp_path / "other.json"
    other.write_bytes(b"{}")
    entries = io.preparation_read_entries(
        head=[{"path": str(other), "offset": 0, "bytes": 2}],
        files=[whole, other],
        ranges=[(str(whole), 0, 16), (str(whole), 32, 8)])
    keys = [(e["path"], e["offset"], e["bytes"]) for e in entries]
    assert keys == [(str(other), 0, 2), (str(whole), 0, 64), (str(whole), 32, 8)]
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty file"):
        io.preparation_read_entries(head=[], files=[empty])


def test_the_read_manifest_is_one_phase_prismabuild_accepts(tmp_path):
    from fleet_sdk import require_prismabuild_sdk
    require_prismabuild_sdk()
    import prismabuild.core as core

    rows = [{"path": f"/mnt/shared/fixture/{name}", "offset": 0, "bytes": size,
             "sha256": None} for name, size in (("a", 3), ("b", 5))]
    manifest = io.preparation_read_manifest(
        rows, produced_by={"tool": "test"}, annotations={"k": 1})
    assert manifest["read_plan"]["phases"] == [
        {"name": io.PREPARATION_READ_PHASE, "entry_indices": [0, 1],
         "bytes": 8, "cumulative_bytes": 8}]
    core.validate_data_manifest(manifest)
    assert len(jl.seal_manifest_bytes(manifest)) < io.DATA_MANIFEST_MAX_BYTES
    with pytest.raises(ValueError, match="outside /mnt/shared"):
        io.preparation_read_manifest(
            [{"path": "/elsewhere/a", "offset": 0, "bytes": 1, "sha256": None}],
            produced_by={}, annotations={})


def test_layer_source_header_reads_name_the_index_and_each_header(tmp_path):
    campaign, _receipt, spans = prep_fixture(tmp_path)
    model = json.loads(campaign["plan_path"].read_text())["model"]
    reads = jl.layer_source_header_reads(model, 4, checkpoint_layers_prefix=PREFIX)
    index = Path(model) / "model.safetensors.index.json"
    assert reads[0] == (str(index), 0, index.stat().st_size)
    shards = {str(path): (offset, size) for path, offset, size in reads[1:]}
    assert shards == {
        str(Path(model) / f"layer-{n}.safetensors"): (
            0, 8 + struct.unpack(
                "<Q", (Path(model) / f"layer-{n}.safetensors").read_bytes()[:8])[0])
        for n in range(4)}
    # The spans the generator completes readsets from are unchanged by the
    # refactor that shares the header reader.
    assert jl.read_layer_source_spans(model, 4, checkpoint_layers_prefix=PREFIX) == spans


# -- the template ------------------------------------------------------------

def test_the_template_is_write_only_with_a_zero_window(tmp_path):
    template = io.build_preparation_template(
        metadata_root=tmp_path / "m", tier="prismabuild-stage:x",
        payload_max_bytes=io.preparation_payload_ceiling(3))
    assert template["write_only"] is True
    assert template["output_prefix"] == str(tmp_path / "m")
    assert template["permitted_tiers"] == ["prismabuild-stage:x"]
    assert template["working_demands"] == {
        "prismabuild-stage:x": {"minimum_gib": 0, "window_gib": 0}}
    assert template["slots"] == {io.PREPARATION_SLOT: {"class": "payload"}}
    maxima = template["durable_maxima"]
    assert maxima["payload_max_bytes"] == maxima["temp_max_bytes"] == (
        (3 * io.PREPARATION_FILES_PER_QUANTUM + io.PREPARATION_CONTROL_FILES)
        * io.DATA_MANIFEST_MAX_BYTES)
    assert maxima["checkpoint_max_bytes"] == 0
    other = io.build_preparation_template(
        metadata_root=tmp_path / "n", tier="prismabuild-stage:x",
        payload_max_bytes=maxima["payload_max_bytes"])
    assert template["template_id"].startswith(io.PREPARATION_TEMPLATE_PREFIX + "-")
    assert other["template_id"] != template["template_id"]
    assert io.preparation_payload_ceiling(3, indexes=4) == (
        maxima["payload_max_bytes"] + 3 * io.DATA_MANIFEST_MAX_BYTES)
    with pytest.raises(ValueError):
        io.preparation_payload_ceiling(0)


def test_the_submission_directory_stays_outside_the_metadata_root(tmp_path):
    import stage_b_preparation_submission as sub

    campaign, receipt_path, _spans = prep_fixture(tmp_path)
    metadata = tmp_path / "metadata-generation"
    assert sub.main(submission_args(tmp_path, campaign, receipt_path, metadata,
                                    metadata / "submission")) == 3
    assert not metadata.exists()


# -- the write path without a publication ------------------------------------

def test_publish_files_without_a_publication_writes_as_before(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "sub" / "b.json"
    b.parent.mkdir()
    assert io.publish_files(None, "k", [(a, b"A", "a"), (b, b"B", "b")]) == [a, b]
    assert a.read_bytes() == b"A" and b.read_bytes() == b"B"
    # Present with the same bytes: left alone, not this call's.
    c = tmp_path / "c.json"
    assert io.publish_files(None, "k", [(a, b"A", "a"), (c, b"C", "c")]) == [c]
    # A differing file refuses before anything is written.
    d = tmp_path / "d.json"
    with pytest.raises(ValueError, match="differing bytes"):
        io.publish_files(None, "k", [(d, b"D", "d"), (a, b"other", "a")])
    assert not d.exists()
    with pytest.raises(ValueError, match="cannot be empty"):
        io.publish_files(None, "k", [(d, b"", "d")])
    with pytest.raises(ValueError, match="twice with different bytes"):
        io.publish_files(None, "k", [(d, b"1", "d"), (d, b"2", "d")])
    assert not d.exists()


def test_only_the_flag_binds_a_publication(tmp_path):
    root = tmp_path / "m"
    action = {"PRISMABUILD_ACTION_KEY": "a" * 64}
    # A test shard is an admitted action too: no flag, no binding.
    assert io.bind_preparation_publication(root, required=False, env=action) is None
    with pytest.raises(io.PreparationPublicationRefused, match="PRISMABUILD_ACTION_KEY"):
        io.bind_preparation_publication(root, required=True, env={})
    with pytest.raises(io.PreparationPublicationRefused, match="--metadata-root"):
        io.bind_preparation_publication(None, required=True, env=action)


def test_the_generator_refuses_the_flag_outside_an_action_before_writing(
        tmp_path, monkeypatch):
    import regenerate_joint_quanta as regen

    monkeypatch.delenv("PRISMABUILD_ACTION_KEY", raising=False)
    io.reset_preparation_publications_for_tests()
    campaign, receipt_path, _spans = prep_fixture(tmp_path)
    metadata = tmp_path / "metadata-generation"
    assert regen.main(generator_args(tmp_path, campaign, receipt_path, metadata)
                      + ["--produced-output"]) == 3
    assert not metadata.exists() or not any(
        path.is_file() for path in metadata.rglob("*"))
