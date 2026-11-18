"""Joint rows bind images in their container spec.

Generated campaign rows retain their separate image field.
Archive-backed specs retain their declared image bytes and content digest.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from prismaquant import campaign_container_spec as container_spec
from prismaquant import tessera_campaign_rows as campaign_rows

from stage_a_spool_spec import with_spool

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import dispatch_joint_quanta as joint  # noqa: E402

IMAGE = "sha256:" + "a" * 64
OTHER_IMAGE = "sha256:" + "b" * 64
REPO_IMAGE = "pq/stage-a@sha256:" + "c" * 64
ARCHIVE_SHA = "d" * 64
CONTENT_SHA = "e" * 64
GB10_PYTHON = "/home/rob/gb10-venvs/example/bin/python"

IMAGE_FLAG = "--container-image"


def _write_spec(path: Path, container: dict) -> Path:
    # The Stage A row refuses a spec without the produced spool (PQ #1012).
    path.write_text(json.dumps(with_spool({"container": container, "env": {}})))
    return path


def _embedded_spec(argv: list) -> dict:
    """The spec document the submission actually seals inside its argv."""

    command = argv["argv"] if isinstance(argv, dict) else argv
    return json.loads(command[command.index("--spec") + 1])


@pytest.fixture
def joint_spec(tmp_path, monkeypatch):
    spec = _write_spec(tmp_path / "spec.json", {"image": IMAGE})
    monkeypatch.setattr(joint, "SPEC_PATH", spec)
    return spec


def _quantum_record(tmp_path: Path) -> Path:
    import hashlib
    record_path = tmp_path / "layer-001.json"
    slice_path = tmp_path / "layer-001.data-manifest.json"
    slice_path.write_bytes(json.dumps({"slice": "layer-001"}).encode())
    adjoint_path = tmp_path / "layer-001.adjoint-slice.json"
    adjoint_path.write_bytes(b'{"slice": "fixture"}')
    if not (tmp_path / "plan.json").exists():
        (tmp_path / "plan.json").write_text("{}")  # the row reads its plan
    record_path.write_text(json.dumps({
        "schema": joint.RECORD_SCHEMA, "quantum_id": "layer-001", "layer": 1,
        "identity_sha256": "3" * 64,
        "campaign": {"plan_path": str(tmp_path / "plan.json"),
                     "plan_sha256": "1" * 64,
                     "prepared_path": "/fixture/prepared.json",
                     "prepared_sha256": "2" * 64},
        "read_set": {"manifest_path": str(slice_path),
                     "manifest_sha256": hashlib.sha256(
                         slice_path.read_bytes()).hexdigest()},
        "chunks": [{"name": "layer-001-chunk-000"}],
        # The row's stage-A slice file (PQ #993); quantum_argv checks only
        # that its bytes hash to the bound digest.
        "adjoint": {"slice_path": str(adjoint_path),
                    "slice_sha256": hashlib.sha256(adjoint_path.read_bytes()).hexdigest()}}))
    return record_path


def test_quantum_seals_the_image_in_the_spec(tmp_path, joint_spec):
    record_path = _quantum_record(tmp_path)
    record = json.loads(record_path.read_text())
    argv = joint.quantum_argv(record, timeout_s=3600, record_path=record_path,
                              output_root=tmp_path / "out")
    embedded = _embedded_spec(argv)
    assert embedded["container"]["image"] == IMAGE


# ---------------------------------------------------------------------------
# Generated joint rows carry the manifest field
# ---------------------------------------------------------------------------


def _base_spec(**extra) -> dict:
    spec = {"model": "/models/example",
            "campaign_argv": ["--families", "tessera16"],
            "cwd": "/home/rob/example", "python": GB10_PYTHON,
            "env": {"OMP_NUM_THREADS": "1"}}
    spec.update(extra)
    return spec


def _row(spec: dict, **kwargs) -> dict:
    return campaign_rows.campaign_row(spec, ["--model", "/models/example"], mem_gb=34,
                         timeout_s=None, **kwargs)


def test_a_container_row_declares_its_own_spec_image():
    spec = _base_spec(python="python3", container={"image": IMAGE})
    row = _row(spec)
    embedded = _embedded_spec(row["argv"])
    assert row["container_images"] == [embedded["container"]["image"]]
    assert IMAGE_FLAG not in row["argv"]


def test_a_repository_manifest_digest_row_declares_exactly():
    spec = _base_spec(python="python3", container={"image": REPO_IMAGE})
    assert _row(spec)["container_images"] == [REPO_IMAGE]


def test_a_row_without_a_container_declares_nothing_and_is_unchanged():
    row = _row(_base_spec())
    assert "container_images" not in row


def test_a_class_override_declares_the_classes_own_image():
    """The resolved class owns the image; the two rows never trade images."""

    spec = _base_spec(python="python3", container={"image": IMAGE}, classes={
        "rocm-encode": {"python": GB10_PYTHON, "tags": ["gfx1201"],
                        "wire_shared": False, "weights_only": True,
                        "container": {"image": OTHER_IMAGE,
                                      "gpu_runtime": "rocm-wsl"}}})
    default = _row(spec)
    overridden = _row(spec, module="prismaquant.example_encode",
                      row_class_name="rocm-encode")
    assert default["container_images"] == [IMAGE]
    assert overridden["container_images"] == [OTHER_IMAGE]
    assert _embedded_spec(overridden["argv"])["container"]["image"] == OTHER_IMAGE


def test_an_archive_backed_row_declares_nothing_and_keeps_the_loader_path():
    container = {"image": IMAGE, "content_sha256": CONTENT_SHA,
                 "archive": {"path": "/mnt/shared/images/campaign.tar",
                             "sha256": ARCHIVE_SHA}}
    row = _row(_base_spec(python="python3", container=container))
    assert "container_images" not in row
    embedded = _embedded_spec(row["argv"])
    assert embedded["container"] == container


def test_a_malformed_archive_row_is_refused_not_undeclared():
    container = {"image": IMAGE, "content_sha256": CONTENT_SHA,
                 "archive": {"path": "/mnt/shared/images/campaign.tar"}}
    with pytest.raises(RuntimeError, match="archive"):
        _row(_base_spec(python="python3", container=container))


# ---------------------------------------------------------------------------
# the direct submit path (submit-joint / submit-aqua)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the shared derivation
# ---------------------------------------------------------------------------


def _portable_spec(**extra):
    return _base_spec(python="python3", container={
        "image": "campaign:mutable", "content_sha256": CONTENT_SHA},
        container_admission_reference="content:sha256:" + CONTENT_SHA, **extra)


def test_shared_owner_and_campaign_row_honor_portable_admission():
    spec = _portable_spec()
    expected = spec["container_admission_reference"]
    assert container_spec.admission_image_reference(spec) == expected
    row = _row(spec)
    assert row["container_images"] == [expected]
    assert _embedded_spec(row["argv"])["container_admission_reference"] == expected


def test_class_inherits_admission_only_with_its_container():
    spec = _portable_spec(classes={
        "same-image": {"env": {"EXAMPLE": "1"}},
        "other-image": {"container": {"image": OTHER_IMAGE}},
    })
    inherited = _row(spec, row_class_name="same-image")
    replaced = _row(spec, row_class_name="other-image")
    assert inherited["container_images"] == [spec["container_admission_reference"]]
    assert replaced["container_images"] == [OTHER_IMAGE]
    assert "container_admission_reference" not in _embedded_spec(replaced["argv"])


@pytest.mark.parametrize("reference", ["campaign:tag", "content:sha256:abc", 42])
def test_portable_admission_refuses_bad_reference_through_campaign(reference):
    spec = _portable_spec()
    spec["container_admission_reference"] = reference
    with pytest.raises(RuntimeError, match="portable image admission"):
        _row(spec)


def test_portable_admission_requires_inspected_image_identity():
    spec = _portable_spec()
    del spec["container"]["content_sha256"]
    with pytest.raises(RuntimeError, match="portable image admission"):
        container_spec.admission_image_reference(spec)


def test_explicit_archive_admission_keeps_the_joint_override_semantics():
    spec = _portable_spec()
    spec["container"]["archive"] = {"path": "/mnt/shared/images/campaign.tar",
                                     "sha256": ARCHIVE_SHA}
    assert container_spec.admission_image_reference(spec) == spec["container_admission_reference"]


def test_an_archive_backed_container_yields_no_admission_reference():
    container = {"image": IMAGE, "content_sha256": CONTENT_SHA,
                 "archive": {"path": "/mnt/shared/images/campaign.tar",
                             "sha256": ARCHIVE_SHA}}
    assert container_spec.admission_image_reference(
        {"container": container}) is None


def test_an_image_container_yields_the_exact_reference():
    assert container_spec.admission_image_reference(
        {"container": {"image": REPO_IMAGE}}) == REPO_IMAGE


def test_no_container_yields_no_admission_reference():
    assert container_spec.admission_image_reference({}) is None


def test_a_malformed_container_refuses_rather_than_declines():
    with pytest.raises(RuntimeError):
        container_spec.admission_image_reference({"container": {"mounts": []}})


def test_container_spec_refuses_duplicate_mount_destinations():
    mounts = [{"source": "/input-a", "target": "/data", "readonly": True},
              {"source": "/input-b", "target": "/data", "readonly": True}]
    spec = {"container": {"image": IMAGE, "mounts": mounts}}
    before = json.dumps(spec, sort_keys=True)
    with pytest.raises(RuntimeError, match=r"\Aduplicate container mount target: /data\Z"):
        container_spec.validate_container(spec)
    assert json.dumps(spec, sort_keys=True) == before
