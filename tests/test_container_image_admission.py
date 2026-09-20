"""The campaign container image reaches PrismaBuild's admission contract.

PrismaBuild #714: a GB10-class action was claimed by the Spark that did *not*
hold its pinned Docker image, and died inside ``tools.tessera_campaign_container``
after the attempt was spent.  The image was sealed inside the
``--spec`` JSON and nowhere PB could read it before claiming.  The paired PB
change adds ``pbrun --container-image`` and the campaign row field
``container_images``; this file gates the PrismaQuant callers that must fill
them:

* the joint dispatcher's stage-A and quantum argvs declare exactly the image
  of the same parsed spec they serialize into ``--spec`` (one parse, no
  second-read race), as a pbrun option before the payload separator;
* generated campaign rows carry ``container_images`` from the *resolved*
  row-class container, so a class override declares its own image;
* a spec whose image is supplied by a validated ``container.archive``
  declares nothing -- the launcher's digest-verifying load establishes
  presence inside the action, on whichever worker claims it;
* a row that runs no container is byte-identical to before and declares
  nothing.

These tests are behavioral: they compare the declaration against the spec
bytes actually sealed into the submission (or the row argv), never against a
hard-coded string, so a launcher change and a dispatcher change that drift
apart fail here.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import dispatch_joint_quanta as joint  # noqa: E402
import dispatch_tessera_campaign as campaign  # noqa: E402
import tessera_campaign_container as container_tool  # noqa: E402

IMAGE = "sha256:" + "a" * 64
OTHER_IMAGE = "sha256:" + "b" * 64
REPO_IMAGE = "pq/stage-a@sha256:" + "c" * 64
ARCHIVE_SHA = "d" * 64
CONTENT_SHA = "e" * 64
GB10_PYTHON = "/home/rob/gb10-venvs/example/bin/python"

IMAGE_FLAG = "--container-image"


def _write_spec(path: Path, container: dict) -> Path:
    path.write_text(json.dumps({"container": container, "env": {}}))
    return path


def _embedded_spec(argv: list) -> dict:
    """The spec document the submission actually seals inside its argv."""

    return json.loads(argv[argv.index("--spec") + 1])


def _pbrun_option(argv: list, flag: str) -> str:
    """One pbrun option's value, refused inside the payload instead.

    A declaration the payload carries is invisible to PB, so "present but
    after the separator" must fail exactly like "absent".
    """

    separator = argv.index("--")
    assert flag in argv[:separator], (
        f"{flag} is absent from the pbrun envelope of {argv[0]}")
    return argv[argv.index(flag) + 1]


# ---------------------------------------------------------------------------
# dispatch_joint_quanta: stage A and the quanta
# ---------------------------------------------------------------------------


@pytest.fixture
def joint_spec(tmp_path, monkeypatch):
    spec = _write_spec(tmp_path / "spec.json", {"image": IMAGE})
    monkeypatch.setattr(joint, "SPEC_PATH", spec)
    return spec


def _joint_campaign(tmp_path: Path) -> dict:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"output_root": str(tmp_path / "campaign-root")}))
    return {"plan_sha256": "1" * 64, "prepared_sha256": "2" * 64,
            "plan_path": str(plan), "prepared_path": "/fixture/prepared.json"}


def _quantum_record(tmp_path: Path) -> Path:
    record_path = tmp_path / "layer-001.json"
    record_path.write_text(json.dumps({
        "schema": joint.RECORD_SCHEMA, "quantum_id": "layer-001", "layer": 1,
        "identity_sha256": "3" * 64,
        "read_set": {"manifest_path": "manifests/layer-001.data-manifest.json.gz"},
        "chunks": [{"name": "layer-001-chunk-000"}]}))
    return record_path


def test_stage_a_declares_the_image_it_seals_into_the_spec(tmp_path, joint_spec):
    argv = joint.stage_a_argv(tmp_path / "adjoint.manifest", _joint_campaign(tmp_path))
    embedded = _embedded_spec(argv)
    assert _pbrun_option(argv, IMAGE_FLAG) == embedded["container"]["image"]


def test_quantum_declares_the_image_it_seals_into_the_spec(tmp_path, joint_spec):
    record_path = _quantum_record(tmp_path)
    record = json.loads(record_path.read_text())
    argv = joint.quantum_argv(record, record_path=record_path,
                              output_root=tmp_path / "out")
    embedded = _embedded_spec(argv)
    assert _pbrun_option(argv, IMAGE_FLAG) == embedded["container"]["image"]


def test_the_declaration_comes_from_the_same_parse_as_the_sealed_spec(
        tmp_path, monkeypatch, joint_spec):
    """A second read of the spec file cannot race the sealed bytes.

    The spec file answers with one image on the first read and another on the
    second; a dispatcher that parses twice would seal one identity and declare
    the other.  One parse means one read and one image.
    """

    first = {"container": {"image": IMAGE}, "env": {}}
    second = {"container": {"image": OTHER_IMAGE}, "env": {}}
    reads = {"spec": 0}
    real_read_text = Path.read_text

    def two_faced(self, *args, **kwargs):
        if self == joint_spec:
            reads["spec"] += 1
            return json.dumps(first if reads["spec"] == 1 else second)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", two_faced)
    argv = joint.stage_a_argv(tmp_path / "adjoint.manifest", _joint_campaign(tmp_path))
    embedded = _embedded_spec(argv)
    assert reads["spec"] == 1
    assert embedded["container"]["image"] == IMAGE
    assert _pbrun_option(argv, IMAGE_FLAG) == IMAGE


def test_a_repository_manifest_digest_is_declared_exactly(tmp_path, monkeypatch):
    spec = _write_spec(tmp_path / "spec.json", {"image": REPO_IMAGE})
    monkeypatch.setattr(joint, "SPEC_PATH", spec)
    argv = joint.stage_a_argv(tmp_path / "adjoint.manifest", _joint_campaign(tmp_path))
    assert _pbrun_option(argv, IMAGE_FLAG) == REPO_IMAGE


def test_an_archive_backed_spec_declares_no_image(tmp_path, monkeypatch):
    """The archive supplies the image on any worker; PB must not pre-gate it."""

    container = {"image": IMAGE, "content_sha256": CONTENT_SHA,
                 "archive": {"path": "/mnt/shared/images/campaign.tar",
                             "sha256": ARCHIVE_SHA}}
    spec = _write_spec(tmp_path / "spec.json", container)
    monkeypatch.setattr(joint, "SPEC_PATH", spec)
    argv = joint.stage_a_argv(tmp_path / "adjoint.manifest", _joint_campaign(tmp_path))
    assert IMAGE_FLAG not in argv
    embedded = _embedded_spec(argv)
    assert embedded["container"]["archive"] == container["archive"]
    assert embedded["container"]["content_sha256"] == CONTENT_SHA


def test_a_malformed_archive_is_refused_not_silently_undeclared(tmp_path, monkeypatch):
    """A broken archive is not an excuse to skip the loader's contract."""

    container = {"image": IMAGE, "content_sha256": CONTENT_SHA,
                 "archive": {"path": "images/campaign.tar", "sha256": ARCHIVE_SHA}}
    spec = _write_spec(tmp_path / "spec.json", container)
    monkeypatch.setattr(joint, "SPEC_PATH", spec)
    with pytest.raises(RuntimeError, match="archive"):
        joint.stage_a_argv(tmp_path / "adjoint.manifest", _joint_campaign(tmp_path))


def test_the_dry_run_plan_declares_the_stage_a_image(tmp_path, joint_spec, capsys):
    """End to end through the dispatcher's own plan output."""

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"output_root": str(tmp_path / "campaign-root")}))
    records = tmp_path / "records"
    records.mkdir()
    record_path = _quantum_record(tmp_path)
    record = json.loads(record_path.read_text())
    record["campaign"] = {"plan_path": str(plan), "plan_sha256": "1" * 64,
                          "prepared_path": "/fixture/prepared.json",
                          "prepared_sha256": "2" * 64}
    record["adjoint"] = {"receipt_sha256": None}
    (records / record_path.name).write_text(json.dumps(record))
    code = joint.main(["--records", str(records),
                       "--output-root", str(tmp_path / "out"),
                       "--adjoint-manifest", str(tmp_path / "adjoint.manifest"),
                       "--dry-run"])
    assert code == 0
    plan_out = json.loads(capsys.readouterr().out)
    assert [row["kind"] for row in plan_out["rows"]] == ["stage-a"]
    argv = plan_out["rows"][0]["argv"]
    assert _pbrun_option(argv, IMAGE_FLAG) == _embedded_spec(argv)["container"]["image"]


# ---------------------------------------------------------------------------
# dispatch_tessera_campaign: generated rows carry the manifest field
# ---------------------------------------------------------------------------


def _base_spec(**extra) -> dict:
    spec = {"model": "/models/example",
            "campaign_argv": ["--families", "tessera16"],
            "cwd": "/home/rob/example", "python": GB10_PYTHON,
            "env": {"OMP_NUM_THREADS": "1"}}
    spec.update(extra)
    return spec


def _row(spec: dict, **kwargs) -> dict:
    return campaign._row(spec, ["--model", "/models/example"], mem_gb=34,
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


def _direct_args(tmp_path: Path, container: dict) -> types.SimpleNamespace:
    spec = _write_spec(tmp_path / "spec.json", container)
    return types.SimpleNamespace(
        spec=spec, pbrun="/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py",
        demand="gpu=1,mem_gb=104", cpus=6, tag="sparky", priority=-10,
        timeout_s=None, container_arg=None, residency=None, head_grace_s=1800)


def _direct_argv(tmp_path: Path, container: dict) -> list:
    args = _direct_args(tmp_path, container)
    spec = json.loads(Path(args.spec).read_text())
    return campaign._pbrun_argv(args, manifest=tmp_path / "read.json",
                                inner=["--resume"], container_spec=spec)


def test_a_direct_submission_declares_the_resolved_spec_image(tmp_path):
    argv = _direct_argv(tmp_path, {"image": IMAGE})
    embedded = _embedded_spec(argv)
    assert _pbrun_option(argv, IMAGE_FLAG) == embedded["container"]["image"]


def test_a_direct_submission_with_an_archive_declares_no_image(tmp_path):
    container = {"image": IMAGE, "content_sha256": CONTENT_SHA,
                 "archive": {"path": "/mnt/shared/images/campaign.tar",
                             "sha256": ARCHIVE_SHA}}
    argv = _direct_argv(tmp_path, container)
    assert IMAGE_FLAG not in argv
    assert _embedded_spec(argv)["container"]["archive"] == container["archive"]


def test_a_direct_submission_without_a_container_declares_nothing(tmp_path):
    """The opaque pre-container spec shape some callers still pass."""

    args = _direct_args(tmp_path, {"image": IMAGE})
    argv = campaign._pbrun_argv(args, manifest=tmp_path / "read.json",
                                inner=["--resume"], container_spec={"image": "x"})
    assert IMAGE_FLAG not in argv


# ---------------------------------------------------------------------------
# the shared derivation
# ---------------------------------------------------------------------------


def test_an_archive_backed_container_yields_no_admission_reference():
    container = {"image": IMAGE, "content_sha256": CONTENT_SHA,
                 "archive": {"path": "/mnt/shared/images/campaign.tar",
                             "sha256": ARCHIVE_SHA}}
    assert container_tool.admission_image_reference(
        {"container": container}) is None


def test_an_image_container_yields_the_exact_reference():
    assert container_tool.admission_image_reference(
        {"container": {"image": REPO_IMAGE}}) == REPO_IMAGE


def test_no_container_yields_no_admission_reference():
    assert container_tool.admission_image_reference({}) is None


def test_a_malformed_container_refuses_rather_than_declines():
    with pytest.raises(RuntimeError):
        container_tool.admission_image_reference({"container": {"mounts": []}})
