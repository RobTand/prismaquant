"""A portable tag may resolve differently, but its executable content is fixed."""
from copy import deepcopy
import hashlib
from pathlib import Path
from subprocess import CompletedProcess
import json

import pytest

from container_inspection_fixture import inspection
from prismaquant import container_runtime_identity as identity
from tools import tessera_campaign_container as runner


def test_content_identity_survives_backend_ids_and_local_tags():
    original = inspection()
    copied = deepcopy(original)
    copied.update(Id="sha256:" + "3" * 64, RepoTags=["qualified:copy"], Size=123,
                  GraphDriver={"Name": "overlay2"})
    assert identity.image_content_sha256(original) == identity.image_content_sha256(copied)


@pytest.mark.parametrize("field,value", [
    ("RootFS", {"Type": "layers", "Layers": ["sha256:" + "4" * 64]}),
    ("Config", {"Env": ["PATH=/changed"], "Entrypoint": ["/entry"], "Cmd": []}),
    ("Architecture", "amd64"), ("Os", "windows"), ("Variant", "v8"),
])
def test_executable_content_changes_the_seal(field, value):
    changed = {**inspection(), field: value}
    assert identity.image_content_sha256(changed) != identity.image_content_sha256(inspection())


@pytest.mark.parametrize("field,value", [("Config", None), ("RootFS", {}),
                                        ("Architecture", ""), ("Os", None)])
def test_incomplete_inspection_cannot_mint_content_identity(field, value):
    with pytest.raises(identity.RuntimeIdentityError):
        identity.image_content_sha256({**inspection(), field: value})


def test_resolved_image_is_executed_after_content_check(monkeypatch, capsys):
    observed = inspection()
    digest = identity.image_content_sha256(observed)
    head = "9" * 40
    inspected, executed = [], []
    def run(argv, **kwargs):
        # The launcher also reads the checkout's HEAD on the host (#728), so
        # the double answers per binary instead of answering every call as
        # Docker. Answering git as Docker is what made this test red on any
        # working directory that is a checkout. Whether git is called at all
        # depends on the cwd, so the assertions below cover the Docker calls.
        if argv[0] == "git":
            return CompletedProcess(argv, 0, stdout=head if "rev-parse" in argv else "",
                                    stderr="")
        if argv[0] == "docker":
            inspected.append(argv)
            return CompletedProcess(argv, 0, stdout=json.dumps([observed]), stderr="")
        pytest.fail(f"unexpected subprocess call: {argv}")
    def execute(binary, argv):
        executed.append(argv)
        raise SystemExit(0)
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.os, "execvp", execute)
    spec = {"container": {"image": "qualified:portable", "content_sha256": digest}}
    with pytest.raises(SystemExit) as stopped:
        runner.main(["--spec", json.dumps(spec), "--", "python3", "task.py"])
    assert stopped.value.code == 0
    assert inspected == [["docker", "image", "inspect", "qualified:portable"]]
    assert executed[0][-3:] == [observed["Id"], "python3", "task.py"]
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["image_id"] == observed["Id"]
    assert receipt["image_content_sha256"] == digest
    # The launcher reads HEAD only when the working directory is a checkout;
    # either way the value on the receipt is the one the host reported.
    assert receipt["checkout_commit"] == (head if (Path.cwd() / ".git").exists() else None)


def test_changed_portable_tag_refuses_before_container_launch(monkeypatch):
    expected = identity.image_content_sha256(inspection())
    observed = inspection()
    observed["Config"]["Env"] = ["PATH=/changed"]
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kwargs:
                        CompletedProcess(argv, 0, stdout=json.dumps([observed]), stderr=""))
    def forbidden(*args):
        pytest.fail("changed runtime reached container launch")
    monkeypatch.setattr(runner.os, "execvp", forbidden)
    spec = {"container": {"image": "qualified:portable", "content_sha256": expected}}
    with pytest.raises(RuntimeError, match="content.*differs"):
        runner.main(["--spec", json.dumps(spec), "--", "python3", "task.py"])


@pytest.mark.parametrize("value", [None, "short", "A" * 64])
def test_malformed_declared_seal_refuses(value):
    with pytest.raises(RuntimeError, match="content_sha256"):
        runner.validate_container({"container": {"image": "qualified:portable", "content_sha256": value}})


@pytest.fixture
def actual_derivative_build():
    """The retained, audited real Docker build, not a fabricated inspection."""
    path = (Path(__file__).resolve().parents[1] / "experiments/measurements"
            / "glm-derivative-contract-20260908/image-build-result.json")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "e8426d3554180219a0fff421149118a9e0a17ef7a2091c5a03e21834dd1744c5")
    return json.loads(raw)


def test_actual_original_and_corrected_images_keep_their_content_identity(actual_derivative_build):
    from prismaquant.glm_source_derivative import validate_image_build

    build = actual_derivative_build
    assert identity.image_content_sha256(build["original_image"]) == build["original_image_content_sha256"]
    assert identity.image_content_sha256(build["corrected_image"]) == build["corrected_image_content_sha256"]
    assert validate_image_build(build) is build


def test_actual_build_refuses_changed_runtime_configuration(actual_derivative_build):
    from prismaquant.glm_source_derivative import validate_image_build

    build = actual_derivative_build
    build["corrected_image"]["Config"]["Env"].append("PRISMAQUANT_WRONG_RUNTIME=1")
    assert identity.image_content_sha256(build["corrected_image"]) != build["corrected_image_content_sha256"]
    with pytest.raises(ValueError, match="image build config or layer content differs"):
        validate_image_build(build)


def test_actual_image_refuses_missing_runtime_configuration(actual_derivative_build):
    image = actual_derivative_build["original_image"]
    del image["Config"]
    with pytest.raises(identity.RuntimeIdentityError, match="Docker image has no runtime Config"):
        identity.image_content_sha256(image)
