"""A portable tag may resolve differently, but its executable content is fixed."""
from copy import deepcopy
from subprocess import CompletedProcess
import json

import pytest

from tools import container_runtime_identity as identity
from tools import tessera_campaign_container as runner


def inspection():
    return {"Id": "sha256:" + "1" * 64, "Os": "linux", "Architecture": "arm64",
            "RootFS": {"Type": "layers", "Layers": ["sha256:" + "2" * 64]},
            "Config": {"Env": ["PATH=/bin"], "Entrypoint": ["/entry"], "Cmd": []}}


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
    inspected, executed = [], []
    def inspect(argv, **kwargs):
        inspected.append(argv)
        return CompletedProcess(argv, 0, stdout=json.dumps([observed]), stderr="")
    def execute(binary, argv):
        executed.append(argv)
        raise SystemExit(0)
    monkeypatch.setattr(runner.subprocess, "run", inspect)
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
