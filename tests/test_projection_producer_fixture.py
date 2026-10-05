"""Missing host-specific producer bytes are explicit, not false product failures."""
import errno
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

import projection_producer_fixture as fixture
from prismaquant.tessera_expert_projection import PRODUCER_PYTHON_ENV


@pytest.fixture
def declaration(tmp_path, monkeypatch):
    document = json.loads(fixture.DECLARATION.read_text())
    document["required_host"] = "dl380g10"
    path = tmp_path / "producer.json"
    monkeypatch.setattr(fixture, "DECLARATION", path)
    monkeypatch.delenv(PRODUCER_PYTHON_ENV, raising=False)
    return document, path


@pytest.mark.parametrize("host", ["sparky", "sparklina"])
def test_missing_interpreter_skips_elsewhere_with_named_reason(declaration, monkeypatch, host):
    document, path = declaration
    missing = str(path.parent / "missing-producer" / "bin" / "python")
    document["interpreter"] = missing
    path.write_text(json.dumps(document))
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    with pytest.raises(pytest.skip.Exception, match="missing.*" + missing):
        fixture.require_projection_producer(monkeypatch)


@pytest.mark.parametrize("host", ["dl380g10", "dl380g10.example.test"])
def test_missing_interpreter_fails_on_declared_host(declaration, monkeypatch, host):
    document, path = declaration
    missing = str(path.parent / "missing-producer" / "bin" / "python")
    document["interpreter"] = missing
    path.write_text(json.dumps(document))
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    with pytest.raises(pytest.fail.Exception, match="missing.*" + missing):
        fixture.require_projection_producer(monkeypatch)


@pytest.mark.parametrize("host", ["dl380g10", "sparky"])
@pytest.mark.parametrize("drift", [None, "executable_sha256", "module_sha256", "package_payload_sha256"])
def test_present_interpreter_is_authenticated_on_every_host(declaration, monkeypatch, host, drift):
    document, path = declaration
    interpreter = path.parent / "python"
    interpreter.write_bytes(b"synthetic executable identity probe")
    document["interpreter"] = str(interpreter)
    path.write_text(json.dumps(document))
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    monkeypatch.setattr(fixture, "producer_plan_tool", lambda: document["module"])
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        observed = {key: document[key] for key in (
            "executable_sha256", "module_sha256", "package_payload_sha256")}
        if drift:
            observed[drift] = "0" * 64
        return SimpleNamespace(stdout=json.dumps(observed))

    monkeypatch.setattr(fixture.subprocess, "run", probe)
    if drift:
        with pytest.raises(AssertionError, match=drift + " drift"):
            fixture.require_projection_producer(monkeypatch)
    else:
        assert fixture.require_projection_producer(monkeypatch) == str(interpreter)
        assert fixture.producer_plan_tool() == document["module"]
    assert calls == [[str(interpreter), "-c", fixture.SOURCE_PROBE]]


def test_declared_producer_runs_on_its_required_host(monkeypatch):
    document = json.loads(fixture.DECLARATION.read_text())
    interpreter = fixture.require_projection_producer(monkeypatch)
    assert interpreter == document["interpreter"]
    assert Path(interpreter).is_file()


@pytest.mark.parametrize("host", ["dl380g10", "sparky"])
@pytest.mark.parametrize("error", [
    PermissionError(errno.EACCES, "permission denied"),
    OSError(errno.EIO, "input/output error"),
    NotADirectoryError(errno.ENOTDIR, "ancestor is not a directory"),
], ids=["permission", "input-output", "not-a-directory"])
def test_non_absence_stat_error_fails_with_path_and_error(declaration, monkeypatch, host, error):
    document, path = declaration
    interpreter = path.parent / "python"
    interpreter.write_bytes(b"present producer interpreter")
    document["interpreter"] = str(interpreter)
    path.write_text(json.dumps(document))
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    real_stat = fixture.os.stat

    def broken_stat(target, *args, **kwargs):
        if str(target) == str(interpreter):
            raise error
        return real_stat(target, *args, **kwargs)

    monkeypatch.setattr(fixture.os, "stat", broken_stat)
    with pytest.raises(pytest.fail.Exception) as caught:
        try:
            fixture.require_projection_producer(monkeypatch)
        except pytest.skip.Exception as skipped:
            raise AssertionError("a non-absence stat error became a skip") from skipped
    assert str(interpreter) in str(caught.value)
    assert type(error).__name__ in str(caught.value)


@pytest.mark.parametrize("host", ["dl380g10", "sparky"])
def test_non_regular_interpreter_fails_on_every_host(declaration, monkeypatch, host):
    document, path = declaration
    interpreter = path.parent / "python"
    interpreter.mkdir()
    document["interpreter"] = str(interpreter)
    path.write_text(json.dumps(document))
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    with pytest.raises(pytest.fail.Exception) as caught:
        try:
            fixture.require_projection_producer(monkeypatch)
        except pytest.skip.Exception as skipped:
            raise AssertionError("a non-regular interpreter became a skip") from skipped
    assert str(interpreter) in str(caught.value)
    assert "not a regular file" in str(caught.value)
