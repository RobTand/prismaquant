"""The fanout must execute the sealed source in its declared Docker runtime."""
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch


def spec():
    return {
        "model": "/mnt/shared/model", "cwd": "/original/checkout",
        "python": "python3", "campaign_argv": [],
        "env": {"PYTHONPATH": ".:/producer/src", "OMP_NUM_THREADS": "4"},
        "container": {"image": "qualified:fixed", "mounts": [
            {"source": "/mnt/shared", "target": "/mnt/shared"},
            {"source": "/producer", "target": "/producer", "readonly": True},
        ]},
    }


def test_container_row_keeps_row_arguments_and_declared_runtime():
    row = dispatch._row(spec(), ["--units", "/mnt/shared/units.json"],
                        mem_gb=48, timeout_s=600)
    assert row["argv"][:3] == ["python3", "-m", "tools.tessera_campaign_container"]
    cut = row["argv"].index("--")
    assert row["argv"][cut + 1:] == [
        "python3", "-u", "-m", "prismaquant.tessera_campaign",
        "--units", "/mnt/shared/units.json"]
    assert json.loads(row["argv"][4])["container"] == spec()["container"]
    assert row["demand"] == {"gpu": 1, "cpu": 4, "mem_gb": 48}
    assert row["retry_safe"] is True


def test_container_uses_worker_snapshot_and_host_user_without_shell():
    runner = importlib.import_module("tools.tessera_campaign_container")
    argv = runner.docker_command(spec(), ["python3", "a path;$(touch bad)"],
                                 cwd="/worker/snapshot", uid=1000, gid=1001,
                                 image_id="sha256:resolved")
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--user") + 1] == "1000:1001"
    assert "type=bind,src=/worker/snapshot,dst=/workspace,readonly" in argv
    assert not any("/original/checkout" in arg for arg in argv)
    assert "PYTHONPATH=.:/producer/src" in argv
    assert "type=bind,src=/producer,dst=/producer,readonly" in argv
    assert argv[-3:] == ["sha256:resolved", "python3", "a path;$(touch bad)"]
    assert not any(arg.startswith("--cpuset") or arg == "--cgroup-parent" for arg in argv)


@pytest.mark.parametrize("target", ["/", "/workspace", "/workspace/tools", "/mnt/../workspace"])
def test_spec_refuses_mounts_that_hide_the_sealed_source(tmp_path, target):
    data = spec()
    data["container"]["mounts"][0]["target"] = target
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="workspace|canonical"):
        dispatch.load_spec(path)


def test_host_rows_remain_direct():
    data = spec()
    del data["container"]
    assert dispatch._row(data, [], mem_gb=40, timeout_s=60)["argv"] == [
        "python3", "-u", "-m", "prismaquant.tessera_campaign"]


def test_actual_image_content_cannot_be_overridden_and_cpu_mode_requests_no_gpu():
    runner = importlib.import_module('tools.tessera_campaign_container')
    data = spec()
    argv = runner.docker_command(data, ['python3'], cwd='/snapshot', uid=1, gid=1,
        image_id='sha256:resolved', content_sha256='a'*64, with_gpu=False)
    assert '--gpus' not in argv
    assert 'PRISMAQUANT_CONTAINER_CONTENT_SHA256='+'a'*64 in argv
    data['env']['PRISMAQUANT_CONTAINER_CONTENT_SHA256'] = 'b'*64
    with pytest.raises(RuntimeError, match='inspected launcher'):
        runner.validate_container(data)


def test_image_archive_requires_content_digest_and_refuses_changed_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    runner = importlib.import_module('tools.tessera_campaign_container')
    data = spec()
    path = tmp_path/'image.tar'
    path.write_bytes(b'changed archive')
    data['container']['archive'] = dict(path=str(path), sha256='a'*64)
    with pytest.raises(RuntimeError, match='image content digest'):
        runner.validate_container(data)
    data['container']['content_sha256'] = 'b'*64
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert argv[:3] == ['docker', 'image', 'inspect']
        return SimpleNamespace(returncode=1, stdout='', stderr='absent')
    monkeypatch.setattr(runner.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='archive bytes changed'):
        runner.inspect_or_load(data['container'])
    assert len(calls) == 1
