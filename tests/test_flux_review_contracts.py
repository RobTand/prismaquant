"""Regression contracts for the native Flux review findings."""
import inspect
from types import SimpleNamespace

import pytest

from prismaquant import staged_launch
from prismaquant import ship_gates
from tools import staged_read_stream_ab


@pytest.fixture
def launch_environment(monkeypatch):
    for name in (staged_launch.PATH_ENV, staged_launch.SHA_ENV, staged_launch.READ_ENV,
                 "PRISMAQUANT_READ_CONTRACT_PATH", "PRISMAQUANT_STAGED_ALLOWED_TIERS"):
        monkeypatch.setenv(name, "")


def test_host_local_launch_refuses_an_unpinned_cache_helper(tmp_path, monkeypatch, launch_environment):
    helper = tmp_path / "fluxcache.py"
    helper.write_text("raise SystemExit('modified')\n")
    monkeypatch.setattr(staged_launch, "CACHE_HELPER", str(helper))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    manifest = SimpleNamespace(lifetime={"mode": "host_local_cache", "root": str(tmp_path)}, members=lambda: [])
    monkeypatch.setattr(staged_launch, "validate_binding", lambda binding: manifest)
    calls = []
    monkeypatch.setattr(staged_launch.os, "execv", lambda *args: calls.append(args))
    monkeypatch.setattr(staged_launch.os, "execvpe", lambda *args: calls.append(args))
    binding = {"path": "manifest", "sha256": "a" * 64, "read_contract_sha256": "b" * 64,
               "read_contract_path": "contract", "allowed_tiers": "ssd"}
    with pytest.raises(ValueError, match="cache helper sha256"):
        staged_launch.launch(binding, ["/bin/true"])
    assert calls == []
    assert not (tmp_path / "staged-members.json").exists()


def test_missing_stage_mount_refuses_instead_of_inventing_measurements():
    with pytest.raises(ValueError, match="REFUSED: stage mount /missing has no NFS read counters"):
        staged_read_stream_ab.stage_counters({"/mnt/shared": (12, 8)}, "/missing")
    assert staged_read_stream_ab.stage_counters({"/stage": (42, 13)}, "/stage") == (42, 13)


@pytest.mark.parametrize("argv", [
    ["flux", "submit", "true"], ["/bin/flux", "run", "true"],
    ["sh", "-c", "flux submit true"], ["bash", "-lc", "/bin/flux run true"],
    ["bash", "-c", "true; flux submit true"],
    ["python3", "-m", "prismaquant.flux_jobs"],
    ["python3", "-mprismaquant.flux_jobs"],
    ["python3", "-c", "import prismaquant.flux_jobs as jobs; jobs.submit_job()"],
    ["python3", "-c", "import subprocess; subprocess.run(['flux', 'submit', 'true'])"],
    ["python3", "-c", "import os; os.system('flux run true')"],
    ["bash", "--rcfile", "/dev/null", "-c", "flux submit true"],
    ["bash", "--noprofile", "-lc", "flux run true"],
    ["sh", "-c", "python3 -m prismaquant.flux_jobs"],
    ["bash", "-c", "sh -c 'flux submit true'"],
    ["sh", "-c", 'true; bash -c "flux submit true"'],
    ["env", "SETTING=1", "sh", "-c", "flux submit true"],
])
def test_stage_commands_refuse_nested_submission_forms(argv):
    assert ship_gates._submits_nested_job(argv)


@pytest.mark.parametrize("argv", [
    ["python3", "-m", "prismaquant.ship_gates"], ["/bin/true"],
    ["python3", "-c", "print('flux')"],
    ["python3", "-m", "other_module", "-mprismaquant.flux_jobs"],
    ["bash", "-c", "echo flux-results.json"], ["python3", "flux_report.py"],
])
def test_ordinary_stage_commands_remain_allowed(argv):
    assert not ship_gates._submits_nested_job(argv)


def test_immutable_coverage_has_no_wait_contract():
    from prismaquant import residency_shard_reader as reader
    assert set(inspect.signature(reader.await_staged_spans).parameters) == {
        "resolver", "wanted", "cancel"}
    assert not hasattr(reader, "STAGED_RANGE_WAIT_S")


def test_tensor_span_crossing_adjacent_extents_requires_one_covering_object(tmp_path, monkeypatch):
    from safetensors.torch import save
    import torch
    from prismaquant.residency_shard_reader import staged_shard_opener
    from prismaquant.staged_tier_policy import staged_tier_policy_test_context, TierPolicyRefused
    from staged_native_fixtures import fixture, bind
    raw = save({"weight": torch.arange(8, dtype=torch.float32)})
    base = 8 + int.from_bytes(raw[:8], "little")
    split = base + 16
    source, _data, binding, _contract = fixture(tmp_path, raw, ranges=[(0, split), (split, len(raw))])
    resolver = bind(monkeypatch, binding)
    try:
        assert resolver.staged_range_outcome(source, base, len(raw), declared_size=len(raw)) == (None, "undeclared")
        assert resolver.staged_range_outcome(source, base, split, declared_size=len(raw))[1] == "hit"
        assert resolver.staged_range_outcome(source, split, len(raw), declared_size=len(raw))[1] == "hit"
        def no_pool(*args, **kwargs):
            raise AssertionError("canonical source must not be opened")
        with staged_tier_policy_test_context("ssd"):
            with staged_shard_opener(source, no_pool)(str(source), framework="pt") as reader:
                with pytest.raises(TierPolicyRefused, match="readset-not-staged"):
                    reader.get_tensor("weight")
    finally:
        from prismaquant.residency_map import reset_residency_resolver_for_tests
        reset_residency_resolver_for_tests()


def test_host_local_launch_executes_the_verified_bytes(tmp_path, monkeypatch, launch_environment):
    helper = tmp_path / "fluxcache.py"
    raw = b"import argparse; print('reviewed helper')\n"
    helper.write_bytes(raw)
    monkeypatch.setattr(staged_launch, "CACHE_HELPER", str(helper))
    monkeypatch.setattr(staged_launch, "CACHE_HELPER_SHA256", staged_launch.bytes_sha256hex(raw))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    manifest = SimpleNamespace(lifetime={"mode": "host_local_cache", "root": str(tmp_path)}, members=lambda: [])
    monkeypatch.setattr(staged_launch, "validate_binding", lambda binding: manifest)
    publish = staged_launch.publish_staged_json
    def swap(*args):
        publish(*args)
        helper.write_text("raise SystemExit('swapped')")
    monkeypatch.setattr(staged_launch, "publish_staged_json", swap)
    class Executed(Exception):
        pass
    def execute(path, argv):
        assert argv[:3] == ["/usr/bin/python3", "-I", "-c"]
        assert argv[3].encode() == raw
        import subprocess
        (tmp_path / "argparse.py").write_text("raise RuntimeError('cwd import')\n")
        result = subprocess.run(argv, cwd=tmp_path, env={"PATH": "/usr/bin:/bin"},
                                capture_output=True, text=True, timeout=10, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "reviewed helper\n"
        raise Executed
    monkeypatch.setattr(staged_launch.os, "execv", execute)
    binding = {"path": "manifest", "sha256": "a" * 64, "read_contract_sha256": "b" * 64,
               "read_contract_path": "contract", "allowed_tiers": "ssd"}
    with pytest.raises(Executed):
        staged_launch.launch(binding, ["/bin/true"])


@pytest.mark.parametrize("argv", [
    ["env", "-i", "flux", "submit"],
    ["timeout", "60", "flux", "submit"],
    ["nohup", "flux", "submit"], ["nice", "-n", "5", "flux", "submit"],
    ["time", "flux", "submit"], ["stdbuf", "-oL", "flux", "submit"],
    ["setsid", "flux", "submit"], ["xargs", "flux", "submit"],
    ["wrapper", "/custom/flux_jobs.py"],
    ["sh", "-c", "if true; then flux submit; fi"],
    ["sh", "-c", "for x in a; do flux submit; done"],
    ["sh", "-c", "if false; then true; else flux submit; fi"],
    ["sh", "-c", "! flux submit"], ["sh", "-c", "{ flux submit; }"],
    ["sh", "-c", "echo `flux submit`"],
    ["python3", "-Ic", "import prismaquant.flux_jobs"],
    ["python3", "-Im", "prismaquant.flux_jobs"],
    ["python3", "-X", "dev", "-m", "prismaquant.flux_jobs"],
    ["python3", "-W", "error", "-m", "prismaquant.flux_jobs"],
    ["echo", "flux"], ["bash", "--rcfile", "flux", "-c", "true"],
])
def test_fail_closed_tripwire_keeps_original_and_cluster_forms(argv):
    assert ship_gates._submits_nested_job(argv)
