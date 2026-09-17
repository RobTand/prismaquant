"""The fanout must execute the sealed source in its declared Docker runtime."""
import importlib
import json
import os
from pathlib import Path
import subprocess
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


def _argv_with_budget(budget):
    runner = importlib.import_module("tools.tessera_campaign_container")
    declared = spec()
    if budget is not _ABSENT:
        declared["box_memory_gb"] = budget
    return runner.docker_command(declared, ["python3"], cwd="/worker/snapshot",
                                 uid=1000, gid=1000, image_id="sha256:resolved")


_ABSENT = object()


def test_a_declared_box_budget_becomes_the_cgroup_cap_the_guard_refuses_from():
    """The bound has to be enforced by the kernel, not asserted in a plan.

    ``CaptureMemoryGuard`` holds its margin back from a cgroup cap and refuses
    without one; on GB10 the device and the host share one physical pool, so
    this single number is the aggregate bound rather than 92 GiB plus a
    resident set. The value is the spec's own ``box_memory_gb`` -- the same one
    the demand derivation already refuses to exceed.
    """
    argv = _argv_with_budget(104)
    assert argv[argv.index("--memory") + 1] == "104g"
    # no swap headroom: the row fails rather than spilling past its budget
    assert argv[argv.index("--memory-swap") + 1] == "104g"


def test_a_spec_declaring_no_budget_keeps_the_previous_invocation():
    """Every row that ran before this existed declared none."""
    argv = _argv_with_budget(_ABSENT)
    assert "--memory" not in argv and "--memory-swap" not in argv


@pytest.mark.parametrize("budget", [0, -4, True, "104", float("nan")])
def test_an_unusable_memory_budget_is_refused_not_coerced(budget):
    """A cap that cannot be a cap is a fail-closed error, never a default."""
    with pytest.raises(RuntimeError, match="must be a positive number of GiB"):
        _argv_with_budget(budget)


def _argv_with_fields(cpu_gb=_ABSENT, box_gb=_ABSENT):
    runner = importlib.import_module("tools.tessera_campaign_container")
    declared = spec()
    if cpu_gb is not _ABSENT:
        declared["cpu_memory_gb"] = cpu_gb
    if box_gb is not _ABSENT:
        declared["box_memory_gb"] = box_gb
    return runner.docker_command(declared, ["python3"], cwd="/worker/snapshot",
                                 uid=1000, gid=1000, image_id="sha256:resolved")


def test_the_container_cap_is_the_cpu_field_not_the_box_capacity():
    """Two fields, two meanings, and conflating them breaks one of them.

    ``box_memory_gb`` is the box's total unified capacity, which the demand
    derivation refuses to derive a row above -- for A2 that is the combined
    physical demand a PrismaBuild row reserves. ``cpu_memory_gb`` is what the
    container's cgroup may charge. Reading the cap out of the capacity field
    gives a 114 GiB container cap for a row whose CPU side must fit 34 GiB.
    """
    argv = _argv_with_fields(cpu_gb=34, box_gb=114)
    assert argv[argv.index("--memory") + 1] == "34g"
    assert argv[argv.index("--memory-swap") + 1] == "34g"


def test_a_spec_sealed_before_the_cpu_field_keeps_its_old_cap():
    """Backwards: v3 and every earlier spec declared only box_memory_gb."""
    argv = _argv_with_fields(cpu_gb=_ABSENT, box_gb=104)
    assert argv[argv.index("--memory") + 1] == "104g"


def test_host_adapter_needs_no_prismaquant_dependencies_and_forwards_progress(tmp_path):
    """The host launcher must reach Docker before importing the pinned image.

    Sparklina's minimal action interpreter deliberately has no
    ``compressed_tensors``.  Before #601 importing this module initialized
    ``prismaquant`` first and failed there, even though that dependency belongs
    to the qualified container.  ``-S`` also makes this a minimal interpreter
    check rather than accidentally satisfying the dependency from site-packages.
    """

    root = Path(__file__).resolve().parents[1]
    script = f"""
import json
import os
import sys
sys.path.insert(0, {str(root)!r})
from tools import tessera_campaign_container as runner
spec = {json.dumps(spec())!r}
environ = {{
    "PRISMABUILD_ACTION_PROGRESS_PATH": "/mnt/shared/queue/k.progress",
    "PRISMABUILD_ACTION_PROGRESS_TOKEN": "minted-token",
}}
argv = runner.docker_command(json.loads(spec), ["python3"], cwd="/snapshot",
                            uid=1, gid=1, image_id="sha256:resolved",
                            environ=environ)
assert "PRISMABUILD_ACTION_PROGRESS_PATH=/mnt/shared/queue/k.progress" in argv
assert "PRISMABUILD_ACTION_PROGRESS_TOKEN=minted-token" in argv
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, "-S", "-c", script],
                               cwd=tmp_path, env=environment,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_host_progress_names_match_the_container_reporter():
    """Forwarded names and the in-container writer remain one PB contract."""

    from runpy import run_path

    runner = importlib.import_module("tools.tessera_campaign_container")
    # Read the stdlib-only reporter by file path: importing its package here
    # would hide precisely the minimal-host import boundary this test guards.
    reporter = run_path(str(Path(__file__).resolve().parents[1] /
                            'prismaquant' / 'prismabuild_progress.py'))
    assert runner.PATH_ENV == reporter['PATH_ENV']
    assert runner.TOKEN_ENV == reporter['TOKEN_ENV']


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


def _runner():
    return importlib.import_module('tools.tessera_campaign_container')


def test_a_row_that_reserved_no_gpu_does_not_get_the_device():
    """The defect in one assertion (#430).

    ``pbrun`` empties ``CUDA_VISIBLE_DEVICES`` in the action's environment
    exactly when it granted no GPU slots.  Attaching the device anyway spends
    no PrismaBuild token for it, so its admission arithmetic can seat a
    GPU-reserving row beside this one, and a power reading taken next door has
    an owner it cannot see.
    """

    attached, reason = _runner().gpu_attachment(
        spec(), cpu_only=False, environ={'CUDA_VISIBLE_DEVICES': ''})
    assert attached is False, (
        'the container took the whole GPU for a row that reserved none: '
        'PrismaBuild granted no slots and emptied CUDA_VISIBLE_DEVICES, and '
        'nothing but a remembered --cpu-only withheld --gpus all (#430)')
    assert 'no visible device' in reason


def test_a_spec_that_hides_the_device_from_its_payload_does_not_attach_it():
    """Masking CUDA is not the same as not having the device, and the spec
    saying its payload sees none is a declaration this must honour."""

    data = spec()
    data['env']['CUDA_VISIBLE_DEVICES'] = ''
    attached, reason = _runner().gpu_attachment(data, cpu_only=False, environ={})
    assert attached is False
    assert reason.startswith('container spec env')


def test_a_granted_row_still_gets_the_device():
    """The control.  Without it the assertions above would pass on a function
    that never attaches anything, which would break every campaign row."""

    attached, _ = _runner().gpu_attachment(
        spec(), cpu_only=False, environ={'CUDA_VISIBLE_DEVICES': '0'})
    assert attached is True


def test_an_interactive_run_outside_pbrun_keeps_its_behaviour():
    """Unset is not a declaration: there is no grant to read, so the flag is
    still the way to say no and the default is unchanged."""

    runner = _runner()
    assert runner.gpu_attachment(spec(), cpu_only=False, environ={})[0] is True
    assert runner.gpu_attachment(spec(), cpu_only=True, environ={})[0] is False
    assert runner.gpu_attachment(
        spec(), cpu_only=True, environ={'CUDA_VISIBLE_DEVICES': '0'})[0] is False


def test_main_withholds_the_device_from_a_row_that_reserved_none(monkeypatch, tmp_path):
    """The same property through the real entry point, on the argv it execs.

    The unit assertions above call the decision directly; this one asserts on
    what Docker is actually handed, so a decision that is right in isolation
    and unwired in ``main`` still fails.
    """

    runner = _runner()
    launched = {}
    monkeypatch.setattr(runner, 'inspect_or_load', lambda container: [
        {'Id': 'sha256:' + 'c' * 64, 'RepoDigests': [], 'RootFS': {'Layers': []}}])
    monkeypatch.setattr(runner, 'image_content_sha256', lambda inspected: 'd' * 64)
    monkeypatch.setattr(runner.os, 'execvp', lambda file, argv: launched.setdefault('argv', argv))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    data = spec()
    data['container'].pop('archive', None)
    # main() also refuses a launch that would import PrismaQuant from the
    # sealed checkout rather than the pinned mount (#519), so this row names a
    # pinned tree it can actually reach.
    pinned, checkout = tmp_path / 'pinned', tmp_path / 'checkout'
    for root in (pinned, checkout):
        (root / 'prismaquant').mkdir(parents=True)
        (root / 'prismaquant' / '__init__.py').write_text(f'TREE = {root.name!r}\n')
    data['container']['mounts'] = [
        {'source': str(pinned), 'target': str(pinned), 'readonly': True}]
    data['env']['PYTHONPATH'] = str(pinned)
    monkeypatch.chdir(checkout)
    runner.main(['--spec', json.dumps(data), '--', 'python3', '-c', 'pass'])
    assert '--gpus' not in launched['argv'], (
        'the exec line still maps the whole GPU into a container whose row '
        'reserved none (#430): ' + ' '.join(launched['argv'][:8]))


# --- the bounded capture environment the pass requires before it touches CUDA

def _bounded_spec(**env_overrides):
    data = spec()
    data["container"]["image"] = "qualified:fixed"
    data["cpu_memory_gb"] = 21
    data["env"] = {**data["env"], **env_overrides}
    return data


def test_the_two_bounded_capture_constants_cannot_drift():
    """The launcher's copy is compared against the pass's, so it cannot go stale.

    ``require_bounded_capture_environment`` is called at the joint pass's first
    bounded step -- minutes into loading the model on this campaign -- so a
    launcher that supplied a stale value would kill a run that had already paid
    for its loader. This is the comparison that makes the host-side copy safe.
    """
    from prismaquant import autoscale

    assert _runner().BOUNDED_CAPTURE_ENV == autoscale.BOUNDED_CAPTURE_ENV


def test_the_launcher_supplies_the_bounded_capture_environment(tmp_path):
    """A sealed spec that omits the names still gets a container that can start."""
    data = _bounded_spec()
    assert "MIMALLOC_PURGE_DELAY" not in data["env"]
    argv = _runner().docker_command(
        data, ["python3", "-c", "pass"], cwd=str(tmp_path), uid=1, gid=1,
        image_id="sha256:" + "a" * 64, with_gpu=False, environ={})
    forwarded = [argv[index + 1] for index, value in enumerate(argv)
                 if value == "--env"]
    for name, expected in _runner().BOUNDED_CAPTURE_ENV.items():
        assert f"{name}={expected}" in forwarded, (name, forwarded)
    # The CPU cap itself (`--memory`/`--memory-swap` from the spec's own
    # ``cpu_memory_gb``) is the launcher change on RobTand/prismaquant#663, not
    # this one: this delivery supplies the environment the bounded pass refuses
    # without, and the two land together. Asserting the cap here would make this
    # test fail on a tree that has neither, which is what it did.


def test_a_spec_that_contradicts_the_bounded_capture_contract_refuses():
    """A bounded row's spec may omit it; it may not weaken it.

    The contract binds the BOUNDED path, which is why every call here says so:
    a legacy row that declares its own purge delay is not a bounded capture row
    and keeps the environment it was sealed with.
    """
    data = _bounded_spec(MIMALLOC_PURGE_DELAY="60000")
    with pytest.raises(RuntimeError) as refused:
        _runner().validate_container(data, bounded=True)
    assert "contradicts the bounded capture contract" in str(refused.value)

    ok = _bounded_spec(PRISMAQUANT_RELEASE_SOURCE_PAGES="1",
                       MIMALLOC_PURGE_DELAY="0")
    _runner().validate_container(ok, bounded=True)
    # The same spec on the legacy path is untouched: the rule is the bounded
    # capture path's, and applying it everywhere was refusing unrelated work.
    _runner().validate_container(data)
