"""The fanout must execute the sealed source in its declared rootless Podman runtime."""
import importlib
import json
import re
import os
from pathlib import Path
import subprocess
import sys

import pytest

from prismaquant import tessera_campaign_rows as campaign_rows

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch

from container_admission import ADMISSION, CIDFILE, stub_launch, with_admission
from container_inspection_fixture import inspection


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
    row = campaign_rows.campaign_row(spec(), ["--units", "/mnt/shared/units.json"],
                        mem_gb=48, timeout_s=600)
    assert row["argv"][:3] == ["python3", "-m", "tools.tessera_campaign_container"]
    cut = row["argv"].index("--")
    assert row["argv"][cut + 1:] == [
        "python3", "-u", "-m", "prismaquant.tessera_campaign",
        "--units", "/mnt/shared/units.json"]
    assert json.loads(row["argv"][4])["container"] == spec()["container"]
    assert row["demand"] == {"gpu": 1, "cpu": 4, "mem_gb": 48}
    assert row["retry_safe"] is True


def test_container_uses_worker_snapshot_and_native_cpuset_without_shell():
    runner = importlib.import_module("tools.tessera_campaign_container")
    argv = runner.podman_command(spec(), ["python3", "a path;$(touch bad)"],
                                 cwd="/worker/snapshot", admission=ADMISSION, cidfile=CIDFILE,
                                 image_id="sha256:resolved")
    assert argv[:2] == [runner.PODMAN, "create"]
    assert argv[argv.index("--cidfile") + 1] == CIDFILE
    for flag in ("--pull=never", "--cgroups=disabled", "--network=none", "--env-host=false"):
        assert flag in argv
    assert argv[argv.index("--cpuset-cpus") + 1] == "21,22,23,30"
    assert ("--user=0:0" in argv) or ("--userns=keep-id" in argv and "--user=1000:1000" in argv)
    assert "type=bind,src=/worker/snapshot,dst=/workspace,readonly" in argv
    assert not any("/original/checkout" in arg for arg in argv)
    assert "PYTHONPATH=.:/producer/src" in argv
    assert "type=bind,src=/producer,dst=/producer,readonly" in argv
    assert argv[-3:] == ["sha256:resolved", "python3", "a path;$(touch bad)"]
    assert "docker" not in " ".join(argv) and "--gpus" not in argv and "--device" not in argv


def test_native_publication_identity_crosses_the_explicit_container_environment():
    runner = importlib.import_module("tools.tessera_campaign_container")
    environ = {"FLUX_JOB_ID": "123", "OUT": "/mnt/shared/jobs/owned", "API_TOKEN": "private"}
    argv = runner.podman_command(spec(), ["python3"], cwd="/worker/snapshot",
        admission=ADMISSION, cidfile=CIDFILE, image_id="sha256:resolved", environ=environ)
    forwarded = dict(argv[index + 1].split("=", 1) for index, value in enumerate(argv) if value == "--env")
    assert forwarded["FLUX_JOB_ID"] == "123"
    assert forwarded["OUT"] == environ["OUT"]
    assert "API_TOKEN" not in forwarded


def test_native_admission_refuses_before_namespace_directories_are_created(monkeypatch):
    runner = importlib.import_module("tools.tessera_campaign_container")
    touched = []
    monkeypatch.setattr(runner, "validate_namespace_launch", lambda *a, **kw: touched.append("namespace"))
    def refuse(_environ):
        raise RuntimeError("unadmitted")
    monkeypatch.setattr(runner, "native_admission", refuse)
    with pytest.raises(RuntimeError, match="unadmitted"):
        runner.main(["--spec", json.dumps(spec()), "--", "python3"])
    assert touched == []


def _argv_with_budget(budget, admission=ADMISSION):
    runner = importlib.import_module("tools.tessera_campaign_container")
    declared = spec()
    if budget is not _ABSENT:
        declared["box_memory_gb"] = budget
    return runner.podman_command(declared, ["python3"], cwd="/worker/snapshot",
                                 admission=admission, cidfile=CIDFILE, image_id="sha256:resolved")


_ABSENT = object()


def test_a_declared_budget_within_the_admission_is_recorded_not_kernel_enforced():
    """D1 (probe job fV3wqkMnFq): with ``--cgroups=disabled`` podman accepts ``--memory``
    but does not enforce it, so none is emitted and ``HostConfig.Memory`` is never evidence.
    The bound is the job unit's; the declared cap only has to fit the admitted memory."""
    argv = _argv_with_budget(10)
    assert "--memory" not in argv and "--memory-swap" not in argv


def test_a_declared_budget_above_the_admission_is_refused():
    with pytest.raises(RuntimeError, match="104 GiB exceeds the admitted 16 GiB"):
        _argv_with_budget(104)


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
    return runner.podman_command(declared, ["python3"], cwd="/worker/snapshot",
                                 admission=ADMISSION, cidfile=CIDFILE, image_id="sha256:resolved")


def test_the_container_cap_is_the_cpu_field_not_the_box_capacity():
    """``cpu_memory_gb`` is what the container may charge; ``box_memory_gb`` is capacity."""
    big = with_admission(memory_gib=64)
    runner = importlib.import_module("tools.tessera_campaign_container")
    declared = spec()
    declared["cpu_memory_gb"], declared["box_memory_gb"] = 34, 114
    runner.podman_command(declared, ["python3"], cwd="/worker/snapshot", admission=big,
                          cidfile=CIDFILE, image_id="sha256:resolved")
    with pytest.raises(RuntimeError, match="34 GiB exceeds the admitted 16 GiB"):
        _argv_with_fields(cpu_gb=34, box_gb=114)


def test_a_spec_sealed_before_the_cpu_field_keeps_its_old_cap():
    """Backwards: v3 and every earlier spec declared only box_memory_gb."""
    with pytest.raises(RuntimeError, match="104 GiB exceeds the admitted 16 GiB"):
        _argv_with_fields(cpu_gb=_ABSENT, box_gb=104)


def test_host_adapter_needs_no_prismaquant_dependencies_and_forwards_no_progress(tmp_path):
    """The host launcher must build its container argv before importing the pinned image.

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
admission = {json.dumps(ADMISSION)!r}
argv = runner.podman_command(json.loads(spec), ["python3"], cwd="/snapshot",
                            admission=json.loads(admission), cidfile={CIDFILE!r},
                            image_id="sha256:resolved", environ=environ)
assert not any("PRISMABUILD_ACTION_PROGRESS" in part for part in argv)
assert not hasattr(runner, "progress_environment")
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, "-S", "-c", script],
                               cwd=tmp_path, env=environment,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_host_launcher_validates_with_the_package_spec_contract():
    """One owner: the launcher runs the planners' contract, loaded by path."""
    from prismaquant import campaign_container_spec as contract

    runner = importlib.import_module("tools.tessera_campaign_container")
    assert (Path(runner.validate_container.__code__.co_filename).resolve()
            == Path(contract.__file__).resolve())
    for name in ("SAFE_PATH_ENV", "CHECKOUT_COMMIT_ENV",
                 "BOUNDED_CAPTURE_ENV", "GPU_RUNTIME_FLAGS", "DEFAULT_GPU_RUNTIME"):
        assert getattr(runner, name) == getattr(contract, name), name


@pytest.mark.parametrize("target", ["/", "/workspace", "/workspace/tools", "/mnt/../workspace"])
def test_spec_refuses_mounts_that_hide_the_sealed_source(tmp_path, target):
    data = spec()
    data["container"]["mounts"][0]["target"] = target
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="workspace|canonical"):
        campaign_rows.load_spec(path)


def test_host_rows_remain_direct():
    data = spec()
    del data["container"]
    assert campaign_rows.campaign_row(data, [], mem_gb=40, timeout_s=60)["argv"] == [
        "python3", "-u", "-m", "prismaquant.tessera_campaign"]


@pytest.mark.parametrize('container', [False, True])
def test_materialization_row_preserves_its_subcommand_and_resource_demand(container):
    data = spec()
    data['env']['MIMALLOC_PURGE_DELAY'] = '10'
    data['cpus'] = 2
    data['tags'] = ['gb10']
    if not container:
        del data['container']
    arguments = ['run', '--plan', '/mnt/shared/plan.json', '--group', '0',
                 '--anchor-batch-size', '2']
    row = campaign_rows.campaign_row(data, arguments, module='prismaquant.tessera_materialization',
                        mem_gb=48, timeout_s=600)
    command = ['python3', '-u', '-m', 'prismaquant.tessera_materialization', *arguments]
    if container:
        assert row['argv'][:4] == [
            'python3', '-m', 'tools.tessera_campaign_container', '--spec']
        assert json.loads(row['argv'][4]) == {
            'container': data['container'], 'env': data['env']}
        assert row['argv'][5:] == ['--', *command]
    else:
        assert row['argv'] == command
    assert row['demand'] == {'gpu': 1, 'cpu': 2, 'mem_gb': 48}
    assert row['env'] == data['env']
    assert row['tags'] == ['gb10']
    assert row['timeout_s'] == 600
    assert row['retry_safe'] is True


def test_actual_image_content_cannot_be_overridden_and_cpu_mode_requests_no_gpu():
    runner = importlib.import_module('tools.tessera_campaign_container')
    data = spec()
    argv = runner.podman_command(data, ['python3'], cwd='/snapshot', admission=ADMISSION, cidfile=CIDFILE,
        image_id='sha256:resolved', content_sha256='a'*64)
    assert '--gpus' not in argv
    assert 'PRISMAQUANT_CONTAINER_CONTENT_SHA256='+'a'*64 in argv
    data['env']['PRISMAQUANT_CONTAINER_CONTENT_SHA256'] = 'b'*64
    with pytest.raises(RuntimeError, match='inspected launcher'):
        runner.validate_container(data)


def test_spec_cannot_supply_the_checkout_commit():
    runner = _runner()
    data = spec()
    data['env'][runner.CHECKOUT_COMMIT_ENV] = 'a' * 40
    with pytest.raises(RuntimeError, match='inspected launcher'):
        runner.validate_container(data)


def _git_checkout(root):
    """A committed checkout with a tracked package file; returns its HEAD."""
    (root / 'prismaquant').mkdir(parents=True)
    (root / 'prismaquant' / 'x.py').write_text('X = 1\n')
    env = {**os.environ, 'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t',
           'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t', 'HOME': str(root)}
    for args in (['init', '-q'], ['add', '-A'], ['commit', '-q', '-m', 'seal']):
        subprocess.run(['git', '-C', str(root), *args], check=True, env=env, capture_output=True)
    return subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_checkout_commit_is_read_on_the_host_and_refuses_changed_package_bytes(tmp_path):
    """The container has no git; the launcher reads HEAD and performs the exactness check.

    ``PRISMAQUANT_IDENTITY_GIT_COMMIT`` also disables ``aura_provenance``'s
    checkpoint-owner ``git diff`` check inside the container, so a checkout
    whose tracked package bytes differ from HEAD must be refused here, before a commit the
    bytes do not match is stamped on a checkpoint.
    """
    runner = _runner()
    assert runner.checkout_commit(str(tmp_path)) is None
    root = tmp_path / 'checkout'
    head = _git_checkout(root)
    assert runner.checkout_commit(str(root)) == head
    argv = runner.podman_command(spec(), ['python3'], cwd=str(root), admission=ADMISSION, cidfile=CIDFILE,
                                 image_id='sha256:resolved', checkout_commit=head)
    assert f'{runner.CHECKOUT_COMMIT_ENV}={head}' in argv
    assert not any(item.startswith(runner.CHECKOUT_COMMIT_ENV) for item in
                   runner.podman_command(spec(), ['python3'], cwd=str(root), admission=ADMISSION, cidfile=CIDFILE,
                                         image_id='sha256:resolved'))
    # An untracked file is not part of a commit; the package digest covers it.
    (root / 'prismaquant' / 'untracked.py').write_text('U = 1\n')
    assert runner.checkout_commit(str(root)) == head
    (root / 'prismaquant' / 'x.py').write_text('X = 2\n')
    with pytest.raises(RuntimeError, match='differ from that commit'):
        runner.checkout_commit(str(root))
    (root / '.git' / 'HEAD').write_text('ref: refs/heads/nowhere\n')
    with pytest.raises(RuntimeError, match='cannot read|no commit'):
        runner.checkout_commit(str(root))


def test_main_hands_the_checkout_commit_to_the_container(monkeypatch, tmp_path, capsys):
    runner = _runner()
    launched = []
    monkeypatch.setattr(runner, 'inspect_or_load', lambda container: [inspection()])
    monkeypatch.setattr(runner, 'verify_pinned_import', lambda *args, **kwargs: {})
    stub_launch(monkeypatch, runner, launched)
    data = spec()
    data['container'].pop('archive', None)
    root = tmp_path / 'checkout'
    head = _git_checkout(root)
    monkeypatch.chdir(root)
    assert runner.main(['--spec', json.dumps(data), '--', 'python3', '-c', 'pass']) == 0
    assert f'{runner.CHECKOUT_COMMIT_ENV}={head}' in launched[0]
    assert json.loads(capsys.readouterr().out.splitlines()[0])['checkout_commit'] == head


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
        assert argv[:3] == [runner.PODMAN, 'image', 'inspect']
        return SimpleNamespace(returncode=1, stdout='', stderr='absent')
    monkeypatch.setattr(runner.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='archive bytes changed'):
        runner.inspect_or_load(data['container'])
    assert len(calls) == 1


def _runner():
    return importlib.import_module('tools.tessera_campaign_container')


def test_a_cpu_admission_withholds_the_device_whatever_the_spec_says():
    """Only the native admission decides. An empty or granting CUDA_VISIBLE_DEVICES is not consulted."""
    for visible in (None, '', '0'):
        data = spec()
        if visible is not None:
            data['env']['CUDA_VISIBLE_DEVICES'] = visible
        attached, reason = _runner().gpu_attachment(ADMISSION)
        assert attached is False and 'no GPU' in reason


def test_a_gpu_admission_is_refused_until_unit_4e():
    with pytest.raises(RuntimeError, match='held for unit 4E'):
        _runner().gpu_attachment(with_admission(gpus=1))


def test_a_malformed_admission_is_refused():
    runner = _runner()
    for bad in ({}, {**ADMISSION, 'schema': 'other'}, {**ADMISSION, 'affinity': []},
                {**ADMISSION, 'affinity': [3, 1]}, {k: v for k, v in ADMISSION.items() if k != 'gpus'}):
        with pytest.raises(RuntimeError, match='admission is missing or malformed'):
            runner.gpu_attachment(bad)


def test_main_withholds_the_device_and_launches_with_the_native_cpuset(monkeypatch, tmp_path):
    """The same property through the real entry point, on the argv it would run."""

    runner = _runner()
    launched = []
    monkeypatch.setattr(runner, 'inspect_or_load', lambda container: [inspection()])
    stub_launch(monkeypatch, runner, launched)
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
    assert runner.main(['--spec', json.dumps(data), '--', 'python3', '-c', 'pass']) == 0
    argv = launched[0]
    assert '--gpus' not in argv and '--device' not in argv
    assert argv[argv.index('--cpuset-cpus') + 1] == '21,22,23,30'


def test_no_launch_without_a_native_job(monkeypatch):
    runner = _runner()
    monkeypatch.delenv('FLUX_JOB_ID', raising=False)
    monkeypatch.delenv('OUT', raising=False)
    with pytest.raises(RuntimeError, match='admitted native job'):
        runner.native_admission({})


def test_pb_context_is_never_forwarded_and_a_spec_naming_it_refuses():
    runner = _runner()
    retired = ('PRISMABUILD_ACTION_KEY', 'PRISMABUILD_ACTION_NONCE', 'PRISMABUILD_ACTION_SCOPE',
               'PRISMABUILD_READER_HELPER_ROOT', 'PRISMABUILD_QUEUE_ROOT',
               'PRISMABUILD_ACTION_PROGRESS_PATH', 'PRISMABUILD_ACTION_PROGRESS_TOKEN',
               'PRISMABUILD_LOCAL_SCRATCH_IO', 'PRISMABUILD_LOCAL_SCRATCH_PAIRS')
    for name in retired:
        data = spec()
        data['env'][name] = 'x'
        with pytest.raises(RuntimeError, match='retired PB action/reader/spool context'):
            runner.validate_container(data)
    argv = runner.podman_command(spec(), ['python3'], cwd='/snapshot', admission=ADMISSION,
                                 cidfile=CIDFILE, image_id='sha256:x',
                                 environ={name: 'x' for name in retired})
    assert not any(name in part for part in argv for name in retired)
    assert not hasattr(runner, 'reader_context_environment')


def test_only_the_listed_ambient_names_are_read(tmp_path):
    runner = _runner()

    class Recording(dict):
        seen = None

        def get(self, key, default=None):
            self.seen.add(key)
            return super().get(key, default)

        def __contains__(self, key):
            self.seen.add(key)
            return super().__contains__(key)

        def __getitem__(self, key):
            self.seen.add(key)
            return super().__getitem__(key)

    environ = Recording({'SECRET_TOKEN': 'x', 'HOME': '/home/x', 'PATH': '/bin'})
    environ.seen = set()
    runner.podman_command(spec(), ['python3'], cwd='/snapshot', admission=ADMISSION,
                          cidfile=CIDFILE, image_id='sha256:x', environ=environ)
    assert environ.seen <= set(runner.AMBIENT_SETTINGS), sorted(environ.seen - set(runner.AMBIENT_SETTINGS))


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
    """A sealed spec that omits the names still gets a container that can start.

    ``bounded`` is the caller stating the row is bounded, which ``main`` derives
    from the row's own sealed environment -- the only marker the host-side
    adapter sees -- so this is the decision the launch path makes.
    """
    data = _bounded_spec()
    assert "MIMALLOC_PURGE_DELAY" not in data["env"]
    argv = _runner().podman_command(
        data, ["python3", "-c", "pass"], cwd=str(tmp_path), admission=with_admission(memory_gib=32), cidfile=CIDFILE,
        image_id="sha256:" + "a" * 64, environ={}, bounded=True)
    forwarded = [argv[index + 1] for index, value in enumerate(argv)
                 if value == "--env"]
    for name, expected in _runner().BOUNDED_CAPTURE_ENV.items():
        assert f"{name}={expected}" in forwarded, (name, forwarded)
    # The CPU cap itself (`--memory`/`--memory-swap` from the spec's own
    # ``cpu_memory_gb``) is the launcher change on RobTand/prismaquant#663, not
    # this one: this delivery supplies the environment the bounded pass refuses
    # without, and the two land together. Asserting the cap here would make this
    # test fail on a tree that has neither, which is what it did.


def test_only_a_bounded_row_is_handed_the_bounded_environment(tmp_path):
    """A legacy spec's own value survives into the docker argv, not just the row.

    ``data['env']`` is what dispatch seals, and
    ``test_legacy_manifest_keeps_explicit_allocator_environment`` pins that a
    legacy row keeps the ``MIMALLOC_PURGE_DELAY=10`` its class declares. The
    launcher then merged ``BOUNDED_CAPTURE_ENV`` into the forwarded environment
    of every row, so the container actually started with ``0`` and the sealed
    declaration never reached the process that reads it. The defaults are the
    bounded path's, so they are merged only for a bounded row -- and a bounded
    spec that contradicts them is refused on that same path rather than
    overridden.
    """
    runner = _runner()
    legacy = spec()
    legacy["env"] = {**legacy["env"], "MIMALLOC_PURGE_DELAY": "10"}
    argv = runner.podman_command(
        legacy, ["python3", "-c", "pass"], cwd=str(tmp_path), admission=ADMISSION, cidfile=CIDFILE,
        image_id="sha256:" + "a" * 64, environ={})
    forwarded = [argv[index + 1] for index, value in enumerate(argv)
                 if value == "--env"]
    assert "MIMALLOC_PURGE_DELAY=10" in forwarded, forwarded
    assert "MIMALLOC_PURGE_DELAY=0" not in forwarded, forwarded
    assert not [entry for entry in forwarded
                if entry.startswith("PRISMAQUANT_RELEASE_SOURCE_PAGES=")]

    with pytest.raises(RuntimeError,
                       match="contradicts the bounded capture contract"):
        runner.podman_command(
            legacy, ["python3", "-c", "pass"], cwd=str(tmp_path), admission=ADMISSION, cidfile=CIDFILE,
            image_id="sha256:" + "a" * 64, environ={},
            bounded=True)


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


# -- the immutable staged manifest crosses the container boundary (unit 4d) ---------------------


def _binding(tmp_path):
    from staged_native_fixtures import fixture
    _source, data, binding, _contract = fixture(tmp_path)
    return data, binding


def _environ(binding):
    return {"PRISMAQUANT_STAGED_MANIFEST": binding["path"],
            "PRISMAQUANT_STAGED_MANIFEST_SHA256": binding["sha256"],
            "PRISMAQUANT_READ_CONTRACT_SHA256": binding["read_contract_sha256"],
            "PRISMAQUANT_READ_CONTRACT_PATH": binding["read_contract_path"],
            "PRISMAQUANT_STAGED_ALLOWED_TIERS": "ssd"}


def _staged_spec(mounts):
    declared = spec()
    declared["container"]["mounts"] = mounts
    return declared


def _argv(tmp_path, environ, declared=None):
    runner = importlib.import_module("tools.tessera_campaign_container")
    return runner.podman_command(declared or _staged_spec([]), ["python3"],
                                 cwd="/snapshot", admission=ADMISSION, cidfile=CIDFILE,
                                 image_id="sha256:resolved", environ=environ)


def _mounts(argv):
    return [argv[i + 1] for i, flag in enumerate(argv) if flag == "--mount"]


def _files(binding):
    return [{"source": binding["path"], "target": binding["path"], "readonly": True},
            {"source": binding["read_contract_path"], "target": binding["read_contract_path"], "readonly": True}]


def test_the_manifest_crosses_with_its_phase_namespace_bound_readonly(tmp_path):
    data, binding = _binding(tmp_path)
    argv = _argv(tmp_path, _environ(binding), _staged_spec(_files(binding)))
    namespace = data["lifetime"]["namespace"]
    assert f"type=bind,src={namespace},dst={namespace},readonly" in _mounts(argv)
    forwarded = dict(argv[i + 1].split("=", 1) for i, flag in enumerate(argv) if flag == "--env")
    for name, value in _environ(binding).items():
        assert forwarded[name] == value


def test_manifest_and_contract_paths_are_translated_through_declared_mounts(tmp_path):
    data, binding = _binding(tmp_path)
    declared = _staged_spec([{"source": binding["path"], "target": "/staged/manifest.json", "readonly": True},
                             {"source": binding["read_contract_path"], "target": "/staged/contract.json",
                              "readonly": True}])
    argv = _argv(tmp_path, _environ(binding), declared)
    forwarded = dict(argv[i + 1].split("=", 1) for i, flag in enumerate(argv) if flag == "--env")
    assert forwarded["PRISMAQUANT_STAGED_MANIFEST"] == "/staged/manifest.json"
    assert forwarded["PRISMAQUANT_READ_CONTRACT_PATH"] == "/staged/contract.json"
    assert forwarded["PRISMAQUANT_STAGED_MANIFEST_SHA256"] == binding["sha256"]


def test_without_a_manifest_the_argv_is_byte_identical(tmp_path):
    assert _argv(tmp_path, {}) == _argv(tmp_path, None)
    assert not any("STAGED" in flag or "READ_CONTRACT" in flag for flag in _argv(tmp_path, {}))


def test_a_manifest_outside_every_declared_mount_is_refused(tmp_path):
    _data, binding = _binding(tmp_path)
    declared = _staged_spec([{"source": "/elsewhere", "target": "/elsewhere", "readonly": True}])
    with pytest.raises(RuntimeError, match="not inside any declared container mount"):
        _argv(tmp_path, _environ(binding), declared)


def test_a_partial_binding_is_refused_not_guessed(tmp_path):
    _data, binding = _binding(tmp_path)
    environ = _environ(binding)
    del environ["PRISMAQUANT_READ_CONTRACT_PATH"]
    with pytest.raises(RuntimeError, match="partial staged-input binding"):
        _argv(tmp_path, environ, _staged_spec(_files(binding)))


def test_a_changed_manifest_or_contract_digest_is_refused(tmp_path):
    from prismaquant.staged_manifest import ManifestRefused
    _data, binding = _binding(tmp_path)
    environ = {**_environ(binding), "PRISMAQUANT_STAGED_MANIFEST_SHA256": "0" * 64}
    with pytest.raises(ManifestRefused):
        _argv(tmp_path, environ, _staged_spec(_files(binding)))
    environ = {**_environ(binding), "PRISMAQUANT_READ_CONTRACT_SHA256": "0" * 64}
    with pytest.raises(RuntimeError, match="read contract digest differs"):
        _argv(tmp_path, environ, _staged_spec(_files(binding)))


def test_a_host_local_manifest_cannot_cross_the_container_boundary(tmp_path):
    from prismaquant.staged_manifest import publish_staged_json
    data, binding = _binding(tmp_path)
    local = {**data, "lifetime": {"mode": "host_local_cache", "root": str(tmp_path / "cache"),
                                  "owner_host": "dl380g10"}}
    path = tmp_path / "host-local.json"
    digest = publish_staged_json(path, local)
    environ = {**_environ(binding), "PRISMAQUANT_STAGED_MANIFEST": str(path),
               "PRISMAQUANT_STAGED_MANIFEST_SHA256": digest}
    declared = _staged_spec([{"source": str(path), "target": str(path), "readonly": True},
                             {"source": binding["read_contract_path"],
                              "target": binding["read_contract_path"], "readonly": True}])
    with pytest.raises(RuntimeError, match="cannot cross the container boundary.*descriptor transfer"):
        _argv(tmp_path, environ, declared)


def test_a_writable_mount_over_the_phase_namespace_is_refused(tmp_path):
    _data, binding = _binding(tmp_path)
    declared = _staged_spec([{"source": str(tmp_path), "target": str(tmp_path)}])
    with pytest.raises(RuntimeError, match="readonly identity mount"):
        _argv(tmp_path, _environ(binding), declared)


def test_a_readonly_mount_already_covering_the_namespace_is_not_bound_twice(tmp_path):
    _data, binding = _binding(tmp_path)
    declared = _staged_spec([{"source": str(tmp_path), "target": str(tmp_path), "readonly": True}])
    mounts = _mounts(_argv(tmp_path, _environ(binding), declared))
    assert sum(1 for m in mounts if f"dst={tmp_path}" in m) == 1
    assert sum(1 for m in mounts if f"dst={tmp_path}/phase" in m) == 0


def test_the_launcher_owned_names_are_the_staged_launch_names_and_a_spec_cannot_set_them():
    from prismaquant import staged_manifest
    runner = importlib.import_module("tools.tessera_campaign_container")
    assert runner.STAGED_ENV == (staged_manifest.PATH_ENV, staged_manifest.SHA_ENV, staged_manifest.READ_ENV,
                                 "PRISMAQUANT_READ_CONTRACT_PATH", "PRISMAQUANT_STAGED_ALLOWED_TIERS")
    for name in runner.STAGED_ENV:
        data = spec()
        data["env"][name] = "x"
        with pytest.raises(RuntimeError, match="supplied by the launcher"):
            runner.validate_container(data)


@pytest.mark.parametrize("mode", ["good", "missing", "readonly", "remapped", "nested", "conflict"])
def test_cotangent_scratch_preserves_sealed_local_path_identity(mode):
    runner = importlib.import_module("tools.tessera_campaign_container")
    declared = spec()
    root = "/home/rob/pb-scratch/glm-stageb"
    env = {runner.COTANGENT_SCRATCH_ENV[0]: root,
           runner.COTANGENT_SCRATCH_ENV[1]: str(36 << 30)}
    if mode != "missing":
        declared["container"]["mounts"].append({
            "source": root if mode != "remapped" else "/different/host-path",
            "target": root, "readonly": mode == "readonly"})
    if mode == "nested":
        declared["container"]["mounts"].append({
            "source": "/other", "target": root + "/payload"})
    if mode == "conflict":
        declared["env"][runner.COTANGENT_SCRATCH_ENV[0]] = "/different"
    if mode == "good":
        argv = runner.podman_command(declared, ["python3"], cwd="/worker/snapshot",
            admission=ADMISSION, cidfile=CIDFILE, image_id="sha256:resolved", environ=env)
        assert f"{runner.COTANGENT_SCRATCH_ENV[0]}={root}" in argv
        assert f"{runner.COTANGENT_SCRATCH_ENV[1]}={36 << 30}" in argv
    else:
        with pytest.raises(RuntimeError, match="cotangent scratch"):
            runner.podman_command(declared, ["python3"], cwd="/worker/snapshot",
                admission=ADMISSION, cidfile=CIDFILE, image_id="sha256:resolved", environ=env)


@pytest.mark.parametrize("name", ["PRISMABUILD_PRODUCED_SPOOL_ROOT", "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES",
                                  "PRISMABUILD_PRODUCED_SPOOL_PACED_EXPORT", "PRISMABUILD_PRODUCED_SPOOL_HOST_WINDOW"])
def test_a_spec_naming_a_retired_spool_setting_refuses(name):
    """The spool bound is the launch-bound declaration's; the retired names are refused, never forwarded."""
    runner = importlib.import_module("tools.tessera_campaign_container")
    declared = spec()
    declared["env"][name] = "x"
    with pytest.raises(RuntimeError, match="retired produced-output service context"):
        runner.validate_container(declared)
    assert not hasattr(runner, "produced_spool_environment")
    assert not hasattr(runner, "PRODUCED_SPOOL_ENV")
