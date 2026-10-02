"""Metadata/admission composition; no Docker, GPU, Netdata or native profiler."""
import copy
import json
import subprocess
from pathlib import Path

import pytest

from tests.test_tessera_campaign_namespace_1986 import arguments, replace_fixture_request
from tools import dispatch_tessera_campaign as dispatch
from tools import pq_admitted_profile as profiler
from tools import tessera_campaign_namespace as namespace
from tools import tessera_campaign_container as adapter


@pytest.fixture(autouse=True)
def owned_local_policy(tmp_path, monkeypatch):
    # Isolate metadata fixtures, preserving the same deterministic root rule.
    monkeypatch.setattr(profiler, "PROFILE_LOCAL_ROOT", tmp_path.parent / (tmp_path.name + "-host-local"))


def profiled_arguments(tmp_path):
    kwargs = arguments(tmp_path)
    row = kwargs["requests"][0]
    spec = json.loads(row["argv"][4])
    spec["namespace_profile"] = {
        "schema": "prismaquant.tessera_namespace_profile.v1",
        "profiler": {"path": "/readonly/py-spy", "sha256": "3" * 64},
        "observations": "/old/profile",
        "profile_local": str(profiler.PROFILE_LOCAL_ROOT / "unbound"),
        "row_s": 900,
    }
    local = str(profiler.PROFILE_LOCAL_ROOT)
    spec["container"]["mounts"].append({"source": local, "target": local, "readonly": False})
    row["argv"][4] = json.dumps(spec)
    row["argv"] = profiler.profiled_row_command(row["argv"],
        destination="/old/profile/child-profile.speedscope", profiler_executable="/readonly/py-spy")
    replace_fixture_request(kwargs, row)
    return kwargs


def test_composed_request_is_prepared_before_binding(tmp_path):
    kwargs = profiled_arguments(tmp_path)
    before = copy.deepcopy(kwargs["requests"])
    rows = dispatch.prepare_namespace_requests(**kwargs)
    assert rows == dispatch.prepare_namespace_requests(**kwargs)
    assert kwargs["requests"] == before
    assert "tools.pq_profile_child" in rows[0]["argv"]
    namespace.validate_namespace_request(rows[0])


def test_post_binding_wrapper_still_refuses_actual_request_change(tmp_path):
    row = dispatch.prepare_namespace_requests(**arguments(tmp_path))[0]
    row["argv"] = profiler.profiled_row_command(row["argv"],
        destination=str(tmp_path / "foreign-profile"), profiler_executable="/readonly/py-spy")
    with pytest.raises(RuntimeError):
        namespace.validate_namespace_request(row)


def profile(row):
    return json.loads(row["argv"][4])["namespace_profile"]


def publish_profile_row(tmp_path):
    row = dispatch.prepare_namespace_requests(**profiled_arguments(tmp_path))[0]
    dispatch.publish_namespace_requests([row])
    return row


def launch_args(row, tmp_path):
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps(row))
    record = profile(row)
    return ["--launch", str(launch), "--observations", record["observations"],
            "--profile-local", record["profile_local"],
            "--profiler-executable", record["profiler"]["path"], "--row-s", str(record["row_s"])]


def test_public_composer_preserves_payload_and_input_row(tmp_path):
    kwargs = arguments(tmp_path)
    original = copy.deepcopy(kwargs["requests"][0])
    row = namespace.prepare_namespace_profile_request(original,
        profiler_reference={"path": "/readonly/py-spy", "sha256": "3" * 64},
        observations="/old/profile", profile_local=str(profiler.PROFILE_LOCAL_ROOT / "unbound"), row_s=900)
    assert original == kwargs["requests"][0]
    payload = profiler.profiled_workload_parts(row["argv"],
        destination="/old/profile/child-profile.speedscope", profiler_executable="/readonly/py-spy")
    assert row["argv"][payload:] == original["argv"][6:]
    assert profile(row)["profiler"]["sha256"] == "3" * 64
    with pytest.raises(RuntimeError):
        namespace.prepare_namespace_profile_request(row,
            profiler_reference=profile(row)["profiler"], observations="/other", profile_local="/other", row_s=900)


def test_owned_outputs_and_key_placeholder_are_bound_before_publication(tmp_path):
    row = publish_profile_row(tmp_path)
    binding = namespace.require_namespace_publication(row)
    record = profile(row)
    key = binding["request_key"]
    assert record["observations"] == str(tmp_path / "new" / key / "profile" / "observations")
    assert record["profile_local"] == str(profiler.PROFILE_LOCAL_ROOT / key)
    normalized = profile(binding["normalized_request"])
    assert normalized["observations"] == "{namespace-row}/profile/observations"
    assert normalized["profile_local"] == str(profiler.PROFILE_LOCAL_ROOT / "{namespace-row}")
    assert key not in normalized["profile_local"]
    destinations = namespace.namespace_destinations(binding["request"], binding)
    assert (Path(record["observations"]), True) in destinations
    assert (Path(record["profile_local"]), True) in destinations
    assert "tools.pq_profile_child" in binding["request"]["argv"]


def real_source_row(tmp_path):
    checkout = tmp_path / "source"
    (checkout / "prismaquant").mkdir(parents=True)
    (checkout / "prismaquant" / "__init__.py").write_text("# real Git admission fixture\n")
    for command in (["git", "init", "--quiet", str(checkout)],
                    ["git", "-C", str(checkout), "add", "prismaquant"],
                    ["git", "-C", str(checkout), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "commit", "--quiet", "-m", "source"]):
        subprocess.run(command, check=True, capture_output=True, timeout=30)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    kwargs = profiled_arguments(tmp_path)
    kwargs.update(checkout=str(checkout), executed_commit=commit)
    row = kwargs["requests"][0]
    row["cwd"] = str(checkout)
    replace_fixture_request(kwargs, row)
    row = dispatch.prepare_namespace_requests(**kwargs)[0]
    dispatch.publish_namespace_requests([row])
    return checkout, commit, row


def test_real_git_composed_admission_reaches_docker_boundary_only_after_checks(tmp_path, monkeypatch):
    checkout, commit, row = real_source_row(tmp_path)
    spec = json.loads(row["argv"][4])
    assert adapter.validate_namespace_launch(spec, row["argv"][6:], cwd=str(checkout), environ=row["env"]) == commit
    monkeypatch.chdir(checkout)
    for key, value in row["env"].items():
        monkeypatch.setenv(key, value)
    class DockerBoundary(Exception):
        pass
    reached = []
    def inspect(container):
        reached.append(container)
        raise DockerBoundary()
    monkeypatch.setattr(adapter, "inspect_or_load", inspect)
    with pytest.raises(DockerBoundary):
        adapter.main(["--spec", json.dumps(spec), "--", *row["argv"][6:]])
    assert reached == [spec["container"]]
    payload = profiler.profiled_workload_parts(row["argv"],
        destination=str(Path(profile(row)["observations"]) / "child-profile.speedscope"),
        profiler_executable=profile(row)["profiler"]["path"])
    assert row["argv"][payload:payload + 3] == ["python", "-m", "prismaquant.tessera_campaign"]


@pytest.mark.parametrize("change", ["schema", "extra", "row_s", "binary", "digest", "wrapper", "flag", "nested", "child_out", "payload"])
def test_unsupported_initial_instrumentation_refuses_before_publication(tmp_path, change):
    kwargs = profiled_arguments(tmp_path)
    row = kwargs["requests"][0]
    spec = json.loads(row["argv"][4])
    if change == "schema": spec["namespace_profile"]["schema"] += ".unknown"
    elif change == "extra": spec["namespace_profile"]["extra"] = True
    elif change == "row_s": spec["namespace_profile"]["row_s"] = True
    elif change == "binary": spec["namespace_profile"]["profiler"]["path"] = "py-spy"
    elif change == "digest": spec["namespace_profile"]["profiler"]["sha256"] = "unqualified"
    elif change == "wrapper": row["argv"][9] = "tools.other_profiler"
    elif change == "flag": row["argv"][10] = "--unknown"
    elif change == "nested": row["argv"] = profiler.profiled_row_command(row["argv"], destination="/old/profile/child-profile.speedscope", profiler_executable="/readonly/py-spy")
    elif change == "child_out": row["argv"][13] = "/foreign/profile"
    elif change == "payload": row["argv"][17] = "prismaquant.other_campaign"
    row["argv"][4] = json.dumps(spec)
    replace_fixture_request(kwargs, row)
    with pytest.raises(RuntimeError):
        dispatch.prepare_namespace_requests(**kwargs)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("change", ["argv", "metadata", "missing", "binary"])
def test_published_instrumentation_tamper_refuses_before_docker(tmp_path, monkeypatch, change):
    checkout, _, row = real_source_row(tmp_path)
    spec = json.loads(row["argv"][4])
    if change == "argv": row["argv"][13] += ".other"
    elif change == "metadata": spec["namespace_profile"]["row_s"] += 1
    elif change == "missing": del spec["namespace_profile"]
    elif change == "binary": spec["namespace_profile"]["profiler"]["path"] += ".other"
    row["argv"][4] = json.dumps(spec)
    monkeypatch.chdir(checkout)
    for key, value in row["env"].items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(adapter, "inspect_or_load", lambda *args: pytest.fail("Docker reached"))
    with pytest.raises(RuntimeError):
        adapter.main(["--spec", json.dumps(spec), "--", *row["argv"][6:]])


@pytest.mark.parametrize("mount", ["missing", "readonly", "remapped"])
def test_host_local_metadata_needs_declared_writable_identity_coverage(tmp_path, mount):
    kwargs = profiled_arguments(tmp_path)
    row = kwargs["requests"][0]
    spec = json.loads(row["argv"][4])
    if mount == "missing": spec["container"]["mounts"].pop()
    elif mount == "readonly": spec["container"]["mounts"][-1]["readonly"] = True
    else: spec["container"]["mounts"][-1]["source"] += "-remapped"
    row["argv"][4] = json.dumps(spec)
    replace_fixture_request(kwargs, row)
    with pytest.raises(RuntimeError, match="writable identity"):
        dispatch.prepare_namespace_requests(**kwargs)


@pytest.mark.parametrize("destination", ["observations", "profile_local"])
def test_owned_profile_destination_symlink_refuses(tmp_path, destination):
    row = publish_profile_row(tmp_path)
    path = Path(profile(row)[destination])
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        namespace.validate_namespace_request(row)
    assert list(foreign.iterdir()) == []


def test_declared_profiler_cannot_overlap_owned_writable_space(tmp_path):
    kwargs = profiled_arguments(tmp_path)
    row = kwargs["requests"][0]
    spec = json.loads(row["argv"][4])
    spec["namespace_profile"]["profiler"]["path"] = str(tmp_path / "new" / "binary")
    row["argv"][11] = spec["namespace_profile"]["profiler"]["path"]
    row["argv"][4] = json.dumps(spec)
    replace_fixture_request(kwargs, row)
    with pytest.raises(RuntimeError, match="input.*overlap"):
        dispatch.prepare_namespace_requests(**kwargs)


def test_host_runs_sealed_wrapper_and_observes_campaign_output(tmp_path, monkeypatch):
    row = publish_profile_row(tmp_path)
    calls = []
    def observed(command, observer, observations, **kwargs):
        calls.append((command, observer, observations, kwargs))
        return 0
    monkeypatch.setattr(profiler, "run_observed", observed)
    assert profiler.main(launch_args(row, tmp_path)) == 0
    command, observer, _, kwargs = calls[0]
    assert command == row["argv"]
    assert command.count("tools.pq_profile_child") == 1
    binding = namespace.require_namespace_publication(row)
    expected_cost = binding["root"] + "/" + binding["request_key"] + "/cost.pkl"
    assert kwargs["target_out"] == expected_cost
    assert observer[observer.index("--target-out") + 1] == expected_cost


@pytest.mark.parametrize("argument", ["--observations", "--profile-local", "--profiler-executable", "--row-s"])
def test_host_cli_drift_refuses_before_observer_or_workload(tmp_path, monkeypatch, argument):
    row = publish_profile_row(tmp_path)
    args = launch_args(row, tmp_path)
    args[args.index(argument) + 1] = "901" if argument == "--row-s" else str(tmp_path / "foreign")
    monkeypatch.setattr(profiler, "run_observed", lambda *args, **kwargs: pytest.fail("process owner reached"))
    with pytest.raises(RuntimeError, match="launch arguments"):
        profiler.main(args)


@pytest.mark.parametrize("destination", ["observations", "profile_local"])
def test_profile_output_collision_refuses_before_process_creation(tmp_path, monkeypatch, destination):
    row = publish_profile_row(tmp_path)
    Path(profile(row)[destination]).mkdir(parents=True)
    monkeypatch.setattr(profiler.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("process started"))
    with pytest.raises(FileExistsError):
        profiler.main(launch_args(row, tmp_path))


def test_broad_writable_identity_mount_can_cover_owned_metadata(tmp_path):
    kwargs = profiled_arguments(tmp_path)
    row = kwargs["requests"][0]
    spec = json.loads(row["argv"][4])
    parent = str(tmp_path.parent)
    spec["container"]["mounts"] = [{"source": parent, "target": parent, "readonly": False}]
    row["argv"][4] = json.dumps(spec)
    replace_fixture_request(kwargs, row)
    namespace.validate_namespace_request(dispatch.prepare_namespace_requests(**kwargs)[0])


@pytest.mark.parametrize("schema", ["known", "unknown"])
def test_declared_profile_without_namespace_binding_refuses_before_docker(tmp_path, monkeypatch, schema):
    checkout, _, row = real_source_row(tmp_path)
    spec = json.loads(row["argv"][4])
    del spec["namespace_binding"]
    if schema == "unknown": spec["namespace_profile"]["schema"] += ".unsupported"
    monkeypatch.chdir(checkout)
    class DockerReached(Exception):
        pass
    def inspect(*args):
        raise DockerReached("unbound instrumentation reached Docker")
    monkeypatch.setattr(adapter, "inspect_or_load", inspect)
    with pytest.raises(RuntimeError, match="namespace"):
        adapter.main(["--spec", json.dumps(spec), "--", *row["argv"][6:]])
