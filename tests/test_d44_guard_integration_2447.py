"""Bounded CPU transport integration for PQ #2447, not numerical qualification.

Every guard invocation runs in a fresh interpreter against real temporary owner
files. The stage1 double only records argv/cwd and writes tiny fixture receipts;
it performs no numerical work. The route-plan tests validate requests with the
real installed prismabuild.decomposition, not a test-only validator. No corrected
source is copied into the frozen tree, and no test injects modules into
sys.modules or starts Docker/a GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
from prismabuild import decomposition as dc


_REPO = Path(__file__).resolve().parents[1]
_GUARD = _REPO / "prismaquant" / "d44_guard" / "campaign_launch.py"
_MODULE = "prismaquant.d44_guard.campaign_launch"
_TIMEOUT_SECONDS = 60
_LABEL = "CPU transport only: spaces, quotes ' and unicode \u03c0"
_OUTCOMES = ("dry", "real", "failed")


_STAGE1_DOUBLE = r'''
"""CPU transport double only: these receipts do not qualify any numerics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from transport_source_identity import IDENTITY

OWNERS = Path(__file__).resolve().parent / "default owners"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("encode",))
    parser.add_argument("--device", choices=("cpu",), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--transport-exit-code", type=int, required=True)
    parser.add_argument("--transport-label", required=True)
    args = parser.parse_args()
    require(Path.cwd() == Path(__file__).resolve().parent, "child cwd is not source")
    require(Path("transport-source-token.txt").read_text(encoding="utf-8") == IDENTITY,
            "source-relative input was not readable")
    save(args.root / "transport-invocation.json", {
        "scope": "CPU transport double; not numerical qualification",
        "file": str(Path(__file__).resolve()),
        "entry_arg": sys.argv[0],
        "cwd": str(Path.cwd()),
        "argv": sys.argv[1:],
        "device": args.device,
        "stage": args.stage,
        "batch": str(args.batch),
        "root": str(args.root),
        "label": args.transport_label,
        "source_identity": IDENTITY,
        "dry_run": args.dry_run,
    })
    if args.dry_run:
        save(args.root / "transport-preflight.json", {"dry_run": True})
    else:
        # Even a failed child leaves receipts: exit status must gate publication.
        for task in load(args.batch)["tasks"]:
            name = task["payload"]["qname"].replace(".", "__") + ".json"
            save(args.root / "receipts" / name, {
                "scope": "CPU transport double; not numerical qualification",
                "task_id": task["id"], "label": args.transport_label,
            })
    return args.transport_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
'''


@dataclass(frozen=True)
class _TransportTree:
    source: Path
    owners: Path
    cwd: Path
    ambient: Path
    cache: Path


@pytest.fixture
def transport_tree(tmp_path):
    tmp_path = tmp_path.resolve()
    source = tmp_path / "frozen numerical tree"
    owners = tmp_path / "external delivered owners"
    cwd = tmp_path / "unrelated working directory"
    ambient = tmp_path / "ambient import decoys"
    for directory in (source, owners, cwd, ambient):
        directory.mkdir()
    (source / "stage1.py").write_text(textwrap.dedent(_STAGE1_DOUBLE), encoding="utf-8")
    (source / "transport_source_identity.py").write_text(
        'IDENTITY = "fixture source identity"\n', encoding="utf-8")
    (source / "transport-source-token.txt").write_text(
        "fixture source identity", encoding="utf-8")
    (source / "encode_launch.py").write_text(
        'raise RuntimeError("frozen launcher must be routed, never executed")\n',
        encoding="utf-8")
    default_owners = source / "default owners"
    default_owners.mkdir()
    (default_owners / "v2_launch.py").write_text(
        'raise RuntimeError("default D30 owner must not be imported")\n', encoding="utf-8")
    (owners / "owner_transport_identity.py").write_text(
        'IDENTITY = "external CPU owner double"\n', encoding="utf-8")
    (owners / "v2_launch.py").write_text(
        "from pathlib import Path\n"
        "from owner_transport_identity import IDENTITY\n"
        'TEMPLATE = Path(__file__).with_name("launch.json")\n'
        "def available():\n"
        "    return 16 * 2**30\n", encoding="utf-8")
    (owners / "launch.json").write_text(
        json.dumps({"spec": {"container": {"mounts": []}, "env": {}}}), encoding="utf-8")
    # A cwd/ambient import must not substitute for explicitly bound source/owners.
    # Any eager CUDA/residency dependency on the CPU path also fails immediately.
    for directory in (cwd, ambient):
        for name in ("stage1", "v2_launch", "g3_residency"):
            (directory / f"{name}.py").write_text(
                f'raise RuntimeError("ambient {name} must not be imported")\n',
                encoding="utf-8")
    return _TransportTree(source, owners, cwd, ambient, tmp_path / "test caches")


def _fresh_environment(tree):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("D44_") and key not in {
               "PRISMABUILD_CONTAINER_OWNER", "PYTHONINSPECT", "PYTHONOPTIMIZE",
           }}
    # Deliberately do not inherit an operator's PYTHONPATH or owner configuration.
    env.update({
        "PYTHONPATH": str(_REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": str(tree.cache),
        "TRITON_CACHE_DIR": str(tree.cache / "triton"),
    })
    return env


def _guard_command(entry):
    if entry == "module":
        return [sys.executable, "-m", _MODULE]
    assert entry == "file"
    return [sys.executable, str(_GUARD)]


def _run_fresh(argv, *, cwd, env):
    # The only child backend is the immediate, standard-library transport double.
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=_TIMEOUT_SECONDS, check=False)


def _expect_returncode(result, expected):
    assert result.returncode == expected, (
        f"argv={result.args!r}\nreturncode={result.returncode}, expected={expected}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def _tree_contents(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _write_transport_batch(root):
    root.mkdir(parents=True)
    path = root / "batch.json"
    path.write_text(json.dumps({
        "parent_key": "a" * 64,
        "plan_key": "b" * 64,
        "child_ordinal": 3,
        "result_manifest_path": str(root / "child-result.json"),
        "tasks": [
            {"id": "t0", "output_id": "o0", "payload": {"qname": "model.layer.0.proj"}},
            {"id": "t1", "output_id": "o1", "payload": {"qname": "model.layer.1.proj"}},
        ],
    }), encoding="utf-8")
    return path


def _transport_arguments(root, batch, outcome):
    argv = ["--device", "cpu", "--root", str(root), "--stage", "encode",
            "--batch", str(batch), "--transport-label", _LABEL,
            "--transport-exit-code", "7" if outcome == "failed" else "0"]
    if outcome == "dry":
        argv.append("--dry-run")
    return argv


def _assert_transport_result(tree, root, batch_path, outcome):
    invocation = json.loads((root / "transport-invocation.json").read_text(encoding="utf-8"))
    assert invocation["file"] == str(tree.source / "stage1.py")
    assert invocation["entry_arg"] == str(tree.source / "stage1.py")
    assert invocation["cwd"] == str(tree.source)
    assert invocation["stage"] == "encode"
    assert invocation["device"] == "cpu"
    assert invocation["root"] == str(root)
    assert invocation["batch"] == str(batch_path)
    assert invocation["label"] == _LABEL
    assert invocation["source_identity"] == "fixture source identity"
    assert invocation["dry_run"] is (outcome == "dry")
    assert "--source-dir" not in invocation["argv"]
    assert "--owners-dir" not in invocation["argv"]
    manifest_path = root / "child-result.json"
    if outcome == "dry":
        assert json.loads((root / "transport-preflight.json").read_text()) == {"dry_run": True}
        assert not (root / "receipts").exists()
        assert not manifest_path.exists()
        return
    assert not (root / "transport-preflight.json").exists()
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    expected_results = []
    for task in batch["tasks"]:
        receipt = root / "receipts" / (task["payload"]["qname"].replace(".", "__") + ".json")
        assert json.loads(receipt.read_text(encoding="utf-8")) == {
            "scope": "CPU transport double; not numerical qualification",
            "task_id": task["id"], "label": _LABEL,
        }
        expected_results.append({
            "task_id": task["id"], "output_id": task["output_id"],
            "value_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        })
    if outcome == "failed":
        assert not manifest_path.exists(), "partial receipts must not authorize publication"
    else:
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
            "schema": "prismabuild.child_result_manifest.v1",
            **{key: batch[key] for key in ("parent_key", "plan_key", "child_ordinal")},
            "results": expected_results,
        }


@pytest.mark.parametrize("entry", ("module", "file"))
@pytest.mark.parametrize("subcommand", ((), ("route-plan",)), ids=("launch", "route-plan"))
def test_help_is_lazy_from_an_unrelated_cwd(transport_tree, entry, subcommand):
    tree = transport_tree
    before = _tree_contents(tree.source)
    env = _fresh_environment(tree)
    # Even an explicitly configured but unavailable tree must not break --help.
    env["D44_SOURCE_DIR"] = str(tree.cwd / "absent source")
    env["D44_DELIVERED_OWNERS_DIR"] = str(tree.cwd / "absent owners")
    result = _run_fresh([*_guard_command(entry), *subcommand, "--help"], cwd=tree.cwd, env=env)
    _expect_returncode(result, 0)
    assert "--source-dir" in result.stdout
    assert "--owners-dir" in result.stdout
    assert _tree_contents(tree.source) == before


@pytest.mark.parametrize("entry", ("module", "file"))
@pytest.mark.parametrize("configuration", ("missing", "cli-absent", "env-absent"))
def test_unconfigured_source_refuses_without_scientific_outputs(
        transport_tree, tmp_path, entry, configuration):
    tree = transport_tree
    root = tmp_path / "refused run"
    batch = _write_transport_batch(root)
    before_root, before_source = _tree_contents(root), _tree_contents(tree.source)
    env = _fresh_environment(tree)
    argv = [*_guard_command(entry), *_transport_arguments(root, batch, "real")]
    if configuration == "cli-absent":
        argv.extend(["--source-dir", str(tree.cwd / "absent source")])
    elif configuration == "env-absent":
        env["D44_SOURCE_DIR"] = str(tree.cwd / "absent source")
    result = _run_fresh(argv, cwd=tree.cwd, env=env)
    assert result.returncode != 0, result.stdout
    expected = ("D44 requires --source-dir or D44_SOURCE_DIR" if configuration == "missing"
                else "D44 standalone source is absent")
    assert expected in result.stderr, result.stderr
    assert _tree_contents(root) == before_root
    assert _tree_contents(tree.source) == before_source


@pytest.mark.parametrize("entry", ("module", "file"))
@pytest.mark.parametrize("configuration", ("cli", "env"))
@pytest.mark.parametrize("outcome", _OUTCOMES)
def test_explicit_cpu_source_in_fresh_interpreter(
        transport_tree, tmp_path, entry, configuration, outcome):
    tree = transport_tree
    root = tmp_path / "CPU run root"
    batch = _write_transport_batch(root)
    before_source, before_owners = _tree_contents(tree.source), _tree_contents(tree.owners)
    env = _fresh_environment(tree)
    argv = [*_guard_command(entry), *_transport_arguments(root, batch, outcome)]
    source = os.path.relpath(tree.source, tree.cwd)
    owners = os.path.relpath(tree.owners, tree.cwd)
    if configuration == "cli":
        # CLI wins over a stale deployment environment, including paths with spaces.
        env["D44_SOURCE_DIR"] = str(tree.cwd / "wrong source")
        env["D44_DELIVERED_OWNERS_DIR"] = str(tree.cwd / "wrong owners")
        argv.extend(["--source-dir", source, "--owners-dir", owners])
    else:
        env["D44_SOURCE_DIR"] = source
        env["D44_DELIVERED_OWNERS_DIR"] = owners
    result = _run_fresh(argv, cwd=tree.cwd, env=env)
    _expect_returncode(result, 7 if outcome == "failed" else 0)
    _assert_transport_result(tree, root, batch, outcome)
    assert _tree_contents(tree.source) == before_source
    assert _tree_contents(tree.owners) == before_owners


@pytest.mark.parametrize("configuration", ("explicit", "env"))
def test_external_d30_owner_binding_is_lazy_and_cpu_only(transport_tree, configuration):
    """cuda=True here only imports a tiny owner file; no launcher/device is used."""
    tree = transport_tree
    env = _fresh_environment(tree)
    source = os.path.relpath(tree.source, tree.cwd)
    owners = os.path.relpath(tree.owners, tree.cwd)
    if configuration == "env":
        env["D44_SOURCE_DIR"], env["D44_DELIVERED_OWNERS_DIR"] = source, owners
    else:
        env["D44_SOURCE_DIR"] = str(tree.cwd / "wrong source")
        env["D44_DELIVERED_OWNERS_DIR"] = str(tree.cwd / "wrong owners")
    probe = textwrap.dedent('''
        import importlib.util
        import json
        from pathlib import Path
        import sys

        spec = importlib.util.spec_from_file_location("transport_guard_probe", sys.argv[1])
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        assert guard.S is None and guard.stage1 is None and guard.D30 is None
        kwargs = ({"source_dir": sys.argv[2], "owners_dir": sys.argv[3]}
                  if sys.argv[4] == "explicit" else {})
        cpu_owner = guard.configure_runtime(**kwargs)
        assert cpu_owner is guard.S is guard.stage1
        assert guard.D30 is None
        assert guard.configure_runtime(**kwargs, cuda=True) is cpu_owner
        print(json.dumps({
            "source": str(Path(cpu_owner.__file__).resolve()),
            "owner": str(Path(guard.D30.__file__).resolve()),
            "template": str(Path(guard.D30.TEMPLATE).resolve()),
            "identity": guard.D30.IDENTITY,
            "available_bytes": guard.D30.available(),
        }))
    ''')
    result = _run_fresh([sys.executable, "-c", probe, str(_GUARD), source, owners, configuration],
                        cwd=tree.cwd, env=env)
    _expect_returncode(result, 0)
    assert json.loads(result.stdout) == {
        "source": str(tree.source / "stage1.py"),
        "owner": str(tree.owners / "v2_launch.py"),
        "template": str(tree.owners / "launch.json"),
        "identity": "external CPU owner double", "available_bytes": 16 * 2**30,
    }


@pytest.mark.parametrize("owners_configuration", ("absent", "cli", "env"))
@pytest.mark.parametrize("outcome", _OUTCOMES)
def test_routed_argv_executes_delivered_guard_without_copying_source(
        transport_tree, tmp_path, owners_configuration, outcome):
    tree = transport_tree
    planner_cwd = tmp_path / "independent planner cwd"
    planner_cwd.mkdir()
    root = tmp_path / "routed CPU root"
    batch = _write_transport_batch(root)
    # The frozen entry's parent is the default source; do not pass source-dir to
    # route-plan. Its output must carry that path explicitly into another process.
    launch_args = _transport_arguments(root, dc.TASK_BATCH_PLACEHOLDER, outcome)
    stage_index = launch_args.index("--stage")
    del launch_args[stage_index:stage_index + 2]
    evidence = "cas:sha256:" + "0" * 64
    residency_keys = ("GPU-A", "GPU-B")
    batch_tasks = json.loads(batch.read_text(encoding="utf-8"))["tasks"]
    plan = {
        "schema": dc.LOGICAL_REQUEST_SCHEMA_V1,
        "common": {
            "argv": [sys.executable, "encode_launch.py", *launch_args],
            "cwd": str(tree.source),
            "demand": {"cpu": 1, "mem_gb": 4},
            "gpu_memory_gb": None,
            "data_manifest": None,
            "env": {},
        },
        "roster": {
            "schema": dc.LOGICAL_TASK_ROSTER_SCHEMA_V1,
            "tasks": [
                {
                    **task,
                    "residency_key": residency_keys[index],
                    "estimated_seconds": 8.2,
                    "estimate_evidence": evidence,
                }
                for index, task in enumerate(batch_tasks)
            ],
        },
        "batch_policy": {
            "schema": dc.ROSTER_BATCH_POLICY_SCHEMA_V1,
            "residencies": [
                {"key": key, "setup_seconds": 45.0, "setup_evidence": evidence}
                for key in residency_keys
            ],
            "max_setup_fraction": 0.20,
            "max_estimated_wall_seconds": 300.0,
        },
    }
    plan_path, routed_path = planner_cwd / "plan.json", planner_cwd / "routed.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    before_plan = plan_path.read_bytes()
    before_source, before_owners = _tree_contents(tree.source), _tree_contents(tree.owners)
    env = _fresh_environment(tree)
    route_argv = [*_guard_command("file"), "route-plan", plan_path.name, "--out", routed_path.name]
    if owners_configuration == "cli":
        env["D44_DELIVERED_OWNERS_DIR"] = str(planner_cwd / "wrong owners")
        route_argv.extend(["--owners-dir", os.path.relpath(tree.owners, planner_cwd)])
    elif owners_configuration == "env":
        env["D44_DELIVERED_OWNERS_DIR"] = os.path.relpath(tree.owners, planner_cwd)
    routed_result = _run_fresh(route_argv, cwd=planner_cwd, env=env)
    _expect_returncode(routed_result, 0)
    routed = json.loads(routed_path.read_text(encoding="utf-8"))
    assert dc.validate_logical_request(routed) == routed
    argv = routed["common"]["argv"]
    assert argv[:2] == [sys.executable, str(_GUARD.resolve())]
    assert argv.count(dc.TASK_BATCH_PLACEHOLDER) == 1
    assert argv[argv.index("--batch") + 1] == dc.TASK_BATCH_PLACEHOLDER
    assert argv.count("--source-dir") == 1
    assert argv[argv.index("--source-dir") + 1] == str(tree.source)
    assert argv.count("--stage") == 1
    assert argv[argv.index("--stage") + 1] == "encode"
    if owners_configuration == "absent":
        assert "--owners-dir" not in argv
    else:
        assert argv.count("--owners-dir") == 1
        assert argv[argv.index("--owners-dir") + 1] == str(tree.owners)
    assert [t["residency_key"] for t in routed["roster"]["tasks"]] == ["gpu-a", "gpu-b"]
    assert [r["key"] for r in routed["batch_policy"]["residencies"]] == ["gpu-a", "gpu-b"]
    assert plan_path.read_bytes() == before_plan
    assert not (tree.source / "campaign_launch.py").exists()
    assert not (root / "child-result.json").exists()
    # Bind the real transport batch only at execution, keeping the logical plan's
    # placeholder intact. A stale source environment must lose to routed CLI paths.
    execution_argv = [str(batch) if arg == dc.TASK_BATCH_PLACEHOLDER else arg for arg in argv]
    execution_env = _fresh_environment(tree)
    execution_env["D44_SOURCE_DIR"] = str(tree.cwd / "wrong execution source")
    execution_env["D44_DELIVERED_OWNERS_DIR"] = str(tree.cwd / "wrong execution owners")
    result = _run_fresh(execution_argv, cwd=tree.cwd, env=execution_env)
    _expect_returncode(result, 7 if outcome == "failed" else 0)
    _assert_transport_result(tree, root, batch, outcome)
    assert _tree_contents(tree.source) == before_source
    assert _tree_contents(tree.owners) == before_owners
    assert plan_path.read_bytes() == before_plan
