"""Concurrent first-time runtime identities acknowledge one immutable winner."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import multiprocessing
from pathlib import Path
import sys

import pytest


_IMAGE_REF = "vllm-node@sha256:" + "a" * 64
_IMAGE_ID = "sha256:" + "a" * 64
_COMMIT = "b" * 40


def _load_owner(path):
    name = "_audit_container_runtime_identity"
    spec = importlib.util.spec_from_file_location(name, path)
    owner = importlib.util.module_from_spec(spec)
    sys.modules[name] = owner
    spec.loader.exec_module(owner)
    return owner


def _writer(owner_path, barrier, outcomes, parameters, interface):
    owner = _load_owner(owner_path)
    create_stage = owner.tempfile.NamedTemporaryFile

    def synchronized_stage(*args, **kwargs):
        barrier.wait(timeout=30)
        return create_stage(*args, **kwargs)

    owner.tempfile.NamedTemporaryFile = synchronized_stage
    try:
        if interface == "api":
            payload = owner.write_or_verify_identity(**parameters)
        else:
            arguments = ["write-or-verify"]
            for key, value in parameters.items():
                option = {"identity_path": "identity", "implementation_receipt": "implementation-receipt"}.get(
                    key, key.replace("_", "-"))
                arguments.extend(["--" + option, str(value)])
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = owner.main(arguments)
            if status != 0:
                outcomes.put({"status": "refused", "code": status, "error": stderr.getvalue()})
                return
            payload = json.loads(stdout.getvalue())
        outcomes.put({"status": "accepted", "payload": payload})
    except owner.RuntimeIdentityError as error:
        outcomes.put({"status": "refused", "error": str(error)})
    except BaseException as error:
        outcomes.put({"status": "error", "error": f"{type(error).__name__}: {error}"})
    finally:
        owner.tempfile.NamedTemporaryFile = create_stage


def _race(tmp_path, *, distinct, interface):
    owner_path = Path.cwd() / "prismaquant" / "container_runtime_identity.py"
    assert owner_path.is_file(), "Run this audit from the sealed repository root"
    source = tmp_path / "source"
    package = source / "prismaquant"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('MARKER = "reviewed"\n')
    (package / "data.json").write_text('{"marker": "reviewed"}')
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    identity = checkpoint / "container_runtime_identity.json"
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"git_commit": _COMMIT}))
    parameters = {
        "identity_path": identity, "checkpoint_root": checkpoint,
        "source_root": source, "target": "dsv4", "image_ref": _IMAGE_REF,
        "image_id": _IMAGE_ID, "git_commit": _COMMIT,
        "implementation_receipt": receipt,
    }
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    outcomes = context.Queue()
    variants = [parameters, {**parameters, "image_id": "sha256:" + "c" * 64} if distinct else parameters]
    workers = [context.Process(target=_writer, args=(
        str(owner_path), barrier, outcomes, variant, interface)) for variant in variants]
    try:
        for worker in workers:
            worker.start()
        barrier.wait(timeout=30)
        results = [outcomes.get(timeout=30) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0, results
    finally:
        barrier.abort()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
        outcomes.close()
        outcomes.join_thread()
    assert all(result["status"] != "error" for result in results), results
    return json.loads(identity.read_text()), results


@pytest.mark.parametrize("interface", ["api", "cli"])
def test_matching_concurrent_initial_identities_acknowledge_the_same_bytes(tmp_path, interface):
    saved, results = _race(tmp_path, distinct=False, interface=interface)
    assert [result["status"] for result in results] == ["accepted", "accepted"], results
    assert all(result["payload"] == saved for result in results), results


@pytest.mark.parametrize("interface", ["api", "cli"])
def test_distinct_concurrent_initial_identity_refuses_the_loser(tmp_path, interface):
    saved, results = _race(tmp_path, distinct=True, interface=interface)
    assert sorted(result["status"] for result in results) == ["accepted", "refused"], results
    accepted = next(result for result in results if result["status"] == "accepted")
    refused = next(result for result in results if result["status"] == "refused")
    assert accepted["payload"] == saved
    assert "container runtime differs" in refused["error"]
    assert "image_id" in refused["error"]
    if interface == "cli":
        assert refused["code"] == 2
