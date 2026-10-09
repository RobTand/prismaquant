"""Real SDK5 selected-result controls on private CPU CAS/queue fixtures.

The producer measurements and source review relation are controlled doubles.
PB request sealing, receipt publication, selected-result reads and capture
binding are real. These tests establish no genuine GPU pilot qualification.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from test_dispatch_joint_quanta import campaign, records_dir
from test_joint_dispatch_pilot import _argv, _pilot_spec, _published_pilots, _ready
from tools import dispatch_joint_quanta as dispatch


def _selected_fixture(tmp_path, monkeypatch, sdk, args, canned, fault):
    from prismabuild import core, movement_actions

    for name in ("PRISMABUILD_ACTION_NONCE", "PRISMABUILD_ACTION_SCOPE",
                 "PRISMABUILD_READER_HELPER_ROOT"):
        monkeypatch.delenv(name, raising=False)
    base = tmp_path / "private-pb"
    cas = core.PrismaBuildCAS(base / "cas")

    def source_tree(name, code):
        root = base / name
        root.mkdir(parents=True)
        (root / "fixture.py").write_text("# prior source in the same bundle\n")
        def git(*argv):
            return subprocess.run(["git", "-C", str(root), *argv], check=True,
                                  capture_output=True, text=True).stdout.strip()
        git("init", "-q", "-b", "master")
        git("-c", "user.name=CPU fixture", "-c", "user.email=fixture@example.invalid",
            "add", "fixture.py")
        git("-c", "user.name=CPU fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-q", "-m", "controlled source")
        prior = git("rev-parse", "HEAD")
        (root / "fixture.py").write_text(code)
        git("add", "fixture.py")
        git("-c", "user.name=CPU fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-q", "-m", "reviewed source selection")
        bundle = base / (name + ".bundle")
        git("bundle", "create", str(bundle), "--all")
        descriptor, _ = cas.ingest_input(bundle, input_id="pbrun.checkout-snapshot")
        return root, git("rev-parse", "HEAD"), descriptor, prior

    approved_root, approved_commit, reviewed, approved_parent = source_tree("reviewed", "# reviewed CPU fixture source\n")
    foreign_root, foreign_commit, foreign, foreign_parent = source_tree("foreign", "# foreign CPU fixture source\n")
    checkout, commit = ((foreign_root, foreign_commit) if fault == "source"
                        else (approved_root, approved_commit))
    parent = foreign_parent if fault == "source" else approved_parent
    if fault == "source_commit":
        subprocess.run(["git", "-C", str(approved_root), "checkout", "--detach", approved_parent],
                       check=True, capture_output=True)
        commit, parent = approved_parent, None
    queue = sdk.PoolQueue(base / "queue")
    queue.ensure_layout()
    extra = []
    for index, word in enumerate(args):
        if word == "--pilot-source-contract":
            path = Path(args[index + 1])
            contract = json.loads(path.read_bytes())
            contract["snapshot"] = reviewed
            contract["snapshot_selection"] = {
                "schema": "prismaquant.prismabuild.pbrun_checkout_snapshot.v2",
                "input": reviewed, "subdirectory": ".", "commit": approved_commit,
                "parent": approved_parent, "refs": {}}
            wire = json.dumps(contract, sort_keys=True).encode()
            path.write_bytes(wire)
            extra += [word, str(path), hashlib.sha256(wire).hexdigest()]
    resealed = []
    for candidate in canned.results.values():
        request = candidate["request"]
        command = copy.deepcopy(request["params"]["command"])
        descriptor = foreign if fault == "source" else reviewed
        if fault == "entry":
            command[8] = "foreign.module"
        elif fault == "plan":
            command[command.index("--plan-sha256") + 1] = "e" * 64
        elif fault == "environment":
            spec = json.loads(command[4])
            spec["env"]["PYTHONPATH"] = "/foreign/imports"
            command[4] = json.dumps(spec)
        wrapper = movement_actions.standard_capture_argv(command, "pbrun_result.txt", path_prefix="/usr/bin")
        if fault == "wrapper":
            wrapper = ["/bin/echo", "unrelated result"]
        action = core.seal_action({
            "schema": core.ACTION_SCHEMA_V2,
            "task": {"definition_id": "tests/pilot-result", "definition_version": "v1",
                     "task_class": "generation", "determinism": "deterministic",
                     "artifact_family": "generic", "artifact_kind": "generic",
                     "argv": wrapper, "working_directory": ".", "result_path": "pbrun_result.txt"},
            "inputs": [descriptor], "code_closure": core.build_code_closure(checkout, ["fixture.py"]),
            "params": {"command": command, "cwd": ".", "checkout_snapshot": {
                "schema": "prismaquant.prismabuild.pbrun_checkout_snapshot.v2",
                "input": descriptor, "subdirectory": "other" if fault == "subdirectory" else ".",
                "commit": commit, "parent": parent,
                "refs": {"master": approved_commit} if fault == "refs" else {}}},
            "environment": {"variables": {"PATH": "/usr/bin:/bin"}, "toolchain": {}},
            "execution_scope": {"portability": "portable", "platform_key": None, "host_class": None}})
        assert core.validate_action(action) == action
        resealed.append(action["action_key"])
        completion = json.loads(candidate["payload"])
        counters_path = Path(completion["counters"]["path"])
        counters = json.loads(counters_path.read_bytes())
        counters["pilot"]["action_key"] = action["action_key"]
        wire = json.dumps(counters).encode()
        counters_path.write_bytes(wire)
        digest = hashlib.sha256(wire).hexdigest()
        completion["counters"].update(sha256=digest, bytes=len(wire))
        if fault == "legacy":
            completion = {"passed": True, "units_done": 1}
        payload = json.dumps(completion).encode() + b"\n"
        cas.publish_action_request(action)
        attestation = core.preflight_action(action, cas_root=base / "cas", checkout_root=checkout)
        output = base / (action["action_key"] + ".output")
        output.write_bytes(payload)
        receipt, _ = cas.publish_result(action, output, attestation=attestation, return_execution_receipt=True)
        queue.publish(action_key=action["action_key"], cas_root=base / "cas", checkout_root=checkout,
                      worker_script=base / "worker.py", tags=["x86"], resources={"cpu": 1, "mem_gb": 1})
        queue.claim(tags=["x86"], capacity={"cpu": 8, "mem_gb": 16})
        ending = queue.finish(action["action_key"], status="executed", detail={
            "returncode": 0, "status": "executed", "stderr": "",
            "stdout": json.dumps({"status": "published", "receipt": receipt,
                                   "payload_path": str(cas.blob_path(receipt["result"]["sha256"]))}) + "\n"})
        published = json.loads(ending.read_bytes())["published_unix"]
        extra += ["--pilot-counters", str(counters_path), digest,
                  "--pilot-selector", action["action_key"], repr(published),
                  "2" if fault == "attempt" else "1"]

    class PrivateGateway(dispatch.FakeGateway):
        bind_capture_command = dispatch.Gateway.bind_capture_command

        def read_verified_result(self, selector):
            self.result_reads.append(selector)
            try:
                return sdk.read_verified_action_result(
                    queue, selector.action_key, published_unix=selector.published_unix,
                    attempt=selector.attempt, max_result_bytes=selector.max_result_bytes,
                    max_evidence_bytes=selector.max_evidence_bytes)
            except sdk.ActionResultError as exc:
                from prismaquant.joint_dispatch_pilot import PilotRefused
                raise PilotRefused(str(exc)) from exc

    return extra, PrivateGateway(), resealed


@pytest.mark.parametrize("fault", [None, "source", "source_commit", "subdirectory", "refs", "entry", "plan", "environment", "wrapper", "legacy", "attempt"])
def test_real_selected_result_admits_only_the_reviewed_source_and_invocation(
        tmp_path, campaign, records_dir, monkeypatch, installed_client_sdk, fault):
    receipt = _ready(tmp_path, campaign, records_dir)
    canned = dispatch.FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=canned)
    args, gateway, resealed = _selected_fixture(
        tmp_path, monkeypatch, installed_client_sdk, args, canned, fault)
    assert len(set(resealed)) == 3
    result = dispatch.main(_argv(records_dir, tmp_path / "out", receipt, extra=args), _gateway=gateway)
    assert result == (0 if fault is None else 3)
    assert bool(gateway.submitted) is (fault is None)


#: What the production Gateway refuses each fault for. A control that broke for
#: another reason, such as sealing or setup, would still say "refused"; it would
#: not say this.
_REFUSALS = {
    "source": "has no unique independently reviewed contract",
    "source_commit": "has no unique independently reviewed contract",
    "subdirectory": "has no supported checkout snapshot invocation",
    "refs": "has no unique independently reviewed contract",
    "entry": "is not the supported containerized quantum entry",
    "plan": "sealed invocation differs from the proposed row",
    "environment": "has no unique independently reviewed contract",
    "wrapper": "is not the standard capture recipe",
    "legacy": "carries 0 quantum completion announcements",
    "attempt": "is not the terminal attempt 1",
}


@pytest.mark.parametrize("fault", [None, "source", "source_commit", "subdirectory", "refs", "entry", "plan", "environment", "wrapper", "legacy", "attempt"])
def test_production_gateway_consumes_selected_results_from_the_installed_sdk5_helper(
        tmp_path, campaign, records_dir, monkeypatch, capsys, installed_helper_sdk, fault):
    """The real resolver, reader, binder and Gateway all use one SDK5 install."""
    from prismaquant import staged_lease

    sdk = installed_helper_sdk
    assert staged_lease._INJECTED is None
    assert staged_lease.client_sdk() is sdk
    receipt = _ready(tmp_path, campaign, records_dir)
    canned = dispatch.FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=canned)
    args, _, resealed = _selected_fixture(
        tmp_path, monkeypatch, sdk, args, canned, fault)
    from prismabuild import core, movement_actions, pool

    package = Path(sdk.__file__).resolve().parent
    for module in (core, movement_actions, pool):
        assert Path(module.__file__).resolve().parent == package
    assert staged_lease._INJECTED is None
    # Redirect only the existing queue owner's default, not the reader,
    # binder, resolver or production Gateway. Never touch the fleet queue.
    monkeypatch.setattr(pool, "DEFAULT_POOL_ROOT", tmp_path / "private-pb" / "queue")
    capsys.readouterr()
    output = tmp_path / "consumer-dry-run"
    result = dispatch.main(
        _argv(records_dir, output, receipt, extra=[*args, "--dry-run"]),
        _gateway=dispatch.Gateway())
    captured = capsys.readouterr()
    assert not output.exists()
    assert result == (0 if fault is None else 3)
    if fault is not None:
        assert "refused" in captured.err
        assert _REFUSALS[fault] in captured.err
        assert not captured.out
    else:
        rows = json.loads(captured.out)["rows"]
        assert len(rows) == len(resealed) == 3
        evidence = [row["pilot_admission"]["result"] for row in rows]
        assert {item["action_key"] for item in evidence} == set(resealed)
        assert all(item["attempt"] == 1 for item in evidence)
        assert all(len(item["receipt_sha256"]) == 64 for item in evidence)
