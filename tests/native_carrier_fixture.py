"""Private CPU native-carrier evidence for the SDK5 release smoke.

Adapted from PrismaBuild's test_verified_action_result native fixture. This
constructs evidence through the installed CAS and queue writers; it does not
execute a native worker, qualify a GPU/fleet, or recover historical evidence.
Only the runtime source identities used to *write* this synthetic receipt are
rebound, with monkeypatch restoration. The public reader is never replaced.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


def native_carrier(base: Path, monkeypatch):
    from prismabuild import core as pb
    from importlib import import_module
    from prismabuild import movement_actions, pool, produced_output, reader_lease, resource_scope

    # An outer admitted test action must not supply this private producer's
    # launch identity or influence preflight's interpretation of its demand.
    for name in ("PRISMABUILD_ACTION_NONCE", "PRISMABUILD_ACTION_SCOPE",
                 "PRISMABUILD_READER_HELPER_ROOT"):
        monkeypatch.delenv(name, raising=False)
    checkout = base / "checkout"
    checkout.mkdir(parents=True)
    generation = base / "generation"
    # Use the installed Core/immutable owner bytes, not a served checkout or
    # historical Git object. They are evidence files, never imported here.
    sources = [(pb, "src/prismabuild/core.py", "_LOADED_WORKER_CORE_IDENTITY")]
    # SDK5 predates extraction of immutable_files on some installed pins.
    if hasattr(pb, "_LOADED_WORKER_IMMUTABLE_IDENTITY"):
        sources.append((import_module("prismabuild.immutable_files"),
                        "src/prismabuild/immutable_files.py",
                        "_LOADED_WORKER_IMMUTABLE_IDENTITY"))
    for module, relative, identity in sources:
        target = generation / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(module.__file__).read_bytes())
        target.chmod(0o444)
        monkeypatch.setattr(pb, identity, pb._identify_runtime_source(
            target, where="private CPU carrier runtime"))
    launcher = generation / "tools/prismabuild_worker.py"
    launcher.parent.mkdir()
    launcher.write_text("# Synthetic launcher evidence; never executed.\n")
    launcher.chmod(0o444)
    proxy = generation / "tools/resource_exec.py"
    proxy.write_text("# Synthetic broker proxy evidence; never executed.\n")
    proxy.chmod(0o444)
    (checkout / "task_code.py").write_text("# Private captured-log closure.\n")
    log_line = "1 passed: synthetic CPU native-carrier contract integration"
    command = [sys.executable, "-c", f"print({log_line!r})"]
    demand = {"cpu": 1, "mem_gb": 1}
    action = pb.seal_action({
        "schema": pb.ACTION_SCHEMA_V2,
        "task": {
            "definition_id": "tests/held-producer-sdk5", "definition_version": "v1",
            "task_class": "generation", "determinism": "deterministic",
            "artifact_family": "generic", "artifact_kind": "generic",
            "argv": movement_actions.standard_capture_argv(
                command, "pbrun_result.txt", path_prefix=str(generation / "tools")),
            "working_directory": ".", "result_path": "pbrun_result.txt",
        },
        "inputs": [],
        "code_closure": pb.build_code_closure(checkout, ["task_code.py"]),
        "params": {"command": command, "demand": demand},
        "environment": {"variables": {"PATH": str(generation / "tools") + ":/usr/bin:/bin"},
                        "toolchain": {}},
        "execution_scope": {"portability": "portable", "platform_key": None,
                            "host_class": None},
    })
    cas = pb.PrismaBuildCAS(base / "cas")
    cas.publish_action_request(action)
    attestation = pb.preflight_action(
        action, cas_root=base / "cas", checkout_root=checkout,
        worker_launcher_identity=pb._identify_runtime_source(
            launcher, where="private CPU carrier launcher"))
    payload = (log_line + "\n").encode()
    output = base / "output.txt"
    output.write_bytes(payload)
    receipt, _won = cas.publish_result(
        action, output, attestation=attestation, return_execution_receipt=True)

    key = action["action_key"]
    host = receipt["producer"]["evidence"]["hostname"]
    nonce = "a" * 32
    scope_id = produced_output._broker_scope_id(key, nonce)
    telemetry = {"complete": True, "action_key": key, "nonce": nonce,
                 "scope_unit": scope_id, "host": host, "memory_max_bytes": 1024 ** 3}
    allocation = {"preferred": [0], "fallback": []}
    scope = {"action_key": key, "nonce": nonce, "scope_id": scope_id,
             "memory_max_bytes": 1024 ** 3, "token": "b" * 64,
             "socket_path": str(resource_scope.BROKER_SOCKET),
             "cgroup_path": "/sys/fs/cgroup/prismabuild.slice/" + scope_id}
    export = {"scope_id": scope_id, "stopped_unix": 42.0, "empty": True,
              "tickets_pending": False, "released": True, "retired": False,
              "settled": True}
    queue = pool.PoolQueue(base / "queue")
    # Pool record methods moved to a records owner after the installed SDK5
    # pin; both writers preserve the same immutable evidence contract.
    records = getattr(queue, "records", queue)
    records.ensure_layout()
    queue.publish(action_key=key, cas_root=base / "cas", checkout_root=checkout,
                  worker_script=launcher, tags=["cpu-fixture"], resources=demand)
    # Model a selected claim, not a scheduler run. Claim admission needs a
    # live host measurement census, which is deliberately not fixture input.
    # finish below is the real writer of immutable attempts/logs/attestation.
    ready_path = records.item_path(pool.READY, key)
    claimed_path = records.item_path(pool.CLAIMED, key)
    claimed = json.loads(ready_path.read_text())
    claimed.update({"claimed_host": host, "claimed_by": "cpu-fixture-worker",
                    "claimed_unix": claimed["published_unix"],
                    "resource_scope": scope, "cpu_allocation": allocation,
                    "resource_scope_cleanup": {"complete": True, "nonce": nonce,
                                               "telemetry": telemetry, "export": export}})
    claimed_path.write_bytes(pb._canonical_file_bytes(claimed))
    ready_path.unlink()
    argv = resource_scope._wrapped_scope_argv(
        pool._with_cpu_affinity([sys.executable] + pool.worker_argv(
            worker_script=launcher, action_key=key, cas_root=base / "cas",
            checkout_root=checkout), allocation["preferred"]),
        python=sys.executable, helper=proxy, socket_path=scope["socket_path"],
        action_key=key, nonce=nonce, token=scope["token"])
    queue.finish(key, status="executed", detail={
        "returncode": 0, "status": "executed", "stderr": "",
        "stdout": json.dumps({"status": "published", "receipt": receipt,
                              "payload_path": str(cas.blob_path(receipt["result"]["sha256"]))}) + "\n",
        "resource_telemetry": telemetry, "cpu_allocation": allocation, "argv": argv,
    })
    record = json.loads(records.item_path(pool.DONE, key).read_text())
    attempt = json.loads(records.attempt_path(record, 1).read_text())
    proof = reader_lease.attestation_path(queue, key, nonce)
    assert proof.is_file()
    expected = {
        "queue_root": str(queue.root.resolve()), "action_key": key,
        "published_unix": record["published_unix"], "attempt": attempt["attempt"],
        "generation": records.attempt_generation(record),
        "nonce": nonce, "scope_id": scope_id, "host": attempt["claimed_host"],
        "worker": attempt["claimed_by"], "incarnation": attempt["claimed_by"],
        "helper_root": str(generation), "resources": demand,
        "resources_semantics": "selected-claim-sealed-demand",
        "attempt_source": "selected-immutable-attempt",
        "receipt_sha256": receipt["receipt_sha256"],
        "runtime_sha256": receipt["producer"]["runtime"]["runtime_sha256"],
        "scope_attestation_sha256": pb.raw_sha256(proof.read_bytes()),
    }
    return queue, action, record, receipt, payload, expected
