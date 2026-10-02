"""Pilot admission tests mutate published QuantumCounters receipts.

Power/rate inputs are controlled CPU doubles, not GPU performance evidence.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from test_dispatch_joint_quanta import (
    _receipt, _stamp_receipt, _write_receipt, campaign, records_dir,
)
from tools.dispatch_joint_quanta import FakeGateway, main


@pytest.fixture(autouse=True)
def _pilot_spec(tmp_path, monkeypatch):
    from stage_a_spool_spec import with_spool
    from tools import dispatch_joint_quanta as dispatch

    spec = tmp_path / "pilot-spec.json"
    spec.write_text(json.dumps(with_spool(
        {"container": {"image": "sha256:" + "0" * 64}, "env": {}})))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)


def _argv(records_dir, output_root, receipt, extra=()):
    # No implicit override: unlike legacy dispatcher behavior tests, these
    # calls exercise the production pilot gate.
    return ["--records", str(records_dir), "--output-root", str(output_root),
            "--adjoint-receipt", str(receipt), *extra]


def _ready(tmp_path, campaign, records_dir):
    receipt = tmp_path / "receipt.json"
    _write_receipt(receipt, _receipt(campaign))
    _stamp_receipt(records_dir, receipt)
    return receipt


@pytest.mark.parametrize("dry_run", [False, True])
def test_fanout_without_a_measured_pilot_refuses_before_submission(
        tmp_path, campaign, records_dir, capsys, dry_run):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _argv(records_dir, tmp_path / "out", receipt,
                 extra=["--dry-run"] if dry_run else [])
    assert main(args, _gateway=gateway) == 3
    assert not gateway.submitted
    assert "pilot" in capsys.readouterr().err


def test_one_quantum_can_produce_the_pilot(tmp_path, campaign, records_dir):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    assert main(_argv(records_dir, tmp_path / "out", receipt,
                      extra=["--quantum", "layer-000"]),
                _gateway=gateway) == 0
    assert len(gateway.submitted) == 1
    state = tmp_path / "out" / "layer-quanta" / "campaign-state.json"
    events = [json.loads(line) for line in state.read_text().splitlines()]
    assert events[-1]["pilot_admission"]["mode"] == "pilot"


class _Power:
    interval_s = 1.0

    def __init__(self, now):
        self.times = [now - 30 + i for i in range(31)]
        self.samples = [12.0] * 10 + [70.0] * 21

    def stop(self):
        return {"sample_count": len(self.samples), "gpu_joules": 100.0,
                "gpu_power_w_p50": 70.0, "gpu_power_w_p95": 70.0,
                "gpu_power_w_max": 70.0}


def _published_pilots(tmp_path, records_dir, monkeypatch, mutate=None, *,
                      disabled_kernel_profile=False, gateway=None,
                      mutate_completion=None, quantum_override=None):
    """Use the row's counters and publisher, then mutate their output bytes.

    Returns ``extra`` argv (``--pilot-counters``/``--pilot-selector``). With a
    ``gateway`` the helper also publishes one canned verified result per pilot
    on that gateway, so the real SDK4-bound admission path runs.
    """
    from tools import dispatch_joint_quanta as dispatch
    from prismaquant import joint_dispatch_pilot as pilot
    from prismaquant.joint_cost_quantum import (
        ChunkFrontier, QuantumCounters, publish_quantum_outputs,
        quantum_completion_record,
    )

    monkeypatch.setattr(dispatch, "_pilot_code_sha256", lambda: "f" * 64)
    args = []
    contract_refs = set()
    # The fixture's layers have distinct shapes (different layer/geometry), so
    # each row needs its own pilot action — exactly as a real fanout does when
    # shapes differ. Each pilot gets its own action key so selectors stay
    # unique (PQ #1293).
    record_paths = sorted(records_dir.glob("layer-*.json"))
    for ordinal, record_path in enumerate(record_paths):
        action_key = f"{ordinal:064x}"
        monkeypatch.setenv("PRISMABUILD_ACTION_KEY", action_key)
        published = round(time.time(), 6)
        record = json.loads(record_path.read_text())
        now = time.time()
        windows = [{**w, "render_file_upper_bound_bytes": 300}
                   for w in record["windows"]]
        frontier = ChunkFrontier(chunks=record["chunks"], windows=windows)
        counters = QuantumCounters(
            quantum_id=record["quantum_id"], identity_sha256=record["identity_sha256"],
            chunks=record["chunks"], frontier=frontier,
            sampler=_Power(now), started=now - 30,
            pilot_binding=pilot.pilot_binding(
                record, implementation_sha256="f" * 64,
                execution_plan_sha256=record["campaign"]["plan_sha256"],
                replay_regime=None, cotangent_source="chain", emits_handoff=False))
        if disabled_kernel_profile:
            from prismaquant.joint_adjoint_checkpoints import KernelTimeProfiler
            from prismaquant.joint_cost_quantum import KERNEL_PROFILE_NOT_MEASURED
            counters.kernel_block(KernelTimeProfiler(
                not_measured=KERNEL_PROFILE_NOT_MEASURED))
        counters.gpu_work_started_unix = now - 20
        counters.phases[0]["entered_unix"] = now - 15
        counters.exposed_wait.take("window-load", now - 8, now - 7, {
            "bytes": 300, "work_before_s": 1.0,
            "load_bytes_per_s": 100.0, "first_fill": False})
        block = counters.finish(units_done=1, units_total=1)
        root = tmp_path / f"pilot-{record['layer']}"
        root.mkdir()
        outputs = {"root": str(root), "cost_payload": str(root / "cost.pkl"),
                   "counters": str(root / "counters.json"),
                   "results": str(root / "results.json")}
        publish_quantum_outputs(
            {**record, "output_space": outputs}, payload={"costs": {"unit": {}}},
            result={}, counters=block, units_total=1)
        path = root / "counters.json"
        if mutate is not None:
            document = json.loads(path.read_text())
            mutate(document)
            path.write_text(json.dumps(document))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        args += ["--pilot-counters", str(path), digest,
                 "--pilot-selector", action_key, repr(published), "1"]
        from tools.dispatch_joint_quanta import PilotSelector

        wire = record_path.read_bytes()
        sealed_sha = hashlib.sha256(wire).hexdigest()
        completion = quantum_completion_record(
            {"quantum_id": record["quantum_id"], "passed": True,
             "status": "complete", "units_done": 1, "units_total": 1,
             "counters": {"path": str(path), "sha256": digest,
                          "bytes": path.stat().st_size}},
            record_bytes=wire)
        if quantum_override is not None:
            completion = quantum_override(completion) or completion
        if mutate_completion is not None:
            completion = mutate_completion(completion) or completion
        selector = PilotSelector(action_key, published, 1)
        row_argv = dispatch.quantum_argv(
            record, record_path=record_path, output_root=tmp_path / "out")
        command = row_argv[row_argv.index("--") + 1:]
        # This review input is supplied independently of the result and
        # counter bytes. Its source/entry/environment are the planned ones.
        descriptor = {"id": "pbrun.checkout-snapshot", "sha256": "9" * 64, "bytes": 1234}
        contract = {"schema": pilot.PILOT_SOURCE_SCHEMA, "snapshot": descriptor,
                    "snapshot_commit": "7" * 40,
                    "implementation_sha256": "f" * 64,
                    "launcher_argv": pilot.PILOT_LAUNCHER,
                    "quantum_argv": pilot.PILOT_ENTRY,
                    "container_spec": json.loads(command[command.index("--spec") + 1]),
                    "outer_environment": {"PATH": "/usr/bin:/bin"}}
        source_wire = json.dumps(contract, sort_keys=True).encode()
        source_sha = hashlib.sha256(source_wire).hexdigest()
        source_path = tmp_path / (source_sha + ".pilot-source.json")
        source_path.write_bytes(source_wire)
        if source_sha not in contract_refs:
            args += ["--pilot-source-contract", str(source_path), source_sha]
            contract_refs.add(source_sha)
        result = {
            "schema": "prismabuild.verified_action_result.v1",
            "action_key": action_key, "published_unix": published, "attempt": 1,
            "generation": "g" * 64, "worker_id": "fixture", "host": "fixture",
            "request": {"params": {"command": command, "cwd": ".",
                "checkout_snapshot": {"schema": "prismaquant.prismabuild.pbrun_checkout_snapshot.v2",
                                      "input": descriptor, "subdirectory": ".", "commit": "7" * 40}},
                        "inputs": [descriptor], "task": {"working_directory": "."},
                        "environment": {"variables": {"PATH": "/usr/bin:/bin"}}},
            "receipt": {"receipt_sha256": "c" * 64},
            "payload": json.dumps(completion).encode() + b"\n",
            "inputs": [], "input_payloads": {}}
        if gateway is not None:
            gateway.publish_result(selector, result)
    return args


def test_matching_producer_pilots_admit_and_stamp_the_fanout(
        tmp_path, campaign, records_dir, monkeypatch):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    pilot_args = _published_pilots(tmp_path, records_dir, monkeypatch,
                                   gateway=gateway)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=pilot_args),
                _gateway=gateway) == 0
    assert len(gateway.submitted) == 3
    state = tmp_path / "out" / "layer-quanta" / "campaign-state.json"
    events = [json.loads(line) for line in state.read_text().splitlines()]
    for event in events:
        admission = event["pilot_admission"]
        assert admission["mode"] == "verified"
        assert admission["gpu_envelope_fraction"] == 0.5
        assert len(admission["action_key"]) == 64
        assert len(admission["document"]["sha256"]) == 64
        assert admission["result"]["action_key"] == admission["action_key"]
        assert admission["result"]["attempt"] == 1


def test_optional_kernel_profile_does_not_invalidate_measured_wait(
        tmp_path, campaign, records_dir, monkeypatch):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch,
                             disabled_kernel_profile=True, gateway=gateway)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 0
    assert len(gateway.submitted) == 3


@pytest.mark.parametrize("field,value", [
    ("implementation_sha256", "0" * 64),
    ("replay_regime", {"capture_batch": 2, "accumulation": "per_invocation", "chunk_rows": None}),
    ("row_shape_sha256", "0" * 64),
])
def test_stale_binding_refuses_without_publishing(
        tmp_path, campaign, records_dir, monkeypatch, field, value):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch,
                             mutate=lambda c: c["pilot"]["binding"].update({field: value}),
                             gateway=gateway)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted


def test_before_regime_wait_refuses_with_phase_excess_and_rates(
        tmp_path, campaign, records_dir, monkeypatch, capsys):
    receipt = _ready(tmp_path, campaign, records_dir)

    def before(counters):
        take = counters["exposed_wait"]["bound"]["per_take"][0]
        take["wait_s"] = 6.0
        take["end_unix"] = take["start_unix"] + 6.0

    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, mutate=before,
                             gateway=gateway)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted
    error = capsys.readouterr().err
    assert "layer-002-chunk-000" in error
    assert "excess" in error and "4" in error
    assert "load_bytes_per_s" in error and "100" in error


@pytest.mark.parametrize("defect", ["failed", "incomplete", "power", "rates", "unsampled", "unbounded-spill"])
def test_incomplete_measurement_never_certifies_a_pilot(
        tmp_path, campaign, records_dir, monkeypatch, defect):
    receipt = _ready(tmp_path, campaign, records_dir)

    def break_receipt(c):
        if defect == "failed":
            c["outcome"]["status"] = "gapped"
        elif defect == "incomplete":
            c["units"] = [None, 1]
        elif defect == "power":
            c["gpu_sampler_samples"] = 0
        elif defect == "rates":
            c["exposed_wait"]["bound"]["per_take"][0]["load_bytes_per_s"] = None
        elif defect == "unsampled":
            c["exposed_wait"]["total"].update(unsampled_s=1.0, busy_band_s=0.0)
        else:
            total = c["exposed_wait"]["total"]
            total["by_kind"]["spill-reader"] = {"wait_s": 1.0, "count": 1}
            total.update(wait_s=2.0, busy_band_s=2.0)

    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, mutate=break_receipt,
                             gateway=gateway)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted


def test_a_pilot_without_a_verified_result_refuses(tmp_path, campaign, records_dir, monkeypatch):
    """The counters naming an action with no readable verified result refuse."""
    receipt = _ready(tmp_path, campaign, records_dir)
    args = _published_pilots(tmp_path, records_dir, monkeypatch)
    gateway = FakeGateway()
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted


def test_a_missing_selector_refuses_unbound_counters(tmp_path, campaign, records_dir,
                                                    monkeypatch, capsys):
    """Counters whose action has no --pilot-selector cannot certify the row."""
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=gateway)
    stripped = []
    skip = 0
    for word in args:
        if skip:
            skip -= 1
            continue
        if word == "--pilot-selector":
            skip = 3
            continue
        stripped.append(word)
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=stripped),
                _gateway=gateway) == 3
    assert not gateway.submitted
    assert "unbound counter document" in capsys.readouterr().err


def test_a_duplicate_selector_refuses(tmp_path, campaign, records_dir, monkeypatch):
    """Two selectors naming one action key are ambiguous, never a merge."""
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=gateway)
    # The first row's selector repeated: same action key, a second triple.
    first_key = args[args.index("--pilot-selector") + 1]
    extra = args + ["--pilot-selector", first_key, "1.0", "1"]
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=extra),
                _gateway=gateway) == 3
    assert not gateway.submitted


def test_a_substituted_quantum_record_refuses(tmp_path, campaign, records_dir,
                                              monkeypatch, capsys):
    """A completion inlining other record bytes than the sealed digest refuses."""
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(
        tmp_path, records_dir, monkeypatch, gateway=gateway,
        quantum_override=lambda completion: {
            # Same declared length and digest, different bytes: the validator
            # must reject the block rather than trust the declared fields.
            **completion,
            "quantum_record": _flip_base64(completion["quantum_record"])})
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted
    assert "do not match their digest" in capsys.readouterr().err


def test_a_completion_naming_another_quantum_refuses(tmp_path, campaign, records_dir,
                                                     monkeypatch, capsys):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(
        tmp_path, records_dir, monkeypatch, gateway=gateway,
        mutate_completion=lambda completion: {**completion, "quantum_id": "layer-999"})
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted
    assert "another quantum" in capsys.readouterr().err


def test_a_wrong_attempt_selector_refuses(tmp_path, campaign, records_dir,
                                          monkeypatch, capsys):
    """A selector for an attempt the reader does not hold refuses."""
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=gateway)
    # Rewrite every attempt to 2; no result is published for attempt 2.
    patched = []
    index = 0
    while index < len(args):
        if args[index] == "--pilot-selector":
            patched += ["--pilot-selector", args[index + 1], args[index + 2], "2"]
            index += 4
            continue
        patched.append(args[index])
        index += 1
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=patched),
                _gateway=gateway) == 3
    assert not gateway.submitted
    assert "no verified result published" in capsys.readouterr().err


def _flip_base64(encoded: str) -> str:
    """A same-length base64 string carrying different bytes."""
    import base64

    raw = bytearray(base64.b64decode(encoded, validate=True))
    raw[0] = raw[0] ^ 0x01
    return base64.b64encode(bytes(raw)).decode("ascii")


def test_override_is_explicit_and_persistent(tmp_path, campaign, records_dir):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    assert main(_argv(records_dir, tmp_path / "out", receipt,
                      extra=["--force-unverified-pilot"]), _gateway=gateway) == 0
    state = tmp_path / "out" / "layer-quanta" / "campaign-state.json"
    events = [json.loads(line) for line in state.read_text().splitlines()]
    assert len(events) == 3
    assert all(event["pilot_admission"]["mode"] == "override" for event in events)


@pytest.mark.parametrize("fault", ["entry", "plan", "prepared", "adjoint", "regime"])
def test_sealed_invocation_mismatch_refuses(tmp_path, campaign, records_dir,
                                           monkeypatch, fault):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=gateway)
    for result in gateway.results.values():
        command = result["request"]["params"]["command"]
        if fault == "entry":
            command[command.index("prismaquant.joint_cost_quantum")] = "foreign.module"
        elif fault == "regime":
            position = command.index("--spec") + 1
            spec = json.loads(command[position])
            spec["env"]["PRISMAQUANT_STAGE_B_REPLAY_REGIME"] = "capture_batch=4"
            command[position] = json.dumps(spec)
        else:
            flag = {"plan": "--plan-sha256", "prepared": "--prepared-sha256",
                    "adjoint": "--adjoint-slice-sha256"}[fault]
            command[command.index(flag) + 1] = "e" * 64
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted


def test_replaced_record_shape_refuses(tmp_path, campaign, records_dir, monkeypatch):
    import base64
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=gateway)
    for result in gateway.results.values():
        completion = json.loads(result["payload"])
        record = json.loads(base64.b64decode(completion["quantum_record"]))
        record["chunks"][0]["end_bytes"] += 1
        del record["identity_sha256"]
        record["identity_sha256"] = canonical_json_sha256(record, where="altered record")
        wire = json.dumps(record).encode()
        completion.update(quantum_record=base64.b64encode(wire).decode(),
                          quantum_record_sha256=hashlib.sha256(wire).hexdigest(),
                          quantum_record_bytes=len(wire))
        command = result["request"]["params"]["command"]
        command[command.index("--quantum-sha256") + 1] = completion["quantum_record_sha256"]
        result["payload"] = json.dumps(completion).encode()
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted


def test_real_sdk_capture_binder_proves_wrapper_then_reads_command(
        tmp_path, installed_client_sdk):
    from prismabuild import core, movement_actions
    from tools import dispatch_joint_quanta as dispatch

    code = tmp_path / "closure.py"
    code.write_text("# reviewed test closure\n")
    command = ["python3", "-m", "tools.tessera_campaign_container", "--spec",
               json.dumps({"env": {}, "container": {"image": "sha256:" + "0" * 64}}),
               "--", "python3", "-m", "prismaquant.joint_cost_quantum",
               "--quantum", "/absent/original.json", "--quantum-sha256", "a" * 64,
               "--plan", "/plan", "--plan-sha256", "b" * 64,
               "--prepared", "/prepared", "--prepared-sha256", "c" * 64,
               "--adjoint-slice", "/slice", "--adjoint-slice-sha256", "d" * 64,
               "--data-manifest-sha256", "e" * 64,
               "--allowed-tiers", "ram,stage", "--resume", "--output-root", "/out"]
    request = core.seal_action({
        "schema": core.ACTION_SCHEMA_V2,
        "task": {"definition_id": "tests/pilot-binding", "definition_version": "v1",
                 "task_class": "generation", "determinism": "deterministic",
                 "artifact_family": "generic", "artifact_kind": "generic",
                 "argv": movement_actions.standard_capture_argv(
                     command, "pbrun_result.txt", path_prefix="/tools"),
                 "working_directory": ".", "result_path": "pbrun_result.txt"},
        "inputs": [], "code_closure": core.build_code_closure(tmp_path, ["closure.py"]),
        "params": {"command": command},
        "environment": {"variables": {"PATH": "/tools:/usr/bin:/bin"}, "toolchain": {}},
        "execution_scope": {"portability": "portable", "platform_key": None,
                            "host_class": None}})
    assert installed_client_sdk.bind_standard_capture_command(request) == request["task"]["argv"]
    parsed = dispatch._sealed_quantum_command({"request": request})
    assert parsed["--quantum-sha256"] == "a" * 64


def test_legacy_unbound_result_refuses(tmp_path, campaign, records_dir, monkeypatch):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=gateway)
    for result in gateway.results.values():
        result["payload"] = b'{"passed": true, "units_done": 1}\n'
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args),
                _gateway=gateway) == 3
    assert not gateway.submitted


@pytest.mark.parametrize("fault", ["snapshot", "outer_environment", "image", "self_asserted_code"])
def test_resealed_unreviewed_source_refuses(tmp_path, campaign, records_dir, monkeypatch, fault):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=gateway)
    for result in gateway.results.values():
        request = result["request"]
        if fault == "snapshot":
            request["params"]["checkout_snapshot"]["input"]["sha256"] = "8" * 64
        elif fault == "outer_environment":
            request["environment"]["variables"]["PYTHONPATH"] = "/foreign/imports"
        elif fault == "image":
            command = request["params"]["command"]
            spec = json.loads(command[4])
            spec["container"]["image"] = "sha256:" + "8" * 64
            command[4] = json.dumps(spec)
        else:
            # The exact counter digest/reference still matches the immutable
            # result; its self-asserted code digest cannot replace review.
            request["params"]["checkout_snapshot"]["input"]["sha256"] = "8" * 64
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args), _gateway=gateway) == 3
    assert not gateway.submitted


def test_missing_independent_source_contract_refuses(tmp_path, campaign, records_dir, monkeypatch):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=gateway)
    while "--pilot-source-contract" in args:
        start = args.index("--pilot-source-contract")
        del args[start:start + 3]
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args), _gateway=gateway) == 3
    assert not gateway.submitted



def test_another_output_namespace_uses_only_the_authenticated_record(
        tmp_path, campaign, records_dir, monkeypatch):
    import base64
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=gateway)
    for result in gateway.results.values():
        completion = json.loads(result["payload"])
        record = json.loads(base64.b64decode(completion["quantum_record"]))
        record["output_space"]["root"] = "/other-pilot-output/" + record["quantum_id"]
        del record["identity_sha256"]
        record["identity_sha256"] = canonical_json_sha256(record, where="pilot namespace")
        wire = json.dumps(record, indent=2).encode()
        completion.update(quantum_record=base64.b64encode(wire).decode(),
                          quantum_record_bytes=len(wire),
                          quantum_record_sha256=hashlib.sha256(wire).hexdigest())
        command = result["request"]["params"]["command"]
        command[command.index("--quantum") + 1] = "/absent-pilot-wire/" + record["quantum_id"]
        command[command.index("--quantum-sha256") + 1] = completion["quantum_record_sha256"]
        command[command.index("--output-root") + 1] = "/other-pilot-output"
        path = Path(completion["counters"]["path"])
        counters = json.loads(path.read_bytes())
        counters["identity_sha256"] = record["identity_sha256"]
        raw = json.dumps(counters).encode()
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        completion["counters"].update(sha256=digest, bytes=len(raw))
        start = args.index(str(path))
        args[start + 1] = digest
        result["payload"] = json.dumps(completion).encode()
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args), _gateway=gateway) == 0
    assert len(gateway.submitted) == 3


def test_foreign_selector_refuses(tmp_path, campaign, records_dir, monkeypatch):
    receipt = _ready(tmp_path, campaign, records_dir)
    gateway = FakeGateway()
    args = _published_pilots(tmp_path, records_dir, monkeypatch, gateway=gateway)
    args += ["--pilot-selector", "8" * 64, "1.0", "1"]
    assert main(_argv(records_dir, tmp_path / "out", receipt, extra=args), _gateway=gateway) == 3
    assert not gateway.submitted
