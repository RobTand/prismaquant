from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sys

import pytest

import tools.wait_dsv4_aura_campaign as waiter
from prismaquant.dsv4_campaign_completion import (
    AURA_CHECKPOINT_IDENTITY_SCHEMA,
    AURA_CHECKPOINT_MANIFEST_SCHEMA,
    AURA_CHECKPOINT_UNIT_SCHEMA,
    CampaignCompletionError,
    CompletionContract,
    DEFAULT_SERVICE_UNIT,
    PRODUCER_SCHEMA,
    SERVICE_SCHEMA,
    STREAMED_PAYLOAD_SCHEMA,
    audit_campaign_closure,
    build_completion_receipt,
    canonical_sha256,
    load_completion_receipt,
    publish_completion_receipt,
    validate_terminal_service_evidence,
    verify_receipt_against_current_campaign,
    verify_receipt_for_replay,
    wait_for_bound_systemd_service,
)


TEST_CONTRACT = CompletionContract(
    expected_unit_count=4,
    overlap_layers=(42,),
    units_per_overlap_layer=2,
)


def _write_checkpoint(
    root: Path,
    qnames: list[str],
    *,
    written_qnames: set[str],
    payload_suffix: bytes = b"",
) -> None:
    identity = {
        "schema": AURA_CHECKPOINT_IDENTITY_SCHEMA,
        "git_commit": "1" * 40,
        "scope": qnames,
    }
    identity_sha256 = canonical_sha256(identity)
    rows = []
    (root / "units").mkdir(parents=True)
    for qname in qnames:
        filename = hashlib.sha256(qname.encode()).hexdigest() + ".pkl"
        rows.append({"qname": qname, "file": f"units/{filename}"})
        if qname not in written_qnames:
            continue
        state = {"rows": {"FP8_CB_K32": {"cost": qname}}}
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        if payload_suffix:
            state["suffix"] = payload_suffix
            payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        envelope = {
            "schema": AURA_CHECKPOINT_UNIT_SCHEMA,
            "qname": qname,
            "identity_sha256": identity_sha256,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload": payload,
        }
        (root / "units" / filename).write_bytes(
            pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        )
    manifest = {
        "schema": AURA_CHECKPOINT_MANIFEST_SCHEMA,
        "identity": identity,
        "identity_sha256": identity_sha256,
        "units": rows,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    qnames = [
        "model.layers.42.mlp.experts.0.gate_proj",
        "model.layers.42.mlp.experts.0.up_proj",
        "model.layers.0.self_attn.wq_a",
        "model.layers.0.self_attn.wo_b",
    ]
    work = tmp_path / "aura-cb-reprice-streamed-cached"
    current = work / "checkpoints" / "aura"
    historical = tmp_path / "aura-cb-reprice" / "checkpoints" / "aura"
    _write_checkpoint(current, qnames, written_qnames=set(qnames))
    _write_checkpoint(
        historical,
        qnames,
        written_qnames={name for name in qnames if ".layers.42." in name},
    )
    artifact = work / "artifacts" / "streamed_anchor_aura.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(pickle.dumps({
        "schema": STREAMED_PAYLOAD_SCHEMA,
        "costs": {name: {"FP8_CB_K32": 1.0} for name in qnames},
        "stats": {name: {"n_params": 1} for name in qnames},
        "provenance": {"test": True},
    }, protocol=pickle.HIGHEST_PROTOCOL))
    return work, historical, qnames


def _service(*, result: str = "success") -> dict[str, object]:
    return {
        "schema": SERVICE_SCHEMA,
        "unit": DEFAULT_SERVICE_UNIT,
        "object_path": "/org/freedesktop/systemd1/unit/test",
        "invocation_id": "2" * 32,
        "initial_invocation_id": "2" * 32,
        "main_pid": 4321,
        "main_pid_start_ticks": 9876,
        "initial_active_state": "active",
        "initial_sub_state": "running",
        "restart_policy": "no",
        "terminal_active_state": "inactive",
        "terminal_sub_state": "dead",
        "result": result,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "exec_main_pid": 4321,
        "n_restarts": 0,
    }


def _producer(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": PRODUCER_SCHEMA,
        "snapshot": str((tmp_path / "runtime-snapshot").absolute()),
        "commit": "3" * 40,
        "tree": "4" * 40,
        "closure_sha256": "5" * 64,
    }


def test_completion_receipt_closes_campaign_and_historical_overlap(tmp_path):
    work, historical, _qnames = _fixture(tmp_path)
    assert len(list((historical / "units").iterdir())) == 2
    campaign = audit_campaign_closure(
        work, historical, contract=TEST_CONTRACT,
    )
    assert campaign["unit_count"] == 4
    assert campaign["historical_overlap"]["exact_payload_match_count"] == 2
    receipt = build_completion_receipt(
        service=_service(),
        producer=_producer(tmp_path),
        campaign=campaign,
        contract=TEST_CONTRACT,
    )
    destination = work / "checkpoints" / "aura" / "completion.json"
    publish_completion_receipt(
        destination, receipt, contract=TEST_CONTRACT,
    )
    assert load_completion_receipt(
        destination, contract=TEST_CONTRACT,
    ) == receipt
    assert verify_receipt_against_current_campaign(
        destination,
        work_dir=work,
        historical_root=historical,
        expected_producer_commit="3" * 40,
        contract=TEST_CONTRACT,
    ) == receipt
    assert verify_receipt_for_replay(
        destination,
        work_dir=work,
        historical_root=historical,
        expected_producer_commit="3" * 40,
        contract=TEST_CONTRACT,
    ) == receipt


def test_completion_receipt_refuses_current_unit_tamper(tmp_path):
    work, historical, qnames = _fixture(tmp_path)
    campaign = audit_campaign_closure(
        work, historical, contract=TEST_CONTRACT,
    )
    receipt = build_completion_receipt(
        service=_service(), producer=_producer(tmp_path), campaign=campaign,
        contract=TEST_CONTRACT,
    )
    destination = work / "checkpoints" / "aura" / "completion.json"
    publish_completion_receipt(
        destination, receipt, contract=TEST_CONTRACT,
    )
    unit = hashlib.sha256(qnames[-1].encode()).hexdigest() + ".pkl"
    (work / "checkpoints" / "aura" / "units" / unit).write_bytes(b"tampered")
    with pytest.raises(CampaignCompletionError):
        verify_receipt_against_current_campaign(
            destination,
            work_dir=work,
            historical_root=historical,
            contract=TEST_CONTRACT,
        )


def test_completion_audit_refuses_extra_current_unit_file(tmp_path):
    work, historical, _qnames = _fixture(tmp_path)
    (work / "checkpoints" / "aura" / "units" / "unmanifested.pkl").write_bytes(
        b"extra"
    )
    with pytest.raises(CampaignCompletionError, match="exactly the manifest"):
        audit_campaign_closure(work, historical, contract=TEST_CONTRACT)


def test_completion_audit_refuses_historical_overlap_mismatch(tmp_path):
    work, historical, qnames = _fixture(tmp_path)
    overlap = qnames[0]
    filename = hashlib.sha256(overlap.encode()).hexdigest() + ".pkl"
    path = historical / "units" / filename
    envelope = pickle.loads(path.read_bytes())
    payload = pickle.dumps(
        {"rows": {"FP8_CB_K32": {"cost": "different"}}},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    envelope["payload"] = payload
    envelope["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    with pytest.raises(CampaignCompletionError, match="historical payload differs"):
        audit_campaign_closure(work, historical, contract=TEST_CONTRACT)


def test_completion_receipt_refuses_failed_service_and_self_hash_tamper(tmp_path):
    work, historical, _qnames = _fixture(tmp_path)
    with pytest.raises(CampaignCompletionError, match="result"):
        validate_terminal_service_evidence(_service(result="exit-code"))
    restarted = _service()
    restarted["n_restarts"] = 1
    with pytest.raises(CampaignCompletionError, match="n_restarts"):
        validate_terminal_service_evidence(restarted)
    missing = _service()
    missing.pop("exec_main_status")
    with pytest.raises(CampaignCompletionError, match="exec_main_status"):
        validate_terminal_service_evidence(missing)
    boolean_code = _service()
    boolean_code["exec_main_code"] = True
    with pytest.raises(CampaignCompletionError, match="exec_main_code"):
        validate_terminal_service_evidence(boolean_code)
    for invalid_start_ticks in (0, True):
        bad_start = _service()
        bad_start["main_pid_start_ticks"] = invalid_start_ticks
        with pytest.raises(CampaignCompletionError, match="start time"):
            validate_terminal_service_evidence(bad_start)
    campaign = audit_campaign_closure(
        work, historical, contract=TEST_CONTRACT,
    )
    receipt = build_completion_receipt(
        service=_service(), producer=_producer(tmp_path), campaign=campaign,
        contract=TEST_CONTRACT,
    )
    destination = work / "checkpoints" / "aura" / "completion.json"
    publish_completion_receipt(
        destination, receipt, contract=TEST_CONTRACT,
    )
    raw = json.loads(destination.read_text(encoding="utf-8"))
    raw["service"]["main_pid_start_ticks"] += 1
    destination.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(CampaignCompletionError, match="self-hash"):
        load_completion_receipt(destination, contract=TEST_CONTRACT)


def test_completion_receipt_is_no_clobber(tmp_path):
    work, historical, _qnames = _fixture(tmp_path)
    campaign = audit_campaign_closure(
        work, historical, contract=TEST_CONTRACT,
    )
    receipt = build_completion_receipt(
        service=_service(), producer=_producer(tmp_path), campaign=campaign,
        contract=TEST_CONTRACT,
    )
    destination = work / "checkpoints" / "aura" / "completion.json"
    publish_completion_receipt(
        destination, receipt, contract=TEST_CONTRACT,
    )
    with pytest.raises(CampaignCompletionError, match="will not be replaced"):
        publish_completion_receipt(
            destination, receipt, contract=TEST_CONTRACT,
        )


@pytest.mark.parametrize("occupied", ("file", "symlink"))
def test_completion_stage_refusal_preserves_unowned_entry(tmp_path, occupied):
    import prismaquant.dsv4_campaign_completion as completion

    destination = tmp_path / "completion.json"
    stage = tmp_path / f".completion.json.{completion.os.getpid()}.tmp"
    original = tmp_path / "other-writer"
    original.write_bytes(b"other writer's incomplete receipt")
    if occupied == "symlink":
        stage.symlink_to(original.name)
    else:
        stage.write_bytes(original.read_bytes())
    before = stage.lstat()

    with pytest.raises(FileExistsError):
        completion._atomic_publish_new(destination, b"new receipt")

    assert stage.lstat() == before
    assert stage.read_bytes() == original.read_bytes()
    assert stage.is_symlink() == (occupied == "symlink")
    assert not destination.exists()


@pytest.mark.parametrize("outcome", ("success", "link_error", "concurrent_winner"))
def test_completion_owned_stage_cleanup_preserves_publication(tmp_path, monkeypatch, outcome):
    import prismaquant.dsv4_campaign_completion as completion

    destination = tmp_path / "completion.json"
    stage = tmp_path / f".completion.json.{completion.os.getpid()}.tmp"
    failure = OSError("injected link failure")

    def fail_link(source, target):
        assert source == stage and target == destination
        assert source.read_bytes() == b"new receipt"
        if outcome == "concurrent_winner":
            target.write_bytes(b"winner receipt")
            raise FileExistsError("injected concurrent winner")
        raise failure

    if outcome != "success":
        monkeypatch.setattr(completion.os, "link", fail_link)
    if outcome == "success":
        completion._atomic_publish_new(destination, b"new receipt")
        assert destination.read_bytes() == b"new receipt"
        assert destination.stat().st_mode & 0o777 == 0o600
    elif outcome == "concurrent_winner":
        with pytest.raises(CampaignCompletionError, match="appeared concurrently"):
            completion._atomic_publish_new(destination, b"new receipt")
        assert destination.read_bytes() == b"winner receipt"
    else:
        with pytest.raises(OSError) as observed:
            completion._atomic_publish_new(destination, b"new receipt")
        assert observed.value is failure
        assert not destination.exists()
    assert not stage.exists()


def test_outer_waiter_refuses_dirty_release_tree(monkeypatch):
    repo = Path(waiter.__file__).resolve().parent.parent
    calls = []

    def fake_git(observed_repo, *args):
        calls.append(args)
        assert observed_repo == repo
        if args == ("rev-parse", "--show-toplevel"):
            return str(repo)
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return " M prismaquant/dsv4_campaign_completion.py"
        raise AssertionError(args)

    monkeypatch.setattr(waiter, "_git", fake_git)
    with pytest.raises(waiter.WaiterError, match="must be clean"):
        waiter._outer_reexec(["wait"])
    assert calls[-1] == ("status", "--porcelain", "--untracked-files=all")


def test_waiter_loads_completion_logic_without_package_startup(monkeypatch):
    repo = Path(waiter.__file__).resolve().parent.parent
    closure = "6" * 64
    module_name = f"_prismaquant_release_dsv4_campaign_completion_{closure}"
    monkeypatch.setitem(sys.modules, "prismaquant", None)
    try:
        completion = waiter._load_completion_module({
            "snapshot": str(repo),
            "closure_sha256": closure,
        })
        assert Path(completion.__file__).resolve() == (
            repo / "prismaquant" / "dsv4_campaign_completion.py"
        )
        assert completion.PRODUCER_SCHEMA == PRODUCER_SCHEMA
    finally:
        sys.modules.pop(module_name, None)


def test_systemd_waiter_refs_before_capture_and_always_cleans_up(monkeypatch):
    import prismaquant.dsv4_campaign_completion as completion

    calls = []

    class Manager:
        def Subscribe(self):
            calls.append("subscribe")

        def RefUnit(self, unit):
            assert unit == DEFAULT_SERVICE_UNIT
            calls.append("ref")

        def UnrefUnit(self, unit):
            assert unit == DEFAULT_SERVICE_UNIT
            calls.append("unref")

        def Unsubscribe(self):
            calls.append("unsubscribe")

    manager = Manager()

    class Bus:
        def get_object(self, service, path):
            assert service == "org.freedesktop.systemd1"
            assert path == "/org/freedesktop/systemd1"
            calls.append("manager-object")
            return manager

    class DBus:
        @staticmethod
        def SessionBus():
            calls.append("session-bus")
            return Bus()

        @staticmethod
        def Interface(value, interface):
            assert value is manager
            assert interface == "org.freedesktop.systemd1.Manager"
            calls.append("manager-interface")
            return manager

    def loop_init(*, set_as_default):
        assert set_as_default is True
        calls.append("mainloop-init")

    monkeypatch.setattr(
        completion,
        "_systemd_dbus_dependencies",
        lambda: (DBus, loop_init, object()),
    )

    def capture(**_kwargs):
        calls.append("capture")
        raise CampaignCompletionError("terminal validation failed")

    monkeypatch.setattr(
        completion, "_wait_for_referenced_systemd_service", capture,
    )
    with pytest.raises(CampaignCompletionError, match="terminal validation"):
        wait_for_bound_systemd_service(timeout_seconds=1)
    assert calls == [
        "mainloop-init", "session-bus", "manager-object", "manager-interface",
        "subscribe", "ref", "capture", "unref", "unsubscribe",
    ]
