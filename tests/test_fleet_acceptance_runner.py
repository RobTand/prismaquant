"""Runner entry points behave: list, misuse refusal, scenario table."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "fleet_acceptance_runner.py"


def test_list_names_the_runnable_scenarios():
    done = subprocess.run([sys.executable, str(RUNNER), "list",
                           "--work", "/tmp/fleet-acceptance-unused",
                           "--result", "/tmp/fleet-acceptance-unused.json"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1000:]
    names = done.stdout.split()
    assert "broker-roundtrip" in names
    assert "sdk-first-release" in names
    assert "sdk-recovery-reaper" in names
    assert "sdk-pending-ticket" in names
    assert "sdk-namespace-separation" in names
    assert "sdk-failure-unwind" in names
    assert "sdk-alias-host" in names


def test_unknown_scenario_is_misuse(tmp_path):
    done = subprocess.run(
        [sys.executable, str(RUNNER), "no-such-scenario",
         "--work", str(tmp_path), "--result", str(tmp_path / "r.json")],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 2
    assert not (tmp_path / "r.json").is_file()


def test_pending_ticket_uses_the_broker_scope_and_does_not_mask_bad_input(monkeypatch):
    from types import SimpleNamespace
    import pytest
    import fleet_acceptance_runner as runner

    scope_id = "prismabuild-job" + "a" * 32 + ".slice"
    monkeypatch.setattr(runner, "_open_scope",
                        lambda *args, **kwargs: (None, {"scope_id": scope_id}))
    requests = []
    def refused(request):
        requests.append(request)
        raise ValueError("invalid container intent fields")
    world = SimpleNamespace(broker_request=refused)
    with pytest.raises(ValueError, match="invalid container intent fields"):
        runner.scenario_sdk_pending_ticket(world, {})
    assert requests == [{"op": "container_begin", "scope_id": scope_id}]


def test_pending_ticket_names_only_the_actual_creator_membership_refusal(monkeypatch):
    from types import SimpleNamespace
    import pytest
    import fleet_acceptance_runner as runner

    scope_id = "prismabuild-job" + "b" * 32 + ".slice"
    monkeypatch.setattr(runner, "_open_scope",
                        lambda *args, **kwargs: (None, {"scope_id": scope_id}))
    def outside_scope(request):
        assert request == {"op": "container_begin", "scope_id": scope_id}
        raise PermissionError("container creator is not in its scope")
    world = SimpleNamespace(broker_request=outside_scope)
    with pytest.raises(runner.pins.NonQualified) as refused:
        runner.scenario_sdk_pending_ticket(world, {})
    assert refused.value.detail["scope_id"] == scope_id
    assert refused.value.detail["refusal"] == (
        "PermissionError: container creator is not in its scope")
