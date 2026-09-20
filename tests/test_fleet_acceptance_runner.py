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
