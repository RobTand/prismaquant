"""Standalone pin-scope documentation, independent of the retained GPU package."""

import json
import re
import runpy
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs/measurements/pq1317-qualification-pin-scope.md"
HISTORICAL = "fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb"
OBSERVED_PIN = "aaa417003f0ee690e8ace138acd18df7f9531369"
OBSERVED_PQ = "6d8288adc94b3c4b624c143ed372fb6513f32784"
PIN_PATH = "prismaquant/tessera_runtime/tessera_serving_runtime_pin.json"
RESOLVER = "tools/resolve_tessera_dev_pin.py"
CONTRACT_SOURCE = "prismaquant/tessera_runtime_contract.py"


def _note():
    assert NOTE.is_file(), "Publish the standalone historical-candidate/current-pin scope note"
    return NOTE.read_text(encoding="utf-8")


def test_historical_scope_and_three_operations_are_documented():
    text = _note()
    lower = text.lower()
    # Fixed historical observations, not a timeless assertion of the latest pin.
    for fact in (
        HISTORICAL,
        OBSERVED_PIN,
        OBSERVED_PQ,
        "ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e",
        "candidate.json",
        "roster.json",
        "requalification.json",
        "4e5380f573dff3319fd687a26b82678eaead7dcd",
        PIN_PATH,
        RESOLVER,
        "--source-url",
    ):
        assert fact in text, f"Missing scope fact or evidence locator: {fact}"
    # These are navigation/claim anchors; the reviewer also checks their meaning.
    for concept in (
        r"observed|as of|at prismaquant",
        r"contract",
        r"source",
        r"test",
        r"native-v2",
        r"\bv3\b",
        r"cpu",
        r"offline",
        r"frozen",
        r"versioned",
        r"affected",
        r"fresh",
        r"receipt",
        r"relabel",
        r"component",
        r"whole.model",
    ):
        assert re.search(concept, lower), f"Missing qualification boundary: {concept}"
    assert "NOT qualified by this evidence" in text
    assert "[Test evidence " in text, "Retain run_tests receipts for the command and docs checks"


def _pin_command():
    text = _note()
    commands = [
        body
        for body in re.findall(r"```(?:sh|bash)\s*\n(.*?)\n```", text, re.S)
        if HISTORICAL in body and "resolve_tessera_dev_pin" in body
    ]
    assert len(commands) == 1, "Provide one runnable sh/bash pin-comparison block"
    command = commands[0]
    for forbidden in (
        "/home/", "/mnt/", "/tmp/", "pq1317-gpu-tests/", "git show",
        "git log", "git archive", "pbrun", "pbtest", "pbcampaign",
    ):
        assert forbidden not in command, f"Comparison must be standalone: {forbidden}"
    return command


def _run(command, cwd):
    result = subprocess.run(
        ["bash", "-eu", "-c", command], cwd=cwd, capture_output=True, text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_documented_command_reads_live_pins_without_qualification_package(tmp_path):
    command = _pin_command()
    serving = json.loads((ROOT / PIN_PATH).read_text(encoding="utf-8"))["commit"]
    development = runpy.run_path(str(ROOT / RESOLVER))["resolve_tessera_dev_pin"]()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    output = _run(command, ROOT)
    for value in (serving, development, revision, HISTORICAL):
        assert value in output, f"Comparison did not display {value}: {output}"
    if serving != HISTORICAL or development != HISTORICAL:
        assert "NOT qualified by this evidence" in output

    # Relocate only the existing pin inputs/resolver into a minimal checkout.
    # No historical Git objects, application imports, Tessera, GPU, or package.
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for relative in (RESOLVER, CONTRACT_SOURCE, PIN_PATH):
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=Acceptance", "-c", "user.email=acceptance@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "pin fixture"],
        cwd=checkout, check=True, capture_output=True,
    )
    # Exercise both inputs independently: reading only one must not suffice.
    for dev_pin, serving_pin in (("1" * 40, "2" * 40), ("3" * 40, "4" * 40)):
        source = checkout / CONTRACT_SOURCE
        source.write_text(
            re.sub(rf'(?m)^(TESSERA_DEV_PIN_COMMIT\s*=\s*)[\"\'][0-9a-f]{{40}}[\"\']',
                   rf'\g<1>"{dev_pin}"', source.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        pin = json.loads((checkout / PIN_PATH).read_text(encoding="utf-8"))
        pin["commit"] = serving_pin
        (checkout / PIN_PATH).write_text(json.dumps(pin), encoding="utf-8")
        output = _run(command, checkout)
        for value in (HISTORICAL, dev_pin, serving_pin, "NOT qualified by this evidence"):
            assert value in output, f"Comparison ignored a live binding: {output}"
