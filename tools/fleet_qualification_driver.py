"""Deterministic published-fleet qualification driver (level 2).

Builds the hardware-phase qualification plan from declared pins using
ONLY supported interfaces (``pbrun``/``pbtest``/``pbcampaign`` rows
with explicit per-host placement for host coverage). No scheduler, no
model-layer sharding, no live membership mutation: each row is explicit,
each membership row carries its unadvertised-API requirement, and
``--submit`` (unexecuted in this task) is the only path that touches
the fleet.

``--print-plan`` emits the exact invocations root will run after
accepted code integrates/deploys. Unit coverage stops at plan
construction; hardware rows are not executed here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ENTRYPOINTS = ("tools/pbrun.py", "tools/pbtest.py", "tools/pbcampaign.py")
HOSTS = ("sparky", "dl380g10")


def _row(*, host: str, entrypoint: str, argv: list[str],
         demand: dict, evidence: list[str], requires: str = "") -> dict:
    assert entrypoint in ENTRYPOINTS, entrypoint
    assert host in HOSTS, host
    return {"host": host, "entrypoint": entrypoint, "argv": list(argv),
            "demand": dict(demand), "evidence": list(evidence),
            "requires": requires}


def build_plan(*, pins: dict, hosts: tuple = HOSTS,
               harness_tests: tuple = (
                   "tests/test_fleet_acceptance_level1.py",
                   "tests/test_fleet_acceptance_runner.py",
                   "tests/test_fleet_acceptance_matrix.py",
               )) -> dict:
    """The full qualification plan: stages of explicit per-host rows."""
    for host in hosts:
        assert host in HOSTS, host
    demand = {"cpu": 1, "mem_gb": 8, "gpu": 0}
    stages = [
        {"name": "level1-harness",
         "rows": [_row(host=host, entrypoint="tools/pbtest.py",
                       argv=["--priority", "-10", *harness_tests],
                       demand=demand,
                       evidence=["pytest terminal", "action receipt",
                                 "CAS result"])
                  for host in hosts]},
        {"name": "connected-scenarios-both-sparks",
         "rows": [_row(
             host=host, entrypoint="tools/pbrun.py",
             argv=["run", "--tag", "cpu",
                   "--", "python3", "tests/fleet_acceptance_runner.py",
                   scenario, "--work", "{work}", "--result",
                   "{result}"],
             demand={"cpu": 2, "mem_gb": 8, "gpu": 0},
             evidence=["result JSON (qualified)", "action receipt",
                       "CAS result", "broker replies", "pin/terminal records"])
             for host in hosts
             for scenario in ("sdk-first-release", "sdk-recovery-reaper")]},
        {"name": "native-container-readers-both-sparks",
         "rows": [_row(
             host=host, entrypoint="tools/pbrun.py",
             argv=["run", "--tag", "cpu", "--image", "{image}", "--",
                   "python3", "-m", "prismaquant.tessera_campaign",
                   "--help"],
             demand={"cpu": 2, "mem_gb": 8, "gpu": 0},
             evidence=["container terminal", "action receipt"],
             requires="accepted + deployed reader SDK and image; "
                      "command finalized at acceptance time")
             for host in hosts]},
        {"name": "membership-join-resign-handoff-plan",
         "rows": [_row(
             host=host, entrypoint="tools/pbrun.py",
             argv=["run", "--tag", "cpu"],
             demand={"cpu": 1, "mem_gb": 4, "gpu": 0},
             evidence=[],
             requires="PB728 published membership API (JOIN/RESIGN/chained "
                      "handoff); no live membership mutation until root "
                      "authorizes the acceptance phase")
             for host in hosts]},
    ]
    return {"schema": "prismaquant.fleet_qualification.plan.v1",
            "pins": pins, "hosts": list(hosts), "stages": stages}


def format_invocations(plan: dict) -> list[str]:
    """The exact shell invocations for a stage row, in plan order."""
    lines = []
    for stage in plan["stages"]:
        lines.append(f"# stage: {stage['name']}")
        for row in stage["rows"]:
            if row["requires"]:
                lines.append(f"# requires: {row['requires']}")
                continue
            lines.append(
                f"python3 /mnt/shared/prismabuild-fleet/repo/{row['entrypoint']} "
                + " ".join(row["argv"]))
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", required=True,
                        help="pins JSON (record_pins output)")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    pins_doc = json.loads(Path(args.pins).read_text())
    plan = build_plan(pins=pins_doc)
    if args.out is not None:
        Path(args.out).write_text(json.dumps(plan, indent=1) + "\n")
    if args.print_plan:
        print("\n".join(format_invocations(plan)))
        return 0
    if args.submit:
        for stage in plan["stages"]:
            for row in stage["rows"]:
                if row["requires"]:
                    continue
                done = subprocess.run(
                    ["python3",
                     f"/mnt/shared/prismabuild-fleet/repo/{row['entrypoint']}"]
                    + row["argv"], check=False)
                if done.returncode != 0:
                    return done.returncode
        return 0
    parser.error("one of --print-plan or --submit is required")


if __name__ == "__main__":
    raise SystemExit(main())
