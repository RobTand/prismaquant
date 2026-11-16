"""Committed Tessera sources and native Flux stage records.

New dispatches keep immutable payloads in source/ and execution records in
.flux-state/. Historical PB workspaces are neither read nor modified.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

from prismaquant import tessera_campaign_rows as campaign_rows
from tools.tessera_fleet.model_worker import atomic_json

SOURCE = "source"

STATE = ".flux-state"


def stage_state(workspace, stage):
    if not stage or Path(stage).name != stage or stage in {'.', '..'}:
        raise ValueError('stage must be one path component')
    return Path(workspace) / STATE / stage


EXIT_DONE = 0
EXIT_FAILED = 1
EXIT_RUNNING = 4


def exit_status(table: list[dict], expected: int) -> int:
    statuses = [row.get("status", "") for row in table]
    if len(table) != expected or any(s not in {"succeeded", "running"}
                                     for s in statuses):
        return EXIT_FAILED
    if "running" in statuses:
        return EXIT_RUNNING
    return EXIT_DONE


def initialize_git(workspace: Path, *, message: str) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "--all", "--force"], check=True)
    subprocess.run(["git", "-C", str(workspace), "-c", "user.name=PrismaQuant",
                    "-c", "user.email=prismaquant@localhost", "commit",
                    "-qm", message], check=True)
    subprocess.run(["git", "-C", str(workspace), "checkout", "--detach", "-q"], check=True)


def run_stage(rows: list[dict], workspace: Path, stage: str, *,
              wait_s: float, afterok=()) -> list[dict]:
    """Submit or reconnect, then require native success for the whole stage.

    wait_s chooses a blocking collection or a status query. Each job has its
    own enforced finite walltime; no local scheduler or watchdog is implied.
    """
    state = stage_state(workspace, stage)
    campaign_rows.submit_rows(rows, state, afterok=afterok)
    table = campaign_rows.collect_rows(state, wait=wait_s > 0)
    atomic_json(state / 'endings.json', {'rows': table})
    done = sum(row['status'] == 'succeeded' and row['payload_exit'] == 0
               and not row['query_error'] for row in table)
    if len(table) != len(rows) or done != len(rows):
        raise ValueError(f'{stage}: {done} of {len(rows)} action(s) done; refusing downstream merge')
    return table
