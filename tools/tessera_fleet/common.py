"""Committed Tessera sources and native Flux stage records.

New dispatches keep immutable payloads in source/ and execution records in
.flux-state/. Historical PB workspaces are neither read nor modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
from typing import Mapping

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
EXIT_NOTHING = 3
EXIT_RUNNING = 4


def exit_status(table: list[dict], expected: int) -> int:
    statuses = [row.get("status", "") for row in table]
    if len(table) != expected or any(s not in {"succeeded", "running"}
                                     for s in statuses):
        return EXIT_FAILED
    if "running" in statuses:
        return EXIT_RUNNING
    return EXIT_DONE


def show_status(workspace, stage, *, collect_rows):
    """Report saved stage IDs using the caller's native collector."""
    workspace = Path(workspace)
    state = stage_state(workspace, stage)
    if not (state / 'manifest.json').exists():
        print(f"no {stage} submissions recorded under {workspace / STATE}")
        return EXIT_NOTHING
    try:
        table = collect_rows(state, wait=False)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'native status query failed: {exc}')
        return EXIT_FAILED
    expected = len(json.loads((state / 'manifest.json').read_text())['rows'])
    print(json.dumps(table, indent=2))
    counts = Counter(row.get("status", "?") for row in table)
    print(f"{stage}: {expected} recorded, "
          + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    return exit_status(table, expected)


def shard_range(of_shards: int):
    """An argparse type: ``N`` or ``LO-HI``, inclusive, within ``1..of_shards``.

    A bare ``int()`` accepted ``0``, ``500`` and ``9-4``: the first two name
    shards that do not exist and the third names nothing while reporting
    success. argparse prints an ``ArgumentTypeError`` as a usage error, so the
    domain is stated once and every wrong value gets it.
    """

    domain = f"a shard number or an inclusive LO-HI range within 1-{of_shards}"

    def parse(text: str) -> range:
        low, separator, high = text.partition("-")
        # ``1-`` is a half-typed range, not shard 1.
        if separator and not high:
            raise argparse.ArgumentTypeError(f"expected {domain}, got {text!r}")
        high = high or low
        if not (low.isdigit() and high.isdigit()):
            raise argparse.ArgumentTypeError(f"expected {domain}, got {text!r}")
        low_n, high_n = int(low), int(high)
        if not 1 <= low_n <= high_n <= of_shards:
            raise argparse.ArgumentTypeError(
                f"expected {domain}, got {text!r}: "
                f"{low_n}-{high_n} is empty or names a shard that does not exist")
        return range(low_n, high_n + 1)

    return parse


def stage_workspace(workspace: Path, files: Mapping[str, Path], *,
                    message: str) -> None:
    """Copy actual source bytes into a new committed detached shared tree."""

    workspace = Path(workspace)
    if workspace.exists():
        raise ValueError(f"workspace {workspace} already exists")
    workspace = workspace / SOURCE
    workspace.mkdir(parents=True)
    for name, source in sorted(files.items()):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"staged name escapes the workspace: {name}")
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    initialize_git(workspace, message=message)


def initialize_git(workspace: Path, *, message: str) -> None:
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "--all", "--force"], check=True)
    subprocess.run(["git", "-C", str(workspace), "-c", "user.name=PrismaQuant",
                    "-c", "user.email=prismaquant@localhost", "commit",
                    "-qm", message], check=True)
    subprocess.run(["git", "-C", str(workspace), "checkout", "--detach", "-q"], check=True)


def gb10_row(workspace: Path, argv: list, *, demand: Mapping, env: Mapping) -> dict:
    """One GPU row with the declared resources and environment."""
    return {"cwd": str(Path(workspace) / SOURCE), "argv": list(argv), "demand": dict(demand),
            "tags": ["gb10"], "env": dict(env), "timeout_s": 1800}


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


def encoder_workspace_files(checkout: Path, wrapper: Path, wrapper_name: str,
                            extra: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """The wrapper, every ``.py`` of the checkout's encoder, and ``extra``.

    The encoder tree is the checkout's ``tessera/`` directory, exactly the
    files the old PrismaBuild dispatchers bound into their code closures.
    """

    checkout = Path(checkout)
    files = {wrapper_name: Path(wrapper).resolve(strict=True)}
    for path in sorted((checkout / "tessera").rglob("*.py")):
        files[str(path.relative_to(checkout))] = path
    if len(files) == 1:
        raise ValueError(f"{checkout}/tessera holds no encoder sources")
    files.update(extra or {})
    return files


def submit_detached(rows: list[dict], workspace: Path, stage: str, *,
                    wait_s: float, afterok=()) -> list[dict]:
    """Persist every native ID before returning; resume uses exactly those IDs."""
    from dataclasses import asdict
    submissions = campaign_rows.submit_rows(rows, stage_state(workspace, stage), afterok=afterok)
    return [{**asdict(item), 'record_dir': str(item.record_dir)} for item in submissions]
