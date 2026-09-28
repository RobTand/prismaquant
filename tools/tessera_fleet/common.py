"""What the Tessera fleet drivers share: a sealed workspace and pbcampaign.

A driver never builds an action itself. It stages the bytes an action runs
into a Git workspace, writes one ``pbcampaign`` manifest row per unit of
work, and runs the published ``pbcampaign.py``. ``pbrun`` seals each row over
that workspace's snapshot, so the encoder and wrapper bytes are bound into
every action key without this module knowing how a key is made.

Every stage leaves its manifest and what ``pbcampaign`` printed under the
workspace's ``.pb-state/``, which is what ``tools.tessera_fleet.status``
reads back.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Mapping
import uuid

from prismaquant.pbwait_table import DONE_STATUSES, parse_pbwait_table

#: The published PrismaBuild client. A stale checkout must not become a stale
#: submission client, so the drivers use the generation the fleet runs.
PUBLISHED_TOOLS = Path("/mnt/shared/prismabuild-fleet/repo/tools")

#: The per-workspace directory the drivers write their own records into. The
#: workspace's Git exclude keeps it out of every sealed snapshot.
STATE = ".pb-state"


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    """Copy ``files`` (relative name to source) into a new Git workspace.

    ``pbrun`` refuses a checkout that is not a Git tree and snapshots its
    tracked and untracked bytes, so an empty commit is enough: every staged
    file travels in the snapshot. Nothing here reaches a real repository.
    """

    workspace = Path(workspace)
    if workspace.exists():
        raise ValueError(f"workspace {workspace} already exists")
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
    subprocess.run(["git", "-C", str(workspace), "-c", "user.name=PrismaQuant",
                    "-c", "user.email=prismaquant@localhost", "commit",
                    "--allow-empty", "-qm", message], check=True)
    (Path(workspace) / ".git" / "info" / "exclude").write_text(f"/{STATE}/\n")


def pbcampaign_command(manifest: Path, *, wait_s: float, detach: bool,
                       tools: Path = PUBLISHED_TOOLS) -> list[str]:
    command = [sys.executable, str(Path(tools) / "pbcampaign.py"),
               "--wait-s", str(wait_s)]
    if detach:
        command.append("--detach")
    return command + [str(manifest)]


def run_pbcampaign(rows: Iterable[dict], workspace: Path, stage: str, *,
                   wait_s: float, detach: bool = False) -> tuple[int, str]:
    """Write ``rows`` as ``<stage>-manifest.json`` and run ``pbcampaign``.

    Returns the exit status and everything it printed, which is also kept as
    ``<stage>-pbcampaign.txt``. A refusal of any row, and a failed or
    unfinished action, are a nonzero status; reading the status is the
    caller's job, because an acknowledgement is not a result.
    """

    state = Path(workspace) / STATE
    manifest = state / f"{stage}-manifest.json"
    atomic_json(manifest, list(rows))
    command = pbcampaign_command(manifest, wait_s=wait_s, detach=detach)
    print("[tessera-fleet] " + " ".join(command), flush=True)
    completed = subprocess.run(command, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (state / f"{stage}-pbcampaign.txt").write_text(completed.stdout)
    print(completed.stdout, end="", flush=True)
    return completed.returncode, completed.stdout


def detached_submissions(text: str) -> list[dict]:
    """The one JSON line per row ``pbcampaign --detach`` prints."""

    found = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("action_key"):
                found.append(record)
    return found


def run_stage(rows: list[dict], workspace: Path, stage: str, *,
              wait_s: float) -> list[dict]:
    """Run one stage to its endings, and refuse unless every row is done.

    The table is ``pbwait``'s, kept as ``<stage>-endings.json``. A stage that
    did not finish every row raises, so nothing after it -- an assembly
    behind the complete-set barrier, say -- starts on a partial set.
    """

    returncode, text = run_pbcampaign(rows, workspace, stage, wait_s=wait_s)
    table = parse_pbwait_table(text)
    atomic_json(Path(workspace) / STATE / f"{stage}-endings.json",
                {"returncode": returncode, "rows": table})
    done = [row for row in table if row.get("status") in DONE_STATUSES]
    if returncode != 0 or len(table) != len(rows) or len(done) != len(rows):
        raise ValueError(
            f"{stage}: {len(done)} of {len(rows)} action(s) done "
            f"(pbcampaign exit {returncode}); see {STATE}/{stage}-pbcampaign.txt")
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
                    wait_s: float) -> list[dict]:
    """Submit ``rows`` and return without waiting; keep the keys it printed.

    ``tools.tessera_fleet.status`` reads ``<stage>-submissions.json`` back and
    asks ``pbwait`` where each key got to.
    """

    returncode, text = run_pbcampaign(rows, workspace, stage, wait_s=wait_s,
                                      detach=True)
    submitted = detached_submissions(text)
    atomic_json(Path(workspace) / STATE / f"{stage}-submissions.json",
                {"returncode": returncode, "rows": submitted})
    if returncode != 0 or len(submitted) != len(rows):
        raise ValueError(
            f"{stage}: {len(submitted)} of {len(rows)} row(s) submitted "
            f"(pbcampaign exit {returncode}); see {STATE}/{stage}-pbcampaign.txt")
    return submitted
