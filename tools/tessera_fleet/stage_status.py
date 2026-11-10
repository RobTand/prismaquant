"""Native stage state paths and status presentation, independent of row submission."""
from collections import Counter
import json
from pathlib import Path

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
