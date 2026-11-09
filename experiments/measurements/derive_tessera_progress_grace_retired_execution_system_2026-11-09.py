#!/usr/bin/env python3
"""Derive Tessera pricing watchdog evidence from retained PB terminal records.

The GLM campaign's completed rows retain both the terminal wall time and the
campaign's durable-batch log lines.  This tool makes the selection and linear
fit explicit so a phase allowance is reviewable rather than a number copied
from a PR body.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


DEFAULT_ROOT = Path(
    "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
    "first-proof-anchor-preparation-05/root-action-records-current.json"
)
DEFAULT_TERMINALS = Path("/mnt/shared/prismabuild-fleet/pb-queue/done")
FINAL = re.compile(r"cost\.pkl: (\d+) units, (\d+) priced rungs")
BATCH = re.compile(r"^\[campaign\] r\d+ \d+/\d+ batch=", re.MULTILINE)
ROW = re.compile(r"workspace/rows/(row-\d+)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(rows: list[dict[str, object]]) -> dict[str, float]:
    n = len(rows)
    xs = [float(row["committed_batches"]) for row in rows]
    ys = [float(row["elapsed_s"]) for row in rows]
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise RuntimeError("selected rows have no batch-count variation")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    return {
        "s_per_committed_batch": slope,
        "nonpricing_intercept_s": intercept,
        "max_abs_residual_s": max(abs(value) for value in residuals),
    }


def derive(root_path: Path, terminals: Path) -> dict[str, object]:
    root = json.loads(root_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for action in root["actions"]:
        if action.get("state") != "done":
            continue
        key = str(action["action_key"])
        terminal = terminals / f"{key}.json"
        if not terminal.is_file():
            continue
        payload = json.loads(terminal.read_text(encoding="utf-8"))
        detail = payload.get("detail") or {}
        stdout = detail.get("stdout")
        elapsed = detail.get("elapsed_s")
        if not isinstance(stdout, str) or not isinstance(elapsed, (int, float)):
            continue
        final = FINAL.search(stdout)
        if final is None or final.groups() != ("864", "2592"):
            continue
        row = ROW.search(stdout)
        if row is None:
            raise RuntimeError(f"{key}: completed GLM row has no row identity")
        rows.append({
            "row_id": row.group(1),
            "action_key": key,
            "terminal_sha256": sha256(terminal),
            "committed_batches": len(BATCH.findall(stdout)),
            "elapsed_s": float(elapsed),
        })
    rows.sort(key=lambda item: str(item["row_id"]))
    if len(rows) != 23:
        raise RuntimeError(f"expected 23 completed 864-unit/2592-rung rows, found {len(rows)}")
    return {
        "schema": "prismaquant.tessera_progress_grace_fit.v1",
        "root_action_records": str(root_path),
        "root_action_records_sha256": sha256(root_path),
        "terminal_directory": str(terminals),
        "selection": "successful rows with final '864 units, 2592 priced rungs'",
        "rows": rows,
        "fit": fit(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-action-records", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--terminals", type=Path, default=DEFAULT_TERMINALS)
    args = parser.parse_args()
    print(json.dumps(derive(args.root_action_records, args.terminals), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
