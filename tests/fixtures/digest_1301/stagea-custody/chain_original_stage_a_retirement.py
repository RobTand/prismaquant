# Exact original AST-selected source; boundaries in ledger_chains.json.
# Only exercised consumer functions and required dependencies are retained.

# Original stage_a_retirement.py:58-58
from __future__ import annotations

# Original stage_a_retirement.py:61-61
import json

# Original stage_a_retirement.py:62-62
import os

# Original stage_a_retirement.py:63-63
import shutil

# Original stage_a_retirement.py:66-66
from dataclasses import dataclass, field

# Original stage_a_retirement.py:67-67
from pathlib import Path

# Original stage_a_retirement.py:70-70
from .cost_stage_checkpoint import canonical_json, canonical_json_sha256, publish_new_bytes

# Original stage_a_retirement.py:75-75
RETIREMENT_NAME = "stage-a-retired.json"

# Original stage_a_retirement.py:82-83
class RetirementRefused(RuntimeError):
    """Retirement admission or completion was refused."""

# Original stage_a_retirement.py:86-87
def retirement_record_path(space) -> Path:
    return Path(space) / RETIREMENT_NAME

# Original stage_a_retirement.py:403-410
@dataclass
class Retirement:
    space: Path
    record: dict
    entry_paths: list
    batches: list = field(default_factory=list)       # [(queue, instance, template, row)]
    untouched: list = field(default_factory=list)     # [{"batch_id", "bytes", "owner"}]
    bindings_read: int = 0

# Original stage_a_retirement.py:574-622
def apply_retirement(plan: Retirement) -> dict:
    """Seal the record, remove the files, reclaim each batch. Returns a report."""
    path = retirement_record_path(plan.space)
    payload = (json.dumps(plan.record, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()
    if not publish_new_bytes(path, payload) and path.read_bytes() != payload:
        raise RetirementRefused(f"{path} already seals another retirement")
    po = _sdk()["produced_output"]
    charged_before = {}
    for queue, instance, template, _row in plan.batches:
        key = (str(queue.root), instance["owner_action_key"], instance["template_id"])
        charged_before.setdefault(key, durable_charge(queue, instance, template))

    checkpoints = plan.space / "checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    removed_bytes = removed = 0
    for entry in plan.entry_paths:
        try:
            size = os.lstat(entry).st_size
        except FileNotFoundError:
            continue
        os.unlink(entry)
        removed += 1
        removed_bytes += size

    outcomes = {}
    for queue, instance, template, row in plan.batches:
        out = dict(po.reclaim_origin(queue, instance, template, batch_id=row["batch_id"]))
        outcomes[row["batch_id"]] = "reclaimed" if out.get("ok") else str(out.get("refusal"))
    charged_after = {}
    for queue, instance, template, _row in plan.batches:
        key = (str(queue.root), instance["owner_action_key"], instance["template_id"])
        charged_after.setdefault(key, durable_charge(queue, instance, template))
    refused = {batch: state for batch, state in outcomes.items() if state != "reclaimed"}
    report = {
        "schema": "prismaquant.stage_a.retirement_report.v1",
        "record": str(path), "record_sha256": plan.record["record_sha256"],
        "entries_removed": removed, "entry_bytes_removed": removed_bytes,
        "batches": outcomes,
        "durable_charge_before": sum(charged_before.values()),
        "durable_charge_after": sum(charged_after.values()),
        "untouched_batches": plan.untouched,
        "bindings_read": plan.bindings_read,
    }
    if refused:
        raise RetirementRefused(
            f"PrismaBuild kept {len(refused)} batch(es) charged: {refused}; report: "
            + json.dumps(report, sort_keys=True))
    return report
