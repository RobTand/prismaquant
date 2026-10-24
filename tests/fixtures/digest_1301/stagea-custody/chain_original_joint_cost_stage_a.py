# Exact original AST-selected source; boundaries in ledger_chains.json.
# Only exercised consumer functions and required dependencies are retained.

# Original joint_cost_stage_a.py:36-36
from __future__ import annotations

# Original joint_cost_stage_a.py:39-39
import json

# Original joint_cost_stage_a.py:43-43
import time

# Original joint_cost_stage_a.py:44-44
from contextlib import ExitStack, contextmanager

# Original joint_cost_stage_a.py:45-45
from pathlib import Path

# Original joint_cost_stage_a.py:2327-2348
def _produced_output_block(storage) -> dict:
    """The owner's closing staging facts, as a receipt block that seals.

    This runs after the last layer, so it may only add to the receipt. The
    receipt is sealed with ``json.dumps(allow_nan=False)``: a reason
    PrismaBuild worded with something JSON cannot carry travels as its repr,
    and a report that cannot be taken or carried is recorded as its own
    error. A finished capture is never failed by its own telemetry. An
    unbound owner adds nothing, so its receipt is unchanged.
    """

    produced = getattr(storage, "produced_output_report", None)
    if not callable(produced):
        return {}
    try:
        report = produced()
        if report is None:
            return {}
        return {"produced_output": json.loads(json.dumps(
            report, sort_keys=True, default=repr, allow_nan=False))}
    except Exception as exc:
        return {"produced_output": {"report_error": repr(exc)[:400]}}

# Original joint_cost_stage_a.py:2353-2353
FAILED_PRODUCED_OUTPUT_RECORDS = "produced-output.failures.jsonl"

# Original joint_cost_stage_a.py:2356-2389
@contextmanager
def _failed_produced_output_record(space, storage, *, path=None):
    """Keep the owner's staging records when the capture fails (PQ #1110).

    A finished capture seals them into its receipt
    (:func:`_produced_output_block`). A failed one writes no receipt, so the
    refusals, waits and evictions of its local output spool -- the export a
    barrier refused on and its state among them -- would otherwise survive
    only in the log. Runs after the owner has closed, and appends one line
    per attempt, so a later attempt never overwrites an earlier one's. A
    record that cannot be written is said once and never replaces the
    failure that is propagating.
    """

    try:
        yield
    except BaseException as error:
        try:
            block = _produced_output_block(storage)
            if block:
                line = json.dumps({
                    "schema": "prismaquant.stage_a.produced_output_failure.v1",
                    "unix": time.time(),
                    "error": f"{type(error).__name__}: {error}"[:2000],
                    **block}, sort_keys=True, default=repr, allow_nan=False)
                target = (Path(space) / FAILED_PRODUCED_OUTPUT_RECORDS if path is None
                          else Path(path))
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "a") as out:
                    out.write(line + "\n")
        except Exception as record_error:            # noqa: BLE001
            print("joint_cost_stage_a: the failed capture's produced-output "
                  f"record could not be written: {record_error!r}", flush=True)
        raise
