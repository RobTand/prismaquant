# Exact original AST-selected source; boundaries in ledger_chains.json.
# Only exercised consumer functions and required dependencies are retained.

# Original stage_a_chain_resume.py:63-63
from __future__ import annotations

# Original stage_a_chain_resume.py:66-66
import json

# Original stage_a_chain_resume.py:67-67
import os

# Original stage_a_chain_resume.py:68-68
from pathlib import Path

# Original stage_a_chain_resume.py:71-71
from .cost_stage_checkpoint import canonical_json, canonical_json_sha256, publish_new_bytes

# Original stage_a_chain_resume.py:79-79
RESUME_RECORD_SCHEMA = "prismaquant.stage_a.chain_resume.v1"

# Original stage_a_chain_resume.py:96-97
class ChainResumeRefused(RuntimeError):
    """The relaunch is not a continuation of the run it names."""

# Original stage_a_chain_resume.py:104-105
def resume_directory(space) -> Path:
    return Path(space) / "resumes"

# Original stage_a_chain_resume.py:542-573
def apply_chain_resume(space, plan: ChainResume, *, producer=None, split=None) -> dict:
    """Remove the interrupted attempt's working entries and seal the resume record.

    Runs after every check. Returns the record it sealed. ``split`` (a
    chain split prep, PQ #738) is stamped into the record as ``split``; a
    record without it keeps its bytes.
    """
    for path in plan.leftovers:
        path.unlink(missing_ok=True)
    set_aside = []
    for directory in plan.partials:
        target = directory.with_name(f"{directory.name}.partial-resume-{plan.index:03d}")
        os.rename(directory, target)
        set_aside.append(target.name)
    body = canonical_json({
        "schema": RESUME_RECORD_SCHEMA,
        "index": plan.index,
        "session": plan.session,
        "switch_checkpoint": plan.boundary,
        "implementation_sha256": plan.implementation_sha256,
        "compatibility": plan.compatibility,
        "producer": producer,
        "removed_rolling_entries": len(plan.leftovers),
        "partial_checkpoints_set_aside": set_aside,
        **({"split": split} if split is not None else {}),
    }, where="chain resume record")
    record = {**body, "record_sha256": canonical_json_sha256(body, where="chain resume record")}
    path = resume_directory(space) / f"resume-{plan.index:03d}.json"
    payload = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if not publish_new_bytes(path, payload):
        raise ChainResumeRefused(f"{path} already exists: another relaunch resumed this run")
    return record
