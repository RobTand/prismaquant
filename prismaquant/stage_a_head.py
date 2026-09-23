"""Stage A's head, taken from the prepared completion (PQ #1051).

Stage A (``joint_cost_stage_a.run_adjoint_capture``) needs four things from
its head:

* the unit roster, whose digest names the campaign in the run identity;
* the unit and measured-cell counts the result records;
* the unit count its PrismaBuild progress counter starts from.

The prepared completion (``prepared.json``) holds all of them. The prepare
action wrote it after walking the same plan-bound inputs. Stage A binds the
completion by digest (``--prepared-sha256``) and refuses one prepared under
another plan.

Before #1051, Stage A walked the whole anchor catalog again to re-derive them
(``tessera_joint_aura.load_measured_anchor_input``). On GLM-5.3 (R12) the walk
took about 78 minutes, and its reads were 36,428 of the head phase's 36,441
data-manifest entries (8.10 of 10.97 GB) that the render-free capture never
uses: the five plan inputs it binds and the merged checkpoint's 36,423 unit
parts. Stage A now reads the completion instead, for every root: fresh,
forward recovery, chain resume and seed.

What each walk check becomes:

* **The census model and eager attention.** The prepare checked both on the
  same plan-bound census. Stage A still compares the completion's source
  model identity with the model it builds, and builds it with eager
  attention.
* **The calibration draw against the anchor payload.** The prepare checked
  it. Stage A requires the completion's calibration record to equal the one
  it loads (:func:`prepared_head`), as the Stage B head does.
* **The roster and the cell count.** They come from the completion, which its
  digest binds.
* **The encoder seal and the existing renders.** Not checked. Stage A reads
  no render; Stage B checks both.

The walk remains a verification arm (``run_adjoint_capture(head_walk=True)``):
it runs the old checks, then requires the walk's roster and cell count to
equal the completion's (:func:`walked_head`). The two arms produce the same
head record except for its ``walked`` flag.

A Stage A data manifest declares none of the walk's reads
(:func:`drop_head_walk_reads`), so PrismaBuild stages none of them.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .tessera_joint_aura import _require, _same, head_walk_read_set, is_head_walk_read

STAGE_A_HEAD_SCHEMA = "prismaquant.stage_a.head.v1"


@dataclass(frozen=True)
class StageAHead:
    """What Stage A takes from its head, and the record its receipt carries."""

    formats_by_qname: dict
    measured_cells: int
    progress_units: int
    record: dict

    @property
    def units(self) -> int:
        return len(self.formats_by_qname)


def _record(prepared, *, walked, units, measured_cells) -> dict:
    return {"schema": STAGE_A_HEAD_SCHEMA, "walked": bool(walked),
            "prepared_completion": {"path": str(prepared["path"]),
                                    "sha256": str(prepared["sha256"])},
            "units": int(units), "measured_cells": int(measured_cells)}


def prepared_head(completion, *, prepared, calibration) -> StageAHead:
    """The head from a prepared completion that already passed its preflight.

    ``completion`` is the parsed completion that
    ``tessera_joint_aura.check_prepared_completion`` returned for the bytes
    ``prepared`` pins; the caller has also compared its ``plan_sha256``.
    ``calibration`` is the calibration record Stage A loaded, and must equal
    the completion's.
    """
    _same(completion.get("calibration_input"), calibration, "prepared calibration")
    formats = completion.get("formats_by_qname")
    _require(isinstance(formats, dict) and formats,
             "the prepared completion holds no candidate roster")
    cells = completion.get("measured_cells")
    _require(type(cells) is int and cells >= 0,
             "the prepared completion holds no measured-cell count")
    formats = {str(name): tuple(fmts) for name, fmts in formats.items()}
    return StageAHead(formats, cells, len(formats),
                      _record(prepared, walked=False, units=len(formats),
                              measured_cells=cells))


def walked_head(data, completion, *, prepared, calibration) -> StageAHead:
    """The verification arm's head: the walk's, and it must be the completion's.

    ``data`` is the ``MeasuredAnchorInput`` the walk returned. The walk
    committed its own progress units, so the head continues from its count.
    """
    head = prepared_head(completion, prepared=prepared, calibration=calibration)
    _same({name: list(fmts) for name, fmts in head.formats_by_qname.items()},
          {name: list(fmts) for name, fmts in data.formats_by_qname.items()},
          "prepared exact candidate roster")
    _same(head.measured_cells, len(data.cells), "prepared measured cells")
    return StageAHead(head.formats_by_qname, head.measured_cells,
                      int(data.progress_committed),
                      {**head.record, "walked": True})


def stage_a_roster(formats_by_qname, *, capsule, plan_sha256, prepared_sha256,
                   read_manifest_sha256, calibration_shape, campaign_scope):
    """``(unit_roster_sha256, campaign_scope)`` for Stage A's run identity.

    Without a capsule, the roster digest is the newline-joined sorted names
    and the scope is the plan's. With one (a forward recovery or a seed's),
    both are the campaign the capsule publishes, resolved against this
    roster by ``resolve_forward_campaign``: it refuses a campaign record
    whose bytes do not match the capsule's digest, and a campaign under
    another plan, preparation, read manifest, roster or calibration shape.
    """
    roster = hashlib.sha256("".join(
        f"{name}\n" for name in sorted(formats_by_qname)).encode()).hexdigest()
    if capsule is None:
        return roster, campaign_scope
    from .joint_forward_campaign import resolve_forward_campaign
    from .joint_forward_resume import _read

    document, _ = _read(capsule["path"], capsule["sha256"])
    campaign = resolve_forward_campaign(
        document, plan_sha256=plan_sha256, prepared_sha256=prepared_sha256,
        read_manifest_sha256=read_manifest_sha256, formats_by_qname=formats_by_qname,
        calibration_shape=list(calibration_shape))
    return campaign["unit_roster_sha256"], campaign["campaign_scope"]


def drop_head_walk_reads(manifest, inputs) -> tuple[dict, dict]:
    """A copy of a Stage A data manifest whose ``head`` phase omits the walk's reads.

    ``inputs`` is the plan's ``inputs`` block. An entry leaves the head when
    its path is one of the walk's bound inputs or sits under the merged
    checkpoint's parts directory (``head_walk_read_set``). An entry no phase
    references any more is dropped, the remaining indices are renumbered, and
    every byte count is recomputed. Returns ``(manifest, dropped)``, where
    ``dropped`` counts the entries and bytes removed from the head.
    """
    read_set = head_walk_read_set(inputs)
    manifest = copy.deepcopy(manifest)
    entries = manifest["entries"]
    phases = manifest["read_plan"]["phases"]
    heads = [phase for phase in phases if phase["name"] == "head"]
    _require(len(heads) == 1, "a Stage A data manifest has exactly one head phase")
    head = heads[0]
    kept, removed = [], []
    for index in head["entry_indices"]:
        (removed if is_head_walk_read(entries[index]["path"], read_set)
         else kept).append(index)
    head["entry_indices"] = kept
    used = sorted({index for phase in phases for index in phase["entry_indices"]})
    remap = {old: new for new, old in enumerate(used)}
    manifest["entries"] = [entries[index] for index in used]
    cumulative = 0
    for phase in phases:
        phase["entry_indices"] = [remap[index] for index in phase["entry_indices"]]
        phase["bytes"] = sum(manifest["entries"][index]["bytes"]
                             for index in phase["entry_indices"])
        cumulative += phase["bytes"]
        phase["cumulative_bytes"] = cumulative
    manifest["read_plan"]["read_bytes"] = cumulative
    manifest["entry_count"] = len(manifest["entries"])
    manifest["total_bytes"] = sum(entry["bytes"] for entry in manifest["entries"])
    dropped = {"entries": len(removed),
               "bytes": sum(entries[index]["bytes"] for index in removed)}
    return manifest, dropped


def drop_source_head_walk_reads(manifest, plan) -> tuple[dict, dict]:
    """:func:`drop_head_walk_reads` for a source run's manifest, under its pinned plan.

    The package builders derive a seed's or a forward recovery's manifest
    from a source run's submitted one, which a manifest built before #1051
    still carries the walk's reads in. ``plan`` is the pinned ``{path,
    sha256}`` of the plan the source manifest names. Refuses (``ValueError``)
    a plan whose bytes are not its digest, and a manifest whose
    ``annotations.plan_sha256`` names another plan: its ``inputs`` would name
    another walk's reads.
    """
    raw = Path(plan["path"]).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == plan["sha256"],
             f"the plan {plan['path']} does not have the pinned digest")
    _require((manifest.get("annotations") or {}).get("plan_sha256") == plan["sha256"],
             "the source manifest was built for another plan")
    return drop_head_walk_reads(manifest, json.loads(raw)["inputs"])
