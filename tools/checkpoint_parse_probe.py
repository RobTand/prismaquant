#!/usr/bin/env python3
"""Measure the joint checkpoint's real reader: RSS, sharing, seal and loader.

``json.loads(path.read_text())`` holds the decoded text and the object graph at
the same time. On the 7,198,621,019-byte merged checkpoint that is a 20.73 GiB
peak and 14.03 GiB retained, of which 12.39 GiB is ``identity.units``:
205,243,544 menu-name occurrences that are 5,635 distinct strings because every
unit's menu was a fresh list of fresh ``str`` objects
(``joint-aura-resume/ckpt-subtree-probe-0{1,2}.txt``). The reader this probe
measures is the production one -- ``interned_json.load_json_file``, the stdlib
decoder plus the sharing ``object_pairs_hook`` -- so the numbers are the
loader's, not a prototype's.

``--phase`` decides what a run measures, and each phase is its own process so
the reported peak belongs to that phase alone:

* ``parse`` -- parse the checkpoint, report current and peak RSS, count distinct
  menu ``str`` objects, and recompute the canonical identity digest. A digest
  that does not equal ``--expect-digest`` **refuses** with a nonzero exit,
  because a boolean in a log nobody compares is not an acceptance check.
* ``loader`` -- run the real ``load_measured_anchor_input`` over the plan's
  frozen inputs with ``verify_payloads=False`` and
  ``require_existing_renders=True`` -- metadata only, no synthesis, no wire
  read, no campaign write -- in a process that holds no other graph, so the
  peak is the loader's own, including its parse and the metadata beyond it. A
  refusal is reported as a refusal with the working set it reached, and exits 2.
* ``both`` -- parse and seal, then **drop the parsed graph**, then load. The
  free is explicit and drops both names: ``identity`` is an alias into
  ``manifest``, so dropping only ``manifest`` leaves the whole identity block
  alive and the loader phase would report the sum of two graphs. Because
  ``VmHWM`` never falls, the loader row's ``peak_rss_gib`` in this mode is the
  larger of the two phases; use ``--phase loader`` alone for the loader's own
  peak.

Recorded runs (dl380g10, 2026-09-16, receipts under
``joint-aura-resume/parser-repair/``): ``--phase parse`` 181 s, 13.812 GiB
peak, 2.116 GiB retained, digest exact; ``--phase loader`` 1,093.75 s, 36,423
units, 197,990 cells, 0 synthesized, VmHWM 15.830 GiB inside the 21 GiB CPU
envelope.

Read-only: it opens the checkpoint and the plan, and writes nothing.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from pathlib import Path

PHASES = ("both", "parse", "loader")


def _gib(field: str) -> float:
    from prismaquant.io_spans import read_proc_status

    value = read_proc_status().get(field.rstrip(":"))
    if value is None:
        raise RuntimeError(f"no {field} in /proc/self/status")
    return round(value / 1024 ** 3, 3)


def _report(step: str, **extra) -> None:
    row = {"step": step, "rss_gib": _gib("VmRSS:"), "peak_rss_gib": _gib("VmHWM:")}
    row.update(extra)
    print(json.dumps(row, sort_keys=True), flush=True)


def menu_sample(identity: dict, units: int) -> tuple[int, int]:
    """Occurrences and distinct ``str`` objects in the first ``units`` menus."""
    records = identity.get("units") or {}
    occurrences, objects = 0, set()
    for record in itertools.islice(iter(records.values()), units):
        for item in record.get("menu") or ():
            occurrences += 1
            objects.add(id(item))
    return occurrences, len(objects)


def _load_plan(plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        raise SystemExit(f"{plan_path}: no resolved 'inputs' block to load")
    return inputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expect-digest", default=None,
                        help="the recorded identity seal a parse phase must "
                             "reproduce; required by --phase parse and both")
    parser.add_argument("--phase", choices=PHASES, default="both")
    parser.add_argument("--sample-units", type=int, default=200,
                        help="units whose menu strings are counted; 0 skips")
    parser.add_argument("--loader-plan", default=None,
                        help="resolved joint plan whose 'inputs' are handed to "
                             "load_measured_anchor_input; required by --phase "
                             "loader and both")
    args = parser.parse_args(argv)

    phase_parses = args.phase in ("parse", "both")
    phase_loads = args.phase in ("loader", "both")
    if phase_parses and not args.expect_digest:
        parser.error("--expect-digest is required by --phase parse and both")
    if phase_loads and not args.loader_plan:
        parser.error("--loader-plan is required by --phase loader and both")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256_normalized
    from prismaquant.interned_json import load_json_file

    checkpoint = Path(args.checkpoint)
    _report("start", phase=args.phase, checkpoint=str(checkpoint),
            byte_size=checkpoint.stat().st_size, loader_plan=bool(args.loader_plan))

    manifest = identity = None
    if phase_parses:
        stats: dict = {}
        started = time.time()
        manifest = load_json_file(checkpoint, stats=stats)
        _report("parse", elapsed_s=round(time.time() - started, 2), **stats)

        identity = manifest["identity"]
        if args.sample_units:
            occurrences, objects = menu_sample(identity, args.sample_units)
            print(json.dumps({"step": "menu.sample",
                              "units": args.sample_units, "occurrences": occurrences,
                              "distinct_menu_strings": objects,
                              "objects_per_occurrence": round(objects / max(occurrences, 1), 6)},
                             sort_keys=True), flush=True)

        started = time.time()
        digest = canonical_json_sha256_normalized(identity, where="checkpoint identity")
        _report("seal", elapsed_s=round(time.time() - started, 2),
                identity_sha256=digest, expected_sha256=args.expect_digest,
                matches_recorded=digest == args.expect_digest)
        if digest != args.expect_digest:
            raise SystemExit(
                f"checkpoint identity digest changed: {digest} != {args.expect_digest}. "
                "The seal the loader would compute does not match the recorded identity.")

    if phase_loads:
        if phase_parses:
            # The loader re-parses the same file. `identity` is an alias into
            # `manifest`, so both names have to go or the whole identity block
            # stays alive under the loader and the peak reports two graphs.
            del identity, manifest
            gc.collect()
            _report("parse.graph_dropped")

        plan_path = Path(args.loader_plan)
        inputs = _load_plan(plan_path)
        _report("loader.plan", inputs=sorted(inputs))
        from prismaquant.tessera_joint_aura import load_measured_anchor_input

        started = time.time()
        try:
            data = load_measured_anchor_input(
                inputs, verify_payloads=False, require_existing_renders=True,
                log_every=0, progress_phase=None)
        except Exception as exc:
            _report("loader.refused", elapsed_s=round(time.time() - started, 2),
                    refused_by=type(exc).__name__, reason=str(exc)[:600])
            return 2
        _report("loader.metadata_intake", elapsed_s=round(time.time() - started, 2),
                units=len(data.formats_by_qname), cells=len(data.cells),
                synthesized_now=data.synthesized_now)
        del data
        gc.collect()
        _report("loader.graph_dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
