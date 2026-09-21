"""ONE tiny real Stage A produced-boundary read cycle, on the live fleet.

Runs INSIDE an admitted PrismaBuild action whose request sealed a
produced-output template (``pbrun --produced-output-template``) and
declared one tiny staged input (``--data-manifest`` + ``--residency
stage``). Nothing here simulates a worker: the mover is sealed by
PrismaBuild, claimed and run by the fleet, and this process only reads,
writes, retires and asks for the batch again.

The point is the intersection a private queue fixture cannot prove -- the
real owner launch tuple, the real declaration, the real helper-root SDK
bootstrap inside the container, and the real NFS tier window. Prints one
JSON object on the last line; every identifier in it is PrismaBuild's.

Four things it establishes beyond "the bytes came back", each of them a
thing a correct tensor alone says nothing about:

* the INPUT side actually resolves. The declared tiny input is opened
  through the existing reader helper and the INJECTED map, and its bytes
  are compared to the identity the map publishes -- length and digest,
  not merely that something was returned. A staged input nobody reads
  proves nothing about staging.
* ACC07: a DECLARED PRODUCED ORIGIN is refused when it is read directly.
  Not an arbitrary undeclared file -- bytes this action wrote itself,
  that exist, whose digest it already knows and that it could simply
  open. Asserted after the write and BEFORE any publication or staging,
  through the same reader helper, because that is the only moment the
  claim means anything: strict policy forces even your own bytes through
  the staged path. (An undeclared metadata file is also refused and is
  recorded as a baseline; it is not this gate.)
* the SERVING PATH and TIER are asserted, not just the value. Where the
  bytes were served from is the whole question.
* PROGRESS is reported from durably committed group state -- the batch's
  own materialization receipt -- never from reaching a line of code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: The read phase this action declares to PrismaBuild
#: (``--progress-phase boundary-live-cycle=300``). A phase the submission
#: did not declare raises in the helper rather than being refused silently
#: by the worker.
PHASE = "boundary-live-cycle"


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _declared_entry(map_path: str) -> tuple[str, dict]:
    """The one declared input this action was given, from the MAP itself.

    Read out of the injected map rather than out of the submission's
    manifest: the map is what the reader resolves against, it is the
    document PrismaBuild actually composed for this launch, and inside a
    container it is the copy at the remapped path. No size or digest of
    the input is written into this file.
    """

    payload = json.loads(Path(map_path).read_text())
    entries = payload["entries"]
    if len(entries) != 1:
        raise AssertionError(
            "this cycle declares exactly one staged input; the map has "
            f"{len(entries)}")
    key = next(iter(entries))
    # Keys are '<offset>:<path>'; the path is everything after the first
    # colon, because an absolute path contains none before it.
    return key.split(":", 1)[1], dict(entries[key])


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queue-root", default=None,
        help=("OVERRIDE, and recorded as one; the production path does not "
              "pass it. A produced-output owner binds on its queue, and the "
              "launch signal that names the queue root is "
              "PRISMABUILD_RESIDENCY_MAP, which PrismaBuild injects for an "
              "action with STAGED INPUTS. This cycle declares one, so the "
              "root is derived exactly as a real Stage A capture derives "
              "it. The flag remains for a run with no declared input, "
              "where there is no map to derive from."))
    args = ap.parse_args()

    from prismaquant.calibration_data import _read_calibration_payload
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)
    from prismaquant.prismabuild_progress import report as report_progress
    from prismaquant.residency_map import residency_resolver
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication
    from prismaquant.staged_lease import lease_helper_root, sdk_submodule
    from prismaquant.staged_tier_policy import (
        TierPolicyRefused, activate_staged_tier_policy)

    map_path = os.environ.get("PRISMABUILD_RESIDENCY_MAP")
    report: dict[str, object] = {
        "schema": "prismaquant.stagea_produced_boundaries.live_cycle.v2",
        "owner_action_key": os.environ.get("PRISMABUILD_ACTION_KEY"),
        "helper_root": lease_helper_root(),
        "residency_map": map_path,
    }
    # The container-loading gate, asserted where it actually matters.
    roots = {name: str(Path(sdk_submodule(name).__file__).resolve().parent)
             for name in ("reader_lease", "produced_output", "pool",
                          "storage_tiers", "residency_map")}
    report["sdk_roots"] = roots
    if len(set(roots.values())) != 1:
        report["error"] = "the SDK is a MIXTURE of generations"
        print(json.dumps(report))
        return 1

    activate_staged_tier_policy("ram,ssd")

    # ---- the input side, before anything is produced --------------------
    #
    # Ordered first on purpose: if the declared input does not resolve, the
    # rest of this cycle would be a queue fixture with extra steps.
    if not map_path:
        report["error"] = ("no PRISMABUILD_RESIDENCY_MAP: this action "
                           "declared no staged input, so nothing here can "
                           "prove the injected map resolves")
        print(json.dumps(report))
        return 1
    declared, declared_entry = _declared_entry(map_path)
    report["declared_input"] = {
        "path": declared,
        "bytes": declared_entry.get("bytes"),
        "sha256": declared_entry.get("sha256"),
        "stage_path": declared_entry.get("stage_path"),
        "ram_path": declared_entry.get("ram_path"),
    }

    # BASELINE ONLY, recorded and deliberately NOT the ACC07 gate. The
    # residency map is an undeclared metadata file, so refusing it shows
    # the policy rejects arbitrary files -- which is true and is not the
    # claim that matters. ACC07's claim is stronger and is asserted below,
    # on a DECLARED PRODUCED ORIGIN this action wrote itself: bytes it
    # owns, that exist, whose digest it already knows, and that it could
    # simply open. That is what proves strict policy forces the read
    # through staging.
    baseline: dict[str, object] = {
        "kind": "baseline-only: an undeclared metadata file",
        "origin": map_path,
        "origin_readable": os.access(map_path, os.R_OK)}
    try:
        served = _read_calibration_payload(
            Path(map_path),
            expected_sha256=_digest(Path(map_path).read_bytes()))
    except TierPolicyRefused as refusal:
        # The TYPED refusal only. A bare ``except Exception`` would let an
        # ImportError, a typo or a crash read as "it refused", which is
        # the trivial pass this control exists to avoid; and the reason
        # has to be the not-staged one, not some other refusal that
        # happens to be raised on the way.
        baseline["refused"] = f"{type(refusal).__name__}: {refusal}"
        baseline["refused_ok"] = "readset-not-staged" in str(refusal)
    except Exception as other:                          # noqa: BLE001
        baseline["refused"] = f"NOT A REFUSAL: {type(other).__name__}: {other}"
        baseline["refused_ok"] = False
    else:
        baseline["refused_ok"] = False
        baseline["served_bytes"] = len(served)
    report["undeclared_metadata_baseline"] = baseline

    resolver = residency_resolver()
    report["resolver_bound"] = resolver is not None
    input_read: dict[str, object] = {}
    if resolver is not None:
        lease = resolver.lease_identity()
        input_read["lease_tier_id"] = lease.get("tier_id")
        staged = resolver.staged_read(
            declared, expected_sha256=str(declared_entry.get("sha256")))
        input_read["staged_cover"] = (
            {"stage_path": staged.get("stage_path"),
             "ram_path": staged.get("ram_path"),
             "bytes": staged.get("bytes")} if staged else None)
        try:
            raw = _read_calibration_payload(
                Path(declared),
                expected_sha256=str(declared_entry.get("sha256")))
        except Exception as refusal:                    # noqa: BLE001
            input_read["error"] = f"{type(refusal).__name__}: {refusal}"
        else:
            input_read["read_bytes"] = len(raw)
            input_read["read_sha256"] = _digest(raw)
        # Where it was served from, in the reader's own record. The row's
        # keys are the ones ``residency_map.record_serving_tier`` writes --
        # ``path`` (normalized with os.path.normpath, never resolved) and
        # ``serving_tier`` -- read from that function rather than guessed;
        # filtering on a key it does not write would make the check below
        # pass vacuously by matching nothing.
        want = os.path.normpath(declared)
        input_read["serving_rows"] = [
            row for row in resolver.report().get("serving_tiers", [])
            if os.path.normpath(str(row.get("path", ""))) == want]
    report["declared_input_read"] = input_read

    report["queue_root_source"] = ("explicit override (an operator override; "
                                   "not the production path)"
                                   if args.queue_root else
                                   "derived from PRISMABUILD_RESIDENCY_MAP")
    publication = BoundaryProducedPublication.bind_from_admitted_owner(
        queue_root=args.queue_root)
    report["tier"] = publication.tier
    report["output_prefix"] = str(publication.output_prefix)
    report["durable_maxima"] = publication.durable_maxima()
    admit = publication.admit_window()
    report["admit_window"] = admit
    if not admit.get("ok"):
        report["error"] = "the bounded window was not admitted"
        print(json.dumps(report))
        return 1

    group_size = 4
    entries = Path(publication.output_prefix) / "live-cycle"
    storage = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(entries),
        "max_resident_bytes": 1 << 22, "max_auxiliary_bytes": 1 << 22,
        "max_artifact_bytes": 1 << 22, "prefetch_batches": group_size})
    committed = 0
    with storage:
        storage.bind({"source_model": "live-cycle"}, n_probes=1)
        storage.bind_produced_output(
            publication, group_size=group_size, n_batches=group_size,
            max_entry_tensor_bytes=1 << 14, staging_timeout_s=600.0)
        wanted = [torch.arange(8, dtype=torch.float32) + index
                  for index in range(group_size)]
        references = [storage.write(tensor, batch_index=index,
                                    boundary_index=0)
                      for index, tensor in enumerate(wanted)]
        report["prewritten"] = storage.telemetry["produced_groups_prewritten"]
        report["published_before_first_read"] = storage.telemetry[
            "produced_groups_published"]

        # ---- ACC07, at the only moment it means anything ----------------
        #
        # The origin exists RIGHT NOW: storage.write created it under the
        # prewrite authorization and the reference carries the writer's own
        # inline digest. Nothing has been published, staged or
        # materialized yet. So this is a file this action owns, whose
        # bytes are there and whose digest it already knows -- the exact
        # case where opening it directly would be easiest and would be
        # wrong. Under the strict policy it must refuse, and refuse with
        # the not-staged reason rather than any other.
        #
        # The origin is described from METADATA ONLY: a stat and an access
        # check. Reading it here to "confirm" it, or hashing it, would be
        # the very read the policy forbids.
        probe = references[0]
        acc07: dict[str, object] = {
            "gate": "ACC07: a declared produced origin is not readable "
                    "directly before it is staged",
            "origin_path": str(probe.path),
            "origin_bytes": os.stat(probe.path).st_size,
            "reference_file_bytes": int(probe.file_bytes),
            "origin_readable_by_this_process": os.access(probe.path, os.R_OK),
            "inline_sha256": str(probe.sha256),
            "published_yet": storage.telemetry["produced_groups_published"],
        }
        try:
            _read_calibration_payload(Path(probe.path),
                                      expected_sha256=str(probe.sha256))
        except TierPolicyRefused as refusal:
            acc07["refused"] = f"{type(refusal).__name__}: {refusal}"
            acc07["refused_ok"] = "readset-not-staged" in str(refusal)
        except Exception as other:                      # noqa: BLE001
            acc07["refused"] = (
                f"NOT A REFUSAL: {type(other).__name__}: {other}")
            acc07["refused_ok"] = False
        else:
            acc07["refused_ok"] = False
            acc07["served"] = "the origin was READ directly; strict policy "
            "did not force the read through staging"
        report["produced_origin_refusal"] = acc07

        with storage.prefetch(references) as window:
            for reference, tensor in zip(references, wanted):
                if not torch.equal(storage.get(window, reference), tensor):
                    report["error"] = "first read returned different bytes"
                    print(json.dumps(report))
                    return 1
        batch_id = storage.produced_group_records()[0]["batch_id"]
        report["batch_id"] = batch_id
        first = publication.materialization_state(batch_id=batch_id)
        report["after_first_read"] = first
        report["charge_after_first_read"] = publication.durable_charge()

        # PROGRESS, from committed state and nothing else. The unit is a
        # group whose batch PrismaBuild's OWN record says is committed and
        # whose mover receipt is complete; reaching this line is not the
        # evidence, the record is. Cumulative, as the contract requires.
        if first.get("mover_receipt_complete") is True:
            committed += 1
            report["progress_reported"] = report_progress(
                PHASE, committed, unit="boundary-groups")
        report["committed_groups_after_first_read"] = committed

        # The repeat read ALSO disposes its entries while the window is
        # live -- the production pattern (the tail retires inside the read
        # window, the reverse roll retires the previous cotangent as it
        # writes the next). A cycle that disposes only afterwards has the
        # same blind spot the unit suite had.
        with storage.prefetch(references) as window:
            for reference, tensor in zip(references, wanted):
                if not torch.equal(storage.get(window, reference), tensor):
                    report["error"] = "repeat read returned different bytes"
                    print(json.dumps(report))
                    return 1
            # OBSERVED INSIDE THE WINDOW, BEFORE THE DISPOSAL BELOW. The
            # charge is constant across the cycle -- one logical batch,
            # one durable origin charge -- and that claim is only true
            # while the origins are still there. Read after the in-window
            # disposal it is zero, and asserting both "equal to the first
            # charge" and "zero" of the same number is the contradiction
            # this report used to carry.
            report["charge_in_repeat_window"] = publication.durable_charge()
            for reference in references:
                storage.retire(reference)
            report["in_window_disposal"] = True
        second = publication.materialization_state(batch_id=batch_id)
        report["after_repeat_read"] = second
        report["charge_after_repeat_read"] = publication.durable_charge()
        report["telemetry"] = dict(storage.telemetry)
        report["release_debt"] = storage.produced_release_debt()
        report["group_records"] = storage.produced_group_records()

        report["group_records_after_disposal"] = storage.produced_group_records()
        report["charge_after_disposal"] = publication.durable_charge()
    report["release_instance"] = publication.release()

    # ASSERTED, not printed. A cycle that observes without asserting
    # "passes" on any result, including the ones this run exists to catch.
    checks = {
        "one_sdk_generation": len(set(roots.values())) == 1,
        "declared_input_has_a_staged_cover":
            bool(input_read.get("staged_cover")),
        "declared_input_bytes_and_digest_match_the_map":
            input_read.get("read_bytes") == declared_entry.get("bytes")
            and input_read.get("read_sha256") == declared_entry.get("sha256"),
        "declared_input_was_served_by_a_named_tier":
            bool(input_read.get("serving_rows"))
            and all(str(row.get("serving_tier")) in ("stage", "ram")
                    and bool(row.get("pin_id"))
                    for row in input_read.get("serving_rows") or []),
        "a_declared_produced_origin_is_refused_before_staging":
            acc07.get("refused_ok") is True,
        "the_same_reference_then_read_through_staging":
            acc07.get("origin_bytes", -1) >= 0
            and storage.telemetry["produced_groups_materialized"] >= 1,
        "stage_copy_was_retired_on_window_exit":
            first.get("stage_retired") is True,
        "repeat_read_used_a_fresh_materialization":
            int(second.get("generation") or 0) == 1,
        "successor_mover_is_a_different_row":
            str(second.get("mover_key")) != str(first.get("mover_key")),
        "same_logical_batch":
            str(second.get("manifest_digest")) == str(first.get("manifest_digest"))
            and str(second.get("batch_namespace")) == str(first.get("batch_namespace")),
        "one_publication_only":
            storage.telemetry["produced_groups_published"] == 1,
        "origin_charge_constant_while_the_origins_are_there":
            report["charge_in_repeat_window"] == report["charge_after_first_read"],
        "progress_came_from_a_committed_group": committed == 1,
        "no_release_debt":
            report["release_debt"] == {"pending": {}, "abandoned": {},
                                       "unclassified": {},
                                       "publish_deferred": {}},
        "final_origin_disposal_reclaimed_the_charge":
            report["charge_after_disposal"] == {"payload": 0, "checkpoint": 0,
                                                "temp": 0},
        "instance_released":
            bool(report["release_instance"].get("ok")),
    }
    report["checks"] = checks
    report["failed_checks"] = sorted(k for k, ok in checks.items() if not ok)
    report["ok"] = not report["failed_checks"]
    print(json.dumps(report))
    if report["failed_checks"]:
        raise AssertionError(
            "live produced-boundary cycle failed: "
            + ", ".join(report["failed_checks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
