"""ONE tiny real Stage A produced-boundary read cycle, on the live fleet.

Runs INSIDE an admitted PrismaBuild action whose request sealed a
produced-output template (``pbrun --produced-output-template``). Nothing
here simulates a worker: the mover is sealed by PrismaBuild, claimed and
run by the fleet, and this process only writes, reads, retires and asks
for the batch again.

The point is the intersection a private queue fixture cannot prove -- the
real owner launch tuple, the real declaration, the real helper-root SDK
bootstrap inside the container, and the real NFS tier window. Prints one
JSON object on the last line; every identifier in it is PrismaBuild's.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queue-root", default=None,
        help=("STOPGAP, and recorded as one. A produced-output owner binds "
              "on its queue, and the only launch signal that names the "
              "queue root is PRISMABUILD_RESIDENCY_MAP -- which PrismaBuild "
              "injects for an action with STAGED INPUTS. A real Stage A "
              "capture has them (it runs --residency stage against its "
              "calibration manifest) and derives the root; this tiny cycle "
              "reads nothing, so it has no residency map and must be told. "
              "Passing it is an override, never the production path."))
    args = ap.parse_args()

    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication
    from prismaquant.staged_lease import lease_helper_root, sdk_submodule
    from prismaquant.staged_tier_policy import activate_staged_tier_policy

    report: dict[str, object] = {
        "schema": "prismaquant.stagea_produced_boundaries.live_cycle.v1",
        "owner_action_key": os.environ.get("PRISMABUILD_ACTION_KEY"),
        "helper_root": lease_helper_root(),
        "residency_map": os.environ.get("PRISMABUILD_RESIDENCY_MAP"),
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

    report["queue_root_source"] = ("explicit override (no staged inputs, so "
                                   "no residency map to derive from)"
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
    activate_staged_tier_policy("ram,ssd")
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
        "origin_charge_constant_across_the_cycle":
            report["charge_after_repeat_read"] == report["charge_after_first_read"],
        "no_release_debt":
            report["release_debt"] == {"pending": {}, "abandoned": {}},
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
