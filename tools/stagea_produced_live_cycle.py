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


def _chain_cycle(publication, report, report_progress, *,
                 window_compute_s: float = 0.0) -> int:
    """A miniature Stage A over real staging: forward, tail, two-layer roll.

    Two boundary planes and 4 probe planes of 2 groups each, in the order
    the real capture uses: each boundary plane is written and read back once
    (the forward pass), the incoming cotangent planes are written (the
    tail), and each layer's roll reads its input boundary in all 4 probe
    passes while it replaces every cotangent entry it reads. What is
    asserted is the staging the read-ahead rules promise: one staging of a
    boundary group per roll, no group staged twice in a roll, no release
    debt, and the durable charge back to zero.
    """

    import time

    import torch

    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)

    group_size, groups, n_probes, layers = 4, 2, 4, 2
    n_batches = group_size * groups
    storage = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA,
        "directory": str(Path(publication.output_prefix) / "chain-cycle"),
        "max_resident_bytes": 1 << 22, "max_auxiliary_bytes": 1 << 22,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": group_size})

    def boundary_value(layer, batch):
        return torch.arange(8, dtype=torch.float32) + 1000 * layer + batch

    def windows(references):
        for start in range(0, n_batches, group_size):
            yield start, references[start:start + group_size]

    marks: dict[str, float] = {}
    started = time.monotonic()
    with storage:
        storage.bind({"source_model": "live-chain-cycle"}, n_probes=n_probes)
        storage.bind_produced_output(
            publication, group_size=group_size, n_batches=n_batches,
            max_entry_tensor_bytes=1 << 14, staging_timeout_s=600.0)
        report["window_groups"] = storage._produced_plan["window_groups"]
        report["window_compute_s"] = window_compute_s

        # Forward: write each plane, then read it back once, as the next
        # layer's forward does. The read leaves the plane retired.
        boundary = {}
        for layer in range(layers):
            boundary[layer] = [
                storage.write(boundary_value(layer, batch), batch_index=batch,
                              boundary_index=layer)
                for batch in range(n_batches)]
            for start, refs in windows(boundary[layer]):
                with storage.prefetch(refs) as window:
                    for offset, reference in enumerate(refs):
                        if not torch.equal(
                                storage.get(window, reference),
                                boundary_value(layer, start + offset)):
                            report["error"] = "forward read returned other bytes"
                            print(json.dumps(report))
                            return 1
        marks["forward_s"] = time.monotonic() - started
        after_forward = dict(storage.telemetry)

        # Tail: the incoming cotangent planes at the top boundary.
        cotangent = {probe: [
            storage.write(torch.full((8,), float(10 * probe + 1)) + batch,
                          batch_index=batch, boundary_index=layers,
                          probe_index=probe)
            for batch in range(n_batches)] for probe in range(n_probes)}
        storage.stage_produced_boundary_ahead(layers - 1)
        marks["tail_s"] = time.monotonic() - started

        # Chain: probe outer, window inner, the input boundary re-read in
        # every pass, each cotangent entry replaced as it is read.
        per_layer = []
        for layer in reversed(range(layers)):
            before = dict(storage.telemetry)
            layer_started = time.monotonic()
            if layer > 0:
                storage.stage_produced_boundary_ahead(layer - 1)
            with storage.retain_produced_boundary(layer):
                for probe in range(n_probes):
                    for start, refs in windows(boundary[layer]):
                        incoming = cotangent[probe][start:start + group_size]
                        with storage.prefetch(list(refs) + list(incoming)) as window:
                            time.sleep(window_compute_s)
                            for offset, (b_ref, c_ref) in enumerate(
                                    zip(refs, incoming)):
                                x = storage.get(window, b_ref)
                                g = storage.get(window, c_ref)
                                if not torch.equal(
                                        x, boundary_value(layer, start + offset)):
                                    report["error"] = "chain read returned other bytes"
                                    print(json.dumps(report))
                                    return 1
                                # The walk's last roll is read by nothing,
                                # exactly as the capture declares it.
                                cotangent[probe][start + offset] = storage.write(
                                    g + 1.0, batch_index=start + offset,
                                    boundary_index=layer, probe_index=probe,
                                    previous=c_ref, read_back=layer > 0)
            after = dict(storage.telemetry)
            per_layer.append({
                "layer": layer,
                "seconds": round(time.monotonic() - layer_started, 3),
                **{name: round(after[name] - before[name], 3) for name in (
                    "produced_groups_materialized",
                    "produced_groups_rematerialized",
                    "produced_groups_retired",
                    "produced_groups_published_ahead",
                    "produced_groups_staged_ahead",
                    "produced_group_stage_wait_s",
                    "produced_group_release_wait_s",
                    "produced_group_ahead_wait_s",
                    "produced_group_credit_waits")}})
        marks["chain_s"] = time.monotonic() - started
        report["progress_reported"] = report_progress(
            PHASE, layers, unit="chain-layers")

        # The capture's own order: the last roll's entries are disposed of
        # FIRST, and the stage credit is settled after that, as the owner
        # closes. Settling first would hide a mover racing the disposal.
        for probe in range(n_probes):
            for reference in cotangent[probe]:
                storage.retire(reference)
        marks["last_roll_disposed_s"] = time.monotonic() - started
        storage.settle_produced_releases()
        marks["settled_s"] = time.monotonic() - started
        report["release_debt"] = storage.produced_release_debt()
        report["ahead_refusals"] = storage.produced_ahead_refusals()
        report["release_errors"] = [
            {**entry, "reason": repr(entry.get("reason"))[:600]}
            for entry in storage._produced_release_errors[-24:]]
        report["charge_before_disposal"] = publication.durable_charge()
        for reference in list(storage._references.values()):
            storage.retire(reference)
        report["charge_after_disposal"] = publication.durable_charge()
        report["telemetry"] = dict(storage.telemetry)
        report["group_records"] = storage.produced_group_records()
    report["release_instance"] = publication.release()
    report["per_layer"] = per_layer
    report["marks_s"] = {name: round(value, 3) for name, value in marks.items()}

    telemetry = report["telemetry"]
    groups_read_per_layer = groups + n_probes * groups
    checks = {
        "the_window_funds_read_ahead": report["window_groups"] > 2,
        "every_group_was_published_when_its_last_entry_landed":
            telemetry["produced_groups_published_ahead"]
            == telemetry["produced_groups_published"],
        "the_last_roll_was_never_staged":
            telemetry["produced_groups_published"]
            == layers * groups + n_probes * groups * layers,
        "read_ahead_was_never_refused_or_surrendered":
            telemetry["produced_group_ahead_refusals"] == 0
            and telemetry["produced_groups_ahead_surrendered"] == 0,
        "the_forward_read_each_boundary_group_once":
            after_forward["produced_groups_materialized"] == layers * groups,
        "a_roll_stages_each_group_it_reads_exactly_once": all(
            row["produced_groups_materialized"] == groups_read_per_layer
            for row in per_layer),
        # A re-staging is counted where it is asked for, so the lower
        # plane's count lands in the roll above it: across the chain each
        # plane is staged again once, never once per probe pass.
        "a_boundary_plane_is_restaged_once_per_roll_not_once_per_pass":
            sum(row["produced_groups_rematerialized"] for row in per_layer)
            == layers * groups,
        "the_lower_plane_was_staged_ahead_of_its_roll":
            per_layer[-1]["produced_groups_rematerialized"] == 0
            and telemetry["produced_groups_staged_ahead"] >= groups,
        "no_release_debt": report["release_debt"] == {
            "pending": {}, "abandoned": {}, "unclassified": {},
            "publish_deferred": {}},
        "every_published_group_retired": all(
            record["retired"] for record in report["group_records"]
            if record["manifest_digest"]),
        "durable_charge_reclaimed_to_zero":
            report["charge_after_disposal"] == {
                "payload": 0, "checkpoint": 0, "temp": 0},
        "instance_retained_only_by_its_live_owner":
            report["release_instance"].get("ok") is False
            and report["release_instance"].get("refusal")
            == "owner-active-retain",
    }
    report["checks"] = checks
    report["failed_checks"] = sorted(
        name for name, passed in checks.items() if not passed)
    report["ok"] = not report["failed_checks"]
    print(json.dumps(report))
    return 0 if report["ok"] else 1


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
    ap.add_argument(
        "--data-manifest-sha256", required=True,
        help=("sha256 of the submitted data manifest's wire bytes, passed "
              "by the launcher exactly as the dispatcher passes it to a real "
              "Stage A capture. The resolver refuses every map until the "
              "process names the manifest it was submitted with, so without "
              "it the declared input is refused readset-not-staged. Taken "
              "from the launcher and never from the map, which would make "
              "the binding agree with whatever map was injected."))
    ap.add_argument(
        "--mode", choices=("one-token", "chain"), default="one-token",
        help=("one-token is the single-group cycle. chain is a miniature "
              "forward, tail and two-layer reverse roll over several groups "
              "and 4 probe passes, for the read-ahead rules "
              "(RobTand/prismaquant#887): it needs a template whose window "
              "funds about 24 groups."))
    ap.add_argument(
        "--window-compute-s", type=float, default=0.0,
        help=("chain mode: seconds each read window spends on stand-in "
              "compute. Read-ahead overlaps staging with compute, so with "
              "none there is nothing for it to overlap."))
    args = ap.parse_args()

    from prismaquant.calibration_data import _read_calibration_payload
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)
    from prismaquant.prismabuild_progress import report as report_progress
    from prismaquant.residency_map import (
        bind_residency_manifest, residency_resolver)
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
    # The same call, from the same launcher-supplied flag, that a real
    # capture makes (joint_cost_stage_a.run_adjoint_capture).
    bind_residency_manifest(args.data_manifest_sha256)
    report["data_manifest_sha256"] = args.data_manifest_sha256

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

    if args.mode == "chain":
        return _chain_cycle(publication, report, report_progress,
                            window_compute_s=args.window_compute_s)

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
        # NOT "released". This runs inside the owner, and PrismaBuild's
        # contract is that any live owner claim retains the instance: the
        # release is authorized only after the claim ends, by the broker's
        # containment certificate. ``safe_release_instance`` checks the
        # owner LAST -- after active movers, funding intents and the pin
        # census -- so this exact refusal says every other reason to retain
        # is clear. ``ok`` here would mean PrismaBuild freed an instance
        # under its live owner, and any other refusal names a real leftover.
        "instance_retained_only_by_its_live_owner":
            report["release_instance"].get("ok") is False
            and report["release_instance"].get("refusal")
            == "owner-active-retain",
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
