#!/usr/bin/env python3
"""Fan a Tessera anchor campaign out across the fleet, one row per anchor group.

``prismaquant.tessera_campaign`` prices a rate surface per (unit, family), and
every one of those surfaces is independent work: its anchors, its
leave-one-out gate and its refusal are its own.  Run as one process it is a
single exclusive GPU action, and the second box sits idle beside it
(RobTand/prismaquant#282).  This lays the same campaign out as N PrismaBuild
rows and puts the pieces back together.

The quantum is the **fused anchor group**, not the bare unit.  Anchors are
placed per group: the group shares one rung grid, and the group's worst member
drives every split (``tessera_campaign`` round loop).  A group is therefore the
smallest scope whose measured values do not depend on what else the run priced,
which is what makes the merged table equal to the monolith's rather than merely
similar to it.

Four steps, and each one is separately re-runnable:

``census``
    One cheap GPU row: a calibration forward over the whole scope that counts
    each unit's rows and reports the anchor grouping.  The counts are what let
    every later row stamp the **scope's** ``fit_tokens`` rather than its own
    selection's, so the sharded table carries one Hessian identity -- the same
    one a whole-scope run carries.  The grouping is what lets ``plan`` lay out
    rows without loading the model.

``plan``
    One ``--units`` selection file per row and one pbcampaign manifest.  Rows
    are portable (no host pin), not exclusive, GPU-demanding, and carry a
    memory demand derived from the phase plan the row will check itself
    against -- the plan's bytes, the process floor measured on this fleet, and
    the margin the row's own guard holds back from its cap.

``check``
    The same derivation, run against a manifest that already exists, refusing
    any row whose declared ``demand.mem_gb`` is below it or whose derived
    demand is wider than a GPU box.  ``submit`` runs it first, so a row is
    never queued for an admission that its own guard will decline
    (RobTand/prismaquant#522).

``submit``
    ``pbcampaign`` over that manifest.  Re-running it **is** the resume: a
    finished row is a CAS hit that runs nothing and a running row is attached
    to by its job id, so there is no second dispatcher here deciding what to
    skip.

``merge``
    One ``cost.pkl``, one ``cost.anchors.json`` and one export-inputs cache
    from the rows, refusing on any identity the rows do not already share.  The
    merged Hessian capture is the union of the rows' -- the same H under the
    same counts and the same provenance a whole-scope run writes -- so its
    digest is recomputed rather than asserted, and every row's
    ``capture_sha256`` is re-stamped to it.

Seeding from a campaign already in flight
-----------------------------------------
``--seed-checkpoint`` on a planned row hands the monolith's stored anchors to
that row's own gates: the producer input identity is recomputed from the row's
weights, menu, Hessian and static scale, and the cached wire is re-verified
against it, so a seeded row adopts only bytes it would itself have encoded.
The adaptive state needs nothing else -- ``grid``, the leave-one-out error and
the stop reason are all recomputed from the anchor set at the top of every
round -- so adopting the anchors resumes the group exactly where it stood.

Rows planned before the progress contract
-----------------------------------------
Nothing migrates in place.  A row already in ``ready/`` was sealed with its
own ``execution_timeout_s``; that request is immutable and stays exactly as
it is, still bounded by the number it was submitted with.  The stall policy
is sealed too, so a plan made after it produces different action keys, and a
key that has never been priced is not a cache hit.  What carries the work
across is the journal rather than the queue: the new row resumes from the
same identity-bound ``cost_stage_checkpoint`` directory (or is handed the
monolith's anchors with ``--seed-workspace`` / ``--seed-checkpoint``), so the
anchors already committed under the old key are re-adopted rather than
re-measured.  Withdraw the old rows once the new ones are running; do not
edit them.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path

if __package__:
    from .tessera_campaign_container import (
        CONTAINER_IMAGE_FLAG,
        DEV_MODE_ENV,
        admission_image_reference,
        container_memory_budget_gb,
        validate_container,
    )
else:
    # Direct script execution puts only tools/ on sys.path. Planning also
    # reads the shared calibration contract from the sibling package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tessera_campaign_container import (
        CONTAINER_IMAGE_FLAG,
        DEV_MODE_ENV,
        admission_image_reference,
        container_memory_budget_gb,
        validate_container,
    )

PBCAMPAIGN = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbcampaign.py")

#: What ``plan`` writes beside the manifest, so ``merge`` reads the row layout
#: from the plan rather than from the directory listing it happens to find.
PLAN_SCHEMA = "prismaquant.tessera_campaign_plan.v1"

#: The plan key that declares a census scope deliberately left partly unpriced:
#: the rows the plan dropped and why.  ``merge`` reads only this declaration
#: to expect less than the scope -- never which row directories happen to hold
#: a ``cost.pkl`` -- and writes what it read into the merged table under
#: ``provenance.coverage``.  A plan without it claims the whole scope.
DENSE_ROWS_EXCLUDED_KEY = "dense_rows_excluded"
COVERAGE_SCHEMA = "prismaquant.tessera_campaign_coverage.v1"

#: Provenance fields every row must agree on before a merge is possible.  Each
#: one describes the run, not the selection, so a disagreement means two
#: campaigns are being merged into one table.
SHARED_PROVENANCE = (
    "menu_mode", "tp_degree", "model", "nsamples", "seqlen", "max_act_rows",
    "layer_stride", "anchors_round_one", "max_rounds", "anchor_budget",
    "loo_gate", "max_artifact_bpp", "cost_mode", "rate_band", "calibration_cache",
)

#: Hessian identity fields every row must already agree on.  ``capture_sha256``
#: is deliberately absent: it is the digest of the capture a row wrote for its
#: own units, and reconciling it is what the merge is for.
SHARED_HESSIAN = (
    "supplied", "text_sha", "token_count", "text_sha256", "fit_ids_sha256",
    "fit_tokens", "kwarg", "reference_binding",
)


#: Checkpoint-identity keys the merge RECONCILES instead of requiring equal.
#: Every one of them is per-selection: it describes the units this row was
#: given, not the campaign. Everything outside this set must already agree,
#: because a difference there means the rows priced two different campaigns.
RECONCILED_IDENTITY_KEYS = frozenset({
    "units", "serving_scope", "expert_projection", "stack_sampling_identity",
    "family_restriction",
})


class MergeRefused(RuntimeError):
    """The rows do not describe one campaign."""


class RowClassRefused(RuntimeError):
    """The spec asks for a row this fleet cannot place, or must not place."""


# ---------------------------------------------------------------------------
# Row classes
# ---------------------------------------------------------------------------

#: The class every subcommand builds today, and the one a spec that declares no
#: ``classes`` block is entirely made of.  Naming it does not change a row: the
#: default class resolves to exactly the spec-level ``python`` / ``env`` /
#: ``tags`` / ``cpus`` / ``container`` a row has always been built from, so the
#: row dict -- and therefore the action key a finished row is memoized under --
#: is byte-identical to what the same spec produced before classes existed.
DEFAULT_ROW_CLASS = "default"

#: What a class may declare.  ``python``/``env``/``tags``/``cpus``/``container``
#: override the spec-level value of the same name; ``env`` merges over the
#: spec's rather than replacing it, so a class states only what differs.
ROW_CLASS_FIELDS = frozenset({
    "python", "env", "tags", "cpus", "container",
    "wire_shared", "weights_only", "_why",
})

#: Where the placement facts live: ISA, container GPU runtime and the host
#: interpreters actually observed running work under each tag, each with the
#: PrismaBuild action key that ran it.
FLEET_INTERPRETERS = Path(__file__).resolve().parent / "fleet_interpreters.json"
FLEET_INTERPRETERS_SCHEMA = "prismaquant.fleet_interpreters.v1"

#: Campaign flags that make a row's bytes depend on a Hessian.  A weights-only
#: class may not carry one: the Hessian-aware wire is NOT bit-comparable across
#: ISAs (RobTand/tessera#472 -- ``torch.linalg.cholesky`` diverges between
#: cuSOLVER and rocSOLVER by one fp32 epsilon, ``ldl`` explains 53.78% of the
#: differing rendered elements and the encoder's own consumption of it a
#: further 15.28%), so such a row encoded off-ISA would not be this campaign's
#: bytes.  ``--seed-checkpoint`` / ``--seed-wire-dir`` are here for the same
#: reason: they adopt anchors another box measured under its own Hessian.
HESSIAN_AWARE_FLAGS = (
    "--calibration-census", "--calibration-cache", "--calibration-cache-sha256",
    "--capture-calibration-out", "--seed-checkpoint", "--seed-wire-dir",
)

#: Modules a weights-only class may never run.  ``tessera_campaign`` is the
#: whole campaign: its ``census`` row runs a calibration forward over the scope,
#: its ``capture`` row writes the calibration cache and its pricing rows fit and
#: consume a Hessian, so every one of its row kinds is either Hessian-aware or
#: needs the census environment (a resident model, transformers, a PrismaQuant
#: install).  Listing the module rather than trusting the flags is what makes
#: the refusal hold for a row kind that has not been written yet.
WEIGHTS_ONLY_FORBIDDEN_MODULES = frozenset({"prismaquant.tessera_campaign"})


def load_fleet_interpreters(path=None) -> dict:
    """The placement attestation table, checked for shape before it is trusted."""

    source = Path(path) if path is not None else FLEET_INTERPRETERS
    table = json.loads(source.read_text())
    if not isinstance(table, dict) or table.get("schema") != FLEET_INTERPRETERS_SCHEMA:
        raise RowClassRefused(
            f"{source}: not a {FLEET_INTERPRETERS_SCHEMA} document")
    tags = table.get("tags")
    if not isinstance(tags, dict) or not tags:
        raise RowClassRefused(f"{source}: declares no tags")
    for tag, shape in tags.items():
        if not isinstance(shape, dict):
            raise RowClassRefused(f"{source}: tag {tag!r} is not an object")
        for field in ("isa", "gpu_runtime"):
            if not isinstance(shape.get(field), str) or not shape[field]:
                raise RowClassRefused(f"{source}: tag {tag!r} declares no {field}")
        interpreters = shape.get("interpreters")
        if not isinstance(interpreters, dict):
            raise RowClassRefused(
                f"{source}: tag {tag!r} declares no interpreters mapping")
        for interpreter, record in interpreters.items():
            if not isinstance(record, dict) or not isinstance(
                    record.get("attested_by"), str) or not record["attested_by"]:
                raise RowClassRefused(
                    f"{source}: interpreter {interpreter} on tag {tag!r} names no "
                    "PrismaBuild action key that ran it")
    return table


def row_class(spec: dict, name: str = DEFAULT_ROW_CLASS) -> dict:
    """Resolve one class against the spec-level values it inherits.

    A spec with no ``classes`` block has exactly one class, the default, and it
    IS the spec: same interpreter, same environment, same tags, same container.
    That is the property the action key depends on, so it is stated here rather
    than left to the caller.
    """

    base = {
        "name": name,
        "python": spec["python"],
        "env": dict(spec["env"]),
        "tags": list(spec.get("tags", ["gb10"])),
        "cpus": int(spec.get("cpus", 4)),
        "wire_shared": True,
        "weights_only": False,
    }
    if "container" in spec:
        base["container"] = spec["container"]
    declared = spec.get("classes") or {}
    if name not in declared:
        if name != DEFAULT_ROW_CLASS:
            raise RowClassRefused(
                f"the spec declares no row class {name!r}; it has "
                f"{sorted(declared) or [DEFAULT_ROW_CLASS]}")
        return base
    override = declared[name]
    if not isinstance(override, dict):
        raise RowClassRefused(f"row class {name!r} is not an object")
    unknown = set(override) - ROW_CLASS_FIELDS
    if unknown:
        raise RowClassRefused(
            f"row class {name!r} declares {sorted(unknown)}, which a class does "
            f"not own; a class may set {sorted(ROW_CLASS_FIELDS - {'_why'})}")
    resolved = {**base, **{k: v for k, v in override.items() if k != "_why"}}
    if "env" in override:
        if not isinstance(override["env"], dict):
            raise RowClassRefused(f"row class {name!r} env is not an object")
        resolved["env"] = {**spec["env"], **override["env"]}
    resolved["tags"] = list(resolved["tags"])
    resolved["cpus"] = int(resolved["cpus"])
    for flag in ("wire_shared", "weights_only"):
        if not isinstance(resolved[flag], bool):
            raise RowClassRefused(f"row class {name!r} {flag} must be a boolean")
    if not resolved["tags"]:
        raise RowClassRefused(f"row class {name!r} names no placement tag")
    return resolved


def _class_isa(resolved: dict, fleet: dict) -> str:
    """The one ISA this class's tags place it on, or a refusal naming why not."""

    tags = fleet["tags"]
    found = {}
    for tag in resolved["tags"]:
        shape = tags.get(tag)
        if shape is None:
            raise RowClassRefused(
                f"row class {resolved['name']!r} names tag {tag!r}, which "
                f"{FLEET_INTERPRETERS.name} does not attest; add the tag with "
                "its ISA, its container GPU runtime and the interpreters "
                "observed running there")
        found.setdefault(shape["isa"], []).append(tag)
    if len(found) != 1:
        raise RowClassRefused(
            f"row class {resolved['name']!r} spans {len(found)} instruction "
            f"sets ({', '.join(sorted(found))}); a row is placed on one of its "
            "tags and the wire it writes is not the same object on both")
    return next(iter(found))


def _attest_placement(resolved: dict, fleet: dict) -> None:
    """Refuse a class this fleet cannot actually run where it is sent."""

    tags = fleet["tags"]
    container = resolved.get("container")
    for tag in resolved["tags"]:
        shape = tags[tag]
        if container is not None:
            declared = container.get("gpu_runtime", "nvidia")
            if declared != shape["gpu_runtime"]:
                raise RowClassRefused(
                    f"row class {resolved['name']!r} declares container GPU "
                    f"runtime {declared!r}, and tag {tag!r} attaches its GPU "
                    f"with {shape['gpu_runtime']!r}; the container would start "
                    "without the device it was admitted for")
            continue
        if resolved["python"] not in shape["interpreters"]:
            attested = sorted(shape["interpreters"])
            raise RowClassRefused(
                f"row class {resolved['name']!r} runs {resolved['python']}, "
                f"which is not attested on tag {tag!r}; that tag attests "
                f"{attested or 'no host interpreter'}. Run it there once and "
                f"add it to {FLEET_INTERPRETERS.name} with the PrismaBuild "
                "action key, or give the class a container")


def validate_row_classes(spec: dict, *, fleet=None, where="spec") -> list[dict]:
    """Check every declared class against the fleet, and say what each one is.

    A spec that declares no ``classes`` block is not checked: it is the shape
    every campaign in flight already has, and refusing it here would refuse
    specs this tool accepts today for a reason that has nothing to do with
    them.  A spec that DOES declare classes is checked in full, because the
    only reason to declare one is to place a row somewhere new.

    Two refusals carry the measurement rather than a taste.  A class whose
    bytes must equal the campaign's (``wire_shared``, the default) may not sit
    on an ISA other than the default class's: the Hessian-aware wire is not
    bit-comparable across ISAs, so such a row would merge foreign bytes into
    one table.  And a class that declares it does NOT share the wire must
    declare ``weights_only``, because weights-only is the only encode measured
    byte-identical across the two.
    """

    declared = spec.get("classes")
    if declared is None:
        return []
    if not isinstance(declared, dict) or not declared:
        raise RowClassRefused(f"{where}: classes must be a non-empty object")
    fleet = load_fleet_interpreters() if fleet is None else fleet
    default = row_class(spec, DEFAULT_ROW_CLASS)
    default_isa = _class_isa(default, fleet)
    _attest_placement(default, fleet)
    records = []
    for name in sorted(declared):
        resolved = default if name == DEFAULT_ROW_CLASS else row_class(spec, name)
        isa = default_isa if name == DEFAULT_ROW_CLASS else _class_isa(resolved, fleet)
        if name != DEFAULT_ROW_CLASS:
            _attest_placement(resolved, fleet)
        if resolved["wire_shared"]:
            if isa != default_isa:
                raise RowClassRefused(
                    f"{where}: row class {name!r} is placed on {isa} while this "
                    f"campaign's default class is on {default_isa}, and it "
                    "declares its bytes are the campaign's. A Hessian-aware "
                    "Tessera wire is not bit-comparable across those two "
                    "(RobTand/tessera#472). Declare wire_shared false and "
                    "weights_only true, or keep the class on one ISA")
        elif not resolved["weights_only"]:
            raise RowClassRefused(
                f"{where}: row class {name!r} declares wire_shared false "
                "without weights_only; the only encode measured byte-identical "
                "across instruction sets is the weights-only one")
        records.append({
            "class": name, "isa": isa, "tags": list(resolved["tags"]),
            "python": resolved["python"], "cpus": resolved["cpus"],
            "wire_shared": resolved["wire_shared"],
            "weights_only": resolved["weights_only"],
            "containerized": "container" in resolved,
        })
    return records


# ---------------------------------------------------------------------------
# The run spec
# ---------------------------------------------------------------------------

def load_spec(path: Path) -> dict:
    """Read the shared half of every row: model, campaign flags, fleet demand.

    Everything a row does *not* share -- its selection, its output paths -- is
    computed here, so the spec cannot accidentally pin two rows to one file.
    """
    spec = json.loads(Path(path).read_text())
    for field in ("model", "campaign_argv", "cwd", "python", "env"):
        if field not in spec:
            raise RuntimeError(f"{path}: spec has no {field!r}")
    forbidden = {"--model", "--out", "--cache-dir", "--checkpoint", "--units",
                 "--calibration-census", "--census-out", "--seed-checkpoint",
                 "--seed-wire-dir", "--capture-calibration-out",
                 "--calibration-cache", "--calibration-cache-sha256"}
    named = forbidden.intersection(spec["campaign_argv"])
    if named:
        raise RuntimeError(
            f"{path}: campaign_argv names {sorted(named)}, which this tool "
            "owns per row")
    if "--deadline-seconds" in spec["campaign_argv"]:
        # The in-process deadline stops a run mid-round, in the sorted-key
        # order the round's pending list happens to have; two rows stopped that
        # way price different anchor sets than one run would have. A row's
        # deadline is PrismaBuild's ``timeout_s`` and its retry, which restarts
        # the row against its own journal.
        raise RuntimeError(
            f"{path}: campaign_argv sets --deadline-seconds; a fanned-out row "
            "takes its deadline from the fleet, not from inside the round loop")
    if "container" in spec:
        validate_container(spec)
    for name in sorted(spec.get("classes") or {}):
        resolved = row_class(spec, name)
        if "container" in resolved:
            # The bounded-capture environment contract is stated by the row
            # that is actually bounded: a class whose argv asks for the
            # bounded policy has it here, and a legacy class that declares its
            # own purge delay is not refused for a rule it does not fall under.
            validate_container({"container": resolved["container"],
                                "env": resolved["env"]},
                               bounded=_row_is_bounded(resolved.get("argv") or []))
    validate_row_classes(spec, where=str(path))
    _process_baseline_bytes(spec, where=str(path))
    return spec


#: The process floor a row reserves when its spec declares none, in bytes.
#:
#: Measured, not invented.  Every completed row of the GLM-5.3
#: ``extension-r1024-02`` campaign stamps the floor its own
#: ``CaptureMemoryGuard`` read at its first check onto its ``cost.pkl``
#: (``selected_source_preparation.memory_guard.baseline.bytes``).  Read on
#: 2026-09-12 across rows 0058, 0061, 0062, 0063, 0066, 0074 and 0079, those
#: readings span 0.88-1.15 GB, and this is the top of that range: a
#: reservation is only worth the demand it moves if it covers the worst floor
#: observed, not the average one.  The 1,062,359,040 bytes the
#: ``_row_memory_gb`` docstring cites is an earlier reading, recorded on the
#: RobTand/prismaquant#390 receipt rather than traced to one of these rows; it
#: sits inside this range.
#:
#: Session note: ``pb_mem_gb_must_track_the_checked_phase_plan``, and
#: RobTand/prismaquant#522, which records the three rows this default exists
#: to stop losing.  It is a fleet number with a date on it, so a spec that
#: knows its own box overrides it and says so in ``baseline_policy``.
DEFAULT_PROCESS_BASELINE_BYTES = 1_150_000_000


def _process_baseline(spec: dict, *, where="spec") -> "tuple[int, str]":
    """The per-row process floor this recipe reserves, and where it came from.

    A spec that declares ``process_baseline_bytes`` owns the number, including
    a declared zero, which reserves nothing.  A spec that declares nothing gets
    ``DEFAULT_PROCESS_BASELINE_BYTES``, the worst floor measured on this fleet.
    The two are reported under different ``baseline_policy`` values, so a
    reader of a plan can tell a number an operator chose from a number this
    tool supplied.

    No universal torch-plus-CUDA constant is invented here: the default is a
    reading taken on the boxes these rows run on, and it stays a reservation
    rather than a measurement.  The row still measures its own floor at its
    first ``CaptureMemoryGuard.check`` and stamps it on its receipt.
    """
    from prismaquant.autoscale import (BASELINE_POLICY_EXPLICIT_RESERVATION,
                                       BASELINE_POLICY_MEASURED_DEFAULT_RESERVATION,
                                       validate_process_baseline_bytes)
    declared = "process_baseline_bytes" in spec
    value = validate_process_baseline_bytes(
        spec.get("process_baseline_bytes", DEFAULT_PROCESS_BASELINE_BYTES),
        where=f"{where}: process_baseline_bytes")
    return value, (BASELINE_POLICY_EXPLICIT_RESERVATION if declared
                   else BASELINE_POLICY_MEASURED_DEFAULT_RESERVATION)


def _process_baseline_bytes(spec: dict, *, where="spec") -> int:
    """The reservation alone, for callers that do not record its origin."""
    return _process_baseline(spec, where=where)[0]


def _guard_margin_bytes() -> int:
    """The physical safety margin the row's own guard holds back from the cap.

    Read from ``CaptureMemoryGuard`` rather than restated.  The guard refuses
    at ``cap - margin``, so a demand that does not carry the margin buys an
    admission the guard then declines, which is the failure this derivation
    exists to stop.
    """
    from prismaquant.memory_management import CaptureMemoryGuard
    return int(CaptureMemoryGuard.MARGIN_BYTES)


def _model_bytes(model: str) -> int:
    root = Path(model)
    return sum(path.stat().st_size for path in root.glob("*.safetensors"))


def _row_memory_gb(spec: dict, members: list[str], census: dict, *, selected_source=False) -> int:
    """The row's memory demand, from what the row actually holds.

    Streaming rows use the phase resource plan, including the selected-source
    plan when requested. These are derived byte bounds, not measured peaks.
    The resident-source fallback charges three quantities:

    * the checkpoint, which is loaded whole in ``bfloat16`` and is the same for
      every row;
    * the selection's Hessians, ``in x in`` in fp32 per member, which is the
      accumulator ``_collect_activations`` keeps;
    * the selection's retained scoring rows, ``max_act_rows x in`` in fp32.

    plus the spec's declared headroom for the forward pass and the encoder.

    **The spec's process baseline is charged here, once, outside the deltas.**
    A phase plan states deltas, and the floor those deltas sit on --
    interpreter, torch, the CUDA runtime, the pages the row's process has
    touched -- is a property of the box the row lands on, which this planner
    never enters. So it is still never derived here; it is *reserved*. A spec
    that declares ``process_baseline_bytes`` owns the number, including a
    declared zero; a spec that declares nothing gets
    ``DEFAULT_PROCESS_BASELINE_BYTES``, the worst floor measured on this fleet.
    ``baseline_policy`` records which of the two a plan used. Either way the
    scope travels with the number: it is this fleet's or this recipe's
    reservation, not a universal maximum.

    It is added to the demand and never to ``memory_bytes``, because the
    demand becomes a cgroup cap of exactly that many GiB
    (``prismabuild/pool.py:2850``) while the row refuses unless its plan fits
    under that cap *less* the floor it measures for itself
    (``prismaquant/tessera_campaign.py:4515``). Fold the reservation into the
    plan instead and the predicate compares an inflated delta against an
    inflated cap and nets to zero -- which is exactly why the spec's declared
    headroom, a term inside ``memory_bytes``, could never close this gap.

    Both branches charge it. A process floor exists whether or not a row
    streams, so leaving the resident-source branch out would make the key mean
    one thing on one path and nothing on the other.

    Rounding is not a reservation. ``ceil`` leaves at most one GiB of slack,
    and the floor on the #390 receipt is 1,062,359,040 bytes -- 0.9894 GiB,
    less than the most ``ceil`` can leave -- so before this key a row admitted
    according to where its ``memory_bytes`` landed modulo one GiB, which both
    inspected example rows lost. The row
    still measures its own floor at its first ``CaptureMemoryGuard.check`` and
    stamps it on its receipt (RobTand/prismaquant#390); that reading, not this
    declaration, remains the measured number.

    **The guard's own margin is charged here too.** The row is refused not at
    its cap but at ``cap - margin``: ``CaptureMemoryGuard.check`` compares its
    absolute reading against ``cap_bytes - margin_bytes``
    (``prismaquant/memory_management.py``). A demand that covers the plan and
    the floor but not the margin therefore buys an admission the row's own
    first check declines. The number is read from the guard, never restated,
    so the two cannot drift apart.
    """
    return _row_memory_demand(spec, members, census,
                              selected_source=selected_source)["mem_gb"]


def _row_memory_demand(spec: dict, members: list[str], census: dict, *,
                       selected_source=False) -> dict:
    """The row's demand and every term it is made of.

    ``_row_memory_gb`` is this, reduced to its GiB. The terms are kept because
    a refusal has to name them: a row that dies on the admission predicate is
    diagnosable from the plan, the floor and the margin, and before this they
    were reachable only by unpickling a completed row's ``cost.pkl``
    (RobTand/prismaquant#522).
    """
    gib = 1024 ** 3
    baseline, policy = _process_baseline(spec)
    margin = _guard_margin_bytes()
    if "--streaming" in spec['campaign_argv']:
        resource = _streamed_resource_plan(spec, census, members,
                                           selected_source=selected_source)
        # A selected row that runs the streaming row head holds a window, not
        # the population, and is admitted against the window's plan
        # (RobTand/prismaquant#640).
        stream = bool(selected_source and 'stream_memory_bytes' in resource
                      and _row_head_dependency(spec['campaign_argv']) is None)
        plan_bytes = int(resource['stream_memory_bytes' if stream else 'memory_bytes'])
        headroom_gb = 0
    else:
        shapes = census.get("unit_shapes") or {}
        hessian = sum(int(shapes.get(name, [0, 0])[1]) ** 2 * 4 for name in members)
        rows = sum(int(shapes.get(name, [0, 0])[1]) * int(spec.get("max_act_rows", 512)) * 4
                   for name in members)
        plan_bytes = _model_bytes(spec["model"]) + hessian + rows
        headroom_gb = int(spec.get("headroom_gb", 24))
    demand_bytes = plan_bytes + baseline + margin
    return {
        "plan_bytes": int(plan_bytes),
        "process_baseline_bytes": int(baseline),
        "process_baseline_policy": policy,
        "guard_margin_bytes": int(margin),
        "demand_bytes": int(demand_bytes),
        "headroom_gb": headroom_gb,
        "mem_gb": int(math.ceil(demand_bytes / gib)) + headroom_gb,
    }


def _streamed_resource_plan(spec, census, members, *, selected_source=False):
    from prismaquant.autoscale import streamed_calibration_resources, selected_anchor_resources
    argv = spec['campaign_argv']
    baseline_bytes, baseline_policy = _process_baseline(spec)
    def argument(name, default, convert=int):
        return convert(argv[argv.index(name)+1]) if name in argv else default
    shapes = census.get('unit_shapes') or {}
    counts = census.get('counts') or {}
    options = dict(
        unit_shapes={n: shapes[n] for n in members}, counts=counts,
        max_act_rows=argument('--max-act-rows', int(spec.get('max_act_rows', 512))),
        cache_slots=argument('--streaming-cache-slots', 2),
        prefetch_workers=argument('--streaming-prefetch-workers', 1),
        headroom_gb=max(float(spec.get('headroom_gb', 24)),
                        argument('--streaming-cache-headroom-gb', 24., float)),
        # Recorded beside ``memory_bytes``, never summed into it: the plan
        # stays pure phase deltas and the reservation is charged once, in
        # ``_row_memory_demand``, on the demand.
        process_baseline_bytes=baseline_bytes,
        process_baseline_policy=baseline_policy)
    if selected_source:
        return selected_anchor_resources(spec['model'], **options,
            anchor_batch_size=argument('--anchor-batch-size', 1),
            source_snapshot_policy=argument('--source-snapshot-policy', 'whole-layer-v1', str),
            # The row's own campaign will hold this many host bytes of staged
            # artifacts, so the box that admits the row has to be told. A
            # dispatcher that planned without it would size a worker for a
            # campaign it is not about to run.
            publication_overlap_bytes=argument('--publication-overlap-bytes', 0),
            campaign_identity_bytes=argument('--campaign-identity-bytes', 0),
            # The campaign's default is the CPUs the row is admitted with,
            # which is the class's declared ``cpus`` demand.
            campaign_identity_threads=argument('--campaign-identity-threads',
                                               int(spec.get('cpus', 4))),
            **(dict(capture_load_policy=argument('--capture-load-policy', None, json.loads))
               if '--capture-load-policy' in argv else {}))
    return streamed_calibration_resources(spec['model'], **options,
        nsamples=argument('--nsamples', 8), seqlen=argument('--seqlen', 512),
        capture_policy=argument('--streaming-capture-policy', 'legacy', str))


def _row_head_dependency(argv):
    """What makes a selected row's argv run the load-all head, or ``None``.

    The campaign's own rule (``tessera_row_stream.stream_head_dependency``)
    applied to what argv can say. A plan cannot see a checkpoint an earlier
    attempt leaves behind: a row that resumes one runs the load-all head, and
    its own admission then refuses a stream-sized reservation by name.
    """
    from prismaquant.tessera_row_stream import stream_head_dependency

    def value(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    return stream_head_dependency(
        row_head=value('--row-head', 'stream'), selected_source=True,
        capture_load_policy='--capture-load-policy' in argv,
        export_hessian_reference_policy='--export-hessian-reference-policy' in argv,
        max_rounds=int(value('--max-rounds', 0)), seed_checkpoint=value('--seed-checkpoint'),
        checkpoint_exists=False)


class DemandRefused(RuntimeError):
    """A manifest row asks for less memory than the row it will run needs."""


def _inner_campaign_argv(row: dict) -> list:
    """The campaign argv a manifest row will actually run.

    A row's command is ``python -u -m prismaquant.tessera_campaign <argv>``,
    wrapped by the container launcher when the spec declares one, so the
    campaign argv is whatever follows the LAST ``-m``. Reading it back from
    the row, rather than rebuilding it from the spec, is the point: the
    relaunch that lost three rows carried ``--publication-overlap-bytes`` in
    the manifest while the spec that planned it did not
    (RobTand/prismaquant#522).
    """
    argv = list(row.get("argv") or [])
    if "-m" not in argv:
        raise DemandRefused("row argv runs no python module, so its demand "
                            "cannot be derived")
    index = len(argv) - 1 - argv[::-1].index("-m")
    return argv[index + 2:]


def _row_label(inner_argv: list, index: int) -> str:
    """The row id, taken from the selection file it names."""
    if "--units" in inner_argv:
        return Path(inner_argv[inner_argv.index("--units") + 1]).stem
    return f"row-{index:04d}"


def _units_members(inner_argv: list) -> list:
    selection = json.loads(Path(inner_argv[inner_argv.index("--units") + 1]).read_text())
    return [name for entry in selection["groups"]
            for name in (entry.get("sampled") or entry["members"])]


def verify_row_demand(spec: dict, census: dict, row: dict, *,
                      box_memory_gb=None, label=None) -> dict:
    """Recompute one row's demand from its own argv and refuse an under-declared one.

    Two refusals, and neither is a warning:

    * a declared ``demand.mem_gb`` below the derived one buys an admission the
      row's own guard declines about twenty seconds later, which PrismaBuild
      records as a failed row with no retry;
    * a derived demand above the capacity the fleet's GPU boxes declare can
      never be admitted at all, so it is refused here rather than queued.

    ``box_memory_gb`` is a parameter and not a lookup: this function states
    what the capacity has to be compared against, and the caller states what
    the fleet declares.
    """
    inner = _inner_campaign_argv(row)
    label = label or "row"
    model = (inner[inner.index("--model") + 1] if "--model" in inner
             else spec["model"])
    row_spec = {**spec, "model": model, "campaign_argv": inner}
    members = (_units_members(inner) if "--units" in inner
               else sorted(census.get("counts") or {}))
    demand = _row_memory_demand(row_spec, members, census,
                                selected_source="--streaming" in inner)
    gib = 1024 ** 3
    declared_gb = int(row["demand"]["mem_gb"])
    record = {"row": label, "declared_mem_gb": declared_gb,
              "declared_bytes": declared_gb * gib, **demand}
    terms = (f"plan {demand['plan_bytes']} B + process baseline "
             f"{demand['process_baseline_bytes']} B "
             f"({demand['process_baseline_policy']}) + guard margin "
             f"{demand['guard_margin_bytes']} B = {demand['demand_bytes']} B")
    if box_memory_gb is not None and demand["mem_gb"] > int(box_memory_gb):
        raise DemandRefused(
            f"{label}: derived demand {demand['mem_gb']} GiB is above the "
            f"{int(box_memory_gb)} GiB a GPU box declares, so no admission "
            f"can come: {terms}"
            + (f" + {demand['headroom_gb']} GiB declared headroom"
               if demand["headroom_gb"] else "")
            + f"; declared demand.mem_gb {declared_gb} "
            f"({declared_gb * gib} B). Reduce a plan term or run it on a "
            "wider box; do not shrink the demand to fit.")
    if declared_gb < demand["mem_gb"]:
        raise DemandRefused(
            f"{label}: declared demand.mem_gb {declared_gb} "
            f"({declared_gb * gib} B) is below the {demand['mem_gb']} GiB its "
            f"own argv derives: {terms}"
            + (f" + {demand['headroom_gb']} GiB declared headroom"
               if demand["headroom_gb"] else "")
            + ". PrismaBuild would admit the row and its CaptureMemoryGuard "
            "would then refuse it.")
    return record


def verify_manifest_demands(spec: dict, census: dict, rows: list, *,
                            box_memory_gb=None) -> list:
    """Every row in a manifest, refusing on the whole set rather than the first.

    A campaign is re-queued as a set, so an operator needs every
    under-declared row named at once, not one per run.
    """
    records, refusals = [], []
    for index, row in enumerate(rows):
        label = _row_label(_inner_campaign_argv(row), index)
        try:
            records.append(verify_row_demand(spec, census, row,
                                             box_memory_gb=box_memory_gb,
                                             label=label))
        except DemandRefused as error:
            refusals.append(str(error))
    if refusals:
        raise DemandRefused(
            f"{len(refusals)} of {len(rows)} rows declare a memory demand "
            "their own argv does not support:\n" + "\n".join(refusals))
    return records


def parse_pbrun_demand(demand: str) -> "dict[str, int]":
    """Parse pbrun's ``key=value,key=value`` demand spelling."""
    if not isinstance(demand, str) or not demand.strip():
        raise DemandRefused("--demand must be a non-empty pbrun demand string")
    parsed: dict[str, int] = {}
    for item in demand.split(","):
        name, separator, value = item.partition("=")
        if not separator or not name.strip():
            raise DemandRefused(f"--demand {demand!r} is not key=value pairs")
        try:
            parsed[name.strip()] = int(value)
        except ValueError:
            raise DemandRefused(
                f"--demand {demand!r} has a non-integer {name.strip()!r}"
            ) from None
    return parsed


def joint_submission_memory_bound(spec: dict, plan: dict) -> dict:
    """The combined physical bound the row's own policy states, in bytes.

    A joint row has three numbers and they are not interchangeable. The
    container cap bounds only what the cgroup charges -- on GB10 that is the
    CPU side, measured at 16 GiB charged for a container that held 78.87 GiB
    of model and 6 GiB of KV -- so the demand PrismaBuild reserves has to be
    the combined physical demand, and the device envelope is a *subset* of it
    rather than a second box.

    The plan's ``aggregate_memory_bytes`` is that combined bound and is used
    first. A plan written before it existed gets the same number derived from
    the two bounds it does state, ``cpu_memory_gb`` (whose cgroup cap is what
    the launcher passes as ``--memory``) plus ``max_gpu_bytes``. When neither
    is derivable the record says so rather than inventing a bound.
    """
    gib = 1024 ** 3
    aggregate = plan.get("aggregate_memory_bytes")
    if (isinstance(aggregate, int) and not isinstance(aggregate, bool)
            and aggregate > 0):
        return {"bound_bytes": aggregate,
                "bound_basis": "plan.aggregate_memory_bytes"}
    cap_gb = container_memory_budget_gb(spec)
    gpu_bytes = plan.get("max_gpu_bytes")
    if (cap_gb is not None and isinstance(gpu_bytes, int)
            and not isinstance(gpu_bytes, bool) and gpu_bytes > 0):
        return {"bound_bytes": int(round(cap_gb * gib)) + gpu_bytes,
                "bound_basis": "spec container cap + plan.max_gpu_bytes"}
    return {"bound_bytes": None, "bound_basis": None}


def verify_joint_submission_demand(spec: dict, plan: dict, demand: str, *,
                                   label: str = "joint row") -> dict:
    """Refuse a PrismaBuild reservation below the row's own physical bound.

    Neither of the two mistakes is a warning: reserving the container cap
    under-reserves the box by exactly the device envelope, and reserving
    34 GiB while the row's guard is allowed to hold 80 GiB of device
    residency beside it buys an admission the row then declines.

    Reserving *more* than the bound is allowed and is what A2 does (114 GiB
    for a 34 GiB cap beside an 80 GiB device envelope): the excess is
    conservatism, and it is recorded rather than silently trimmed.
    """
    gib = 1024 ** 3
    parsed = parse_pbrun_demand(demand)
    reserved_gb = parsed.get("mem_gb")
    if reserved_gb is None:
        raise DemandRefused(
            f"{label}: --demand {demand!r} reserves no mem_gb. A joint row's "
            "device residency is not charged to its container cap, so the "
            "reservation has to state the combined physical demand.")
    record = {"row": label, "demand": demand, "reserved_mem_gb": reserved_gb,
              **joint_submission_memory_bound(spec, plan)}
    bound = record["bound_bytes"]
    if bound is not None and reserved_gb * gib < bound:
        raise DemandRefused(
            f"{label}: --demand reserves {reserved_gb} GiB, below the "
            f"{math.ceil(bound / gib)} GiB combined physical bound this row "
            f"states ({record['bound_basis']}: {bound} B). The container cap "
            "bounds the CPU side alone, so reserving it here under-reserves "
            "the box by the device envelope; reserve the combined demand, or "
            "lower the plan's own bound first.")
    return record


#: The scope a joint pass evaluates, and the two shapes it can take.  A
#: campaign-scoped pass evaluates every unit and every window the campaign's
#: census defines; a diagnostic pass evaluates an explicitly frozen window
#: subset of that same roster.  The scope is *derived* from the census and
#: campaign plan the joint plan binds by sha256 -- never from a tally the plan
#: states about itself -- so a roster or window set narrowed without moving the
#: declared count cannot read as complete.  Two rosters of equal length with
#: different members are different scopes; so are two window sets of equal size
#: with different membership.
CAMPAIGN_SCOPE_SCHEMA = "prismaquant.tessera_joint_campaign_scope.v1"
COMPLETE_CAMPAIGN_SCOPE = "complete_campaign"
DIAGNOSTIC_SCOPE = "diagnostic_window_subset"
CAMPAIGN_SCOPE_KINDS = (COMPLETE_CAMPAIGN_SCOPE, DIAGNOSTIC_SCOPE)

CENSUS_SCHEMA = "prismaquant.tessera_campaign_census.v1"
PANEL_SCHEMA = "prismaquant.tessera_joint_eval_panel.v1"
PANEL_STATUS = "diagnostic_pilot"


class ScopeRefused(RuntimeError):
    """A joint row's evaluated scope is not the scope it is submitted for."""


def _bound_json(plan: dict, key: str, *, label: str) -> dict:
    """The JSON artifact a joint plan binds by path and sha256, re-checked."""
    declared = (plan.get("inputs") or {}).get(key)
    if (not isinstance(declared, dict) or not isinstance(declared.get("path"), str)
            or not isinstance(declared.get("sha256"), str)):
        raise ScopeRefused(f"{label}: joint plan inputs.{key} is not a bound artifact")
    path = Path(declared["path"])
    actual = _sha256_of(path)
    if actual != declared["sha256"]:
        raise ScopeRefused(
            f"{label}: plan inputs.{key} is {path}, which hashes to {actual}, "
            f"not the bound {declared['sha256']}")
    return json.loads(path.read_text())


def _bound_digest(container: dict, key: str, *, label: str, where: str) -> str:
    """One artifact a plan binds by path and sha256, re-checked, as its digest.

    The same check ``_bound_json`` makes, for the artifacts a scope *names* but
    never parses: the calibration tokens, the canonical capture manifest and the
    merged campaign checkpoint are large or non-JSON, and a digest is all the
    identity needs.  Omitting the check would let a plan bind a digest it does
    not hold.
    """
    declared = (container or {}).get(key)
    if (not isinstance(declared, dict) or not isinstance(declared.get("path"), str)
            or not isinstance(declared.get("sha256"), str) or not declared["sha256"]):
        raise ScopeRefused(f"{label}: {where} is not a bound artifact")
    path = Path(declared["path"])
    actual = _sha256_of(path)
    if actual != declared["sha256"]:
        raise ScopeRefused(
            f"{label}: {where} is {path}, which hashes to {actual}, not the "
            f"bound {declared['sha256']}")
    return actual


def _bound_pickle(container: dict, key: str, *, label: str, where: str) -> dict:
    """The pickled artifact a plan binds by path and sha256, re-read and checked.

    ``_bound_json``'s check and then the object itself, for the campaign cost
    table: the plan's own record of which ``(unit, format)`` cells the campaign
    priced.  The digest is what makes that record the plan's rather than the
    caller's -- an edited table no longer hashes to what the plan declares --
    and reading it is one sequential pass over a few hundred MiB, not the
    multi-GiB candidate roster a per-cell re-derivation from the checkpoint
    would need.
    """
    import pickle

    declared = (container or {}).get(key)
    if (not isinstance(declared, dict) or not isinstance(declared.get("path"), str)
            or not isinstance(declared.get("sha256"), str) or not declared["sha256"]):
        raise ScopeRefused(f"{label}: {where} is not a bound artifact")
    path = Path(declared["path"])
    actual = _sha256_of(path)
    if actual != declared["sha256"]:
        raise ScopeRefused(
            f"{label}: {where} is {path}, which hashes to {actual}, not the "
            f"bound {declared['sha256']}; the campaign cost table the plan names "
            "is not the one on disk, so the cells it priced are not the plan's")
    with path.open("rb") as handle:
        blob = pickle.load(handle)
    if not isinstance(blob, dict):
        raise ScopeRefused(f"{label}: {where} ({path}) is not a cost payload")
    return blob


#: The census fields that identify the DRAW, not merely its size. ``nsamples``
#: and ``seqlen`` say how many windows; these say which ones, over which corpus
#: revision and which tokenizer ids, and they are exactly what
#: ``tessera_campaign.require_census_draw`` holds a producer run to at encode
#: time. Two censuses with the same roster and the same window count that differ
#: here are two different calibrations, and a scope that did not bind them would
#: let one stand in for the other.
CENSUS_DRAW_FIELDS = ("model", "text_sha256", "fit_ids_sha256",
                      "seed", "layer_stride")


def _census_draw(census: dict, *, label: str) -> dict:
    """The frozen draw a census was taken on, or a refusal naming the field."""
    draw: dict = {}
    for field in ("model", "text_sha256", "fit_ids_sha256"):
        value = census.get(field)
        if not isinstance(value, str) or not value:
            raise ScopeRefused(
                f"{label}: the census declares no {field}, so the calibration "
                "draw it was taken on is unidentified; a roster and a window "
                "count are not a draw")
        draw[field] = value
    for field in ("seed", "layer_stride"):
        value = census.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ScopeRefused(
                f"{label}: the census declares no integer {field}, so the "
                "calibration draw it was taken on is unidentified")
        draw[field] = value
    return draw


def joint_campaign_scope(plan: dict, *, label: str = "joint row") -> dict:
    """The exact roster and window identity one joint plan evaluates.

    Read from the plan's own bound census and campaign plan.  A plan whose
    census roster is not the campaign plan's roster, whose ``calib_seqlen`` is
    not the census ``seqlen``, or whose declared unit/group tallies do not
    equal the roster it actually bound refuses here -- the counts are checked
    against the identity, not accepted in place of it.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    inputs = plan.get("inputs")
    if not isinstance(inputs, dict):
        raise ScopeRefused(f"{label}: the joint plan declares no inputs block")
    census = _bound_json(plan, "census", label=label)
    campaign = _bound_json(plan, "campaign_plan", label=label)
    if census.get("schema") != CENSUS_SCHEMA:
        raise ScopeRefused(
            f"{label}: census schema is {census.get('schema')!r}, expected {CENSUS_SCHEMA}")
    if Path(campaign["census"]).resolve() != Path(inputs["census"]["path"]).resolve():
        raise ScopeRefused(
            f"{label}: the campaign plan and the joint plan name different censuses")
    roster = census.get("unit_shapes")
    groups = census.get("anchor_groups")
    if (not isinstance(roster, dict) or not roster
            or not isinstance(groups, dict) or not groups):
        raise ScopeRefused(f"{label}: the census carries no exact roster")
    owners: set = set()
    for row in campaign.get("rows") or []:
        owners.update(row.get("members") or [])
    if owners != set(roster):
        raise ScopeRefused(
            f"{label}: the census roster is not the campaign plan's roster "
            f"({len(owners)} campaign members, {len(roster)} census units)")
    # The census is the window authority: a plan that narrows its own
    # ``n_calib_samples`` is a narrower scope, not the campaign's.
    campaign_windows = census.get("nsamples")
    seqlen = census.get("seqlen")
    if (isinstance(campaign_windows, bool) or not isinstance(campaign_windows, int)
            or campaign_windows <= 0 or not isinstance(seqlen, int) or seqlen <= 0):
        raise ScopeRefused(
            f"{label}: the census declares no window count or sequence length")
    execution = plan.get("execution") or {}
    if execution.get("calib_seqlen") != seqlen:
        raise ScopeRefused(
            f"{label}: plan calib_seqlen {execution.get('calib_seqlen')!r} is not "
            f"the census seqlen {seqlen!r}")
    evaluated = execution.get("n_calib_samples")
    panel = plan.get("joint_eval")
    if panel is None:
        kind, selection = COMPLETE_CAMPAIGN_SCOPE, None
        windows = evaluated
    else:
        kind = DIAGNOSTIC_SCOPE
        if (not isinstance(panel, dict) or panel.get("schema") != PANEL_SCHEMA
                or panel.get("status") != PANEL_STATUS):
            raise ScopeRefused(
                f"{label}: a narrow scope needs an explicit {PANEL_SCHEMA} "
                f"panel with status {PANEL_STATUS!r}")
        selected = panel.get("selection")
        windows = selected.get("size") if isinstance(selected, dict) else None
        if (isinstance(windows, bool) or not isinstance(windows, int)
                or not 0 < windows < campaign_windows):
            raise ScopeRefused(
                f"{label}: a diagnostic scope selects a strict, sized subset of the "
                f"{campaign_windows}-window campaign, got {windows!r}")
        if panel.get("shape") != [windows, seqlen]:
            raise ScopeRefused(
                f"{label}: the diagnostic panel shape {panel.get('shape')!r} is not "
                f"[{windows}, {seqlen}]")
        selection = canonical_json_sha256(panel, where=f"{label} joint evaluation panel")
    if isinstance(windows, bool) or not isinstance(windows, int) or windows <= 0:
        raise ScopeRefused(f"{label}: the plan evaluates no windows")
    if kind == COMPLETE_CAMPAIGN_SCOPE and windows != campaign_windows:
        raise ScopeRefused(
            f"{label}: a campaign-scoped plan evaluates all {campaign_windows} "
            f"windows, got {windows}; freeze the subset as an explicit diagnostic panel")
    declared = (inputs.get("required_source_units"),
                inputs.get("required_campaign_groups"))
    if declared != (len(roster), len(groups)):
        raise ScopeRefused(
            f"{label}: the plan declares {declared[0]!r}/{declared[1]!r} units/groups "
            f"but bound {len(roster)}/{len(groups)}; a count is not the roster")
    # The draw and the capture are the other half of "the same calibration
    # contract": the roster says which Linears were priced, this says over which
    # windows, which corpus revision and which tokenizer ids, and which captured
    # activations were read. Read from the plan's own bound artifacts, so a plan
    # cannot restate one it does not hold.
    calibration_sha256 = canonical_json_sha256({
        "census_draw": _census_draw(census, label=label),
        "window_count": campaign_windows,
        "calib_seqlen": seqlen,
        "calibration_input_sha256": _bound_digest(
            plan, "calibration_input", label=label,
            where="joint plan calibration_input"),
        "canonical_capture_sha256": _bound_digest(
            plan, "canonical_capture", label=label,
            where="joint plan canonical_capture"),
    }, where=f"{label} calibration identity")
    # The candidate roster is `identity.units[unit].menu` inside the campaign's
    # merged checkpoint, and the joint loader refuses any priced rung outside
    # it. Binding that artifact binds the exact (unit, format) roster the pass
    # is admitted against: a reduced or substituted menu is different bytes, so
    # it is a different campaign. The bind costs ONE sequential read of the
    # artifact per submission (``_bound_digest`` hashes it rather than trusting
    # the plan's own number); what it avoids is parsing the identity into the
    # graph, which is the memory the joint loader's streaming seal exists to
    # remove.
    campaign_checkpoint_sha256 = _bound_digest(
        inputs, "merged_checkpoint", label=label,
        where="joint plan inputs.merged_checkpoint")
    return {
        "schema": CAMPAIGN_SCOPE_SCHEMA,
        "kind": kind,
        "source_roster_sha256": canonical_json_sha256(
            sorted(roster), where=f"{label} source roster"),
        "source_unit_count": len(roster),
        "campaign_group_roster_sha256": canonical_json_sha256(
            groups, where=f"{label} campaign group roster"),
        "campaign_group_count": len(groups),
        "window_count": windows,
        "campaign_window_count": campaign_windows,
        "calib_seqlen": seqlen,
        "selection_sha256": selection,
        "calibration_sha256": calibration_sha256,
        "campaign_checkpoint_sha256": campaign_checkpoint_sha256,
    }


CAMPAIGN_IDENTITY_SCHEMA = "prismaquant.tessera_joint_campaign_identity.v1"

#: The fields of the frozen campaign identity a submission has to reproduce.
#: ``window_count`` is deliberately absent: the campaign identity names the
#: campaign's own window total, and whether a plan evaluates all of it is the
#: ``kind`` the caller requires, not a property of the campaign.
#: ``selection_sha256`` is absent for the same reason -- a diagnostic subset is
#: a different ``kind`` of the same campaign, and its panel identity travels in
#: the scope the submission is stamped with. ``calibration_sha256`` is NOT
#: absent: the pilot and the full continuation read one draw and one capture, so
#: both have to reproduce it, and it deliberately excludes the panel so that
#: they can.
CAMPAIGN_IDENTITY_FIELDS = (
    "source_unit_count", "source_roster_sha256",
    "campaign_group_count", "campaign_group_roster_sha256",
    "campaign_window_count", "calib_seqlen",
    "calibration_sha256", "campaign_checkpoint_sha256")


def campaign_identity(scope: dict) -> dict:
    """The frozen identity of the campaign a scope belongs to.

    Derived from a plan's bound census so that the pilot and the full
    continuation can be held to one roster and one window total without either
    of them restating it.  An operator seals this once and passes it to every
    joint submission; the generic path never hardcodes the numbers.
    """
    record = {"schema": CAMPAIGN_IDENTITY_SCHEMA,
              **{field: scope[field] for field in CAMPAIGN_IDENTITY_FIELDS}}
    return record


def verify_joint_campaign_scope(plan: dict, *, require_scope: str,
                                campaign: dict, label: str = "joint row") -> dict:
    """Refuse a joint submission whose scope is not the scope it claims.

    ``require_scope`` is what the caller is submitting *for*, and a diagnostic
    subset is not the campaign's score: asking for ``complete_campaign`` and
    handing it a plan that evaluates 16 of 512 windows refuses here, before the
    read set is built, rather than surfacing as a shortfall in the merged cost
    after the GPU window has closed.

    ``campaign`` is the frozen campaign identity the submission belongs to.  It
    is required: a plan is self-consistent with whatever census it binds, so
    without an identity that is fixed *outside* the plan a coherently narrowed
    roster would still read as the whole campaign.  Every identity field is
    compared exactly -- the counts are a convenience of the record, not the
    acceptance.
    """
    if require_scope not in CAMPAIGN_SCOPE_KINDS:
        raise ScopeRefused(
            f"{label}: unknown required scope {require_scope!r}; expected one of "
            f"{', '.join(CAMPAIGN_SCOPE_KINDS)}")
    if not isinstance(campaign, dict) or campaign.get("schema") != CAMPAIGN_IDENTITY_SCHEMA:
        raise ScopeRefused(
            f"{label}: a joint submission must bind a {CAMPAIGN_IDENTITY_SCHEMA} "
            "campaign identity; a plan's own census cannot show its own scope is "
            "the campaign's")
    # A field the identity does not carry cannot be compared, and "absent"
    # compares unequal to every value -- which is how an identity sealed before
    # the calibration and candidate-roster binding would refuse the *right*
    # plan with a message about the wrong thing. Name the re-seal instead.
    absent = [field for field in CAMPAIGN_IDENTITY_FIELDS if field not in campaign]
    if absent:
        raise ScopeRefused(
            f"{label}: the frozen campaign identity carries no "
            f"{', '.join(absent)}; it predates binding the calibration draw, the "
            "capture and the campaign's candidate roster, and must be re-sealed "
            "from a campaign-scoped plan before it can stand for the campaign")
    scope = joint_campaign_scope(plan, label=label)
    for field in CAMPAIGN_IDENTITY_FIELDS:
        declared = campaign.get(field)
        if declared != scope[field]:
            raise ScopeRefused(
                f"{label}: the plan's {field} is {scope[field]!r}, not the frozen "
                f"campaign identity's {declared!r}; a roster of equal length with "
                "different members is a different campaign")
    if scope["kind"] != require_scope:
        raise ScopeRefused(
            f"{label}: this submission requires a {require_scope} scope, but the "
            f"plan evaluates {scope['kind']} ({scope['window_count']} of "
            f"{scope['campaign_window_count']} windows over "
            f"{scope['source_unit_count']} units, selection "
            f"{scope['selection_sha256']}). A diagnostic subset is its own "
            "evidence; submit the campaign-scoped plan for the campaign's score.")
    return scope


def _joint_row_binds_cell(row: dict, name: str, fmt: str, *, label: str) -> bool:
    """Whether this row is a joint A-side **for this cell**.

    The predicate itself is the shared one
    (:func:`prismaquant.allocator_candidates.joint_row_binds_cell`) so that the
    submission-time gate and the stage that computes coverage cannot drift; the
    only thing added here is this tool's own refusal type and label.
    """
    from prismaquant.allocator_candidates import joint_row_binds_cell

    try:
        return joint_row_binds_cell(row, name, fmt, where=label)
    except ValueError as error:
        raise ScopeRefused(str(error)) from error


def _bound_joint_cells(costs: dict, *, label: str) -> set:
    """The ``(unit, format)`` cells whose own joint A-side is bound to them.

    One walk, one answer, for both readers of "this artifact already carries an
    activation term at this cell": the coverage roster below, and ``submit-aqua``
    deciding whether a stage would add anything. A row that does not bind its own
    key is a refusal here rather than a cell either reader may count, so the two
    cannot disagree about which cells are covered.
    """
    bound: set = set()
    for name, entry in costs.items():
        if not isinstance(entry, dict):
            continue
        for fmt, row in entry.items():
            if not isinstance(row, dict):
                continue
            if _joint_row_binds_cell(row, name, fmt, label=label):
                bound.add((name, fmt))
    return bound


def aqua_requested_cells(plan: dict, payload: dict, formats: "list[str]",
                         *, label: str = "aqua row",
                         accept_joint_cells_outside_plan: bool = False,
                         ) -> "tuple[dict, frozenset]":
    """The exact ``(unit, format)`` roster AQUA is asked to price, from the plan.

    Returns ``(record, cells)``. ``cells`` is the requested roster itself; the
    record is the same thing as counts and a digest, so a submission can stamp
    it without carrying hundreds of thousands of tuples in its summary.

    The roster is the **plan's**, read from the cost table the plan binds at
    ``inputs.merged_cost`` and re-checked against the sha256 it declares there.
    That table is the campaign's own priced surface: the cells the allocation is
    allowed to select from. ``--cost-in`` -- the weight-only or joint-merged
    table the stage merges an A-side into -- has to reproduce it unit for unit.

    Deriving the requested set from the artifact under test is the hole this
    closes. The stage states its acceptance against the cells it is given, so a
    table narrowed to the cells that already carry a price leaves the gate
    holding a denominator that the narrowing moved: deleting one unit's cell
    while another unit still carries that format leaves the unit roster exact
    and the carried-format union exact, and if the remainder is joint-priced the
    run reads as complete coverage while a planned cell has no A-side at all.
    Binding to the plan's own table makes that case a refusal, and makes a cost
    table edited after the plan was sealed a refusal of its own.

    ``formats`` is the campaign's menu and has to be exactly the union of the
    plan's cells. Naming a subset would move the denominator the same way: the
    cells it left out are exactly the ones whose A-side would then go unchecked.

    Extra cells are read the same way round. A cell the plan never priced that
    is *not* already joint-priced would be priced by this stage, which makes the
    gate stricter rather than narrower -- but it is still a row the campaign's
    own table does not have, so it refuses.

    A cell the plan never priced that already carries its own joint A-side is a
    joint pass's addition. Being joint-priced is not by itself evidence that the
    row belongs to *this* plan: the row is internally valid and its own cell is
    the one it names, but nothing in it was compared with the plan's draw, its
    capture or its candidate menu, so a pass over a wider roster would place
    prices this campaign never priced beside the plan's own table and the
    artifact would read as the plan's surface. It refuses, naming the re-seal
    instead, unless the caller passes
    ``accept_joint_cells_outside_plan`` -- which is how an operator reusing an
    artifact sealed against an older roster keeps that reuse explicit and
    recorded as unverified rather than silent. A row whose operator coordinate
    is not the key it was found under is a refusal either way.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    census = _bound_json(plan, "census", label=label)
    roster = census.get("unit_shapes")
    if not isinstance(roster, dict) or not roster:
        raise ScopeRefused(f"{label}: the plan's bound census carries no roster")
    planned = _bound_pickle(
        plan.get("inputs") or {}, "merged_cost", label=label,
        where="joint plan inputs.merged_cost")
    planned_costs = planned.get("costs")
    if not isinstance(planned_costs, dict) or not planned_costs:
        raise ScopeRefused(
            f"{label}: the cost table the plan binds carries no 'costs' table, "
            "so it does not say which cells the campaign priced")
    costs = payload.get("costs")
    if not isinstance(costs, dict) or not costs:
        raise ScopeRefused(f"{label}: the cost artifact carries no 'costs' table")
    # One walk, before anything is counted: every joint row has to be its own
    # cell. A row that is not raises here, so no later reader sees a set with a
    # row whose coordinate this function never checked.
    bound_joints = _bound_joint_cells(costs, label=f"{label} cost artifact")
    # The unit roster is checked on BOTH tables against the plan's bound census.
    # A unit the plan priced and the artifact lacks would leave the campaign
    # short an A-side while reading as complete; a unit the artifact carries
    # that the plan never priced is not the campaign's cell at all.
    for what, table in (("the cost table the plan binds", planned_costs),
                        ("the cost artifact", costs)):
        missing = sorted(set(roster) - set(table))
        extra = sorted(set(table) - set(roster))
        if missing or extra:
            raise ScopeRefused(
                f"{label}: {what} is not the plan's roster -- "
                f"{len(missing)} unit(s) the plan priced are absent (e.g. "
                f"{missing[:3]}) and {len(extra)} unit(s) it never priced are "
                f"carried (e.g. {extra[:3]}).")
    wanted = [str(item) for item in formats]
    if len(set(wanted)) != len(wanted) or not wanted:
        raise ScopeRefused(f"{label}: --formats must name each format once")

    cells: set = set()
    planned_formats: set = set()
    units_missing: list = []
    missing_total = 0
    units_extra: list = []
    extra_total = 0
    joint_extra = 0
    for name in sorted(roster):
        planned_entry = planned_costs[name]
        if not isinstance(planned_entry, dict):
            raise ScopeRefused(
                f"{label}: the cost table the plan binds has no per-format row "
                f"for {name}")
        entry = costs[name]
        if not isinstance(entry, dict):
            raise ScopeRefused(f"{label}: {name} has no per-format row")
        expected = {fmt for fmt, row in planned_entry.items()
                    if isinstance(row, dict)}
        carried = {fmt for fmt, row in entry.items() if isinstance(row, dict)}
        planned_formats |= expected
        # The requested roster is the plan's cells, so it is the same digest for
        # the same plan whether or not a joint pass has since added rows of its
        # own.
        cells.update((name, fmt) for fmt in expected)
        gone = sorted(expected - carried)
        if gone:
            missing_total += len(gone)
            units_missing.append(
                # An empty entry is named as such rather than by one of the
                # cells it lost: it is the shape a truncating write or a
                # dropped merge produces, and the unit is present exactly as
                # the roster demands.
                f"{name}@<entry empty>" if not carried else f"{name}@{gone[0]}")
        outside = carried - expected
        invented = []
        outside_joints = []
        for fmt in sorted(outside):
            if (name, fmt) in bound_joints:
                outside_joints.append(fmt)
                joint_extra += 1
            else:
                invented.append(fmt)
        if outside_joints and not accept_joint_cells_outside_plan:
            joints = ", ".join(f"{name}@{fmt}" for fmt in outside_joints[:5])
            raise ScopeRefused(
                f"{label}: the cost artifact carries {len(outside_joints)} "
                f"joint-priced cell(s) the plan's own cost table never priced "
                f"(e.g. {joints}). A joint row is bound to the coordinate it "
                "names and to nothing else -- it is not compared with this "
                "plan's draw, capture or candidate menu -- so a pass over a "
                "wider roster would place this campaign's table beside prices "
                "it never priced. Seal a requested roster that names them, or "
                "pass --accept-joint-cells-outside-plan to reuse the artifact "
                "as unverified.")
        if invented:
            extra_total += len(invented)
            units_extra.append(f"{name}@{invented[0]}")
    if missing_total or extra_total:
        # Bounded, like every other refusal here: a campaign has tens of
        # thousands of units, and naming each one floods the log the operator
        # has to read. The totals are the contract; the sample is what makes
        # them checkable.
        missing_units, extra_units = len(units_missing), len(units_extra)
        units_missing, units_extra = units_missing[:5], units_extra[:5]
        raise ScopeRefused(
            f"{label}: the cost artifact is not the plan's priced surface -- it "
            f"does not carry {missing_total} planned (unit, format) cell(s) "
            f"across {missing_units} unit(s) (e.g. "
            f"{', '.join(units_missing) or 'none'}) and carries {extra_total} "
            f"cell(s) the plan's cost table never priced across "
            f"{extra_units} unit(s) (e.g. "
            f"{', '.join(units_extra) or 'none'}). The stage states its "
            "acceptance against the requested cells, so an artifact that drops "
            "one unit's cell while another unit still carries that format would "
            "narrow the denominator instead of filling the hole.")
    unnamed = sorted(planned_formats - set(wanted))
    absent = sorted(set(wanted) - planned_formats)
    if unnamed or absent:
        raise ScopeRefused(
            f"{label}: --formats {sorted(wanted)} is not the menu the plan "
            f"prices -- the plan's cost table holds {unnamed} that are unnamed "
            f"and none of {absent}. The requested set is the denominator the "
            "coverage gate reads, so it is bound to the plan's cells rather "
            "than narrowed to the ones that already have a price.")
    record = {"requested_cells": len(cells),
              "requested_units": len(roster),
              "formats": sorted(wanted),
              # Joint rows a joint pass added beyond the plan's own table. They
              # are not part of the requested roster, and they are accepted only
              # when the caller asked for that reuse explicitly: the flag is
              # recorded here so an artifact holding cells this campaign never
              # priced carries that fact into every receipt that stamps this
              # record rather than reading as the plan's own surface.
              "joint_cells_outside_plan": joint_extra,
              "joint_cells_outside_plan_accepted_unverified": bool(
                  joint_extra and accept_joint_cells_outside_plan),
              "roster_sha256": canonical_json_sha256(
                  {"units": sorted(roster), "cells": sorted(cells)},
                  where=f"{label} requested roster")}
    return record, frozenset(cells)


def partition_rows_by_fit(row_memory_gb: "dict[str, int]", per_box: int,
                          budget) -> "tuple[list[str], list[dict]]":
    """Split the planned rows into the ones a box holds and the ones it does not.

    Concurrency is a property of the row's demand, not a flag: PrismaBuild
    admits as many rows as a box's memory holds.  So this checks rather than
    sets -- shrinking a row's declared demand to force co-residency would be
    reserving less than the row holds.

    A row wider than the box is **declined**, not a reason to refuse the
    campaign.  The rows that fit are work the fleet can do now, and the ones
    that do not are a demand to report at the width it was derived at, while
    the limit they name is worked separately.  Returns the admissible row ids
    in plan order and one record per declined row.  A plan with nothing
    admissible refuses: there is no campaign to submit.
    """
    if per_box < 1:
        raise RuntimeError("--rows-per-box must be at least 1")
    widest = max(row_memory_gb.values())
    print(f"[dispatch] widest row demands {widest} GB; "
          f"--rows-per-box {per_box} needs {widest * per_box} GB per box"
          + (f" (spec declares {int(budget)} GB)" if budget is not None else
             " (the spec declares no box budget, so this is unchecked)"))
    admissible: list[str] = []
    declined: list[dict] = []
    for row_id, mem_gb in row_memory_gb.items():
        if budget is None or int(mem_gb) * per_box <= int(budget):
            admissible.append(row_id)
            continue
        declined.append({
            "row_id": row_id, "mem_gb": int(mem_gb), "rows_per_box": per_box,
            "box_memory_gb": int(budget),
            "reason": (f"demands {int(mem_gb)} GB, and --rows-per-box "
                       f"{per_box} needs {int(mem_gb) * per_box} GB, over the "
                       f"{int(budget)} GB box the spec declares"),
        })
    if not admissible:
        raise RuntimeError(
            f"--rows-per-box {per_box} fits no planned row: the widest row "
            f"demands {widest} GB and the spec declares a {int(budget)} GB "
            f"box, so at most {int(budget) // widest} of these rows are "
            "co-resident. Reduce --groups-per-row, or make the quantum hold "
            "less than the whole checkpoint.")
    print(f"[dispatch] {len(admissible)} of {len(row_memory_gb)} rows are "
          f"admissible, {len(declined)} declined")
    for record in declined:
        print(f"[dispatch]   {record['row_id']} {record['reason']}")
    return admissible, declined


#: The quiet a pricing row is allowed in each phase, in the order it walks
#: them. Chosen with margin from a least-squares fit of ``elapsed_s`` against
#: committed batches over the 23 completed 864-unit GLM pricing rows in the
#: fleet's terminal records (2026-09-10) gives 18.6728 s of wall clock per
#: committed batch and an 836.1 s non-pricing intercept; the greatest absolute
#: residual is 160.5 s. The retained-record extraction and its exact row and
#: terminal hashes are committed in ``docs/measurements/pq480_progress_grace_fit_2026-09-10.md``.
#:
#: So: ``pricing`` permits 48.2 fitted commit intervals, and ``startup`` and
#: ``finalize`` each exceed the fitted non-pricing interval including its
#: greatest residual (996.6 s). The sum, 6300 s, is the longest a row can
#: run having committed nothing -- less than half the 14,400 s that killed
#: row-0050 and row-0065 while they were committing anchors every 18 s.
#:
#: There is no flag to override this, on purpose: the number is a measurement
#: of one workload and a flag would invite a guess.  A campaign whose rows
#: measurably behave differently edits its planned manifest -- ``plan`` writes
#: ``progress_phases`` into every row and ``submit`` reads it back -- or
#: re-fits this constant against its own terminal records.
CAMPAIGN_PROGRESS_PHASES = (("startup", 3600), ("pricing", 900), ("finalize", 1800))


def _row_is_bounded(argv: list[str]) -> bool:
    """Whether this row runs the bounded capture path.

    One predicate, two readers: the row builder that merges the bounded
    capture environment, and the container validation that holds a spec to the
    SAME environment. A legacy row that declares its own purge delay is not a
    bounded row and is not refused for a contract it does not fall under.
    """
    policy_flag = '--streaming-capture-policy'
    # ``--streaming-capture-policy`` takes a value, and a malformed argv that
    # ends on the flag has no next element: reading it blindly raised
    # IndexError, which is a crash rather than the refusal a malformed flag
    # deserves. A flag with no value is not the bounded policy.
    index = argv.index(policy_flag) if policy_flag in argv else None
    named = (argv[index + 1] if index is not None and index + 1 < len(argv)
             else None)
    bounded = (policy_flag + '=shared-inputs-bounded-v1' in argv
               or named == 'shared-inputs-bounded-v1')
    return bounded or all(flag in argv for flag in
        ('--streaming', '--units', '--calibration-cache', '--calibration-cache-sha256'))


def _row(spec: dict, argv: list[str], *, mem_gb: int, timeout_s: int | None,
         progress_phases: tuple[tuple[str, int], ...] = CAMPAIGN_PROGRESS_PHASES,
         module: str = "prismaquant.tessera_campaign",
         row_class_name: str = DEFAULT_ROW_CLASS) -> dict:
    """One PrismaBuild row, built for one class of the spec.

    The class supplies the interpreter, the environment, the placement tags,
    the CPU count and the container; the default class supplies exactly the
    spec-level values, so a row built for it is byte-identical to the row this
    function returned before classes existed.  Nothing about the class is
    written INTO the row: ``submit`` re-run is the resume, a finished row is a
    CAS hit on its action key, and a key that moved is a row that re-runs.
    Nothing records the class either: every subcommand today builds the
    ``default`` class, so there is no second class in any plan to record, and
    the first row kind that is built for another one records it in its own
    plan entry when it lands.
    """
    resolved = row_class(spec, row_class_name)
    if resolved["weights_only"]:
        if module in WEIGHTS_ONLY_FORBIDDEN_MODULES:
            raise RowClassRefused(
                f"row class {row_class_name!r} is weights-only and {module} is "
                "not: its census row runs a calibration forward over the whole "
                "scope, its capture row writes the calibration cache, and its "
                "pricing rows fit and consume a Hessian, whose wire is not "
                "bit-comparable across instruction sets (RobTand/tessera#472)")
        named = [flag for flag in HESSIAN_AWARE_FLAGS
                 if any(arg == flag or arg.startswith(flag + "=")
                        for arg in argv)]
        if named:
            raise RowClassRefused(
                f"row class {row_class_name!r} is weights-only and this row's "
                f"argv names {named}; those bytes depend on a Hessian this "
                "class may not have measured or adopted")
    env = dict(resolved['env'])
    bounded = _row_is_bounded(argv)
    if bounded:
        from prismaquant.autoscale import BOUNDED_CAPTURE_ENV, require_bounded_capture_environment
        env = {**BOUNDED_CAPTURE_ENV, **env}
        require_bounded_capture_environment(env)
    command = [resolved["python"], "-u", "-m", module, *argv]
    container_image = None
    if "container" in resolved:
        container_spec = {"container": resolved["container"], "env": env}
        validate_container(container_spec, bounded=bounded)
        command = ["python3", "-m", "tools.tessera_campaign_container", "--spec",
                   json.dumps(container_spec, sort_keys=True), "--", *command]
        # The class owns the image: whatever container this row resolved runs
        # is what PrismaBuild must find on the claiming box before the claim
        # (RobTand/prismabuild#714).  ``None`` for an archive-backed class --
        # its loader establishes the image inside the action, so no
        # local-presence prerequisite may gate placement.
        container_image = admission_image_reference(container_spec)
    row = {
        "argv": command,
        "cwd": spec["cwd"],
        "demand": {"gpu": 1, "cpu": int(resolved["cpus"]), "mem_gb": int(mem_gb)},
        "env": env,
        "tags": list(resolved["tags"]),
        # A row is one memoized action and a retry re-runs the same argv over
        # the same checkpoint, which is exactly what the journal is for.  The
        # policy is sealed into the action key, so it is spelled even though
        # pbcampaign submits every row detached and cannot retry one itself.
        "retry_safe": True,
    }
    if container_image is not None:
        row["container_images"] = [container_image]
    if progress_phases:
        # What bounds this row is whether it is still committing anchors, not
        # how long it has been running.  ``tessera_campaign`` reports each
        # journal flush through ``prismaquant.prismabuild_progress``; PB then
        # applies no total-duration limit while the count advances, and ends
        # the row within the declared allowance when it stops.
        row["progress_phases"] = [f"{name}={grace}" for name, grace in progress_phases]
    if timeout_s is not None:
        # Only when somebody asked for one.  A blanket default here is what
        # sealed 14,400 s into every pricing row and killed two of them mid
        # round (PB #480); the ceiling a row needs is not a property of the
        # dispatcher.
        row["timeout_s"] = int(timeout_s)
    return row


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------

def cmd_census(args) -> int:
    spec = load_spec(Path(args.spec))
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    census_path = workspace / "census.json"
    manifest = workspace / "census-manifest.json"
    row = _row(
        spec,
        ["--model", spec["model"],
         "--out", str(workspace / "census-unused.pkl"),
         "--cache-dir", str(workspace / "census-cache"),
         "--census-out", str(census_path),
         *spec["campaign_argv"]],
        mem_gb=_row_memory_gb(spec, [], {}),
        timeout_s=int(args.timeout_s),
        # Census exits before the pricing journal/reporter exists.  Its
        # explicit wall-clock deadline is the only bound it declares.
        progress_phases=(),
    )
    manifest.write_text(json.dumps([row], indent=2) + "\n")
    if '--streaming' in spec['campaign_argv']:
        (workspace/'census-resources.json').write_text(json.dumps(
            _streamed_resource_plan(spec, {}, []), indent=2, sort_keys=True)+'\n')
    print(f"[dispatch] census manifest {manifest}")
    if args.submit:
        return _pbcampaign(manifest, wait_s=args.wait_s,
                           receipts=workspace / "census-receipts.json")
    return 0


def cmd_capture(args) -> int:
    """Submit one dependent full-scope capture through the existing PB adapter."""
    spec = load_spec(Path(args.spec))
    workspace = Path(args.workspace)
    census_path = workspace / "census.json"
    census = json.loads(census_path.read_text())
    if census.get("model") != spec["model"]:
        raise RuntimeError("capture census and spec name different models")
    manifest = workspace / "capture-manifest.json"
    row = _row(spec, ["--model", spec["model"],
        "--out", str(workspace / "capture-unused.pkl"),
        "--cache-dir", str(workspace / "capture-cache"),
        "--calibration-census", str(census_path),
        "--capture-calibration-out", str(workspace / "calibration-cache"),
        *spec["campaign_argv"]],
        mem_gb=_row_memory_gb(spec, sorted(census["counts"]), census),
        timeout_s=int(args.timeout_s),
        # Capture is likewise not an anchor-pricing row and makes no durable
        # anchor-counter reports.
        progress_phases=())
    manifest.write_text(json.dumps([row], indent=2) + "\n")
    if '--streaming' in spec['campaign_argv']:
        (workspace/'capture-resources.json').write_text(json.dumps(
            _streamed_resource_plan(spec, census, sorted(census['counts'])),
            indent=2, sort_keys=True)+'\n')
    if args.submit:
        return _pbcampaign(manifest, wait_s=args.wait_s,
                           receipts=workspace / "capture-receipts.json")
    return 0


def _calibration_cache_binding(path, census_path):
    from prismaquant.tessera_calibration_cache import require_capture_contract, sha256
    if not path:
        return None
    path = Path(path).resolve()
    capture = require_capture_contract(path)
    if capture["identity"].get("census_sha256") != sha256(census_path):
        raise RuntimeError("planning requires a complete capture bound to this census")
    return dict(path=str(path), sha256=sha256(path))


def _pbcampaign(manifest: Path, *, wait_s: int, receipts: Path | None = None) -> int:
    """Run the campaign and keep the fleet's own row table.

    The table is what says a row *ran*, as opposed to having been accepted:
    every row reports a key, the host it executed on and its exit status, and
    ``merge`` refuses without it.  A submission acknowledgement is not a result.
    """
    # No ``--transport``: the fleet's own default carries these rows, and the
    # rows say what they need.  Every row declares progress phases, which
    # ``pbcampaign`` refuses at manifest load on SLURM because the stall
    # watchdog is the pull-queue worker's.  Pinning ``--transport pool`` here
    # would instead submit into a queue a cut-over fleet might not drain; the
    # refusal is the outcome we want, and it names the reason.
    command = [sys.executable, str(PBCAMPAIGN), "--wait-s", str(wait_s), str(manifest)]
    print("[dispatch] " + " ".join(command), flush=True)
    completed = subprocess.run(command, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout, flush=True)
    if receipts is not None:
        rows = _parse_row_table(completed.stdout)
        receipts.write_text(json.dumps(
            {"manifest": str(manifest), "returncode": completed.returncode,
             "rows": rows}, indent=2) + "\n")
        print(f"[dispatch] {len(rows)} row receipts -> {receipts}")
    return completed.returncode


def _parse_row_table(text: str) -> list[dict]:
    """The ``key status transport job host elapsed rc receipt note`` table.

    Read by the header's own column offsets rather than by splitting on
    whitespace: ``pbwait`` left-justifies every cell to a common width, and a
    cell can hold a space -- ``rc`` renders ``1 (action 137)`` when the
    launcher's status and the action's differ, which is exactly the failing
    row a whitespace split would drop.
    """
    rows: list[dict] = []
    header: list[tuple[str, int, int]] | None = None
    for line in text.splitlines():
        if header is None:
            if line.split()[:2] != ["key", "status"]:
                continue
            names = line.split()
            starts = []
            cursor = 0
            for name in names:
                cursor = line.index(name, cursor)
                starts.append(cursor)
                cursor += len(name)
            ends = starts[1:] + [1 << 20]
            header = list(zip(names, starts, ends))
            continue
        if not line.strip():
            continue
        rows.append({name: line[start:end].strip()
                     for name, start, end in header})
    return rows


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def load_probe_h_trace(path) -> dict:
    """Read original packed probe rows, preserving the allocator's multiplier.

    Sampling needs the full per-expert Fisher vector AND packed topology. An
    expanded per-expert probe cannot establish that identity and is refused.
    """
    import pickle

    probe = pickle.loads(Path(path).read_bytes())
    stats = probe.get("stats") if isinstance(probe, dict) else None
    if not isinstance(stats, dict):
        raise RuntimeError(f"--probe {path}: no 'stats' map to read h_trace from")
    return {str(name): row for name, row in stats.items()
            if isinstance(row, dict) and row.get("_packed_experts_module")}


def stack_expert_counts(census, frame) -> dict:
    """Per-expert routed-row counts from the census, summed over projections.

    The census counts every unit's calibration rows, so an expert's size is
    the sum over its projections.  It is a routed-token proxy for ``h_trace``
    and this function never calls it one: the caller records which vector a
    draw was proportional to (``design``, ``sizes.source``), because "we drew
    proportional to counts" and "we drew proportional to Fisher" are different
    designs with different variance arguments, and only one of them needs a
    probe to exist.
    """
    counts = census.get("counts") or {}
    sizes = {}
    for expert, members in sorted(frame.members.items()):
        missing = [m for m in members if m not in counts]
        if missing:
            raise RuntimeError(
                f"{frame.packed_qname}: the census has no row count for "
                f"{missing[0]}; --stack-sample-sizes counts needs every "
                "expert's own count, and a missing one would draw it with "
                "probability zero")
        sizes[str(int(expert))] = float(sum(int(counts[m]) for m in members))
    if not any(value > 0.0 for value in sizes.values()):
        raise RuntimeError(
            f"{frame.packed_qname}: every expert's routed-row count is zero; "
            "there is no size to draw proportional to")
    return sizes


def sample_stack_groups(groups, probe_rows, *, profile, stack_sample: int,
                        seed: int, audit_rate: int, sizes: str = "probe",
                        census=None) -> dict:
    """Draw once per profile-defined packed parameter, across all its roles.

    The same expert IDs and full-frame inclusion probabilities are persisted
    for every projection and rung. The original probe remains the allocator
    input; no per-expert expansion changes its topology or Fisher currency.

    ``sizes`` chooses what the PPS draw is proportional to.  ``probe`` is the
    per-expert Fisher vector and is the default, so a plan written without the
    flag is byte-identical to every plan written before it.  ``counts`` draws
    on the census's per-expert routed-row counts instead, but still requires
    the original packed probe with ``h_trace_per_expert`` to construct and
    validate the full-frame sampling record.  A ``counts`` draw
    declares itself: ``design`` gains a ``_counts`` suffix and the record
    carries the size vector and its digest, so nothing has to infer from an
    inclusion probability which vector produced it.
    """
    from prismaquant.tessera_campaign import (
        STACK_SAMPLE_COUNTS_SUFFIX, STACK_SAMPLE_SIZE_SOURCES,
        audit_subsample, draw_stack_sample, stack_sample_from_probe,
        _validate_stack_sample, selection_stack_samples)

    if sizes not in STACK_SAMPLE_SIZE_SOURCES:
        raise RuntimeError(
            f"--stack-sample-sizes {sizes}: not one of "
            f"{list(STACK_SAMPLE_SIZE_SOURCES)}")
    if sizes == "counts" and census is None:
        raise RuntimeError(
            "--stack-sample-sizes counts needs the census: the per-expert "
            "sizes are its routed-row counts")
    sampled = {}
    for key, members in sorted(groups.items()):
        if not str(key).startswith("s:"):
            continue
        records, drawn, audit, pi = {}, set(), set(), {}
        for name, row in sorted(probe_rows.items()):
            if "s:" + str(row.get("_packed_experts_module")) != key:
                continue
            frame = stack_sample_from_probe(
                name, row, profile, sampled_experts=range(int(row["num_experts"])),
                inclusion_prob={e: 1.0 for e in range(int(row["num_experts"]))},
                seed=seed, design="census")
            _validate_stack_sample(frame)
            if sizes == "counts":
                size_vector = stack_expert_counts(census, frame)
            else:
                size_vector = {str(e): h
                               for e, h in enumerate(frame.h_trace_per_expert)}
            draw = draw_stack_sample(size_vector, stack_sample, seed=seed,
                                     stack=name)
            audit_ids = audit_subsample(draw["units"], rate=audit_rate,
                                       seed=seed, stack=name)
            experts = sorted(int(e) for e in draw["units"])
            # Only fields read by the constructor: JSON-portable values copied
            # exactly from the probe, rather than a second normalized weight.
            probe_row = {
                "_packed_experts_module": frame.packed_experts_module,
                "_packed_param": frame.packed_param,
                "num_experts": frame.num_experts,
                "h_trace": frame.stack_h_trace,
                "h_trace_per_expert": list(frame.h_trace_per_expert),
            }
            records[name] = {
                "probe_row": probe_row, "sampled_experts": experts,
                "inclusion_prob": dict(draw["inclusion_probability"]),
                "seed": seed,
                "design": (draw["method"] if sizes == "probe"
                           else draw["method"] + STACK_SAMPLE_COUNTS_SUFFIX),
                "draw": draw,
                "audit_experts": sorted(int(e) for e in audit_ids),
                # Written only for a non-default size source, so a probe-sized
                # plan stays byte-identical to the ones already on disk.
                **({} if sizes == "probe" else {"sizes": {
                    "source": sizes, "sha256": draw["size_sha256"],
                    "values": dict(size_vector)}}),
            }
            for expert, names in frame.members.items():
                for member in names:
                    pi[member] = draw["inclusion_probability"][str(expert)]
                    if expert in experts:
                        drawn.add(member)
                    if str(expert) in audit_ids:
                        audit.add(member)
        if not records:
            raise RuntimeError(
                f"anchor group {key}: original packed probe rows with "
                "h_trace_per_expert are required; expanded probes cannot price stacks")
        entry = {"key": key, "members": sorted(members),
                 "sampled": sorted(drawn), "audit": sorted(audit),
                 "inclusion_probability": dict(sorted(pi.items())),
                 "stack_samples": records}
        selection_stack_samples({"groups": [entry]}, profile)
        sampled[key] = {k: v for k, v in entry.items() if k not in ("key", "members")}
    return sampled


#: The two selection schemas, spelled here so that ``plan`` stays a CPU-side
#: step: importing ``prismaquant.tessera_campaign`` for two strings would drag
#: torch into a command whose whole job is to write JSON and a manifest.  They
#: are pinned to the campaign's own constants by
#: ``test_the_planner_and_the_campaign_agree_on_the_selection_schemas``, which
#: runs where the package is importable.
UNITS_SCHEMA = "prismaquant.tessera_campaign_units.v1"
UNITS_SCHEMA_V2 = "prismaquant.tessera_campaign_units.v2"


def _seed_workspace_rows(path, *, census, calibration_cache):
    """Bind a previous plan; each matching row keeps its own checkpoint owner."""
    import hashlib
    root = Path(path).resolve()
    plan_path = root/'plan.json'
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    if (plan.get('schema') != PLAN_SCHEMA or plan.get('model') != census['model'] or
            json.loads(Path(plan['census']).read_text()) != census or
            plan.get('calibration_cache') != calibration_cache):
        raise RuntimeError('seed workspace model, census or capture differs')
    rows = {}
    for row in plan['rows']:
        key = tuple(sorted(row['groups']))
        if not key or key in rows:
            raise RuntimeError('seed workspace has empty or duplicate group bundles')
        rows[key] = row
    return rows, {'path': str(root), 'plan_sha256': hashlib.sha256(raw).hexdigest()}


def _seed_for_selection(rows, bundle, selection):
    """Only unchanged group membership and sampling may inherit this journal."""
    import hashlib
    row = rows.get(tuple(sorted(bundle)))
    if row is None or json.loads(Path(row['units']).read_text()) != selection:
        raise RuntimeError('seed workspace selection differs; preserve group bundles and sampling')
    checkpoint = Path(row['dir'])/'cost.anchors.json'
    if not checkpoint.is_file():
        return None
    raw = checkpoint.read_bytes()
    manifest = json.loads(raw)
    return {'checkpoint': str(checkpoint), 'wire_dir': str(Path(row['dir'])/'cache/wire'),
            'manifest_sha256_at_plan': hashlib.sha256(raw).hexdigest(),
            'identity_sha256': manifest['identity_sha256'], 'row_id': row['row_id']}


def cmd_plan(args) -> int:
    spec = load_spec(Path(args.spec))
    workspace = Path(args.workspace)
    census = json.loads((workspace / "census.json").read_text())
    if census.get("model") != spec["model"]:
        raise RuntimeError(
            f"census was taken on {census.get('model')!r}, the spec names "
            f"{spec['model']!r}")
    calibration_cache = _calibration_cache_binding(
        getattr(args, "calibration_cache", None), workspace / "census.json")
    selected_source = '--streaming' in spec['campaign_argv']
    if selected_source and calibration_cache is None:
        raise RuntimeError('streaming anchor rows require a hash-bound complete calibration cache')
    seed_rows, seed_workspace = None, None
    if getattr(args, 'seed_workspace', None):
        if args.seed_checkpoint or args.seed_wire_dir:
            raise RuntimeError('seed workspace is exclusive with a global seed checkpoint/wire directory')
        seed_rows, seed_workspace = _seed_workspace_rows(args.seed_workspace,
            census=census, calibration_cache=calibration_cache)
    groups = census["anchor_groups"]
    if not groups:
        raise RuntimeError("census reports no anchor group to price")

    stack_sample: dict[str, dict] = {}
    if args.stack_sample is not None:
        size_source = getattr(args, "stack_sample_sizes", "probe") or "probe"
        if not args.probe:
            raise RuntimeError(
                "--stack-sample needs --probe: the stack row's currency is the "
                "packed probe's h_trace, whatever the draw is proportional to")
        from prismaquant.model_profiles import detect_profile
        stack_sample = sample_stack_groups(
            groups, load_probe_h_trace(args.probe), profile=detect_profile(spec["model"]),
            stack_sample=int(args.stack_sample), seed=int(args.stack_sample_seed),
            audit_rate=int(args.audit_rate), sizes=size_source, census=census)
        priced = sum(len(entry["sampled"]) for entry in stack_sample.values())
        frame = sum(len(groups[key]) for key in stack_sample)
        print(f"[dispatch] sampled {priced} of {frame} routed expert units "
              f"across {len(stack_sample)} stack(s), "
              f"{sum(len(e['audit']) for e in stack_sample.values())} audited")

    units_dir = workspace / "units"
    ordered = sorted(groups)
    bundles = [ordered[index:index + args.groups_per_row]
               for index in range(0, len(ordered), args.groups_per_row)]

    rows: list[dict] = []
    planned: list[dict] = []
    selection_writes: list[tuple[Path, str]] = []
    for index, bundle in enumerate(bundles):
        row_id = f"row-{index:04d}"
        entries = []
        for key in bundle:
            entry = {"key": key, "members": sorted(groups[key])}
            if key in stack_sample:
                entry.update(stack_sample[key])
            entries.append(entry)
        selection = {
            # A file that samples says so in its schema; one that does not
            # stays byte-identical to what every row before 2026-09-06 read.
            "schema": (UNITS_SCHEMA_V2 if stack_sample else UNITS_SCHEMA),
            "model": spec["model"],
            "layer_stride": census["layer_stride"],
            "groups": entries,
        }
        units_path = units_dir / f"{row_id}.json"
        selection_writes.append((units_path, json.dumps(selection, indent=2, sort_keys=True) + "\n"))
        row_dir = workspace / "rows" / row_id
        members = [name for entry in entries
                   for name in (entry.get("sampled") or entry["members"])]
        argv = [
            "--model", spec["model"],
            "--out", str(row_dir / "cost.pkl"),
            "--cache-dir", str(row_dir / "cache"),
            "--checkpoint", str(row_dir / "cost.anchors.json"),
            "--units", str(units_path),
            "--calibration-census", str(workspace / "census.json"),
            *spec["campaign_argv"],
        ]
        if calibration_cache:
            argv += ["--calibration-cache", calibration_cache["path"],
                     "--calibration-cache-sha256", calibration_cache["sha256"]]
        row_seed = (_seed_for_selection(seed_rows, bundle, selection)
                    if seed_rows is not None else None)
        if row_seed is not None:
            argv += ['--seed-checkpoint', row_seed['checkpoint'],
                     '--seed-wire-dir', row_seed['wire_dir']]
        if args.seed_checkpoint:
            argv += ["--seed-checkpoint", str(args.seed_checkpoint)]
            if args.seed_wire_dir:
                argv += ["--seed-wire-dir", str(args.seed_wire_dir)]
        rows.append(_row(spec, argv,
                         mem_gb=_row_memory_gb(spec, members, census, selected_source=selected_source),
                         timeout_s=(None if args.timeout_s is None
                                    else int(args.timeout_s))))
        planned.append({"row_id": row_id, "groups": bundle, "members": sorted(members),
                        "dir": str(row_dir), "units": str(units_path),
                        **({'seed': row_seed} if row_seed is not None else {}),
                        **({'resources': _streamed_resource_plan(spec, census, members,
                            selected_source=True)} if selected_source else {})})

    # PB alone admits and places these independently retryable rows according
    # to their actual source/capture preparation and resident encoding demand.
    # All this decides is which rows it is handed: a row too wide for the box
    # is declined here rather than submitted for an admission that cannot
    # come, and the rest of the plan goes on being work.
    per_box = int(args.rows_per_box)
    row_memory_gb = {entry["row_id"]: int(row["demand"]["mem_gb"])
                     for entry, row in zip(planned, rows)}
    admissible, inadmissible = partition_rows_by_fit(
        row_memory_gb, per_box, spec.get("box_memory_gb"))
    members_by_row = {entry["row_id"]: entry["members"] for entry in planned}
    for record in inadmissible:
        record["members"] = members_by_row[record["row_id"]]
    admitted = set(admissible)
    for entry in planned:
        entry["admissible"] = entry["row_id"] in admitted

    manifest = workspace / "manifest.json"
    plan = {
        "schema": PLAN_SCHEMA,
        "model": spec["model"],
        # The spec this plan was derived from, so ``check`` and ``submit`` can
        # re-derive every row's demand without being told again. A plan
        # written before this field exists is checked with an explicit
        # ``--spec``.
        "spec": str(args.spec),
        "census": str(workspace / "census.json"),
        "calibration_cache": calibration_cache,
        "manifest": str(manifest),
        "groups_per_row": int(args.groups_per_row),
        "rows_per_box": per_box,
        "row_memory_gb": row_memory_gb,
        # The reservation those demands carry, stated once for the whole plan
        # because it is a per-row constant, with where it came from. Zero
        # means the spec declared none, and every row's phase plan records the
        # same thing in its own ``baseline_policy``, so a reader cannot
        # mistake an absent reservation for a covered one.
        "process_baseline_bytes": _process_baseline(spec)[0],
        "process_baseline_policy": _process_baseline(spec)[1],
        # The other term outside the phase deltas: the margin the row's own
        # guard holds back from the cap. Recorded because a demand that does
        # not carry it is admitted and then refused.
        "guard_margin_bytes": _guard_margin_bytes(),
        # The rows the manifest does not hold, at the demand they were derived
        # at. A reader of the plan sees the whole layout; a reader of the
        # manifest sees only what was submitted.
        "inadmissible_rows": inadmissible,
        **({'seed_workspace': seed_workspace} if seed_workspace is not None else {}),
        "seed_checkpoint": (None if not args.seed_checkpoint
                            else str(args.seed_checkpoint)),
        # The draw itself, whole: which experts stand for their stack, under
        # what inclusion probability, from which probe and which seed. It is
        # here as well as in every units file because the plan is the thing a
        # reader audits, and an estimate built on a sample is only checkable
        # against the pi it was drawn under.
        "stack_sample": {
            "size": (None if args.stack_sample is None else int(args.stack_sample)),
            "seed": int(args.stack_sample_seed),
            "audit_rate": int(args.audit_rate),
            "probe": (None if not args.probe else str(args.probe)),
            # Which per-expert vector the draw was proportional to, written
            # only when it is not the probe's Fisher vector -- so a plan made
            # without the flag is byte-identical to the ones already on disk,
            # and an absent field means ``probe`` exactly as an absent
            # ``sizes`` block on a record does.
            **({} if (getattr(args, "stack_sample_sizes", "probe") or "probe")
               == "probe" else {"sizes": args.stack_sample_sizes}),
            "stacks": stack_sample,
        },
        "rows": planned,
    }

    # Every published row names the bytes PrismaBuild's prewarm loop warms it
    # from, so a subset copied out of manifest.json carries them too. Only the
    # producer knows a row's read set, and a row whose read set cannot be
    # derived refuses the plan here.
    published_rows, manifest_writes = planned_data_manifests(
        workspace, plan,
        {str(path): json.loads(text) for path, text in selection_writes},
        [row for entry, row in zip(planned, rows) if entry["admissible"]])

    # A refused fit check, or a row with no derivable read set, must not
    # rewrite selections still named by an existing published manifest.
    # Derive every row and every data manifest before publishing bytes.
    units_dir.mkdir(parents=True, exist_ok=True)
    for units_path, selection_text in selection_writes:
        units_path.write_text(selection_text)
    for path, blob in manifest_writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    manifest.write_text(json.dumps(published_rows, indent=2) + "\n")
    (workspace / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"[dispatch] planned {len(rows)} rows over {len(ordered)} anchor "
          f"groups, {len(admissible)} submitted -> {manifest}")
    return 0


def _checked_manifest(args, *, manifest: Path) -> list:
    """Re-derive every row's demand from its own argv, or refuse to go on.

    A manifest is an editable file and the plan that wrote it is not
    authoritative over what it now says. So the check reads the rows as they
    stand: a hand-edited argv, a hand-edited ``mem_gb``, or a plan term that
    moved since are all the same question, asked of the bytes about to be
    submitted.
    """
    workspace = Path(args.workspace)
    plan_path = workspace / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    spec_path = getattr(args, "spec", None) or plan.get("spec")
    if not spec_path:
        raise DemandRefused(
            f"{manifest} cannot be checked: neither --spec nor a 'spec' field "
            f"in {plan_path}. Pass the spec these rows were planned from.")
    spec = load_spec(Path(spec_path))
    census_path = getattr(args, "census", None) or plan.get("census") or (
        workspace / "census.json")
    census = json.loads(Path(census_path).read_text())
    box_memory_gb = getattr(args, "box_memory_gb", None)
    if box_memory_gb is None:
        box_memory_gb = spec.get("box_memory_gb")
    rows = json.loads(Path(manifest).read_text())
    records = verify_manifest_demands(spec, census, rows,
                                      box_memory_gb=box_memory_gb)
    for record in records:
        print(f"[dispatch] {record['row']} demands {record['declared_mem_gb']} "
              f"GiB, derives {record['mem_gb']} GiB "
              f"(plan {record['plan_bytes']} B, baseline "
              f"{record['process_baseline_bytes']} B, margin "
              f"{record['guard_margin_bytes']} B)")
    return records


def cmd_check(args) -> int:
    """Recompute every manifest row's demand and refuse an under-declared one."""
    manifest = Path(args.manifest) if getattr(args, "manifest", None) else (
        Path(args.workspace) / "manifest.json")
    records = _checked_manifest(args, manifest=manifest)
    print(f"[dispatch] {len(records)} rows in {manifest} declare a demand "
          "their own argv supports")
    rows = json.loads(Path(manifest).read_text())
    require_data_manifests(rows, where=manifest)
    print(f"[dispatch] {len(rows)} rows in {manifest} name a data manifest "
          "PrismaBuild can warm them from")
    return 0

#: Where ``plan`` writes each row's read set, and the manifest ``submit`` hands
#: the fleet.  ``submit`` re-derives the read sets and refuses one that no
#: longer matches the bytes ``plan`` published.
DATA_MANIFEST_DIR = "data-manifests"
SUBMITTED_MANIFEST = "manifest.submitted.json"


class DataManifestRefused(RuntimeError):
    """A row would reach PrismaBuild without the read set it is warmed from."""


def _manifest_producer():
    """The campaign's data-manifest producer, imported from ``experiments/``.

    It is imported here rather than at module load because it reads the
    campaign's plan and capture manifest, which only ``plan``, ``check`` and
    ``submit`` need.
    """
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments import glm_data_manifests

    return glm_data_manifests


def _derived_data_manifests(workspace: Path, rows: list[dict], campaign, *,
                            out_dir: Path) -> list[tuple[str, Path, bytes]]:
    """Each row's id, manifest path and manifest bytes, or a refusal.

    Writes nothing, so a caller can refuse the whole set before any byte of it
    is published.
    """
    producer = _manifest_producer()
    provenance = producer.deterministic_provenance(
        str(workspace), campaign, "stat")
    derived = []
    for index, row in enumerate(rows):
        row_id = producer.row_id_of(row)
        if row_id is None:
            raise RuntimeError(
                f"row {index} names no single units/row-XXXX.json in its argv, "
                "so its read set cannot be derived; refusing to submit it "
                "without a data manifest")
        if row_id not in campaign.rows:
            raise RuntimeError(
                f"{row_id} is not a row of {workspace}/plan.json")
        manifest = producer.build_manifest(
            campaign, row_id, provenance, row.get("argv"))
        blob = producer.check_manifest_bytes(
            json.dumps(manifest, indent=1, sort_keys=False).encode() + b"\n",
            where=row_id)
        derived.append((row_id, out_dir / f"{row_id}.data-manifest.json", blob))
    return derived


def planned_data_manifests(workspace: Path, plan: dict,
                           selections: dict[str, dict], rows: list[dict]
                           ) -> tuple[list[dict], list[tuple[Path, bytes]]]:
    """The rows ``plan`` publishes, each naming its manifest, and the bytes to write.

    Derived from the plan and the selections ``plan`` is about to write, not
    from disk: until the whole set is derived nothing is published, and the
    files on disk may still be the previous plan's.  ``argv`` and ``demand``
    are the row's action identity and are returned byte-identical; the only
    change to a row is ``data_manifest``, appended last.
    """
    producer = _manifest_producer()
    campaign = producer.Campaign(str(workspace), plan=plan,
                                 selections=selections)
    derived = _derived_data_manifests(workspace, rows, campaign,
                                      out_dir=workspace / DATA_MANIFEST_DIR)
    return ([{**row, "data_manifest": str(path)}
             for row, (_, path, _) in zip(rows, derived)],
            [(path, blob) for _, path, blob in derived])


def require_data_manifests(rows: list[dict], *, where) -> None:
    """Refuse a manifest any of whose rows PrismaBuild could not warm.

    ``pbcampaign`` submits a row with no ``data_manifest`` without a word, and
    the row then reads its inputs off cold spindles: every partial release of
    ``extension-r1024-02`` and of the 09-15 census did, because each was a
    filtered copy of a ``manifest.json`` planned before ``plan`` attached the
    manifests.  This is the check a subset goes through before it is released,
    so it names every such row at once.  It reads the manifest each row names
    and holds it to the rules PrismaBuild applies, and to belonging to the row
    that names it; it does not re-derive the read set, which ``submit`` does.
    """
    producer = _manifest_producer()
    refusals = []
    for index, row in enumerate(rows):
        label = _row_label(_inner_campaign_argv(row), index)
        named = row.get("data_manifest")
        if not named:
            refusals.append(f"{label}: names no data manifest")
            continue
        try:
            blob = Path(named).read_bytes()
            manifest = producer.check_manifest(
                json.loads(producer.check_manifest_bytes(blob, where=label)),
                where=label)
        except (OSError, ValueError, SystemExit) as error:
            refusals.append(f"{label}: {named}: {error}")
            continue
        owner = manifest["annotations"].get("row_id")
        if owner != producer.row_id_of(row):
            refusals.append(f"{label}: {named} is the read set of {owner}")
    if refusals:
        raise DataManifestRefused(
            f"{len(refusals)} of {len(rows)} rows in {where} would reach "
            "PrismaBuild without a data manifest its prewarm loop can warm "
            "them from:\n" + "\n".join(refusals) + "\nA subset copied from a "
            "manifest.json that plan published carries one; re-plan a "
            "workspace planned before plan attached them.")


def attach_data_manifests(workspace: Path, rows: list[dict], *,
                          out_dir: Path | None = None) -> list[dict]:
    """Give every row the byte list PrismaBuild needs to warm it, or refuse.

    Only the producer knows a row's read set: the capture files its members
    name, the byte extents of those members' weights inside the safetensors
    shards, and the seed wire the row's own argv points at.  Without that list
    a row is invisible to the fleet's prewarm loop and starts against cold
    spindles -- measured at 26 MB/s over 64 GB on sparky (row-0074,
    2026-09-12), about 40 minutes of idle GPU per row.

    The manifest is a ``pbrun`` input, not part of the campaign's own
    checkpoint identity, so the row's ``argv`` is returned byte-identical to
    what ``plan`` wrote; only the ``data_manifest`` key is added.  A row whose
    manifest cannot be built is refused here, where the reason is readable,
    rather than submitted blind.

    ``plan`` already attached a manifest to every row it published, so for
    those rows this is a re-derivation: the row must name the path derived
    now, and the file there must hold exactly the bytes derived now.  A row
    that does not is refused rather than rewritten, because a subset of the
    plan may already have sealed the published bytes into an action key, and
    the same row submitted under different bytes would run twice.  A row
    planned before ``plan`` attached manifests names none, and gets one here.
    """
    producer = _manifest_producer()
    campaign = producer.Campaign(str(workspace))
    out_dir = out_dir or workspace / DATA_MANIFEST_DIR
    derived = _derived_data_manifests(workspace, rows, campaign, out_dir=out_dir)

    drifted = []
    for row, (row_id, path, blob) in zip(rows, derived):
        named = row.get("data_manifest")
        if named is None:
            continue
        if named != str(path):
            drifted.append(f"{row_id}: names {named}, derives {path}")
            continue
        try:
            current = Path(named).read_bytes()
        except OSError as error:
            drifted.append(f"{row_id}: {named} is unreadable: {error}")
            continue
        if current != blob:
            drifted.append(f"{row_id}: {named} differs from the read set "
                           "derived now")
    if drifted:
        raise DataManifestRefused(
            f"{len(drifted)} of {len(rows)} rows name a data manifest that is "
            "not the one their read set derives now:\n" + "\n".join(drifted)
            + "\nThe row's inputs or the producer changed since plan. "
            "Submitting under new bytes would re-key rows a released subset "
            "may already run; re-plan the workspace instead.")

    out_dir.mkdir(parents=True, exist_ok=True)
    attached: list[dict] = []
    for row, (_, path, blob) in zip(rows, derived):
        if row.get("data_manifest") is None:
            path.write_bytes(blob)
        attached.append({**row, "data_manifest": str(path)})
    return attached


def cmd_submit(args) -> int:
    workspace = Path(args.workspace)
    manifest = workspace / "manifest.json"
    # An under-declared row is admitted and then refused by its own guard
    # about twenty seconds in, which PrismaBuild records as failed with no
    # retry (RobTand/prismaquant#522). Nothing about that is cheaper to find
    # out later, so the demands are re-derived before any row is submitted.
    # This reads the planned rows, and re-deriving their manifests below
    # changes neither ``argv`` nor ``demand``, so what is checked is what is
    # sent.
    _checked_manifest(args, manifest=manifest)
    rows = attach_data_manifests(workspace, json.loads(manifest.read_text()))
    submitted = workspace / SUBMITTED_MANIFEST
    submitted.write_text(json.dumps(rows, indent=2) + "\n")
    plural = "" if len(rows) == 1 else "s"
    print(f"[dispatch] data manifests attached to {len(rows)} row{plural} "
          f"-> {submitted}")
    # Re-running the manifest IS the resume: a finished row is a cache hit and
    # a running row is re-attached, both by pbcampaign itself.  The manifests
    # are a deterministic function of the campaign and the tree, so a second
    # submit addresses the same action keys as the first.
    return _pbcampaign(submitted, wait_s=args.wait_s,
                       receipts=workspace / "receipts.json")



# ---------------------------------------------------------------------------
# post-campaign GPU submissions
# ---------------------------------------------------------------------------

#: The published PrismaBuild client. ``pbrun`` snapshots the checkout and
#: places the action; it is not vendored here so a stale copy cannot become a
#: stale submission.
PBRUN = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py")

JOINT_ENTRY_POINT = "prismaquant.tessera_joint_aura"
ALLOCATION_ENTRY_POINT = "prismaquant.tessera_joint_allocation"
AQUA_ENTRY_POINT = "prismaquant.aqua_activation_cost"
EXPORT_ENTRY_POINT = "tessera.experiments.export_tessera_serving"


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_sha256(path: Path, declared: str | None, *, label: str) -> str:
    """The file's digest, checked against what the caller declared.

    The entry point binds every input by digest, so a submission that names a
    different one fails twenty seconds into an admitted action rather than
    here. Computing it costs one read of a file the submitter already has.

    In dev mode (the default; PQ #1147) a declared digest is a RECORD, not a
    gate: ``seal_check`` prints the mismatch and the ACTUAL digest is used.
    The pass's own gates -- not the submitter -- decide what runs, and a dev
    submission's argv already names the environment it ran under.
    """
    actual = _sha256_of(path)
    if declared is not None:
        from prismaquant.dev_mode import seal_check
        seal_check(f"{label} declared digest", declared, actual, where=str(path),
                   refusal=lambda: RuntimeError(
                       f"{label}: {path} hashes to {actual}, not the declared {declared}"))
    return actual


def _pbrun_argv(args, *, manifest: Path, inner: list[str],
                progress_phases=(), gpu_memory_gb=None, container_spec=None,
                residency=None) -> list[str]:
    """The submission command, with ``--data-manifest`` before ``--detach``.

    Everything after ``--`` is the action; ``--data-manifest`` is an option of
    ``pbrun`` itself, so it has to precede the separator. ``pbrun`` validates
    the manifest, ingests it as a second content-addressed input and seals its
    summary into the action, which is also why ``produced_by`` carries nothing
    run-specific: the manifest's digest is part of the action key.

    The container image the spec resolves is declared the same way
    (``--container-image``, also before the separator): it is derived from the
    SAME parsed document that is serialized into ``--spec``, so the row cannot
    seal one image and be admitted against another, and an archive-backed spec
    declares nothing because its launcher loads and verifies the image inside
    the action (RobTand/prismabuild#714).
    """
    if container_spec is None:
        spec = Path(args.spec).read_text()
        parsed_spec = json.loads(spec)
    else:
        parsed_spec = container_spec
        spec = json.dumps(container_spec, sort_keys=True)
    container_image = admission_image_reference(parsed_spec)
    argv = ["python3", str(args.pbrun), "--demand", args.demand]
    if gpu_memory_gb is not None:
        # The device envelope is a *subset* of the unified reservation on
        # GB10, not a second box: without this PB caps the GPU subset at the
        # whole reservation and the plan's own device bound is not enforced.
        argv += ["--gpu-memory-gb", str(gpu_memory_gb)]
    if args.cpus is not None:
        argv += ["--cpus", str(args.cpus)]
    if args.tag:
        argv += ["--tag", args.tag]
    if residency is not None:
        # A pbrun option, so it precedes the ``--`` separator like the manifest
        # does. PB derives the tier demand from the manifest this same command
        # seals, which is why nothing here names bytes or a tier. It is a
        # parameter rather than a read of ``args`` because only the caller
        # knows which entry point is being submitted, and only one of them
        # reads through the stage.
        argv += ["--residency", str(residency)]
    argv += ["--priority", str(args.priority)]
    if args.timeout_s is not None:
        argv += ["--timeout-s", str(args.timeout_s)]
    head_grace = getattr(args, "head_grace_s", None) or 1800
    for name in progress_phases:
        # The allowance bounds an uncommitted unit, not an arbitrary number of
        # logging lines. The head walk reports one unit per resolved anchor
        # roster entry since #678 -- for a run as well as a prepare since #741,
        # where that walk measurably took 4 h 14 min of the 6 h 30 min before
        # the first capture line on ``ad8803aa`` -- so the allowance no longer
        # has to cover a silent pass over the whole roster. It still covers
        # what stays unreported inside `head`: whole-source authentication,
        # streamed model construction and, on a run, the production cache's
        # unpickle between the last roster entry and the first boundary write. Their cost is a property of the plan and the
        # box, so the allowance stays a caller input rather than a constant
        # sized from one run.
        argv += ["--progress-phase",
                 f"{name}={head_grace if name == 'head' else 900}"]
    if container_image is not None:
        argv += [CONTAINER_IMAGE_FLAG, container_image]
    argv += ["--data-manifest", str(manifest), "--detach", "--",
             "python3", "-m", "tools.tessera_campaign_container"]
    argv += list(args.container_arg or [])
    argv += ["--spec", spec, "--", *inner]
    return argv


def _declared_phase_names(manifest: dict) -> tuple[str, ...]:
    """The phase names of a sealed read set, for the progress policy.

    One reader for both joint commands, because the property that matters is
    that the names the action may report are the names the residency plan
    carries: ``residency_plan.remaining`` reads a name its plan does not hold
    as "the consumer has passed nothing", so a second list built any other way
    is a window that never advances.
    """
    names = tuple(row["name"] for row in manifest["annotations"]["phases"])
    if len(names) > 2048:
        raise RuntimeError("joint read plan exceeds 2048 sealed PB phases")
    return names


def _manifest_path(args, plan: dict, *, entry_point: str, command: str) -> Path:
    """Where the manifest is written.

    The brief's ``<output-root>/data-manifests`` is the default, but a plan's
    output root can be a frozen campaign directory this tool may not write to,
    so ``--manifest-dir`` names somewhere else. A dry run never writes, and
    names the path it would have used.
    """
    if args.manifest_dir:
        directory = Path(args.manifest_dir)
    else:
        root = plan.get("output_root")
        if not root:
            raise RuntimeError(
                "the plan declares no 'output_root'; pass --manifest-dir")
        directory = Path(root) / DATA_MANIFEST_DIR
    extension = ".json.gz" if entry_point == JOINT_ENTRY_POINT else ".json"
    return directory / f"{entry_point}.{command}{extension}"


def _submit_gpu_action(args, *, entry_point: str, command: str, inner: list[str],
                       plan: dict, build, scope: dict | None = None) -> int:
    """Build the read set, write it, and run the chain-style pbrun command.

    Order matters for a dry run: the command shape is printed before the
    manifest is built, so a plan whose inputs are not on disk yet still shows
    what would be submitted.
    """
    residency = getattr(args, "residency", None)
    if residency is not None and entry_point != JOINT_ENTRY_POINT:
        # ``--residency stage`` is not advice. It reserves cluster-scoped tier
        # capacity and narrows placement to the boxes that mount the tier, and
        # it is paid for by reads that go through the residency map. Only the
        # joint pass reads through the resolver today
        # (``prismaquant/residency_map.py``, wired into
        # ``ProductionWeightCache._load_file_tensor`` and the Tessera wire
        # reader); allocation, export and AQUA read every byte from the pool.
        # Accepting the flag for them would buy a reservation nothing consumes
        # and hide it, which is the failure this whole change exists to stop.
        raise RuntimeError(
            f"--residency is joint-only; {entry_point}:{command} reads every "
            "byte from the pool, so a stage reservation would narrow its "
            "placement and be read by nothing. Submit it without --residency")
    container_spec = json.loads(Path(args.spec).read_text())
    if (entry_point == JOINT_ENTRY_POINT
            and (plan.get("qualification_window") is not None
                 or plan.get("execution", {}).get("retained_operator_windows") is not None)):
        from tools.tessera_campaign_container import BOUNDED_CAPTURE_ENV, validate_container
        validate_container(container_spec, bounded=True)
        container_spec = {**container_spec,
                          "env": {**BOUNDED_CAPTURE_ENV, **container_spec.get("env", {})}}
    # Dev mode (Rob, 2026-09-19; the default since PQ #1147): the switch
    # travels to the container exactly the way PRISMAQUANT_LAYER_READ_THREADS
    # does -- sealed in the spec env the launcher receives -- so the pass
    # gates inside read the same switch this submitter read. The spec FILE is
    # never rewritten; only the in-memory copy moves. This changes the
    # submission's argv, and therefore its action key: a dev submission is a
    # different action from a certified one over the same plan, by design.
    # A certified submitter (PRISMAQUANT_DEV_MODE=0) seals "0": the container
    # does not inherit this environment, and unset there means dev mode.
    from prismaquant.dev_mode import CERTIFIED_VALUE, dev_mode_enabled
    if dev_mode_enabled():
        from prismaquant.dev_mode import dev_warning
        declared = (container_spec.get("env") or {}).get(DEV_MODE_ENV)
        if declared is not None and declared != "1":
            dev_warning(
                f"spec env {DEV_MODE_ENV}={declared!r} contradicts the "
                "submitting environment's '1'; the submitting environment wins")
        container_spec = {**container_spec,
                          "env": {**(container_spec.get("env") or {}),
                                  DEV_MODE_ENV: "1"}}
        dev_warning(
            "submitting under dev mode (PRISMAQUANT_DEV_MODE is not 0): digest "
            "arguments are records and the pass's provenance gates stamp instead "
            "of refusing; every artifact this produces is dev-uncertified")
    else:
        container_spec = {**container_spec,
                          "env": {**(container_spec.get("env") or {}),
                                  DEV_MODE_ENV: CERTIFIED_VALUE}}
    manifest_path = _manifest_path(args, plan, entry_point=entry_point,
                                   command=command)
    manifest = build()
    if scope is not None:
        # The scope is part of the submission's identity, not a comment beside
        # it: the manifest's bytes are content-addressed into the action key, so
        # a pinned read set that was acknowledged as a diagnostic subset cannot
        # later be re-presented as the campaign's score.
        manifest["annotations"]["campaign_scope"] = scope
    # Derive what this row's own policy says it holds before reserving it, so
    # a reservation below the container cap or below the combined physical
    # bound is refused here rather than admitted and then declined by the row.
    demand_record = verify_joint_submission_demand(
        container_spec, plan, args.demand,
        label=f"{entry_point}:{command}")
    gpu_memory_gb = None
    gpu_bytes = plan.get("max_gpu_bytes")
    if (isinstance(gpu_bytes, int) and not isinstance(gpu_bytes, bool)
            and gpu_bytes > 0):
        gpu_memory_gb = math.ceil(gpu_bytes / 1024 ** 3)
    # ``source_identity_cache_host`` stays in the manifest as provenance -- the
    # box whose manifest build adopted the cached full-source SHA -- and as a
    # placement constraint, because that proof names a mount instance and not
    # just a path. Reusing it from another mount is not authorized by anything
    # this tree can verify (an NFSv4 client's statfs fsid is 0 here), so the row
    # has to run where the proof was made rather than silently re-reading or
    # silently trusting a device-free comparison.
    cache_host = manifest["annotations"].get("source_identity_cache_host")
    if cache_host is not None and args.tag != cache_host:
        raise RuntimeError(
            f"source identity cache proof is local to {cache_host}; "
            f"submit with --tag {cache_host}, not {args.tag!r}")
    # PB accepts one gzip member up to 64 MiB stored / 512 MiB expanded.
    # The full joint read set exceeds the plain limit, so seal its compressed
    # bytes deterministically; the CAS action key includes their SHA256.
    decoded = json.dumps(manifest, separators=(",", ":"), sort_keys=False).encode() + b"\n"
    blob = (gzip.compress(decoded, mtime=0) if manifest_path.suffix == ".gz"
            else decoded)
    phase_names = ()
    if entry_point == JOINT_ENTRY_POINT and command == "run":
        # The run declares the phase table it will read, for the same reason
        # the prepare does and for one more: PrismaBuild's residency window
        # publishes the next phase's movers and releases the finished ones on
        # the consumer's *accepted progress*, matched by phase NAME against
        # this very table (``residency_plan.remaining``). Submitted without it,
        # ``ad8803aa`` staged 19 of 46 phases, egressed 1, and filled a 744 GB
        # stage to 0 B available (RobTand/prismabuild#632). The names come from
        # the manifest this same call seals, so the table the window walks and
        # the names the action may report are one list.
        phase_names = _declared_phase_names(manifest)
    if entry_point == JOINT_ENTRY_POINT and command == "prepare":
        # A resumed submission seals the order it will actually read -- the
        # qualification journal's units in their replay parts, then the layer
        # walk over the rest -- so its phase table is as true as a fresh one
        # and the action can be given it (RobTand/prismaquant#607).
        if (manifest["annotations"].get("phase_start_units")
                and manifest["annotations"].get("source_authentication_mode") ==
                    "verified_streamed_identity_cache"):
            phase_names = _declared_phase_names(manifest)
            inner = [*inner, "--prewarm-manifest", str(manifest_path),
                     "--prewarm-manifest-sha256", hashlib.sha256(blob).hexdigest()]
    if residency is not None:
        # The action has to know which read set it was submitted with, or a
        # residency map composed for another manifest could answer for it. The
        # digest is the manifest's own, the one pbrun seals into the action
        # key, so this adds no fact -- it puts a fact the action could not see
        # where the action can see it. Added only under --residency, so a
        # submission that asks for no stage keeps the argv, and therefore the
        # action key, it has today.
        inner = [*inner, "--data-manifest-sha256", hashlib.sha256(blob).hexdigest()]
    argv = _pbrun_argv(args, manifest=manifest_path, inner=inner,
                       progress_phases=phase_names, residency=residency,
                       gpu_memory_gb=gpu_memory_gb, container_spec=container_spec)
    summary = {
        "entry_point": f"{entry_point}:{command}",
        "data_manifest": str(manifest_path),
        # What was reserved, and what the row's own plan says it holds. The
        # difference is the conservatism, stated rather than implied.
        # ``residency`` rides here because a stage reservation is part of what
        # this row asks the fleet for, not a comment beside it. It does not
        # change the host demand, and ``verify_joint_submission_demand`` is
        # unchanged: the plan's own bound still decides cpu/gpu/mem_gb.
        "resource_demand": {**demand_record,
                            "gpu_memory_gb": gpu_memory_gb,
                            **({"residency": residency}
                               if residency is not None else {})},
        "manifest_bytes": len(blob),
        "decoded_manifest_bytes": len(decoded),
        "manifest_sha256": hashlib.sha256(blob).hexdigest(),
        "entry_count": manifest["entry_count"],
        "total_bytes": manifest["total_bytes"],
        "counts": manifest["annotations"]["counts"],
        "bytes": manifest["annotations"]["bytes"],
        # Cells the campaign adopted rather than encoded: the head decodes a
        # shard from each of those wires, so the number belongs beside the
        # phase table a reader uses to size the warm.
        **{key: manifest["annotations"][key]
           for key in ("renders_absent", "synthesized_render_wire_bytes")
           if key in manifest["annotations"]},
        "phases": manifest["annotations"]["phases"],
    }
    if scope is not None:
        summary["campaign_scope"] = scope
    if args.dry_run:
        print("[dry-run] " + " ".join(shlex.quote(item) for item in argv))
        print(json.dumps(summary, indent=1))
        # Checked after the summary is printed, so a read set the fleet would
        # refuse still reports the size and the phase boundaries that make the
        # refusal readable.
        _manifest_producer().check_manifest_bytes(
            blob, where=f"{entry_point}:{command}")
        return 0
    _manifest_producer().check_manifest_bytes(blob, where=f"{entry_point}:{command}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_bytes(blob)
    os.replace(temporary, manifest_path)
    print(json.dumps(summary, indent=1))
    # Flush before pbrun inherits the descriptors: under redirection this
    # stream is block-buffered, and pbrun's own stderr lines would otherwise
    # land glued to the buffered summary in the caller's log (2026-09-18: a
    # refusal read as a silent exit 1 and was re-run with its text in argv).
    print("[submit] " + " ".join(shlex.quote(item) for item in argv), flush=True)
    return subprocess.run(argv, check=False).returncode


def cmd_submit_joint(args) -> int:
    producer = _manifest_producer()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    plan_sha256 = _bound_sha256(plan_path, args.plan_sha256, label="joint plan")
    # The scope is derived from the plan's own bound census, and the caller has
    # to state what it is submitting *for*. A 16-of-512 diagnostic plan handed
    # to a campaign-scoped submission refuses here rather than after the GPU
    # window has been spent producing a subset the campaign cannot score.
    identity_path = Path(args.campaign_identity).resolve()
    _bound_sha256(identity_path, args.campaign_identity_sha256,
                  label="joint campaign identity")
    campaign = json.loads(identity_path.read_text())
    scope = verify_joint_campaign_scope(
        plan, require_scope=args.require_scope, campaign=campaign,
        label=f"joint {args.command}: {plan_path.name}")
    scope = {**scope, "campaign_identity_sha256": _sha256_of(identity_path)}
    inner = ["python3", "-u", "-m", JOINT_ENTRY_POINT, args.command,
             "--plan", str(plan_path), "--plan-sha256", plan_sha256]
    prepared = None
    if args.command == "run":
        if not args.prepared:
            raise RuntimeError("the run command requires --prepared")
        prepared_path = Path(args.prepared).resolve()
        prepared_sha256 = _bound_sha256(prepared_path, args.prepared_sha256,
                                        label="prepared completion")
        inner += ["--prepared", str(prepared_path),
                  "--prepared-sha256", prepared_sha256]
        prepared = str(prepared_path)
    if args.resume:
        inner += ["--resume"]
    source_transition = None
    if args.source_transition_sha256 and not args.source_transition:
        raise RuntimeError(
            "--source-transition-sha256 names a digest for no receipt; pass --source-transition")
    if args.source_transition:
        # A source transition is a run-only, resume-only admission (the pass
        # refuses it otherwise), so the submission refuses the same shapes here
        # rather than after the GPU window has been reserved.
        if args.command != "run" or not args.resume:
            raise RuntimeError(
                "--source-transition is admitted only by `run --resume`; the pass "
                "loads it as a resume under a receipt that binds the sealed prepare")
        transition_path = Path(args.source_transition).resolve()
        transition_sha256 = _bound_sha256(transition_path, args.source_transition_sha256,
                                          label="source transition receipt")
        inner += ["--source-transition", str(transition_path),
                  "--source-transition-sha256", transition_sha256]
        source_transition = str(transition_path)
    provenance = producer.deterministic_entry_provenance(
        f"{JOINT_ENTRY_POINT}:{args.command}", plan=str(plan_path),
        plan_sha256=plan_sha256,
        workspace=str(Path(plan["inputs"]["campaign_plan"]["path"]).parent))
    return _submit_gpu_action(
        args, entry_point=JOINT_ENTRY_POINT, command=args.command, inner=inner,
        plan=plan, scope=scope,
        build=lambda: producer.build_joint_pass_manifest(
            str(plan_path), command=args.command, produced_by=provenance,
            argv=inner, prepared=prepared, source_transition=source_transition))


def cmd_submit_aqua(args) -> int:
    """Submit the campaign's AQUA stage, with strict per-cell coverage.

    The A-side is a stage of the same campaign, so it is submitted the same way
    the joint pass is: the plan's bound census and the frozen campaign identity
    decide the roster, the read set the action will consume is declared to
    PrismaBuild before it is queued, and the reservation is checked against the
    row's own physical bound rather than taken from habit.

    Two things differ from the joint pass, and both are refusals rather than
    options. ``--require-complete-coverage`` is mandatory: the campaign's
    requested ``(unit, format)`` cells either carry their own activation term
    or the run refuses, because a weight-only cell with a positive surrogate is
    what silently buys 4-bit on a route the lane declares W4A4. And the
    requested roster is the plan's, not the artifact's: the plan binds the
    campaign's own cost table (``inputs.merged_cost``) by sha256, ``--cost-in``
    has to reproduce that table's ``(unit, format)`` cells unit for unit, and
    ``--formats`` has to name exactly the menu it prices. Any of those drifting
    refuses here -- the stage's acceptance is stated against the cells it is
    given, so a narrowed artifact would decide its own coverage.

    A campaign whose cells are ALL already joint-priced is a fulfilled
    artifact, not a nearly-empty run: nothing is submitted and the caller is
    told which payload already carries the A-side.
    """
    producer = _manifest_producer()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    plan_sha256 = _bound_sha256(plan_path, args.plan_sha256, label="joint plan")
    identity_path = Path(args.campaign_identity).resolve()
    _bound_sha256(identity_path, args.campaign_identity_sha256,
                  label="joint campaign identity")
    campaign = json.loads(identity_path.read_text())
    scope = verify_joint_campaign_scope(
        plan, require_scope=args.require_scope, campaign=campaign,
        label=f"aqua {args.require_scope}: {plan_path.name}")
    scope = {**scope, "campaign_identity_sha256": _sha256_of(identity_path)}

    cost_in = Path(args.cost_in).resolve()
    cost_sha256 = _bound_sha256(cost_in, args.cost_in_sha256,
                                label="aqua cost artifact")
    cost_out = Path(args.cost_out).resolve()
    if cost_out == cost_in:
        raise RuntimeError(
            f"--cost-out is --cost-in ({cost_in}); the stage writes a new "
            "payload beside the weight-only one so that arm stays reproducible")
    if cost_out.exists():
        raise RuntimeError(
            f"--cost-out {cost_out} already exists. The stage writes it with a "
            "plain truncating write, so a path left by an interrupted run "
            "cannot be told from a finished one; remove it or name another "
            "path, and do not re-run against an artifact this tool did not "
            "publish.")
    card = Path(args.card).resolve()
    card_sha256 = _bound_sha256(card, args.card_sha256,
                                label="sensitivity card")
    act_dir = str(Path(args.act_dir).resolve()) if args.act_dir else None
    formats = [item.strip() for item in (args.formats or "").split(",")
               if item.strip()]
    if not formats:
        raise RuntimeError(
            "--formats is required: the campaign's menu is the requested set "
            "the coverage gate is stated against, and it is not inferred from "
            "the artifact's own keys")
    with cost_in.open("rb") as handle:
        payload = pickle.load(handle)
    roster, requested = aqua_requested_cells(
        plan, payload, formats, label=f"aqua: {plan_path.name}",
        accept_joint_cells_outside_plan=args.accept_joint_cells_outside_plan)
    costs = payload["costs"]
    # The fulfilled-artifact shortcut reads the same bound set the roster above
    # validated, so a joint row that names another cell can neither short-circuit
    # this submission nor be reported as one of the campaign's priced cells.
    joint_cells = _bound_joint_cells(costs, label=f"aqua: {plan_path.name}")
    summary = {
        "entry_point": f"{AQUA_ENTRY_POINT}:{args.require_scope}",
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "campaign_scope": scope,
        "cost_in": str(cost_in),
        "cost_in_sha256": cost_sha256,
        "cost_out": str(cost_out),
        "card": str(card),
        "card_sha256": card_sha256,
        "act_dir": act_dir,
        "requested_cells": roster["requested_cells"],
        "requested_units": roster["requested_units"],
        "requested_roster_sha256": roster["roster_sha256"],
        # The whole record, not a hand-copied subset of it. The fulfilled-
        # artifact shortcut below returns before the submission path that
        # stamps the scope, so a summary that flattened only the counts would
        # print a "satisfied" artifact that reused joint cells the plan never
        # priced while saying nothing about them. Whatever the gate decided
        # travels here, and the submitted path carries the same record into the
        # sealed manifest's ``campaign_scope`` annotation.
        "requested_roster": roster,
        "joint_cells_already_priced": len(joint_cells & requested),
    }
    if requested <= joint_cells:
        # The stage's own acceptance: every requested cell already carries its
        # activation term, so its merge would add nothing and its read set is
        # the model source it would stream for no result. Nothing is queued.
        print(json.dumps({**summary, "submitted": False,
                          "reason": (
                              "every requested cell already carries a joint "
                              "A-side; the AQUA requirement is satisfied by "
                              "the artifact named by --cost-in, so no stage "
                              "was queued and --cost-out was not written")},
                         indent=1))
        return 0
    inner = ["python3", "-u", "-m", AQUA_ENTRY_POINT,
             "--card", str(card), "--model-path", str(plan["model"]),
             "--cost-in", str(cost_in), "--cost-out", str(cost_out),
             "--formats", ",".join(formats), "--require-complete-coverage"]
    if args.serving_lane:
        inner += ["--serving-lane", args.serving_lane]
    else:
        inner += ["--lane-executes-all-activation-grids"]
    if act_dir:
        inner += ["--act-dir", act_dir]
    provenance = producer.deterministic_entry_provenance(
        AQUA_ENTRY_POINT, plan=str(plan_path), plan_sha256=plan_sha256,
        workspace=str(Path(plan["inputs"]["campaign_plan"]["path"]).parent))
    return _submit_gpu_action(
        args, entry_point=AQUA_ENTRY_POINT, command="price",
        inner=inner, plan=plan, scope={**scope, **roster},
        build=lambda: producer.build_aqua_manifest(
            str(plan_path), card=str(card), cost_in=str(cost_in),
            act_dir=act_dir, produced_by=provenance, argv=inner))


def cmd_submit_allocation(args) -> int:
    producer = _manifest_producer()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    plan_sha256 = _bound_sha256(plan_path, args.plan_sha256, label="joint plan")
    joint_cost = Path(args.joint_cost).resolve()
    joint_sha256 = _bound_sha256(joint_cost, args.joint_cost_sha256,
                                 label="joint cost")
    inner = ["python3", "-u", "-m", ALLOCATION_ENTRY_POINT,
             "--joint-cost", str(joint_cost), "--joint-cost-sha256", joint_sha256,
             "--plan", str(plan_path), "--plan-sha256", plan_sha256,
             "--output", str(Path(args.output).resolve())]
    provenance = producer.deterministic_entry_provenance(
        ALLOCATION_ENTRY_POINT, plan=str(plan_path), plan_sha256=plan_sha256,
        workspace=str(Path(plan["inputs"]["campaign_plan"]["path"]).parent))
    return _submit_gpu_action(
        args, entry_point=ALLOCATION_ENTRY_POINT, command="handoff", inner=inner,
        plan=plan,
        build=lambda: producer.build_allocation_manifest(
            str(joint_cost), str(plan_path), produced_by=provenance, argv=inner))


def cmd_submit_export(args) -> int:
    producer = _manifest_producer()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    plan_sha256 = _bound_sha256(plan_path, args.plan_sha256, label="joint plan")
    inner = list(args.inner or [])
    if inner and inner[0] == "--":
        inner = inner[1:]
    if not inner:
        raise RuntimeError(
            "the export entry point lives in the Tessera tree, so its command "
            "is not derived here; pass it after --")
    provenance = producer.deterministic_entry_provenance(
        EXPORT_ENTRY_POINT, plan=str(plan_path), plan_sha256=plan_sha256,
        workspace=str(Path(plan["inputs"]["campaign_plan"]["path"]).parent))
    return _submit_gpu_action(
        args, entry_point=EXPORT_ENTRY_POINT, command="export", inner=inner,
        plan=plan,
        build=lambda: producer.build_export_manifest(
            str(plan_path), assignment=str(Path(args.assignment).resolve()),
            allocation_cost=str(Path(args.allocation_cost).resolve()),
            produced_by=provenance, argv=inner))


def _add_submission_arguments(parser) -> None:
    """The demand and placement every post-campaign submission declares.

    None of these carries a default demand. A demand copied from habit is what
    #522 and the fleet note `pb_demand_must_be_measured_not_habitual` are
    about: the numbers belong to the pass being submitted, and the caller
    measures them.
    """
    parser.add_argument("--spec", required=True,
                        help="container spec JSON; its text is passed to "
                             "tools.tessera_campaign_container --spec")
    parser.add_argument("--pbrun", default=str(PBRUN),
                        help="the published pbrun client")
    parser.add_argument("--demand", required=True,
                        help="pbrun --demand for this pass, measured from what "
                             "it holds (for example gpu=1,mem_gb=104)")
    parser.add_argument("--cpus", type=int, default=None,
                        help="cores the action reserves")
    parser.add_argument("--tag", default=None,
                        help="placement tag; the joint pass needs a GB10 box")
    parser.add_argument("--priority", type=int, default=-10,
                        help="queue band; agent and post-campaign work runs at "
                             "-10 so it never displaces campaign rows")
    parser.add_argument("--residency", default=None, choices=("stage",),
                        help="submit-joint only: ask PrismaBuild to make this "
                             "row's declared read set resident on its SSD "
                             "stage tier before the row runs "
                             "(RobTand/prismabuild#583). The action reads the "
                             "stage through the residency map the launcher "
                             "injects; omitted, nothing about the read set "
                             "changes. Reserves cluster-scoped tier capacity, "
                             "so it is a placement input, not a hint -- which "
                             "is why submit-allocation, submit-export and "
                             "submit-aqua refuse it: they read every byte "
                             "from the pool.")
    parser.add_argument("--timeout-s", type=int, default=None,
                        help="hard wall-clock cap for the action")
    parser.add_argument("--container-arg", action="append", default=None,
                        help="extra argument for tools.tessera_campaign_container, "
                             "repeatable (for example --container-arg --cpu-only)")
    parser.add_argument("--manifest-dir", default=None,
                        help="where the data manifest is written; the plan's "
                             "output_root/data-manifests by default. A frozen "
                             "output root needs this.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the command and the manifest summary; "
                             "write nothing and submit nothing")


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def _require_equal(field: str, values: dict) -> object:
    distinct = {json.dumps(value, sort_keys=True, default=str) for value in values.values()}
    if len(distinct) != 1:
        detail = ", ".join(f"{row}={json.dumps(value, sort_keys=True, default=str)[:160]}"
                           for row, value in sorted(values.items()))
        raise MergeRefused(f"rows disagree on {field}: {detail}")
    return next(iter(values.values()))


def _hessian_identities(payload: dict) -> list[dict]:
    return [row["hessian_identity"]
            for rows in payload["costs"].values() for row in rows.values()
            if "hessian_identity" in row]


def declared_coverage(plan: dict) -> dict | None:
    """The coverage a plan declares, or ``None`` when it claims the scope.

    A plan prices its census's whole scope unless it says otherwise under
    ``dense_rows_excluded``.  When it does, the anchor groups the merge expects
    are exactly the union of the plan's remaining rows' ``groups`` -- read from
    the plan, never from which rows exist on disk -- and the declaration's
    reason travels with them into the merged table.  A malformed declaration
    refuses rather than reading as either shape.
    """
    declared = plan.get(DENSE_ROWS_EXCLUDED_KEY)
    if declared is None:
        return None
    rows = declared.get("rows") if isinstance(declared, dict) else None
    reason = declared.get("reason") if isinstance(declared, dict) else None
    if (not isinstance(rows, list) or not rows
            or not all(isinstance(row, str) for row in rows)
            or not isinstance(reason, str) or not reason.strip()):
        raise MergeRefused(
            f"plan.json {DENSE_ROWS_EXCLUDED_KEY} must list the excluded row ids "
            "and give a reason")
    planned = {entry["row_id"] for entry in plan["rows"]}
    both = sorted(set(rows) & planned)
    if both:
        raise MergeRefused(
            f"plan.json {DENSE_ROWS_EXCLUDED_KEY} names planned row(s): " + ", ".join(both[:8]))
    expected = sorted({key for entry in plan["rows"] for key in entry["groups"]})
    if not expected:
        raise MergeRefused("plan.json declares no priced anchor group")
    return {"expected_groups": expected, "excluded_rows": list(rows), "reason": reason}


def merge_payloads(row_payloads: dict, *, census: dict, capture_sha256: str,
                   plan_coverage: dict | None = None) -> dict:
    """One cost payload from N rows, refusing anything they do not share.

    The merged table is the monolith's on every field the monolith's rows would
    carry: the union of the per-unit prices, one Hessian identity, and a
    coverage block rebuilt over the **scope** rather than over any one row's
    selection.

    ``plan_coverage`` is what :func:`declared_coverage` read from the plan.
    Without it the rows must price every anchor group of the scope; with it
    they must price exactly the groups the plan declares, and the merged table
    says which scope groups it leaves unpriced and why under
    ``provenance.coverage``, with ``unit_selection.selected`` True.
    """
    from prismaquant.tessera_campaign import (
        SCHEMA, campaign_population_block, canonical_refusals, selection_stack_samples,
        parse_family_restriction)
    from prismaquant.tessera_campaign import ExpertPopulation
    # The keys a merged payload must land under are the ones the campaign and
    # the allocation share.  Spelling them here as literals is how a merge
    # writes a block nothing reads: POPULATION_KEY is "population", not
    # "tessera_population", and the allocation reads only the former.
    from prismaquant.tessera_expert_projection import (
        EXPERT_WIRES_KEY, POPULATION_KEY, PROJECTION_KEY)

    for row_id, payload in row_payloads.items():
        if payload.get("schema") != SCHEMA:
            raise MergeRefused(f"{row_id}: not a {SCHEMA} payload")

    provenances = {row: payload["provenance"] for row, payload in row_payloads.items()}
    for row_id, provenance in provenances.items():
        if provenance.get("research_exact_member_scope") is not None:
            raise MergeRefused(
                f"{row_id}: research exact-member scalar cannot be merged as a full group or stack estimate")
    family_policies, restricted_structures = {}, {}
    for row_id, prov in provenances.items():
        restriction = prov.get("family_restriction")
        if restriction is None:
            family_policies[row_id] = None
            continue
        if not isinstance(restriction, dict) or set(restriction) != {"policy", "structure_by_unit"}:
            raise MergeRefused(f"{row_id}: invalid family restriction provenance")
        try:
            policy = parse_family_restriction(restriction["policy"])
        except (ValueError, TypeError) as exc:
            raise MergeRefused(f"{row_id}: invalid family restriction policy: {exc}") from exc
        structures = restriction["structure_by_unit"]
        members = {name for group in prov["unit_selection"]["groups"]
                   for name in group.get("sampled", group["members"])}
        if (policy is None or not isinstance(structures, dict) or set(structures) != members
                or any(s not in ("dense", "routed_moe") for s in structures.values())):
            raise MergeRefused(f"{row_id}: family restriction must cover exact selected unit structures")
        for name, structure in structures.items():
            if name in restricted_structures:
                raise MergeRefused(f"{row_id}: family restriction repeats selected unit {name}")
            if name in prov["campaign_scope"]["expert_targets"] and structure != "routed_moe":
                raise MergeRefused(f"{row_id}: family restriction contradicts projected expert {name}")
            restricted_structures[name] = structure
        family_policies[row_id] = policy
    family_policy = _require_equal("provenance.family_restriction.policy", family_policies)
    for field in SHARED_PROVENANCE:
        _require_equal(f"provenance.{field}",
                       {row: prov.get(field) for row, prov in provenances.items()})
    _require_equal("provenance.hessian.calibration_identity",
                   {row: prov["hessian"]["calibration_identity"]
                    for row, prov in provenances.items()})
    _require_equal("provenance.activation_static_scales.policy",
                   {row: prov["activation_static_scales"]["policy"]
                    for row, prov in provenances.items()})
    _require_equal("currency", {row: payload["currency"]
                                for row, payload in row_payloads.items()})
    scope = _require_equal(
        "provenance.campaign_scope",
        {row: {key: value for key, value in prov["campaign_scope"].items()}
         for row, prov in provenances.items()})

    # One Hessian identity across every priced row, on every field but the
    # capture digest the merge is about to replace.
    for row_id, payload in row_payloads.items():
        for field in SHARED_HESSIAN:
            _require_equal(
                f"hessian_identity.{field}",
                {f"{row_id}:{index}": identity.get(field)
                 for index, identity in enumerate(_hessian_identities(payload))}
                or {row_id: None})
    for field in SHARED_HESSIAN:
        _require_equal(
            f"hessian_identity.{field} across rows",
            {row: (_hessian_identities(payload)[0].get(field)
                   if _hessian_identities(payload) else None)
             for row, payload in row_payloads.items()})

    # Coverage: every expected group priced exactly once.  The expectation is
    # the scope, or the groups the plan declares when it declares fewer.
    selected: dict[str, str] = {}
    selection_entries = {}
    for row_id, prov in provenances.items():
        for entry in prov["unit_selection"]["groups"]:
            key = entry["key"]
            if key in selected:
                raise MergeRefused(
                    f"anchor group {key!r} is priced by both {selected[key]} and {row_id}")
            if key not in scope["anchor_groups"] or sorted(entry["members"]) != sorted(scope["anchor_groups"][key]):
                raise MergeRefused(f"{row_id}: selection {key!r} differs from campaign scope")
            selected[key] = row_id
            selection_entries[key] = entry
    if plan_coverage is None:
        expected = set(scope["anchor_groups"])
        expected_by = "scope"
    else:
        expected = set(plan_coverage["expected_groups"])
        expected_by = "plan"
        outside = sorted(expected - set(scope["anchor_groups"]))
        if outside:
            raise MergeRefused(
                f"the plan declares {len(outside)} anchor group(s) outside the campaign "
                "scope: " + ", ".join(outside[:8]))
    missing = sorted(expected - set(selected))
    if missing:
        raise MergeRefused(
            f"the rows do not cover {len(missing)} anchor group(s) of the {expected_by}: "
            + ", ".join(missing[:8]))
    undeclared = sorted(set(selected) - expected)
    if undeclared:
        raise MergeRefused(
            f"the rows price {len(undeclared)} anchor group(s) the plan does not declare: "
            + ", ".join(undeclared[:8]))
    unpriced = sorted(set(scope["anchor_groups"]) - expected)

    sampled = any("stack_samples" in entry or entry.get("sampled")
                  for entry in selection_entries.values())
    merged_selection = {
        "schema": "prismaquant.tessera_campaign_units.v2" if sampled else "prismaquant.tessera_campaign_units.v1",
        # False is the whole-scope claim a monolith writes; a table the plan
        # left partly unpriced is a selection and says so.
        "selected": bool(unpriced),
        "groups": [selection_entries[key] if sampled else
                   {"key": key, "members": list(scope["anchor_groups"][key])}
                   for key in sorted(selection_entries)],
    }
    stack_samples, profile = {}, None
    if sampled:
        from prismaquant.model_profiles import detect_profile
        profile = detect_profile(next(iter(provenances.values()))["model"])
        stack_samples = selection_stack_samples(merged_selection, profile)
    costs: dict[str, dict] = {}
    loo: dict[str, dict] = {}
    surfaces: dict[str, dict] = {}
    anchor_counts: dict[str, dict] = {}
    menu_sizes: dict[str, int] = {}
    anchor_groups: dict[str, list] = {}
    non_interpolable: list[dict] = []
    expert_wires: dict[str, dict] = {}
    # Evidence, not prices: rows a shard adopted from another campaign whose
    # rungs its menu does not admit.  The union is taken here for the same
    # reason the prices are -- the reference row's block describes one slice.
    unservable: dict[str, dict] = {}
    formats: set[str] = set()
    stopped_early = False
    wall_seconds = 0.0
    rounds_run = 0
    seeds: list[dict] = []
    projection_block = None
    serving_by_unit: dict[str, dict] = {}
    serving_target = None
    for row_id in sorted(row_payloads):
        payload = row_payloads[row_id]
        prov = payload["provenance"]
        for qname, rows in payload["costs"].items():
            if qname in costs:
                raise MergeRefused(f"unit {qname} is priced by more than one row")
            costs[qname] = {
                fmt: {**row, "hessian_identity": {**row["hessian_identity"],
                                                  "capture_sha256": capture_sha256}}
                if "hessian_identity" in row else row
                for fmt, row in rows.items()
            }
        formats.update(payload["formats"])
        loo.update(payload["leave_one_anchor_out"])
        non_interpolable.extend(payload["non_interpolable"])
        surfaces.update(prov["surfaces"])
        anchor_groups.update(prov["anchor_groups"])
        anchor_counts.update(payload["anchor_counts"])
        menu_sizes.update(payload["menu_sizes"])
        expert_wires.update(payload.get(EXPERT_WIRES_KEY, {}))
        for qname, rungs in (prov.get("unservable") or {}).items():
            held = unservable.setdefault(qname, {})
            for fmt, record in rungs.items():
                if fmt in held and held[fmt] != record:
                    raise MergeRefused(
                        f"{row_id}: it carries different unservable evidence for "
                        f"{qname} {fmt} than an earlier row")
                held[fmt] = record
        stopped_early = stopped_early or bool(prov["stopped_early"])
        wall_seconds += float(prov["wall_seconds"])
        rounds_run = max(rounds_run, int(prov["rounds_run"]))
        if prov.get("seed_checkpoint"):
            seeds.append({"row": row_id, **prov["seed_checkpoint"]})
        projection = prov.get(PROJECTION_KEY)
        if projection is not None:
            # Every row carries the SCOPE's projection block, because the
            # allocation rebinds the producer's answer over every stack the
            # block names. Two different blocks would be two producer answers.
            if projection_block is None:
                projection_block = projection
            elif projection_block != projection:
                raise MergeRefused(
                    f"{row_id}: its producer expert projection differs from the "
                    "other rows'; they did not read one census projection")
        serving = prov.get("tessera_serving_scope")
        if serving:
            serving_target = serving["target"] if serving_target is None else serving_target
            if serving["target"] != serving_target:
                raise MergeRefused("rows disagree on the serving target")
            serving_by_unit.update(serving["by_unit"])

    reference = provenances[sorted(provenances)[0]]
    provenance = {key: value for key, value in reference.items()}
    provenance.pop("identity_migration", None)
    carried_migration = merge_identity_migrations(
        {row: prov.get("identity_migration") for row, prov in provenances.items()})
    if carried_migration is not None:
        provenance["identity_migration"] = carried_migration
    if family_policy is not None:
        provenance["family_restriction"] = {"policy": family_policy,
            "structure_by_unit": dict(sorted(restricted_structures.items()))}
    provenance.update({
        "surfaces": dict(sorted(surfaces.items())),
        "anchor_groups": dict(sorted(anchor_groups.items())),
        "unservable": {name: {fmt: rungs[fmt] for fmt in sorted(rungs)}
                       for name, rungs in sorted(unservable.items())},
        "stopped_early": stopped_early,
        "wall_seconds": wall_seconds,
        "rounds_run": rounds_run,
        "unit_selection": merged_selection,
        "activation_static_scales": dict(reference["activation_static_scales"]),
        "hessian": {**reference["hessian"], "capture_sha256": capture_sha256},
        "campaign_fanout": {
            "schema": PLAN_SCHEMA,
            "rows": {row_id: sorted(
                entry["key"] for entry in provenances[row_id]["unit_selection"]["groups"])
                for row_id in sorted(provenances)},
            "seed_checkpoints": seeds,
        },
    })
    if plan_coverage is not None:
        provenance["coverage"] = {
            "schema": COVERAGE_SCHEMA,
            "scope_groups": len(scope["anchor_groups"]),
            "priced_groups": len(expected),
            "unpriced_groups": unpriced,
            "excluded_rows": list(plan_coverage["excluded_rows"]),
            "reason": plan_coverage["reason"],
        }
    if any("no_admitted_rung" in prov for prov in provenances.values()):
        provenance["no_admitted_rung"] = sorted({name for prov in provenances.values()
                                               for name in prov.get("no_admitted_rung", [])})
    if any("unit_selection_sample" in prov for prov in provenances.values()):
        audit, probabilities = set(), {}
        for row_id, prov in provenances.items():
            sample = prov.get("unit_selection_sample", {})
            audit.update(sample.get("audit_units", []))
            for name, probability in sample.get("inclusion_probability", {}).items():
                if name in probabilities and probabilities[name] != probability:
                    raise MergeRefused(f"{row_id}: different inclusion probability for {name}")
                probabilities[name] = probability
        provenance["unit_selection_sample"] = {
            "audit_units": sorted(audit), "inclusion_probability": dict(sorted(probabilities.items()))}
    if serving_target is not None:
        provenance["tessera_serving_scope"] = {
            "target": serving_target, "by_unit": dict(sorted(serving_by_unit.items()))}

    payload = {
        **{key: value for key, value in row_payloads[sorted(row_payloads)[0]].items()
           if key not in {"costs", "formats", "leave_one_anchor_out",
                          "non_interpolable", "menu_sizes", "anchor_counts",
                          "provenance", EXPERT_WIRES_KEY}},
        "schema": SCHEMA,
        "costs": dict(sorted(costs.items())),
        "formats": sorted(formats),
        "leave_one_anchor_out": dict(sorted(loo.items())),
        "non_interpolable": canonical_refusals(non_interpolable),
        "menu_sizes": dict(sorted(menu_sizes.items())),
        "anchor_counts": dict(sorted(anchor_counts.items())),
        "provenance": provenance,
    }
    if expert_wires:
        payload[EXPERT_WIRES_KEY] = dict(sorted(expert_wires.items()))
    population = ExpertPopulation(
        members=(),
        declared={stack: {name: tuple(shape) for name, shape in units.items()}
                  for stack, units in scope["declared_stacks"].items()},
        packed_in_scope={name: tuple(shape) for name, shape
                         in scope["packed_in_scope"].items()},
        omitted_outside_layer_stride={
            name: tuple(shape) for name, shape
            in scope["packed_outside_layer_stride"].items()},
    )
    # Overwrites the reference row's block, which describes that row's slice.
    payload["provenance"][POPULATION_KEY] = campaign_population_block(
        dense_targets=scope["dense_targets"], expert_targets=scope["expert_targets"],
        dense_all=scope["dense_all"], pinned=scope["pinned"],
        population=population, layer_stride=int(reference["layer_stride"]),
        costs=payload["costs"], menus=menu_sizes,
        stack_samples=stack_samples, profile=profile)
    return payload


def _merge_export_hessian_references(row_dirs, payloads, *, out_cache, identity,
                                     policy, static_scales, census):
    from prismaquant import tessera_calibration_cache as store
    from prismaquant.tessera_campaign import write_export_inputs

    def accepted_rows():
        for row_id in sorted(row_dirs):
            path = Path(row_dirs[row_id])/'cache'/'hessian_capture.references.json'
            payload = payloads[row_id]
            with store.open_hessian_reference(path) as owner:
                owner.require_census(census)
                owner.require_provenance({**identity,'hessian_role':'fit'})
                descriptor = owner.descriptor
                provenance = payload['provenance']
                if provenance.get('calibration_cache') != descriptor['canonical_capture']:
                    raise MergeRefused(f'{row_id}: reference does not bind the row canonical capture')
                if provenance['hessian'].get('reference_binding') != owner.binding():
                    raise MergeRefused(f'{row_id}: reference binding differs from the row provenance')
                identities = _hessian_identities(payload)
                if (not identities or any(row.get('capture_sha256') != descriptor['capture_sha256'] or
                        row.get('reference_binding') != owner.binding() for row in identities)):
                    raise MergeRefused(f'{row_id}: reference commitments do not bind the exact priced row seals')
                groups = (provenance.get('unit_selection') or {}).get('groups')
                if not isinstance(groups, list) or not groups:
                    raise MergeRefused(f'{row_id}: reference row has no selected unit roster')
                expected = {name for group in groups for name in group.get('sampled', group['members'])}
                if set(owner) != expected:
                    raise MergeRefused(f'{row_id}: reference H roster differs from the exact selected members')
                if owner.receipt()['loaded_entries'] != 0:
                    raise MergeRefused('Hessian reference merge unexpectedly consumed tensor bytes')
            yield descriptor

    try:
        descriptor = store.merge_hessian_reference_descriptors(accepted_rows())
        out_cache.mkdir(parents=True, exist_ok=True)
        path = out_cache/'hessian_capture.references.json'
        digest = store.write_hessian_reference(path, descriptor)
        _, scales, _ = write_export_inputs(out_cache, hessians=None, hessian_rows=census['counts'],
            hessian_identity=identity, static_scales=static_scales, static_scale_policy=policy)
    except (ValueError, RuntimeError, OSError) as error:
        if isinstance(error, MergeRefused):
            raise
        raise MergeRefused(f'canonical Hessian reference merge refused: {error}') from error
    return path, scales, digest


def merge_export_inputs(row_dirs: dict, payloads: dict, *, out_cache: Path,
                        identity: dict, policy: str, static_scales: dict,
                        census: dict):
    """Union the rows' Hessian captures into the capture a whole run writes.

    The rows priced disjoint units of one draw, so their captures hold disjoint
    ``H`` under the same ``counts`` and the same provenance.  The union is
    therefore the object a whole-scope run writes, and its digest is recomputed
    from the union rather than carried over from any row.

    Four refusals stand between "the rows agree" and "this is that object":
    every row's own capture must still seal to the digest its cost rows carry;
    every row's capture provenance must be the same dict, not merely the same
    digested triple; no unit may be captured twice with different bytes; and
    the union's ``counts`` must be the census's, over the census's roster.
    """
    reference_modes = [(payloads[row].get('provenance',{}).get('hessian') or {}).get('reference_binding')
                       for row in sorted(row_dirs)]
    if any(value is not None for value in reference_modes):
        if any(value is None for value in reference_modes):
            raise MergeRefused('cannot merge legacy and canonical-reference Hessian handoffs')
        return _merge_export_hessian_references(row_dirs, payloads, out_cache=out_cache,
            identity=identity, policy=policy, static_scales=static_scales, census=census)
    import torch

    from prismaquant.tessera_campaign import write_export_inputs
    from prismaquant.tessera_export_lane import hessian_capture_sha256

    hessians: dict[str, object] = {}
    counts = None
    provenance = None
    for row_id in sorted(row_dirs):
        capture = Path(row_dirs[row_id]) / "cache" / "hessian_capture.pt"
        if not capture.is_file():
            stamped = {row["hessian_identity"].get("capture_sha256")
                       for rows in payloads[row_id]["costs"].values()
                       for row in rows.values() if "hessian_identity" in row}
            if stamped - {None}:
                raise MergeRefused(
                    f"{row_id}: its rows carry a capture digest but it wrote no "
                    "Hessian capture")
            continue
        blob = torch.load(capture, map_location="cpu", weights_only=False)
        own = hessian_capture_sha256(blob["H"], blob["provenance"])
        stamped = {row["hessian_identity"].get("capture_sha256")
                   for rows in payloads[row_id]["costs"].values()
                   for row in rows.values() if "hessian_identity" in row}
        if stamped and stamped != {own}:
            raise MergeRefused(
                f"{row_id}: its cost rows carry capture digests {sorted(stamped)} "
                f"but its capture seals to {own}")
        if counts is None:
            counts, provenance = dict(blob["counts"]), dict(blob["provenance"])
        else:
            if dict(blob["counts"]) != counts:
                raise MergeRefused(
                    f"{row_id}: Hessian capture counts differ from the other rows'; "
                    "the rows did not see one calibration census")
            if dict(blob["provenance"]) != provenance:
                raise MergeRefused(
                    f"{row_id}: Hessian capture provenance differs from the other "
                    "rows'; the rows describe two calibrations")
        for name, tensor in blob["H"].items():
            if name in hessians and not torch.equal(hessians[name], tensor):
                raise MergeRefused(
                    f"{name}: two rows captured different Hessians for one unit")
            hessians[name] = tensor
    if counts is not None and dict(counts) != dict(census["counts"]):
        raise MergeRefused(
            "the merged capture's counts are not the census's; the rows did not "
            "price the scope this census describes")
    out_cache.mkdir(parents=True, exist_ok=True)
    capture_path, scales_path, capture_sha256 = write_export_inputs(
        out_cache,
        hessians=(hessians if hessians else None),
        hessian_rows=(counts or {}),
        hessian_identity=identity,
        static_scales=static_scales,
        static_scale_policy=policy,
    )
    return capture_path, scales_path, capture_sha256


def merge_checkpoint(row_dirs: dict, out_manifest: Path) -> dict:
    """One journal from the rows', under the identity their union describes.

    The rows' identities differ only where the selection does: the ``units``
    map, the serving scope's ``by_unit`` and the producer's projected stacks.
    Everything else must already be equal, and the union of the three is what a
    whole-scope run of this code computes -- so the merged journal is one a
    later whole-scope invocation can resume, and refuses by field if it is not.
    """
    from prismaquant.cost_stage_checkpoint import (
        atomic_write_bytes, canonical_json, canonical_json_sha256, unit_path,
        MANIFEST_SCHEMA, prepare_journal, write_unit,
    )

    identities = {}
    migrations = {}
    states: dict[str, dict] = {}
    stage = "Tessera campaign"
    for row_id in sorted(row_dirs):
        manifest_path = Path(row_dirs[row_id]) / "cost.anchors.json"
        manifest = json.loads(manifest_path.read_text())
        identities[row_id] = manifest["identity"]
        migrations[row_id] = manifest.get("identity_migration")
        parts = manifest_path.with_name(manifest_path.name + ".parts")
        listed: list[str] = []
        for entry in manifest["units"]:
            qname = entry["qname"]
            listed.append(qname)
            shard = parts / entry["file"]
            if not shard.is_file():
                raise MergeRefused(
                    f"{row_id}: its journal names {qname} and the shard "
                    f"{shard} is not there; the row's anchors would be "
                    "dropped from the merged journal")
            if shard != unit_path(parts, qname):
                raise MergeRefused(f"{row_id}: unit {qname} names a noncanonical shard")
        expected = set(manifest["identity"]["units"])
        if len(listed) != len(set(listed)) or set(listed) != expected:
            raise MergeRefused(
                f"{row_id}: manifest units differ from its checkpoint identity units")
        # Reuse the journal's reader: it validates the manifest and every
        # envelope before returning state. Rehashing unchecked payload bytes
        # here would turn corrupt or foreign shards into a trusted journal.
        try:
            _, _, completed = prepare_journal(
                parts, stage=stage, resume=True, identity=manifest["identity"],
                qnames=sorted(expected), manifest_path=manifest_path)
        except RuntimeError as exc:
            raise MergeRefused(f"{row_id}: {exc}") from exc
        for qname, state in completed.items():
            if qname in states:
                raise MergeRefused(f"unit {qname} has a journal shard in two rows")
            states[qname] = state

    merged_identity = None
    for row_id, identity in sorted(identities.items()):
        if merged_identity is None:
            merged_identity = {key: value for key, value in identity.items()}
            merged_identity["units"] = dict(identity["units"])
            continue
        for key in sorted(set(merged_identity) | set(identity)):
            if key in RECONCILED_IDENTITY_KEYS:
                continue
            if (key not in merged_identity or key not in identity
                    or merged_identity[key] != identity[key]):
                raise MergeRefused(
                    f"{row_id}: checkpoint identity differs at {key!r}")
        for name, unit in identity["units"].items():
            if name in merged_identity["units"] and merged_identity["units"][name] != unit:
                raise MergeRefused(f"{row_id}: two rows bind different inputs for {name}")
            merged_identity["units"][name] = unit
        if "stack_sampling_identity" in merged_identity or "stack_sampling_identity" in identity:
            combined = dict(merged_identity.get("stack_sampling_identity", {}))
            for name, sample in identity.get("stack_sampling_identity", {}).items():
                if name in combined and combined[name] != sample:
                    raise MergeRefused(f"{row_id}: different stack_sampling_identity for {name}")
                combined[name] = sample
            merged_identity["stack_sampling_identity"] = dict(sorted(combined.items()))
        merged_identity["serving_scope"] = _merge_scope(
            merged_identity.get("serving_scope"), identity.get("serving_scope"), row_id)
        merged_identity["expert_projection"] = _merge_projection(
            merged_identity.get("expert_projection"), identity.get("expert_projection"),
            row_id)
        merged_identity["family_restriction"] = _merge_family_restriction(
            merged_identity.get("family_restriction"),
            identity.get("family_restriction"), row_id)
        if merged_identity["family_restriction"] is None:
            del merged_identity["family_restriction"]
    merged_identity["units"] = dict(sorted(merged_identity["units"].items()))

    canonical = canonical_json(merged_identity, where="merged campaign identity")
    identity_sha256 = canonical_json_sha256(canonical, where="merged campaign identity")
    parts = out_manifest.with_name(out_manifest.name + ".parts")
    for qname, state in sorted(states.items()):
        write_unit(parts, stage=stage, qname=qname,
                   identity_sha256=identity_sha256, state=state)
    manifest = {
        "schema": MANIFEST_SCHEMA, "stage": stage,
        "identity_sha256": identity_sha256, "identity": canonical,
        "units": [{"qname": qname,
                   "file": str(unit_path(parts, qname).relative_to(parts))}
                  for qname in sorted(merged_identity["units"])],
    }
    carried = merge_identity_migrations(migrations)
    if carried is not None:
        manifest["identity_migration"] = carried
    atomic_write_bytes(out_manifest, json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False).encode("utf-8"))
    return manifest


def merge_identity_migrations(per_row: dict) -> "list | None":
    """The rows' ``identity_migration`` records, merged by
    ``cost_stage_checkpoint.merge_identity_migrations``."""
    from prismaquant.cost_stage_checkpoint import merge_identity_migrations as merge
    return merge(per_row, error=MergeRefused)


def _merge_family_restriction(left, right, row_id):
    """One policy, and the union of the rows' per-unit structure maps.

    ``structure_by_unit`` is keyed by the row's OWN selected units, so a dense
    row and a routed row of one campaign never carry the same map. Comparing
    the whole restriction for equality therefore refuses every census that
    fans dense and routed units onto different rows, which is every GLM census
    (RobTand/prismaquant#487). ``merge_payloads`` already reconciles the same
    field this way; this is the journal side of it.
    """
    if left is None or right is None:
        if left != right:
            raise MergeRefused(
                f"{row_id}: one row restricts families and another does not")
        return left
    if left["policy"] != right["policy"]:
        raise MergeRefused(f"{row_id}: rows disagree on the family restriction policy")
    for name in left["structure_by_unit"].keys() & right["structure_by_unit"].keys():
        if left["structure_by_unit"][name] != right["structure_by_unit"][name]:
            raise MergeRefused(
                f"{row_id}: different restricted structure for {name}")
    return {"policy": left["policy"],
            "structure_by_unit": dict(sorted(
                {**left["structure_by_unit"], **right["structure_by_unit"]}.items()))}


def _merge_scope(left, right, row_id):
    if left is None or right is None:
        if left != right:
            raise MergeRefused(f"{row_id}: one row has a serving scope and another does not")
        return left
    if left["target"] != right["target"]:
        raise MergeRefused(f"{row_id}: rows disagree on the serving target")
    for name in left["by_unit"].keys() & right["by_unit"].keys():
        if left["by_unit"][name] != right["by_unit"][name]:
            raise MergeRefused(f"{row_id}: different serving context for {name}")
    return {"target": left["target"],
            "by_unit": dict(sorted({**left["by_unit"], **right["by_unit"]}.items()))}


def _merge_projection(left, right, row_id):
    if left is None:
        return right
    if right is None:
        return left
    if left["source"] != right["source"]:
        raise MergeRefused(f"{row_id}: rows projected different source checkpoints")
    for name in left["stacks"].keys() & right["stacks"].keys():
        if left["stacks"][name] != right["stacks"][name]:
            raise MergeRefused(f"{row_id}: different producer projection for stack {name}")
    stacks = {**left["stacks"], **right["stacks"]}
    return {"source": left["source"], "stacks": dict(sorted(stacks.items()))}


def _require_receipts(workspace: Path, expected: int) -> None:
    """Refuse to merge a row the fleet did not report as executed.

    A ``cost.pkl`` on disk says a process wrote a file; the fleet's row table
    says which action it was, where it ran and what it exited with.  Both, or
    neither.
    """
    path = workspace / "receipts.json"
    if not path.is_file():
        raise MergeRefused(
            f"no fleet receipts at {path}; submit the manifest before merging")
    receipts = json.loads(path.read_text())
    rows = receipts.get("rows") or []
    if len(rows) != expected:
        raise MergeRefused(
            f"{path} reports {len(rows)} rows and the plan has {expected}")
    # ``pbwait.verdict`` is the fleet's own reading of the table: 0 when every
    # row's work is done, and a memoized ``cache_hit`` counts as done there.
    # It is the gate, because a re-submitted row that was already priced
    # reports no launcher status of its own and renders ``rc`` as ``-``.
    if receipts.get("returncode") not in {0, "0"}:
        raise MergeRefused(
            f"{path} records pbcampaign exit {receipts.get('returncode')!r}; "
            "not every row is done")
    failed = [row for row in rows
              if row.get("rc") not in {"0", 0, "-", ""}]
    if failed:
        raise MergeRefused(
            "the fleet reports a non-zero exit for "
            + ", ".join(f"{row.get('key')} (rc={row.get('rc')})" for row in failed))
    where = sorted({f"{row.get('host') or '?'} ({row.get('status')})"
                    for row in rows})
    print(f"[dispatch] {len(rows)} rows: " + ", ".join(where))


def cmd_merge(args) -> int:
    workspace = Path(args.workspace)
    plan = json.loads((workspace / "plan.json").read_text())
    census = json.loads(Path(plan["census"]).read_text())
    row_dirs = {entry["row_id"]: entry["dir"] for entry in plan["rows"]}
    missing = sorted(row for row, path in row_dirs.items()
                     if not (Path(path) / "cost.pkl").is_file())
    if missing:
        raise MergeRefused(
            f"{len(missing)} planned row(s) wrote no cost.pkl: " + ", ".join(missing[:8]))
    _require_receipts(workspace, len(row_dirs))
    payloads = {}
    for row_id, path in row_dirs.items():
        with open(Path(path) / "cost.pkl", "rb") as handle:
            payloads[row_id] = pickle.load(handle)

    reference = payloads[sorted(payloads)[0]]["provenance"]
    # Under a census every row calibrated the SCOPE's static scales, so this is
    # an equality check and not a union: two rows that disagree here priced two
    # different A-side contracts for one fused module.
    static_scales = _require_equal(
        "provenance.activation_static_scales.units",
        {row: payload["provenance"]["activation_static_scales"]["units"]
         for row, payload in payloads.items()})
    out_cache = Path(args.out).parent / "cache"
    _capture, _scales, capture_sha256 = merge_export_inputs(
        row_dirs, payloads, out_cache=out_cache,
        identity=reference["hessian"]["calibration_identity"],
        policy=reference["activation_static_scales"]["policy"],
        static_scales=static_scales, census=census)
    coverage = declared_coverage(plan)
    merged = merge_payloads(payloads, census=census, capture_sha256=capture_sha256,
                            plan_coverage=coverage)
    if coverage is not None:
        block = merged["provenance"]["coverage"]
        print(f"[dispatch] coverage: {block['priced_groups']} of {block['scope_groups']} "
              f"anchor groups priced; {len(block['unpriced_groups'])} left unpriced by "
              f"the plan's {DENSE_ROWS_EXCLUDED_KEY}: {block['reason']}")
    merged["provenance"]["cache_dir"] = str(out_cache)
    merged["provenance"]["wire_dir"] = str(out_cache / "wire")
    merged["provenance"]["hessian"]["capture_path"] = (
        None if _capture is None else str(_capture))
    merged["provenance"]["activation_static_scales"]["path"] = (
        None if _scales is None else str(_scales))

    wire_out = out_cache / "wire"
    wire_out.mkdir(parents=True, exist_ok=True)
    linked = 0
    for path in sorted(row_dirs.values()):
        source_dir = Path(path) / "cache" / "wire"
        if not source_dir.is_dir():
            continue
        for blob in sorted(source_dir.iterdir()):
            target = wire_out / blob.name
            if target.exists():
                continue
            try:
                os.link(blob, target)
            except OSError:
                target.write_bytes(blob.read_bytes())
            linked += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as handle:
        pickle.dump(merged, handle)
    manifest = merge_checkpoint(row_dirs, out.with_suffix(".anchors.json"))

    from prismaquant.tessera_menu import assert_uniform_hessian_identity

    identity = assert_uniform_hessian_identity(merged["costs"])
    total = sum(len(rows) for rows in merged["costs"].values())
    print(f"[dispatch] merged {len(payloads)} rows -> {out}: "
          f"{len(merged['costs'])} units, {total} priced rungs, "
          f"{len(merged['formats'])} formats, {linked} wire blobs, "
          f"checkpoint {manifest['identity_sha256'][:12]}")
    print(f"[dispatch] one Hessian identity: capture_sha256="
          f"{identity.get('capture_sha256')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    census = sub.add_parser("census", help="take the scope's calibration census")
    census.add_argument("--spec", required=True)
    census.add_argument("--workspace", required=True)
    census.add_argument("--timeout-s", type=int, default=7200)
    census.add_argument("--wait-s", type=int, default=14400)
    census.add_argument("--submit", action="store_true")
    census.set_defaults(func=cmd_census)

    capture = sub.add_parser("capture", help="capture full-census X/H once before planning rows")
    capture.add_argument("--spec", required=True)
    capture.add_argument("--workspace", required=True)
    capture.add_argument("--timeout-s", type=int, default=7200)
    capture.add_argument("--wait-s", type=int, default=14400)
    capture.add_argument("--submit", action="store_true")
    capture.set_defaults(func=cmd_capture)

    plan = sub.add_parser("plan", help="lay the campaign out as pbcampaign rows")
    plan.add_argument("--spec", required=True)
    plan.add_argument("--workspace", required=True)
    plan.add_argument("--calibration-cache", default=None,
                      help="complete capture manifest to hash-bind into every row")
    plan.add_argument("--groups-per-row", type=int, default=1)
    plan.add_argument("--rows-per-box", type=int, default=1,
                      help="how many of these rows one box is meant to run at "
                           "once. It does not change a row's demand -- PB "
                           "places on the demand, and shrinking it to force "
                           "co-residency would be reserving less than the row "
                           "holds. It is checked against the spec's "
                           "'box_memory_gb', when the spec declares one: a row "
                           "that does not fit is left out of the manifest and "
                           "recorded in the plan, and only a plan with no "
                           "admissible row at all refuses.")
    plan.add_argument("--timeout-s", type=int, default=None,
                      help="a hard wall-clock deadline for every row, ending "
                           "it whatever it is doing. Unset by default: rows "
                           "declare the phases they walk and the quiet they "
                           "are allowed in each instead, so a row that keeps "
                           "committing anchors keeps running and one that "
                           "stops ends within "
                           f"{sum(g for _, g in CAMPAIGN_PROGRESS_PHASES)}s. "
                           "Set it only to cap a row's cost deliberately")
    plan.add_argument("--stack-sample", type=int, default=None,
                      help="price each routed stack from this many experts "
                           "per role, drawn proportional to the probe's "
                           "h_trace. Unset prices every expert.")
    plan.add_argument("--stack-sample-sizes", choices=("probe", "counts"),
                      default="probe",
                      help="what the PPS draw is proportional to: the probe's "
                           "per-expert h_trace (the default, and what every "
                           "plan on disk used), or the census's per-expert "
                           "routed-row counts. counts is a routed-token proxy "
                           "for h_trace, not h_trace; the draw records which "
                           "one it used and the digest of the vector.")
    plan.add_argument("--stack-sample-seed", type=int, default=0,
                      help="the draw's seed; the same seed and the same probe "
                           "draw the same experts.")
    plan.add_argument("--audit-rate", type=int, default=10,
                      help="one sampled expert in this many gets a third "
                           "anchor and a leave-one-out check.")
    plan.add_argument("--probe", default=None,
                      help="a probe pickle carrying per-expert h_trace.")
    seeds = plan.add_mutually_exclusive_group()
    seeds.add_argument('--seed-workspace', default=None,
                       help='reuse matching rows from a prior plan through the ordinary seed gates; '
                            'completed and partial checkpoints are supported')
    seeds.add_argument("--seed-checkpoint", default=None,
                      help="a campaign checkpoint whose measured anchors every "
                           "row may adopt, subject to its own row gates")
    plan.add_argument("--seed-wire-dir", default=None)
    plan.set_defaults(func=cmd_plan)

    check = sub.add_parser(
        "check", help="re-derive every manifest row's memory demand")
    check.add_argument("--workspace", required=True)
    check.add_argument("--manifest", default=None,
                       help="the manifest to check; the workspace's "
                            "manifest.json by default")
    check.add_argument("--spec", default=None,
                       help="the spec the rows were planned from; taken from "
                            "the workspace's plan.json when it records one")
    check.add_argument("--census", default=None,
                       help="the census the rows were planned against; taken "
                            "from plan.json or the workspace by default")
    check.add_argument("--box-memory-gb", type=int, default=None,
                       help="what a GPU box in the fleet declares, in GiB. A "
                            "row deriving more than this can never be "
                            "admitted and is refused. Defaults to the spec's "
                            "'box_memory_gb'; unset on both, capacity is not "
                            "checked.")
    check.set_defaults(func=cmd_check)

    submit = sub.add_parser(
        "submit", help="submit the manifest; re-running it is the resume")
    submit.add_argument("--workspace", required=True)
    submit.add_argument("--wait-s", type=int, default=86400)
    submit.add_argument("--spec", default=None,
                       help="the spec the rows were planned from. Every row's "
                            "demand is re-derived before submission, so a "
                            "plan.json without a 'spec' field needs this.")
    submit.add_argument("--census", default=None)
    submit.add_argument("--box-memory-gb", type=int, default=None,
                       help="what a GPU box in the fleet declares, in GiB; "
                            "defaults to the spec's 'box_memory_gb'.")
    submit.set_defaults(func=cmd_submit)

    joint = sub.add_parser(
        "submit-joint",
        help="submit a joint AURA pass with the read set it will consume")
    joint.add_argument("command", choices=("prepare", "run"))
    joint.add_argument("--plan", required=True,
                       help="the resolved joint plan the pass runs from")
    joint.add_argument("--plan-sha256", default=None,
                       help="the digest the pass binds the plan by; computed "
                            "when omitted and checked when given")
    joint.add_argument("--prepared", default=None,
                       help="the prepared completion the run command consumes")
    joint.add_argument("--prepared-sha256", default=None)
    joint.add_argument("--head-grace-s", type=int, default=None,
                       help="stall allowance in seconds for the 'head' phase; "
                            "the default 1800 is sized for a fresh run, and a "
                            "resume's cell-roster pass commits nothing while it "
                            "runs (#678), so size this from the observed rate")
    joint.add_argument("--resume", action="store_true",
                       help="forwarded to the pass, which resumes from its "
                            "identity-bound checkpoints")
    joint.add_argument("--source-transition", default=None,
                       help="run --resume only: the closed source-transition "
                            "receipt (prismaquant.joint_aura_transitions) under "
                            "which this checkout may consume a prepared record "
                            "sealed by an earlier implementation; forwarded to "
                            "the pass and declared in the head of the read set")
    joint.add_argument("--source-transition-sha256", default=None,
                       help="the digest the receipt is bound by; computed when "
                            "omitted and checked when given")
    joint.add_argument("--require-scope", required=True,
                       choices=CAMPAIGN_SCOPE_KINDS,
                       help="what this submission is for. The scope itself is "
                            "derived from the plan's bound census; this states "
                            "the intent it has to match, so a diagnostic "
                            "window subset cannot be submitted as the "
                            "campaign's score")
    joint.add_argument("--campaign-identity", required=True,
                       help="the frozen "
                            f"{CAMPAIGN_IDENTITY_SCHEMA} the plan's roster, "
                            "group roster and window total have to reproduce "
                            "exactly. A plan is self-consistent with whatever "
                            "census it binds, so the campaign it belongs to has "
                            "to come from outside the plan")
    joint.add_argument("--campaign-identity-sha256", default=None,
                       help="the digest the campaign identity is bound by; "
                            "computed when omitted and checked when given")
    _add_submission_arguments(joint)
    joint.set_defaults(func=cmd_submit_joint)

    allocation = sub.add_parser(
        "submit-allocation",
        help="submit the allocation handoff with the read set it will consume")
    allocation.add_argument("--plan", required=True)
    allocation.add_argument("--plan-sha256", default=None)
    allocation.add_argument("--joint-cost", required=True)
    allocation.add_argument("--joint-cost-sha256", default=None)
    allocation.add_argument("--output", required=True,
                            help="where the handoff writes its own table")
    _add_submission_arguments(allocation)
    allocation.set_defaults(func=cmd_submit_allocation)

    export = sub.add_parser(
        "submit-export",
        help="submit the serving export with the read set it will consume")
    export.add_argument("--plan", required=True)
    export.add_argument("--plan-sha256", default=None)
    export.add_argument("--assignment", required=True,
                        help="the layer_config.json the export ships")
    export.add_argument("--allocation-cost", required=True,
                        help="the allocation handoff's table")
    _add_submission_arguments(export)
    export.add_argument("inner", nargs=argparse.REMAINDER,
                        help="the export command itself, after --; it lives "
                             "in the Tessera tree and is not derived here")
    export.set_defaults(func=cmd_submit_export)

    aqua = sub.add_parser(
        "submit-aqua",
        help="submit the campaign's AQUA stage with strict per-cell coverage")
    aqua.add_argument("--plan", required=True,
                      help="the same sealed joint plan the campaign runs from; "
                           "its bound census decides the requested roster")
    aqua.add_argument("--plan-sha256", default=None)
    aqua.add_argument("--require-scope", required=True,
                      choices=CAMPAIGN_SCOPE_KINDS,
                      help="what this submission is for, checked against the "
                           "plan's derived scope exactly as submit-joint does")
    aqua.add_argument("--campaign-identity", required=True,
                      help="the frozen campaign identity the plan's roster, "
                           "group roster, draw, capture and candidate menu "
                           "have to reproduce")
    aqua.add_argument("--campaign-identity-sha256", default=None)
    aqua.add_argument("--cost-in", required=True,
                      help="the cost payload the A-side is merged into: the "
                           "weight-only table, or the joint pass's merged "
                           "table. Left untouched")
    aqua.add_argument("--cost-in-sha256", default=None)
    aqua.add_argument("--cost-out", required=True,
                      help="where the merged table is written; it must not "
                           "exist, so an interrupted write cannot be mistaken "
                           "for a finished one")
    aqua.add_argument("--card", required=True,
                      help="the sensitivity card .npz the activation term is "
                           "computed from")
    aqua.add_argument("--card-sha256", default=None)
    aqua.add_argument("--formats", required=True,
                      help="the campaign's menu, comma-separated; it has to "
                           "name exactly the formats the payload carries, "
                           "because the stage states its acceptance against "
                           "the requested cells")
    aqua.add_argument("--act-dir", default=None,
                      help="cached real activations, for measured pricing")
    aqua.add_argument("--accept-joint-cells-outside-plan", action="store_true",
                      help="accept joint-priced cells the plan's own cost table "
                           "never priced, recording them as unverified. A joint "
                           "row is bound to the coordinate it names and to "
                           "nothing else, so by default such a cell refuses: "
                           "sealing a requested roster that names it is the "
                           "answer, and this flag is how an artifact sealed "
                           "against an older roster keeps that reuse explicit")
    lane = aqua.add_mutually_exclusive_group(required=True)
    lane.add_argument("--serving-lane", default=None,
                      help="lane id whose served_activation_quantization "
                           "declares which formats' activation grid the "
                           "runtime executes; there is no default")
    lane.add_argument("--lane-executes-all-activation-grids",
                      action="store_true",
                      help="assert that the lane executes every format's "
                           "activation grid fused")
    _add_submission_arguments(aqua)
    aqua.set_defaults(func=cmd_submit_aqua)

    merge = sub.add_parser("merge", help="one cost.pkl and journal from the rows")
    merge.add_argument("--workspace", required=True)
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=cmd_merge)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
