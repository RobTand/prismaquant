"""PrismaBuild rows for a Tessera campaign spec.

One owner for what every campaign and materialization row shares: the run spec,
its row classes and their placement attestations, the census-derived memory
demand a row is admitted with, and the PrismaBuild row itself. Fleet tools keep
argv, submission and printing; package planners build rows here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .campaign_container_spec import admission_image_reference, validate_container
from .tessera_campaign_cli import explicit_campaign_options


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
#: PrismaBuild action key that ran it.  Fleet data maintained beside the
#: fleet tools; this owner reads it and imports nothing from ``tools``.
FLEET_INTERPRETERS = (Path(__file__).resolve().parents[1] / "tools"
                      / "fleet_interpreters.json")
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
    """Validate active placement attestations and separately retained history."""

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
        for field in ("interpreters", "retired_interpreters"):
            interpreters = shape.get(field, {} if field == "retired_interpreters" else None)
            if not isinstance(interpreters, dict):
                raise RowClassRefused(
                    f"{source}: tag {tag!r} declares no {field} mapping")
            for interpreter, record in interpreters.items():
                if not isinstance(record, dict) or not isinstance(
                        record.get("attested_by"), str) or not record["attested_by"]:
                    raise RowClassRefused(
                        f"{source}: interpreter {interpreter} on tag {tag!r} names no "
                        "PrismaBuild action key that ran it")
        overlap = shape["interpreters"].keys() & shape.get("retired_interpreters", {}).keys()
        if overlap:
            raise RowClassRefused(
                f"{source}: tag {tag!r} lists interpreters as both active and retired: "
                f"{sorted(overlap)}")
    return table


def row_class(spec: dict, name: str = DEFAULT_ROW_CLASS) -> dict:
    """Resolve one class against the spec-level values it inherits.

    A spec with no ``classes`` block has exactly one class, the default, and it
    IS the spec: same interpreter, same environment, same tags, same container.
    That is the property the action key depends on, so it is stated here rather
    than left to the caller. A portable admission reference belongs to that
    container: a class replacing the container does not inherit its reference.
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
    if "container_admission_reference" in spec:
        base["container_admission_reference"] = spec["container_admission_reference"]
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
    if "container" in override:
        resolved.pop("container_admission_reference", None)
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
    """Check device capability; stamp or refuse recorded host attestation."""

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
            from prismaquant.dev_mode import seal_check

            attested = sorted(shape["interpreters"])
            history = (" This interpreter is retired; its retained receipt is history, "
                       "not placement admission."
                       if resolved["python"] in shape.get("retired_interpreters", {})
                       else "")
            seal_check(
                "host interpreter placement attestation", attested, resolved["python"],
                where=f"row class {resolved['name']!r} on tag {tag!r}", same=False,
                refusal=lambda: RowClassRefused(
                    f"row class {resolved['name']!r} runs {resolved['python']}, "
                    f"which is not attested on tag {tag!r}; that tag attests "
                    f"{attested or 'no host interpreter'}. Run it there once and "
                    f"add it to {FLEET_INTERPRETERS.name} with the PrismaBuild "
                    f"action key, or give the class a container.{history}"))


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
                 "--calibration-cache", "--calibration-cache-sha256",
                 "--source-identity-cache", "--source-identity-cache-sha256"}
    try:
        options = explicit_campaign_options(spec["campaign_argv"])
    except ValueError as exc:
        raise RuntimeError(f"{path}: campaign_argv: {exc}") from exc
    named = {flag for flag in forbidden if flag[2:].replace('-', '_') in options}
    if named:
        raise RuntimeError(
            f"{path}: campaign_argv names {sorted(named)}, which this tool "
            "owns per row")
    if "deadline_seconds" in options:
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
        try:
            resolved = row_class(spec, name)
        except RowClassRefused as exc:
            raise RowClassRefused(f"{path}: {exc}") from exc
        if "container" in resolved:
            # The bounded-capture environment contract is stated by the row
            # through shared campaign argv. Classes may override environment,
            # but do not own argv. Legacy rows keep their own purge delay.
            validate_container({"container": resolved["container"],
                                "env": resolved["env"]},
                               bounded=row_is_bounded(
                                   spec["campaign_argv"],
                                   where=f"{path}: row class {name!r}"))
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
#: ``row_memory_gb`` docstring cites is an earlier reading, recorded on the
#: RobTand/prismaquant#390 receipt rather than traced to one of these rows; it
#: sits inside this range.
#:
#: Session note: ``pb_mem_gb_must_track_the_checked_phase_plan``, and
#: RobTand/prismaquant#522, which records the three rows this default exists
#: to stop losing.  It is a fleet number with a date on it, so a spec that
#: knows its own box overrides it and says so in ``baseline_policy``.
DEFAULT_PROCESS_BASELINE_BYTES = 1_150_000_000


def process_baseline(spec: dict, *, where="spec") -> "tuple[int, str]":
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
    return process_baseline(spec, where=where)[0]


def guard_margin_bytes() -> int:
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


def row_memory_gb(spec: dict, members: list[str], census: dict, *, selected_source=False) -> int:
    """The row's memory demand, from what the row actually holds.

    Streaming rows use the phase resource plan, including the selected-source
    plan when requested. These are derived byte bounds, not measured peaks.
    The resident-source fallback charges three quantities:

    * the checkpoint, which is loaded whole in ``bfloat16`` and is the same for
      every row;
    * the selection's Hessians, ``in x in`` in fp32 per member, which is the
      full calibration accumulator;
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
    (``prismaquant/tessera_campaign.py``). Fold the reservation into the
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
    return row_memory_demand(spec, members, census,
                              selected_source=selected_source)["mem_gb"]


def row_memory_demand(spec: dict, members: list[str], census: dict, *,
                       selected_source=False) -> dict:
    """The row's demand and every term it is made of.

    ``row_memory_gb`` is this, reduced to its GiB. The terms are kept because
    a refusal has to name them: a row that dies on the admission predicate is
    diagnosable from the plan, the floor and the margin, and before this they
    were reachable only by unpickling a completed row's ``cost.pkl``
    (RobTand/prismaquant#522).
    """
    gib = 1024 ** 3
    baseline, policy = process_baseline(spec)
    margin = guard_margin_bytes()
    if explicit_campaign_options(spec['campaign_argv']).get("streaming", False):
        resource = streamed_resource_plan(spec, census, members,
                                           selected_source=selected_source)
        # A selected row that runs the streaming row head holds a window, not
        # the population, and is admitted against the window's plan
        # (RobTand/prismaquant#640).
        from prismaquant.tessera_row_stream import MEMORY_PLANS, ROW_HEAD_LOAD_ALL, ROW_HEAD_STREAM
        stream = bool(selected_source and MEMORY_PLANS[ROW_HEAD_STREAM] in resource
                      and row_head_dependency(spec['campaign_argv']) is None)
        plan_bytes = int(resource[MEMORY_PLANS[ROW_HEAD_STREAM if stream else ROW_HEAD_LOAD_ALL]])
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


def streamed_resource_plan(spec, census, members, *, selected_source=False):
    from prismaquant.autoscale import streamed_calibration_resources, selected_anchor_resources
    options = explicit_campaign_options(spec['campaign_argv'])
    baseline_bytes, baseline_policy = process_baseline(spec)
    shapes = census.get('unit_shapes') or {}
    counts = census.get('counts') or {}
    resource_options = dict(
        unit_shapes={n: shapes[n] for n in members}, counts=counts,
        max_act_rows=options.get('max_act_rows', int(spec.get('max_act_rows', 512))),
        cache_slots=options.get('streaming_cache_slots', 2),
        prefetch_workers=options.get('streaming_prefetch_workers', 1),
        headroom_gb=max(float(spec.get('headroom_gb', 24)),
                        options.get('streaming_cache_headroom_gb', 24.)),
        # Recorded beside ``memory_bytes``, never summed into it: the plan
        # stays pure phase deltas and the reservation is charged once, in
        # ``row_memory_demand``, on the demand.
        process_baseline_bytes=baseline_bytes,
        process_baseline_policy=baseline_policy)
    if selected_source:
        return selected_anchor_resources(spec['model'], **resource_options,
            anchor_batch_size=options.get('anchor_batch_size', 1),
            source_snapshot_policy=options.get('source_snapshot_policy', 'whole-layer-v1'),
            # A scoped row (``--source-scope mtp``) is planned over its own
            # source, as its runner reads it (PQ #1316).
            **({'source_scope': options.get('source_scope', None)}
               if 'source_scope' in options else {}),
            # The row's own campaign will hold this many host bytes of staged
            # artifacts, so the box that admits the row has to be told. A
            # dispatcher that planned without it would size a worker for a
            # campaign it is not about to run.
            publication_overlap_bytes=options.get('publication_overlap_bytes', 0),
            campaign_identity_bytes=options.get('campaign_identity_bytes', 0),
            # The campaign's default is the CPUs the row is admitted with,
            # which is the class's declared ``cpus`` demand.
            campaign_identity_threads=options.get('campaign_identity_threads', int(spec.get('cpus', 4))),
            **(dict(capture_load_policy=options.get('capture_load_policy', None))
               if 'capture_load_policy' in options else {}))
    return streamed_calibration_resources(spec['model'], **resource_options,
        nsamples=options.get('nsamples', 8), seqlen=options.get('seqlen', 512),
        capture_policy=options.get('streaming_capture_policy', 'legacy'),
        # A streamed capture reads its source through a recording owner; the
        # plan reports the owner's retained clean pages beside, not in, its
        # memory demand (PQ #1896).
        source_recording=bool(options.get('capture_calibration_out')))


def row_head_dependency(argv):
    """Apply the campaign's own head rule to the options it will execute."""
    from prismaquant.tessera_row_stream import stream_head_dependency
    options = explicit_campaign_options(argv)
    return stream_head_dependency(
        row_head=options.get('row_head', 'stream'), selected_source=True,
        capture_load_policy=options.get('capture_load_policy'),
        export_hessian_reference_policy=options.get('export_hessian_reference_policy'),
        max_rounds=options.get('max_rounds', 0), seed_checkpoint=options.get('seed_checkpoint'))


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


def row_is_bounded(argv: list[str], *, where: str = "campaign row") -> bool:
    """Whether this row runs the bounded capture path.

    One predicate, two readers: the row builder that merges the bounded
    capture environment, and the container validation that holds a spec to the
    SAME environment. A legacy row that declares its own purge delay is not a
    bounded row and is not refused for a contract it does not fall under.
    """
    try:
        options = explicit_campaign_options(argv)
    except ValueError as exc:
        raise RuntimeError(f"{where}: {exc}") from exc
    return (options.get('streaming_capture_policy') == 'shared-inputs-bounded-v1'
            or all(options.get(field) for field in (
                'streaming', 'units', 'calibration_cache', 'calibration_cache_sha256')))


def prismabuild_row(spec: dict, argv: list[str], *, mem_gb: int, timeout_s: int | None,
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
    from prismaquant.digests import DIRECT_ASCII_SPACED_LAX
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
    bounded = (module == "prismaquant.tessera_campaign" and row_is_bounded(
        argv, where=f"row class {row_class_name!r}"))
    if bounded:
        from prismaquant.autoscale import BOUNDED_CAPTURE_ENV, require_bounded_capture_environment
        env = {**BOUNDED_CAPTURE_ENV, **env}
        require_bounded_capture_environment(env)
    command = [resolved["python"], "-u", "-m", module, *argv]
    container_image = None
    if "container" in resolved:
        container_spec = {"container": resolved["container"], "env": env}
        if resolved.get("container_admission_reference") is not None:
            container_spec["container_admission_reference"] = resolved["container_admission_reference"]
        validate_container(container_spec, bounded=bounded)
        command = ["python3", "-m", "tools.tessera_campaign_container", "--spec",
                   DIRECT_ASCII_SPACED_LAX.text(container_spec), "--", *command]
        # The class owns the image: whatever container this row resolved runs
        # is what PrismaBuild must find on the claiming box before the claim
        # (RobTand/prismabuild#714). Without an explicit portable override,
        # an archive-backed class declares nothing: its loader establishes
        # the image inside the action rather than requiring local presence.
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
