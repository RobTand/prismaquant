#!/usr/bin/env python3
"""Publish the distributed joint-AURA cost campaign's PB rows (§5).

A receipt-driven **submitter**, never a scheduler. It publishes rows PB
owns; it never claims, places, retries, reorders, or steers work, holds no
long-running state, and reads no capacity to choose a box. Placement is the
static policy one shared GB10 class tag on every quantum row; PB's
ready-order, loop counts, and tier tokens do the balancing. If the static
policy starves a box, that is a PB placement capability gap to file, not a
knob to turn here.

Order (§5.2): stage A first; then quanta, published descending by layer id
as stage A seals them. A quantum is publishable once a sealed stage-A proof
covers its checkpoint -- the completed ``adjoint-capture.json`` or, before
it lands, the checkpoint band of the quantum's checkpoint (PQ #993) -- and
its record's slice validates against that proof (the per-record slice
gate). A validated proof also stops stage-A republication. Re-run it (cron, a shell loop, or a
human) as stage A completes; within a run every publishable row is
published — the tool never waits on a worker.

Idempotence: ``<output_root>/layer-quanta/campaign-state.json`` is the
campaign's own machine-readable state (atomic append of submission events,
never edits). A re-run publishes nothing already terminally executed; any
other re-publication is a CAS attach to the same sealed action key, never a
repartition.

``--dry-run`` prints the submission plan with digests and submits nothing.

Shapes owned elsewhere (fixtures here, never imports): the layer-quantum
record (§3, built in parallel by the producer) and the stage-A receipt
(§3.3, built in parallel by stage A). Real dispatch is the coordinator's
call after the runtime's cutover check; this tool's tests use fixtures.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

if __package__:
    from tools.tessera_campaign_container import (
        CONTAINER_IMAGE_FLAG,
        STAGE_B_SPILL_ENV,
        container_cache_environment,
        admission_image_reference,
        local_scratch_environment,
        produced_spool_environment,
        stage_b_spill_environment,
    )
    from prismaquant.joint_layer_quanta import (
        canonical_sha256 as _canonical_receipt_sha256,
    )
    from prismaquant.joint_quantum_handoff import HANDOFF_LOAD_PHASE
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tessera_campaign_container import (
        CONTAINER_IMAGE_FLAG,
        STAGE_B_SPILL_ENV,
        container_cache_environment,
        admission_image_reference,
        local_scratch_environment,
        produced_spool_environment,
        stage_b_spill_environment,
    )
    from prismaquant.joint_layer_quanta import (
        canonical_sha256 as _canonical_receipt_sha256,
    )
    from prismaquant.joint_quantum_handoff import HANDOFF_LOAD_PHASE

PBRUN = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py")
PBWAIT = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbwait.py")

#: §5.1 plan defaults. One shared GB10 class tag on every quantum row; PB
#: owns which box claims.  PB's matcher requires *every* tag a row lists
#: (``wanted.issubset(offer.tags)``), and each live Spark offers ``gb10``
#: plus its own host name -- so the host pair this default used to carry
#: (``sparky``, ``sparklina``) admitted neither box (#831).
CONSUMER_TAGS = ("gb10",)
SUBMISSION_PRIORITY = -5
ADJOINT_TAG = "sparky"
DEV_MODE_ENV = "PRISMAQUANT_DEV_MODE=1"

#: The sealed staged-tier declaration every campaign row carries as the
#: payload ``--allowed-tiers`` flag (never ambient env: the container
#: forwards no ambient action environment into the payload). RAM first,
#: SSD explicitly emitted; pool/HDD bulk opens refuse under it. The value
#: must equal ``prismaquant.staged_tier_policy.DEFAULT_ALLOWED_TIERS``
#: and the joint entrypoints' parser default — pinned by
#: ``tests/test_dispatch_joint_quanta.py``.
STAGED_ALLOWED_TIERS = "ram,ssd"

#: §5.2: each chunk's progress phase seals a 900-second stall allowance.
CHUNK_PROGRESS_GRACE_S = 900
#: The head phase's stall allowance. Not pinned by the contract (only the
#: chunk grace is); the sealed default below is overridable via
#: --head-grace-s and pinned by the dispatcher's own tests.
HEAD_PROGRESS_GRACE_S = 1800

RECORD_SCHEMA = "prismaquant.joint_layer_quanta.v1"
ADJOINT_SCHEMA = "prismaquant.joint_adjoint_capture.v1"
#: The schema of the stage-A data manifest the dispatcher binds
#: (``prismabuild.core.DATA_MANIFEST_SCHEMA_V1``, restated: this module
#: stays free of the fleet runtime import).
DATA_MANIFEST_SCHEMA_V1 = "prismaquant.prismabuild.data_manifest.v1"
#: The read-plan schema (``prismabuild.core.DATA_MANIFEST_SCHEMA_V2``):
#: same entries plus ``read_plan.phases`` with ``entry_indices`` for
#: repeated read order. v2 forbids ``annotations.phases``; both schemas
#: keep ``annotations`` carrying the parent read-set digest and the sealed
#: plan/prepared digests this dispatcher checks.
DATA_MANIFEST_SCHEMA_V2 = "prismaquant.prismabuild.data_manifest.v2"
#: gzip magic, for the transparent manifest read below: detection is by
#: header, never by suffix. The bound digest always covers the wire bytes
#: pbrun ingests (compressed when compressed); parsing decompresses.
_GZIP_MAGIC = b"\x1f\x8b"
_HEX64 = frozenset("0123456789abcdef")
#: The campaign container spec every row seals unless ``--spec`` names
#: another. It declares the produced output spool (PQ #1012), which the
#: Stage A row requires; the spec it replaced, ``spec-hostcap32-ram-dev.json``
#: beside it, declared none and is left unchanged.
DEFAULT_SPEC_PATH = Path(
    "/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913"
    "/allocation/joint-panel/spec-hostcap32-ram-dev-spool.json")
SPEC_PATH = DEFAULT_SPEC_PATH
#: The produced output spool's root and byte bound (``produced_output_spool``).
PRODUCED_SPOOL_ROOT_ENV = "PRISMABUILD_PRODUCED_SPOOL_ROOT"
PRODUCED_SPOOL_MAX_ENV = "PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"
#: Bytes per element of the execution dtypes a model config may name.
_CONFIG_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
#: The host spool window opt-in (PB #910), which the Stage A row seals.
PRODUCED_SPOOL_HOST_WINDOW_ENV = "PRISMABUILD_PRODUCED_SPOOL_HOST_WINDOW"
#: Opt-ins PrismaBuild reads from the producer's sealed environment, each "0"
#: or "1": the paced export (PB #891) and the host spool window (PB #910).
PRODUCED_SPOOL_OPT_IN_ENV = ("PRISMABUILD_PRODUCED_SPOOL_PACED_EXPORT",
                             PRODUCED_SPOOL_HOST_WINDOW_ENV)
STATE_FILENAME = "campaign-state.json"

#: Refusal exits: 3 = the stage-A precondition (or the campaign binding)
#: failed closed; 1 = a submission itself failed.
EXIT_PRECONDITION_REFUSED = 3
EXIT_SUBMIT_FAILED = 1


class DispatchRefused(Exception):
    """Fail closed: no receipt, a stale receipt, or a mixed campaign."""


#: The sealed static prepared-input annotation a production-accepted
#: executable manifest carries (PQ #917). Mirrors
#: ``prismaquant.joint_layer_quanta.PREPARED_INPUT_SCHEMA``; this submitter
#: deliberately does not import the producer package, so the comparison is
#: by exact string (a mover-reference validator output never carries it).
PREPARED_INPUT_SCHEMA = "prismaquant.joint_layer_quanta.prepared_input.v1"


class ExecutableBindingUnsupported(DispatchRefused):
    """An executable quantum row names no acceptable output binding.

    No PB produced-output binding validator is accepted yet (the PB732/735
    stacks are still unaccepted), so no executable manifest -- however
    plausible its ``render_prerequisite`` dictionary looks -- can be
    admitted for production dispatch. Sequencing/phase artifacts remain
    useful for read-plan qualification; production dispatch of the legacy
    slice rows is unchanged.
    """


class ProducedOutputDeclarationUnsupported(DispatchRefused):
    """The client in front of this dispatcher cannot SEAL a template.

    A produced-output owner is admitted only when the queue row carries the
    template AND the sealed request carries the matching declaration,
    validated against an input ingested under
    ``core.PRODUCED_OUTPUT_TEMPLATE_INPUT_ID``. The deployed ``pbrun`` does
    all of that behind ``--produced-output-template``; an older one does
    none of it.

    So this refusal is CONDITIONAL on the client in hand, never blanket:
    threading a flag an older client ignores would submit a capture that
    writes its boundary entries and then cannot read them, and adding a
    payload flag instead would advertise a declaration that admitted
    nothing.
    """


def _pbrun_seals_produced_output(pbrun: Path = PBRUN) -> bool:
    """Does THIS client carry the produced-output template seal?

    Asked of the client that will actually run, by its own help, rather
    than assumed from a version or a date. A client that cannot be asked
    is treated as not carrying it -- the fail-closed direction.
    """

    try:
        probe = subprocess.run([sys.executable, str(pbrun), "--help"],
                               capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--produced-output-template" in (probe.stdout or "")


def build_stage_a_produced_template(*, output_prefix, tier: str,
                                    artifact_max_bytes: int,
                                    group_size: int,
                                    max_entry_tensor_bytes: int,
                                    concurrent_groups: int = 2) -> dict:
    """The Stage A boundary template this submission would declare.

    Derived, never constant: ``artifact_max_bytes`` is the EFFECTIVE
    artifact budget this submission forwards (the plan's sealed
    ``boundary_storage.max_artifact_bytes`` or the run's
    ``--artifact-budget-bytes`` override, whichever the payload carries),
    and it becomes the durable origin class maximum. The tier window is
    derived from the ACTUAL maximum publication group, which is a
    different quantity from the retained origin peak.

    ``concurrent_groups`` is how many groups' worth of stage space the
    window funds. Two is the synchronous loop. Every group past two is
    read-ahead credit the bound owner spends on publishing at
    write-complete, retaining a layer's input boundary across probe passes
    and staging the next plane ahead (#887); the owner derives the count
    back from the sealed ``window_gib``, so this is the one place it is
    chosen.

    The single implementation lives with the runtime binding
    (``prismaquant.stage_a_produced_output.build_boundary_template``) so
    the submitted declaration and the bound instance cannot drift into two
    spellings of one contract.
    """

    from prismaquant.stage_a_produced_output import build_boundary_template

    return build_boundary_template(
        output_prefix=output_prefix, tier=tier,
        artifact_max_bytes=int(artifact_max_bytes),
        group_size=int(group_size),
        max_entry_tensor_bytes=int(max_entry_tensor_bytes),
        concurrent_groups=int(concurrent_groups))


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path, *, where: str) -> dict:
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise DispatchRefused(f"{where}: unreadable JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DispatchRefused(f"{where}: not a JSON object at {path}")
    return value


def load_records(records_dir: Path) -> list[tuple[Path, dict]]:
    """Load the sealed layer records, highest layer first (§5.2 order).

    Every record must share one campaign block (plan, prepared, manifest,
    scope, roster digests), and all are unbound (pre-A) or all bound to
    their stage-A slices (PQ #993); anything else is a mixed campaign and
    refuses."""
    paths = sorted(records_dir.glob("layer-*.json"))
    if not paths:
        raise DispatchRefused(f"no layer records at {records_dir}")
    loaded = [(path, _load_json(path, where="layer record")) for path in paths]
    for path, record in loaded:
        if record.get("schema") != RECORD_SCHEMA:
            raise DispatchRefused(
                f"{path}: record schema is not {RECORD_SCHEMA!r}")
    ids = [record.get("quantum_id") for _, record in loaded]
    if len(set(ids)) != len(ids):
        raise DispatchRefused("layer records carry duplicated quantum ids")
    first = loaded[0][1]["campaign"]
    for path, record in loaded[1:]:
        if record.get("campaign") != first:
            raise DispatchRefused(
                f"{path}: campaign block differs (mixed campaign)")
    for path, record in loaded:
        if record.get("adjoint", {}).get("receipt_sha256") is not None:
            raise DispatchRefused(
                f"{path}: record binds a whole stage-A receipt, not its slice "
                "(PQ #993): regenerate")
    bindings = {record.get("adjoint", {}).get("slice_sha256") is not None
                for _, record in loaded}
    if len(bindings) != 1:
        raise DispatchRefused(
            "layer records mix unbound and slice-bound stage-A bindings")
    # Before stage A publishes, the sealed records are unbound (§5.2 seals
    # records first); after, each binds its own slice. Mixed bindings are a
    # mixed campaign.
    ordered = sorted(loaded,
                     key=lambda item: item[1].get("layer", -1),
                     reverse=True)
    layers = [record.get("layer") for _, record in ordered]
    if any(not isinstance(layer, int) for layer in layers):
        raise DispatchRefused("a layer record carries no integer layer")
    return ordered




def _is_hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(char in _HEX64 for char in value))


def _stage_manifest_binding(adjoint_manifest: Path, campaign: Mapping) -> dict:
    """Validate the stage-A data manifest and derive its submission binding.

    The payload's tier redirect and progress window both hang off this
    document, so both are derived here, from its validated annotations --
    never from a hardcoded list:

    * ``data_manifest_sha256``: sha256 of the submitted manifest wire bytes
      (compressed when compressed), the digest the residency map will carry
      and ``bind_residency_manifest`` must equal for any redirect;
    * ``read_manifest_sha256``: the annotated parent read-set digest
      (``annotations.parent_manifest_sha256``), the run's read-parent
      identity, 64-hex checked;
    * ``phases``: the read-phase names in manifest order -- v1 from
      ``annotations.phases``, v2 from ``read_plan.phases`` (v2 forbids
      ``annotations.phases``; ``prismabuild.core`` holds that rule, this
      reader mirrors it). pbrun's linear rule requires every read name
      declared, in order; the caller declares exactly this list.

    Anything else -- an unreadable file, a non-manifest, a phase table that
    is empty, unnamed, or duplicated, a malformed parent digest, a manifest
    built against another plan/prepared pair, an entry count that drifted
    from the entries -- is a mixed or corrupt campaign and refuses before
    anything publishes.

    This reader is deliberately scoped rather than shared with
    ``prismaquant.joint_layer_quanta.phase_ranges``: that helper covers the
    v1 annotation table only, demands non-empty positive-byte entries, and
    raises ``ValueError`` -- it cannot validate v2 read plans, empty fixture
    manifests, or the campaign cross-checks, and importing it would drag the
    producer package into the submitter.
    """
    path = Path(adjoint_manifest)
    try:
        wire = path.read_bytes()
    except OSError as exc:
        raise DispatchRefused(
            f"stage-A data manifest unreadable at {path}: {exc}") from exc
    data_sha256 = _sha_bytes(wire)
    raw = wire
    if raw[:2] == _GZIP_MAGIC:
        import gzip
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError) as exc:
            raise DispatchRefused(
                f"stage-A data manifest is not valid gzip at {path}: "
                f"{exc}") from exc
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise DispatchRefused(
            f"stage-A data manifest is not JSON at {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DispatchRefused(
            f"stage-A data manifest is not a JSON object at {path}")
    schema = manifest.get("schema")
    if schema not in (DATA_MANIFEST_SCHEMA_V1, DATA_MANIFEST_SCHEMA_V2):
        raise DispatchRefused(
            f"stage-A data manifest schema is not a data manifest at {path}: "
            f"{schema!r}")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise DispatchRefused(
            f"stage-A data manifest has no entries at {path}")
    count = manifest.get("entry_count")
    if count is not None and count != len(entries):
        raise DispatchRefused(
            f"stage-A data manifest entry_count {count!r} names "
            f"{len(entries)} entries at {path}")
    annotations = manifest.get("annotations")
    if not isinstance(annotations, dict):
        raise DispatchRefused(
            f"stage-A data manifest has no annotations at {path}")
    if schema == DATA_MANIFEST_SCHEMA_V2:
        if "phases" in annotations:
            raise DispatchRefused(
                f"stage-A data manifest v2 uses read_plan, not "
                f"annotations.phases, at {path}")
        read_plan = manifest.get("read_plan")
        if not isinstance(read_plan, dict):
            raise DispatchRefused(
                f"stage-A data manifest v2 has no read_plan at {path}")
        raw_phases = read_plan.get("phases")
        table = "read_plan.phases"
    else:
        raw_phases = annotations.get("phases")
        table = "annotations.phases"
    if not isinstance(raw_phases, list) or not raw_phases:
        raise DispatchRefused(
            f"stage-A data manifest {table} is empty at {path}")
    phases: list[str] = []
    for entry in raw_phases:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise DispatchRefused(
                f"stage-A data manifest {table} names an unnamed phase "
                f"at {path}")
        if name in phases:
            raise DispatchRefused(
                f"stage-A data manifest {table} repeats phase {name!r} "
                f"at {path}")
        phases.append(name)
    parent = annotations.get("parent_manifest_sha256")
    if not _is_hex64(parent):
        raise DispatchRefused(
            f"stage-A data manifest parent_manifest_sha256 is not a digest "
            f"at {path}")
    sealed_parent = campaign.get("read_manifest_sha256")
    if parent != sealed_parent:
        # A manifest from another lineage would otherwise launch: the run's
        # receipt binds this read parent, so an incompatible one is a mixed
        # campaign, refused here rather than at the receipt.
        raise DispatchRefused(
            f"stage-A data manifest parent {parent[:12]}... is not the "
            f"sealed campaign read parent {sealed_parent!r} at {path}: "
            f"mixed campaign")
    for key in ("plan_sha256", "prepared_sha256"):
        sealed = annotations.get(key)
        if sealed != campaign.get(key):
            raise DispatchRefused(
                f"stage-A data manifest {key} {sealed!r} is not the sealed "
                f"campaign {campaign.get(key)!r} at {path}: mixed campaign")
    return {"data_manifest_sha256": data_sha256,
            "read_manifest_sha256": parent, "phases": phases}


def _executable_row_parts(record: dict, *, output_root: Path,
                            head_grace_s: int):
    """Pure executable-row construction from sealed inputs (no gate).

    Resolves the row's executable manifest, verifies its wire bytes hash to
    the sealed digest, and derives the read-phase progress declarations in
    manifest order. Consults no output binding and bypasses no production
    gate: production dispatch (:func:`quantum_argv`) refuses every
    executable row with :class:`ExecutableBindingUnsupported` before
    reaching here. Tests exercise manifest/phase propagation through this
    helper directly.
    """
    quantum_id = record["quantum_id"]
    executable = record.get("executable_readset")
    manifest = Path(executable.get("manifest_path", ""))
    if not manifest.is_absolute():
        manifest = output_root / manifest
    staged_sha256 = _executable_manifest_digest(
        record, output_root=output_root)
    phases = executable.get("phases")
    if not isinstance(phases, list) or not phases or any(
            type(name) is not str or not name for name in phases):
        raise DispatchRefused(
            f"quantum {quantum_id!r} seals no executable phase list")
    progress = [("head", head_grace_s)]
    for name in phases:
        if name == "head":
            continue
        grace = (HEAD_PROGRESS_GRACE_S if name == "checkpoint-load"
                 else CHUNK_PROGRESS_GRACE_S)
        progress.append((name, grace))
    return manifest, staged_sha256, progress


def _executable_manifest_digest(record: dict, *, output_root: Path) -> str:
    """The sealed executable-manifest digest a bound quantum row binds.

    Mirrors :func:`_slice_manifest_digest` for the post-capture executable
    readset: the row's ``executable_readset.manifest_sha256`` names the one
    manifest pbrun stages for this row, and its wire bytes must hash to the
    sealed digest; a drifted or absent manifest refuses before anything
    publishes. Records without the block keep the legacy slice path.
    """
    block = record.get("executable_readset")
    if not isinstance(block, dict):
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} carries no executable "
            "readset")
    manifest = block.get("manifest_path")
    if not isinstance(manifest, str) or not manifest:
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} names no executable "
            "manifest")
    path = Path(manifest)
    if not path.is_absolute():
        path = Path(output_root) / path
    declared = block.get("manifest_sha256")
    if not _is_hex64(declared):
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} seals no executable digest")
    try:
        actual = _sha_bytes(path.read_bytes())
    except OSError as exc:
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} executable manifest "
            f"unreadable at {path}: {exc}") from exc
    if actual != declared:
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} executable manifest at "
            f"{path} does not hash to the sealed digest")
    return actual


def _executable_prepared_input(record: dict, *,
                                 output_root: Path) -> tuple[dict, dict]:
    """Load and validate the sealed static prepared-input contract (PQ #917).

    Returns the bound manifest document and its ``prepared_input``
    annotation. The wire bytes must hash to the sealed digest (checked
    first, through the existing owner); the document must then carry a
    complete prepared-input contract bound to this exact row:

    * the annotation schema is the sealed prepared-input schema (a
      produced-output mover reference never carries it);
    * its production-pickle and unit-roster digests equal the manifest's
      own sealed render prerequisite (a foreign roster refuses);
    * its prepared digest equals the row's sealed campaign digest;
    * its windows cover exactly the row's sealed retained-window indices
      with non-empty member rosters and entry references;
    * every window's ``render-{w:02d}`` phase exists with exactly the
      window's entries, immediately before that window's first replay
      phase (source phases stay source-only);
    * the row's bound block agrees with the manifest (phases and the
      prepared-input annotation itself).

    Anything else raises :class:`ExecutableBindingUnsupported`: legacy
    sequencing-only rows (no annotation at all) keep the historical
    refusal message, while a malformed or foreign prepared contract
    names its own gate. No mover-reference validator is consulted here:
    passing ``validate_produced_output_batch`` proves a mover reference,
    never a consumer read capability.
    """
    quantum_id = record.get("quantum_id")
    block = record.get("executable_readset")
    if not isinstance(block, dict):
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} carries no executable readset block")
    staged_sha256 = _executable_manifest_digest(
        record, output_root=output_root)
    manifest = block.get("manifest_path")
    path = Path(manifest) if isinstance(manifest, str) else None
    if path is None or not manifest:
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} names no executable manifest")
    if not path.is_absolute():
        path = Path(output_root) / path
    try:
        wire = path.read_bytes()
    except OSError as exc:
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} executable manifest unreadable at "
            f"{path}: {exc}") from exc
    if hashlib.sha256(wire).hexdigest() != staged_sha256:
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} executable manifest at {path} does "
            "not hash to the sealed digest")
    try:
        manifest_doc = json.loads(gzip.decompress(wire).decode("utf-8"))
    except (OSError, EOFError, ValueError) as exc:
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} executable manifest at {path} is not "
            f"the sealed manifest wire: {exc}") from exc
    if not isinstance(manifest_doc, dict):
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} executable manifest is not an object")
    annotations = manifest_doc.get("annotations")
    if not isinstance(annotations, dict):
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} executable manifest has no annotations")
    prepared = annotations.get("prepared_input")
    if not isinstance(prepared, dict) or \
            prepared.get("schema") != PREPARED_INPUT_SCHEMA:
        # R3: no invented admission. No accepted PB output-binding
        # validator exists (PB732/735 are still unaccepted stacks), so a
        # sequencing-only executable plan proves no capability. Refuse
        # before any staged read.
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} carries an executable readset, but no "
            "accepted PB produced-output binding validator exists "
            "(PB732/735 stacks unaccepted): executable plans are "
            "sequencing-only and not production-runnable -- refusing")
    prerequisite = annotations.get("render_prerequisite")
    if not isinstance(prerequisite, dict):
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} prepared contract names no sealed "
            "render prerequisite: refusing")
    for key in ("production_pkl_sha256", "unit_roster_sha256"):
        if prepared.get(key) != prerequisite.get(key):
            raise ExecutableBindingUnsupported(
                f"quantum {quantum_id!r} prepared contract names a foreign "
                f"{key}, not the sealed render prerequisite: refusing")
    campaign = record.get("campaign")
    if not isinstance(campaign, dict) or \
            prepared.get("prepared_sha256") != campaign.get("prepared_sha256"):
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} prepared contract names another "
            "prepared payload, not the sealed campaign: refusing")
    bound_slice = block.get("slice_sha256")
    adjoint = record.get("adjoint")
    if bound_slice is None or annotations.get("slice_sha256") != bound_slice or \
            not isinstance(adjoint, dict) or \
            adjoint.get("slice_sha256") != bound_slice:
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} prepared contract binds another "
            "stage-A slice, not this row's sealed capture: refusing")
    windows = record.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} seals no retained windows: refusing")
    indices = [window.get("window_index") if isinstance(window, dict)
               else None for window in windows]
    sealed_windows = prepared.get("windows")
    if not isinstance(sealed_windows, list) or \
            [window.get("window_index") if isinstance(window, dict)
             else None for window in sealed_windows] != indices:
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} prepared contract does not cover the "
            f"sealed retained windows {indices!r}: refusing")
    phases = manifest_doc.get("read_plan", {}).get("phases")
    if not isinstance(phases, list) or not phases:
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} executable manifest has no read plan: "
            "refusing")
    replay_mode = block.get("replay_mode")
    if replay_mode != (manifest_doc.get("annotations") or {}).get("replay_mode") \
            or replay_mode not in (None, "spill"):
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} bound replay mode {replay_mode!r} is not "
            "the sealed manifest's: refusing")
    by_name = {phase.get("name"): phase for phase in phases
               if isinstance(phase, dict)}
    order = [phase.get("name") for phase in phases
             if isinstance(phase, dict)]
    entries = manifest_doc.get("entries")
    if not isinstance(entries, list):
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} executable manifest has no entries: "
            "refusing")
    for window in sealed_windows:
        window_index = window.get("window_index")
        members = window.get("members")
        refs = window.get("entry_indices")
        if not isinstance(members, list) or not members or any(
                not isinstance(pair, (list, tuple)) or len(pair) != 2
                for pair in members):
            raise ExecutableBindingUnsupported(
                f"quantum {quantum_id!r} prepared window {window_index!r} "
                "seals no member roster: refusing")
        if not isinstance(refs, list) or not refs or any(
                type(index) is not int or isinstance(index, bool)
                or not 0 <= index < len(entries) for index in refs):
            raise ExecutableBindingUnsupported(
                f"quantum {quantum_id!r} prepared window {window_index!r} "
                "seals no render entries: refusing")
        render_name = f"render-{window_index:02d}"
        # The render phase sits immediately before the phase that consumes
        # its window next: the window's first replay (windowed), or under
        # the one-pass spill the first probe capture for window 0 and the
        # next window's render after it (PQ #1011).
        if replay_mode == "spill":
            replay_name = ("spill-p0" if window_index == 0 else
                           f"render-{window_index + 1:02d}"
                           if window_index + 1 < len(sealed_windows) else None)
        else:
            replay_name = f"replay-{window_index:02d}-p0"
        render_phase = by_name.get(render_name)
        if not isinstance(render_phase, dict) or \
                list(render_phase.get("entry_indices", None)
                     or []) != list(refs):
            raise ExecutableBindingUnsupported(
                f"quantum {quantum_id!r} prepared window {window_index!r} "
                f"stages no {render_name} phase with exactly its render "
                "entries: refusing")
        if render_name not in order:
            follows = False
        elif replay_name is None:
            follows = order.index(render_name) + 1 == len(order)
        else:
            follows = (replay_name in order and
                       order.index(render_name) + 1 == order.index(replay_name))
        if not follows:
            raise ExecutableBindingUnsupported(
                f"quantum {quantum_id!r} prepared window {window_index!r} "
                f"does not stage {render_name} immediately before its "
                "replay phases: refusing")
    manifest_names = [phase.get("name") for phase in phases
                      if isinstance(phase, dict)]
    if list(block.get("phases", None) or []) != manifest_names:
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} bound phase list is not the sealed "
            "manifest order: refusing")
    block_prepared = block.get("prepared_input")
    if not isinstance(block_prepared, dict):
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} names no bound prepared contract: "
            "refusing")
    for key in ("schema", "production_pkl_sha256", "unit_roster_sha256",
                "prepared_sha256"):
        if block_prepared.get(key) != prepared.get(key):
            raise ExecutableBindingUnsupported(
                f"quantum {quantum_id!r} bound prepared contract is not "
                "the sealed manifest annotation: refusing")
    block_windows = block_prepared.get("windows")
    if not isinstance(block_windows, list) or len(block_windows) != len(
            sealed_windows):
        raise ExecutableBindingUnsupported(
            f"quantum {quantum_id!r} bound prepared contract is not the "
            "sealed manifest annotation: refusing")
    for bound_window, sealed_window in zip(block_windows, sealed_windows):
        if not isinstance(bound_window, dict):
            raise ExecutableBindingUnsupported(
                f"quantum {quantum_id!r} bound prepared contract is not "
                "the sealed manifest annotation: refusing")
        for key in ("window_index", "members", "entry_indices"):
            if bound_window.get(key) != sealed_window.get(key):
                raise ExecutableBindingUnsupported(
                    f"quantum {quantum_id!r} bound prepared contract is "
                    "not the sealed manifest annotation: refusing")
        staged = []
        for index in sealed_window.get("entry_indices", []):
            staged.append(
                {key: entries[index][key]
                 for key in ("path", "offset", "bytes", "sha256")})
        if bound_window.get("entries") != staged:
            raise ExecutableBindingUnsupported(
                f"quantum {quantum_id!r} bound prepared entries are not "
                "the sealed manifest entries: refusing")
    return manifest_doc, prepared


def _slice_manifest_digest(record: dict, *, output_root: Path) -> str:
    """The sealed slice digest a quantum row actually binds.

    The row's ``read_set.manifest_sha256`` names the slice manifest pbrun
    stages for this row -- not the campaign parent the record also carries.
    The file is read where the row reads it (relative manifests resolve
    against the output root, as the row builder does) and its wire bytes
    must hash to the sealed digest; a drifted or absent slice refuses
    before anything publishes.
    """
    read_set = record.get("read_set")
    if not isinstance(read_set, dict):
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} carries no read_set")
    manifest = read_set.get("manifest_path")
    if not isinstance(manifest, str) or not manifest:
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} names no slice manifest")
    path = Path(manifest)
    if not path.is_absolute():
        path = Path(output_root) / path
    declared = read_set.get("manifest_sha256")
    if not _is_hex64(declared):
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} seals no slice digest")
    try:
        actual = _sha_bytes(path.read_bytes())
    except OSError as exc:
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} slice manifest unreadable "
            f"at {path}: {exc}") from exc
    if actual != declared:
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} slice manifest bytes "
            f"do not hash to the sealed {declared[:12]} at {path}")
    return declared


def _row_manifest_sha256(record: dict) -> str:
    """The digest the submitted row actually stages (PQ #917).

    Prepared-input executable rows stage their bound executable manifest;
    every other row stages its legacy slice manifest.
    """
    executable = record.get("executable_readset")
    if isinstance(executable, dict) and _is_hex64(
            executable.get("manifest_sha256")):
        return executable["manifest_sha256"]
    return record["read_set"]["manifest_sha256"]


#: The Stage A row's host memory reservation when its plan states no
#: combined bound (``aggregate_memory_bytes``): the reservation every Stage A
#: row sealed before the plan's bound was read (PQ #997).
STAGE_A_UNBOUNDED_MEMORY_GIB = 104


def stage_a_memory_gib(campaign: Mapping) -> int:
    """The Stage A row's ``mem_gb``: the plan's combined physical bound, in GiB.

    ``aggregate_memory_bytes`` is the bound the plan states for the capture's
    host side and its device envelope together
    (``dispatch_tessera_campaign.joint_submission_memory_bound`` reads it
    first too), rounded up to the whole GiB ``--demand`` counts in. The GLM
    plan states 108447924224 B, so the row reserves 101 GiB. A constant 104
    reserved 3 GiB more than the plan's own bound, and a GB10 whose live offer
    was 102 GiB could never place the row (PQ #997, 2026-09-23). A plan that
    states no bound keeps ``STAGE_A_UNBOUNDED_MEMORY_GIB``.
    """
    plan = json.loads(Path(campaign["plan_path"]).read_text())
    bound = plan.get("aggregate_memory_bytes")
    if bound is None:
        return STAGE_A_UNBOUNDED_MEMORY_GIB
    if isinstance(bound, bool) or not isinstance(bound, int) or bound <= 0:
        raise DispatchRefused(
            f"plan {campaign['plan_path']} states aggregate_memory_bytes "
            f"{bound!r}, not a positive integer byte count")
    return -(-bound // 1024 ** 3)


def stage_a_spool_window_bytes(campaign: Mapping) -> int:
    """The Stage A row's local output window: two cotangent planes (PQ #1110).

    The reverse chain reads the cotangent plane it wrote one layer earlier
    from the producing box's own spool, so the spool holds one live plane
    and one more for the writes and exports turning over
    (``produced_output_spool.two_plane_window_bytes``). A plane is the
    plan's ``n_probes`` x one entry per batch of ``probe_microbatch`` rows
    of its ``n_calib_samples`` (``produced_output_spool.plane_partitions``,
    the partition the capture writes and binds, PQ #1121). Each entry is
    reserved at the writer's bound: one full batch's boundary tensor bytes
    plus the per-entry file envelope, a partial last batch included,
    because PrismaBuild reserves every entry of a group at the bound entry
    size (``boundary_group_ceiling_bytes``). The
    tensor is ``rows x calib_seqlen x hidden_size x hc_mult`` elements of the
    model config's dtype; ``hc_mult`` is the residual-stream count the GLM
    profile expands to, read off the same config key it reads, and 1 for a
    config that declares none. At R12's shape (4 probes, 512 one-row
    batches, 512 tokens, 4096 x 4 bf16) that is 68,987,912,192 B (64.25 GiB);
    at ``probe_microbatch`` 4 it is 128 four-row entries a plane,
    68,786,585,600 B.

    This is the planning derivation the row seals. The capture recomputes it
    from the live model at bind and refuses a sealed window below it
    (``StreamedBoundaryArtifacts._require_local_window``), so a config this
    reads differently from the runner fails before any forward work.
    """

    from prismaquant.produced_output_spool import (plane_partitions,
                                                   two_plane_window_bytes)

    plan = json.loads(Path(campaign["plan_path"]).read_text())
    try:
        execution = plan["execution"]
        n_probes = int(execution["n_probes"])
        n_rows = int(execution["n_calib_samples"])
        seqlen = int(execution["calib_seqlen"])
        microbatch = int(execution.get("probe_microbatch", 0))
        group_size = int(execution["boundary_storage"]["prefetch_batches"])
        model = Path(plan["model"])
        rows, row_offsets = plane_partitions(n_rows=n_rows,
                                             probe_microbatch=microbatch)
    except (KeyError, TypeError, ValueError) as exc:
        raise DispatchRefused(
            f"plan {campaign['plan_path']} does not state the Stage A plane "
            f"geometry the spool window is derived from: {exc!r}") from exc
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config") or config
    dtype = text.get("dtype") or text.get("torch_dtype") or config.get(
        "dtype") or config.get("torch_dtype")
    if dtype not in _CONFIG_DTYPE_BYTES or not isinstance(
            text.get("hidden_size"), int):
        raise DispatchRefused(
            f"model config {model / 'config.json'} does not state a hidden "
            f"size and a known dtype (dtype {dtype!r}); the Stage A spool "
            "window cannot be derived")
    tensor_bytes = (rows * seqlen * int(text["hidden_size"])
                    * int(text.get("hc_mult") or 1) * _CONFIG_DTYPE_BYTES[dtype])
    # The capture binds n_batches=len(row_offsets) from the same partition
    # (joint_cost_stage_a.py bind_produced_output) and derives its need from
    # it, so the seal and the bind's need are one number.
    return two_plane_window_bytes(
        n_probes=n_probes, n_batches=len(row_offsets), group_size=group_size,
        max_entry_tensor_bytes=tensor_bytes)


def _plan_output_root(campaign: Mapping) -> Path:
    """The plan's sealed output_root: the only root the stage-A capture will
    write into (its identity guard refuses any other --output-root), and the
    root whose ``layer-quanta/adjoint/adjoint-capture.json`` is the receipt
    this dispatcher validates."""
    plan = json.loads(Path(campaign["plan_path"]).read_text())
    return Path(plan["output_root"])


def require_staged_wait_below_grace(spec: Mapping,
                                    progress: Sequence[tuple[str, int]]) -> None:
    """Refuse a row whose fallback staged-range wait could outlast a phase grace.

    Where PrismaBuild publishes a landing record for the reader's ranges
    (PB #989), the landing record bounds the wait, not this setting: the
    reader waits while the range's mover is queued or copying, refuses at
    once on a mover that ends without a receipt, and declares the wait so
    PrismaBuild's ``no_progress`` rung does not count it. Only where no
    landing record covers the range (an older PrismaBuild generation) does
    the reader wait up to ``PRISMAQUANT_STAGED_RANGE_WAIT_S`` and then
    refuse with the staging error. PrismaBuild kills a phase that commits no
    progress for its grace, so that fallback wait at or above the smallest
    grace turns a staging stall into a no-progress kill, which names no
    range (R10: 900 s wait against a 900 s chain grace). The wait is read
    from the spec's ``env`` block with the reader's own rules, so the value
    compared here is the value the reader will use; the admissible bound is
    derived from the grace this row declares.
    """
    from prismaquant.residency_shard_reader import (
        STAGED_RANGE_WAIT_ENV, staged_range_wait_from_env)

    env = spec.get("env") or {}
    try:
        wait = staged_range_wait_from_env(env)
    except ValueError as exc:
        raise DispatchRefused(f"campaign spec: {exc}") from exc
    name, grace = min(progress, key=lambda phase: phase[1])
    if not wait < grace:
        raise DispatchRefused(
            f"campaign spec {STAGED_RANGE_WAIT_ENV}={wait:g} s is not below "
            f"the {grace} s progress grace of phase {name!r}. This is the "
            "reader's fallback wait, used where no PrismaBuild landing record "
            "covers a range (the landing record bounds every other wait); at "
            "or above the grace a staging stall would end as a no-progress "
            f"kill instead of the reader's staging refusal. Set it below {grace} s")


def _require_replay_regime(spec: dict, *, emits_handoff: bool = False) -> None:
    """Validate the Stage B replay regime the sealed spec declares (#994).

    One spec wraps every quantum of a dispatch, so the regime is uniform by
    construction. It changes the statistics arithmetic, and it replays only
    from the spill, so the spec must declare the spill beside it. A
    band-serial producer (#996) runs only a batch-1 capture.
    """
    from prismaquant.joint_replay_regime import (
        handoff_regime_refusal, replay_regime_from_environment)

    env = spec.get("env", {})
    regime = replay_regime_from_environment(env)
    if regime is not None and not stage_b_spill_environment(spec, env):
        raise RuntimeError("a non-default Stage B replay regime replays from the "
                           "spill; declare the spill in the same spec")
    if emits_handoff and handoff_regime_refusal(regime):
        raise RuntimeError(handoff_regime_refusal(regime))


def produced_spool_row_environment(spec: Mapping) -> dict:
    """The produced output spool a row seals, from its sealed spec.

    The spool's root and byte bound, and each opt-in the spec declares, go
    into the pbrun request as ``--env``. PrismaBuild reads them from the
    producer's sealed environment, and the campaign container refuses to
    launch a spec that declares the spool without them
    (``tessera_campaign_container.produced_spool_environment``). The
    container check runs here too, so a spec whose root has no writable
    identity bind refuses before anything is published. The root must be
    local to the executing box: a root under ``/mnt/shared`` would write
    every entry into the pool the spool exists to keep writes out of
    (PQ #1012). Returns ``{}`` for a spec that declares no spool.
    """

    declared = spec.get("env", {})
    try:
        forwarded = produced_spool_environment(spec, declared)
    except RuntimeError as exc:
        raise DispatchRefused(str(exc)) from exc
    if not forwarded:
        stray = [name for name in PRODUCED_SPOOL_OPT_IN_ENV if name in declared]
        if stray:
            raise DispatchRefused(
                f"spec env declares {', '.join(stray)} without "
                f"{PRODUCED_SPOOL_ROOT_ENV}")
        return {}
    root = PurePosixPath(forwarded[PRODUCED_SPOOL_ROOT_ENV])
    if PurePosixPath("/mnt/shared") in root.parents:
        raise DispatchRefused(
            f"produced spool root {root} is under /mnt/shared; it must be a "
            "directory on the executing box's own disk")
    for name in PRODUCED_SPOOL_OPT_IN_ENV:
        if name in declared:
            if declared[name] not in ("0", "1"):
                raise DispatchRefused(
                    f"spec env {name} must be \"0\" or \"1\", not {declared[name]!r}")
            forwarded[name] = declared[name]
    return forwarded


class OverlayCacheWarning(UserWarning):
    """A spec points a container cache at the overlay or /tmp (PQ #1072)."""


def _warn_overlay_caches(spec: dict, scratch: dict) -> None:
    """Warn, but do not refuse, when a spec pins a cache to the overlay.

    When the row declares bounded local scratch, the launcher binds every
    cache the spec leaves unset under the scratch root
    (``container_cache_environment``). A value set in the spec wins over
    that default. A value on ``/tmp``, ``/var/tmp`` or an unmounted path
    writes to the container overlay, which is unbounded and invisible to
    PrismaBuild. The warning fires with or without declared scratch.
    """
    _defaults, pinned = container_cache_environment(spec, scratch)
    if pinned:
        env = spec.get("env", {})
        named = ", ".join(f"{name}={env[name]}" for name in pinned)
        hint = ("unset them to bind them under the declared scratch root"
                if scratch else
                "declare a bounded local scratch root to bind them there")
        warnings.warn(
            f"the campaign spec pins container caches to the overlay: {named}; "
            f"these writes are unbounded and invisible to PrismaBuild; {hint}",
            OverlayCacheWarning, stacklevel=3)


def _container_wrap(spec_path: Path, payload: list[str], *,
                    progress: Sequence[tuple[str, int]],
                    resource_policy=None,
                    spool_max_bytes: int | None = None,
                    spill_bound: Mapping | None = None) -> tuple[list[str], str | None]:
    """Run a payload inside the qualified campaign container.

    The projection backend's runtime identity check (and the workload's own
    torch/CUDA requirement) qualify one image; a bare ``python3 -m ...``
    executes unidentified and refuses.  The single-run path wraps every
    command in ``tools.tessera_campaign_container`` with the campaign spec;
    the distributed rows are the same workload and take the same wrapper.

    Returns the wrapped payload and the image reference PrismaBuild must
    admit the row against, derived from ONE parse of the spec -- the same
    parse whose bytes are serialized into ``--spec``.  A second read could
    race a spec rewrite and seal one image while declaring another; the
    caller adds the reference to the pbrun envelope (``--container-image``
    before the payload separator), never inside the payload.

    ``progress`` is the row's ``--progress-phase`` list; the same parse is
    checked against it (:func:`require_staged_wait_below_grace`).

    ``spool_max_bytes`` replaces the spec's produced-spool byte bound (the
    Stage A row's two-plane window, :func:`stage_a_spool_window_bytes`), in
    the one parse that is sealed. It replaces only a well-formed bound: a
    spec that declares a spool root with no bound, or with one that is not a
    positive decimal byte count, is left as it is, so the row's spool check
    (:func:`produced_spool_row_environment`) refuses it as before.

    With the bound it seals the host window opt-in
    (``PRISMABUILD_PRODUCED_SPOOL_HOST_WINDOW=1``, PB #910), so PrismaBuild
    charges that window to the executing box's ``spool_gb`` at placement
    and two rows cannot together overrun one box's spool disk (PQ #1120). A
    spec that declares the opt-in off refuses: the row reads its planes
    back from the spool, and an uncharged window is refused only at bind.

    ``spill_bound`` is the row's sealed Stage B spill bound
    (:func:`_sealed_spill_bound`). Its reservation replaces the spec's
    ``PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES`` in the same one parse, so the
    ceiling the quantum checks and the ``spool_gb`` PrismaBuild charges for
    the pair (PB #911) are the sealed need, not a spec literal. A spec that
    declares no spill, or launches another capture batch than the bound
    counts parts for, refuses.
    """
    spec = json.loads(Path(spec_path).read_text())
    if spill_bound is not None:
        env = spec.get("env", {})
        if not env.get(STAGE_B_SPILL_ENV[0]):
            raise DispatchRefused(
                "this row seals a Stage B spill bound, but the spec declares "
                f"no {STAGE_B_SPILL_ENV[0]}")
        from prismaquant.joint_replay_regime import (
            ReplayRegimeRefused, normalize_replay_regime, replay_regime_from_environment)
        try:
            capture_batch = normalize_replay_regime(
                replay_regime_from_environment(env))["capture_batch"]
        except ReplayRegimeRefused as exc:
            raise DispatchRefused(str(exc)) from exc
        if capture_batch != spill_bound["capture_batch"]:
            raise DispatchRefused(
                f"the spec launches capture batch {capture_batch}, but the row's "
                f"spill bound counts parts for {spill_bound['capture_batch']}; "
                "regenerate the executable readsets with the spec's regime")
        spec["env"] = {**env, STAGE_B_SPILL_ENV[1]: str(int(spill_bound["reservation_bytes"]))}
    declared = spec.get("env", {}).get(PRODUCED_SPOOL_MAX_ENV)
    if (spool_max_bytes is not None
            and PRODUCED_SPOOL_ROOT_ENV in spec.get("env", {})
            and isinstance(declared, str) and declared.isascii()
            and declared.isdigit() and int(declared) > 0):
        window = spec["env"].get(PRODUCED_SPOOL_HOST_WINDOW_ENV)
        if window == "0":
            raise DispatchRefused(
                f"spec {spec_path} declares {PRODUCED_SPOOL_HOST_WINDOW_ENV}=0, "
                "but this row reads its cotangent planes back from its spool: "
                "its window must be charged to the box at placement (PQ #1120)")
        spec["env"] = {**spec["env"],
                       PRODUCED_SPOOL_MAX_ENV: str(int(spool_max_bytes))}
        # A value other than "0" or "1" is left for the row's spool check,
        # which refuses it.
        if window in (None, "1"):
            spec["env"][PRODUCED_SPOOL_HOST_WINDOW_ENV] = "1"
    require_staged_wait_below_grace(spec, progress)
    # Validate a declared workspace before publishing the row. These same
    # inlined spec bytes supply its outer PB environment below; no ambient
    # coordinator environment or second spec read participates.
    try:
        scratch = local_scratch_environment(spec, spec.get("env", {}))
        _warn_overlay_caches(spec, scratch)
        _require_replay_regime(spec, emits_handoff="--emit-adjoint-handoff" in payload)
        # The bf16 reduction flag is sealed in the same spec, so it is
        # uniform across every quantum of the dispatch (PQ #1028).
        from prismaquant.matmul_arithmetic import bf16_reduction_from_environment
        bf16_reduction_from_environment(spec.get("env", {}))
    except (ValueError, RuntimeError) as exc:
        raise DispatchRefused(str(exc)) from exc
    if resource_policy is not None:
        limits = resource_policy["limits"]
        if (float(spec.get("cpu_memory_gb", -1)) * 1024 ** 3 != limits["host_bytes"]
                or float(spec.get("env", {}).get("PRISMAQUANT_MAX_GPU_MEM_GB", -1)) * 1024 ** 3 != limits["gpu_bytes"]):
            raise DispatchRefused("container host/device envelope differs from the bound Stage B resource policy")
    argv = ["python3", "-m", "tools.tessera_campaign_container",
            "--spec", json.dumps(spec, sort_keys=True),
            "--", *payload]
    default_admission = admission_image_reference(spec)
    admission = spec.get("container_admission_reference")
    if admission is not None:
        if (not isinstance(admission, str) or not admission.startswith("content:sha256:")
                or not _is_hex64(admission.removeprefix("content:sha256:"))
                or not _is_hex64(spec.get("container", {}).get("content_sha256"))):
            raise DispatchRefused("explicit portable image admission requires content SHA and inspected scientific image identity")
    return argv, admission or default_admission

def _sealed_spill_bound(record: Mapping) -> Mapping | None:
    """The row's validated spill bound, or ``None`` for a row that does not spill.

    An executable row sealed for the spill replay must seal its bound: the
    row's spill ceiling is set from it, never from the spec. A bound on any
    other row refuses. Rows without an executable readset keep the spec's
    ceiling (the legacy slice path).
    """
    quantum_id = record.get("quantum_id")
    block = record.get("executable_readset")
    if not isinstance(block, Mapping):
        return None
    bound = block.get("spill_bound")
    if block.get("replay_mode") != "spill":
        if bound is not None:
            raise DispatchRefused(
                f"quantum {quantum_id!r} seals a spill bound on a readset that is "
                "not sealed for the spill replay")
        return None
    if bound is None:
        raise DispatchRefused(
            f"quantum {quantum_id!r} is sealed for the spill replay but seals no "
            "spill bound; regenerate its executable readset")
    from prismaquant.joint_replay_spill import SpillBoundRefused, check_spill_bound
    try:
        check_spill_bound(bound)
    except SpillBoundRefused as exc:
        raise DispatchRefused(f"quantum {quantum_id!r}: {exc}") from exc
    return bound


def quantum_argv(record: dict, *, record_path: Path, output_root: Path,
                 priority: int = SUBMISSION_PRIORITY,
                 head_grace_s: int = HEAD_PROGRESS_GRACE_S,
                 consumer_tags: Sequence[str] = CONSUMER_TAGS,
                 band: Mapping | None = None) -> list[str]:
    """The exact §5.2 submission argv for one quantum. Pinned by tests: a
    drift here breaks placement.  ``consumer_tags`` is the effective §5.1
    placement policy, a conjunction PB matches against a worker's offered
    tags; PB alone decides which matching box claims the row.

    Every binding the consumer CLI requires is threaded from sealed sources:
    plan/prepared paths+digests from the record's campaign block, the record
    file's own wire digest as ``--quantum-sha256`` (the consumer checks raw
    bytes first; the canonical body check inside stays), the record's
    stage-A slice file and its digest as ``--adjoint-slice`` /
    ``--adjoint-slice-sha256`` (PQ #993: a quantum reads its slice, never a
    whole receipt), the verified staged-manifest
    digest, and ``--resume``. Files are read where the row reads them; an
    unreadable or drifting file refuses before anything publishes (#838).
    A record carrying ``executable_readset`` without the sealed
    static prepared-input contract (PQ #917) is refused with
    :class:`ExecutableBindingUnsupported`: no PB produced-output binding
    validator is accepted yet, so those executable rows are
    sequencing/phase artifacts only, never production-runnable. A row
    whose bound manifest carries the complete prepared-input contract --
    the sealed prepared/production-pickle digest and roster, one render
    phase per retained window sealed before its replays, and a bound
    block that agrees with the manifest -- submits through the existing
    ``pbrun --data-manifest`` path with manifest-order progress. Without
    the block the row keeps the legacy slice manifest with head/chunk
    progress. Tier flags, tags, demand and environment are identical in
    all lanes.

    ``band`` (PQ #996) is this row's band-serial role, from
    :func:`fresh_band_role` or :func:`recorded_band_role`. Its
    ``handoff`` makes the row a consumer: the payload gains
    ``--adjoint-handoff`` and its digest, and the row stages the derived
    band-serial readset, with its phases as the progress declaration,
    instead of the sealed chain manifest. Its ``emit_template`` makes the
    row a producer: the payload gains ``--emit-adjoint-handoff`` and the
    envelope declares the handoff's produced-output template. Without a
    band the argv is the chain-mode one, byte for byte.
    """
    quantum_id = record["quantum_id"]
    handoff = (band or {}).get("handoff")
    emit_template = (band or {}).get("emit_template")
    if (handoff is not None or emit_template is not None) and \
            record.get("executable_readset") is None:
        raise DispatchRefused(
            f"quantum {quantum_id!r} is not an executable row: band-serial "
            "quanta stage their reads through an executable readset")
    if emit_template is not None and not _pbrun_seals_produced_output():
        raise ProducedOutputDeclarationUnsupported(
            f"quantum {quantum_id!r} must declare its handoff template "
            f"({emit_template}), but the client at {PBRUN} carries no "
            "--produced-output-template")
    resource_policy = None
    executable = record.get("executable_readset")
    if record.get("catalog_extension") is not None and executable is None:
        raise DispatchRefused("catalog extension requires executable prepared-input readsets before publication")
    if executable is not None:
        # PQ #917: the complete static prepared-input contract is an
        # ordinary immutable input staging -- the manifest names existing
        # prepared bytes, and PWC's strict resolver/lease read stays the
        # single read mechanism. Anything less (sequencing-only rows, a
        # foreign or malformed prepared contract, a bare mover reference)
        # refuses below with the typed refusal; manifest/phase propagation
        # stays exercised through _executable_row_parts directly.
        _executable_prepared_input(record, output_root=output_root)
        manifest, staged_sha256, progress = _executable_row_parts(
            record, output_root=output_root, head_grace_s=head_grace_s)
        if handoff is not None:
            # The sealed chain manifest passed its gate above; the row stages
            # the readset derived from it and this handoff.
            manifest = Path(handoff["manifest_path"])
            staged_sha256 = handoff["manifest_sha256"]
            progress = [("head", head_grace_s)] + [
                (name, HEAD_PROGRESS_GRACE_S if name == HANDOFF_LOAD_PHASE
                 else CHUNK_PROGRESS_GRACE_S)
                for name in handoff["phases"] if name != "head"]
    else:
        manifest = Path(record["read_set"]["manifest_path"])
        if not manifest.is_absolute():
            manifest = output_root / manifest
        staged_sha256 = _slice_manifest_digest(record, output_root=output_root)
        progress = [("head", head_grace_s)]
        for chunk in record.get("chunks", []):
            progress.append((chunk["name"], CHUNK_PROGRESS_GRACE_S))
    campaign = record.get("campaign")
    if not isinstance(campaign, dict):
        raise DispatchRefused(
            f"quantum {quantum_id!r} carries no campaign block")
    for key in ("plan_path", "plan_sha256", "prepared_path", "prepared_sha256"):
        value = campaign.get(key)
        if not isinstance(value, str) or not value:
            raise DispatchRefused(
                f"quantum {quantum_id!r} seals no campaign {key}")
    try:
        plan_raw = Path(campaign["plan_path"]).read_bytes()
        plan = json.loads(plan_raw)
    except (OSError, ValueError) as exc:
        raise DispatchRefused(f"quantum {quantum_id!r} source plan is unreadable: {exc}") from exc
    if plan.get("stage_b_resource_policy") is not None:
        if hashlib.sha256(plan_raw).hexdigest() != campaign["plan_sha256"]:
            raise DispatchRefused("resource-bound plan differs from its quantum seal")
        from prismaquant.joint_stageb_resources import verify_policy
        try:
            resource_policy = verify_policy(plan["stage_b_resource_policy"])
        except (ValueError, OSError) as exc:
            raise DispatchRefused(f"invalid Stage B resource policy: {exc}") from exc
    try:
        record_sha256 = _sha_bytes(Path(record_path).read_bytes())
    except OSError as exc:
        raise DispatchRefused(
            f"quantum {quantum_id!r} record unreadable at {record_path}: "
            f"{exc}") from exc
    adjoint = record.get("adjoint", {})
    slice_path, slice_sha256 = adjoint.get("slice_path"), adjoint.get("slice_sha256")
    if not isinstance(slice_path, str) or not _is_hex64(slice_sha256):
        raise DispatchRefused(
            f"quantum {quantum_id!r} is unbound (pre-A): regenerate it against "
            "its stage-A slice before publishing")
    try:
        if _sha_bytes(Path(slice_path).read_bytes()) != slice_sha256:
            raise DispatchRefused(
                f"quantum {quantum_id!r} stage-A slice at {slice_path} does not "
                "hash to the sealed digest")
    except OSError as exc:
        raise DispatchRefused(
            f"quantum {quantum_id!r} stage-A slice unreadable at "
            f"{slice_path}: {exc}") from exc
    band_payload: list[str] = []
    if handoff is not None:
        band_payload += ["--adjoint-handoff", str(handoff["path"]),
                         "--adjoint-handoff-sha256", str(handoff["sha256"])]
    if emit_template is not None:
        band_payload.append("--emit-adjoint-handoff")
    wrapped, container_image = _container_wrap(SPEC_PATH, [
        "python3", "-m", "prismaquant.joint_cost_quantum",
        "--quantum", str(record_path),
        "--quantum-sha256", record_sha256,
        "--plan", str(campaign["plan_path"]),
        "--plan-sha256", str(campaign["plan_sha256"]),
        "--prepared", str(campaign["prepared_path"]),
        "--prepared-sha256", str(campaign["prepared_sha256"]),
        "--adjoint-slice", str(slice_path),
        "--adjoint-slice-sha256", slice_sha256,
        "--data-manifest-sha256", staged_sha256,
        "--allowed-tiers", STAGED_ALLOWED_TIERS,
        "--resume",
        "--output-root", str(output_root), *band_payload], progress=progress,
        resource_policy=resource_policy, spill_bound=_sealed_spill_bound(record))
    argv = [sys.executable, str(PBRUN)]
    for tag in consumer_tags:
        argv += ["--tag", str(tag)]
    argv += ["--data-manifest", str(manifest),
             "--residency", "stage", "--residency-ram", "auto"]
    for name, grace in progress:
        argv += ["--progress-phase", f"{name}={grace}"]
    sealed_spec = json.loads(wrapped[wrapped.index("--spec") + 1])
    mem_gib, gpu_gib, cpus = "104", "80", 10
    if resource_policy is not None:
        limits = resource_policy["limits"]
        mem_gib, gpu_gib = (f"{limits[key] / 1024 ** 3:g}" for key in ("physical_bytes", "gpu_bytes"))
        cpus = max(int(sealed_spec.get("env", {}).get("PRISMAQUANT_LAYER_READ_THREADS", 1)),
                   int(plan["source_prefetch"]["prefetch_workers"]) + 1,
                   int(plan["execution"]["operator_windows"]["prefetch_workers"]) + 1)
    argv += ["--priority", str(priority),
             "--demand", f"gpu=1,mem_gb={mem_gib}", "--gpu-memory-gb", gpu_gib,
             "--cpus", str(cpus)]
    if emit_template is not None:
        # An envelope option, as for stage A: pbrun seals the declaration
        # into the request and derives the window's tier demand from it.
        argv += ["--produced-output-template", str(emit_template)]
    if container_image is not None:
        # A pbrun option, so it precedes the separator like the manifest: PB
        # must admit the row only where this image is already present, or
        # leave it ready for a box that has it (RobTand/prismabuild#714).
        argv += [CONTAINER_IMAGE_FLAG, container_image]
    # Every bounded-local scratch pair and PrismaBuild's list of them (PB
    # #911), so pbrun charges each ceiling to the executing box's disk
    # budget at claim. Nothing declared, nothing added.
    for name, value in local_scratch_environment(
            sealed_spec, sealed_spec.get("env", {})).items():
        argv += ["--env", f"{name}={value}"]
    # A spec that declares the produced spool seals it into the request too:
    # the container refuses a declared spool the action does not carry, and
    # a band-serial producer writes its handoff group through it (#1015).
    for name, value in produced_spool_row_environment(sealed_spec).items():
        argv += ["--env", f"{name}={value}"]
    argv += ["--env", DEV_MODE_ENV, "--detach", "--", *wrapped]
    return argv


def stage_a_argv(adjoint_manifest: Path, campaign: Mapping,
                 *, tag: str = ADJOINT_TAG,
                 prefetch_override: Path | None = None,
                 artifact_budget_bytes: int | str | None = None,
                 produced_output_template: Path | None = None,
                 binding: dict | None = None) -> list[str]:
    """The §5.2 stage-A submission argv: the adjoint capture goes first and
    alone; quanta wait on its receipt.  The campaign binding every record
    carries names the plan and prepared inputs (with digests) the capture's
    own CLI requires -- the records are the single source of those paths.

    ``prefetch_override`` (optional, #819): names the explicit prefetch-
    override document the capture reads through its own ``--prefetch-override``
    flag -- the IO-side #809 seam for a frozen plan whose sealed budget
    starves the capture.  The payload flag is the channel that crosses the
    container boundary (``tessera_campaign_container`` forwards no ambient
    action environment into the payload), so the dispatcher threads the
    flag, not ``--env``.  Absent: the plan's sealed budget, argv unchanged.

    ``artifact_budget_bytes`` (optional, #882): the explicit durable
    artifact ceiling in bytes the capture reads through its own
    ``--artifact-budget-bytes`` flag -- the durable-side #809 seam for a
    frozen plan whose sealed ``max_artifact_bytes`` cannot hold the retained
    peak. Strict positive-integer bytes; the payload flag is the channel
    that crosses the container boundary, so the dispatcher threads the
    flag, not ``--env``. Absent: the plan's sealed ceiling, argv unchanged.

    The manifest binding is derived the same way: ``--data-manifest-sha256``
    and ``--read-manifest-sha256`` ride the payload (a run that bound no
    manifest digest gets no tier redirect), and the manifest's read phases
    become the row's ``--progress-phase`` declarations in manifest order --
    the worker refuses undeclared names, so the list is derived, never
    hardcoded (#835).

    ``produced_output_template`` (optional): the pre-submit produced-output
    declaration that lets the capture read its OWN boundary entries back
    through PrismaBuild. It rides as a pbrun ENVELOPE option, so the client
    ingests it as a declared input, seals the matching declaration into the
    request params and carries the template onto the queue row -- and
    derives the bounded window's tier demand from the template rather than
    from anything restated here. A client without that flag refuses
    (:class:`ProducedOutputDeclarationUnsupported`) rather than submitting
    a capture that could write its entries and not read them.

    ``binding`` (optional) is a precomputed :func:`_stage_manifest_binding`
    for this manifest and campaign, so a caller that also records the
    digests does not read the manifest twice.

    The spec must declare the produced output spool (PQ #1012): its root on
    the executing box's own disk and its byte bound, sealed into the request
    with the opt-ins the spec declares (:func:`produced_spool_row_environment`).
    A spec without it refuses, because the owner would otherwise write every
    boundary entry synchronously into the pool. The row also seals the
    storage box's RAM tier (``--residency-ram auto``), as the quantum row
    does.
    """
    if produced_output_template is not None:
        if not Path(produced_output_template).is_file():
            raise DispatchRefused(
                "stage-A produced-output template is not a file: "
                f"{produced_output_template}")
        if not _pbrun_seals_produced_output():
            raise ProducedOutputDeclarationUnsupported(
                "stage-A submission names a produced-output template "
                f"({produced_output_template}), but the client at {PBRUN} "
                "carries no --produced-output-template: the owner's request "
                "must seal the declaration against an ingested "
                "prismabuild.produced-output-template input and the queue "
                "row must carry the template itself. Refusing rather than "
                "submitting a capture that would write its boundary entries "
                "and then be unable to read them")
    if binding is None:
        binding = _stage_manifest_binding(adjoint_manifest, campaign)
    if artifact_budget_bytes is not None:
        if isinstance(artifact_budget_bytes, bool):
            raise DispatchRefused(
                "stage-A artifact budget must be a positive integer byte "
                f"count (bytes), got bool {artifact_budget_bytes!r}")
        if isinstance(artifact_budget_bytes, int):
            _budget = artifact_budget_bytes
        elif isinstance(artifact_budget_bytes, str):
            text = artifact_budget_bytes.strip()
            if not text or any(ch not in "0123456789" for ch in text):
                raise DispatchRefused(
                    "stage-A artifact budget must be a positive integer byte "
                    "count (ASCII decimal bytes), got "
                    f"{artifact_budget_bytes!r}")
            _budget = int(text, 10)
        else:
            raise DispatchRefused(
                "stage-A artifact budget must be a positive integer byte "
                "count (bytes) as int or decimal string, not "
                f"{type(artifact_budget_bytes).__name__} "
                f"{artifact_budget_bytes!r}")
        if _budget <= 0:
            raise DispatchRefused(
                "stage-A artifact budget must be a positive integer byte "
                f"count (bytes), got {_budget!r}")
        artifact_budget_bytes = _budget
    payload = [
        "python3", "-m", "prismaquant.joint_adjoint_capture",
        "--plan", str(campaign["plan_path"]),
        "--plan-sha256", str(campaign["plan_sha256"]),
        "--prepared", str(campaign["prepared_path"]),
        "--prepared-sha256", str(campaign["prepared_sha256"]),
        "--data-manifest-sha256", binding["data_manifest_sha256"],
        "--read-manifest-sha256", binding["read_manifest_sha256"],
        "--allowed-tiers", STAGED_ALLOWED_TIERS,
        "--output-root", str(_plan_output_root(campaign)),
        "--resume"]
    if prefetch_override is not None:
        payload += ["--prefetch-override", str(prefetch_override)]
    if artifact_budget_bytes is not None:
        payload += ["--artifact-budget-bytes", str(artifact_budget_bytes)]
    progress = [(phase, HEAD_PROGRESS_GRACE_S if phase == "head"
                 else CHUNK_PROGRESS_GRACE_S) for phase in binding["phases"]]
    # The spool's window is the plan's two cotangent planes, not the spec's
    # bound (PQ #1110): the chain reads its own planes back from it. The
    # wrapper also seals the host window opt-in, so placement charges that
    # window to the box (PQ #1120).
    wrapped, container_image = _container_wrap(
        SPEC_PATH, payload, progress=progress,
        spool_max_bytes=stage_a_spool_window_bytes(campaign))
    sealed_spec = json.loads(wrapped[wrapped.index("--spec") + 1])
    spool = produced_spool_row_environment(sealed_spec)
    if not spool:
        raise DispatchRefused(
            f"stage-A spec {SPEC_PATH} declares no {PRODUCED_SPOOL_ROOT_ENV}: "
            "without the produced output spool every boundary entry is "
            "written synchronously into the pool (PQ #1012)")
    # The ram leg promotes landed stage ranges into the RAM tier the storage
    # box announces, as the Stage B row does (#640, PQ #1012).
    argv = [sys.executable, str(PBRUN),
            "--tag", tag,
            "--data-manifest", str(adjoint_manifest),
            "--residency", "stage", "--residency-ram", "auto"]
    for phase, grace in progress:
        argv += ["--progress-phase", f"{phase}={grace}"]
    argv += ["--demand", f"gpu=1,mem_gb={stage_a_memory_gib(campaign)}",
             "--gpu-memory-gb", "80", "--cpus", "10"]
    if produced_output_template is not None:
        # An ENVELOPE option, before the payload separator, like every
        # other pbrun flag. Not a payload flag: the seal has to happen on
        # the request and the queue row, which is the submitting client's
        # job, and the tier demand for the bounded window is derived by
        # pbrun from the template itself -- so nothing here adds it again.
        argv += ["--produced-output-template", str(produced_output_template)]
    if container_image is not None:
        # Before the separator, like every other pbrun option; see the
        # quantum row above and RobTand/prismabuild#714.
        argv += [CONTAINER_IMAGE_FLAG, container_image]
    for name, value in spool.items():
        argv += ["--env", f"{name}={value}"]
    argv += ["--env", DEV_MODE_ENV, "--detach", "--", *wrapped]
    return argv


def check_stage_a_proofs(proofs: Sequence[tuple[Path, dict]],
                         records: list[tuple[Path, dict]]) -> dict[str, str]:
    """The per-record slice gate (PQ #993): which quanta a sealed proof covers.

    ``proofs`` are the completed receipt and/or checkpoint bands, as
    ``(path, document)``. They must share one run header, and that header
    must answer for the records' campaign (or, for a catalog extension, the
    original capture the extension binds). Then each record is publishable
    exactly when a proof carries its checkpoint AND the slice that proof
    gives the record's layer is the slice the record binds, byte for byte,
    AND the slice file at ``adjoint.slice_path`` hashes to it. A record with
    no covering proof is pending, never refused; a covered record binding
    any other slice refuses the run. Returns ``{quantum_id: slice_sha256}``.
    """
    from prismaquant.joint_adjoint_slices import (
        AdjointSliceRefused, adjoint_slice_sha256, band_set,
        load_adjoint_slice, stage_a_receipt_kind, stage_a_run_header,
        stage_a_run_header_sha256, stage_a_slice)
    from prismaquant.joint_layer_quanta import check_adjoint_run_identity
    if not proofs:
        return {}
    campaign = records[0][1]["campaign"]
    extensions = [record.get("catalog_extension") for _, record in records]
    if any(extension != extensions[0] for extension in extensions):
        raise DispatchRefused("quantum catalog extension bindings differ")
    if any(record.get("adjoint", {}).get("slice_sha256") is None for _, record in records):
        raise DispatchRefused(
            "the layer records are unbound (pre-A): regenerate them against the "
            "stage-A proof before publishing quanta")
    try:
        complete = None
        bands = []
        for path, document in proofs:
            if stage_a_receipt_kind(document) == "complete":
                if complete is not None:
                    raise DispatchRefused("two completed stage-A receipts named")
                complete = document
            else:
                bands.append(document)
        indexed = band_set(bands) if bands else {}
        headers = {stage_a_run_header_sha256(stage_a_run_header(document))
                   for _, document in proofs}
        if len(headers) != 1:
            raise DispatchRefused("stage-A proofs carry different run headers: mixed runs")
        header = stage_a_run_header(proofs[0][1])
        try:
            # The stride is not checked here: the dispatcher derives none.
            # load_adjoint_slice below checks that the header's stride
            # places each record at its checkpoint.
            check_adjoint_run_identity(
                header, plan_sha256=campaign["plan_sha256"],
                prepared_sha256=campaign["prepared_sha256"],
                scope=campaign["campaign_scope"],
                catalog_extension=extensions[0])
        except (ValueError, OSError, KeyError) as exc:
            raise DispatchRefused(f"stage-A proof does not answer for this campaign: {exc}") from exc
        publishable: dict[str, str] = {}
        for path, record in records:
            adjoint = record["adjoint"]
            layer, boundary = int(record["layer"]), int(adjoint["checkpoint_boundary"])
            proof = complete if complete is not None else indexed.get(boundary)
            if proof is None:
                continue
            expected = adjoint_slice_sha256(stage_a_slice(proof, layer))
            if adjoint["slice_sha256"] != expected:
                raise DispatchRefused(
                    f"{path}: record binds another stage-A slice than the sealed "
                    f"proof gives layer {layer}")
            load_adjoint_slice(adjoint["slice_path"], expected, layer=layer,
                               checkpoint_boundary=boundary)
            publishable[record["quantum_id"]] = expected
    except AdjointSliceRefused as exc:
        raise DispatchRefused(f"stage-A slice gate refused: {exc}") from exc
    except OSError as exc:
        raise DispatchRefused(f"stage-A slice unreadable: {exc}") from exc
    return publishable


def load_stage_a_proofs(receipt_path: Path | None,
                        band_paths: Sequence[Path], *,
                        receipt_named: bool = True) -> list[tuple[Path, dict]]:
    """Read the stage-A proofs.

    A named receipt or band that is missing, or is not a completed receipt
    or sealed band, refuses. The implicit default receipt path
    (``receipt_named=False``) is only a place a receipt may have landed: a
    missing or still-running document there is no proof yet, never a
    refusal, so bands publish while stage A runs.
    """
    from prismaquant.joint_adjoint_slices import load_stage_a_receipt_like
    proofs = []
    if receipt_path is not None:
        try:
            proofs.append((Path(receipt_path), load_stage_a_receipt_like(receipt_path)))
        except (OSError, ValueError, RuntimeError) as exc:
            if receipt_named:
                raise DispatchRefused(
                    f"{receipt_path}: not a completed stage-A receipt: {exc}") from exc
    for path in band_paths:
        try:
            proofs.append((Path(path), load_stage_a_receipt_like(path)))
        except (OSError, ValueError, RuntimeError) as exc:
            raise DispatchRefused(f"{path}: not a sealed stage-A band: {exc}") from exc
    return proofs


# --------------------------------------------------------------------------
# Band-serial Stage B (PQ #996)
# --------------------------------------------------------------------------
#
# Inside a checkpoint band, quantum L-1 can take quantum L's final input
# cotangent (its handoff) instead of rebuilding the chain from the band's
# checkpoint. PrismaBuild has no dependency between actions, so the edge is
# expressed here, as publication order: L carries a declared produced output
# (its handoff), and L-1 is published only once L has executed, reported
# complete and published a handoff that binds L-1. L-1 then declares the
# handoff's bytes as ordinary staged inputs of its own data manifest. Nothing
# here places, schedules or moves bytes; bands stay independent.

#: The dispatcher's own control files for band-serial rows: the per-producer
#: produced-output templates and the per-consumer derived readsets.
BAND_SERIAL_DIRECTORY = "band-serial"


def _band_serial_root(output_root: Path) -> Path:
    return Path(output_root) / "layer-quanta" / BAND_SERIAL_DIRECTORY


def _publish_control_bytes(path: Path, payload: bytes, *, what: str) -> Path:
    """Publish a derived control file once; other bytes at its name refuse."""
    from prismaquant.cost_stage_checkpoint import publish_new_bytes

    if not publish_new_bytes(Path(path), payload):
        try:
            existing = Path(path).read_bytes()
        except OSError as exc:
            raise DispatchRefused(f"{what} at {path} is unreadable: {exc}") from exc
        if existing != payload:
            raise DispatchRefused(
                f"{what} at {path} holds other bytes: refusing to replace it")
    return Path(path)


def _read_bound_slice(record: Mapping) -> dict:
    """The record's Stage A slice, read where the row reads it and hashed."""
    adjoint = record.get("adjoint", {})
    path, digest = adjoint.get("slice_path"), adjoint.get("slice_sha256")
    try:
        raw = Path(path).read_bytes()
    except (OSError, TypeError) as exc:
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} stage-A slice unreadable at "
            f"{path}: {exc}") from exc
    if _sha_bytes(raw) != digest:
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} stage-A slice at {path} does "
            "not hash to the sealed digest")
    return json.loads(raw)


def band_serial_roles(records: Sequence[tuple[Path, dict]]) -> dict[str, dict]:
    """Which quantum hands its cotangent to which, per checkpoint band.

    ``hands_to`` names quantum ``L - 1`` when it shares ``L``'s checkpoint
    boundary; ``takes_from`` is the inverse. A band's top (the layer just
    below its checkpoint) takes from nobody: its incoming cotangent is the
    checkpoint itself.
    """
    by_layer = {record["layer"]: record for _, record in records}

    def boundary(record):
        return record.get("adjoint", {}).get("checkpoint_boundary")

    roles = {}
    for _, record in records:
        below = by_layer.get(record["layer"] - 1)
        above = by_layer.get(record["layer"] + 1)
        roles[record["quantum_id"]] = {
            "hands_to": (below["quantum_id"] if below is not None
                         and boundary(below) == boundary(record) else None),
            "takes_from": (above["quantum_id"] if above is not None
                           and boundary(above) == boundary(record) else None),
        }
    return roles


#: The prefix of a Stage B handoff template's derived id.
HANDOFF_TEMPLATE_ID_PREFIX = "pq-stageb-handoff-"


def handoff_template_id(quantum_id: str, template: Mapping) -> str:
    """``pq-stageb-handoff-<quantum id>-<digest>``: a handoff template's id.

    The digest covers every field of the template except its id, so two
    bodies never share an id and one body always gets the same one. The
    body names the handoff directory inside the quantum's own output space,
    so the same quantum id under another root gets another id (PQ #1054).
    """
    body = {key: value for key, value in template.items() if key != "template_id"}
    digest = _sha_bytes(json.dumps(body, sort_keys=True,
                                   separators=(",", ":")).encode())
    return f"{HANDOFF_TEMPLATE_ID_PREFIX}{quantum_id}-{digest[:16]}"


def handoff_template_path(record: Mapping, *, plan: Mapping,
                          adjoint_slice: Mapping, tier: str,
                          output_root: Path,
                          template_id: str | None = None) -> Path:
    """Write and return a producer row's handoff produced-output template.

    Derived from the numbers the producer's emitter binds with: the plan's
    ``execution.boundary_storage`` normalized onto the producer's handoff
    directory (artifact maximum and prefetch group), and the largest
    checkpoint-plane tensor in the producer's slice. ``tier`` is the stage
    tier the declaration permits, a fleet fact the submitter names. The file
    is named by its content digest, so a changed tier or plan writes a new
    template and never rewrites one a submitted row already declared.

    The template is write-only (PrismaBuild #912, PQ #1075): the producer
    never reads its handoff back, so it reserves no stage window and commits
    each group at its origin for the consumer to stage as an input.

    ``template_id`` defaults to :func:`handoff_template_id`, derived from the
    template's own body. PrismaBuild files a template under its id and
    refuses a different template under the same id, so the id changes
    exactly when the body does: the same quantum under another root (another
    ``output_prefix``), tier or plan files a template of its own, and a
    re-dispatch of the same row files the same one again (PQ #1054). An
    explicit ``template_id`` overrides the derived one.
    """
    from prismaquant.cost_streaming import normalize_boundary_storage
    from prismaquant.joint_quantum_handoff import handoff_root
    from prismaquant.stage_a_produced_output import build_boundary_template

    quantum_id = record["quantum_id"]
    storage = plan.get("execution", {}).get("boundary_storage")
    if not isinstance(storage, dict):
        raise DispatchRefused(
            f"quantum {quantum_id!r}: the plan seals no boundary storage for a "
            "handoff")
    try:
        policy = normalize_boundary_storage({
            **storage,
            "directory": str(handoff_root(record["output_space"]["root"]))})
        tensors = [int(entry["tensor_bytes"]) for entry in
                   adjoint_slice["checkpoint"]["activation_entries"]]

        def build(name: str) -> dict:
            return build_boundary_template(
                output_prefix=policy["directory"], tier=str(tier),
                artifact_max_bytes=int(policy["max_artifact_bytes"]),
                group_size=int(policy["prefetch_batches"]),
                max_entry_tensor_bytes=max(tensors), template_id=name,
                write_only=True)

        template = build(HANDOFF_TEMPLATE_ID_PREFIX + str(quantum_id)
                         if template_id is None else template_id)
        if template_id is None:
            template = build(handoff_template_id(quantum_id, template))
    except (KeyError, TypeError, ValueError) as exc:
        raise DispatchRefused(
            f"quantum {quantum_id!r}: no handoff template derives from the "
            f"plan and slice: {exc}") from exc
    payload = (json.dumps(template, sort_keys=True, indent=2) + "\n").encode()
    path = (_band_serial_root(output_root)
            / f"{quantum_id}.handoff-template.{_sha_bytes(payload)[:16]}.json")
    return _publish_control_bytes(path, payload, what="handoff template")


def bind_consumer_handoff(record: Mapping, *, path: str, sha256: str,
                          producer: str, output_root: Path) -> dict:
    """Bind a published handoff to its consumer row, or refuse.

    Runs the consumer's own checks (:func:`load_quantum_handoff`) and
    derives, then writes, the band-serial readset the row stages. A handoff
    the consumer would refuse is refused here: publishing it would only exit
    3 on a GPU box.
    """
    from prismaquant.joint_quantum_handoff import (
        QuantumHandoffRefused, band_serial_manifest_bytes, load_quantum_handoff)

    quantum_id = record["quantum_id"]
    adjoint_slice = _read_bound_slice(record)
    try:
        handoff = load_quantum_handoff(path, sha256, record=record,
                                       adjoint_slice=adjoint_slice)
        wire = band_serial_manifest_bytes(record, handoff,
                                          adjoint_slice["checkpoint"],
                                          output_root=output_root)
    except QuantumHandoffRefused as exc:
        raise DispatchRefused(
            f"quantum {quantum_id!r} refuses the handoff {producer!r} "
            f"published: {exc}") from exc
    manifest_path = _publish_control_bytes(
        _band_serial_root(output_root)
        / f"{quantum_id}.{handoff['handoff_sha256'][:16]}.executable.json.gz",
        wire, what="band-serial readset")
    phases = [phase["name"] for phase in
              json.loads(gzip.decompress(wire))["read_plan"]["phases"]]
    return {"path": str(path), "sha256": str(sha256), "producer": str(producer),
            "handoff_sha256": handoff["handoff_sha256"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha_bytes(wire), "phases": phases}


def _producer_handoff(producer: Mapping, *, key: str | None,
                      gateway: "Gateway") -> tuple[dict | None, str]:
    """What an executed producer published: ``(handoff, "")``, ``(None, "chain")``
    or ``(None, reason)`` while its consumer must wait."""
    quantum_id = producer["quantum_id"]
    if key is None or not gateway.is_terminal_executed(key):
        return None, f"waits on {quantum_id}, which has not executed"
    space = producer["output_space"]
    status_path = Path(space.get("status", str(Path(space["root"]) / "status.json")))
    try:
        status = json.loads(status_path.read_bytes())
        results = json.loads(Path(space["results"]).read_bytes())
    except (OSError, ValueError) as exc:
        return None, f"waits on {quantum_id}, whose outputs are unreadable: {exc}"
    if not isinstance(status, dict) or status.get("status") != "complete" or \
            status.get("identity_sha256") != producer["identity_sha256"]:
        return None, f"waits on {quantum_id}, which has not reported complete"
    published = results.get("handoff") if isinstance(results, dict) else None
    if published is None:
        return None, "chain"
    try:
        digest = _sha_bytes(Path(published["path"]).read_bytes())
    except (OSError, KeyError, TypeError) as exc:
        return None, f"waits on {quantum_id}, whose handoff is unreadable: {exc}"
    if digest != published.get("sha256"):
        return None, (f"waits on {quantum_id}, whose handoff does not hash to "
                      "the digest its results name")
    return dict(published), ""


def fresh_band_role(record: Mapping, *, roles: Mapping, by_id: Mapping,
                    last_submission: Mapping, submitted_keys: Mapping,
                    gateway: "Gateway", tier: str | None,
                    output_root: Path) -> tuple[dict | None, str | None]:
    """The band-serial role of a row never submitted before.

    Returns ``(band, None)`` to publish or ``(None, reason)`` while the row
    waits for its producer. A consumer takes its producer's handoff; a
    producer emits one only for a successor never yet submitted, since a
    submitted row keeps the mode it was submitted in.
    """
    from prismaquant.joint_quantum_handoff import handoff_chain_regime_refusal

    quantum_id = record["quantum_id"]
    role = roles[quantum_id]
    adjoint_slice = _read_bound_slice(record)
    refusal = handoff_chain_regime_refusal(adjoint_slice.get("run_identity"))
    if refusal is not None:
        raise DispatchRefused(f"quantum {quantum_id!r} cannot run band-serial: {refusal}")
    band: dict = {}
    source = role["takes_from"]
    if source is not None:
        published, reason = _producer_handoff(
            by_id[source], key=submitted_keys.get(source), gateway=gateway)
        if published is None and reason != "chain":
            return None, reason
        if published is not None:
            band["handoff"] = bind_consumer_handoff(
                record, path=published["path"], sha256=published["sha256"],
                producer=source, output_root=output_root)
    successor = role["hands_to"]
    if successor is not None and successor not in last_submission:
        if tier is None:
            raise DispatchRefused("--band-serial needs --handoff-tier")
        plan = _load_json(Path(record["campaign"]["plan_path"]), where="plan")
        band["emit_template"] = handoff_template_path(
            record, plan=plan, adjoint_slice=adjoint_slice, tier=tier,
            output_root=output_root)
    return band, None


def recorded_band_role(record: Mapping, event: Mapping, *,
                       output_root: Path) -> dict:
    """Rebuild the role a row was submitted with, from its state event.

    A resubmission must be the same sealed action: a row republished in
    another mode would be a second action for the same output space. So a
    row keeps its recorded handoff and template whether or not this run
    passes ``--band-serial``; events written before PQ #996 are chain rows.
    """
    band: dict = {}
    template = event.get("handoff_template")
    if template is not None:
        if not Path(template).is_file():
            raise DispatchRefused(
                f"quantum {record['quantum_id']!r} was submitted with the "
                f"handoff template {template}, which is gone")
        band["emit_template"] = Path(template)
    source = event.get("cotangent_source") or {"mode": "chain"}
    if source.get("mode") == "handoff":
        band["handoff"] = bind_consumer_handoff(
            record, path=source["path"], sha256=source["sha256"],
            producer=source["producer"], output_root=output_root)
    return band


def _cotangent_source(band: Mapping | None) -> dict:
    handoff = (band or {}).get("handoff")
    if handoff is None:
        return {"mode": "chain"}
    return {"mode": "handoff", "path": handoff["path"],
            "sha256": handoff["sha256"], "producer": handoff["producer"],
            "handoff_sha256": handoff["handoff_sha256"]}


class Gateway:
    """How submissions reach PB. The default shells to the published
    client; tests inject :class:`FakeGateway`."""

    def submit(self, argv: list[str]) -> dict:
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"pbrun exited {proc.returncode}: {proc.stderr[-2000:]}")
        try:
            answer = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"pbrun printed no JSON line: {proc.stdout[-500:]}") from exc
        return {"action_key": answer["action_key"], "status": answer.get("status", "")}

    def is_terminal_executed(self, action_key: str) -> bool:
        proc = subprocess.run(
            [sys.executable, str(PBWAIT), "--wait-s", "0", action_key],
            capture_output=True, text=True)
        try:
            answer = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return False
        return answer.get("status") == "executed" or (
            isinstance(answer.get("terminal"), dict)
            and answer["terminal"].get("status") == "executed")


class FakeGateway(Gateway):
    """Fixture gateway: records submissions, never touches the fleet."""

    def __init__(self, *, terminal: bool = False):
        self.submitted: list[dict] = []
        self._terminal: set[str] = set()
        self._auto_terminal = terminal
        self._counter = 0

    def mark_terminal(self, action_key: str) -> None:
        self._terminal.add(action_key)

    def submit(self, argv: list[str]) -> dict:
        self._counter += 1
        key = f"fake-action-key-{self._counter:04d}"
        kind = ("stage-a" if any("joint_adjoint_capture" in word for word in argv)
                else "quantum")
        entry: dict = {"kind": kind, "argv": argv, "action_key": key}
        if kind == "quantum":
            entry["quantum_id"] = argv[argv.index("--quantum") + 1].split("/")[-1].replace(".json", "")
        self.submitted.append(entry)
        if self._auto_terminal:
            self._terminal.add(key)
        return {"action_key": key, "status": "submitted"}

    def is_terminal_executed(self, action_key: str) -> bool:
        return action_key in self._terminal


def _read_state(state_path: Path) -> list[dict]:
    if not state_path.exists():
        return []
    events = []
    for line in state_path.read_text().splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _append_state(state_path: Path, event: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps({**event, "unix": time.time()}, sort_keys=True) + "\n").encode()
    with open(state_path, "ab") as handle:
        handle.write(line)


def _plan_block(plan_path: Path | None) -> dict:
    if plan_path is None:
        return {}
    plan = _load_json(plan_path, where="plan")
    block = plan.get("distributed_campaign", {})
    return block if isinstance(block, dict) else {}


def plan_consumer_tags(block: Mapping) -> tuple[str, ...]:
    """The effective §5.1 placement tags for the quantum rows.

    The plan's ``distributed_campaign.consumer_tags`` when it declares them,
    else the shared GB10 class tag.  PB requires *every* listed tag
    (``wanted.issubset(offer.tags)``), so the list is a conjunction, never a
    menu of acceptable boxes: an empty list would publish an unconstrained
    row, and two host names would publish a row no single box can claim (the
    defect this default fixes).  A plan that pins one box names that host
    tag alone; anything ill-typed refuses at dispatch time, before a row is
    sealed."""
    raw = block.get("consumer_tags")
    if raw is None:
        return CONSUMER_TAGS
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise DispatchRefused(
            "plan distributed_campaign.consumer_tags must be a list of tags")
    if not raw or any(not isinstance(tag, str) or not tag for tag in raw):
        raise DispatchRefused(
            "plan distributed_campaign.consumer_tags must be non-empty strings")
    return tuple(raw)


def _coverage_row(record: dict, *, output_root: Path, band: dict | None) -> dict:
    """What the source-read coverage check needs of one executable row."""
    handoff = (band or {}).get("handoff")
    if handoff is not None:
        # A band-serial consumer takes its cotangent from the handoff and
        # installs only its own layer.
        return {"record": record, "manifest_path": str(handoff["manifest_path"]),
                "manifest_sha256": handoff["manifest_sha256"],
                "order": (record["layer"],)}
    manifest = Path(record["executable_readset"].get("manifest_path", ""))
    if not manifest.is_absolute():
        manifest = output_root / manifest
    return {"record": record, "manifest_path": str(manifest),
            "manifest_sha256": record["executable_readset"]["manifest_sha256"],
            "order": None}


def check_source_coverage(coverage_rows: list[dict], *, coverage=None) -> None:
    """Refuse a publication whose readsets miss a source read (PQ #1095).

    Every executable row's manifest has to declare every source tensor its
    quantum's streaming loader reads: the resident head in ``head`` and each
    installed layer in its source phase (``prismaquant.readset_coverage``).
    The check reads headers only and needs no GPU. It lists every gap of
    every row, then refuses once; a dry run refuses the same way.
    """
    if not coverage_rows:
        return
    if coverage is None:
        from prismaquant.readset_coverage import quantum_rows_gaps as coverage
    gaps = coverage(coverage_rows)
    if gaps:
        from prismaquant.readset_coverage import gaps_report
        print(json.dumps(gaps_report(gaps, rows=len(coverage_rows)),
                         indent=2, sort_keys=True, default=str), file=sys.stderr)
        raise DispatchRefused(
            f"{len(gaps)} source read(s) of {len(coverage_rows)} row(s) are "
            "not declared by their readsets (listed above): rebuild the "
            "readsets from the loader's source plan")


def main(argv: list[str] | None = None, _gateway: Gateway | None = None,
         _coverage=None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish distributed joint-AURA campaign rows (§5).")
    parser.add_argument("--records", required=True,
                        help="directory of sealed layer-quantum records")
    parser.add_argument("--output-root", required=True,
                        help="campaign output root (state + quantum spaces)")
    parser.add_argument("--adjoint-receipt", default=None,
                        help="stage-A adjoint-capture.json, when published")
    parser.add_argument("--adjoint-band", action="append", default=[],
                        help="a sealed stage-A checkpoint band (repeatable; PQ "
                             "#993): publishes the quanta of that band's layers "
                             "before the receipt lands")
    parser.add_argument("--adjoint-manifest", default=None,
                        help="stage-A read manifest (defaults beside records)")
    parser.add_argument("--plan", default=None,
                        help="plan file carrying the distributed_campaign block")
    parser.add_argument("--priority", type=int, default=SUBMISSION_PRIORITY)
    parser.add_argument("--head-grace-s", type=int, default=HEAD_PROGRESS_GRACE_S)
    parser.add_argument("--stage-a-prefetch-override", type=Path, default=None,
                        help="explicit source_prefetch override document for "
                             "the stage-A action (#819): threaded into the "
                             "payload's --prefetch-override; the run stamps "
                             "the deviation into its provenance")
    parser.add_argument("--stage-a-artifact-budget-bytes", default=None,
                        help="explicit durable artifact ceiling in bytes for "
                             "the stage-A action (#882): threaded into the "
                             "payload's --artifact-budget-bytes; the run stamps "
                             "the deviation into its provenance; the sealed "
                             "plan is unchanged")
    parser.add_argument("--stage-a-produced-output-template", type=Path,
                        default=None,
                        help="produced-output template JSON declaring the "
                             "bounded window the stage-A action will stage "
                             "for the boundary entries it produces itself "
                             "(RobTand/prismaquant#881). Sealed by pbrun as "
                             "a declared input plus request params and "
                             "carried onto the queue row, so the capture can "
                             "read its OWN entries back; its window demand is "
                             "derived by pbrun from the template. Without it "
                             "the capture writes entries it cannot read.")
    parser.add_argument("--band-serial", action="store_true",
                        help="PQ #996: inside a checkpoint band, quantum L-1 "
                             "takes quantum L's final input cotangent (its "
                             "handoff) instead of rebuilding the chain. "
                             "Producer rows declare the handoff as a produced "
                             "output; a consumer row is published once its "
                             "producer executed and published a handoff. "
                             "Executable rows only")
    parser.add_argument("--handoff-tier", default=None,
                        help="the stage tier a producer row's handoff "
                             "template permits (for example "
                             "prismabuild-stage:dl380g10); required with "
                             "--band-serial")
    parser.add_argument("--spec", default=None,
                        help="campaign spec for the container wrapper (default: the joint-panel dev spec)")
    parser.add_argument("--state", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    gateway = _gateway if _gateway is not None else Gateway()
    if args.spec:
        global SPEC_PATH
        SPEC_PATH = Path(args.spec)
    records_dir = Path(args.records)
    output_root = Path(args.output_root)
    state_path = (Path(args.state) if args.state is not None
                  else output_root / "layer-quanta" / STATE_FILENAME)
    try:
        block = _plan_block(Path(args.plan) if args.plan else None)
        tags = plan_consumer_tags(block)
        records = load_records(records_dir)
    except DispatchRefused as exc:
        print(f"dispatch_joint_quanta: refused: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION_REFUSED
    priority = int(block.get("submission_priority", args.priority))
    adjoint_tag = str(block.get("adjoint", {}).get("tag", ADJOINT_TAG)
                      if isinstance(block.get("adjoint"), dict)
                      else ADJOINT_TAG)
    events = _read_state(state_path)
    submitted_keys = {event.get("quantum_id"): event.get("action_key")
                      for event in events if event.get("event") == "quantum-submitted"}
    last_submission = {event.get("quantum_id"): event for event in events
                       if event.get("event") == "quantum-submitted"}
    if args.band_serial and args.handoff_tier is None:
        print("dispatch_joint_quanta: refused: --band-serial needs --handoff-tier",
              file=sys.stderr)
        return EXIT_PRECONDITION_REFUSED
    if args.band_serial and any(record.get("executable_readset") is None
                                for _, record in records):
        print("dispatch_joint_quanta: refused: --band-serial runs executable "
              "rows only, and these records carry no executable readset",
              file=sys.stderr)
        return EXIT_PRECONDITION_REFUSED
    roles = band_serial_roles(records)
    by_id = {record["quantum_id"]: record for _, record in records}
    band_pending: list[dict] = []
    stage_a_keys = [event.get("action_key") for event in events
                    if event.get("event") == "stage-a-submitted"]

    receipt_path = (Path(args.adjoint_receipt) if args.adjoint_receipt
                    else _plan_output_root(records[0][1]["campaign"])
                    / "layer-quanta" / "adjoint" / "adjoint-capture.json")
    publishable: dict[str, str] = {}
    try:
        publishable = check_stage_a_proofs(
            load_stage_a_proofs(receipt_path, [Path(p) for p in args.adjoint_band],
                                receipt_named=args.adjoint_receipt is not None),
            records)
    except DispatchRefused as exc:
        # A named band that fails its gate always refuses. A stale receipt
        # fails closed only once stage A is terminally done (otherwise the
        # receipt simply has not landed yet and the run reports pending).
        if args.adjoint_band or (
                stage_a_keys and gateway.is_terminal_executed(stage_a_keys[-1])):
            print(f"dispatch_joint_quanta: refused: {exc}", file=sys.stderr)
            return EXIT_PRECONDITION_REFUSED
    receipt_ok = bool(publishable)

    rows: list[dict] = []
    coverage_rows: list[dict] = []
    # A stage A that was submitted but did not terminally execute (failed,
    # withdrawn, lost) is republished: the state file records the attempt,
    # never the outcome, and retry is free (#5 contract).  Only a terminally
    # executed capture, or a validated receipt, stops republication.
    stage_a_done = bool(stage_a_keys) and gateway.is_terminal_executed(stage_a_keys[-1])
    stage_a_binding: dict | None = None
    try:
        if not stage_a_done and not receipt_ok:
            manifest = (Path(args.adjoint_manifest) if args.adjoint_manifest
                        else records_dir / "adjoint.data-manifest.json.gz")
            stage_a_binding = _stage_manifest_binding(
                manifest, records[0][1]["campaign"])
            rows.append({"kind": "stage-a",
                         "argv": stage_a_argv(manifest, records[0][1]["campaign"],
                                              tag=adjoint_tag,
                                              prefetch_override=args.stage_a_prefetch_override,
                                              artifact_budget_bytes=args.stage_a_artifact_budget_bytes,
                                              produced_output_template=(
                                                  args.stage_a_produced_output_template),
                                              binding=stage_a_binding)})
        if receipt_ok:
            for record_path, record in records:
                quantum_id = record["quantum_id"]
                if quantum_id not in publishable:
                    continue
                key = submitted_keys.get(quantum_id)
                if key is not None and gateway.is_terminal_executed(key):
                    continue
                band = None
                if quantum_id in last_submission:
                    band = recorded_band_role(
                        record, last_submission[quantum_id],
                        output_root=output_root)
                elif args.band_serial:
                    band, waiting = fresh_band_role(
                        record, roles=roles, by_id=by_id,
                        last_submission=last_submission,
                        submitted_keys=submitted_keys, gateway=gateway,
                        tier=args.handoff_tier, output_root=output_root)
                    if waiting is not None:
                        band_pending.append({"quantum_id": quantum_id,
                                             "reason": waiting})
                        continue
                handoff = (band or {}).get("handoff")
                template = (band or {}).get("emit_template")
                if record.get("executable_readset") is not None:
                    coverage_rows.append(_coverage_row(
                        record, output_root=output_root, band=band))
                rows.append({"kind": "quantum", "quantum_id": quantum_id,
                             "identity_sha256": record["identity_sha256"],
                             "manifest_sha256": (
                                 handoff["manifest_sha256"] if handoff
                                 else _row_manifest_sha256(record)),
                             "cotangent_source": _cotangent_source(band),
                             "handoff_template": (
                                 None if template is None else str(template)),
                             "argv": quantum_argv(
                                 record, record_path=record_path,
                                 output_root=output_root, priority=priority,
                                 head_grace_s=args.head_grace_s,
                                 consumer_tags=tags, band=band)})
        check_source_coverage(coverage_rows, coverage=_coverage)
    except DispatchRefused as exc:
        print(f"dispatch_joint_quanta: refused: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION_REFUSED

    if args.dry_run:
        by_id = {record["quantum_id"]: record for _, record in records}
        print(json.dumps({"consumer_tags": list(tags), "priority": priority,
                          "stage_a_terminal": bool(stage_a_keys) and bool(
                              receipt_ok),
                          "stage_a_pending": [record["quantum_id"] for _, record in records
                                              if record["quantum_id"] not in publishable],
                          "band_serial_pending": band_pending,
                          "rows": [{"kind": row["kind"],
                                    "quantum_id": row.get("quantum_id"),
                                    "identity_sha256": row.get("identity_sha256"),
                                    "manifest_sha256": row.get("manifest_sha256"),
                                    **({"cotangent_source": row["cotangent_source"],
                                        "handoff_template": row["handoff_template"]}
                                       if row["kind"] == "quantum" else {}),
                                    **({"slice_sha256": publishable[row["quantum_id"]],
                                        **{key: by_id[row["quantum_id"]]["adjoint"][key]
                                           for key in ("checkpoint_boundary", "chain_layers")}}
                                       if row["kind"] == "quantum" else {}),
                                    "argv": row["argv"]}
                                   for row in rows]},
                         indent=2, sort_keys=True))
        return 0

    try:
        for row in rows:
            answer = gateway.submit(row["argv"])
            if row["kind"] == "stage-a":
                _append_state(state_path, {"event": "stage-a-submitted",
                                           "action_key": answer["action_key"],
                                           "data_manifest_sha256": (
                                               stage_a_binding or {}
                                           ).get("data_manifest_sha256"),
                                           "read_manifest_sha256": (
                                               stage_a_binding or {}
                                           ).get("read_manifest_sha256"),
                                           "prefetch_override": (
                                               str(args.stage_a_prefetch_override)
                                               if args.stage_a_prefetch_override
                                               else None),
                                           "artifact_budget_bytes": (
                                               str(args.stage_a_artifact_budget_bytes)
                                               if args.stage_a_artifact_budget_bytes
                                               is not None
                                               else None)})
            else:
                _append_state(state_path,
                              {"event": "quantum-submitted",
                               "quantum_id": row["quantum_id"],
                               "identity_sha256": row["identity_sha256"],
                               "data_manifest_sha256": row.get("manifest_sha256"),
                               "cotangent_source": row["cotangent_source"],
                               "handoff_template": row["handoff_template"],
                               "action_key": answer["action_key"]})
            print(json.dumps({"published": row.get("quantum_id", "stage-a"),
                              "action_key": answer["action_key"],
                              "status": answer.get("status", "")},
                             sort_keys=True))
    except (RuntimeError, OSError) as exc:
        print(f"dispatch_joint_quanta: submission failed: {exc}", file=sys.stderr)
        return EXIT_SUBMIT_FAILED
    for waiting in band_pending:
        print(json.dumps({"band_serial_pending": waiting["quantum_id"],
                          "reason": waiting["reason"]}, sort_keys=True))
    if not rows:
        print(json.dumps({"published": [], "note": "nothing publishable"},
                         sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
