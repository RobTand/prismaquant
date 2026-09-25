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
Its stdout is that one JSON document; every log line goes to stderr.
It writes nothing either: a band-serial row's handoff template and readset
are derived and printed with their digests, never published (PQ #1200).

Shapes owned elsewhere (fixtures here, never imports): the layer-quantum
record (§3, built in parallel by the producer) and the stage-A receipt
(§3.3, built in parallel by stage A). Real dispatch is the coordinator's
call after the runtime's cutover check; this tool's tests use fixtures.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
import time
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
#: An executable row's phases inherited it; its compute phases now derive
#: their own (COMPUTE_PHASE_BOUND, PQ #1165), and its source and
#: read-only render phases keep it.
CHUNK_PROGRESS_GRACE_S = 900
#: The head phase's stall allowance. Not pinned by the contract (only the
#: chunk grace is); the sealed default below is overridable via
#: --head-grace-s and pinned by the dispatcher's own tests. It is also the
#: blanket grace a load phase takes when no floor applies (below).
HEAD_PROGRESS_GRACE_S = 1800

#: checkpoint-load and handoff-load commit no progress units: a read into a
#: disposable scratch is not durable work (PB #480). Their grace is therefore
#: the phase's whole time budget, derived per row as W + ceil(bytes / floor):
#:
#: * W is the spec's ``PRISMAQUANT_STAGED_RANGE_WAIT_S``, read with the
#:   reader's own rules. The reader sets one deadline, start + W, for every
#:   staged wait in the phase (prismaquant/joint_adjoint_checkpoints.py:1869 in
#:   load_adjoint_checkpoint, prismaquant/joint_quantum_handoff.py:1034 in
#:   load_handoff_inputs). Without a PrismaBuild landing record the phase
#:   waits at most W in total; with one (PB #989) its waits are declared and
#:   exempt from the no-progress clock.
#: * bytes is the phase's byte count in the row's staged read plan, and at
#:   the floor rate or faster the transfer takes at most ceil(bytes / floor).
#:
#: So W + ceil(bytes / floor) bounds the phase's non-exempt time, and
#: :func:`require_staged_wait_below_grace` holds for any phase with bytes.
LOAD_PHASE_GRACE_SCHEMA = "prismaquant.load_phase_grace.v1"
LOAD_PHASE_FLOOR_SCHEMA = "prismaquant.load_phase_floor.v1"
LOAD_PHASE_BOUND = (
    "grace = W + ceil(bytes / floor). The reader sets one deadline, start + W, "
    "for every staged wait in the phase (prismaquant/joint_adjoint_checkpoints.py"
    ":1869 load_adjoint_checkpoint; prismaquant/joint_quantum_handoff.py:1034 "
    "load_handoff_inputs), so without a PrismaBuild landing record the phase "
    "waits at most W in total, and with one (PB #989) the waits are declared "
    "and exempt. At or above the floor rate the transfer takes at most "
    "ceil(bytes / floor).")
#: Who counts as a reader of the link, for every stamp.
LINK_READERS_SCOPE = (
    "this dispatch's own rows only: readers from other workloads on the same "
    "link, and rows an earlier dispatch published, are not counted")
#: The measured floor for one reader on the dl380g10 link: the slowest 30 s
#: window of the reading process's ``read_bytes`` over its checkpoint-load
#: phase, from the host samplers of the two R13 layer-044 gates on sparky
#: (``/home/rob/tmp/ws-sb4/out/<key12>/io.tsv``, 10 s samples). v4 read into
#: a host dict under memory pressure; v5 read into the cotangent scratch.
#: Measured with ONE reader: it does not apply at any other concurrency.
LOAD_PHASE_FLOOR_ONE_READER = {
    "schema": LOAD_PHASE_FLOOR_SCHEMA,
    "floor_bytes_per_s": 62_954_973,
    "readers": 1,
    "scope": "one reader on the dl380g10 link",
    "method": ("slowest 30 s window of the reading process's /proc read_bytes "
               "over its checkpoint-load phase, 10 s host samples"),
    "sources": [
        {"action_key": "70e7baeb6e96564ed5fd05b8eb90f1f709a550dcc213298a3dff5db5dd5dea8b",
         "gate": "R13 layer-044 v4", "host": "sparky",
         "window_start_utc": "2026-09-24T03:18:35Z", "bytes_per_s": 62_954_973},
        {"action_key": "2dc145299e0eebfef1b3fbc159307e07c4a5a2ca0cf394c187018a1e9d907e86",
         "gate": "R13 layer-044 v5", "host": "sparky",
         "window_start_utc": "2026-09-24T03:43:30Z", "bytes_per_s": 77_212_876},
    ],
}
#: The payload flag that carries a row's grace stamps (load phases and
#: compute phases) into the quantum, which copies them into results.json and
#: counters.json.
PROGRESS_GRACE_FLAG = "--progress-grace-derivation"

#: Stage B's compute phases (PQ #1165). A chain layer's backward roll
#: (``chain-NNN-bound``), a probe's spill capture (``spill-pP``), a spill
#: window's replay (``render-NN`` for NN >= 1 in a spill row) and a windowed
#: replay (``replay-NN-pP``) run one pass whose work lands in disposable
#: scratch: rolled rows in the cotangent scratch, spilled rows in the
#: ``O_TMPFILE`` spill. A resume recomputes both, so the pass has no durable
#: unit to commit (PB #480) and PrismaBuild sees no progress until the next
#: phase. Its grace is therefore the pass's whole time budget: the read term
#: plus each compute term, with each term's derivation in the stamp.
#: Before #1165 these phases took CHUNK_PROGRESS_GRACE_S, the legacy chunk
#: lane's allowance, which bounds no pass.
#: The quantum's tail after its last window (the ``payload`` and ``teardown``
#: spans, PQ #1187) declares no phase and commits no unit, so it runs on the
#: last declared phase's clock after that phase's last commit. That phase's
#: grace carries it as one more term, ``tail`` (PQ #1190).
COMPUTE_CEILING_SCHEMA = "prismaquant.compute_unit_ceiling.v1"
COMPUTE_PHASE_GRACE_SCHEMA = "prismaquant.compute_phase_grace.v1"
COMPUTE_PHASE_BOUND = (
    "grace = read + the sum of the compute terms. read is W + ceil(bytes / "
    "floor) with W counted once. The one-deadline property of the load-phase "
    "bound is shown only for load_adjoint_checkpoint and load_handoff_inputs; "
    "in a compute phase the staged waits are covered because PrismaBuild "
    "landing records (PB #989) declare them and exempt them from the "
    "no-progress clock. More than one undeclared wait is not covered. A "
    "measured compute term is ceil(units x unit_s), where unit_s is the "
    "slowest window of at least 30 s of a pass on the same regime and device "
    "class. An unmeasured term takes the blanket HEAD_PROGRESS_GRACE_S. The "
    "pass commits no unit (nothing in it is durable), so the grace is its "
    "whole time budget. The row's last declared phase adds a tail term: the "
    "quantum's tail after its last window (the payload and teardown spans) "
    "commits no unit and runs on that phase's clock after its last commit, "
    "so the term is ceil(1 x unit_s), unit_s the slowest measured tail "
    "(payload plus teardown wall_s) on the same regime and device class, or "
    "the blanket.")
#: What each compute kind counts, and where an unmeasured kind's time will
#: come from once a run records it.
COMPUTE_KINDS = {
    "chain-roll": {
        "unit": "one rolled cotangent row, a (probe, stored batch) of the "
                "chain layer's backward roll",
        "measured_by": "io.tsv write_bytes over the chain-NNN-bound phase"},
    "spill-capture": {
        "unit": "one capture group, capture_batch stored batches of one "
                "probe's spill capture",
        "measured_by": "the #1151 workspace profile's capture-b<N> groups; "
                       "counters.json spill telemetry capture_wall_s"},
    "spill-replay": {
        "unit": "one probe's replay of one window from the spill",
        "measured_by": "counters.json spill telemetry replay_wall_s and each "
                       "window's wall_s"},
    "windowed-replay": {
        "unit": "one stored batch's replay backward, with statistics hooks, "
                "for one probe",
        "measured_by": "counters.json each window's wall_s"},
    "tail": {
        "unit": "one quantum's tail after its last window: the band-serial "
                "handoff write, the payload assembly, the final check of "
                "every row, the runner shutdown and the residency report",
        "measured_by": "counters.json io_spans: the payload and teardown "
                       "spans' wall_s, summed (PQ #1187)"},
}
#: The tail's one term (PQ #1190), which :func:`compute_phase_work` appends to
#: the last declared phase's work.
TAIL_WORK = {"kind": "tail", "units": 1,
             "work": ("the quantum's tail after its last window, which "
                      "declares no phase and commits no unit")}
#: The per-row entry that carries the bound and the ceiling documents once,
#: so each compute stamp names its ceiling by kind and the payload stays
#: small.
COMPUTE_GRACE_BASIS_SCHEMA = "prismaquant.compute_phase_grace_basis.v1"
#: The measured unit ceilings the dispatcher applies when a row is in their
#: scope. ``scope.equal`` fields must match the row exactly and
#: ``scope.at_most`` fields must not exceed the measured value; ``basis`` says
#: why a row in scope takes no longer per unit. Any other row takes the
#: blanket for that term and the stamp names the field that differs.
#: ``--compute-ceiling FILE`` replaces the built-in document of its kind.
SPILL_CAPTURE_CEILING_GB10 = {
    "schema": COMPUTE_CEILING_SCHEMA,
    "kind": "spill-capture",
    "unit_s": 34.316,
    "method": ("slowest window of at least 30 s over the pass's own capture "
               "groups, in seconds per group; one group took 34.316 s, so it "
               "is its own window"),
    "samples": {"groups": 16, "median_s": 10.301, "mean_s": 13.025,
                "max_s": 34.316},
    "scope": {
        "consumer_tag": "gb10",
        "equal": {"replay_regime.capture_batch": 4,
                  "replay_regime.accumulation": "operator_gemm",
                  "replay_regime.chunk_rows": 65536,
                  "spill.capture_batch": 4, "spill.element_dtype": "bfloat16",
                  "spill.block": 4096, "spill.geometry.element_size": 2,
                  "spill.geometry.experts_per_token": 8,
                  "spill.geometry.max_batch_tokens": 2048,
                  "spill.geometry.largest_tensor_bytes": 16777216,
                  "spill.geometry.n_probes": 4},
        "at_most": {"spill.geometry.batch_bytes": 629145600,
                    "spill.geometry.batch_x_bytes": 327155712},
        "basis": ("GLM-5.3's routed layers share one shape, so a group's "
                  "forward and backward is the same work at the same tokens "
                  "per batch and routing fan-out; the spill writes per group "
                  "scale with batch_bytes. Measured on layer 44 only; the "
                  "at_most fields are an assumption this measurement does "
                  "not test"),
    },
    "sources": [{
        "action_key": "8f5422dc1d1b243e5d5756c9416ba827be905e35393f703b1ee6e3ab54e69f61",
        "receipt": ("/mnt/shared/tessera-measurements/glm-campaign-takeover-"
                    "20260913/ws-sb4-1151/20260924T060449Z-ec053c1ab79b/"
                    "profile.json"),
        "receipt_sha256": "fe42262bff4c5a14888e37bb67bbab93cfcae33843fcceceb7fbcdb923f31e0e",
        "setting": "capture-b4", "quantum_id": "layer-044",
        "record_identity_sha256": "51087d963d723f2f5ae928415b98d9b42491019f5a9bcf7238e4a73856393ef9",
        "host": "sparky", "device": "NVIDIA GB10",
        "git_commit": "7595bd350c34abbcd71d50a24bee77f806e27e33"}],
}
CHAIN_ROLL_CEILING_GB10 = {
    "schema": COMPUTE_CEILING_SCHEMA,
    "kind": "chain-roll",
    "unit_s": 0.8231,
    "method": ("slowest window of at least 30 s of the rolling process's "
               "/proc write_bytes over its chain-NNN-bound phase, 10 s host "
               "samples, as seconds per row of entry_bytes (16779369 B at "
               "20386452 B/s, rounded up)"),
    "samples": {"rows": 1585, "phase_s": 964, "mean_s": 0.608},
    "scope": {
        "consumer_tag": "gb10",
        "equal": {"chain_regime.batch_size": 4,
                  "chain_regime.probe_fusion": True, "n_probes": 4,
                  "entry_bytes": 16779369},
        "basis": ("a row is one probe's cotangent for one stored batch of "
                  "entry_bytes; at one chain regime and row size each row is "
                  "the same backward work. Measured on chain layer 44 only; "
                  "the scope does not tell a dense chain layer from a routed "
                  "one"),
    },
    "sources": [{
        "action_key": "93247fc291c0996013ce2606728df0f19b705199444637c360e57d7f619f43df",
        "samples": "/home/rob/tmp/ws-sb4/out/93247fc291c0/io.tsv",
        "samples_sha256": "d7368bc7e5d34b88ffa8341006639c1a533eb83e196e4c33ec431fb743dbeaa8",
        "quantum_id": "layer-043", "phase": "chain-044-bound",
        "window_unix": [1790237681, 1790237712], "host": "sparky",
        "device": "NVIDIA GB10",
        "git_commit": "b7ae25d091c0d2b89b205fa8b9e4d08867468ded"}],
}
COMPUTE_UNIT_CEILINGS = (SPILL_CAPTURE_CEILING_GB10, CHAIN_ROLL_CEILING_GB10)

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


def with_execution_plan(records: list[tuple[Path, dict]],
                        execution_plan: Path) -> tuple[list[tuple[Path, dict]], dict]:
    """The records as this dispatch runs them under ``--execution-plan`` (PQ #1191).

    A re-declared Stage B plan (a resource re-declare, for example) gets its
    own path. The sealed plan stays at the record's ``campaign.plan_path``,
    where the band prepare reads it against the sealed digest; writing the
    re-declared plan over that path blocked every later prepare.

    Returns the records with their in-memory ``campaign.plan_path`` naming
    the execution plan, so every execution-time plan read of this dispatch
    (the row's ``--plan`` and its digest, the Stage A memory bound and spool
    window, the plan's output root, the handoff template, the readset
    coverage check) takes that file. ``campaign.plan_sha256`` stays the
    sealed digest: the Stage A proof gate and the manifest binding compare
    with it, and the argv digest is derived from the bytes read, as for a
    re-declared plan at the sealed path. The record files are never written.
    The second value is the stamp the dry run and the state events carry.

    The records share one campaign (:func:`load_records`), so the plan is
    read and compared once. A digest that differs from the sealed one is a
    run seal (PQ #1147): dev mode prints it, certified mode refuses it.
    """
    from prismaquant.dev_mode import seal_check

    try:
        raw = Path(execution_plan).read_bytes()
        json.loads(raw)
    except (OSError, ValueError) as exc:
        raise DispatchRefused(
            f"--execution-plan {execution_plan}: unreadable JSON: {exc}") from exc
    campaign = records[0][1].get("campaign")
    if not isinstance(campaign, dict):
        raise DispatchRefused("the layer records carry no campaign block")
    sealed = campaign.get("plan_sha256")
    actual = _sha_bytes(raw)
    seal_check("execution plan", sealed, actual,
               where=f"--execution-plan {execution_plan}",
               refusal=lambda: DispatchRefused(
                   f"execution plan {execution_plan} ({actual}) differs from "
                   f"the records' sealed plan {campaign.get('plan_path')} "
                   f"({sealed})"))
    stamp = {"path": str(execution_plan), "sha256": actual,
             "sealed_plan_path": campaign.get("plan_path"),
             "sealed_plan_sha256": sealed}
    running = [(path, {**record, "campaign": {**record["campaign"],
                                              "plan_path": str(execution_plan)}})
               for path, record in records]
    return running, stamp


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
                            head_grace_s: int, load_grace=None,
                            compute_grace=None):
    """Pure executable-row construction from sealed inputs (no gate).

    Resolves the row's executable manifest, verifies its wire bytes hash to
    the sealed digest, and derives the read-phase progress declarations in
    manifest order. Consults no output binding and bypasses no production
    gate: production dispatch (:func:`quantum_argv`) refuses every
    executable row with :class:`ExecutableBindingUnsupported` before
    reaching here. Tests exercise manifest/phase propagation through this
    helper directly.

    ``load_grace(name, phase_bytes)`` returns the checkpoint-load grace from
    the phase's byte count in this manifest (:func:`load_phase_grace`).
    Without it the phase takes the blanket :data:`HEAD_PROGRESS_GRACE_S`.
    ``compute_grace(name, facts, annotations)`` returns a compute phase's
    grace (:func:`compute_phase_grace`), or ``None`` for a phase that runs no
    compute; every other phase, and every phase without it, takes
    :data:`CHUNK_PROGRESS_GRACE_S`.
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
    # The wire just hashed to the sealed digest above; its read plan names
    # each phase's bytes and entries.
    facts, annotations = (
        _manifest_phase_facts(manifest, quantum_id=quantum_id)
        if load_grace is not None or compute_grace is not None else ({}, {}))
    progress = [("head", head_grace_s)]
    for name in phases:
        if name == "head":
            continue
        fact = facts.get(name) or {}
        if name == "checkpoint-load":
            grace = (HEAD_PROGRESS_GRACE_S if load_grace is None
                     else load_grace(name, fact.get("bytes")))
        else:
            grace = (None if compute_grace is None
                     else compute_grace(name, fact, annotations))
            if grace is None:
                grace = CHUNK_PROGRESS_GRACE_S
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


def stage_a_spool_window_bytes(campaign: Mapping,
                               batch_range: Sequence[int] | None = None) -> int:
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

    ``batch_range`` (a chain split quantum, PQ #738) is the ``(start, stop)``
    of the batches the row owns: its planes hold those batches only, and the
    capture derives its need from the same range.
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
    n_batches = len(row_offsets)
    if batch_range is not None:
        start, stop = (int(value) for value in batch_range)
        if not 0 <= start < stop <= n_batches:
            raise DispatchRefused(
                f"batch range {start}:{stop} is not inside the plan's {n_batches} batches")
        n_batches = stop - start
    return two_plane_window_bytes(
        n_probes=n_probes, n_batches=n_batches, group_size=group_size,
        max_entry_tensor_bytes=tensor_bytes)


def _argv_file_sha256(campaign: Mapping, key: str, *, where: str,
                      raw: bytes | None = None) -> str:
    """The digest a row's argv names for the campaign's plan or prepared file.

    The record seals the digest of each file it was cut from, and the worker
    checks the file against the digest its argv names. In dev mode that
    record digest is a run seal (PQ #1147): the argv names the file on disk
    by the digest of its bytes, and ``seal_check`` prints the difference, so
    a re-declared file runs instead of dispatching only to refuse. A file dev
    mode cannot read keeps the record's digest, and the worker reports it.
    """
    from prismaquant.dev_mode import dev_mode_enabled, seal_check

    sealed = str(campaign[f"{key}_sha256"])
    # Certified mode names the record's digest, byte-identical to main; a
    # re-declared file refuses at the worker (joint_cost_quantum.py:229).
    if not dev_mode_enabled():
        return sealed
    if raw is None:
        try:
            raw = Path(campaign[f"{key}_path"]).read_bytes()
        except OSError:
            return sealed
    actual = hashlib.sha256(raw).hexdigest()
    seal_check(f"record {key}", sealed, actual, where=f"{where} argv")
    return actual


def _plan_output_root(campaign: Mapping) -> Path:
    """The plan's sealed output_root: the only root the stage-A capture will
    write into (its identity guard refuses any other --output-root), and the
    root whose ``layer-quanta/adjoint/adjoint-capture.json`` is the receipt
    this dispatcher validates."""
    plan = json.loads(Path(campaign["plan_path"]).read_text())
    return Path(plan["output_root"])


def load_phase_floor(document: object) -> dict:
    """A validated load-phase floor document, or :class:`DispatchRefused`.

    The built-in :data:`LOAD_PHASE_FLOOR_ONE_READER` and any document given
    with ``--checkpoint-load-floor`` take the same shape: the floor in bytes
    per second, the number of concurrent link readers it was measured at, a
    scope, a method and the action keys it came from.
    """
    def refuse(why):
        raise DispatchRefused(f"load-phase floor document: {why}")

    if not isinstance(document, Mapping):
        raise DispatchRefused("load-phase floor document: not a JSON object")
    if document.get("schema") != LOAD_PHASE_FLOOR_SCHEMA:
        refuse(f"schema is not {LOAD_PHASE_FLOOR_SCHEMA}")
    floor = document.get("floor_bytes_per_s")
    if type(floor) is not int or floor <= 0:
        refuse("floor_bytes_per_s must be a positive integer")
    readers = document.get("readers")
    if type(readers) is not int or readers <= 0:
        refuse("readers must be a positive integer")
    for key in ("scope", "method"):
        if not isinstance(document.get(key), str) or not document[key]:
            refuse(f"{key} must be a non-empty string")
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources or not all(
            isinstance(source, Mapping) and _is_hex64(source.get("action_key"))
            for source in sources):
        refuse("sources must list the measurements, each with its 64-hex "
               "action_key")
    return dict(document)


def link_readers(*, rows: int, declared: int | None = None,
                 floor_document: object = None) -> dict:
    """How many of this dispatch's rows can share the link, and their floor.

    ``rows`` is the number of quantum rows the dispatch publishes, the most
    of its own readers that can read the link at once; ``declared``
    (``--link-readers``) replaces it. A floor document must be measured at
    exactly that many readers. Without one, the built-in floor applies to
    one reader only, and any other count takes the blanket grace.
    """
    readers = int(rows if declared is None else declared)
    if readers < 0 or (declared is not None and readers < 1):
        raise DispatchRefused(f"--link-readers must be at least 1, not {readers}")
    source = ("--link-readers" if declared is not None
              else "the quantum rows this dispatch publishes")
    if readers == 0:
        return {"readers": 0, "readers_source": source, "floor": None}
    if floor_document is not None:
        floor = load_phase_floor(floor_document)
        if floor["readers"] != readers:
            raise DispatchRefused(
                f"the load-phase floor was measured at {floor['readers']} "
                f"link reader(s), but this dispatch has {readers} ({source}); "
                "a floor applies only at the concurrency it was measured at")
    elif readers == LOAD_PHASE_FLOOR_ONE_READER["readers"]:
        floor = load_phase_floor(LOAD_PHASE_FLOOR_ONE_READER)
    else:
        floor = None
    return {"readers": readers, "readers_source": source, "floor": floor}


def load_phase_grace(name: str, *, phase_bytes: int, staged_wait_s: float,
                     link: Mapping | None) -> dict:
    """The stall allowance of one load phase, with the stamp that explains it.

    ``mode`` is ``derived`` (W + ceil(bytes / floor), see
    :data:`LOAD_PHASE_BOUND`) when a floor measured at this dispatch's
    reader count applies, and ``blanket`` (:data:`HEAD_PROGRESS_GRACE_S`)
    otherwise, with the reason. No concurrency discount is ever applied.
    """
    counted = type(phase_bytes) is int and phase_bytes >= 0
    stamp = {"schema": LOAD_PHASE_GRACE_SCHEMA, "phase": str(name),
             "phase_bytes": phase_bytes if counted else None,
             "staged_wait_s": staged_wait_s,
             "readers": None if link is None else link["readers"],
             "readers_source": None if link is None else link["readers_source"],
             "readers_scope": LINK_READERS_SCOPE}
    blanket = {**stamp, "mode": "blanket", "grace_s": HEAD_PROGRESS_GRACE_S}
    if link is None:
        return {**blanket,
                "reason": "the row was built without a link-reader count"}
    floor = link.get("floor")
    if floor is None:
        return {**blanket,
                "reason": (f"no floor measured at {link['readers']} link "
                           "readers; the built-in floor is for one reader")}
    if not counted:
        return {**blanket,
                "reason": "the read plan declares no byte count for this phase"}
    if phase_bytes == 0:
        # W + 0 would equal the reader's own deadline, and the no-progress
        # clock would race it (require_staged_wait_below_grace).
        return {**blanket, "reason": "the read plan declares 0 bytes for this "
                                     "phase, so there is no transfer to bound"}
    transfer = -(-phase_bytes // int(floor["floor_bytes_per_s"]))
    return {**stamp, "mode": "derived",
            "grace_s": math.ceil(staged_wait_s) + transfer,
            "transfer_s": transfer,
            "floor_bytes_per_s": int(floor["floor_bytes_per_s"]),
            "floor_readers": int(floor["readers"]),
            "floor_scope": floor["scope"], "floor_method": floor["method"],
            "floor_sources": [dict(source) for source in floor["sources"]],
            # The --checkpoint-load-floor file (path, sha256); None for the
            # built-in one-reader floor.
            "floor_document": floor.get("document"),
            "bound": LOAD_PHASE_BOUND}


def compute_unit_ceiling(document: object) -> dict:
    """A validated compute ceiling document, or :class:`DispatchRefused`.

    The built-in documents in :data:`COMPUTE_UNIT_CEILINGS` and any document
    given with ``--compute-ceiling`` take the same shape: a kind from
    :data:`COMPUTE_KINDS`, the measured seconds per unit, the method, the
    scope the measurement covers, and the action keys it came from.
    """
    def refuse(why):
        raise DispatchRefused(f"compute ceiling document: {why}")

    if not isinstance(document, Mapping):
        raise DispatchRefused("compute ceiling document: not a JSON object")
    if document.get("schema") != COMPUTE_CEILING_SCHEMA:
        refuse(f"schema is not {COMPUTE_CEILING_SCHEMA}")
    if document.get("kind") not in COMPUTE_KINDS:
        refuse(f"kind must be one of {sorted(COMPUTE_KINDS)}")
    unit_s = document.get("unit_s")
    if (type(unit_s) not in (int, float) or not math.isfinite(unit_s)
            or unit_s <= 0):
        refuse("unit_s must be a positive number of seconds")
    if not isinstance(document.get("method"), str) or not document["method"]:
        refuse("method must be a non-empty string")
    scope = document.get("scope")
    if not isinstance(scope, Mapping):
        refuse("scope must be an object")
    if not isinstance(scope.get("consumer_tag"), str) or not scope["consumer_tag"]:
        refuse("scope.consumer_tag must name the device class it was measured on")
    if not isinstance(scope.get("basis"), str) or not scope["basis"]:
        refuse("scope.basis must say why a row in scope takes no longer per unit")
    for key in ("equal", "at_most"):
        if not isinstance(scope.get(key, {}), Mapping):
            refuse(f"scope.{key} must be an object")
    if any(type(value) not in (int, float) or isinstance(value, bool)
           for value in scope.get("at_most", {}).values()):
        refuse("scope.at_most values must be numbers")
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources or not all(
            isinstance(source, Mapping) and _is_hex64(source.get("action_key"))
            for source in sources):
        refuse("sources must list the measurements, each with its 64-hex "
               "action_key")
    return dict(document)


def compute_ceilings(documents: Sequence[Mapping] = ()) -> dict:
    """The ceiling per kind: the built-ins, each replaced by a given document."""
    by_kind = {}
    for document in (*COMPUTE_UNIT_CEILINGS, *documents):
        checked = compute_unit_ceiling(document)
        by_kind[checked["kind"]] = checked
    return by_kind


_CHAIN_BOUND_PHASE = re.compile(r"chain-(\d{3,})-bound")
_SPILL_PHASE = re.compile(r"spill-p(\d+)")
_RENDER_PHASE = re.compile(r"render-(\d{2,})")
_REPLAY_PHASE = re.compile(r"replay-(\d{2,})-p(\d+)")


def phase_work_entries(name: str, fact: Mapping, annotations: Mapping):
    """The entries a phase's pass counts its work by, from its read-plan facts.

    A phase's entry count, less the incoming plane entries a band-serial
    spill readset moved into it (``annotations.band_serial.streamed_incoming``,
    PQ #1143): those add read bytes to ``spill-pP``, not capture groups.
    ``None`` when the plan does not count the phase's entries.
    """
    entries = fact.get("entries") if isinstance(fact, Mapping) else None
    band = annotations.get("band_serial") if isinstance(annotations, Mapping) else None
    streamed = band.get("streamed_incoming") if isinstance(band, Mapping) else None
    moved = streamed.get(name) if isinstance(streamed, Mapping) else None
    if type(entries) is not int or moved is None:
        return entries
    if type(moved) is not int or not 0 <= moved <= entries:
        return None
    return entries - moved


def compute_phase_work(name: str, *, replay_mode: str, entries,
                       n_probes, capture_batch,
                       runs_tail: bool = False) -> list[dict] | None:
    """The compute one phase's pass runs, or ``None`` for a phase with none.

    Each term is ``{"kind", "units", "work"}``; ``units`` is ``None`` when
    the read plan or the row does not count them. The order follows
    ``joint_statistics_replay.observe_and_project_retained_windows``: a spill
    row captures each probe under ``spill-pP`` and replays the first window
    for that probe under the same phase, and replays every later window under
    its ``render-NN`` phase. A resume that has committed window 0 still
    captures under ``spill-pP`` and replays nothing there (PQ #1172), so the
    replay term of ``spill-pP`` is an upper bound for it. ``render-00`` of a
    spill row, and every ``render-NN`` of a windowed row, only read.

    ``runs_tail`` marks the row's last declared phase: the quantum's tail
    after its last window runs on its clock (PQ #1187), so :data:`TAIL_WORK`
    is its last term, with or without a pass of its own (PQ #1190).
    """
    def counted(*values):
        return all(type(value) is int and value > 0 for value in values)

    tail = [dict(TAIL_WORK)] if runs_tail else []
    match = _CHAIN_BOUND_PHASE.fullmatch(name)
    if match:
        return [{"kind": "chain-roll",
                 "units": entries * n_probes if counted(entries, n_probes) else None,
                 "work": (f"chain layer {int(match.group(1))}'s backward roll: "
                          "one row per probe and stored batch")}, *tail]
    match = _SPILL_PHASE.fullmatch(name)
    if match and replay_mode == "spill":
        probe = int(match.group(1))
        return [{"kind": "spill-capture",
                 "units": (-(-entries // capture_batch)
                           if counted(entries, capture_batch) else None),
                 "work": (f"probe {probe}'s spill capture: one group per "
                          "capture_batch stored batches")},
                {"kind": "spill-replay", "units": 1,
                 "work": f"the first window's spill replay for probe {probe}"},
                *tail]
    match = _RENDER_PHASE.fullmatch(name)
    if match and replay_mode == "spill" and int(match.group(1)) >= 1:
        return [{"kind": "spill-replay",
                 "units": n_probes if counted(n_probes) else None,
                 "work": (f"window {int(match.group(1))}'s spill replay for "
                          "each probe")}, *tail]
    match = _REPLAY_PHASE.fullmatch(name)
    if match and replay_mode == "windowed":
        return [{"kind": "windowed-replay",
                 "units": entries if counted(entries) else None,
                 "work": (f"window {int(match.group(1))}'s replay for probe "
                          f"{int(match.group(2))}: one backward per stored "
                          "batch")}, *tail]
    return tail or None


def _scope_misses(scope: Mapping, context: Mapping) -> list[str]:
    """Why a row is outside a measurement's scope; empty when it is inside."""
    misses = []
    tags = list(context.get("consumer_tags") or ())
    if scope["consumer_tag"] not in tags:
        misses.append(f"the row's consumer tags {tags} do not include the "
                      f"measured device class {scope['consumer_tag']!r}")
    for key, want in sorted(scope.get("equal", {}).items()):
        have = context.get(key)
        if type(have) is not type(want) or have != want:
            misses.append(f"{key} is {have!r}, measured at {want!r}")
    for key, most in sorted(scope.get("at_most", {}).items()):
        have = context.get(key)
        if type(have) not in (int, float) or isinstance(have, bool) or have > most:
            misses.append(f"{key} is {have!r}, above the measured {most!r}")
    return misses


def compute_phase_grace(name: str, *, work: Sequence[Mapping], phase_bytes,
                        staged_wait_s: float, link: Mapping | None,
                        context: Mapping, ceilings: Mapping) -> dict:
    """The stall allowance of one compute phase, with the stamp that explains it.

    ``grace_s`` is the read term (:func:`load_phase_grace` for the phase's
    bytes) plus each compute term: ``ceil(units x unit_s)`` from the ceiling
    of its kind when the row is in that ceiling's scope, and the blanket
    :data:`HEAD_PROGRESS_GRACE_S` otherwise, with the reason. ``mode`` is
    ``derived`` only when every term is. See :data:`COMPUTE_PHASE_BOUND`.
    """
    read = load_phase_grace(name, phase_bytes=phase_bytes,
                            staged_wait_s=staged_wait_s, link=link)
    read_term = {key: read[key] for key in (
        "mode", "grace_s", "phase_bytes", "staged_wait_s", "transfer_s",
        "floor_bytes_per_s", "readers", "reason") if key in read}
    terms = []
    for item in work:
        kind = item["kind"]
        term = {"kind": kind, "work": item["work"], "units": item["units"],
                "unit": COMPUTE_KINDS[kind]["unit"]}
        ceiling = ceilings.get(kind)
        misses = [] if ceiling is None else _scope_misses(ceiling["scope"], context)
        if ceiling is None:
            reason = (f"no measurement of {kind} on this regime; "
                      f"{COMPUTE_KINDS[kind]['measured_by']} will measure it")
        elif item["units"] is None:
            reason = "the read plan and the row do not count this term's units"
        elif misses:
            reason = "outside the measurement's scope: " + "; ".join(misses)
        else:
            reason = None
        if reason is not None:
            terms.append({**term, "mode": "blanket",
                          "grace_s": HEAD_PROGRESS_GRACE_S, "reason": reason})
            continue
        terms.append({**term, "mode": "derived",
                      "grace_s": math.ceil(item["units"] * ceiling["unit_s"]),
                      "unit_s": ceiling["unit_s"], "ceiling": kind})
    derived = read["mode"] == "derived" and all(
        term["mode"] == "derived" for term in terms)
    return {"schema": COMPUTE_PHASE_GRACE_SCHEMA, "phase": str(name),
            "mode": "derived" if derived else "blanket",
            "grace_s": read["grace_s"] + sum(term["grace_s"] for term in terms),
            "read": read_term, "compute": terms}


def compute_grace_basis(stamps: Sequence[Mapping], *, ceilings: Mapping,
                        link: Mapping | None) -> dict:
    """The row's one entry carrying the bound, the floor and every ceiling
    its compute stamps name, so no stamp repeats them."""
    named = sorted({term["ceiling"] for stamp in stamps
                    for term in stamp.get("compute", ())
                    if term.get("ceiling") is not None})
    floor = None if link is None else link.get("floor")
    return {"schema": COMPUTE_GRACE_BASIS_SCHEMA, "bound": COMPUTE_PHASE_BOUND,
            "load_bound": LOAD_PHASE_BOUND, "readers_scope": LINK_READERS_SCOPE,
            "floor": None if floor is None else dict(floor),
            "ceilings": {kind: ceilings[kind] for kind in named}}


def _row_compute_context(record: Mapping, *, spec: Mapping,
                         consumer_tags: Sequence[str],
                         annotations: Mapping,
                         emits_handoff: bool = False) -> dict:
    """The fields a compute ceiling's scope is checked against, for one row.

    It refuses nothing: a spec regime, spill bound or slice this cannot read
    leaves its fields out, the terms that need them take the blanket, and
    the row's own checks in :func:`quantum_argv` refuse it as before.
    ``emits_handoff`` says the row is a band-serial producer, whose tail
    writes the handoff, so a ``tail`` measurement can scope it (PQ #1190).
    """
    from prismaquant.joint_adjoint_slices import (
        AdjointSliceRefused, chain_regime_of)
    from prismaquant.joint_replay_regime import (
        ReplayRegimeRefused, normalize_replay_regime,
        replay_regime_from_environment)

    context: dict = {"consumer_tags": [str(tag) for tag in consumer_tags],
                     "n_probes": annotations.get("n_probes"),
                     "emits_handoff": bool(emits_handoff)}
    try:
        regime = normalize_replay_regime(
            replay_regime_from_environment(spec.get("env") or {}))
    except ReplayRegimeRefused:
        regime = {}
    context.update({f"replay_regime.{key}": value
                    for key, value in regime.items()})
    try:
        spill = _sealed_spill_bound(record)
    except DispatchRefused:
        spill = None
    if spill is not None:
        context.update({f"spill.{key}": spill.get(key)
                        for key in ("capture_batch", "element_dtype", "block")})
        context.update({f"spill.geometry.{key}": value for key, value
                        in (spill.get("geometry") or {}).items()})
    try:
        run_identity = _read_bound_slice(record).get("run_identity")
        chain = (chain_regime_of(dict(run_identity))
                 if isinstance(run_identity, Mapping) else {})
    except (DispatchRefused, AdjointSliceRefused, ValueError, AttributeError):
        chain = {}
    context.update({f"chain_regime.{key}": value
                    for key, value in chain.items()})
    return context


def _manifest_phase_facts(path: Path, *, quantum_id) -> tuple[dict, dict]:
    """Each read-plan phase's bytes and entries, and the manifest annotations.

    The caller has verified the wire against the row's digest
    (:func:`_executable_manifest_digest`), or published it in this dispatch
    (:func:`bind_consumer_handoff`); this reads the plan only.
    """
    try:
        wire = Path(path).read_bytes()
    except OSError as exc:
        raise DispatchRefused(
            f"quantum {quantum_id!r} executable manifest unreadable at "
            f"{path}: {exc}") from exc
    return _read_plan_phase_facts(wire, where=f"quantum {quantum_id!r} {path}")


def _read_plan_phase_facts(wire: bytes, *, where: str) -> tuple[dict, dict]:
    """``({phase: {"bytes", "entries", "entry_bytes"}}, annotations)``.

    ``entries`` is the phase's entry count and ``entry_bytes`` the one size
    its entries share (``None`` when they differ).
    """
    try:
        body = json.loads(gzip.decompress(wire))
        phases = body["read_plan"]["phases"]
    except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
        raise DispatchRefused(f"{where}: no readable read plan: {exc}") from exc
    entries = body.get("entries")
    entries = entries if isinstance(entries, list) else []
    facts = {}
    for phase in phases:
        if not isinstance(phase, Mapping) or "name" not in phase:
            continue
        indices = phase.get("entry_indices")
        sizes = set()
        if isinstance(indices, list):
            for index in indices:
                entry = (entries[index] if type(index) is int
                         and 0 <= index < len(entries) else None)
                sizes.add(entry.get("bytes") if isinstance(entry, Mapping)
                          else None)
        facts[phase["name"]] = {
            "bytes": phase.get("bytes"),
            "entries": len(indices) if isinstance(indices, list) else None,
            "entry_bytes": (next(iter(sizes)) if len(sizes) == 1
                            and type(next(iter(sizes))) is int else None)}
    annotations = body.get("annotations")
    return facts, dict(annotations) if isinstance(annotations, Mapping) else {}


def _read_plan_phase_bytes(wire: bytes, *, where: str) -> dict:
    try:
        phases = json.loads(gzip.decompress(wire))["read_plan"]["phases"]
    except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
        raise DispatchRefused(f"{where}: no readable read plan: {exc}") from exc
    return {phase["name"]: phase.get("bytes") for phase in phases
            if isinstance(phase, Mapping) and "name" in phase}


def progress_grace_of(argv: Sequence[str]) -> list | None:
    """The grace stamps (load and compute phases) a row's payload carries, if any."""
    if PROGRESS_GRACE_FLAG not in argv:
        return None
    return json.loads(argv[list(argv).index(PROGRESS_GRACE_FLAG) + 1])


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


def _require_replay_regime(spec: dict, *, emits_handoff: bool = False,
                           chain_batch_size: int | None = None) -> None:
    """Validate the Stage B replay regime the sealed spec declares (#994).

    One spec wraps every quantum of a dispatch, so the regime is uniform by
    construction. It changes the statistics arithmetic, and it replays only
    from the spill, so the spec must declare the spill beside it. A
    band-serial producer (#996) captures at ``chain_batch_size``, the batch
    size its slice's Stage A chain regime rolls at (#997), so the plane it
    hands off is the one its consumer's chain rebuild ends on; any other
    capture batch refuses (``joint_replay_regime.handoff_regime_refusal``).
    """
    from prismaquant.joint_replay_regime import (
        handoff_regime_refusal, replay_regime_from_environment)

    env = spec.get("env", {})
    regime = replay_regime_from_environment(env)
    if regime is not None and not stage_b_spill_environment(spec, env):
        raise RuntimeError("a non-default Stage B replay regime replays from the "
                           "spill; declare the spill in the same spec")
    if emits_handoff:
        if chain_batch_size is None:
            raise RuntimeError("a band-serial producer row names the chain batch "
                               "size its slice rolls at")
        refusal = handoff_regime_refusal(regime, chain_batch_size=chain_batch_size)
        if refusal:
            raise RuntimeError(refusal)


def _spec_capture_batch(spec_path: Path) -> int:
    """The capture batch the sealed spec's replay regime launches (default 1)."""
    from prismaquant.joint_replay_regime import (
        ReplayRegimeRefused, normalize_replay_regime, replay_regime_from_environment)

    try:
        spec = json.loads(Path(spec_path).read_text())
    except (OSError, ValueError) as exc:
        raise DispatchRefused(f"campaign spec {spec_path}: {exc}") from exc
    try:
        return normalize_replay_regime(
            replay_regime_from_environment(spec.get("env") or {}))["capture_batch"]
    except ReplayRegimeRefused as exc:
        raise DispatchRefused(f"campaign spec {spec_path}: {exc}") from exc


def _spec_kda_capture_kernel(spec_path: Path) -> str | None:
    """The mode the sealed spec's consumers bind a handoff in (PQ #1214, #1252).

    One dispatch seals one spec for every row, so a band it publishes runs in
    one mode. A consumer binds a handoff only in the mode of the launch it
    is published into: the kernel the spec names, ``None`` for ``fallback``,
    or ``FOLLOW_PRODUCER`` when the spec leaves the setting unset, since the
    launch then takes its producer's mode.
    """
    from prismaquant.glm_kda_capture_kernel import (
        KdaCaptureKernelRefused, consumer_handoff_mode, kda_capture_kernel_setting)

    try:
        spec = json.loads(Path(spec_path).read_text())
    except (OSError, ValueError) as exc:
        raise DispatchRefused(f"campaign spec {spec_path}: {exc}") from exc
    try:
        return consumer_handoff_mode(kda_capture_kernel_setting(spec.get("env") or {}))
    except KdaCaptureKernelRefused as exc:
        raise DispatchRefused(f"campaign spec {spec_path}: {exc}") from exc


def _slice_chain_batch_size(record: Mapping) -> int:
    """The batch size the record's Stage A slice rolls its chain at (#997)."""
    from prismaquant.joint_adjoint_slices import ChainRegimeRefused, chain_regime_of

    adjoint_slice = _read_bound_slice(record)
    try:
        return chain_regime_of(dict(adjoint_slice["run_identity"]))["batch_size"]
    except (ChainRegimeRefused, KeyError, TypeError) as exc:
        raise DispatchRefused(
            f"quantum {record.get('quantum_id')!r} stage-A slice chain regime: "
            f"{exc}") from exc


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


#: The spec field that admits a container cache pinned to the overlay, with
#: the reason (PQ #1129). A spec with no pin may not name one.
OVERLAY_CACHE_REASON_FIELD = "overlay_cache_reason"
#: The stamp the dispatcher seals beside that reason: the pins it admits.
#: Derived from the spec, never declared by one.
OVERLAY_CACHE_ADMISSION_FIELD = "overlay_cache_admission"


def _admit_overlay_caches(spec: dict, scratch: dict) -> None:
    """Refuse a spec that pins a container cache to the overlay (PQ #1129).

    When the row declares bounded local scratch, the launcher binds every
    cache the spec leaves unset under the scratch root
    (``container_cache_environment``). A value set in the spec wins over
    that default. A value on ``/tmp``, ``/var/tmp`` or a path no writable
    mount covers writes to the container overlay, which is unbounded and
    invisible to PrismaBuild, with or without declared scratch.

    Such a pin refuses, naming each pin and :data:`OVERLAY_CACHE_REASON_FIELD`,
    unless the spec names why in that field. An admitted spec (``spec`` is
    the parse being sealed) gains :data:`OVERLAY_CACHE_ADMISSION_FIELD`, which
    repeats the reason beside the pins it admits, so the sealed request
    carries the waiver. PQ #1072 only warned here; the warning fired on every
    Stage B prepare of R13 and nothing acted on it.
    """
    if OVERLAY_CACHE_ADMISSION_FIELD in spec:
        raise RuntimeError(
            f"spec field {OVERLAY_CACHE_ADMISSION_FIELD} is derived by the "
            f"dispatcher from the spec's pins and {OVERLAY_CACHE_REASON_FIELD}, "
            "not declared by a spec")
    _defaults, pinned = container_cache_environment(spec, scratch)
    declared = OVERLAY_CACHE_REASON_FIELD in spec
    reason = spec.get(OVERLAY_CACHE_REASON_FIELD)
    if not pinned:
        if declared:
            raise RuntimeError(
                f"the campaign spec names an {OVERLAY_CACHE_REASON_FIELD}, but "
                "pins no container cache to the overlay; drop the reason")
        return
    env = spec.get("env", {})
    if not isinstance(reason, str) or not reason.strip():
        named = ", ".join(f"{name}={env[name]}" for name in pinned)
        hint = ("unset them to bind them under the declared scratch root"
                if scratch else
                "declare a bounded local scratch root to bind them there")
        raise RuntimeError(
            f"the campaign spec pins container caches to the overlay: {named}; "
            f"these writes are unbounded and invisible to PrismaBuild. {hint}, "
            f"or name why in the spec's {OVERLAY_CACHE_REASON_FIELD} "
            "(a non-empty string, sealed with the pins it admits)")
    spec[OVERLAY_CACHE_ADMISSION_FIELD] = {
        "pinned": {name: env[name] for name in pinned}, "reason": reason}


def _container_wrap(spec_path: Path, payload: list[str], *,
                    progress: Sequence[tuple[str, int]],
                    resource_policy=None,
                    spool_max_bytes: int | None = None,
                    spill_bound: Mapping | None = None,
                    spec: dict | None = None,
                    handoff_chain_batch_size: int | None = None,
                    ) -> tuple[list[str], str | None]:
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

    ``spec`` is the caller's one parse of ``spec_path``, when the caller had
    to read it first (the quantum row derives its load grace from the
    spec's staged wait); the wrapper then seals that parse and reads nothing.
    """
    spec = (json.loads(Path(spec_path).read_text()) if spec is None
            else json.loads(json.dumps(spec)))
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
        _admit_overlay_caches(spec, scratch)
        _require_replay_regime(spec, emits_handoff="--emit-adjoint-handoff" in payload,
                               chain_batch_size=handoff_chain_batch_size)
        # The bf16 reduction flag is sealed in the same spec, so it is
        # uniform across every quantum of the dispatch (PQ #1028).
        from prismaquant.matmul_arithmetic import bf16_reduction_from_environment
        bf16_reduction_from_environment(spec.get("env", {}))
        # So is the KDA capture kernel (PQ #1199): a kernel name, fallback,
        # or unset for the default each quantum resolves (PQ #1252).
        from prismaquant.glm_kda_capture_kernel import kda_capture_kernel_setting
        kda_capture_kernel_setting(spec.get("env", {}))
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

#: A checkpoint's shard index, by model directory. A band publishes many rows
#: of one model; the dispatcher reads the index once.
_CHECKPOINT_INDEX: dict[str, dict] = {}
#: Shard headers already read, by shard path. Header bytes only.
_SHARD_HEADERS: dict[str, dict] = {}


def _checkpoint_weight_shapes(model: Path, names: Sequence[str], *,
                              where: str) -> dict[str, tuple[int, ...]]:
    """Each target's weight shape, from the checkpoint's safetensors headers.

    ``names`` are roster qnames; a target's weight is the tensor
    ``<qname>.weight``. Reads the shard index and the headers of the shards
    that hold the targets, never a payload byte. A target the headers do not
    hold refuses in both modes: a shape the dispatcher cannot read is not a
    qualified shape, and it is not a seal either (PQ #1175).
    """
    from prismaquant.source_read_plan import read_safetensors_header

    root = str(model)
    index_path = Path(root) / "model.safetensors.index.json"
    if root not in _CHECKPOINT_INDEX:
        try:
            if index_path.is_file():
                weight_map = json.loads(index_path.read_text())["weight_map"]
            else:
                weight_map = None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DispatchRefused(f"{where}: checkpoint index {index_path} is "
                                  f"unreadable: {exc}") from exc
        _CHECKPOINT_INDEX[root] = {"weight_map": weight_map}
    weight_map = _CHECKPOINT_INDEX[root]["weight_map"]
    by_shard: dict[str, list[str]] = {}
    missing = []
    for name in names:
        tensor = f"{name}.weight"
        shard = ("model.safetensors" if weight_map is None
                 else weight_map.get(tensor))
        if not isinstance(shard, str):
            missing.append(name)
            continue
        by_shard.setdefault(shard, []).append(name)
    shapes: dict[str, tuple[int, ...]] = {}
    for shard, members in sorted(by_shard.items()):
        path = str(Path(root) / shard)
        if path not in _SHARD_HEADERS:
            try:
                _SHARD_HEADERS[path] = read_safetensors_header(path)[0]
            except (OSError, ValueError) as exc:
                raise DispatchRefused(f"{where}: checkpoint shard header {path} "
                                      f"is unreadable: {exc}") from exc
        header = _SHARD_HEADERS[path]
        for name in members:
            row = header.get(f"{name}.weight")
            shape = row.get("shape") if isinstance(row, dict) else None
            if not isinstance(shape, list):
                missing.append(name)
                continue
            shapes[name] = tuple(int(size) for size in shape)
    if missing:
        raise DispatchRefused(
            f"{where}: the checkpoint headers at {root} hold no weight for "
            f"{len(missing)} target(s), so their projection shapes cannot be "
            f"checked against the kernel's qualification: {sorted(missing)[:4]}")
    return shapes


def check_projection_shapes(record: Mapping, *, plan: Mapping,
                            prepared_input: Mapping | None) -> list:
    """Compare the row's projection shapes with its kernel's qualification.

    PQ #1175. The plan's ``execution.projection_backend`` selects the
    reduction kernel. The reference ``torch`` backend accepts every shape,
    and this reads nothing for it. For ``fused_fp32_v1`` the row's shapes
    are the ``(out, in)`` weight shapes of the quantum's joint statistics
    targets (every ``product_sum`` operand has its target's weight shape,
    ``joint_aura.py``). The targets are the members of the record's sealed
    prepared-input windows, or, for a record without them, the prepared
    roster's units in the record's layer. Their shapes come from the
    safetensors headers of the plan's ``model``.

    Certified mode refuses with :class:`DispatchRefused` before anything is
    submitted. Dev mode prints one ``[DEV-MODE]`` line and the row runs
    those shapes on the reference arithmetic (PQ #1176). Returns the shapes
    that will run on the reference arithmetic.
    """
    from prismaquant.joint_projection_backend import (
        check_qualified_shapes, qualified_shapes)

    quantum_id = record.get("quantum_id")
    where = f"quantum {quantum_id!r}"
    config = (plan.get("execution") or {}).get("projection_backend")
    try:
        if qualified_shapes(config) is None:
            return []
    except ValueError as exc:
        raise DispatchRefused(f"{where}: {exc}") from exc
    if prepared_input is not None:
        names = [str(name) for window in prepared_input.get("windows", [])
                 for name, _fmt in window.get("members", [])]
    else:
        from prismaquant.joint_layer_quanta import qname_layer
        campaign = record.get("campaign") or {}
        try:
            prepared = json.loads(Path(campaign["prepared_path"]).read_bytes())
            roster = prepared["formats_by_qname"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DispatchRefused(f"{where}: prepared roster is unreadable: "
                                  f"{exc}") from exc
        names = [name for name in roster if qname_layer(name) == record.get("layer")]
    names = sorted(set(names))
    if not names:
        raise DispatchRefused(f"{where}: no joint statistics target to check "
                              "against the projection kernel's qualification")
    model = plan.get("model")
    if not isinstance(model, str) or not model:
        raise DispatchRefused(f"{where}: the plan names no source model")
    shapes = _checkpoint_weight_shapes(Path(model), names, where=where)
    malformed = sorted(name for name, shape in shapes.items() if len(shape) != 2)
    if malformed:
        raise DispatchRefused(f"{where}: target weights are not matrices: "
                              f"{malformed[:4]}")
    return check_qualified_shapes(config, shapes.values(), where=where,
                                  refusal=DispatchRefused)


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
                 band: Mapping | None = None,
                 link: Mapping | None = None,
                 ceilings: Mapping | None = None) -> list[str]:
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

    ``link`` (:func:`link_readers`) is how many of this dispatch's rows can
    share the link, and the floor measured at that count. It sets the
    checkpoint-load or handoff-load grace (:func:`load_phase_grace`); the
    stamps ride the payload as ``--progress-grace-derivation``. Without it
    a load phase takes the blanket grace and says so.

    ``ceilings`` (:func:`compute_ceilings`; default the built-in
    :data:`COMPUTE_UNIT_CEILINGS`) sets each compute phase's grace, the read
    term plus the pass's compute (:func:`compute_phase_grace`, PQ #1165).
    Its stamps ride the same flag, with one basis entry
    (:func:`compute_grace_basis`) that carries the bound and the ceilings.
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
    prepared_input = None
    executable = record.get("executable_readset")
    # At most one parse of the campaign spec, made when a load phase first
    # needs its staged wait; _container_wrap then seals that same parse.
    spec_parse: dict = {}
    grace_stamps: list[dict] = []
    compute_stamps: list[dict] = []
    ceiling_docs = compute_ceilings() if ceilings is None else dict(ceilings)

    def parsed_spec():
        if "spec" not in spec_parse:
            spec_parse["spec"] = json.loads(Path(SPEC_PATH).read_text())
        return spec_parse["spec"]

    def staged_wait():
        from prismaquant.residency_shard_reader import staged_range_wait_from_env

        try:
            return staged_range_wait_from_env(parsed_spec().get("env") or {})
        except ValueError as exc:
            raise DispatchRefused(f"campaign spec: {exc}") from exc

    def load_grace(name, phase_bytes):
        stamp = load_phase_grace(name, phase_bytes=phase_bytes,
                                 staged_wait_s=staged_wait(), link=link)
        grace_stamps.append(stamp)
        return stamp["grace_s"]

    row_context: dict = {}
    # The quantum's tail after its last window runs on the last declared
    # phase's clock (PQ #1187), so that phase's grace carries it (PQ #1190).
    declared_phases = (handoff if handoff is not None
                       else executable or {}).get("phases")
    tail_phase = (declared_phases[-1] if isinstance(declared_phases, list)
                  and declared_phases else None)

    def compute_grace(name, fact, annotations):
        from prismaquant.joint_layer_quanta import normalize_replay_mode

        if "context" not in row_context:
            row_context["context"] = _row_compute_context(
                record, spec=parsed_spec(), consumer_tags=consumer_tags,
                annotations=annotations,
                emits_handoff=emit_template is not None)
        context = row_context["context"]
        work = compute_phase_work(
            name, replay_mode=normalize_replay_mode(
                (executable or {}).get("replay_mode")),
            entries=phase_work_entries(name, fact, annotations),
            n_probes=context.get("n_probes"),
            capture_batch=context.get("replay_regime.capture_batch"),
            runs_tail=name == tail_phase)
        if work is None:
            return None
        stamp = compute_phase_grace(
            name, work=work, phase_bytes=fact.get("bytes"),
            staged_wait_s=staged_wait(), link=link,
            context={**context, "entry_bytes": fact.get("entry_bytes")},
            ceilings=ceiling_docs)
        compute_stamps.append(stamp)
        return stamp["grace_s"]

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
        _, prepared_input = _executable_prepared_input(record, output_root=output_root)
        if handoff is not None:
            # The sealed chain manifest passes its gate here; the row stages
            # the readset derived from it and this handoff.
            _executable_row_parts(record, output_root=output_root,
                                  head_grace_s=head_grace_s)
            manifest = Path(handoff["manifest_path"])
            staged_sha256 = handoff["manifest_sha256"]
            phase_bytes = handoff.get("phase_bytes") or {}
            # A dry run's readset is unpublished: its bytes ride the handoff.
            wire = handoff.get("manifest_wire")
            try:
                facts, annotations = (
                    _manifest_phase_facts(manifest, quantum_id=quantum_id)
                    if wire is None else _read_plan_phase_facts(
                        wire, where=f"quantum {quantum_id!r} {manifest}"))
            except DispatchRefused:
                # Grace derivation only: the terms take the blanket, and
                # the row's staging checks stay what they were.
                facts, annotations = {}, {}
            progress = [("head", head_grace_s)]
            for name in handoff["phases"]:
                if name == "head":
                    continue
                if name == HANDOFF_LOAD_PHASE:
                    grace = load_grace(name, phase_bytes.get(name))
                else:
                    grace = compute_grace(name, facts.get(name) or {},
                                          annotations)
                progress.append((name, CHUNK_PROGRESS_GRACE_S
                                 if grace is None else grace))
        else:
            manifest, staged_sha256, progress = _executable_row_parts(
                record, output_root=output_root, head_grace_s=head_grace_s,
                load_grace=load_grace, compute_grace=compute_grace)
        if compute_stamps:
            grace_stamps.extend(compute_stamps)
            grace_stamps.append(compute_grace_basis(
                compute_stamps, ceilings=ceiling_docs, link=link))
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
    # PQ #1175: before anything is submitted, the row's projection shapes
    # against its kernel's qualification. Certified mode refuses here.
    check_projection_shapes(record, plan=plan, prepared_input=prepared_input)
    resource_bound = plan.get("stage_b_resource_policy") is not None
    if not resource_bound:
        plan_sha256 = _argv_file_sha256(campaign, "plan", raw=plan_raw,
                                        where=f"quantum {quantum_id!r}")
    prepared_sha256 = _argv_file_sha256(campaign, "prepared",
                                        where=f"quantum {quantum_id!r}")
    if resource_bound:
        # The record's plan digest is a run seal (PQ #1147): dev mode prints a
        # re-declared plan and dispatches under the plan on disk, by the
        # digest of the bytes just read. Certified mode refuses unless the
        # two are equal, so its argv is unchanged.
        from prismaquant.dev_mode import seal_check
        plan_sha256 = hashlib.sha256(plan_raw).hexdigest()
        seal_check("resource-bound plan", campaign["plan_sha256"], plan_sha256,
                   where=f"quantum {quantum_id!r}",
                   refusal=lambda: DispatchRefused(
                       "resource-bound plan differs from its quantum seal"))
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
    handoff_chain_batch_size = None
    if emit_template is not None:
        band_payload.append("--emit-adjoint-handoff")
        handoff_chain_batch_size = _slice_chain_batch_size(record)
    if grace_stamps:
        band_payload += [PROGRESS_GRACE_FLAG,
                         json.dumps(grace_stamps, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False)]
    wrapped, container_image = _container_wrap(SPEC_PATH, [
        "python3", "-m", "prismaquant.joint_cost_quantum",
        "--quantum", str(record_path),
        "--quantum-sha256", record_sha256,
        "--plan", str(campaign["plan_path"]),
        "--plan-sha256", plan_sha256,
        "--prepared", str(campaign["prepared_path"]),
        "--prepared-sha256", prepared_sha256,
        "--adjoint-slice", str(slice_path),
        "--adjoint-slice-sha256", slice_sha256,
        "--data-manifest-sha256", staged_sha256,
        "--allowed-tiers", STAGED_ALLOWED_TIERS,
        "--resume",
        "--output-root", str(output_root), *band_payload], progress=progress,
        resource_policy=resource_policy, spill_bound=_sealed_spill_bound(record),
        spec=spec_parse.get("spec"),
        handoff_chain_batch_size=handoff_chain_batch_size)
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
                 binding: dict | None = None,
                 batch_range: Sequence[int] | None = None) -> list[str]:
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

    ``batch_range`` (optional, PQ #738) is a chain split quantum's
    ``(start, stop)``: the row's spool window is two planes of those
    batches (:func:`stage_a_spool_window_bytes`), not of the whole run.

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
        "--plan-sha256", _argv_file_sha256(campaign, "plan", where="stage-A"),
        "--prepared", str(campaign["prepared_path"]),
        "--prepared-sha256", _argv_file_sha256(campaign, "prepared", where="stage-A"),
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
        spool_max_bytes=stage_a_spool_window_bytes(campaign, batch_range))
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

#: The export-rate family every Stage B handoff template declares (PQ #1225,
#: #1254). PrismaBuild learns a producer's export slots per template (#999),
#: and each row's handoff template is new, so no row was ever measured and
#: each ran its exports one at a time. The template's ``export_rate_family``
#: (PrismaBuild #1126) keys the learned rate by this one string instead, so
#: each row starts from what the earlier rows' exports measured.
HANDOFF_EXPORT_RATE_FAMILY = "pq-stageb-handoff"


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


def handoff_template(record: Mapping, *, plan: Mapping,
                     adjoint_slice: Mapping, tier: str, output_root: Path,
                     template_id: str | None = None) -> tuple[Path, bytes]:
    """A producer row's handoff produced-output template: ``(path, bytes)``.

    Derived from the numbers the producer's emitter binds with: the plan's
    ``execution.boundary_storage`` normalized onto the producer's handoff
    directory (artifact maximum and prefetch group), and the largest
    checkpoint-plane tensor in the producer's slice. ``tier`` is the stage
    tier the declaration permits, a fleet fact the submitter names. The file
    is named by its content digest, so a changed tier or plan writes a new
    template and never rewrites one a submitted row already declared.

    The handoff directory and the file derive from one root, ``output_root``
    (PQ #1200): the handoff directory is inside the quantum's output space
    under it, ``{output_root}/layer-quanta/{quantum id}``, the space the
    quantum's identity gate requires its record to name when it runs under
    that ``--output-root``; the file goes under
    ``{output_root}/layer-quanta/band-serial``. Writes nothing:
    :func:`handoff_template_path` publishes it.

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
    from prismaquant.joint_layer_quanta import quantum_id as layer_quantum_id
    from prismaquant.joint_quantum_handoff import handoff_root
    from prismaquant.stage_a_produced_output import build_boundary_template

    quantum_id = record["quantum_id"]
    storage = plan.get("execution", {}).get("boundary_storage")
    if not isinstance(storage, dict):
        raise DispatchRefused(
            f"quantum {quantum_id!r}: the plan seals no boundary storage for a "
            "handoff")
    try:
        space = (Path(output_root) / "layer-quanta"
                 / layer_quantum_id(record["layer"]))
        policy = normalize_boundary_storage({
            **storage, "directory": str(handoff_root(space))})
        tensors = [int(entry["tensor_bytes"]) for entry in
                   adjoint_slice["checkpoint"]["activation_entries"]]

        def build(name: str) -> dict:
            return build_boundary_template(
                output_prefix=policy["directory"], tier=str(tier),
                artifact_max_bytes=int(policy["max_artifact_bytes"]),
                group_size=int(policy["prefetch_batches"]),
                max_entry_tensor_bytes=max(tensors), template_id=name,
                write_only=True, export_rate_family=HANDOFF_EXPORT_RATE_FAMILY)

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
    return path, payload


def handoff_template_path(record: Mapping, *, plan: Mapping,
                          adjoint_slice: Mapping, tier: str,
                          output_root: Path,
                          template_id: str | None = None) -> Path:
    """Write and return a producer row's handoff template (:func:`handoff_template`)."""
    path, payload = handoff_template(
        record, plan=plan, adjoint_slice=adjoint_slice, tier=tier,
        output_root=output_root, template_id=template_id)
    return _publish_control_bytes(path, payload, what="handoff template")


def bind_consumer_handoff(record: Mapping, *, path: str, sha256: str,
                          producer: str, output_root: Path,
                          kda_capture_kernel: str | None,
                          publish: bool = True) -> dict:
    """Bind a published handoff to its consumer row, or refuse.

    Runs the consumer's own checks (:func:`load_quantum_handoff`) and
    derives, then writes, the band-serial readset the row stages. A handoff
    the consumer would refuse is refused here: publishing it would only exit
    3 on a GPU box. ``kda_capture_kernel`` is the mode the consumer launches
    in (:func:`_spec_kda_capture_kernel`): a handoff produced in the other
    mode refuses (PQ #1214).

    ``publish=False`` (a dry run, PQ #1200) derives the same readset and
    writes nothing: the result names the path it would be published at and
    carries its bytes as ``manifest_wire``, which the row's grace derivation
    and the source-coverage check read instead of the file.
    """
    from prismaquant.joint_quantum_handoff import (
        QuantumHandoffRefused, band_serial_manifest_bytes, load_quantum_handoff)

    quantum_id = record["quantum_id"]
    adjoint_slice = _read_bound_slice(record)
    try:
        handoff = load_quantum_handoff(path, sha256, record=record,
                                       adjoint_slice=adjoint_slice,
                                       kda_capture_kernel=kda_capture_kernel)
        wire = band_serial_manifest_bytes(record, handoff,
                                          adjoint_slice["checkpoint"],
                                          output_root=output_root)
    except QuantumHandoffRefused as exc:
        raise DispatchRefused(
            f"quantum {quantum_id!r} refuses the handoff {producer!r} "
            f"published: {exc}") from exc
    manifest_path = (
        _band_serial_root(output_root)
        / f"{quantum_id}.{handoff['handoff_sha256'][:16]}.executable.json.gz")
    if publish:
        _publish_control_bytes(manifest_path, wire, what="band-serial readset")
    phases = [phase["name"] for phase in
              json.loads(gzip.decompress(wire))["read_plan"]["phases"]]
    return {"path": str(path), "sha256": str(sha256), "producer": str(producer),
            "handoff_sha256": handoff["handoff_sha256"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha_bytes(wire), "phases": phases,
            # The load grace reads the handoff-load phase's bytes from here.
            "phase_bytes": _read_plan_phase_bytes(
                wire, where=f"quantum {quantum_id!r} band-serial readset"),
            **({} if publish else {"manifest_wire": wire})}


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
                    output_root: Path, capture_batch: int = 1,
                    publish: bool = True,
                    ) -> tuple[dict | None, str | None]:
    """The band-serial role of a row never submitted before.

    Returns ``(band, None)`` to publish or ``(None, reason)`` while the row
    waits for its producer. A consumer takes its producer's handoff; a
    producer emits one only for a successor never yet submitted, since a
    submitted row keeps the mode it was submitted in. ``capture_batch`` is
    the spec's launch regime's (:func:`_spec_capture_batch`): a band runs
    serial only when its slice's chain regime rolls at that batch size
    (PQ #994, #997), the batch the handed-off plane is captured at.

    ``publish=False`` (a dry run, PQ #1200) derives the template and the
    readset and writes neither; the band carries the template's digest and
    body so the dry run can print them.
    """
    from prismaquant.joint_quantum_handoff import handoff_chain_regime_refusal

    quantum_id = record["quantum_id"]
    role = roles[quantum_id]
    adjoint_slice = _read_bound_slice(record)
    refusal = handoff_chain_regime_refusal(adjoint_slice.get("run_identity"),
                                           capture_batch=capture_batch)
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
                producer=source, output_root=output_root,
                kda_capture_kernel=_spec_kda_capture_kernel(SPEC_PATH),
                publish=publish)
    successor = role["hands_to"]
    if successor is not None and successor not in last_submission:
        if tier is None:
            raise DispatchRefused("--band-serial needs --handoff-tier")
        plan = _load_json(Path(record["campaign"]["plan_path"]), where="plan")
        path, payload = handoff_template(
            record, plan=plan, adjoint_slice=adjoint_slice, tier=tier,
            output_root=output_root)
        if publish:
            _publish_control_bytes(path, payload, what="handoff template")
        band["emit_template"] = path
        band["emit_template_sha256"] = _sha_bytes(payload)
        band["emit_template_document"] = json.loads(payload)
    return band, None


def recorded_band_role(record: Mapping, event: Mapping, *,
                       output_root: Path, publish: bool = True) -> dict:
    """Rebuild the role a row was submitted with, from its state event.

    A resubmission must be the same sealed action: a row republished in
    another mode would be a second action for the same output space. So a
    row keeps its recorded handoff and template whether or not this run
    passes ``--band-serial``; events written before PQ #996 are chain rows.
    ``publish=False`` (a dry run) re-derives a consumer's readset without
    writing it (:func:`bind_consumer_handoff`).
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
            producer=source["producer"], output_root=output_root,
            kda_capture_kernel=_spec_kda_capture_kernel(SPEC_PATH),
            publish=publish)
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
        """Whether PB has finished this action's work, as ``pbwait`` says.

        ``pbwait`` prints a row table and returns its verdict as the exit
        code, 0 when the work is done; it prints no JSON (PQ #1197). Both
        must say done: the verdict, and this key's own row, ``executed`` or
        ``cache_hit`` (a memoized result is the same result). The caller
        then reads the action's outputs and checks them itself.
        """
        from prismaquant.pbwait_table import DONE_STATUSES, parse_pbwait_table

        proc = subprocess.run(
            [sys.executable, str(PBWAIT), "--wait-s", "0", action_key],
            capture_output=True, text=True)
        if proc.returncode != 0:
            return False
        # The table names a key by its 12-character prefix.
        rows = [row for row in parse_pbwait_table(proc.stdout)
                if len(row.get("key", "")) >= 12 and action_key.startswith(row["key"])]
        return len(rows) == 1 and rows[0].get("status") in DONE_STATUSES


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
        # installs only its own layer. A dry run's readset is unpublished
        # and checked from its bytes (PQ #1200).
        return {"record": record, "manifest_path": str(handoff["manifest_path"]),
                "manifest_sha256": handoff["manifest_sha256"],
                "order": (record["layer"],),
                **({"manifest_wire": handoff["manifest_wire"]}
                   if "manifest_wire" in handoff else {})}
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
    parser.add_argument("--execution-plan", type=Path, default=None,
                        help="PQ #1191: a re-declared plan (for example a "
                             "Stage B resource re-declare) at its own path. "
                             "Every row runs under it (--plan PATH and its "
                             "digest) while the records keep naming the "
                             "sealed plan, which is never rewritten. Dev mode "
                             "stamps a digest that differs from the sealed "
                             "one; certified mode refuses it")
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
    parser.add_argument("--quantum", action="append", default=[],
                        metavar="QUANTUM_ID",
                        help="publish only this quantum's row (repeatable). "
                             "The link-reader count is then the rows named "
                             "and publishable, not every row of the records "
                             "directory. An id no record carries refuses")
    parser.add_argument("--link-readers", type=int, default=None,
                        help="how many of this dispatch's rows can read the "
                             "storage link at once. Default: the number of "
                             "quantum rows this dispatch publishes. The "
                             "built-in load-phase floor applies to 1 reader "
                             "only")
    parser.add_argument("--checkpoint-load-floor", type=Path, default=None,
                        help="a load-phase floor document "
                             f"({LOAD_PHASE_FLOOR_SCHEMA}) measured at this "
                             "dispatch's link-reader count; without one, any "
                             "count other than 1 takes the blanket "
                             f"{HEAD_PROGRESS_GRACE_S} s load grace")
    parser.add_argument("--compute-ceiling", type=Path, action="append",
                        default=[], metavar="FILE",
                        help="a compute ceiling document "
                             f"({COMPUTE_CEILING_SCHEMA}) that sets the "
                             "ceiling of its kind, replacing any built-in "
                             "one (repeatable). A compute term outside every "
                             "ceiling's scope takes the blanket "
                             f"{HEAD_PROGRESS_GRACE_S} s")
    parser.add_argument("--state", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        return _dispatch(args, _gateway=_gateway, _coverage=_coverage,
                         report=sys.stdout)
    # A dry run's stdout is its plan, one JSON document (PQ #1087). Every
    # line printed while deriving it -- a helper's progress, a [DEV-MODE]
    # stamp -- is a log, and goes to stderr.
    report = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        return _dispatch(args, _gateway=_gateway, _coverage=_coverage,
                         report=report)


def _dispatch(args, *, _gateway: Gateway | None, _coverage, report) -> int:
    """Run ``main`` on parsed ``args``; ``report`` receives the dry-run plan."""
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
        execution_stamp = None
        if args.execution_plan is not None:
            records, execution_stamp = with_execution_plan(
                records, args.execution_plan)
        wanted = set(args.quantum)
        unknown = sorted(wanted - {record["quantum_id"] for _, record in records})
        if unknown:
            raise DispatchRefused(
                f"--quantum names {unknown}, which no record in {records_dir} "
                "carries")
        floor_document = None
        if args.checkpoint_load_floor is not None:
            try:
                raw = args.checkpoint_load_floor.read_bytes()
                floor_document = json.loads(raw)
            except (OSError, ValueError) as exc:
                raise DispatchRefused(
                    f"--checkpoint-load-floor {args.checkpoint_load_floor}: "
                    f"{exc}") from exc
            floor_document = {**load_phase_floor(floor_document),
                              "document": {"path": str(args.checkpoint_load_floor),
                                           "sha256": _sha_bytes(raw)}}
        ceiling_documents = []
        for path in args.compute_ceiling:
            try:
                raw = path.read_bytes()
                document = json.loads(raw)
            except (OSError, ValueError) as exc:
                raise DispatchRefused(f"--compute-ceiling {path}: {exc}") from exc
            ceiling_documents.append(
                {**compute_unit_ceiling(document),
                 "document": {"path": str(path), "sha256": _sha_bytes(raw)}})
        ceilings = compute_ceilings(ceiling_documents)
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
    band_capture_batch = 1
    if args.band_serial:
        try:
            band_capture_batch = _spec_capture_batch(SPEC_PATH)
        except DispatchRefused as exc:
            print(f"dispatch_joint_quanta: refused: {exc}", file=sys.stderr)
            return EXIT_PRECONDITION_REFUSED
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
            # Pass 1: the quantum rows this dispatch publishes. Their count
            # is the most of its own readers that can share the link.
            publishing = []
            for record_path, record in records:
                quantum_id = record["quantum_id"]
                if quantum_id not in publishable:
                    continue
                if wanted and quantum_id not in wanted:
                    continue
                key = submitted_keys.get(quantum_id)
                if key is not None and gateway.is_terminal_executed(key):
                    continue
                band = None
                # A dry run derives the band's control files and publishes
                # none of them (PQ #1200).
                if quantum_id in last_submission:
                    band = recorded_band_role(
                        record, last_submission[quantum_id],
                        output_root=output_root, publish=not args.dry_run)
                elif args.band_serial:
                    band, waiting = fresh_band_role(
                        record, roles=roles, by_id=by_id,
                        last_submission=last_submission,
                        submitted_keys=submitted_keys, gateway=gateway,
                        tier=args.handoff_tier, output_root=output_root,
                        capture_batch=band_capture_batch,
                        publish=not args.dry_run)
                    if waiting is not None:
                        band_pending.append({"quantum_id": quantum_id,
                                             "reason": waiting})
                        continue
                publishing.append((record_path, record, band))
            fresh_link = link_readers(rows=len(publishing),
                                      declared=args.link_readers,
                                      floor_document=floor_document)
            # Pass 2: the rows. A resubmitted row keeps the link count it was
            # first submitted with, as it keeps its band role: the same row
            # must stay the same sealed action whatever else this run holds.
            for record_path, record, band in publishing:
                quantum_id = record["quantum_id"]
                recorded = (last_submission.get(quantum_id) or {}).get("link")
                link = recorded if isinstance(recorded, dict) else fresh_link
                handoff = (band or {}).get("handoff")
                template = (band or {}).get("emit_template")
                if record.get("executable_readset") is not None:
                    coverage_rows.append(_coverage_row(
                        record, output_root=output_root, band=band))
                argv = quantum_argv(
                    record, record_path=record_path,
                    output_root=output_root, priority=priority,
                    head_grace_s=args.head_grace_s,
                    consumer_tags=tags, band=band, link=link,
                    ceilings=ceilings)
                rows.append({"kind": "quantum", "quantum_id": quantum_id,
                             "identity_sha256": record["identity_sha256"],
                             "manifest_sha256": (
                                 handoff["manifest_sha256"] if handoff
                                 else _row_manifest_sha256(record)),
                             "cotangent_source": _cotangent_source(band),
                             "handoff_template": (
                                 None if template is None else str(template)),
                             # What this run derived; a dry run prints it
                             # instead of publishing it (PQ #1200).
                             "handoff_template_sha256": (
                                 band or {}).get("emit_template_sha256"),
                             "handoff_template_document": (
                                 band or {}).get("emit_template_document"),
                             "band_serial_readset": (
                                 None if handoff is None else {
                                     "path": handoff["manifest_path"],
                                     "sha256": handoff["manifest_sha256"]}),
                             "link": link,
                             "progress_grace": progress_grace_of(argv),
                             "argv": argv})
        check_source_coverage(coverage_rows, coverage=_coverage)
    except DispatchRefused as exc:
        print(f"dispatch_joint_quanta: refused: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION_REFUSED

    if args.dry_run:
        by_id = {record["quantum_id"]: record for _, record in records}
        print(json.dumps({"consumer_tags": list(tags), "priority": priority,
                          "execution_plan": execution_stamp,
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
                                        "handoff_template": row["handoff_template"],
                                        "handoff_template_sha256": row[
                                            "handoff_template_sha256"],
                                        "handoff_template_document": row[
                                            "handoff_template_document"],
                                        "band_serial_readset": row[
                                            "band_serial_readset"],
                                        "link": row["link"],
                                        "progress_grace": row["progress_grace"]}
                                       if row["kind"] == "quantum" else {}),
                                    **({"slice_sha256": publishable[row["quantum_id"]],
                                        **{key: by_id[row["quantum_id"]]["adjoint"][key]
                                           for key in ("checkpoint_boundary", "chain_layers")}}
                                       if row["kind"] == "quantum" else {}),
                                    "argv": row["argv"]}
                                   for row in rows]},
                         indent=2, sort_keys=True), file=report)
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
                                               else None),
                                           "execution_plan": execution_stamp})
            else:
                _append_state(state_path,
                              {"event": "quantum-submitted",
                               "quantum_id": row["quantum_id"],
                               "identity_sha256": row["identity_sha256"],
                               "data_manifest_sha256": row.get("manifest_sha256"),
                               "cotangent_source": row["cotangent_source"],
                               "handoff_template": row["handoff_template"],
                               "link": row["link"],
                               "progress_grace": row["progress_grace"],
                               "execution_plan": execution_stamp,
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
