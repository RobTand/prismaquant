"""Stage A chain resume: a relaunch of the same run (RobTand/prismaquant#1001).

A Stage A run seals its tail checkpoint and then one checkpoint every
``stride`` layers on the way down. A run that dies partway through the
reverse chain has done everything above its lowest sealed checkpoint ``b``.
A chain resume relaunches that run from ``b``:

* It **is** the same run. The relaunch adopts the original run header and
  boundary session byte for byte, so every checkpoint it seals below ``b``
  carries the generation the checkpoints above ``b`` carry, and bands from
  both sides of the resume form one set (``band_set``).
* It is **self-contained**. At the tail checkpoint a fresh run writes a
  sealed chain-state file (``chain-state.json``): the run header, the bind
  identity, the arithmetic stamp and every forward boundary entry record.
  A resume reads that file under a pinned digest and never reconstructs
  them from anywhere else.
* It **refuses** in both modes when the chain's layout or data differs: the
  stride, the partition and layer counts, the chain regime (batch size, probe
  fusion), the calibration draw, the probes or the unit roster.
* Its **run seals** go through ``seal_check`` (PQ #1147): the arithmetic
  stamp, the plan, the preparation, the read manifest, the implementation,
  the campaign scope, the source model, the artifact budget, the boundary
  storage policy, the capsule binding and the generation status. Certified
  mode (``PRISMAQUANT_DEV_MODE=0``) refuses on each as before. Dev mode prints
  one ``[DEV-MODE]`` line per difference and continues with the stored chain.
  A relaunch may pass ``NOT_COMPUTED`` for a seal input it would derive only
  to compare it.

**Same implementation.** A resume under the implementation that sealed
checkpoint ``b`` is a plain bitwise continuation: the receipt, every
checkpoint and every band slice are the bytes the uninterrupted run writes.

**Different implementation.** Allowed in dev mode only. Since PQ #1147 dev
mode is the default and needs no declaration: ``seal_check`` prints both
implementations and the resume continues. An operator may still declare the
switch (``--resume-implementation-compatibility FROM:TO``); certified mode
(``PRISMAQUANT_DEV_MODE=0``) refuses the switch either way. The header keeps
the original ``implementation_sha256``. The switch -- both implementations,
the checkpoint where it happened and, when undeclared, ``declared: false`` --
is recorded outside the header digest: in the receipt and in every band sealed
below the switch, under ``resume_compatibility``. The precedent is the
forward-recovery capsule's ``implementation_compatibility``.

**What a resume removes.** The interrupted attempt's rolling cotangent
entries (``cotangent-*-at-*`` in the run's own generation directory) are
working entries no receipt names, and the resumed chain rewrites the same
names, so they are unlinked. Partial checkpoint directories below ``b`` are
renamed aside, never deleted. Both happen only after every check passed.
Forward boundary entries and sealed checkpoints are never touched.

**PrismaBuild.** A relaunch is a new PrismaBuild owner. It reads the
original's boundary entries, the checkpoint entries and this file as
declared inputs, exactly as a forward-recovery capsule's inputs are read,
and the original owner must be contained first (``require_contained``).
That dispatcher contract is not exercised by the fixture tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re

from .cost_stage_checkpoint import canonical_json, canonical_json_sha256, publish_new_bytes
from .dev_mode import NOT_COMPUTED, seal_check
from .joint_adjoint_slices import checkpoint_cotangent_plane, checkpoint_is_referenced
from .matmul_arithmetic import BF16_REDUCTION_FIELD

CHAIN_STATE_SCHEMA = "prismaquant.stage_a.chain_state.v1"
CHAIN_ARITHMETIC_SCHEMA = "prismaquant.stage_a.chain_arithmetic.v1"
RESUME_RECORD_SCHEMA = "prismaquant.stage_a.chain_resume.v1"
RESUME_COMPATIBILITY_SCHEMA = "prismaquant.stage_a.resume_compatibility.v1"
RESUME_COMPATIBILITY_SCOPE = "stage-a-chain-continuation"
#: The receipt and band field that carries every declaration. Outside the run
#: header, so a declaration never changes a slice or a band set's header.
RESUME_COMPATIBILITY_KEY = "resume_compatibility"

#: The chain-state fields a resume compares with what it recomputes.
_COMPARED = ("run_identity", "stride", "bind_identity", "arithmetic",
             "n_batches", "num_layers", "artifact_budget_override")
_STATE_FIELDS = frozenset({
    "schema", *_COMPARED, "boundary_storage", "boundary_entries", "producer",
    "tail_checkpoint", "chain_state_sha256"})
_ROLLING_ENTRY = re.compile(r"cotangent-\d+-\d+-at-\d+\.pt(\.tmp)?")
_RESUME_RECORD = re.compile(r"resume-(\d{3,})\.json")


class ChainResumeRefused(RuntimeError):
    """The relaunch is not a continuation of the run it names."""


def chain_state_path(space) -> Path:
    return Path(space) / "chain-state.json"


def resume_directory(space) -> Path:
    return Path(space) / "resumes"


def chain_arithmetic_stamp(runner, extra=None) -> dict:
    """What decides the chain's rounding beyond the code and the regime.

    The run's measurement dtype and device, the float32 matmul settings the
    Stage A entry point pins, and the Torch/CUDA build. The entry point adds
    the executing container image and the projection backend identity.
    """
    import torch

    from .matmul_arithmetic import bf16_reduction_stamp

    device = torch.device(runner.device)
    stamp = {
        "schema": CHAIN_ARITHMETIC_SCHEMA,
        "dtype": str(runner.dtype),
        "device_type": device.type,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        # Absent unless the bf16 reduced-precision reduction flag is off
        # (PQ #1028), so a chain state written before it still compares equal.
        **bf16_reduction_stamp(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        stamp["device_name"] = torch.cuda.get_device_name(device)
        stamp["device_capability"] = list(torch.cuda.get_device_capability(device))
    for key, value in dict(extra or {}).items():
        if key in stamp:
            raise ValueError(f"arithmetic stamp field {key!r} is the runner's own")
        stamp[key] = value
    return canonical_json(stamp, where="Stage A arithmetic stamp")


#: Chain-state fields that fix how many checkpoints, partitions and layers
#: the stored chain holds: a relaunch that differs in one cannot continue it,
#: so they stay walls in both modes (PQ #1147).
_LAYOUT_FIELDS = frozenset({"stride", "n_batches", "num_layers"})
#: The keys of ``run_identity`` and ``bind_identity`` that name a recorded run
#: identity: run seals (PQ #1147). Every other key names the data the chain
#: was computed from (calibration, probes, seed, token scope, partition,
#: roster, chain regime) and stays a wall. ``arithmetic`` and
#: ``artifact_budget_override`` are run seals as a whole: the arithmetic
#: stamp names the device, the build, the image and the projection backend,
#: which a human judges from the ``[DEV-MODE]`` line (the numerics exception).
_SEAL_KEYS = {
    "run_identity": frozenset({
        "plan_sha256", "prepared_sha256", "read_manifest_sha256",
        "implementation_sha256", "campaign_scope", BF16_REDUCTION_FIELD}),
    "bind_identity": frozenset({"source_model", "producer_source_sha256"}),
}


def _chain_wall(name, recorded, recomputed) -> bool:
    """Whether a differing chain-state field refuses in dev mode too."""
    if name in _LAYOUT_FIELDS:
        return True
    seal_keys = _SEAL_KEYS.get(name)
    if seal_keys is None or (isinstance(recomputed, str) and recomputed == NOT_COMPUTED):
        return False
    if not isinstance(recorded, dict) or not isinstance(recomputed, dict):
        return True
    return any(recorded.get(key) != recomputed.get(key)
               for key in set(recorded) | set(recomputed) if key not in seal_keys)


def _seal(document: dict) -> dict:
    body = {key: value for key, value in document.items() if key != "chain_state_sha256"}
    return {**body, "chain_state_sha256": canonical_json_sha256(body, where="chain state")}


def build_chain_state(*, run_identity, stride, boundary_storage, bind_identity, arithmetic,
                      boundary_entries, n_batches, num_layers, artifact_budget_override,
                      tail_checkpoint, producer=None) -> dict:
    """The sealed chain state a fresh run writes after its tail checkpoint."""
    document = {
        "schema": CHAIN_STATE_SCHEMA,
        "run_identity": run_identity,
        "stride": stride,
        "boundary_storage": boundary_storage,
        "bind_identity": bind_identity,
        "arithmetic": arithmetic,
        "boundary_entries": boundary_entries,
        "n_batches": int(n_batches),
        "num_layers": int(num_layers),
        "artifact_budget_override": artifact_budget_override,
        "tail_checkpoint": {"boundary": int(tail_checkpoint["boundary"]),
                            "cotangent_sha256": tail_checkpoint["cotangent_sha256"]},
        "producer": producer,
    }
    return _seal(canonical_json(document, where="chain state"))


def write_chain_state(space, document) -> dict:
    """Publish the chain state once; a second writer refuses."""
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    path = chain_state_path(space)
    if not publish_new_bytes(path, payload):
        raise ChainResumeRefused(
            f"{path} already exists: a run's chain state is written once, at its "
            "tail checkpoint")
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def load_chain_state(space, sha256) -> dict:
    """The chain state under its pinned digest, self-seal verified."""
    path = chain_state_path(space)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ChainResumeRefused(
            f"the run has no chain state at {path}: only a run that sealed its "
            "tail checkpoint under #1001 can resume its chain") from exc
    if hashlib.sha256(raw).hexdigest() != str(sha256):
        raise ChainResumeRefused(f"{path} does not have the pinned digest {sha256}")
    document = json.loads(raw)
    if (not isinstance(document, dict) or document.get("schema") != CHAIN_STATE_SCHEMA
            or set(document) != _STATE_FIELDS):
        raise ChainResumeRefused(f"{path} is not a {CHAIN_STATE_SCHEMA} document")
    if _seal(document)["chain_state_sha256"] != document["chain_state_sha256"]:
        raise ChainResumeRefused(f"{path} does not seal its own content")
    return document


def read_chain_state(space) -> dict:
    """The chain state as written, its own seal checked; no pinned digest.

    For a reader that needs the run's numbers (a split join, PQ #738), not a
    relaunch that must be the same run: the bytes must seal their own
    content, and nothing is compared with a recorded digest.
    """
    path = chain_state_path(space)
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ChainResumeRefused(f"the run has no readable chain state at {path}") from exc
    if (not isinstance(document, dict) or document.get("schema") != CHAIN_STATE_SCHEMA
            or set(document) != _STATE_FIELDS
            or _seal(document)["chain_state_sha256"] != document["chain_state_sha256"]):
        raise ChainResumeRefused(f"{path} is not a {CHAIN_STATE_SCHEMA} document that "
                                 "seals its own content")
    return document


def parse_declaration(text) -> dict | None:
    """``FROM:TO`` implementation digests, or ``None``."""
    if text is None:
        return None
    parts = str(text).split(":")
    if (len(parts) != 2 or any(len(part) != 64 for part in parts)
            or any(not re.fullmatch(r"[0-9a-f]{64}", part) for part in parts)):
        raise ChainResumeRefused(
            "--resume-implementation-compatibility takes FROM:TO, two sha256 "
            "implementation digests")
    return {"from": parts[0], "to": parts[1]}


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_bytes())


def resume_records(space, session) -> list[dict]:
    """Every sealed resume record of ``session``, in resume order.

    A record of another session refuses: a space holds one run's chain state.
    """
    directory = resume_directory(space)
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.iterdir()):
        match = _RESUME_RECORD.fullmatch(path.name)
        if match is None:
            continue
        record = _read_json(path)
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        if (record.get("schema") != RESUME_RECORD_SCHEMA
                or record.get("record_sha256") != canonical_json_sha256(
                    body, where="chain resume record")
                or record.get("index") != int(match[1])):
            raise ChainResumeRefused(f"{path} is not a sealed chain resume record")
        if record["session"] != session:
            # One space holds one run's chain state, so one session resumes in it.
            raise ChainResumeRefused(f"{path} resumed another run's session")
        records.append(record)
    indices = [record["index"] for record in records]
    if indices != list(range(1, len(records) + 1)):
        raise ChainResumeRefused(f"the chain resume records of the run are not 1..n: {indices}")
    return records


def resume_declarations(space, session, *, below=None) -> list[dict]:
    """The implementation declarations of ``session``'s resumes.

    ``below`` keeps those whose switch happened above that checkpoint: the
    declarations a band of checkpoint ``below`` was sealed under.
    """
    return [record["compatibility"] for record in resume_records(space, session)
            if record["compatibility"] is not None
            and (below is None or record["compatibility"]["switch_checkpoint"] > int(below))]


@dataclass
class ChainResume:
    """A checked resume: everything it will read, and what it will remove."""

    document: dict
    boundary: int
    checkpoints: list
    checkpoint_directories: list
    compatibility: dict | None
    implementation_sha256: str
    index: int
    leftovers: list = field(default_factory=list)
    partials: list = field(default_factory=list)

    @property
    def session(self) -> dict:
        return self.document["boundary_storage"]["session"]


def _generation_directory(document) -> Path:
    storage = document["boundary_storage"]
    return Path(storage["directory"]) / str(storage["session"]["generation"])


def _sealed_checkpoints(space, marker, boundaries):
    """``(sealed {boundary: (record, directory)}, partial directories)``."""
    from .joint_adjoint_band import BandRefused, read_sealed_checkpoint
    from .joint_adjoint_checkpoints import occupied_checkpoint_directories

    sealed, partial = {}, []
    for directory in occupied_checkpoint_directories(space):
        boundary = int(directory.name.removeprefix("boundary-"))
        if not (directory / "checkpoint.json").is_file():
            partial.append((boundary, directory))
            continue
        try:
            record, _binding = read_sealed_checkpoint(Path(space), boundary)
        except (BandRefused, OSError, ValueError, KeyError) as exc:
            raise ChainResumeRefused(
                f"checkpoint {boundary} has a manifest that is not sealed: {exc}") from exc
        if record["session"] != marker:
            raise ChainResumeRefused(
                f"checkpoint {boundary} was sealed by another generation "
                f"({record['session'].get('generation')})")
        if boundary not in boundaries:
            raise ChainResumeRefused(
                f"checkpoint {boundary} is not a stride checkpoint of the run")
        sealed[boundary] = (record, directory)
    return sealed, partial


def plan_chain_resume(space, document, *, recomputed, running_implementation_sha256,
                      declaration=None, resume_from=None) -> ChainResume:
    """Check a relaunch against its run's sealed chain state. Writes nothing.

    ``document`` is the chain state ``load_chain_state`` verified under its
    pinned digest. ``recomputed`` holds what the relaunch derives from its
    own inputs for every compared chain-state field, with ``run_identity``
    and ``bind_identity`` built under the header's own implementation.
    """
    from .joint_adjoint_checkpoints import adjoint_receipt_path, require_dev_mode

    space = Path(space)
    from .stage_a_retirement import refuse_retired_space
    refuse_retired_space(space, ChainResumeRefused,
                         what="a retired run has no checkpoints to resume from")
    if adjoint_receipt_path(space).exists():
        raise ChainResumeRefused(
            f"{adjoint_receipt_path(space)} exists: the run completed and has no "
            "chain left to resume")
    differing = [name for name in _COMPARED if recomputed.get(name) != document[name]]
    if differing:
        refusal = ChainResumeRefused(
            "the relaunch is not the run its chain state seals; it differs in "
            + ", ".join(differing))
        if any(_chain_wall(name, document[name], recomputed.get(name)) for name in differing):
            raise refusal
        # Only run seals differ (PQ #1147). A relaunch may pass NOT_COMPUTED
        # for an input it would derive only to compare it here.
        for name in differing:
            seal_check(f"chain state {name}", document[name], recomputed.get(name),
                       where="Stage A chain resume", refusal=refusal)
    storage = document["boundary_storage"]
    seal_check("boundary storage policy", storage["policy"], recomputed.get("boundary_policy"),
               where="Stage A chain resume",
               refusal=ChainResumeRefused("the relaunch runs another boundary storage policy"))
    if recomputed.get("boundary_directory") != storage["directory"]:
        raise ChainResumeRefused("the relaunch writes another boundary directory")
    session = storage["session"]
    marker = {"generation": session["generation"], "kind": "adjoint_checkpoint",
              "run_identity_sha256": session["run_identity_sha256"]}
    boundaries = [int(mark) for mark in document["stride"]["boundaries"]]
    sealed, partial = _sealed_checkpoints(space, marker, boundaries)
    tail = document["tail_checkpoint"]
    if tail["boundary"] not in sealed or (
            sealed[tail["boundary"]][0]["cotangent_sha256"] != tail["cotangent_sha256"]):
        raise ChainResumeRefused("the tail checkpoint the chain state names is not sealed")
    boundary = min(sealed)
    if sorted(sealed) != sorted(mark for mark in boundaries if mark >= boundary):
        raise ChainResumeRefused(
            f"the sealed checkpoints {sorted(sealed)} are not every stride checkpoint "
            f"from {boundary} up: a gap cannot be resumed across")
    if resume_from is not None and int(resume_from) != boundary:
        raise ChainResumeRefused(
            f"the relaunch names checkpoint {resume_from}; the lowest sealed checkpoint "
            f"of the run is {boundary}")
    above = [mark for mark, _directory in partial if mark >= boundary]
    if above:
        raise ChainResumeRefused(
            f"checkpoint directories {above} are unsealed at or above the lowest sealed "
            f"checkpoint {boundary}")

    records = resume_records(space, session)
    sealed_by = (records[-1]["implementation_sha256"] if records
                 else document["run_identity"]["implementation_sha256"])
    running = str(running_implementation_sha256)
    compatibility = None
    if running == sealed_by:
        if declaration is not None:
            raise ChainResumeRefused(
                "an implementation declaration names a switch that is not happening: "
                f"checkpoint {boundary} was sealed by the running implementation")
    else:
        if declaration is None:
            # The implementation binding is a run seal (PQ #1147). Dev mode
            # resumes across implementations without a declaration; the switch
            # is still recorded below, so a human can decide whether it
            # changed stored numerics.
            seal_check(
                "implementation", sealed_by, running,
                where=f"Stage A chain resume at checkpoint {boundary}",
                refusal=ChainResumeRefused(
                    f"checkpoint {boundary} was sealed by implementation {sealed_by} and "
                    f"this relaunch runs {running}: resuming across implementations needs "
                    "an explicit --resume-implementation-compatibility declaration"))
        else:
            try:
                require_dev_mode("a Stage A chain resume under an implementation declaration")
            except RuntimeError as exc:
                raise ChainResumeRefused(str(exc)) from exc
            if declaration != {"from": sealed_by, "to": running}:
                raise ChainResumeRefused(
                    f"the declaration {declaration['from']}:{declaration['to']} is not the "
                    f"switch this relaunch makes, {sealed_by}:{running}")
        compatibility = {
            "schema": RESUME_COMPATIBILITY_SCHEMA,
            "scope": RESUME_COMPATIBILITY_SCOPE,
            "from_implementation_sha256": sealed_by,
            "to_implementation_sha256": running,
            "switch_checkpoint": boundary,
            **({} if declaration is not None else {"declared": False}),
        }

    generation = _generation_directory(document)
    try:
        status = _read_json(generation / "generation.json")
    except (OSError, ValueError) as exc:
        raise ChainResumeRefused(f"the run's generation has no status file: {exc}") from exc
    if status.get("session") != session:
        raise ChainResumeRefused("the run's generation names another session")
    seal_check("generation status", "running or failed", status.get("status"),
               where="Stage A chain resume",
               same=status.get("status") in ("running", "failed"),
               refusal=ChainResumeRefused(
                   f"the run's generation status is {status.get('status')!r}; only an "
                   "interrupted run resumes"))
    for producer in [document["producer"], *(record["producer"] for record in records)]:
        if producer is not None:
            require_producer_contained(producer)

    entries = generation / "entries"
    # A referenced checkpoint (PQ #1036) names some of these rolling
    # entries: they are the checkpoint's plane now, and the resume reads them.
    pinned = set()
    for mark in sealed:
        record = sealed[mark][0]
        if checkpoint_is_referenced(record):
            try:
                plane = checkpoint_cotangent_plane(record)
            except ValueError as exc:
                raise ChainResumeRefused(
                    f"checkpoint {mark} names a plane the resume cannot hold: {exc}") from exc
            pinned.update(Path(row["path"]) for row in plane.values())
    leftovers = sorted(path for path in entries.iterdir()
                       if _ROLLING_ENTRY.fullmatch(path.name) and path not in pinned)
    order = sorted(sealed, key=lambda mark: (mark != tail["boundary"], -mark))
    return ChainResume(
        document=document, boundary=boundary,
        checkpoints=[sealed[mark][0] for mark in order],
        checkpoint_directories=[sealed[mark][1] for mark in order],
        compatibility=compatibility, implementation_sha256=running,
        index=len(records) + 1, leftovers=leftovers,
        partials=[directory for _mark, directory in sorted(partial)])


def require_producer_contained(producer) -> None:
    """The PrismaBuild owner that wrote the interrupted attempt is gone.

    Removing its rolling entries while it still runs would pull files out
    from under a live writer; PrismaBuild's containment certificate is the
    proof it will not write again.
    """
    from .joint_forward_resume import ForwardRecoveryRefused, _sdk, require_contained

    sdk = _sdk()
    try:
        require_contained(sdk["pool"].PoolQueue(producer["queue_root"]),
                          {"owner_action_key": producer["owner_action_key"],
                           "owner_attempt": producer["owner_attempt"]}, sdk)
    except ForwardRecoveryRefused as exc:
        raise ChainResumeRefused(
            f"the owner {producer['owner_action_key'][:12]} that wrote the "
            f"interrupted attempt is not contained: {exc}") from exc


def producer_binding(publication) -> dict | None:
    """The PrismaBuild owner of this attempt, for a later resume's containment check."""
    if publication is None:
        return None
    return {"owner_action_key": str(publication.instance["owner_action_key"]),
            "owner_attempt": publication.instance["owner_attempt"],
            "queue_root": str(publication.queue.root)}


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
