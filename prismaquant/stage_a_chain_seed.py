"""Stage A seed mode: continue another run's sealed checkpoint in a scratch root.

RobTand/prismaquant#1016, part of #997. A chain resume (#1001) continues the
same run under the implementation that sealed it, or under an explicit
declaration, and it rebinds that run's own session. That cannot measure a
new implementation against a campaign run that is still sealed under an old
one: the new implementation's source digest is part of the bind identity,
so the relaunch can neither rebind the old session nor pass the capsule's
``validate_forward_state``. A **seed** run does the measurement instead:

* It binds a **fresh session in a scratch output root** that is neither the
  source run's root nor inside it, nor contains it. It never writes a byte
  under the source run's root.
* It **borrows exactly two things, each under a pinned sha256**: the source
  run's sealed checkpoint ``b`` (its manifest, cotangent plane and shared
  states), and the forward boundary rows ``through .. b - 1`` of the
  source run's forward-recovery capsule. It runs no forward pass and no
  tail.
* It **checks the science**. The seed's own bind identity, with the
  implementation that sealed checkpoint ``b`` put in its
  ``producer_source_sha256``, must hash to that checkpoint's session. The
  capsule must pass ``validate_forward_state`` under the same substitution.
  So the source run and the seed differ in nothing but the implementation
  (and the chain regime, which the seed stamps in its own run identity).
  A different implementation must be declared (``implementation_compatibility``,
  ``FROM`` and ``TO``); it is never inferred.
* It rolls the chain from ``b`` down to ``through``, sealing its own
  checkpoints at the stride boundaries on the way and at ``through``, its
  result, and stops. With a
  ``compare`` checkpoint of the source run at ``through``, it hashes each
  rolled ``(probe, batch)`` tensor payload as it writes it and each of the
  reference's, and records both in its receipt.
* It is **not a campaign run**. It writes ``chain-seed.json`` before it
  binds, and ``seed-receipt.json`` (never ``adjoint-capture.json``) when it
  completes. It writes no chain state. The band tool refuses a space that
  holds a seed marker, so a seed run can never feed Stage B.

Dev mode only. The seed spec is a sealed JSON document
(``prismaquant.stage_a.chain_seed.v1``) passed to the Stage A CLI with
``--chain-seed`` and ``--chain-seed-sha256``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .cost_stage_checkpoint import canonical_json, canonical_json_sha256, publish_new_bytes

SEED_SPEC_SCHEMA = "prismaquant.stage_a.chain_seed.v1"
SEED_MARKER_SCHEMA = "prismaquant.stage_a.chain_seed_marker.v1"
SEED_RECEIPT_SCHEMA = "prismaquant.stage_a.seed_receipt.v1"
SEED_COMPARISON_SCHEMA = "prismaquant.stage_a.seed_plane_comparison.v1"
PLANE_DISTANCE_SCHEMA = "prismaquant.stage_a.checkpoint_plane_distance.v1"
SEED_COMPATIBILITY_SCOPE = "stage-a-chain-seed"
SEED_MARKER_NAME = "chain-seed.json"
SEED_RECEIPT_NAME = "seed-receipt.json"
_SPEC_FIELDS = frozenset({"schema", "checkpoint", "capsule", "through",
                          "implementation_compatibility", "compare"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ChainSeedRefused(RuntimeError):
    """The seed is not a measurement continuation of the checkpoint it names."""


def seed_marker_path(space) -> Path:
    return Path(space) / SEED_MARKER_NAME


def seed_receipt_path(space) -> Path:
    return Path(space) / SEED_RECEIPT_NAME


def _pinned(binding, where) -> dict:
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str) or not binding["path"]
            or not isinstance(binding["sha256"], str)
            or not _SHA256.fullmatch(binding["sha256"])):
        raise ChainSeedRefused(f"the seed's {where} is not a {{path, sha256}} binding")
    return {"path": binding["path"], "sha256": binding["sha256"]}


def normalize_seed_spec(spec) -> dict:
    """The seed spec's fields, validated; refuses anything else."""
    if not isinstance(spec, dict) or set(spec) != _SPEC_FIELDS:
        raise ChainSeedRefused(
            f"a chain seed is a {SEED_SPEC_SCHEMA} document with exactly the fields "
            f"{sorted(_SPEC_FIELDS)}")
    if spec["schema"] != SEED_SPEC_SCHEMA:
        raise ChainSeedRefused(f"a chain seed is a {SEED_SPEC_SCHEMA} document")
    through = spec["through"]
    if type(through) is not int or through < 0:
        raise ChainSeedRefused("the seed's through boundary is a nonnegative integer")
    declared = spec["implementation_compatibility"]
    if declared is not None and (
            not isinstance(declared, dict) or set(declared) != {"from", "to"}
            or not all(isinstance(declared[key], str) and _SHA256.fullmatch(declared[key])
                       for key in ("from", "to"))):
        raise ChainSeedRefused(
            "the seed's implementation_compatibility is null or {from, to}, two sha256 "
            "implementation digests")
    return {
        "schema": SEED_SPEC_SCHEMA,
        "checkpoint": _pinned(spec["checkpoint"], "checkpoint"),
        "capsule": _pinned(spec["capsule"], "capsule"),
        "through": through,
        "implementation_compatibility": (None if declared is None
                                         else {"from": declared["from"],
                                               "to": declared["to"]}),
        "compare": None if spec["compare"] is None else _pinned(spec["compare"], "compare"),
    }


def load_seed_spec(path, sha256) -> dict:
    """The seed spec under its pinned digest."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ChainSeedRefused(f"the chain seed spec is not readable at {path}") from exc
    if hashlib.sha256(raw).hexdigest() != str(sha256):
        raise ChainSeedRefused(f"{path} does not have the pinned digest {sha256}")
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise ChainSeedRefused(f"{path} is not JSON") from exc
    return normalize_seed_spec(document)


def _sealed_checkpoint(binding, where) -> tuple[dict, Path]:
    """``(record, source run root)`` of a pinned ``checkpoint.json``.

    The bytes must be the one serialization the checkpoint writer publishes
    of its record (``checkpoint_manifest_bytes``), the record must seal its
    own fields, and the file must sit at
    ``<root>/layer-quanta/adjoint/checkpoints/boundary-NNN/checkpoint.json``
    beside the entries it lists.
    """
    from .joint_adjoint_checkpoints import (
        adjoint_space,
        checkpoint_manifest_bytes,
        checkpoint_manifest_entry,
        checkpoint_seal_sha256,
    )

    path = Path(binding["path"])
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ChainSeedRefused(f"the seed's {where} is not readable at {path}") from exc
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ChainSeedRefused(
            f"the seed's {where} {path} does not have the pinned digest "
            f"{binding['sha256']}")
    try:
        record = json.loads(raw)
        if checkpoint_manifest_bytes(record) != raw:
            raise ValueError("its bytes are not the writer's serialization")
        manifest = checkpoint_manifest_entry(record)
    except (ValueError, TypeError, KeyError) as exc:
        raise ChainSeedRefused(
            f"the seed's {where} {path} is not a sealed adjoint checkpoint: {exc}") from exc
    if record["cotangent_sha256"] != checkpoint_seal_sha256(record):
        raise ChainSeedRefused(f"the seed's {where} {path} does not seal its own manifest")
    if record["session"].get("kind") != "adjoint_checkpoint":
        raise ChainSeedRefused(f"the seed's {where} {path} is not an adjoint checkpoint")
    if Path(manifest["path"]) != path:
        raise ChainSeedRefused(
            f"the seed's {where} {path} is not beside the entries it lists")
    root = path.parents[4]
    if adjoint_space(root) != path.parents[2]:
        raise ChainSeedRefused(
            f"the seed's {where} {path} is not under a Stage A output root")
    from .stage_a_retirement import refuse_retired_space
    refuse_retired_space(path.parents[2], ChainSeedRefused,
                         what=f"the seed's {where} is retired")
    return record, root


def _plane(record, where) -> dict:
    """``{(probe, batch): row}`` of a copied (v1) or referenced (v2) checkpoint."""
    from .joint_adjoint_slices import checkpoint_cotangent_plane
    try:
        return checkpoint_cotangent_plane(record)
    except (ValueError, TypeError, KeyError) as exc:
        raise ChainSeedRefused(f"the {where} lists a non-cotangent entry: {exc}") from exc


def _whole_plane(record, *, n_probes, n_batches, where) -> dict:
    plane = _plane(record, f"seed's {where}")
    if set(plane) != {(probe, batch) for probe in range(n_probes)
                      for batch in range(n_batches)}:
        raise ChainSeedRefused(
            f"the seed's {where} is not a whole {n_probes} x {n_batches} cotangent plane")
    return plane


def _disjoint(output_root, source_root) -> None:
    ours, theirs = Path(output_root).resolve(), Path(source_root).resolve()
    if ours == theirs or theirs in ours.parents or ours in theirs.parents:
        raise ChainSeedRefused(
            f"a seed writes into a scratch output root; {ours} is the source run's "
            f"root {theirs}, inside it, or around it")


def _fresh(space) -> None:
    from .joint_adjoint_checkpoints import (
        adjoint_receipt_path,
        occupied_checkpoint_directories,
    )
    from .stage_a_chain_resume import chain_state_path, resume_directory

    held = [path for path in (adjoint_receipt_path(space), seed_marker_path(space),
                              seed_receipt_path(space), chain_state_path(space),
                              resume_directory(space)) if path.exists()]
    held += occupied_checkpoint_directories(space)
    generations = Path(space) / "exact-boundaries"
    if generations.is_dir():
        held += sorted(path for path in generations.iterdir() if path.is_dir())
    if held:
        raise ChainSeedRefused(
            "a seed binds a fresh scratch root, and this one already holds "
            + ", ".join(str(path) for path in held))


def _require_dev_mode() -> None:
    from .joint_adjoint_checkpoints import require_dev_mode
    try:
        require_dev_mode("a Stage A chain seed")
    except RuntimeError as exc:
        raise ChainSeedRefused(str(exc)) from exc


def preflight_seed_root(output_root, spec) -> Path:
    """Refuse a scratch root that is not a seed's to take; return the source root.

    The Stage A CLI calls this before the head walk and the model load, so a
    wrong root refuses before any GPU work, and the core calls it before its
    first ``mkdir``, so a root inside the source run's root refuses before a
    directory is created there. :func:`plan_chain_seed` repeats every check
    after the bind identity is known.
    """
    from .joint_adjoint_checkpoints import adjoint_space

    _require_dev_mode()
    spec = normalize_seed_spec(spec)
    _, source_root = _sealed_checkpoint(spec["checkpoint"], "checkpoint")
    _disjoint(output_root, source_root)
    _fresh(adjoint_space(output_root))
    return source_root


@dataclass(frozen=True)
class ChainSeed:
    """A checked seed: everything it will borrow, and what it records."""

    spec: dict
    boundary: int
    through: int
    record: dict
    source_root: Path
    sealed_by: str
    rows: dict
    plane: dict
    compare: dict | None
    binding: dict

    @property
    def source_space(self) -> Path:
        from .joint_adjoint_checkpoints import adjoint_space
        return adjoint_space(self.source_root)


def plan_chain_seed(space, output_root, spec, *, bind_identity, campaign_identity,
                    running_implementation_sha256, n_batches, n_probes,
                    num_layers) -> ChainSeed:
    """Check a seed against the checkpoint and capsule it borrows. Writes nothing."""
    from .joint_forward_resume import (
        ForwardRecoveryRefused,
        _read,
        chain_records,
        require_published_campaign,
        validate_forward_state,
    )

    _require_dev_mode()
    spec = normalize_seed_spec(spec)
    record, source_root = _sealed_checkpoint(spec["checkpoint"], "checkpoint")
    _disjoint(output_root, source_root)
    _fresh(space)
    boundary, through = int(record["boundary"]), spec["through"]
    if not through < boundary <= num_layers:
        raise ChainSeedRefused(
            f"a seed from checkpoint {boundary} of a {num_layers}-layer model rolls "
            f"down to a boundary below it; through is {through}")

    declared = spec["implementation_compatibility"]
    running = str(running_implementation_sha256)
    if declared is None:
        sealed_by = running
    else:
        if declared["to"] != running:
            raise ChainSeedRefused(
                f"the seed declares implementation {declared['to']}, and this run is "
                f"{running}")
        if declared["from"] == running:
            raise ChainSeedRefused(
                "the seed's declaration names no switch: FROM is the running "
                "implementation")
        sealed_by = declared["from"]
    source_identity = {**dict(bind_identity), "producer_source_sha256": sealed_by}
    if canonical_json_sha256(source_identity, where="exact boundary source") != (
            record["session"]["run_identity_sha256"]):
        raise ChainSeedRefused(
            f"checkpoint {boundary} was not sealed by this science under implementation "
            f"{sealed_by}: the bind identity with that implementation does not hash to "
            "its session; a different implementation needs an explicit "
            "implementation_compatibility declaration")
    plane = _whole_plane(record, n_probes=n_probes, n_batches=n_batches,
                         where=f"checkpoint {boundary}")

    try:
        document, _ = _read(spec["capsule"]["path"], spec["capsule"]["sha256"])
        require_published_campaign(document, bind_identity=source_identity,
                                   campaign_identity=campaign_identity)
        validate_forward_state(document, bind_identity=source_identity,
                               campaign_identity=campaign_identity)
        # Every row is pinned by the capsule's digest, as a Stage B owner's
        # attached chain is (``attached_chain``): no PB reads here. Each read
        # still checks the entry's own metadata and waits on PB's map.
        records = chain_records(document)
    except (ForwardRecoveryRefused, RuntimeError, OSError) as exc:
        # RuntimeError: the published-campaign resolver's own refusals.
        raise ChainSeedRefused(f"the seed's capsule refuses: {exc}") from exc
    missing = [layer for layer in range(through, boundary) if str(layer) not in records]
    if missing:
        raise ChainSeedRefused(
            f"the seed's capsule (frontier {document['frontier']}) does not hold the "
            f"forward boundaries {missing} a chain from {boundary} to {through} reads")
    rows = {layer: records[str(layer)] for layer in range(through, boundary)}
    if any(len(column) != n_batches for column in rows.values()):
        raise ChainSeedRefused("the seed's capsule rows are not one per calibration partition")

    compare = None
    if spec["compare"] is not None:
        reference, reference_root = _sealed_checkpoint(spec["compare"], "compare checkpoint")
        if reference_root != source_root or reference["session"] != record["session"]:
            raise ChainSeedRefused(
                "the seed's compare checkpoint is not the source run's own checkpoint")
        if int(reference["boundary"]) != through:
            raise ChainSeedRefused(
                f"the seed's compare checkpoint is boundary {reference['boundary']}; the "
                f"seed stops at {through}")
        compare = {"record": reference,
                   "plane": _whole_plane(reference, n_probes=n_probes, n_batches=n_batches,
                                         where="compare checkpoint")}

    binding = canonical_json({
        "checkpoint": {**spec["checkpoint"], "boundary": boundary,
                       "session": record["session"],
                       "cotangent_sha256": record["cotangent_sha256"]},
        "capsule": {**spec["capsule"], "frontier": document["frontier"],
                    "rows": [through, boundary - 1]},
        "through": through,
        "source_root": str(source_root),
        "implementation_compatibility": (None if declared is None else {
            "scope": SEED_COMPATIBILITY_SCOPE,
            "from_implementation_sha256": sealed_by,
            "to_implementation_sha256": running,
            "seed_checkpoint": boundary}),
        "compare": (None if compare is None else {
            **spec["compare"], "boundary": through,
            "cotangent_sha256": compare["record"]["cotangent_sha256"]}),
    }, where="Stage A chain seed")
    return ChainSeed(spec=spec, boundary=boundary, through=through, record=record,
                     source_root=source_root, sealed_by=sealed_by, rows=rows,
                     plane=plane, compare=compare, binding=binding)


def write_seed_marker(space, plan: ChainSeed, *, run_identity) -> dict:
    """Mark the scratch root as a seed run's before anything else is written."""
    document = {"schema": SEED_MARKER_SCHEMA, "seed": plan.binding,
                "run_identity": run_identity}
    payload = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False)
               + "\n").encode()
    path = seed_marker_path(space)
    if not publish_new_bytes(path, payload):
        raise ChainSeedRefused(f"{path} already exists: a scratch root holds one seed")
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def write_seed_receipt(space, receipt) -> dict:
    """Seal the seed receipt once, beside the marker; never ``adjoint-capture.json``."""
    if receipt.get("schema") != SEED_RECEIPT_SCHEMA:
        raise ChainSeedRefused("only a seed receipt is written as one")
    if not seed_marker_path(space).is_file():
        raise ChainSeedRefused(f"{space} holds no seed marker")
    payload = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)
               + "\n").encode()
    path = seed_receipt_path(space)
    if not publish_new_bytes(path, payload):
        raise ChainSeedRefused(f"{path} already exists: a seed receipt is written once")
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def tensor_payload_sha256(tensor) -> str:
    """sha256 of a tensor's contiguous payload bytes, dtype and shape aside."""
    import torch

    data = tensor.detach().to("cpu").contiguous().reshape(-1)
    return hashlib.sha256(data.view(torch.uint8).numpy()).hexdigest()


def compare_seed_plane(plan: ChainSeed, digests: dict, *, max_resident_bytes=None) -> dict:
    """Every rolled ``(probe, batch)`` payload digest against the reference's.

    ``digests`` are the seed's own, taken as it wrote the plane at
    ``plan.through``. The reference entries stream back through the same
    staged path a checkpoint load takes, in windows under
    ``max_resident_bytes`` (one entry at a time without one), after its
    manifest is checked against the pinned record.
    """
    from contextlib import closing

    from .joint_adjoint_checkpoints import (
        _verified_checkpoint_manifest,
        checkpoint_entry_session,
        stream_exact_entry_tensors,
    )
    from .residency_shard_reader import staged_range_wait_s
    import time

    if plan.compare is None:
        return None
    reference = plan.compare["record"]
    deadline = time.monotonic() + staged_range_wait_s()
    _verified_checkpoint_manifest(plan.source_space, reference, deadline=deadline)
    if set(digests) != set(plan.compare["plane"]):
        raise ChainSeedRefused(
            "the seed did not roll a whole plane at the compare boundary")
    entries = []
    keys = sorted(plan.compare["plane"])
    with closing(stream_exact_entry_tensors(
            [plan.compare["plane"][key] for key in keys],
            expected_session=checkpoint_entry_session(reference),
            max_resident_bytes=max_resident_bytes, deadline=deadline)) as stream:
        for (probe, batch), (_row, tensor) in zip(keys, stream):
            entries.append({"probe": probe, "batch": batch,
                            "seed_sha256": digests[probe, batch],
                            "reference_sha256": tensor_payload_sha256(tensor)})
            del tensor
    equal = sum(entry["seed_sha256"] == entry["reference_sha256"] for entry in entries)
    return {
        "schema": SEED_COMPARISON_SCHEMA,
        "boundary": plan.through,
        "reference": plan.binding["compare"],
        "entries": entries,
        "equal": equal,
        "different": len(entries) - equal,
        "bitwise_equal": equal == len(entries),
    }


def load_pinned_checkpoint(binding, where="checkpoint") -> dict:
    """The record of a pinned, self-sealing ``checkpoint.json``.

    The same checks a seed makes of the checkpoint it borrows: the pinned
    digest, the writer's own serialization, the self-seal, and the file's
    place beside its entries under a Stage A output root.
    """
    record, _ = _sealed_checkpoint(_pinned(binding, where), where)
    return record


def _cotangent_plane(record, where) -> dict:
    plane = _plane(record, where)
    if not plane:
        raise ChainSeedRefused(f"the {where} holds no cotangent entry")
    return plane


def _quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def checkpoint_plane_distance(reference, candidate, *, device="cpu",
                              read_ahead=8) -> dict:
    """Per-entry distance of one sealed cotangent plane from another (PQ #997).

    ``reference`` and ``candidate`` are ``{path, sha256}`` bindings of two
    sealed ``checkpoint.json`` files at the same boundary, for example two
    seed runs of one checkpoint under different chain regimes or matmul
    reduction settings. They must be sealed under one bind identity (the
    chain regime and the reduction flag are outside it), sit at one
    boundary, and list the same ``(probe, batch)`` entries,
    each of one shape and dtype. Every entry is read digest-verified, each
    plane streamed in leased windows that hold at most ``read_ahead`` of its
    entries (PQ #1142), and compared in fp32 with fp64 sums: ``relative_l2`` is
    ``||candidate - reference|| / ||reference||``, and ``max_abs`` the
    largest elementwise difference. Each batch of a Stage A plane is one
    calibration sample, so an entry is one (probe, sample).
    """
    from contextlib import closing
    import torch

    from .joint_adjoint_checkpoints import checkpoint_entry_session, stream_exact_entry_tensors

    records = {"reference": load_pinned_checkpoint(reference, "reference checkpoint"),
               "candidate": load_pinned_checkpoint(candidate, "candidate checkpoint")}
    if records["reference"]["boundary"] != records["candidate"]["boundary"]:
        raise ChainSeedRefused(
            f"the reference checkpoint is boundary {records['reference']['boundary']} "
            f"and the candidate {records['candidate']['boundary']}")
    if (records["reference"]["session"]["run_identity_sha256"]
            != records["candidate"]["session"]["run_identity_sha256"]):
        raise ChainSeedRefused(
            "the two checkpoints were sealed under different bind identities: they "
            "are not arms of one science")
    planes = {name: _cotangent_plane(record, f"{name} checkpoint")
              for name, record in records.items()}
    if set(planes["reference"]) != set(planes["candidate"]):
        raise ChainSeedRefused("the two checkpoints do not hold the same cotangent entries")
    for key, row in planes["reference"].items():
        other = planes["candidate"][key]
        if (row["shape"], row["dtype"]) != (other["shape"], other["dtype"]):
            raise ChainSeedRefused(
                f"cotangent {key} is {row['dtype']} {row['shape']} in the reference and "
                f"{other['dtype']} {other['shape']} in the candidate")
    if type(read_ahead) is not int or read_ahead < 1:
        raise ValueError("read_ahead is a positive number of entries per plane")

    keys = sorted(planes["reference"])
    largest = max(int(planes[name][key]["tensor_bytes"])
                  for name in planes for key in keys)

    def stream(name):
        return closing(stream_exact_entry_tensors(
            [planes[name][key] for key in keys],
            expected_session=checkpoint_entry_session(records[name]),
            max_resident_bytes=read_ahead * largest))

    entries = []
    with stream("reference") as reference_plane, stream("candidate") as candidate_plane:
        for key, (_ours_row, ours), (_theirs_row, theirs) in zip(
                keys, reference_plane, candidate_plane):
            equal = tensor_payload_sha256(ours) == tensor_payload_sha256(theirs)
            ref = ours.to(device=device, dtype=torch.float32)
            diff = theirs.to(device=device, dtype=torch.float32) - ref
            norm = float(ref.double().pow(2).sum().sqrt())
            dist = float(diff.double().pow(2).sum().sqrt())
            entries.append({
                "probe": key[0], "batch": key[1], "bitwise_equal": equal,
                "relative_l2": (0.0 if dist == 0.0 else
                                float("inf") if norm == 0.0 else dist / norm),
                "max_abs": float(diff.abs().max()) if diff.numel() else 0.0,
            })
            del ours, theirs, ref, diff
    relative = [entry["relative_l2"] for entry in entries]
    return {
        "schema": PLANE_DISTANCE_SCHEMA,
        "boundary": records["reference"]["boundary"],
        "reference": {**_pinned(reference, "reference checkpoint"),
                      "session": records["reference"]["session"]},
        "candidate": {**_pinned(candidate, "candidate checkpoint"),
                      "session": records["candidate"]["session"]},
        "entries": entries,
        "equal": sum(entry["bitwise_equal"] for entry in entries),
        "different": sum(not entry["bitwise_equal"] for entry in entries),
        "relative_l2": {"mean": sum(relative) / len(relative),
                        "median": _quantile(relative, 0.5),
                        "p99": _quantile(relative, 0.99),
                        "max": max(relative)},
        "max_abs": max(entry["max_abs"] for entry in entries),
    }
