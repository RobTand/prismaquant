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
as the stage-A receipt lands. A quantum is publishable once stage A's
terminal record says executed AND ``adjoint-capture.json`` validates
(digests match the sealed records). Re-run it (cron, a shell loop, or a
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
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__:
    from tools.tessera_campaign_container import (
        CONTAINER_IMAGE_FLAG,
        admission_image_reference,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tessera_campaign_container import (
        CONTAINER_IMAGE_FLAG,
        admission_image_reference,
    )

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
SPEC_PATH = Path("/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913"
                "/allocation/joint-panel/spec-hostcap32-ram-dev.json")
STATE_FILENAME = "campaign-state.json"

#: Refusal exits: 3 = the stage-A precondition (or the campaign binding)
#: failed closed; 1 = a submission itself failed.
EXIT_PRECONDITION_REFUSED = 3
EXIT_SUBMIT_FAILED = 1


class DispatchRefused(Exception):
    """Fail closed: no receipt, a stale receipt, or a mixed campaign."""


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
    scope, roster digests) and one stage-A receipt digest; anything else is
    a mixed campaign and refuses."""
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
    receipts = {record.get("adjoint", {}).get("receipt_sha256")
                for _, record in loaded}
    if len(receipts) != 1:
        raise DispatchRefused(
            "layer records do not share one stage-A receipt digest")
    # Before stage A publishes, the sealed records bind a null receipt
    # digest (§5.2 seals records first); after, they uniformly bind the
    # receipt file's digest. Mixed bindings are a mixed campaign.
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


def _plan_output_root(campaign: Mapping) -> Path:
    """The plan's sealed output_root: the only root the stage-A capture will
    write into (its identity guard refuses any other --output-root), and the
    root whose ``layer-quanta/adjoint/adjoint-capture.json`` is the receipt
    this dispatcher validates."""
    plan = json.loads(Path(campaign["plan_path"]).read_text())
    return Path(plan["output_root"])


def _container_wrap(spec_path: Path,
                    payload: list[str]) -> tuple[list[str], str | None]:
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
    """
    spec = json.loads(Path(spec_path).read_text())
    argv = ["python3", "-m", "tools.tessera_campaign_container",
            "--spec", json.dumps(spec, sort_keys=True),
            "--", *payload]
    return argv, admission_image_reference(spec)

def quantum_argv(record: dict, *, record_path: Path, output_root: Path,
                 priority: int = SUBMISSION_PRIORITY,
                 head_grace_s: int = HEAD_PROGRESS_GRACE_S,
                 consumer_tags: Sequence[str] = CONSUMER_TAGS) -> list[str]:
    """The exact §5.2 submission argv for one quantum. Pinned by tests: a
    drift here breaks placement.  ``consumer_tags`` is the effective §5.1
    placement policy, a conjunction PB matches against a worker's offered
    tags; PB alone decides which matching box claims the row."""
    quantum_id = record["quantum_id"]
    manifest = Path(record["read_set"]["manifest_path"])
    if not manifest.is_absolute():
        manifest = output_root / manifest
    slice_sha256 = _slice_manifest_digest(record, output_root=output_root)
    wrapped, container_image = _container_wrap(SPEC_PATH, [
        "python3", "-m", "prismaquant.joint_cost_quantum",
        "--quantum", str(record_path),
        "--quantum-sha256", record["identity_sha256"],
        "--data-manifest-sha256", slice_sha256,
        "--output-root", str(output_root)])
    argv = [sys.executable, str(PBRUN)]
    for tag in consumer_tags:
        argv += ["--tag", str(tag)]
    argv += ["--data-manifest", str(manifest),
             "--residency", "stage", "--residency-ram", "auto",
             "--progress-phase", f"head={head_grace_s}"]
    for chunk in record.get("chunks", []):
        argv += ["--progress-phase",
                 f"{chunk['name']}={CHUNK_PROGRESS_GRACE_S}"]
    argv += ["--priority", str(priority),
             "--demand", "gpu=1,mem_gb=104", "--gpu-memory-gb", "80",
             "--cpus", "10"]
    if container_image is not None:
        # A pbrun option, so it precedes the separator like the manifest: PB
        # must admit the row only where this image is already present, or
        # leave it ready for a box that has it (RobTand/prismabuild#714).
        argv += [CONTAINER_IMAGE_FLAG, container_image]
    argv += ["--env", DEV_MODE_ENV, "--detach", "--", *wrapped]
    return argv


def stage_a_argv(adjoint_manifest: Path, campaign: Mapping,
                 *, tag: str = ADJOINT_TAG,
                 prefetch_override: Path | None = None,
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

    The manifest binding is derived the same way: ``--data-manifest-sha256``
    and ``--read-manifest-sha256`` ride the payload (a run that bound no
    manifest digest gets no tier redirect), and the manifest's read phases
    become the row's ``--progress-phase`` declarations in manifest order --
    the worker refuses undeclared names, so the list is derived, never
    hardcoded (#835).

    ``binding`` (optional) is a precomputed :func:`_stage_manifest_binding`
    for this manifest and campaign, so a caller that also records the
    digests does not read the manifest twice.
    """
    if binding is None:
        binding = _stage_manifest_binding(adjoint_manifest, campaign)
    payload = [
        "python3", "-m", "prismaquant.joint_adjoint_capture",
        "--plan", str(campaign["plan_path"]),
        "--plan-sha256", str(campaign["plan_sha256"]),
        "--prepared", str(campaign["prepared_path"]),
        "--prepared-sha256", str(campaign["prepared_sha256"]),
        "--data-manifest-sha256", binding["data_manifest_sha256"],
        "--read-manifest-sha256", binding["read_manifest_sha256"],
        "--output-root", str(_plan_output_root(campaign)),
        "--resume"]
    if prefetch_override is not None:
        payload += ["--prefetch-override", str(prefetch_override)]
    wrapped, container_image = _container_wrap(SPEC_PATH, payload)
    argv = [sys.executable, str(PBRUN),
            "--tag", tag,
            "--data-manifest", str(adjoint_manifest),
            "--residency", "stage"]
    for phase in binding["phases"]:
        grace = (HEAD_PROGRESS_GRACE_S if phase == "head"
                 else CHUNK_PROGRESS_GRACE_S)
        argv += ["--progress-phase", f"{phase}={grace}"]
    argv += ["--demand", "gpu=1,mem_gb=104", "--gpu-memory-gb", "80",
             "--cpus", "10"]
    if container_image is not None:
        # Before the separator, like every other pbrun option; see the
        # quantum row above and RobTand/prismabuild#714.
        argv += [CONTAINER_IMAGE_FLAG, container_image]
    argv += ["--env", DEV_MODE_ENV, "--detach", "--", *wrapped]
    return argv


def check_adjoint_receipt(receipt_path: Path, records: list[tuple[Path, dict]]) -> dict:
    """Validate the stage-A receipt against the sealed records: schema, the
    shared campaign digests, and the receipt digest every record binds."""
    receipt = _load_json(receipt_path, where="stage-A receipt")
    if receipt.get("schema") != ADJOINT_SCHEMA:
        raise DispatchRefused(
            f"{receipt_path}: receipt schema is not {ADJOINT_SCHEMA!r}")
    campaign = records[0][1]["campaign"]
    for key in ("plan_sha256", "prepared_sha256"):
        if receipt.get(key) != campaign[key]:
            raise DispatchRefused(
                f"{receipt_path}: receipt {key} is not this campaign's "
                "(stale receipt)")
    digest = _sha_bytes(receipt_path.read_bytes())
    expected = records[0][1]["adjoint"]["receipt_sha256"]
    if expected is not None and digest != expected:
        raise DispatchRefused(
            f"{receipt_path}: digest {digest} does not match the sealed "
            f"receipt digest {expected} (moved or mismatched receipt)")
    return receipt


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


def main(argv: list[str] | None = None, _gateway: Gateway | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish distributed joint-AURA campaign rows (§5).")
    parser.add_argument("--records", required=True,
                        help="directory of sealed layer-quantum records")
    parser.add_argument("--output-root", required=True,
                        help="campaign output root (state + quantum spaces)")
    parser.add_argument("--adjoint-receipt", default=None,
                        help="stage-A adjoint-capture.json, when published")
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
    stage_a_keys = [event.get("action_key") for event in events
                    if event.get("event") == "stage-a-submitted"]

    receipt_path = (Path(args.adjoint_receipt) if args.adjoint_receipt
                    else _plan_output_root(records[0][1]["campaign"])
                    / "layer-quanta" / "adjoint" / "adjoint-capture.json")
    receipt_ok = False
    if receipt_path is not None:
        try:
            check_adjoint_receipt(receipt_path, records)
            receipt_ok = True
        except DispatchRefused as exc:
            # A receipt is named but stale: fail closed only once stage A is
            # terminally done (otherwise the receipt simply has not landed
            # yet and the run reports pending).
            if stage_a_keys and gateway.is_terminal_executed(stage_a_keys[-1]):
                print(f"dispatch_joint_quanta: refused: {exc}", file=sys.stderr)
                return EXIT_PRECONDITION_REFUSED
            receipt_ok = False

    rows: list[dict] = []
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
                                              binding=stage_a_binding)})
        if receipt_ok:
            for record_path, record in records:
                quantum_id = record["quantum_id"]
                key = submitted_keys.get(quantum_id)
                if key is not None and gateway.is_terminal_executed(key):
                    continue
                rows.append({"kind": "quantum", "quantum_id": quantum_id,
                             "identity_sha256": record["identity_sha256"],
                             "manifest_sha256": record["read_set"]["manifest_sha256"],
                             "argv": quantum_argv(
                                 record, record_path=record_path,
                                 output_root=output_root, priority=priority,
                                 head_grace_s=args.head_grace_s,
                                 consumer_tags=tags)})
    except DispatchRefused as exc:
        print(f"dispatch_joint_quanta: refused: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION_REFUSED

    if args.dry_run:
        print(json.dumps({"consumer_tags": list(tags), "priority": priority,
                          "stage_a_terminal": bool(stage_a_keys) and bool(
                              receipt_ok),
                          "rows": [{"kind": row["kind"],
                                    "quantum_id": row.get("quantum_id"),
                                    "identity_sha256": row.get("identity_sha256"),
                                    "manifest_sha256": row.get("manifest_sha256"),
                                    "receipt_sha256": records[0][1]["adjoint"]["receipt_sha256"],
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
                                               else None)})
            else:
                _append_state(state_path,
                              {"event": "quantum-submitted",
                               "quantum_id": row["quantum_id"],
                               "identity_sha256": row["identity_sha256"],
                               "data_manifest_sha256": row.get("manifest_sha256"),
                               "action_key": answer["action_key"]})
            print(json.dumps({"published": row.get("quantum_id", "stage-a"),
                              "action_key": answer["action_key"],
                              "status": answer.get("status", "")},
                             sort_keys=True))
    except (RuntimeError, OSError) as exc:
        print(f"dispatch_joint_quanta: submission failed: {exc}", file=sys.stderr)
        return EXIT_SUBMIT_FAILED
    if not rows:
        print(json.dumps({"published": [], "note": "nothing publishable"},
                         sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
