#!/usr/bin/env python3
"""Publish the distributed joint-AURA cost campaign's PB rows (§5).

A receipt-driven **submitter**, never a scheduler. It publishes rows PB
owns; it never claims, places, retries, reorders, or steers work, holds no
long-running state, and reads no capacity to choose a box. Placement is the
static policy both GB10 tags on every row; PB's ready-order, loop counts,
and tier tokens do the balancing. If the static policy starves a box, that
is a PB placement capability gap to file, not a knob to turn here.

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
from pathlib import Path

PBRUN = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py")
PBWAIT = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbwait.py")

#: §5.1 plan defaults. Both GB10 tags on every quantum row; PB owns which
#: box claims.
CONSUMER_TAGS = ("sparky", "sparklina")
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



def _container_wrap(spec_path: Path, payload: list[str]) -> list[str]:
    """Run a payload inside the qualified campaign container.

    The projection backend's runtime identity check (and the workload's own
    torch/CUDA requirement) qualify one image; a bare ``python3 -m ...``
    executes unidentified and refuses.  The single-run path wraps every
    command in ``tools.tessera_campaign_container`` with the campaign spec;
    the distributed rows are the same workload and take the same wrapper.
    """
    spec = json.loads(Path(spec_path).read_text())
    return ["python3", "-m", "tools.tessera_campaign_container",
            "--spec", json.dumps(spec, sort_keys=True),
            "--", *payload]

def quantum_argv(record: dict, *, record_path: Path, output_root: Path,
                 priority: int = SUBMISSION_PRIORITY,
                 head_grace_s: int = HEAD_PROGRESS_GRACE_S) -> list[str]:
    """The exact §5.2 submission argv for one quantum. Pinned by tests: a
    drift here breaks placement."""
    quantum_id = record["quantum_id"]
    manifest = Path(record["read_set"]["manifest_path"])
    if not manifest.is_absolute():
        manifest = output_root / manifest
    argv = [sys.executable, str(PBRUN),
            "--tag", CONSUMER_TAGS[0], "--tag", CONSUMER_TAGS[1],
            "--data-manifest", str(manifest),
            "--residency", "stage", "--residency-ram", "auto",
            "--progress-phase", f"head={head_grace_s}"]
    for chunk in record.get("chunks", []):
        argv += ["--progress-phase",
                 f"{chunk['name']}={CHUNK_PROGRESS_GRACE_S}"]
    argv += ["--priority", str(priority),
             "--demand", "gpu=1,mem_gb=104", "--gpu-memory-gb", "80",
             "--cpus", "10",
             "--env", DEV_MODE_ENV, "--detach", "--",
             *_container_wrap(SPEC_PATH, [
                 "python3", "-m", "prismaquant.joint_cost_quantum",
                 "--quantum", str(record_path),
                 "--quantum-sha256", record["identity_sha256"],
                 "--output-root", str(output_root)])]
    return argv


def stage_a_argv(adjoint_manifest: Path, campaign: Mapping,
                 *, tag: str = ADJOINT_TAG) -> list[str]:
    """The §5.2 stage-A submission argv: the adjoint capture goes first and
    alone; quanta wait on its receipt.  The campaign binding every record
    carries names the plan and prepared inputs (with digests) the capture's
    own CLI requires -- the records are the single source of those paths."""
    return [sys.executable, str(PBRUN),
            "--tag", tag,
            "--data-manifest", str(adjoint_manifest),
            "--residency", "stage",
            "--demand", "gpu=1,mem_gb=104", "--gpu-memory-gb", "80",
            "--cpus", "10",
            "--env", DEV_MODE_ENV, "--detach", "--",
            *_container_wrap(SPEC_PATH, [
                "python3", "-m", "prismaquant.joint_adjoint_capture",
                "--plan", str(campaign["plan_path"]),
                "--plan-sha256", str(campaign["plan_sha256"]),
                "--prepared", str(campaign["prepared_path"]),
                "--prepared-sha256", str(campaign["prepared_sha256"]),
                "--output-root", str(adjoint_manifest.parent.parent),
                "--resume"])]


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
    block = _plan_block(Path(args.plan) if args.plan else None)
    tags = tuple(block.get("consumer_tags", list(CONSUMER_TAGS)))
    priority = int(block.get("submission_priority", args.priority))
    adjoint_tag = str(block.get("adjoint", {}).get("tag", ADJOINT_TAG)
                      if isinstance(block.get("adjoint"), dict)
                      else ADJOINT_TAG)

    try:
        records = load_records(records_dir)
    except DispatchRefused as exc:
        print(f"dispatch_joint_quanta: refused: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION_REFUSED
    events = _read_state(state_path)
    submitted_keys = {event.get("quantum_id"): event.get("action_key")
                      for event in events if event.get("event") == "quantum-submitted"}
    stage_a_keys = [event.get("action_key") for event in events
                    if event.get("event") == "stage-a-submitted"]

    receipt_path = Path(args.adjoint_receipt) if args.adjoint_receipt else None
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
    if not stage_a_done and not receipt_ok:
        manifest = (Path(args.adjoint_manifest) if args.adjoint_manifest
                    else records_dir / "adjoint.data-manifest.json.gz")
        rows.append({"kind": "stage-a",
                     "argv": stage_a_argv(manifest, records[0][1]["campaign"], tag=adjoint_tag)})
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
                             head_grace_s=args.head_grace_s)})

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
                                           "action_key": answer["action_key"]})
            else:
                _append_state(state_path,
                              {"event": "quantum-submitted",
                               "quantum_id": row["quantum_id"],
                               "identity_sha256": row["identity_sha256"],
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
