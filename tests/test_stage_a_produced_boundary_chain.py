"""Stage A's own boundary entries, written and then read back through PB.

The live failure: an admitted Stage A action writes exact boundary entries
with the existing writer and then refuses to read them -- the strict
allowed-tier path resolves the process residency map, which is the map of
the run's SEALED INPUTS, so bytes this action produced are
``staged-not-serving``. There is no own-session exemption.

Every test here drives the REAL chain on tiny tensors and refuses every
shortcut: the pinned PrismaBuild candidate supplies the produced-output
API, ``Pool.execute`` runs the REAL sealed mover through claim/execute/
finish, the reader is the UNCHANGED strict prefetch, and the reader
context is composed from the fragments that mover actually filed.

FAILING-BEFORE, stated as a property rather than an implementation, so the
same body fails against pristine main and passes here:
``test_strict_read_of_an_unpublished_boundary_entry_refuses`` writes an
entry exactly as Stage A does and asks the strict reader for it with no
produced binding -- the staged-not-serving refusal, which is the whole
defect. ``test_own_boundary_group_publishes_stages_and_reads_back`` is the
same write and the same reader with the binding in place.

The BOUNDED CYCLE is closed and measured here rather than declared:
``test_the_bounded_cycle_reads_retires_and_reads_again`` reads a group,
lets the window exit return its stage copy, and reads the SAME unchanged
references again through a successor mover PrismaBuild sealed over its own
materialization generation -- one logical batch, one durable origin
charge, no second publication and no origin read.

NOT ESTABLISHED HERE, and said plainly rather than implied: the wait for
an asynchronous mover polls PrismaBuild's own records
(``materialization_state``'s receipt field) because PB publishes no
blocking library wait for a produced mover's row. The budget is bounded
and its expiry is a named ``BoundaryStagingTimeout``, but a blocking wait
would be better than a polled one and is a PB capability request, not a
thing this lane should build.

HARNESS PROVENANCE. The produced-output API's SOURCE is root-accepted and
its global application is a separate, pending step; what any particular
runtime generation carries is a dated observation in the pin record, not a
claim this file makes, because a fixture that asserts a live-fleet fact
goes stale the moment the fleet moves. It is
pinned by file digest against an IMMUTABLE bundle
(``stagea_produced_pb_pin.json``) cut from the owning lane's committed
tree, never against that lane's live worktree, and these tests skip loudly
rather than implying support that does not exist.

OWN PROCESS (PQ #1008). Resolving the pinned candidate inserts its ``src``
on ``sys.path`` and leaves ``prismabuild`` in ``sys.modules``, and one process
holds only one ``prismabuild``. So this module and every module that imports
it are marked ``own_process``: in a session that collects other modules too,
``tests/conftest.py`` runs their tests in a child pytest of their own and
reports each outcome under the test's own node id. A caller that is not
marked, and finds another ``prismabuild`` already imported, FAILS naming the
marker; it used to skip, which hid 25 tests in a green multi-file shard.
Only a missing bundle (no pinned candidate resolves on this box) still skips.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time

import pytest

# PQ #1008: one pinned prismabuild per process (tests/conftest.py).
pytestmark = pytest.mark.own_process

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TIER = "prismabuild-stage:dl380g10"
KIND = "stage_gib"
PIN_PATH = Path(__file__).resolve().parent / "stagea_produced_pb_pin.json"

#: The fixture's publication group: small enough to run in seconds, large
#: enough that a group is a group. The production group is the read
#: window's own batch count (``prefetch_batches``, 64); neither the adapter
#: nor the owner knows that number, which is why this can be 4.
GROUP_SIZE = 4

#: An owner that owes nothing. Three buckets, not two: an egress refusal
#: this lane cannot classify is reported in its own, because folding it
#: into "abandoned" would say a decision was made when none was.
_NO_DEBT = {"pending": {}, "abandoned": {}, "unclassified": {},
            "publish_deferred": {}}


# -- pinned candidate resolution -------------------------------------------

def _pin_roots(pin: dict) -> list[Path]:
    roots: list[Path] = []
    for entry in pin["search_paths"]:
        if "env" in entry:
            value = os.environ.get(entry["env"], "")
            if value:
                roots.append(Path(value))
        elif "glob" in entry:
            import glob as _glob
            roots += [Path(p) for p in sorted(_glob.glob(entry["glob"]),
                                              reverse=True)]
        elif "path" in entry:
            roots.append(Path(entry["path"]))
    return roots


def _export_pythonpath(*roots: Path) -> None:
    """Put these roots on ``PYTHONPATH`` for child processes.

    ``Pool.execute`` spawns the canonical worker as a CHILD, which resolves
    ``prismabuild`` from the environment rather than from this
    interpreter's ``sys.path``. Without this the pin would cover only half
    the run.
    """

    current = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
               if p]
    for root in roots:
        if str(root) not in current:
            current.insert(0, str(root))
    os.environ["PYTHONPATH"] = os.pathsep.join(current)


def _pb_source() -> tuple[Path, Path]:
    """Resolve the pinned candidate bundle, or skip loudly.

    Returns ``(src_dir, bundle_root)``; ``bundle_root`` carries
    ``tools/fleet``, which the distribution never installs and the mover
    needs.
    """

    pin = json.loads(PIN_PATH.read_text())

    def _digest_ok(root: Path) -> bool:
        try:
            return all((root / name).is_file() and hashlib.sha256(
                (root / name).read_bytes()).hexdigest() == digest
                for name, digest in pin["files"].items())
        except OSError:
            return False

    roots = [root for root in _pin_roots(pin) if _digest_ok(root)]
    already = sys.modules.get("prismabuild")
    if already is not None and roots:
        resolved = Path(already.__file__).resolve().parent
        if not resolved.is_relative_to((roots[0] / "src").resolve()):
            # A failure, not a skip (PQ #1008): a skip here hid 25 tests in
            # a green multi-file shard. A module marked ``own_process`` never
            # gets here in a shared session (tests/conftest.py).
            pytest.fail(
                f"a different prismabuild ({resolved}) is already imported in "
                "this process and would shadow the pinned candidate bundle: "
                "mark the calling module `pytestmark = "
                "pytest.mark.own_process`")
    for root in roots:
        src = root / "src"
        fleet = root / "tools" / "fleet"
        for entry in (fleet, src):
            if str(entry) not in sys.path:
                sys.path.insert(0, str(entry))
        _export_pythonpath(src, fleet)
        from prismabuild import produced_output as _po
        if not Path(_po.__file__).resolve().is_relative_to(src.resolve()):
            pytest.fail(
                f"the resolved prismabuild ({_po.__file__}) shadows the pinned "
                "bundle: mark the calling module `pytestmark = "
                "pytest.mark.own_process`")
        return src, root
    pytest.skip(
        "no PrismaBuild produced-output candidate resolves against "
        "tests/stagea_produced_pb_pin.json (the named candidate is PENDING "
        "QUALIFICATION and no runtime generation carries it)")


# -- PB-side fixtures (the shapes pbrun/tier_loop produce) -----------------

@pytest.fixture(autouse=True)
def _isolated_launch_context(monkeypatch):
    """A synthetic action never inherits the outer PB launch tuple.

    These tests run INSIDE a live PrismaBuild action, so the real tuple is
    in the environment. ``Pool.execute`` and the launcher it spawns must
    derive each synthetic action's OWN identity; a leaked outer tuple would
    bind the fixture's publication to the wrong owner.
    """

    for name in ("PRISMABUILD_ACTION_NONCE", "PRISMABUILD_ACTION_SCOPE",
                 "PRISMABUILD_READER_HELPER_ROOT", "PRISMABUILD_ACTION_KEY"):
        monkeypatch.delenv(name, raising=False)
    from prismaquant.residency_map import reset_residency_resolver_for_tests
    from prismaquant.staged_lease import set_lease_helper_root
    from prismaquant.staged_tier_policy import (
        deactivate_staged_tier_policy_for_tests)
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    set_lease_helper_root(None)
    yield
    reset_residency_resolver_for_tests()
    deactivate_staged_tier_policy_for_tests()
    set_lease_helper_root(None)


def _queue(tmp_path: Path, gib: int = 4):
    from prismabuild import pool
    q = pool.PoolQueue(tmp_path / "pb-queue")
    q.ensure_layout()
    q.mint_tier_capacity(TIER, {KIND: gib})
    return q


def _broker_control(q, owner: str) -> dict:
    from prismabuild import pool, resource_scope
    nonce = secrets.token_hex(16)
    scope = resource_scope.ResourceScope(
        owner, nonce, 1 * 1024 ** 3, q.root / "telemetry" / f"{owner}.json")
    token = secrets.token_hex(32)
    unit = ("prismabuild-job"
            + hashlib.sha256((owner + nonce).encode()).hexdigest()[:32]
            + ".slice")
    scope._adopt_created_scope({
        "scope_id": unit, "token": token,
        "cgroup_path": str(Path("/sys/fs/cgroup/prismabuild.slice") / unit)})
    control = scope.control_record()
    path = q.item_path(pool.CLAIMED, owner)
    live = pool._read_json(path)
    assert live is not None
    live["resource_scope"] = control
    pool._write_json_atomic(path, live)
    return control


def _template(prefix: str, *, payload_max_bytes: int = 1 << 20,
              window_gib: int = 2) -> dict:
    from prismabuild import produced_output as po
    return po.validate_template({
        "schema": po.TEMPLATE_SCHEMA_V1,
        "version": 1,
        "template_id": "pq-stagea-boundary-entries-v1",
        "output_prefix": prefix,
        "slots": {"boundary_entries": {"class": "payload"}},
        # The temp ceiling scales WITH the payload ceiling rather than
        # sitting at a default: every final passes through a staging file
        # of its own size before the rename, so the relationship is
        # explicit instead of a coincidence.
        "durable_maxima": {"payload_max_bytes": payload_max_bytes,
                           "checkpoint_max_bytes": 1 << 10,
                           "temp_max_bytes": payload_max_bytes},
        "working_demands": {TIER: {"minimum_gib": 1,
                                   "window_gib": int(window_gib)}},
        "permitted_tiers": [TIER]})


def _sealed_producer_request(tmp_path: Path, cas_root: Path,
                             pb_repo: Path, template: dict, *,
                             producer_environment=None) -> str:
    """The owner's own sealed request, filed once (the parent identity).

    The movement template ``publish_prepaid_batch`` derives at runtime
    comes from exactly this request: the code closure covers a checkout
    carrying the fleet tools, sealed ONCE per producer and never re-sealed
    per group.

    The request must also SEAL ITS PRODUCED-OUTPUT DECLARATION, and this is
    load-bearing rather than ceremonial. A produced-output owner carries
    its window as ordinary tier demand (``owner_demand_terms``), and the
    egress classifies a claimed row as a movement node by tier demand
    alone. The only thing that then distinguishes the producer from a
    corrupt mover is the verified hold ``stage_release`` checks: the claim
    row's ``produced_output`` ref against the declaration sealed in
    ``params`` and validated against this request's own inputs. A request
    without it is not a verified hold, taints the egress ownership pass,
    and ``retire_batch`` answers ``egress-incomplete`` with zero live pins
    -- which this fixture measured and briefly mistook for a PrismaBuild
    defect. It is not one; it is what an unsealed producer looks like from
    the egress's side.
    """

    from prismabuild import core as pb
    from prismabuild import produced_output as po
    checkout = tmp_path / "mover-checkout"
    tools = checkout / "tools" / "fleet"
    tools.mkdir(parents=True, exist_ok=True)
    for name in ("stage_move.py", "prewarm_loop.py", "stage_release.py"):
        (tools / name).write_bytes(
            (pb_repo / "tools" / "fleet" / name).read_bytes())
    cas = pb.PrismaBuildCAS(cas_root)
    envelope = tmp_path / "produced-template.json"
    envelope.write_text(json.dumps(template, sort_keys=True))
    template_input, _ = cas.ingest_input(
        envelope, input_id=pb.PRODUCED_OUTPUT_TEMPLATE_INPUT_ID)
    declaration = po.build_declaration(template, template_input)
    body = {
        "schema": pb.ACTION_SCHEMA_V2,
        "task": {"definition_id": "tests/pq-stagea-boundary-producer",
                 "definition_version": "v1", "task_class": "generation",
                 "determinism": "deterministic",
                 "artifact_family": "generic", "artifact_kind": "generic",
                 "argv": ["/bin/true"], "working_directory": ".",
                 "result_path": "result"},
        "inputs": [template_input],
        "code_closure": pb.build_code_closure(
            checkout, ["tools/fleet/stage_move.py",
                       "tools/fleet/prewarm_loop.py",
                       "tools/fleet/stage_release.py"]),
        "params": {"cwd": ".", "command": ["/bin/true"],
                   "produced_output_template": declaration},
        "environment": {"variables": {"PATH": "/usr/bin:/bin",
                                      **(producer_environment or {})},
                        "toolchain": {}},
        "execution_scope": {"portability": "portable", "platform_key": None,
                            "host_class": None}}
    action = pb.seal_action(body)
    cas.publish_action_request(action)
    return str(action["action_key"])


def _announce_tier(q, stage_root: Path, pb_repo: Path) -> None:
    """The tier record the storage role announces (tier_loop shape)."""

    from prismabuild import pool
    stage_root.mkdir(parents=True, exist_ok=True)
    registered = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[4]);"
         "sys.path.insert(0, sys.argv[5]);"
         "from prismabuild import pool;"
         "import stage_release;"
         "print(stage_release.register_stage_root("
         "pool.PoolQueue(sys.argv[1]), tier_id=sys.argv[2],"
         " stage_root=sys.argv[3]))",
         str(q.root), TIER, str(stage_root),
         str(pb_repo / "tools" / "fleet"), str(pb_repo / "src")],
        capture_output=True, text=True, timeout=120)
    assert registered.returncode == 0, registered.stdout + registered.stderr
    assert "registered" in registered.stdout, registered.stdout
    record = {"tier": "stage", "tier_id": TIER, "host": socket.gethostname(),
              "mountpoint": str(stage_root),
              "mover_python": sys.executable,
              "mover_tools_root": str(pb_repo / "tools" / "fleet")}
    path = Path(q.root) / "tiers" / f"{TIER}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    pool._write_json_atomic(path, record)


def _tier_host(q) -> str:
    for record in q.tiers():
        if isinstance(record, dict) and record.get("tier_id") == TIER:
            return str(record.get("host") or "")
    return ""


def _execute_mover(q, mover: str) -> dict:
    """The REAL worker seam: claim -> ``Pool.execute`` -> ``finish``.

    Executing the same argv directly is NOT the same as going through the
    pool: it stages the same bytes and skips the funding-intent lifecycle
    entirely, so the batch's window share is never returned and the
    producer's next refill refuses forever. The frozen render lane nearly
    filed that as a PrismaBuild counter defect; the counter was sound.
    """

    from prismabuild import pool
    claimed = q.claim(owner=f"w-mover-{mover[:8]}", tags=[_tier_host(q)])
    assert claimed is not None and claimed["action_key"] == mover, (
        "the mover row must be claimable on its tier host", claimed)
    row = pool._read_json(q.item_path(pool.CLAIMED, mover))
    assert isinstance(row, dict), "mover row must be claimed to execute"
    outcome = q.execute(row, timeout_s=240)
    assert outcome.get("returncode") == 0, (
        outcome.get("returncode"), outcome.get("stdout"),
        outcome.get("stderr"), outcome.get("status"))
    receipt = q.move_record(mover)
    assert isinstance(receipt, dict), "mover recorded no receipt"
    q.finish(mover, status="executed")
    return receipt


# -- the PQ-side fixture: a bound owner over a bound publication -----------

def _bound_owner(tmp_path: Path, *, n_batches: int = GROUP_SIZE,
                 payload_max_bytes: int = 1 << 20, window_gib: int = 2,
                 gib: int = 4, staging_timeout_s: float = 900.0,
                 published=False, producer_environment=None, claim_capacity=None,
                 n_probes=1, read_order=None, storage_config=None,
                 max_entry_tensor_bytes=1 << 14):
    """A real queue, admitted owner, declared template, bound publication,
    and a real ``StreamedBoundaryArtifacts`` writing inside its prefix.

    ``n_probes`` and ``read_order`` are the owner's probe count and the
    roll's read order (``bind_produced_output``); the defaults are the
    probe-major roll every test here was written for. ``storage_config``
    overrides the owner's byte bounds and ``max_entry_tensor_bytes`` the
    entry bound, for a fixture with larger entries (PQ #1128's profile)."""

    _src, pb_repo = _pb_source()
    from prismaquant.staged_lease import set_lease_helper_root
    # The adapter resolves EVERY prismabuild.* module through the sealed
    # generation the reader SDK names, so the generation has to be named
    # before anything binds -- not later, when `_strict` activates the
    # policy. Same bundle either way; this is where it is declared.
    set_lease_helper_root(str(pb_repo))
    from prismabuild import produced_output as po
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)
    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublication)

    cas_root = tmp_path / "cas"
    q = _queue(tmp_path, gib=gib)
    prefix = tmp_path / "outputs"
    prefix.mkdir(parents=True, exist_ok=True)
    template = _template(str(prefix), payload_max_bytes=payload_max_bytes,
                         window_gib=window_gib)
    # The template first: the owner's sealed request carries its declaration.
    owner = _sealed_producer_request(
        tmp_path, cas_root, pb_repo, template,
        producer_environment=producer_environment)
    terms = po.owner_demand_terms(template)
    q.publish(action_key=owner, cas_root=str(cas_root),
              worker_script=str(pb_repo / "tools" / "prismabuild_worker.py"),
              checkout_root=str(tmp_path / "mover-checkout"),
              resources={"cpu": 1, "mem_gb": 1, **terms},
              produced_output_template=template)
    claimed = q.claim(owner="w-owner", capacity=claim_capacity)
    assert claimed is not None and claimed["action_key"] == owner
    control = _broker_control(q, owner)
    env = {"PRISMABUILD_ACTION_KEY": owner,
           "PRISMABUILD_ACTION_NONCE": control["nonce"],
           "PRISMABUILD_ACTION_SCOPE": control["scope_id"],
           **(producer_environment or {})}
    po.declare_template(q.root, template)
    publication = BoundaryProducedPublication.bind_from_admitted_owner(
        queue_root=q.root, tier=TIER, env=env, command_extra=("--unpaced",))
    assert publication.admit_window().get("ok") is True
    _announce_tier(q, tmp_path / "stage", pb_repo)

    storage = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA,
        "directory": str(prefix / "exact"),
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": GROUP_SIZE,
        **(storage_config or {})})
    storage.bind({"source_model": "fixture"}, n_probes=n_probes, published=published)
    storage.bind_produced_output(
        publication, group_size=GROUP_SIZE, n_batches=n_batches,
        max_entry_tensor_bytes=max_entry_tensor_bytes,
        staging_timeout_s=staging_timeout_s,
        **({"read_order": read_order} if read_order is not None else {}))
    return storage, publication, q, env, pb_repo


def _strict(monkeypatch, env, pb_repo, q, *, tiers="ram,ssd"):
    """Activate the real strict policy with the real launch identity.

    ``PRISMABUILD_RESIDENCY_MAP`` is part of that identity, not decoration:
    the SDK's ``injected_context`` refuses ``no-map-context`` without it and
    derives the QUEUE ROOT from it (``Path(map_path).parent.parent``). PB's
    launcher sets it to the action's own input map, whose parent's parent is
    the queue root, so a real action always has it. The fixture sets the
    same thing. The file need not exist -- the context reads the string, not
    the map -- and the produced reads never resolve through it: they carry
    their own namespaced resolver as an argument. This is exactly why the
    supplemental context cannot simply BE this variable: the queue root is
    derived from its shape, so a produced batch's map pointed at here would
    send the SDK looking for the queue somewhere else.
    """

    from prismaquant.residency_map import ENV_VAR
    from prismaquant.staged_lease import set_lease_helper_root
    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(ENV_VAR, str(q.residency_map_path(
        env["PRISMABUILD_ACTION_KEY"])))
    set_lease_helper_root(str(pb_repo))
    activate_staged_tier_policy(tiers)


def _write_group(storage, *, boundary_index=0, count=GROUP_SIZE, first=0):
    import torch
    return [storage.write(torch.arange(8, dtype=torch.float32) + index,
                          batch_index=index, boundary_index=boundary_index)
            for index in range(first, first + count)]


_FLEET_DRIVER = """
import json, os, sys, time
from pathlib import Path
root, tag, stop, budget = sys.argv[1], sys.argv[2], Path(sys.argv[3]), float(sys.argv[4])
from prismabuild import pool
q = pool.PoolQueue(Path(root))
capacity = json.loads(sys.argv[5]) if len(sys.argv) > 5 else None
end = time.monotonic() + budget
while time.monotonic() < end and not stop.exists():
    claimed = q.claim(owner="w-fleet", tags=[tag], capacity=capacity)
    if claimed is None:
        time.sleep(0.05)
        continue
    key = str(claimed["action_key"])
    row = pool._read_json(q.item_path(pool.CLAIMED, key))
    outcome = q.execute(row, timeout_s=240)
    rc = outcome.get("returncode")
    q.finish(key, status="executed" if rc == 0 else "failed")
    print(json.dumps({"mover": key, "rc": rc,
                      "stderr": (outcome.get("stderr") or "")[-800:]}),
          flush=True)
"""


class _Fleet:
    """A real, ASYNCHRONOUS fleet for this queue, in its own process.

    Staging is asynchronous in production -- publishing seals a mover row
    and some worker claims, places and runs it later -- and a fixture that
    drives the mover by hand before the read hides exactly that. This runs
    the pool's own claim/execute/finish loop CONCURRENTLY with the read, so
    the read really does wait on PrismaBuild's receipt.

    It is a separate PROCESS, not a thread, for one measured reason: the
    worker the pool spawns derives its own strict identity from the
    environment it inherits, and refuses "rather than binding a strict
    identity from half of one" when an owner's launch tuple is present. A
    thread shares ``os.environ`` with the reader, which needs that tuple;
    a child process gets its own copy with the tuple removed.
    """

    def __init__(self, q, tag: str, tmp_path: Path, *, budget_s: float = 240.0, capacity=None):
        self._q = q
        self._tag = tag
        self._stop = tmp_path / "fleet.stop"
        self._budget = budget_s
        self._capacity = capacity
        self._proc = None

    def __enter__(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE",
                            "PRISMABUILD_ACTION_SCOPE", "PRISMABUILD_RESIDENCY_MAP",
                            "PRISMABUILD_READER_HELPER_ROOT")}
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _FLEET_DRIVER, str(self._q.root), self._tag,
             str(self._stop), str(self._budget), json.dumps(self._capacity)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.write_text("stop")
        try:
            out, err = self._proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            out, err = self._proc.communicate()
        self.stdout, self.stderr = out, err
        return False


def _fleet(q, tmp_path: Path, *, capacity=None):
    return _Fleet(q, _tier_host(q), tmp_path, capacity=capacity)


def _stage_groups(storage, q):
    """Seal each pending group's mover and run it through the real pool.

    The publish is what seals the mover; a real producer's fleet then
    admits and executes it. Here the fixture is the fleet, and it uses the
    pool's own claim/execute/finish rather than the sealed argv.

    MUST RUN BEFORE the strict launch identity is exported. ``Pool.execute``
    spawns the mover as a CHILD, which inherits this process's environment
    and derives its OWN identity from it; an owner's launch tuple sitting in
    the environment is a tuple that belongs to a different action, and the
    launcher refuses it -- "refusing rather than binding a strict identity
    from half of one". That refusal is correct, so the fixture orders
    around it instead of weakening it. Measured, not reasoned: the mover
    exited 1 with exactly that message until the order changed.
    """

    for key, group in list(storage._produced_groups.items()):
        if group["published"] is not None:
            continue
        storage._produced_publish(key, group)
        _execute_mover(q, str(group["published"]["mover_key"]))


# -- the gate --------------------------------------------------------------

def test_strict_read_of_an_unpublished_boundary_entry_refuses(
        tmp_path, monkeypatch):
    """FAILING-BEFORE: the defect, stated as the property it violates.

    An entry this action just wrote, asked for through the strict reader
    with no produced binding, is not in the run's input map and refuses.
    Nothing is mocked: the writer, the policy and the refusal are the
    production ones.
    """

    import torch
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)
    from prismaquant.residency_map import ENV_VAR
    from prismaquant.staged_lease import LeaseRefused
    from prismaquant.staged_tier_policy import activate_staged_tier_policy

    storage = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA,
        "directory": str(tmp_path / "exact"),
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": 1})
    storage.bind({"source_model": "fixture"}, n_probes=1)
    reference = storage.write(torch.arange(8, dtype=torch.float32),
                              batch_index=0, boundary_index=0)
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "map-absent.json"))
    activate_staged_tier_policy("ram,ssd")
    with pytest.raises(LeaseRefused, match="staged-not-serving"):
        with storage.prefetch([reference]):
            pass


def test_own_boundary_group_publishes_stages_and_reads_back(
        tmp_path, monkeypatch):
    """The whole chain, asynchronous, with no fallback and one charge.

    Writer -> prewrite-before-the-first-byte -> deferred publish at the
    first read -> a REAL sealed mover claimed and run by a REAL fleet in
    another process, concurrently -> the bounded wait on PrismaBuild's own
    receipt -> composed reader context -> the UNCHANGED strict prefetch ->
    byte-equal tensors. Nothing in this test stages anything by hand.
    """

    import torch
    storage, publication, q, env, pb_repo = _bound_owner(tmp_path)
    references = _write_group(storage)
    assert storage.telemetry["produced_groups_prewritten"] == 1, (
        "one publication group, one prewrite -- never one per entry")
    assert storage.telemetry["produced_groups_published"] == 0, (
        "publishing at write time would spend the stage credit the first "
        "read needs")
    from prismaquant.perturbed_x_cache import exact_lease_counters
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        leases_before = exact_lease_counters()
        with storage.prefetch(references) as window:
            for index, reference in enumerate(references):
                assert torch.equal(storage.get(window, reference),
                                   torch.arange(8, dtype=torch.float32) + index)
        leases = {name: count - leases_before[name]
                  for name, count in exact_lease_counters().items()}
    assert storage.telemetry["produced_groups_published"] == 1
    assert storage.telemetry["produced_groups_materialized"] == 1
    assert storage.telemetry["produced_group_release_failures"] == 0, (
        storage._produced_release_errors)
    # One material namespace, one tier: the window's entries share ONE
    # pinned lease, not one each (PQ #997).
    assert leases == {"windows_batched": 1, "entries_batched": len(references),
                      "entries_single": 0, "batch_fallbacks": 0}, leases
    assert len(references) > 1
    report = storage.produced_group_records()
    assert len(report) == 1 and report[0]["retired"] is True, (
        "the window's stage copy is a LOAN: it goes back when the pins do",
        report)


def test_the_bounded_cycle_reads_retires_and_reads_again(
        tmp_path, monkeypatch):
    """The acceptance chain: read -> release stage -> read again.

    One logical batch, ONE durable origin charge, across the forward and
    the reverse read. The second read is not a second publication and not
    a fallback: the group's stage copy went back to free when its pins
    released, and the repeat read takes it again through PrismaBuild's own
    repeat-materialization surface -- ``materialization_state`` to ask,
    ``ensure_batch_materialized`` to drive -- under a NEW sealed mover that
    the same real fleet claims and runs.
    """

    import torch
    storage, publication, q, env, pb_repo = _bound_owner(tmp_path)
    references = _write_group(storage)
    expected = [torch.arange(8, dtype=torch.float32) + index
                for index in range(len(references))]
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        with storage.prefetch(references) as window:
            for reference, want in zip(references, expected):
                assert torch.equal(storage.get(window, reference), want)
        batch_id = storage.produced_group_records()[0]["batch_id"]
        charge_after_first = publication.durable_charge()
        first = publication.materialization_state(batch_id=batch_id)
        assert first["stage_retired"] is True, (
            "the stage copy must be back in free before the repeat read "
            "can be a re-materialization at all", first)
        assert storage.telemetry["produced_groups_retired"] == 1

        # The SAME unchanged references, through a NEW materialization.
        with storage.prefetch(references) as window:
            for reference, want in zip(references, expected):
                assert torch.equal(storage.get(window, reference), want)

    second = publication.materialization_state(batch_id=batch_id)
    assert storage.telemetry["produced_groups_rematerialized"] == 1
    assert int(second["generation"]) == 1, (
        "PrismaBuild's own materialization generation, not a caller's",
        second)
    assert second["mover_key"] != first["mover_key"], (
        "a successor mover, sealed by PB over the filed generation", second)
    assert second["manifest_digest"] == first["manifest_digest"], (
        "the same logical batch: same descriptors, same manifest", second)
    assert second["batch_namespace"] == first["batch_namespace"]
    assert publication.durable_charge() == charge_after_first, (
        "one logical batch, one durable origin charge, constant across "
        "forward and reverse staging")
    assert storage.telemetry["produced_groups_published"] == 1, (
        "the repeat read must NOT be a second publication")


def _claimed_consumer(tmp_path: Path, cas_root: Path, q, pb_repo: Path,
                      *, name: str) -> str:
    """A second, REAL claimed action: the holder the pin belongs to.

    A reader pin names an owner, and PrismaBuild refuses one whose owner is
    not a live claim (``ownership-uncertain: bad owner``) -- correctly, since
    an unowned pin is a pin nothing can ever be held responsible for. So the
    holder in this test is an actual published and claimed action, not a
    string that looks like one.
    """

    from prismabuild import core as pb
    checkout = tmp_path / "mover-checkout"
    body = {
        "schema": pb.ACTION_SCHEMA_V2,
        "task": {"definition_id": f"tests/{name}",
                 "definition_version": "v1", "task_class": "generation",
                 "determinism": "deterministic",
                 "artifact_family": "generic", "artifact_kind": "generic",
                 "argv": ["/bin/true"], "working_directory": ".",
                 "result_path": "result"},
        "inputs": [],
        "code_closure": pb.build_code_closure(
            checkout, ["tools/fleet/stage_move.py"]),
        "params": {"cwd": ".", "command": ["/bin/true"]},
        "environment": {"variables": {"PATH": "/usr/bin:/bin"},
                        "toolchain": {}},
        "execution_scope": {"portability": "portable", "platform_key": None,
                            "host_class": None}}
    action = pb.seal_action(body)
    pb.PrismaBuildCAS(cas_root).publish_action_request(action)
    key = str(action["action_key"])
    q.publish(action_key=key, cas_root=str(cas_root),
              worker_script=str(pb_repo / "tools" / "prismabuild_worker.py"),
              checkout_root=str(checkout),
              resources={"cpu": 1, "mem_gb": 1})
    claimed = q.claim(owner=f"w-{name}")
    assert claimed is not None and claimed["action_key"] == key, claimed
    return key


def _extra_pin(q, publication, batch_id: str, state: dict, *,
               holder_key: str) -> dict:
    """A SECOND, independent reader pin on the group's staged bytes.

    Not a mock refusal: PrismaBuild's egress refuses to evict material a
    live reader still holds, so this is how a real ``egress-incomplete``
    is produced on demand -- by being a real reader.
    """

    from prismabuild import produced_output as po
    from prismabuild import reader_lease as rlc

    manifest = str(state["manifest_digest"])
    return rlc.acquire(
        q, consumer_action_key=po.batch_namespace(
            publication.instance, batch_id, manifest),
        attempt={"nonce": secrets.token_hex(16), "scope_id": "holder-scope"},
        tier_id=TIER, epoch="",
        span={"start_bytes": 0, "end_bytes": int(state["total_bytes"])},
        holder={"host": socket.gethostname(), "worker": "holder",
                "pid": os.getpid()},
        acquire_token=secrets.token_hex(16),
        covers=[{"mover_action_key": str(state["mover_key"]),
                 "manifest_sha256": manifest}],
        expected=None, owner_action_key=holder_key,
        residency_root=str(publication.fragment_root()))


def test_a_refused_release_is_recorded_and_drained_not_dropped(
        tmp_path, monkeypatch):
    """A refusal first, then an allowed release, with the window accounted.

    The failure shape this guards is specific: a returned ``{ok: False}``
    is a RESULT. With a four-token window, one group that quietly failed
    to give its two tokens back is the whole next advance, so a dropped
    non-ok does not degrade -- it stalls. The refusal here is a real one:
    an independent reader holds a pin on the staged bytes, and
    PrismaBuild's own egress refuses to evict material a reader holds.
    """

    import torch
    storage, publication, q, env, pb_repo = _bound_owner(tmp_path)
    references = _write_group(storage)
    holder_key = _claimed_consumer(tmp_path, tmp_path / "cas", q, pb_repo,
                                   name="pq-stagea-boundary-holder")
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        with storage.prefetch(references) as window:
            storage.get(window, references[0])
        batch_id = storage.produced_group_records()[0]["batch_id"]
        state = publication.materialization_state(batch_id=batch_id)
        assert state["stage_retired"] is True
        # Take the group back, then hold a real pin on it so the next
        # release cannot succeed.
        with storage.prefetch(references) as window:
            storage.get(window, references[0])
        # ^ that window's exit already tried to retire and, with no other
        # reader, succeeded. Re-materialize and hold a pin across the exit.
        live = publication.materialization_state(batch_id=batch_id)
        assert live["stage_retired"] is True

        holder = None
        try:
            with storage.prefetch(references) as window:
                storage.get(window, references[0])
                held = _extra_pin(
                    q, publication, batch_id,
                    publication.materialization_state(batch_id=batch_id),
                    holder_key=holder_key)
                assert held.get("ok") is True, held
                assert "pin_id" in held and "ref_id" in held, (
                    "the holder must file a real pin, not a pinless proof",
                    sorted(held))
                # Copy the identifiers out NOW. The returned record is
                # PrismaBuild's, and the egress this test is about touches
                # the same pin: a test must not depend on a foreign dict
                # still reading the same way later.
                holder = (str(held["pin_id"]), str(held["ref_id"]))
            debt = storage.produced_release_debt()
            assert list(debt["pending"]) == [batch_id], debt
            reason = debt["pending"][batch_id]
            assert reason.get("refusal") == "egress-incomplete", reason
            assert storage.telemetry["produced_group_release_failures"] >= 1
            assert storage.produced_group_records()[0]["retired"] is False, (
                "a refused retirement is never reported as retired")
        finally:
            from prismabuild import reader_lease as rlc
            if holder is not None:
                assert rlc.release(
                    q, holder[0], holder[1],
                    consumer_action_key=holder_key,
                    residency_root=str(publication.fragment_root())) is True

        # The leak that lives ONLY in the exceptional branch: the group's
        # entries are disposed while its retirement is still refused, so
        # the reference-keyed reclaim gate can never see this group again
        # -- its references are unlinked and out of the lookup index.
        for reference in references:
            storage.retire(reference)
        assert storage.produced_group_records()[0]["origin_reclaimed"] is False, (
            "nothing is reclaimed while the stage copy is still held")
        assert storage.produced_group_records()[0]["retired"] is False

        # The pin is gone, so the drain re-drives the SAME PrismaBuild
        # retire -- by group key, because there is no live reference left
        # to find it with -- and the charge check rides with it.
        storage._drain_produced_releases()
    assert storage.telemetry["produced_group_release_retries"] >= 1
    assert storage.produced_release_debt() == _NO_DEBT
    assert storage.produced_group_records()[0]["retired"] is True
    assert storage.produced_group_records()[0]["origin_reclaimed"] is True, (
        "the drained retire released the charge of a group whose origins "
        "went while its retirement was refused")
    assert publication.durable_charge()["payload"] == 0


def test_entries_disposed_INSIDE_a_window_still_return_their_credits(
        tmp_path, monkeypatch):
    """The production pattern, and the one a late-disposal test misses.

    Stage A disposes entries WHILE a read window is live: the tail retires
    an activation inside ``prefetched_boundary_batches``, and the reverse
    roll retires the previous cotangent as it writes the next. A retired
    reference leaves the lookup index -- correctly, it is dead -- so a
    window that resolved its groups again at exit would find nothing for a
    group whose entries all went, offer it to no retirement, raise no
    release debt, and keep its stage credits forever. The window therefore
    owns its groups from the moment it opens.
    """

    import torch
    storage, publication, q, env, pb_repo = _bound_owner(tmp_path)
    references = _write_group(storage)
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        with storage.prefetch(references) as window:
            for index, reference in enumerate(references):
                assert torch.equal(storage.get(window, reference),
                                   torch.arange(8, dtype=torch.float32) + index)
            for reference in references:
                storage.retire(reference)
            assert storage._produced_group_for(references[0]) == (None, None), (
                "a disposed reference is out of the lookup index; that is "
                "the condition this test exists for")
    record = storage.produced_group_records()[0]
    assert storage.telemetry["produced_groups_retired"] == 1, (
        "the window's stage copy must come back even though nothing it "
        "read is resolvable any more", storage.telemetry)
    assert storage.produced_release_debt() == _NO_DEBT
    assert record["retired"] is True and record["origin_reclaimed"] is True
    assert publication.durable_charge() == {"payload": 0, "checkpoint": 0,
                                            "temp": 0}


@pytest.mark.parametrize("gib", [2, 1])
def test_repeated_distinct_group_turnover_on_a_single_stage_token(
        tmp_path, monkeypatch, gib):
    """Four distinct groups through ONE token, and only that.

    What this establishes: repeated credit turnover across distinct
    publication groups. The template's window is ONE token, so only one
    group's stage copy is funded at a time -- each window's mover takes
    that token by exact transfer, retirement returns it, and the next
    window refills. Four groups against one window credit: if a stage copy
    were not returned, window 1 could not fund and this stops.

    Run at TWO tier tokens and at ONE, because the difference between them
    was a PrismaBuild defect and the one-token run is its acceptance.
    Steady-state occupancy is ONE either way -- the ledger shows held=1
    while a group is staged, held=0 with the token back in free at every
    window close, and at two tokens the second is never held by anyone.
    The second token was never overlap; it covered an INTERMITTENT
    STAGE-TOKEN LOSS in produced-output egress. The one-token parameter is
    what fails if that loss is back, and it is pinned against the repaired
    candidate rather than asserted.

    Measured, not inferred (tools/audit_produced_window_tokens.py, and
    repeats of this test at one token: 1 failure in 6, then reproduced
    under instrumentation on iteration 2 of 12). At the failure the tier
    has NO token anywhere: free empty, every holder directory empty, and a
    30 s poll never sees it return. The refusal is refill_window's
    availability branch -- ``{'refusal': 'tier-reservation-unavailable',
    'available': {}, 'window_gib': 1, 'held': 0, 'outstanding': 0}``, so
    room = 1 - 0 - 0 = 1, NOT the ``room <= 0`` branch that returns ok
    with acquired 0.

    PrismaBuild's own egress receipt names the mechanism. A healthy
    retirement reports ``tokens_released=1, tokens_decharged=0,
    entries_deleted=4, entries_shared=0``. The failing one reports
    ``tokens_released=0, tokens_decharged=1, entries_deleted=0,
    entries_shared=4, shared_with=['in-flight-copy']``: the staged entries
    were still shared with an in-flight copy, so nothing was deleted and
    the token was DECHARGED out of the mint instead of released to free,
    with nothing scheduled to return it (``retiring: false``,
    ``deferred_handoffs: []``). The tier's capacity silently shrank by one.

    Root traced it the rest of the way, in PrismaBuild and not here: the
    mover misreads ITSELF as another pending copy, because
    ``stage_release._evict_owned`` calls ``_claimed_paths`` without the
    ``exclude`` argument that call takes -- three lines from a
    ``_fragment_owners`` call that already self-excludes -- and the final
    move receipt exists before the row leaves CLAIMED. It drops its only
    proof and destroys its own token. That repair is assigned elsewhere.

    The repair landed in PrismaBuild and this lane re-pinned onto it:
    ``tools/fleet/stage_release.py`` moves between the two bundles
    (42f2cfb874077da66c27401e718d3636903a8a40
    6b94e738f78af6b98d61107f129e1364aa0a3bec13a165996b535b63aaf53bce ->
    83e8d502ad9305177ee6e0501ff7c88ee036579d
    900436256d57a78a43620385b60b5934a8197d42d0037cffd4140f429eb059a7),
    and ``src/prismabuild/produced_output.py`` with it. The one-token
    parameter is therefore a real acceptance against the new candidate,
    not a restored assumption: at the old one it failed 1 run in 6, so a
    single green here is evidence and not proof, and the flake it would
    catch is a regression of exactly that defect.

    What this does NOT establish: full cotangent rollover. The previous
    group's entries are disposed inside the NEXT group's window, after the
    previous group's stage copy has already been retired. Disposal of the
    CURRENT staged group's own references is a different transition and is
    test_entries_disposed_INSIDE_a_window_still_return_their_credits.
    """

    import torch
    storage, publication, q, env, pb_repo = _bound_owner(
        tmp_path, n_batches=4 * GROUP_SIZE, payload_max_bytes=1 << 22,
        window_gib=1, gib=gib)
    groups = [_write_group(storage, count=GROUP_SIZE, first=index * GROUP_SIZE)
              for index in range(4)]
    assert storage.telemetry["produced_groups_prewritten"] == 4
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        previous = None
        for index, references in enumerate(groups):
            with storage.prefetch(references) as window:
                assert torch.equal(
                    storage.get(window, references[0]),
                    torch.arange(8, dtype=torch.float32) + index * GROUP_SIZE)
                if previous is not None:
                    for reference in previous:
                        storage.retire(reference)
            previous = references
            assert storage.produced_release_debt() == _NO_DEBT, (
                "a window that could not give its credit back would starve "
                "the next one", index)
    records = {record["batch_id"]: record
               for record in storage.produced_group_records()}
    assert len(records) == 4
    assert storage.telemetry["produced_groups_retired"] == 4
    assert sum(1 for record in records.values()
               if record["origin_reclaimed"]) == 3, (
        "the three rolled-over groups released their charge; the last one "
        "still holds its entries", records)


def test_a_groups_durable_charge_is_reclaimed_when_its_last_origin_goes(
        tmp_path, monkeypatch):
    """Group-final reclaim: files gone AND stage copy retired, in that order.

    The rollover gate. A reverse walk replaces a cotangent plane boundary
    by boundary; without this the replaced planes keep their durable charge
    for the life of the instance and the origin class fills with bytes that
    are not there. It fires only when the group's LAST origin is unlinked
    and its stage copy is already retired -- a live materialization is
    material PrismaBuild still holds over those origins.
    """

    import torch
    storage, publication, q, env, pb_repo = _bound_owner(tmp_path)
    references = _write_group(storage)
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        with storage.prefetch(references) as window:
            storage.get(window, references[0])
    record = storage.produced_group_records()[0]
    assert record["retired"] is True and record["origin_reclaimed"] is False
    charged = publication.durable_charge()
    assert charged["payload"] > 0, charged

    for index, reference in enumerate(references):
        storage.retire(reference)
        expected = index == len(references) - 1
        assert storage.produced_group_records()[0]["origin_reclaimed"] is expected, (
            "the charge is released by the LAST origin going, never by the "
            "first", index)
    assert publication.durable_charge()["payload"] == 0, (
        "a batch whose origins are provably absent stops counting against "
        "the durable origin class")


def test_multi_window_initial_writes_exceed_the_window_and_fit_durable(
        tmp_path, monkeypatch):
    """More initial writes than one read window, all under the durable max.

    The property the first boundary needs: every group's writes land
    before the first read, charged only against the durable class budget,
    without the stage credits being consumed ahead of that read. Two
    groups are prewritten and neither is published until a read asks.
    """

    import torch
    storage, publication, q, env, pb_repo = _bound_owner(
        tmp_path, n_batches=2 * GROUP_SIZE)
    references = _write_group(storage, count=2 * GROUP_SIZE)
    assert storage.telemetry["produced_groups_prewritten"] == 2
    assert storage.telemetry["produced_groups_published"] == 0
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        for start in (0, GROUP_SIZE):
            window_refs = references[start:start + GROUP_SIZE]
            with storage.prefetch(window_refs) as window:
                assert torch.equal(
                    storage.get(window, window_refs[0]),
                    torch.arange(8, dtype=torch.float32) + start)
    assert storage.telemetry["produced_groups_published"] == 2
    assert storage.telemetry["produced_groups_materialized"] == 2
    assert storage.telemetry["produced_groups_retired"] == 2, (
        "each window gives its own stage copy back; holding both is how a "
        "bounded window silently becomes an unbounded one")


def test_a_foreign_session_reference_is_refused(tmp_path):
    """A reference from another generation never enters a bound window."""

    import torch
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)

    storage, _publication, _q, _env, _pb = _bound_owner(tmp_path)
    foreign = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA,
        "directory": str(tmp_path / "foreign"),
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": GROUP_SIZE})
    foreign.bind({"source_model": "other"}, n_probes=1)
    alien = foreign.write(torch.arange(8, dtype=torch.float32),
                          batch_index=0, boundary_index=0)
    with pytest.raises(RuntimeError, match="stale or belongs to another"):
        with storage.prefetch([alien]):
            pass


def test_an_entry_directory_outside_the_prefix_is_refused(tmp_path):
    """An own-generation path must sit inside the bound output prefix."""

    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)

    _storage, publication, _q, _env, _pb = _bound_owner(tmp_path)
    escaped = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA,
        "directory": str(tmp_path / "elsewhere"),
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": GROUP_SIZE})
    escaped.bind({"source_model": "fixture"}, n_probes=1)
    with pytest.raises(RuntimeError, match="outside the bound output prefix"):
        escaped.bind_produced_output(
            publication, group_size=GROUP_SIZE, n_batches=GROUP_SIZE,
            max_entry_tensor_bytes=1 << 14)


def test_a_mutated_origin_is_refused_before_it_can_be_staged(
        tmp_path, monkeypatch):
    """The writer's own inline digest is the contract PB enforces on copy.

    The descriptor carries the digest ``write_exact_activation_cache_entry``
    computed while serializing -- no reread, no sealing pass -- so it
    travels into the batch's data manifest. Mutating the origin after the
    batch is published, keeping its LENGTH, therefore has to be refused at
    the copy: this is a digest proof, not a size check.

    SCOPE, stated because it is easy to overclaim: this is the refusal
    BEFORE staging. A mutation after the bytes are already staged is NOT
    caught by a read, and honestly cannot be -- the reader is served the
    staged copy, which still matches the receipt the producer signed, and
    the origin it no longer reads is not re-examined. Re-checking the
    origin's identity belongs to the re-materialization path, which
    PrismaBuild does not publish yet (see the skipped acceptance below).
    """

    storage, _publication, q, _env, _pb = _bound_owner(tmp_path)
    references = _write_group(storage, count=1)
    key, group = next(iter(storage._produced_groups.items()))
    storage._produced_publish(key, group)
    origin = Path(references[0].path)
    payload = bytearray(origin.read_bytes())
    payload[-1] ^= 0xFF
    origin.write_bytes(bytes(payload))
    assert origin.stat().st_size == references[0].file_bytes, (
        "the mutation must preserve the length, or this proves a size "
        "check rather than a digest")
    from prismabuild import pool
    mover = str(group["published"]["mover_key"])
    claimed = q.claim(owner=f"w-mover-{mover[:8]}", tags=[_tier_host(q)])
    assert claimed is not None and claimed["action_key"] == mover
    row = pool._read_json(q.item_path(pool.CLAIMED, mover))
    outcome = q.execute(row, timeout_s=240)
    assert outcome.get("returncode") != 0, (
        "a mover must not stage bytes that disagree with the digest the "
        "producer signed", outcome.get("returncode"), outcome.get("stderr"))


def test_a_read_only_attachment_cannot_declare_an_owner_prewrite(tmp_path):
    """An attached generation stays an ordinary input-map reader.

    A read-only attachment writes nothing, so it has nothing to declare;
    taking a produced binding would let a foreign generation charge this
    owner's durable budget. It refuses.
    """

    import torch
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)

    storage, publication, _q, _env, _pb = _bound_owner(tmp_path)
    storage.write(torch.arange(8, dtype=torch.float32),
                  batch_index=0, boundary_index=0)
    attached = StreamedBoundaryArtifacts(dict(storage.config))
    attached.attach(storage.session, n_probes=1)
    assert attached._produced is None
    with pytest.raises(RuntimeError, match="read-only"):
        attached.bind_produced_output(
            publication, group_size=GROUP_SIZE, n_batches=GROUP_SIZE,
            max_entry_tensor_bytes=1 << 14)


def test_the_sdk_loads_one_generation_even_when_the_install_lacks_it():
    """An old installed distribution must not serve half the SDK.

    The deployed container carries a pip distribution at PrismaQuant's
    reviewed reader-lease pin, and that distribution has NO
    ``produced_output`` module at all -- the produced-output API is an
    unqualified candidate. Production forwards an immutable
    ``PRISMABUILD_READER_HELPER_ROOT`` and mounts it but does not populate
    ``sys.path``, so a bare ``import prismabuild.produced_output`` is
    exactly where a run would silently pick up the wrong generation, or a
    MIXTURE across modules.

    So this asserts the property that matters: with the helper root named,
    every module the publication uses resolves inside ONE package
    directory; with no helper root and no test injection, the loader
    refuses instead of falling back to whatever is installed.
    """

    from prismaquant.staged_lease import (
        LeaseRefused, set_lease_helper_root, sdk_submodule)

    _src, pb_repo = _pb_source()
    set_lease_helper_root(str(pb_repo))
    roots = set()
    for name in ("reader_lease", "produced_output", "pool", "storage_tiers",
                 "residency_map"):
        module = sdk_submodule(name)
        roots.add(str(Path(module.__file__).resolve().parent))
    assert len(roots) == 1, ("the SDK must be ONE generation, not a mixture",
                             sorted(roots))
    assert Path(roots.pop()) == (pb_repo / "src" / "prismabuild").resolve()

    set_lease_helper_root(None)
    with pytest.raises(LeaseRefused, match="lease-helper-unavailable"):
        sdk_submodule("produced_output")


def test_the_dispatcher_seals_the_template_the_client_supports(
        tmp_path, monkeypatch):
    """The seal rides the SUBMISSION, and only on a client that has it.

    An admitted produced-output owner needs the template on the queue row
    AND the matching declaration in the sealed request -- which is the
    submitting client's job, not a payload flag's. The deployed pbrun does
    it behind --produced-output-template; an older one does not, so the
    refusal is conditional on the client in hand rather than blanket. The
    derived template is checked here too: the durable origin class maximum
    is the configured artifact max, and the window comes from the actual
    maximum group.
    """

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    _src, pb_repo = _pb_source()
    from prismaquant.staged_lease import set_lease_helper_root
    set_lease_helper_root(str(pb_repo))
    from dispatch_joint_quanta import (
        ProducedOutputDeclarationUnsupported, build_stage_a_produced_template,
        stage_a_argv)

    artifact_max = 640 * (1 << 30)          # a PLANNING figure, passed in
    template = build_stage_a_produced_template(
        output_prefix="/home/rob/stage-a-out", tier=TIER,
        artifact_max_bytes=artifact_max,
        group_size=64, max_entry_tensor_bytes=16 << 20)
    assert template["durable_maxima"]["payload_max_bytes"] == artifact_max, (
        "the configured artifact max is the single source of the durable "
        "origin class maximum")
    assert template["working_demands"][TIER]["window_gib"] == 4, (
        "the window is the actual maximum group (64 x 16 MiB + envelope = "
        "2 tokens) for the current and the next group -- never derived "
        "from the retained origin peak", template["working_demands"])

    # The deployed client seals it; an older one does not, and the
    # dispatcher asks the client in hand rather than assuming either.
    import dispatch_joint_quanta as djq
    document = tmp_path / "produced-output-template.json"
    document.write_text(json.dumps(template, sort_keys=True))

    monkeypatch.setattr(djq, "_pbrun_seals_produced_output", lambda *a: False)
    with pytest.raises(ProducedOutputDeclarationUnsupported,
                       match="carries no --produced-output-template"):
        stage_a_argv(Path("/nonexistent/manifest.json"), {},
                     produced_output_template=document)

    monkeypatch.setattr(djq, "_pbrun_seals_produced_output", lambda *a: True)
    monkeypatch.setattr(djq, "_stage_manifest_binding", lambda *a, **k: {
        "data_manifest_sha256": "a" * 64, "read_manifest_sha256": "b" * 64,
        "phases": ["head"]})
    # The stub keeps the real wrapper's keyword surface: the dispatcher
    # passes the declared progress phases (and, after #991, the Stage B
    # resource policy). A narrower stub fails on the call, not on the seal.
    # The wrapper seals a spec; the Stage A row reads its spool from it
    # (PQ #1012).
    from stage_a_spool_spec import with_spool
    sealed = json.dumps(with_spool({"container": {"image": "sha256:" + "0" * 64}}))
    monkeypatch.setattr(djq, "_container_wrap",
                        lambda spec, payload, *, progress, resource_policy=None,
                        spool_max_bytes=None:
                        (["python3", "-m", "tools.tessera_campaign_container",
                          "--spec", sealed, "--", *payload], None))
    monkeypatch.setattr(djq, "_plan_output_root", lambda campaign: "/tmp/out")
    # The row reserves the plan's own memory bound (#997), so the plan is read.
    plan = tmp_path / "plan.json"
    # It also derives the spool window from the plan's geometry (#1110).
    from stage_a_spool_spec import stage_a_plan
    plan.write_text(json.dumps(stage_a_plan(
        tmp_path, aggregate_memory_bytes=108447924224)))
    argv = stage_a_argv(Path("/nonexistent/manifest.json"),
                        {"plan_path": str(plan), "plan_sha256": "c" * 64,
                         "prepared_path": "/q", "prepared_sha256": "d" * 64},
                        produced_output_template=document)
    assert "--produced-output-template" in argv, argv
    flag = argv.index("--produced-output-template")
    assert argv[flag + 1] == str(document)
    assert flag < argv.index("--"), (
        "the seal is an ENVELOPE option the client acts on, never a payload "
        "flag the capture would merely read", argv)
    assert "--produced-output-template" not in argv[argv.index("--"):]

    # pbrun derives the bounded window's tier demand from the template, so
    # the dispatcher must not restate it.
    demands = [argv[i + 1] for i, word in enumerate(argv) if word == "--demand"]
    assert not any("stage" in value for value in demands), demands


def test_the_pinned_candidate_provenance_is_immutable():
    """No fake green: the run names the immutable bundle it resolved.

    The bundle is read-only and digest-pinned; the owning lane's live
    worktree is deliberately not a search path, because a committed
    fixture must not resolve against a tree that exists to move.
    """

    src, root = _pb_source()
    pin = json.loads(PIN_PATH.read_text())
    assert not str(root).startswith("/home/rob/tmp/"), (
        "a committed fixture must not resolve against a scratch tree")
    for name, digest in pin["files"].items():
        path = root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        assert not os.access(path, os.W_OK), f"{name} is not immutable"
    import prismabuild
    assert Path(prismabuild.__file__).resolve().is_relative_to(src.resolve())


# -- the funding transient -------------------------------------------------
#
# Several PrismaBuild steps take a transition lock with
# ``blocking=False`` and answer ``{"ok": False, "refusal":
# "funding-race-deferred"}`` when it is contended: ``stage_output_intent``
# and ``fund_output_batch`` are the two on this path, and the same typed
# answer appears on the drive, release and adoption paths. Both
# ``publish_prepaid_batch`` and ``ensure_batch_materialized`` surface it
# verbatim, stamped with the step that met the contention -- MEASURED here
# as ``step="stage"`` when the owner lock is held before the publish, and
# observed in production as ``step="fund"``. Which step meets it is a race;
# that it was a lock and not a verdict is not. The transient is narrow:
# THAT non-blocking call moved no tokens. It does not mean nothing
# happened -- the request is sealed and filed, the funding intent staged
# and the READY row published before the funding step runs, and a deferral
# rolls none of that back. The re-drive is licensed by the content-
# addressed mover key and PrismaBuild's step-wise idempotency, not by an
# absence of side effects. The adapter therefore keys on the REFUSAL,
# never on the step.
#
# The contention here is REAL and it is PrismaBuild's own lock, taken from
# another PROCESS -- the fcntl leg, which is how a producer and the fleet's
# claim loop actually collide. Nothing substitutes the refusal.


_LOCK_HOLDER = """
import sys, time
from pathlib import Path
root, key, ready, stop = (sys.argv[1], sys.argv[2],
                          Path(sys.argv[3]), Path(sys.argv[4]))
from prismabuild import pool
q = pool.PoolQueue(Path(root))
with q._transition_locked(key, blocking=True) as held:
    if not held:
        raise SystemExit("holder did not acquire the transition lock")
    ready.write_text("held")
    while not stop.exists():
        time.sleep(0.02)
"""


@contextmanager
def _owner_lock_held(q, action_key: str, tmp_path: Path):
    """Hold PrismaBuild's own transition lock for one key, from elsewhere.

    A separate process on purpose. ``posix_lock.held`` serializes threads
    of ONE process on a per-path ``RLock`` before it ever reaches the
    inode, so a holder thread would prove the thread leg and not the
    ``fcntl.LOCK_EX | LOCK_NB`` leg that a real producer-versus-fleet
    collision takes.

    It calls ``_transition_locked`` -- private -- because that is the lock
    PrismaBuild itself takes on the OWNER key at the site under test, and
    the published ``mover_transition_lock`` is the other leg on a key that
    does not exist until the publish this is contending with. Re-deriving
    the lock path here would be inventing an interface instead of using
    one.

    Yields a ``release`` callable; also releases on exit.
    """

    ready = tmp_path / "owner-transition-lock.ready"
    stop = tmp_path / "owner-transition-lock.stop"
    for path in (ready, stop):
        if path.exists():
            path.unlink()
    env = {k: v for k, v in os.environ.items()
           if k not in ("PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE",
                        "PRISMABUILD_ACTION_SCOPE",
                        "PRISMABUILD_RESIDENCY_MAP")}
    proc = subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER, str(q.root), str(action_key),
         str(ready), str(stop)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    released = False

    def release():
        nonlocal released
        if released:
            return
        released = True
        stop.write_text("stop")
        try:
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()

    try:
        _until(lambda: ready.exists() or proc.poll() is not None, 30.0,
               "the holder process never took the transition lock")
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise AssertionError(f"holder exited early: {err or out}")
        yield release
    finally:
        release()


def _until(predicate, budget_s: float, message: str, poll_s: float = 0.02):
    """Wait for a real state change, bounded. Never a bare sleep."""

    deadline = time.monotonic() + float(budget_s)
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(poll_s)
    raise AssertionError(f"{message} (within {budget_s}s)")


def test_a_contended_funding_lock_is_waited_out_and_publishes_once(
        tmp_path, monkeypatch):
    """The observed transient, forced, and then survived without a double charge.

    What was observed in a suite run and NOT handled: one group's
    ``publish_prepaid_batch`` answered ``{'ok': False, 'refusal':
    'funding-race-deferred', 'step': 'fund'}`` and the adapter raised,
    ending the run. The adapter, not PrismaBuild, was the defect: every
    not-ok was terminal, with no discrimination between a documented
    transient and a real refusal.

    Forced here, deterministically: another process holds the OWNER
    transition lock, the read publishes into that contention, and the lock
    is released only once the publication reports it actually waited on a
    deferral -- a state change, never a sleep. The re-drive passes the same
    batch id, the same descriptors and the same instance, so PrismaBuild
    re-derives the same content-addressed mover key.

    The no-double-charge claim is the tier's own arithmetic, before and
    after: capacity unchanged (nothing minted, nothing decharged) and the
    free pool back where it started once the group retires, with exactly
    one group record and one mover.
    """

    import torch
    storage, publication, q, env, pb_repo = _bound_owner(
        tmp_path, staging_timeout_s=120.0)
    references = _write_group(storage)
    owner = env["PRISMABUILD_ACTION_KEY"]
    ledger = q.tier_ledger(TIER)
    capacity_before = ledger.capacity().get(KIND)
    free_before = ledger.available().get(KIND)
    outcome = {}

    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        with _owner_lock_held(q, owner, tmp_path) as release:

            def read():
                try:
                    with storage.prefetch(references) as window:
                        outcome["tensor"] = storage.get(window, references[0])
                    outcome["ok"] = True
                except BaseException as exc:            # noqa: BLE001
                    outcome["error"] = exc

            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            _until(lambda: publication.funding_deferrals >= 1
                   or "error" in outcome, 60.0,
                   "a held owner lock must make PrismaBuild defer the funding")
            assert "error" not in outcome, outcome.get("error")
            release()
            reader.join(180.0)

    assert not reader.is_alive(), "the publish never returned after the lock went"
    assert "error" not in outcome, outcome.get("error")
    assert torch.equal(outcome["tensor"], torch.arange(8, dtype=torch.float32))
    assert publication.funding_deferrals >= 1, (
        "the transient must have actually fired; otherwise this test proves "
        "nothing about waiting it out")
    assert storage.telemetry["produced_group_funding_deferrals"] >= 1
    assert storage.telemetry["produced_groups_published"] == 1, (
        "one publication, not one per re-drive")
    records = storage.produced_group_records()
    assert len(records) == 1
    assert storage.produced_release_debt() == _NO_DEBT
    mover = str(storage._produced_groups[
        next(iter(storage._produced_groups))]["published"]["mover_key"])
    assert ledger.capacity().get(KIND) == capacity_before, (
        "a retried funding must not mint or decharge capacity")
    assert ledger.holder_tokens(mover).get(KIND, 0) == 0, (
        "the one mover this published gave its token back")
    # Conservation, which is what "no duplicate charge" means on a ledger
    # that never mints: free plus this owner's remaining window is the
    # whole tier. A second charge would be parked under some holder and
    # this sum would fall short. It is NOT free_before: retirement returns
    # the staged token to FREE rather than to the owner's holding, so the
    # free pool ends HIGHER than it started by exactly that token --
    # measured (2 -> 3 of 4 at the fixture's window of 2), not assumed.
    assert (ledger.available().get(KIND)
            + ledger.holder_tokens(owner).get(KIND, 0)) == capacity_before, (
        "a token is parked somewhere it should not be", ledger.available(),
        ledger.holder_tokens(owner))
    assert ledger.available().get(KIND) > free_before, (
        "the staged token came back to free", ledger.available())


def test_an_inherited_spent_deadline_never_touches_prismabuild(
        monkeypatch):
    """ZERO calls when the budget arrived already spent. Not one.

    The rule this pins, stated where it is easiest to break: a deadline
    bounds SIDE EFFECTS, not attempts -- and that includes the FIRST one.
    A step handed a deadline a previous step already spent has no licence
    to seal a request, publish a row or take a reservation in order to be
    handed a refusal it can already see. An expiry observed locally is a
    fact; performing doomed work to have it confirmed is how a bounded run
    stops being bounded.

    The stub refuses EVERY attribute, so "no mutation" is not a count that
    could be read wrong: touching PrismaBuild at all fails the test.
    """

    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublication, BoundaryStagingTimeout)

    class _RefusesEveryCall:
        def __init__(self):
            self.touched = []

        def __getattr__(self, name):
            self.touched.append(name)
            raise AssertionError(
                f"a spent deadline reached PrismaBuild: {name}")

    clock = {"now": 100.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    publication = BoundaryProducedPublication.__new__(
        BoundaryProducedPublication)
    stub = _RefusesEveryCall()
    publication._po = stub
    spent = clock["now"] - 1.0

    with pytest.raises(BoundaryStagingTimeout) as caught:
        publication.publish(batch_id="b-inherited", descriptors=[{"x": 1}],
                            deadline=spent)
    assert stub.touched == [], stub.touched
    assert "did not START" in str(caught.value)

    with pytest.raises(BoundaryStagingTimeout):
        publication.ensure_batch_materialized(batch_id="b-inherited",
                                              deadline=spent)
    assert stub.touched == [], stub.touched

    # And the legacy one-shot is untouched: no budget, nothing bounded.
    calls = []
    assert publication._drive_prepaid_step(
        lambda: calls.append(1) or {"ok": True},
        batch_id="b-inherited", deadline=None) == {"ok": True}
    assert len(calls) == 1


def test_a_spent_budget_starts_no_further_prepaid_call(monkeypatch):
    """Zero calls after the deadline. Checked where the side effect is.

    The third variant of one bug class in this work, after the recursive
    budget reset and the unpaced loop: a budget read AFTER the call, or
    only after the sleep that reached it, still issues one more call past
    expiry -- and each of these calls can seal a request, stage a funding
    intent or publish a row. Counted on a FAKE clock, so what is asserted
    is the ordering and not a wall-clock coincidence.

    The method is driven directly on an unconstructed publication because
    it touches nothing else on it: the poll floor and the per-instance
    deferral count. Building a real owner here would measure the fixture.
    """

    from prismaquant.stage_a_produced_output import (
        FUNDING_RACE_REFUSAL, BoundaryProducedFundingDeferred,
        BoundaryProducedPublication)

    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(time, "sleep",
                        lambda seconds: clock.__setitem__(
                            "now", clock["now"] + float(seconds)))
    publication = BoundaryProducedPublication.__new__(
        BoundaryProducedPublication)
    deadline = 0.4
    at = []

    def deferred_call():
        at.append(clock["now"])
        return {"ok": False, "step": "fund", "refusal": FUNDING_RACE_REFUSAL,
                "attempt": len(at)}

    with pytest.raises(BoundaryProducedFundingDeferred) as caught:
        publication._drive_prepaid_step(
            deferred_call, batch_id="b-spent", deadline=deadline)

    assert at, "it must have driven the step at least once"
    assert [t for t in at if t >= deadline] == [], (
        "no call may START once the absolute deadline has passed", at,
        deadline)
    assert caught.value.outcome == {
        "ok": False, "step": "fund", "refusal": FUNDING_RACE_REFUSAL,
        "attempt": len(at)}, (
        "the LAST ACTUAL outcome is reported; a refusal nobody returned "
        "would be a fabrication", caught.value.outcome)
    assert caught.value.attempts == len(at)


def test_a_terminal_publish_refusal_is_not_retried(tmp_path, monkeypatch):
    """Everything that is not the typed transient fails at once.

    The danger in fixing the transient is fixing too much: a retry loop
    that treats any not-ok as retryable turns a real refusal -- an
    unavailable reservation, a rejected descriptor -- into a silent stall
    that ends in a timeout naming the wrong cause. One call, one failure,
    no wait.
    """

    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublicationFailed)

    storage, publication, q, env, pb_repo = _bound_owner(
        tmp_path, staging_timeout_s=120.0)
    references = _write_group(storage)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(time.monotonic())
        return {"ok": False, "step": "fund",
                "refusal": "tier-reservation-unavailable"}

    monkeypatch.setattr(publication._po, "publish_prepaid_batch", refuse)
    _strict(monkeypatch, env, pb_repo, q)
    started = time.monotonic()
    with pytest.raises(BoundaryProducedPublicationFailed) as caught:
        with storage.prefetch(references) as window:
            storage.get(window, references[0])
    elapsed = time.monotonic() - started
    assert len(calls) == 1, ("a terminal refusal is not re-driven", calls)
    assert elapsed < publication.FUNDING_DEFERRAL_POLL_S, (
        "it returned immediately rather than after a poll", elapsed)
    assert "tier-reservation-unavailable" in str(caught.value)
    assert publication.funding_deferrals == 0
    assert storage.produced_release_debt()["publish_deferred"] == {}


def test_a_funding_lock_that_never_clears_leaves_diagnosed_debt(
        tmp_path, monkeypatch):
    """The budget runs out and the held window is NAMED, not dropped.

    Contention that never clears is not a thing this lane may wait on
    forever, and it is not a thing it may forget either: the batch's
    commit is unfinished and its credit is still accounted somewhere, so a
    held window that nothing names is the invisible half of a leak. One
    absolute budget, a typed failure carrying what it waited for, and the
    group in its own debt bucket.
    """

    from prismaquant.stage_a_produced_output import (
        FUNDING_RACE_REFUSAL, BoundaryProducedFundingDeferred)

    storage, publication, q, env, pb_repo = _bound_owner(
        tmp_path, staging_timeout_s=1.0)
    references = _write_group(storage)
    owner = env["PRISMABUILD_ACTION_KEY"]
    ledger = q.tier_ledger(TIER)
    held_before = ledger.holder_tokens(owner).get(KIND, 0)
    batch_id = storage.produced_group_records()[0]["batch_id"]

    with _owner_lock_held(q, owner, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        started = time.monotonic()
        with pytest.raises(BoundaryProducedFundingDeferred) as caught:
            with storage.prefetch(references) as window:
                storage.get(window, references[0])
        elapsed = time.monotonic() - started

    # The owner lock is taken at the staging step before the funding one,
    # so this fixture meets the contention at "stage" while the production
    # occurrence met it at "fund". Same refusal, same nothing-moved, same
    # idempotent re-drive: the step is recorded, never dispatched on.
    assert caught.value.step in ("stage", "fund"), caught.value.step
    assert caught.value.attempts >= 2, (
        "it must have re-driven, not just failed once", caught.value.attempts)
    assert elapsed >= 1.0, ("it must have spent the budget it reports", elapsed)
    assert elapsed < 30.0, ("and ONLY that budget", elapsed)
    assert FUNDING_RACE_REFUSAL in str(caught.value)
    debt = storage.produced_release_debt()["publish_deferred"]
    assert list(debt) == [batch_id], debt
    assert debt[batch_id]["step"] == caught.value.step
    # MEASURED, not inferred: the owner's stage holding is what it was
    # before the attempt. That is the fact; what it is NOT is a claim that
    # the attempt left no trace. A sealed request, a staged intent and a
    # published row may all exist -- this lane makes no statement about
    # them and rolls nothing back.
    assert ledger.holder_tokens(owner).get(KIND, 0) == held_before, (
        "the owner still holds the credit it held, which is exactly why "
        "the unfinished commit has to be reported rather than dropped")


# -- the egress outcome seam ----------------------------------------------
#
# PrismaBuild's conservative egress fix makes an evicted mover's own live
# CLAIMED copy a deferred handoff instead of a second co-owner: bytes,
# proof and full credit are kept, and ordinary retry returns the token once
# that child mover reaches terminal. The receipt says so in a NEW top-level
# key, ``deferred_own`` -- a sibling of ``deferred_handoffs``,
# ``live_pins``, ``entries_deferred`` and ``errors``, not an entry in any
# of them. ``retire_batch`` still returns refusal "egress-incomplete", so
# the refusal is the category and the receipt is the cause.
#
# These tests drive the REAL bound owner and the real release path; only
# the egress receipt is substituted, because provoking a genuine own-copy
# deferral needs the PrismaBuild candidate that is not pinned yet. The
# shapes below are the confirmed ones, not invented ones.


def _staged_group(tmp_path, monkeypatch, **kwargs):
    """A real owner with one published, staged, read group."""

    storage, publication, q, env, pb_repo = _bound_owner(tmp_path, **kwargs)
    references = _write_group(storage)
    return storage, publication, q, env, pb_repo, references


def _incomplete(**receipt) -> dict:
    """What retire_batch returns when the egress did not complete."""

    body = {"complete": False, "reason": "egress", "live_pins": [],
            "deferred_handoffs": [], "entries_deferred": 0, "errors": []}
    body.update(receipt)
    return {"ok": False, "refusal": "egress-incomplete", "receipt": body}


def test_an_own_copy_deferral_is_waited_out_and_then_succeeds(
        tmp_path, monkeypatch):
    """The one case this lane waits on: keep asking until PB gives it back.

    PrismaBuild defers rather than destroying anything, so the credit is
    still there and ordinary retry is what returns it. The adapter must
    re-drive inside the budget it already has -- not a second one -- and
    must never declare the group retired on its own.
    """

    storage, publication, q, env, pb_repo, references = _staged_group(
        tmp_path, monkeypatch)
    monkeypatch.setattr(storage, "PRODUCED_DEFERRAL_POLL_S", 0.01)
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        with storage.prefetch(references) as window:
            storage.get(window, references[0])
            real = publication.retire
            calls = {"n": 0}

            def deferring(batch_id, **kwargs):
                calls["n"] += 1
                if calls["n"] <= 2:
                    return _incomplete(deferred_own=["own-copy-in-flight"],
                                       entries_deferred=GROUP_SIZE)
                return real(batch_id, **kwargs)

            monkeypatch.setattr(publication, "retire", deferring)
        # The condition, not a count: both scripted deferrals were served
        # and the retirement that followed them completed. The real
        # ``retire`` behind call three may itself defer while the fleet's
        # own mover is still live, and the adapter re-drives it through
        # this same wrapper, so pinning ``calls["n"]`` to 3 asserted that
        # the fleet was fast, not that the adapter waited (PQ #890).
        assert calls["n"] >= 3, ("two deferrals, then the real retire", calls)
        assert storage.produced_group_records()[0]["retired"] is True
    assert storage.produced_group_records()[0]["retired"] is True
    assert storage.produced_release_debt() == _NO_DEBT, (
        "a deferral that cleared owes nothing",
        storage.produced_release_debt())
    assert storage.telemetry["produced_group_release_deferrals"] == 1
    # FAILING-BEFORE (PQ #890): a deferral that cleared left no stage copy
    # standing, so it is not a release failure. Before the fix the counter
    # recorded one here -- and recorded none for the identical receipt seen
    # by the unwaited poll path, which is why
    # ``test_own_boundary_group_publishes_stages_and_reads_back`` was red
    # or green depending on how fast the in-process fleet was.
    assert storage.telemetry["produced_group_release_failures"] == 0, (
        "a deferral that cleared is not a stage copy left standing",
        storage._produced_release_errors)
    assert not storage.produced_output_report()["release_errors"]


def test_an_own_copy_deferral_that_never_clears_raises_inside_its_budget(
        tmp_path, monkeypatch):
    """Never continue quietly into a refill that cannot fund.

    The other half of the same contract: exhaustion is loud, it carries the
    elapsed time and the re-drive count, and the credit is reported as
    still owed rather than written off.
    """

    from prismaquant.stage_a_produced_output import (
        BoundaryProducedReleaseDeferred)

    storage, publication, q, env, pb_repo, references = _staged_group(
        tmp_path, monkeypatch)
    monkeypatch.setattr(storage, "PRODUCED_DEFERRAL_POLL_S", 0.05)
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        with pytest.raises(BoundaryProducedReleaseDeferred) as caught:
            with storage.prefetch(references) as window:
                storage.get(window, references[0])
                # ONE budget: the staging wait above spent it honestly at
                # its bound value, and the deferral below is shortened here
                # so the test does not sit for fifteen minutes proving a
                # bound it shares.
                storage._produced_plan["staging_timeout_s"] = 0.3
                monkeypatch.setattr(
                    publication, "retire",
                    lambda batch_id, **kwargs: _incomplete(
                        deferred_own=["own-copy-in-flight"]))
    assert caught.value.attempts >= 1, (
        "it must actually have re-driven, not merely timed out", caught.value)
    assert caught.value.waited_s >= 0.05, (
        "the paced wait is the point: a bounded attempt count that burns "
        "the budget in milliseconds reports a timeout it never waited for",
        caught.value.waited_s)
    assert storage.produced_group_records()[0]["retired"] is False
    assert storage.produced_release_debt()["abandoned"], (
        "the credit is still owed and says so",
        storage.produced_release_debt())
    assert storage.telemetry["produced_group_release_failures"] >= 1, (
        "a deferral that never cleared IS a stage copy left standing, and "
        "the counter says so where the wait gives up")


def test_a_receipt_without_the_deferred_own_key_is_surfaced_not_decided(
        tmp_path, monkeypatch):
    """A missing key is not an empty list.

    An older PrismaBuild's receipt carries no ``deferred_own`` at all and
    could not have reported an own-copy deferral. Reading that silence as
    "no deferral" -- which ``receipt.get("deferred_own", [])`` would do --
    is the exact fail-open shape that destroyed a stage token per
    occurrence. The handler written for it refuses to repeat it.
    """

    from prismaquant.stage_a_produced_output import BoundaryEgressUnclassified

    storage, publication, q, env, pb_repo, references = _staged_group(
        tmp_path, monkeypatch)
    old_receipt = {"complete": False, "reason": "egress", "live_pins": [],
                   "deferred_handoffs": [], "errors": []}
    assert "deferred_own" not in old_receipt
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        with pytest.raises(BoundaryEgressUnclassified) as caught:
            with storage.prefetch(references) as window:
                storage.get(window, references[0])
                monkeypatch.setattr(
                    publication, "retire",
                    lambda batch_id, **kwargs: {
                        "ok": False, "refusal": "egress-incomplete",
                        "receipt": dict(old_receipt)})
    assert "deferred_own" in str(caught.value)
    assert caught.value.kind == "unknown"
    assert storage.produced_release_debt()["unclassified"], (
        "reported in its own bucket, because folding it into 'abandoned' "
        "would claim a decision was made")


def test_an_unrecognised_deferral_reason_is_surfaced_not_waited_on(
        tmp_path, monkeypatch):
    """A deferral this lane cannot name is visible, never sat on.

    The inverse fail-open of the missing key, and the more expensive one:
    ``if receipt.get("deferred_own"):`` would read ANY non-empty list as
    "PrismaBuild will clear this", and then spend the whole staging budget
    waiting for something that was never that. The wait is licensed by one
    known reason -- the evicted mover's own in-flight copy, which ordinary
    retry returns -- and by nothing else.

    Driven through the REAL release path. The malformed shape (a
    ``deferred_own`` that is not a list) takes the same branch and is
    covered at the classifier, because what differs there is the receipt,
    not the handling.
    """

    from prismaquant.stage_a_produced_output import BoundaryEgressUnclassified

    storage, publication, q, env, pb_repo, references = _staged_group(
        tmp_path, monkeypatch, staging_timeout_s=30.0)
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        started = time.monotonic()
        with pytest.raises(BoundaryEgressUnclassified) as caught:
            with storage.prefetch(references) as window:
                storage.get(window, references[0])
                monkeypatch.setattr(
                    publication, "retire",
                    lambda batch_id, **kwargs: _incomplete(
                        deferred_own=["frobnicated-hold"]))
        elapsed = time.monotonic() - started
    assert caught.value.kind == "egress-deferral-unrecognised"
    assert "frobnicated-hold" in str(caught.value)
    assert elapsed < 30.0, (
        "it must not have spent the deferral budget on a reason it cannot "
        "name", elapsed)
    assert storage.telemetry["produced_group_release_deferrals"] == 0, (
        "nothing here is a deferral this lane waits on")
    assert storage.produced_release_debt()["unclassified"], (
        "reported in its own bucket: no decision was made about it")


def test_a_deferred_own_receipt_that_turns_into_a_foreign_pin_stops_waiting(
        tmp_path, monkeypatch):
    """Whatever it becomes, it is handled as that.

    A real foreign pin is preserved, not waited out: the wait exists for
    PrismaBuild's own child mover and for nothing else.
    """

    storage, publication, q, env, pb_repo, references = _staged_group(
        tmp_path, monkeypatch)
    monkeypatch.setattr(storage, "PRODUCED_DEFERRAL_POLL_S", 0.01)
    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        calls = {"n": 0}

        def turning(batch_id, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _incomplete(deferred_own=["own-copy-in-flight"])
            return _incomplete(live_pins=[{"pin_id": "a-real-reader"}])

        with storage.prefetch(references) as window:
            storage.get(window, references[0])
            monkeypatch.setattr(publication, "retire", turning)
    assert calls["n"] >= 2
    debt = storage.produced_release_debt()
    assert debt["pending"] or debt["abandoned"], (
        "a foreign pin is recorded and drained, never waited on", debt)
    assert not debt["unclassified"], (
        "a live pin is a positive cause; it is not the unknown case", debt)
    assert storage.produced_group_records()[0]["retired"] is False


def test_the_egress_classifier_reads_the_confirmed_receipt_shape():
    """Every row of the table, and only the first one waits.

    The reason string is matched EXACTLY, against the one PrismaBuild's
    egress writes (``stage_release.py``: ``"deferred_own":
    ["own-copy-in-flight"] if own_deferred else []``). Truthiness is not a
    match and neither is a substring: a deferral this lane has never seen
    is not one it knows will clear, and a field of the wrong shape is not
    a field at all.
    """

    from prismaquant.stage_a_produced_output import (
        OWN_COPY_IN_FLIGHT, OWN_EGRESS_IN_FLIGHT, classify_egress_outcome)

    assert OWN_COPY_IN_FLIGHT == "own-copy-in-flight"
    assert OWN_EGRESS_IN_FLIGHT == "own-egress-in-flight"
    assert classify_egress_outcome({"ok": True}) == "retired"
    # 1. The ones that wait: the mover's own live copy, and the retirement's
    #    own egress action on the tier host (RobTand/prismabuild#801), which
    #    an owner on a GPU host always sees because only the tier host can
    #    delete from the stage.
    assert classify_egress_outcome(_incomplete(
        deferred_own=["own-copy-in-flight"])) == "own-copy-deferral"
    assert classify_egress_outcome(_incomplete(
        deferred_own=["own-egress-in-flight"])) == "own-copy-deferral"
    # 2. Non-empty, unrecognised reason: visible, never waited on.
    assert classify_egress_outcome(_incomplete(
        deferred_own=["some-other-hold"])) == "egress-deferral-unrecognised"
    assert classify_egress_outcome(_incomplete(
        deferred_own=["own-copy-in-flight", "some-other-hold"])) == (
            "egress-deferral-unrecognised"), (
        "one unrecognised reason in the list is still unrecognised")
    # 3. Malformed -- not a list. A bare string that reads like the reason
    #    is the trap: it is truthy, it contains the right characters, and
    #    it is not this field.
    assert classify_egress_outcome(_incomplete(
        deferred_own="own-copy-in-flight")) == "egress-deferral-malformed"
    assert classify_egress_outcome(_incomplete(
        deferred_own={"reason": "own-copy-in-flight"})) == (
            "egress-deferral-malformed")
    assert classify_egress_outcome(_incomplete(
        deferred_own=True)) == "egress-deferral-malformed"
    # 4. Absent, with nothing else positive: an older PrismaBuild with no
    #    opinion, not a negative answer.
    assert classify_egress_outcome({
        "ok": False, "refusal": "egress-incomplete",
        "receipt": {"complete": False, "live_pins": [],
                    "deferred_handoffs": [], "errors": []}}) == "unknown"
    # 5. Empty with a live pin: a real foreign pin. Fail, do not wait.
    assert classify_egress_outcome(_incomplete(
        deferred_own=[], live_pins=[{"pin_id": "r"}])) == "foreign-pin"
    assert classify_egress_outcome(_incomplete(
        deferred_own=[],
        deferred_handoffs=["promotion-handoff"])) == "promotion-handoff"
    assert classify_egress_outcome(_incomplete(
        deferred_own=[], errors=["something"])) == "egress-error"
    # deferred_own present and EMPTY, nothing else: a complete answer.
    # PrismaBuild looked and found no own-copy deferral, which is a
    # positive observation and not the silence above.
    assert classify_egress_outcome(_incomplete(
        deferred_own=[])) == "egress-incomplete"
    # An older receipt that DOES name a cause still classifies by it: a
    # live pin is observed, not inferred from the missing key, and it is
    # reported identically by every generation. This is also the deployed
    # generation's only path, since none of them publishes deferred_own.
    assert classify_egress_outcome({
        "ok": False, "refusal": "egress-incomplete",
        "receipt": {"complete": False,
                    "live_pins": [{"pin_id": "r"}]}}) == "foreign-pin"


def test_mixed_outcomes_cannot_extend_the_declared_deferral_bound(
        tmp_path, monkeypatch):
    """One absolute deadline for the whole wait, whatever it sees.

    The failure this guards is the inverse of a tight loop and just as
    unbounded. A wait that hands a CHANGED outcome back by re-driving the
    retirement gets a fresh answer; when that answer is another deferral,
    the handler starts a NEW full budget, and a source that alternates
    deferral and something-else extends the bound forever while reporting
    the original one. It never spins and it never ends.

    Alternating outcomes here, a 0.3 s budget, and the assertion is that
    the whole wait still ends inside a small multiple of it.
    """

    from prismaquant.stage_a_produced_output import (
        BoundaryProducedReleaseDeferred)

    storage, publication, q, env, pb_repo, references = _staged_group(
        tmp_path, monkeypatch)
    monkeypatch.setattr(storage, "PRODUCED_DEFERRAL_POLL_S", 0.02)
    budget = 0.3
    calls = {"n": 0}

    def alternating(batch_id, **kwargs):
        calls["n"] += 1
        if calls["n"] % 2:
            return _incomplete(deferred_own=["own-copy-in-flight"])
        return _incomplete(live_pins=[{"pin_id": "a-real-reader"}])

    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        import time as _time
        started = _time.monotonic()
        try:
            with storage.prefetch(references) as window:
                storage.get(window, references[0])
                storage._produced_plan["staging_timeout_s"] = budget
                monkeypatch.setattr(publication, "retire", alternating)
        except BoundaryProducedReleaseDeferred:
            pass
        elapsed = _time.monotonic() - started
    assert elapsed < budget * 6, (
        "the wait must be bounded by the budget it declares, not by one "
        "budget per changed outcome", elapsed, budget, calls)
    assert storage.produced_group_records()[0]["retired"] is False


def test_a_changed_outcome_is_handled_not_re_driven_unseen(
        tmp_path, monkeypatch):
    """The outcome the wait observed is the outcome that is recorded.

    Re-driving to "find out" what it already saw costs an extra egress and
    replaces the observed refusal with a newer one, so a foreign pin seen
    inside the wait could be reported as whatever the next call returned.
    Here the retirement defers once, then reports a live pin, then would
    report a DIFFERENT cause if asked again -- and it must not be asked.
    """

    storage, publication, q, env, pb_repo, references = _staged_group(
        tmp_path, monkeypatch)
    monkeypatch.setattr(storage, "PRODUCED_DEFERRAL_POLL_S", 0.01)
    calls = {"n": 0}

    def sequence(batch_id, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _incomplete(deferred_own=["own-copy-in-flight"])
        if calls["n"] == 2:
            return _incomplete(live_pins=[{"pin_id": "the-observed-pin"}])
        return _incomplete(errors=["a third call would hide the pin"])

    with _fleet(q, tmp_path):
        _strict(monkeypatch, env, pb_repo, q)
        with storage.prefetch(references) as window:
            storage.get(window, references[0])
            monkeypatch.setattr(publication, "retire", sequence)
    assert calls["n"] == 2, (
        "exactly one retirement per decision: the deferral, then the "
        "outcome that changed. A third call means the wait asked again "
        "instead of handling what it saw", calls)
    debt = storage.produced_release_debt()
    recorded = json.dumps(debt)
    assert "the-observed-pin" in recorded, (
        "the recorded reason must be the outcome the wait actually saw",
        debt)
    assert "a third call would hide the pin" not in recorded, (
        "an extra retirement would have overwritten it", debt)


@pytest.mark.parametrize("lifetime", ["consumed", None])
def test_a_referenced_checkpoint_refuses_a_consumed_origin_batch(tmp_path, lifetime):
    """PQ #1036: the pin lives in PQ, so a consumed origin batch fails closed.

    PrismaBuild's retirement tick unlinks the origins of an origin-only batch
    committed ``consumed`` once its consumers succeed. Stage A commits staged
    batches today. If a group a checkpoint names were ever filed consumed,
    the seal's reference wait (``await_checkpoint_references``) refuses. One
    filed without a lifetime (retain) seals. The record is PB's own
    ``commitments.json`` of the real bound instance, written as PB files an
    origin-only batch.
    """

    import torch

    storage, publication, q, _env, _repo = _bound_owner(tmp_path, n_probes=1)
    references = [storage.write(torch.full((8,), float(batch)), batch_index=batch,
                                boundary_index=3, probe_index=0)
                  for batch in range(GROUP_SIZE)]
    _key, group = storage._produced_group_for(references[0])
    assert group is not None
    path = Path(publication._po.instance_dir(q.root, publication.instance)) / (
        "commitments.json")
    record = json.loads(path.read_text()) if path.exists() else {"batches": {}}
    entry = {**(record["batches"].get(group["batch_id"]) or {}), "origin_only": True}
    if lifetime is not None:
        entry["lifetime"] = lifetime
    record["batches"][group["batch_id"]] = entry
    path.write_text(json.dumps(record))

    assert publication.origin_only_lifetimes()[group["batch_id"]] == (
        lifetime or "retain")
    if lifetime == "consumed":
        with pytest.raises(RuntimeError, match="lifetime consumed"):
            storage.await_checkpoint_references(references)
    else:
        storage.await_checkpoint_references(references)
