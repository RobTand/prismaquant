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

NOT ESTABLISHED HERE, and deliberately not faked: re-staging an unchanged
group after its stage copy is released. ``publish_prepaid_batch``
short-circuits on the filed commitment, so a second publish answers the
committed duplicate instead of sealing a successor mover, and this lane
will not invent a successor key or a second durable charge to fake it. The
scenario is declared, skipped and names the exact call it needs (see
``test_repeat_read_after_release_needs_pb_rematerialization``).

HARNESS PROVENANCE. The produced-output API is an unqualified candidate:
no deployed runtime generation carries ``produced_output.py`` at all. It is
pinned by file digest against an IMMUTABLE bundle
(``stagea_produced_pb_pin.json``) cut from the owning lane's committed
tree, never against that lane's live worktree, and these tests skip loudly
rather than implying support that does not exist.

KNOWN HARNESS DEFECT, recorded rather than tolerated: resolving the pinned
candidate inserts its ``src`` on ``sys.path`` and leaves ``prismabuild`` in
``sys.modules``, so in a shared pytest process this module shadows the
deployed runtime generation for every other test that imports PrismaBuild.
Run it in its own pytest invocation. This is the same defect the frozen
produced-render harness records; it is not fixed blind here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys

import pytest

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
            pytest.skip(
                "a different prismabuild is already imported in this "
                "process and would shadow the pinned candidate bundle")
    for root in roots:
        src = root / "src"
        fleet = root / "tools" / "fleet"
        for entry in (fleet, src):
            if str(entry) not in sys.path:
                sys.path.insert(0, str(entry))
        _export_pythonpath(src, fleet)
        from prismabuild import produced_output as _po
        if not Path(_po.__file__).resolve().is_relative_to(src.resolve()):
            pytest.skip("the resolved prismabuild shadows the pinned bundle")
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
                             pb_repo: Path) -> str:
    """The owner's own sealed request, filed once (the parent identity).

    The movement template ``publish_prepaid_batch`` derives at runtime
    comes from exactly this request: the code closure covers a checkout
    carrying the fleet tools, sealed ONCE per producer and never re-sealed
    per group.
    """

    from prismabuild import core as pb
    checkout = tmp_path / "mover-checkout"
    tools = checkout / "tools" / "fleet"
    tools.mkdir(parents=True, exist_ok=True)
    for name in ("stage_move.py", "prewarm_loop.py", "stage_release.py"):
        (tools / name).write_bytes(
            (pb_repo / "tools" / "fleet" / name).read_bytes())
    body = {
        "schema": pb.ACTION_SCHEMA_V2,
        "task": {"definition_id": "tests/pq-stagea-boundary-producer",
                 "definition_version": "v1", "task_class": "generation",
                 "determinism": "deterministic",
                 "artifact_family": "generic", "artifact_kind": "generic",
                 "argv": ["/bin/true"], "working_directory": ".",
                 "result_path": "result"},
        "inputs": [],
        "code_closure": pb.build_code_closure(
            checkout, ["tools/fleet/stage_move.py",
                       "tools/fleet/prewarm_loop.py",
                       "tools/fleet/stage_release.py"]),
        "params": {"cwd": "."},
        "environment": {"variables": {"PATH": "/usr/bin:/bin"},
                        "toolchain": {}},
        "execution_scope": {"portability": "portable", "platform_key": None,
                            "host_class": None}}
    action = pb.seal_action(body)
    pb.PrismaBuildCAS(cas_root).publish_action_request(action)
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
                 gib: int = 4):
    """A real queue, admitted owner, declared template, bound publication,
    and a real ``StreamedBoundaryArtifacts`` writing inside its prefix."""

    _src, pb_repo = _pb_source()
    from prismabuild import produced_output as po
    from prismaquant.cost_streaming import (
        BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts)
    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublication)

    cas_root = tmp_path / "cas"
    owner = _sealed_producer_request(tmp_path, cas_root, pb_repo)
    q = _queue(tmp_path, gib=gib)
    prefix = tmp_path / "outputs"
    prefix.mkdir(parents=True, exist_ok=True)
    template = _template(str(prefix), payload_max_bytes=payload_max_bytes,
                         window_gib=window_gib)
    terms = po.owner_demand_terms(template)
    q.publish(action_key=owner, cas_root=str(cas_root),
              worker_script=str(pb_repo / "tools" / "prismabuild_worker.py"),
              checkout_root=str(tmp_path / "mover-checkout"),
              resources={"cpu": 1, "mem_gb": 1, **terms},
              produced_output_template=template)
    claimed = q.claim(owner="w-owner")
    assert claimed is not None and claimed["action_key"] == owner
    control = _broker_control(q, owner)
    env = {"PRISMABUILD_ACTION_KEY": owner,
           "PRISMABUILD_ACTION_NONCE": control["nonce"],
           "PRISMABUILD_ACTION_SCOPE": control["scope_id"]}
    po.declare_template(q.root, template)
    publication = BoundaryProducedPublication.bind_from_admitted_owner(
        queue_root=q.root, tier=TIER, env=env, command_extra=("--unpaced",))
    assert publication.admit_window().get("ok") is True
    _announce_tier(q, tmp_path / "stage", pb_repo)

    storage = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA,
        "directory": str(prefix / "exact"),
        "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": GROUP_SIZE})
    storage.bind({"source_model": "fixture"}, n_probes=1)
    storage.bind_produced_output(
        publication, group_size=GROUP_SIZE, n_batches=n_batches,
        max_entry_tensor_bytes=1 << 14)
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
    """The whole chain, with no fallback and one durable charge.

    Writer -> prewrite-before-the-first-byte -> deferred publish -> REAL
    sealed mover through claim/execute/finish -> composed reader context
    -> the UNCHANGED strict prefetch -> byte-equal tensors -> stage
    release.
    """

    import torch
    storage, publication, q, env, pb_repo = _bound_owner(tmp_path)
    references = _write_group(storage)
    assert storage.telemetry["produced_groups_prewritten"] == 1, (
        "one publication group, one prewrite -- never one per entry")
    assert storage.telemetry["produced_groups_published"] == 0, (
        "publishing at write time would spend the stage credit the first "
        "read needs")
    _stage_groups(storage, q)
    _strict(monkeypatch, env, pb_repo, q)
    with storage.prefetch(references) as window:
        for index, reference in enumerate(references):
            assert torch.equal(storage.get(window, reference),
                               torch.arange(8, dtype=torch.float32) + index)
    assert storage.telemetry["produced_groups_materialized"] == 1
    report = storage.produced_group_records()
    assert len(report) == 1 and report[0]["staged"] is True
    released = storage.release_produced_group(references[0])
    assert released.get("ok") is True, released


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
    _stage_groups(storage, q)
    _strict(monkeypatch, env, pb_repo, q)
    for start in (0, GROUP_SIZE):
        window_refs = references[start:start + GROUP_SIZE]
        with storage.prefetch(window_refs) as window:
            assert torch.equal(
                storage.get(window, window_refs[0]),
                torch.arange(8, dtype=torch.float32) + start)
    assert storage.telemetry["produced_groups_published"] == 2
    assert storage.telemetry["produced_groups_materialized"] == 2


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


@pytest.mark.skip(reason=(
    "BLOCKED on the PrismaBuild repeat-materialization entry point (the "
    "parallel produced-output lane). publish_prepaid_batch short-circuits "
    "on the filed commitment, so a second publish after retire_batch "
    "replays the committed duplicate instead of sealing a successor mover. "
    "The CAPABILITY needed -- re-stage an unchanged logical batch from the "
    "owner's bound (queue, instance, template) plus its EXISTING batch_id, "
    "tier and cas_root, returning the successor mover key, the PB-sealed "
    "materialization sequence and the batch namespace, idempotent on "
    "replay. The NAME is the owning PB lane's to settle; this lane calls "
    "no such symbol and invents none. Declared, never faked green."))
def test_repeat_read_after_release_needs_pb_rematerialization(tmp_path):
    raise AssertionError("declared acceptance, not implemented")


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
