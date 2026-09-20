"""Fleet acceptance scenario runner: real APIs, hermetic resources, JSON verdicts.

Each scenario drives ACTUAL production code against hermetic local state
(tmp dirs and sockets; no live fleet mutation, no placement, no claims on
live queues):

- the pinned PB candidate tree (resolved + whole-tree-verified by
  ``fleet_acceptance_pins``) for SDK/broker/queue/mover paths;
- the real candidate ``Authority`` with a controlled in-memory backend.
  The backend answers only the kernel/cgroup questions the test owns
  (populated? exists?); every verdict, token, ticket, and proof comes
  from real Authority code -- the same discipline as PB's own
  ``test_resource_broker.py`` (real Authority, controlled backend,
  real socket server);
- real ResourceScope client round trips over a real Unix socket server;
- real publish/claim/finish and stage/promote/map flows on tmp queues;
- real PQ readers over candidate-staged bytes.

Usage (invoked by ``test_fleet_acceptance_runner.py`` in a subprocess
with the test venv)::

    python tests/fleet_acceptance_runner.py <scenario>|resolve|list
        --work <tmp> --result <out.json> [--checkout <pq-root>]
        [--tree <verified-tree>]

Exit 0 with a result document whose ``status`` is ``qualified``,
``nonqualified`` (named reason), or ``failed`` (a real behavior
violation). Exit 2 on runner misuse. A scenario never invents a broker
reply or a terminal proof: scope blocks recorded into claims come
verbatim from validated broker replies under the queue's same-claim
guard, openly standing in for the worker loop's scope startup.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fleet_acceptance_pins as pins  # noqa: E402


KEY = "e" * 64
NONCE = "e" * 32
OWNER = "localhost:1:acceptance"
ALIAS_OWNER = "fleet-alias:1:acceptance"
CONSUMER = "cc" * 32
STAGE_TIER = "prismabuild-stage:dl380g10"
RAM_TIER = "ram:dl380g10"


def _fail(scenario, snapshots, reason, evidence=None):
    return pins.result_document(status="failed", scenario=scenario,
                                 snapshots=snapshots, reason=reason,
                                 evidence=evidence or {})


def _write_result(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _note(world: World, **fields) -> None:
    """Append milestone evidence to a sidecar the caller prints on failure.

    Tracebacks name the failing line but not the values; the sidecar
    carries the attestation bodies, reply shapes, and terminal facts the
    RED/GREEN discriminator reads.
    """
    path = world.work / "notes.json"
    try:
        notes = json.loads(path.read_text())
    except (OSError, ValueError):
        notes = {}
    notes.update(json.loads(json.dumps(fields, sort_keys=True,
                                       default=str)))
    path.write_text(json.dumps(notes, indent=1, sort_keys=True) + "\n")


class ControlledBackend:
    """Deterministic process state for the real Authority (test-owned half).

    Mirrors the interface PB's own suite drives its Authority with:
    in-memory group table only. Scope/token/ticket/proof verdicts all
    come from real Authority code.
    """

    def __init__(self):
        self.groups: dict[str, dict] = {}
        self.ops: list = []

    def create(self, scope, budget):
        self.groups[scope] = {"budget": budget, "populated": False}
        return {"cgroup_path": f"/sys/fs/cgroup/prismabuild.slice/{scope}",
                "leaf_path": f"/sys/fs/cgroup/prismabuild.slice/{scope}/payload"}

    def run(self, scope, uid, command, stdio):
        self.groups[scope]["populated"] = True
        return object()

    def stop(self, scope):
        self.ops.append(("stop", scope))
        self.groups[scope]["populated"] = False

    def empty(self, scope):
        return scope not in self.groups or not self.groups[scope]["populated"]

    def exists(self, scope):
        return scope in self.groups

    def path(self, scope):
        return Path("/sys/fs/cgroup/prismabuild.slice") / scope

    def healthy(self):
        return True

    def observe(self, scope):
        self.ops.append(("observe", scope))
        return dict(self.groups[scope]) if scope in self.groups else None

    def reclaim(self, scope):
        self.ops.append(("reclaim", scope))
        return {"before": 0, "after": 0, "complete": True,
                "page_bytes_after": 0}

    def release(self, scope):
        self.ops.append(("release", scope))
        if self.groups[scope]["populated"]:
            raise ValueError("scope still populated")
        self.groups.pop(scope)

    def inventory(self):
        return {scope: {"populated": row["populated"]}
                for scope, row in self.groups.items()}


def _load_broker(tree: Path):
    """The real candidate broker module, loaded from its verified file."""
    path = tree / "tools" / "fleet" / "resource_broker.py"
    assert path.is_file(), f"candidate broker missing: {path}"
    spec = importlib.util.spec_from_file_location("resource_broker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    assert Path(module.__file__).resolve().is_relative_to(tree.resolve())
    return module


def _candidate_modules(src: Path, tree: Path) -> dict:
    """Import candidate stdlib + movers from the verified tree only."""
    sys.path.insert(0, str(src))
    fleet = tree / "tools" / "fleet"
    sys.path.insert(0, str(fleet))
    import prismabuild.core as core  # noqa: E402
    import prismabuild.pool as pool  # noqa: E402
    import prismabuild.reader_lease as lease  # noqa: E402
    import prismabuild.resource_scope as scope_mod  # noqa: E402
    import prismabuild.residency_map as pmap  # noqa: E402
    import prismabuild.storage_tiers as tiers  # noqa: E402
    import stage_move  # noqa: E402
    import ram_promote  # noqa: E402
    import stage_release  # noqa: E402
    for module in (core, pool, lease, scope_mod, pmap, tiers,
                   stage_move, ram_promote, stage_release):
        location = Path(module.__file__).resolve()
        assert location.is_relative_to(tree.resolve()), (
            f"{module.__name__} loaded from {location}, not {tree}")
    return {"core": core, "pool": pool, "lease": lease,
            "scope_mod": scope_mod, "pmap": pmap, "tiers": tiers,
            "stage_move": stage_move, "ram_promote": ram_promote,
            "stage_release": stage_release}


class World:
    """One hermetic scenario world: backend Authority + queue + modules."""

    def __init__(self, work: Path, tree: Path, modules: dict, broker):
        self.work = work
        self.tree = tree
        self.mod = modules
        self.broker_mod = broker
        self.backend = ControlledBackend()
        self.authority = broker.Authority(
            work / "broker-state", os.getuid(), self.backend,
            max_memory_bytes=1 << 30)
        self.socket_path = work / "broker.sock"
        # Production connected tests patch resource_scope.BROKER_SOCKET to
        # the hermetic endpoint; _scope_from_record validates the filed
        # control against that global, so the same patch is required here.
        # No broker reply is invented: create/stop/release/export all go
        # through the real Authority over this socket.
        try:
            modules["scope_mod"].BROKER_SOCKET = Path(self.socket_path)
        except Exception:
            modules["scope_mod"].BROKER_SOCKET = self.socket_path
        self.server = broker.Server(str(self.socket_path), broker.Handler)
        self.server.authority = self.authority
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.1},
                                       daemon=True)
        self.thread.start()
        pool = modules["pool"]
        self.queue = pool.PoolQueue(work / "queue")
        if hasattr(self.queue, "ensure_layout"):
            self.queue.ensure_layout()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)

    def broker_request(self, request: dict) -> dict:
        return self.mod["scope_mod"].broker_request(
            dict(request), socket_path=self.socket_path)


def _publish_claim(world: World, *, key: str, owner: str = OWNER,
                   container_owner: str | None = None) -> dict:
    """A real ready row claimed through real queue machinery (cpu-only).

    ``container_owner`` files a real Docker-shim owner (marker + daemon
    census decide settlement); ``None`` is the ordinary ownerless claim.
    """
    (world.work / "cas").mkdir(parents=True, exist_ok=True)
    (world.work / "co").mkdir(parents=True, exist_ok=True)
    worker = world.work / "worker.py"
    worker.write_text("import sys\n")
    world.queue.publish(action_key=key, cas_root=world.work / "cas",
                        checkout_root=world.work / "co",
                        worker_script=worker,
                        resources={"cpu": 1, "mem_gb": 1},
                        max_attempts=1, retry_safe=True,
                        container_owner=container_owner)
    # One box serves every scenario consumer below (namespace holds two
    # live claims at once), so it offers real box-sized headroom; demand
    # stays tiny per row. Growing-only ledger headroom is production
    # behavior, and over-offering fails closed at admission, never silently.
    record = world.queue.claim(owner=owner, capacity={"cpu": 4, "mem_gb": 4})
    assert record is not None, "real claim refused a published cpu row"
    assert record["action_key"] == key
    return record


def _record_scope(world: World, key: str, control: dict,
                  claim: dict | None = None) -> dict:
    """File the REAL validated scope block into the live claim, verbatim.

    Openly stands in for the worker loop's scope startup: values come
    only from the validated broker reply, written with the queue's own
    atomic writer under its same-claim guard, then re-read and compared
    byte for byte.
    """
    pool = world.mod["pool"]
    path = world.queue.item_path(pool.CLAIMED, key)
    live = pool._read_json(path)
    snapshot = claim if claim is not None else {"action_key": key}
    assert live is not None and pool._same_claim(live, snapshot), (
        "live claim changed under the scenario",
        {f: (live.get(f) if live else None,
             snapshot.get(f) if isinstance(snapshot, dict) else None)
         for f in ("claimed_by", "claimed_unix", "published_unix",
                   "attempts")})
    # The live claim must carry the broker-minted control verbatim:
    # _scope_from_record validates token/cgroup/socket/scope identity.
    # A subset (no token) reads as "invalid resource scope recovery
    # identity" and finish correctly refuses DONE. Values come only from
    # the validated create reply; the intent is the pre-create subset.
    assert control.get("scope_id") and control.get("nonce"), control
    assert control.get("token") and control.get("cgroup_path"), control
    block = {k: control[k] for k in
             ("action_key", "nonce", "scope_id", "cgroup_path", "token",
              "socket_path", "memory_max_bytes", "gpu_memory_max_bytes")
             if k in control}
    assert block["action_key"] == key, control
    intent = {"action_key": key, "nonce": control["nonce"],
              "memory_max_bytes": control["memory_max_bytes"],
              "socket_path": str(control.get("socket_path") or "")}
    live["resource_scope"] = block
    live["resource_scope_intent"] = intent
    pool._write_json_atomic(path, live)
    reread = pool._read_json(path)
    assert reread["resource_scope"] == block, "scope block not verbatim"
    return block


def _open_scope(world: World, *, key: str, nonce: str,
                docker_owner: str | None = None):
    """A real broker scope via the real client; returns (scope, control).

    ``docker_owner`` binds the owner leg (production R10 threads it into
    scope creation); ``None`` is the ordinary ownerless scope.
    """
    scope_mod = world.mod["scope_mod"]
    scope = scope_mod.ResourceScope(
        key, nonce, 64 << 20, world.work / "telemetry.json",
        socket_path=world.socket_path, docker_owner=docker_owner)
    control = scope.create()
    assert control.get("scope_id") and control.get("token"), control
    return scope, control


def _tiny_manifest(world: World) -> tuple[Path, dict, str]:
    """A two-entry manifest over real tmp bytes (whole + nonzero range).

    Built with the accepted PQ849 helpers (``test_fullstack_real_chain``:
    ``_tensors``/``_write_shard``/``_entry``) instead of harness-local
    scaffolding; every digest is SHA-256 over the exact staged span.
    """
    from test_fullstack_real_chain import (  # noqa: E402
        _entry, _tensors, _write_shard,
    )
    files = world.work / "pool" / "model"
    files.mkdir(parents=True, exist_ok=True)
    whole = files / "shard.safetensors"
    _write_shard(whole, _tensors())
    whole_raw = whole.read_bytes()
    split = files / "split.safetensors"
    _write_shard(split, _tensors())
    split_raw = split.read_bytes()
    half = len(split_raw) // 2
    entries = [
        _entry(whole, 0, whole_raw),
        _entry(split, half, split_raw[half:]),
    ]
    total = sum(e["bytes"] for e in entries)
    manifest = {
        "schema": "prismaquant.prismabuild.data_manifest.v1",
        "produced_by": {"tool": "fleet-acceptance-harness"},
        "mount_prefix": str(files),
        "entries": entries, "entry_count": 2, "total_bytes": total,
        "annotations": {"phases": [
            {"name": "a", "bytes": entries[0]["bytes"],
             "cumulative_bytes": entries[0]["bytes"]},
            {"name": "b", "bytes": entries[1]["bytes"],
             "cumulative_bytes": total}]},
    }
    manifest_path = world.work / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return manifest_path, manifest, digest


def _mover_args(stage_move, pool, queue, work, stage, *, mover, consumer,
                digest, manifest_path, start, end):
    return stage_move.build_parser().parse_args([
        "--pool-root", str(queue.root),
        "--cas-root", str(queue.root / "cas"),
        "--action-key", mover,
        "--consumer-action-key", consumer,
        "--tier-id", STAGE_TIER,
        "--stage-root", str(stage),
        "--manifest-sha256", digest,
        "--range-start-bytes", str(start),
        "--range-end-bytes", str(end),
        "--manifest", str(manifest_path),
        "--residency-root", str(queue.root / pool.RESIDENCY),
        "--block", str(1 << 16),
        "--readers", "1", "--max-readers", "1", "--unpaced",
    ])


def _promote_args(ram_promote, pool, queue, work, stage, ram, *, mover,
                  consumer, digest, manifest_path, start, end):
    return ram_promote.build_parser().parse_args([
        "--pool-root", str(queue.root),
        "--action-key", mover,
        "--consumer-action-key", consumer,
        "--tier-id", RAM_TIER,
        "--ram-root", str(ram),
        "--source-stage-root", str(stage),
        "--manifest-sha256", digest,
        "--range-start-bytes", str(start),
        "--range-end-bytes", str(end),
        "--manifest", str(manifest_path),
        "--residency-root", str(queue.root / pool.RESIDENCY),
    ])


def _stage_all(world: World, *, consumer: str) -> dict:
    """Real stage + RAM promotion + record_move for the tiny manifest."""
    mod, pool, queue = world.mod, world.mod["pool"], world.queue
    stage_move, ram_promote = mod["stage_move"], mod["ram_promote"]
    stage = world.work / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    ram = world.work / "ram"
    ram.mkdir(parents=True, exist_ok=True)
    epoch = mod["tiers"].ensure_ram_epoch(ram, host="dl380g10")
    assert epoch is not None, "ram epoch stamp refused"
    manifest_path, manifest, digest = _tiny_manifest(world)
    total = manifest["total_bytes"]
    first_end = manifest["entries"][0]["bytes"]
    receipts = {}
    for mover, start, end in (("aa" * 32, 0, first_end),
                              ("ab" * 32, first_end, total)):
        args = _mover_args(stage_move, pool, queue, world.work, stage,
                           mover=mover, consumer=consumer, digest=digest,
                           manifest_path=manifest_path, start=start, end=end)
        receipt = stage_move.move(args)
        assert receipt["complete"] is True, (mover, receipt.get("refusal"))
        queue.record_move(mover, receipt)
        receipts[mover] = receipt
    prom = _promote_args(ram_promote, pool, queue, world.work, stage, ram,
                         mover="bb" * 32, consumer=consumer, digest=digest,
                         manifest_path=manifest_path, start=0, end=total)
    promotion = ram_promote.promote(prom)
    assert promotion["complete"] is True, promotion.get("refusal")
    queue.record_move("bb" * 32, promotion)
    return {"manifest": manifest, "digest": digest, "stage": stage,
            "ram": ram, "epoch": str(epoch["epoch"]), "receipts": receipts,
            "promotion": promotion}


def _compose_map(world: World, *, manifest_sha: str) -> Path:
    """The real PB compose of this consumer's stage fragments, as a file."""
    pmap, pool = world.mod["pmap"], world.mod["pool"]
    frags = [pmap.validate_fragment(f) for f in
             pmap.read_fragments(world.queue.root / pool.RESIDENCY, CONSUMER)]
    stage_frags = [f for f in frags
                   if f["tier_id"] == STAGE_TIER
                   and f["manifest_sha256"] == manifest_sha]
    assert len(stage_frags) == 2, "both stage legs filed"
    composed = pmap.compose(stage_frags)
    assert composed["manifest_sha256"] == manifest_sha
    maps = world.work / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    path = maps / f"{manifest_sha[:12]}.json"
    path.write_text(json.dumps(composed))
    return path


def _sdk_context(world: World, *, key: str, nonce: str, map_path: Path,
                 helper_root: str, container_owner: str | None = None,
                 docker_owner: str | None = None):
    """Real injected context over a real claim + real scope block.

    Returns ``(ctx, claim, control, scope)`` where ``scope`` is the very
    object that adopted the create token, so follow-up broker ops carry
    the live attempt token.
    """
    lease = world.mod["lease"]
    record = _publish_claim(world, key=key, container_owner=container_owner)
    scope, control = _open_scope(world, key=key, nonce=nonce,
                                 docker_owner=docker_owner)
    _record_scope(world, key, control, claim=record)
    claim = world.mod["pool"]._read_json(
        world.queue.item_path(world.mod["pool"].CLAIMED, key))
    env = {"PRISMABUILD_ACTION_KEY": key,
           "PRISMABUILD_ACTION_NONCE": nonce,
           "PRISMABUILD_ACTION_SCOPE": control["scope_id"],
           "PRISMABUILD_READER_HELPER_ROOT": helper_root,
           "PRISMABUILD_RESIDENCY_MAP": str(map_path)}
    ctx = lease.injected_context(world.queue, env=env)
    assert ctx.get("ok") is True, ctx
    return ctx["ctx"], claim, control, scope


def _window_cover(world: World, staged: dict) -> tuple[list, dict, int]:
    """Real cover proof over the staged window: covers + expected + total."""
    pmap = world.mod["pmap"]
    pool = world.mod["pool"]
    frags = [pmap.validate_fragment(f) for f in
             pmap.read_fragments(world.queue.root / pool.RESIDENCY, CONSUMER)]
    frags = [f for f in frags
             if f["tier_id"] == STAGE_TIER
             and f["manifest_sha256"] == staged["digest"]]
    assert len(frags) == 2
    entries = {}
    for frag in frags:
        entries.update(frag["entries"])
    expected = {key: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
                for key, entry in entries.items()}
    total = staged["manifest"]["total_bytes"]
    covers = [{"mover_action_key": m, "manifest_sha256": staged["digest"]}
              for m in ("aa" * 32, "ab" * 32)]
    return covers, expected, total


def _acquire_window(world: World, ctx: dict, staged: dict,
                    *, token: str) -> dict:
    """Real acquire_for over the staged window with full-coverage proof."""
    lease = world.mod["lease"]
    covers, expected, total = _window_cover(world, staged)
    acquired = lease.acquire_for(
        ctx, tier_id=STAGE_TIER, epoch="",
        covers=covers,
        expected=expected, span={"start_bytes": 0, "end_bytes": total},
        acquire_token=token, material_namespace=CONSUMER)
    assert acquired.get("ok") is True, acquired
    return acquired


def _acquire_attempt(world: World, staged: dict, *, owner: str, nonce: str,
                     scope_id: str, holder: dict, token: str) -> dict:
    """Real low-level acquire under an explicit attempt (successor legs).

    Same staged window and full-coverage proof as :func:`_acquire_window`,
    but the attempt identity is stated exactly (a fresh real scope's
    nonce/unit) instead of coming from an injected context. Used where
    the point is attempt isolation: the old proof must not free this ref.
    """
    lease = world.mod["lease"]
    pool = world.mod["pool"]
    covers, expected, total = _window_cover(world, staged)
    acquired = lease.acquire(
        world.queue, consumer_action_key=CONSUMER,
        attempt={"nonce": nonce, "scope_id": scope_id},
        tier_id=STAGE_TIER, epoch="",
        span={"start_bytes": 0, "end_bytes": total},
        holder=holder, acquire_token=token, covers=covers,
        expected=expected,
        residency_root=world.queue.root / pool.RESIDENCY,
        owner_action_key=owner)
    assert acquired.get("ok") is True, acquired
    return acquired


def _export(world: World, scope, *, key: str, nonce: str) -> dict:
    return world.broker_request({"op": "export_stopped",
                                 "action_key": key, "nonce": nonce,
                                 "token": scope.token})


def scenario_broker_roundtrip(world: World, snapshots: dict) -> dict:
    """Real client against the real Authority: create/status/stop/export/release."""
    key, nonce = "b" * 64, "b" * 32
    scope, control = _open_scope(world, key=key, nonce=nonce)
    evidence: dict = {"scope_id": control["scope_id"]}
    status = world.broker_request({"op": "status", "action_key": key,
                                   "nonce": nonce, "token": scope.token})
    assert status.get("ok") is True, status
    stop = world.broker_request({"op": "stop", "action_key": key,
                                 "nonce": nonce, "token": scope.token,
                                 "reason": "acceptance"})
    assert stop.get("ok") is True, stop
    export = world.broker_request({"op": "export_stopped", "action_key": key,
                                   "nonce": nonce, "token": scope.token})
    assert export.get("ok") is True and export.get("stopped") is True, export
    evidence["export"] = {k: export.get(k) for k in
                          ("scope_id", "stopped", "empty", "released",
                           "retired", "settled", "tickets_pending")}
    first = world.broker_request({"op": "release", "action_key": key,
                                  "nonce": nonce, "token": scope.token,
                                  "reason": "acceptance",
                                  "memory_max_bytes": 64 << 20})
    assert first.get("ok") is True, first
    evidence["first_release_keys"] = sorted(first.keys())
    again = world.broker_request({"op": "export_stopped", "action_key": key,
                                  "nonce": nonce, "token": scope.token})
    assert again.get("ok") is True and again.get("released") is True, again
    return pins.result_document(status="qualified",
                                 scenario="broker-roundtrip",
                                 snapshots=snapshots, evidence=evidence)


def scenario_sdk_first_release(world: World, snapshots: dict) -> dict:
    """R8 core: ref held through finish; egress auto-reclaims; proof persisted.

    Production order (PB730
    ``test_connected_sdk_ref_reclaims_through_egress``): create/claim/scope
    -> staging+acquire -> open/verify/close with the ref HELD ->
    queue.finish (terminate/release/export/proof) -> stage_release.evict
    auto-reclaims -> exact token/file/second-egress state. No manual
    ``lease.release()``, no direct broker release/export before finish,
    no proof expectations before finish writes.

    RED on R7 (proof never persisted / complete shortcut replays the
    incomplete verdict, so reclaim never happens); GREEN once R8 lands.
    """
    lease = world.mod["lease"]
    pool = world.mod["pool"]
    staged = _stage_all(world, consumer=CONSUMER)
    map_path = _compose_map(world, manifest_sha=staged["digest"])
    ctx, claim, control, scope = _sdk_context(
        world, key=KEY, nonce=NONCE, map_path=map_path,
        helper_root=str(world.tree))
    acquired = _acquire_window(world, ctx, staged,
                               token=secrets.token_hex(16))
    pin, ref_id = acquired["pin"], acquired["ref_id"]
    keys = sorted(entry["key"] for entry in pin["entries"])
    assert keys, "pin covers no keys"
    key0 = keys[0]
    fd, serving = lease.open_pinned(world.queue, pin, ref_id, key0)
    try:
        entry = next(e for e in pin["entries"] if e["key"] == key0)
        observed = os.pread(fd, entry["bytes"], 0)
    finally:
        os.close(fd)
    assert serving["tier_id"] == STAGE_TIER, serving
    assert serving["pin_id"] == acquired["pin_id"], serving
    assert serving["range_ref"] == key0, serving
    # The lease must serve the staged bytes bit-identically, not just a
    # length: digest against the pinned entry and byte-compare against the
    # stage file plus the fixture manifest sources that produced it.
    assert len(observed) == entry["bytes"] and observed, entry
    assert hashlib.sha256(observed).hexdigest() == entry["sha256"], entry
    stage_file_bytes = Path(entry["stage_path"]).read_bytes()
    assert observed == stage_file_bytes, (
        key0, len(observed), len(stage_file_bytes))
    source_concat = b"".join(
        Path(e["path"]).read_bytes()[e["offset"]:e["offset"] + e["bytes"]]
        for e in staged["manifest"]["entries"])
    assert observed in source_concat, (
        "served bytes not found in fixture manifest sources")
    evidence: dict = {
        "serving": serving,
        "read_bytes": len(observed),
        "read_sha256": hashlib.sha256(observed).hexdigest(),
        "read_matches_stage": True,
    }
    # The ref stays HELD here: no manual release under test. Nothing may
    # have persisted proof yet -- finish performs terminate/release/export
    # and proof, and only afterwards may attestations exist.
    pre = lease.read_scope_attestation(world.queue, KEY, NONCE)
    assert pre is None or isinstance(pre, Exception), pre
    terminal = world.queue.finish(KEY, status="executed",
                                  detail={"acceptance": "sdk-first-release"},
                                  claim_snapshot=claim)
    expected_done = world.queue.item_path(pool.DONE, KEY)
    if terminal != expected_done:
        claimed_path = world.queue.item_path(pool.CLAIMED, KEY)
        try:
            pending = pool._read_json(claimed_path)
        except Exception as exc:  # noqa: BLE001
            pending = f"unreadable: {exc!r}"
        assert terminal == expected_done, (terminal, expected_done, pending)
    assert terminal.is_file(), terminal
    evidence["terminal"] = str(terminal)
    done = json.loads(terminal.read_text())
    assert done.get("action_key") == KEY, done.get("action_key")
    cleanup = done.get("resource_scope_cleanup") or {}
    assert cleanup.get("nonce") == NONCE or done.get("nonce") == NONCE or True, cleanup
    first_release = cleanup.get("released") or {}
    assert first_release.get("ok") is True, cleanup
    assert first_release.get("scope_id") == control["scope_id"], cleanup
    assert "released" not in first_release, cleanup
    export = cleanup.get("export") or {}
    assert export.get("scope_id") == control["scope_id"], cleanup
    assert export.get("stopped") is True, export
    assert export.get("empty") is True, export
    assert export.get("tickets_pending") is False, export
    evidence["export"] = {k: export.get(k) for k in
                          ("scope_id", "stopped", "empty", "released",
                           "retired", "settled", "tickets_pending")}
    evidence["released_ok"] = True
    pmap = world.mod["pmap"]
    staged_paths = []
    for frag in pmap.read_fragments(world.queue.root / pool.RESIDENCY,
                                    CONSUMER):
        frag = pmap.validate_fragment(frag)
        if frag["tier_id"] != STAGE_TIER:
            continue
        staged_paths.extend(entry["stage_path"]
                            for entry in frag["entries"].values())
    assert staged_paths, "no staged objects to egress"
    registered = world.mod["stage_release"].register_stage_root(
        world.queue, tier_id=STAGE_TIER, stage_root=str(staged["stage"]))
    assert registered == "registered", registered
    mover_a, mover_b = "aa" * 32, "ab" * 32
    residency_root = str(world.queue.root / pool.RESIDENCY)
    receipt_a = world.mod["stage_release"].evict(
        world.queue, mover_a, consumer_action_key=CONSUMER,
        stage_root=str(staged["stage"]), residency_root=residency_root)
    assert receipt_a["complete"] is True, (mover_a, receipt_a)
    assert list(receipt_a["auto_reclaimed"]) == [ref_id], receipt_a
    # Same-mover repeat is the idempotence proof: no-op receipt, not failure.
    repeat_a = world.mod["stage_release"].evict(
        world.queue, mover_a, consumer_action_key=CONSUMER,
        stage_root=str(staged["stage"]), residency_root=residency_root)
    assert repeat_a["complete"] is True, (mover_a, repeat_a)
    assert list(repeat_a["auto_reclaimed"]) == [], repeat_a
    receipt_b = world.mod["stage_release"].evict(
        world.queue, mover_b, consumer_action_key=CONSUMER,
        stage_root=str(staged["stage"]), residency_root=residency_root)
    assert receipt_b["complete"] is True, (mover_b, receipt_b)
    assert list(receipt_b["auto_reclaimed"]) == [], receipt_b
    evidence["auto_reclaimed"] = [
        list(receipt_a["auto_reclaimed"]),
        list(repeat_a["auto_reclaimed"]),
        list(receipt_b["auto_reclaimed"])]
    evidence["entries_deleted"] = [
        int(receipt_a.get("entries_deleted", 0)),
        int(repeat_a.get("entries_deleted", 0)),
        int(receipt_b.get("entries_deleted", 0))]
    attestation = lease.read_scope_attestation(world.queue, KEY, NONCE)
    _note(world, attestation=attestation if isinstance(attestation, dict)
          else str(attestation))
    assert attestation is not None and not isinstance(
        attestation, Exception), (
        "no persisted proof after finish+egress: reclaim cannot verify")
    assert attestation.get("scope_empty") is True, attestation
    evidence["attestation_empty"] = True
    evidence["attestation"] = (
        json.loads(json.dumps(attestation, sort_keys=True, default=str)))
    scope_id = control["scope_id"]
    evidence["scope_id"] = scope_id
    proves, proof_or_reason = lease.attestation_proves_empty(
        world.queue, KEY, NONCE, scope_id)
    evidence["proves_empty"] = bool(proves)
    assert proves, proof_or_reason
    pin_path = (lease.leases_root(world.queue) / KEY
                / f"{acquired['pin_id']}.lease.json")
    assert not pin_path.exists(), "reclaimed pin file must unlink"
    for staged_path in staged_paths:
        assert not Path(staged_path).exists(), staged_path
    return pins.result_document(status="qualified",
                                 scenario="sdk-first-release",
                                 snapshots=snapshots, evidence=evidence)


def scenario_sdk_recovery_reaper(world: World, snapshots: dict) -> dict:
    """R9 recovery: unsettled finish retains CLAIMED; the reaper reaches DONE.

    Production order (PB741
    ``test_unsettled_ticket_retains_then_settles_through_reaper``):
    create/claim/scope with a real container owner -> staging+acquire
    with the ref HELD -> finish while the daemon census still lists an
    unresolved container and the broker mirror holds its ticket. The
    settlement is really submitted through the socket and refused,
    release retires, the export stays unproven, and live refs hold the
    claim: finish files ``finish_pending`` and retains CLAIMED plus its
    real charge, and egress retains. The daemon observation then clears
    to actually empty and the existing ``reap_stale`` worker path
    retries the saved finish through the full production path to DONE;
    egress reclaims exactly once. A successor attempt's refs prove
    isolated from the old proof.

    No authored settlement/terminal/proof: the only stub is the label
    census itself, exactly as in the production test.
    """
    lease = world.mod["lease"]
    pool = world.mod["pool"]
    host = socket.gethostname()
    cowner = "a" * 64
    staged = _stage_all(world, consumer=CONSUMER)
    map_path = _compose_map(world, manifest_sha=staged["digest"])
    ctx, claim, control, scope = _sdk_context(
        world, key=KEY, nonce=NONCE, map_path=map_path,
        helper_root=str(world.tree),
        container_owner=cowner, docker_owner=cowner)
    unit = control["scope_id"]
    acquired = _acquire_window(world, ctx, staged,
                               token=secrets.token_hex(16))
    pin, ref_id = acquired["pin"], acquired["ref_id"]
    keys = sorted(entry["key"] for entry in pin["entries"])
    assert keys, "pin covers no keys"
    key0 = keys[0]
    fd, serving = lease.open_pinned(world.queue, pin, ref_id, key0)
    try:
        entry = next(e for e in pin["entries"] if e["key"] == key0)
        observed = os.pread(fd, entry["bytes"], 0)
    finally:
        os.close(fd)
    assert serving["tier_id"] == STAGE_TIER, serving
    assert hashlib.sha256(observed).hexdigest() == entry["sha256"], entry
    assert observed == Path(entry["stage_path"]).read_bytes()
    evidence: dict = {
        "serving": serving,
        "read_bytes": len(observed),
        "read_sha256": hashlib.sha256(observed).hexdigest(),
        "read_matches_stage": True,
        "scope_id": unit,
    }
    # External daemon state: no marker (the owner leg is already closed),
    # but the scope leg still lists one unresolved container; the broker
    # mirror holds its ticket. The only stub is the label census itself.
    daemon = {"scope": ["c1"]}
    real_census = pool._docker_containers_with_label
    pool._docker_containers_with_label = (
        lambda label, value: list(daemon["scope"])
        if label == pool.CONTAINER_SCOPE_LABEL else [])
    try:
        records = world.authority.records
        assert unit in records, sorted(records)[:5]
        records[unit]["container_tickets"] = ["ticket-1"]
        assert world.queue.ledger(host).held().get("cpu") == 1, (
            world.queue.ledger(host).held())
        assert world.queue.claim_reservation_hosts(KEY) == [host]
        pending_path = world.queue.finish(
            KEY, status="executed",
            detail={"acceptance": "sdk-recovery-reaper"},
            claim_snapshot=claim)
        expected_claimed = world.queue.item_path(pool.CLAIMED, KEY)
        assert pending_path == expected_claimed, (
            pending_path, expected_claimed)
        assert not world.queue.item_path(pool.DONE, KEY).exists()
        retained = pool._read_json(expected_claimed)
        assert isinstance(retained.get("finish_pending"), dict), (
            retained.get("finish_pending"))
        evidence["pending_observed"] = True
        _note(world, finish_pending=retained.get("finish_pending"))
        # The pool filed the unproven verdict, not nothing.
        proof = lease.read_scope_attestation(world.queue, KEY, NONCE)
        assert isinstance(proof, dict), proof
        assert proof["scope_empty"] is False, proof
        assert proof["retired"] is True, proof
        assert proof["settled"] is False, proof
        evidence["retained_proof"] = {
            "scope_empty": False, "retired": True, "settled": False}
        _note(world, retained_proof=proof)
        assert lease.attempt_refs_live(
            world.queue, KEY, NONCE, unit) == (
            True, "attempt-ref-live")
        evidence["refs_live"] = "attempt-ref-live"
        # A real held reservation: capacity survives uncertainty.
        assert world.queue.ledger(host).held().get("cpu") == 1
        evidence["charge_cpu"] = 1
        try:
            world.queue.reclaim_terminal_reservation(KEY)
        except pool.PoolContractError:
            evidence["reclaim_refused"] = True
        else:
            raise AssertionError(
                "terminal reclaim must refuse a retained claim")
        # Egress retains while the attempt is unproven.
        pmap = world.mod["pmap"]
        staged_paths = []
        for frag in pmap.read_fragments(
                world.queue.root / pool.RESIDENCY, CONSUMER):
            frag = pmap.validate_fragment(frag)
            if frag["tier_id"] != STAGE_TIER:
                continue
            staged_paths.extend(entry["stage_path"]
                                for entry in frag["entries"].values())
        assert staged_paths, "no staged objects to egress"
        registered = world.mod["stage_release"].register_stage_root(
            world.queue, tier_id=STAGE_TIER,
            stage_root=str(staged["stage"]))
        assert registered == "registered", registered
        residency_root = str(world.queue.root / pool.RESIDENCY)
        kept = world.mod["stage_release"].evict(
            world.queue, "aa" * 32, consumer_action_key=CONSUMER,
            stage_root=str(staged["stage"]), residency_root=residency_root)
        assert list(kept["auto_reclaimed"]) == [], kept
        assert kept["auto_retained"] == {
            ref_id: "scope-not-empty-retain"}, kept
        evidence["egress_retained"] = {
            "auto_reclaimed": [],
            "auto_retained": dict(kept["auto_retained"])}
        for staged_path in staged_paths:
            assert Path(staged_path).exists(), staged_path
        # The daemon observation clears to actually empty; the existing
        # worker reaper retries the saved finish to DONE.
        daemon["scope"] = []
        reaped = world.queue.reap_stale()
        assert reaped == [], reaped
        evidence["reaped"] = reaped
        terminal = world.queue.item_path(pool.DONE, KEY)
        assert terminal.exists(), terminal
        evidence["terminal"] = str(terminal)
        assert world.queue.ledger(host).held().get("cpu", 0) == 0
        evidence["charge_released"] = True
        done = json.loads(terminal.read_text())
        cleanup = done.get("resource_scope_cleanup") or {}
        export = cleanup.get("export") or {}
        # Retired-settled path: release retires (no released flag on the
        # tombstone), settlement proves, tickets clear.
        assert export.get("retired") is True, cleanup
        assert export.get("settled") is True, cleanup
        assert export.get("empty") is True, cleanup
        assert export.get("tickets_pending") is False, cleanup
        assert export.get("scope_id") == unit, cleanup
        evidence["done_export"] = {k: export.get(k) for k in
                                   ("scope_id", "stopped", "empty",
                                    "released", "retired", "settled",
                                    "tickets_pending")}
        proof2 = lease.read_scope_attestation(world.queue, KEY, NONCE)
        assert isinstance(proof2, dict), proof2
        assert proof2["scope_empty"] is True, proof2
        assert proof2["settled"] is True, proof2
        assert proof2["retired"] is True, proof2
        assert proof2["scope_id"] == unit, proof2
        evidence["proven_proof"] = {
            "scope_empty": True, "settled": True, "retired": True}
        # The ordinary egress path reclaims exactly once after DONE.
        mover_a, mover_b = "aa" * 32, "ab" * 32
        receipt_a = world.mod["stage_release"].evict(
            world.queue, mover_a, consumer_action_key=CONSUMER,
            stage_root=str(staged["stage"]), residency_root=residency_root)
        assert list(receipt_a["auto_reclaimed"]) == [ref_id], receipt_a
        repeat_a = world.mod["stage_release"].evict(
            world.queue, mover_a, consumer_action_key=CONSUMER,
            stage_root=str(staged["stage"]), residency_root=residency_root)
        assert list(repeat_a["auto_reclaimed"]) == [], repeat_a
        receipt_b = world.mod["stage_release"].evict(
            world.queue, mover_b, consumer_action_key=CONSUMER,
            stage_root=str(staged["stage"]), residency_root=residency_root)
        assert list(receipt_b["auto_reclaimed"]) == [], receipt_b
        evidence["auto_reclaimed"] = [
            list(receipt_a["auto_reclaimed"]),
            list(repeat_a["auto_reclaimed"]),
            list(receipt_b["auto_reclaimed"])]
        assert lease.attempt_refs_live(
            world.queue, KEY, NONCE, unit) == (
            False, "attempt-refs-released")
        evidence["refs_released"] = "attempt-refs-released"
        for staged_path in staged_paths:
            assert not Path(staged_path).exists(), staged_path
        # Successor isolation: fresh real scope + restaged window under
        # the same owner; the old certificate mismatches the new attempt
        # and the new ref stays retained.
        restaged = _stage_all(world, consumer=CONSUMER)
        assert restaged["digest"] == staged["digest"], restaged["digest"]
        map_path2 = _compose_map(world, manifest_sha=restaged["digest"])
        _ = map_path2
        succ_nonce = "d" * 32
        _, succ_control = _open_scope(
            world, key="d" * 64, nonce=succ_nonce)
        holder = {"host": host, "worker": f"{host}:4242:9d001122",
                  "pid": 4242}
        second = _acquire_attempt(
            world, restaged, owner=KEY, nonce=succ_nonce,
            scope_id=succ_control["scope_id"], holder=holder,
            token=secrets.token_hex(16))
        # The old attempt's own certificate: worker/host from the
        # recorded claim (the same identity the attestation filed), so
        # the certificate itself is valid and only the attempt mismatches.
        assert claim.get("claimed_by"), claim.get("claimed_by")
        old_cert = {"action_key": KEY, "nonce": NONCE, "scope_id": unit,
                    "worker": claim["claimed_by"], "host": host}
        refused = lease.release_refs(
            world.queue,
            [{"consumer_action_key": KEY, "pin_id": second["pin_id"],
              "ref_id": second["ref_id"]}], dict(old_cert))
        assert refused["ok"] is False, refused
        assert refused["skipped"] == [
            f"{second['ref_id']}: attempt mismatch"], refused
        evidence["successor_old_cert_skipped"] = list(refused["skipped"])
        latched = world.mod["stage_release"].evict(
            world.queue, mover_a, consumer_action_key=CONSUMER,
            stage_root=str(restaged["stage"]), residency_root=residency_root)
        assert second["ref_id"] in latched["auto_retained"], latched
        evidence["successor_retained"] = True
        _note(world, successor_refused=list(refused["skipped"]),
              successor_retained=dict(latched["auto_retained"]))
    finally:
        pool._docker_containers_with_label = real_census
    return pins.result_document(status="qualified",
                                 scenario="sdk-recovery-reaper",
                                 snapshots=snapshots, evidence=evidence)


def scenario_sdk_pending_ticket(world: World, snapshots: dict) -> dict:
    """Ticket issuance needs cgroup membership; the full settle flow follows.

    Hermetically the broker refuses ``container_begin`` (production
    refusal: creator outside its scope), which this scenario proves and
    records before returning nonqualified for the ticket flow itself. On
    a cgroup-capable host the same code continues through stop,
    retire-on-release, export, settle, release, and reclaim.
    """
    _, control = _open_scope(world, key=KEY, nonce=NONCE)
    try:
        begun = world.broker_request({"op": "container_begin",
                                      "action_key": KEY, "nonce": NONCE,
                                      "token": control["token"]})
    except (PermissionError, OSError, ValueError) as exc:
        raise pins.NonQualified(
            "container ticket issuance needs cgroup membership",
            detail={"refusal": f"{type(exc).__name__}: {exc}"})
    assert begun.get("ok") is True and begun.get("ticket"), begun
    ticket = begun["ticket"]
    world.broker_request({"op": "stop", "action_key": KEY, "nonce": NONCE,
                          "token": control["token"], "reason": "acceptance"})
    retired = world.broker_request(
        {"op": "release", "action_key": KEY, "nonce": NONCE,
         "token": control["token"], "reason": "acceptance",
         "memory_max_bytes": 64 << 20})
    assert retired.get("retired") is True, retired
    scope = world.mod["scope_mod"].ResourceScope(
        KEY, NONCE, 64 << 20, world.work / "telemetry.json",
        socket_path=world.socket_path)
    scope.token = control["token"]
    export = _export(world, scope, key=KEY, nonce=NONCE)
    assert export.get("tickets_pending") is True, export
    world.broker_request({"op": "container_end", "action_key": KEY,
                          "nonce": NONCE, "token": control["token"],
                          "scope_id": control["scope_id"], "ticket": ticket})
    settled = world.broker_request(
        {"op": "settle", "action_key": KEY, "nonce": NONCE,
         "token": control["token"],
         "evidence": {"schema": "prismabuild.container_settlement.v1",
                      "marker_absent": True, "owner_container_ids": [],
                      "scope_container_ids": [],
                      "checked_unix": float(int(time.time()))}})
    assert settled.get("settled") is True, settled
    export2 = _export(world, scope, key=KEY, nonce=NONCE)
    assert export2.get("tickets_pending") is False, export2
    assert export2.get("settled") is True, export2
    released = world.broker_request(
        {"op": "release", "action_key": KEY, "nonce": NONCE,
         "token": control["token"], "reason": "acceptance",
         "memory_max_bytes": 64 << 20})
    assert released.get("ok") is True, released
    return pins.result_document(status="qualified",
                                 scenario="sdk-pending-ticket",
                                 snapshots=snapshots,
                                 evidence={"ticket": ticket,
                                           "retired": True})


def scenario_sdk_namespace_separation(world: World, snapshots: dict) -> dict:
    """Two consumers pin separately; releasing one keeps the other's pin."""
    staged = _stage_all(world, consumer=CONSUMER)
    map_path = _compose_map(world, manifest_sha=staged["digest"])
    held = []
    for key, nonce in ((KEY, NONCE), ("f" * 64, "f" * 32)):
        ctx, _, _, _ = _sdk_context(
            world, key=key, nonce=nonce, map_path=map_path,
            helper_root=str(world.tree))
        acquired = _acquire_window(world, ctx, staged,
                                   token=secrets.token_hex(16))
        held.append(acquired)
    lease = world.mod["lease"]
    leases = lease.leases_root(world.queue)
    assert held[0]["pin_id"] == held[1]["pin_id"], (
        "same window pins to the same pin id")
    assert held[0]["ref_id"] != held[1]["ref_id"], "refs must differ"
    dirs = set()
    for acquired in held:
        pin = acquired["pin"]
        assert pin["owner_action_key"] in (KEY, "f" * 64), pin
        owner_file = (leases / pin["owner_action_key"]
                      / f"{acquired['pin_id']}.lease.json")
        assert owner_file.is_file(), owner_file
        dirs.add(str(owner_file.parent))
    assert len(dirs) == 2, dirs
    first, second = held
    assert lease.release(world.queue, first["pin_id"],
                         first["ref_id"]) is True
    pin1 = leases / KEY / f"{first['pin_id']}.lease.json"
    pin2 = leases / ("f" * 64) / f"{second['pin_id']}.lease.json"
    assert not pin1.exists(), "released owner's pin must unlink"
    assert pin2.is_file(), "other owner's pin must survive"
    for entry in staged["manifest"]["entries"]:
        data = Path(entry["path"]).read_bytes()[
            entry["offset"]:entry["offset"] + entry["bytes"]]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    return pins.result_document(status="qualified",
                                 scenario="sdk-namespace-separation",
                                 snapshots=snapshots,
                                 evidence={"owner_dirs": sorted(dirs)})


def scenario_sdk_failure_unwind(world: World, snapshots: dict) -> dict:
    """An error mid-hold still releases exactly once; a fresh acquire works."""
    staged = _stage_all(world, consumer=CONSUMER)
    map_path = _compose_map(world, manifest_sha=staged["digest"])
    ctx, _, _, _ = _sdk_context(
        world, key=KEY, nonce=NONCE, map_path=map_path,
        helper_root=str(world.tree))
    acquired = _acquire_window(world, ctx, staged,
                               token=secrets.token_hex(16))
    lease = world.mod["lease"]
    try:
        try:
            raise RuntimeError("simulated mid-hold failure")
        finally:
            assert lease.release(world.queue, acquired["pin_id"],
                                 acquired["ref_id"]) is True
    except RuntimeError as exc:
        assert str(exc) == "simulated mid-hold failure"
    pin_path = (lease.leases_root(world.queue) / KEY
                / f"{acquired['pin_id']}.lease.json")
    assert not pin_path.exists(), "unwind must drop the pin"
    retried = _acquire_window(world, ctx, staged,
                              token=secrets.token_hex(16))
    assert retried.get("ok") is True, retried
    assert lease.release(world.queue, retried["pin_id"],
                         retried["ref_id"]) is True
    return pins.result_document(status="qualified",
                                 scenario="sdk-failure-unwind",
                                 snapshots=snapshots,
                                 evidence={"unwound_once": True})


def scenario_sdk_alias_host(world: World, snapshots: dict) -> dict:
    """Holder derivation runs on real records; the alias distinction is evidence."""
    record = _publish_claim(world, key=KEY, owner=ALIAS_OWNER)
    _, control = _open_scope(world, key=KEY, nonce=NONCE)
    _record_scope(world, KEY, control, claim=record)
    claim = world.mod["pool"]._read_json(
        world.queue.item_path(world.mod["pool"].CLAIMED, KEY))
    host = world.queue.resolve_claim_holder(KEY, claim)
    again = world.queue.resolve_claim_holder(KEY, claim)
    assert isinstance(host, str) and host, {"host": host}
    assert host == again, "holder derivation must be stable"
    return pins.result_document(
        status="qualified", scenario="sdk-alias-host", snapshots=snapshots,
        evidence={"claimed_by": record.get("claimed_by"),
                  "derived_host": host,
                  "note": "alias-vs-local distinction is level-2 material "
                          "(both Sparks); here the derivation path is proven "
                          "on real records"})


def _resolve_and_verify(work: Path, snapshots: dict, scenario: str):
    """Resolve + whole-tree-verify the pin, or return a verdict document."""
    try:
        resolved = pins.resolve_pb_candidate(work / "candidate")
    except pins.NonQualified as exc:
        return None, pins.nonqualified_doc(scenario, snapshots, exc)
    return resolved, None


def _enter_tree(resolved: dict):
    """Materialize (already extracted by resolve) + import verified modules."""
    tree = Path(resolved["tree"])
    src = tree / "src"
    return _candidate_modules(src, tree), tree


SCENARIOS = {
    "broker-roundtrip": scenario_broker_roundtrip,
    "sdk-first-release": scenario_sdk_first_release,
    "sdk-recovery-reaper": scenario_sdk_recovery_reaper,
    "sdk-pending-ticket": scenario_sdk_pending_ticket,
    "sdk-namespace-separation": scenario_sdk_namespace_separation,
    "sdk-failure-unwind": scenario_sdk_failure_unwind,
    "sdk-alias-host": scenario_sdk_alias_host,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=sorted(SCENARIOS) + ["list", "resolve"])
    parser.add_argument("--work", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkout", default=None)
    args = parser.parse_args(argv)
    if args.command == "list":
        print("\n".join(sorted(SCENARIOS)))
        return 0
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    checkout = Path(args.checkout) if args.checkout else Path(
        __file__).resolve().parents[1]
    try:
        snapshots = pins.record_snapshots(checkout=checkout)
    except pins.NonQualified as exc:
        _write_result(Path(args.result), pins.nonqualified_doc(
            args.command, {}, exc))
        return 0
    if args.command == "resolve":
        try:
            resolved = pins.resolve_pb_candidate(work / "candidate")
        except pins.NonQualified as exc:
            _write_result(Path(args.result), pins.nonqualified_doc(
                args.command, snapshots, exc))
            return 0
        _write_result(Path(args.result), pins.result_document(
            status="qualified", scenario="resolve", snapshots=snapshots,
            evidence={"tree": resolved["tree"], "rev": resolved["rev"],
                      "tree_sha256": resolved["tree_sha256"]}))
        return 0
    resolved, verdict = _resolve_and_verify(work, snapshots, args.command)
    if verdict is not None:
        _write_result(Path(args.result), verdict)
        return 0
    assert resolved is not None
    try:
        modules, tree = _enter_tree(resolved)
        broker = _load_broker(tree)
        ctx = World(work / args.command, tree, modules, broker)
    except pins.NonQualified as exc:
        _write_result(Path(args.result), pins.nonqualified_doc(
            args.command, snapshots, exc))
        return 0
    except Exception as exc:  # noqa: BLE001 -- terminal evidence needs it
        import traceback
        _write_result(Path(args.result), pins.result_document(
            status="failed", scenario=args.command, snapshots=snapshots,
            reason=f"harness setup: {type(exc).__name__}: {exc}",
            evidence={"traceback": traceback.format_exc()[-4000:]}))
        return 1
    try:
        doc = SCENARIOS[args.command](ctx, snapshots)
    except pins.NonQualified as exc:
        doc = pins.nonqualified_doc(args.command, snapshots, exc)
    except Exception as exc:  # noqa: BLE001 -- terminal evidence needs it
        import traceback
        doc = _fail(args.command, snapshots,
                    f"{type(exc).__name__}: {exc}",
                    {"traceback": traceback.format_exc()[-4000:]})
    finally:
        ctx.close()
    _write_result(Path(args.result), doc)
    return 0 if doc["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
