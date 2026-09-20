"""live_reader_qualify.main TRUE positive on a real isolated claim (PQ #869).

R5 proved only refusals: every main-path test stopped at
``lease-context-unavailable`` because no launch identity existed. This
module builds one for real, with the accepted PQ864 fleet-acceptance
machinery (``fleet_acceptance_runner``): a real PoolQueue publish/claim on
a tmp queue, a real broker Authority over a hermetic Unix socket, the real
validated scope control recorded verbatim into the live claim — exactly
what ``reader_lease.injected_context`` requires, nothing forged.

The bytes are real too: a tiny safetensors calibration artifact staged
through the real SDK writers (fragment/material/compose/overlay) under the
claimed action key, the composed map at the production queue layout
``<queue>/residency/<key>.map.json``, and the sealed helper root named as
a view of the reviewed installed distribution (provenance proven by
``inject_installed_sdk_for_tests`` before the view is built).

Only platform mount observations are simulated (``_mountstats`` /
``_mount_of``), per the qualification contract. The tool keeps its own
calibration→lease→open→release path; no acquire/open/release is replaced.

Covers: the true positive (exit 0 with exact-attempt held refs and clean
post-release census, explicit SSD-only), wrong-draw rejection after a real
read, a foreign attempt's ref never counting as ours, a tainted census
refusing while open, RAM-first serving, and the typed RAM-availability
fallback accepted only on the resolver's own recorded evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fleet_acceptance_runner as far  # noqa: E402
from test_live_reader_main import (  # noqa: E402
    MOUNT, N_SAMPLES, SEQLEN, _artifact, _read_argv, _run_main, _stats,
)
import tools.live_reader_qualify as probe  # noqa: E402

KEY = "b" * 64
NONCE = "a" * 32
MOVER = "c" * 64
RAM_MOVER = "f" * 64
MANIFEST = "9" * 64
STAGE_TIER = "prismabuild-stage:fixture"
RAM_HOST = "fixturesh"


@pytest.fixture(autouse=True)
def _isolated_probe_env(monkeypatch):
    """Each main run starts with no identity, map, policy, or injection."""

    from prismaquant.residency_map import reset_residency_resolver_for_tests
    from prismaquant.staged_lease import (
        clear_injected_sdk_for_tests,
        set_lease_helper_root,
    )
    from prismaquant.staged_tier_policy import (
        deactivate_staged_tier_policy_for_tests,
    )

    def scrub() -> None:
        for name in ("PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE",
                     "PRISMABUILD_ACTION_SCOPE",
                     "PRISMABUILD_READER_HELPER_ROOT",
                     "PRISMABUILD_RESIDENCY_MAP",
                     "PRISMABUILD_RESIDENCY_TIERS_DIR"):
            monkeypatch.delenv(name, raising=False)
        set_lease_helper_root(None)
        clear_injected_sdk_for_tests()
        reset_residency_resolver_for_tests()
        deactivate_staged_tier_policy_for_tests()

    scrub()
    yield
    scrub()


@pytest.fixture(scope="session")
def pinned_tree(tmp_path_factory):
    """The pinned PB candidate tree (accepted 864 resolution), for the broker.

    A missing dependency is a SKIP with the reason, never a silent green.
    """
    import fleet_acceptance_pins as pins

    try:
        return pins.resolve_pb_candidate(tmp_path_factory.mktemp("pb-pin"))
    except pins.NonQualified as exc:
        pytest.skip(f"PB candidate unavailable: {exc.reason}")


@pytest.fixture()
def world(tmp_path, pinned_tree, monkeypatch):
    """The accepted hermetic world over the INSTALLED pinned distribution.

    The 864 runner loads candidate modules from the verified tree because it
    tests that tree; the qualification tool runs the installed distribution,
    so every ``prismabuild.*`` module here is the installed one and the
    sealed-tree coherence check holds. The real broker Authority, socket
    server, claim machinery and scope client are the 864 machinery itself.
    """

    import prismabuild.core as core
    import prismabuild.pool as pool
    import prismabuild.reader_lease as lease
    import prismabuild.residency_map as pmap
    import prismabuild.resource_scope as scope_mod
    import prismabuild.storage_tiers as tiers

    tree = Path(pinned_tree["tree"])
    broker = far._load_broker(tree)
    modules = {"core": core, "pool": pool, "lease": lease,
               "scope_mod": scope_mod, "pmap": pmap, "tiers": tiers}
    saved_socket = getattr(scope_mod, "BROKER_SOCKET", None)
    w = far.World(tmp_path, tree, modules, broker)
    try:
        yield w
    finally:
        w.close()
        if saved_socket is None:
            if hasattr(scope_mod, "BROKER_SOCKET"):
                try:
                    del scope_mod.BROKER_SOCKET
                except AttributeError:
                    pass
        else:
            scope_mod.BROKER_SOCKET = saved_socket


@pytest.fixture()
def helper_root(tmp_path):
    """The reviewed install named as the sealed generation root.

    ``inject_installed_sdk_for_tests`` proves the install's provenance
    (non-editable git install at the pinned commit, no shadow). The symlink
    view then presents that one distribution in the production generation
    layout ``<root>/src/prismabuild``, so ``main`` runs the real sealed-tree
    resolution (``_sdk_from_tree``) rather than any test-only injection.
    """

    from prismaquant.staged_lease import inject_installed_sdk_for_tests

    module = inject_installed_sdk_for_tests()
    package = Path(module.__file__).resolve().parent
    view = tmp_path / "helper"
    (view / "src").symlink_to(package.parent, target_is_directory=True)
    return view


def _mount_patches(monkeypatch):
    """Simulate only the platform mount observations (contract-allowed)."""

    monkeypatch.setattr(probe, "_mountstats", lambda: _stats(41))
    monkeypatch.setattr(probe, "_mount_of", lambda _path: MOUNT)


def _identity_env(world, monkeypatch, map_path, control):
    """Launch identity = the real claim's real broker control, verbatim."""

    monkeypatch.setenv("PRISMABUILD_ACTION_KEY", KEY)
    monkeypatch.setenv("PRISMABUILD_ACTION_NONCE", NONCE)
    monkeypatch.setenv("PRISMABUILD_ACTION_SCOPE", control["scope_id"])
    monkeypatch.setenv("PRISMABUILD_RESIDENCY_MAP", str(map_path))


def _stage(world, work: Path, artifact: Path, digest: str,
           *, ram: dict | None = None):
    """Real SDK-writer staging under the claimed key; map at queue layout.

    The material namespace is the claimed action key itself, matching
    ``acquire_for``'s default (the reading action is the consumer whose
    manifest named the bytes). The map lands at
    ``<queue>/residency/<key>.map.json`` so the strict reader's derived
    queue root is this isolated queue.
    """

    lease = world.mod["lease"]
    pool = world.mod["pool"]
    pmap = world.mod["pmap"]
    root = world.queue.root / pool.RESIDENCY
    stage = (work / "stage").resolve()
    stage.mkdir(parents=True, exist_ok=True)
    staged = stage / "calib.safetensors"
    staged.write_bytes(artifact.read_bytes())
    declared_dir = (work / "declared").resolve()
    declared_dir.mkdir(parents=True, exist_ok=True)
    declared = declared_dir / "calib.safetensors"
    declared.write_bytes(artifact.read_bytes())
    size = staged.stat().st_size
    key = pmap.residency_map_key(str(declared), 0)
    entry = {"stage_path": str(staged), "bytes": size,
             "sha256": digest, "offset": 0}
    fragment = {
        "schema": pmap.RESIDENCY_MAP_FRAGMENT_SCHEMA_V1,
        "consumer_action_key": KEY, "mover_action_key": MOVER,
        "tier_id": STAGE_TIER, "stage_root": str(stage),
        "manifest_sha256": MANIFEST, "entries": {key: dict(entry)}}
    pmap.write_fragment(root, fragment)
    lease.write_material(
        root, consumer_action_key=KEY, mover_action_key=MOVER,
        tier_id=STAGE_TIER, stage_root=str(stage), manifest_sha256=MANIFEST,
        generation=lease.mint_generation(),
        entries={key: {**entry, "file_id": lease.stat_identity(str(staged))}})
    composed = pmap.compose([fragment])
    if ram is not None:
        ram_fragment = {
            "schema": pmap.RESIDENCY_MAP_FRAGMENT_SCHEMA_V1,
            "consumer_action_key": KEY, "mover_action_key": RAM_MOVER,
            "tier_id": ram["tier_id"], "stage_root": str(ram["root"]),
            "manifest_sha256": MANIFEST, "epoch": ram["epoch"],
            "entries": {key: {"stage_path": str(ram["path"]),
                              "bytes": size, "sha256": digest, "offset": 0}}}
        if ram.get("publish"):
            ram_path = str(ram["path"])
            lease.write_material(
                root, consumer_action_key=KEY, mover_action_key=RAM_MOVER,
                tier_id=ram["tier_id"], stage_root=str(ram["root"]),
                manifest_sha256=MANIFEST, epoch=ram["epoch"],
                generation=lease.mint_generation(),
                entries={key: {"stage_path": ram_path, "bytes": size,
                               "sha256": digest, "offset": 0,
                               "file_id": lease.stat_identity(ram_path)}})
        composed = pmap.overlay_ram(
            composed, [ram_fragment], ram_tier_id=ram["tier_id"],
            ram_root=str(ram["root"]), ram_epoch=ram["epoch"])
    map_path = root / f"{KEY}.map.json"
    pmap.write_map(map_path, composed)
    return {"declared": declared, "declared_key": key, "entry": entry,
            "map_path": map_path, "residency_root": root,
            "size": size, "staged": staged}


def _announce_ram(world, ram_root: Path) -> dict:
    """A real epoch marker plus the isolated tier record the readers read.

    The epoch comes from the SDK's own ``ensure_ram_epoch`` (a real marker
    file under the tmpfs root). The tier record is fixture authority
    configuration for this isolated tier: real schema, the real announced
    epoch, and no admission pretense. Only schema/tier_id/epoch are read
    by the strict path under test.
    """

    tiers = world.mod["tiers"]
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) > 2 and fields[0] == "tmpfs" and fields[1] == "/dev/shm":
            break
    else:
        pytest.skip("no tmpfs /dev/shm on this worker")
    marker = tiers.ensure_ram_epoch(ram_root, host=RAM_HOST)
    assert marker is not None, "ram epoch stamp refused"
    tier_id = tiers.tier_id("ram", RAM_HOST)
    record = {
        "schema": tiers.TIER_RECORD_SCHEMA_V1, "tier": "ram",
        "tier_id": tier_id, "host": RAM_HOST,
        "mountpoint": str(ram_root),
        "mount_options": ["rw", "nosuid", "nodev"],
        "epoch": str(marker["epoch"]),
        "size_bytes": 1, "ceiling_bytes": 1, "capacity_bytes": 1,
        "ram_admission": {"admissible": False,
                          "reason": "fixture-isolated-tier-record"},
    }
    tiers_dir = world.queue.root / "tiers"
    tiers_dir.mkdir(parents=True, exist_ok=True)
    (tiers_dir / f"{tier_id}.json").write_text(
        json.dumps(record, sort_keys=True) + "\n")
    return {"tier_id": tier_id, "epoch": str(marker["epoch"])}


def _ram_root() -> Path:
    path = Path("/dev/shm") / f"pq-live-reader-qual-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)
    return path


def _run_positive(world, monkeypatch, capsys, tmp_path, staged, *,
                  tiers="ssd", draw=None, artifact_digest=None,
                  helper_root_path=None, extra=()):
    _mount_patches(monkeypatch)
    artifact_digest = artifact_digest or _artifact(tmp_path)[1]
    draw = draw if draw is not None else _artifact(tmp_path)[2]
    monkeypatch.setenv("PRISMABUILD_READER_HELPER_ROOT",
                       str(helper_root_path or tmp_path / "helper"))
    argv = _read_argv(staged["declared"], artifact_digest, draw,
                      tiers=tiers, extra=extra)
    return _run_main(monkeypatch, capsys, argv)


def test_true_positive_read_holds_and_releases(
        world, monkeypatch, capsys, tmp_path, helper_root) -> None:
    """One real read: exit 0, exact-attempt hold, clean release, zero pool.

    The full production path runs: strict calibration read through a real
    acquired window, then the hold proof on a second real window, then the
    post-release census — all on the real SDK at the pinned install, with
    the lease filed and released on the isolated queue only.
    """

    artifact, digest, draw = _artifact(tmp_path)
    staged = _stage(world, tmp_path, artifact, digest)
    record = far._publish_claim(world, key=KEY)
    _scope, control = far._open_scope(world, key=KEY, nonce=NONCE)
    far._record_scope(world, KEY, control, claim=record)
    _identity_env(world, monkeypatch, staged["map_path"], control)
    code, result = _run_positive(
        world, monkeypatch, capsys, tmp_path, staged,
        tiers="ssd", draw=draw, artifact_digest=digest,
        helper_root_path=helper_root)
    assert code == 0, result
    assert result["ok"] is True
    assert result["draw_match"] is True
    assert result["shape"] == [N_SAMPLES, SEQLEN]
    assert result["sdk_importable"] is True
    assert Path(result["sdk_file"]).parent.name == "prismabuild"
    assert result["serving"]["tier_id"] == STAGE_TIER
    assert result["serving_tier_check"] == "ssd-served"
    held = result["held_refs_for_attempt"]
    assert isinstance(held, list) and len(held) == 1
    served_path = os.path.normpath(str(staged["staged"]))
    assert result["pinned_while_open"] == {served_path: [
        result["serving"]["pin_id"]]}
    assert result["tainted_while_open"] == []
    assert result["lingering_refs_for_attempt"] == []
    assert result["pins_after_release"] == {}
    assert result["tainted_after_release"] == []
    assert result["pool_reads_observed"] is True
    assert result["pool_client_read_delta"] == 0
    leases = staged["residency_root"] / "leases"
    assert not leases.exists() or not list(leases.rglob("*.lease.json"))


def test_wrong_draw_rejected_after_real_read(
        world, monkeypatch, capsys, tmp_path, helper_root) -> None:
    """A served, digest-vouched read with a foreign expected draw: exit 2.

    The rejection happens AFTER the real read proves the draw: the result
    carries the actual serving record and the actual draw hash, so this is
    a post-read rejection, not a staged-refusal shortcut.
    """

    artifact, digest, draw = _artifact(tmp_path)
    staged = _stage(world, tmp_path, artifact, digest)
    record = far._publish_claim(world, key=KEY)
    _scope, control = far._open_scope(world, key=KEY, nonce=NONCE)
    far._record_scope(world, KEY, control, claim=record)
    _identity_env(world, monkeypatch, staged["map_path"], control)
    code, result = _run_positive(
        world, monkeypatch, capsys, tmp_path, staged,
        tiers="ssd", draw="e" * 64, artifact_digest=digest,
        helper_root_path=helper_root)
    assert code == probe.UNQUALIFIED == 2
    assert result["ok"] is False
    assert result["draw_match"] is False
    assert result["draw_sha256"] == draw
    assert result["serving"]["tier_id"] == STAGE_TIER
    assert "draw" in str(result["finding"])


def test_foreign_attempt_ref_never_counts_as_ours(
        world, monkeypatch, capsys, tmp_path, helper_root) -> None:
    """A live foreign ref on the shared pin is invisible to the hold proof.

    A real low-level acquire under a foreign attempt (different nonce and
    scope, real SDK, real pin) holds the same deterministic pin while main
    runs. Main must pass by counting ONLY this attempt's refs — and the
    foreign ref's survival after release must read as an unrelated healthy
    reader, not as taint or as our own lingering lease.
    """

    lease = world.mod["lease"]
    artifact, digest, draw = _artifact(tmp_path)
    staged = _stage(world, tmp_path, artifact, digest)
    foreign = lease.acquire(
        world.queue, consumer_action_key=KEY,
        attempt={"nonce": "d" * 32, "scope_id": "foreign-scope"},
        tier_id=STAGE_TIER, epoch="",
        span={"start_bytes": 0, "end_bytes": staged["size"]},
        holder={"host": "fixture-foreign", "worker": "fixture-foreign:1:x",
                "pid": os.getpid()},
        acquire_token="foreign-token",
        covers=[{"mover_action_key": MOVER, "manifest_sha256": MANIFEST}],
        expected={staged["declared_key"]: {
            "bytes": staged["size"], "sha256": digest}},
        residency_root=staged["residency_root"], owner_action_key=KEY)
    assert foreign.get("ok") is True, foreign
    try:
        record = far._publish_claim(world, key=KEY)
        _scope, control = far._open_scope(world, key=KEY, nonce=NONCE)
        far._record_scope(world, KEY, control, claim=record)
        _identity_env(world, monkeypatch, staged["map_path"], control)
        code, result = _run_positive(
            world, monkeypatch, capsys, tmp_path, staged,
            tiers="ssd", draw=draw, artifact_digest=digest,
            helper_root_path=helper_root)
        assert code == 0, result
        held = result["held_refs_for_attempt"]
        assert len(held) == 1 and foreign["ref_id"] not in held
        served_path = os.path.normpath(str(staged["staged"]))
        assert result["pins_after_release"] == {served_path: [
            foreign["pin_id"]]}
        assert result["tainted_after_release"] == []
        assert result["lingering_refs_for_attempt"] == []
    finally:
        assert lease.release(
            world.queue, foreign["pin_id"], foreign["ref_id"],
            consumer_action_key=KEY, residency_root=staged["residency_root"])


def test_tainted_census_refuses_while_open(
        world, monkeypatch, capsys, tmp_path, helper_root) -> None:
    """An unreadable pin anywhere taints the census: exit 2, no hold proof.

    A real corrupt ``*.lease.json`` under a foreign owner directory in the
    isolated queue's leases tree makes ``live_for`` taint the pass; main
    must refuse the hold proof rather than count an unknowable census.
    """

    artifact, digest, draw = _artifact(tmp_path)
    staged = _stage(world, tmp_path, artifact, digest)
    leases = staged["residency_root"] / "leases" / "zz-foreign-owner"
    leases.mkdir(parents=True)
    (leases / "corrupt.lease.json").write_text("{not json")
    record = far._publish_claim(world, key=KEY)
    _scope, control = far._open_scope(world, key=KEY, nonce=NONCE)
    far._record_scope(world, KEY, control, claim=record)
    _identity_env(world, monkeypatch, staged["map_path"], control)
    code, result = _run_positive(
        world, monkeypatch, capsys, tmp_path, staged,
        tiers="ssd", draw=draw, artifact_digest=digest,
        helper_root_path=helper_root)
    assert code == probe.UNQUALIFIED == 2
    assert result["ok"] is False
    assert "held proof incomplete" in str(result["finding"])
    assert result["tainted_while_open"]


def test_ram_first_serves_ram(
        world, monkeypatch, capsys, tmp_path, helper_root) -> None:
    """RAM offered, allowed, and published: the window serves the tmpfs leg.

    Real ram copy, real epoch marker, real ram fragment/material under the
    announced epoch; the resolver's own RAM selection acquires the ram
    window and the open serves the ram tier.
    """

    artifact, digest, draw = _artifact(tmp_path)
    ram = _ram_root()
    try:
        shutil.copy2(artifact, ram / "calib.safetensors")
        announced = _announce_ram(world, ram)
        staged = _stage(world, tmp_path, artifact, digest, ram={
            "root": ram, "path": ram / "calib.safetensors",
            "tier_id": announced["tier_id"], "epoch": announced["epoch"],
            "publish": True})
        record = far._publish_claim(world, key=KEY)
        _scope, control = far._open_scope(world, key=KEY, nonce=NONCE)
        far._record_scope(world, KEY, control, claim=record)
        _identity_env(world, monkeypatch, staged["map_path"], control)
        code, result = _run_positive(
            world, monkeypatch, capsys, tmp_path, staged,
            tiers="ram,ssd", draw=draw, artifact_digest=digest,
            helper_root_path=helper_root)
        assert code == 0, result
        assert result["ok"] is True
        assert result["ram_offered"]
        assert result["serving"]["tier_id"] == announced["tier_id"]
        assert result["serving_tier_check"] == "ram-served"
        assert result["tainted_after_release"] == []
        leases = staged["residency_root"] / "leases"
        assert not leases.exists() or not list(leases.rglob("*.lease.json"))
    finally:
        shutil.rmtree(ram, ignore_errors=True)


def test_ram_offered_without_published_material_falls_back_typed(
        world, monkeypatch, capsys, tmp_path, helper_root) -> None:
    """RAM offered but its covers unpublished: recorded fallback, SSD serves.

    The ram leg refuses availability (nothing published for it), the
    resolver records the typed fallback itself, and main accepts the SSD
    service only on that recorded evidence — the RAM-first gate stays
    honest because the record is the resolver's, never a bare ram_path.
    """

    artifact, digest, draw = _artifact(tmp_path)
    ram = _ram_root()
    try:
        shutil.copy2(artifact, ram / "calib.safetensors")
        announced = _announce_ram(world, ram)
        staged = _stage(world, tmp_path, artifact, digest, ram={
            "root": ram, "path": ram / "calib.safetensors",
            "tier_id": announced["tier_id"], "epoch": announced["epoch"],
            "publish": False})
        record = far._publish_claim(world, key=KEY)
        _scope, control = far._open_scope(world, key=KEY, nonce=NONCE)
        far._record_scope(world, KEY, control, claim=record)
        _identity_env(world, monkeypatch, staged["map_path"], control)
        code, result = _run_positive(
            world, monkeypatch, capsys, tmp_path, staged,
            tiers="ram,ssd", draw=draw, artifact_digest=digest,
            helper_root_path=helper_root)
        assert code == 0, result
        assert result["ok"] is True
        assert result["ram_offered"]
        assert result["serving"]["tier_id"] == STAGE_TIER
        assert result["serving_tier_check"] == "ssd-served"
    finally:
        shutil.rmtree(ram, ignore_errors=True)
