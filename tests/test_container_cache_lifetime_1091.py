"""Versioned PB lifetime intent owns cache-pair workspaces, not legacy rows."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_container_cache_charge_1091 import CACHE, CONTAINER_FIXTURE_ROOT, _layout, _runner
from fullstack_pb_generation import PIN_PATH
from test_container_cache_roots_1072 import _spec
from test_dispatch_joint_quanta import campaign  # noqa: F401

LIFETIME_ENV = "PRISMABUILD_EPHEMERAL_SCRATCH_DECLARATIONS"
SELECTION_SCHEMA = "prismabuild.scratch_lifetime_selection.v1"
pytestmark = pytest.mark.own_process


@pytest.fixture
def lifetime_sdk(installed_client_sdk):
    return installed_client_sdk


def test_cache_pair_selects_the_actual_tmp_root_and_persistent_caches(lifetime_sdk):
    runner = _runner()
    spec, env = _layout(CONTAINER_FIXTURE_ROOT)
    selection = runner.scratch_lifetime_selection(
        spec, runner.local_scratch_environment(spec, env))
    assert json.loads(selection[LIFETIME_ENV]) == {
        "schema": SELECTION_SCHEMA,
        "entries": [
            {"root_env": "PRISMAQUANT_STAGE_B_SPILL_ROOT", "name": "row-tmp",
             "lifetime": "ephemeral"},
            {"root_env": CACHE[0], "name": "compile", "lifetime": "persistent"},
        ],
    }


def test_legacy_row_selects_no_lifetime_intent():
    assert _runner().scratch_lifetime_selection(_spec({}, roots=()), {}) is None


def test_spec_cannot_declare_the_lifetime_control():
    spec, _ = _layout(CONTAINER_FIXTURE_ROOT)
    spec["env"][LIFETIME_ENV] = "[]"
    with pytest.raises(RuntimeError, match="derived from the declared"):
        _runner().validate_container(spec)


def test_tmpdir_without_selected_support_refuses(monkeypatch):
    from prismaquant import staged_lease

    monkeypatch.delenv(staged_lease.HELPER_ROOT_ENV_VAR, raising=False)
    staged_lease.set_lease_helper_root(None)
    spec, env = _layout(CONTAINER_FIXTURE_ROOT)
    with pytest.raises(RuntimeError, match="lifetime.*support|lease-helper"):
        _runner().scratch_lifetime_selection(
            spec, _runner().local_scratch_environment(spec, env))


def test_sdk_without_lifetime_capability_refuses(lifetime_sdk, monkeypatch):
    monkeypatch.setattr(lifetime_sdk, "CAPABILITIES", ())
    spec, env = _layout(CONTAINER_FIXTURE_ROOT)
    with pytest.raises(RuntimeError, match="scratch-lifetime-v1"):
        _runner().scratch_lifetime_selection(
            spec, _runner().local_scratch_environment(spec, env))


def _registered_client(spec, env, monkeypatch, lifetime_sdk, *, registration="complete"):
    """Replace only public claim reads; the real SDK owns names and declarations."""
    runner = _runner()
    scratch = runner.local_scratch_environment(spec, env)
    selected = json.loads(runner.scratch_lifetime_selection(spec, scratch)[LIFETIME_ENV])
    key, nonce = "a" * 64, "b" * 32
    declaration = {
        "schema": lifetime_sdk.EPHEMERAL_SCRATCH_SCHEMA_V1, "lifetime": "ephemeral",
        "root_env": "PRISMAQUANT_STAGE_B_SPILL_ROOT",
        "max_env": "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES",
        "root": env["PRISMAQUANT_STAGE_B_SPILL_ROOT"], "max_bytes": 1 << 30,
        "name": "row-tmp", "owner_action_key": key, "owner_published_unix": 1.0,
        "owner_host": "fixture-host", "owner_attempt": {"nonce": nonce, "scope_id": ""},
    }
    import hashlib
    declaration["owner_attempt"]["scope_id"] = (
        "prismabuild-job" + hashlib.sha256((key + nonce).encode()).hexdigest()[:32] + ".slice")
    path = str(lifetime_sdk.ephemeral_scratch_path(declaration))
    entry = {"root_env": declaration["root_env"], "name": "row-tmp",
             "lifetime": "ephemeral", "declaration": declaration,
             "identity": [{"fixture": "registered"}], "cleaned": False}
    record = {"schema": lifetime_sdk.SCRATCH_LIFETIME_RECORD_SCHEMA_V1,
              "registration_complete": registration == "complete", "entries": [entry]}
    claim = {"action_key": key, lifetime_sdk.SCRATCH_LIFETIME_FIELD: record}
    if registration == "missing":
        del claim[lifetime_sdk.SCRATCH_LIFETIME_FIELD]
    elif registration == "wrong-attempt":
        entry["declaration"] = {**declaration, "owner_published_unix": 2.0}
    monkeypatch.setattr(lifetime_sdk, "PoolQueue", lambda root: SimpleNamespace(root=root))
    monkeypatch.setattr(lifetime_sdk, "read_claimed_record", lambda queue, action: claim)
    monkeypatch.setattr(lifetime_sdk, "bind_ephemeral_scratch", lambda *a, **kw: declaration)
    launch = {**env, LIFETIME_ENV: json.dumps(selected),
              runner.QUEUE_ROOT_ENV: "/fixture-queue", runner.ACTION_KEY_ENV: key,
              runner.ACTION_NONCE_ENV: nonce,
              runner.ACTION_SCOPE_ENV: declaration["owner_attempt"]["scope_id"]}
    launch[runner.READER_HELPER_ROOT_ENV] = json.loads(PIN_PATH.read_text())["bundle_root"]
    return launch, path


def test_registered_attempt_path_replaces_only_tmpdir(lifetime_sdk, monkeypatch):
    runner = _runner()
    spec, env = _layout(CONTAINER_FIXTURE_ROOT)
    launch, path = _registered_client(spec, env, monkeypatch, lifetime_sdk)
    result = runner.scratch_workspace_environment(spec, runner.local_scratch_environment(spec, env), launch)
    assert result == {"PRISMAQUANT_TMPDIR": path}
    assert Path(path).is_relative_to(Path(env["PRISMAQUANT_STAGE_B_SPILL_ROOT"]))
    assert "/prismabuild-ephemeral/" in path
    assert path != env["PRISMAQUANT_TMPDIR"]
    defaults, _ = runner.container_cache_environment(spec, runner.local_scratch_environment(spec, env))
    assert defaults["TRITON_CACHE_DIR"] == env[CACHE[0]] + "/triton"
    assert defaults["TORCHINDUCTOR_CACHE_DIR"] == env[CACHE[0]] + "/inductor"


@pytest.mark.parametrize("registration", ["missing", "incomplete", "wrong-attempt"])
def test_unregistered_or_foreign_attempt_never_gets_a_workspace(
        lifetime_sdk, monkeypatch, registration):
    runner = _runner()
    spec, env = _layout(CONTAINER_FIXTURE_ROOT)
    launch, _ = _registered_client(spec, env, monkeypatch, lifetime_sdk, registration=registration)
    with pytest.raises(RuntimeError, match="scratch lifetime"):
        runner.scratch_workspace_environment(spec, runner.local_scratch_environment(spec, env), launch)


def test_legacy_quantum_row_seals_no_lifetime_intent(tmp_path, campaign):
    from test_dispatch_joint_quanta import _scratch_quantum_argv, _outer_env

    cot, spill = "/home/rob/pb-scratch/cot", "/home/rob/pb-scratch/spill"
    env = {"PRISMAQUANT_STAGE_B_COTANGENT_ROOT": cot,
           "PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES": "1",
           "PRISMAQUANT_STAGE_B_SPILL_ROOT": spill,
           "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES": str(1 << 30)}
    argv = _scratch_quantum_argv(tmp_path, campaign, env, (cot, spill))()
    assert LIFETIME_ENV not in dict(item.split("=", 1) for item in _outer_env(argv))


def test_cache_pair_quantum_seals_versioned_intent_outside_the_container_spec(
        tmp_path, campaign, lifetime_sdk):
    from test_dispatch_joint_quanta import _scratch_quantum_argv, _outer_env

    spec, env = _layout(CONTAINER_FIXTURE_ROOT)
    roots = tuple(mount["source"] for mount in spec["container"]["mounts"]
                  if mount["source"].startswith(str(CONTAINER_FIXTURE_ROOT)))
    argv = _scratch_quantum_argv(tmp_path, campaign, env, roots)()
    sealed = dict(item.split("=", 1) for item in _outer_env(argv))
    intent = json.loads(sealed[LIFETIME_ENV])
    assert intent["schema"] == SELECTION_SCHEMA
    assert intent["entries"][0]["root_env"] == "PRISMAQUANT_STAGE_B_SPILL_ROOT"
    launched = json.loads(argv[argv.index("--spec") + 1])
    assert LIFETIME_ENV not in launched["env"]
    assert launched["env"]["PRISMAQUANT_TMPDIR"] == env["PRISMAQUANT_TMPDIR"]


@pytest.fixture
def connected_lifetime(tmp_path, monkeypatch, pinned_pb_source):
    """Run the pinned pool owner with its broker and process fixture endpoints."""
    import importlib.util
    from fullstack_pb_generation import reader_sdk_bound, require_paths

    root = Path(require_paths()["root"])
    source = root / "tests" / "test_scratch_declaration_record.py"
    module_spec = importlib.util.spec_from_file_location("_pq1091_pool_fixture", source)
    fixture = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(fixture)
    core, pool, scope = fixture.pb, fixture.pool, fixture.resource_scope
    seal = core.seal_action
    runner = _runner()
    temporary, cache = str(tmp_path / "temporary"), str(tmp_path / "persistent")
    env = {"PRISMAQUANT_STAGE_B_SPILL_ROOT": temporary,
           "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES": "1024",
           CACHE[0]: cache, CACHE[1]: "2048",
           "PRISMAQUANT_TMPDIR": temporary + "/row-temp"}
    spec = _spec(env, roots=(temporary, cache))
    scratch = runner.local_scratch_environment(spec, env)
    with reader_sdk_bound():
        selection = runner.scratch_lifetime_selection(spec, scratch)

        def seal_row(request):
            request["environment"]["variables"] = {**scratch, **selection}
            return seal(request)

        monkeypatch.setattr(core, "seal_action", seal_row)
        claim = pool.PoolQueue.claim
        monkeypatch.setattr(pool.PoolQueue, "claim", lambda self, **kw:
                            claim(self, **{"tags": ["scratch-lifetime-v1"], **kw}))
        create, calls = fixture.runtime.__wrapped__(tmp_path, monkeypatch)
        queue, item, _ = create()
        Path(temporary).mkdir()
        Path(cache).mkdir()
        (Path(cache) / "compiled").write_bytes(b"persistent")
        monkeypatch.setattr(scope.ResourceScope, "export_stopped_verdict", lambda value: {
            "ok": True, "scope_id": value.unit, "stopped": True, "empty": True,
            "released": True, "retired": False, "settled": True,
            "tickets_pending": False, "stopped_unix": 1.0})

        def launch(returncode=0):
            observed = {}

            def payload(argv, kwargs):
                live = fixture.live_record(queue, item)
                launch_env = {**scratch, **selection, **fixture.context(live),
                              runner.QUEUE_ROOT_ENV: str(queue.root)}
                forwarded = runner.scratch_workspace_environment(spec, scratch, launch_env)
                path = Path(forwarded["PRISMAQUANT_TMPDIR"])
                assert path.is_dir()
                (path / "payload-temp").write_bytes(b"temporary")
                observed["path"] = path

            fixture.process(monkeypatch, payload, returncode=returncode)
            outcome = queue.execute(item, containment=True)
            return outcome, observed["path"]

        yield queue, item, cache, launch, pool


@pytest.mark.parametrize("returncode", [0, 1])
def test_terminal_pool_cleans_the_forwarded_tmpdir_and_preserves_compile_cache(
        connected_lifetime, returncode):
    queue, item, cache, launch, pool = connected_lifetime
    outcome, path = launch(returncode)
    queue.finish(item["action_key"], status=outcome["status"], detail=outcome, claim_snapshot=item)
    assert not path.exists()
    assert (Path(cache) / "compiled").read_bytes() == b"persistent"
    assert queue.ledger().held() == {}


def test_stale_owner_recovery_cleans_only_the_registered_attempt(connected_lifetime):
    queue, item, cache, launch, pool = connected_lifetime
    outcome, path = launch()
    lease_path = queue.lease_path(item["action_key"])
    lease = json.loads(lease_path.read_text())
    lease["heartbeat_unix"] = 1.0
    pool._write_json_atomic(lease_path, lease)
    queue.reap_stale(timeout_s=1)
    assert not path.exists()
    assert (Path(cache) / "compiled").read_bytes() == b"persistent"
    assert queue.ledger().held() == {}


def test_predecessor_finish_preserves_the_next_attempt_workspace(connected_lifetime, monkeypatch):
    queue, old, cache, launch, pool = connected_lifetime
    from prismaquant.staged_lease import client_sdk

    outcome, old_path = launch(1)
    queue.finish(old["action_key"], status="failed", detail=outcome, claim_snapshot=old)
    assert not old_path.exists()
    successor = queue.claim(capacity=old["resources"], tags=["scratch-lifetime-v1"])
    queue.execute(successor, containment=True)
    record = successor[client_sdk().SCRATCH_LIFETIME_FIELD]
    new_path = client_sdk().ephemeral_scratch_path(record["entries"][0]["declaration"])
    assert new_path != old_path and new_path.is_dir()
    sentinel = new_path / "new-attempt"
    sentinel.write_bytes(b"successor")
    held = queue.ledger().held()
    queue.finish(old["action_key"], status="failed", detail=outcome, claim_snapshot=old)
    assert sentinel.read_bytes() == b"successor"
    assert queue.ledger().held() == held
    assert (Path(cache) / "compiled").read_bytes() == b"persistent"
