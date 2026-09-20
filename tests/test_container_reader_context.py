"""The campaign container forwards PB's injected reader identity (PQ #856).

PB's resource_exec proxy injects a public attempt identity plus the sealed
helper generation root into the action environment
(`PRISMABUILD_ACTION_KEY/NONCE/SCOPE/READER_HELPER_ROOT`); the reader SDK
checks the live claim against the nonce/scope pair and imports sealed
helpers from beneath the helper root. The wrapper used to build container
env from the declared spec alone, dropping all four. These tests pin the
transport over actual ``docker_command`` construction: complete bundle
forwarded with the generation mounted read-only, spec forgeries refused,
partial bundles refused, legacy absence byte-identical, and no broker
capability crossing the boundary.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
runner = importlib.import_module("tools.tessera_campaign_container")


KEY = "a" * 64
NONCE = "b" * 32
SCOPE = "scope-7"

NAMES = ("PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE",
         "PRISMABUILD_ACTION_SCOPE", "PRISMABUILD_READER_HELPER_ROOT")


def _spec(tmp_path, *, env=None, mounts=None):
    declared = {
        "model": "/mnt/shared/model", "cwd": "/original/checkout",
        "python": "python3", "campaign_argv": [],
        "env": {"PYTHONPATH": ".:/producer/src"},
        "container": {"image": "qualified:fixed", "mounts": (
            mounts if mounts is not None else [
                {"source": str(tmp_path), "target": str(tmp_path)},
            ])},
    }
    if env:
        declared["env"].update(env)
    return declared


def _helper_root(tmp_path, name="gen-1"):
    """An immutable-generation layout: <root>/src/prismabuild beneath it."""
    root = tmp_path / name
    package = root / "src" / "prismabuild"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "reader_lease.py").write_text("# sealed\n")
    return Path(os.path.realpath(root))


def _bundle(helper, *, key=KEY, nonce=NONCE, scope=SCOPE):
    return {"PRISMABUILD_ACTION_KEY": key,
            "PRISMABUILD_ACTION_NONCE": nonce,
            "PRISMABUILD_ACTION_SCOPE": scope,
            "PRISMABUILD_READER_HELPER_ROOT": str(helper)}


def _argv(spec, environ):
    return runner.docker_command(spec, ["python3"], cwd="/snapshot",
                                 uid=1, gid=1, image_id="sha256:resolved",
                                 environ=environ)


def _forwarded_env(argv):
    """The container environment as ``docker_command`` built it."""
    out = {}
    seen = list(argv)
    while "--env" in seen:
        i = seen.index("--env")
        pair = seen[i + 1]
        del seen[i:i + 2]
        key, _, value = pair.partition("=")
        out[key] = value
    return out


def _mounts(argv):
    out = []
    seen = list(argv)
    while "--mount" in seen:
        i = seen.index("--mount")
        out.append(seen[i + 1])
        del seen[i:i + 2]
    return out


def test_complete_bundle_is_forwarded_with_its_generation_mounted_read_only(tmp_path):
    """Exact launch values cross; the generation binds read-only at its path."""
    helper = _helper_root(tmp_path)
    spec = _spec(tmp_path)
    argv = _argv(spec, _bundle(helper))
    env = _forwarded_env(argv)
    assert env["PRISMABUILD_ACTION_KEY"] == KEY
    assert env["PRISMABUILD_ACTION_NONCE"] == NONCE
    assert env["PRISMABUILD_ACTION_SCOPE"] == SCOPE
    assert env["PRISMABUILD_READER_HELPER_ROOT"] == str(helper)
    assert (f"type=bind,src={helper},dst={helper},readonly"
            in _mounts(argv))


def test_legacy_absence_keeps_a_byte_identical_argv(tmp_path):
    """No strict signal in the launcher env: no identity, no helper mount."""
    spec = _spec(tmp_path)
    assert _argv(spec, {}) == _argv(spec, None)
    argv = _argv(spec, {})
    env = _forwarded_env(argv)
    assert not any(name in env for name in NAMES)
    assert not any("prismabuild-fleet" in mount and "readonly" in mount
                   for mount in _mounts(argv))


def test_key_only_launcher_env_is_the_legacy_shape_not_a_partial_bundle(tmp_path):
    """Published PB sets the public key on every action env without the rest."""
    spec = _spec(tmp_path)
    argv = _argv(spec, {"PRISMABUILD_ACTION_KEY": KEY})
    env = _forwarded_env(argv)
    assert not any(name in env for name in NAMES)


def test_module_names_match_the_pb_producer_contract():
    """The four forwarded names are PB's protected residency names exactly.

    Cross-checked read-only against candidate PB730
    (``core.ACTION_{KEY,NONCE,SCOPE}_ENV``,
    ``core.READER_HELPER_ROOT_ENV``, ``resource_exec.payload_identity_env``):
    the literals below equal those spellings, and the helper-root value is
    the bare generation directory in both (the ``/src`` suffix PB730
    injected is the shape root asked that author to fix; the SDK appends
    ``/src`` itself). Where the published PB core is importable its
    existing key name must agree; the newer names postdate it.
    """
    assert runner.ACTION_KEY_ENV == "PRISMABUILD_ACTION_KEY"
    assert runner.ACTION_NONCE_ENV == "PRISMABUILD_ACTION_NONCE"
    assert runner.ACTION_SCOPE_ENV == "PRISMABUILD_ACTION_SCOPE"
    assert runner.READER_HELPER_ROOT_ENV == "PRISMABUILD_READER_HELPER_ROOT"
    assert runner.READER_CONTEXT_ENV == (
        "PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE",
        "PRISMABUILD_ACTION_SCOPE", "PRISMABUILD_READER_HELPER_ROOT")
    core = None
    try:
        found = importlib.util.find_spec("prismabuild.core")
    except (ImportError, ModuleNotFoundError):
        found = None
    if found is None:
        pytest.skip("published PB core is not importable here")
    import prismabuild.core as published
    assert published.ACTION_KEY_ENV == runner.ACTION_KEY_ENV


@pytest.mark.parametrize("name", NAMES)
def test_a_spec_declaring_reader_identity_is_refused(tmp_path, name):
    """A sealed identity name is a forgery, never a default (both entries)."""
    helper = _helper_root(tmp_path)
    forged = {name: "forged"}
    with pytest.raises(RuntimeError, match=name):
        runner.docker_command(_spec(tmp_path, env=forged), ["python3"],
                              cwd="/snapshot", uid=1, gid=1,
                              image_id="sha256:resolved",
                              environ=_bundle(helper))
    with pytest.raises(RuntimeError, match=name):
        runner.reader_context_environment(_spec(tmp_path, env=forged),
                                          _bundle(helper))


@pytest.mark.parametrize("drop", NAMES)
def test_a_partial_bundle_is_refused_not_downgraded(tmp_path, drop):
    """Any strict half missing: refuse rather than bind a guessed identity."""
    helper = _helper_root(tmp_path)
    bundle = _bundle(helper)
    del bundle[drop]
    with pytest.raises(RuntimeError, match="partial"):
        runner.docker_command(_spec(tmp_path), ["python3"], cwd="/snapshot",
                              uid=1, gid=1, image_id="sha256:resolved",
                              environ=bundle)


@pytest.mark.parametrize("key", ["short", "G" * 64, ""])
def test_a_malformed_action_key_is_refused(tmp_path, key):
    """The key is 64 lowercase hex or there is no launch identity at all."""
    helper = _helper_root(tmp_path)
    bundle = _bundle(helper, key=key)
    with pytest.raises(RuntimeError, match="ACTION_KEY|partial"):
        _argv(_spec(tmp_path), bundle)


def test_a_src_suffixed_helper_root_is_refused(tmp_path):
    """The value names the generation dir; the SDK appends ``/src`` itself."""
    helper = _helper_root(tmp_path)
    bundle = _bundle(helper / "src")
    with pytest.raises(RuntimeError, match="src/prismabuild"):
        _argv(_spec(tmp_path), bundle)


def test_a_symlinked_helper_root_is_refused(tmp_path):
    """A spelling that resolves elsewhere (like the mutable /repo link)."""
    helper = _helper_root(tmp_path)
    link = tmp_path / "repo"
    try:
        os.symlink(helper, link)
    except OSError:
        pytest.skip("this host cannot create a symlink")
    with pytest.raises(RuntimeError, match="symlink|canonical"):
        _argv(_spec(tmp_path), _bundle(link))


@pytest.mark.parametrize("target", ["nested", "exact"])
def test_mounts_shadowing_the_helper_tree_are_refused(tmp_path, target):
    """Nothing declared may sit at or beneath the helper generation root."""
    helper = _helper_root(tmp_path)
    victim = helper if target == "exact" else helper / "src"
    mounts = [{"source": str(tmp_path), "target": str(tmp_path)},
              {"source": str(tmp_path), "target": str(victim)}]
    with pytest.raises(RuntimeError, match="helper"):
        _argv(_spec(tmp_path, mounts=mounts), _bundle(helper))


def test_a_broad_rw_mount_still_gets_the_readonly_helper_bind(tmp_path):
    """A covering rw mount is not a read-only helper mount: bind it anyway."""
    helper = _helper_root(tmp_path)
    argv = _argv(_spec(tmp_path), _bundle(helper))
    assert (f"type=bind,src={helper},dst={helper},readonly"
            in _mounts(argv))


@pytest.mark.parametrize("layout", ["same-source", "remapped-source", "nested-rw"])
def test_parent_mounts_cannot_replace_or_make_the_helper_writable(tmp_path, layout):
    """The exact helper source stays readonly despite ancestor mappings."""
    parent = tmp_path / "generations"
    parent.mkdir()
    helper = _helper_root(parent)
    source = tmp_path if layout != "remapped-source" else tmp_path / "other"
    source.mkdir(exist_ok=True)
    mounts = [{"source": str(source), "target": str(tmp_path),
               "readonly": True}]
    if layout == "nested-rw":
        mounts.append({"source": str(parent), "target": str(parent),
                       "readonly": False})
    argv = _argv(_spec(tmp_path, mounts=mounts), _bundle(helper))
    binds = [m for m in _mounts(argv) if f"dst={helper}" in m]
    assert binds == [f"type=bind,src={helper},dst={helper},readonly"]
    assert _mounts(argv)[-1] == binds[0]


def test_no_broker_capability_crosses_the_boundary(tmp_path):
    """Only the four public names cross; tokens and sockets stay out."""
    helper = _helper_root(tmp_path)
    environ = _bundle(helper)
    environ.update({"PRISMABUILD_BROKER_TOKEN": "secret",
                    "PRISMABUILD_RESOURCE_TOKEN": "secret",
                    "PRISMABUILD_BROKER_SOCKET": "/run/broker.sock"})
    env = _forwarded_env(_argv(_spec(tmp_path), environ))
    assert set(env) == {"PYTHONSAFEPATH", "PYTHONPATH",
                        *NAMES}
    for name in env:
        assert "TOKEN" not in name and "SOCKET" not in name


def test_reader_context_rides_beside_the_residency_map(tmp_path, monkeypatch):
    """Both channels cross together; neither mount disturbs the other."""
    monkeypatch.setattr(runner, "_stage_root_mounted", lambda root: True)
    # The stage lives outside the declared mount, as /stage/prewarm does;
    # the helper generation lives beneath it, under a covering rw mount.
    stage = tmp_path.parent / f"{tmp_path.name}-stage"
    stage.mkdir(exist_ok=True)
    helper = _helper_root(tmp_path)
    map_path = tmp_path / "queue" / "k.map.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps({"schema": "prismabuild.residency-map.v1",
                                    "stage_root": str(stage), "entries": {}}))
    spec = _spec(tmp_path)
    environ = {"PRISMABUILD_RESIDENCY_MAP": str(map_path),
               **_bundle(helper)}
    argv = _argv(spec, environ)
    env = _forwarded_env(argv)
    assert env["PRISMABUILD_RESIDENCY_MAP"] == str(map_path)
    assert env["PRISMABUILD_ACTION_KEY"] == KEY
    assert env["PRISMABUILD_READER_HELPER_ROOT"] == str(helper)
    mounts = _mounts(argv)
    assert f"type=bind,src={stage},dst={stage},readonly" in mounts
    assert (f"type=bind,src={helper},dst={helper},readonly"
            in mounts)
