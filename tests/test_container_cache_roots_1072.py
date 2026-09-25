"""Container caches under the declared local scratch (PQ #1072, #1014 item f).

When a row declares bounded local scratch (PB #911), the launcher binds HF,
Triton, inductor, XDG and PrismaQuant's temp parent under
``<scratch root>/container-cache/`` for each one the spec leaves unset. A spec
value wins. A row with no scratch launches exactly as before. The dispatcher
warned when a spec pinned one of them to ``/tmp``, ``/var/tmp`` or a path no
writable mount covers; it now refuses unless the spec names a reason
(PQ #1129, ``tests/test_overlay_cache_refusal_1129.py``).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from test_dispatch_joint_quanta import _COTANGENT, _SPILL  # noqa: E402

COT_ROOT = "/home/rob/pb-scratch/cot"
SPILL_ROOT = "/home/rob/pb-scratch/spill"
CACHES = ("HF_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
          "XDG_CACHE_HOME", "PRISMAQUANT_TMPDIR")


def _runner():
    return importlib.import_module("tools.tessera_campaign_container")


def _spec(env=None, roots=()):
    return {"env": {"OMP_NUM_THREADS": "4", **(env or {})},
            "container": {"image": "qualified:fixed", "mounts": [
                {"source": "/mnt/shared", "target": "/mnt/shared"},
                {"source": "/producer", "target": "/producer", "readonly": True},
                *({"source": root, "target": root} for root in roots)]}}


def _docker_env(declared, environ):
    argv = _runner().docker_command(declared, ["python3"], cwd="/worker/snapshot",
                                    uid=1000, gid=1000, image_id="sha256:resolved",
                                    environ=environ)
    return dict(argv[i + 1].split("=", 1) for i, value in enumerate(argv[:-1])
                if value == "--env")


def _scratch(names, root, ceiling):
    return {names[0]: root, names[1]: str(ceiling)}


@pytest.mark.parametrize("names,root", [(_COTANGENT, COT_ROOT), (_SPILL, SPILL_ROOT)],
                         ids=["cotangent", "spill"])
def test_declared_scratch_binds_every_unset_cache_under_its_root(names, root):
    environ = _scratch(names, root, 1 << 30)
    env = _docker_env(_spec(roots=(root,)), environ)
    base = f"{root}/container-cache"
    assert {name: env[name] for name in CACHES} == {
        "HF_HOME": f"{base}/hf", "TRITON_CACHE_DIR": f"{base}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{base}/inductor",
        "XDG_CACHE_HOME": f"{base}/xdg", "PRISMAQUANT_TMPDIR": f"{base}/tmp"}


def test_with_both_kinds_the_cotangent_root_holds_the_caches():
    environ = {**_scratch(_COTANGENT, COT_ROOT, 1 << 30),
               **_scratch(_SPILL, SPILL_ROOT, 2 << 30)}
    env = _docker_env(_spec(roots=(COT_ROOT, SPILL_ROOT)), environ)
    assert all(env[name].startswith(f"{COT_ROOT}/container-cache/") for name in CACHES)


def test_a_spec_value_wins_over_the_default_and_is_reported():
    runner = _runner()
    environ = _scratch(_SPILL, SPILL_ROOT, 1 << 30)
    declared = _spec({"HF_HOME": "/tmp/pq-hf",
                      "TRITON_CACHE_DIR": "/mnt/shared/triton"}, roots=(SPILL_ROOT,))
    env = _docker_env(declared, environ)
    assert env["HF_HOME"] == "/tmp/pq-hf"
    assert env["TRITON_CACHE_DIR"] == "/mnt/shared/triton"
    assert env["XDG_CACHE_HOME"] == f"{SPILL_ROOT}/container-cache/xdg"
    _defaults, pinned = runner.container_cache_environment(
        declared, runner.stage_b_spill_environment(declared, environ))
    # /mnt/shared is a writable mount, so only the /tmp pin is the overlay.
    assert pinned == ["HF_HOME"]


@pytest.mark.parametrize("value,overlay", [
    ("/tmp/x", True), ("/var/tmp/x", True), ("/scratch/unmounted", True),
    ("/producer/cache", True), ("/mnt/shared/cache", False)],
    ids=["tmp", "var-tmp", "unmounted", "readonly-mount", "writable-mount"])
def test_overlay_pins_are_named(value, overlay):
    _defaults, pinned = _runner().container_cache_environment(
        _spec({"TORCHINDUCTOR_CACHE_DIR": value}), {})
    assert pinned == (["TORCHINDUCTOR_CACHE_DIR"] if overlay else [])


def test_no_scratch_launches_exactly_as_before(monkeypatch):
    runner = _runner()
    declared = _spec()
    argv = runner.docker_command(declared, ["python3"], cwd="/worker/snapshot",
                                 uid=1000, gid=1000, image_id="sha256:resolved", environ={})
    monkeypatch.setattr(runner, "CONTAINER_CACHE_ENV", ())
    assert runner.docker_command(declared, ["python3"], cwd="/worker/snapshot",
                                 uid=1000, gid=1000, image_id="sha256:resolved",
                                 environ={}) == argv
    assert not any(part.split("=", 1)[0] in CACHES for part in argv)
