"""A spec that pins a container cache to the overlay refuses (PQ #1129).

PQ #1072 bound HF, Triton, inductor, XDG and ``PRISMAQUANT_TMPDIR`` under the
declared scratch root for each one a spec leaves unset, and only warned
(``OverlayCacheWarning``) when a spec pinned one to ``/tmp``, ``/var/tmp`` or
a path no writable mount covers. The warning fired on every Stage B prepare
of R13 (``stage-b-spec.r13.json`` pinned all five under
``/var/tmp/pq-stageb-full512``) and nothing acted on it. A warning that
nothing consumes is not a gate.

The dispatcher's spec check (``_container_wrap``, which the Stage B prepare's
``check_stage_b_spec`` also runs) now refuses such a spec, unless the spec
names why in ``overlay_cache_reason``. The refusal names each pin and that
field. An admitted spec is sealed with an ``overlay_cache_admission`` stamp
that repeats the reason beside the pins it admits, so the sealed request
carries the waiver.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from test_dispatch_joint_quanta import (  # noqa: E402,F401  (fixtures)
    _SPILL, _outer_env, _scratch_quantum_argv, _portable_spec, campaign)

SPILL_ROOT = "/home/rob/pb-scratch/stage-b-spill"
#: The five pins R13's Stage B spec carried (PQ #1129's warning, verbatim).
R13_PINS = {
    "HF_HOME": "/var/tmp/pq-stageb-full512/hf",
    "TRITON_CACHE_DIR": "/var/tmp/pq-stageb-full512/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/var/tmp/pq-stageb-full512/inductor",
    "XDG_CACHE_HOME": "/var/tmp/pq-stageb-full512/xdg",
    "PRISMAQUANT_TMPDIR": "/var/tmp/pq-stageb-full512/joint",
}
REASON = ("layer-44 measurement gate: compare memory with meta-045-m1060 "
          "under the same pins")


def _dispatch():
    return importlib.import_module("dispatch_joint_quanta")


def _spill_env(**pins):
    return {_SPILL[0]: SPILL_ROOT, _SPILL[1]: str(1 << 30), **pins}


def _build(tmp_path, campaign, env, *, roots=(SPILL_ROOT,), **fields):
    """A quantum row's argv builder under a spec with top-level ``fields``."""
    build = _scratch_quantum_argv(tmp_path, campaign, env, roots)
    dispatch = _dispatch()
    spec = json.loads(dispatch.SPEC_PATH.read_text())
    dispatch.SPEC_PATH.write_text(json.dumps({**spec, **fields}))
    return build


def _sealed(argv):
    return json.loads(argv[argv.index("--spec") + 1])


def test_a_pinned_spec_refuses_and_names_the_pins_and_the_reason_field(
        tmp_path, campaign):
    dispatch = _dispatch()
    build = _build(tmp_path, campaign, _spill_env(**R13_PINS))
    with pytest.raises(dispatch.DispatchRefused) as refused:
        build()
    message = str(refused.value)
    for name, value in R13_PINS.items():
        assert f"{name}={value}" in message, message
    assert "overlay_cache_reason" in message
    assert "unset them to bind them under the declared scratch root" in message


def test_a_pinned_spec_without_scratch_refuses_too(tmp_path, campaign):
    dispatch = _dispatch()
    build = _build(tmp_path, campaign, {"HF_HOME": "/tmp/pq-hf"}, roots=())
    with pytest.raises(dispatch.DispatchRefused,
                       match=r"HF_HOME=/tmp/pq-hf.*declare a bounded local"):
        build()


def test_the_same_spec_with_a_reason_is_admitted_and_stamped(tmp_path, campaign):
    env = _spill_env(**R13_PINS)
    argv = _build(tmp_path, campaign, env, overlay_cache_reason=REASON)()
    sealed = _sealed(argv)
    assert sealed["overlay_cache_admission"] == {
        "pinned": R13_PINS, "reason": REASON}
    # The declaration is sealed as written: the pins, the reason, and no
    # default derived into the env.
    assert sealed["env"] == env
    assert sealed["overlay_cache_reason"] == REASON


def test_an_unpinned_spec_binds_under_the_scratch_root(tmp_path, campaign):
    env = _spill_env()
    argv = _build(tmp_path, campaign, env)()
    sealed = _sealed(argv)
    assert "overlay_cache_admission" not in sealed
    assert sealed["env"] == env
    # The launcher, given the scratch pair pbrun seals, binds all five
    # under the spill root.
    runner = importlib.import_module("tools.tessera_campaign_container")
    environ = dict(item.split("=", 1) for item in _outer_env(argv))
    docker = runner.docker_command(
        sealed, ["python3"], cwd="/worker/snapshot", uid=1000, gid=1000,
        image_id="sha256:resolved", environ=environ)
    forwarded = dict(docker[i + 1].split("=", 1)
                     for i, word in enumerate(docker[:-1]) if word == "--env")
    base = f"{SPILL_ROOT}/container-cache"
    assert {name: forwarded[name] for name in R13_PINS} == {
        "HF_HOME": f"{base}/hf", "TRITON_CACHE_DIR": f"{base}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{base}/inductor",
        "XDG_CACHE_HOME": f"{base}/xdg", "PRISMAQUANT_TMPDIR": f"{base}/tmp"}


def test_a_pin_on_a_writable_mount_needs_no_reason(tmp_path, campaign):
    """A declared writable mount is not the overlay: nothing to admit."""
    cache = "/home/rob/pb-scratch/hf"
    argv = _build(tmp_path, campaign, _spill_env(HF_HOME=cache),
                  roots=(SPILL_ROOT, cache))()
    assert "overlay_cache_admission" not in _sealed(argv)


@pytest.mark.parametrize("reason", ["", "   ", 7, ["why"], None],
                         ids=["empty", "blank", "int", "list", "null"])
def test_a_reason_that_says_nothing_refuses(tmp_path, campaign, reason):
    dispatch = _dispatch()
    build = _build(tmp_path, campaign, _spill_env(HF_HOME="/tmp/pq-hf"),
                   overlay_cache_reason=reason)
    with pytest.raises(dispatch.DispatchRefused, match="overlay_cache_reason"):
        build()


def test_a_reason_with_nothing_pinned_refuses(tmp_path, campaign):
    """A stale waiver would ride every later revision of the spec."""
    dispatch = _dispatch()
    build = _build(tmp_path, campaign, _spill_env(), overlay_cache_reason=REASON)
    with pytest.raises(dispatch.DispatchRefused,
                       match="pins no container cache to the overlay"):
        build()


def test_a_spec_cannot_declare_the_admission_stamp(tmp_path, campaign):
    dispatch = _dispatch()
    build = _build(tmp_path, campaign, _spill_env(HF_HOME="/tmp/pq-hf"),
                   overlay_cache_reason=REASON,
                   overlay_cache_admission={"pinned": {}, "reason": REASON})
    with pytest.raises(dispatch.DispatchRefused,
                       match="overlay_cache_admission is derived"):
        build()


def test_the_stage_b_prepare_check_refuses_a_pinned_spec(tmp_path):
    """``check_stage_b_spec`` runs the dispatcher's wrapper: the pins that
    warned on every R13 prepare now refuse before any metadata is made."""
    from tests.test_prepare_extended_joint_quanta import _POLICY, _stage_b_spec
    from tools.prepare_extended_joint_quanta import check_stage_b_spec

    path, spec = _stage_b_spec(tmp_path, 600)
    check_stage_b_spec(path, spec, _POLICY)
    spec["env"].update(R13_PINS)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=r"refused by the quantum dispatcher: "
                                         r"the campaign spec pins"):
        check_stage_b_spec(path, spec, _POLICY)
    spec["overlay_cache_reason"] = REASON
    path.write_text(json.dumps(spec))
    check_stage_b_spec(path, spec, _POLICY)
