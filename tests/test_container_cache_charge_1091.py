"""An explicit cache pair is charged separately, not mistaken for crash cleanup."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from test_container_cache_roots_1072 import _docker_env, _scratch, _spec
from test_dispatch_joint_quanta import _COTANGENT, _SPILL

CACHE = ("PRISMAQUANT_CONTAINER_CACHE_ROOT", "PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES")
COMPILATION = ("HF_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME")
# Pure Docker-argv fixtures declare container paths, not pytest's /tmp overlay.
# Filesystem/symlink refusal tests below deliberately retain their real tmp_path.
CONTAINER_FIXTURE_ROOT = Path("/pq-fixture-cache-charge-1091")


def _runner():
    return importlib.import_module("tools.tessera_campaign_container")


def _layout(tmp_path, *, cache=None, pins=None):
    cot, spill = str(tmp_path / "cot"), str(tmp_path / "spill")
    cache = str(cache or tmp_path / "compile")
    env = {**_scratch(_COTANGENT, cot, 1), **_scratch(_SPILL, spill, 1 << 30),
           **_scratch(CACHE, cache, (1 << 30) + 1),
           "PRISMAQUANT_TMPDIR": f"{spill}/row-tmp", **(pins or {})}
    return _spec(env, roots=tuple(dict.fromkeys((cot, spill, cache)))), env


def test_cache_pair_is_priced_independently_without_creating_directories(tmp_path):
    runner = _runner()
    spec, env = _layout(tmp_path)
    declared = runner.local_scratch_environment(spec, env)
    pairs = declared[runner.LOCAL_SCRATCH_PAIRS_ENV]
    assert pairs == ",".join(f"{a}:{b}" for a, b in (_COTANGENT, _SPILL, CACHE))
    assert {name: declared[name] for name in CACHE} == {name: env[name] for name in CACHE}
    if importlib.util.find_spec("prismabuild") is None:
        pytest.skip("public PB SDK unavailable; pricing requires a qualified PB-worker receipt")
    # A present but incompatible/broken SDK must fail, not be skipped.
    from prismabuild.local_scratch import scratch_terms
    assert scratch_terms(declared) == {"spool_gb": 4}
    assert list(tmp_path.iterdir()) == []


def test_cache_forwarding_and_defaults_do_not_reuse_spill_or_tmp():
    spec, env = _layout(CONTAINER_FIXTURE_ROOT)
    inside = _docker_env(spec, env)
    assert {name: inside[name] for name in CACHE} == {name: env[name] for name in CACHE}
    assert {name: inside[name] for name in COMPILATION} == {
        "HF_HOME": env[CACHE[0]] + "/hf",
        "TRITON_CACHE_DIR": env[CACHE[0]] + "/triton",
        "TORCHINDUCTOR_CACHE_DIR": env[CACHE[0]] + "/inductor",
        "XDG_CACHE_HOME": env[CACHE[0]] + "/xdg"}
    assert inside["PRISMAQUANT_TMPDIR"] == env["PRISMAQUANT_TMPDIR"]
    assert "PRISMABUILD_LOCAL_SCRATCH_PAIRS" not in inside


@pytest.mark.parametrize("cache_relation", ["same", "ancestor", "descendant"])
def test_distinct_cache_refuses_overlapping_scratch_roots(tmp_path, cache_relation):
    root = {"same": tmp_path / "spill", "ancestor": tmp_path,
            "descendant": tmp_path / "spill" / "compile"}[cache_relation]
    spec, env = _layout(tmp_path, cache=root)
    # A descendant is also a nested mount beneath spill: the existing
    # per-kind guard may reject it before the cache-specific disjoint check.
    with pytest.raises(RuntimeError, match="disjoint|nested container mounts"):
        _runner().local_scratch_environment(spec, env)


def test_explicitly_empty_cache_pair_is_not_treated_as_absent(tmp_path):
    spec, env = _layout(tmp_path)
    for name in CACHE:
        spec["env"][name] = env[name] = ""
    with pytest.raises(RuntimeError, match="positive byte ceiling"):
        _runner().local_scratch_environment(spec, env)


@pytest.mark.parametrize("missing", CACHE)
def test_partial_cache_declaration_refuses(tmp_path, missing):
    spec, env = _layout(tmp_path)
    del spec["env"][missing]
    del env[missing]
    with pytest.raises(RuntimeError):
        _runner().local_scratch_environment(spec, env)


@pytest.mark.parametrize("value", ["0", "-1", "", "١", "１", "1.0"])
def test_cache_ceiling_uses_the_public_pb_ascii_positive_contract(tmp_path, value):
    spec, env = _layout(tmp_path)
    spec["env"][CACHE[1]] = env[CACHE[1]] = value
    with pytest.raises(RuntimeError, match="positive byte ceiling"):
        _runner().local_scratch_environment(spec, env)


@pytest.mark.parametrize("pin", ["/tmp/hf", "/var/tmp/hf", "/mnt/shared/hf", "/unmounted/hf"])
def test_an_overlay_waiver_cannot_route_distinct_cache_outside_its_charge(pin):
    spec, env = _layout(CONTAINER_FIXTURE_ROOT, pins={"HF_HOME": pin})
    spec["overlay_cache_reason"] = "fixture: must not waive accounting"
    with pytest.raises(RuntimeError, match="HF_HOME.*cache root"):
        _docker_env(spec, env)


def test_tmpdir_requires_an_explicit_other_charged_root():
    spec, env = _layout(CONTAINER_FIXTURE_ROOT)
    del spec["env"]["PRISMAQUANT_TMPDIR"]
    with pytest.raises(RuntimeError, match="TMPDIR.*separate"):
        _docker_env(spec, env)


@pytest.mark.parametrize("bad_tmp", ["compile/row-tmp", "outside/row-tmp"])
def test_tmpdir_cannot_reuse_cache_or_an_uncharged_directory(bad_tmp):
    spec, env = _layout(CONTAINER_FIXTURE_ROOT,
                        pins={"PRISMAQUANT_TMPDIR": str(CONTAINER_FIXTURE_ROOT / bad_tmp)})
    with pytest.raises(RuntimeError, match="TMPDIR.*separate"):
        _docker_env(spec, env)


def test_distinct_cache_refuses_existing_symlink_ancestors(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    spec, env = _layout(tmp_path, cache=alias / "cache")
    with pytest.raises(RuntimeError, match="symlink"):
        _runner().local_scratch_environment(spec, env)
    assert list(actual.iterdir()) == []


@pytest.mark.parametrize("mode", ["readonly", "remapped", "nested", "unmounted"])
def test_cache_requires_an_unambiguous_writable_identity_bind(tmp_path, mode):
    spec, env = _layout(tmp_path)
    mounts = spec["container"]["mounts"]
    mount = next(m for m in mounts if m["target"] == env[CACHE[0]])
    if mode == "readonly":
        mount["readonly"] = True
    elif mode == "remapped":
        mount["source"] = str(tmp_path / "elsewhere")
    elif mode == "nested":
        mounts.append({"source": str(tmp_path / "elsewhere"),
                       "target": env[CACHE[0]] + "/hf", "readonly": False})
    else:
        mounts.remove(mount)
    with pytest.raises(RuntimeError, match="container cache.*(identity bind|nested)"):
        _runner().local_scratch_environment(spec, env)


@pytest.mark.parametrize("overlay", ["/tmp", "/var/tmp"])
def test_identity_mount_does_not_waive_overlay_cache_refusal(overlay):
    spec, env = _layout(Path(overlay) / "pq-cache-fixture")
    with pytest.raises(RuntimeError, match="HF_HOME.*cache root"):
        _docker_env(spec, env)


def test_contained_cache_pins_are_preserved():
    cache = CONTAINER_FIXTURE_ROOT / "compile"
    spec, env = _layout(CONTAINER_FIXTURE_ROOT, pins={"HF_HOME": str(cache / "pinned-hf")})
    inside = _docker_env(spec, env)
    assert inside["HF_HOME"] == str(cache / "pinned-hf")
    assert inside["TRITON_CACHE_DIR"] == str(cache / "triton")


def test_spec_and_sealed_cache_pair_must_agree(tmp_path):
    spec, env = _layout(tmp_path)
    env[CACHE[1]] = "4"
    with pytest.raises(RuntimeError, match="disagrees"):
        _runner().local_scratch_environment(spec, env)
