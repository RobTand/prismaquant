"""A Stage B quantum touches no pool input before its readset is active (PQ #1024).

``_load_plan`` hashed the plan's ``source_identity_cache`` and resolved the
``boundary_storage`` directory (an lstat per path component) before the quantum
bound its data manifest, so neither could resolve through residency. The
quantum now loads the plan with ``defer_pool_reads=True``: the identity cache is
digest-checked where it is read (the head slice's declared entry, or
``_seed_source_identity_cache`` on the legacy walk), and the boundary directory
is resolved by the quantum's own storage setup, after the bind. The head slice
itself, the quantum's first head read, now runs after the bind too.

The audit covers ``open`` through ``sys.addaudithook`` and ``stat``/``lstat``
through wrappers on ``os``: Python raises no audit event for a stat, so a hook
alone would pass this test for the wrong reason.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from test_joint_cost_quantum_runtime import (  # noqa: E402,F401 (fixtures)
    _offline_tier_policy,
    _valid_record,
    identity_files,
)
from test_joint_projection_backend import _plan  # noqa: E402
from test_quantum_chain_readset import _argv, _executable  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pool_plan(tmp_path, *, identity_sha256=None):
    """A loadable plan whose identity cache and boundaries live on a 'pool'."""
    pool = tmp_path / "pool"
    (pool / "boundaries").mkdir(parents=True)
    identity = pool / "source-identity.json"
    identity.write_text(json.dumps({"fixture": True}))
    config = _plan(tmp_path / "run")
    config["source_identity_cache"] = {
        "path": str(identity), "sha256": identity_sha256 or _sha(identity)}
    config["execution"]["boundary_storage"] = {
        "schema": "prismaquant.aura.boundary_storage.v1",
        "directory": str(pool / "boundaries"),
        "max_resident_bytes": 1, "max_auxiliary_bytes": 1,
        "max_artifact_bytes": 1, "prefetch_batches": 1}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(config))
    return path, config, pool


class _Touches:
    """Every path opened, stat'ed or lstat'ed while armed."""

    def __init__(self, monkeypatch):
        self.paths, self.armed = [], False
        for name in ("stat", "lstat"):
            real = getattr(os, name)

            def wrapped(path, *args, _real=real, **kwargs):
                self._note(path)
                return _real(path, *args, **kwargs)

            monkeypatch.setattr(os, name, wrapped)
        sys.addaudithook(self._audit)

    def _note(self, raw):
        if self.armed and isinstance(raw, (str, bytes, os.PathLike)):
            self.paths.append(os.path.abspath(os.fsdecode(raw)))

    def _audit(self, event, args):
        # An audit hook cannot be removed; it goes quiet when disarmed.
        if event in {"open", "os.listdir", "os.scandir"} and args:
            self._note(args[0])

    def under(self, root):
        root = os.path.abspath(root)
        return sorted({p for p in self.paths if p == root or p.startswith(root + "/")})


def test_deferred_plan_load_touches_nothing_on_the_pool(tmp_path, monkeypatch):
    from prismaquant.tessera_joint_aura import _load_plan

    path, config, pool = _pool_plan(tmp_path)
    touches = _Touches(monkeypatch)
    touches.armed = True
    try:
        assert _load_plan(path, _sha(path), defer_pool_reads=True) == config
    finally:
        touches.armed = False
    assert touches.paths, "the audit observed no read at all"
    assert touches.under(pool) == []
    # The instrument bites: the default load hashes the identity cache.
    touches.armed = True
    try:
        _load_plan(path, _sha(path))
    finally:
        touches.armed = False
    assert config["source_identity_cache"]["path"] in touches.under(pool)


def test_default_plan_load_still_binds_the_identity_cache(tmp_path):
    """Stage A, the prepare and the tools keep the eager digest check."""
    from prismaquant.tessera_joint_aura import _load_plan

    path, _, _ = _pool_plan(tmp_path, identity_sha256="e" * 64)
    with pytest.raises(ValueError, match="source identity cache: artifact checksum"):
        _load_plan(path, _sha(path))
    # Deferred, the plan loads; the quantum's own read of the cache refuses.
    assert _load_plan(path, _sha(path), defer_pool_reads=True)


def test_deferred_plan_load_still_refuses_a_malformed_identity_binding(tmp_path):
    from prismaquant.tessera_joint_aura import _load_plan

    path, config, _ = _pool_plan(tmp_path)
    config["source_identity_cache"] = {"path": config["source_identity_cache"]["path"]}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="independently bound path/SHA256"):
        _load_plan(path, _sha(path), defer_pool_reads=True)


def test_plan_load_never_stats_the_boundary_directory(tmp_path, monkeypatch):
    """The policy check discards its result, so it resolves nothing."""
    from prismaquant.tessera_joint_aura import _load_plan

    path, config, pool = _pool_plan(tmp_path)
    config.pop("source_identity_cache")
    path.write_text(json.dumps(config))
    touches = _Touches(monkeypatch)
    touches.armed = True
    try:
        _load_plan(path, _sha(path))
    finally:
        touches.armed = False
    assert touches.under(pool) == []


def test_boundary_policy_check_refuses_what_normalize_refuses():
    from prismaquant.cost_streaming import (
        check_boundary_storage, normalize_boundary_storage)

    good = {"schema": "prismaquant.aura.boundary_storage.v1", "directory": "rel/dir",
            "max_resident_bytes": 1, "max_auxiliary_bytes": 1,
            "max_artifact_bytes": 1, "prefetch_batches": 1}
    assert check_boundary_storage(good) == good
    assert normalize_boundary_storage(good)["directory"] == str(Path("rel/dir").resolve())
    assert check_boundary_storage(None) is None
    for bad in ({**good, "prefetch_batches": 0}, {**good, "directory": " "},
                {**good, "schema": "other"}, {**good, "extra": 1}):
        for check in (check_boundary_storage, normalize_boundary_storage):
            with pytest.raises(ValueError, match="exact boundary storage"):
                check(bad)


class _Stop(Exception):
    pass


def test_quantum_cli_loads_its_plan_deferred(identity_files, monkeypatch):
    import prismaquant.tessera_joint_aura as aura_mod
    from prismaquant.joint_cost_quantum import main

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    seen = {}

    def spy(path, digest, **kwargs):
        seen.update(kwargs)
        raise _Stop()

    monkeypatch.setattr(aura_mod, "_load_plan", spy)
    record = _executable(_valid_record(identity_files))
    with pytest.raises(_Stop):
        main(_argv(identity_files, record, "--data-manifest-sha256",
                   record["executable_readset"]["manifest_sha256"]))
    assert seen.get("defer_pool_reads") is True


def test_head_slice_is_read_after_the_manifest_binds(tmp_path, monkeypatch):
    """The first head read resolves through a bound residency map."""
    import prismaquant.gpu_guard as gpu_guard
    import prismaquant.joint_stage_b_head as head_mod
    from prismaquant import residency_map
    from prismaquant.joint_cost_quantum import run_layer_quantum

    digest = "d" * 64
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: None)
    monkeypatch.setenv(residency_map.ENV_VAR, str(tmp_path / "residency-map.json"))
    residency_map.reset_residency_resolver_for_tests()
    bound = []

    def read_slice(*_args, **_kwargs):
        bound.append(residency_map.residency_resolver()._manifest_sha256)
        raise _Stop()

    monkeypatch.setattr(head_mod, "read_quantum_head_slice", read_slice)
    record = {"layer": 1, "quantum_id": "q", "campaign": {},
              "output_space": {"root": str(tmp_path / "space")},
              "executable_readset": {"head_slice": {"path": "slice.json",
                                                    "sha256": "a" * 64}}}
    try:
        with pytest.raises(_Stop):
            run_layer_quantum(
                {"execution": {}}, record=record, adjoint_slice={},
                plan_sha256="c" * 64, prepared={"path": "p", "sha256": "b" * 64},
                output_root=tmp_path, data_manifest_sha256=digest)
    finally:
        residency_map.reset_residency_resolver_for_tests()
    assert bound == [digest]
