"""Stage B's post-intake prepared check honors PRISMAQUANT_DEV_MODE (#771).

``run_layer_quantum`` compares the prepared completion with the running pass
twice: once at startup through ``_preflight_run_prepared`` and once after
the head intake. #771 made the plan and implementation digests records under
dev mode through one shared reader, ``_prepared_digest_recorded``. The Stage
B runtime (#778) landed after #771 with a bare ``_same`` at the second site,
so under dev mode it printed the [DEV-MODE] line at startup, ran the whole
head intake, and then refused on the same digest.

The GLM extended prepared completion copies the original prepared record,
so it carries the original ``implementation_sha256`` (192e73f9...). No tree
that can run Stage B hashes to that value. Both sites must agree: certified
mode refuses at both, dev mode records at both.

CPU-only.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from prismaquant import dev_mode
from prismaquant import tessera_joint_aura as bridge

ROOT = Path(__file__).resolve().parents[1]
DEV_ENV = dev_mode.DEV_MODE_ENV


def _completion(*, plan_sha256="a" * 64, implementation_sha256="b" * 64):
    return {"schema": bridge.PREPARED_SCHEMA, "status": "complete",
            "plan_sha256": plan_sha256,
            "implementation_sha256": implementation_sha256}


def test_certified_mode_refuses_a_digest_mismatch(monkeypatch):
    monkeypatch.setenv(DEV_ENV, "0")
    with pytest.raises(ValueError, match="prepared implementation_sha256"):
        bridge.require_prepared_digests(_completion(), plan_sha256="a" * 64,
                                        implementation_sha256="d" * 64)
    with pytest.raises(ValueError, match="prepared plan_sha256"):
        bridge.require_prepared_digests(_completion(), plan_sha256="c" * 64,
                                        implementation_sha256="b" * 64)


def test_certified_mode_admits_matching_digests(monkeypatch):
    monkeypatch.setenv(DEV_ENV, "0")
    bridge.require_prepared_digests(_completion(), plan_sha256="a" * 64,
                                    implementation_sha256="b" * 64)


def test_dev_mode_records_both_digests(monkeypatch, capsys):
    monkeypatch.setenv(DEV_ENV, "1")
    bridge.require_prepared_digests(_completion(), plan_sha256="c" * 64,
                                    implementation_sha256="d" * 64)
    out = capsys.readouterr().out
    assert "DEV-MODE" in out
    for stored, running in (("a" * 64, "c" * 64), ("b" * 64, "d" * 64)):
        assert stored in out and running in out


def _run_layer_quantum():
    source = (ROOT / "prismaquant" / "joint_cost_quantum.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "run_layer_quantum":
            return node
    raise AssertionError("run_layer_quantum not found")


def _string_args(call):
    return [arg.value for arg in ast.walk(call)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]


def test_stage_b_post_intake_check_uses_the_shared_reader():
    function = _run_layer_quantum()
    bare = []
    shared = 0
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", getattr(node.func, "attr", None))
        if name == "_same" and any(value in ("implementation_sha256", "plan_sha256")
                                   for value in _string_args(node)):
            bare.append(node.lineno)
        if name == "require_prepared_digests":
            shared += 1
    assert bare == [], f"bare prepared digest walls at lines {bare}"
    assert shared == 1


@pytest.mark.parametrize("site", ["completion", "cache"])
@pytest.mark.parametrize("certified", [False, True])
def test_prepared_backend_change_is_one_seal_at_completion_and_cache(
        monkeypatch, capsys, site, certified):
    from copy import deepcopy
    from prismaquant import joint_projection_backend as backend
    from prismaquant.production_weight_cache import ProductionWeightCache

    qualification, digest = backend._qualification()
    stored = dict(schema=backend.SCHEMA, name=backend.FUSED_NAME,
                  qualification_sha256=digest, build=qualification['build'],
                  runtime=qualification['runtime'],
                  qualified_shapes=qualification['qualified_shapes'],
                  ineligible_layout='torch_reference')
    running = deepcopy(backend.REFERENCE_IDENTITY)
    completion = {**_completion(), 'reader_identity': {'reader': 'same'},
                  'projection_backend': deepcopy(stored)}
    cache = ProductionWeightCache({}, {}, metadata={'projection_backend': deepcopy(stored)})
    before = deepcopy((completion, cache.metadata))
    monkeypatch.setenv(DEV_ENV, "0" if certified else "1")

    def intake():
        if site == "completion":
            return bridge.check_prepared_completion(completion,
                plan_sha256="a" * 64, implementation_sha256="b" * 64,
                reader_identity={'reader': 'same'}, projection_backend=running)
        return bridge.require_prepared_binding('projection_backend',
            cache.metadata['projection_backend'], running,
            where='prepared backend identity')

    if certified:
        with pytest.raises(ValueError, match='prepared (projection_backend|backend identity)'):
            intake()
        assert 'DEV-MODE' not in capsys.readouterr().out
    else:
        intake()
        out = capsys.readouterr().out
        assert 'DEV-MODE' in out and 'prepared projection_backend' in out
        # The shared seal reports its first differing field, not a full dump
        # of both identities. The fused build is absent from the reference.
        assert 'build' in out and 'None' in out
        assert qualification['build']['binary_sha256'][:16] in out
    # Recording a seal must not rewrite either prepared record or its cache.
    assert (completion, cache.metadata) == before


@pytest.mark.parametrize("key", ['reader_identity', 'source_model_identity',
                                'calibration_input', 'formats_by_qname'])
def test_backend_recording_does_not_relax_other_prepared_fields(monkeypatch, key):
    monkeypatch.setenv(DEV_ENV, "1")
    with pytest.raises(ValueError, match='prepared ' + key):
        bridge.require_prepared_binding(key, {'stored': True}, {'running': True})
