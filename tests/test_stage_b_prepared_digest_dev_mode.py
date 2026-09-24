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
