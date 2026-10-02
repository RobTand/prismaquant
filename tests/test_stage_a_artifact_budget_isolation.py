
"""A budget fixture must preserve dispatcher globals borrowed by other tests.

The real budget teardown runs between collection-time dispatcher imports and
portable-spec use. Guard the original spec before any read so the regression
cannot reach a live campaign's artifacts.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

import test_dispatch_joint_quanta as dispatch_tests
import test_stage_a_artifact_budget as budget_tests


def _run_budget_predecessor(tmp_path):
    cleanup = budget_tests._release_process_state_this_file_installs.__wrapped__()
    next(cleanup)
    budget_tests.test_resolve_defaults_to_plan_verbatim(tmp_path)
    with pytest.raises(StopIteration):
        next(cleanup)


def test_budget_teardown_preserves_a_retained_quantum_callable(tmp_path, monkeypatch):
    retained = dispatch_tests.quantum_argv
    original = sys.modules["dispatch_joint_quanta"]
    original_spec = retained.__globals__["SPEC_PATH"]
    read_text = Path.read_text

    def forbid_original_spec(path, *args, **kwargs):
        if path == original_spec:
            pytest.fail("retained quantum_argv reached the original SPEC_PATH")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", forbid_original_spec)
    # Restore the borrowed module even on the deliberately failing baseline.
    monkeypatch.setitem(sys.modules, "dispatch_joint_quanta", original)
    _run_budget_predecessor(tmp_path)
    dispatch_tests._portable_spec.__wrapped__(tmp_path, monkeypatch)
    campaign = dispatch_tests.campaign.__wrapped__(tmp_path)
    build = dispatch_tests._scratch_quantum_argv(tmp_path, campaign, {}, ())
    argv = build()
    sealed = json.loads(argv[argv.index("--spec") + 1])
    assert sys.modules["dispatch_joint_quanta"] is original
    assert sys.modules["dispatch_joint_quanta"].quantum_argv is retained
    assert sealed["env"] == {}


def test_budget_teardown_removes_only_a_dispatcher_it_installed(tmp_path, monkeypatch):
    with monkeypatch.context() as context:
        context.delitem(sys.modules, "dispatch_joint_quanta")
        cleanup = budget_tests._release_process_state_this_file_installs.__wrapped__()
        next(cleanup)
        importlib.import_module("dispatch_joint_quanta")
        with pytest.raises(StopIteration):
            next(cleanup)
        assert "dispatch_joint_quanta" not in sys.modules
