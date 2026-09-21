"""Stage A holds no whole-capture ``torch.profiler`` session (PQ #899).

On 2026-09-21 the 512-sample capture stopped its profiler in the ``finally``
of a run that had already failed. The stop grew host memory by 22 GB in two
minutes, the kernel killed the process, and the capture's own exception never
printed.
"""
from __future__ import annotations

import ast
import inspect

import pytest
import torch

from prismaquant import joint_cost_stage_a as stage_a
from prismaquant.joint_adjoint_checkpoints import KernelTimeProfiler


class _Event:
    def __init__(self, microseconds):
        self.self_device_time_total = microseconds


class _Session:
    opened = 0

    def __init__(self, **_kwargs):
        type(self).opened += 1

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def key_averages(self):
        return [_Event(1_500_000), _Event(500_000)]


@pytest.fixture
def sessions(monkeypatch):
    _Session.opened = 0
    monkeypatch.setattr(torch.profiler, "profile", _Session)
    return _Session


def test_the_capture_opens_no_session_unless_asked(sessions, monkeypatch):
    monkeypatch.delenv(stage_a.KERNEL_PROFILE_ENV, raising=False)
    kernel = stage_a._stage_a_kernel_profiler()
    kernel.__enter__()
    kernel.__exit__(None, None, None)
    assert sessions.opened == 0
    block = kernel.block()
    assert block["kernel_active_s"] is None
    assert "#899" in block["profiler_error"]


def test_a_profiler_told_not_to_measure_reports_why_and_not_a_zero(sessions):
    kernel = KernelTimeProfiler(not_measured="not measured: a reason")
    with kernel:
        pass
    assert sessions.opened == 0
    assert kernel.block() == {"kernel_active_s": None,
                              "profiler_error": "not measured: a reason"}


def test_an_operator_can_opt_in(sessions, monkeypatch):
    monkeypatch.setenv(stage_a.KERNEL_PROFILE_ENV, "1")
    kernel = stage_a._stage_a_kernel_profiler()
    kernel.__enter__()
    kernel.__exit__(None, None, None)
    assert sessions.opened == 1
    assert kernel.block() == {"kernel_active_s": 2.0}


def test_a_bounded_scope_still_measures_by_default(sessions):
    with KernelTimeProfiler() as kernel:
        pass
    assert sessions.opened == 1
    assert kernel.block() == {"kernel_active_s": 2.0}


def _capture_function() -> ast.FunctionDef:
    tree = ast.parse(inspect.getsource(stage_a.run_adjoint_capture))
    return tree.body[0]


def test_the_capture_builds_its_profiler_through_the_opt_in():
    calls = {node.func.id for node in ast.walk(_capture_function())
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "_stage_a_kernel_profiler" in calls
    assert "KernelTimeProfiler" not in calls


def test_the_capture_says_why_it_failed_before_it_tears_anything_down():
    guarded = [node for node in ast.walk(_capture_function())
               if isinstance(node, ast.Try) and node.finalbody]
    teardown = [block for block in guarded if any(
        isinstance(node, ast.Attribute) and node.attr == "__exit__"
        for statement in block.finalbody for node in ast.walk(statement))]
    assert len(teardown) == 1
    handlers = teardown[0].handlers
    assert [handler.type.id for handler in handlers] == ["BaseException"]
    body = handlers[0].body
    assert isinstance(body[0], ast.Expr) and body[0].value.func.id == "print"
    assert isinstance(body[-1], ast.Raise) and body[-1].exc is None
