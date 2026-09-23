"""Stage B opens no kernel-time profiler session unless asked (PQ #1029).

Stage B wrapped the render-free chain and every retained window in a
``torch.profiler`` session. Each session's close sums every CUDA kernel it
recorded through ``key_averages``, which on a GLM-shaped proxy took 15.6 s of
one quantum's main thread. Stage A made the same profiler opt-in after #899;
Stage B now does the same, under ``PRISMAQUANT_STAGE_B_KERNEL_PROFILE=1``.

These tests drive the real layer quantum over the spill suite's fixture, with
``torch.profiler.profile`` replaced by a session that counts its openings.
"""
from __future__ import annotations

import pytest
import torch

import prismaquant.joint_cost_quantum as core

from test_stageb_one_pass_spill import (  # noqa: F401 (module fixture)
    _chain,
    _clear_output,
    _quantum,
    _spill_root,
    campaign,
)

SESSION_S = 0.25


class _Event:
    self_device_time_total = SESSION_S * 1e6


class _Session:
    opened = 0

    def __init__(self, **_kwargs):
        type(self).opened += 1

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def key_averages(self):
        return [_Event()]


@pytest.fixture
def sessions(monkeypatch):
    _Session.opened = 0
    monkeypatch.setattr(torch.profiler, "profile", _Session)
    return _Session


def _run(campaign, monkeypatch, tmp_path, *, spill):
    _clear_output(campaign, 0)
    kwargs = dict(spill_root=_spill_root(tmp_path), ceiling=1 << 30) if spill else {}
    payload, run = _quantum(campaign, monkeypatch, layer=0, **kwargs)
    assert payload is not None, _chain(run.error)
    return run.counters_block


@pytest.mark.parametrize("value", [None, "0", "true", "yes"],
                         ids=["unset", "zero", "true", "yes"])
@pytest.mark.parametrize("spill", [False, True], ids=["windowed", "spill"])
def test_a_quantum_opens_no_session_unless_asked(sessions, campaign, monkeypatch,
                                                 tmp_path, spill, value):
    if value is None:
        monkeypatch.delenv(core.KERNEL_PROFILE_ENV, raising=False)
    else:
        monkeypatch.setenv(core.KERNEL_PROFILE_ENV, value)
    block = _run(campaign, monkeypatch, tmp_path, spill=spill)
    assert sessions.opened == 0
    # Not measured reads as None and says why, never as a zero.
    assert block["kernel_active_s"] is None
    assert block["kernel_active_ratio"] is None
    assert "#1029" in block["kernel_profiler_error"]
    assert block["chain"]["kernel_active_s"] is None
    assert block["windows"] and all(
        window["kernel_active_s"] is None for window in block["windows"])
    # The power sampler is not the profiler; it stays on.
    assert block["gpu_power_envelope_w"] == 140.0


@pytest.mark.parametrize("spill", [False, True], ids=["windowed", "spill"])
def test_an_operator_can_opt_in(sessions, campaign, monkeypatch, tmp_path, spill):
    monkeypatch.setenv(core.KERNEL_PROFILE_ENV, "1")
    block = _run(campaign, monkeypatch, tmp_path, spill=spill)
    windows = len(block["windows"])
    assert windows > 1
    # One session around the chain, then one around each retained window.
    assert sessions.opened == 1 + windows
    assert block["chain"]["kernel_active_s"] == SESSION_S
    assert all(window["kernel_active_s"] == SESSION_S for window in block["windows"])
    assert block["kernel_active_s"] == pytest.approx(SESSION_S * (1 + windows))
    assert "kernel_profiler_error" not in block


def test_a_chain_sum_that_was_not_measured_stays_none():
    """A later measured step cannot turn an unmeasured chain into a number."""
    frontier = core.ChunkFrontier(chunks=[{"name": "c", "start_bytes": 0, "end_bytes": 1}],
                                  windows=[])
    counters = core.QuantumCounters(quantum_id="layer-000", identity_sha256="0" * 64,
                                    chunks=[{"name": "c", "start_bytes": 0, "end_bytes": 1}],
                                    frontier=frontier)
    counters.chain_step(layers=1, backwards=1, wall_s=1.0, kernel_active_s=None)
    counters.chain_step(layers=1, backwards=1, wall_s=1.0, kernel_active_s=2.0)
    block = counters.finish(units_done=0, units_total=0)
    assert block["chain"] == {"layers": 2, "backwards": 2, "wall_s": 2.0,
                              "kernel_active_s": None}
