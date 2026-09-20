"""The cuda head walk returns its retained decode pool at unit boundaries (#823).

Stage A v12 (action 7b68ec96, sparky GB10) was reaped by GPU containment at
90.6 GiB against an 80 GiB budget while the head walk was re-verifying and
synthesizing units: the decode path keeps no tensor by reference (the shard
is written to the render file), but the caching allocator retains every
freed block, per-cell shapes vary enough that the pool only grows, and on
shared-system GPUs the retained pool is charged to the action's GPU budget.

Three properties pin the fix:

* **Gap-gated** -- a walk whose freed-but-cached pool is under
  ``HEAD_WALK_RECLAIM_GAP_BYTES`` never pays the reclaim; one past it
  returns the pool to the driver and says so once.
* **Wired at the boundary** -- every unit commitment reaches the reclaim
  hook (the device label decides whether it acts), so a cuda walk cannot
  accumulate retention between boundaries.
* **Never fatal** -- a reclaim that fails is logged and the walk continues;
  memory charge is an observability contract, not a correctness one.
"""

from __future__ import annotations

import pytest

from prismaquant import tessera_joint_aura


class _ScriptedCuda:
    """torch.cuda stand-in with a scripted retained pool."""

    def __init__(self, *, reserved, allocated, fail=None):
        self._reserved = reserved
        self._allocated = allocated
        self._fail = fail
        self.empty_cache_calls = 0

    # torch.cuda API surface the reclaim reads
    def is_available(self):
        return True

    def memory_reserved(self):
        return self._reserved

    def memory_allocated(self):
        return self._allocated

    def empty_cache(self):
        if self._fail is not None:
            raise self._fail
        self.empty_cache_calls += 1
        self._reserved = self._allocated


def _install(monkeypatch, capsys, scripted):
    class _FakeTorch:
        cuda = scripted

    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    capsys.readouterr()
    return scripted


def test_below_gap_never_reclaims(monkeypatch, capsys):
    scripted = _install(
        monkeypatch, capsys,
        _ScriptedCuda(reserved=tessera_joint_aura.HEAD_WALK_RECLAIM_GAP_BYTES - 1,
                      allocated=0))
    tessera_joint_aura._reclaim_head_walk_allocator("cuda")
    assert scripted.empty_cache_calls == 0
    assert capsys.readouterr().out == ""


def test_past_gap_reclaims_once_and_names_the_pool(monkeypatch, capsys):
    scripted = _install(
        monkeypatch, capsys,
        _ScriptedCuda(reserved=tessera_joint_aura.HEAD_WALK_RECLAIM_GAP_BYTES + (512 << 20),
                      allocated=256 << 20))
    tessera_joint_aura._reclaim_head_walk_allocator("cuda")
    assert scripted.empty_cache_calls == 1
    out = capsys.readouterr().out
    assert "head-walk allocator reclaim" in out
    assert "2560 MiB" in out  # the gap, not the allocation, names the pool


def test_cpu_device_is_a_no_op_and_unavailable_cuda_is_silent(monkeypatch, capsys):
    class _Unavailable:
        @staticmethod
        def is_available():
            return False

    class _FakeTorch:
        cuda = _Unavailable()

    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    capsys.readouterr()
    tessera_joint_aura._reclaim_head_walk_allocator("cpu")
    tessera_joint_aura._reclaim_head_walk_allocator("cuda")  # unavailable -> silent
    assert capsys.readouterr().out == ""


def test_failing_reclaim_is_logged_not_raised(monkeypatch, capsys):
    scripted = _install(
        monkeypatch, capsys,
        _ScriptedCuda(reserved=tessera_joint_aura.HEAD_WALK_RECLAIM_GAP_BYTES * 4,
                      allocated=0, fail=RuntimeError("driver refused")))
    tessera_joint_aura._reclaim_head_walk_allocator("cuda")  # must not raise
    assert scripted.empty_cache_calls == 0
    assert "reclaim skipped" in capsys.readouterr().out


def test_every_unit_commitment_reaches_the_reclaim(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        tessera_joint_aura, "_reclaim_head_walk_allocator",
        lambda device, **_: calls.append(device), raising=True)
    from tests.test_tessera_joint_aura import fixture

    config, names, _fmt, _payload, _states = fixture(tmp_path)
    result = tessera_joint_aura.load_measured_anchor_input(
        config, verify_payloads=False, progress_phase=None)
    assert len(result.cells) >= 2
    assert calls == ["cpu"] * len(result.cells)
