"""Sample-major produced reads for the fused chain roll (RobTand/prismaquant#997).

The fused roll reads, in every window, one boundary group and one incoming
cotangent group per probe, and the next window of the same pass reads the
same groups again until they end. These tests drive the real bound owner
over a real queue, as ``test_stage_a_produced_readahead`` does: the read
order's share of the sealed window, its refusal when the window cannot hold
one read, and the retention that keeps every group staged across window
exits so a group is staged once, not once per window.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import test_stage_a_produced_boundary_chain as chain
# Autouse, as in the read-ahead tests: it isolates each test's launch tuple
# and strict tier policy from the next test's.
from test_stage_a_produced_boundary_chain import (  # noqa: F401
    _isolated_launch_context)
from test_stage_a_produced_readahead import _stager_mode  # noqa: F401

GROUP_SIZE = chain.GROUP_SIZE
PROBES = 2


def _fused_owner(tmp_path, *, window_gib=8):
    return chain._bound_owner(
        tmp_path, n_batches=GROUP_SIZE, window_gib=window_gib,
        gib=max(2 * window_gib, 4), payload_max_bytes=1 << 22,
        n_probes=PROBES, read_order="sample_major")


def _value(boundary, probe, batch):
    return torch.arange(8, dtype=torch.float32) + 100 * boundary + 10 * (probe + 1) + batch


def _write_planes(storage):
    """Boundary 1's group and both probes' incoming groups at boundary 2."""
    boundary = [storage.write(_value(1, -1, batch), batch_index=batch, boundary_index=1)
                for batch in range(GROUP_SIZE)]
    incoming = [[storage.write(_value(2, probe, batch), batch_index=batch,
                               boundary_index=2, probe_index=probe)
                 for batch in range(GROUP_SIZE)] for probe in range(PROBES)]
    batches = [SimpleNamespace(activations_cpu={1: reference}) for reference in boundary]
    return batches, incoming


def _fused_read(storage, batches, incoming, *, window_batches):
    from prismaquant.cost_streaming import prefetched_fused_boundary_windows

    seen = []
    with prefetched_fused_boundary_windows(
            storage, batches, 1, incoming=incoming,
            window_batches=window_batches) as windows:
        for indices, boundary, incoming_of in windows:
            for batch in indices:
                assert torch.equal(boundary(batch), _value(1, -1, batch))
                for probe in range(PROBES):
                    assert torch.equal(incoming_of(probe, batch), _value(2, probe, batch))
                seen.append(batch)
    return seen


def test_a_sample_major_read_owns_one_group_per_probe_beside_the_boundary(tmp_path):
    storage, *_rest = _fused_owner(tmp_path)
    plan = storage._produced_plan
    assert plan["read_groups"] == 1 + PROBES
    assert plan["ahead_groups"] == plan["window_groups"] - (1 + PROBES)
    report = storage.produced_output_report()
    assert report["read_order"] == "sample_major"
    assert report["read_groups"] == 1 + PROBES


def test_a_window_that_cannot_hold_one_fused_read_is_refused_at_bind(tmp_path):
    with pytest.raises(ValueError, match="sample_major read window holds 3"):
        _fused_owner(tmp_path, window_gib=2)


def test_fused_windows_stage_each_group_once_across_the_pass(tmp_path, monkeypatch):
    """Two windows of two batches read the same three groups: staged once."""
    storage, _publication, q, env, pb_repo = _fused_owner(tmp_path)
    batches, incoming = _write_planes(storage)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        assert _fused_read(storage, batches, incoming, window_batches=2) == \
            list(range(GROUP_SIZE))
        storage.settle_produced_releases()
    telemetry = storage.telemetry
    assert telemetry["produced_groups_materialized"] == 1 + PROBES
    assert telemetry["produced_groups_rematerialized"] == 0, (
        "the second window found every group still staged")
    assert telemetry["produced_groups_retained"] >= 1 + PROBES
    assert all(record["retired"] for record in storage.produced_group_records())
    assert storage.produced_release_debt() == chain._NO_DEBT
    assert storage._produced_retained_reads == frozenset(), (
        "leaving the roll retains nothing")


def test_without_retention_every_window_stages_its_groups_again(tmp_path, monkeypatch):
    """The driver, mutated: with ``retain_produced_reads`` a no-op, the second
    window stages all three groups again. This is what the retention saves."""
    storage, _publication, q, env, pb_repo = _fused_owner(tmp_path)
    monkeypatch.setattr(storage, "retain_produced_reads", lambda references: 0)
    batches, incoming = _write_planes(storage)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        _fused_read(storage, batches, incoming, window_batches=2)
        storage.settle_produced_releases()
    assert storage.telemetry["produced_groups_rematerialized"] == 1 + PROBES
    assert storage.produced_release_debt() == chain._NO_DEBT
