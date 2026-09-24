"""A split quantum's produced output stays inside its own range (PQ #738).

Several quanta of one Stage A run write into one generation, each through
its own PrismaBuild owner. A produced-output group is 64 entries of one
plane keyed by ``batch // group``, and PrismaBuild refuses a second live
owner of a path. So a quantum must claim, prewrite and publish only the
groups of its own range: a prewrite ahead of the writer that reached into
the next range would claim another quantum's paths. These run the real
bound owner over a real queue, as the produced-output chain tests do.
"""
from __future__ import annotations

import pytest

import test_stage_a_produced_boundary_chain as chain
# Autouse (RobTand/prismaquant#889): drops the outer launch tuple before each
# test and deactivates the strict tier policy after it.
from test_stage_a_produced_boundary_chain import (  # noqa: F401
    _isolated_launch_context)
from test_stage_a_produced_stager import closing  # noqa: F401  (fixture)

pytestmark = pytest.mark.own_process

GROUP = chain.GROUP_SIZE


def _owner(tmp_path, batch_range):
    return chain._bound_owner(
        tmp_path, n_batches=3 * GROUP, window_gib=8, gib=16,
        payload_max_bytes=1 << 22, batch_range=batch_range)


def test_a_ranged_owner_never_claims_the_next_ranges_group(tmp_path, closing):
    storage, _publication, _q, _env, _pb = _owner(tmp_path, (GROUP, 2 * GROUP))
    closing(storage)
    assert storage._stager is not None, "the wide window runs the stager"
    chain._write_group(storage, count=1, first=GROUP)
    assert storage.drain_produced_stager(60.0)
    assert sorted(key[3] for key in storage._produced_groups) == [1], (
        "the first entry of the range's last group claims nothing past the range")
    assert storage.telemetry["produced_groups_prewritten_ahead"] == 0
    chain._write_group(storage, count=GROUP - 1, first=GROUP + 1)
    assert storage.drain_produced_stager(60.0)
    (group,) = storage._produced_groups.values()
    assert len(group["references"]) == GROUP


def test_a_ranged_owner_still_prewrites_ahead_inside_its_range(tmp_path, closing):
    storage, _publication, _q, _env, _pb = _owner(tmp_path, (0, 2 * GROUP))
    closing(storage)
    chain._write_group(storage, count=1)
    assert storage.drain_produced_stager(60.0)
    assert sorted(key[3] for key in storage._produced_groups) == [0, 1]
    assert storage.telemetry["produced_groups_prewritten_ahead"] == 1


def test_a_ranged_owner_refuses_an_entry_outside_its_range(tmp_path, closing):
    storage, _publication, _q, _env, _pb = _owner(tmp_path, (GROUP, 2 * GROUP))
    closing(storage)
    with pytest.raises(RuntimeError, match="outside this owner's produced range"):
        chain._write_group(storage, count=1, first=2 * GROUP)
    assert storage._produced_groups == {}


@pytest.mark.parametrize("batch_range", [(1, GROUP), (0, GROUP + 1), (GROUP, GROUP)])
def test_a_range_is_whole_groups(tmp_path, closing, batch_range):
    with pytest.raises(ValueError, match="is not whole groups"):
        _owner(tmp_path, batch_range)
