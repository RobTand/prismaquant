"""The Stage A chain's entry I/O off the compute thread (RobTand/prismaquant#1128).

These tests pin, one behavior each, what the compute thread no longer does:

* it makes no export poll per write: the stager looks at the oldest live
  export once per write, and progress still advances only for acknowledged
  entries;

The owner, the writer, the spool adapter, the queue and the fleet are the
real ones; only PrismaBuild's asynchronous exporter is replaced by a
transport whose receipt the test controls (``test_produced_output_spool``).
"""
from __future__ import annotations

import collections
import threading

import pytest

import test_produced_output_spool as spool_tests
import test_stage_a_same_box_readback as readback
# Autouse, and it must apply HERE too (RobTand/prismaquant#889).
from test_stage_a_produced_boundary_chain import (  # noqa: F401
    _isolated_launch_context)
from test_stage_a_same_box_readback import closing  # noqa: F401

pytestmark = pytest.mark.own_process

GROUP_SIZE = readback.GROUP_SIZE
TOP = readback.TOP


class CountingExport(spool_tests.ControlledExport):
    """Lands a group only when the test says so; counts polls per thread."""

    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
        self.poll_threads = collections.Counter()

    def poll_group(self, batch_id):
        self.poll_threads[threading.current_thread().name] += 1
        return super().poll_group(batch_id)


def _cotangent_groups(storage, groups, *, probe=0):
    return [storage.write(readback._tail(probe, b), batch_index=b,
                          boundary_index=TOP, probe_index=probe)
            for b in range(groups * GROUP_SIZE)]


def test_the_compute_thread_polls_no_export_per_write(tmp_path, closing):
    """RED on the parent: every write polled every live export twice.

    Two groups are written while the first one's export is live. The
    compute thread asks PrismaBuild nothing about it; the stager does, and
    progress reports each entry once its group is acknowledged.
    """

    storage, _pub, _q, _env, _pb, backend = readback._spool_owner(
        tmp_path, backend_type=CountingExport)
    closing(storage)
    assert storage._stager is not None, "the fixture's window funds a stager"
    progress = []

    class Progress:
        def entry(self, **kwargs):
            progress.append(kwargs)

    storage.watch_progress(Progress())
    compute = threading.current_thread().name
    _cotangent_groups(storage, 2)
    assert storage.drain_produced_stager(60.0)
    assert backend.poll_threads[compute] == 0, (
        "the compute thread polled an export per write", dict(backend.poll_threads))
    assert sum(backend.poll_threads.values()) > 0, "nothing polled the exports"
    assert progress == []
    for batch_id, group in backend.groups.items():
        if group["entries"]:
            backend.acknowledge(batch_id)
    storage.settle_local_output()
    assert len(progress) == 2 * GROUP_SIZE
