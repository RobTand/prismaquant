"""The campaign may publish an anchor's files on another thread, in order.

Every timing assertion here is an ORDERING assertion held open by a barrier
the test controls.  A barrier proves that one thing can begin before another
has finished; it does not measure how much wall clock that is worth on a real
endpoint, and nothing in this file should be read as a speed result.  The
whole-cycle profiles and the both-host power series are collected separately,
on the boxes, against the real arms.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from prismaquant.tessera_publication import (
    BoundedPublisher, PublicationError, PublicationJob,
)


# A barrier a test holds and a writer waits on.  ``timeout`` everywhere, so a
# defect in the code under test fails the test instead of hanging the shard.
WAIT = 20.0


def _publisher(**kwargs):
    kwargs.setdefault("budget_bytes", 1 << 20)
    return BoundedPublisher(**kwargs)


# ---------------------------------------------------------------------------
# The publisher on its own
# ---------------------------------------------------------------------------

def test_jobs_publish_in_submission_order():
    order = []
    pub = _publisher()
    try:
        for index in range(8):
            pub.submit(PublicationJob(
                key=index, charged_bytes=1,
                publish=lambda index=index: order.append(index)))
        assert pub.drain() == list(range(8))
    finally:
        pub.close()
    assert order == list(range(8)), (
        "one writer thread and a FIFO queue is what makes recovery "
        f"deterministic; got {order}")


def test_the_budget_blocks_a_submit_until_the_writer_catches_up():
    release = threading.Event()
    started = threading.Event()

    def slow():
        started.set()
        assert release.wait(WAIT), "writer never released"

    pub = BoundedPublisher(budget_bytes=100)
    try:
        pub.submit(PublicationJob(key="a", charged_bytes=60, publish=slow))
        assert started.wait(WAIT)
        blocked = threading.Event()

        def second():
            pub.submit(PublicationJob(key="b", charged_bytes=60,
                                      publish=lambda: None))
            blocked.set()

        worker = threading.Thread(target=second)
        worker.start()
        # 60 + 60 > 100, and the first job is still resident, so the second
        # submit has to wait rather than stage past the bound.
        assert not blocked.wait(0.5), (
            "an over-budget submit returned; staging is not bounded")
        release.set()
        assert blocked.wait(WAIT), "submit never unblocked after the writer drained"
        worker.join(WAIT)
        assert pub.drain() == ["a", "b"]
        assert pub.stats()["peak_charged_bytes"] <= 100
        assert pub.stats()["submit_blocked_seconds"] > 0.0
    finally:
        release.set()
        pub.close()


def test_an_artifact_larger_than_the_whole_budget_still_publishes():
    pub = BoundedPublisher(budget_bytes=8)
    try:
        pub.submit(PublicationJob(key="huge", charged_bytes=4096,
                                  publish=lambda: None))
        assert pub.drain() == ["huge"], (
            "an artifact bigger than the budget must wait for an empty queue, "
            "not for room that cannot appear")
    finally:
        pub.close()


def test_a_writer_failure_drops_what_was_queued_behind_it_unwritten():
    written = []
    release = threading.Event()

    def first():
        assert release.wait(WAIT)
        raise OSError("no space left on device")

    pub = _publisher()
    try:
        pub.submit(PublicationJob(key="a", charged_bytes=1, publish=first))
        pub.submit(PublicationJob(
            key="b", charged_bytes=1, publish=lambda: written.append("b")))
        pub.submit(PublicationJob(
            key="c", charged_bytes=1, publish=lambda: written.append("c")))
        release.set()
        with pytest.raises(PublicationError) as caught:
            pub.drain()
        assert isinstance(caught.value.__cause__, OSError)
        assert written == [], (
            "work queued behind a failed write must not be written: "
            f"{written} landed after the failure")
    finally:
        release.set()
        pub.close()


def test_what_landed_before_a_failure_is_still_reported_for_recovery():
    release = threading.Event()
    pub = _publisher()
    try:
        pub.submit(PublicationJob(key="done", charged_bytes=1,
                                  publish=lambda: None))

        def boom():
            assert release.wait(WAIT)
            raise OSError("no space left on device")

        pub.submit(PublicationJob(key="bad", charged_bytes=1, publish=boom))
        release.set()
        with pytest.raises(PublicationError):
            pub.drain()
        # The first job's files exist. A resume that is told about them skips
        # work it really did; a resume that is not repeats it.
        assert pub.completed() == ["done"]
        assert isinstance(pub.failure, OSError)
    finally:
        release.set()
        pub.close()


def test_a_failure_reaches_a_submit_that_is_blocked_on_the_budget():
    release = threading.Event()

    def boom():
        assert release.wait(WAIT)
        raise OSError("no space left on device")

    pub = BoundedPublisher(budget_bytes=100)
    try:
        pub.submit(PublicationJob(key="bad", charged_bytes=60, publish=boom))
        raised = []

        def second():
            try:
                pub.submit(PublicationJob(key="next", charged_bytes=60,
                                          publish=lambda: None))
            except BaseException as exc:  # noqa: BLE001
                raised.append(exc)

        worker = threading.Thread(target=second)
        worker.start()
        time.sleep(0.2)
        release.set()
        worker.join(WAIT)
        assert not worker.is_alive(), (
            "a caller blocked on the budget must wake into the failure, not "
            "wait on a writer that has stopped")
        assert raised and isinstance(raised[0], PublicationError)
    finally:
        release.set()
        pub.close()


def test_a_zero_budget_is_refused_rather_than_silently_synchronous():
    with pytest.raises(ValueError):
        BoundedPublisher(budget_bytes=0)


# ---------------------------------------------------------------------------
# The campaign's two writes, with and without a publisher
# ---------------------------------------------------------------------------

class _Spec:
    act_dtype_name = "a16"

    def bits_for_shape(self, shape):
        return 4 * shape[0] * shape[1]

    def memory_bytes_for_shape(self, shape):
        return shape[0] * shape[1] // 2


class _Family:
    name = "TQ"


class _Cache:
    def __init__(self, cache_dir):
        self.cache_dir = str(cache_dir)
        self.weights = {}
        self.metadata = {}


def _prepared():
    return dict(spec=_Spec(), family=_Family(), rung=1088,
                activation_qdq=None, input_scale=None, activation_kwargs=None)


def _finish(tmp_path, monkeypatch, *, publisher, qname="model.layers.0.q",
            barrier=None, blob=b"wire-bytes"):
    """Run ``_finish_anchor`` over fakes, optionally behind a save barrier."""
    import torch

    from prismaquant import production_weight_cache as pwc
    from prismaquant import tessera_campaign as tc

    monkeypatch.setattr(
        pwc, "_local_forward_render_score",
        lambda **kwargs: (0.25, "output_mse", False, 0))
    if barrier is not None:
        real_save = torch.save

        def held(obj, path, *args, **kwargs):
            assert barrier.wait(WAIT), "save barrier never released"
            return real_save(obj, path, *args, **kwargs)

        monkeypatch.setattr(torch, "save", held)

    cache_dir = tmp_path / "cache"
    wire_dir = tmp_path / "wire"
    cache_dir.mkdir(parents=True, exist_ok=True)
    wire_dir.mkdir(parents=True, exist_ok=True)
    weight = torch.zeros((8, 8), dtype=torch.bfloat16)
    return tc._finish_anchor(
        qname=qname, weight=weight, activations=torch.zeros((4, 8)),
        # Uppercase because the cache canonicalises its key and the wire path
        # does not; a mixed-case name would name two different things.
        format_name="NVFP4", cache=_Cache(cache_dir), wire_dir=wire_dir,
        prepared=_prepared(), render=torch.zeros((8, 8)), blob=blob,
        elapsed=1.0, publisher=publisher), cache_dir, wire_dir


def test_a_blocked_writer_blocks_the_encode_thread_when_publication_is_synchronous(
        tmp_path, monkeypatch):
    """The baseline this branch exists to change, stated as a test."""
    barrier = threading.Event()
    returned = threading.Event()

    def run():
        _finish(tmp_path, monkeypatch, publisher=None, barrier=barrier)
        returned.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert not returned.wait(0.5), (
            "the synchronous path returned while its writer was blocked")
        barrier.set()
        assert returned.wait(WAIT)
    finally:
        barrier.set()
        worker.join(WAIT)


def test_a_publisher_lets_the_next_unit_start_while_the_writer_is_blocked(
        tmp_path, monkeypatch):
    barrier = threading.Event()
    pub = _publisher()
    try:
        anchor, cache_dir, wire_dir = _finish(
            tmp_path, monkeypatch, publisher=pub, barrier=barrier,
            qname="model.layers.0.q")
        # The encode thread is back with a scored anchor while the writer is
        # still inside torch.save. That is the whole point of the change.
        assert anchor.qname == "model.layers.0.q"
        assert pub.outstanding == 1
        assert not (wire_dir / "model__layers__0__q__NVFP4.tessera").exists(), (
            "the wire landed before the render it is sequenced behind")
        assert pub.completed() == [], (
            "a job reported complete while its writer was still blocked")
        barrier.set()
        assert pub.drain() == [("model.layers.0.q", "NVFP4")]
    finally:
        barrier.set()
        pub.close()
    # The drain leaves ordinary files, in the ordinary places, with the
    # ordinary names: nothing here is a new store.
    assert (wire_dir / "model__layers__0__q__NVFP4.tessera").read_bytes() == b"wire-bytes"
    assert len(list(cache_dir.glob("*"))) == 1
    assert not list(cache_dir.glob("*.tmp"))


def test_the_published_render_is_the_tensor_the_synchronous_path_stores(
        tmp_path, monkeypatch):
    sync_anchor, sync_cache, sync_wire = _finish(
        tmp_path / "sync", monkeypatch, publisher=None)
    pub = _publisher()
    try:
        async_anchor, async_cache, async_wire = _finish(
            tmp_path / "async", monkeypatch, publisher=pub)
        pub.drain()
    finally:
        pub.close()
    sync_file = next(iter(sync_cache.glob("*")))
    async_file = next(iter(async_cache.glob("*")))
    assert sync_file.name == async_file.name
    assert sync_file.read_bytes() == async_file.read_bytes(), (
        "the staged tensor is not the tensor the synchronous path stores")
    assert next(iter(sync_wire.glob("*"))).read_bytes() == (
        next(iter(async_wire.glob("*"))).read_bytes())
    for field in ("dloss", "wire_bytes", "bits_per_param", "memory_bytes",
                  "activation_contract", "hessian_applied"):
        assert getattr(sync_anchor, field) == getattr(async_anchor, field), field


def test_a_failed_publication_surfaces_at_the_next_submit(tmp_path, monkeypatch):
    import torch

    pub = _publisher()
    try:
        real_save = torch.save
        calls = []

        def failing(obj, path, *args, **kwargs):
            calls.append(Path(path).name)
            if len(calls) == 2:
                raise OSError("no space left on device")
            return real_save(obj, path, *args, **kwargs)

        monkeypatch.setattr(torch, "save", failing)
        _finish(tmp_path / "one", monkeypatch, publisher=pub, qname="a.b")
        pub.drain()
        _finish(tmp_path / "two", monkeypatch, publisher=pub, qname="c.d")
        with pytest.raises(PublicationError):
            pub.drain()
        # Fail closed: the second unit's wire must not exist, because its
        # render never landed and the two are one publication.
        assert not list((tmp_path / "two" / "wire").glob("*.tessera")), (
            "the wire was published after its render failed")
    finally:
        pub.close()


def test_the_temporary_file_is_not_left_behind_by_a_completed_publication(
        tmp_path, monkeypatch):
    pub = _publisher()
    try:
        _, cache_dir, wire_dir = _finish(tmp_path, monkeypatch, publisher=pub)
        pub.drain()
    finally:
        pub.close()
    assert [p.name for p in wire_dir.glob("*")] == [
        "model__layers__0__q__NVFP4.tessera"]
    assert not [p.name for p in cache_dir.glob("*.tmp")]
    assert os.listdir(cache_dir)
