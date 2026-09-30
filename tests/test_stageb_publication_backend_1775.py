"""CPU-only publisher ownership/ordering gates; not performance measurements."""
from concurrent.futures import Future
import os
import stat
import threading
import weakref

import pytest

from prismaquant import aura_cost
from prismaquant.io_engine import IOEngine
from prismaquant.tessera_publication import (
    BoundedPublisher, PublicationError, PublicationJob,
)

WAIT = 10


class _Payload:
    pass


@pytest.fixture
def engine():
    # An isolated instance for this test only; never shut down global ENGINE.
    own = IOEngine()
    own.width = 1
    yield own
    if own._pool is not None:
        own._pool.shutdown(wait=True)


def _pool_publisher(engine, *, budget=64, jobs=4):
    return BoundedPublisher(budget_bytes=budget, max_jobs=jobs,
                            submit_task=engine.submit,
                            name="stage-b-checkpoint-publication")


def _stage(pub, key, publish, charge=1):
    pub.reserve(charge)
    pub.submit(PublicationJob(key, charge, publish))


def test_success_releases_payload_before_reusable_credit(monkeypatch):
    pub = BoundedPublisher(budget_bytes=8)
    held = _Payload()
    reference = weakref.ref(held)
    observed = []
    release = threading.Event()
    notify = pub._cond.notify_all

    def credit():
        if pub._published and pub._charged == 0:
            observed.append(reference() is None)
        notify()

    monkeypatch.setattr(pub._cond, "notify_all", credit)
    def publish(payload=held):
        assert release.wait(WAIT)

    pub.reserve(8)
    pub.submit(PublicationJob("unit", 8, publish))
    del publish, held
    release.set()
    try:
        assert pub.drain() == ["unit"]
    finally:
        release.set()
        pub.close()
    assert observed and all(observed), "credit returned while payload was still held"


def test_failure_does_not_retain_payload_in_traceback():
    pub = BoundedPublisher(budget_bytes=8)
    held = _Payload()
    reference = weakref.ref(held)

    def fail(payload=held):
        raise OSError("injected publication failure")

    pub.reserve(8)
    pub.submit(PublicationJob("unit", 8, fail))
    del fail, held
    try:
        with pytest.raises(PublicationError):
            pub.drain()
    finally:
        pub.close()
    assert reference() is None, "stored failure traceback retained uncharged payload"
    assert isinstance(pub.failure, OSError)


def test_pool_backend_is_fifo_and_leaves_idle_worker_for_reads(engine, tmp_path):
    pub = _pool_publisher(engine)
    order, worker_names = [], []
    path = tmp_path / "read.bin"
    path.write_bytes(b"verified source")
    try:
        # No publication task waits indefinitely for the first job.
        assert engine.submit(path.read_bytes).result(WAIT) == b"verified source"
        for index in range(3):
            def publish(index=index):
                order.append(index)
                worker_names.append(threading.current_thread().name)
            _stage(pub, index, publish)
        assert pub.drain() == [0, 1, 2]
        assert order == [0, 1, 2]
        assert all(name.startswith("pq-io") for name in worker_names)
        # The finite drain task relinquishes the sole worker after a batch.
        assert engine.submit(path.read_bytes).result(WAIT) == b"verified source"
        _stage(pub, 3, lambda: order.append(3))
        assert pub.drain() == [3]
    finally:
        pub.close()
    # Closing a publisher must not shut down its shared execution backend.
    assert engine.submit(path.read_bytes).result(WAIT) == b"verified source"


def test_pool_failure_reports_prefix_and_drops_tail(engine):
    pub = _pool_publisher(engine)
    began, release = threading.Event(), threading.Event()
    written = []

    def fail():
        began.set()
        assert release.wait(WAIT)
        raise OSError("injected disk failure")

    try:
        _stage(pub, "first", lambda: written.append("first"))
        _stage(pub, "bad", fail)
        assert began.wait(WAIT)
        _stage(pub, "tail", lambda: written.append("tail"))
        release.set()
        with pytest.raises(PublicationError):
            pub.drain()
        assert pub.completed() == ["first"]
        assert written == ["first"]
    finally:
        release.set()
        pub.close()
    assert engine.submit(lambda: "read remains live").result(WAIT) == "read remains live"


def test_zero_byte_jobs_have_bounded_unreaped_completions(engine):
    pub = _pool_publisher(engine, jobs=1)
    reserved, waiting = threading.Event(), threading.Event()
    errors = []
    worker = None
    try:
        _stage(pub, "first", lambda: None, charge=0)
        # Fence behind the finite writer on the width-one engine.
        engine.submit(lambda: None).result(WAIT)

        def next_reservation():
            try:
                waiting.set()
                pub.reserve(0)
                reserved.set()
                pub.release(0)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=next_reservation)
        worker.start()
        assert waiting.wait(WAIT)
        assert not reserved.wait(0.1), "unreaped completion exceeded the job bound"
        assert pub.completed() == ["first"]
        assert reserved.wait(WAIT)
        worker.join(WAIT)
        assert not worker.is_alive() and not errors
    finally:
        pub.completed()
        pub.close()
        if worker is not None:
            worker.join(WAIT)


def test_pool_submission_failure_releases_reservation():
    def unavailable(task):
        raise RuntimeError("executor unavailable")

    pub = BoundedPublisher(budget_bytes=8, max_jobs=1, submit_task=unavailable)
    try:
        pub.reserve(8)
        with pytest.raises(PublicationError):
            pub.submit(PublicationJob("unit", 8, lambda: None))
        # Staging cleanup after cancellation must not make charges negative.
        pub.release(8)
        assert pub.stats()["charged_bytes"] == 0
        with pytest.raises(PublicationError):
            pub.drain()
    finally:
        pub.close()


def test_cancelled_pool_task_fails_without_hanging():
    pending = Future()
    pub = BoundedPublisher(budget_bytes=8, max_jobs=1,
                            submit_task=lambda task: pending)
    try:
        _stage(pub, "unit", lambda: pytest.fail("cancelled job ran"))
        assert pending.cancel()
        with pytest.raises(PublicationError):
            pub.drain()
    finally:
        pub.close()
    assert pub.stats()["charged_bytes"] == 0


def test_encoded_unit_is_immutable_and_matches_existing_writer(tmp_path):
    state = {"components": [{"costs": [1.0, 2.0]}],
             "identity": {"source": ["sha256", "verified"]}}
    qname, identity = "layer.0.unit", "a" * 64
    aura_cost._write_aura_unit_checkpoint(
        tmp_path, qname=qname, identity_sha256=identity, state=state)
    path = aura_cost._aura_unit_checkpoint_path(tmp_path, qname)
    expected = path.read_bytes()
    encoded = aura_cost._encode_aura_unit_checkpoint(
        qname=qname, identity_sha256=identity, state=state)
    assert type(encoded) is bytes and encoded == expected
    state["components"][0]["costs"].append(999.0)
    state["identity"]["source"].append("mutated")
    assert encoded == expected


def test_encoded_cpu_tensor_unit_is_a_snapshot(tmp_path):
    import torch

    state = {"tensor": torch.arange(8, dtype=torch.float32)}
    qname, identity = "tensor.unit", "b" * 64
    aura_cost._write_aura_unit_checkpoint(
        tmp_path, qname=qname, identity_sha256=identity, state=state)
    expected = aura_cost._aura_unit_checkpoint_path(tmp_path, qname).read_bytes()
    encoded = aura_cost._encode_aura_unit_checkpoint(
        qname=qname, identity_sha256=identity, state=state)
    assert encoded == expected
    state["tensor"].fill_(999)
    assert encoded == expected


def test_shared_drain_yields_after_its_job_bound(engine):
    pub = _pool_publisher(engine, budget=8, jobs=2)
    first, second, third = (threading.Event() for _ in range(3))
    release_first, release_second = threading.Event(), threading.Event()
    reader_seen = threading.Event()
    observations = []
    reader = None

    def one():
        first.set()
        assert release_first.wait(WAIT)

    def two():
        second.set()
        assert release_second.wait(WAIT)

    def three():
        observations.append(reader_seen.is_set())
        third.set()

    try:
        _stage(pub, "first", one)
        assert first.wait(WAIT)
        _stage(pub, "second", two)
        reader = engine.submit(reader_seen.set)
        release_first.set()
        assert second.wait(WAIT)
        assert pub.completed() == ["first"]
        _stage(pub, "third", three)  # refill while the original drain is active
        release_second.set()
        assert third.wait(WAIT)
        assert observations == [True], "shared drain monopolized queued IO work"
        assert pub.drain() == ["second", "third"]
    finally:
        release_first.set()
        release_second.set()
        pub.close()
        if reader is not None:
            reader.result(timeout=WAIT)


def test_close_waits_across_rearmed_owned_drains(engine):
    pub = _pool_publisher(engine, budget=8, jobs=2)
    first, second, third = (threading.Event() for _ in range(3))
    release_first, release_second, release_third = (
        threading.Event() for _ in range(3))
    closing, closed = threading.Event(), threading.Event()
    closer = None

    def held(began, release):
        began.set()
        assert release.wait(WAIT)

    def close():
        closing.set()
        pub.close()
        closed.set()

    try:
        _stage(pub, "first", lambda: held(first, release_first))
        assert first.wait(WAIT)
        _stage(pub, "second", lambda: held(second, release_second))
        release_first.set()
        assert second.wait(WAIT)
        assert pub.completed() == ["first"]
        _stage(pub, "third", lambda: held(third, release_third))
        closer = threading.Thread(target=close)
        closer.start()
        assert closing.wait(WAIT)
        release_second.set()
        assert third.wait(WAIT)
        assert not closed.wait(0.1), "close joined only the first dispatch"
        release_third.set()
        assert closed.wait(WAIT)
        closer.join(WAIT)
        assert not closer.is_alive()
        assert pub.drain() == ["second", "third"]
    finally:
        release_first.set()
        release_second.set()
        release_third.set()
        pub.close()
        if closer is not None:
            closer.join(WAIT)


def test_pool_byte_bound_blocks_before_next_staging(engine):
    pub = _pool_publisher(engine, budget=8)
    began, release = threading.Event(), threading.Event()
    waiting, staged = threading.Event(), threading.Event()
    worker = None
    errors = []

    def held():
        began.set()
        assert release.wait(WAIT)

    def second():
        try:
            waiting.set()
            _stage(pub, "second", lambda: None, charge=8)
            staged.set()
        except BaseException as exc:
            errors.append(exc)

    try:
        _stage(pub, "first", held, charge=8)
        assert began.wait(WAIT)
        worker = threading.Thread(target=second)
        worker.start()
        assert waiting.wait(WAIT)
        assert not staged.wait(0.1)
        release.set()
        assert staged.wait(WAIT)
        worker.join(WAIT)
        assert not worker.is_alive() and not errors
        assert pub.drain() == ["first", "second"]
        assert pub.stats()["peak_charged_bytes"] <= 8
    finally:
        release.set()
        pub.close()
        if worker is not None:
            worker.join(WAIT)


def test_atomic_unit_is_not_acknowledged_before_directory_fsync(
        engine, tmp_path, monkeypatch):
    from prismaquant.cost_stage_checkpoint import atomic_write_bytes

    began, release = threading.Event(), threading.Event()
    original = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            began.set()
            assert release.wait(WAIT)
        original(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    pub = _pool_publisher(engine, budget=4096)
    path = tmp_path / "unit.bin"
    try:
        pub.reserve(4096)
        encoded = aura_cost._encode_aura_unit_checkpoint(
            qname="unit", identity_sha256="a" * 64, state={"value": 1})
        assert len(encoded) <= 4096
        pub.submit(PublicationJob("unit", 4096,
                                  lambda: atomic_write_bytes(path, encoded)))
        assert began.wait(WAIT)
        assert path.read_bytes() == encoded  # rename is NOT acknowledgement
        assert pub.completed() == [] and pub.outstanding == 1
        release.set()
        assert pub.drain() == ["unit"]
    finally:
        release.set()
        pub.close()


@pytest.mark.parametrize("kwargs", [
    {"max_jobs": 0, "submit_task": lambda task: Future()},
    {"max_jobs": 1, "submit_task": "not callable"},
    {"submit_task": lambda task: Future()},
])
def test_pool_backend_refuses_unbounded_or_invalid_execution(kwargs):
    with pytest.raises((ValueError, TypeError)):
        BoundedPublisher(budget_bytes=8, **kwargs)
