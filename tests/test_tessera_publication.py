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


def _stage(publisher, job):
    """Reserve then submit, the way a producer must."""
    publisher.reserve(job.charged_bytes)
    publisher.submit(job)


# ---------------------------------------------------------------------------
# The publisher on its own
# ---------------------------------------------------------------------------

def test_jobs_publish_in_submission_order():
    order = []
    pub = _publisher()
    try:
        for index in range(8):
            _stage(pub, PublicationJob(
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
        _stage(pub, PublicationJob(key="a", charged_bytes=60, publish=slow))
        assert started.wait(WAIT)
        blocked = threading.Event()

        def second():
            _stage(pub, PublicationJob(key="b", charged_bytes=60,
                                       publish=lambda: None))
            blocked.set()

        worker = threading.Thread(target=second)
        worker.start()
        # 60 + 60 > 100, and the first job is still resident, so the second
        # submit has to wait rather than stage past the bound.
        assert not blocked.wait(0.5), (
            "an over-budget reservation returned; staging is not bounded")
        release.set()
        assert blocked.wait(WAIT), "submit never unblocked after the writer drained"
        worker.join(WAIT)
        assert pub.drain() == ["a", "b"]
        assert pub.stats()["peak_charged_bytes"] <= 100
        assert pub.stats()["submit_blocked_seconds"] > 0.0
    finally:
        release.set()
        pub.close()


def test_an_artifact_larger_than_the_whole_budget_is_refused(monkeypatch):
    """The declared bound is the bound, or it is not a bound."""
    published = []
    pub = BoundedPublisher(budget_bytes=8)
    try:
        with pytest.raises(PublicationError, match="does not fit"):
            pub.reserve(4096)
        # And nothing was staged behind the refusal.
        _stage(pub, PublicationJob(key="small", charged_bytes=8,
                                   publish=lambda: published.append("small")))
        assert pub.drain() == ["small"]
        assert published == ["small"]
        assert pub.stats()["peak_charged_bytes"] == 8
    finally:
        pub.close()


def test_a_reservation_is_needed_before_charged_bytes_are_submitted():
    pub = BoundedPublisher(budget_bytes=64)
    try:
        with pytest.raises(PublicationError, match="reserve"):
            pub.submit(PublicationJob(key="unreserved", charged_bytes=16,
                                      publish=lambda: None))
        # A zero-charge job carries no bytes and needs no room: the receipt
        # and journal jobs are ordering, not staging.
        pub.submit(PublicationJob(key="free", charged_bytes=0,
                                  publish=lambda: None))
        assert pub.drain() == ["free"]
    finally:
        pub.close()


def test_a_released_reservation_gives_its_room_back():
    pub = BoundedPublisher(budget_bytes=64)
    try:
        pub.reserve(64)
        pub.release(64)
        # If the release had not landed this would block until the timeout.
        _stage(pub, PublicationJob(key="after", charged_bytes=64,
                                   publish=lambda: None))
        assert pub.drain() == ["after"]
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
        _stage(pub, PublicationJob(key="a", charged_bytes=1, publish=first))
        _stage(pub, PublicationJob(
            key="b", charged_bytes=1, publish=lambda: written.append("b")))
        _stage(pub, PublicationJob(
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
        _stage(pub, PublicationJob(key="done", charged_bytes=1,
                                   publish=lambda: None))

        def boom():
            assert release.wait(WAIT)
            raise OSError("no space left on device")

        _stage(pub, PublicationJob(key="bad", charged_bytes=1, publish=boom))
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
        _stage(pub, PublicationJob(key="bad", charged_bytes=60, publish=boom))
        raised = []

        def second():
            try:
                _stage(pub, PublicationJob(key="next", charged_bytes=60,
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
        assert pub.drain() == [("files", "model.layers.0.q", "NVFP4")]
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


def test_the_budget_bounds_the_encode_thread_with_the_writer_blocked(
        tmp_path, monkeypatch):
    """One artifact in flight, and the next one not yet made."""
    from prismaquant import production_weight_cache as pwc

    barrier = threading.Event()
    # 8x8 BF16 render plus a ten byte blob: room for exactly one.
    pub = BoundedPublisher(budget_bytes=8 * 8 * 2 + 10)
    returned = threading.Event()
    # Only the producer's copies. The writer canonicalises again inside
    # _store_rendered_weight_entry, and that call is an identity on a tensor
    # this test is not asking about.
    copies = []
    real_canonical = pwc._canonical_rendered_weight_tensor

    def counted(tensor, **kwargs):
        if not threading.current_thread().name.startswith("tessera-publication"):
            copies.append(1)
        return real_canonical(tensor, **kwargs)

    monkeypatch.setattr(pwc, "_canonical_rendered_weight_tensor", counted)
    try:
        _finish(tmp_path / "one", monkeypatch, publisher=pub, barrier=barrier,
                qname="a.b")

        def second():
            _finish(tmp_path / "two", monkeypatch, publisher=pub,
                    barrier=barrier, qname="c.d")
            returned.set()

        worker = threading.Thread(target=second)
        worker.start()
        assert not returned.wait(0.5), (
            "a second artifact was staged while the first still occupied the "
            "whole budget; the bound is not a bound")
        assert copies == [1], (
            "the second render was copied to host while the budget was full: "
            "reserving after the copy leaves the producer holding one "
            "artifact more than the bound admits")
        assert pub.stats()["peak_charged_bytes"] == 8 * 8 * 2 + 10
        barrier.set()
        assert returned.wait(WAIT)
        worker.join(WAIT)
        pub.drain()
    finally:
        barrier.set()
        pub.close()


def test_a_render_larger_than_the_budget_is_refused_before_it_is_copied(
        tmp_path, monkeypatch):
    pub = BoundedPublisher(budget_bytes=16)
    try:
        with pytest.raises(PublicationError, match="does not fit"):
            _finish(tmp_path, monkeypatch, publisher=pub, qname="a.b")
        assert not list((tmp_path / "cache").glob("*.pt")), (
            "a refused artifact still reached the cache")
        assert pub.stats()["peak_charged_bytes"] == 0
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


# ---------------------------------------------------------------------------
# The rule that a receipt never precedes the file it describes
# ---------------------------------------------------------------------------

class _Anchor:
    def __init__(self, qname, format_name="NVFP4"):
        self.qname = qname
        self.format_name = format_name


def _ledger(publisher, journalled):
    from prismaquant.tessera_campaign import _AnchorPublicationLedger

    return _AnchorPublicationLedger(
        publisher=publisher,
        make_record=lambda anchor, identity: f"record:{identity}",
        journal_anchor=lambda anchor, record: journalled.append(
            (anchor.qname, record)))


def _files_job(publisher, name, publish):
    """Stand in for the render/wire job ``_finish_anchor`` submits."""
    from prismaquant.tessera_campaign import FILES_JOB

    _stage(publisher, PublicationJob(
        key=(FILES_JOB, name, "NVFP4"), charged_bytes=1, publish=publish))


def test_without_a_publisher_the_ledger_journals_where_it_always_did():
    journalled = []
    ledger = _ledger(None, journalled)
    assert ledger.active is False
    ledger.record(_Anchor("a.b"), "id-a")
    assert journalled == [("a.b", "record:id-a")], (
        "the default path must journal at the same point it always has")
    assert ledger.apply_completed() == 0
    assert ledger.drain() == 0
    assert ledger.staged == {}


def test_without_a_publisher_a_checkpoint_is_written_where_it_always_was():
    written = []
    ledger = _ledger(None, [])
    ledger.submit_checkpoint(lambda: written.append("now"))
    assert written == ["now"]


def test_a_staged_anchor_is_not_journalled_while_its_writer_is_blocked():
    journalled = []
    barrier = threading.Event()
    pub = _publisher()
    ledger = _ledger(pub, journalled)
    try:
        _files_job(pub, "a.b", lambda: barrier.wait(WAIT))
        ledger.record(_Anchor("a.b"), "id-a")
        assert ledger.apply_completed() == 0
        assert journalled == [], (
            "an anchor row reached the journal before its own bytes reached "
            "the disk; a resume would then read a receipt for a file that "
            "does not exist")
        barrier.set()
        assert ledger.drain() == 1
        assert journalled == [("a.b", "record:id-a")]
        assert ledger.staged == {}
    finally:
        barrier.set()
        pub.close()


def test_a_checkpoint_write_is_ordered_behind_the_receipts_it_cites():
    order = []
    barrier = threading.Event()
    pub = _publisher()
    ledger = _ledger(pub, order)
    try:
        _files_job(pub, "a.b", lambda: (barrier.wait(WAIT),
                                        order.append("files")))
        ledger.record(_Anchor("a.b"), "id-a")
        ledger.submit_checkpoint(lambda: order.append("checkpoint"))
        barrier.set()
        ledger.drain()
    finally:
        barrier.set()
        pub.close()
    # The receipt is made on the writer, between the two, and the row it
    # produced is applied by the caller; what matters here is that the journal
    # write did not run before the bytes it cites were written.
    assert order.index("files") < order.index("checkpoint")


def test_the_ledger_journals_in_publication_order():
    journalled = []
    release = threading.Event()
    pub = _publisher()
    ledger = _ledger(pub, journalled)
    try:
        for name in ("a.b", "c.d", "e.f"):
            _files_job(pub, name, lambda: release.wait(WAIT))
            ledger.record(_Anchor(name), f"id-{name}")
        release.set()
        assert ledger.drain() == 3
    finally:
        release.set()
        pub.close()
    assert journalled == [("a.b", "record:id-a.b"), ("c.d", "record:id-c.d"),
                          ("e.f", "record:id-e.f")]


def test_a_failed_publication_leaves_its_anchor_unjournalled():
    journalled = []
    release = threading.Event()
    pub = _publisher()
    ledger = _ledger(pub, journalled)
    try:
        _files_job(pub, "good", lambda: release.wait(WAIT))
        ledger.record(_Anchor("good"), "id-good")

        def boom():
            raise OSError("no space left on device")

        _files_job(pub, "bad", boom)
        ledger.record(_Anchor("bad"), "id-bad")
        release.set()
        with pytest.raises(PublicationError):
            ledger.drain()
        assert ("bad", "record:id-bad") not in journalled, (
            "an anchor whose files never landed was journalled anyway")
        # The unit that did land is still recoverable: applying the completed
        # keys journals it, so a resume skips work that was really done.
        assert ledger.apply_completed() == 1
        assert journalled == [("good", "record:id-good")]
        assert ("bad", "NVFP4") in ledger.staged
    finally:
        release.set()
        pub.close()


def test_a_completion_nothing_staged_is_refused_rather_than_ignored():
    pub = _publisher()
    ledger = _ledger(pub, [])
    try:
        pub.submit(PublicationJob(key=("something-else", 1), charged_bytes=0,
                                  publish=lambda: None))  # ordering job, no bytes
        with pytest.raises(RuntimeError, match="unknown job"):
            ledger.drain()
    finally:
        pub.close()


def test_one_publication_key_cannot_hold_two_anchors():
    journalled = []
    pub = _publisher()
    ledger = _ledger(pub, journalled)
    try:
        _files_job(pub, "a.b", lambda: None)
        ledger.record(_Anchor("a.b"), "first")
        with pytest.raises(RuntimeError, match="already staged"):
            ledger.record(_Anchor("a.b"), "second")
    finally:
        pub.close()


def test_the_dispatcher_carries_the_staging_bound_into_the_plan(monkeypatch):
    """A row's declared demand has to include what its campaign will stage.

    The flag reaches ``selected_anchor_resources`` twice over: the campaign
    passes its own, and the dispatcher reads it back off the row's argv when
    it sizes the worker that will admit the row.  Only the second is testable
    without running a campaign, and it is the one a planning change breaks
    silently, because a worker sized without the term still runs -- it just
    runs a campaign holding bytes nobody reserved.
    """
    from prismaquant import autoscale
    from tools import dispatch_tessera_campaign as dispatch

    seen: dict = {}

    def selected(model, **kwargs):
        seen.update(kwargs)
        return {"memory_bytes": 100}

    monkeypatch.setattr(autoscale, "selected_anchor_resources", selected)
    dispatch._streamed_resource_plan(
        dict(model="/source", campaign_argv=[
            "--streaming", "--publication-overlap-bytes", "8388608",
            "--campaign-identity-bytes", "536870912"]),
        dict(unit_shapes={"a": [3, 4]}, counts={"a": 9}), ["a"],
        selected_source=True)
    assert seen["publication_overlap_bytes"] == 8388608
    assert seen["campaign_identity_bytes"] == 536870912

    seen.clear()
    dispatch._streamed_resource_plan(
        dict(model="/source", campaign_argv=["--streaming"]),
        dict(unit_shapes={"a": [3, 4]}, counts={"a": 9}), ["a"],
        selected_source=True)
    assert seen["publication_overlap_bytes"] == 0
    assert seen["campaign_identity_bytes"] == 0


def test_the_plan_charges_the_staging_bound_where_the_anchors_live(monkeypatch):
    """And the number lands in the phase the staged bytes are resident in."""
    from prismaquant import autoscale

    # The source census is the part that needs a real checkpoint on disk;
    # the term under test is added after it, so it is stood in for here.
    monkeypatch.setattr(autoscale, "streamed_calibration_resources",
                        lambda *_args, **_kwargs: dict(
                            live_layer_prefix="layers.",
                            terms=dict(nonbody_source_bytes=100,
                                       declared_headroom_bytes=200),
                            body_layer_bytes={"0": 1000},
                            body_loader_transient_bytes={"0": 100},
                            body_source_file_bytes={"0": 900},
                            unit_source_weight_bytes={"layers.0.proj": 24},
                            full_hessian_bytes=64, full_prefix_bytes=32,
                            source_header_sha256="a" * 64))
    kwargs = dict(unit_shapes={"layers.0.proj": [3, 4]},
                  counts={"layers.0.proj": 9},
                  max_act_rows=2, cache_slots=2, prefetch_workers=1,
                  headroom_gb=0)
    off = autoscale.selected_anchor_resources("/source", **kwargs)
    on = autoscale.selected_anchor_resources(
        "/source", **kwargs, publication_overlap_bytes=8388608,
        campaign_identity_bytes=16777216)
    assert off["phases"]["resident_anchors"].get(
        "publication_staging_bytes", 0) == 0
    assert on["phases"]["resident_anchors"]["publication_staging_bytes"] == 8388608
    assert on["phases"]["resident_anchors"]["campaign_identity_metadata_bytes"] == 16777216
    # Nothing else moved: the term is additive, not a re-sizing.
    assert {k: v for k, v in on["phases"]["resident_anchors"].items()
            if k not in {"publication_staging_bytes", "campaign_identity_metadata_bytes"}} == {
        k: v for k, v in off["phases"]["resident_anchors"].items()
        if k not in {"publication_staging_bytes", "campaign_identity_metadata_bytes"}}


# ---------------------------------------------------------------------------
# A deferred identity: post-work the writer does behind the anchor's own files
# ---------------------------------------------------------------------------

def test_a_deferred_identity_is_derived_on_the_writer_behind_the_files():
    """``record(anchor, derive=...)`` runs the derivation inside the receipt job.

    Ordering, not speed: the derivation must not run on the caller's thread,
    must run after this anchor's file job, and must seal the receipt that is
    journalled.  The thread name is the writer's, which is what a profile of
    the real arm attributes the phase to.
    """
    journalled, order = [], []
    barrier = threading.Event()
    pub = _publisher()
    ledger = _ledger(pub, journalled)
    try:
        _files_job(pub, "a.b", lambda: (barrier.wait(WAIT), order.append("files")))

        def derive():
            order.append(("derive", threading.current_thread().name))
            return "id-a"

        ledger.record(_Anchor("a.b"), derive=derive)
        assert order == [], "the identity was derived on the encode thread"
        assert ledger.apply_completed() == 0
        barrier.set()
        assert ledger.drain() == 1
        assert order == ["files", ("derive", "tessera-publication")]
        assert journalled == [("a.b", "record:id-a")]
    finally:
        barrier.set()
        pub.close()


def test_without_a_publisher_a_deferred_identity_is_derived_inline():
    journalled, threads = [], []
    ledger = _ledger(None, journalled)

    def derive():
        threads.append(threading.current_thread().name)
        return "id-a"

    ledger.record(_Anchor("a.b"), derive=derive)
    assert threads == [threading.current_thread().name]
    assert journalled == [("a.b", "record:id-a")]


def test_record_takes_exactly_one_form_of_identity():
    ledger = _ledger(None, [])
    with pytest.raises(ValueError, match="exactly one"):
        ledger.record(_Anchor("a.b"))
    with pytest.raises(ValueError, match="exactly one"):
        ledger.record(_Anchor("a.b"), "id-a", derive=lambda: "id-a")


def test_a_refused_deferred_identity_fails_the_publication_and_journals_nothing():
    """An identity gate that refuses on the writer is a publication failure.

    On the synchronous path the same refusal raises on the encode thread
    before anything is journalled.  Deferred, it is raised by the writer and
    surfaces at the next barrier, and the anchor whose identity was refused
    has no journal row: no receipt, no row.
    """
    journalled = []
    pub = _publisher()
    ledger = _ledger(pub, journalled)
    try:
        _files_job(pub, "a.b", lambda: None)

        def derive():
            raise RuntimeError("checkpoint anchor is outside the current menu")

        ledger.record(_Anchor("a.b"), derive=derive)
        with pytest.raises(PublicationError):
            ledger.drain()
        assert journalled == []
        assert pub.failure is not None
    finally:
        pub.close()
