"""Host staging guards and cancellation used by retained-window publication."""
from __future__ import annotations

import pickle
import threading
import weakref

import pytest

from prismaquant.aura_cost import _encode_aura_unit_checkpoint
from prismaquant.joint_cost_quantum import quantum_runtime_execution
from prismaquant.tessera_publication import PublicationError, PublicationJob
from test_stageb_publication_backend_1775 import engine, _pool_publisher, _stage, WAIT


@pytest.mark.parametrize("state", [{"x": [1.0, 2.0], "nested": {"a": "b"}},
                                   {"text": "\u03bb" * 1000, "raw": b"a" * 1000}])
def test_capped_encoder_matches_exact_existing_bytes(state):
    expected = _encode_aura_unit_checkpoint(qname="unit", identity_sha256="a" * 64,
                                           state=state)
    actual = _encode_aura_unit_checkpoint(qname="unit", identity_sha256="a" * 64,
                                         state=state, max_bytes=len(expected))
    assert actual == expected


def test_capped_encoder_refuses_oversize():
    with pytest.raises(ValueError, match="checkpoint.*(bound|limit|bytes)"):
        _encode_aura_unit_checkpoint(qname="unit", identity_sha256="a" * 64,
                                     state={"blob": b"x" * 100_000}, max_bytes=4096)


@pytest.mark.parametrize("budget", [True, False, -1, 1.5, "1024"])
def test_launch_refuses_invalid_publication_budget(budget):
    with pytest.raises(ValueError, match="checkpoint_publication_budget_bytes"):
        quantum_runtime_execution(
            {"execution": {"checkpoint_publication_budget_bytes": budget}},
            replay_regime={})


def test_cancel_keeps_running_charge_and_drops_unstarted_payload(engine):
    publisher = _pool_publisher(engine, budget=16, jobs=3)
    started, release = threading.Event(), threading.Event()
    writes = []

    class Payload:
        pass

    tail = Payload()
    reference = weakref.ref(tail)

    def first():
        started.set()
        assert release.wait(WAIT)
        writes.append("first")

    def second(payload=tail):
        writes.append("tail")

    try:
        _stage(publisher, "first", first, charge=8)
        assert started.wait(WAIT)
        _stage(publisher, "tail", second, charge=8)
        del tail, second
        assert publisher.cancel_pending() == ["tail"]
        assert reference() is None, "cancel returned credit with tail payload held"
        assert publisher.stats()["charged_bytes"] == 8
        assert publisher.outstanding == 1
        with pytest.raises(PublicationError):
            publisher.reserve(1)
        release.set()
        publisher.close()
        assert publisher.completed() == ["first"]
        assert publisher.stats()["charged_bytes"] == 0
        assert writes == ["first"]
        with pytest.raises(PublicationError):
            publisher.drain()
        assert engine.submit(lambda: "reader").result(timeout=WAIT) == "reader"
    finally:
        release.set()
        publisher.close()


def test_snapshot_graph_refuses_unbounded_or_custom_reducers():
    from prismaquant.joint_checkpoint_publication import snapshot_bound

    with pytest.raises(ValueError, match="tensor-free"):
        snapshot_bound({"custom": object()}, limit=16 << 20)
    with pytest.raises(ValueError, match="staging bytes"):
        snapshot_bound({"blob": "x" * 100_000}, limit=1 << 20)
    graph = None
    for _ in range(70):
        graph = [graph]
    with pytest.raises(ValueError, match="depth"):
        snapshot_bound(graph, limit=16 << 20)


def test_construction_bound_refuses_large_shape_before_copying():
    from prismaquant.joint_checkpoint_publication import check_construction

    with pytest.raises(ValueError, match="construction"):
        check_construction(references={}, rows=3, probes=10_000,
                           seed_base=7000, limit=16 << 20)
    with pytest.raises(ValueError, match="staging bytes"):
        check_construction(references={"resident": "x" * 1_000_000},
                           rows=3, probes=3, seed_base=7000, limit=16 << 20)
    check_construction(references={"x": [1.0, 2.0]}, rows=3, probes=3,
                       seed_base=7000, limit=16 << 20)


def test_ledger_reserves_before_snapshot_and_never_borrows_nested_graph(monkeypatch, tmp_path):
    import prismaquant.aura_cost as aura
    from prismaquant.joint_checkpoint_publication import CheckpointPublicationLedger

    entered, release = threading.Event(), threading.Event()
    consumer = threading.get_ident()
    writers = []
    durable, acknowledgements, windows = set(), [], []
    source = {"nested": {"components": [1.0, 2.0]}}
    atomic = aura.atomic_write_bytes
    encode = aura._encode_aura_unit_checkpoint
    encoders = []

    def checked_encode(**kwargs):
        encoders.append(threading.get_ident())
        assert threading.get_ident() != consumer, "checkpoint serialization ran on consumer"
        return encode(**kwargs)

    def held(path, body):
        writers.append(threading.get_ident())
        entered.set()
        assert release.wait(WAIT)
        atomic(path, body)

    monkeypatch.setattr(aura, "atomic_write_bytes", held)
    monkeypatch.setattr(aura, "_encode_aura_unit_checkpoint", checked_encode)
    ledger = CheckpointPublicationLedger(
        checkpoint_root=tmp_path, identity_sha256="a" * 64,
        windows=[{"names": ["unit"]}], completed=durable,
        acknowledge=lambda: acknowledgements.append(set(durable)),
        window_done=windows.append, budget_bytes=16 << 20)

    def state(limit):
        assert limit == ledger.stats()["slot_bytes"]
        assert ledger.stats()["charged_bytes"] >= ledger.stats()["slot_bytes"]
        return source

    try:
        ledger.start_window(0, ["unit"])
        assert ledger.submit("unit", state)
        assert entered.wait(WAIT)
        source["nested"]["components"][0] = 99.0
        ledger.poll()
        assert durable == set() and acknowledgements == [] and windows == []
        release.set()
        ledger.flush()
        assert durable == {"unit"}
        assert acknowledgements == [{"unit"}] and windows == [0]
        raw = aura._aura_unit_checkpoint_path(tmp_path, "unit").read_bytes()
        envelope = pickle.loads(raw)
        assert pickle.loads(envelope["payload"]) == {"nested": {"components": [1.0, 2.0]}}
        assert writers and all(worker != consumer for worker in writers)
        assert encoders and all(worker != consumer for worker in encoders)
        assert ledger.stats()["charged_bytes"] == 0
    finally:
        release.set()
        ledger.close()


def test_encoder_owns_snapshot_before_source_mutates(monkeypatch, tmp_path):
    import prismaquant.aura_cost as aura
    from prismaquant.joint_checkpoint_publication import CheckpointPublicationLedger

    entered, release = threading.Event(), threading.Event()
    source = {"nested": {"components": [1.0, 2.0]}}
    encode = aura._encode_aura_unit_checkpoint
    expected = encode(qname="unit", identity_sha256="a" * 64, state=source)

    def held_encode(**kwargs):
        entered.set()
        assert release.wait(WAIT)
        return encode(**kwargs)

    monkeypatch.setattr(aura, "_encode_aura_unit_checkpoint", held_encode)
    durable = set()
    ledger = CheckpointPublicationLedger(
        checkpoint_root=tmp_path, identity_sha256="a" * 64,
        windows=[{"names": ["unit"]}], completed=durable,
        acknowledge=lambda: None, window_done=lambda index: None,
        budget_bytes=16 << 20)
    try:
        ledger.start_window(0, ["unit"])
        assert ledger.submit("unit", lambda limit: source)
        assert entered.wait(WAIT)
        source["nested"]["components"][0] = 99.0
        assert durable == set()
        release.set()
        ledger.flush()
        actual = aura._aura_unit_checkpoint_path(tmp_path, "unit").read_bytes()
        assert actual == expected
        assert ledger.stats()["encoded_bytes"] == len(expected)
        assert ledger.stats()["charged_bytes"] == 0
    finally:
        release.set()
        ledger.close()
