"""Owned cyclic snapshots retire before reusable staging credit, without GC."""
from __future__ import annotations

import gc
import threading

import pytest

import prismaquant.aura_cost as aura
import prismaquant.joint_checkpoint_publication as publication

WAIT = 15


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel", "reject"])
def test_cyclic_snapshot_retirement_precedes_credit(monkeypatch, tmp_path, outcome):
    owned_ids = set()
    copy = publication.deepcopy
    encode = aura._encode_aura_unit_checkpoint
    entered, release = threading.Event(), threading.Event()
    marker = f"snapshot-lifetime-{outcome}"
    source: dict[str, object] = {"marker": marker}
    source["self"] = source

    def tracked_copy(value):
        owned = copy(value)
        owned_ids.add(id(owned))
        return owned

    def controlled_encode(**kwargs):
        entered.set()
        assert release.wait(WAIT)
        if outcome == "failure":
            raise OSError("injected cyclic snapshot encoder failure")
        return encode(**kwargs)

    monkeypatch.setattr(publication, "deepcopy", tracked_copy)
    monkeypatch.setattr(aura, "_encode_aura_unit_checkpoint", controlled_encode)
    names = ["first", "tail-a", "tail-b"]
    durable = set()
    ledger = publication.CheckpointPublicationLedger(
        checkpoint_root=tmp_path, identity_sha256="a" * 64,
        windows=[{"names": names}], completed=durable,
        acknowledge=lambda: None, window_done=lambda index: None,
        budget_bytes=16 << 20, max_jobs=3)
    enabled = gc.isenabled()
    gc.disable()
    try:
        ledger.start_window(0, names)
        if outcome == "reject":
            def reject(job):
                raise RuntimeError("injected snapshot submission refusal")
            monkeypatch.setattr(ledger._publisher, "submit", reject)
            with pytest.raises(RuntimeError, match="submission refusal"):
                ledger.submit("first", lambda limit: source)
        else:
            assert ledger.submit("first", lambda limit: source)
            assert entered.wait(WAIT)
            assert ledger.submit("tail-a", lambda limit: source)
            assert ledger.submit("tail-b", lambda limit: source)
            if outcome == "cancel":
                assert ledger._publisher.cancel_pending() == ["tail-a", "tail-b"]
                # Queued owners must already have retired, while the running
                # owner still retains one slot and its original charge.
                live = [id(item) for item in gc.get_objects()
                        if type(item) is dict and id(item) in owned_ids
                        and item.get("marker") == marker]
                assert len(live) == 1, "cancel returned credit for live cyclic snapshots"
                assert ledger.stats()["charged_bytes"] == ledger.stats()["slot_bytes"]
            release.set()
            if outcome == "failure":
                with pytest.raises(RuntimeError):
                    ledger.flush()
            elif outcome == "success":
                ledger.flush()
                assert durable == set(names)
        ledger.close()
        assert ledger.stats()["charged_bytes"] == 0
        assert not any(type(item) is dict and id(item) in owned_ids
                       and item.get("marker") == marker for item in gc.get_objects()), (
                           "publication returned credit for live cyclic snapshots")
        # Disposal is confined to the owned copy, not the consumer's graph.
        assert source["self"] is source and source["marker"] == marker
        if outcome in ("failure", "reject"):
            assert not durable
    finally:
        release.set()
        ledger.close()
        source.clear()
        if enabled:
            gc.enable()
        gc.collect()
