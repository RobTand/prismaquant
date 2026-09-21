"""Read-ahead in the Stage A produced-boundary owner loop (#887).

The synchronous loop publishes a group when a read first asks for it, waits
for PrismaBuild's mover, and at window exit waits again for the retirement.
Measured on the fleet (cycle a3, 2026-09-21) each of those round trips is a
queue claim plus 4 s to 5 s of action, and the 512-sample panel has about
3,240 of each, so the GPU would wait on the queue at every window.

These tests drive the real bound owner over a real queue, as
``test_stage_a_produced_boundary_chain`` does, with a sealed window wider
than two groups. Two groups is the geometry ``build_boundary_template``
seals by default, and at two groups every rule below is inert.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import test_stage_a_produced_boundary_chain as chain

GROUP_SIZE = chain.GROUP_SIZE


def _owner(tmp_path, *, groups: int, window_gib: int):
    return chain._bound_owner(
        tmp_path, n_batches=groups * GROUP_SIZE, window_gib=window_gib,
        gib=max(2 * window_gib, 4), payload_max_bytes=1 << 22)


def _expected(first: int = 0, count: int = GROUP_SIZE):
    return [torch.arange(8, dtype=torch.float32) + index
            for index in range(first, first + count)]


def _read(storage, references, expected):
    with storage.prefetch(references) as window:
        for reference, want in zip(references, expected):
            assert torch.equal(storage.get(window, reference), want)


def test_the_default_two_group_window_publishes_nothing_ahead(tmp_path):
    """Backwards compatible: the sealed default is the synchronous loop."""

    storage, _publication, _q, _env, _pb = chain._bound_owner(tmp_path)
    chain._write_group(storage)
    (group,) = storage._produced_groups.values()
    assert storage._produced_plan["window_groups"] == 2
    assert group["published"] is None
    assert storage.telemetry["produced_groups_published_ahead"] == 0


def test_a_group_is_published_when_its_last_entry_lands(tmp_path, monkeypatch):
    storage, _publication, q, env, pb_repo = _owner(
        tmp_path, groups=2, window_gib=8)
    partial = chain._write_group(storage, count=GROUP_SIZE - 1)
    (group,) = storage._produced_groups.values()
    assert group["published"] is None, "a group with a slot missing is not whole"

    references = partial + chain._write_group(
        storage, count=1, first=GROUP_SIZE - 1)
    assert group["published"] is not None, (
        "the last entry is durable and every fact the publication needs is "
        "on the references: the mover should be running already")
    assert storage.telemetry["produced_groups_published_ahead"] == 1

    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        _read(storage, references, _expected())
    assert storage.telemetry["produced_groups_published"] == 1, (
        "the read found the group published and did not publish it again")
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_a_full_window_falls_back_to_publishing_at_first_read(
        tmp_path, monkeypatch):
    """Three groups fund one ahead: the rest are backpressure, not failure."""

    storage, _publication, q, env, pb_repo = _owner(
        tmp_path, groups=3, window_gib=3)
    assert storage._produced_plan["ahead_groups"] == 1
    written = [chain._write_group(storage, first=index * GROUP_SIZE)
               for index in range(3)]
    published = [group["published"] is not None
                 for group in storage._produced_groups.values()]
    assert published == [True, False, False], (
        "two groups' worth of the sealed window stays with the read path",
        published)

    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        for index, references in enumerate(written):
            _read(storage, references, _expected(index * GROUP_SIZE))
        storage.settle_produced_releases()
    assert storage.telemetry["produced_groups_published"] == 3
    assert all(record["retired"] for record in storage.produced_group_records())
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_a_retained_boundary_group_is_staged_once_across_four_passes(
        tmp_path, monkeypatch):
    storage, publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    references = chain._write_group(storage, boundary_index=0)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        with storage.retain_produced_boundary(0):
            for _probe in range(4):
                _read(storage, references, _expected())
            assert storage.telemetry["produced_groups_materialized"] == 1
            assert storage.telemetry["produced_groups_retired"] == 0, (
                "another probe pass reads this group: it stays staged")
        storage.settle_produced_releases()
    assert storage.telemetry["produced_groups_rematerialized"] == 0
    assert storage.telemetry["produced_groups_retired"] == 1, (
        "leaving the scope asks for the retirement, once")
    batch_id = storage.produced_group_records()[0]["batch_id"]
    assert publication.materialization_state(
        batch_id=batch_id)["stage_retired"] is True
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_a_retirement_not_waited_for_is_settled_before_its_group_is_read_again(
        tmp_path, monkeypatch):
    """The window exit returns at once; the copy is never read mid-egress."""

    storage, publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    monkeypatch.setattr(storage, "PRODUCED_DEFERRAL_POLL_S", 0.01)
    monkeypatch.setattr(storage, "PRODUCED_RELEASE_POLL_S", 0.0)
    references = chain._write_group(storage)
    real = publication.retire
    calls = {"n": 0}

    def deferring(batch_id, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 5:
            return chain._incomplete(deferred_own=["own-egress-in-flight"],
                                     entries_deferred=GROUP_SIZE)
        return real(batch_id, **kwargs)

    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        with storage.prefetch(references) as window:
            storage.get(window, references[0])
            monkeypatch.setattr(publication, "retire", deferring)
        assert calls["n"] == 1, (
            "the window exit asked once and did not wait", calls)
        assert storage.produced_group_records()[0]["retired"] is False
        assert list(storage.produced_release_debt()["pending"]) == [
            storage.produced_group_records()[0]["batch_id"]]

        _read(storage, references, _expected())
        assert calls["n"] >= 6, (
            "the second read waited the retirement out before it touched "
            "a copy an egress might be deleting", calls)
        assert storage.telemetry["produced_groups_rematerialized"] == 1
        storage.settle_produced_releases()
    debt = storage.produced_release_debt()
    assert debt["abandoned"] == {}, (
        "a poll of PrismaBuild's own in-flight work is not a failed attempt",
        debt)
    assert debt == chain._NO_DEBT
    assert storage.telemetry["produced_group_release_deferrals"] >= 1


def test_a_retired_plane_is_staged_ahead_of_its_read(tmp_path, monkeypatch):
    storage, _publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    references = chain._write_group(storage, boundary_index=3)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        _read(storage, references, _expected())
        assert storage.produced_group_records()[0]["retired"] is True

        assert storage.stage_produced_boundary_ahead(3) == 1
        assert storage.stage_produced_boundary_ahead(3) == 0, (
            "a group that already holds credit is left alone")
        assert storage.telemetry["produced_groups_rematerialized"] == 1
        _read(storage, references, _expected())
        assert storage.telemetry["produced_groups_rematerialized"] == 1, (
            "the read found the plane staged and did not stage it again")
        storage.settle_produced_releases()
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_closing_the_owner_returns_credit_no_read_consumed(
        tmp_path, monkeypatch):
    storage, publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    chain._write_group(storage)
    batch_id = storage.produced_group_records()[0]["batch_id"]
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        publication.await_materialized(
            batch_id=batch_id, timeout_s=120.0)
        storage.settle_produced_releases()
    assert storage._produced_held == set()
    assert publication.materialization_state(
        batch_id=batch_id)["stage_retired"] is True


def test_a_window_of_one_group_is_refused_at_bind(tmp_path):
    storage, publication, *_rest = chain._bound_owner(tmp_path)
    storage._produced = None        # bind again, as a fresh owner would
    with pytest.raises(ValueError, match="at least 2"):
        storage.bind_produced_output(
            publication, group_size=GROUP_SIZE, n_batches=GROUP_SIZE,
            max_entry_tensor_bytes=1 << 14, window_groups=1)


def test_the_window_the_dispatcher_seals_is_the_group_count_the_owner_derives():
    """One place chooses the read-ahead width: the sealed template."""

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    _src, pb_repo = chain._pb_source()
    from prismaquant.staged_lease import set_lease_helper_root
    set_lease_helper_root(str(pb_repo))
    from dispatch_joint_quanta import build_stage_a_produced_template
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.stage_a_produced_output import (
        boundary_group_ceiling_bytes)

    geometry = {"group_size": 64, "max_entry_tensor_bytes": 16 << 20}
    for groups in (2, 56):
        template = build_stage_a_produced_template(
            output_prefix="/home/rob/stage-a-out", tier=chain.TIER,
            artifact_max_bytes=640 * (1 << 30), concurrent_groups=groups,
            **geometry)
        assert template["working_demands"][chain.TIER]["window_gib"] == (
            2 * groups), "the production group is 2 tokens"
        publication = SimpleNamespace(
            template=template, tier=chain.TIER,
            group_ceiling_bytes=lambda *, entries, max_entry_tensor_bytes: (
                boundary_group_ceiling_bytes(
                    group_size=entries,
                    max_entry_tensor_bytes=max_entry_tensor_bytes)))
        assert StreamedBoundaryArtifacts._sealed_window_groups(
            publication, **geometry) == groups


# ---------------------------------------------------------------------------
# Review findings (2026-09-21). Each test fails on the change as first written.
# ---------------------------------------------------------------------------

def _foreign_pin(batch_id, **kwargs):
    return chain._incomplete(live_pins=[{"pin_id": "a-real-reader"}])


def test_read_ahead_leaves_the_read_path_its_two_groups_when_one_is_stuck(
        tmp_path, monkeypatch):
    """The share is bounded by the WHOLE window, not only by itself.

    A group whose retirement PrismaBuild refused holds credit without being
    read-ahead. Counting only the share let read-ahead fill the window
    around it, and the next read then failed to fund: terminal, and caused
    by an optional step.
    """

    storage, publication, q, env, pb_repo = _owner(
        tmp_path, groups=6, window_gib=5)
    assert storage._produced_plan["ahead_groups"] == 3
    # One entry short of whole, so the READ publishes it: a read-path
    # group, which is what a refused window retirement leaves behind.
    stuck = chain._write_group(storage, count=GROUP_SIZE - 1)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        real = publication.retire
        stuck_id = storage.produced_group_records()[0]["batch_id"]

        def refusing_the_stuck_one(batch_id, **kwargs):
            if batch_id == stuck_id:
                return _foreign_pin(batch_id)
            return real(batch_id, **kwargs)

        monkeypatch.setattr(publication, "retire", refusing_the_stuck_one)
        _read(storage, stuck, _expected(0, GROUP_SIZE - 1))
        assert len(storage._produced_held) == 1, "refused: it keeps its credit"
        assert storage._produced_ahead == set()

        written = [chain._write_group(storage, first=index * GROUP_SIZE)
                   for index in range(1, 6)]
        assert len(storage._produced_held) <= (
            storage._produced_plan["window_groups"] - 2), (
            "read-ahead spent the two groups the read path is owed",
            sorted(storage._produced_held))
        for index, references in enumerate(written, start=1):
            _read(storage, references, _expected(index * GROUP_SIZE))
        storage.settle_produced_releases()


def test_a_read_the_ledger_will_not_fund_takes_read_ahead_credit_back(
        tmp_path, monkeypatch):
    """The owner's count is an estimate; PrismaBuild's ledger is the fact."""

    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublicationFailed)

    storage, _publication, q, env, pb_repo = _owner(
        tmp_path, groups=3, window_gib=8)
    ahead = chain._write_group(storage)
    wanted = chain._write_group(storage, first=GROUP_SIZE, count=GROUP_SIZE - 1)
    assert storage.telemetry["produced_groups_published_ahead"] == 1
    real = storage._produced_publish
    refused = {"n": 0}

    def refusing_once(key, group, deadline=None):
        if group["published"] is None and not refused["n"]:
            refused["n"] += 1
            raise BoundaryProducedPublicationFailed(
                batch_id=group["batch_id"],
                refusal={"step": "refill",
                         "refusal": "tier-reservation-unavailable"})
        return real(key, group, deadline=deadline)

    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        monkeypatch.setattr(storage, "_produced_publish", refusing_once)
        _read(storage, wanted, _expected(GROUP_SIZE, GROUP_SIZE - 1))
        assert refused["n"] == 1
        assert storage.telemetry["produced_groups_ahead_surrendered"] == 1, (
            "the group staged ahead gave its credit to the read that needed it")
        _read(storage, ahead, _expected())
        assert storage.telemetry["produced_groups_rematerialized"] == 1, (
            "and its own read staged it again: the synchronous loop")
        storage.settle_produced_releases()
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_a_census_prismabuild_could_not_take_is_asked_again_not_paid_for(
        tmp_path, monkeypatch):
    """Many movers in flight, a row caught mid-transition: not a shortfall."""

    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublicationFailed)

    storage, _publication, q, env, pb_repo = _owner(
        tmp_path, groups=3, window_gib=8)
    monkeypatch.setattr(storage, "PRODUCED_CENSUS_POLL_S", 0.01)
    chain._write_group(storage)
    wanted = chain._write_group(storage, first=GROUP_SIZE, count=GROUP_SIZE - 1)
    real = storage._produced_publish
    refused = {"n": 0}

    def census_unknown_twice(key, group, deadline=None):
        if group["published"] is None and refused["n"] < 2:
            refused["n"] += 1
            raise BoundaryProducedPublicationFailed(
                batch_id=group["batch_id"],
                refusal={"step": "refill",
                         "refusal": "unknown-retain: funding-census"})
        return real(key, group, deadline=deadline)

    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        monkeypatch.setattr(storage, "_produced_publish", census_unknown_twice)
        _read(storage, wanted, _expected(GROUP_SIZE, GROUP_SIZE - 1))
        assert refused["n"] == 2
        assert storage.telemetry["produced_group_read_refunds"] == 2
        assert storage.telemetry["produced_groups_ahead_surrendered"] == 0, (
            "nothing was short, so nothing staged ahead was given up")
        assert [entry["kind"] for entry in storage.produced_ahead_refusals()] == [
            "census", "census"]
        storage.settle_produced_releases()
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_a_refusal_that_is_not_about_credit_propagates_with_read_ahead_held(
        tmp_path, monkeypatch):
    """Giving credit back cannot fix it, so none is given."""

    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublicationFailed)

    storage, _publication, q, env, pb_repo = _owner(
        tmp_path, groups=3, window_gib=8)
    chain._write_group(storage)
    wanted = chain._write_group(storage, first=GROUP_SIZE, count=GROUP_SIZE - 1)
    assert len(storage._produced_ahead) == 1
    real = storage._produced_publish

    def refusing(key, group, deadline=None):
        if group["published"] is None:
            raise BoundaryProducedPublicationFailed(
                batch_id=group["batch_id"],
                refusal={"ok": False, "step": "origin",
                         "refusal": "origin-identity-changed"})
        return real(key, group, deadline=deadline)

    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        monkeypatch.setattr(storage, "_produced_publish", refusing)
        with pytest.raises(BoundaryProducedPublicationFailed):
            with storage.prefetch(wanted):
                pass
        assert storage.telemetry["produced_groups_ahead_surrendered"] == 0
        assert storage.telemetry["produced_group_read_refunds"] == 0
        monkeypatch.setattr(storage, "_produced_publish", real)
        storage.settle_produced_releases()


def test_a_refused_read_with_nothing_to_surrender_fails_as_it_did_before(
        tmp_path, monkeypatch):
    from prismaquant.stage_a_produced_output import (
        BoundaryProducedPublicationFailed)

    storage, _publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    references = chain._write_group(storage, count=GROUP_SIZE - 1)

    def refusing(key, group, deadline=None):
        raise BoundaryProducedPublicationFailed(
            batch_id=group["batch_id"],
            refusal={"step": "refill",
                     "refusal": "tier-reservation-unavailable"})

    monkeypatch.setattr(storage, "_produced_publish", refusing)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        with pytest.raises(BoundaryProducedPublicationFailed):
            with storage.prefetch(references):
                pass


def test_an_optional_step_that_times_out_does_not_end_the_capture(
        tmp_path, monkeypatch):
    from prismaquant.stage_a_produced_output import BoundaryStagingTimeout

    storage, publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    references = chain._write_group(storage, boundary_index=2)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        _read(storage, references, _expected())
        storage.settle_produced_releases()
        real = publication.ensure_batch_materialized

        def late(**kwargs):
            raise BoundaryStagingTimeout("the ahead budget ran out")

        monkeypatch.setattr(publication, "ensure_batch_materialized", late)
        assert storage.stage_produced_boundary_ahead(2) == 0
        assert storage.telemetry["produced_group_ahead_refusals"] == 1
        assert "BoundaryStagingTimeout" in (
            storage.produced_ahead_refusals()[0]["reason"])
        monkeypatch.setattr(publication, "ensure_batch_materialized", real)
        _read(storage, references, _expected())
        storage.settle_produced_releases()
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_an_origin_is_not_unlinked_under_a_mover_that_has_not_copied_it(
        tmp_path, monkeypatch):
    """Production's order: dispose of the origins, THEN settle at exit.

    A group staged ahead was only asked for. Its mover opens the origin
    when it runs, so the disposal waits for the receipt first.
    """

    storage, publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    references = chain._write_group(storage)
    batch_id = storage.produced_group_records()[0]["batch_id"]
    assert storage.telemetry["produced_groups_published_ahead"] == 1
    awaited = []
    real = publication.await_materialized

    def watching(**kwargs):
        awaited.append(kwargs["batch_id"])
        return real(**kwargs)

    monkeypatch.setattr(publication, "await_materialized", watching)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        for reference in references:
            storage.retire(reference)
        assert awaited == [batch_id], (
            "waited once, before the first unlink, and not per entry", awaited)
        assert publication.materialization_state(
            batch_id=batch_id)["mover_receipt_complete"] is True
        storage.settle_produced_releases()
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_a_write_nothing_reads_back_is_not_published_ahead(tmp_path):
    storage, _publication, _q, _env, _pb = _owner(
        tmp_path, groups=1, window_gib=8)
    for index in range(GROUP_SIZE):
        storage.write(torch.arange(8, dtype=torch.float32) + index,
                      batch_index=index, boundary_index=0, probe_index=0,
                      read_back=False)
    (group,) = storage._produced_groups.values()
    assert group["published"] is None
    assert storage.telemetry["produced_groups_published_ahead"] == 0


def test_the_default_window_settles_nothing_at_exit(tmp_path, monkeypatch):
    """Inert means inert: a refused retirement stays the debt it was."""

    storage, publication, q, env, pb_repo = chain._bound_owner(tmp_path)
    references = chain._write_group(storage)
    calls = {"n": 0}

    def counting(batch_id, **kwargs):
        calls["n"] += 1
        return _foreign_pin(batch_id)

    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        with storage.prefetch(references) as window:
            storage.get(window, references[0])
            monkeypatch.setattr(publication, "retire", counting)
        asked, failures = calls["n"], storage.telemetry[
            "produced_group_release_failures"]
        debt = storage.produced_release_debt()
        storage.settle_produced_releases()
        storage._settle_produced_releases_at_exit(None)
    assert calls["n"] == asked, "no further retirement was asked for"
    assert storage.telemetry["produced_group_release_failures"] == failures
    assert storage.produced_release_debt() == debt


def test_a_settle_that_cannot_finish_does_not_fail_a_finished_capture(
        tmp_path, monkeypatch, capsys):
    storage, publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    chain._write_group(storage)

    def broken(batch_id, **kwargs):
        return {"ok": False, "refusal": "egress-incomplete",
                "receipt": {"complete": False}}        # no cause: unclassified

    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        monkeypatch.setattr(publication, "retire", broken)
        storage._settle_produced_releases_at_exit(None)
    assert "settle at exit did not finish" in capsys.readouterr().out
    assert storage.produced_release_debt()["unclassified"], (
        "reported as debt, not raised out of a finished run")


def test_a_failing_run_asks_for_its_retirements_and_keeps_its_own_error(
        tmp_path, monkeypatch):
    storage, publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    chain._write_group(storage)
    asked = []
    real = publication.retire

    def recording(batch_id, **kwargs):
        asked.append(batch_id)
        return real(batch_id, **kwargs)

    primary = RuntimeError("the capture failed")
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        monkeypatch.setattr(publication, "retire", recording)
        storage._settle_produced_releases_at_exit(primary)
        assert len(asked) == 1, "asked once, and not waited for"
        monkeypatch.setattr(publication, "retire", real)
        storage.settle_produced_releases()
    assert storage.produced_release_debt() == chain._NO_DEBT


def test_the_waits_account_for_the_settle(tmp_path, monkeypatch):
    storage, _publication, q, env, pb_repo = _owner(
        tmp_path, groups=1, window_gib=8)
    chain._write_group(storage)
    with chain._fleet(q, tmp_path):
        chain._strict(monkeypatch, env, pb_repo, q)
        assert storage.telemetry["produced_group_ahead_wait_s"] > 0.0
        before = storage.telemetry["produced_group_release_wait_s"]
        storage.settle_produced_releases()
    assert storage.telemetry["produced_group_release_wait_s"] > before
