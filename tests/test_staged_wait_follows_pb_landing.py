"""The staged-range reader waits on PrismaBuild's landing record (PQ #1107, PB #989).

Before this change a strict read refused a declared range 300 s after it
started waiting, whatever the range's mover was doing.  With three Stage A
consumers on one stage, the third's next range copies behind the other two
and lands after that constant: the reader refused while the mover was still
queued, and hours of GPU work ended by name.

Now PrismaBuild writes ``<consumer>.landing.json`` beside the map.  For every
pending range it names the mover, its state and when the tier expects it to
land.  The reader waits while the range's mover is ``ready`` or ``claimed``,
or while the range is ``unpublished`` and the tier loop is alive.  It refuses
on evidence only: the range is ``terminal-no-receipt``, or the tier loop has
not announced its tier within PB's own liveness bound.  The expectation is
information, never a deadline.  A generation that writes no landing record
keeps the bounded wait, and its refusal says so.

The clock is faked, so these tests take no wall time.
"""
from __future__ import annotations

import json

import pytest

import prismaquant.residency_shard_reader as reader
from prismaquant.residency_map import RANGE_HIT, RANGE_UNCOVERED

MOVER = "ab" * 32
TIER = "prismabuild-stage:dl380g10"


class _Clock:
    """``time`` for the reader module: sleeping advances a fake clock."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(float(seconds), 0.0)


class _Resolver:
    """A resolver whose span lands on poll ``lands_at``, or never."""

    def __init__(self, *, lands_at: int | None, record: dict | None,
                 tier_age_s: float | None = 5.0):
        self.lands_at = lands_at
        self.record = record
        self.tier_age_s = tier_age_s
        self.polls = 0
        self.waits: list[dict] = []

    def staged_range_outcome(self, declared, start, end, declared_size=None):
        self.polls += 1
        if self.lands_at is not None and self.polls >= self.lands_at:
            return {"offset": start, "bytes": end - start}, RANGE_HIT
        return None, RANGE_UNCOVERED

    def record_range_wait(self, declared, **kwargs):
        self.waits.append(dict(kwargs))

    def record_range_refusal(self, declared, detail):
        self.waits.append({"served": False, "detail": detail})

    # What PrismaBuild publishes, and where the span sits in its read order.
    def landing_record(self):
        return self.record

    def read_order_positions(self, declared, start, end):
        return [(1_000 + start, 1_000 + end)]

    def tier_record_age(self, tier_id):
        return self.tier_age_s


def _record(state: str, *, expected_in_s: float | None, now: float) -> dict:
    queued = state in ("ready", "claimed")
    return {
        "schema": "prismaquant.prismabuild.residency_landing.v1",
        "consumer_action_key": "cd" * 32, "tier_id": TIER,
        "manifest_sha256": "ef" * 32, "written_unix": now,
        "landing_bytes_per_s": 134e6, "landing_basis": "measured",
        "rates_measured_bytes_per_s": [134e6, 239e6],
        "rate_min_bytes_per_s": 134e6, "rate_max_bytes_per_s": 239e6,
        "report_latency_s": 90.0, "tier_loop_liveness_s": 120.0,
        "ranges": [{
            "mover_action_key": MOVER, "phase": "chain-033", "chunk_index": None,
            "range_start_bytes": 0, "range_end_bytes": 10 ** 9,
            "state": state,
            "expected_landing_unix": (now + expected_in_s) if queued else None,
            "queue_position": 2 if queued else None,
            "bytes_ahead": 46 * 10 ** 9 if queued else None,
            "claimed_unix": now - 10 if state == "claimed" else None,
            "waiting_for": "" if queued else "the window",
        }],
    }


@pytest.fixture
def clock(monkeypatch, tmp_path):
    fake = _Clock()
    monkeypatch.setattr(reader, "time", fake)
    monkeypatch.delenv(reader.STAGED_RANGE_WAIT_ENV, raising=False)
    progress = tmp_path / "action.progress"
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PATH", str(progress))
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", "t" * 32)
    fake.progress = progress
    return fake


def _wait(resolver, clock):
    return reader.await_staged_spans(
        resolver, [("/pool/shard-1.safetensors", 0, 64, 128)],
        deadline=clock.monotonic() + reader.staged_range_wait_s())


def test_a_claimed_mover_expected_400s_out_is_waited_past_the_300s_default(clock):
    """The acceptance case: the default bound would refuse at 300 s."""

    resolver = _Resolver(lands_at=420, record=_record(
        "claimed", expected_in_s=400.0, now=clock.now))

    assert _wait(resolver, clock) == RANGE_HIT
    assert clock.now - 1_000_000.0 > reader.STAGED_RANGE_WAIT_S
    assert resolver.waits[-1]["served"] is True


def test_a_copy_slower_than_every_receipt_is_still_waited_on(clock):
    """The expectation passed long ago; the mover is claimed and the loop alive."""

    resolver = _Resolver(lands_at=1_800, record=_record(
        "claimed", expected_in_s=-600.0, now=clock.now))

    assert _wait(resolver, clock) == RANGE_HIT


def test_the_wait_is_declared_to_prismabuild_while_it_lasts(clock):
    """The worker's no_progress rung reads this record (PB #989)."""

    seen = {}
    record = _record("ready", expected_in_s=400.0, now=clock.now)

    class _Watching(_Resolver):
        def staged_range_outcome(self, *args, **kwargs):
            path = str(clock.progress) + ".staged-wait"
            if self.polls == 350:
                with open(path) as handle:
                    seen.update(json.load(handle))
            return super().staged_range_outcome(*args, **kwargs)

    assert _wait(_Watching(lands_at=420, record=record), clock) == RANGE_HIT
    assert seen["schema"] == "prismabuild.staged_wait.v1"
    assert seen["token"] == "t" * 32
    assert seen["movers"] == [MOVER]
    # Cleared once the range landed.
    assert not (clock.progress.parent / "action.progress.staged-wait").exists()


def test_an_unpublished_range_is_waited_on_while_the_tier_loop_is_alive(clock):
    resolver = _Resolver(lands_at=700, record=_record(
        "unpublished", expected_in_s=None, now=clock.now))

    assert _wait(resolver, clock) == RANGE_HIT


def test_a_terminal_mover_without_a_receipt_refuses_at_once_and_says_so(clock):
    resolver = _Resolver(lands_at=None, record=_record(
        "terminal-no-receipt", expected_in_s=None, now=clock.now))

    assert _wait(resolver, clock) == RANGE_UNCOVERED
    assert clock.now - 1_000_000.0 < 5.0
    detail = resolver.waits[-1]["detail"]
    assert "terminal-no-receipt" in detail and MOVER[:12] in detail


def test_a_silent_tier_loop_refuses_and_names_the_state(clock):
    resolver = _Resolver(lands_at=None, record=_record(
        "claimed", expected_in_s=400.0, now=clock.now), tier_age_s=600.0)

    assert _wait(resolver, clock) == RANGE_UNCOVERED
    detail = resolver.waits[-1]["detail"]
    assert "claimed" in detail and "tier loop" in detail
    assert "expected" in detail


def test_without_a_landing_record_the_bounded_wait_applies_and_says_so(clock):
    """An older PrismaBuild generation publishes nothing to wait on."""

    resolver = _Resolver(lands_at=None, record=None)

    assert _wait(resolver, clock) == RANGE_UNCOVERED
    waited = clock.now - 1_000_000.0
    assert reader.STAGED_RANGE_WAIT_S <= waited < reader.STAGED_RANGE_WAIT_S + 2
    detail = resolver.waits[-1]["detail"]
    assert "no landing record" in detail
    assert f"{reader.STAGED_RANGE_WAIT_S:g} s" in detail


def test_the_dispatcher_gate_says_the_bound_is_the_fallback_wait():
    """The grace still bounds the constant; the landing record bounds the rest."""

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import dispatch_joint_quanta as dispatch

    with pytest.raises(dispatch.DispatchRefused, match="landing record"):
        dispatch.require_staged_wait_below_grace(
            {"env": {reader.STAGED_RANGE_WAIT_ENV: "900"}}, [("chain-000", 900)])
    assert "landing record" in dispatch.require_staged_wait_below_grace.__doc__


# -- the real resolver: files on disk, the read order through PB's helper ---


def _real_resolver(tmp_path, monkeypatch, *, state, tier_age_s=5.0):
    """A ``ResidencyResolver`` reading a landing record and a tier record
    from disk, with the sealed read order cut by ``load_sealed_read_order``.

    Only the two hops that need a live PrismaBuild claim are stood in for:
    the sealed manifest (``_load_sealed_payload``) and the generation's
    ``storage_tiers`` module, whose v1 read order is list order.
    """
    import time as real_time
    from types import SimpleNamespace

    from prismaquant import staged_lease
    from prismaquant.residency_map import ResidencyResolver

    manifest = "ef" * 32
    entries = [{"path": "/pool/a", "offset": 0, "bytes": 100},
               {"path": "/pool/b", "offset": 0, "bytes": 50},
               {"path": "/pool/a", "offset": 100, "bytes": 100}]
    monkeypatch.setattr(staged_lease, "_load_sealed_payload",
                        lambda digest: {"schema": "v1", "entries": entries})
    monkeypatch.setattr(staged_lease, "sdk_submodule", lambda name: SimpleNamespace(
        manifest_read_entries=lambda payload: list(payload["entries"])))
    consumer = "cd" * 32
    root = tmp_path / "residency"
    root.mkdir(parents=True)
    tiers = tmp_path / "tiers"
    tiers.mkdir(parents=True)
    now = real_time.time()
    (tiers / f"{TIER}.json").write_text(json.dumps({
        "schema": "prismabuild.storage_tier.v1", "tier_id": TIER,
        "announced_unix": now - tier_age_s}))
    record = _record(state, expected_in_s=400.0 if state in ("ready", "claimed")
                     else None, now=now)
    record["consumer_action_key"] = consumer
    record["manifest_sha256"] = manifest
    # The third entry, a[100, 200), is read-order bytes [150, 250).
    record["ranges"][0].update({"range_start_bytes": 150, "range_end_bytes": 250})
    (root / f"{consumer}.landing.json").write_text(json.dumps(record))
    resolver = ResidencyResolver(root / f"{consumer}.map.json", tiers_dir=tiers)
    resolver.bind_manifest_sha256(manifest)
    return resolver


def test_the_real_resolver_places_a_span_in_the_read_order_and_waits(
        tmp_path, monkeypatch):
    resolver = _real_resolver(tmp_path, monkeypatch, state="claimed")

    assert resolver.read_order_positions("/pool/a", 120, 130) == [(170, 180)]
    assert resolver.read_order_positions("/pool/b", 10, 20) == [(110, 120)]
    kind, why, movers = reader.landing_verdict(
        resolver, [("/pool/a", 120, 130, 200)])
    assert (kind, movers) == ("wait", (MOVER,))
    assert "claimed" in why
    assert resolver.report()["read_order"] == {"state": "bound", "paths": 2}
    # A span the record does not list keeps the bounded wait.
    assert reader.landing_verdict(
        resolver, [("/pool/a", 10, 20, 200)])[0] == "absent"


def test_the_real_resolver_refuses_a_terminal_range_and_a_silent_loop(
        tmp_path, monkeypatch):
    terminal = _real_resolver(tmp_path / "t", monkeypatch,
                              state="terminal-no-receipt")
    assert reader.landing_verdict(
        terminal, [("/pool/a", 120, 130, 200)])[0] == "refuse"
    silent = _real_resolver(tmp_path / "s", monkeypatch, state="claimed",
                            tier_age_s=600.0)
    kind, why, _movers = reader.landing_verdict(
        silent, [("/pool/a", 120, 130, 200)])
    assert kind == "refuse" and "tier loop" in why


def test_a_landing_record_for_another_manifest_is_not_followed(
        tmp_path, monkeypatch):
    resolver = _real_resolver(tmp_path, monkeypatch, state="claimed")
    resolver.bind_manifest_sha256("12" * 32)

    assert resolver.landing_record() is None
    assert reader.landing_verdict(
        resolver, [("/pool/a", 120, 130, 200)])[0] == "absent"


@pytest.mark.parametrize("state", ["evicted", "done-not-resident"])
def test_a_finished_or_evicted_range_is_waited_on_by_the_real_resolver(
        tmp_path, monkeypatch, state):
    """PB #1009 review: a copy that finished but is not resident waits on
    adoption, and an evicted range on the window.  Neither is a refusal."""
    resolver = _real_resolver(tmp_path, monkeypatch, state=state)

    assert resolver.landing_record() is not None
    kind, why, movers = reader.landing_verdict(
        resolver, [("/pool/a", 120, 130, 200)])
    assert (kind, movers) == ("wait", (MOVER,))
    assert state in why
