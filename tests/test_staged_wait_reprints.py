"""A staged-range wait keeps saying where it stands (PQ #1167).

The layer-43 Stage B profile (PB ``93247fc291c0``) printed ``expected to land
in 8 s`` once, at 08:06:23Z, and then nothing for 494 s, until the span
landed.  The wait loop printed only when ``(kind, first clause)`` changed,
and the first clause, ``mover … is claimed``, stayed the same while the
expectation passed and PrismaBuild revised it.  From the log, a wait that
outlives its expectation sixty times over looks exactly like a hang.

The campaign's observability rule is that no phase runs two minutes with no
output.  So the wait prints again when the expectation it printed passes,
and at a bounded interval while it lasts, with the record's current
expectation and the rate and basis PrismaBuild priced it on.

The clock is faked, so these tests take no wall time.
"""
from __future__ import annotations

import pytest

import prismaquant.residency_shard_reader as reader
from prismaquant.residency_map import RANGE_HIT, RANGE_UNCOVERED

MOVER = "ab" * 32
TIER = "prismabuild-stage:dl380g10"
#: The campaign's observability rule: no phase runs two minutes silent.
SILENCE_BOUND_S = 120.0
START = 1_000_000.0


class _Clock:
    """``time`` for the reader module: sleeping advances a fake clock."""

    def __init__(self, start: float = START):
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(float(seconds), 0.0)


class _Resolver:
    """A resolver whose span lands on poll ``lands_at``, with a record that
    ``revise`` may replace at a given poll."""

    def __init__(self, *, lands_at: int | None, record: dict | None,
                 revise: tuple[int, dict] | None = None):
        self.lands_at = lands_at
        self.record = record
        self.revise = revise
        self.polls = 0

    def staged_range_outcome(self, declared, start, end, declared_size=None):
        self.polls += 1
        if self.revise is not None and self.polls >= self.revise[0]:
            self.record, self.revise = self.revise[1], None
        if self.lands_at is not None and self.polls >= self.lands_at:
            return {"offset": start, "bytes": end - start}, RANGE_HIT
        return None, RANGE_UNCOVERED

    def record_range_wait(self, declared, **kwargs):
        pass

    def record_range_refusal(self, declared, detail):
        pass

    def landing_record(self):
        return self.record

    def read_order_positions(self, declared, start, end):
        return [(1_000 + start, 1_000 + end)]

    def tier_record_age(self, tier_id):
        return 5.0


def _record(state: str, *, expected_unix: float | None, now: float,
            rate: float = 134e6, basis: str = "measured") -> dict:
    return {
        "schema": "prismaquant.prismabuild.residency_landing.v1",
        "consumer_action_key": "cd" * 32, "tier_id": TIER,
        "manifest_sha256": "ef" * 32, "written_unix": now,
        "landing_bytes_per_s": rate, "landing_basis": basis,
        "rates_measured_bytes_per_s": [134e6, 239e6],
        "rate_min_bytes_per_s": 134e6, "rate_max_bytes_per_s": 239e6,
        "report_latency_s": 90.0, "tier_loop_liveness_s": 120.0,
        "ranges": [{
            "mover_action_key": MOVER, "phase": "chain-044", "chunk_index": None,
            "range_start_bytes": 0, "range_end_bytes": 10 ** 9,
            "state": state, "expected_landing_unix": expected_unix,
            "queue_position": 0 if state == "claimed" else None,
            "bytes_ahead": 0 if state == "claimed" else None,
            "claimed_unix": now - 10 if state == "claimed" else None,
            "waiting_for": "",
        }],
    }


@pytest.fixture
def clock(monkeypatch, tmp_path):
    fake = _Clock()
    monkeypatch.setattr(reader, "time", fake)
    monkeypatch.delenv(reader.STAGED_RANGE_WAIT_ENV, raising=False)
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PATH", str(tmp_path / "action.progress"))
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", "t" * 32)
    return fake


@pytest.fixture
def lines(monkeypatch, clock):
    """Every wait line the reader prints, with the fake clock's time."""

    said: list[tuple[float, str]] = []

    def capture(*args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        if text.startswith("[residency] staged-range wait"):
            said.append((clock.now, text))

    monkeypatch.setattr(reader, "print", capture, raising=False)
    return said


def _wait(resolver, clock):
    return reader.await_staged_spans(
        resolver, [("/pool/shard-1.safetensors", 0, 64, 128)],
        deadline=clock.monotonic() + reader.staged_range_wait_s())


def _longest_silence(said, *, end: float) -> float:
    times = [START] + [when for when, _text in said] + [end]
    return max(later - earlier for earlier, later in zip(times, times[1:]))


def test_the_layer_43_wait_is_never_silent_past_the_bound(clock, lines):
    """The acceptance case: 8 s expected, 495 s taken."""

    resolver = _Resolver(lands_at=496, record=_record(
        "claimed", expected_unix=START + 8.0, now=START))

    assert _wait(resolver, clock) == RANGE_HIT
    assert clock.now - START >= 495.0
    silence = _longest_silence(lines, end=clock.now)
    assert silence <= SILENCE_BOUND_S, (
        f"the wait printed {len(lines)} line(s) and was silent {silence:.0f} s: "
        + " | ".join(text for _when, text in lines))
    # A line a minute, not a line a poll.
    assert len(lines) <= 495.0 / reader.STAGED_WAIT_REPORT_S + 3, lines


def test_a_passed_expectation_is_said_at_once(clock, lines):
    """The first poll past the printed expectation prints the record again."""

    resolver = _Resolver(lands_at=100, record=_record(
        "claimed", expected_unix=START + 8.0, now=START))

    assert _wait(resolver, clock) == RANGE_HIT
    after = [(when, text) for when, text in lines if when >= START + 8.0]
    assert after, [text for _when, text in lines]
    when, text = after[0]
    assert when <= START + 8.0 + 2 * reader.STAGED_RANGE_POLL_S, lines
    assert "ago" in text, text


def test_a_revised_expectation_is_said_and_followed(clock, lines):
    """PrismaBuild revises the expectation; the wait prints the new one, and
    prints again when that one passes too."""

    revised = START + 200.0
    resolver = _Resolver(
        lands_at=400,
        record=_record("claimed", expected_unix=START + 8.0, now=START),
        revise=(30, _record("claimed", expected_unix=revised, now=START + 29.0,
                            rate=30.8e6, basis="reported")))

    assert _wait(resolver, clock) == RANGE_HIT
    named = [(when, text) for when, text in lines
             if START + 29.0 <= when < revised and "expected to land in" in text]
    assert named, [text for _when, text in lines]
    assert "30.8 MB/s" in named[0][1] and "reported" in named[0][1], named[0][1]
    after = [when for when, _text in lines if when >= revised]
    assert after and after[0] <= revised + 2 * reader.STAGED_RANGE_POLL_S, lines


def test_a_wait_with_no_expectation_still_reports(clock, lines):
    """An unpublished range with no priced landing is waited on while the tier
    loop lives; the log still says so at the bounded interval."""

    resolver = _Resolver(lands_at=400, record=_record(
        "unpublished", expected_unix=None, now=START))

    assert _wait(resolver, clock) == RANGE_HIT
    silence = _longest_silence(lines, end=clock.now)
    assert silence <= SILENCE_BOUND_S, [text for _when, text in lines]
    assert all("no expected landing time" in text for _when, text in lines), lines


def test_a_bounded_wait_reports_what_is_left(clock, lines):
    """With no landing record the wait is bounded; each line says how much of
    the bound is left, and the bound still refuses exactly as before."""

    resolver = _Resolver(lands_at=None, record=None)

    assert _wait(resolver, clock) == RANGE_UNCOVERED
    assert clock.now - START == pytest.approx(reader.STAGED_RANGE_WAIT_S)
    silence = _longest_silence(lines, end=clock.now)
    assert silence <= SILENCE_BOUND_S, [text for _when, text in lines]
    assert all("bounded wait of 300 s" in text for _when, text in lines), lines
    assert any("s left" in text for _when, text in lines[1:]), lines
