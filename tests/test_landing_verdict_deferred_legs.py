"""The reader follows every leg PrismaBuild lists, and refuses what it cannot (#1113).

PB #1018 lists every leg a claimed consumer has still to read in its landing
record while the consumer has a refill horizon, not only the legs inside it.
A leg past the horizon is ``unpublished`` with ``deferred_by: horizon``, and
the record's ``horizon`` block says where the window stops and the
consumption rate that moves it.  Such a record carries the ``horizon`` key
(``null`` when the horizon is undefined); a record without it predates #1018.

What the reader does with that:

* A deferred leg is waited on while the tier loop lives, as any unpublished
  range is, and the wait's log line says it is deferred, where the horizon
  ends and what moves it.
* A span outside the bound read order is refused at once, naming the span:
  no leg of the consumer's plan covers it, so nothing will stage it.  A v2
  manifest can declare an entry that no read phase names, which is such a
  span.  Before #1113 it was ``absent`` and the reader waited out its bounded
  clock first.
* With no record, or a record without the ``horizon`` key, the reader does
  exactly what it did before: the clock.

The clock is faked, so these tests take no wall time.
"""
from __future__ import annotations

import hashlib
import json
import time as real_time
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import prismaquant.residency_shard_reader as reader
from prismaquant import layer_streaming, staged_lease
from prismaquant.residency_map import (
    ENV_VAR, RANGE_HIT, RANGE_UNCOVERED, SCHEMA, TIERS_DIR_ENV_VAR,
    ResidencyResolver, bind_residency_manifest, reset_residency_resolver_for_tests,
    residency_map_key, residency_resolver)

TIER = "prismabuild-stage:dl380g10"
MANIFEST = "ef" * 32
CONSUMER = "cd" * 32
MOVER = "ab" * 32
HEAD_BYTES = 1000
#: The consumption rate R12 read at on 2026-09-22.
RATE = 20.7e6


class _Clock:
    """``time`` for the reader module: sleeping advances a fake clock.

    ``at(seconds, action)`` runs ``action`` once the fake clock has advanced
    ``seconds`` past its start: the mover landing while the reader waits.
    """

    def __init__(self, start: float = 1_000_000.0):
        self.start = self.now = start
        self.pending: list[tuple[float, object]] = []

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(float(seconds), 0.0)
        for due, action in list(self.pending):
            if self.now - self.start >= due:
                self.pending.remove((due, action))
                action()

    def at(self, seconds: float, action) -> None:
        self.pending.append((seconds, action))


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


@pytest.fixture(autouse=True)
def _forget_resolver(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    reset_residency_resolver_for_tests()
    yield
    reset_residency_resolver_for_tests()


def _shard(tmp_path):
    """One real safetensors file on the pool, and its tensors."""
    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    tensors = {
        "f32": torch.linspace(-3, 3, 256, dtype=torch.float32).reshape(8, 32),
        "bf16": (torch.arange(512, dtype=torch.int32).reshape(16, 32)
                 .to(torch.bfloat16)),
        "i8": torch.arange(-32, 32, dtype=torch.int8).reshape(8, 8),
    }
    path = pool / "model-00001-of-00002.safetensors"
    save_file(tensors, str(path))
    return path, tensors


#: A v2 manifest entry that no read phase names: the sealed readset declares
#: it, and the read order, which is the read plan's expansion, does not.
UNREAD = "/pool/unread.safetensors"


def _read_order(monkeypatch, shard):
    """The sealed manifest: a head entry, the shard whole, and one unread entry.

    Only the hops that need a live PrismaBuild claim are stood in for, as
    ``test_staged_wait_follows_pb_landing`` does: the sealed manifest and the
    generation's ``storage_tiers``.  Its read order is the read plan's, which
    names the head and the shard and not ``UNREAD``, so the shard is
    read-order bytes ``[HEAD_BYTES, HEAD_BYTES + size)``.
    """
    read = [{"path": "/pool/head", "offset": 0, "bytes": HEAD_BYTES},
            {"path": str(shard), "offset": 0, "bytes": shard.stat().st_size}]
    entries = read + [{"path": UNREAD, "offset": 0, "bytes": 64}]
    monkeypatch.setattr(staged_lease, "_load_sealed_payload",
                        lambda digest: {"schema": "v2", "entries": entries})
    monkeypatch.setattr(staged_lease, "sdk_submodule", lambda name: SimpleNamespace(
        manifest_read_entries=lambda payload: list(read)))


def _landing(size: int, *, now: float, lists_every_leg: bool = True) -> dict:
    """PB's record for a consumer reading ``head``, the shard deferred.

    ``lists_every_leg`` False is a record written before PB #1018: no
    ``horizon`` block and no ``deferred_by``.
    """
    row = {
        "mover_action_key": MOVER, "phase": "layer-0", "chunk_index": None,
        "range_start_bytes": HEAD_BYTES, "range_end_bytes": HEAD_BYTES + size,
        "state": "unpublished", "expected_landing_unix": None,
        "queue_position": None, "bytes_ahead": None, "claimed_unix": None,
        "waiting_for": (f"the refill horizon, which ends at byte {HEAD_BYTES}: "
                        "the window publishes this range once the consumer's "
                        "accepted progress past head brings it inside"),
    }
    record = {
        "schema": "prismaquant.prismabuild.residency_landing.v1",
        "consumer_action_key": CONSUMER, "tier_id": TIER,
        "manifest_sha256": MANIFEST, "written_unix": now,
        "landing_bytes_per_s": 134e6, "landing_basis": "measured",
        "rates_measured_bytes_per_s": [134e6],
        "rate_min_bytes_per_s": 134e6, "rate_max_bytes_per_s": 134e6,
        "report_latency_s": 35.0, "tier_loop_liveness_s": 120.0,
        "publish_s": 0.01, "ranges": [row],
    }
    if lists_every_leg:
        row["deferred_by"] = "horizon"
        record["horizon"] = {
            "end_bytes": HEAD_BYTES, "accepted_phase": "head",
            "reading_phase": "head", "advance_mover_action_key": MOVER,
            "consumption_bytes_per_s": RATE, "consumption_basis": "measured",
            "readahead_bytes": 0, "reach_end_bytes": HEAD_BYTES,
            "declared_wait_end_bytes": None}
    return record


def _write_map(root, stage_root, rows):
    """The composed map: ``rows`` is [(declared Path, staged Path)], whole files."""
    entries = {}
    for declared, staged in rows:
        blob = declared.read_bytes()
        entries[residency_map_key(str(declared), 0)] = {
            "stage_path": str(staged), "bytes": len(blob), "offset": 0,
            "sha256": hashlib.sha256(blob).hexdigest()}
    path = root / f"{CONSUMER}.map.json"
    path.write_text(json.dumps({
        "schema": SCHEMA, "tier_id": TIER, "stage_root": str(stage_root),
        "manifest_sha256": MANIFEST, "leads": ["d" * 64], "generation": 3,
        "entries": entries}))
    return path


def _consumer(tmp_path, monkeypatch, *, lists_every_leg=True, tier_age_s=5.0):
    """A bound resolver: map, landing record and tier record on disk."""
    shard, tensors = _shard(tmp_path)
    _read_order(monkeypatch, shard)
    root = tmp_path / "residency"
    root.mkdir(parents=True)
    tiers = tmp_path / "tiers"
    tiers.mkdir(parents=True)
    stage_root = tmp_path / "stage" / "prewarm"
    stage_root.mkdir(parents=True)
    now = real_time.time()
    (tiers / f"{TIER}.json").write_text(json.dumps({
        "schema": "prismabuild.storage_tier.v1", "tier_id": TIER,
        "announced_unix": now - tier_age_s}))
    (root / f"{CONSUMER}.landing.json").write_text(json.dumps(_landing(
        shard.stat().st_size, now=now, lists_every_leg=lists_every_leg)))
    map_path = _write_map(root, stage_root, [])
    monkeypatch.setenv(ENV_VAR, str(map_path))
    monkeypatch.setenv(TIERS_DIR_ENV_VAR, str(tiers))
    reset_residency_resolver_for_tests()
    bind_residency_manifest(MANIFEST)
    resolver = residency_resolver()
    assert isinstance(resolver, ResidencyResolver)
    return SimpleNamespace(resolver=resolver, shard=shard, tensors=tensors,
                           root=root, stage_root=stage_root, map_path=map_path)


def _span(shard, name):
    raw = shard.read_bytes()
    size = int.from_bytes(raw[:8], "little")
    row = json.loads(raw[8:8 + size])[name]
    base = 8 + size
    return base + row["data_offsets"][0], base + row["data_offsets"][1]


def _wait(resolver, clock, rows):
    return reader.await_staged_spans(
        resolver, rows, deadline=clock.monotonic() + reader.staged_range_wait_s())


# -- a deferred leg is waited on, and then read --------------------------------

def test_a_leg_deferred_by_the_horizon_is_waited_for_then_read_bitwise(
        tmp_path, monkeypatch, clock, capsys):
    """The acceptance case: the leg lands after three of the reader's clocks.

    The wait follows the record while the tier loop lives, its log line
    says the leg is deferred by the refill horizon and what moves it, and
    the bytes read afterwards are the pool's, bit for bit.
    """
    run = _consumer(tmp_path, monkeypatch)
    start, end = _span(run.shard, "f32")
    size = run.shard.stat().st_size
    landed_after = 3 * reader.STAGED_RANGE_WAIT_S + 30.0

    def land():
        staged = run.stage_root / run.shard.name
        staged.write_bytes(run.shard.read_bytes())
        _write_map(run.root, run.stage_root, [(run.shard, staged)])

    clock.at(landed_after, land)

    verdict = _wait(run.resolver, clock, [(str(run.shard), start, end, size)])

    assert verdict == RANGE_HIT
    assert clock.now - clock.start >= landed_after
    said = capsys.readouterr().out
    assert "following PrismaBuild's landing record" in said, said
    assert "deferred by the refill horizon" in said, said
    assert f"at byte {HEAD_BYTES}" in said, said
    assert f"{RATE / 1e6:.1f} MB/s" in said, said
    with layer_streaming._source_safe_open(str(run.shard), framework="pt") as got:
        with safe_open(str(run.shard), framework="pt") as want:
            for name in sorted(run.tensors):
                mine, theirs = got.get_tensor(name), want.get_tensor(name)
                assert mine.dtype == theirs.dtype and mine.shape == theirs.shape
                assert torch.equal(mine.view(torch.uint8), theirs.view(torch.uint8))
    assert run.resolver.report()["range_hits"] >= len(run.tensors)


def test_a_deferred_leg_is_refused_once_the_tier_loop_is_silent(
        tmp_path, monkeypatch, clock):
    run = _consumer(tmp_path, monkeypatch, tier_age_s=600.0)
    start, end = _span(run.shard, "f32")

    kind, why, movers = reader.landing_verdict(
        run.resolver, [(str(run.shard), start, end, 0)])

    assert (kind, movers) == ("refuse", ())
    assert "tier loop" in why and "deferred by the refill horizon" in why


# -- a span outside the bound read order ----------------------------------------

def test_a_span_outside_the_bound_read_order_refuses_at_once_naming_it(
        tmp_path, monkeypatch, clock):
    """No leg of the plan covers it, so no mover will stage it.

    Before #1113 the verdict was ``absent``, and the reader waited out its
    300 s clock before it refused the same span.
    """
    run = _consumer(tmp_path, monkeypatch)
    stray = UNREAD
    assert run.resolver.staged_range_outcome(stray, 10, 20)[1] == RANGE_UNCOVERED

    kind, why, movers = reader.landing_verdict(
        run.resolver, [(stray, 10, 20, 64)])

    assert (kind, movers) == ("refuse", ())
    assert stray in why and "[10, 20)" in why and "read order" in why

    verdict = _wait(run.resolver, clock, [(stray, 10, 20, 64)])
    assert verdict == RANGE_UNCOVERED
    assert clock.now - clock.start < 5.0
    report = run.resolver.report()
    assert stray in str(report["range_wait_refusal"]), report


# -- what a reader without the new record does ---------------------------------

def test_a_record_written_before_every_leg_was_listed_keeps_the_clock(
        tmp_path, monkeypatch, clock):
    """An older PrismaBuild's record: a span it does not list, or one outside
    the read order, is ``absent``, and the bounded wait applies as before."""
    run = _consumer(tmp_path, monkeypatch, lists_every_leg=False)
    stray = UNREAD

    kind, _why, _movers = reader.landing_verdict(
        run.resolver, [(stray, 10, 20, 64)])
    assert kind == "absent"
    kind, _why, _movers = reader.landing_verdict(
        run.resolver, [("/pool/head", 0, 10, HEAD_BYTES)])
    assert kind == "absent"

    verdict = _wait(run.resolver, clock, [(stray, 10, 20, 64)])
    assert verdict == RANGE_UNCOVERED
    waited = clock.now - clock.start
    assert reader.STAGED_RANGE_WAIT_S <= waited < reader.STAGED_RANGE_WAIT_S + 2


def test_without_a_record_a_span_outside_the_read_order_keeps_the_clock(
        tmp_path, monkeypatch, clock):
    run = _consumer(tmp_path, monkeypatch)
    (run.root / f"{CONSUMER}.landing.json").unlink()
    stray = UNREAD

    verdict = _wait(run.resolver, clock, [(stray, 10, 20, 64)])

    assert verdict == RANGE_UNCOVERED
    waited = clock.now - clock.start
    assert reader.STAGED_RANGE_WAIT_S <= waited < reader.STAGED_RANGE_WAIT_S + 2
