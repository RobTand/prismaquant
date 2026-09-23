"""A lease sees what a stage mover has published NOW, not at first look (PQ #905).

Stage A r2, 2026-09-21: one PrismaBuild stage mover covered three shards. It
republishes its fragment and its material sidecar as entries land, all under
one material generation for its run. The reader leased the 1 MiB header entry
of ``model-00057``, waited 37 s for the same mover's 5 GB body entry, saw its
map row, and was refused ``staged-tier-forbidden: unpublished`` -- for bytes
that were on the stage, proven by a sidecar that was on disk. The SDK's
``context`` pre-check cache, kept process-wide by this side, still answered
with the mover's documents from the first look.

The first two cases run the real chain on tiny real files, like
``test_stage_minimal_covers``: the installed SDK, a real claim row, fragments
and material written by the real PrismaBuild writers, the real resolver.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.residency_map import (  # noqa: E402
    ENV_VAR, RANGE_HIT, RANGE_UNCOVERED, bind_residency_manifest,
    residency_map_key, residency_resolver,
    reset_residency_resolver_for_tests)
from prismaquant.staged_lease import (  # noqa: E402
    acquire_entry_window, clear_injected_sdk_for_tests,
    set_lease_helper_root)
from prismaquant.staged_tier_policy import (  # noqa: E402
    activate_staged_tier_policy, deactivate_staged_tier_policy_for_tests)

from test_strict_reader_tier_enforcement import (  # noqa: E402
    MANIFEST, STAGE_TIER, _hex64, _launch_env, _pb, _pb_queue, _write_map)

HEADER = b"\x01header-entry" * 32
BODY = b"\x02body-entry\x00\x00" * 512


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    for name in ("PRISMABUILD_ACTION_KEY", "PRISMABUILD_ACTION_NONCE",
                 "PRISMABUILD_ACTION_SCOPE", "PRISMABUILD_READER_HELPER_ROOT",
                 ENV_VAR):
        monkeypatch.delenv(name, raising=False)

    def scrub():
        reset_residency_resolver_for_tests()
        deactivate_staged_tier_policy_for_tests()
        set_lease_helper_root(None)
        clear_injected_sdk_for_tests()

    scrub()
    yield
    scrub()


class _Mover:
    """One stage mover publishing as its entries land, under one generation."""

    def __init__(self, tmp_path, monkeypatch):
        self.rl, pool_mod, self.map_mod = _pb()
        self.tmp_path, self.monkeypatch = tmp_path, monkeypatch
        self.consumer = _hex64(f"consumer-{tmp_path}")
        self.mover = _hex64(f"mover-{tmp_path}")
        _pb_queue(tmp_path, pool_mod, self.consumer)
        self.stage = tmp_path / "stage" / "prewarm"
        self.root = tmp_path / "residency"
        pool = tmp_path / "pool"
        pool.mkdir(parents=True, exist_ok=True)
        self.files = {}
        for name, blob in (("header", HEADER), ("body", BODY)):
            source, staged = pool / f"{name}.safetensors", self.stage / f"{name}.safetensors"
            source.write_bytes(blob)
            staged.write_bytes(blob)
            self.files[name] = (source, staged)
        self.generation = self.rl.mint_generation()
        _launch_env(monkeypatch, self.consumer)

    def _documents(self, names):
        fragment, material = {}, {}
        for name in names:
            source, staged = self.files[name]
            blob = staged.read_bytes()
            row = {"stage_path": str(staged), "bytes": len(blob),
                   "sha256": hashlib.sha256(blob).hexdigest()}
            key = residency_map_key(str(source), 0)
            fragment[key] = dict(row, offset=0)
            material[key] = dict(row, file_id=self.rl.stat_identity(str(staged)))
        return fragment, material

    def write_fragment(self, names):
        fragment, _material = self._documents(names)
        self.map_mod.write_fragment(self.root, {
            "schema": self.map_mod.RESIDENCY_MAP_FRAGMENT_SCHEMA_V1,
            "consumer_action_key": self.consumer,
            "mover_action_key": self.mover, "tier_id": STAGE_TIER,
            "stage_root": str(self.stage), "manifest_sha256": MANIFEST,
            "entries": fragment})

    def write_material(self, names):
        _fragment, material = self._documents(names)
        self.rl.write_material(
            self.root, consumer_action_key=self.consumer,
            mover_action_key=self.mover, tier_id=STAGE_TIER,
            stage_root=str(self.stage), manifest_sha256=MANIFEST,
            generation=self.generation, entries=material)

    def compose_map(self, names):
        rows = {name: (*self.files[name], None) for name in names}
        path = _write_map(self.tmp_path, rows, leads=[self.mover])
        # A recomposed map is a new file to the resolver's identity gate even
        # when the rewrite lands inside one mtime tick.
        os.utime(path, ns=(time.time_ns(), time.time_ns()))
        self.monkeypatch.setenv(ENV_VAR, str(path))
        reset_residency_resolver_for_tests()
        bind_residency_manifest(MANIFEST)
        return residency_resolver()

    def entry(self, resolver, name):
        source, staged = self.files[name]
        entry = resolver.staged_range(source, 0, staged.stat().st_size)
        assert entry is not None, f"the composed map covers {name}"
        return source, entry


def _read(resolver, source, entry):
    window, key = acquire_entry_window(resolver, source, entry)
    with window:
        fd, _serving = window.open(key)
        try:
            return os.read(fd, int(entry["bytes"]))
        finally:
            window.close_fd(fd)


def test_an_entry_the_mover_landed_after_the_first_lease_is_served(
        tmp_path, monkeypatch):
    mover = _Mover(tmp_path, monkeypatch)
    activate_staged_tier_policy("ssd")
    mover.write_fragment(["header"])
    mover.write_material(["header"])
    resolver = mover.compose_map(["header"])
    assert _read(resolver, *mover.entry(resolver, "header")) == HEADER

    # The same mover, the same run, the same generation: the body lands.
    mover.write_fragment(["header", "body"])
    mover.write_material(["header", "body"])
    resolver = mover.compose_map(["header", "body"])

    assert _read(resolver, *mover.entry(resolver, "body")) == BODY
    assert resolver.report()["bytes_from_pool"] == 0


def test_a_row_whose_sidecar_is_not_written_yet_is_not_published(
        tmp_path, monkeypatch):
    from prismaquant.staged_lease import stage_cover_is_published
    mover = _Mover(tmp_path, monkeypatch)
    activate_staged_tier_policy("ssd")
    mover.write_fragment(["header"])
    mover.write_material(["header"])
    # Fragment first, sidecar second: the map can name the body in between.
    mover.write_fragment(["header", "body"])
    resolver = mover.compose_map(["header", "body"])
    source, entry = mover.entry(resolver, "body")

    assert stage_cover_is_published(resolver, *mover.entry(resolver, "header"))
    assert not stage_cover_is_published(resolver, source, entry)

    mover.write_material(["header", "body"])

    assert stage_cover_is_published(resolver, source, entry)
    assert _read(resolver, source, entry) == BODY


class _Rows:
    """A resolver that covers every span with one entry per declared path."""

    def staged_range_outcome(self, declared, start, end, declared_size=None):
        return {"offset": 0, "bytes": 4096, "stage_path": f"/stage/{declared}"}, RANGE_HIT

    def record_range_wait(self, declared, *, polls, seconds, served):
        self.waited = (str(declared), polls, served)


def test_the_wait_holds_a_covered_span_until_its_proof_is_published(monkeypatch):
    from prismaquant import residency_shard_reader as reader
    monkeypatch.setattr(reader, "STAGED_RANGE_POLL_S", 0.001)
    resolver, asked = _Rows(), []

    def published(_resolver, declared, entry):
        asked.append((declared, entry["offset"]))
        return declared == "ready" or len(asked) > 4

    wanted = [("ready", 0, 10, 4096), ("ready", 10, 20, 4096),
              ("landing", 0, 10, 4096), ("landing", 10, 20, 4096)]
    verdict = reader.await_staged_spans(
        resolver, wanted, deadline=time.monotonic() + 30, published=published)

    assert verdict == RANGE_HIT
    _declared, polls, served = resolver.waited
    assert polls == 3 and served is True
    # Once per staged entry per poll, never per tensor, and never again
    # after a yes.
    assert asked.count(("ready", 0)) == 1
    assert asked.count(("landing", 0)) == len(asked) - 1 >= 2


def test_the_wait_ends_at_its_deadline_and_the_read_decides(monkeypatch):
    from prismaquant import residency_shard_reader as reader
    monkeypatch.setattr(reader, "STAGED_RANGE_POLL_S", 0.001)
    resolver = _Rows()
    verdict = reader.await_staged_spans(
        resolver, [("landing", 0, 10, 4096)],
        deadline=time.monotonic() + 0.02,
        published=lambda *_row: False)
    assert verdict == RANGE_UNCOVERED
    assert resolver.waited[2] is False


def test_without_a_proof_question_the_wait_is_what_it_was():
    from prismaquant import residency_shard_reader as reader
    verdict = reader.await_staged_spans(
        _Rows(), [("any", 0, 10, 4096)], deadline=time.monotonic() + 30)
    assert verdict == RANGE_HIT


def test_the_wait_asks_one_batched_proof_question_per_poll(monkeypatch):
    """PQ #997: one cover lookup for every entry a poll still needs. A
    batched ``None`` (it cannot say which entry is missing) asks each entry
    alone for that poll, so the verdict is the per-entry wait's verdict."""
    from prismaquant import residency_shard_reader as reader
    monkeypatch.setattr(reader, "STAGED_RANGE_POLL_S", 0.001)
    batches, singles = [], []

    def published(_resolver, declared, entry):
        singles.append(declared)
        return declared == "ready" or len(batches) > 2

    def published_batch(_resolver, items):
        batches.append(sorted(declared for declared, _entry in items))
        return True if len(batches) > 2 else None

    wanted = [("ready", 0, 10, 4096), ("ready", 10, 20, 4096),
              ("landing", 0, 10, 4096), ("landing", 10, 20, 4096)]
    resolver = _Rows()
    verdict = reader.await_staged_spans(
        resolver, wanted, deadline=time.monotonic() + 30, published=published,
        published_batch=published_batch)
    assert verdict == RANGE_HIT
    # Poll 1 asks both entries at once, then each alone; "ready" is proven
    # and never asked again. Poll 2 asks "landing" alone in the batch (a
    # None again, so once more on its own). Poll 3's batch proves it.
    assert batches == [["landing", "ready"], ["landing"], ["landing"]]
    assert singles == ["ready", "landing", "landing"]
    assert resolver.waited[1] == 2


def test_a_batched_yes_proves_the_whole_window_in_one_question():
    from prismaquant import residency_shard_reader as reader
    batches = []
    verdict = reader.await_staged_spans(
        _Rows(), [(f"entry-{i}", 0, 10, 4096) for i in range(64)],
        deadline=time.monotonic() + 30,
        published=lambda *_row: pytest.fail("a batched yes needs no per-entry question"),
        published_batch=lambda _resolver, items: batches.append(len(items)) or True)
    assert verdict == RANGE_HIT and batches == [64]


def test_the_batched_proof_is_the_real_cover_lookup(tmp_path, monkeypatch):
    """Real SDK: every entry published proves at once; one entry whose
    sidecar is not written yet makes the batched answer ``None``, and the
    per-entry question then names it."""
    from prismaquant.staged_lease import (
        stage_cover_is_published, stage_covers_are_published)
    mover = _Mover(tmp_path, monkeypatch)
    activate_staged_tier_policy("ssd")
    mover.write_fragment(["header", "body"])
    mover.write_material(["header"])
    resolver = mover.compose_map(["header", "body"])
    header, body = mover.entry(resolver, "header"), mover.entry(resolver, "body")

    assert stage_covers_are_published(resolver, [header]) is True
    assert stage_covers_are_published(resolver, [header, body]) is None
    assert stage_cover_is_published(resolver, *header)
    assert not stage_cover_is_published(resolver, *body)

    mover.write_material(["header", "body"])
    assert stage_covers_are_published(resolver, [header, body]) is True
