"""A retained PWC window's per-file load overhead (PQ #1210).

On GLM-5.3 Stage B row 041 the main thread spent 27% of every render window
waiting on its own window's loads, and the loader threads spent about 55% of
their time on work repeated for every 16 MiB file rather than on reading it:
one reader-lease acquire and release per file (the lease's ``retiring_for``
scan and its lock), a second parse of every archive's central directory on
the pool path, a fresh loader pool per four-file quantum, and one PrismaBuild
cover lookup per entry before the window opened.

These tests run the real strict chain the quantum runs: the pinned
PrismaBuild SDK, real fragments and material from the real writers, a real
claim row, the strict tier policy and a digest-bound PWC. They pin what each
load still does -- every file opened through the SDK under a live pin, its
bytes hashed against the bound digest, released when the window's loads are
done -- and what it no longer repeats.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
from pathlib import Path

import pytest
import torch

import test_strict_reader_tier_enforcement as strict
from test_strict_reader_tier_enforcement import _forget_state  # noqa: F401

from prismaquant import io_engine
from prismaquant import perturbed_x_cache as pxc
from prismaquant import production_weight_cache as pwc

FMT = "TESSERA_E4M3_K1_R1024"
#: Six files at two loaders make three load quanta, so a per-quantum cost
#: shows up as three of something and a per-window cost as one.
COUNT = 6


@pytest.fixture
def workers():
    """Two loaders, or what the shard was given (PQ #1067)."""
    return max(1, min(2, len(os.sched_getaffinity(0))))


def _render(index, shape=(8, 16)):
    return (torch.arange(shape[0] * shape[1], dtype=torch.float32)
            .reshape(shape).to(torch.bfloat16) + index)


def _strict_window(tmp_path, monkeypatch, *, count=COUNT, shape=(8, 16)):
    """A digest-bound PWC whose renders are staged and leased for real."""
    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    stage = strict._stage_root(tmp_path)
    paths, expected, staged_files = {}, {}, {}
    for index in range(count):
        key = (f"model.layers.41.mlp.experts.{index}.down_proj", FMT)
        path = pool / f"unit{index}.pt"
        tensor = _render(index, shape)
        torch.save(tensor, path)
        paths[key], expected[key] = path, tensor
        staged_files[f"u{index}"] = (path, strict._stage_whole(stage, path), None)
    resolver, consumer, _mover = strict._leased_fixture(
        tmp_path, monkeypatch, staged_files)
    cache = pwc.ProductionWeightCache(
        weights={key: str(path) for key, path in paths.items()}, levers={})
    total = sum(path.stat().st_size for path in paths.values())
    cache.enable_lru(4 * total)
    cache.require_file_load_sha256(
        {key: hashlib.sha256(path.read_bytes()).hexdigest()
         for key, path in paths.items()},
        max_file_bytes=max(path.stat().st_size for path in paths.values()))
    return cache, paths, expected, resolver, consumer, total


def _open_window(cache, paths, total, workers, **kwargs):
    return cache.retained_window(
        list(paths), max_resident_bytes=2 * total, max_workers=workers,
        max_load_buffer_bytes=2 * total, release_file_pages=True, **kwargs)


def _sdk_calls(monkeypatch, *names):
    """Count calls into the pinned SDK the strict reader actually makes."""
    sdk = strict._pb()[0]
    calls = {name: [] for name in names}
    for name in names:
        original = getattr(sdk, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name].append(threading.current_thread().name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(sdk, name, counted)
    return calls


def _loaded(cache, paths):
    """Every resident render and its detached file-load receipt."""
    out = {}
    for key in paths:
        tensor = cache.get_resident(*key)
        out[key] = (tensor.clone(), cache.file_load_receipt(key, tensor),
                    cache.resident_render_identity(*key, tensor))
    return out


# --------------------------------------------------------------------------
# One reader-lease window per retained window
# --------------------------------------------------------------------------

def test_a_retained_window_pins_its_staged_renders_under_one_lease(
        tmp_path, monkeypatch, workers):
    cache, paths, expected, resolver, consumer, total = _strict_window(
        tmp_path, monkeypatch)
    calls = _sdk_calls(monkeypatch, "acquire_for", "open_pinned", "release")
    with _open_window(cache, paths, total, workers) as receipt:
        # The window's loads are done and the pin is gone before the consumer
        # runs: a pin protects the reads, not the consumer's compute.
        assert strict._pins_live(tmp_path, consumer) == []
        loaded = _loaded(cache, paths)
        assert receipt["resident_bytes"] <= receipt["budget_bytes"]
    # One acquire and one release for the window, not one per file ...
    assert len(calls["acquire_for"]) == 1, calls["acquire_for"]
    assert len(calls["release"]) == 1
    # ... and still one SDK open per file, each under that pin.
    assert len(calls["open_pinned"]) == len(paths)
    for key, (tensor, file_receipt, _identity) in loaded.items():
        assert torch.equal(tensor.view(torch.uint8), expected[key].view(torch.uint8))
        assert file_receipt["serving_tier"] == "stage"
        assert file_receipt["sha256"] == hashlib.sha256(
            paths[key].read_bytes()).hexdigest()
    report = resolver.report()
    assert report["bytes_from_pool"] == 0
    assert report["bytes_from_stage"] == sum(p.stat().st_size for p in paths.values())
    assert all(record.get("pin_id") for record in report["serving_tiers"])
    assert all(isinstance(value, str) for value in cache.weights.values())


def test_a_refused_window_lease_reads_each_render_under_its_own_lease(
        tmp_path, monkeypatch, workers):
    """The batched lease is an optimization: a refusal re-reads per entry."""
    from prismaquant import staged_lease

    cache, paths, expected, _resolver, consumer, total = _strict_window(
        tmp_path, monkeypatch)

    def refuse(*_args, **_kwargs):
        raise staged_lease.LeaseRefused("unpublished: forced", kind="availability")

    monkeypatch.setattr(staged_lease, "acquire_entries_window", refuse)
    calls = _sdk_calls(monkeypatch, "acquire_for", "release")
    with _open_window(cache, paths, total, workers):
        loaded = _loaded(cache, paths)
    assert len(calls["acquire_for"]) == len(paths)
    assert len(calls["release"]) == len(paths)
    assert strict._pins_live(tmp_path, consumer) == []
    for key, (tensor, file_receipt, _identity) in loaded.items():
        assert torch.equal(tensor.view(torch.uint8), expected[key].view(torch.uint8))
        assert file_receipt["serving_tier"] == "stage"


def test_a_bad_staged_copy_fails_its_window_by_name_and_releases_the_pin(
        tmp_path, monkeypatch, workers):
    """A copy that no longer matches its pin refuses; nothing falls back."""
    from prismaquant.staged_tier_policy import TierPolicyRefused

    cache, paths, _expected, resolver, consumer, total = _strict_window(
        tmp_path, monkeypatch)
    victim = list(paths.values())[3]
    staged = strict._stage_root(tmp_path) / victim.name
    raw = bytearray(staged.read_bytes())
    raw[-40] ^= 0xFF
    real_open = strict._pb()[0].open_pinned
    corrupted = []

    def open_then_damage(queue, pin, ref_id, key, **kwargs):
        # Damage after the acquire proved the copy, so the check that
        # refuses is the read's own, on the bytes it hashed.
        if str(victim) in key and not corrupted:
            staged.write_bytes(bytes(raw))
            corrupted.append(key)
        return real_open(queue, pin, ref_id, key, **kwargs)

    monkeypatch.setattr(strict._pb()[0], "open_pinned", open_then_damage)
    with pytest.raises((TierPolicyRefused, RuntimeError)):
        with _open_window(cache, paths, total, workers):
            pass
    assert corrupted
    assert strict._pins_live(tmp_path, consumer) == []
    assert resolver.report()["bytes_from_pool"] == 0
    assert getattr(cache, "_resident_window_files", None) is None
    assert all(isinstance(value, str) for value in cache.weights.values())


# --------------------------------------------------------------------------
# One archive parse per file
# --------------------------------------------------------------------------

def test_each_render_archive_is_parsed_once_on_the_bytes_it_loads(
        tmp_path, monkeypatch, workers):
    cache, paths, _expected, _resolver, _consumer, total = _strict_window(
        tmp_path, monkeypatch)
    scans = []
    original = pxc.torch_archive_storage_bytes

    def counted(source, **kwargs):
        scans.append(("path" if isinstance(source, (str, Path)) else "bytes",
                      threading.current_thread() is threading.main_thread()))
        return original(source, **kwargs)

    monkeypatch.setattr(pxc, "torch_archive_storage_bytes", counted)
    with _open_window(cache, paths, total, workers):
        pass
    # Once per file, on the bytes the loader hashed and deserializes; never a
    # second open of the declared pool file to read its directory.
    assert len(scans) == len(paths), scans
    assert {kind for kind, _main in scans} == {"bytes"}
    assert not [main for _kind, main in scans if main]


def test_a_retained_window_still_charges_every_file_before_its_first_load(
        tmp_path, monkeypatch, workers):
    """The priced bound is the file length the sealed plan charges."""
    cache, paths, _expected, _resolver, _consumer, total = _strict_window(
        tmp_path, monkeypatch)
    budgets = []

    def record(state):
        budgets.append(dict(state))

    with _open_window(cache, paths, total, workers, before_load_quantum=record):
        pass
    assert budgets[0]["remaining_incoming_storage_bytes"] >= sum(
        pxc.torch_archive_storage_bytes(path) for path in paths.values())
    assert budgets[0]["remaining_incoming_storage_bytes"] <= total
    # A budget one byte short of the charged bound refuses before any load.
    loads = []
    monkeypatch.setattr(cache, "prefetch", lambda *a, **k: loads.append(a) or 0)
    with pytest.raises(RuntimeError, match="retained resident storage exceeds budget"):
        with cache.retained_window(list(paths), max_resident_bytes=total - 1,
                                   max_workers=workers, max_load_buffer_bytes=total):
            pass
    assert loads == []


# --------------------------------------------------------------------------
# The IO engine's one pool, within the window's buffer cap
# --------------------------------------------------------------------------

def test_a_retained_window_loads_on_the_io_engines_one_pool(
        tmp_path, monkeypatch, workers):
    """Every load runs on the process's one IO pool, never the main thread.

    The pool is the IO engine's (PQ #1294): its width follows the CPUs this
    process was given, and no caller states a worker count.
    """
    cache, paths, _expected, _resolver, _consumer, total = _strict_window(
        tmp_path, monkeypatch)
    threads = []
    original = io_engine.load_file

    def counted(path, limit, **kwargs):
        threads.append(threading.current_thread().name)
        return original(path, limit, **kwargs)

    monkeypatch.setattr(io_engine, "load_file", counted)
    with _open_window(cache, paths, total, workers) as receipt:
        assert len(receipt["load_quanta"]) > 1
    assert len(threads) == len(paths)
    assert all(name.startswith("pq-io") for name in threads), threads
    assert len(set(threads)) <= io_engine.ENGINE.width


def test_the_load_buffers_in_flight_never_exceed_the_windows_buffer_cap(
        tmp_path, monkeypatch, workers):
    """A serialized buffer lives from its read to the end of its decode.

    The IO engine starts a read only while the buffers already in flight
    leave room for it under the window's cap (PQ #1291), so however many
    reads run at once, their buffers stay within what the window charged.
    """
    cache, paths, _expected, _resolver, _consumer, total = _strict_window(
        tmp_path, monkeypatch)
    sizes = {str(path.absolute()): path.stat().st_size for path in paths.values()}
    live = []
    peaks = []
    lock = threading.Lock()
    original = io_engine.load_file

    def counted(path, limit, **kwargs):
        with lock:
            live.append(sizes[str(path)])
            peaks.append(sum(live))
        try:
            return original(path, limit, **kwargs)
        finally:
            with lock:
                live.remove(sizes[str(path)])

    monkeypatch.setattr(io_engine, "load_file", counted)
    largest = max(sizes.values())
    buffer_cap = workers * largest
    with cache.retained_window(list(paths), max_resident_bytes=2 * total,
                               max_workers=workers,
                               max_load_buffer_bytes=buffer_cap) as receipt:
        assert receipt["load_buffer_capacity_bytes"] <= buffer_cap
        assert receipt["resident_bytes"] <= receipt["budget_bytes"]
    assert peaks and max(peaks) <= buffer_cap


# --------------------------------------------------------------------------
# The same bytes, either way
# --------------------------------------------------------------------------

def test_the_window_lease_serves_the_same_bytes_as_per_file_leases(
        tmp_path, monkeypatch, workers):
    """Real-shaped renders: one 16 MiB bf16 expert projection per file."""
    from prismaquant import staged_lease

    arms = {}
    for arm in ("per_file", "window"):
        with monkeypatch.context() as patch:
            cache, paths, expected, _resolver, consumer, total = _strict_window(
                tmp_path / arm, patch, count=3, shape=(2048, 4096))
            if arm == "per_file":
                def refuse(*_args, **_kwargs):
                    raise staged_lease.LeaseRefused("unpublished: arm",
                                                    kind="availability")
                patch.setattr(staged_lease, "acquire_entries_window", refuse)
            with _open_window(cache, paths, total, workers, render_identities=True):
                loaded = _loaded(cache, paths)
            assert strict._pins_live(tmp_path / arm, consumer) == []
        arms[arm] = {
            key[0]: (hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
                     tuple(tensor.shape), str(tensor.dtype),
                     {k: v for k, v in receipt.items() if k != "path"}, identity)
            for key, (tensor, receipt, identity) in loaded.items()}
        for key, (tensor, _receipt, _identity) in loaded.items():
            assert torch.equal(tensor.view(torch.uint8), expected[key].view(torch.uint8))
    assert arms["window"] == arms["per_file"]


# --------------------------------------------------------------------------
# One cover proof per window before it opens
# --------------------------------------------------------------------------

class _Progress:
    def __init__(self):
        self.phases = []

    def enter_read_phase(self, name):
        self.phases.append(name)


def test_the_window_readiness_proves_its_entries_in_one_cover_lookup(
        tmp_path, monkeypatch):
    from prismaquant.joint_cost_quantum import prepare_retained_window_read

    _cache, paths, _expected, _resolver, _consumer, _total = _strict_window(
        tmp_path, monkeypatch)
    record = {"executable_readset": {"prepared_input": {"windows": [{
        "window_index": 3,
        "entries": [{"path": str(path), "offset": 0, "bytes": path.stat().st_size}
                    for path in paths.values()]}]}}}
    calls = _sdk_calls(monkeypatch, "covers_for_keys")
    progress = _Progress()
    assert prepare_retained_window_read(3, record=record, progress=progress) == "ready"
    assert len(calls["covers_for_keys"]) == 1, calls
    assert progress.phases


def test_the_window_readiness_asks_per_entry_when_the_batch_cannot_say(
        tmp_path, monkeypatch):
    """A batched proof that refuses falls back to one lookup per entry."""
    from prismaquant import staged_lease
    from prismaquant.joint_cost_quantum import prepare_retained_window_read

    _cache, paths, _expected, _resolver, _consumer, _total = _strict_window(
        tmp_path, monkeypatch)
    record = {"executable_readset": {"prepared_input": {"windows": [{
        "window_index": 0,
        "entries": [{"path": str(path), "offset": 0, "bytes": path.stat().st_size}
                    for path in paths.values()]}]}}}
    monkeypatch.setattr(staged_lease, "stage_covers_are_published",
                        lambda resolver, items: None)
    calls = _sdk_calls(monkeypatch, "covers_for_keys")
    assert prepare_retained_window_read(0, record=record, progress=_Progress()) == "ready"
    assert len(calls["covers_for_keys"]) == len(paths)
