"""Retained PWC bytes, budgets and native descriptor lifetimes (PQ #1210)."""
from __future__ import annotations

import hashlib
import io
import os
import threading
from pathlib import Path

import pytest
import torch

from staged_native_fixtures import bind, native_staged_state, publish_inputs
from prismaquant.staged_tier_policy import staged_tier_policy_test_context
from test_streamed_metadata_staged_reads import _deny_pool_opens

from prismaquant import io_engine
from prismaquant import perturbed_x_cache as pxc
from prismaquant import production_weight_cache as pwc

FMT = "TESSERA_E4M3_K1_R1024"
#: Six files at two loaders make three load quanta, so a per-quantum cost
#: shows up as three of something and a per-window cost as one.
COUNT = 6


@pytest.fixture(autouse=True)
def _native_policy(native_staged_state):
    with staged_tier_policy_test_context('ssd'):
        yield


@pytest.fixture
def workers():
    """Two loaders, or what the shard was given (PQ #1067)."""
    return max(1, min(2, len(os.sched_getaffinity(0))))


def _render(index, shape=(8, 16)):
    return (torch.arange(shape[0] * shape[1], dtype=torch.float32)
            .reshape(shape).to(torch.bfloat16) + index)


def _strict_window(tmp_path, monkeypatch, *, count=COUNT, shape=(8, 16)):
    """A digest-bound PWC backed by real immutable published renders."""
    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    paths, expected = {}, {}
    for index in range(count):
        key = (f"model.layers.41.mlp.experts.{index}.down_proj", FMT)
        path = pool / f"unit{index}.pt"
        tensor = _render(index, shape)
        torch.save(tensor, path)
        paths[key], expected[key] = path, tensor
    resolver = bind(monkeypatch, publish_inputs(tmp_path, paths.values()))
    cache = pwc.ProductionWeightCache(
        weights={key: str(path) for key, path in paths.items()}, levers={})
    total = sum(path.stat().st_size for path in paths.values())
    cache.enable_lru(4 * total)
    cache.require_file_load_sha256(
        {key: hashlib.sha256(path.read_bytes()).hexdigest()
         for key, path in paths.items()},
        max_file_bytes=max(path.stat().st_size for path in paths.values()))
    return cache, paths, expected, resolver, total


def _open_window(cache, paths, total, workers, *, release_file_pages=True, **kwargs):
    return cache.retained_window(
        list(paths), max_resident_bytes=2 * total, max_workers=workers,
        max_load_buffer_bytes=2 * total, release_file_pages=release_file_pages, **kwargs)


def _observe_windows(monkeypatch, resolver):
    """Observe actual opens, live-owner barriers and immediate FD closure."""
    from prismaquant.staged_lease import LeaseWindow
    entered, exited, opened = [], [], []
    real_enter, real_exit = LeaseWindow.__enter__, LeaseWindow.__exit__
    real_open, real_close = LeaseWindow.open, LeaseWindow.close_fd
    real_os_close = os.close
    closed = []

    def enter(window):
        result = real_enter(window)
        entered.append(window)
        with pytest.raises(RuntimeError, match='live reader windows'):
            resolver.close()
        return result

    def exit(window, *args):
        try:
            return real_exit(window, *args)
        finally:
            assert not window._fds
            exited.append(window)

    def open(window, key):
        fd, serving = real_open(window, key)
        assert os.fstat(fd).st_size == window.entries[key]['object_bytes']
        opened.append(window)
        return fd, serving

    def close(window, fd):
        marker = (threading.get_ident(), fd)
        start = len(closed)
        real_close(window, fd)
        assert marker in closed[start:] and fd not in window._fds

    def syscall_close(fd):
        result = real_os_close(fd)
        closed.append((threading.get_ident(), fd))
        return result

    monkeypatch.setattr(LeaseWindow, '__enter__', enter)
    monkeypatch.setattr(LeaseWindow, '__exit__', exit)
    monkeypatch.setattr(LeaseWindow, 'open', open)
    monkeypatch.setattr(LeaseWindow, 'close_fd', close)
    monkeypatch.setattr(os, 'close', syscall_close)
    return entered, exited, opened


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
    cache, paths, expected, resolver, total = _strict_window(
        tmp_path, monkeypatch)
    digests = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    entered, exited, opened = _observe_windows(monkeypatch, resolver)
    _deny_pool_opens(monkeypatch, tmp_path / 'pool', names=[p.name for p in paths.values()])
    with _open_window(cache, paths, total, workers, release_file_pages=False) as receipt:
        # The window's loads are done and the pin is gone before the consumer
        # runs: a pin protects the reads, not the consumer's compute.
        assert resolver._windows == 0
        loaded = _loaded(cache, paths)
        assert receipt["resident_bytes"] <= receipt["budget_bytes"]
    # One acquire and one release for the window, not one per file ...
    assert len(entered) == 1 and entered == exited
    assert opened == [entered[0]] * len(paths)
    for key, (tensor, file_receipt, _identity) in loaded.items():
        assert torch.equal(tensor.view(torch.uint8), expected[key].view(torch.uint8))
        assert file_receipt["serving_tier"] == "stage"
        assert file_receipt["sha256"] == digests[key]
    report = resolver.report()
    assert report["bytes_from_pool"] == 0
    assert report["bytes_from_stage"] == sum(p.stat().st_size for p in paths.values())
    assert all(record.get("pin_id") for record in report["serving_tiers"])
    assert all(isinstance(value, str) for value in cache.weights.values())


def test_an_ungrouped_window_reads_each_render_under_its_own_lease(
        tmp_path, monkeypatch, workers):
    """An absent optional grouping still uses real per-entry windows."""
    cache, paths, expected, resolver, total = _strict_window(
        tmp_path, monkeypatch)
    monkeypatch.setattr(io_engine, 'pin_group', lambda *_args, **_kwargs: None)
    entered, exited, opened = _observe_windows(monkeypatch, resolver)
    with _open_window(cache, paths, total, workers):
        loaded = _loaded(cache, paths)
    assert len(entered) == len(exited) == len(opened) == len(paths)
    assert set(entered) == set(exited) == set(opened)
    assert resolver._windows == 0
    for key, (tensor, file_receipt, _identity) in loaded.items():
        assert torch.equal(tensor.view(torch.uint8), expected[key].view(torch.uint8))
        assert file_receipt["serving_tier"] == "stage"


@pytest.mark.parametrize("failure_point", ["allocation", "serving_record"])
def test_failed_read_closes_its_descriptor_without_releasing_group_pin(
        tmp_path, monkeypatch, failure_point):
    from prismaquant.staged_lease import LeaseWindow

    _cache, paths, _expected, resolver, _total = _strict_window(
        tmp_path, monkeypatch, count=2)
    members = [(path, hashlib.sha256(path.read_bytes()).hexdigest())
               for path in paths.values()]
    live_leases = []
    descriptors = []
    original_open = LeaseWindow.open

    def record_open(self, key):
        fd, serving = original_open(self, key)
        descriptors.append(fd)
        return fd, serving

    def refuse_setup(*args, **kwargs):
        assert resolver._windows == 1
        raise MemoryError("read setup failed")

    monkeypatch.setattr(LeaseWindow, "open", record_open)
    if failure_point == "allocation":
        monkeypatch.setattr(io_engine.SealedBuffer, "__init__", refuse_setup)
    else:
        monkeypatch.setattr(resolver, "record_serving_tier", refuse_setup)
    try:
        leases = io_engine.pin_group(members, live_leases)
        assert leases is not None
        path = members[0][0]
        with pytest.raises(MemoryError, match="read setup failed"):
            io_engine.read_file(
                path, path.stat().st_size, staged=leases[path][2],
                lease=leases[path], sealed=True)
        assert len(descriptors) == 1
        with pytest.raises(OSError):
            os.fstat(descriptors[0])
        assert resolver._windows == 1 and not live_leases[0]._fds
        with pytest.raises(RuntimeError, match='live reader windows'):
            resolver.close()
    finally:
        io_engine.unpin(live_leases)
    assert resolver._windows == 0
    assert resolver.report()["bytes_from_pool"] == 0


def test_a_bad_staged_copy_fails_its_window_by_name_and_releases_the_pin(
        tmp_path, monkeypatch, workers):
    """A copy that no longer matches its pin refuses; nothing falls back."""
    from prismaquant.staged_tier_policy import TierPolicyRefused

    from prismaquant.staged_lease import LeaseWindow
    cache, paths, _expected, resolver, total = _strict_window(
        tmp_path, monkeypatch)
    victim = list(paths.values())[3]
    staged = Path(resolver.staged_read(victim)['stage_path'])
    raw = bytearray(staged.read_bytes())
    raw[-40] ^= 0xFF
    real_open = LeaseWindow.open
    corrupted = []

    def open_then_damage(window, key):
        # Damage after the acquire proved the copy, so the check that
        # refuses is the read's own, on the bytes it hashed.
        fd, serving = real_open(window, key)
        if window.entries[key]['declared_path'] == str(victim) and not corrupted:
            staged.chmod(0o600)
            staged.write_bytes(bytes(raw))
            corrupted.append(key)
        return fd, serving

    monkeypatch.setattr(LeaseWindow, "open", open_then_damage)
    with pytest.raises((TierPolicyRefused, RuntimeError)):
        with _open_window(cache, paths, total, workers):
            pass
    assert corrupted
    assert resolver._windows == 0
    assert resolver.report()["bytes_from_pool"] == 0
    assert getattr(cache, "_resident_window_files", None) is None
    assert all(isinstance(value, str) for value in cache.weights.values())


# --------------------------------------------------------------------------
# One archive parse per file
# --------------------------------------------------------------------------

def test_each_render_archive_is_parsed_once_on_the_bytes_it_loads(
        tmp_path, monkeypatch, workers):
    cache, paths, _expected, _resolver, total = _strict_window(
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
    cache, paths, _expected, _resolver, total = _strict_window(
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
    cache, paths, _expected, _resolver, total = _strict_window(
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
    cache, paths, _expected, _resolver, total = _strict_window(
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
    arms = {}
    for arm in ("per_file", "window"):
        with monkeypatch.context() as patch:
            cache, paths, expected, resolver, total = _strict_window(
                tmp_path / arm, patch, count=3, shape=(2048, 4096))
            if arm == "per_file":
                patch.setattr(io_engine, 'pin_group', lambda *_args, **_kwargs: None)
            with _open_window(cache, paths, total, workers, render_identities=True):
                loaded = _loaded(cache, paths)
            assert resolver._windows == 0
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


def test_the_window_readiness_covers_every_exact_declared_span(
        tmp_path, monkeypatch):
    from prismaquant.joint_cost_quantum import prepare_retained_window_read

    _cache, paths, _expected, resolver, _total = _strict_window(
        tmp_path, monkeypatch)
    record = {"executable_readset": {"prepared_input": {"windows": [{
        "window_index": 3,
        "entries": [{"path": str(path), "offset": 0, "bytes": path.stat().st_size}
                    for path in paths.values()]}]}}}
    progress = _Progress()
    assert prepare_retained_window_read(3, record=record, progress=progress) == "ready"
    for path in paths.values():
        entry = resolver.staged_range(path, 0, path.stat().st_size)
        assert entry['_resolver'] is resolver and entry['declared_path'] == str(path)
    assert progress.phases


def test_the_window_readiness_refuses_one_undeclared_entry(
        tmp_path, monkeypatch):
    """Declared siblings cannot grant authority to a foreign span."""
    from prismaquant.joint_cost_quantum import prepare_retained_window_read

    _cache, paths, _expected, resolver, _total = _strict_window(
        tmp_path, monkeypatch)
    record = {"executable_readset": {"prepared_input": {"windows": [{
        "window_index": 0,
        "entries": [{"path": str(path), "offset": 0, "bytes": path.stat().st_size}
                    for path in paths.values()]}]}}}
    assert prepare_retained_window_read(0, record=record, progress=_Progress()) == "ready"
    entry = record['executable_readset']['prepared_input']['windows'][0]['entries'][-1]
    entry['path'] = str(tmp_path / 'never-declared.pt')
    assert prepare_retained_window_read(0, record=record, progress=_Progress()) == 'unready-undeclared'
    assert resolver._windows == 0 and resolver.report()['bytes_from_pool'] == 0


def test_an_active_group_refusal_is_not_hidden_by_single_entry_reads(tmp_path, monkeypatch, workers):
    from prismaquant import staged_lease
    cache, paths, _expected, resolver, total = _strict_window(tmp_path, monkeypatch)

    def refuse(*_args, **_kwargs):
        raise staged_lease.LeaseRefused('unpublished: forced', kind='availability')

    monkeypatch.setattr(staged_lease, 'acquire_entries_window', refuse)
    with pytest.raises(staged_lease.LeaseRefused, match='unpublished: forced'):
        with _open_window(cache, paths, total, workers):
            pytest.fail('refused group became readable')
    assert resolver._windows == 0
    assert resolver.report()['bytes_from_stage'] == resolver.report()['bytes_from_pool'] == 0
    assert all(isinstance(value, str) for value in cache.weights.values())
