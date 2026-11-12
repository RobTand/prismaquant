"""Native policy, grouped exact reads and descriptor lifetime obligations."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading

import pytest
import torch

from prismaquant import perturbed_x_cache as pxc
from prismaquant import joint_adjoint_checkpoints as checkpoints
from prismaquant.staged_lease import LeaseRefused, LeaseWindow, acquire_entry_window
from prismaquant.staged_manifest import load_staged_manifest
from prismaquant.staged_tier_policy import (
    DEFAULT_ALLOWED_TIERS, TierPolicyRefused, activate_staged_tier_policy,
    active_policy, deactivate_staged_tier_policy_for_tests,
    parse_allowed_tiers, staged_tier_policy_test_context,
)
from staged_native_fixtures import bind, native_staged_state, publish_inputs
from test_native_strict_app_consumers import _checkpoint
from test_native_strict_reader_contract import _forbid_source

pytestmark = pytest.mark.usefixtures('native_staged_state')

def test_parse_allowed_tiers_grammar():
    assert parse_allowed_tiers("ssd") == frozenset({"ssd"})
    assert parse_allowed_tiers(" SSD ") == frozenset({"ssd"})
    assert parse_allowed_tiers(DEFAULT_ALLOWED_TIERS) == frozenset({"ssd"})
    for bad in ("", "ram", "ram,ssd", "pool", "hdd", "ssd,hdd", "nvme", 42, None):
        with pytest.raises(ValueError):
            parse_allowed_tiers(bad)


def test_policy_context_lifetime_is_explicit_and_thread_global():
    assert active_policy() is None
    seen = {}
    with staged_tier_policy_test_context("ssd") as allowed:
        assert allowed == frozenset({"ssd"})
        worker = threading.Thread(
            target=lambda: seen.setdefault("active", active_policy()))
        worker.start()
        worker.join()
        assert seen["active"] == frozenset({"ssd"})
        with pytest.raises(TierPolicyRefused):
            raise TierPolicyRefused("sentinel")
    assert active_policy() is None
    assert seen["active"] == frozenset({"ssd"})
    with pytest.raises(RuntimeError, match="boom"):
        with staged_tier_policy_test_context("ssd"):
            raise RuntimeError("boom")
    assert active_policy() is None


def test_nested_scope_restores_the_prior_verdict():
    with staged_tier_policy_test_context("ssd"):
        with staged_tier_policy_test_context("ssd"):
            assert active_policy() == frozenset({"ssd"})
        assert active_policy() == frozenset({"ssd"})
    assert active_policy() is None
    activate_staged_tier_policy("ssd")
    try:
        with staged_tier_policy_test_context("ssd"):
            assert active_policy() == frozenset({"ssd"})
        assert active_policy() == frozenset({"ssd"})
    finally:
        deactivate_staged_tier_policy_for_tests()
    assert active_policy() is None


def test_overlapping_scopes_on_threads_refuse():
    with staged_tier_policy_test_context("ssd"):
        outcome = {}

        def enter_elsewhere():
            try:
                with staged_tier_policy_test_context("ssd"):
                    outcome["entered"] = True
            except RuntimeError as exc:
                outcome["refused"] = str(exc)

        worker = threading.Thread(target=enter_elsewhere)
        worker.start()
        worker.join()
        assert "entered" not in outcome
        assert "unsupported" in outcome["refused"]
        assert active_policy() == frozenset({"ssd"})
    assert active_policy() is None


def test_activation_needs_an_explicit_sealed_value():
    with pytest.raises(TypeError):
        activate_staged_tier_policy()

def test_parallel_reads_raise_the_first_failure_in_window_order():
    """The lowest-indexed failure is raised even when a later one fails first."""
    from prismaquant.perturbed_x_cache import _run_in_order
    pool = ThreadPoolExecutor(max_workers=2)
    started, released = [], threading.Event()

    def ok(index):
        started.append(index)
        return index

    def slow_failure():
        started.append(1)
        assert released.wait(10)
        raise ValueError("entry 1")

    def fast_failure():
        started.append(2)
        released.set()
        raise ValueError("entry 2")

    try:
        with pytest.raises(ValueError, match="entry 1"):
            _run_in_order(pool, [lambda: ok(0), slow_failure, fast_failure])
        assert _run_in_order(pool, [lambda i=i: ok(i) for i in range(6)]) == list(range(6))
        assert _run_in_order(None, [lambda: ok(7)]) == [7]
    finally:
        pool.shutdown(wait=True)


class _StartAtSubmitPool:
    """A pool that starts each call the moment it is submitted.

    The worst case for cancelling after a failure: every call has been
    picked up before the caller can see the first one fail.
    """

    def submit(self, fn, *args):
        from concurrent.futures import Future
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def test_calls_after_a_failure_that_have_not_started_never_start():
    from concurrent.futures import ThreadPoolExecutor
    from prismaquant.perturbed_x_cache import _run_in_order
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        for pool in (_StartAtSubmitPool(), worker):
            ran = []

            def fail():
                ran.append("fail")
                raise ValueError("first")

            with pytest.raises(ValueError, match="first"):
                _run_in_order(pool, [fail] + [lambda i=i: ran.append(i) for i in range(4)])
            assert ran == ["fail"], (pool, ran)
    finally:
        worker.shutdown(wait=True)


def test_calls_before_a_later_failure_still_run_and_raise_first():
    from prismaquant.perturbed_x_cache import _run_in_order
    ran = []

    def fail(index):
        ran.append(index)
        raise ValueError(f"entry {index}")

    # Calls 0 and 2 fail; the serial loop raises entry 0 and never starts 1.
    with pytest.raises(ValueError, match="entry 0"):
        _run_in_order(_StartAtSubmitPool(), [lambda: fail(0), lambda: ran.append(1),
                                             lambda: fail(2)])
    assert ran == [0], ran


def test_concurrent_readers_borrow_distinct_buffers_and_release_frees_them():
    from prismaquant.perturbed_x_cache import EntryReadScratch
    scratch = EntryReadScratch()
    with scratch.lend() as first:
        assert first is scratch
        with scratch.lend() as second:
            assert second is not first
            assert first.buffer(16) is not second.buffer(16)
    with scratch.lend() as again:
        assert again is scratch or again is second
    assert len(scratch._extra) == 1
    scratch.release()
    assert len(scratch.buffer(0)) == 0 and scratch._extra == []


def test_exact_entry_windows_follow_the_budget():
    from prismaquant.joint_adjoint_checkpoints import exact_entry_windows
    records = [{"name": f"e{i}", "tensor_bytes": 10} for i in range(5)]
    windows, ahead = exact_entry_windows(records, max_resident_bytes=None)
    assert [len(w) for w in windows] == [1] * 5 and ahead is False
    windows, ahead = exact_entry_windows(records, max_resident_bytes=40)
    assert [len(w) for w in windows] == [2, 2, 1] and ahead is True
    windows, ahead = exact_entry_windows(records, max_resident_bytes=15)
    assert [len(w) for w in windows] == [1] * 5 and ahead is False
    with pytest.raises(RuntimeError, match="exceeds the stream's resident budget"):
        exact_entry_windows(records, max_resident_bytes=9)
    with pytest.raises(ValueError):
        exact_entry_windows(records, max_resident_bytes=0)
    assert exact_entry_windows([], max_resident_bytes=40) == ([], False)


def _native_entries(tmp_path, monkeypatch):
    _, record, expected, _, _, files = _checkpoint(tmp_path)
    resolver = bind(monkeypatch, publish_inputs(tmp_path, files))
    refs = [checkpoints.reference_from_record(row) for row in record['activation_entries']]
    opened = [_forbid_source(monkeypatch, path) for path in files]
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '2')
    return record, refs, expected, resolver, opened


def test_grouped_and_single_exact_reads_keep_independent_known_tensor_bytes(tmp_path, monkeypatch):
    record, refs, expected, resolver, opened = _native_entries(tmp_path, monkeypatch)
    fds, windows = [], []
    original = LeaseWindow.open
    def observe(window, key):
        fd, serving = original(window, key)
        fds.append(fd)
        windows.append(window)
        assert resolver._windows == 1
        return fd, serving
    monkeypatch.setattr(LeaseWindow, 'open', observe)
    held = [0]
    def charge(delta):
        held[0] += delta
        assert 0 <= held[0] <= 192
    session = checkpoints.checkpoint_entry_session(record)
    with staged_tier_policy_test_context('ssd'):
        with pxc.prefetch_exact_activation_cache_entries(refs, max_tensor_bytes=192,
                expected_session=session, residency_check=charge) as window:
            assert held == [192] and resolver._windows == 0
            tensors = [window.get(ref) for ref in refs]
            for tensor, coordinates in zip(tensors, sorted(expected)):
                assert torch.equal(tensor.view(torch.uint8), expected[coordinates].view(torch.uint8))
        assert held == [0]
        with pytest.raises(RuntimeError, match='not ready'):
            window.get(refs[0])
        assert len({id(window) for window in windows}) == 1
        assert all(not window._fds for window in windows)
        windows.clear()
        for ref, want in zip(refs, tensors):
            with pxc.prefetch_exact_activation_cache_entries([ref], max_tensor_bytes=48,
                    expected_session=session) as single:
                assert torch.equal(single.get(ref).view(torch.uint8), want.view(torch.uint8))
        assert len({id(window) for window in windows}) == 4
    assert resolver._windows == 0 and resolver.report()['bytes_from_pool'] == 0
    assert resolver.report()['bytes_from_stage'] == 2 * sum(ref.file_bytes for ref in refs)
    assert all(not calls for calls in opened)
    assert len(fds) == 8 and all(not window._fds for window in windows)
    resolver.close()
    assert resolver._fds == {}
    for tensor, coordinates in zip(tensors, sorted(expected)):
        assert torch.equal(tensor.view(torch.uint8), expected[coordinates].view(torch.uint8))


def test_parallel_failed_group_joins_started_reads_before_releasing_window(tmp_path, monkeypatch):
    record, refs, _, resolver, opened = _native_entries(tmp_path, monkeypatch)
    started, failed = threading.Event(), threading.Event()
    original_read, original_exit = pxc._read_exact_entry, LeaseWindow.__exit__
    active, exits = set(), []
    lock = threading.Lock()
    def reading(ref, **kwargs):
        with lock:
            active.add(ref.name)
        try:
            if ref is refs[0]:
                started.set()
                assert failed.wait(10)
                return original_read(ref, **kwargs)
            if ref is refs[1]:
                assert started.wait(10)
                failed.set()
                raise RuntimeError('owned exact read failure')
            return original_read(ref, **kwargs)
        finally:
            with lock:
                active.remove(ref.name)
    def exiting(window, *args):
        with lock:
            exits.append(set(active))
        return original_exit(window, *args)
    monkeypatch.setattr(pxc, '_read_exact_entry', reading)
    monkeypatch.setattr(LeaseWindow, '__exit__', exiting)
    held, exposed = [0], []
    def charge(delta):
        held[0] += delta
    with staged_tier_policy_test_context('ssd'):
        with pytest.raises(RuntimeError, match='owned exact read failure'):
            with pxc.prefetch_exact_activation_cache_entries(refs, max_tensor_bytes=192,
                    expected_session=checkpoints.checkpoint_entry_session(record),
                    residency_check=charge) as window:
                exposed.append(window.get(refs[0]))
    assert failed.is_set() and started.is_set() and not active
    assert exposed == [] and exits == [set()] and held == [0]
    assert resolver._windows == 0 and resolver.report()['bytes_from_pool'] == 0
    assert all(not calls for calls in opened)


def _window(tmp_path, monkeypatch):
    source = tmp_path / 'known-window'
    payload = b'0123456789-window-owned-by-test'
    source.write_bytes(payload)
    binding = publish_inputs(tmp_path, [source])
    resolver = bind(monkeypatch, binding)
    entry = resolver.staged_read(source)
    return resolver, entry, source, payload


@pytest.mark.parametrize('operation', ['close', 'open', 'resolver'])
def test_forked_reader_refuses_without_closing_parent_owned_descriptor(tmp_path, monkeypatch, operation):
    resolver, entry, source, payload = _window(tmp_path, monkeypatch)
    with staged_tier_policy_test_context('ssd'):
        window, key = acquire_entry_window(resolver, source, entry)
        with window:
            fd, _ = window.open(key)
            read, write = os.pipe()
            child = os.fork()
            if child == 0:
                os.close(read)
                try:
                    if operation == 'close':
                        window.close_fd(fd)
                    elif operation == 'open':
                        window.open(key)
                    else:
                        resolver.close()
                except RuntimeError as exc:
                    os.write(write, str(exc).encode())
                    os._exit(0)
                os._exit(1)
            os.close(write)
            try:
                _, status = os.waitpid(child, 0)
                assert os.waitstatus_to_exitcode(status) == 0
                assert b'forked child' in os.read(read, 4096)
            finally:
                os.close(read)
            assert os.pread(fd, len(payload), 0) == payload
            assert resolver._windows == 1
        assert resolver._windows == 0


def test_closed_window_does_not_adopt_reused_descriptor_of_another_reader(tmp_path, monkeypatch):
    resolver, entry, source, payload = _window(tmp_path, monkeypatch)
    with staged_tier_policy_test_context('ssd'):
        first, key = acquire_entry_window(resolver, source, entry)
        second, other_key = acquire_entry_window(resolver, source, entry)
        first.__enter__()
        second.__enter__()
        old, _ = first.open(key)
        first.close_fd(old)
        real_open = resolver.open_entry
        def reuse(entry):
            fresh = real_open(entry)
            if fresh != old:
                os.dup2(fresh, old)
                os.close(fresh)
            return old
        monkeypatch.setattr(resolver, 'open_entry', reuse)
        new, _ = second.open(other_key)
        try:
            assert new == old
            first.close_fd(old)
            first.__exit__(None, None, None)
            assert os.pread(new, len(payload), 0) == payload
            assert new in second._fds and resolver._windows == 1
        finally:
            second.__exit__(None, None, None)
        with pytest.raises(OSError):
            os.fstat(new)
        assert resolver._windows == 0


def test_two_windows_have_independent_positions_for_the_same_staged_object(tmp_path, monkeypatch):
    resolver, entry, source, payload = _window(tmp_path, monkeypatch)
    with staged_tier_policy_test_context('ssd'):
        first, key = acquire_entry_window(resolver, source, entry)
        second, other = acquire_entry_window(resolver, source, entry)
        with first, second:
            a, _ = first.open(key)
            b, _ = second.open(other)
            assert a != b
            assert os.read(a, 3) == payload[:3]
            assert os.read(b, len(payload) + 1) == payload
            assert os.read(a, len(payload)) == payload[3:]
            assert resolver._windows == 2
        assert resolver._windows == 0


@pytest.mark.parametrize('fault', ['changed', 'replacement'])
def test_live_descriptor_rejects_mutation_or_same_bytes_path_replacement(tmp_path, monkeypatch, fault):
    resolver, entry, source, payload = _window(tmp_path, monkeypatch)
    with staged_tier_policy_test_context('ssd'):
        window, key = acquire_entry_window(resolver, source, entry)
        with window:
            fd, _ = window.open(key)
            obj = Path(entry['stage_path'])
            if fault == 'changed':
                obj.chmod(0o600)
                obj.write_bytes(b'X' + payload[1:])
            else:
                replacement = obj.with_name('replacement')
                replacement.write_bytes(payload)
                replacement.replace(obj)
            try:
                with pytest.raises(RuntimeError, match='changed'):
                    window.check_fd(fd)
            finally:
                window.close_fd(fd)
        assert resolver._windows == 0


def test_reentry_and_double_exit_refuse_without_releasing_another_window(tmp_path, monkeypatch):
    resolver, entry, source, _ = _window(tmp_path, monkeypatch)
    with staged_tier_policy_test_context('ssd'):
        window, _ = acquire_entry_window(resolver, source, entry)
        with window:
            with pytest.raises(RuntimeError, match='re-enter'):
                window.__enter__()
            assert resolver._windows == 1
        with pytest.raises(RuntimeError, match='re-exit'):
            window.__exit__(None, None, None)
        assert resolver._windows == 0


