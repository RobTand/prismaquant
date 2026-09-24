"""Finite research windows reuse PWC loads and never fault on consumption."""
import pickle
import os
import weakref
import zipfile

import pytest
import torch

from prismaquant.production_weight_cache import ProductionWeightCache
from test_pwc_file_load_receipts import make_cache


#: The window loader count these tests were written against.
FIXTURE_WINDOW_WORKERS = 2


@pytest.fixture
def workers():
    """Window loaders that fit the CPUs this process was assigned.

    ``ProductionWeightCache._window_limits`` and ``prefetch`` refuse more
    loaders than ``len(os.sched_getaffinity(0))``, and that refusal is correct:
    a shard admitted with one CPU may not open two loaders. A test that
    hardcodes two asserts on the shard's width, not on the code, so these
    tests ask for two or for what the shard was given (PQ #1067, as PQ #888
    did for the operator-window fixture). A resident window's width follows
    its byte budgets, not its loader count (#693), so its splits hold at
    either count. A retained window's load quanta are bounded by both, so
    that test derives its quanta from this count.
    """
    return max(1, min(FIXTURE_WINDOW_WORKERS, len(os.sched_getaffinity(0))))


def _retained_cache(tmp_path, count):
    """``make_cache`` with an LRU that holds ``count`` whole files.

    A retained window charges each file its length, the bound the sealed
    plan charges (PQ #1210), so its budgets are counted in file lengths; the
    fixture's tensors store 32 bytes each in equal-length archives.
    """
    cache, paths, tensors = make_cache(tmp_path, count, budget=1)
    each = max(path.stat().st_size for path in paths.values())
    assert all(path.stat().st_size == each for path in paths.values())
    cache.enable_lru(count * each)
    return cache, paths, tensors, each


def test_plan_resolves_aliases_and_bounds_existing_prefetch(tmp_path, monkeypatch, workers):
    cache, paths, expected = make_cache(tmp_path, 5, budget=100000)
    keys = tuple(paths)
    bound = 2 * max(path.stat().st_size for path in paths.values())
    aliases = [(name + '.weight', fmt) for name, fmt in keys]
    windows = cache.plan_resident_windows(aliases + [aliases[0]],
        max_resident_bytes=bound, max_workers=workers)
    assert windows == (keys[:2], keys[2:4], keys[4:])
    original = cache.prefetch
    calls = []
    def bounded(keys, max_workers):
        calls.append(tuple(keys))
        assert max_workers == workers
        return original(keys, max_workers=max_workers)
    monkeypatch.setattr(cache, 'prefetch', bounded)
    refs = []
    for window in windows:
        with cache.resident_window(window, max_resident_bytes=bound, max_workers=workers) as receipt:
            assert receipt['keys'] == window and receipt['loaded'] == len(window)
            assert receipt['resident_bytes'] == 32 * len(window)
            for key in window:
                value = cache.get_resident(*key)
                torch.testing.assert_close(value, expected[key])
                refs.append(weakref.ref(value))
                del value
        assert all(ref() is None for ref in refs)
        assert all(isinstance(cache.weights[key], str) for key in window)
    assert calls == list(windows)


def test_resident_lookup_refuses_missing_or_evicted_without_load(tmp_path, monkeypatch):
    cache, paths, _ = make_cache(tmp_path, 2, budget=32)
    a, b = paths
    cache.get(*a)
    cache.get(*b)
    monkeypatch.setattr(cache, '_load_file_tensor', lambda *args: pytest.fail('hidden load'))
    with pytest.raises(RuntimeError, match='resident'):
        cache.get_resident(*a)
    with pytest.raises(RuntimeError, match='missing'):
        cache.get_resident('unknown', a[1])
    assert cache.get_resident(*b) is cache.weights[b]


def test_full_backing_storage_and_aliases_are_accounted(workers):
    pool = torch.zeros(100)
    keys = [('a', 'FP8'), ('b', 'FP8')]
    cache = ProductionWeightCache(dict(zip(keys, (pool[:2], pool[2:4]))), {})
    with pytest.raises(RuntimeError, match='budget'):
        cache.plan_resident_windows(keys, max_resident_bytes=399, max_workers=workers)
    assert cache.plan_resident_windows(keys, max_resident_bytes=400, max_workers=workers) == (tuple(keys),)
    with cache.resident_window(keys, max_resident_bytes=400, max_workers=workers) as receipt:
        assert receipt['resident_bytes'] == 400
    assert all(isinstance(cache.weights[key], torch.Tensor) for key in keys)


def test_window_accounting_does_not_walk_entire_file_roster_each_time(tmp_path):
    class CountedWeights(dict):
        scans = 0

        def values(self):
            self.scans += 1
            return super().values()

    keys = [('first', 'FP8'), ('second', 'FP8')]
    weights = CountedWeights({(f'absent-{index}', 'FP8'): 'not-read'
                              for index in range(10000)})
    weights.update({key: torch.zeros(4) for key in keys})
    cache = ProductionWeightCache(weights, {})
    for key in keys:
        with cache.resident_window([key], max_resident_bytes=32, max_workers=1):
            assert cache.get_resident(*key) is weights[key]
    assert weights.scans <= 1, 'full roster was scanned at every window boundary'


def test_window_accounting_tracks_direct_mutations_and_rebound_weights():
    key = ('resident', 'FP8')
    cache = ProductionWeightCache({key: torch.zeros(4)}, {})
    assert cache.plan_resident_windows([key], max_resident_bytes=16, max_workers=1)
    pool = torch.zeros(100)
    cache.weights['alias', 'FP8'] = pool[:1]
    cache.weights.update({('alias2', 'FP8'): pool[1:2]})
    with pytest.raises(RuntimeError, match='budget'):
        cache.plan_resident_windows([key], max_resident_bytes=399, max_workers=1)
    del cache.weights['alias', 'FP8']
    cache.weights.pop(('alias2', 'FP8'))
    assert cache.plan_resident_windows([key], max_resident_bytes=16, max_workers=1)
    cache.weights = {key: pool[:1]}
    with pytest.raises(RuntimeError, match='budget'):
        cache.plan_resident_windows([key], max_resident_bytes=399, max_workers=1)


def test_window_accounting_rechecks_rebound_view_storage_and_pickle():
    key = ('resident', 'FP8')
    tensor = torch.zeros(4)
    cache = ProductionWeightCache({key: tensor}, {})
    assert cache.plan_resident_windows([key], max_resident_bytes=16, max_workers=1)
    tensor.set_(torch.zeros(100)[:1])
    with pytest.raises(RuntimeError, match='budget'):
        cache.plan_resident_windows([key], max_resident_bytes=399, max_workers=1)
    # Resident indexing is local bookkeeping; a saved cache has plain weights
    # and builds a fresh index from its actual restored storage when needed.
    clone = pickle.loads(pickle.dumps(cache))
    assert type(clone.weights) is dict
    with pytest.raises(RuntimeError, match='budget'):
        clone.plan_resident_windows([key], max_resident_bytes=399, max_workers=1)


def test_invalid_roster_or_oversize_refuses_before_loading(tmp_path, monkeypatch):
    cache, paths, _ = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    monkeypatch.setattr(cache, '_load_file_tensor', lambda *args: pytest.fail('hidden load'))
    with pytest.raises(RuntimeError, match='budget'):
        cache.plan_resident_windows([key], max_resident_bytes=path.stat().st_size - 1, max_workers=1)
    with pytest.raises(RuntimeError, match='missing'):
        cache.plan_resident_windows([key, ('missing', key[1])], max_resident_bytes=10000, max_workers=1)
    with pytest.raises((TypeError, ValueError), match='finite|sequence'):
        cache.plan_resident_windows(iter([key]), max_resident_bytes=10000, max_workers=1)


def test_selected_release_preserves_unrelated_residents_and_receipts(tmp_path, workers):
    cache, paths, _ = make_cache(tmp_path, 2)
    a, b = paths
    cache.enable_file_load_receipts(max_file_bytes=10000)
    cache.prefetch([a, b], max_workers=workers)
    retained = cache.get_resident(*b)
    assert cache.release_resident_tensors([a]) == 1
    assert isinstance(cache.weights[a], str)
    assert cache.get_resident(*b) is retained
    cache.file_load_receipt(b, retained)
    assert cache._lru_bytes == 32 and cache._lru_order == [b]


def test_window_releases_on_consumer_failure_and_receipt_checks_mutation(tmp_path):
    cache, paths, _ = make_cache(tmp_path)
    key = next(iter(paths))
    cache.enable_file_load_receipts(max_file_bytes=10000)
    with pytest.raises(RuntimeError, match='changed'):
        with cache.resident_window([key], max_resident_bytes=10000, max_workers=1):
            value = cache.get_resident(*key)
            value.add_(1)
            cache.get_resident(*key)
    assert isinstance(cache.weights[key], str)
    assert not cache._file_load_receipts


def test_lru_eviction_cannot_yield_partial_window(tmp_path, workers):
    cache, paths, _ = make_cache(tmp_path, 2, budget=32)
    with pytest.raises(RuntimeError, match='resident|LRU|budget'):
        with cache.resident_window(tuple(paths), max_resident_bytes=10000, max_workers=workers):
            pytest.fail('partial resident window exposed')
    assert all(isinstance(value, str) for value in cache.weights.values())


def test_unrelated_resident_storage_is_in_the_budget_before_load(tmp_path, monkeypatch):
    cache, paths, _ = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    cache.weights[('unrelated', 'FP8')] = torch.zeros(100)[:1]
    monkeypatch.setattr(cache, '_load_file_tensor', lambda *args: pytest.fail('hidden load'))
    with pytest.raises(RuntimeError, match='budget'):
        with cache.resident_window([key], max_resident_bytes=path.stat().st_size + 399, max_workers=1):
            pytest.fail('unrelated backing storage was omitted')


def test_nested_window_refuses_without_releasing_outer_owners(tmp_path):
    cache, paths, _ = make_cache(tmp_path, 2)
    a, b = paths
    with cache.resident_window([a], max_resident_bytes=10000, max_workers=1):
        outer = cache.get_resident(*a)
        with pytest.raises(RuntimeError, match='nested'):
            with cache.resident_window([b], max_resident_bytes=10000, max_workers=1):
                pytest.fail('nested window')
        with pytest.raises(RuntimeError, match='outside'):
            cache.get(*b)
        assert cache.get_resident(*a) is outer
    assert all(isinstance(value, str) for value in cache.weights.values())


def test_existing_lru_owner_cannot_be_evicted_by_window(tmp_path, monkeypatch):
    cache, paths, _ = make_cache(tmp_path, 2, budget=32)
    a, b = paths
    old = cache.get(*a)
    monkeypatch.setattr(cache, '_load_file_tensor', lambda *args: pytest.fail('hidden load'))
    with pytest.raises(RuntimeError, match='LRU budget'):
        with cache.resident_window([b], max_resident_bytes=10000, max_workers=1):
            pytest.fail('unrelated LRU owner evicted')
    assert cache.get_resident(*a) is old


@pytest.mark.parametrize('kind', ['compressed', 'opaque', 'legacy', 'symlink'])
def test_unaccountable_disk_inputs_refuse_without_deserializing(tmp_path, monkeypatch, kind):
    cache, paths, _ = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    if kind == 'compressed':
        with zipfile.ZipFile(path) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in entries.items():
                archive.writestr(name, value)
    elif kind == 'opaque':
        cache.weights[key] = object()
    elif kind == 'legacy':
        torch.save(torch.zeros(4), path, _use_new_zipfile_serialization=False)
    else:
        target = path.with_suffix('.target'); path.rename(target); path.symlink_to(target)
    monkeypatch.setattr(cache, '_load_file_tensor', lambda *args: pytest.fail('hidden load'))
    with pytest.raises(RuntimeError, match='archive|unaccountable|regular'):
        cache.plan_resident_windows([key], max_resident_bytes=10000, max_workers=1)


@pytest.mark.parametrize('value', [torch.empty(1, device='meta'), torch.sparse_coo_tensor([[0]], [1.], (1,))])
def test_unaccountable_tensor_storages_refuse(value):
    cache = ProductionWeightCache({('a', 'FP8'): value}, {})
    with pytest.raises(RuntimeError, match='unaccountable'):
        cache.plan_resident_windows([('a', 'FP8')], max_resident_bytes=10000, max_workers=1)


def test_serialized_buffers_have_a_separate_aggregate_limit(tmp_path, monkeypatch, workers):
    # The cap is enforced where the width is now decided: the planner splits a
    # key set its serialized budget cannot read at once, and the one-quantum
    # window then refuses that key set rather than reading it (#693).
    cache, paths, _ = make_cache(tmp_path, 2)
    keys = tuple(paths)
    size = sum(path.stat().st_size for path in paths.values())
    monkeypatch.setattr(cache, '_load_file_tensor', lambda *args: pytest.fail('hidden load'))
    assert cache.plan_resident_windows(keys, max_resident_bytes=10000, max_workers=workers,
                                       max_load_buffer_bytes=size - 1) == (keys[:1], keys[1:])
    assert cache.plan_resident_windows(keys, max_resident_bytes=10000, max_workers=workers,
                                       max_load_buffer_bytes=size) == (keys,)
    with pytest.raises(RuntimeError, match='one nonempty planned quantum'):
        with cache.resident_window(keys, max_resident_bytes=10000, max_workers=workers,
                                   max_load_buffer_bytes=size - 1):
            pytest.fail('oversize buffers')
    with pytest.raises(RuntimeError, match='single serialized load buffer'):
        cache.plan_resident_windows(keys, max_resident_bytes=10000, max_workers=workers,
                                    max_load_buffer_bytes=size // 2 - 1)


def test_page_advice_uses_verified_load_stat_and_cleanup(tmp_path, monkeypatch):
    from prismaquant import perturbed_x_cache
    cache, paths, _ = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    seen = []
    def advice(candidate, *, expected_stat):
        value = cache.get_resident(*key)
        assert cache.file_load_receipt(key, value)['bytes'] == expected_stat.st_size
        assert str(path) == candidate
        seen.append(candidate)
    monkeypatch.setattr(perturbed_x_cache, 'release_activation_cache_file_pages', advice)
    with cache.resident_window([key], max_resident_bytes=10000, max_workers=1,
                               release_file_pages=True) as receipt:
        assert receipt['file_pages_advised'] == 1
    assert seen == [str(path)] and not cache._file_load_receipts


def test_window_load_failure_cleans_partial_prefetch_and_allows_fresh_context(tmp_path, monkeypatch, workers):
    cache, paths, _ = make_cache(tmp_path, 2)
    a, b = paths
    original = cache._validate_loaded_cb_pair_tensor
    def refused(key, tensor):
        if key == b:
            raise RuntimeError('synthetic integrity refusal')
        return original(key, tensor)
    monkeypatch.setattr(cache, '_validate_loaded_cb_pair_tensor', refused)
    with pytest.raises(RuntimeError, match='integrity'):
        with cache.resident_window(tuple(paths), max_resident_bytes=10000, max_workers=workers):
            pytest.fail('partial load exposed')
    assert all(isinstance(value, str) for value in cache.weights.values())
    assert not cache._file_load_receipts
    monkeypatch.setattr(cache, '_validate_loaded_cb_pair_tensor', original)
    with cache.resident_window([a], max_resident_bytes=10000, max_workers=1):
        assert isinstance(cache.get_resident(*a), torch.Tensor)


def test_resident_cb_lookup_preserves_both_existing_validators(monkeypatch):
    key = ('a', 'FP8_CB_K28')
    cache = ProductionWeightCache({key: torch.zeros(2, 2)}, {})
    calls = []
    monkeypatch.setattr('prismaquant.production_weight_cache._is_cb_format_name', lambda fmt: True)
    monkeypatch.setattr(cache, 'validate_cb_render_identity', lambda **kwargs: calls.append('identity'))
    monkeypatch.setattr(cache, '_validate_loaded_cb_pair_tensor', lambda *args: calls.append('tensor'))
    assert cache.get_resident(*key) is cache.weights[key]
    assert calls == ['identity', 'tensor']


def test_existing_disk_loaded_tensor_can_enter_without_enabling_receipts(tmp_path):
    cache, paths, _ = make_cache(tmp_path)
    key = next(iter(paths))
    tensor = cache.get(*key)
    assert not cache._file_load_receipts
    with cache.resident_window([key], max_resident_bytes=32, max_workers=1) as receipt:
        assert receipt['loaded'] == 0
        assert cache.get_resident(*key) is tensor
    assert isinstance(cache.weights[key], str)


def test_invalidated_window_receipt_cannot_be_bypassed_by_a_second_lookup(tmp_path):
    cache, paths, _ = make_cache(tmp_path)
    key = next(iter(paths))
    with cache.resident_window([key], max_resident_bytes=10000, max_workers=1):
        cache.get_resident(*key).add_(1)
        for _ in range(2):
            with pytest.raises(RuntimeError, match='receipt|changed'):
                cache.get_resident(*key)


def test_window_load_rejects_file_drift_after_preflight(tmp_path, monkeypatch):
    cache, paths, _ = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    original = cache.prefetch
    def altered(keys, max_workers):
        with path.open('ab') as stream:
            stream.write(b'changed')
        return original(keys, max_workers=max_workers)
    monkeypatch.setattr(cache, 'prefetch', altered)
    with pytest.raises(RuntimeError, match='bound|changed'):
        with cache.resident_window([key], max_resident_bytes=10000, max_workers=1):
            pytest.fail('changed file admitted')
    assert isinstance(cache.weights[key], str) and not cache._file_load_receipts


def test_selected_release_does_not_adopt_same_named_unverified_file(tmp_path):
    from prismaquant.production_weight_cache import _cache_weight_filename
    key = ('unit', 'FP8')
    tensor = torch.ones(4)
    torch.save(torch.zeros(4), tmp_path / _cache_weight_filename(*key))
    cache = ProductionWeightCache({key: tensor}, {}, cache_dir=str(tmp_path))
    assert cache.release_resident_tensors([key]) == 0
    assert cache.get_resident(*key) is tensor


def test_retained_window_keeps_more_keys_than_workers_for_repeated_passes(tmp_path, monkeypatch, workers):
    cache, paths, expected, file_size = _retained_cache(tmp_path, 5)
    keys = tuple(paths)
    # A resident window is still ONE quantum: it refuses a key set its budgets
    # cannot hold at once. What no longer bounds it is the loader count, so the
    # contrast with a retained lifetime is the budget, not the CPU count (#693).
    assert cache.plan_resident_windows(keys, max_resident_bytes=5 * file_size,
                                       max_workers=workers) == (keys,)
    with pytest.raises(RuntimeError, match='one nonempty planned quantum'):
        with cache.resident_window(keys, max_resident_bytes=2 * file_size, max_workers=workers):
            pytest.fail('window unexpectedly retained a key set over its budget')
    cache.enable_file_load_receipts(max_file_bytes=file_size)
    original_prefetch, original_load = cache.prefetch, cache._load_file_tensor
    quanta, reads = [], []
    def bounded_prefetch(selected, max_workers, **kwargs):
        quanta.append(tuple(selected))
        # A retained load quantum IS bounded by the loader count, and by the
        # serialized buffer budget: unlike a resident window (#693).
        assert len(selected) <= max_workers == workers
        assert sum(paths[key].stat().st_size for key in selected) <= 2 * file_size
        return original_prefetch(selected, max_workers=max_workers, **kwargs)
    def counted_load(value, key=None):
        reads.append(str(value))
        return original_load(value, key)
    monkeypatch.setattr(cache, 'prefetch', bounded_prefetch)
    monkeypatch.setattr(cache, '_load_file_tensor', counted_load)
    aliases = [(name + '.weight', fmt) for name, fmt in keys]
    planned = cache.plan_retained_window(aliases + [aliases[0]],
        max_resident_bytes=5 * file_size, max_workers=workers,
        max_load_buffer_bytes=2 * file_size)
    # Both bounds allow `workers` keys per quantum: two keys fit the buffer.
    assert planned == tuple(keys[start:start + workers]
                            for start in range(0, len(keys), workers))
    with cache.retained_window(aliases + [aliases[0]],
            max_resident_bytes=5 * file_size, max_workers=workers,
            max_load_buffer_bytes=2 * file_size) as receipt:
        assert receipt['keys'] == keys and receipt['loaded'] == len(keys)
        assert receipt['load_quanta'] == planned == tuple(quanta)
        assert receipt['resident_bytes'] == 5 * 32
        assert receipt['load_buffer_capacity_bytes'] <= 2 * file_size
        owners = {key: cache.get_resident(*key) for key in keys}
        for _ in range(4):
            for key in keys:
                assert cache.get_resident(*key) is owners[key]
                torch.testing.assert_close(owners[key], expected[key])
                assert cache.file_load_receipt(key, owners[key])['bytes'] == paths[key].stat().st_size
        del owners
    assert len(reads) == len(keys) and len(set(reads)) == len(keys)
    assert all(isinstance(cache.weights[key], str) for key in keys)
    assert not cache._file_load_receipts


def test_retained_window_preflights_all_keys_and_serialized_quanta(tmp_path, monkeypatch, workers):
    cache, paths, _, size = _retained_cache(tmp_path, 3)
    keys = tuple(paths)
    monkeypatch.setattr(cache, '_load_file_tensor', lambda *args: pytest.fail('preflight loaded a tensor'))
    with pytest.raises(RuntimeError, match='missing'):
        with cache.retained_window(keys + (('missing', keys[0][1]),),
                max_resident_bytes=3 * size, max_workers=workers,
                max_load_buffer_bytes=size):
            pytest.fail('incomplete roster admitted')
    with pytest.raises(RuntimeError, match='resident storage'):
        with cache.retained_window(keys, max_resident_bytes=3 * size - 1,
                max_workers=workers, max_load_buffer_bytes=size):
            pytest.fail('oversized roster admitted')
    with pytest.raises(RuntimeError, match='serialized'):
        with cache.retained_window(keys, max_resident_bytes=3 * size,
                max_workers=workers, max_load_buffer_bytes=size - 1):
            pytest.fail('oversized read admitted')
    assert cache.plan_retained_window(keys, max_resident_bytes=3 * size,
        max_workers=workers, max_load_buffer_bytes=size) == tuple((key,) for key in keys)


def test_retained_window_refuses_late_symlink_before_any_load(tmp_path, monkeypatch):
    cache, paths, _, size = _retained_cache(tmp_path, 3)
    keys = tuple(paths)
    path = paths[keys[-1]]
    target = path.with_suffix('.original')
    path.rename(target)
    path.symlink_to(target)
    monkeypatch.setattr(cache, '_load_file_tensor',
                        lambda *args: pytest.fail('first file loaded before late refusal'))
    with pytest.raises(RuntimeError, match='regular'):
        with cache.retained_window(keys, max_resident_bytes=3 * size,
                max_workers=1, max_load_buffer_bytes=10000):
            pytest.fail('unaccountable file admitted')


def test_retained_window_refuses_compressed_archive_before_deserializing_it(
        tmp_path, monkeypatch):
    """A compressed archive refuses at its own read, never deserialized.

    The window charges each file its length before the first load, which
    bounds any uncompressed archive's storage, and parses each archive once,
    on the bytes its loader read (PQ #1210). A compressed archive is not an
    accountable load, so it refuses there, before ``torch.load``, and the
    window releases the files it had already loaded.
    """
    cache, paths, _, _size = _retained_cache(tmp_path, 3)
    keys = tuple(paths)
    path = paths[keys[-1]]
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    cache.enable_lru(sum(p.stat().st_size for p in paths.values()))
    budget = sum(p.stat().st_size for p in paths.values())
    loads = []
    original = torch.load
    def counted_load(*args, **kwargs):
        loads.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', counted_load)
    with pytest.raises(RuntimeError, match='uncompressed Torch archive'):
        with cache.retained_window(keys, max_resident_bytes=budget,
                max_workers=1, max_load_buffer_bytes=10000):
            pytest.fail('unaccountable file admitted')
    # One loader, one key per quantum: the first two loaded, the third never
    # reached ``torch.load``.
    assert len(loads) == 2
    assert all(isinstance(cache.weights[key], str) for key in keys)
    assert cache._lru_bytes == 0 and cache._resident_window_files is None


def test_retained_window_rechecks_late_file_and_releases_failed_load(tmp_path, monkeypatch):
    cache, paths, _, size = _retained_cache(tmp_path, 3)
    keys = tuple(paths)
    cache.enable_file_load_receipts(max_file_bytes=size)
    original = cache.prefetch
    calls = 0
    def changed_late(selected, max_workers, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            before = paths[keys[1]].stat()
            os.utime(paths[keys[1]], ns=(before.st_atime_ns, before.st_mtime_ns + 1))
        return original(selected, max_workers=max_workers, **kwargs)
    monkeypatch.setattr(cache, 'prefetch', changed_late)
    with pytest.raises(RuntimeError, match='changed'):
        with cache.retained_window(keys, max_resident_bytes=3 * size,
                max_workers=1, max_load_buffer_bytes=size):
            pytest.fail('partially loaded roster exposed')
    assert all(isinstance(cache.weights[key], str) for key in keys)
    assert not cache._file_load_receipts
    assert cache._resident_window_files is None


def test_retained_window_keeps_unrelated_owners_and_refuses_nested(tmp_path):
    cache, paths, _, size = _retained_cache(tmp_path, 3)
    a, b, c = paths
    outside = cache.get(*a)
    # The unrelated owner's 32 stored bytes plus the window's two files.
    held = 32 + 2 * size
    with cache.retained_window((b, c), max_resident_bytes=held,
            max_workers=1, max_load_buffer_bytes=size):
        with pytest.raises(RuntimeError, match='nested'):
            with cache.resident_window([a], max_resident_bytes=held,
                    max_workers=1):
                pytest.fail('nested window')
        assert cache.get_resident(*a) is outside
    assert cache.get_resident(*a) is outside
    assert isinstance(cache.weights[b], str) and isinstance(cache.weights[c], str)
    cache._lru_max_bytes = held - 1
    with pytest.raises(RuntimeError, match='LRU budget'):
        with cache.retained_window((b, c), max_resident_bytes=held,
                max_workers=1, max_load_buffer_bytes=size):
            pytest.fail('unrelated LRU owner evicted')


def test_retained_key_costs_price_only_selected_entries(tmp_path):
    cache, paths, _ = make_cache(tmp_path, 2, budget=64)
    a, b = paths
    cache.weights[('unselected', 'FP8')] = 'missing-file.pt'
    aliases = [(name + '.weight', fmt) for name, fmt in (a, b)]
    costs = cache.retained_key_costs(aliases)
    assert tuple(costs) == (a, b)
    # Charged as the retained window and the sealed plan charge it: its
    # length, which bounds its 32 stored bytes (PQ #1210).
    assert costs[a] == {'incoming_storage_bytes': paths[a].stat().st_size,
                        'serialized_bytes': paths[a].stat().st_size}
    cache.get(*a)
    assert cache.retained_key_costs([a])[a] == {
        'incoming_storage_bytes': 0, 'serialized_bytes': 0}


def test_retained_window_advices_each_quantum_before_next_admission(tmp_path, monkeypatch):
    from prismaquant import perturbed_x_cache
    cache, paths, _, size = _retained_cache(tmp_path, 3)
    keys = tuple(paths)
    cache.enable_file_load_receipts(max_file_bytes=size)
    original_prefetch = cache.prefetch
    events = []
    def before_load(state):
        events.append(('guard', dict(state)))
    def prefetch(selected, max_workers, **kwargs):
        events.append(('load', selected[0]))
        return original_prefetch(selected, max_workers=max_workers, **kwargs)
    def advice(path, *, expected_stat):
        key = next(key for key, candidate in paths.items() if str(candidate) == path)
        tensor = cache.get_resident(*key)
        assert cache.file_load_receipt(key, tensor)['bytes'] == expected_stat.st_size
        events.append(('advice', key))
    monkeypatch.setattr(cache, 'prefetch', prefetch)
    monkeypatch.setattr(perturbed_x_cache, 'release_activation_cache_file_pages', advice)
    with cache.retained_window(keys, max_resident_bytes=3 * size,
            max_workers=1, max_load_buffer_bytes=size,
            release_file_pages=True, before_load_quantum=before_load) as receipt:
        assert receipt['file_pages_advised'] == 3
    assert [kind for kind, _ in events] == ['guard', 'load', 'advice'] * 3
    # Resident bytes are what is stored; the remaining incoming charge is
    # the files' lengths, as the sealed plan charges them (PQ #1210).
    assert [value for kind, value in events if kind == 'guard'] == [
        {'resident_bytes': 0, 'remaining_incoming_storage_bytes': 3 * size,
         'next_serialized_bytes': paths[keys[0]].stat().st_size},
        {'resident_bytes': 32, 'remaining_incoming_storage_bytes': 2 * size,
         'next_serialized_bytes': paths[keys[1]].stat().st_size},
        {'resident_bytes': 64, 'remaining_incoming_storage_bytes': size,
         'next_serialized_bytes': paths[keys[2]].stat().st_size},
    ]


def test_retained_window_midload_guard_refusal_releases_selected_owners(tmp_path, monkeypatch):
    from prismaquant import perturbed_x_cache
    cache, paths, _, size = _retained_cache(tmp_path, 3)
    keys = tuple(paths)
    cache.enable_file_load_receipts(max_file_bytes=size)
    original_load = cache._load_file_tensor
    reads, advice, guards = [], [], []
    def load(value, key=None):
        reads.append(str(value))
        return original_load(value, key)
    def advise(path, *, expected_stat):
        advice.append(path)
    def guard(state):
        guards.append(dict(state))
        if len(guards) == 2:
            raise RuntimeError('host reserve refused')
    monkeypatch.setattr(cache, '_load_file_tensor', load)
    monkeypatch.setattr(perturbed_x_cache, 'release_activation_cache_file_pages', advise)
    with pytest.raises(RuntimeError, match='host reserve refused'):
        with cache.retained_window(keys, max_resident_bytes=3 * size,
                max_workers=1, max_load_buffer_bytes=size,
                release_file_pages=True, before_load_quantum=guard):
            pytest.fail('guard refusal exposed a partial window')
    assert reads == [str(paths[keys[0]])]
    assert advice == [str(paths[keys[0]])]
    assert len(guards) == 2 and guards[1]['resident_bytes'] == 32
    assert all(isinstance(cache.weights[key], str) for key in keys)
    assert cache._lru_bytes == 0 and not cache._file_load_receipts
    assert cache._resident_window_files is None


def test_retained_window_advices_shared_file_once_after_last_read(tmp_path, monkeypatch):
    from prismaquant import perturbed_x_cache
    cache, paths, _, size = _retained_cache(tmp_path, 2)
    first, last = paths
    shared = ('shared', first[1])
    cache.weights[shared] = str(paths[first])
    # Two keys read the first file: each is charged its own load.
    cache.enable_lru(3 * size)
    advice = []
    monkeypatch.setattr(perturbed_x_cache, 'release_activation_cache_file_pages',
                        lambda path, *, expected_stat: advice.append(path))
    with cache.retained_window((first, shared, last), max_resident_bytes=3 * size,
            max_workers=1, max_load_buffer_bytes=size,
            release_file_pages=True) as receipt:
        assert receipt['file_pages_advised'] == 2
    assert advice == [str(paths[first]), str(paths[last])]


def test_window_width_follows_the_admitted_bytes_not_the_loader_count(tmp_path, workers):
    """A quantum holds every key its two byte budgets admit (#693).

    The joint-AURA walk hands one unit's five renders to the planner under a
    budget that fits all five. Bounding the quantum by the loader count instead
    split it into four keys at a fraction of the residency it was admitted for,
    plus a one-key quantum that read a single file on a single thread.
    """
    cache, paths, expected = make_cache(tmp_path, 5, budget=100000)
    keys = tuple(paths)
    each = max(path.stat().st_size for path in paths.values())
    assert all(path.stat().st_size == each for path in paths.values())
    plenty = 5 * each

    # Both budgets admit all five, and the loader count is at most two.
    assert cache.plan_resident_windows(keys, max_resident_bytes=plenty,
                                       max_workers=workers) == (keys,)
    assert cache.plan_resident_windows(keys, max_resident_bytes=plenty,
                                       max_load_buffer_bytes=plenty,
                                       max_workers=workers) == (keys,)

    # The full-width quantum is a window the context manager accepts, and the
    # pool still sees exactly one call with every key in it.
    loads = []
    original = cache.prefetch
    def counted(window_keys, max_workers):
        loads.append((tuple(window_keys), max_workers))
        return original(window_keys, max_workers=max_workers)
    cache.prefetch = counted
    try:
        with cache.resident_window(keys, max_resident_bytes=plenty,
                                   max_load_buffer_bytes=plenty,
                                   max_workers=workers) as receipt:
            assert receipt['keys'] == keys and receipt['loaded'] == 5
            assert receipt['resident_bytes'] == 32 * 5
            for key in keys:
                torch.testing.assert_close(cache.get_resident(*key), expected[key])
    finally:
        del cache.prefetch
    assert loads == [(keys, workers)]

    # MUTATE THE DRIVER: each budget must still bite on its own axis.
    assert cache.plan_resident_windows(keys, max_resident_bytes=3 * each,
                                       max_load_buffer_bytes=plenty,
                                       max_workers=workers) == (keys[:3], keys[3:])
    assert cache.plan_resident_windows(keys, max_resident_bytes=plenty,
                                       max_load_buffer_bytes=2 * each,
                                       max_workers=workers) == (keys[:2], keys[2:4], keys[4:])
    with pytest.raises(RuntimeError, match='single serialized load buffer'):
        cache.plan_resident_windows(keys, max_resident_bytes=plenty,
                                    max_load_buffer_bytes=each - 1, max_workers=workers)
    with pytest.raises(RuntimeError, match='single entry exceeds resident window'):
        cache.plan_resident_windows(keys, max_resident_bytes=each - 1, max_workers=workers)
    with pytest.raises(ValueError, match='serialized buffer budget'):
        cache.plan_resident_windows(keys, max_resident_bytes=plenty,
                                    max_load_buffer_bytes=0, max_workers=workers)
    # A window whose serialized buffers exceed their cap is split by the plan,
    # so the context manager refuses the oversized key set instead of loading it.
    with pytest.raises(RuntimeError, match='one nonempty planned quantum'):
        with cache.resident_window(keys, max_resident_bytes=plenty,
                                   max_load_buffer_bytes=2 * each, max_workers=workers):
            pass


def test_window_preflight_scans_each_archive_once_per_lifetime(tmp_path, monkeypatch, workers):
    """Preflight prices a file's archive storage once, not once per call (#693).

    The joint walk reaches the same key three times before it reads anything:
    the caller's plan, the window's own re-plan, and the window's file table.
    Each one opened the archive and read its central directory, which over
    cold NFS was 11.0% of the prepare's main-thread wall time.
    """
    from pathlib import Path as _Path
    from prismaquant import perturbed_x_cache as pxc

    cache, paths, expected = make_cache(tmp_path, 3, budget=100000)
    keys = tuple(paths)
    plenty = 3 * max(path.stat().st_size for path in paths.values())
    scans = []
    original = pxc.torch_archive_storage_bytes

    def counted(source, **kwargs):
        if isinstance(source, (str, _Path)):
            scans.append(str(source))
        return original(source, **kwargs)

    monkeypatch.setattr(pxc, 'torch_archive_storage_bytes', counted)

    windows = cache.plan_resident_windows(keys, max_resident_bytes=plenty,
                                          max_load_buffer_bytes=plenty, max_workers=workers)
    assert windows == (keys,)
    with cache.resident_window(keys, max_resident_bytes=plenty,
                               max_load_buffer_bytes=plenty, max_workers=workers):
        for key in keys:
            torch.testing.assert_close(cache.get_resident(*key), expected[key])
    assert sorted(scans) == sorted(str(path.absolute()) for path in paths.values())

    # The memo does not outlive the window it served.
    scans.clear()
    cache.plan_resident_windows(keys, max_resident_bytes=plenty,
                                max_load_buffer_bytes=plenty, max_workers=workers)
    assert len(scans) == len(keys)

    # MUTATE THE DRIVER: a same-size rewrite is a miss, and only it rescans.
    scans.clear()
    cache.plan_resident_windows(keys, max_resident_bytes=plenty,
                                max_load_buffer_bytes=plenty, max_workers=workers)
    assert scans == []
    changed = keys[1]
    size_before = paths[changed].stat().st_size
    torch.save(expected[changed] + 100, paths[changed])
    # Same dtype, same shape, same archive size: the stat signature's mtime and
    # ctime are the only thing that can catch this, and they are the memo key.
    assert paths[changed].stat().st_size == size_before
    cache.plan_resident_windows(keys, max_resident_bytes=plenty,
                                max_load_buffer_bytes=plenty, max_workers=workers)
    assert scans == [str(paths[changed].absolute())]

    # Compaction drops it too, so a pickled cache carries no file metadata.
    scans.clear()
    cache.compact_for_pickle()
    assert cache._window_archive_bytes is None
    cache.plan_resident_windows(keys, max_resident_bytes=plenty,
                                max_load_buffer_bytes=plenty, max_workers=workers)
    assert len(scans) == len(keys)
