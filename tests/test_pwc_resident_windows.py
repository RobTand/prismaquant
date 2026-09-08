"""Finite research windows reuse PWC loads and never fault on consumption."""
import weakref
import zipfile

import pytest
import torch

from prismaquant.production_weight_cache import ProductionWeightCache
from test_pwc_file_load_receipts import make_cache


def test_plan_resolves_aliases_and_bounds_existing_prefetch(tmp_path, monkeypatch):
    cache, paths, expected = make_cache(tmp_path, 5, budget=100000)
    keys = tuple(paths)
    bound = 2 * max(path.stat().st_size for path in paths.values())
    aliases = [(name + '.weight', fmt) for name, fmt in keys]
    windows = cache.plan_resident_windows(aliases + [aliases[0]],
        max_resident_bytes=bound, max_workers=2)
    assert windows == (keys[:2], keys[2:4], keys[4:])
    original = cache.prefetch
    calls = []
    def bounded(keys, max_workers):
        calls.append(tuple(keys))
        assert len(keys) <= max_workers == 2
        return original(keys, max_workers=max_workers)
    monkeypatch.setattr(cache, 'prefetch', bounded)
    refs = []
    for window in windows:
        with cache.resident_window(window, max_resident_bytes=bound, max_workers=2) as receipt:
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


def test_full_backing_storage_and_aliases_are_accounted():
    pool = torch.zeros(100)
    keys = [('a', 'FP8'), ('b', 'FP8')]
    cache = ProductionWeightCache(dict(zip(keys, (pool[:2], pool[2:4]))), {})
    with pytest.raises(RuntimeError, match='budget'):
        cache.plan_resident_windows(keys, max_resident_bytes=399, max_workers=2)
    assert cache.plan_resident_windows(keys, max_resident_bytes=400, max_workers=2) == (tuple(keys),)
    with cache.resident_window(keys, max_resident_bytes=400, max_workers=2) as receipt:
        assert receipt['resident_bytes'] == 400
    assert all(isinstance(cache.weights[key], torch.Tensor) for key in keys)


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


def test_selected_release_preserves_unrelated_residents_and_receipts(tmp_path):
    cache, paths, _ = make_cache(tmp_path, 2)
    a, b = paths
    cache.enable_file_load_receipts(max_file_bytes=10000)
    cache.prefetch([a, b], max_workers=2)
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


def test_lru_eviction_cannot_yield_partial_window(tmp_path):
    cache, paths, _ = make_cache(tmp_path, 2, budget=32)
    with pytest.raises(RuntimeError, match='resident|LRU|budget'):
        with cache.resident_window(tuple(paths), max_resident_bytes=10000, max_workers=2):
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


def test_serialized_buffers_have_a_separate_aggregate_limit(tmp_path, monkeypatch):
    cache, paths, _ = make_cache(tmp_path, 2)
    size = sum(path.stat().st_size for path in paths.values())
    monkeypatch.setattr(cache, '_load_file_tensor', lambda *args: pytest.fail('hidden load'))
    with pytest.raises(RuntimeError, match='serialized'):
        with cache.resident_window(tuple(paths), max_resident_bytes=10000, max_workers=2,
                                   max_load_buffer_bytes=size - 1):
            pytest.fail('oversize buffers')


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


def test_window_load_failure_cleans_partial_prefetch_and_allows_fresh_context(tmp_path, monkeypatch):
    cache, paths, _ = make_cache(tmp_path, 2)
    a, b = paths
    original = cache._validate_loaded_cb_pair_tensor
    def refused(key, tensor):
        if key == b:
            raise RuntimeError('synthetic integrity refusal')
        return original(key, tensor)
    monkeypatch.setattr(cache, '_validate_loaded_cb_pair_tensor', refused)
    with pytest.raises(RuntimeError, match='integrity'):
        with cache.resident_window(tuple(paths), max_resident_bytes=10000, max_workers=2):
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
