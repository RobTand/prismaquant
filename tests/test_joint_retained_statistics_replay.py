"""Retained joint replay keeps the PWC owner while replacing each probe lease."""
from collections import Counter
import os

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import io_engine
from prismaquant.joint_retained_window_plan import RetainedWindowBudget
from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
from prismaquant.joint_statistics_replay import (
    SCHEMA, RetainedRenderDeviceCache, observe_and_project_retained_windows,
    observe_and_project_windows,
)
from prismaquant.production_weight_cache import ProductionWeightCache


FORMATS = ('FP8_E4M3', 'NVFP4A16')


def _fixture(tmp_path):
    torch.manual_seed(17)
    modules = {name: torch.nn.Linear(16, 16, bias=False).eval()
               for name in ('first', 'second')}
    for module in modules.values():
        module.weight.requires_grad_(False)
    specs = {name: {fmt: fr.get_format(fmt) for fmt in FORMATS}
             for name in modules}
    paths = {}
    for name, module in modules.items():
        for index, fmt in enumerate(FORMATS):
            path = tmp_path / f'{name}-{index}.pt'
            torch.save(module.weight.detach().clone() + (index + 1) / 128, path)
            paths[name, fmt] = path
    cache = ProductionWeightCache(
        {key: str(path) for key, path in paths.items()}, {},
        activation_max_abs={name: 1.0 for name in modules})
    maximum_file = max(path.stat().st_size for path in paths.values())
    # The quantum sizes the LRU to the retained render cap, which the plan
    # counts in file lengths, as the retained window charges them (PQ #1210).
    cache.enable_lru(4 * maximum_file)
    full = plan_joint_statistics_target_windows(
        modules, specs, max_statistics_bytes=1 << 20,
        activation_max_abs=cache.activation_max_abs)
    one_target = max(target.statistics_bytes for target in full.targets)
    assert len(full.targets) == 2
    policy = dict(
        schema=SCHEMA, max_statistics_bytes=one_target,
        max_candidate_bytes=16 * 16 * 4,
        max_render_resident_bytes=4 * maximum_file,
        max_load_buffer_bytes=maximum_file,
        workspace_reserve_bytes=1 << 14,
        max_replay_cotangent_bytes=1 << 14,
        prefetch_workers=1,
    )
    budget = RetainedWindowBudget(
        physical_limit_bytes=1 << 24,
        safety_margin_bytes=1 << 12,
        metadata_reserve_bytes=1 << 12,
        runtime_reserve_bytes=1 << 12,
        workspace_reserve_bytes=1 << 14,
        boundary_reserve_bytes=0,
        auxiliary_reserve_bytes=0,
        load_buffer_bytes=maximum_file,
        read_page_reserve_bytes=1 << 13,
        candidate_delta_bytes=16 * 16 * 4,
        statistics_cap_bytes=one_target,
        retained_render_cap_bytes=4 * maximum_file,
        max_windows_per_layer=2,
    )
    return modules, specs, cache, paths, policy, budget


def _spy_reads(monkeypatch, reads):
    """Record every file the IO engine reads (PQ #1294), on any thread."""
    load = io_engine.load_file

    def spy(path, limit, **kwargs):
        reads.append(str(path))
        return load(path, limit, **kwargs)

    monkeypatch.setattr(io_engine, 'load_file', spy)


def _backward(modules, *, probe_index, final, lease):
    # The exact same signed source/cotangent arithmetic feeds both replay
    # orders. The input needs grad so real output observers see backward.
    x = (torch.arange(16, dtype=torch.float32).reshape(1, 16) / 11
         + probe_index / 7).requires_grad_(True)
    loss = sum((module(x).square().sum() * (probe_index + 1))
               for module in modules.values())
    loss.backward()


def _run_retained(modules, specs, cache, policy, budget, **options):
    calls, consumed, records = [], [], []
    def backward(*, probe_index, final, lease):
        calls.append((probe_index, final))
        _backward(modules, probe_index=probe_index, final=final, lease=lease)
    def record(name, fmt, source, rendered):
        records.append((name, fmt))
    def consume(probe_index, terms, diagnostics, receipt):
        consumed.append((probe_index, terms, diagnostics, receipt))
    result = observe_and_project_retained_windows(
        modules, specs, cache, policy, retained_budget=budget,
        n_probes=4, source_bytes=1 << 12, backward=backward,
        record_operator=record, consume_probe=consume,
        collect_col_energy=True, backend=None, **options)
    return result, calls, consumed, records


def _run_probe_major(modules, specs, cache, policy):
    per_probe = []
    for probe_index in range(4):
        def backward(*, final, lease):
            _backward(modules, probe_index=probe_index, final=final, lease=lease)
        per_probe.append(observe_and_project_windows(
            modules, specs, cache, policy, backward=backward,
            record_operator=lambda *args: None, collect_col_energy=True,
            backend=None))
    return per_probe


def test_retained_replay_matches_probe_major_signed_components_and_loads_once(
        tmp_path, monkeypatch):
    modules, specs, cache, paths, policy, budget = _fixture(tmp_path)
    loaded = []
    _spy_reads(monkeypatch, loaded)
    result, calls, consumed, records = _run_retained(
        modules, specs, cache, policy, budget)
    assert len(result['plan']['windows']) == 2
    assert result['plan']['archive_admission'] is True
    assert calls == ([(probe, False) for probe in range(4)]
                     + [(probe, True) for probe in range(4)])
    assert [(index, receipt['window_index']) for index, _, _, receipt in consumed] == (
        [(probe, window) for window in range(2) for probe in range(4)])
    assert Counter(loaded) == Counter(str(path) for path in paths.values())
    # Recorded at each window's first probe and once more at its close
    # (PQ #1348), not on every probe.
    assert Counter(records) == Counter({key: 2 for key in paths})
    assert all(isinstance(value, str) for value in cache.weights.values())
    assert cache._lru_bytes == 0 and not cache._file_load_receipts

    old_cache = ProductionWeightCache(
        {key: str(path) for key, path in paths.items()}, {},
        activation_max_abs={name: 1.0 for name in modules})
    old_cache.enable_lru(4 * 16 * 16 * 4)
    original = _run_probe_major(modules, specs, old_cache, policy)
    for probe_index in range(4):
        actual_terms = {}
        actual_diagnostics = {}
        for observed_index, terms, diagnostics, _ in consumed:
            if observed_index == probe_index:
                actual_terms.update(terms)
                actual_diagnostics.update(diagnostics)
        old_terms, old_diagnostics, _ = original[probe_index]
        assert actual_terms.keys() == old_terms.keys()
        for key in old_terms:
            assert actual_terms[key] == pytest.approx(old_terms[key], rel=0, abs=0)
        for name in old_diagnostics:
            assert actual_diagnostics[name]['g_trace'] == pytest.approx(
                old_diagnostics[name]['g_trace'], rel=0, abs=0)
            torch.testing.assert_close(
                actual_diagnostics[name]['col_energy'],
                old_diagnostics[name]['col_energy'], rtol=0, atol=0)


def test_retained_replay_source_mutation_refuses_and_releases_cache(tmp_path):
    modules, specs, cache, _, policy, budget = _fixture(tmp_path)
    consumed = []
    def backward(*, probe_index, final, lease):
        _backward(modules, probe_index=probe_index, final=final, lease=lease)
    def consume(probe_index, terms, diagnostics, receipt):
        consumed.append((probe_index, receipt['window_index']))
        if len(consumed) == 4:
            with torch.no_grad():
                modules['second'].weight.add_(1)
    with pytest.raises(RuntimeError, match='source changed'):
        observe_and_project_retained_windows(
            modules, specs, cache, policy, retained_budget=budget,
            n_probes=4, source_bytes=1 << 12, backward=backward,
            record_operator=lambda *args: None, consume_probe=consume,
            collect_col_energy=False, backend=None)
    assert consumed == [(probe, 0) for probe in range(4)]
    assert all(isinstance(value, str) for value in cache.weights.values())
    assert cache._lru_bytes == 0 and cache._resident_window_files is None


def test_retained_replay_refuses_nonempty_pwc_before_selected_file_load(tmp_path, monkeypatch):
    modules, specs, cache, _, policy, budget = _fixture(tmp_path)
    cache.weights['unrelated', 'FP8'] = torch.zeros(16)
    monkeypatch.setattr(io_engine, 'load_file',
                        lambda *args, **kwargs: pytest.fail('nonempty baseline loaded a file'))
    with pytest.raises(RuntimeError, match='empty PWC resident baseline'):
        _run_retained(modules, specs, cache, policy, budget)


def test_retained_replay_guard_reserves_each_side_and_unwinds_a_loaded_window(
        tmp_path, monkeypatch):
    """The window's loads are charged once, each reservation on its side.

    The IO engine reads a window's renders without a barrier between load
    quanta (PQ #1291), so the window charges everything it has not read yet
    before its first load: the incoming renders, their serialized buffers,
    the read pages and the boundaries on the host, the statistics, the
    workspace and the candidate delta on the device. The sums are the ones
    the lumped reservation charged. A refusal after the loads, at the first
    probe, releases the whole window.
    """
    import prismaquant.joint_statistics_replay as replay
    modules, specs, cache, paths, policy, budget = _fixture(tmp_path)
    checks = []
    def check(guard, label, *, reserve_bytes, reserve_device_bytes=0):
        assert guard is marker
        checks.append((label, reserve_bytes, reserve_device_bytes))
        if len(checks) == 2:
            raise RuntimeError('physical guard refused the first probe')
    marker = object()
    monkeypatch.setattr(replay, 'check_operator_allocation', check)
    first_keys = tuple(key for key in paths if key[0] == 'first')
    costs = cache.retained_key_costs(first_keys)
    remaining = sum(cost['incoming_storage_bytes'] for cost in costs.values())
    host = (budget.boundary_reserve_bytes + budget.load_buffer_bytes
            + budget.read_page_reserve_bytes)
    device = (budget.statistics_cap_bytes + budget.workspace_reserve_bytes
              + budget.candidate_delta_bytes)
    with pytest.raises(RuntimeError, match='physical guard refused'):
        _run_retained(modules, specs, cache, policy, budget, guard=marker)
    assert checks == [
        ('before_joint_retained_candidate_load', remaining + host, device),
        ('before_joint_retained_statistics_probe', budget.boundary_reserve_bytes, device),
    ]
    assert all(isinstance(value, str) for value in cache.weights.values())
    assert cache._lru_bytes == 0 and cache._resident_window_files is None


def test_retained_replay_file_change_between_probes_refuses_and_cleans(tmp_path):
    modules, specs, cache, paths, policy, budget = _fixture(tmp_path)
    consumed = []
    def consume(probe_index, terms, diagnostics, receipt):
        consumed.append((probe_index, receipt['window_index']))
        if len(consumed) == 1:
            path = paths['first', FORMATS[0]]
            before = path.stat()
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
    with pytest.raises(RuntimeError, match='changed'):
        observe_and_project_retained_windows(
            modules, specs, cache, policy, retained_budget=budget,
            n_probes=4, source_bytes=1 << 12,
            backward=lambda *, probe_index, final, lease:
                _backward(modules, probe_index=probe_index, final=final, lease=lease),
            record_operator=lambda *args: None, consume_probe=consume,
            collect_col_energy=False, backend=None)
    assert consumed == [(0, 0)]
    assert all(isinstance(value, str) for value in cache.weights.values())
    assert cache._lru_bytes == 0 and cache._resident_window_files is None


def test_resume_keeps_original_windows_and_only_final_active_updates_cotangents(tmp_path, monkeypatch):
    modules, specs, cache, paths, policy, budget = _fixture(tmp_path)
    loads, entered, committed = [], [], []
    _spy_reads(monkeypatch, loads)
    result, calls, consumed, _ = _run_retained(modules, specs, cache, policy, budget,
        completed_names={'first'},
        before_window=lambda index, names: entered.append((index, names)),
        after_window=lambda index, names: committed.append((index, names)))
    assert entered == [(0, ('first',)), (1, ('second',))]
    assert committed == [(1, ('second',))]
    assert calls == [(probe, True) for probe in range(4)]
    assert len(loads) == 2 and all('second' in path for path in loads)
    assert [w['names'] for w in result['plan']['windows']] == [('first',), ('second',)]
    assert all(set(diagnostics) == {'second'} for _, _, diagnostics, _ in consumed)
    assert not cache._window_resident_storages()


# -- the device render cache (PQ #1348) ------------------------------------


def _by_probe(consumed):
    return [(probe, receipt['window_index'], terms,
             {name: (d['g_trace'], d['col_energy']) for name, d in diagnostics.items()})
            for probe, terms, diagnostics, receipt in consumed]


def _assert_same_bytes(actual, expected):
    assert len(actual) == len(expected)
    for (probe, window, terms, diagnostics), (probe0, window0, terms0, diagnostics0) in zip(
            actual, expected):
        assert (probe, window) == (probe0, window0)
        assert terms.keys() == terms0.keys()
        for key in terms0:
            assert terms[key] == terms0[key]
        for name, (g_trace, col_energy) in diagnostics0.items():
            assert diagnostics[name][0] == g_trace
            assert torch.equal(diagnostics[name][1], col_energy)


def _run_with(tmp_path, render_cache=None, consume_hook=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    modules, specs, cache, paths, policy, budget = _fixture(tmp_path)
    consumed, records, reads = [], [], []

    def backward(*, probe_index, final, lease):
        _backward(modules, probe_index=probe_index, final=final, lease=lease)

    def record(name, fmt, source, rendered):
        records.append((name, fmt))

    def consume(probe_index, terms, diagnostics, receipt):
        consumed.append((probe_index, terms, diagnostics, receipt))
        if consume_hook is not None:
            consume_hook(probe_index, receipt['window_index'])

    observe_and_project_retained_windows(
        modules, specs, cache, policy, retained_budget=budget,
        n_probes=4, source_bytes=1 << 12, backward=backward,
        record_operator=record, consume_probe=consume,
        collect_col_energy=True, backend=None, render_cache=render_cache)
    return _by_probe(consumed), records, paths


def test_a_device_render_cache_changes_no_signed_component(tmp_path, monkeypatch):
    """Renders kept across a window's probes give the uncached run's bytes.

    Each window's first probe copies its two renders and keeps them; the
    other three probes widen the kept copies. Every signed component and
    diagnostic equals the uncached run's exactly, each render file is read
    once, and nothing is kept once the pass ends.
    """
    loaded = []
    _spy_reads(monkeypatch, loaded)
    plain, plain_records, _paths = _run_with(tmp_path / 'plain')
    loaded.clear()
    render_cache = RetainedRenderDeviceCache(lambda: 1 << 30)
    cached, records, paths = _run_with(tmp_path / 'cached', render_cache)
    _assert_same_bytes(cached, plain)
    assert Counter(records) == Counter(plain_records) == Counter({key: 2 for key in paths})
    assert Counter(loaded) == Counter(str(path) for path in paths.values())
    render_bytes = 16 * 16 * 4
    assert render_cache.counters == {
        'hits': 2 * 2 * 3, 'misses': 2 * 2, 'admitted': 2 * 2, 'refused': 0,
        'reclaims': 0, 'reclaimed_bytes': 0, 'peak_bytes_held': 2 * render_bytes}
    assert render_cache.bytes_held == 0
    assert render_cache.last_window_peak_bytes == 2 * render_bytes


def test_a_render_cache_without_headroom_is_the_uncached_path(tmp_path):
    plain, _records, _paths = _run_with(tmp_path / 'plain')
    render_cache = RetainedRenderDeviceCache(lambda: 0)
    cached, _records, _paths = _run_with(tmp_path / 'refused', render_cache)
    _assert_same_bytes(cached, plain)
    # Every probe but each window's last asked to keep, and was refused.
    assert render_cache.counters['hits'] == render_cache.counters['admitted'] == 0
    assert render_cache.counters['misses'] == 2 * 2 * 4
    assert render_cache.counters['refused'] == 2 * 2 * 3


def test_a_reclaimed_render_is_copied_again_with_the_same_bytes(tmp_path):
    plain, _records, _paths = _run_with(tmp_path / 'plain')
    render_cache = RetainedRenderDeviceCache(lambda: 1 << 30)

    def reclaim_after_probe_one(probe_index, window_index):
        if probe_index == 1:
            assert render_cache.reclaim(1) == 16 * 16 * 4

    cached, _records, _paths = _run_with(tmp_path / 'reclaimed', render_cache,
                                         reclaim_after_probe_one)
    _assert_same_bytes(cached, plain)
    counters = render_cache.counters
    # One render of each window dropped after probe 1, copied and kept
    # again at probe 2.
    assert counters['reclaims'] == 2 and counters['reclaimed_bytes'] == 2 * 16 * 16 * 4
    assert counters['misses'] == 2 * (2 + 1) and counters['admitted'] == 2 * (2 + 1)
    assert counters['hits'] == 2 * (2 * 3 - 1)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_a_kept_render_widens_to_the_uncached_delta_bytes(dtype):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(3)
    rendered = (torch.randn(64, 96) * 3).to(dtype)
    rendered[0, :4] = torch.tensor([0.0, -0.0, 1e-8, -65504.0]).to(dtype)
    source = torch.randn(64, 96, device=device)
    expected = rendered.to(device=device, dtype=torch.float32, copy=True)
    expected.sub_(source)
    render_cache = RetainedRenderDeviceCache(lambda: 1 << 30)
    first = render_cache.delta('k', rendered, source, keep=True)
    again = render_cache.delta('k', lambda: pytest.fail('a hit reads no render'), source,
                               keep=False)
    for delta in (first, again):
        assert delta.dtype == torch.float32 and delta.device == source.device
        assert torch.equal(delta.view(torch.int32), expected.view(torch.int32))
    assert render_cache.counters['hits'] == render_cache.counters['misses'] == 1
    assert render_cache.reclaim(1) == rendered.numel() * rendered.element_size()
    assert render_cache.bytes_held == 0
