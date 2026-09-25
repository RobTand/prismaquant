"""Retained joint replay keeps the PWC owner while replacing each probe lease."""
from collections import Counter
import os

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant.joint_retained_window_plan import RetainedWindowBudget
from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows
from prismaquant.joint_statistics_replay import (
    SCHEMA, observe_and_project_retained_windows, observe_and_project_windows,
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
    original_load = cache._load_file_tensor
    loaded = []
    def read_once(value, key=None):
        loaded.append(str(value))
        return original_load(value, key)
    monkeypatch.setattr(cache, '_load_file_tensor', read_once)
    result, calls, consumed, records = _run_retained(
        modules, specs, cache, policy, budget)
    assert len(result['plan']['windows']) == 2
    assert result['plan']['archive_admission'] is True
    assert calls == ([(probe, False) for probe in range(4)]
                     + [(probe, True) for probe in range(4)])
    assert [(index, receipt['window_index']) for index, _, _, receipt in consumed] == (
        [(probe, window) for window in range(2) for probe in range(4)])
    assert Counter(loaded) == Counter(str(path) for path in paths.values())
    assert Counter(records) == Counter({key: 4 for key in paths})
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
    monkeypatch.setattr(cache, '_load_file_tensor',
                        lambda *args: pytest.fail('nonempty baseline loaded a file'))
    with pytest.raises(RuntimeError, match='empty PWC resident baseline'):
        _run_retained(modules, specs, cache, policy, budget)


def test_retained_replay_guard_reserves_remaining_window_and_unwinds_midload(
        tmp_path, monkeypatch):
    import prismaquant.joint_statistics_replay as replay
    modules, specs, cache, paths, policy, budget = _fixture(tmp_path)
    checks = []
    def check(guard, label, *, reserve_bytes):
        assert guard is marker
        checks.append((label, reserve_bytes))
        if len(checks) == 2:
            raise RuntimeError('physical guard refused next load')
    marker = object()
    monkeypatch.setattr(replay, 'check_operator_allocation', check)
    first_keys = tuple(key for key in paths if key[0] == 'first')
    costs = cache.retained_key_costs(first_keys)
    remaining = sum(cost['incoming_storage_bytes'] for cost in costs.values())
    common = (budget.statistics_cap_bytes + budget.workspace_reserve_bytes
              + budget.boundary_reserve_bytes + budget.load_buffer_bytes
              + budget.read_page_reserve_bytes + budget.candidate_delta_bytes)
    with pytest.raises(RuntimeError, match='physical guard refused'):
        _run_retained(modules, specs, cache, policy, budget, guard=marker)
    assert checks == [
        ('before_joint_retained_candidate_load', remaining + common),
        ('before_joint_retained_candidate_load',
         remaining - costs[first_keys[0]]['incoming_storage_bytes'] + common),
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
    loader = cache._load_file_tensor
    def load(path, key=None):
        loads.append(str(path))
        return loader(path, key)
    monkeypatch.setattr(cache, '_load_file_tensor', load)
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
