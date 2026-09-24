"""Integrated target replay preserves probes while bounding matrix/cache owners."""
import os
from functools import partial

import pytest
import torch

from prismaquant.joint_statistics_replay import SCHEMA, normalize_operator_windows
import prismaquant.aura_cost as aura
from test_joint_aura_streamed import _fixture, _run


#: The window worker count this fixture asks for when the shard is wide
#: enough to hold it. Two is what these suites were written against.
FIXTURE_PREFETCH_WORKERS = 2


def admissible_prefetch_workers(want=FIXTURE_PREFETCH_WORKERS):
    """Window workers that fit the CPUs this process was assigned.

    ``ProductionWeightCache._window_limits`` refuses a window whose worker
    count exceeds ``len(os.sched_getaffinity(0))``, and that refusal is
    correct production behavior: a shard admitted with one CPU may not open
    two loaders. A test that hardcodes the count therefore asserts on the
    host's CPU count rather than on the code (PQ #888), so the fixture asks
    for what it was given. The mask is read the same bare way the guard
    reads it, so the two can never disagree.
    """

    return max(1, min(int(want), len(os.sched_getaffinity(0))))


def policy(**overrides):
    # Merged, not splatted: a keyword this already names must override it
    # rather than raise `dict() got multiple values`, so a test that wants
    # a wider window than the mask allows can still ask for one by name.
    base = dict(schema=SCHEMA, max_statistics_bytes=2048, max_candidate_bytes=1024,
        max_render_resident_bytes=1024*1024, max_load_buffer_bytes=1024*1024,
        workspace_reserve_bytes=1024*1024, max_replay_cotangent_bytes=1024*1024,
        prefetch_workers=admissible_prefetch_workers())
    base.update(overrides)
    return base


def test_dense_windows_match_independent_full_model_fp64_oracle(monkeypatch):
    import test_joint_aura_streamed as original
    monkeypatch.setattr(original, '_run', partial(_run, operator_windows=policy()))
    original.test_joint_streamed_matches_full_model_output_residual_oracle(monkeypatch)


def test_packed_windows_match_independent_fp64_oracle(monkeypatch):
    import test_joint_aura_packed as original
    monkeypatch.setattr(original, '_run', partial(original._run, operator_windows=policy()))
    original.test_packed_joint_rows_match_full_model_residual_oracle()


def test_replay_reuses_installed_source_and_preserves_input_cotangents():
    import test_joint_aura_packed as packed
    def run(windowed):
        model, context, runner, profile, cache, _ = packed._fixture()
        observed = {}
        forward = runner.isolated_layer
        def isolated(batch, layer, hidden, *, pass_state):
            hidden.register_hook(lambda g: observed.setdefault(layer, []).append(g.detach().clone()))
            return forward(batch, layer, hidden, pass_state=pass_state)
        runner.isolated_layer = isolated
        payload = packed._run(runner, profile, cache,
            **({'operator_windows': policy()} if windowed else {}))
        assert all(parameter.grad is None for parameter in model.parameters())
        assert context.active == set()
        return payload, observed, context.install_calls
    expected, original, reads = run(False)
    actual, replayed, replay_reads = run(True)
    assert reads == replay_reads
    assert actual['provenance']['probe_identity'] != expected['provenance']['probe_identity']
    receipts = actual['provenance']['joint_operator_windows']
    assert all(len(row['plan']['windows']) > 1 for row in receipts)
    for layer, gradients in original.items():
        windows = len(next(r for r in receipts if r['layer'] == layer)['plan']['windows'])
        assert len(replayed[layer]) == len(gradients)*windows
        for index, observed in enumerate(replayed[layer]):
            torch.testing.assert_close(observed, gradients[index//windows], rtol=0, atol=0)


def test_window_resume_refuses_changed_budget_before_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(aura, '_checkpoint_git_commit', lambda: '1'*40)
    _, _, runner, cache = _fixture()
    first = _run(runner, cache, operator_windows=policy(), checkpoint_dir=tmp_path)
    _, context, runner, cache = _fixture()
    resumed = _run(runner, cache, operator_windows=policy(), checkpoint_dir=tmp_path, resume=True)
    assert first['costs'] == resumed['costs']
    assert context.install_calls == 0
    _, context, runner, cache = _fixture()
    changed = policy(); changed['max_statistics_bytes'] *= 2
    with pytest.raises(RuntimeError, match='identity mismatch'):
        _run(runner, cache, operator_windows=changed, checkpoint_dir=tmp_path, resume=True)
    assert context.install_calls == 0


def test_partial_resume_preserves_signed_samples(tmp_path, monkeypatch):
    import test_joint_aura_streamed as original
    monkeypatch.setattr(original, '_run', partial(_run, operator_windows=policy()))
    original.test_joint_interrupted_resume_preserves_signed_samples(tmp_path, monkeypatch)


@pytest.mark.parametrize('change', [{'extra': 1}, {'max_statistics_bytes': True},
    {'max_load_buffer_bytes': 0}, {'max_candidate_bytes': float('inf')}])
def test_policy_refuses_unknown_or_nonfinite_budgets(change):
    value = policy(); value.update(change)
    with pytest.raises(ValueError, match='policy|budgets'):
        normalize_operator_windows(value)


def test_window_refuses_training_or_rng_source(monkeypatch):
    model, _, runner, cache = _fixture()
    model.train()
    with pytest.raises(ValueError, match='eval'):
        _run(runner, cache, operator_windows=policy())
    model, _, runner, cache = _fixture()
    original = runner.isolated_layer
    def stochastic(*args, **kwargs):
        torch.rand(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'isolated_layer', stochastic)
    with pytest.raises(RuntimeError, match='RNG'):
        _run(runner, cache, operator_windows=policy())


def test_disk_candidates_are_released_and_aliases_keep_operator_names(tmp_path):
    _, _, runner, cache = _fixture()
    expected = _run(runner, cache, operator_windows=policy())
    _, _, runner, cache = _fixture()
    paths = {}
    for index, (key, tensor) in enumerate(cache.weights.items()):
        path = tmp_path / f'{index}.pt'
        torch.save(tensor, path)
        paths[(key[0], 'FP8' if key[1] == 'FP8_E4M3' else key[1])] = str(path)
    cache.weights = paths
    cache.enable_lru(1 << 20)
    bound = max(__import__('pathlib').Path(path).stat().st_size for path in paths.values())
    config = policy(); config['max_load_buffer_bytes'] = bound
    actual = _run(runner, cache, operator_windows=config)
    assert all(isinstance(value, str) for value in cache.weights.values())
    for name, rows in expected['costs'].items():
        for fmt, row in rows.items():
            assert actual['costs'][name][fmt]['signed_components_per_probe'] == row['signed_components_per_probe']


def test_shared_cotangent_forks_preserve_multi_consumer_probe_sums(tmp_path, monkeypatch):
    from test_streamed_boundary_artifacts import _shared_run
    expected = _shared_run(tmp_path / 'legacy')
    original = aura.compute_aura_cost_streamed
    def bounded(runner, *args, **kwargs):
        runner.model.eval()
        config = policy(); config['max_statistics_bytes'] = 1024  # one target per window
        return original(runner, *args, operator_windows=config, **kwargs)
    monkeypatch.setattr(aura, 'compute_aura_cost_streamed', bounded)
    actual = _shared_run(tmp_path / 'windows')
    assert any(len(row['plan']['windows']) > 1 for row in actual['provenance']['joint_operator_windows'])
    for name, rows in expected['costs'].items():
        for fmt, row in rows.items():
            assert actual['costs'][name][fmt]['signed_per_probe'] == pytest.approx(
                row['signed_per_probe'], rel=3e-5, abs=3e-8)
    assert actual['provenance']['streamed_boundary_storage']['telemetry']['peak_shared_cotangent_reservation_bytes'] > 0


def test_campaign_admits_only_explicit_candidate_windows_and_checks_later_donor(tmp_path):
    from types import SimpleNamespace
    from prismaquant.tessera_joint_aura import _admit_candidate_phase
    small = tmp_path/'small'; small.write_bytes(b'1'*64)
    large = tmp_path/'large'; large.write_bytes(b'2'*256)
    data = SimpleNamespace(cells={('a','fmt'): {'render': str(small)},
                                  ('b','fmt'): {'render': str(large)}})
    config = {'execution': {}, 'max_render_bytes': 256}
    with pytest.raises(ValueError, match='largest measured candidate layer'):
        _admit_candidate_phase('run', config, data, {0: 320})
    window = policy(); window['max_render_resident_bytes'] = 256; window['max_load_buffer_bytes'] = 256
    config['execution'].update(operator_windows=window, boundary_storage={'explicit': 'owner'})
    assert _admit_candidate_phase('run', config, data, {0: 320}) == window
    with pytest.raises(ValueError, match='largest measured candidate layer'):
        _admit_candidate_phase('prepare', config, data, {0: 320})
    window['max_load_buffer_bytes'] = 128
    with pytest.raises(ValueError, match='read buffer budget'):
        _admit_candidate_phase('run', config, data, {0: 320})
    del config['execution']['boundary_storage']
    with pytest.raises(ValueError, match='exact boundary'):
        _admit_candidate_phase('run', config, data, {0: 320})


def test_passthrough_only_target_refuses_instead_of_emitting_unmeasured_diagnostics():
    from test_streamed_cost_checkpoints import _model_identity
    _, context, runner, cache = _fixture()
    with pytest.raises(ValueError, match='measured candidate for every target'):
        aura.compute_aura_cost_streamed(runner, torch.tensor([[1,2,3,4]]), ['BF16'],
            n_probes=3, min_free_gib=0, joint_activation=True, production_cache=cache,
            model_identity=_model_identity('joint-source'), operator_windows=policy(),
            collect_col_energy=True)
    assert context.install_calls == 0


def test_guarded_operator_phases_release_inactive_allocator_reservation(tmp_path, monkeypatch):
    """Retired CUDA blocks must not consume the next phase's future budget."""
    import prismaquant.joint_statistics_replay as replay
    import prismaquant.memory_management as memory
    scope = tmp_path/'job'; scope.mkdir()
    cap = 4*1024**3
    (scope/'memory.max').write_text(str(cap))
    (scope/'memory.current').write_text(str(1024**3))
    (scope/'memory.stat').write_text("anon 0\nfile 0\nshmem 0\nfile_dirty 0\nfile_writeback 0\n")  # no page cache (PQ #1157)
    membership = tmp_path/'membership'; membership.write_text('0::/job\n')
    guard = memory.CaptureMemoryGuard('cuda:0', cgroup_root=tmp_path, membership=membership)
    state = {'inactive': 2*1024**3, 'releases': 0}
    labels = []
    monkeypatch.setattr(memory, '_host_memory_info', lambda: (32*1024**3, 64*1024**3))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: state['inactive'])
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    def empty_cache():
        state['inactive'] = 0
        state['releases'] += 1
    monkeypatch.setattr(torch.cuda, 'empty_cache', empty_cache)
    checked = guard.check
    def check(label, *, reserve_bytes=0):
        result = checked(label, reserve_bytes=reserve_bytes)
        labels.append(label)
        # Model the just-completed phase leaving only inactive CUDA blocks.
        state['inactive'] = 2*1024**3
        return result
    monkeypatch.setattr(guard, 'check', check)
    monkeypatch.setattr(replay, 'operator_window_guard', lambda device, **envelope: guard)
    _, _, runner, cache = _fixture()
    result = _run(runner, cache, operator_windows=policy())
    assert result['costs']
    assert {'before_joint_statistics_window', 'before_joint_candidate_load',
            'before_joint_window_backward'} <= set(labels)
    assert state['releases'] >= len(labels)


def test_operator_reverse_owns_exact_lookahead_when_cache_has_extra_slots(monkeypatch):
    import copy
    model, context, runner, cache = _fixture()
    model.model.layers.extend([copy.deepcopy(model.model.layers[0]) for _ in range(2)])
    context.num_layers = runner.num_layers = 4
    runner.prefetch_lookahead = 1
    for layer in (2, 3):
        name = f'model.layers.{layer}.proj'
        cache.activation_max_abs[name] = 1.0
        for fmt in ('FP8_E4M3', 'NVFP4A16'):
            cache.weights[name, fmt] = model.model.layers[layer].proj.weight.detach().clone()+0.03125
    original = context.install
    state = {'previous': -1, 'reverse': False}
    futures = set()
    settled = []
    def install(layer, *, require_prefetched=False, prefetch_following=True):
        if layer <= state['previous']:
            state['reverse'] = True
        state['previous'] = layer
        futures.discard(layer)
        if state['reverse'] and prefetch_following:
            # An adaptive three-slot cache can enqueue two successors while
            # this runner's explicitly budgeted reverse lookahead is one.
            futures.update(range(max(0, layer-2), layer))
        return original(layer, require_prefetched=require_prefetched)
    def schedule(layer):
        if state['reverse']:
            futures.add(layer)
    def settle(indices, *, retry_availability=False):
        expected = set(indices)
        if futures != expected:
            raise RuntimeError(f'unexpected source owners: {futures} != {expected}')
        settled.append(expected)
    monkeypatch.setattr(context, 'install', install)
    monkeypatch.setattr(context, 'schedule_prefetch', schedule)
    monkeypatch.setattr(context, 'settle_prefetched_layers', settle, raising=False)
    result = _run(runner, cache, operator_windows=policy())
    assert len(result['costs']) == 4
    assert settled == [{2}, {1}, {0}, set()]


def test_candidate_windows_hold_what_the_operator_policy_admits(tmp_path):
    """The COST read path plans on both byte budgets, not the loader count (#693).

    ``run`` reaches every render through this seam, so a quantum capped at
    ``prefetch_workers`` keys read the campaign's 16.78 MB renders four at a
    time while holding 12.5% of the 512 MB it was admitted for.
    """
    import torch
    from prismaquant.joint_statistics_replay import resident_candidates
    from prismaquant.production_weight_cache import ProductionWeightCache

    paths, weights, expected = {}, {}, {}
    for index in range(6):
        key = (f'unit{index}', 'TESSERA_E4M3_K1_R1024')
        value = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) + index
        path = tmp_path / f'render{index}.pt'
        torch.save(value, path)
        paths[key], weights[key], expected[key] = path, str(path), value
    keys = tuple(weights)
    each = max(path.stat().st_size for path in paths.values())

    def windows_for(**overrides):
        window = policy(**overrides)
        cache = ProductionWeightCache(weights=dict(weights), levers={})
        cache.enable_lru(100000)
        seen = []
        with resident_candidates(cache, keys, window) as iterator:
            for window, receipt in iterator:
                seen.append(window)
                assert receipt['keys'] == window
                for pair in window:
                    torch.testing.assert_close(cache.get_resident(*pair), expected[pair])
        return seen

    # The window is closed by the two byte budgets, not by the loader count,
    # so both of them admit all six whatever the shard's affinity allows.
    assert windows_for() == [keys]

    # MUTATE THE DRIVER: each budget still closes a quantum on its own axis, and
    # the residency budget is no longer shortened by the serialized one.
    assert windows_for(max_render_resident_bytes=3 * each) == [keys[:3], keys[3:]]
    assert windows_for(max_load_buffer_bytes=2 * each) == [keys[:2], keys[2:4], keys[4:]]
    assert windows_for(max_render_resident_bytes=6 * each,
                       max_load_buffer_bytes=each) == [(key,) for key in keys]


def test_policy_prefetch_workers_fit_the_assigned_affinity(monkeypatch):
    """FAILING-BEFORE (PQ #888): the fixture asked for a CPU it was not given.

    ``ProductionWeightCache._window_limits`` refuses more window workers
    than ``os.sched_getaffinity(0)`` holds, so a fixture that names a
    constant makes every suite importing it assert on the shard width. The
    default now fits the mask; an explicit request is still honored, so a
    test that deliberately asks for a wider window keeps its meaning.
    """

    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0})
    assert policy()['prefetch_workers'] == 1
    assert policy(prefetch_workers=2)['prefetch_workers'] == 2

    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 1, 2, 3})
    assert policy()['prefetch_workers'] == FIXTURE_PREFETCH_WORKERS


def test_policy_windows_open_under_a_one_cpu_shard(tmp_path, monkeypatch):
    """The refusal itself: the policy default must open a real window.

    Not a re-statement of the arithmetic above -- this drives the
    production guard the eleven listed tests died on, with the mask the
    one-CPU shard actually has.
    """

    import torch
    from prismaquant.joint_statistics_replay import resident_candidates
    from prismaquant.production_weight_cache import ProductionWeightCache

    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0})
    key = ('unit0', 'TESSERA_E4M3_K1_R1024')
    value = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    path = tmp_path / 'render0.pt'
    torch.save(value, path)
    cache = ProductionWeightCache(weights={key: str(path)}, levers={})
    cache.enable_lru(100000)
    with resident_candidates(cache, (key,), policy()) as iterator:
        assert [window for window, _ in iterator] == [(key,)]
