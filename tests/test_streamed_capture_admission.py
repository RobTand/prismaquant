"""Phase admission must follow the collector and prefetch ownership contracts."""
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from prismaquant.streaming_model import StreamingContext
from test_glm_campaign_streaming import glm_checkpoint


def context(futures, cached=()):
    value = object.__new__(StreamingContext)
    value.num_layers = 4
    value._inflight_lock = Lock()
    value._inflight = futures
    value.layer_cache = SimpleNamespace(peek=lambda layer: layer in cached)
    return value


def test_settlement_awaits_existing_future_without_claiming_or_reloading():
    release = Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: release.wait(timeout=2) and {'source': 'resident'})
        value = context({1: future}, cached={2})
        original_result = future.result
        def result(*args, **kwargs):
            assert not future.done()
            release.set()
            return original_result(*args, **kwargs)
        future.result = result
        records = value.settle_prefetched_layers([1, 2])
        assert records == [{'layer': 1, 'owner': 'prefetch_future'},
                           {'layer': 2, 'owner': 'layer_cache'}]
        assert value._inflight == {1: future}
        assert future.done() and not future.cancelled()


@pytest.mark.parametrize('failure', ['missing', 'refused', 'error', 'unexpected'])
def test_settlement_refuses_an_unproven_loader_boundary(failure):
    futures = {}
    if failure != 'missing':
        future = Future()
        if failure == 'error':
            future.set_exception(RuntimeError('source load failed'))
        else:
            future.set_result(None if failure == 'refused' else {'source': 'resident'})
        futures[3 if failure == 'unexpected' else 1] = future
    value = context(dict(futures))
    with pytest.raises(RuntimeError):
        value.settle_prefetched_layers([1])
    assert value._inflight == futures


@pytest.mark.parametrize('indices', [[-1], [4], [1, 1], [True]])
def test_settlement_rejects_invalid_windows(indices):
    with pytest.raises(ValueError):
        context({}).settle_prefetched_layers(indices)


def test_shared_plan_prices_full_device_prefix_and_each_independent_cpu_output(glm_checkpoint):
    from prismaquant.autoscale import streamed_calibration_resources
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile
    from prismaquant.routed_experts import profile_declared_packed_expert_projections
    model, source = glm_checkpoint
    members = profile_declared_packed_expert_projections(model, Glm5NextProfile())
    shapes = {member.qname: list(member.weight.shape) for member in members}
    counts = {name: 2 for name in shapes}  # deliberately shorter than the buffer
    common = dict(unit_shapes=shapes, counts=counts, nsamples=1, seqlen=2,
                  max_act_rows=7, cache_slots=2, prefetch_workers=1, headroom_gb=1)
    legacy = streamed_calibration_resources(source, **common)
    plan = streamed_calibration_resources(source, **common,
                                          capture_policy='shared-inputs-bounded-v1')
    assert legacy['schema'].endswith('.v1') and plan['schema'].endswith('.v2')
    assert 'terms' in legacy and 'terms' not in plan
    # Derive the actual owners from live packed module handles and expert IDs,
    # independently of the planner's profile/name classification.
    actual = {}
    for member in members:
        actual.setdefault((id(member.module), member.expert_id, member.param_name), []).append(member.qname)
    assert {tuple(sorted(names)) for names in plan['input_groups'].values()} == {
        tuple(sorted(names)) for names in actual.values()}
    device_h = device_x = drain = 0
    for names in actual.values():
        columns = shapes[names[0]][1]
        h, x, output_x = 4*columns**2, 4*7*columns, 4*2*columns
        device_h += h
        device_x += x
        drain += max(h+x, len(names)*(h+output_x))
    forward, output = plan['phases']['forward'], plan['phases']['materialization']
    assert forward['layer_hessian_bytes'] == device_h
    assert forward['layer_prefix_bytes'] == device_x
    assert output['layer_materialization_bytes'] == drain
    assert forward['loader_transient_bytes'] == legacy['terms']['loader_transient_bytes'] > 0
    assert output['loader_transient_bytes'] == 0
    assert output['source_window_bytes'] == max(plan['body_layer_bytes'].values())
    assert plan['memory_bytes'] == max(sum(phase.values()) for phase in plan['phases'].values())
    assert plan['disk_bytes'] == legacy['disk_bytes']
    assert plan['full_hessian_bytes'] == legacy['full_hessian_bytes']


@pytest.fixture
def memory_guard(tmp_path, monkeypatch):
    from prismaquant import memory_management as memory
    gib = 1024**3
    root = tmp_path/'cgroup'
    child = root/'job'
    child.mkdir(parents=True)
    (root/'memory.max').write_text(str(16*gib))
    (root/'memory.current').write_text(str(gib))
    (child/'memory.max').write_text('max')
    membership = tmp_path/'membership'
    membership.write_text('0::/job\n')
    monkeypatch.setattr(memory.torch.cuda, 'memory_reserved', lambda _device: 2*gib)
    monkeypatch.setattr(memory, '_host_memory_info', lambda: (20*gib, 32*gib))
    guard = memory.CaptureMemoryGuard('cuda', cgroup_root=root, membership=membership)
    return guard, root, memory


def test_guard_prices_the_bounded_ancestor_and_the_entire_cuda_reservation(memory_guard):
    guard, root, _memory = memory_guard
    observed = guard.check('start')
    assert observed['conservative_cgroup_plus_cuda_reserved_bytes'] == 3*1024**3
    assert observed['refusal_threshold_bytes'] == 14*1024**3
    assert guard.scope == root
    assert guard.snapshot()['peak_conservative_bytes'] == 3*1024**3


@pytest.mark.parametrize('failure', ['sum', 'host_floor', 'missing', 'tightened_cap'])
def test_guard_latches_physical_refusal(memory_guard, monkeypatch, failure):
    guard, root, memory = memory_guard
    guard.check('start')
    if failure == 'sum':
        (root/'memory.current').write_text(str(13*1024**3))
    elif failure == 'host_floor':
        monkeypatch.setattr(memory, '_host_memory_info', lambda: (7*1024**3, 32*1024**3))
    elif failure == 'missing':
        (root/'memory.current').unlink()
    else:
        (root/'memory.max').write_text(str(4*1024**3))
    with pytest.raises((RuntimeError, OSError)):
        guard.check('refuse')
    (root/'memory.current').write_text('0')
    (root/'memory.max').write_text(str(100*1024**3))
    monkeypatch.setattr(memory, '_host_memory_info', lambda: (20*1024**3, 32*1024**3))
    with pytest.raises(RuntimeError):
        guard.check('must not recover after a latched refusal')


def test_guard_refuses_unbounded_cuda_capture(tmp_path):
    from prismaquant.memory_management import CaptureMemoryGuard
    root = tmp_path/'cgroup'
    root.mkdir()
    membership = tmp_path/'membership'
    membership.write_text('0::/\n')
    with pytest.raises(RuntimeError, match='finite cgroup memory budget'):
        CaptureMemoryGuard('cuda', cgroup_root=root, membership=membership)


def test_actual_collector_refuses_unproven_sharing_before_forward(glm_checkpoint, monkeypatch):
    import torch
    from prismaquant.tessera_campaign import _collect_activations
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile
    model, _source = glm_checkpoint
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and '.mlp.' in name][:2]
    assert len(targets) == 2
    called = []
    with pytest.raises(RuntimeError, match='actual capture input groups'):
        _collect_activations(model, targets, [torch.ones(1, 2, dtype=torch.long)], 1, 'cpu',
            profile=Glm5NextProfile(), want_hessian=True, shared_packed_inputs=True,
            expected_shared_input_groups={targets[0]: targets},
            forward_batch=lambda _batch: called.append(True), on_forwards_complete=lambda: None)
    assert not called
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_guard_reserves_future_allocations_before_they_begin(memory_guard):
    guard, root, _memory = memory_guard
    with pytest.raises(RuntimeError, match='physical memory refusal'):
        guard.check('before growth', reserve_bytes=12*1024**3)
    assert int((root/'memory.current').read_text()) == 1024**3
    assert guard.last['future_allocation_bytes'] == 12*1024**3


def test_projected_source_advice_waits_for_byte_check_and_mapping_release(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from torch.multiprocessing.reductions import StorageWeakRef
    from prismaquant import tessera_campaign as campaign
    from prismaquant import tessera_expert_projection as projection
    from prismaquant import layer_streaming
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file({'expert.weight': weight}, str(tmp_path/'source.safetensors'))
    unit = {'source_tensor': 'expert.weight', 'rows': 3, 'cols': 4}
    bound = {'stack': {'expert': unit}}
    source = {'tensors': {'expert.weight': 'source.safetensors'}}
    original_read = projection.source_unit_weight
    storage = []
    calls = []
    def read(*args, **kwargs):
        value = original_read(*args, **kwargs)
        storage.append(StorageWeakRef(value.untyped_storage()))
        return value
    def advise(path, keys, expected):
        assert all(ref.expired() for ref in storage)
        assert str(path) == str(tmp_path/'source.safetensors')
        assert keys == ['expert.weight']
        assert expected.st_size == (tmp_path/'source.safetensors').stat().st_size
        calls.append(True)
    monkeypatch.setattr(projection, 'source_unit_weight', read)
    monkeypatch.setattr(layer_streaming, '_advise_consumed_safetensors_pages', advise)
    assert campaign._checked_projected_units(bound, weights={'expert': weight},
        model_path=tmp_path, source=source, release_source_pages=True) == {'expert': unit}
    assert calls == [True]
    calls.clear()
    with pytest.raises(RuntimeError, match='disagrees byte-for-byte'):
        campaign._checked_projected_units(bound, weights={'expert': weight+1},
            model_path=tmp_path, source=source, release_source_pages=True)
    assert not calls
