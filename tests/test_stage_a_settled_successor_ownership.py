"""A settled chain successor stays owned until its install claims it (#1124).

R13 (PB 03f50d8e390b) stopped at chain-042 with "streamed layer 42 is not
resident after its required prefetch". The forward walk left layers 42..44
in the LayerCache as ordinary read entries. At chain-043 the chain scheduled
its successors [42, 41]: ``schedule_prefetch(42)`` returned None for the
resident layer, so nothing owned it, and settlement accepted it on a cache
peek. Layer 41's read then made room under the three-entry cap by evicting
the least recently used unpinned entry, which was 42, the settled successor,
and not 44, the layer the chain had just finished.

CPU fixture: a real StreamingContext and LayerCache over one safetensors file
per layer, and the real Stage A core with operator windows (the settled
branch) and its real roll. Only the prefetch pool is controlled: each read
runs when its future is first awaited. That is the order CPython gave R13,
where settlement peeks the resident successor before it awaits the sibling
read.
"""
from concurrent.futures import Future, ThreadPoolExecutor
import sys

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.cost_streaming import StreamedCausalLM
from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
from prismaquant.joint_layer_quanta import adjoint_read_plan_phase_names
from prismaquant.joint_run_progress import JointRunProgress
from prismaquant.layer_streaming import LayerCache, _build_install_resolver, _unload
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.streaming_model import StreamingContext
from test_joint_cost_quantum_runtime import _execution
from test_layer_major_boundary_capture import draw
from test_streamed_cost_checkpoints import _DenseTinyLM, _model_identity

LAYERS = 4


class _AwaitedPool:
    """Run each submitted read when its future is first awaited."""

    def submit(self, fn, *args):
        future = _AwaitedFuture(fn, args)
        return future

    def shutdown(self, wait=True):
        return None


class _AwaitedFuture(Future):
    def __init__(self, fn, args):
        super().__init__()
        self._call = (fn, args)

    def result(self, timeout=None):
        call, self._call = self._call, None
        if call is not None and self.set_running_or_notify_cancel():
            fn, args = call
            try:
                self.set_result(fn(*args))
            except BaseException as exc:
                self.set_exception(exc)
        return super().result(timeout)


def _streamed(tmp_path, pool, *, layers=LAYERS, slots=3, lookahead=2):
    torch.manual_seed(1124)
    model = _DenseTinyLM(layers=layers).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weights = {}
    for index, layer in enumerate(model.model.layers):
        name = f'model.layers.{index}.proj.weight'
        path = tmp_path / f'layer-{index}.safetensors'
        save_file({name: layer.proj.weight.detach().clone()}, str(path))
        weights[name] = str(path)
    cache = LayerCache(max_bytes=1 << 20, max_entries=slots)
    context = StreamingContext(
        model=model, base_model=model.model, layers=model.model.layers,
        layers_prefix='model.layers.', num_layers=layers,
        install_resolvers=[_build_install_resolver(model, f'model.layers.{i}')
                           for i in range(layers)],
        weight_shard=weights, weight_ckpt={name: name for name in weights},
        layer_cache=cache, prefetch_pool=pool,
        device=torch.device('cpu'), dtype=torch.float32,
        offload_folder=str(tmp_path / 'offload'), estimated_layer_bytes=1024,
        prefetch_workers=3)
    _unload(model, ['model.layers.'])
    reads = []
    worker = context._prefetch_worker

    def read(index):
        reads.append(index)
        return worker(index)

    context._prefetch_worker = read
    runner = StreamedCausalLM(context, DefaultProfile(), prefetch_lookahead=lookahead,
                              require_prefetched_residency=True)
    return model, context, cache, runner, reads


def _progress(runner):
    return JointRunProgress(
        layers=runner.num_layers, partitions=1, base_units=0,
        log=lambda _message: None, commit=lambda _phase, _units: True,
        phases=adjoint_read_plan_phase_names(runner.num_layers))


def _record_evictions(cache, progress):
    """Each eviction victim, with the cache method that chose it and the phase."""
    victims = []
    pick = cache._pick_evict_candidate

    def recorded():
        victim = pick()
        victims.append((sys._getframe(1).f_code.co_name, progress.phase, victim))
        return victim

    cache._pick_evict_candidate = recorded
    return victims


def _run_core(tmp_path, runner, progress):
    return run_adjoint_capture_core(
        runner, draw()[0:1], execution=_execution(tmp_path),
        output_root=tmp_path / 'campaign', stride=2,
        source_model_identity=_model_identity('joint-source'),
        unit_roster_sha256='a' * 64, plan_sha256='d' * 64,
        prepared_sha256='e' * 64, read_manifest_sha256='f' * 64,
        implementation_sha256='0' * 64, progress=progress)


def test_a_settled_resident_successor_survives_its_siblings_read(tmp_path):
    _model, context, cache, runner, reads = _streamed(tmp_path, _AwaitedPool())
    progress = _progress(runner)
    victims = _record_evictions(cache, progress)
    try:
        try:
            _run_core(tmp_path, runner, progress)
        except RuntimeError as exc:
            pytest.fail(f'{exc}; evictions={victims}; reads={reads}')
        # The forward reads each layer once; the chain reads only layer 0,
        # the one layer the forward walk's three-entry cache no longer held.
        assert reads == [0, 1, 2, 3, 0]
        # Layer 0's read at chain-002 takes layer 3, which the chain has
        # finished, never a layer the chain has yet to install.
        chain = [(method, phase, victim) for method, phase, victim in victims
                 if phase.startswith('chain-')]
        assert chain == [('prepare_for_load', 'chain-002', 3)]
        assert cache.evicted_pinned == 0
        assert cache.pressure_evictions == 0
        assert context.prefetch_delivered_unretained == 0
        assert context._inflight == {}
    finally:
        context.shutdown()


def test_a_threaded_pool_installs_every_chain_layer_from_one_read(tmp_path):
    _model, context, cache, runner, reads = _streamed(
        tmp_path, ThreadPoolExecutor(max_workers=3))
    try:
        _run_core(tmp_path, runner, _progress(runner))
        assert sorted(reads) == [0, 0, 1, 2, 3]
        assert context._inflight == {}
    finally:
        context.shutdown()


def test_scheduling_a_resident_layer_owns_it_through_any_cache_drop(tmp_path):
    model, context, cache, _runner, reads = _streamed(tmp_path, _AwaitedPool(), layers=2)
    try:
        assert context.install(0, prefetch_following=False) == 'cold'
        expected = model.model.layers[0].proj.weight.detach().clone()
        context.unload(0)
        assert cache.peek(0) and 0 not in cache._pinned_until_read
        owner = context.schedule_prefetch(0)
        # The schedule takes ownership of the resident bytes and pins them,
        # so eviction prefers any other entry.
        assert owner is not None and owner.done()
        assert context._inflight == {0: owner}
        assert 0 in cache._pinned_until_read
        assert context.schedule_prefetch(0) is owner
        # A pressure trim spends pins, but not the owner.
        cache.configure_pressure_threshold(1 << 62)
        cache.trim_for_memory_pressure()
        assert not cache.peek(0)
        cache.configure_pressure_threshold(0)
        assert context.install(0, require_prefetched=True, prefetch_following=False) == 'wait'
        assert torch.equal(model.model.layers[0].proj.weight, expected)
        assert context.prefetch_delivered_unretained == 1
        assert context._inflight == {}
        assert reads == []
    finally:
        context.shutdown()


def test_the_delivery_lever_keeps_the_old_resident_schedule(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_PREFETCH_DELIVERY', '0')
    _model, context, cache, _runner, _reads = _streamed(tmp_path, _AwaitedPool(), layers=2)
    try:
        context.install(0, prefetch_following=False)
        assert context.schedule_prefetch(0) is None
        assert context._inflight == {} and 0 not in cache._pinned_until_read
    finally:
        context.shutdown()


def test_each_chain_boundary_logs_the_prefetch_counters(tmp_path, capsys):
    _model, context, _cache, runner, _reads = _streamed(tmp_path, _AwaitedPool())
    try:
        _run_core(tmp_path, runner, _progress(runner))
    finally:
        context.shutdown()
    lines = [line for line in capsys.readouterr().out.splitlines()
             if 'chain layer' in line and 'Prefetch:' in line]
    assert [line.split('chain layer ', 1)[1].split(' ', 1)[0] for line in lines] == [
        '3', '2', '1', '0']
    for field in ('mem_skips=', 'released_stale=', 'delivered_unretained=',
                  'pressure_evictions=', 'evicted_pinned='):
        assert all(field in line for line in lines)
