"""The Stage A loader barrier retries landed speculation before capture (#911).

CPU/meta fixture, real StreamingContext, real layer-major visitor and Stage A
phase observer, real strict reader and installed PB lease SDK. Only mover
publication timing is controlled; no injected availability exception.
"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.cost_streaming import StreamedCausalLM
from prismaquant.joint_cost_stage_a import stage_a_forward_observer
from prismaquant.layer_streaming import LayerCache, _build_install_resolver, _unload
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.streaming_model import StreamingContext
from prismaquant.staged_tier_policy import StagedRangeNotLanded
from test_layer_major_boundary_capture import draw, owner
from test_streamed_cost_checkpoints import _DenseTinyLM
from test_strict_reader_tier_enforcement import (
    _forget_state, _mid_flight_fixture, _pins_live, _stage_root,
    _stage_whole, _whole_file, STAGED_RANGE_WAIT_ENV,
)


def test_stage_a_settles_landed_speculation_inside_loading_phase(tmp_path, monkeypatch):
    torch.manual_seed(85)
    model = _DenseTinyLM().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    expected = {index: layer.proj.weight.detach().clone()
                for index, layer in enumerate(model.model.layers)}

    paths, staged, weights = {}, {}, {}
    for index, layer in enumerate(model.model.layers):
        name = f'model.layers.{index}.proj.weight'
        path = tmp_path / f'layer-{index}.safetensors'
        save_file({name: layer.proj.weight.detach().clone()}, str(path))
        paths[index] = path
        staged[index] = _stage_whole(_stage_root(tmp_path), path)
        weights[name] = str(path)
    resolver, consumer, _digest, publish = _mid_flight_fixture(
        tmp_path, monkeypatch, declared=[row for path in paths.values()
                                       for row in _whole_file(path)])
    publish({'0': (paths[0], staged[0])})
    # A zero wait proves declared-but-unlanded refusal cheaply; campaign
    # production wait/progress allowances are not changed by this fixture.
    monkeypatch.setenv(STAGED_RANGE_WAIT_ENV, '0')
    context = StreamingContext(
        model=model, base_model=model.model, layers=model.model.layers,
        layers_prefix='model.layers.', num_layers=2,
        install_resolvers=[_build_install_resolver(model, f'model.layers.{i}')
                           for i in range(2)],
        weight_shard=weights, weight_ckpt={name: name for name in weights},
        layer_cache=LayerCache(max_bytes=1 << 20, max_entries=4),
        prefetch_pool=ThreadPoolExecutor(max_workers=1),
        device=torch.device('cpu'), dtype=torch.float32,
        offload_folder=str(tmp_path / 'offload'), estimated_layer_bytes=1024,
        prefetch_workers=1)
    _unload(model, ['model.layers.'])
    runner = StreamedCausalLM(context, DefaultProfile(), prefetch_lookahead=1,
                              require_prefetched_residency=True)
    phases, reads, failed = [], [], []
    active = {'phase': None}
    worker = context._prefetch_worker

    def observed_worker(index):
        assert active['phase'][0] == 'source_loading'
        reads.append(index)
        return worker(index)

    monkeypatch.setattr(context, '_prefetch_worker', observed_worker)
    schedule = context.schedule_prefetch

    def land_after_speculation(index):
        future = schedule(index)
        if index == 1 and not failed:
            with pytest.raises(StagedRangeNotLanded):
                future.result(timeout=30)
            failed.append(future)
            publish({str(i): (paths[i], staged[i]) for i in paths})
        return future

    monkeypatch.setattr(context, 'schedule_prefetch', land_after_speculation)
    progress_phases = []
    observe = stage_a_forward_observer(SimpleNamespace(
        enter=progress_phases.append, flush=lambda **_kwargs: None))

    class ReachedCapture(Exception):
        """Stop at the certified load boundary, before activation I/O/compute."""

    def source_phase(stage, layer, auxiliary):
        active['phase'] = (stage, layer)
        phases.append((stage, layer))
        observe(stage, layer, auxiliary)
        if (stage, layer) == ('capture_forward', 0):
            # Settlement must retain the replacement delivery future and
            # cache pin, not consume them as ensure_loaded would.
            delivered = context._inflight[1]
            assert delivered is not failed[0]
            assert delivered.done() and delivered.result()
            assert torch.equal(delivered.result()['model.layers.1.proj.weight'], expected[1])
            assert 1 in context.layer_cache._pinned_until_read
            assert model.model.layers[1].proj.weight.is_meta
            assert torch.equal(model.model.layers[0].proj.weight, expected[0])
            raise ReachedCapture()

    try:
        with pytest.raises(ReachedCapture):
            with owner(tmp_path / 'boundaries') as storage:
                runner.capture_layer_major_boundaries(
                    [draw()[0:1]], storage=storage, source_phase=source_phase)
        assert phases == [('source_loading', 0), ('capture_forward', 0)]
        assert progress_phases == ['forward-000', 'forward-000']
        # A later consumer claims precisely those already checked tensors;
        # settlement itself must not spend either owner or pin.
        context.install(1, require_prefetched=True, prefetch_following=False)
        assert torch.equal(model.model.layers[1].proj.weight, expected[1])
        assert 1 not in context._inflight
        assert 1 not in context.layer_cache._pinned_until_read
        assert reads == [0, 1, 1]
        assert resolver.report()['bytes_from_pool'] == 0
        assert resolver.report()['bytes_from_stage'] > 0
        assert _pins_live(tmp_path, consumer) == []
    finally:
        context.shutdown()


@pytest.mark.parametrize('failure', ['integrity', 'retiring', 'unknown', 'cancelled'])
def test_loader_opt_in_never_retries_other_failures(monkeypatch, failure):
    from concurrent.futures import CancelledError
    from prismaquant.staged_lease import LeaseRefused
    from test_stagea_speculative_availability_retry import _make_ctx

    errors = {'integrity': LeaseRefused('digest-changed', kind='integrity'),
              'retiring': LeaseRefused('retiring', kind='availability'),
              'unknown': RuntimeError('readset-not-staged'),
              'cancelled': CancelledError('owned wait cancelled')}
    reads = []

    def reader(*args, **kwargs):
        reads.append(3)
        raise errors[failure]

    context = _make_ctx(monkeypatch, reader)
    try:
        future = context.schedule_prefetch(3)
        with pytest.raises(type(errors[failure])):
            context.settle_prefetched_layers([3], retry_availability=True)
        assert context._inflight[3] is future
        assert reads == [3]
    finally:
        context.shutdown()


def test_settlement_and_later_demand_share_one_retry_budget(monkeypatch):
    from test_stagea_speculative_availability_retry import _make_ctx
    reads = []

    def reader(*args, **kwargs):
        reads.append(3)
        raise StagedRangeNotLanded('declared-source', 0, 1)

    context = _make_ctx(monkeypatch, reader)
    try:
        first = context.schedule_prefetch(3)
        for consume in (
            lambda: context.settle_prefetched_layers([3], retry_availability=True),
            lambda: context.settle_prefetched_layers([3], retry_availability=True),
            lambda: context.ensure_loaded(3, require_prefetched=True),
        ):
            with pytest.raises(StagedRangeNotLanded):
                consume()
        assert context._inflight[3] is not first
        assert reads == [3, 3]
    finally:
        context.shutdown()


@pytest.mark.parametrize('condition', ['default', 'cancelled', 'admission_refused'])
def test_a_failed_owner_stays_bounded_when_retry_is_not_available(monkeypatch, condition):
    from concurrent.futures import CancelledError
    from threading import Event
    from test_stagea_speculative_availability_retry import _make_ctx
    reads = []

    def reader(*args, **kwargs):
        reads.append(3)
        raise StagedRangeNotLanded('declared-source', 0, 1)

    context = _make_ctx(monkeypatch, reader)
    context._staged_wait_cancel = Event()
    try:
        future = context.schedule_prefetch(3)
        with pytest.raises(StagedRangeNotLanded):
            future.result()
        scheduled = []
        if condition == 'cancelled':
            context._staged_wait_cancel.set()
        if condition == 'admission_refused':
            monkeypatch.setattr(context, 'schedule_prefetch',
                                lambda layer: scheduled.append(layer))
        for _ in range(2):
            with pytest.raises(CancelledError if condition == 'cancelled'
                               else StagedRangeNotLanded):
                context.settle_prefetched_layers(
                    [3], retry_availability=condition != 'default')
        assert context._inflight[3] is future
        assert reads == [3]
        assert scheduled == ([3] if condition == 'admission_refused' else [])
    finally:
        context.shutdown()


def test_stage_a_core_opts_in_at_both_forward_and_reverse_loader_barriers(tmp_path):
    """Full CPU core wiring; strict source byte/owner proof is the test above."""
    from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
    from prismaquant.joint_layer_quanta import adjoint_read_plan_phase_names
    from prismaquant.joint_run_progress import JointRunProgress
    from test_joint_cost_quantum_runtime import _execution
    from test_layer_major_boundary_capture import fixture
    from test_streamed_cost_checkpoints import _model_identity

    _model, context, runner, _cache = fixture()
    progress = JointRunProgress(
        layers=runner.num_layers, partitions=1, base_units=0,
        log=lambda _message: None, commit=lambda _phase, _units: True,
        phases=adjoint_read_plan_phase_names(runner.num_layers))
    settled = []

    def settle(indices, *, retry_availability=False):
        assert retry_availability is True
        settled.append((progress.phase, tuple(indices)))

    context.settle_prefetched_layers = settle
    run_adjoint_capture_core(
        runner, draw()[0:1], execution=_execution(tmp_path),
        output_root=tmp_path / 'campaign', stride=2,
        source_model_identity=_model_identity('joint-source'),
        unit_roster_sha256='a' * 64, plan_sha256='d' * 64,
        prepared_sha256='e' * 64, read_manifest_sha256='f' * 64,
        implementation_sha256='0' * 64, progress=progress)
    assert settled == [('forward-000', (1,)), ('forward-001', ()),
                       ('chain-001', (0,)), ('chain-000', ())]


def test_retry_releases_failed_loader_temporaries_before_allocating_again(monkeypatch):
    import weakref
    from test_stagea_speculative_availability_retry import _make_ctx, _tensors
    partial = []

    def reader(*args, **kwargs):
        if not partial:
            # A multi-shard read can populate part of its output before a
            # later declared shard refuses. Future.exception's traceback
            # then owns this tensor even after the worker has finished.
            payload = torch.ones(4096)
            partial.append(weakref.ref(payload))
            raise StagedRangeNotLanded('later-shard', 0, 1)
        assert partial[0]() is None, 'retry overlaps failed loader tensor storage'
        return _tensors(3)

    context = _make_ctx(monkeypatch, reader)
    try:
        first = context.schedule_prefetch(3)
        with pytest.raises(StagedRangeNotLanded):
            first.result()
        assert partial[0]() is not None  # the old future really holds it
        context.settle_prefetched_layers([3], retry_availability=True)
        assert context._inflight[3] is not first
        assert partial[0]() is None
    finally:
        context.shutdown()
