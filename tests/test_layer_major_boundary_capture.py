"""Layer-major baseline capture preserves each original source batch exactly."""
import hashlib
from pathlib import Path
import pytest
import torch

from prismaquant import aura_cost
from prismaquant.cost_streaming import (
    LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts,
    normalize_boundary_storage,
)
from prismaquant.cost_streaming import StreamedCausalLM
from prismaquant.model_profiles.default import DefaultProfile
from test_joint_aura_streamed import _fixture
from test_streamed_boundary_artifacts import _policy
from test_streamed_cost_checkpoints import _DenseTinyLM, _FakeStreamingContext, _model_identity


def policy(path, *, window=2, cap=None):
    return {**_policy(path, window=window, cap=cap or (2*window+1)*256),
            'schema': LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA, 'capture_order': 'layer_major'}


def owner(path, **kwargs):
    value = StreamedBoundaryArtifacts(policy(path, **kwargs))
    value.bind({'fixture': 'layer-major-source'}, n_probes=4)
    return value


def fixture():
    model, context, runner, cache = _fixture()
    install = context.install
    context.install = lambda layer, *, require_prefetched=False, prefetch_following=True: install(
        layer, require_prefetched=require_prefetched)
    runner.require_prefetched_residency = True
    runner.prefetch_lookahead = 1
    return model, context, runner, cache


def draw():
    return torch.tensor([[1,2,3,4], [4,3,2,1], [2,3,4,1], [3,4,1,2], [1,3,2,4]])


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def test_source_install_count_is_one_per_layer_for_the_full_draw(tmp_path):
    _, context, runner, _ = fixture()
    with owner(tmp_path) as storage:
        batches = runner.capture_layer_major_boundaries([row[None] for row in draw()], storage=storage)
        assert len(batches) == 5
        assert all(len(batch.activations_cpu) == runner.num_layers+1 for batch in batches)
    layers = range(runner.num_layers)
    assert context.install_calls == runner.num_layers
    # Each layer is speculated once ahead of its turn and re-asserted once,
    # immediately before its install, through the runner's idempotent
    # schedule_prefetch; nothing at or beyond num_layers is ever scheduled.
    prefetches = [layer for kind, layer in context.events if kind == 'prefetch']
    assert sorted(prefetches) == sorted([*layers, *layers])
    for layer in layers:
        install = context.events.index(('install', layer))
        assert context.events[install - 1] == ('prefetch', layer)
        assert ('prefetch', layer) in context.events[:install - 1]
    assert runner.layer_major_prefetch_retries == ()
    assert context.active == set()


class _RunnerResidency(_FakeStreamingContext):
    """Refuse install the way ensure_loaded does: resident or in flight, else refuse.

    schedule_prefetch mirrors the runner: None for a hot layer, the live read
    for an in-flight one, a fresh read otherwise. `defect` injects the two ways
    a speculative call leaves nothing behind for install to find: the layer
    was hot when speculated and the LRU evicted it before its turn, or the
    pressure floor refused the speculative read outright.
    """

    def __init__(self, model, *, defect):
        super().__init__(model)
        self.defect = defect
        self.resident = {1} if defect == 'evicted' else set()
        self.in_flight = {}
        self.refusals = 0
        self.reads = []

    def schedule_prefetch(self, layer):
        super().schedule_prefetch(layer)
        layer = int(layer)
        if layer in self.resident:
            return None
        if layer in self.in_flight:
            return self.in_flight[layer]
        if self.defect == 'refused' and layer == 1 and self.refusals == 0:
            self.refusals += 1
            return None
        read = object()
        self.in_flight[layer] = read
        self.reads.append(layer)
        return read

    def install(self, layer, *, require_prefetched=False, prefetch_following=True):
        layer = int(layer)
        if require_prefetched and layer not in self.resident and layer not in self.in_flight:
            raise RuntimeError(f'streamed layer {layer} is not resident after its required prefetch')
        self.in_flight.pop(layer, None)
        self.resident.add(layer)
        return super().install(layer, require_prefetched=require_prefetched)

    def unload(self, layer):
        released = super().unload(layer)
        if self.defect == 'evicted' and int(layer) == 0:
            self.resident.discard(1)
        return released


@pytest.mark.parametrize('defect', ['evicted', 'refused'])
def test_visitor_reasserts_speculation_the_runner_no_longer_holds(tmp_path, defect):
    torch.manual_seed(85)
    model = _DenseTinyLM().eval()
    for layer in model.model.layers:
        layer._fixture_requires_stream_residency = True
    context = _RunnerResidency(model, defect=defect)
    runner = StreamedCausalLM(context, DefaultProfile(), prefetch_lookahead=1,
                              require_prefetched_residency=True)
    with owner(tmp_path) as storage:
        batches = runner.capture_layer_major_boundaries([row[None] for row in draw()], storage=storage)
        assert len(batches) == 5
        assert all(len(batch.activations_cpu) == runner.num_layers+1 for batch in batches)
    assert context.install_calls == runner.num_layers
    assert context.install_require_prefetched == [True] * runner.num_layers
    assert runner.layer_major_prefetch_retries == (1,)
    assert context.reads.count(1) == 1
    assert context.active == set()


@pytest.mark.parametrize('window', [1, 2, 3])
def test_per_batch_source_tensors_costs_and_cotangents_are_exact(tmp_path, monkeypatch, window):
    written = []
    original_write = StreamedBoundaryArtifacts.write
    def write(self, tensor, **kwargs):
        coordinates = tuple(kwargs.get(key) for key in ('probe_index','batch_index','boundary_index'))
        written.append((coordinates, digest(tensor)))
        return original_write(self, tensor, **kwargs)
    monkeypatch.setattr(StreamedBoundaryArtifacts, 'write', write)
    def run(layer_major):
        written.clear()
        _, context, runner, cache = fixture()
        calls = []
        source = runner._call
        def call(layer, hidden, *, batch, pass_state):
            before = digest(hidden)
            output = source(layer, hidden, batch=batch, pass_state=pass_state)
            calls.append((tuple(batch.input_ids.flatten().tolist()), layer,
                          torch.is_grad_enabled(), before, digest(output)))
            return output
        runner._call = call
        settings = policy(tmp_path/str(layer_major), window=window)
        if not layer_major:
            settings = _policy(tmp_path/'baseline', window=window, cap=(2*window+1)*256)
        result = aura_cost.compute_aura_cost_streamed(runner, draw(),
            ['FP8_DYNAMIC','NVFP4A16','BF16'], n_probes=4, probe_microbatch=1,
            seed_base=7000, min_free_gib=0, production_cache=cache,
            joint_activation=True, collect_col_energy=True,
            model_identity=_model_identity('layer-major-source'), boundary_storage=settings)
        assert context.active == set()
        return result, calls, list(written), context.install_calls
    before, old_calls, old_written, old_installs = run(False)
    after, new_calls, new_written, new_installs = run(True)
    assert before['costs'] == after['costs']
    assert old_installs == 12 and new_installs == 4  # Capture plus unchanged reverse.
    assert [x for x in old_calls if x[2]] == [x for x in new_calls if x[2]]
    assert old_calls[:10] != new_calls[:10]  # Intentional global traversal change.
    for row in map(tuple, draw().tolist()):
        assert [x for x in old_calls if x[0] == row] == [x for x in new_calls if x[0] == row]
    assert dict(old_written) == dict(new_written)
    assert [x for x in old_written if x[0][0] is not None] == [x for x in new_written if x[0][0] is not None]
    for name, stats in after['stats'].items():
        assert stats['h_trace'] == before['stats'][name]['h_trace']
        assert torch.equal(stats['fisher_col'], before['stats'][name]['fisher_col'])
    assert after['provenance']['probe_identity'] == before['provenance']['probe_identity']


@pytest.mark.parametrize('stage', ['prepare', 'forward', 'new_pass', 'capture_pass'])
def test_layer_major_refuses_observed_torch_rng_consumption_and_cleans(tmp_path, stage):
    _, context, runner, _ = fixture()
    obj, name = {'prepare': (runner, '_prepare'), 'forward': (runner, '_call'),
                 'new_pass': (runner.profile, 'new_forward_pass_state'),
                 'capture_pass': (runner.profile, 'capture_forward_pass_state')}[stage]
    original = getattr(obj, name)
    def consuming(*args, **kwargs):
        torch.rand(1)
        return original(*args, **kwargs)
    setattr(obj, name, consuming)
    with pytest.raises(RuntimeError, match='RNG consumption'):
        with owner(tmp_path) as storage:
            runner.capture_layer_major_boundaries([draw()[0:1]], storage=storage)
    assert context.active == set()
    assert not list(tmp_path.rglob('*.pt'))


@pytest.mark.parametrize('defect', ['training', 'nonresident', 'window_budget', 'corrupt'])
def test_capture_refuses_invalid_execution_before_unchecked_forward(tmp_path, defect):
    model, context, runner, _ = fixture()
    storage = owner(tmp_path, cap=256 if defect == 'window_budget' else None)
    if defect == 'training':
        model.train()
    if defect == 'nonresident':
        runner.require_prefetched_residency = False
    if defect == 'corrupt':
        install = context.install
        def corrupt(*args, **kwargs):
            install(*args, **kwargs)
            Path(next(iter(storage._references.values())).path).unlink()
        context.install = corrupt
    with pytest.raises((RuntimeError, FileNotFoundError)):
        with storage:
            runner.capture_layer_major_boundaries([row[None] for row in draw()], storage=storage)
    assert context.active == set()
    assert not list(tmp_path.rglob('*.pt'))


@pytest.mark.parametrize('defect', ['missing', 'repeat', 'reorder'])
def test_existing_visitor_checks_apply_to_exact_windows(tmp_path, defect):
    _, context, runner, _ = fixture()
    ids = [row[None] for row in draw()]
    def visit(_layer, forward):
        selected = ids[:-1] if defect == 'missing' else ids+ids[:1] if defect == 'repeat' else ids[::-1]
        for batch in selected:
            forward(batch)
    with pytest.raises(RuntimeError, match='visitor'):
        with owner(tmp_path) as storage:
            runner.visit_layer_batches(ids, visit, boundary_storage=storage)
    assert context.active == set()


def test_v2_policy_is_closed_and_v1_default_cannot_change_order(tmp_path):
    good = policy(tmp_path)
    assert normalize_boundary_storage(good) == good
    for altered in ({**good, 'capture_order': 'batch_major'},
                    {key:value for key,value in good.items() if key != 'capture_order'},
                    {**_policy(tmp_path), 'capture_order': 'layer_major'}):
        with pytest.raises(ValueError):
            normalize_boundary_storage(altered)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA token-device regression')
def test_original_cpu_tokens_compare_with_prepared_cuda_tokens(tmp_path):
    _, context, runner, _ = fixture()
    prepare = runner._prepare
    def prepare_cuda_ids(ids):
        prepared, positions, hidden, embeddings, mask = prepare(ids)
        return prepared.cuda(), positions, hidden, embeddings, mask
    runner._prepare = prepare_cuda_ids
    with owner(tmp_path) as storage:
        batches = runner.capture_layer_major_boundaries([draw()[0:1]], storage=storage)
        assert len(batches) == 1
    assert context.active == set()


def test_actual_shared_state_profile_keeps_every_batch_and_shared_adjoint(tmp_path, monkeypatch):
    from test_streamed_boundary_artifacts import _shared_run
    seen = []
    original_write = StreamedBoundaryArtifacts.write
    def write(self, tensor, **kwargs):
        if kwargs.get('probe_index') is not None:
            seen.append(((kwargs['probe_index'], kwargs['batch_index'], kwargs['boundary_index']), digest(tensor)))
        return original_write(self, tensor, **kwargs)
    monkeypatch.setattr(StreamedBoundaryArtifacts, 'write', write)
    before = _shared_run(tmp_path/'before')
    original = list(seen)
    seen.clear()
    after = _shared_run(tmp_path/'after', layer_major=True)
    assert after['costs'] == before['costs']
    assert seen == original
    telemetry = after['provenance']['streamed_boundary_storage']['telemetry']
    assert telemetry['peak_shared_cotangent_reservation_bytes'] > 0
    with pytest.raises(RuntimeError, match='auxiliary/shared-state'):
        _shared_run(tmp_path/'refused', layer_major=True, aux=1024)
    assert not list(tmp_path.rglob('*.pt'))
