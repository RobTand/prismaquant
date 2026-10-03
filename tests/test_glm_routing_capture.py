"""CPU source/traversal fixtures; these are not GLM or GPU measurements."""
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from prismaquant.cost_streaming import StreamedCausalLM
from prismaquant.glm_routing_capture import capture_streamed_glm_routes, visit_routed_boundaries
from prismaquant.model_profiles.default import DefaultProfile
from prismaquant.streaming_model import _StreamingInitializationAudit
from prismaquant.tessera_calibration_cache import CaptureSourceAuthentication
from test_native_prefix_intake import fixture


class TinyGlmProfile(DefaultProfile):
    @property
    def name(self):
        return 'glm5_next'


class Glm5NextTextExperts(nn.Module):
    # Tiny arithmetic at the native panel's real input/route geometry. There
    # are no expert matrices: this test measures traversal, never expert cost.
    num_experts = 288
    hidden_dim = 4096
    intermediate_dim = 2048
    swiglu_limit = 10.0

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states + 0.125


class Glm5NextTextTopkRouter(nn.Module):
    top_k = 8
    num_group = topk_group = 1
    norm_topk_prob = True
    routed_scaling_factor = 2.5

    def __init__(self):
        super().__init__()
        self.e_score_correction_bias = nn.Parameter(torch.zeros(288), requires_grad=False)

    def forward(self, hidden):
        weights = torch.ones(hidden.shape[0], 8, dtype=torch.float32)
        denominator = weights.sum(-1, keepdim=True) + 1e-20
        ids = torch.arange(8).expand(hidden.shape[0], 8)
        return ids, weights / denominator * self.routed_scaling_factor


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Glm5NextTextTopkRouter()
        self.experts = Glm5NextTextExperts()


class Layer(nn.Module):
    def __init__(self, routed):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), dtype=torch.bfloat16), requires_grad=False)
        if routed:
            self.mlp = TinyMlp()

    def forward(self, hidden_states, **kwargs):
        hidden = hidden_states * self.weight
        if hasattr(self, 'mlp'):
            flat = hidden.reshape(-1, hidden.shape[-1])
            ids, weights = self.mlp.gate(flat)
            hidden = self.mlp.experts(flat, ids, weights).reshape_as(hidden)
        return hidden


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(layer_types=())
        self.embed_tokens = nn.Embedding(2, 4096, dtype=torch.bfloat16)
        self.norm = nn.LayerNorm(4096, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([Layer(i >= 3) for i in range(45)])


class TinyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = TinyBase()


class Glm5NextForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation='eager',
            to_dict=lambda: {'fixture': True},
            text_config=SimpleNamespace(n_shared_experts=1, scoring_func='sigmoid',
                topk_method='noaux_tc', hidden_act='silu'))
        self.model = TinyWrapper()
        self.lm_head = nn.Linear(4096, 2, dtype=torch.bfloat16)
        self.eval()

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class Context:
    """A two-slot source fixture using the real first-use authentication owner."""
    def __init__(self, model, owner, paths):
        self.model, self.owner, self.paths = model, owner, paths
        self.base_model = model.model.language_model
        self.layers = self.base_model.layers
        self.layers_prefix = 'model.language_model.layers.'
        self.num_layers = len(self.layers)
        self.device, self.dtype = torch.device('cpu'), torch.bfloat16
        self.weight_ckpt = {name: name for name, _ in model.named_parameters()}
        self.weight_shard = {name: str(paths[0]) for name in self.weight_ckpt}
        self.install_resolvers = [{name: None for name in self.weight_ckpt
                                  if name.startswith(f'{self.layers_prefix}{i}.')}
                                 for i in range(45)]
        self.ready, self.events = {}, []
        self.max_ready = 0

    def begin_source_initialization_audit(self):
        self.audit = _StreamingInitializationAudit(self)

    def source_prefix_initialization_contract(self, layer):
        return self.audit.complete_prefix(layer)

    def source_initialization_contract(self):
        return self.audit.complete()

    def schedule_prefetch(self, layer):
        if layer >= self.num_layers or layer in self.ready:
            return
        with self.owner.safe_open(safe_open, self.paths[layer], framework='pt', device='cpu') as reader:
            self.ready[layer] = reader.get_tensor('weight')
        self.events.append(('verified', layer))
        self.max_ready = max(self.max_ready, len(self.ready))
        assert self.max_ready <= 2

    def install(self, layer, *, require_prefetched=False, **kwargs):
        assert require_prefetched and layer in self.ready
        self.layers[layer].weight.data.copy_(self.ready.pop(layer))
        self.audit.observe_layer(layer, self.install_resolvers[layer])
        self.events.append(('install', layer))

    def unload(self, layer):
        self.events.append(('unload', layer))


@pytest.fixture
def source(tmp_path):
    import hashlib

    paths = [tmp_path / f'layer-{i:03d}.safetensors' for i in range(45)]
    for path in paths:
        save_file({'weight': torch.ones((), dtype=torch.bfloat16)}, str(path))
    expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    identity = {'source_files': expected, 'census_sha256': 'b' * 64}
    producer = {'files': expected, 'tensors': {'weight': paths[0].name},
                'config_sha256': 'c' * 64, 'auxiliary_sha256': {}}
    owner = CaptureSourceAuthentication(tmp_path, identity, producer, manifest_sha256='a' * 64)
    context = Context(Glm5NextForConditionalGeneration(), owner, paths)
    profile = TinyGlmProfile()
    runner = StreamedCausalLM(context, profile, prefetch_lookahead=1,
                              require_prefetched_residency=True)
    yield runner, producer
    owner.close()


def test_single_traversal_captures_all_owners_before_reading_late_source(source):
    runner, producer = source
    acquisition, _ = fixture()
    acquisition = {key: acquisition[key] for key in
                   ('dev_uncertified', 'dev_mode', 'source_cache_reuse')}
    tokens = torch.zeros(512, 512, dtype=torch.int64)
    calibration = {'shape': [512, 512], 'dtype': 'torch.int64', 'calibration_sha256': 'e' * 64}
    captured = []
    expected = runner.base_model.embed_tokens(tokens[:1]).detach().reshape(512, 4096).clone()

    def consume(layer, result):
        context = runner.context
        assert ('verified', layer) in context.events
        # No complete-checkpoint payload barrier: first capture happens while
        # late source entries remain unread. Read-ahead is bounded to one.
        assert not any(kind == 'verified' and index > layer + 1 for kind, index in context.events)
        metadata = result['metadata']
        if layer == 44:
            assert metadata['model_load_contract']['num_layers'] == 45
        else:
            assert metadata['model_load_contract']['observed_layers'] == list(range(layer + 1))
        x = result['tensors']['inputs']
        assert x.shape == (512, 4096) and x.dtype == torch.bfloat16
        assert torch.equal(x, expected)  # Includes every original preceding expert forward.
        expected.add_(0.125)
        assert result['tensors']['coordinates'][:, 0].eq(0).all()
        assert metadata['replay']['layer'] == layer
        captured.append(layer)

    assert capture_streamed_glm_routes(runner, tokens, calibration=calibration,
        producer_source=producer, source_acquisition=acquisition, consume=consume) == list(range(3, 45))
    assert captured == list(range(3, 45))
    assert sum(kind == 'install' for kind, _ in runner.context.events) == 45
    assert runner.context.max_ready <= 2
    assert len(runner.context.owner.receipt()['verified_files']) == 45
    assert all(not m._forward_pre_hooks for m in runner.model.modules())


def test_corrupt_later_entry_refuses_before_its_consumer(source):
    runner, _ = source
    path = runner.context.paths[9]
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    runner.context.begin_source_initialization_audit()
    consumed = []
    with pytest.raises(RuntimeError, match='content differs'):
        visit_routed_boundaries(runner, torch.zeros(1, 512, dtype=torch.int64),
                                 lambda layer, *_: consumed.append(layer))
    assert consumed and max(consumed) < 9
    assert all(not m._forward_pre_hooks for m in runner.model.modules())


def test_consumer_failure_unhooks_and_releases_current_layer(source):
    runner, _ = source
    runner.context.begin_source_initialization_audit()

    def refuse(*args):
        raise ValueError('consumer refusal')

    with pytest.raises(ValueError, match='consumer refusal'):
        visit_routed_boundaries(runner, torch.zeros(1, 512, dtype=torch.int64), refuse)
    assert runner.context.events[-1] == ('unload', 3)
    assert all(not m._forward_pre_hooks for m in runner.model.modules())


def test_cold_fallback_is_not_allowed(source):
    runner, _ = source
    runner.require_prefetched_residency = False
    with pytest.raises(ValueError, match='prefetched'):
        visit_routed_boundaries(runner, torch.zeros(1, 512, dtype=torch.int64), lambda *_: None)


def test_route_record_records_the_observed_tensor_device():
    """The routing record must carry the device the boundary tensors were
    observed on, not the runner's declared device (#1536).

    ``joint_cost_quantum.build_quantum_source_runner`` builds the streamed
    runner with ``torch.device("cuda")``, so ``str(runner.device)`` is the
    unindexed ``"cuda"``, while ``validate_glm_routing`` requires the indexed
    device a real tensor reports.  The first real GPU capture refused at
    its first published boundary on exactly that string.  The device is
    observed inside ``select_original_routes`` BEFORE its host copy, so the
    record still names where the boundary actually ran even though the
    captured tensors travel on the host.  This CPU test drives the real
    capture path; the CUDA distinction is covered by the CUDA test below.
    """
    from prismaquant.glm_routing_replay import glm_route_record, select_original_routes

    model = Glm5NextForConditionalGeneration()
    mlp = model.model.language_model.layers[3].mlp
    inputs = torch.zeros(512, 4096, dtype=torch.bfloat16)
    captured = select_original_routes(
        mlp.experts, (inputs,
                      torch.zeros(512, 8, dtype=torch.int64),
                      torch.zeros(512, 8, dtype=torch.float32)),
        {}, sequence_length=512)
    assert captured['observed_device'] == str(inputs.device)
    observed = captured['observed_device']
    runner = SimpleNamespace(device=torch.device('cuda'), model=model)
    record = glm_route_record(runner, mlp.experts, mlp.gate, captured, layer=3,
        calibration={'calibration_sha256': '0' * 64, 'shape': [1, 512],
                     'dtype': 'torch.int64'},
        producer_source={'tensors': {}, 'files': []},
        epsilon=1e-20, model_load_contract={'fixture': True},
        replay_source='fresh_streamed_bf16_source_pass')
    routing = record['metadata']['routing']
    assert routing['device'] == observed
    assert routing['input_dtype'] == 'torch.bfloat16'
    assert routing['topk_weights_dtype'] == 'torch.float32'
    assert routing['topk_ids_dtype'] == 'torch.int64'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a live CUDA device')
def test_cuda_boundary_round_trips_at_the_observed_indexed_device():
    """A bf16 boundary observed on ``cuda:0`` must survive its host copy.

    ``select_original_routes`` copies the routed tensors to the host before
    ``glm_route_record`` runs, so reading the tensors' device at record time
    reports ``cpu`` and the native validator refuses (review round 1 of
    PR #1538, #1536).  The record must instead carry the device observed
    before the copy, and the full CUDA round trip -- capture, record,
    ``validate_glm_routing`` -- must pass with it.
    """
    from prismaquant.glm_routing_replay import glm_route_record, select_original_routes
    from prismaquant.native_moe_panel import validate_glm_routing

    model = Glm5NextForConditionalGeneration()
    mlp = model.model.language_model.layers[3].mlp
    inputs = torch.zeros(512, 4096, dtype=torch.bfloat16, device='cuda')
    captured = select_original_routes(
        mlp.experts, (inputs,
                      torch.zeros(512, 8, dtype=torch.int64, device='cuda'),
                      torch.zeros(512, 8, dtype=torch.float32, device='cuda')),
        {}, sequence_length=512)
    assert all(t.device.type == 'cpu'
               for t in captured.values() if isinstance(t, torch.Tensor))
    runner = SimpleNamespace(device=torch.device('cuda'), model=model)
    record = glm_route_record(runner, mlp.experts, mlp.gate, captured, layer=3,
        calibration={'calibration_sha256': '0' * 64, 'shape': [1, 512],
                     'dtype': 'torch.int64'},
        producer_source={'tensors': {}, 'files': []},
        epsilon=1e-20, model_load_contract={'fixture': True},
        replay_source='fresh_streamed_bf16_source_pass')
    routing = record['metadata']['routing']
    assert routing['device'] == 'cuda:0'
    validate_glm_routing(routing)


def test_validate_glm_routing_names_the_failing_field():
    """An anonymous protocol refusal cost a 408 s GPU slot (#1536); the
    message must name the field that failed, with its observed value."""
    from prismaquant.native_moe_panel import validate_glm_routing
    from test_native_moe_glm_geometry import glm_routing

    with pytest.raises(ValueError,
                       match=r"protocol: device='cuda' \(expected 'cuda:0'\)"):
        validate_glm_routing(glm_routing(device='cuda'))


def test_actual_bias_snapshot_is_reused_and_coordinates_remain_cpu_bookkeeping():
    from prismaquant.glm_routing_replay import glm_route_record, select_original_routes

    model = Glm5NextForConditionalGeneration()
    mlp = model.model.language_model.layers[3].mlp
    captured = select_original_routes(mlp.experts,
        (torch.ones(512, 4096, dtype=torch.bfloat16), torch.arange(8).expand(512, 8),
         torch.full((512, 8), 2.5 / 8, dtype=torch.bfloat16)), {}, sequence_length=512,
        expert_bias=mlp.gate.e_score_correction_bias)
    snapshot = captured['expert_bias']
    assert captured['observed_device'] == 'cpu'
    assert captured['coordinates'].device.type == 'cpu'
    assert captured['coordinates'].dtype == torch.int64
    assert torch.equal(captured['coordinates'][:, 1], torch.arange(512))
    record = glm_route_record(SimpleNamespace(model=model, device=torch.device('cuda')),
        mlp.experts, mlp.gate, captured, layer=3,
        calibration={'calibration_sha256': '0' * 64, 'shape': [512, 512], 'dtype': 'torch.int64'},
        producer_source={'tensors': {}, 'files': []}, epsilon=1e-20,
        model_load_contract={'fixture': True}, replay_source='synthetic_cpu_protocol_control')
    assert record['tensors']['expert_bias'] is snapshot
    assert record['metadata']['routing']['device'] == 'cpu'


@pytest.mark.parametrize('change', ['dtype', 'shape', 'device', 'not_tensor'])
def test_original_live_bias_must_match_actual_source_boundary(change):
    from prismaquant.glm_routing_replay import select_original_routes

    module = Glm5NextTextExperts()
    bias = torch.zeros(288, dtype=torch.float32)
    if change == 'dtype':
        bias = bias.bfloat16()
    elif change == 'shape':
        bias = bias[:287]
    elif change == 'device':
        bias = torch.empty(288, dtype=torch.float32, device='meta')
    else:
        bias = [0.0] * 288
    with pytest.raises(ValueError, match='four source tensors'):
        select_original_routes(module,
            (torch.zeros(512, 4096, dtype=torch.bfloat16), torch.zeros(512, 8, dtype=torch.int64),
             torch.zeros(512, 8, dtype=torch.float32)), {}, sequence_length=512, expert_bias=bias)


@pytest.mark.parametrize('change', ['missing_context', 'mixed_dev'])
def test_original_producer_rejects_missing_or_mixed_independent_context_before_traversal(source, change):
    runner, producer = source
    kwargs = {'original_authority_input': {'path': '/synthetic-authority', 'sha256': '1' * 64}}
    if change == 'mixed_dev':
        kwargs['source_acquisition'] = {'dev_uncertified': True, 'dev_mode': {'PRISMAQUANT_DEV_MODE': '1'},
                                      'source_cache_reuse': {}}
    with pytest.raises(ValueError, match='independent|mutually exclusive'):
        capture_streamed_glm_routes(runner, torch.zeros(512, 512, dtype=torch.int64),
            calibration={'shape': [512, 512], 'dtype': 'torch.int64'},
            producer_source=producer, consume=lambda *_: pytest.fail('unadmitted entry'), **kwargs)
    assert runner.context.events == []
    assert not hasattr(runner.context, 'audit')


def capture_cli_arguments(tmp_path):
    return ['--plan', str(tmp_path / 'plan.json'), '--plan-sha256', '1' * 64,
            '--prepared', str(tmp_path / 'prepared.json'), '--prepared-sha256', '2' * 64,
            '--output-root', str(tmp_path / 'output')]


@pytest.mark.parametrize('extra', [
    ['--original-authority', '/authority'],
    ['--original-authority-sha256', '3' * 64],
    ['--original-authority', '/authority', '--original-authority-sha256', '3' * 64],
    ['--session-preparation', '/session', '--session-preparation-sha256', '4' * 64],
])
def test_original_cli_requires_independent_paired_authority_and_session_inputs(tmp_path, extra):
    from tools.capture_glm_routed_layers import main

    with pytest.raises(SystemExit) as error:
        main(capture_cli_arguments(tmp_path) + extra)
    assert error.value.code == 2
    assert not (tmp_path / 'output').exists()


def test_original_cli_refuses_effective_dev_mode_before_control_or_output(tmp_path, monkeypatch):
    from tools.capture_glm_routed_layers import main

    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '1')
    with pytest.raises(ValueError, match='mutually exclusive'):
        main(capture_cli_arguments(tmp_path) + [
            '--original-authority', '/missing-authority', '--original-authority-sha256', '3' * 64,
            '--session-preparation', '/missing-session', '--session-preparation-sha256', '4' * 64])
    assert not (tmp_path / 'output').exists()


def test_original_cli_strict_control_decoder_rejects_duplicate_authority_before_output(tmp_path, monkeypatch):
    import hashlib

    from tools.capture_glm_routed_layers import main

    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '0')
    authority = tmp_path / 'authority.json'
    raw = b'{"schema":"first","schema":"second"}'
    authority.write_bytes(raw)
    with pytest.raises(RuntimeError, match='duplicate key'):
        main(capture_cli_arguments(tmp_path) + [
            '--original-authority', str(authority), '--original-authority-sha256', hashlib.sha256(raw).hexdigest(),
            '--session-preparation', '/missing-session', '--session-preparation-sha256', '4' * 64])
    assert not (tmp_path / 'output').exists()
    assert not torch.cuda.is_initialized()


def test_original_serialization_has_an_actual_bounded_torch_writer():
    import io

    from tools.capture_glm_routed_layers import _BoundedCaptureBuffer

    value = {'tensor': torch.arange(32)}
    with _BoundedCaptureBuffer(16384) as buffer:
        torch.save(value, buffer)
        assert torch.equal(torch.load(io.BytesIO(buffer.getvalue()), weights_only=True)['tensor'], value['tensor'])
    with _BoundedCaptureBuffer(16) as buffer, pytest.raises(RuntimeError, match='serialization exceeds'):
        torch.save(value, buffer)


@pytest.mark.parametrize('allow', [False, True])
def test_original_arithmetic_pins_only_sealed_process_flags_without_cuda_initialization(tmp_path, monkeypatch, allow):
    from test_native_moe_panel import original_protocol_case
    from tools.capture_glm_routed_layers import _pin_original_arithmetic

    runtime = original_protocol_case(tmp_path)['payload']['boundary_metadata']['source_acquisition']['runtime']['source_runtime']
    runtime['arithmetic']['allow_bf16_reduced_precision_reduction'] = allow
    if allow:
        monkeypatch.delenv('PRISMAQUANT_BF16_REDUCED_PRECISION_REDUCTION', raising=False)
    else:
        monkeypatch.setenv('PRISMAQUANT_BF16_REDUCED_PRECISION_REDUCTION', 'off')
    before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    initialized = torch.cuda.is_initialized()
    try:
        _pin_original_arithmetic(runtime)
        assert torch.get_float32_matmul_precision() == 'highest'
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is allow
        assert torch.cuda.is_initialized() is initialized is False
    finally:
        torch.set_float32_matmul_precision(before[0])
        torch.backends.cuda.matmul.allow_tf32 = before[1]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = before[2]


def test_original_arithmetic_rejects_ambient_disagreement_before_changing_flags(tmp_path, monkeypatch):
    from test_native_moe_panel import original_protocol_case
    from tools.capture_glm_routed_layers import _pin_original_arithmetic

    runtime = original_protocol_case(tmp_path)['payload']['boundary_metadata']['source_acquisition']['runtime']['source_runtime']
    runtime['arithmetic']['allow_bf16_reduced_precision_reduction'] = True
    monkeypatch.setenv('PRISMAQUANT_BF16_REDUCED_PRECISION_REDUCTION', 'off')
    before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    with pytest.raises(ValueError, match='independently sealed runtime'):
        _pin_original_arithmetic(runtime)
    assert before == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                      torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    assert not torch.cuda.is_initialized()
