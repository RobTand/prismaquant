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
