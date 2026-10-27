"""Genuine initialization is explicit and cannot escape its calling context."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from transformers import PretrainedConfig, PreTrainedModel

from prismaquant import genuine_weight_initialization, initialize_model_runtime


@pytest.fixture
def probe():
    initialize_model_runtime()

    class Config(PretrainedConfig):
        model_type = 'pq_initialization_probe'

    class Model(PreTrainedModel):
        config_class = Config
        base_model_prefix = 'probe'

        def __init__(self, config):
            super().__init__(config)
            self.weight = torch.nn.Parameter(torch.empty(2))
            self.initialized = []
            self.post_init()

        def _init_weights(self, module):
            self.initialized.append(type(module).__name__)
            if module is self:
                module.weight.fill_(7.)

    return lambda: Model(Config())


def test_genuine_context_initializes_bare_parameters_and_restores_skeleton_policy(probe):
    assert probe().initialized == []
    with genuine_weight_initialization():
        inside = probe()
    assert inside.initialized
    assert torch.equal(inside.weight, torch.full((2,), 7.))
    assert probe().initialized == []


def test_initializer_permission_resets_after_a_construction_exception(probe):
    with pytest.raises(RuntimeError, match='construction failed'):
        with genuine_weight_initialization():
            assert probe().initialized
            raise RuntimeError('construction failed')
    assert probe().initialized == []


def test_nested_genuine_context_keeps_the_outer_initialization_permission(probe):
    with genuine_weight_initialization():
        with genuine_weight_initialization():
            assert probe().initialized
        assert probe().initialized
    assert probe().initialized == []


def test_genuine_permission_does_not_initialize_another_threads_skeleton(probe):
    with ThreadPoolExecutor(max_workers=1) as pool:
        with genuine_weight_initialization():
            other = pool.submit(probe).result(timeout=10)
            assert other.initialized == []
            assert probe().initialized
        assert pool.submit(probe).result(timeout=10).initialized == []
