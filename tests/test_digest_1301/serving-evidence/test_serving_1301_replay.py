"""Raw-source identity contracts: frozen originals, not candidate expectations."""
import ast
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock

import pytest
import torch

from prismaquant import glm_routing_replay as replay

FIXTURES = Path(__file__).resolve().parents[2] / 'fixtures/digest_1301/serving-evidence'
REGISTRY = json.loads((FIXTURES / 'replay_registry.json').read_text())


def record(monkeypatch):
    # Only unrelated metadata dependencies are replaced; source acquisition and
    # both digest owners run through the real glm_route_record call.
    for name, attr, value in (
        ('prismaquant.joint_aura', 'source_execution_identity', lambda model: {'source': 'test'}),
        ('prismaquant.production_weight_cache', '_cb_cache_tensor_identity',
         lambda tensor: {'content_sha256': '0' * 64, 'dtype': str(tensor.dtype)}),
        ('transformers', '__version__', 'test'),
    ):
        module = ModuleType(name)
        setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    router = SimpleNamespace(e_score_correction_bias=torch.zeros(2), norm_topk_prob=True,
        routed_scaling_factor=1., num_group=1, topk_group=1, top_k=1)
    module = SimpleNamespace(num_experts=2, hidden_dim=2, intermediate_dim=2, swiglu_limit=None)
    runner = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {})))
    captured = dict(inputs=torch.zeros(1, 2), top_k_weights=torch.ones(1, 1),
                    top_k_index=torch.zeros(1, 1, dtype=torch.int64), observed_device='cpu')
    return lambda: replay.glm_route_record(runner, module, router, captured, layer=3,
        calibration={'calibration_sha256': 'a'*64, 'shape': [1, 2], 'dtype': 'torch.float32'},
        producer_source={}, epsilon=1e-20, model_load_contract={}, replay_source='test')


@pytest.mark.parametrize('text', ['', 'class Router:\n    pass\n', '# café λ\r\n', '\ufeffx = 1\n', 'no final newline'])
@pytest.mark.parametrize('raw', [b'', b'\x00\xff\r\n', b'# source\n', b'# source'])
def test_source_acquisition_matches_original(monkeypatch, text, raw):
    call = record(monkeypatch)
    source = Mock(return_value=text)
    read = Mock(return_value=raw)
    monkeypatch.setattr(replay.inspect, 'getsource', source)
    monkeypatch.setattr(Path, 'read_bytes', lambda path: read(path))
    result = call()['metadata']
    read.assert_called_once_with(Path(replay.__file__))
    assert source.call_count == read.call_count == 1
    assert source.call_args.args == (SimpleNamespace,)
    # Fixtures execute the exact original raw expressions, not digest owners.
    for entry in REGISTRY:
        original = runpy.run_path(str(FIXTURES / entry['snippet']))
        if entry['owner_route'].endswith('text_sha256hex'):
            expected = original['original'](SimpleNamespace())
            assert result['routing']['source_protocol']['router_source_sha256'] == expected
        else:
            original['original'].__globals__['__file__'] = replay.__file__
            assert result['capture_source_sha256'] == original['original'](None)


@pytest.mark.parametrize('site,error', [('inspect', OSError('source unavailable')),
    ('inspect', TypeError('unsupported object')), ('read', PermissionError('denied')),
    ('read', FileNotFoundError('gone'))])
def test_source_acquisition_errors_propagate(monkeypatch, site, error):
    call = record(monkeypatch)
    source = Mock(return_value='source')
    read = Mock(return_value=b'source')
    (source if site == 'inspect' else read).side_effect = error
    monkeypatch.setattr(replay.inspect, 'getsource', source)
    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(type(error)) as raised:
        call()
    assert raised.value is error
    assert source.call_count == 1
    assert read.call_count == (site == 'read')


def test_source_encoding_error_propagates(monkeypatch):
    call = record(monkeypatch)
    monkeypatch.setattr(replay.inspect, 'getsource', lambda obj: '\ud800')
    read = Mock()
    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(UnicodeEncodeError):
        call()
    read.assert_not_called()


def test_completion_provenance_matches_frozen_research_registry():
    frozen = json.loads((FIXTURES.parents[1] / 'digest_parent_completion_1301.json').read_text())
    calls = {row['id']: row for row in frozen['calls']
             if row['file'] == 'prismaquant/glm_routing_replay.py'}
    ledger = json.loads((FIXTURES / 'ledger_replay.json').read_text())
    assert len(REGISTRY) == len(ledger['replacements']) == len(calls) == 2
    for rows in (REGISTRY, ledger['replacements']):
        assert {row['original_id'] for row in rows} == set(calls)
        for row in rows:
            assert row['source_commit'] == frozen['source_commit']
            assert row['source_blob'] == calls[row['original_id']]['source_blob']
            assert 'blob' not in row


def test_whole_file_call_accounting():
    tree = ast.parse(Path(replay.__file__).read_text())
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(c.startswith('hashlib.') for c in calls)
    assert calls.count('text_sha256hex') == 1
    assert calls.count('bytes_sha256hex') == 1
    assert len(REGISTRY) == 2
    for entry in REGISTRY:
        snippet = ast.parse((FIXTURES / entry['snippet']).read_text())
        assert sum(isinstance(n, ast.Call) and ast.unparse(n.func) == 'hashlib.sha256'
                   for n in ast.walk(snippet)) == 1
