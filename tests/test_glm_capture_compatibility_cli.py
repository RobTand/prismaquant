"""CPU control-flow/refusal tests; fixture identities never authorize real captures."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import glm_capture_compatibility as compatibility
from prismaquant import glm_source_derivative as derivative


def bound(path, value):
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture_plan(tmp_path):
    producer = {name: bound(tmp_path/(name+'.json'), dict(fixture=name))
                for name in ('request', 'image_inspection', 'modeling_source')}
    producer.update(terminal=None, receipt=None, output=None)
    forward = {name: bound(tmp_path/(name+'.json'), dict(fixture=name))
               for name in ('cpu_reproduction', 'original_layer0', 'corrected_layer0', 'corrected_graph')}
    return dict(schema=compatibility.ISSUANCE_PLAN_SCHEMA,
        model_config=bound(tmp_path/'config.json', dict(model_type='glm5_next')),
        source_derivative=dict(schema=derivative.SCHEMA, version=derivative.VERSION,
                               image_build=bound(tmp_path/'build.json', dict(fixture='build'))),
        capture=None, producer=producer, forward_equivalence=forward, output=str(tmp_path/'issued.json'))


def complete_fixture(plan, tmp_path, *, status='complete'):
    from prismaquant import tessera_calibration_cache as cache
    plan['capture'] = bound(tmp_path/'capture.json', dict(schema=cache.SCHEMA, status=status))
    for name in ('terminal', 'receipt', 'output'):
        plan['producer'][name] = bound(tmp_path/('completed-'+name+'.json'), dict(fixture=name))
    return plan


@pytest.mark.parametrize('mutation', ['plan_sha', 'extra', 'relative_model', 'missing_native', 'mixed_completion',
                                    'unknown_derivative', 'no_derivative', 'relative_output', 'bad_digest'])
def test_closed_plan_refusals_happen_before_model_construction(tmp_path, monkeypatch, mutation):
    plan = fixture_plan(tmp_path)
    if mutation == 'extra': plan['allow_incomplete'] = True
    elif mutation == 'relative_model': plan['model_config']['path'] = 'config.json'
    elif mutation == 'missing_native': del plan['forward_equivalence']['corrected_graph']
    elif mutation == 'mixed_completion': plan['capture'] = bound(tmp_path/'capture.json', {})
    elif mutation == 'unknown_derivative': plan['source_derivative']['version'] = 'other'
    elif mutation == 'no_derivative': plan['source_derivative'] = None
    elif mutation == 'relative_output': plan['output'] = 'receipt.json'
    elif mutation == 'bad_digest': plan['producer']['request']['sha256'] = 'g'*64
    binding = bound(tmp_path/'plan.json', plan)
    if mutation == 'plan_sha': binding['sha256'] = '0'*64
    monkeypatch.setattr(compatibility, '_issuance_model', lambda *a: pytest.fail('invalid plan constructed a model'))
    with pytest.raises(ValueError):
        compatibility.execute_issuance_plan(binding)
    assert not Path(plan['output']).exists()


def test_pending_issue_refuses_before_any_source_read_or_model(tmp_path, monkeypatch):
    plan = fixture_plan(tmp_path)
    monkeypatch.setattr(compatibility, '_issuance_static_inputs', lambda *a: pytest.fail('pending issuance read source'))
    monkeypatch.setattr(compatibility, '_issuance_model', lambda *a: pytest.fail('pending issuance constructed model'))
    with pytest.raises(ValueError, match='original capture completion evidence is pending'):
        compatibility.execute_issuance_plan(bound(tmp_path/'plan.json', plan), issue=True)
    assert not Path(plan['output']).exists()


@pytest.mark.parametrize('issue', [False, True])
def test_actual_incomplete_capture_validator_still_refuses_before_model(tmp_path, monkeypatch, issue):
    plan = complete_fixture(fixture_plan(tmp_path), tmp_path, status='running')
    monkeypatch.setattr(compatibility, '_producer', lambda *a: pytest.fail('incomplete capture reached producer proof'))
    monkeypatch.setattr(compatibility, '_issuance_model', lambda *a: pytest.fail('incomplete capture constructed model'))
    with pytest.raises(RuntimeError, match='not a complete canonical calibration capture'):
        compatibility.execute_issuance_plan(bound(tmp_path/'plan.json', plan), issue=issue)
    assert not Path(plan['output']).exists()


@pytest.fixture
def controlled(tmp_path, monkeypatch):
    from prismaquant import joint_aura, tessera_calibration_cache as cache
    plan, calls = fixture_plan(tmp_path), []
    model = torch.nn.Linear(2, 2, device='meta')
    observed = dict(gates={'fixture': {'lower_bound': -5}}, fixture='unit-test-only')
    execution = dict(fixture='unit-test-only', source_derivative=observed)
    plan['forward_equivalence']['corrected_graph'] = bound(tmp_path/'corrected_graph.json', dict(source_execution=execution))
    monkeypatch.setattr(compatibility, '_issuance_static_inputs', lambda p: calls.append('static') or {'fixture': 'config'})
    monkeypatch.setattr(compatibility, '_issuance_model', lambda c, p: calls.append('model') or model)
    monkeypatch.setattr(compatibility, 'source_derivative_identity', lambda m: observed if m is model else None)
    monkeypatch.setattr(joint_aura, 'source_execution_identity', lambda m: execution)
    monkeypatch.setattr(compatibility, '_forward_and_graph', lambda e, d: calls.append('native') or dict(fixture='proof'))
    monkeypatch.setattr(compatibility, '_producer', lambda e, c: calls.append('producer'))
    def capture(path, **kwargs):
        assert kwargs == {'expected_sha256': plan['capture']['sha256']}
        calls.append('complete-capture')
        return dict(identity={'fixture': 'capture'})
    monkeypatch.setattr(cache, 'require_capture_contract', capture)
    return plan, calls, model, observed, execution


def test_pending_preflight_reports_missing_completion_without_issuing(controlled, tmp_path):
    plan, calls, *_ = controlled
    result = compatibility.execute_issuance_plan(bound(tmp_path/'plan.json', plan))
    assert result['status'] == 'preflight_pending_original_capture'
    assert result['pending'] == ['capture', 'producer.output', 'producer.receipt', 'producer.terminal']
    assert result['receipt'] is None and calls == ['static', 'model', 'native']
    assert not Path(plan['output']).exists()


def test_complete_plan_uses_existing_issuer_and_preserves_capture(controlled, tmp_path, monkeypatch):
    plan, calls, model, observed, _ = controlled
    complete_fixture(plan, tmp_path)
    capture_before = Path(plan['capture']['path']).read_bytes()
    def verify(record, *, capture, derivative):
        assert capture == plan['capture'] and derivative == observed
        assert record['producer'] == plan['producer'] and record['forward_equivalence'] == plan['forward_equivalence']
        calls.append('existing-issuer-revalidation')
    monkeypatch.setattr(compatibility, '_verify', verify)
    binding = bound(tmp_path/'plan.json', plan)
    result = compatibility.execute_issuance_plan(binding, issue=True)
    assert calls == ['complete-capture', 'producer', 'static', 'model', 'native', 'existing-issuer-revalidation']
    assert result['status'] == 'receipt_issued' and result['pending'] == []
    receipt = derivative.bound_json(result['receipt'], 'fixture receipt')
    assert receipt['schema'] == compatibility.SCHEMA and receipt['capture'] == plan['capture']
    assert Path(plan['capture']['path']).read_bytes() == capture_before
    with pytest.raises(ValueError, match='receipt output already exists'):
        compatibility.execute_issuance_plan(binding, issue=True)


def test_producer_refusal_precedes_model_even_for_structurally_complete_plan(controlled, tmp_path, monkeypatch):
    plan, calls, *_ = controlled
    complete_fixture(plan, tmp_path)
    def refuse(*args):
        raise ValueError('original capture has not completed successfully')
    monkeypatch.setattr(compatibility, '_producer', refuse)
    with pytest.raises(ValueError, match='not completed successfully'):
        compatibility.execute_issuance_plan(bound(tmp_path/'plan.json', plan), issue=True)
    assert calls == ['complete-capture'] and not Path(plan['output']).exists()


def test_native_execution_mismatch_refuses_publication(controlled, tmp_path):
    plan, *_ = controlled
    complete_fixture(plan, tmp_path)
    plan['forward_equivalence']['corrected_graph'] = bound(tmp_path/'foreign_graph.json', dict(source_execution={}))
    with pytest.raises(ValueError, match='issuance execution differs'):
        compatibility.execute_issuance_plan(bound(tmp_path/'plan.json', plan), issue=True)
    assert not Path(plan['output']).exists()


@pytest.mark.parametrize('available,initialized', [(True, False), (False, True)])
def test_cpu_gate_refuses_available_or_initialized_cuda(monkeypatch, available, initialized):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: available)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: initialized)
    with pytest.raises(ValueError, match='CUDA unavailable and uninitialized'):
        compatibility._require_cpu_issuance()


def test_cli_reports_preflight_and_requires_a_bound_plan(controlled, tmp_path, capsys):
    plan, *_ = controlled
    binding = bound(tmp_path/'plan.json', plan)
    result = compatibility.main(['preflight', '--plan', binding['path'], '--plan-sha256', binding['sha256']])
    assert json.loads(capsys.readouterr().out) == result
    with pytest.raises(SystemExit):
        compatibility.main(['preflight', '--plan', binding['path']])


@pytest.mark.parametrize('auto', [False, True])
def test_shared_skeleton_constructor_keeps_models_on_meta_and_preserves_resolver(auto, monkeypatch):
    import transformers
    from prismaquant import streaming_model as streaming
    config, resolved, calls = object(), object(), []
    class Model:
        @staticmethod
        def _from_config(actual, **kwargs):
            calls.append((actual, kwargs))
            return torch.nn.Linear(2, 2)
    class Auto:
        @staticmethod
        def from_config(actual, **kwargs):
            calls.append((actual, kwargs))
            return torch.nn.Linear(2, 2)
    monkeypatch.setattr(transformers, 'AutoModelForCausalLM', Auto)
    def resolve(actual, **kwargs):
        assert actual is config and kwargs == dict(multimodal=False, log_prefix='fixture')
        return resolved, Auto if auto else Model
    monkeypatch.setattr(streaming, '_skeleton_config_and_class', resolve)
    result = streaming.build_streaming_skeleton(config, multimodal=False, log_prefix='fixture', attn_implementation='eager')
    assert all(parameter.is_meta for parameter in result.parameters())
    assert calls == [(resolved, dict(attn_implementation='eager', **({'trust_remote_code': True} if auto else {})))]
