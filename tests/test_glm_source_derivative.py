"""Closed derivative identity and immutable original-capture compatibility."""
import copy
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from prismaquant import glm_source_derivative as derivative
from prismaquant import glm_capture_compatibility as compatibility
from prismaquant.joint_aura import source_execution_identity
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.model_profiles.default import DefaultProfile

EVIDENCE = Path(__file__).resolve().parents[1] / 'experiments/measurements/glm-derivative-contract-20260908'


def write_bound(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture_policy(tmp_path):
    build = json.loads((EVIDENCE/'image-build-result.json').read_text())
    return dict(schema=derivative.SCHEMA, version=derivative.VERSION,
                image_build=write_bound(tmp_path/'build.json', build))


@pytest.fixture
def observed(tmp_path, monkeypatch):
    policy = fixture_policy(tmp_path)
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(_attn_implementation='eager')
    current = dict(declaration=derivative.declaration(), gates={'unit': {'lower_bound': -5}},
                   image_content_sha256=derivative.CORRECTED_IMAGE_CONTENT_SHA256)
    monkeypatch.setattr(derivative, '_observe', lambda *_: copy.deepcopy(current))
    monkeypatch.setenv('PRISMAQUANT_CONTAINER_CONTENT_SHA256', derivative.CORRECTED_IMAGE_CONTENT_SHA256)
    return model, policy, current


def test_original_runtime_identity_is_exact_v1_and_declaration_does_not_enable():
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(_attn_implementation='eager', _experts_implementation='eager')
    assert Glm5NextProfile().source_derivative_contract() == derivative.declaration()
    assert DefaultProfile().source_derivative_contract() is None
    assert source_execution_identity(model) == {
        'schema': 'prismaquant.joint_aura.source_execution.v1',
        'modules': {'': {'attention': 'eager', 'experts': 'eager'}}}


@pytest.mark.parametrize('changed', ['co_filename', 'co_flags', 'co_consts', 'nested_filename', 'none'])
def test_callable_auth_compares_source_metadata_flags_constants_and_nested_code(changed):
    import types
    source = 'def outer():\n    def inner():\n        return 7\n    return inner()\n'
    scope = {}
    exec(compile(source, '/pinned/source.py', 'exec', dont_inherit=True), scope)
    function = scope['outer']
    expected = derivative._code_at(compile(source, '/pinned/source.py', 'exec', dont_inherit=True), ('outer',))
    if changed == 'none':
        derivative._require_code(function, expected, 'fixture')
        return
    if changed == 'nested_filename':
        constants = tuple(value.replace(co_filename='/foreign.py') if isinstance(value, types.CodeType) else value
                          for value in expected.co_consts)
        expected = expected.replace(co_consts=constants)
    else:
        replacement = {'co_filename': '/foreign.py', 'co_flags': expected.co_flags ^ 0x1000000,
                       'co_consts': expected.co_consts + (99,)}[changed]
        expected = expected.replace(**{changed: replacement})
    with pytest.raises(ValueError, match='callable code changed'):
        derivative._require_code(function, expected, 'fixture')


@pytest.mark.parametrize('entry', ['identity', 'bind_none', 'capture_consumer'])
def test_corrected_module_cannot_silently_use_original_identity(tmp_path, monkeypatch, entry):
    name = 'transformers.models.glm5_next.modeling_glm5_next'
    module = ModuleType(name)
    path = tmp_path/'modeling.py'
    path.write_bytes(b'corrected source fixture')
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(derivative, 'CORRECTED_MODELING_SHA256', derivative.sha256(path))
    model = type('GlmFixture', (torch.nn.Module,), {'__module__': name})()
    with pytest.raises(ValueError, match='explicit derivative binding'):
        if entry == 'identity':
            source_execution_identity(model)
        elif entry == 'bind_none':
            derivative.bind_source_derivative(model, Glm5NextProfile(), None)
        else:
            compatibility.require_capture_compatibility(None, capture={}, model=model)


def test_binding_is_model_local_and_enters_v2_identity(observed):
    model, policy, current = observed
    value = derivative.bind_source_derivative(model, Glm5NextProfile(), policy)
    before = source_execution_identity(model)
    assert before['schema'] == 'prismaquant.joint_aura.source_execution.v2'
    assert before['source_derivative'] == value
    value['gates']['unit']['lower_bound'] = -9
    policy['version'] = 'mutated caller dictionary'
    assert source_execution_identity(model) == before
    assert source_execution_identity(torch.nn.Linear(2, 2))['schema'].endswith('.v1')
    current['gates']['unit']['lower_bound'] = -4
    with pytest.raises(ValueError, match='execution changed'):
        source_execution_identity(model)


@pytest.mark.parametrize('mutation', ['image', 'profile', 'policy', 'build', 'config'])
def test_binding_refuses_unreviewed_inputs(observed, monkeypatch, mutation):
    model, policy, _ = observed
    profile = Glm5NextProfile()
    if mutation == 'image':
        monkeypatch.setenv('PRISMAQUANT_CONTAINER_CONTENT_SHA256', '0'*64)
    elif mutation == 'profile':
        profile = DefaultProfile()
    elif mutation == 'policy':
        policy['enable_anything'] = True
    else:
        build = json.loads(Path(policy['image_build']['path']).read_text())
        if mutation == 'build':
            build['changed_payload_files'].append('/extra.py')
        else:
            build['corrected_image']['Config']['Env'].append('UNREVIEWED=1')
        policy['image_build'] = write_bound(Path(policy['image_build']['path']), build)
    with pytest.raises(ValueError):
        derivative.bind_source_derivative(model, profile, policy)


def test_binding_refuses_later_manifest_mutation_or_removal(observed):
    model, policy, _ = observed
    derivative.bind_source_derivative(model, Glm5NextProfile(), policy)
    with pytest.raises(ValueError, match='cannot remove'):
        derivative.bind_source_derivative(model, Glm5NextProfile(), None)
    Path(policy['image_build']['path']).write_text('{}')
    with pytest.raises(ValueError, match='image build bytes changed'):
        source_execution_identity(model)


def test_closed_transform_refuses_other_source_bytes(monkeypatch):
    raw = ('before\n'+derivative.ORIGINAL_EXPRESSION+'\nafter\n').encode()
    expected = raw.replace(derivative.ORIGINAL_EXPRESSION.encode(), derivative.CORRECTED_EXPRESSION.encode())
    monkeypatch.setattr(derivative, 'ORIGINAL_MODELING_SHA256', hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(derivative, 'CORRECTED_MODELING_SHA256', hashlib.sha256(expected).hexdigest())
    assert derivative.corrected_source(raw) == expected
    with pytest.raises(ValueError, match='original modeling source differs'):
        derivative.corrected_source(raw+b'additional code')


def test_real_image_build_preserves_config_layers_and_source():
    build = json.loads((EVIDENCE/'image-build-result.json').read_text())
    assert derivative.validate_image_build(build) == build
    assert len(build['changed_payload_files']) == 1


def test_real_pb_receipt_binds_canonical_body_payload_and_source():
    audit = json.loads((EVIDENCE/'image-build-audit.json').read_text())
    output = (EVIDENCE/'image-build-output.log').read_bytes()
    compatibility._verify_cas_receipt(audit['receipt'], audit['source_snapshot'], output)
    forged = copy.deepcopy(audit['receipt'])
    forged['result']['bytes'] += 1
    with pytest.raises(ValueError, match='body digest'):
        compatibility._verify_cas_receipt(forged, audit['source_snapshot'], output)
    other = copy.deepcopy(audit['source_snapshot'])
    other['input']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='source snapshot input'):
        compatibility._verify_cas_receipt(audit['receipt'], other, output)


def test_reviewed_producer_completion_is_python_repr_not_json():
    actual_shape = "{'path': '/mnt/shared/capture.json', 'sha256': '"+'a'*64+"'}"
    assert compatibility._completion_literal(actual_shape) == dict(path='/mnt/shared/capture.json', sha256='a'*64)


@pytest.mark.parametrize('value', ["dict(path='x', sha256='y')", "{'path': __import__('os'), 'sha256': 'a'}",
                                  "{'path':'x','path':'y'}", "{'path':'x','sha256':'a','extra':'b'}", 'x'*4097])
def test_completion_parser_does_not_execute_or_accept_unreviewed_repr(value):
    with pytest.raises((ValueError, SyntaxError)):
        compatibility._completion_literal(value)


def test_corrected_capture_consumer_requires_receipt(observed):
    model, policy, _ = observed
    derivative.bind_source_derivative(model, Glm5NextProfile(), policy)
    with pytest.raises(ValueError, match='completed original-capture compatibility receipt'):
        compatibility.require_capture_compatibility(None, capture={}, model=model)


def test_original_consumer_rejects_unrequested_compatibility_receipt():
    with pytest.raises(ValueError, match='explicitly corrected consumer'):
        compatibility.require_capture_compatibility({'path': 'unused', 'sha256': 'a'*64},
                                                     capture={}, model=torch.nn.Linear(2, 2))


def test_incomplete_capture_refuses_before_any_producer_proof(tmp_path, monkeypatch):
    from prismaquant import tessera_calibration_cache as cache
    capture = write_bound(tmp_path/'capture.json', {'schema': cache.SCHEMA, 'status': 'running'})
    value = dict(schema=compatibility.SCHEMA, version=derivative.VERSION, capture=capture,
                 derivative_identity_sha256=compatibility._digest({}), producer={}, forward_equivalence={})
    with pytest.raises(RuntimeError, match='not a complete canonical calibration capture'):
        compatibility._verify(value, capture=capture, derivative={})


def native_schedule_fixture():
    result = dict(backwards=[], replay_diagnostics=[], primary_outputs=[])
    for layer in (0, 3, 4):
        for row in (0, 511):
            output = dict(sha256=f'output-{layer}-{row}')
            result['primary_outputs'].append(dict(layer=layer, original_row=row,
                statistics=dict(nonfinite=0), identity=output))
            for seed in range(7000, 7004):
                for arm in ('unobserved_isolated_baseline', 'nonfinal_fork_replay', 'final_original_owner_replay'):
                    key = dict(layer=layer, original_row=row, seed=seed, arm=arm)
                    result['backwards'].append(dict(**key, output=output,
                        cotangent=dict(sha256=f'gradient-{layer}-{row}-{seed}'), stimulus=dict(sha256=f'stimulus-{seed}'),
                        activity=None if arm == 'unobserved_isolated_baseline' else dict(router='same-route')))
                    result['replay_diagnostics'].append(dict(**key, output_identity=output,
                        output_matches_primary=True, backward_completed=True, output=dict(nonfinite=0),
                        leaf_gradient=dict(nonfinite=0, finite_nonzero=1)))
    return result


def test_native_receipt_consumption_rechecks_complete_schedule_and_equal_arms():
    compatibility._verify_native_schedule(native_schedule_fixture(), diagnostic=False)


@pytest.mark.parametrize('mutation', ['duplicate', 'omit', 'diagnostic_schedule', 'cotangent', 'stimulus',
                                    'route', 'output', 'nonfinite', 'allzero', 'incomplete', 'primary'])
def test_native_receipt_refuses_partial_or_changed_measured_evidence(mutation):
    result = native_schedule_fixture()
    if mutation == 'duplicate':
        result['backwards'][-1] = result['backwards'][0]
    elif mutation == 'omit':
        result['backwards'].pop()
    elif mutation == 'diagnostic_schedule':
        result['replay_diagnostics'][-1]['seed'] = 7000
    elif mutation in ('cotangent', 'stimulus', 'output'):
        result['backwards'][1][mutation] = dict(sha256='changed')
    elif mutation == 'route':
        result['backwards'][1]['activity'] = dict(router='different-route')
    elif mutation in ('nonfinite', 'allzero'):
        result['replay_diagnostics'][0]['leaf_gradient'][
            'nonfinite' if mutation == 'nonfinite' else 'finite_nonzero'] = 1 if mutation == 'nonfinite' else 0
    elif mutation == 'incomplete':
        result['replay_diagnostics'][0]['backward_completed'] = False
    else:
        result['primary_outputs'][0]['original_row'] = 511
    with pytest.raises(ValueError):
        compatibility._verify_native_schedule(result, diagnostic=False)


def test_completed_capture_validation_is_not_replaced_with_runtime_overrides(tmp_path, monkeypatch):
    from prismaquant import tessera_calibration_cache as cache
    from prismaquant import tessera_joint_aura as bridge
    model = torch.nn.Module()
    capture = {'path': str(tmp_path/'capture.json'), 'sha256': 'a'*64}
    data = SimpleNamespace(payload={'provenance': {'calibration_cache': capture}})
    runner = SimpleNamespace(model=model, device='cpu')
    monkeypatch.setattr(bridge, '_bound', lambda *_: tmp_path/'capture.json')
    def original_validator(*args, **kwargs):
        assert kwargs == {'expected_sha256': capture['sha256']}
        raise RuntimeError('original producer/runtime gate retained')
    monkeypatch.setattr(cache, 'require_capture_contract', original_validator)
    with pytest.raises(RuntimeError, match='original producer/runtime gate retained'):
        bridge.prepare_cache(runner, data, capture=capture, max_render_bytes=1024)


def test_streamed_builder_checks_binding_and_shuts_down_on_refusal(monkeypatch):
    from prismaquant import cost_streaming, streaming_model
    calls = []
    context = SimpleNamespace(max_cache_slots=2)
    runner = SimpleNamespace(model=torch.nn.Linear(2, 2), shutdown=lambda: calls.append('shutdown'))
    monkeypatch.setattr(streaming_model, '_build_streaming_context', lambda *a, **k: context)
    monkeypatch.setattr(cost_streaming, 'StreamedCausalLM', lambda *a, **k: runner)
    def bind(model, profile, policy):
        assert model is runner.model and policy == {'closed': 'requested'}
        calls.append('binding')
        raise ValueError('corrected runtime cannot use original execution')
    monkeypatch.setattr(derivative, 'bind_source_derivative', bind)
    with pytest.raises(ValueError, match='cannot use original'):
        cost_streaming.build_streamed_causal_lm('fixture', device=torch.device('cpu'), dtype=torch.bfloat16,
            offload_folder='unused', profile=Glm5NextProfile(), source_derivative={'closed': 'requested'})
    assert calls == ['binding', 'shutdown']


@pytest.mark.parametrize('failure', ['constructor', 'lookahead'])
def test_streamed_builder_retires_context_when_runner_construction_fails(monkeypatch, failure):
    from prismaquant import cost_streaming, streaming_model
    calls = []
    context = SimpleNamespace(max_cache_slots=2, shutdown=lambda: calls.append('shutdown'))
    monkeypatch.setattr(streaming_model, '_build_streaming_context', lambda *a, **k: context)
    def constructor(*args, **kwargs):
        raise RuntimeError('constructor failed')
    monkeypatch.setattr(cost_streaming, 'StreamedCausalLM', constructor)
    with pytest.raises((RuntimeError, ValueError)):
        cost_streaming.build_streamed_causal_lm('fixture', device=torch.device('cpu'), dtype=torch.bfloat16,
            offload_folder='unused', profile=Glm5NextProfile(),
            prefetch_lookahead='invalid' if failure == 'lookahead' else 2)
    assert calls == ['shutdown']
