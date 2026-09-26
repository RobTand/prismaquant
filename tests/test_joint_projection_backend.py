"""Fail-closed projection selection, prewarm and serialized arithmetic policy."""
from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch

from prismaquant import joint_projection_backend as backend
from prismaquant import format_registry as fr
from prismaquant.joint_aura import SignedJointProjectionLease, arithmetic_identity, identity_sha256


def selector(path='missing.so'):
    qualification, _ = backend._qualification()
    return {'name': backend.FUSED_NAME,
            'binary': {'path': str(path), 'sha256': qualification['build']['binary_sha256']}}


def test_reference_remains_available_without_cuda_compiler_or_artifact(monkeypatch):
    monkeypatch.setattr(backend, '_qualification', lambda: pytest.fail('reference read fused qualification'))
    monkeypatch.setattr(backend.kernel, 'load_backend', lambda: pytest.fail('reference compiled GPU code'))
    monkeypatch.setattr(backend, '_runtime_identity', lambda _: pytest.fail('reference inspected GPU runtime'))
    prepared = backend.prewarm_projection_backend(None, device='cpu')
    layer = torch.nn.Linear(8, 4, bias=False)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8).requires_grad_()
    delta = torch.ones_like(layer.weight)
    with SignedJointProjectionLease({'unit': layer}, {'unit': {'BF16': fr.get_format('BF16')}},
                                    {('unit', 'BF16'): delta}, projection_backend=prepared) as lease:
        lease.begin_probe()
        layer(x).sum().backward()
        values = lease.finish_probe()['unit', 'BF16']
    assert values['weight'] == float((torch.ones((2, 4)).T @ x.detach() * delta).sum())
    assert values['activation'] == values['mixed'] == 0.0
    assert prepared.identity == arithmetic_identity(torch.float32)['projection_backend'] == backend.REFERENCE_IDENTITY


@pytest.mark.parametrize('config', ['fused_fp32_v1', {}, {'name': 'surprise'}, {'name': 'torch', 'enable': True},
    {'name': 'fused_fp32_v1'}, {'name': 'fused_fp32_v1', 'binary': {'path': '/tmp/x', 'sha256': 'a' * 64}},
    {'name': 'fused_fp32_v1', 'binary': {'path': '/tmp/x', 'sha256': None}}])
def test_selector_refuses_unknown_or_unqualified_inputs(config):
    with pytest.raises(ValueError):
        backend.normalize_projection_backend(config)


def test_fused_lease_refuses_missing_explicit_prewarm():
    layer = torch.nn.Linear(8, 4, bias=False)
    with pytest.raises(RuntimeError, match='explicitly prewarmed before the lease'):
        SignedJointProjectionLease({'unit': layer}, {'unit': {'BF16': fr.get_format('BF16')}},
            {('unit', 'BF16'): torch.zeros_like(layer.weight)}, projection_backend=selector())
    assert not layer._forward_hooks


@pytest.mark.parametrize('field', ['torch', 'torch_git', 'cuda', 'machine', 'device', 'headers', 'compiler'])
def test_prewarm_refuses_each_unqualified_runtime_dimension(monkeypatch, field):
    qualification, _ = backend._qualification()
    actual = deepcopy(qualification['runtime'])
    actual[field] = 'different-runtime'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(backend, '_runtime_identity', lambda _: actual)
    monkeypatch.setattr(backend.importlib.util, 'module_from_spec', lambda _: pytest.fail('unqualified runtime loaded code'))
    with pytest.raises(RuntimeError, match='unqualified runtime identity: ' + field):
        backend.prewarm_projection_backend(selector(), device='cuda:0')


def qualify_mock_runtime(monkeypatch):
    qualification, _ = backend._qualification()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(backend, '_runtime_identity', lambda _: deepcopy(qualification['runtime']))
    monkeypatch.setattr(backend.kernel, '_source_digest', lambda: qualification['build']['source_sha256'])
    monkeypatch.setattr(backend.importlib.util, 'module_from_spec', lambda _: pytest.fail('invalid input loaded code'))
    return qualification


@pytest.mark.parametrize('field', ['CUDA_FLAGS', 'CPP_FLAGS', 'source'])
def test_prewarm_refuses_changed_source_or_build_flags(monkeypatch, field):
    qualify_mock_runtime(monkeypatch)
    if field == 'source':
        monkeypatch.setattr(backend.kernel, '_source_digest', lambda: '0' * 64)
    else:
        monkeypatch.setattr(backend.kernel, field, getattr(backend.kernel, field) + ['unqualified'])
    with pytest.raises(RuntimeError, match='source/compiler flags differ'):
        backend.prewarm_projection_backend(selector(), device='cuda:0')


def test_binary_bytes_are_checked_before_loading(tmp_path, monkeypatch):
    qualify_mock_runtime(monkeypatch)
    path = tmp_path / 'foreign.so'
    path.write_bytes(b'not the hash-bound qualified binary')
    with pytest.raises(RuntimeError, match='binary bytes differ'):
        backend.prewarm_projection_backend(selector(path), device='cuda:0')


def test_missing_artifact_cannot_trigger_implicit_build(tmp_path, monkeypatch):
    qualify_mock_runtime(monkeypatch)
    monkeypatch.setattr(backend.kernel, 'load_backend', lambda: pytest.fail('missing artifact triggered JIT'))
    with pytest.raises(FileNotFoundError):
        backend.prewarm_projection_backend(selector(tmp_path / 'missing.so'), device='cuda:0')


def test_serialized_backend_identity_is_complete_and_changes_probe_contract():
    qualification, digest = backend._qualification()
    identity = {'schema': backend.SCHEMA, 'name': backend.FUSED_NAME,
                'qualification_sha256': digest, 'build': qualification['build'],
                'runtime': qualification['runtime'], 'qualified_shapes': qualification['qualified_shapes'],
                'ineligible_layout': 'torch_reference'}
    backend.validate_projection_backend_identity(identity)
    fused = arithmetic_identity(torch.float32, SimpleNamespace(identity=identity))
    assert identity_sha256(fused) != identity_sha256(arithmetic_identity(torch.float32))
    for field in identity:
        invalid = deepcopy(identity)
        invalid.pop(field)
        with pytest.raises(ValueError, match='unqualified backend identity'):
            backend.validate_projection_backend_identity(invalid)


def test_legacy_v1_signed_rows_require_fresh_recomputation():
    from test_joint_aura_assignment_diagnostics import _row
    from prismaquant.joint_aura import validate_joint_aura_entry
    # This test uses the same valid row fixture as assignment diagnostics.
    row = _row("model.layers.0.mlp.down_proj", [1.0, 2.0])
    row['joint_operator_identity']['schema'] = 'prismaquant.joint_aura.operator.v1'
    row['joint_operator_identity_sha256'] = identity_sha256(row['joint_operator_identity'])
    with pytest.raises(ValueError, match='legacy artifacts require fresh prepare and recompute'):
        validate_joint_aura_entry(row)


def test_loader_cannot_substitute_another_binary(tmp_path, monkeypatch):
    qualification = qualify_mock_runtime(monkeypatch)
    intended = tmp_path / 'qualified.so'
    intended.write_bytes(b'fixture, hash mocked below')
    foreign = tmp_path / 'substituted.so'
    foreign.write_bytes(b'foreign fixture')
    monkeypatch.setattr(backend, '_sha', lambda path: qualification['build']['binary_sha256']
                        if Path(path) == intended else '0' * 64)
    monkeypatch.setattr(backend.importlib.util, 'module_from_spec', lambda _: SimpleNamespace(__file__=str(foreign)))
    monkeypatch.setattr(backend.importlib.machinery.ExtensionFileLoader, 'exec_module', lambda *_: None)
    with pytest.raises(RuntimeError, match='actually loaded a different binary'):
        backend.prewarm_projection_backend(selector(intended), device='cuda:0')


def test_prewarmer_device_scope_cannot_be_reused_for_another_device():
    fused = backend._FusedProjection(None, torch.device('cuda:0'), {'qualified_shapes': [[2, 2]]},
                                    seal=backend._PREWARM_SEAL)
    with pytest.raises(RuntimeError, match='not prewarmed for this device'):
        backend.require_prewarmed_projection(fused, device='cuda:1')


def _stub_device_envelope(monkeypatch, bridge):
    """State the device envelope without a device (see tests/test_tessera_joint_aura.py)."""
    monkeypatch.setattr(bridge, "_apply_device_envelope", lambda device, device_bytes, **kw: {
        "enforced": False, "stub": True, "device": str(device),
        "device_envelope_bytes": int(device_bytes)})


def _plan(tmp_path):
    from prismaquant.tessera_joint_aura import SCHEMA
    return {'schema': SCHEMA, 'model': 'fixture', 'inputs': {}, 'output_root': str(tmp_path),
        'canonical_capture': {'path': 'fixture-capture', 'sha256': 'b' * 64},
        'calibration_input': {'path': 'fixture', 'sha256': 'a' * 64},
        'execution': {'n_calib_samples': 512, 'calib_seqlen': 512, 'probe_microbatch': 1,
                      'n_probes': 4, 'seed_base': 7000, 'token_scope': 'all', 'temperature': 1.0,
                      'production_act_scales': '0'}, 'profile_tool': 'cprofile',
        'max_render_bytes': 1024, 'max_gpu_bytes': 2048, 'min_free_gib': 0,
        'source_prefetch': {'max_cache_slots': 24, 'prefetch_workers': 4, 'prefetch_lookahead': 4,
                           'cache_headroom_gb': 4.0, 'prefetch_min_available_gb': 2.0,
                           'require_prefetched_residency': True}}


def test_plan_admits_reference_default_but_refuses_unknown_backend(tmp_path):
    from prismaquant.tessera_joint_aura import _load_plan, _sha
    config = _plan(tmp_path)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(config))
    assert _load_plan(path, _sha(path)) == config
    config['execution']['projection_backend'] = {'name': 'unqualified'}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='unsupported joint projection backend'):
        _load_plan(path, _sha(path))


@pytest.mark.parametrize('legacy_schema', [True, False])
def test_cost_refuses_legacy_or_backend_changed_preparation_before_cache_adoption(tmp_path, monkeypatch, legacy_schema):
    from prismaquant import tessera_joint_aura as bridge, calibration_data, cost_streaming, gpu_guard
    from prismaquant import model_profiles, aura_cost, joint_aura
    monkeypatch.setattr(gpu_guard, 'require_cuda_hot_path', lambda *_: None)
    _stub_device_envelope(monkeypatch, bridge)
    monkeypatch.setattr(model_profiles, 'detect_profile', lambda _: object())
    draw = dict(fit_ids_sha256='a' * 64, text_sha256='b' * 64, nsamples=512, seqlen=512, seed=0)
    calibration = {'provenance': draw}
    source = {'fixture': 'source-model-identity'}
    execution = {'fixture': 'source-execution'}
    data = SimpleNamespace(census={'model': 'fixture', 'attention_implementation': 'eager'},
        payload={'provenance': {'hessian': {'calibration_identity': draw}}},
        layer_render_bytes=lambda _: {0: 64}, formats_by_qname={'unit': ['BF16']}, cells={('unit', 'BF16'): {'render_origin': 'encoded'}},
        unit_scope=None, render_mirror_root=None, synthesized_now=0,
        head_walk_workers=None, head_walk_resumed_units=0)
    monkeypatch.setattr(bridge, 'load_measured_anchor_input', lambda *_args, **_kwargs: data)
    monkeypatch.setattr(calibration_data, 'load_calibration_input', lambda *_args, **_kwargs:
        (torch.zeros((512, 512), dtype=torch.int64), calibration))
    runner = SimpleNamespace(model=torch.nn.Module(), layer_index_for_qname=lambda _: 0, shutdown=lambda: None)
    monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', lambda *_args, **_kwargs: runner)
    monkeypatch.setattr(cost_streaming, 'build_streamed_model_identity', lambda *_args, **_kwargs: source)
    monkeypatch.setattr(joint_aura, 'source_execution_identity', lambda _: execution)
    monkeypatch.setattr(aura_cost, '_aura_source_sha256', lambda: 'c' * 64)
    monkeypatch.setattr(aura_cost, 'compute_aura_cost_streamed', lambda *_args, **_kwargs:
        pytest.fail('foreign preparation reached the adjoint'))
    monkeypatch.setattr(bridge.pickle, 'loads', lambda _: pytest.fail('foreign preparation adopted PWC'))
    record = {'schema': 'prismaquant.tessera_joint_aura.prepared.v1' if legacy_schema else bridge.PREPARED_SCHEMA,
        'status': 'complete', 'plan_sha256': 'd' * 64, 'implementation_sha256': 'c' * 64,
        'source_model_identity': source, 'source_execution': execution, 'calibration_input': calibration,
        'measured_cells': 1, 'reader_identity': None, 'projection_backend': {'name': 'foreign'},
        'render_origins': {'encoded': 1, 'synthesized_from_wire': 0},
        'render_comparisons': {'independent_render_vs_wire': 1, 'wire_round_trip_only': 0}}
    path = tmp_path / 'prepared.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='fresh prepare and recompute' if legacy_schema else 'prepared projection_backend'):
        bridge.execute('run', _plan(tmp_path), plan_sha256='d' * 64,
                       prepared={'path': str(path), 'sha256': bridge._sha(path)})
    assert json.loads((tmp_path / 'run/results.json').read_text())['passed'] is False


def test_the_gpu_pass_binds_the_digest_quantum_and_refuses_to_hash_its_source(tmp_path, monkeypatch):
    """PQ #1374: a prepare hashed the whole source under its GPU reservation.
    The pass now hands the identity build the plan's bound digest cache and a
    refusal naming the CPU-only quantum, so nothing is hashed under the GPU."""
    from prismaquant import tessera_joint_aura as bridge, calibration_data, cost_streaming, gpu_guard
    from prismaquant import model_profiles
    monkeypatch.setattr(gpu_guard, 'require_cuda_hot_path', lambda *_: None)
    _stub_device_envelope(monkeypatch, bridge)
    monkeypatch.setattr(model_profiles, 'detect_profile', lambda _: object())
    draw = dict(fit_ids_sha256='a' * 64, text_sha256='b' * 64, nsamples=512, seqlen=512, seed=0)
    data = SimpleNamespace(census={'model': 'fixture', 'attention_implementation': 'eager'},
        payload={'provenance': {'hessian': {'calibration_identity': draw}}},
        layer_render_bytes=lambda _: {0: 64}, formats_by_qname={'unit': ['BF16']},
        cells={('unit', 'BF16'): {'render_origin': 'encoded'}},
        unit_scope=None, render_mirror_root=None, synthesized_now=0,
        head_walk_workers=None, head_walk_resumed_units=0, encoder_source_reuse=None)
    monkeypatch.setattr(bridge, 'load_measured_anchor_input', lambda *_args, **_kwargs: data)
    monkeypatch.setattr(calibration_data, 'load_calibration_input', lambda *_args, **_kwargs:
        (torch.zeros((512, 512), dtype=torch.int64), {'provenance': draw}))
    monkeypatch.setattr(bridge, '_prepare_source_owner', lambda *_args, **_kwargs: None)
    runner = SimpleNamespace(model=torch.nn.Module(), layer_index_for_qname=lambda _: 0, shutdown=lambda: None)
    monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', lambda *_args, **_kwargs: runner)
    seen = {}

    class Reached(Exception):
        pass

    def identity(*_args, **kwargs):
        seen.update(kwargs)
        raise Reached
    monkeypatch.setattr(cost_streaming, 'build_streamed_model_identity', identity)
    digests = tmp_path / 'digests.json'
    digests.write_text('{}')
    config = _plan(tmp_path)
    config['source_digest_cache'] = {'path': str(digests), 'sha256': bridge._sha(digests)}
    with pytest.raises(Reached):
        bridge.execute('prepare', config, plan_sha256='d' * 64)
    assert Path(seen['digest_cache_path']) == digests
    assert 'identity --model fixture' in seen['refuse_uncovered']
    assert 'source_digest_cache' in seen['refuse_uncovered']


def test_repeated_row_admission_uses_prewarmed_metadata_without_file_io(monkeypatch):
    backend._qualification.cache_clear()
    qualification, digest = backend._qualification()
    identity = {'schema': backend.SCHEMA, 'name': backend.FUSED_NAME,
                'qualification_sha256': digest, 'build': qualification['build'],
                'runtime': qualification['runtime'], 'qualified_shapes': qualification['qualified_shapes'],
                'ineligible_layout': 'torch_reference'}
    monkeypatch.setattr(Path, 'read_bytes', lambda _: pytest.fail('row admission reopened package metadata'))
    backend.validate_projection_backend_identity(identity)
    backend.validate_projection_backend_identity(deepcopy(identity))


def _campaign_runtime(monkeypatch, **changed):
    qualification, _ = backend._qualification()
    actual = deepcopy(qualification['runtime'])
    actual.update(changed)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(backend, '_runtime_identity', lambda _: actual)
    monkeypatch.setattr(backend.importlib.util, 'module_from_spec',
                        lambda _: pytest.fail('unqualified runtime loaded code'))
    return qualification, actual


def test_identity_gate_refuses_before_the_head_phase_writes_any_render(tmp_path, monkeypatch):
    """#553: the campaign paid 5h51m of head phase for a refusal it could make first."""
    from prismaquant import tessera_joint_aura as bridge, gpu_guard
    qualification, _ = _campaign_runtime(monkeypatch, compiler={
        'nvcc_version': qualification_nvcc(),
        # The measured difference between the qualifying and campaign images.
        'cxx_version': 'c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\n'})
    monkeypatch.setattr(gpu_guard, 'require_cuda_hot_path', lambda *_: None)
    _stub_device_envelope(monkeypatch, bridge)
    monkeypatch.setattr(bridge, 'load_measured_anchor_input',
                        lambda *_args, **_kwargs: pytest.fail('the head phase ran before the identity gate'))
    config = _plan(tmp_path)
    config['execution']['projection_backend'] = selector()
    with pytest.raises(RuntimeError, match='unqualified runtime identity: compiler'):
        bridge.execute('prepare', config, plan_sha256='d' * 64)
    written = sorted(path.relative_to(tmp_path).as_posix()
                     for path in tmp_path.rglob('*') if path.is_file())
    assert written == ['prepare/profile.pstats', 'prepare/profile.txt', 'prepare/results.json'], written
    assert json.loads((tmp_path / 'prepare/results.json').read_text())['passed'] is False


def qualification_nvcc():
    return backend._qualification()[0]['runtime']['compiler']['nvcc_version']


def test_plan_preflight_refuses_an_unqualified_image_without_a_device(tmp_path, monkeypatch):
    """Step 3a loads the plan in the campaign's own container, before the GB10."""
    from prismaquant.tessera_joint_aura import _load_plan, _sha
    qualification, _ = backend._qualification()
    environment = {key: value for key, value in qualification['runtime'].items() if key != 'device'}
    config = _plan(tmp_path)
    config['execution']['projection_backend'] = selector()
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(config))
    # A CPU-only preflight container reads no device properties, so the device
    # block is the one axis it cannot compare; every other axis it can.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(backend, '_environment_identity', lambda: deepcopy(environment))
    assert _load_plan(path, _sha(path)) == config
    unqualified = dict(environment, compiler={'nvcc_version': qualification_nvcc(),
                                              'cxx_version': 'c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0\n'})
    monkeypatch.setattr(backend, '_environment_identity', lambda: deepcopy(unqualified))
    with pytest.raises(RuntimeError, match='unqualified runtime identity: compiler'):
        _load_plan(path, _sha(path))


def test_the_synthesize_stage_is_not_held_to_the_projection_runtime(tmp_path, monkeypatch):
    """#552's decode runs off the qualified box on purpose; it loads no backend."""
    from prismaquant.tessera_joint_aura import _load_plan, _sha
    config = _plan(tmp_path)
    config['execution']['projection_backend'] = selector()
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(backend, '_environment_identity',
                        lambda: pytest.fail('synthesize read the projection runtime identity'))
    assert _load_plan(path, _sha(path), projection_runtime=False) == config


def test_refusal_names_the_qualified_and_the_executing_image(monkeypatch):
    qualification, _ = backend._qualification()
    record = qualification['image']
    assert qualification['runtime']['image'] == record['content_sha256']
    monkeypatch.setenv(backend.CONTAINER_CONTENT_ENV, 'f' * 64)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(backend, '_environment_identity',
                        lambda: dict({key: value for key, value in qualification['runtime'].items()
                                      if key != 'device'}, image=backend.executing_image()))
    with pytest.raises(RuntimeError, match='unqualified runtime identity: image; qualified in image '
                       + re.escape(record['reference']) + r'.*executing in image content sha256 f{64}'):
        backend.require_qualified_environment()


def test_an_unlaunched_process_cannot_claim_the_qualified_image(monkeypatch):
    monkeypatch.delenv(backend.CONTAINER_CONTENT_ENV, raising=False)
    assert backend.executing_image() is None
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='executing in image unidentified'):
        backend.require_qualified_environment()


class _RecordingBinary:
    """Stands in for the loaded fused binary and records the shapes it ran."""

    def __init__(self):
        self.shapes = []

    def mul_sum(self, left, right):
        self.shapes.append(tuple(left.shape))
        return (left * right).sum()


def _fused_on_cpu(binary, shapes=((2, 2),)):
    return backend._FusedProjection(binary, torch.device('cpu'),
                                    {'qualified_shapes': [list(shape) for shape in shapes]},
                                    seal=backend._PREWARM_SEAL)


def test_dev_mode_runs_an_unqualified_shape_on_the_reference_arithmetic(monkeypatch, capsys):
    # PQ #1176: the packaged qualification is a seal. It certifies that the
    # binary equals (left * right).sum() bit for bit on the shapes it lists.
    # With sealing off, an unlisted shape runs that reference arithmetic,
    # never the binary, and the difference is printed once per shape.
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '1')
    monkeypatch.setattr(backend.kernel, 'fast_path_eligible', lambda left, right: True)
    binary = _RecordingBinary()
    fused = _fused_on_cpu(binary)
    generator = torch.Generator().manual_seed(1176)
    left, right = (torch.randn(3, 4, generator=generator) for _ in range(2))
    square = torch.ones(2, 2)
    with torch.no_grad():
        results = [fused.product_sum(left, right) for _ in range(2)]
        fused.product_sum(square, square)
    reference = (left * right).sum()
    assert all(torch.equal(result, reference) for result in results)
    assert binary.shapes == [(2, 2)]
    printed = [line for line in capsys.readouterr().out.splitlines()
               if line.startswith('[DEV-MODE]')]
    assert len(printed) == 1
    assert '(3, 4)' in printed[0] and '(2, 2)' in printed[0]


def test_certified_mode_refuses_an_unqualified_shape_that_dev_mode_recorded(monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '1')
    fused = _fused_on_cpu(_RecordingBinary())
    operand = torch.ones(3, 4)
    with torch.no_grad():
        fused.product_sum(operand, operand)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '0')
    with pytest.raises(RuntimeError, match='outside the packaged qualification'), torch.no_grad():
        fused.product_sum(operand, operand)


def test_certified_mode_refuses_an_unqualified_shape():
    fused = _fused_on_cpu(_RecordingBinary())
    operand = torch.ones(3, 4)
    with pytest.raises(RuntimeError, match='outside the packaged qualification'), torch.no_grad():
        fused.product_sum(operand, operand)


def test_dev_mode_still_refuses_operands_of_different_shapes(monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '1')
    fused = _fused_on_cpu(_RecordingBinary())
    with pytest.raises(RuntimeError, match='outside the packaged qualification'), torch.no_grad():
        fused.product_sum(torch.ones(2, 2), torch.ones(4, 1))


def test_dev_mode_reference_path_keeps_the_autograd_refusal(monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', '1')
    fused = _fused_on_cpu(_RecordingBinary())
    operand = torch.ones(3, 4, requires_grad=True)
    with pytest.raises(RuntimeError, match='no autograd registration'):
        fused.product_sum(operand, operand)
