"""CPU policy/geometry gates; plans never allocate statistics or retain source."""
from dataclasses import FrozenInstanceError, replace
import gc
import json
import weakref

import pytest
import torch
from torch import nn

from prismaquant import format_registry as fr
from prismaquant.joint_aura import JointOperatorStatisticsLease, SignedJointProjectionLease
from prismaquant.joint_projection_backend import prewarm_projection_backend
from prismaquant.joint_statistics_plan import plan_joint_statistics_target_windows as plan
from prismaquant.nvfp4_activation_contract import (
    ActivationScaleContractError, NVFP4_SERVED_ACTIVATION_CONTRACT)
from prismaquant.routed_experts import PackedExpertProjection
from test_joint_aura_projection import _spec


def linear(rows=4, columns=16):
    return nn.Linear(columns, rows, bias=False, device='meta')


def choices():
    contract = replace(NVFP4_SERVED_ACTIVATION_CONTRACT, measured_as_served=True)
    a4 = replace(fr.get_format('NVFP4'), name='a4', static_activation_contract=contract)
    return {'a4': a4, 'a8': fr.get_format('FP8_E4M3'), 'identity': fr.get_format('BF16')}


def test_whole_target_greedy_windows_have_complete_deterministic_coverage():
    modules = {name: linear(rows) for name, rows in [('z', 2), ('a', 1), ('m', 3)]}
    specs = {name: {'identity': fr.get_format('BF16')} for name in modules}
    result = plan(modules, specs, max_statistics_bytes=256)
    assert result.windows == (('a', 'm'), ('z',))
    assert result.window_statistics_bytes == (256, 128)
    assert result.total_statistics_bytes == 384
    other = plan(dict(reversed(list(modules.items()))), specs, max_statistics_bytes=256)
    assert other.as_dict() == result.as_dict()
    assert other.identity_sha256 == result.identity_sha256
    assert [name for window in result.windows for name in window] == ['a', 'm', 'z']
    assert result.as_dict()['projection_backend']['name'] == 'torch'


def test_actual_glm_shapes_two_qdq_groups_produce_three_windows_without_allocation(monkeypatch):
    # Full routed+shared layer census: 289 down and 578 gate/up projections.
    modules = {f'unit.{index:04}': linear(4096 if index < 289 else 2048,
                                        2048 if index < 289 else 4096)
               for index in range(867)}
    specs = {name: choices() for name in modules}
    maxima = dict.fromkeys(modules, 1.0)
    backend = prewarm_projection_backend({'name': 'torch'}, device='meta')
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: pytest.fail('planner allocated a tensor'))
    monkeypatch.setattr(nn.Module, 'register_forward_hook', lambda *a, **k: pytest.fail('planner installed hook'))
    result = plan(modules, specs, max_statistics_bytes=32*1024**3,
                  activation_max_abs=maxima, projection_backend=backend)
    assert list(map(len, result.windows)) == [341, 341, 185]
    assert result.total_statistics_bytes == 87_275_077_632
    assert all(target.statistics_bytes == 96*1024**2 for target in result.targets)
    assert all(len(target.groups) == 3 for target in result.targets)
    assert result.window_statistics_bytes == (341*96*1024**2, 341*96*1024**2, 185*96*1024**2)


def test_widest_glm_dense_target_remains_indivisible():
    module = linear(4096, 12288)
    specs = {'unit': choices()}
    kwargs = dict(activation_max_abs={'unit': 1.0})
    with pytest.raises(RuntimeError, match='target unit.*exceeding statistics budget'):
        plan({'unit': module}, specs, max_statistics_bytes=576*1024**2-1, **kwargs)
    result = plan({'unit': module}, specs, max_statistics_bytes=576*1024**2, **kwargs)
    assert result.windows == (('unit',),)
    assert result.total_statistics_bytes == 576*1024**2


def test_static_owner_shares_group_despite_distinct_unused_dynamic_callables():
    source = choices()['a4']
    specs = {'second': replace(source, activation_quantize_dequantize=lambda x: x),
             'first': replace(source, activation_quantize_dequantize=lambda x: x*0)}
    result = plan({'unit': linear()}, {'unit': specs}, max_statistics_bytes=512,
                  activation_max_abs={'unit': 1.0})
    assert result.total_statistics_bytes == 512
    assert result.targets[0].groups[0].formats == ('second', 'first')
    assert result.targets[0].groups[0].as_dict()['activation']['input_global_scale'] is not None


def test_dynamic_callable_distinctions_survive_equal_receipts_without_pointer_identity():
    def factory():
        return lambda x: x
    left, right = factory(), factory()
    specs = {'left': _spec('left', left), 'left_alias': _spec('left_alias', left),
             'right': _spec('right', right), 'identity': _spec('identity', left, act_bits=None)}
    result = plan({'unit': linear()}, {'unit': specs}, max_statistics_bytes=768)
    groups = result.targets[0].groups
    assert [group.formats for group in groups] == [('left', 'left_alias'), ('right',), ('identity',)]
    assert groups[0].activation_identity_json == groups[1].activation_identity_json
    assert result.total_statistics_bytes == 768
    # Recreated callable objects preserve equality partition and persisted identity.
    newer_left, newer_right = factory(), factory()
    newer = {fmt: replace(spec, activation_quantize_dequantize=(newer_right if fmt=='right' else newer_left))
             for fmt, spec in specs.items()}
    again = plan({'unit': linear()}, {'unit': newer}, max_statistics_bytes=768)
    assert again.identity_sha256 == result.identity_sha256
    encoded = json.dumps(result.as_dict())
    assert str(id(left)) not in encoded and str(id(right)) not in encoded


def test_plan_and_both_leases_share_requirements_and_preserve_group_order():
    module = nn.Linear(16, 4, bias=False)
    specs = choices()
    result = plan({'unit': module}, {'unit': specs}, max_statistics_bytes=768,
                  activation_max_abs={'unit': 1.0})
    operator = JointOperatorStatisticsLease({'unit': module}, {'unit': specs},
        max_statistics_bytes=768, max_candidate_bytes=256, activation_max_abs={'unit': 1.0})
    signed = SignedJointProjectionLease({'unit': module}, {'unit': specs},
        {('unit', fmt): torch.zeros_like(module.weight) for fmt in specs}, activation_max_abs={'unit': 1.0})
    assert operator.statistics_capacity_bytes == result.total_statistics_bytes
    expected = [group.formats for group in result.targets[0].groups]
    assert [tuple(formats) for _,formats in operator.groups['unit']] == expected
    assert [tuple(formats) for _,formats in signed.groups['unit']] == expected
    assert not module._forward_hooks


def test_plan_does_not_retain_modules_specs_tensors_or_callables():
    def construct():
        module = linear()
        tensor = torch.ones(1)
        def quantizer(x):
            return x*tensor
        spec = _spec('q', quantizer)
        refs = [weakref.ref(value) for value in (module, module.weight, tensor, quantizer, spec)]
        return plan({'unit': module}, {'unit': {'q': spec}}, max_statistics_bytes=512), refs
    result, refs = construct()
    gc.collect()
    assert all(ref() is None for ref in refs)
    original = result.identity_sha256
    exported = result.as_dict()
    exported['targets'][0]['groups'][0]['activation']['act_bits'] = 1
    assert result.identity_sha256 == original
    with pytest.raises(FrozenInstanceError):
        result.windows = ()


@pytest.mark.parametrize('cap', [None, 0, -1, True, 1.0, float('inf')])
def test_invalid_caps_refuse_before_source_access(cap):
    with pytest.raises(ValueError, match='positive integer'):
        plan({'unit': object()}, {'unit': {}}, max_statistics_bytes=cap)


@pytest.mark.parametrize('case', ['empty','missing','extra','no_specs','unknown_spec','unknown_module','alias'])
def test_unknown_coverage_types_and_aliases_refuse(case):
    module = linear()
    modules, specs = {'unit': module}, {'unit': {'BF16': fr.get_format('BF16')}}
    if case == 'empty': modules, specs = {}, {}
    elif case == 'missing': specs = {}
    elif case == 'extra': specs['extra'] = specs['unit']
    elif case == 'no_specs': specs['unit'] = {}
    elif case == 'unknown_spec': specs['unit'] = {'bad': object()}
    elif case == 'unknown_module': modules['unit'] = object()
    else:
        modules['alias'] = module
        specs['alias'] = specs['unit']
    with pytest.raises((ValueError, TypeError)):
        plan(modules, specs, max_statistics_bytes=1024)
    assert not module._forward_hooks


def test_unprewarmed_backend_refuses():
    with pytest.raises(RuntimeError, match='prewarmed'):
        plan({'unit': linear()}, {'unit': choices()}, max_statistics_bytes=1024,
             activation_max_abs={'unit': 1.0}, projection_backend={'name': 'torch'})


def test_static_scale_and_geometry_refuse():
    with pytest.raises(ActivationScaleContractError):
        plan({'unit': linear()}, {'unit': choices()}, max_statistics_bytes=1024)
    with pytest.raises(ValueError, match='geometry'):
        plan({'unit': linear(columns=15)}, {'unit': choices()}, max_statistics_bytes=1024,
             activation_max_abs={'unit': 1.0})


def test_distinct_packed_experts_are_counted_but_exact_alias_refuses():
    owner = nn.Module()
    owner.weight = nn.Parameter(torch.empty(2, 4, 16, device='meta'))
    def target(name, expert):
        return PackedExpertProjection(name, 'experts.weight', 'experts', owner,
                                      'weight', expert, 'down_proj', owner.weight[expert])
    modules = {'one': target('one', 0), 'two': target('two', 1)}
    specs = {name: {'BF16': fr.get_format('BF16')} for name in modules}
    result = plan(modules, specs, max_statistics_bytes=256)
    assert result.windows == (('one',), ('two',))
    modules['two'] = target('two', 0)
    with pytest.raises(ValueError, match='aliased packed'):
        plan(modules, specs, max_statistics_bytes=512)
