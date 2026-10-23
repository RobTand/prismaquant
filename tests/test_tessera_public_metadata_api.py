"""The producer metadata API supplies facts without importing serving code."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tessera import metadata

from prismaquant import lane_eligibility as lane
from prismaquant.tessera_runtime_contract import installed_runtime_snapshot, RuntimeContractSnapshot


@pytest.fixture
def fresh_snapshot():
    return RuntimeContractSnapshot(installed_runtime_snapshot().raw, 'fresh-api-boundary')


def test_missing_api_version_refuses_the_table(monkeypatch, fresh_snapshot):
    monkeypatch.setattr(metadata, 'METADATA_API_VERSION', 'unknown')
    with pytest.raises(lane.LaneEligibilityError, match='incompatible Tessera metadata API'):
        fresh_snapshot.eligibility


def test_missing_public_capability_refuses_the_table(monkeypatch, fresh_snapshot):
    monkeypatch.delattr(metadata, 'parse_smoke_record')
    with pytest.raises(lane.LaneEligibilityError, match='callable.*parse_smoke_record'):
        fresh_snapshot.eligibility


def test_missing_public_module_refuses_without_a_serving_fallback(monkeypatch, fresh_snapshot):
    monkeypatch.setitem(sys.modules, 'tessera.metadata', None)
    with pytest.raises(lane.LaneEligibilityError, match='public metadata API'):
        fresh_snapshot.eligibility


def test_the_public_grammar_returns_caller_owned_record_rows():
    record = {'instrument': 'tools/smoke.py', 'rule': 'owned observed scoring rule',
              'reference': None, 'rows': [{'prompt': 'p', 'form': 'pure_greedy',
              'interface': 'raw_completion', 'status': 'recorded', 'reference_status': None}]}
    before = copy.deepcopy(record)
    result = metadata.parse_smoke_record(record, control=None, where='owned.record')
    result['rows'][0]['status'] = 'repetitive'
    assert record == before
    assert metadata.derive_smoke_status({'status': 'recorded', 'record': record}) == 'recorded'


def test_pinned_metadata_parses_in_a_fresh_serving_free_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    child = '''
import sys
class ForbiddenRuntime:
    def find_spec(self, name, path=None, target=None):
        if name == 'tessera.serving' or name.startswith('tessera.serving.') or name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm'}:
            raise AssertionError('forbidden runtime import: ' + name)
sys.meta_path.insert(0, ForbiddenRuntime())
from prismaquant.tessera_runtime_contract import installed_runtime_snapshot
snapshot = installed_runtime_snapshot()
assert len(snapshot.eligibility.cells) == 22
assert snapshot.sha256 == 'ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e'
snapshot.require_exact_pin()
from prismaquant.tessera_runtime_contract import TESSERA_DEV_PIN_COMMIT
from prismaquant.tessera_serving_runtime_pin import load_tessera_serving_runtime_pin
pin = load_tessera_serving_runtime_pin()
assert pin.commit == pin.producer_commit == TESSERA_DEV_PIN_COMMIT == 'aaa417003f0ee690e8ace138acd18df7f9531369'
assert not any(name == 'tessera.serving' or name.startswith('tessera.serving.') for name in sys.modules)
print('PUBLIC-METADATA-OK')
'''
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join([str(Path(metadata.__file__).resolve().parents[1]), str(root)])
    result = subprocess.run([sys.executable, '-c', child], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'PUBLIC-METADATA-OK'


@pytest.mark.parametrize('consumer', ['fused', 'image'])
def test_producer_consumers_execute_without_serving_or_tensor_imports(tmp_path, consumer):
    root = Path(__file__).resolve().parents[1]
    child = '''
import sys
class ForbiddenRuntime:
    def find_spec(self, name, path=None, target=None):
        if name == 'tessera.serving' or name.startswith('tessera.serving.') or name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm'}:
            raise AssertionError('forbidden runtime import: ' + name)
sys.meta_path.insert(0, ForbiddenRuntime())
'''
    if consumer == 'fused':
        child += '''
from tessera.metadata import fused_module
from prismaquant.tessera_lane import glm_fused_sibling_group, glm_fused_sibling_leaf_mapping
expected = [
    ('gate_up_proj', ('gate_proj', 'up_proj')),
    ('in_proj_qkvbfg_a', ('q_proj', 'k_proj', 'v_proj', 'b_proj', 'f_a_proj', 'g_a_proj')),
    ('fused_qkv_a_proj', ('q_a_proj', 'kv_a_proj_with_mqa')),
    ('wk_weights_proj', ('wk', 'weights_proj')),
]
assert list(glm_fused_sibling_leaf_mapping().items()) == expected
for prefix in ('model.layers.0.', 'model.language_model.layers.0.'):
    attention = prefix + 'self_attn.'
    config = {'text_config': {'layer_types': ['linear_attention']}}
    for leaf in (*expected[1][1], 'forget_gate.f_a_proj'):
        assert glm_fused_sibling_group(attention + leaf, config=config) == attention + 'in_proj_qkvbfg_a'
        assert glm_fused_sibling_group(attention + leaf + '.weight', config=config) == attention + 'in_proj_qkvbfg_a'
    assert config == {'text_config': {'layer_types': ['linear_attention']}}
    for config in (None, {}, {'layer_types': ['full_attention']},
                   {'text_config': {'layer_types': ['deepseek_sparse_attention']}}):
        assert glm_fused_sibling_group(attention + 'q_proj', config=config) is None
    for leaf in expected[2][1]:
        assert glm_fused_sibling_group(attention + leaf) == attention + 'fused_qkv_a_proj'
        assert glm_fused_sibling_group(attention + leaf, config={'text_config': {'q_lora_rank': None}}) is None
    for leaf in expected[3][1]:
        assert glm_fused_sibling_group(attention + 'indexer.' + leaf) == attention + 'indexer.wk_weights_proj'
    assert glm_fused_sibling_group(attention + 'o_proj', config={'layer_types': ['linear_attention']}) is None
    assert glm_fused_sibling_group(prefix + 'mlp.shared_experts.gate_proj') == prefix + 'mlp.shared_experts.gate_up_proj'
    tensor = attention + 'q_proj.weight'
    architecture = 'Glm5NextForConditionalGeneration'
    assert fused_module(tensor)[0] == attention + 'qkv_proj'
    assert fused_module(tensor, architecture, tensor_names=[]) is None
    assert fused_module(tensor, architecture, tensor_names=[attention + 'b_proj.weight'])[0] == attention + 'in_proj_qkvbfg_a'
    assert fused_module(tensor, architecture, config={'layer_types': ['full_attention']},
                        tensor_names=[attention + 'b_proj.weight']) is None
'''
    else:
        child += '''
from copy import deepcopy
import json
import os
from pathlib import Path
from pytest import MonkeyPatch
from tessera import image_metadata as image
from prismaquant.tessera_route_receipt import (
    _check_runtime_image_declaration, check_scoped_route_receipt, TesseraRouteReceiptError)
from test_tessera_scoped_route_receipt import fixture
requested = 'example/runtime@sha256:' + 'a' * 64
other = 'example/runtime@sha256:' + 'b' * 64
record = {'schema': 'tessera.runtime_image/1', 'refused': False,
          'resolved_reference': requested, 'repo_digests': [requested],
          'local_id': 'fixture-local-id'}
environment = {image.CENSUS_IMAGE_ENV: requested,
               image.CENSUS_DECLARATION_ENV: json.dumps(record)}
declaration = image.declared_reference(requested, env=environment)
assert declaration == {'schema': image.DECLARATION_SCHEMA, 'source': image.DECLARATION_SOURCE,
                       'image': requested, 'env': [image.CENSUS_IMAGE_ENV, image.CENSUS_DECLARATION_ENV],
                       'record': record}
before = deepcopy(declaration)
os.environ[image.CENSUS_IMAGE_ENV] = other
os.environ[image.CENSUS_DECLARATION_ENV] = 'not-json'
assert _check_runtime_image_declaration({'runtime_image_declaration': declaration}, requested) is None
assert declaration == before
with MonkeyPatch.context() as patch:
    for structure in ('dense', 'routed_moe'):
        data = fixture(Path.cwd(), patch, structure=structure)
        verdict = check_scoped_route_receipt(data[0], data[1], build=data[2], model_dir=data[3])
        assert verdict['n_records'] == 2
        assert verdict['served_routes'] and verdict['served_decoders']
for updates, target, reason in (
    ({'refused': True, 'reason': 'image_absent'}, requested, 'image_declaration_refused'),
    ({'schema': 'unknown'}, requested, 'image_declaration_unreadable'),
    ({'resolved_reference': other}, requested, 'image_declaration_inconsistent'),
    ({'repo_digests': []}, requested, 'image_declaration_inconsistent'),
    ({}, other, 'image_declaration_mismatch'),
):
    changed = {**record, **updates}
    try:
        image.declared_reference(target, env={image.CENSUS_IMAGE_ENV: requested,
            image.CENSUS_DECLARATION_ENV: json.dumps(changed)})
    except image.RuntimeImageError as error:
        expected = f'{reason}: {error}'
        assert error.payload == {'refused': True, 'reason': reason, 'requested': target,
                                 **({'env': [image.CENSUS_DECLARATION_ENV]} if reason == 'image_declaration_unreadable'
                                    else {'declared': requested})}
    else:
        raise AssertionError('invalid declaration admitted')
    try:
        _check_runtime_image_declaration({'runtime_image_declaration': {**declaration, 'record': changed}}, target)
    except TesseraRouteReceiptError as error:
        assert str(error) == expected
        assert isinstance(error.__cause__, image.RuntimeImageError)
    else:
        raise AssertionError('invalid receipt declaration admitted')
for environment in ({}, {image.CENSUS_IMAGE_ENV: requested},
                    {image.CENSUS_DECLARATION_ENV: json.dumps(record)}):
    try:
        image.declared_reference(requested, env=environment)
    except image.RuntimeImageError as error:
        assert error.payload == {'refused': True, 'reason': 'image_declaration_missing',
                                 'requested': requested, 'env': [image.CENSUS_IMAGE_ENV, image.CENSUS_DECLARATION_ENV]}
    else:
        raise AssertionError('missing env pair admitted')
for raw in ('not-json', '[]', 'true', '{}'):
    try:
        image.declared_reference(requested, env={image.CENSUS_IMAGE_ENV: requested,
            image.CENSUS_DECLARATION_ENV: raw})
    except image.RuntimeImageError as error:
        assert error.payload == {'refused': True, 'reason': 'image_declaration_unreadable',
                                 'requested': requested, 'env': [image.CENSUS_DECLARATION_ENV]}
    else:
        raise AssertionError('malformed declaration admitted')
'''
    child += '''
assert not any(name == 'tessera.serving' or name.startswith('tessera.serving.') or
               name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm'} for name in sys.modules)
print('PUBLIC-CONSUMER-OK')
'''
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join([
        str(root), str(root / 'tests'), str(Path(metadata.__file__).resolve().parents[1])])
    result = subprocess.run([sys.executable, '-c', child], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'PUBLIC-CONSUMER-OK'


def test_package_keeps_the_serving_runtime_outside_the_dependency_boundary():
    """Production modules consume metadata without importing Tessera serving."""
    import ast

    root = Path(__file__).resolve().parents[1]
    violations = []
    for path in sorted((root / 'prismaquant').rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or '']
                if node.module == 'tessera':
                    imported.extend('tessera.' + alias.name for alias in node.names)
            elif isinstance(node, ast.Call):
                function = ast.unparse(node.func)
                if function.endswith(('.files', '.import_module')) or function == '__import__':
                    imported = [arg.value for arg in node.args
                                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            if any(name == 'tessera.serving' or name.startswith('tessera.serving.')
                   for name in imported):
                violations.append(f'{path.relative_to(root)}:{node.lineno}')
    assert not violations, violations
