"""Closed opt-in GLM KDA derivative identity; never mutates model code or dispatch."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
import types
import weakref

VERSION = 'glm_kda_causal_exp_v1'
SCHEMA = 'prismaquant.glm_source_derivative.v1'
ORIGINAL_MODELING_SHA256 = '2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b'
CORRECTED_MODELING_SHA256 = '416bd6168b3c42858c0e22622dd2e85ff04e9460703eadf64e5abef84aac24a3'
ORIGINAL_IMAGE_CONTENT_SHA256 = 'eb8592abd71390231b49aba119e36f02ad91ea867b06df1c67af3833004d07bd'
CORRECTED_IMAGE_CONTENT_SHA256 = 'd0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026'
ORIGINAL_HUB_KERNELS_SHA256 = 'fa5143bbbc6a928c70f7e05580b358caae16e88f53f49059c7002f0d01f6c832'
ORIGINAL_ACCELERATE_INTEGRATION_SHA256 = '4469496da61fdc632faf9cacfc128729b12030eb03c5ef4bb70a66b6012b3a82'
ORIGINAL_EXPRESSION = '(g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()'
CORRECTED_EXPRESSION = '(g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(mask.triu(diagonal=1).unsqueeze(-1), 0).exp().float()'
_BINDINGS = weakref.WeakKeyDictionary()


def _require(ok, message):
    if not ok:
        raise ValueError('GLM source derivative: ' + message)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bound_json(binding, label):
    _require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, label + ' requires path/SHA256')
    raw = Path(binding['path']).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == binding['sha256'], label + ' bytes changed')
    return json.loads(raw)


def declaration():
    return dict(schema=SCHEMA, version=VERSION,
                original_modeling_sha256=ORIGINAL_MODELING_SHA256,
                corrected_modeling_sha256=CORRECTED_MODELING_SHA256,
                original_image_content_sha256=ORIGINAL_IMAGE_CONTENT_SHA256,
                corrected_image_content_sha256=CORRECTED_IMAGE_CONTENT_SHA256,
                transform='strict_upper_triangle_zero_before_exp_preserve_diagonal',
                dispatch='original_decorated_torch_fallback')


def normalize_source_derivative(value):
    if value is None:
        return None
    _require(isinstance(value, dict) and set(value) == {'schema', 'version', 'image_build'}, 'closed policy required')
    _require(value['schema'] == SCHEMA and value['version'] == VERSION, 'unknown derivative contract')
    build = value['image_build']
    _require(isinstance(build, dict) and set(build) == {'path', 'sha256'}, 'image build must be byte-bound')
    return json.loads(json.dumps(value, allow_nan=False))


def corrected_source(raw):
    _require(hashlib.sha256(raw).hexdigest() == ORIGINAL_MODELING_SHA256, 'original modeling source differs')
    old, new = ORIGINAL_EXPRESSION.encode(), CORRECTED_EXPRESSION.encode()
    _require(raw.count(old) == 1, 'reviewed source expression is not unique')
    result = raw.replace(old, new, 1)
    _require(hashlib.sha256(result).hexdigest() == CORRECTED_MODELING_SHA256, 'corrected source differs')
    return result


def validate_image_build(build):
    from tools.container_runtime_identity import image_content_sha256
    _require(build.get('schema') == 'prismaquant.glm_derivative_image_build.v1' and build.get('status') == 'complete',
             'complete corrected image build required')
    _require(build.get('original_image_content_sha256') == ORIGINAL_IMAGE_CONTENT_SHA256 and
             build.get('corrected_image_content_sha256') == CORRECTED_IMAGE_CONTENT_SHA256 and
             build.get('hub_kernels_sha256') == ORIGINAL_HUB_KERNELS_SHA256 and
             build.get('original_modeling_sha256') == ORIGINAL_MODELING_SHA256 and
             build.get('corrected_modeling_sha256') == CORRECTED_MODELING_SHA256 and
             build.get('changed_payload_files') == [build.get('modeling_path')], 'unreviewed image change')
    before, after = build['original_image'], build['corrected_image']
    _require(image_content_sha256(before) == ORIGINAL_IMAGE_CONTENT_SHA256 and
             image_content_sha256(after) == CORRECTED_IMAGE_CONTENT_SHA256 and
             before['Config'] == after['Config'] and before['RootFS']['Layers'] == after['RootFS']['Layers'][:-1] and
             after['RootFS']['Layers'][-1] == 'sha256:' + build['added_layer_sha256'],
             'image build config or layer content differs')
    return build


def _code_at(root, names):
    for name in names:
        matches = [x for x in root.co_consts if isinstance(x, types.CodeType) and x.co_name == name]
        _require(len(matches) == 1, 'ambiguous original callable code')
        root = matches[0]
    return root


def _require_code(function, expected, label):
    # Marshal embeds reference/interning state; equivalent imported/recompiled
    # code can serialize differently. Compare every public immutable code field
    # recursively, including source location, compiler flags and nested bodies.
    def equal(actual, wanted):
        if type(actual) is not type(wanted):
            return False
        if isinstance(actual, types.CodeType):
            fields = [name for name in dir(actual) if name.startswith('co_') and
                      not callable(getattr(actual, name))]
            return all(equal(getattr(actual, name), getattr(wanted, name)) for name in fields)
        if isinstance(actual, tuple):
            return len(actual) == len(wanted) and all(equal(a, b) for a, b in zip(actual, wanted))
        return actual == wanted
    _require(isinstance(function, types.FunctionType) and equal(function.__code__, expected),
             label + ' callable code changed')


def _observe(model, build):
    """Inspect real closure dispatch and live gates without calling unwrapped code."""
    from transformers.models.glm5_next import modeling_glm5_next as modeling
    from transformers.integrations import hub_kernels
    from transformers.integrations import accelerate
    model_path, hub_path = Path(modeling.__file__), Path(hub_kernels.__file__)
    raw, hub_raw = model_path.read_bytes(), hub_path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == CORRECTED_MODELING_SHA256, 'actual modeling source differs')
    _require(hashlib.sha256(hub_raw).hexdigest() == build['hub_kernels_sha256'], 'actual dispatch source differs')
    # Authenticate the target module's compiler flags, without inheriting this
    # verifier's future-annotations flag into unrelated Transformers modules.
    compiled = compile(raw, str(model_path), 'exec', dont_inherit=True)
    hub_compiled = compile(hub_raw, str(hub_path), 'exec', dont_inherit=True)
    accelerate_path = Path(accelerate.__file__)
    accelerate_raw = accelerate_path.read_bytes()
    _require(hashlib.sha256(accelerate_raw).hexdigest() == ORIGINAL_ACCELERATE_INTEGRATION_SHA256,
             'actual accelerate wrapper source differs')
    accelerate_compiled = compile(accelerate_raw, str(accelerate_path), 'exec', dont_inherit=True)
    function = modeling.chunk_kimi_delta_attention
    _require_code(function, _code_at(hub_compiled, ('use_kernel_func_from_hub_with_fallback', 'decorator', 'wrapped')),
                  'decorated fallback')
    closure = inspect.getclosurevars(function).nonlocals
    original = closure.get('torch_function')
    _require(closure.get('implementation') is original and closure.get('is_new_implementation') is False,
             'actual KDA dispatch is not the decorated Torch fallback')
    _require_code(original, _code_at(compiled, ('chunk_kimi_delta_attention',)), 'Torch fallback')
    _require(original.__globals__ is vars(modeling), 'fallback globals differ')
    gates = {}
    for name, module in model.named_modules():
        if type(module).__name__ != 'Glm5NextTextLinearAttention':
            continue
        _require(type(module) is modeling.Glm5NextTextLinearAttention, 'attention class substitution')
        forward = module.forward
        _require('forward' not in vars(module), 'instance forward substitution')
        _require_code(forward.__func__, _code_at(accelerate_compiled,
            ('force_accelerate_hooks', 'decorator', 'wrapped')), 'attention accelerate wrapper')
        _require(forward.__func__.__globals__ is vars(accelerate), 'attention wrapper globals differ')
        attention_closure = inspect.getclosurevars(forward.__func__).nonlocals
        _require(attention_closure.get('child_module_names') == ['conv1d'], 'attention wrapper child list differs')
        attention_forward = attention_closure.get('forward_func')
        _require_code(attention_forward, _code_at(compiled, ('Glm5NextTextLinearAttention', 'forward')), 'attention forward')
        _require(attention_forward.__globals__ is vars(modeling), 'attention dispatch globals differ')
        gate = module.forget_gate
        _require(type(gate) is modeling.Glm5NextTextForgetGate, 'forget-gate class substitution')
        _require('forward' not in vars(gate), 'instance gate forward substitution')
        _require_code(gate.forward.__func__, _code_at(compiled, ('Glm5NextTextForgetGate', 'forward')), 'forget gate')
        _require(gate.forward.__func__.__globals__ is vars(modeling), 'forget gate globals differ')
        config = getattr(model.config, 'text_config', model.config)
        configured = getattr(config, 'linear_lower_bound', 'missing')
        bound = gate.safe_gate_lower_bound
        _require(configured == bound, 'live gate bound differs from actual config')
        if bound is not None:
            _require(type(bound) in (int, float) and math.isfinite(bound) and bound <= 0,
                     'gate lower bound must be finite and nonpositive')
            proof = dict(branch='safe_lower_bound_times_sigmoid', lower_bound=bound)
        else:
            proof = dict(branch='negative_exp_A_times_nonnegative_softplus', lower_bound=None)
        gates[name] = dict(**proof, heads=gate.num_heads, head_dim=gate.head_dim)
    _require(bool(gates), 'no actual GLM KDA modules observed')
    return dict(declaration=declaration(), modeling_sha256=CORRECTED_MODELING_SHA256,
                hub_kernels_sha256=build['hub_kernels_sha256'], gates=gates,
                accelerate_integration_sha256=ORIGINAL_ACCELERATE_INTEGRATION_SHA256,
                image_content_sha256=build['corrected_image_content_sha256'])


def bind_source_derivative(model, profile, value):
    """Issue a model-local binding only after observing the declared runtime."""
    policy = normalize_source_derivative(value)
    if policy is None:
        _require(model not in _BINDINGS, 'cannot remove a live derivative binding')
        _reject_unbound_corrected_runtime(model)
        return None
    _require(profile.source_derivative_contract() == declaration(), 'profile does not declare this correction')
    build = validate_image_build(bound_json(policy['image_build'], 'image build'))
    _require(os.environ.get('PRISMAQUANT_CONTAINER_CONTENT_SHA256') == build['corrected_image_content_sha256'],
             'actual container image content differs from build')
    observed = _observe(model, build)
    identity = dict(**observed, image_build_sha256=policy['image_build']['sha256'])
    prior = _BINDINGS.get(model)
    _require(prior is None or prior['identity'] == identity, 'cannot change a live derivative binding')
    _BINDINGS[model] = dict(identity=identity, policy=policy, build=build)
    return json.loads(json.dumps(identity))


def _reject_unbound_corrected_runtime(model):
    seen = set()
    for _name, module in model.named_modules():
        name = type(module).__module__
        if not name.startswith('transformers.models.glm5_next.') or name in seen:
            continue
        seen.add(name)
        loaded = sys.modules.get(name)
        path = getattr(loaded, '__file__', None)
        _require(path is not None, 'actual GLM source module is unavailable')
        _require(sha256(path) != CORRECTED_MODELING_SHA256,
                 'corrected GLM runtime requires an explicit derivative binding')


def source_derivative_identity(model):
    try:
        binding = _BINDINGS.get(model)
    except TypeError:
        binding = None  # Legacy identity accepts lightweight non-weakrefable model fixtures.
    if binding is None:
        _reject_unbound_corrected_runtime(model)
        return None
    build = bound_json(binding['policy']['image_build'], 'image build')
    _require(build == binding['build'], 'image build changed after binding')
    _require(os.environ.get('PRISMAQUANT_CONTAINER_CONTENT_SHA256') == build['corrected_image_content_sha256'],
             'actual image evidence changed after binding')
    observed = _observe(model, build)
    identity = dict(**observed, image_build_sha256=binding['policy']['image_build']['sha256'])
    _require(identity == binding['identity'], 'source derivative execution changed')
    return identity
