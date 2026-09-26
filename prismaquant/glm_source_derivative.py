"""Closed opt-in GLM KDA derivative identity; never mutates model code.

Dispatch changes in one declared place only: ``CaptureKernelDispatch`` runs a
capture kernel this contract declares (``capture_kernel_declaration``) in
place of the verified Torch fallback for the length of one ``with`` block,
and restores the fallback on exit (PQ #1199).
"""
from __future__ import annotations

import hashlib
import importlib
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


#: Every definition in the module, as a whole (PQ #1341).
_MODULE_LEVEL = '<module>'


def original_source(raw):
    """Invert :func:`corrected_source`, refusing unless the result is the pinned original.

    The reviewed correction is one expression. A corrected file that differs
    from the pinned original anywhere else does not invert to it, so its
    reach cannot be derived and it is refused.
    """
    _require(hashlib.sha256(raw).hexdigest() == CORRECTED_MODELING_SHA256, 'corrected modeling source differs')
    old, new = ORIGINAL_EXPRESSION.encode(), CORRECTED_EXPRESSION.encode()
    _require(raw.count(new) == 1, 'reviewed corrected expression is not unique')
    result = raw.replace(new, old, 1)
    _require(hashlib.sha256(result).hexdigest() == ORIGINAL_MODELING_SHA256,
             'corrected source changes more than the reviewed expression')
    return result


def _definitions(raw):
    """``{key: (ast dump, referenced names)}`` for each definition in a module.

    Keys are a top-level function's name, ``Class.method`` for a method,
    ``Class.<body>`` for a class's bases, decorators and other statements,
    and ``<module>`` for all remaining module-level code. The references are
    every name a definition loads, plus ``Class.attr`` for ``self.attr`` or
    ``Class.attr`` inside class ``Class``.
    """
    import ast

    def refs(nodes, owner):
        names = set()
        for node in nodes:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
                elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
                    base = owner if sub.value.id in ('self', 'cls') else sub.value.id
                    if base is not None:
                        names.add(f'{base}.{sub.attr}')
        return names

    tree = ast.parse(raw)
    found, module_level = {}, []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = (ast.dump(node), refs([node], None))
        elif isinstance(node, ast.ClassDef):
            body = []
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f'{node.name}.{member.name}'] = (ast.dump(member), refs([member], node.name))
                else:
                    body.append(member)
            head = [*node.bases, *node.keywords, *node.decorator_list, *body]
            found[f'{node.name}.<body>'] = (repr([ast.dump(x) for x in head]), refs(head, node.name))
        else:
            module_level.append(node)
    found[_MODULE_LEVEL] = (repr([ast.dump(x) for x in module_level]), refs(module_level, None))
    return found


def correction_reach(original, corrected):
    """The classes whose code can run a definition the correction changed.

    A definition reaches the correction when it changed, or when it names a
    reaching function, class or method. Names are followed to a fixed point,
    so the result is closed under calls, subclassing and decoration within
    the module. The result is ``frozenset({'<module>'})`` when module-level
    code reaches it: then every class in the file is treated as reached.
    """
    before, after = _definitions(original), _definitions(corrected)
    reached = {key for key in set(before) | set(after) if before.get(key, (None,))[0] != after.get(key, (None,))[0]}

    def aliases(key):
        if '.' not in key:
            return {key}
        owner, member = key.split('.', 1)
        return {owner} | ({key} if member != '<body>' else set())

    while True:
        names = set().union(*(aliases(key) for key in reached)) if reached else set()
        grown = {key for key, (_dump, used) in after.items() if key not in reached and used & names}
        if not grown:
            break
        reached |= grown
    if _MODULE_LEVEL in reached:
        return frozenset({_MODULE_LEVEL})
    return frozenset(key.split('.', 1)[0] if '.' in key else key for key in reached)


_REACH = {}


def _corrected_reach(path):
    """``correction_reach`` of a loaded corrected modeling file, once per content."""
    raw = Path(path).read_bytes()
    key = (hashlib.sha256(raw).hexdigest(), ORIGINAL_MODELING_SHA256, CORRECTED_MODELING_SHA256)
    if key not in _REACH:
        _REACH[key] = correction_reach(original_source(raw), raw)
    return _REACH[key]


def _reject_unbound_corrected_runtime(model):
    """Refuse a model that can run the corrected source without a binding.

    Only the classes the correction reaches (:func:`correction_reach`) can
    run it. A model without an instance of one runs identical code on either
    source, so its identity is honestly unbound (PQ #1341: the GLM MTP layer
    has attention, MoE and norms from this file, and no KDA module).
    """
    reach = {}
    for _name, module in model.named_modules():
        name = type(module).__module__
        if not name.startswith('transformers.models.glm5_next.'):
            continue
        if name not in reach:
            loaded = sys.modules.get(name)
            path = getattr(loaded, '__file__', None)
            _require(path is not None, 'actual GLM source module is unavailable')
            if sha256(path) != CORRECTED_MODELING_SHA256:
                reach[name] = frozenset()
            else:
                try:
                    reach[name] = _corrected_reach(path)
                except ValueError as error:
                    raise ValueError('GLM source derivative: corrected GLM runtime requires an explicit '
                                     f'derivative binding; its reach cannot be derived ({error})') from error
        classes = reach[name]
        _require(_MODULE_LEVEL not in classes and type(module).__name__ not in classes,
                 f'corrected GLM runtime requires an explicit derivative binding '
                 f'({type(module).__name__} reaches the correction)')


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


#: Capture kernels this contract declares (PQ #1199). A declared kernel
#: computes this derivative's semantics (``implements``) with its own
#: rounding, and runs only inside a ``CaptureKernelDispatch`` block. Its
#: scope is every Stage B pass of a KDA layer, the target's capture passes
#: and the chain rolls, in a launch that names it (kernel mode, PQ #1214);
#: Stage A never runs it.
CAPTURE_KERNEL_SCHEMA = 'prismaquant.glm_source_derivative.capture_kernel.v1'
_CAPTURE_KERNELS = {
    'kda_gram_v1': dict(module='prismaquant.kernels.kda_chunk',
                        entry='chunk_kimi_delta_attention'),
}


def capture_kernel_declaration(name):
    """The closed declaration of one capture kernel; an undeclared name refuses."""
    _require(isinstance(name, str) and name in _CAPTURE_KERNELS, f'undeclared capture kernel {name!r}')
    return dict(schema=CAPTURE_KERNEL_SCHEMA, name=name, implements=VERSION,
                replaces='chunk_kimi_delta_attention',
                dispatch='module_global_substituted_for_one_block_then_restored',
                scope='stage_b_kda_layer_passes', **_CAPTURE_KERNELS[name])


class CaptureKernelDispatch:
    """Run a declared capture kernel in place of the verified fallback, one block at a time.

    Construction observes the bound runtime through ``source_derivative_identity``,
    so the model must carry this derivative's binding and the module global
    ``chunk_kimi_delta_attention`` must be the decorated Torch fallback. Each
    ``with`` block then points that global, which the attention forward reads
    at call time, at the kernel entry, and points it back at the fallback on
    exit. A global that is not the fallback on entry, or not the kernel on
    exit, refuses: something else changed dispatch. Blocks do not nest.
    Outside a block, ``source_derivative_identity`` observes the fallback as
    before, so every identity check between passes is unchanged; inside one,
    it refuses.
    """

    def __init__(self, model, name):
        declaration = capture_kernel_declaration(name)
        identity = source_derivative_identity(model)
        _require(identity is not None, 'a capture kernel requires the bound GLM derivative')
        _require(identity['declaration']['version'] == declaration['implements'],
                 'capture kernel implements another derivative')
        from transformers.models.glm5_next import modeling_glm5_next as modeling
        entry = getattr(importlib.import_module(declaration['module']), declaration['entry'])
        _require(isinstance(entry, types.FunctionType), 'capture kernel entry is not a function')
        self.declaration = declaration
        self.derivative = identity
        self._modeling = modeling
        # The object _observe authenticated just now, in this thread.
        self._fallback = modeling.chunk_kimi_delta_attention
        self._entry = entry
        self._active = False

    def __enter__(self):
        _require(not self._active, 'capture kernel blocks do not nest')
        _require(self._modeling.chunk_kimi_delta_attention is self._fallback,
                 'KDA dispatch changed outside a capture kernel block')
        self._modeling.chunk_kimi_delta_attention = self._entry
        self._active = True
        return self

    def __exit__(self, *exc_info):
        current = self._modeling.chunk_kimi_delta_attention
        self._modeling.chunk_kimi_delta_attention = self._fallback
        self._active = False
        _require(current is self._entry, 'KDA dispatch changed inside a capture kernel block')
        return False
