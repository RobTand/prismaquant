import ast
import base64
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import allocator, digests
from tools import build_glm_derivative_image, capture_glm_routed_layers, capture_glm_routing_replay
from tools import glm_mtp_capture, materialize_wikitext_corpus

FIXTURE = json.loads(r'''{"archive_member":"config.json","bytes_hex":["","636166c3a90a","00ff0a610062","610a","61"],"json_cases":[{"entries":[["b",1],["a",["caf\u00e9",true,null]]],"name":"ordered-unicode"},{"entries":[[1,"x"],["a",2]],"name":"mixed-keys"},{"entries":[["x","NaN"]],"name":"nonfinite","nan_keys":["x"]},{"entries":[["x","\ud800"]],"name":"surrogate"}],"texts":["","caf\u00e9","\ud800","a\nb\n"]}''')
SITES = {
    1: (allocator, 'main', 'assign', 'digest', 'digest_src'),
    4: (build_glm_derivative_image, 'main', 'assign', 'config_digest', 'encoded'),
    5: (build_glm_derivative_image, 'main', 'verify', 'config_digest', 'outer'),
    6: (build_glm_derivative_image, 'main', 'verify', 'layer_sha256', 'actual_layer'),
    7: (build_glm_derivative_image, 'main', 'keyword', 'hub_kernels_sha256', 'read'),
    8: (build_glm_derivative_image, 'main', 'assign', 'layer_sha256', 'added_layer'),
    9: (capture_glm_routed_layers, 'main.publish', 'field', 'sha256', 'raw'),
    10: (capture_glm_routing_replay, 'main', 'field', 'boundary.sha256', 'raw'),
    11: (glm_mtp_capture, 'projection_phase', 'field', 'projection.sha256', 'raw'),
    13: (materialize_wikitext_corpus, 'main', 'field', 'sha256', 'raw'),
}


def _fields(node, keys):
    if not keys:
        yield node
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == keys[0]:
                yield from _fields(value, keys[1:])


def _expressions(scope, kind, selector, input_name):
    for node in ast.walk(scope):
        if kind == 'assign' and isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == selector for t in node.targets):
                if any(isinstance(n, ast.Name) and n.id == input_name for n in ast.walk(node.value)):
                    yield node.value
        elif kind == 'verify' and isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Compare) and ast.unparse(test.comparators[0]) == selector:
                yield test.left
        elif kind == 'keyword' and isinstance(node, ast.keyword) and node.arg == selector:
            yield node.value
        elif kind == 'field':
            yield from _fields(node, selector.split('.'))


@pytest.fixture(params=SITES, ids=lambda row: f'site-{row}')
def site(request):
    row = request.param
    module, scope_path, kind, selector, input_name = SITES[row]
    path = Path(module.__file__)
    scope = ast.parse(path.read_bytes(), filename=str(path))
    bindings = vars(module).copy()
    for name in scope_path.split('.'):
        scope, = (node for node in scope.body if isinstance(node, ast.FunctionDef) and node.name == name)
        for node in scope.body:
            if isinstance(node, ast.ImportFrom) and any(a.name == 'bytes_sha256hex' for a in node.names):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), bindings)
    expression, = _expressions(scope, kind, selector, input_name)
    return row, input_name, bindings, compile(ast.Expression(expression), str(path), 'eval')


def _inputs(row, name):
    if row == 1:
        for text in FIXTURE['texts']:
            yield {name: text}
        return
    member = FIXTURE['archive_member']
    for encoded in FIXTURE['bytes_hex']:
        value = bytes.fromhex(encoded)
        if row == 5:
            reader = SimpleNamespace(extractfile={member: io.BytesIO(value)}.__getitem__)
            yield {name: reader, 'config_name': member}
        elif row == 7:
            yield {name: {'hub': base64.b64encode(value).decode()}}
        else:
            yield {name: value}


def _outcomes(site, monkeypatch):
    row, name, bindings, expression = site
    sha256 = hashlib.sha256
    outcomes = []
    for locals_ in _inputs(row, name):
        inputs = []

        def observe(value):
            inputs.append(value.hex())
            return sha256(value)

        outcome = {'input_hex': inputs}
        with monkeypatch.context() as patch:
            patch.setattr(hashlib, 'sha256', observe)
            try:
                outcome['return'] = eval(expression, bindings, locals_)
            except UnicodeEncodeError as error:
                cls = type(error)
                outcome.update(type=f'{cls.__module__}.{cls.__qualname__}', message=str(error))
        outcomes.append(outcome)
    return outcomes


def _record_hash(value):
    return hashlib.sha256((json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
                           + '\n').encode('utf-8')).hexdigest()


def test_live_byte_digest_values_2646(site, monkeypatch):
    assert _record_hash(FIXTURE) == '18a216544bbe780f48b2163c67d505c72b563103af4df87938bbfbc60bb30e9c'
    outcomes = _outcomes(site, monkeypatch)
    if site[0] == 1:
        assert outcomes[0]['return'] == 'e3b0c44298fc'
        assert outcomes[2] == {
            'input_hex': [],
            'type': 'builtins.UnicodeEncodeError',
            'message': "'utf-8' codec can't encode character '\\ud800' in position 0: surrogates not allowed",
        }
        assert _record_hash(outcomes) == 'cb6a899d95e77aca843849c279e8212e479efc67e2ddea6ec358eb308df874e2'
    else:
        assert outcomes[0]['return'].startswith('e3b0c44298fc1c14')
        assert _record_hash(outcomes) == '69fd082555691010f9ed26e81eb5cf2a79a01b75f017427508e23825b41d38ae'


def test_live_byte_digest_routes_2646(site, monkeypatch):
    row, _, bindings, _ = site
    owner = digests
    if SITES[row][0] is build_glm_derivative_image:
        owner = sys.modules['digests']
        assert Path(owner.__file__).resolve() == Path(digests.__file__).resolve()
    delegate = bindings.get('bytes_sha256hex')
    assert delegate is owner.bytes_sha256hex
    calls = []

    def observe(value):
        calls.append(value.hex())
        return delegate(value)

    bindings['bytes_sha256hex'] = observe
    outcomes = _outcomes(site, monkeypatch)
    assert calls and calls == [value for outcome in outcomes for value in outcome['input_hex']]
