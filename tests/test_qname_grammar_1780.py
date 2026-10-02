"""Exact roster grammar ownership; re.compile's cache is not routing proof."""
import ast
from importlib import import_module
from pathlib import Path
import re

import pytest


CALLERS = [
    ("prismaquant.joint_cost_read_schedule", "_LAYER"),
    ("prismaquant.joint_layer_quanta", "_QNAME_LAYER"),
]
NAMES = [
    "model.layers.0", "model.layers.001.attn.q_proj", "a.layers.1.layers.37.x",
    "model.layers.٣.weight", "模型.layers.４.attn", ".layers.7.",
    "layers.7.x", "model.layers.-1.x", "model.layers.1x", "model.layers..x",
    "model.layers.².x", "model.layers.3\n", "model\n.layers.3.x", "",
    "model.layers.3.\n", "model.layers.1.layers.bad", "model.layers.999.x",
]


@pytest.mark.parametrize("module_name,alias", CALLERS)
def test_callers_route_to_shared_grammar(module_name, alias):
    module = import_module(module_name)
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.level == 1 and node.module == "qnames"]
    assert [(name.name, name.asname) for node in imports for name in node.names] == [
        ("LAYER_QNAME", alias)]
    # An independent raw compile would silently appear identical through re's cache.
    assert not any(isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == alias
                           for target in node.targets) for node in tree.body)


def _match_record(pattern, name):
    match = pattern.match(name)
    return None if match is None else (match.group(0), match.groups(), match.span())


@pytest.mark.parametrize("name", NAMES)
def test_literal_old_grammar_match_identity(name):
    from prismaquant.qnames import LAYER_QNAME
    legacy = re.compile(r"^.*\.layers\.(\d+)(?:\.|$)")
    assert LAYER_QNAME.pattern == legacy.pattern
    assert LAYER_QNAME.flags == legacy.flags == re.UNICODE
    assert LAYER_QNAME.groups == legacy.groups == 1
    assert _match_record(LAYER_QNAME, name) == _match_record(legacy, name)
    for module_name, alias in CALLERS:
        assert getattr(import_module(module_name), alias) is LAYER_QNAME


class StringSubclass(str):
    pass


@pytest.mark.parametrize("value", [None, True, 1, [], StringSubclass("m.layers.1.x")])
def test_exact_string_policy_stays_with_caller(value):
    from prismaquant.joint_layer_quanta import qname_layer
    assert qname_layer(value) is None


@pytest.mark.parametrize("name", NAMES)
def test_qname_layer_return_policy_matches_literal(name):
    from prismaquant.joint_layer_quanta import qname_layer
    match = re.compile(r"^.*\.layers\.(\d+)(?:\.|$)").match(name)
    assert qname_layer(name) == (None if match is None else int(match.group(1)))


def test_roster_grouping_and_refusal_stay_with_caller():
    from prismaquant.joint_layer_quanta import _layer_qnames
    assert _layer_qnames({"formats_by_qname": {
        "m.layers.2.z": {}, "m.layers.1.x": {}, "m.layers.2.a": {}}}) == {
            2: ["m.layers.2.a", "m.layers.2.z"], 1: ["m.layers.1.x"]}
    with pytest.raises(ValueError, match="roster qname 'not-a-layer' names no layer"):
        _layer_qnames({"formats_by_qname": {"not-a-layer": {}}})


@pytest.mark.parametrize('name', NAMES)
def test_dotted_component_preserves_the_distinct_literal_search(name):
    from prismaquant.qnames import DOTTED_LAYER_QNAME
    from prismaquant.sensitivity_card_build import _layer_of
    legacy = re.compile(r'\.layers\.(\d+)\.')
    expected, actual = legacy.search(name), DOTTED_LAYER_QNAME.search(name)
    def record(match):
        return None if match is None else (match.group(0), match.groups(), match.span())
    assert record(actual) == record(expected)
    assert _layer_of(name) == (int(expected.group(1)) if expected else None)


@pytest.mark.parametrize('stride', [0, 1, 2, 7])
def test_campaign_layer_stride_keeps_its_own_unmatched_name_policy(stride):
    from prismaquant.tessera_campaign import _campaign_layer_scope
    legacy = re.compile(r'\.layers\.(\d+)\.')
    expected = [name for name in NAMES if stride <= 1 or legacy.search(name) is None
                or int(legacy.search(name).group(1)) % stride == 0]
    assert _campaign_layer_scope(NAMES, stride) == expected


def test_all_three_dotted_search_sites_import_the_same_owner():
    from prismaquant import sensitivity_card_build as card
    from prismaquant.qnames import DOTTED_LAYER_QNAME
    assert card._LAYER_RE is DOTTED_LAYER_QNAME
    tree = ast.parse(Path(card.__file__).read_text())
    assert any(isinstance(node, ast.ImportFrom) and node.module == 'qnames'
               and any(alias.name == 'DOTTED_LAYER_QNAME' and alias.asname == '_LAYER_RE'
                       for alias in node.names) for node in tree.body)
    assert not any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                   and target.id == '_LAYER_RE' for target in node.targets) for node in tree.body)
    for module_name, function_name in [('prismaquant.tessera_campaign', '_campaign_layer_scope'),
            ('prismaquant.export_native_compressed', '_bf16_packed_expert_ignore_regex')]:
        module = import_module(module_name)
        tree = ast.parse(Path(module.__file__).read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == function_name)
        assert any(isinstance(node, ast.ImportFrom) and node.module == 'qnames'
                   and any(alias.name == 'DOTTED_LAYER_QNAME' for alias in node.names)
                   for node in ast.walk(fn))
        assert not any(isinstance(node, ast.Constant) and node.value == r'\.layers\.(\d+)\.'
                       for node in ast.walk(fn))


@pytest.mark.parametrize('prefix', ['model', 'mtp'])
def test_export_bf16_packed_ignore_keeps_body_mtp_namespaces(prefix):
    from prismaquant.export_native_compressed import _bf16_packed_expert_ignore_regex
    patterns = _bf16_packed_expert_ignore_regex(f'{prefix}.layers.7.moe.experts.gate_up_proj', None)
    assert patterns
    compiled = [re.compile(pattern.removeprefix('re:')) for pattern in patterns]
    assert any(pattern.search(f'{prefix}.layers.7.moe.experts.1.gate_proj') for pattern in compiled)
    assert not any(pattern.search(f'{prefix}.layers.8.moe.experts.1.gate_proj') for pattern in compiled)
    foreign_prefix = 'model' if prefix == 'mtp' else 'mtp'
    assert not any(pattern.search(f'{foreign_prefix}.layers.7.moe.experts.1.gate_proj') for pattern in compiled)
