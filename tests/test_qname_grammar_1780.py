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
