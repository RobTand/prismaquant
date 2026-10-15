"""Move existing replay bodies and every live caller to routed_experts.

Run on a pre-move checkout. Repeating on the migrated checkout is a no-op.
The JSON evidence records body equivalence after the public-name substitution.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re


RENAMES = {
    "_enumerate_packed_experts": "enumerate_packed_experts",
    "_packed_experts_parent_module": "packed_experts_parent_module",
    "_packed_experts_router": "packed_experts_router",
    "_packed_router_topk": "packed_router_topk",
    "_packed_experts_forward_with_weights": "packed_experts_forward_with_weights",
    "derive_per_expert_activations": "derive_per_expert_activations",
    "_routed_pair_order": "routed_pair_order",
}
RECEIPT_OPERATION = "prismaquant.measure_quant_cost._packed_experts_forward_with_weights"


def rename(text):
    for old, new in RENAMES.items():
        text = re.sub(r"\b" + old + r"\b", new, text)
    return text


def functions(text):
    return {node.name: node for node in ast.parse(text).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


class PublicNames(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = RENAMES.get(node.id, node.id)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.name = RENAMES.get(node.name, node.name)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            if isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        return node


def body_identity(node):
    normalized = PublicNames().visit(copy.deepcopy(node))
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.cwd
    cost_path = root / "prismaquant/measure_quant_cost.py"
    owner_path = root / "prismaquant/routed_experts.py"
    cost = cost_path.read_text()
    original = functions(cost)
    if "_enumerate_packed_experts" not in original:
        if all(name in functions(owner_path.read_text()) for name in RENAMES.values()):
            print("Routed replay already has one public owner")
            return
        raise RuntimeError("incomplete routed replay ownership")
    changed = []
    tests_before = {}
    for path in (root / "tests").glob("test_*.py"):
        tests_before[str(path.relative_to(root))] = sorted(
            node.name for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"))

    def write(relative, value):
        path = root / relative
        if path.read_text() != value:
            path.write_text(value)
            changed.append(relative)

    lines = cost.splitlines(keepends=True)
    start = original["_enumerate_packed_experts"].lineno - 1
    end = original["_routed_pair_order"].end_lineno
    moved = rename("".join(lines[start:end])).replace(
        "sensitivity_probe._is_packed_experts_module", "probe_inventory.is_packed_experts_module")
    cost = "".join(lines[:start] + lines[end + 2:])
    cost = cost.replace("from . import io_spans\n", "from . import io_spans\nfrom . import routed_experts\n", 1)
    for old, new in RENAMES.items():
        cost = re.sub(r"\b" + old + r"\b", "routed_experts." + new, cost)
    write("prismaquant/measure_quant_cost.py", cost)

    owner = owner_path.read_text().replace("import torch.nn as nn\n", "import torch.nn as nn\nimport torch.nn.functional as F\n", 1)
    owner = owner.replace("\n\ndef resolve_routed_expert_profile", "\n\n" + moved + "\n\n\ndef resolve_routed_expert_profile", 1)
    owner = owner.replace("    from .measure_quant_cost import _enumerate_packed_experts\n", "")
    owner = rename(owner)
    owner = owner.replace("__all__ = [\n", "__all__ = [\n" + "".join(f'    "{name}",\n' for name in RENAMES.values()), 1)
    owner = owner.replace('"""Profile-owned routed-expert discovery shared by cost estimators.',
                          '"""Profile-owned routed-expert discovery, routing and explicit replay.')
    write("prismaquant/routed_experts.py", owner)

    for relative in ["prismaquant/production_weight_cache.py", "prismaquant/tessera_campaign.py"]:
        text = (root / relative).read_text()
        text = text.replace("from prismaquant.measure_quant_cost import (\n        derive_per_expert_activations,",
                            "from prismaquant.routed_experts import (\n        derive_per_expert_activations,")
        text = text.replace("from .measure_quant_cost import (\n            _packed_experts_parent_module, derive_per_expert_activations,",
                            "from .routed_experts import (\n            _packed_experts_parent_module, derive_per_expert_activations,")
        write(relative, rename(text))

    relative = "prismaquant/native_moe_panel.py"
    text = (root / relative).read_text().replace(RECEIPT_OPERATION, "PRESERVED_ROUTED_RECEIPT_OPERATION")
    text = text.replace("from .measure_quant_cost import _packed_experts_forward_with_weights",
                        "from .routed_experts import _packed_experts_forward_with_weights")
    write(relative, rename(text).replace("PRESERVED_ROUTED_RECEIPT_OPERATION", RECEIPT_OPERATION))

    for relative in ["tests/test_derive_per_expert_routing_1931.py", "tests/test_packed_expert_hook_scope.py"]:
        text = (root / relative).read_text()
        text = text.replace("from prismaquant import measure_quant_cost as mqc", "from prismaquant import routed_experts")
        text = text.replace("import prismaquant.measure_quant_cost as mqc", "from prismaquant import routed_experts")
        text = text.replace("mqc.", "routed_experts.").replace("setattr(mqc,", "setattr(routed_experts,")
        text = text.replace("from prismaquant.measure_quant_cost import _packed_experts_router, _packed_router_topk",
                            "from prismaquant.routed_experts import _packed_experts_router, _packed_router_topk")
        write(relative, rename(text))

    for relative in ["experiments/glm_shared_input_replay.py", "tests/test_glm_shared_input_replay.py"]:
        write(relative, (root / relative).read_text().replace("measure_quant_cost", "routed_experts"))

    relative = "experiments/profile_expert_routing_1931.py"
    text = (root / relative).read_text().replace("measure_quant_cost as mqc", "routed_experts")
    write(relative, text.replace("mqc.", "routed_experts."))

    for relative in ["experiments/lfm_packed_joint_screen.py", "tests/test_lfm_parent_router_bias.py",
                     "tests/test_native_moe_panel.py", "tests/test_streaming_production_cache.py"]:
        text = (root / relative).read_text()
        for name in RENAMES:
            text = text.replace("from prismaquant.measure_quant_cost import " + name,
                                "from prismaquant.routed_experts import " + name)
        write(relative, rename(text))

    relative = "tests/test_measure_quant_cost_packed_experts.py"
    text = (root / relative).read_text()
    for name in ["_packed_experts_forward_with_weights", "_packed_router_topk"]:
        text = text.replace("    " + name + ",\n", "")
    text = text.replace("\n\nclass _StubHDetail", "\nfrom prismaquant.routed_experts import (\n    packed_experts_forward_with_weights,\n    packed_router_topk,\n)\n\n\nclass _StubHDetail", 1)
    write(relative, rename(text))

    relative = "prismaquant/tessera_campaign.py"
    text = (root / relative).read_text().replace(
        "            resolve_routed_expert_profile,\n        )\n        from .routed_experts import (\n            packed_experts_parent_module, derive_per_expert_activations,",
        "            resolve_routed_expert_profile,\n            packed_experts_parent_module,\n            derive_per_expert_activations,")
    write(relative, text)

    relative = "docs/ARCHITECTURE.md"
    text = (root / relative).read_text()
    paragraph = ("Re-stamped 2026-10-15: `routed_experts` owns packed discovery, parent/router\n"
                 "lookup, router protocol normalization, explicit-weight replay, deterministic\n"
                 "per-expert activation derivation and stable routed-pair ordering. Measurement,\n"
                 "cache fill, campaign capture and native panel reference replay use its public\n"
                 "functions. Historical native panel receipt operation names and versions remain\n"
                 "unchanged; this ownership move changes no routing or numerical policy.\n\n")
    write(relative, text.replace("# PrismaQuant Architecture\n\n", "# PrismaQuant Architecture\n\n" + paragraph, 1))

    after = functions(owner_path.read_text())
    evidence = {old: {"owner": new, "before": body_identity(original[old]),
                      "after": body_identity(after[new])} for old, new in RENAMES.items()}
    if any(row["before"] != row["after"] for row in evidence.values()):
        raise RuntimeError("routed replay bodies differ beyond names and docstrings")
    test_changes = {}
    for name, previous in tests_before.items():
        current = sorted(node.name for node in ast.walk(ast.parse((root / name).read_text()))
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"))
        if current != previous:
            test_changes[name] = {"before": previous, "after": current}
    if test_changes:
        raise RuntimeError(f"test deletion/renaming unexpected: {test_changes}")
    args.evidence.write_text(json.dumps({"bodies": evidence, "changed_files": sorted(set(changed)),
                                         "test_method_changes": test_changes,
                                         "receipt_operation": RECEIPT_OPERATION}, indent=2) + "\n")
    print(json.dumps({"moved_bodies": len(evidence), "changed_files": len(set(changed)), "test_method_changes": 0}))


if __name__ == "__main__":
    main()
