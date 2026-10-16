"""Native numerical inputs are explicit: no codec function chooses its recipe.

E1 binds every native render to one immutable ``NativeRenderLevers``.  A codec
function that falls back to the environment, a static default or exporter
globals when a caller omits the value silently renders a different recipe --
which is how the streamed cache fill priced fused NVFP4 siblings on a joint
global export never used.  These checks keep that fallback unrepresentable:
every native numerical function takes its recipe as a required keyword, no
codec module resolves one, and every production call site names it, including
paths only a GPU or a large model reaches.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEC_MODULES = (
    "prismaquant/native_render_codec.py",
    "prismaquant/export_batched_gptq.py",
    "prismaquant/gguf_gptq.py",
    "prismaquant/kernels/nvfp4_fused.py",
)
#: Recipe-taking helpers outside the codec modules whose callers include the
#: cache fills (the fused-sibling joint global and the render-score records).
RECIPE_HELPERS = {
    "prismaquant/export_native_compressed.py": {
        "_compute_nvfp4_joint_global", "_compute_layer_joint_nvfp4",
    },
    "prismaquant/production_weight_cache.py": {
        "_render_score_record", "_packed_expert_render_score_record",
    },
}
#: Diagnostic switches that change logging only, never a rendered byte.
DIAGNOSTIC_ENVIRONMENT = {
    "PRISMAQUANT_DAMP_SWEEP_LOG", "PRISMAQUANT_DO_NO_HARM_VERBOSE",
}


def _tree(relative: str) -> ast.Module:
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def _levers_functions(relative: str, only=None):
    for node in ast.walk(_tree(relative)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if only is not None and node.name not in only:
            continue
        arguments = node.args
        if any(a.arg == "levers" for a in arguments.posonlyargs + arguments.args):
            yield node, "positional"
            continue
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            if argument.arg == "levers":
                yield node, "required" if default is None else "defaulted"


def _required_recipe_functions() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for relative in CODEC_MODULES:
        for node, kind in _levers_functions(relative):
            found.setdefault(node.name, set()).add(relative)
    for relative, names in RECIPE_HELPERS.items():
        for node, kind in _levers_functions(relative, only=names):
            found.setdefault(node.name, set()).add(relative)
    return found


def test_native_numerical_functions_require_their_recipe():
    offenders = []
    for relative in CODEC_MODULES:
        for node, kind in _levers_functions(relative):
            if kind != "required":
                offenders.append(f"{relative}:{node.lineno} {node.name} ({kind})")
    for relative, names in RECIPE_HELPERS.items():
        seen = set()
        for node, kind in _levers_functions(relative, only=names):
            seen.add(node.name)
            if kind != "required":
                offenders.append(f"{relative}:{node.lineno} {node.name} ({kind})")
        assert seen == names, f"{relative}: recipe helpers moved: {names - seen}"
    assert not offenders, "levers must be a required keyword:\n" + "\n".join(offenders)


def test_codec_modules_resolve_no_recipe_and_read_no_numerical_environment():
    offenders = []
    for relative in CODEC_MODULES:
        for node in ast.walk(_tree(relative)):
            if (isinstance(node, ast.Attribute) and node.attr in {"resolve", "from_record"}
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "NativeRenderLevers"):
                offenders.append(f"{relative}:{node.lineno} NativeRenderLevers.{node.attr}")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"get", "__getitem__"}
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "environ"):
                name = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
                if name not in DIAGNOSTIC_ENVIRONMENT:
                    offenders.append(f"{relative}:{node.lineno} os.environ read {name!r}")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "getenv"):
                offenders.append(f"{relative}:{node.lineno} os.getenv")
    assert not offenders, (
        "codec modules must receive their recipe, never resolve it:\n"
        + "\n".join(offenders)
    )


def test_every_live_call_site_names_the_recipe():
    required = _required_recipe_functions()
    defined_elsewhere = set()
    owners = set(CODEC_MODULES) | set(RECIPE_HELPERS)
    for top in ("prismaquant", "tools"):
        for path in sorted((ROOT / top).rglob("*.py")):
            relative = str(path.relative_to(ROOT))
            if relative in owners:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in required:
                    defined_elsewhere.add(node.name)
    names = set(required) - defined_elsewhere
    assert {"_quantize_2d", "quantize_dequantize_nvfp4", "gptq_fields",
            "_compute_nvfp4_joint_global"} <= names

    offenders = []
    for top in ("prismaquant", "tools", "experiments"):
        for path in sorted((ROOT / top).rglob("*.py")):
            if "archive" in path.parts:
                continue
            relative = str(path.relative_to(ROOT))
            tree = ast.parse(path.read_text(encoding="utf-8"))
            aliases = {}
            for node in ast.walk(tree):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)):
                    value = node.value
                    target = (value.attr if isinstance(value, ast.Attribute)
                              else value.id if isinstance(value, ast.Name) else None)
                    if target in names:
                        aliases[node.targets[0].id] = target
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (func.attr if isinstance(func, ast.Attribute)
                        else func.id if isinstance(func, ast.Name) else None)
                name = aliases.get(name, name) if isinstance(func, ast.Name) else name
                if name not in names:
                    continue
                keywords = {keyword.arg for keyword in node.keywords}
                if "levers" not in keywords and None not in keywords:
                    offenders.append(f"{relative}:{node.lineno} {name}")
    assert not offenders, "call sites must pass levers=:\n" + "\n".join(offenders)
