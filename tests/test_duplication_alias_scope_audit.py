"""Real digest_sites on owned source trees; source snippets are never executed.

No scanner/callee/alias mocks or doubles. The test imports the actual public
inventory API only when root's admitted PB run executes this file. It establishes
static primitive-site classification, not runtime digest or campaign behavior.
"""
from pathlib import Path

import pytest

from tools import duplication_inventory as inventory


def _sites(tmp_path: Path, source: str) -> list[str]:
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "case.py").write_text(source, encoding="utf-8")
    return inventory.digest_sites(tmp_path)


@pytest.mark.parametrize("family,import_line,expression", [
    ("hashlib", "import hashlib as h", "h.sha256(b'owned input').hexdigest()"),
    ("sorted-json", "import json as h", "h.dumps({'owned': 1}, sort_keys=True)"),
], ids=["hashlib", "literal_sorted_json"])
@pytest.mark.parametrize("placement", ["module", "outer_function"])
def test_unrelated_local_import_cannot_hide_outer_primitive_site(
    tmp_path: Path, family: str, import_line: str, expression: str, placement: str
):
    if placement == "module":
        outer = f"result = {expression}\n"
        scope = "<module>"
    else:
        outer = f"def outer():\n    return {expression}\n"
        scope = "outer"
    source = (
        f"{import_line}\n{outer}\n"
        "def unrelated_local():\n"
        "    import math as h\n"
        "    return h.sqrt(4.0)\n"
    )
    assert _sites(tmp_path, source) == [f"{family}:tools/case.py::{scope}"], (
        "A different function's local import must not replace the outer "
        "primitive import binding."
    )


@pytest.mark.parametrize("placement", ["module", "outer_function"])
def test_unrelated_aliases_preserve_both_existing_primitive_families(tmp_path: Path, placement: str):
    expression = "h.sha256(j.dumps({'owned': 1}, sort_keys=True).encode()).hexdigest()"
    if placement == "module":
        outer = f"result = {expression}\n"
        scope = "<module>"
    else:
        outer = f"def outer():\n    return {expression}\n"
        scope = "outer"
    source = (
        "import hashlib as h\nimport json as j\n"
        f"{outer}\n"
        "def unrelated_local():\n"
        "    import math as other\n"
        "    return other.sqrt(4.0)\n"
    )
    assert _sites(tmp_path, source) == [
        f"hashlib:tools/case.py::{scope}",
        f"sorted-json:tools/case.py::{scope}",
    ]


@pytest.mark.parametrize("local_alias", ["h", "digestlib"], ids=["same_name_cannot_leak", "different_name_control"])
def test_function_local_digest_import_cannot_relabel_outer_math_call(tmp_path: Path, local_alias: str):
    source = (
        "import math as h\n"
        "def ordinary_outer():\n"
        "    return h.sqrt(4.0)\n\n"
        "def local_digest():\n"
        f"    import hashlib as {local_alias}\n"
        f"    return {local_alias}.sha256(b'owned input').hexdigest()\n"
    )
    assert _sites(tmp_path, source) == ["hashlib:tools/case.py::local_digest"], (
        "Local hashlib import may classify its own call only; an outer math "
        "call must not become a false primitive digest site."
    )


@pytest.mark.parametrize("source,expected", [
    (
        "import hashlib as h\n"
        "def outer():\n"
        "    def inner():\n"
        "        return h.sha256(b'inner')\n"
        "    return h.sha256(b'outer'), inner\n",
        ["hashlib:tools/case.py::outer", "hashlib:tools/case.py::outer.inner"],
    ),
    (
        "import hashlib as h\n"
        "def outer():\n"
        "    def inner():\n"
        "        import json as h\n"
        "        return h.dumps({}, sort_keys=True)\n"
        "    return h.sha256(b'outer'), inner\n",
        ["hashlib:tools/case.py::outer", "sorted-json:tools/case.py::outer.inner"],
    ),
    (
        "import hashlib as h\n"
        "class Owner:\n"
        "    import math as h\n"
        "    value = h.sqrt(4)\n"
        "    def method(self):\n"
        "        return h.sha256(b'module binding')\n",
        ["hashlib:tools/case.py::Owner.method"],
    ),
    (
        "import math as h\n"
        "class Owner:\n"
        "    import hashlib as h\n"
        "    value = h.sha256(b'class binding')\n"
        "    def method(self):\n"
        "        return h.sqrt(4)\n",
        ["hashlib:tools/case.py::Owner"],
    ),
    (
        "from hashlib import sha256 as digest\n"
        "from json import JSONEncoder as Encoder\n"
        "def outer():\n"
        "    def inner():\n"
        "        from math import sqrt as digest\n"
        "        return digest(4)\n"
        "    return digest(b'outer'), Encoder(sort_keys=True), inner\n",
        ["hashlib:tools/case.py::outer", "sorted-json:tools/case.py::outer"],
    ),
    (
        "def outer():\n"
        "    try:\n"
        "        import hashlib as h\n"
        "    except ImportError:\n"
        "        raise\n"
        "    def inner():\n"
        "        return h.sha256(b'enclosing function')\n"
        "    return inner\n",
        ["hashlib:tools/case.py::outer.inner"],
    ),
    (
        "def outer():\n"
        "    import hashlib as h\n"
        "    class Owner:\n"
        "        import math as h\n"
        "        def method(self):\n"
        "            return h.sha256(b'enclosing function')\n"
        "    return Owner\n",
        ["hashlib:tools/case.py::outer.Owner.method"],
    ),
], ids=["module-inheritance", "nested-local", "method-skips-class",
        "class-body-import", "imported-constructors", "same-owner-control-flow",
        "method-enclosing-function"])
def test_lexical_owners_classify_only_their_import_environments(tmp_path, source, expected):
    assert _sites(tmp_path, source) == expected


@pytest.mark.parametrize("source,scope", [
    (
        "import hashlib as h\n"
        "def defaulted(value=h.sha256(b'default')):\n"
        "    import math as h\n"
        "    return h.sqrt(4)\n",
        "defaulted",
    ),
    (
        "import hashlib as h\n"
        "@decorate(h.sha256(b'decorator'))\n"
        "def decorated():\n"
        "    import math as h\n"
        "    return h.sqrt(4)\n",
        "decorated",
    ),
    (
        "import hashlib as h\n"
        "class Derived(base_for(h.sha256(b'base'))):\n"
        "    import math as h\n"
        "    value = h.sqrt(4)\n",
        "Derived",
    ),
    (
        "import math as h\n"
        "class Owner:\n"
        "    import hashlib as h\n"
        "    def method(self, value=h.sha256(b'class default')):\n"
        "        return h.sqrt(4)\n",
        "Owner.method",
    ),
], ids=["function-default", "function-decorator", "class-base", "method-default"])
def test_definition_headers_use_enclosing_aliases_without_renaming_sites(tmp_path, source, scope):
    assert _sites(tmp_path, source) == [f"hashlib:tools/case.py::{scope}"]


@pytest.mark.parametrize("expression", [
    "(lambda: h.sha256(b'lambda'))()",
    "[h.sha256(b'list') for item in ()]",
    "{h.sha256(b'set') for item in ()}",
    "{item: h.sha256(b'dict') for item in ()}",
    "tuple(h.sha256(b'generator') for item in ())",
])
def test_class_expression_scopes_use_the_enclosing_module_import(tmp_path, expression):
    source = (
        "import hashlib as h\n"
        "class Owner:\n"
        "    import math as h\n"
        f"    value = {expression}\n"
    )
    assert _sites(tmp_path, source) == ["hashlib:tools/case.py::Owner"]


def test_class_comprehension_first_iterator_keeps_class_import(tmp_path):
    source = (
        "import math as h\n"
        "class Owner:\n"
        "    import hashlib as h\n"
        "    value = [h.sqrt(4) for item in values(h.sha256(b'iterator'))]\n"
    )
    assert _sites(tmp_path, source) == ["hashlib:tools/case.py::Owner"]
