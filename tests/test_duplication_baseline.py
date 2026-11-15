"""Duplication can shrink without a fixture edit (PQ #1295, #1508).

The fixture is the absolute ceiling for near-duplicate pairs, same-name
helper members, and primitive digest sites outside the digest owners.
The PQ graph-impact adapter also refuses growth against the live base
inventory. PR mode uses the merge base. Queue mode uses the exact batch base.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import duplication_inventory as inventory  # noqa: E402


CONSOLIDATED_HELPERS = [
    ("file_sha256hex", "prismaquant/digests.py"),
    ("_canon_fmt", "prismaquant/streaming_production_cache.py"),
    ("nvfp4_activation_qdq_served", "prismaquant/nvfp4_activation_contract.py"),
    ("_optional_str", "prismaquant/model_profiles/structure.py"),
]
DIGEST_FREE = [
    "prismaquant/visual_linears.py",
    "prismaquant/tessera_census_cache.py",
]


@pytest.mark.parametrize("name,owner", CONSOLIDATED_HELPERS)
def test_a_consolidated_helper_is_defined_only_in_its_owner(name, owner):
    assert [path.relative_to(ROOT).as_posix() for path in inventory._files(ROOT)
            if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
                   for node in ast.parse(path.read_text(encoding="utf-8")).body)] == [owner]


@pytest.mark.parametrize("module", DIGEST_FREE)
def test_a_digest_free_module_has_no_primitive_digest_site(module):
    assert module not in inventory.DIGEST_OWNERS
    assert not inventory._digest_sites(module, ast.parse((ROOT / module).read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def root_inventory():
    """One read-only root report; synthetic tests scan only their tmp roots."""
    return MappingProxyType(inventory.scan())


def _baseline():
    return json.loads(inventory.BASELINE.read_text(encoding="utf-8"))


def test_scanner_finds_a_renamed_copy_and_ignores_same_file_pairs(tmp_path):
    body = (
        "    total = 0\n"
        "    for item in items:\n"
        "        if item.ready:\n"
        "            total += item.size\n"
        "        else:\n"
        "            total -= helper(item)\n"
        "    return {'total': total, 'count': len(items)}\n"
    )
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "prismaquant" / "a.py").write_text(
        "def _sha(items):\n" + body + "\ndef other(things):\n" + body.replace("items", "things"))
    (tmp_path / "tools" / "b.py").write_text(
        "def _sha(rows):\n" + body.replace("items", "rows").replace("total", "acc"))
    live = inventory.scan(tmp_path)
    assert ["prismaquant/a.py::_sha", "tools/b.py::_sha"] in live["near_duplicates"]
    assert ["prismaquant/a.py::other", "tools/b.py::_sha"] in live["near_duplicates"]
    assert all(a.split("::")[0] != b.split("::")[0] for a, b in live["near_duplicates"])
    assert live["same_name_helpers"] == {"_sha": ["prismaquant/a.py", "tools/b.py"]}


@pytest.mark.parametrize("empty", [False, True])
def test_duplication_only_shrinks(root_inventory, empty):
    live = {key: type(value)() for key, value in root_inventory.items()} if empty else root_inventory
    assert not inventory.growth(_baseline(), live), "new duplication: reuse the existing owner"


def test_must_differ_pairs_are_live_and_give_a_reason():
    """A pair kept as two implementations says why, and still exists (#1302)."""
    base = _baseline()
    pairs = {tuple(p) for p in base["near_duplicates"]}
    seen = set()
    for row in base.get("must_differ", []):
        assert set(row) == {"pair", "reason"}, row
        pair = tuple(row["pair"])
        assert pair in pairs, f"must_differ names a pair the baseline lacks: {pair}"
        assert pair not in seen, f"must_differ repeats {pair}"
        seen.add(pair)
        assert len(row["reason"].split()) >= 8, f"{pair} needs a real reason"


def test_the_digest_site_ratchet_fails_on_a_new_raw_site_and_passes_the_owner(tmp_path):
    """A raw digest site fails outside its owner and passes inside its owner."""
    raw = (
        "import hashlib\nimport json\n"
        "def _digest(value):\n"
        "    return hashlib.sha256(\n"
        "        json.dumps(value, sort_keys=True).encode()).hexdigest()\n"
    )
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "prismaquant" / "digests.py").write_text(raw)
    (tmp_path / "tools" / "fresh_site.py").write_text(raw)
    live = set(inventory.scan(tmp_path)["primitive_digest_sites"])
    assert not any("prismaquant/digests.py" in site for site in live)
    assert sorted(live - set()) == [
        "hashlib:tools/fresh_site.py::_digest",
        "sorted-json:tools/fresh_site.py::_digest",
    ]


def test_image_side_patch_scripts_are_outside_the_scan_and_their_loader_is_not(tmp_path):
    """Image-side scripts stay exempt. Their loader stays in the scan."""
    raw = ("import hashlib\nfrom pathlib import Path\n"
           "DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()\n")
    patch_dir = tmp_path / "prismaquant" / "serving_runtime_patches" / "some_set"
    patch_dir.mkdir(parents=True)
    (tmp_path / "tools").mkdir()
    (patch_dir / "patch_some_set.py").write_text(raw)
    (tmp_path / "prismaquant" / "serving_runtime_patch_set.py").write_text(raw)
    live = set(inventory.scan(tmp_path)["primitive_digest_sites"])
    assert live == {"hashlib:prismaquant/serving_runtime_patch_set.py::<module>"}


def test_shared_stdlib_core_has_an_independent_helper_namespace(tmp_path):
    body = ('import hashlib\n'
            'def summary(rows):\n'
            '    total = 0\n'
            '    for row in rows:\n'
            '        if row.ready:\n'
            '            total += row.size\n'
            '        else:\n'
            '            total -= row.size\n'
            '    return hashlib.sha256(str(total).encode()).hexdigest()\n')
    namespace, = inventory.STANDALONE_HELPER_NAMESPACES
    core_path = '/'.join(namespace.split('.')) + '.py'
    adapter = str(Path(core_path).with_name('ordinary_adapter.py'))
    ordinary = 'tools/ordinary.py'
    paths = [core_path, adapter, ordinary]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    live = inventory.scan(tmp_path)
    assert live['same_name_helpers']['summary'] == [adapter, ordinary]
    assert [f'{core_path}::summary', f'{ordinary}::summary'] in live['near_duplicates']
    assert f'hashlib:{core_path}::summary' in live['primitive_digest_sites']
