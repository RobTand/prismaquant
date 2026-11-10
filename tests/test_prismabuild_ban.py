"""PrismaBuild is retired: no live source may carry one of its mechanisms.

A mechanism is an import or dynamic load, a ``PRISMABUILD_*`` name, a ``pbrun``/``pbtest``/
``pbcampaign``/``pbwait``/``pbgang`` command, a ``prismabuild-fleet`` path, the ``pbsdk`` dependency
group, or a dependency declaration. Prose about history is allowed unless it describes current
behaviour; that half is review, this test is the mechanical half. The frozen wire ids
``prismaquant.prismabuild.<name>`` are allowed by name.

Excluded by explicit lists in ``tests/prismabuild_scan.py``:

- ``HISTORY_PREFIXES``: append-only records of retired runs and decisions;
- ``BAN_DEFINITIONS``: the files that define the ban or prove that a retired name is refused, and so
  must spell the name;
- ``CI_DATA``: measured gate timings keyed by retired test node ids;
"""
from __future__ import annotations

import ast
import re

from tests import prismabuild_scan as scan


# Exact token carve-outs for refusal fixtures and frozen evidence readers.
# The rest of each file remains subject to the ban.
REFUSAL_TOKENS = {
    "prismaquant/campaign_container_spec.py": ('"PRISMABUILD_PRODUCED_"', '"PRISMABUILD_"'),
    "prismaquant/shipcard_slots.py": ('producer_id = matched[0].get("pb_action")',),
    "experiments/build_kda_kernel_qualification.py": ('"pb_action": key',),
    "tests/test_graph_shipcard_receipt.py": ('arm.pop("pb_action")', 'arm["pb_action"] = recorded_id if mismatch else "different-recorded-id"'),
    "tests/graph_receipt_fixtures.py": ('"pb_action": "fixture-equality-action"', 'graph["pb_action"]'),
    "tests/test_pq1317_gpu_qualification_package.py": ('["pb_action"]',),
    "tests/test_bazel_inputs.py": ("'PRISMABUILD_'",),
    "tests/test_container_cache_charge_1091.py": ('"PRISMABUILD_"',),
    "tests/test_dispatch_joint_quanta.py": ('"PRISMABUILD_"', "'PRISMABUILD_'",
        '"PRISMABUILD_PRODUCED_SPOOL_ROOT"', '"PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"',
        "'PRISMABUILD_LOCAL_SCRATCH_PAIRS'", "'PRISMABUILD_LOCAL_SCRATCH_IO'"),
    "tests/test_forward_recovery_chain_tools.py": ("'pbrun'",),
    "tests/test_joint_run_progress.py": ('"PRISMABUILD_ACTION_PROGRESS_PHASES"',
        '"PRISMABUILD_ACTION_PROGRESS_PATH"', '"PRISMABUILD_ACTION_PROGRESS_TOKEN"'),
    "tests/test_native_staged_readers.py": ("'PRISMABUILD_ACTION_KEY'", "'PRISMABUILD_READER_HELPER_ROOT'",
        "'client_sdk'", "'sdk_submodule'"),
    "tests/test_tessera_campaign_container.py": (
        '"PRISMABUILD_ACTION_PROGRESS_PATH"', '"PRISMABUILD_ACTION_PROGRESS_TOKEN"',
        '"PRISMABUILD_ACTION_PROGRESS"', "'PRISMABUILD_ACTION_KEY'", "'PRISMABUILD_ACTION_NONCE'",
        "'PRISMABUILD_ACTION_SCOPE'", "'PRISMABUILD_READER_HELPER_ROOT'", "'PRISMABUILD_QUEUE_ROOT'",
        "'PRISMABUILD_ACTION_PROGRESS_PATH'", "'PRISMABUILD_ACTION_PROGRESS_TOKEN'",
        "'PRISMABUILD_LOCAL_SCRATCH_IO'", "'PRISMABUILD_LOCAL_SCRATCH_PAIRS'",
        '"PRISMABUILD_PRODUCED_SPOOL_ROOT"', '"PRISMABUILD_PRODUCED_SPOOL_MAX_BYTES"',
        '"PRISMABUILD_PRODUCED_SPOOL_PACED_EXPORT"', '"PRISMABUILD_PRODUCED_SPOOL_HOST_WINDOW"'),
    "tests/test_tessera_native_rows.py": ("'PRISMABUILD_ACTION_KEY'",),
    "tests/test_flux_jobs.py": ("'PB_ACTION_KEY'",),
    "tests/g3job/test_g3_submit.py": ('"pbrun"',),
}

_WIRE = re.compile(r"(['\"])prismaquant\.prismabuild\.[a-z_0-9.]+\1")
MECHANISM = re.compile(
    r"PRISMABUILD_|\bpb(?:run|test|campaign|wait|gang)(?:\.py)?\b|prismabuild-fleet"
    r"|(?:^|[^.\w])(?:import|from)[ ]+(?:prismabuild|fleet_sdk)\b|\bfleet_sdk\b|\bpbsdk\b"
    r"|prismabuild_imports|\bsdk_submodule\b|\bclient_sdk\("
    r"|\bpb_action\b|\bPB_ACTION_KEY\b|\bpb(?:mcp|cas)\b|/mnt/shared/prismabuild/"
    r"|(?:from|import)\s+prismaquant(?:\.prismabuild\w*(?:\.\w+)*|\s+import\s+prismabuild\w*)"
    r"|(?:find_spec|import_module|__import__)\(\s*['\"](?:prismaquant\.)?prismabuild")


def mechanisms(text: str) -> list[str]:
    """The mechanism tokens in ``text``; frozen wire ids are removed first."""
    return [m.group(0).strip() for line in text.splitlines()
            for m in MECHANISM.finditer(_WIRE.sub("", line))]








def current_text(path: str, text: str) -> str:
    if path == "docs/ARCHITECTURE.md":
        top = text.split("<!-- flux-current-start -->", 1)[1].split("<!-- flux-current-end -->", 1)[0]
        family = text.split("### 5.7 ", 1)[1].split("## 6. ", 1)[0]
        submit = text.split("#### 7.0.2 ", 1)[1].split("### 7.1 ", 1)[0]
        text = "\n".join((top, family, submit))
    for token in REFUSAL_TOKENS.get(path, ()):
        text = text.replace(token, "")
    return text


def live_findings() -> dict[str, list[str]]:
    found = {}
    for path in scan.tracked():
        if scan.excluded(path):
            continue
        text = scan.read_text(path)
        if text is None:
            continue
        hits = mechanisms(current_text(path, text))
        if hits:
            found[path] = sorted(set(hits))
    return found


def test_no_live_source_carries_a_prismabuild_mechanism():
    assert live_findings() == {}


def test_the_ban_sees_every_mechanism_form_and_spares_wire_ids():
    planted = {
        "PRISMABUILD_ACTION_KEY=1": "PRISMABUILD_",
        "pb_action = key": "pb_action",
        "import prismaquant.prismabuild.client": "import prismaquant.prismabuild.client",
        "from prismaquant import prismabuild_progress": "from prismaquant import prismabuild_progress",
        "importlib.util.find_spec('prismabuild')": "find_spec('prismabuild",
        "__import__('prismabuild')": "__import__('prismabuild",
        "python pbrun.py --x": "pbrun.py",
        "pbtest --json r": "pbtest",
        "run pbcampaign now": "pbcampaign",
        "x = '/mnt/shared/prismabuild-fleet/cas'": "prismabuild-fleet",
        "from prismabuild import pool": "from prismabuild",
        "import prismabuild.core": "import prismabuild",
        "import fleet_sdk": "import fleet_sdk",
        "pip install --group pbsdk": "pbsdk",
    }
    for text, token in planted.items():
        assert token in " ".join(mechanisms(text)), text
    for text in ("a pbwait row", "pbgang x"):
        assert mechanisms(text)
    spared = "schema = 'prismaquant.prismabuild.cas_receipt.v3'  # prose: PrismaBuild was retired"
    assert mechanisms(spared) == []


def test_a_planted_violation_in_a_live_file_is_a_finding(tmp_path, monkeypatch):
    module = scan
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "prismaquant" / "planted.py").write_text("import prismabuild.pool\n")
    (tmp_path / "docs" / "results").mkdir(parents=True)
    (tmp_path / "docs" / "results" / "old.md").write_text("pbrun --old\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "tracked", lambda: ["prismaquant/planted.py", "docs/results/old.md"])
    assert live_findings() == {"prismaquant/planted.py": ["import prismabuild"]}


def test_token_carveouts_do_not_exempt_the_rest_of_a_file():
    for path in REFUSAL_TOKENS:
        assert not scan.excluded(path)
        assert mechanisms(current_text(path, "pb_action = new_live_job")) == ["pb_action"]


def test_only_explicit_recorded_evidence_files_are_exempt():
    assert all(scan.excluded(path) for path in scan.RECORDED_EVIDENCE)
    assert not scan.excluded("prismaquant/kernels/new_qualification.json")
    assert not scan.excluded("docs/design/operator_runbook.md")
    assert not scan.excluded("docs/design/old_run_2026-09-01.md")
    assert not scan.excluded("docs/design/operator_runbook_2026-11-10.md")


def test_token_carveouts_still_name_existing_source():
    for path, tokens in REFUSAL_TOKENS.items():
        text = scan.read_text(path)
        assert text is not None
        for token in tokens:
            assert token in text, (path, token)


def test_recorded_evidence_exemptions_still_have_mechanism_hits():
    for path in scan.RECORDED_EVIDENCE:
        text = scan.read_text(path)
        assert text is not None, path
        assert mechanisms(text), path


def dynamic_loads(source: str) -> list[str]:
    """Imports of ``prismabuild``/``fleet_sdk`` and loads through ``import_module``/``sdk_submodule``."""
    found = []
    def retired(name):
        return (name == "fleet_sdk" or name == "prismabuild" or name.startswith("prismabuild.")
                or name.startswith("prismaquant.prismabuild"))
    tree = ast.parse(source)
    aliases = {alias.asname for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module in {'importlib', 'importlib.util', 'builtins'}
               for alias in node.names if alias.asname and alias.name in {'import_module', 'find_spec', '__import__'}}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or "", *[(node.module or "") + "." + a.name for a in node.names]]
            if isinstance(node, ast.Import):
                found += [n for n in names if retired(n)]
            else:
                found += [names[0]] if retired(names[0]) else [n for n in names[1:] if retired(n)]
        elif isinstance(node, ast.Call) and node.args:
            callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            arg = node.args[0]
            head = (arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    else arg.values[0].value if isinstance(arg, ast.JoinedStr) and arg.values
                    and isinstance(arg.values[0], ast.Constant) else "")
            if callee == "sdk_submodule" or (callee in {"import_module", "find_spec", "__import__", *aliases} and retired(str(head))):
                found.append(callee)
    return found


def import_findings() -> dict[str, list[str]]:
    found = {}
    for path in scan.tracked():
        if not path.endswith(".py") or scan.excluded(path):
            continue
        source = scan.read_text(path)
        hits = dynamic_loads(source) if source is not None else []
        if hits:
            found[path] = sorted(set(hits))
    return found


def test_no_live_module_loads_prismabuild():
    assert import_findings() == {}


def test_the_scanner_sees_every_load_form():
    source = ("import importlib\nimport prismabuild.pool\nfrom prismabuild import core\n"
              "importlib.import_module(f'prismabuild.{name}')\nsdk_submodule('core')\n"
              "importlib.import_module('json')\nimport fleet_sdk\n")
    assert sorted(dynamic_loads(source)) == [
        "fleet_sdk", "import_module", "prismabuild", "prismabuild.pool", "sdk_submodule"]


def test_reexported_and_dynamic_package_loads_are_banned():
    cases = {
        "import prismaquant.prismabuild.client": "prismaquant.prismabuild.client",
        "from prismaquant import prismabuild_progress": "prismaquant.prismabuild_progress",
        "importlib.util.find_spec('prismabuild')": "find_spec",
        "__import__('prismabuild')": "__import__",
        "from importlib.util import find_spec as discover; discover('prismabuild')": "discover",
    }
    for source, expected in cases.items():
        assert expected in dynamic_loads(source)
    assert dynamic_loads("importlib.util.find_spec('json'); __import__('pathlib')") == []
