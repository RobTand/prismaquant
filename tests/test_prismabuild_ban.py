"""PrismaBuild is retired: no live source may carry one of its mechanisms.

A mechanism is an import or dynamic load, a ``PRISMABUILD_*`` name, a ``pbrun``/``pbtest``/
``pbcampaign``/``pbwait``/``pbgang`` command, a ``prismabuild-fleet`` path, the ``pbsdk`` dependency
group, or a dependency declaration. Prose about history is allowed unless it describes current
behaviour; that half is review, this test is the mechanical half. The frozen wire ids
``prismaquant.prismabuild.<name>`` are allowed by name.

Excluded, each by an explicit list below (never by pattern guessing):

- ``HISTORY_PREFIXES``: append-only records of retired runs and decisions;
- ``BAN_DEFINITIONS``: the files that define the ban or prove that a retired name is refused, and so
  must spell the name;
- ``CI_DATA``: measured gate timings keyed by retired test node ids;
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HISTORY_PREFIXES = (
    "docs/audits/", "docs/results/", "docs/measurements/", "docs/reports/", "docs/reviews/",
    "docs/research/", "docs/archive/", "docs/experiments/",
    "CHANGELOG.md", "experiments/measurements/", "tests/fixtures/", "tests/fixture-data/",
    "reports/", "archive/", "fullstack-real-pq-report.md", "tools/fleet_interpreters.json",
    "prismaquant/serving_runtime_patches/", "tests/g3job/data-manifest.json",
)
BAN_DEFINITIONS = ("tests/test_prismabuild_ban.py", "tests/test_prismabuild_boundary.py")
# Recorded kernel evidence is immutable; these are not execution instructions.
RECORDED_EVIDENCE = (
    "prismaquant/kernels/kda_chunk_qualification.json",
    "prismaquant/kernels/joint_projection_reduce_qualification.json",
)
# Exact token carve-outs for refusal fixtures and frozen evidence readers.
# The rest of each file remains subject to the ban.
REFUSAL_TOKENS = {
    "prismaquant/campaign_container_spec.py": ('"PRISMABUILD_PRODUCED_"', '"PRISMABUILD_"'),
    "prismaquant/shipcard_slots.py": ('matched[0].get("pb_action")',),
    "experiments/build_kda_kernel_qualification.py": ('"pb_action": key',),
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
CI_DATA = ("tools/ci/gate_durations.json",)  # measured timings keyed by node ids of tests that no longer exist

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


def excluded(path: str) -> bool:
    if path.startswith("docs/design/") and re.search(r"20\d{2}(?:-\d{2}-\d{2}|\d{4})", Path(path).name):
        return True
    return (path.startswith(HISTORY_PREFIXES) or path in BAN_DEFINITIONS or path in CI_DATA or path in RECORDED_EVIDENCE
            or path.endswith("BUILD.bazel"))


def tracked() -> list[str]:
    if (ROOT / ".git").exists():
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], check=True,
                             capture_output=True).stdout.decode()
        return [p for p in out.split("\0") if p]
    return [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts]


def read_text(path: str) -> str | None:
    try:
        return (ROOT / path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


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
    for path in tracked():
        if excluded(path):
            continue
        text = read_text(path)
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
    import sys
    module = sys.modules[__name__]
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "prismaquant" / "planted.py").write_text("import prismabuild.pool\n")
    (tmp_path / "docs" / "results").mkdir(parents=True)
    (tmp_path / "docs" / "results" / "old.md").write_text("pbrun --old\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "tracked", lambda: ["prismaquant/planted.py", "docs/results/old.md"])
    assert module.live_findings() == {"prismaquant/planted.py": ["import prismabuild"]}


def test_token_carveouts_do_not_exempt_the_rest_of_a_file():
    for path in REFUSAL_TOKENS:
        assert not excluded(path)
        assert mechanisms(current_text(path, "pb_action = new_live_job")) == ["pb_action"]


def test_only_explicit_recorded_evidence_files_are_exempt():
    assert all(excluded(path) for path in RECORDED_EVIDENCE)
    assert not excluded("prismaquant/kernels/new_qualification.json")
    assert not excluded("docs/design/operator_runbook.md")
    assert excluded("docs/design/old_run_2026-09-01.md")
