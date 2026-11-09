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
    "docs/research/", "docs/archive/", "docs/experiments/", "docs/design/", "docs/ARCHITECTURE.md",
    "CHANGELOG.md", "experiments/measurements/", "tests/fixtures/", "tests/fixture-data/",
    "reports/", "archive/", "fullstack-real-pq-report.md", "tools/fleet_interpreters.json",
    "prismaquant/serving_runtime_patches/", "tests/g3job/data-manifest.json",
)
HISTORY_EXCEPT = ("docs/design/runtime_flags.md",)
BAN_DEFINITIONS = (
    "tests/test_prismabuild_ban.py", "tests/test_prismabuild_boundary.py",
    "prismaquant/campaign_container_spec.py", "tests/test_bazel_inputs.py",
    "tests/test_container_cache_charge_1091.py", "tests/test_dispatch_joint_quanta.py",
    "tests/test_forward_recovery_chain_tools.py", "tests/test_joint_run_progress.py",
    "tests/test_native_staged_readers.py", "tests/test_tessera_campaign_container.py",
    "tests/test_tessera_native_rows.py", "tests/g3job/test_g3_submit.py",
)
CI_DATA = ("tools/ci/gate_durations.json",)  # measured timings keyed by node ids of tests that no longer exist

_WIRE = re.compile(r"prismaquant\.prismabuild\.[a-z_0-9.]+")
MECHANISM = re.compile(
    r"PRISMABUILD_|\bpb(?:run|test|campaign|wait|gang)(?:\.py)?\b|prismabuild-fleet"
    r"|(?:^|[^.\w])(?:import|from)[ ]+(?:prismabuild|fleet_sdk)\b|\bfleet_sdk\b|\bpbsdk\b"
    r"|prismabuild_imports|\bsdk_submodule\b|\bclient_sdk\(")


def mechanisms(text: str) -> list[str]:
    """The mechanism tokens in ``text``; frozen wire ids are removed first."""
    return [m.group(0).strip() for line in text.splitlines()
            for m in MECHANISM.finditer(_WIRE.sub("", line))]


def excluded(path: str) -> bool:
    if path in HISTORY_EXCEPT:
        return False
    return (path.startswith(HISTORY_PREFIXES) or path in BAN_DEFINITIONS or path in CI_DATA
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


def live_findings() -> dict[str, list[str]]:
    found = {}
    for path in tracked():
        if excluded(path):
            continue
        text = read_text(path)
        if text is None:
            continue
        hits = mechanisms(text)
        if hits:
            found[path] = sorted(set(hits))
    return found


def test_no_live_source_carries_a_prismabuild_mechanism():
    assert live_findings() == {}


def test_the_ban_sees_every_mechanism_form_and_spares_wire_ids():
    planted = {
        "PRISMABUILD_ACTION_KEY=1": "PRISMABUILD_",
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
