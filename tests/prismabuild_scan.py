"""Shared live-source scope and byte readers for the two retirement scans."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

HISTORY_PREFIXES = (
    "docs/audits/", "docs/results/", "docs/measurements/", "docs/reports/", "docs/reviews/",
    "docs/research/", "docs/archive/", "docs/experiments/",
    "CHANGELOG.md", "experiments/measurements/", "tests/fixtures/", "tests/fixture-data/",
    "reports/", "archive/", "fullstack-real-pq-report.md", "tools/fleet_interpreters.json",
    "prismaquant/serving_runtime_patches/", "tests/g3job/data-manifest.json",
)

BAN_DEFINITIONS = ("tests/test_prismabuild_ban.py",)

RECORDED_EVIDENCE = (
    "docs/design/distributed_campaign_2026-09-19.md",
    "docs/design/domain_numerics_census_1303_20261002.md",
    "docs/design/joint_quanta_765_688_2026-09-19.md",

    "prismaquant/kernels/kda_chunk_qualification.json",
    "prismaquant/kernels/joint_projection_reduce_qualification.json",
)

CI_DATA = ("tools/ci/gate_durations.json",)  # measured timings keyed by node ids of tests that no longer exist


def excluded(path: str) -> bool:
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
