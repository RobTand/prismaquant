"""Whole-tree candidate reporting must leave the shrink-only gate unchanged."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import duplication_inventory as inventory


BODY = """    total = 0
    for item in items:
        if item.ready:
            total += item.size
        else:
            total -= helper(item)
    return {'total': total, 'count': len(items)}
"""


@pytest.fixture
def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    files = {
        "prismaquant/a.py": "def first(items):\n" + BODY,
        "tools/b.py": "def first(items):\n" + BODY,
        "tests/helpers.py": "def left(items):\n" + BODY + "\ndef right(items):\n" + BODY,
        "experiments/check.py": "def first(items):\n" + BODY,
        "launch.py": "def launch(items):\n" + BODY,
        "archive/old.py": "this is preserved, not valid Python",
        "prismaquant/serving_runtime_patches/image.py": "sealed bytes",
        "docs/policy.md": "# Policy\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "fixture"], check=True)
    (tmp_path / "untracked.py").write_text("untracked bytes", encoding="utf-8")
    return tmp_path, files


def test_report_accounts_for_tracked_paths_and_finds_omitted_candidates(repository):
    root, files = repository
    before = inventory.scan(root)
    report = inventory.coverage_report(root)
    assert report["schema"] == "prismaquant.duplication_coverage.v1"
    assert report["source_commit"] == subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert report["tracked_dirty"] is False
    rows = {row["path"]: row for row in report["files"]}
    assert sorted(rows) == sorted(files)
    assert rows["docs/policy.md"]["analysis"] == "not_python"
    assert rows["archive/old.py"]["analysis"] == "excluded"
    assert rows["archive/old.py"]["excluded_parts"] == ["archive"]
    assert rows["prismaquant/serving_runtime_patches/image.py"]["analysis"] == "excluded"
    assert rows["tests/helpers.py"] == {
        "path": "tests/helpers.py", "analysis": "python", "in_default_gate": False,
    }
    assert rows["prismaquant/a.py"]["in_default_gate"] is True
    pairs = report["candidates"]["near_duplicates"]
    assert ["tests/helpers.py:1::left", "tests/helpers.py:10::right"] in pairs
    assert ["experiments/check.py:1::first", "launch.py:1::launch"] in pairs
    assert report["functions"]["tests/helpers.py:1::left"] == {"line": 1, "end_line": 8}
    assert report["candidates"]["same_name_helpers"]["first"] == [
        "experiments/check.py", "prismaquant/a.py", "tools/b.py",
    ]
    assert before == inventory.scan(root) == {
        "near_duplicates": [["prismaquant/a.py::first", "tools/b.py::first"]],
        "same_name_helpers": {"first": ["prismaquant/a.py", "tools/b.py"]},
        "primitive_digest_sites": [],
    }
    assert report == inventory.coverage_report(root)


def test_report_does_not_hide_dirty_or_unreadable_tracked_python(repository):
    root, _ = repository
    path = root / "launch.py"
    path.write_text("def broken(\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        inventory.coverage_report(root)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        inventory.coverage_report(root)
    path.write_text("# changed\n", encoding="utf-8")
    assert inventory.coverage_report(root)["tracked_dirty"] is True


def test_report_requires_git_metadata(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        inventory.coverage_report(tmp_path)


def test_report_cli_writes_audit_without_rewriting_baseline(repository, monkeypatch, capsys):
    root, _ = repository
    output = root / "coverage.json"
    baseline = root / "baseline.json"
    baseline.write_bytes(b"retained baseline\n")
    monkeypatch.setattr(inventory, "ROOT", root)
    monkeypatch.setattr(inventory, "BASELINE", baseline)
    monkeypatch.setattr(sys, "argv", ["duplication_inventory.py", "--coverage-report", str(output)])
    assert inventory.main() == 0
    assert json.loads(output.read_text())["tracked_dirty"] is False
    assert baseline.read_bytes() == b"retained baseline\n"
    assert "tracked paths" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["duplication_inventory.py", "--coverage-report", str(output),
                                     "--write-baseline"])
    with pytest.raises(SystemExit) as error:
        inventory.main()
    assert error.value.code == 2
    assert baseline.read_bytes() == b"retained baseline\n"


def test_report_keeps_repeated_qualified_names_distinct(repository):
    root, _ = repository
    path = root / "tests/helpers.py"
    path.write_text("def repeated(items):\n" + BODY + "\ndef repeated(items):\n" + BODY,
                    encoding="utf-8")
    report = inventory.coverage_report(root)
    assert ["tests/helpers.py:1::repeated", "tests/helpers.py:10::repeated"] in report["candidates"]["near_duplicates"]
    assert report["functions"]["tests/helpers.py:1::repeated"] == {"line": 1, "end_line": 8}
    assert report["functions"]["tests/helpers.py:10::repeated"] == {"line": 10, "end_line": 17}


def test_report_refuses_tracked_python_symlink(repository):
    root, _ = repository
    path = root / "launch.py"
    path.unlink()
    path.symlink_to(root / "prismaquant/a.py")
    with pytest.raises(ValueError, match="tracked Python symlink"):
        inventory.coverage_report(root)
