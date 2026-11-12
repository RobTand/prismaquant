"""Pin TL3's pytest command and the boundary between test edits and production."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def harness() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["tl3"]["harness"]


def full_match(name: str, pattern: str) -> bool:
    """Use PurePath.full_match semantics, including on Python 3.12 workers."""
    if hasattr(PurePosixPath, "full_match"):
        return PurePosixPath(name).full_match(pattern)
    assert not set("?[]") & set(pattern), f"unsupported 3.12 pattern: {pattern}"
    segments = pattern.split("/")
    regex = ""
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "**":
            regex += ".*" if last else "(?:[^/]+/)*"
        else:
            regex += re.escape(segment).replace(r"\*", "[^/]*") + ("" if last else "/")
    return re.fullmatch(regex, name) is not None


def python_files() -> list[str]:
    return sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*.py"))


def test_command_is_the_pytest_module():
    assert harness()["command"] == ["python", "-m", "pytest"]


def test_patterns_have_the_shape_tl3_accepts():
    table = harness()
    assert set(table) == {"command", "test-files"}
    patterns = table["test-files"]
    assert patterns == ["tests/**/*.py"]
    for pattern in patterns:
        path = PurePosixPath(pattern)
        assert not path.is_absolute() and ".." not in path.parts and "\\" not in pattern
        assert path.suffix == ".py" and path.name != "conftest.py"


def test_every_test_and_helper_is_a_test_edit():
    patterns = harness()["test-files"]
    tests = [name for name in python_files() if name.startswith("tests/")]
    assert "tests/test_tl3_harness_table.py" in tests
    assert "tests/conftest.py" in tests
    assert [name for name in tests if not any(full_match(name, p) for p in patterns)] == []


def test_no_production_file_is_a_test_edit():
    patterns = harness()["test-files"]
    production = [name for name in python_files() if not name.startswith("tests/")]
    assert production
    assert [name for name in production if any(full_match(name, p) for p in patterns)] == []
