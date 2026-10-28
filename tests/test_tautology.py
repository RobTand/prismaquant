import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from prismaquant.tautology import Finding, main, scan
from tests import test_no_tautological_tests as guard


@pytest.mark.parametrize(
    ("body", "rule", "line"),
    [
        ("self.assertEqual(actual, actual)", "self_compare", 2),
        ("assert actual == actual", "self_compare", 2),
        ("assert actual is actual", "self_compare", 2),
        ("assert actual >= actual", "self_compare", 2),
        ("assert actual <= actual", "self_compare", 2),
        ("assert True", "constant", 2),
        ("self.assertTrue(1)", "constant", 2),
        ("self.assertFalse([])", "constant", 2),
        ("self.assertIsNone(None)", "constant", 2),
        ("self.assertEqual(1, 1.0)", "constant", 2),
        ("try:\n    assert actual == expected\nexcept AssertionError:\n    pass",
         "swallowed", 4),
        ("try:\n    self.assertEqual(actual, expected)\nexcept Exception:\n    return",
         "swallowed", 4),
        ("assert actual == expected", None, None),
        ("assert read() == read()", None, None),
        ("self.assertEqual(read(), read())", None, None),
        ("assert False", None, None),
        ("self.assertTrue(False)", None, None),
        ("try:\n    assert actual == expected\nexcept Exception:\n    raise",
         None, None),
        ("try:\n    assert actual == expected\nexcept ValueError:\n    pass",
         None, None),
    ],
)
 def test_scan_mechanical_tautologies(tmp_path, body, rule, line):
    source = "def test_sample():\n" + textwrap.indent(body, "    ") + "\n"
    (tmp_path / "test_sample.py").write_text(source, encoding="utf-8")
    expected = [] if rule is None else [Finding("test_sample.py", line, "test_sample", rule)]
    assert scan(tmp_path) == expected


def test_scan_nested_test_trees_without_executing_source(tmp_path):
    archive = tmp_path / "archive" / "old" / "tests"
    archive.mkdir(parents=True)
    (archive / "sample_test.py").write_text(
        "raise RuntimeError('scanner executed source')\n"
        "async def test_sample():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    (tmp_path / "test_strings.py").write_text(
        "def test_fixture():\n    source = 'assert True'\n",
        encoding="utf-8",
    )
    assert scan(tmp_path) == [
        Finding("archive/old/tests/sample_test.py", 3, "test_sample", "constant")
    ]


def test_cli_reports_every_finding_and_clean_exit(tmp_path, capsys):
    target = tmp_path / "test_sample.py"
    target.write_text(
        "def test_sample():\n    self.assertEqual(actual, actual)\n    assert True\n",
        encoding="utf-8",
    )
    assert main([str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert [json.loads(line) for line in output.out.splitlines()] == [
        {"path": "test_sample.py", "line": 2, "test": "test_sample", "rule": "self_compare"},
        {"path": "test_sample.py", "line": 3, "test": "test_sample", "rule": "constant"},
    ]
    assert json.loads(output.err) == {"total": 2}
    target.write_text("def test_sample():\n    assert actual == expected\n", encoding="utf-8")
    assert main([str(tmp_path)]) == 0
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err) == {"total": 0}


def test_vendored_guard_runs_with_isolated_standard_library_python(tmp_path):
    detector = Path(__file__).resolve().parents[1] / "prismaquant" / "tautology.py"
    (tmp_path / "test_sample.py").write_text(
        "def test_sample():\n    self.assertEqual(actual, actual)\n", encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(detector), str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout) == {
        "path": "test_sample.py", "line": 2, "test": "test_sample", "rule": "self_compare"
    }
    assert json.loads(result.stderr) == {"total": 1}


def test_repository_guard_reports_all_planted_findings(tmp_path, monkeypatch):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_planted.py").write_text(
        "def test_planted():\n"
        "    self.assertEqual(actual, actual)\n"
        "    assert True\n"
        "    try:\n"
        "        assert actual == expected\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "test_historical.py").write_text(
        "def test_historical():\n    assert True\n", encoding="utf-8"
    )
    monkeypatch.setattr(guard, "__file__", str(tests / "test_no_tautological_tests.py"))
    with pytest.raises(AssertionError) as failure:
        guard.test_no_tautological_tests()
    assert str(failure.value) == (
        "archive/test_historical.py:2: test_historical: constant\n"
        "tests/test_planted.py:2: test_planted: self_compare\n"
        "tests/test_planted.py:3: test_planted: constant\n"
        "tests/test_planted.py:6: test_planted: swallowed"
    )
