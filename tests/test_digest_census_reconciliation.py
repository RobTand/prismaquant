"""Current observations must not manufacture digest migration evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _source(root: Path, text: str) -> None:
    (root / "prismaquant").mkdir(exist_ok=True)
    (root / "tools").mkdir(exist_ok=True)
    (root / "prismaquant" / "consumer.py").write_text(text, encoding="utf-8")


def _refresh(*expressions: str) -> dict:
    return {
        "source_commit": "historical-main",
        "calls": [
            {"id": f"old-{index}", "family": "hashlib",
             "file": "prismaquant/consumer.py", "scope": "identity",
             "expression": expression, "classification": "load-bearing",
             "disposition": "migrate-exact-route", "source_blob": "old-blob"}
            for index, expression in enumerate(expressions)
        ],
        "supplemental_calls": [],
        "module_owners": [{"file": "prismaquant/consumer.py", "epic": "prismaquant#2541"}],
    }


def test_absent_call_is_not_a_completed_migration(tmp_path):
    from tools.reconcile_digest_census import reconcile

    _source(tmp_path, "def identity(value):\n    return value\n")
    report = reconcile(tmp_path, _refresh("hashlib.sha256(value)"))
    assert report["not_observed_historical_ids"] == ["old-0"]
    assert report["calls"] == []
    assert report["acceptance"] == "observations-only"


def test_line_motion_and_spelling_do_not_establish_unchanged_module(tmp_path):
    from tools.reconcile_digest_census import reconcile

    _source(tmp_path, "import hashlib\n\n\ndef identity(value):\n    return hashlib.sha256( value ).hexdigest()\n")
    report = reconcile(tmp_path, _refresh("hashlib.sha256(value)"))
    row, = report["calls"]
    assert row["historical_ids"] == ["old-0"]
    assert row["match"] == "expression-only"
    assert row["review_required"] is True
    assert row["epic"] == "prismaquant#2541"
    assert row["line"] == 5


def test_repeated_expressions_are_ambiguous_even_at_old_line_numbers(tmp_path):
    from tools.reconcile_digest_census import reconcile

    _source(tmp_path, "import hashlib\ndef identity(value):\n    return hashlib.sha256(value)\n")
    row, = reconcile(tmp_path, _refresh("hashlib.sha256(value)", "hashlib.sha256(value)"))["calls"]
    assert row["match"] == "ambiguous"
    assert row["historical_ids"] == ["old-0", "old-1"]
    assert row["review_required"] is True


def test_one_old_call_cannot_account_for_two_current_calls(tmp_path):
    from tools.reconcile_digest_census import reconcile

    _source(tmp_path, "import hashlib\ndef identity(value):\n    hashlib.sha256(value)\n    return hashlib.sha256(value)\n")
    rows = reconcile(tmp_path, _refresh("hashlib.sha256(value)"))["calls"]
    assert len(rows) == 2
    assert all(row["match"] == "ambiguous" for row in rows)


def test_identical_source_is_still_an_observation_not_acceptance(tmp_path):
    from prismaquant.digests import bytes_sha256hex
    from tools.reconcile_digest_census import reconcile

    _source(tmp_path, "import hashlib\ndef identity(value):\n    return hashlib.sha256(value)\n")
    refresh = _refresh("hashlib.sha256(value)")
    refresh["source_blobs"] = {"prismaquant/consumer.py": {
        "git_blob": "old-blob",
        "sha256": bytes_sha256hex((tmp_path / "prismaquant/consumer.py").read_bytes()),
    }}
    row, = reconcile(tmp_path, refresh)["calls"]
    assert row["match"] == "source-identical"
    assert row["review_required"] is True
    assert "classification" not in row


def test_new_expression_cannot_inherit_old_classification(tmp_path):
    from tools.reconcile_digest_census import reconcile

    _source(tmp_path, "import hashlib\ndef identity(value):\n    return hashlib.sha256(value + b'new')\n")
    row, = reconcile(tmp_path, _refresh("hashlib.sha256(value)"))["calls"]
    assert row["historical_ids"] == []
    assert row["match"] == "new-or-changed"
    assert row["review_required"] is True


def test_supplemental_json_does_not_expand_the_ratchet(tmp_path):
    from tools.reconcile_digest_census import reconcile

    _source(tmp_path, "import json as j\nfrom hashlib import sha256 as h\ndef identity(value):\n    h(value)\n    j.dumps(value, sort_keys=True)\n    j.dump(value, None, sort_keys=True)\n    j.dumps(value, sort_keys=False)\n")
    report = reconcile(tmp_path, _refresh())
    assert len(report["primitive_sites"]) == 2
    assert [row["family"] for row in report["calls"]] == ["hashlib", "sorted-json"]
    assert len(report["supplemental_calls"]) == 2
    assert all(row["site"] is None for row in report["supplemental_calls"])


def test_duplicate_historical_ids_are_rejected(tmp_path):
    from tools.reconcile_digest_census import reconcile

    _source(tmp_path, "")
    refresh = _refresh("hashlib.sha256(value)")
    refresh["calls"].append(copy.deepcopy(refresh["calls"][0]))
    with pytest.raises(ValueError, match="duplicate historical id"):
        reconcile(tmp_path, refresh)


def test_duplicate_module_owners_are_rejected(tmp_path):
    from tools.reconcile_digest_census import reconcile

    _source(tmp_path, "")
    refresh = _refresh()
    refresh["module_owners"].append({"file": "prismaquant/consumer.py", "epic": "another"})
    with pytest.raises(ValueError, match="duplicate module owner"):
        reconcile(tmp_path, refresh)


def test_output_cannot_overwrite_the_historical_census(tmp_path):
    from tools.reconcile_digest_census import main

    census = tmp_path / "census.json"
    original = json.dumps({"refresh_2540": _refresh()})
    census.write_text(original)
    with pytest.raises(SystemExit) as error:
        main(["--root", str(tmp_path), "--census", str(census), "--output", str(census)])
    assert error.value.code == 2
    assert census.read_text() == original


def test_check_detects_source_drift_without_changing_inputs(tmp_path):
    from tools.reconcile_digest_census import main

    _source(tmp_path, "import hashlib\ndef identity(value):\n    return hashlib.sha256(value)\n")
    census = tmp_path / "census.json"
    census.write_text(json.dumps({"refresh_2540": _refresh("hashlib.sha256(value)")}))
    report = tmp_path / "report.json"
    args = ["--root", str(tmp_path), "--census", str(census)]
    assert main([*args, "--output", str(report)]) == 0
    original = report.read_bytes()
    assert main([*args, "--check", str(report)]) == 0
    with (tmp_path / "prismaquant" / "consumer.py").open("a") as handle:
        handle.write("# source has changed\n")
    assert main([*args, "--check", str(report)]) == 1
    assert report.read_bytes() == original


def test_cli_runs_from_an_unrelated_directory(tmp_path):
    tool = Path(__file__).resolve().parents[1] / "tools" / "reconcile_digest_census.py"
    result = subprocess.run([sys.executable, str(tool), "--help"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--check" in result.stdout
