"""The observer records real CLI imports and preserves CLI failures."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[1] / "tools" / "capture_pact_replay_imports.py"


def _observe(tmp_path, entry, arguments):
    output = tmp_path / "capture"
    # A fresh process isolates sys.modules from pytest and previous observations.
    code = (
        "import runpy, sys; "
        "observer = runpy.run_path(sys.argv[1]); "
        "observer['capture_cli'](sys.argv[2], sys.argv[4:], "
        "{'fixture': sys.argv[3]}, sys.argv[3] + '/capture', 'fixture-head')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(DRIVER), str(entry), str(tmp_path), *arguments],
        capture_output=True, text=True,
    )
    return result, json.loads((output / "cli-imports.json").read_bytes())


def test_observer_executes_main_and_records_dynamic_import(tmp_path):
    dependency = tmp_path / "science_dependency.py"
    dependency.write_text("value = 41\n")
    unused = tmp_path / "latent_dependency.py"
    unused.write_text("raise RuntimeError('Do not import latent source')\n")
    entry = tmp_path / "entry.py"
    entry.write_text(
        "import sys\n"
        "if __name__ == '__main__':\n"
        "    import importlib\n"
        "    science = importlib.import_module(sys.argv[1])\n"
        "    print(science.value + 1)\n"
    )
    result, capture = _observe(tmp_path, entry, ["science_dependency"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "42"
    observed = capture["modules"]["science_dependency"]
    assert observed["origin"] == str(dependency)
    assert observed["sha256"] == hashlib.sha256(dependency.read_bytes()).hexdigest()
    assert observed["loaded_before_cli"] is False
    assert "latent_dependency" not in capture["modules"]
    assert capture["argv"][1:] == [str(entry), "science_dependency"]


@pytest.mark.parametrize("failure", ["raise ValueError('science refusal')", "raise SystemExit(7)"])
def test_observer_keeps_cli_failure(tmp_path, failure):
    entry = tmp_path / "entry.py"
    entry.write_text("if __name__ == '__main__':\n    " + failure + "\n")
    result, capture = _observe(tmp_path, entry, [])
    assert result.returncode == (7 if "SystemExit" in failure else 1)
    assert capture["returncode"] == result.returncode
    if "ValueError" in failure:
        assert capture["error"] == {"type": "ValueError", "message": "science refusal"}
        assert "science refusal" in result.stderr
