"""``tools/resolve_prismabuild_dev_pin.py`` prints the PB commit PQ pins.

pbtest runs each ``tools/resolve_<module>_dev_pin.py`` in a shard before
pytest and refuses an interpreter whose installed module is another commit.
The resolver must print the same literal ``staged_lease`` checks at runtime,
read without importing PrismaQuant (PQ #1929).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_resolver_prints_the_staged_lease_pin_without_importing_prismaquant():
    from prismaquant.staged_lease import PB_READER_LEASE_PIN_COMMIT

    completed = subprocess.run(
        [sys.executable, "-S", "-c",
         "import runpy, sys; sys.modules['prismaquant'] = None; "
         "sys.path.insert(0, 'tools'); "
         "runpy.run_path('tools/resolve_prismabuild_dev_pin.py', run_name='__main__')"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == PB_READER_LEASE_PIN_COMMIT


def test_the_resolver_refuses_a_source_with_two_pins(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from resolve_tessera_dev_pin import resolve_literal_pin
    finally:
        sys.path.remove(str(ROOT / "tools"))
    source = tmp_path / "staged_lease.py"
    pin = "PB_READER_LEASE_PIN_COMMIT"
    source.write_text(f'{pin} = "{"a" * 40}"\n{pin} = "{"b" * 40}"\n')
    with pytest.raises(SystemExit, match="expected exactly one literal"):
        resolve_literal_pin(source, pin)


@pytest.mark.parametrize("mode", ["direct-safe", "direct-isolated", "module-safe"])
def test_resolver_in_safe_path_mode_without_script_directory(tmp_path, mode):
    from prismaquant.staged_lease import PB_READER_LEASE_PIN_COMMIT
    env = dict(os.environ, PYTHONSAFEPATH="1", PYTHONPATH=str(ROOT) if mode == "module-safe" else "")
    if mode == "module-safe":
        command = [sys.executable, "-P", "-m", "tools.resolve_prismabuild_dev_pin"]
    else:
        command = [sys.executable, "-I" if mode == "direct-isolated" else "-P",
                   str(ROOT / "tools/resolve_prismabuild_dev_pin.py")]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == PB_READER_LEASE_PIN_COMMIT
