"""Single-owner pin parser and safe script invocation, without PB machinery."""
from pathlib import Path
import subprocess
import sys

import pytest
from tools.resolve_tessera_dev_pin import PIN_NAME, PIN_SOURCE, resolve_literal_pin


@pytest.mark.parametrize('body', ['X = 1\n', 'X = "a"\nX = "b"\n', 'X = 42\n', 'X = "ABC"\n'])
def test_invalid_or_duplicate_pin_refuses(tmp_path, body):
    source = tmp_path / 'pin.py'; source.write_text(body)
    with pytest.raises(SystemExit):
        resolve_literal_pin(source, 'X')


@pytest.mark.parametrize('mode', ['-I', '-s'])
def test_pinned_script_does_not_need_its_directory_on_import_path(tmp_path, mode):
    script = Path(__file__).resolve().parents[1] / 'tools/resolve_tessera_dev_pin.py'
    done = subprocess.run([sys.executable, mode, str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == resolve_literal_pin(PIN_SOURCE, PIN_NAME)
