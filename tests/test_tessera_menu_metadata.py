"""CLI menu metadata needs no numerical allocator or tensor runtime."""
from pathlib import Path
import subprocess
import sys


def test_menu_modes_and_refusals_work_in_a_fresh_numerical_free_process():
    root = Path(__file__).resolve().parents[1]
    script = '''
import os
import sys
sys.path.insert(0, ROOT)
class NoNumericalRuntime:
    def find_spec(self, name, path=None, target=None):
        if name.startswith('tessera.serving') or name in {'prismaquant.allocator_solver', 'prismaquant.format_registry'} or name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm'}:
            raise AssertionError('menu metadata imported numerical runtime: ' + name)
sys.meta_path.insert(0, NoNumericalRuntime())
from prismaquant import tessera_menu as menu
os.environ.pop(menu.MENU_MODE_ENV, None)
assert menu.menu_mode() == 'attested'
os.environ[menu.MENU_MODE_ENV] = ' readable '
assert menu.menu_mode() == 'readable'
assert menu.menu_mode(' RESEARCH ') == 'research'
try:
    menu.menu_mode('unsupported')
except menu.TesseraMenuError as exc:
    assert 'not a Tessera menu mode' in str(exc)
else:
    raise AssertionError('unknown pricing menu mode was admitted')
assert menu.MENU_MODES == {'attested', 'readable', 'research'}
print('COLD-MENU-MODES-OK')
'''.replace('ROOT', repr(str(root)))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'COLD-MENU-MODES-OK'
