"""CPU regression for profiling under PB's inherited NoNewPrivileges contract."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import tools.pq_admitted_profile as admitted

ROOT = Path(__file__).resolve().parents[1]
OBSERVER = ROOT / 'tools/pq_row_profile_observer.py'


def test_observer_preflight_works_with_no_new_privileges(tmp_path):
    # Execute the actual command helper and actual preflight statements, not
    # the observer CLI's hardware/Netdata probes. The isolated child applies
    # exactly PB resource_payload.py's prctl(38, 1) before any sudo invocation.
    code = r'''
import ast, ctypes, json, pathlib, subprocess, sys
from types import SimpleNamespace
assert ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) == 0
assert 'NoNewPrivs:\t1' in pathlib.Path('/proc/self/status').read_text()
source = pathlib.Path(sys.argv[1])
tree = ast.parse(source.read_text())
command = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'command')
statements = [n for n in tree.body if
    (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
     and isinstance(n.value.func, ast.Name) and n.value.func.id == 'command')
    or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
        t.id in ('version', 'first_power') for t in n.targets))]
assert statements, 'fixture must execute the real observer preflight'
space = {'subprocess': subprocess}
exec(compile(ast.Module(body=[command], type_ignores=[]), str(source), 'exec'), space)
actual = space['command']
def run(argv, **kwargs):
    if argv[0] == 'nvidia-smi': return '20'
    return actual(argv, **kwargs)
space.update(command=run, profile_dir=pathlib.Path(sys.argv[2]),
             a=SimpleNamespace(profiler_executable=sys.executable))
try:
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), 'exec'), space)
except subprocess.CalledProcessError as error:
    print('ACTUAL ADMITTED PREFLIGHT STDERR:', error.stderr, file=sys.stderr)
    raise
print('NoNewPrivs=1; actual unprivileged preflight succeeded')
'''
    result = subprocess.run([sys.executable, '-c', code, str(OBSERVER), str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'NoNewPrivs=1' in result.stdout


def test_observer_never_attempts_sudo_under_admission():
    tree = ast.parse(OBSERVER.read_text())
    sudo = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and n.value == 'sudo']
    assert not sudo, f'admitted observer still requires privilege escalation at {sudo}'


def test_container_row_is_profiled_as_same_uid_child(tmp_path, monkeypatch):
    original = ['python3', '-m', 'tools.tessera_campaign_container', '--spec', '{}',
                '--', 'python3', '-u', '-m', 'prismaquant.tessera_campaign',
                '--out', '/mnt/shared/owned/cost.pkl', '--units', '/mnt/shared/owned/units.json']
    launch = tmp_path / 'launch.json'
    launch.write_text(json.dumps({'argv': original}))
    calls = []
    monkeypatch.setattr(admitted, 'run_observed',
                        lambda command, observer, observations, **kwargs: calls.append((command, observer)) or 0)
    assert admitted.main(['--launch', str(launch), '--observations', str(tmp_path / 'obs'),
                          '--profile-local', str(tmp_path / 'local')]) == 0
    command, observer = calls[0]
    boundary = original.index('--')
    assert command[:boundary + 1] == original[:boundary + 1]
    assert 'tools.pq_profile_child' in command[boundary + 1:], 'profiler must parent the real in-container Python, not attach through sudo'
    inner_boundary = command.index('--', boundary + 1)
    assert command[inner_boundary + 1:] == original[boundary + 1:]
    assert '--child-profile' in observer
