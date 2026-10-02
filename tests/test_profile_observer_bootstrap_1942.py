"""Fresh host interpreter executes the actual observer import chain without Torch."""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_profile_observer_import_without_container_dependencies():
    code = r'''
import importlib.util
from pathlib import Path
import sys
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('compressed_tensors') is None
path = Path('tools/pq_row_profile_observer.py').resolve()
spec = importlib.util.spec_from_file_location('actual_profile_observer', path)
observer = importlib.util.module_from_spec(spec)
sys.argv = [str(path), '--help']
try:
    spec.loader.exec_module(observer)
except SystemExit as result:
    assert result.code == 0, result.code
else:
    raise AssertionError('actual observer did not reach its CLI')
assert 'prismaquant' not in sys.modules
assert 'torch' not in sys.modules
assert 'compressed_tensors' not in sys.modules
sampler_owner = sys.modules[observer.PeriodicSampler.__module__]
assert Path(sampler_owner.__file__).resolve() == Path('prismaquant/io_spans.py').resolve()
assert observer.PeriodicSampler is sampler_owner.PeriodicSampler
print('actual observer imports the stdlib sampler without production initialization')
'''
    result = subprocess.run([sys.executable, '-S', '-c', code], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'actual observer imports the stdlib sampler' in result.stdout
