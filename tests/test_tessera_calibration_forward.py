"""Public calibration results expose full statistics and bounded scoring rows."""
from pathlib import Path
import subprocess
import sys


def test_importing_capture_contract_does_not_load_models_or_commands(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = '''
import sys
sys.path.insert(0, ROOT)
class ForbiddenRuntime:
    def find_spec(self, name, path=None, target=None):
        if name == 'prismaquant.tessera_campaign' or name.startswith('tessera.serving') or name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm', 'tools', 'experiments'}:
            raise AssertionError('capture contract dependency: ' + name)
sys.meta_path.insert(0, ForbiddenRuntime())
from prismaquant.tessera_calibration_forward import CalibrationStatistics, capture_activation_statistics
assert 'torch' not in sys.modules and 'prismaquant.tessera_campaign' not in sys.modules
print('CAPTURE-CONTRACT-OK')
'''.replace('ROOT', repr(str(root)), 1)
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'CAPTURE-CONTRACT-OK'


def test_named_statistics_keep_full_rows_for_hessian_counts_and_maximum():
    import torch
    from prismaquant.tessera_calibration_forward import capture_activation_statistics

    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    inputs = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    result = capture_activation_statistics(model, ['0'], [inputs], 1, 'cpu', want_hessian=True)
    assert torch.equal(result.rows['0'], torch.tensor([[1., 2.]]))
    assert torch.equal(result.hessians['0'], torch.tensor([[35., 44.], [44., 56.]]))
    assert result.token_counts == {'0': 3}
    assert result.max_abs == {'0': 6.}
    assert not model[0]._forward_pre_hooks
