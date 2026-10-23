"""Metadata can resolve existing cache leaves without the numerical runtime."""
from pathlib import Path
import subprocess
import sys


def test_existing_cache_paths_and_collision_refusal_work_in_a_cold_process(tmp_path):
    child = '''
import sys
from pathlib import Path

class MetadataOnly:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm', 'tools', 'experiments'} or name == 'prismaquant.production_weight_cache':
            raise AssertionError('cache metadata requested numerical owner: ' + name)

sys.meta_path.insert(0, MetadataOnly())
sys.path.insert(0, sys.argv[1])
from prismaquant.render_cache_paths import (
    cache_weight_filename, cache_weight_leaf, require_injective_cache_filenames,
)

root = Path(sys.argv[2])
legacy = root / 'model__layers_0_q_proj__INT4_W4A16_g128.pt'
legacy.write_bytes(b'existing render archive')
assert cache_weight_leaf('model/layers.0.q_proj') == 'model__layers_0_q_proj'
assert (root / cache_weight_filename('model/layers.0.q_proj', 'INT4_W4A16_g128')).read_bytes() == b'existing render archive'
assert cache_weight_filename(42, 'verbatim_fmt') == '42__verbatim_fmt.pt'
require_injective_cache_filenames([('a.b', 'BF16'), ('a_b', 'NVFP4')], where='metadata open')
try:
    require_injective_cache_filenames([('a.b', 'NVFP4'), ('a_b', 'NVFP4')], where='metadata open')
except ValueError as error:
    assert str(error).startswith('metadata open: coordinates ')
    assert "'a_b__NVFP4.pt'" in str(error)
else:
    raise AssertionError('colliding metadata roster was admitted')
assert not any(name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm'} for name in sys.modules)
print('COLD-CACHE-PATHS-OK')
'''
    checkout = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, '-I', '-S', '-c', child, str(checkout), str(tmp_path)],
        text=True, capture_output=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'COLD-CACHE-PATHS-OK'
