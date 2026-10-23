"""The package union validates metadata without numerical or CLI dependencies."""
from pathlib import Path
import subprocess
import sys


def test_cold_union_preserves_streamed_identity_and_refuses_missing_inputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = '''
import sys
sys.path.insert(0, ROOT)
class ForbiddenRuntime:
    def find_spec(self, name, path=None, target=None):
        if name == 'prismaquant.tessera_campaign' or name == 'prismaquant.tessera_joint_aura' or name.startswith('tessera.serving') or name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm', 'tools'}:
            raise AssertionError('union dependency: ' + name)
sys.meta_path.insert(0, ForbiddenRuntime())
from pathlib import Path
from prismaquant.tessera_census_union import MenuInterner, UnionRefused, build_plan, identity_sha256, manifest_chunks, read_manifest
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.tessera_campaign_table import STAGE, MANIFEST_NAME
identity = {'units': {'u': {'menu': ['TESSERA_E4M3_K1_R832'], 'weight': {'sha256': 'a' * 64}}}, 'currency': 'output_mse'}
assert identity_sha256(identity) == canonical_json_sha256(identity, where='test identity')
manifest = {'identity': identity, 'identity_sha256': identity_sha256(identity), 'stage': STAGE, 'units': []}
path = Path(MANIFEST_NAME)
path.write_bytes(b''.join(manifest_chunks(manifest)))
loaded = read_manifest(path, MenuInterner())
assert loaded[0] == manifest
try:
    build_plan(['one=missing', 'two=also-missing'], Path('out'))
except UnionRefused as error:
    assert 'cost.pkl is missing' in str(error)
else:
    raise AssertionError('missing table admitted')
assert not Path('out').exists()
print('UNION-OWNER-OK')
'''.replace('ROOT', repr(str(root)), 1)
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'UNION-OWNER-OK'
