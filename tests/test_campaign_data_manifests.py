"""Actual campaign readsets need neither models nor experimental commands."""
from pathlib import Path
import subprocess
import sys


def test_cold_owner_reads_real_tensor_extents_and_emits_a_valid_manifest(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = '''
import sys
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + '/tests')
class ForbiddenRuntime:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'tessera', 'torch', 'transformers', 'triton', 'vllm', 'tools', 'experiments'}:
            raise AssertionError('readset dependency: ' + name)
sys.meta_path.insert(0, ForbiddenRuntime())
from pathlib import Path
from test_glm_arc_prewarm_weight_extents import _campaign
from prismaquant import campaign_data_manifests as owner
owner.SHARED_MOUNT = str(Path.cwd())
campaign, shard = _campaign(Path.cwd())
manifest = owner.build_campaign_row_manifest(campaign, 'row-0000', {'tool': 'owned-fixture'})
owner.check_manifest(manifest)
assert manifest['entry_count'] == len(manifest['entries'])
assert manifest['total_bytes'] == sum(row['bytes'] for row in manifest['entries'])
extent, = [row for row in manifest['entries'] if row['path'] == str(shard)]
assert extent['offset'] == 0 and extent['bytes'] == shard.stat().st_size
assert [row['name'] for row in manifest['annotations']['phases']] == ['captures', 'weight_extents', 'seeds']
print('READSET-OWNER-OK')
'''.replace('ROOT', repr(str(root)))
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'READSET-OWNER-OK'
