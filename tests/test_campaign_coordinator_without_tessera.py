"""A fleet coordinator plans and checks campaigns without the Tessera producer.

The dispatcher and the data-manifest producer run on the coordinator's
interpreter, which need not carry ``tessera`` at all; only the admitted rows
encode. Each check runs in a fresh interpreter whose import system refuses the
whole ``tessera`` root.
"""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

NO_TESSERA = '''
import sys
class NoTessera:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'tessera':
            raise ModuleNotFoundError('No module named ' + repr(name), name=name)
sys.meta_path.insert(0, NoTessera())
sys.path[:0] = [ROOT, ROOT + '/tests']
'''.replace('ROOT', repr(str(ROOT)))


def _run(script, cwd):
    return subprocess.run([sys.executable, '-c', NO_TESSERA + script], cwd=cwd,
                          capture_output=True, text=True)


@pytest.mark.parametrize('launch', [
    "runpy.run_module('tools.dispatch_tessera_campaign', run_name='__main__', alter_sys=True)",
    "runpy.run_path(ROOT + '/experiments/glm_data_manifests.py', run_name='__main__')",
], ids=['dispatcher', 'experiment-manifests'])
def test_coordinator_help_needs_no_tessera(launch, tmp_path):
    script = ("import runpy\nsys.argv = ['coordinator', '--help']\n"
              + launch.replace('ROOT', repr(str(ROOT))) + "\n")
    result = _run(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith('usage:')


def test_dispatcher_derives_a_row_manifest_without_tessera(tmp_path):
    script = '''
from pathlib import Path
from tools import dispatch_tessera_campaign as dispatch
from test_glm_arc_prewarm_weight_extents import _campaign
producer = dispatch.campaign_data_manifests
producer.SHARED_MOUNT = str(Path.cwd())
campaign, shard = _campaign(Path.cwd())
options = dispatch.explicit_campaign_options(['--menu-mode', 'research', '--units', 'units/row-0000.json'])
assert options == {'menu_mode': 'research', 'units': 'units/row-0000.json'}
provenance = producer.deterministic_provenance(str(Path.cwd()), campaign, 'stat')
manifest = producer.build_campaign_row_manifest(campaign, 'row-0000', provenance)
producer.check_manifest(manifest)
assert any(row['path'] == str(shard) for row in manifest['entries'])
assert not [name for name in sys.modules if name.split('.')[0] == 'tessera']
print('COORDINATOR-WITHOUT-TESSERA-OK')
'''
    result = _run(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'COORDINATOR-WITHOUT-TESSERA-OK'
