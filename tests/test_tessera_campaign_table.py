"""Campaign metadata decisions work without model or serving runtime imports."""
from pathlib import Path
import subprocess
import sys


def test_cold_table_owner_builds_honest_measured_and_interpolated_rows(tmp_path):
    root = Path(__file__).resolve().parents[1]
    child = '''
import sys
sys.path.insert(0, ROOT)
class ForbiddenRuntime:
    def find_spec(self, name, path=None, target=None):
        if name == 'prismaquant.tessera_campaign' or name == 'tessera.serving' or name.startswith('tessera.serving.') or name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm', 'tools'}:
            raise AssertionError('numerical or tool dependency: ' + name)
sys.meta_path.insert(0, ForbiddenRuntime())
from prismaquant.tessera_campaign_table import CampaignAnchor, campaign_cost_payload, campaign_acquisition_scope
from dataclasses import dataclass
@dataclass(frozen=True)
class Activation:
    activation_contract: str
@dataclass(frozen=True)
class MenuRow:
    format_name: str
    family: str
    body_rate_q256: int
    admission: Activation
family = 'TESSERA_E4M3_K1'
unit = 'layers.0.q_proj'
def name(rate):
    return family + '_R' + str(rate)
def anchor(rate, loss):
    return CampaignAnchor(unit, family, name(rate), rate, loss, 0.0, 100, rate / 256, 'a8', True, 100, 1.0)
def rung(rate):
    return MenuRow(name(rate), family, rate, Activation('a8'))
payload = campaign_cost_payload({unit: {family: [anchor(384, 0.01), anchor(640, 0.001)]}},
    {unit: [rung(rate) for rate in (128, 384, 512, 640, 896)]}, loo={}, provenance={})
rows = payload['costs'][unit]
assert set(rows) == {name(384), name(512), name(640)}
assert rows[name(384)]['output_mse'] == 0.01
assert rows[name(384)]['output_mse_measured'] is True
assert rows[name(512)]['output_mse_measured'] is False
assert 0.001 < rows[name(512)]['output_mse'] < 0.01
assert rows[name(512)]['cost_source'] == 'tessera_campaign_interpolated'
assert all('predicted_dloss' not in row for row in rows.values())
assert payload['currency'] == 'output_mse_under_route_activation_contract'
assert payload['provenance']['cost_mode'] == 'production-render-score'
acquisition = {'requests': {'a': {}, 'b': {}}, 'source_weights': {'a': {}, 'b': {}}}
assert campaign_acquisition_scope(acquisition, {'g': ['a', 'b']}) == (['g'], {'a', 'b'})
try:
    campaign_acquisition_scope(acquisition, {'g': ['a', 'b']}, selected={'a'})
except ValueError as error:
    assert 'atomic-expanded scope' in str(error)
else:
    raise AssertionError('partial atomic acquisition admitted')
assert not any(name == 'prismaquant.tessera_campaign' or name.startswith('tools.') or name == 'torch' or name == 'tessera.serving' or name.startswith('tessera.serving.') for name in sys.modules)
print('TABLE-OWNER-OK')
'''.replace('ROOT', repr(str(root)), 1)
    result = subprocess.run([sys.executable, '-c', child], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'TABLE-OWNER-OK'
