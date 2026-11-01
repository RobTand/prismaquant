"""Wire custody stays independent of model loading and command implementations."""
from pathlib import Path
import subprocess
import sys


def test_wire_custody_is_cold_and_reserves_before_seed_reads(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = '''
import sys
sys.path.insert(0, ROOT)
class MissingRuntime:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'tessera':
            raise ImportError('producer is absent in this dependency witness')
        if name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm', 'tools', 'experiments'}:
            raise AssertionError('wire custody imported runtime: ' + name)
sys.meta_path.insert(0, MissingRuntime())
from pathlib import Path
from types import SimpleNamespace
from prismaquant import tessera_campaign_wire as wire
root = Path.cwd()
cache = SimpleNamespace(weights={}, cache_dir=None)
wire.register_wire_coordinates(cache, root, [('a.b', 'FMT')])
try:
    wire.register_wire_coordinates(cache, root, [('a__b', 'FMT')])
except ValueError as exc:
    assert 'one wire destination' in str(exc)
else:
    raise AssertionError('colliding wire reservation was admitted')
assert not list(root.iterdir())
render_cache = SimpleNamespace(weights={}, cache_dir=str(root))
try:
    wire.require_injective_anchor_filenames(render_cache, root,
                                            [('a.b', 'FMT'), ('a_b', 'FMT')])
except ValueError as exc:
    assert 'one rendered cache destination' in str(exc)
else:
    raise AssertionError('colliding rendered destination was admitted')
try:
    wire.require_checkpoint_identity_api()
except RuntimeError as exc:
    assert "producer's cached_unit input/byte receipt API" in str(exc)
else:
    raise AssertionError('missing producer identity capability was admitted')
seed, target = root / 'seed', root / 'target'
seed.mkdir(); target.mkdir()
filename = wire.wire_path(seed, 'layer.a', 'FMT').name
(seed / filename).write_bytes(b'actual seeded wire')
wire.link_seed_wire(seed, target, filename, qname='layer.a', format_name='FMT')
assert (target / filename).read_bytes() == b'actual seeded wire'
try:
    wire.link_seed_wire(root / 'absent', target, '../other', qname='layer.a', format_name='FMT')
except RuntimeError as exc:
    assert 'unusable file' in str(exc)
else:
    raise AssertionError('unbound seed filename was admitted')
print('WIRE-CUSTODY-OK')
'''.replace('ROOT', repr(str(root)))
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'WIRE-CUSTODY-OK'
