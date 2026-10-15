"""Metadata imports and actual model construction have separate boundaries."""
from pathlib import Path
import os
import subprocess
import sys


def test_metadata_import_does_not_load_numerical_runtime(tmp_path):
    root = Path(__file__).resolve().parents[1]
    child = '''
import sys, os
from pathlib import Path
class NoNumericalRuntime:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm'}:
            raise AssertionError('metadata requested numerical runtime: ' + name)
sys.meta_path.insert(0, NoNumericalRuntime())
import prismaquant
from prismaquant import validate_pretrained_initialization_contract, validate_source_initialization_contract
from prismaquant.tessera_runtime_contract import installed_runtime_snapshot
from prismaquant.tessera_serving_runtime_pin import load_tessera_serving_runtime_pin
snapshot = installed_runtime_snapshot()
assert snapshot.published_formats
assert load_tessera_serving_runtime_pin().contract_sha256 == snapshot.sha256
record = {'schema': 'prismaquant.pretrained_initialization.v1', 'scope': 'checkpoint_missing_state',
          'status': 'completed', 'transformers_version': 'owned'}
assert validate_source_initialization_contract(record) == record
assert not any(name.split('.')[0] in {'torch', 'transformers', 'triton', 'vllm'} for name in sys.modules)
assert not Path(os.environ['TRITON_CACHE_DIR']).exists()
print('METADATA-OK')
'''
    environment = dict(os.environ, PYTHONPATH=str(root), TRITON_CACHE_DIR=str(tmp_path / 'unused-triton-cache'))
    result = subprocess.run([sys.executable, '-c', child], cwd=tmp_path,
                            env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'METADATA-OK'


def test_streaming_skeleton_initializes_checkpoint_policy_before_construction():
    import torch
    from transformers import Qwen3Config
    from prismaquant.streaming_model import build_streaming_skeleton

    config = Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=4)
    skeleton = build_streaming_skeleton(config, multimodal=False)
    assert all(parameter.is_meta for parameter in skeleton.parameters())
    assert skeleton.config.vocab_size == 16
    assert next(skeleton.parameters()).dtype == torch.float32


def test_every_hf_loader_boundary_explicitly_initializes_the_model_runtime():
    """Keep actual dependency entry points explicit as new loaders are added."""
    import ast

    root = Path(__file__).resolve().parents[1]
    missing = []
    for directory in ('prismaquant', 'tools', 'experiments'):
        for path in (root / directory).rglob('*.py'):
            if 'vendored' in path.parts:
                continue
            tree = ast.parse(path.read_text())
            functions = [node for node in ast.walk(tree)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                name = ast.unparse(call.func)
                if not name.endswith(('.from_pretrained', '.from_config', '._from_config')):
                    continue
                if any(word in name for word in ('Tokenizer', 'Processor', 'Embedding')):
                    continue
                owners = [fn for fn in functions if fn.lineno <= call.lineno <= fn.end_lineno]
                owner = min(owners, key=lambda fn: fn.end_lineno - fn.lineno) if owners else tree
                initialized = any(
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'initialize_model_runtime' and node.lineno < call.lineno
                    for node in ast.walk(owner))
                if not initialized:
                    missing.append(f'{path.relative_to(root)}:{call.lineno}:{name}')
    assert not missing, missing
