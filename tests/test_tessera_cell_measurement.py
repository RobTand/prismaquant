"""Cell measurements admit complete recipes before publishing priced bytes."""
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from prismaquant import tessera_cell_measurement as measurement
from prismaquant import tessera_render


FORMAT = "TESSERA_E4M3_K1_R1024"


def test_owner_import_does_not_load_campaign_or_serving():
    code = """
import importlib.abc
import sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == 'prismaquant.tessera_campaign' or
                fullname.startswith('tessera.serving') or
                fullname.split('.')[0] in {'tools', 'experiments'}):
            raise AssertionError(fullname)

sys.meta_path.insert(0, Block())
from prismaquant.tessera_cell_measurement import CellRecipe, measure_cells
assert 'prismaquant.tessera_campaign' not in sys.modules
assert not any(name.startswith('tessera.serving') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_batch_admits_every_hessian_before_encoding_or_publishing(monkeypatch, tmp_path):
    inputs = torch.ones(2, 32)
    weights = [torch.ones(32, 32), torch.full((32, 32), 2.0)]
    originals = [weight.clone() for weight in weights]
    seen = []

    def factors(name, plane):
        seen.append((name, plane))
        return {"ldl": torch.eye(32)} if name == "first" else None

    monkeypatch.setattr(tessera_render, "encode_tessera_units",
                        lambda *_args, **_kwargs: pytest.fail("encoded an unadmitted batch"))
    cache = SimpleNamespace(weights={}, cache_dir=None)
    with pytest.raises(tessera_render.HessianContractError, match="no Hessian"):
        measurement.measure_cells(qnames=["first", "second"], weights=weights,
            activations=[inputs, inputs], format_name=FORMAT, cache=cache,
            wire_dir=tmp_path, activation_kwargs_for=factors)
    assert [name for name, _plane in seen] == ["first", "second"]
    assert seen[0][1] == seen[1][1]
    assert cache.weights == {} and list(tmp_path.iterdir()) == []
    assert all(torch.equal(weight, original) for weight, original in zip(weights, originals))


@pytest.mark.parametrize("leaf,contract,dtype", [
    ("indexer.weights_proj", "a32", torch.float32),
    ("kv_b_proj", "a16", torch.bfloat16),
])
def test_direct_measurement_uses_public_producer_without_serving(
        monkeypatch, tmp_path, leaf, contract, dtype):
    import importlib.abc

    class BlockServing(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("tessera.serving"):
                raise AssertionError(fullname)

    monkeypatch.setattr(sys, "meta_path", [BlockServing(), *sys.meta_path])
    profile = SimpleNamespace(name="glm5_next",
        declared_architectures=lambda: ("Glm5NextForConditionalGeneration",))
    source = torch.randn(32, 32, generator=torch.Generator().manual_seed(194)).bfloat16()
    original = source.clone()
    cache = SimpleNamespace(weights={}, cache_dir=None)
    name = "model.layers.0.self_attn." + leaf
    anchor = measurement.measure_cell(qname=name, weight=source,
        activations=torch.ones(2, 32), format_name=FORMAT, cache=cache,
        wire_dir=tmp_path, hessian_required=False, profile=profile, structure="dense")
    assert cache.weights[(name, FORMAT)].dtype == dtype
    assert anchor.activation_contract == contract
    assert anchor.activation_quantized is False
    assert torch.equal(source, original)
