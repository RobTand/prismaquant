"""Cell measurements admit complete recipes before publishing priced bytes."""
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from prismaquant import tessera_cell_measurement as measurement
from prismaquant import tessera_render


FORMAT = "TESSERA_E4M3_K1_R1024"
GLM = SimpleNamespace(name="glm5_next",
                      declared_architectures=lambda: ("Glm5NextForConditionalGeneration",))
#: GLM-5.3's two direct consumers: leaf -> (shape, activation contract,
#: retained dtype, the module and role its wire decodes as). The shapes differ,
#: as they do in the model, so the compatibility planner batches each apart.
DIRECT_CONSUMERS = {
    "indexer.weights_proj": ((32, 64), "a32", torch.float32,
                             "indexer.wk_weights_proj", "weights_proj"),
    "kv_b_proj": ((64, 64), "a16", torch.bfloat16, "kv_b_proj", "kv_b_proj"),
}


def _direct_consumer_cells(root, *, batch_size):
    """Measure two layers of each GLM direct consumer through the planner.

    ``batch_size`` 1 measures every unit on its own (``measure_cells`` hands a
    one-unit batch to ``measure_cell``); a larger one encodes each compatible
    batch in one producer call, as the campaign does. Weights are BF16 and
    activations FP32, so a quantized or rounded activation would move a score.
    """
    from prismaquant.production_weight_cache import ProductionWeightCache

    units = {}
    for layer in (1, 3):
        for seed, (leaf, (shape, *_rest)) in enumerate(DIRECT_CONSUMERS.items()):
            generator = torch.Generator().manual_seed(1000 * layer + seed)
            units[f"model.layers.{layer}.self_attn.{leaf}"] = (
                torch.randn(shape, generator=generator).bfloat16(),
                torch.randn(2, shape[1], generator=generator))
    originals = {name: tuple(value.clone() for value in pair) for name, pair in units.items()}
    cache_dir, wire_dir = root / "cache", root / "wire"
    cache_dir.mkdir(parents=True)
    wire_dir.mkdir()
    cache = ProductionWeightCache(weights={}, levers={}, cache_dir=str(cache_dir), metadata={})
    family, rung = FORMAT.rsplit("_R", 1)
    structures = dict.fromkeys(units, "dense")
    batches = measurement.compatible_batches(
        [(name, family, int(rung)) for name in units],
        weights={name: weight for name, (weight, _inputs) in units.items()},
        batch_size=batch_size, structures=structures)
    anchors = {}
    for batch in batches:
        names = [name for name, _family, _rung in batch]
        for anchor in measurement.measure_cells(
                qnames=names, weights=[units[name][0] for name in names],
                activations=[units[name][1] for name in names], format_name=FORMAT,
                cache=cache, wire_dir=wire_dir, hessian_required=False,
                structures=structures, profile=GLM):
            anchors[anchor.qname] = anchor
    return SimpleNamespace(units=units, originals=originals, batches=batches,
                           anchors=anchors, cache=cache, cache_dir=cache_dir,
                           wire_dir=wire_dir)


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


#: Run as a fresh interpreter: the blocker is installed before anything else
#: is imported, so a ``tessera.serving`` module an earlier test loaded into
#: this process cannot be returned from ``sys.modules`` behind its back, and an
#: attempt the code under test swallows is still recorded.
COLD_DIRECT_MEASUREMENT = """
import importlib.abc
import json
import sys


def forbidden(name):
    return name == "tessera.serving" or name.startswith("tessera.serving.")


attempts = []


class BlockServing(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            attempts.append(fullname)
            raise AssertionError(fullname)
        return None


preloaded = sorted(name for name in sys.modules if forbidden(name))
sys.meta_path.insert(0, BlockServing())

from pathlib import Path

import torch

from tests import test_tessera_cell_measurement as cells

root = Path(sys.argv[1])
measured = {}
for batch_size in (1, 4):
    run = cells._direct_consumer_cells(root / str(batch_size), batch_size=batch_size)
    measured[batch_size] = sorted(
        [name, anchor.activation_contract, anchor.encoding_batch_size,
         anchor.activation_quantized,
         str(torch.load(run.cache_dir / run.cache.weights[(name, cells.FORMAT)],
                        weights_only=True).dtype)]
        for name, anchor in run.anchors.items())
print(json.dumps({"preloaded": preloaded, "attempts": attempts, "measured": measured,
                  "loaded": sorted(name for name in sys.modules if forbidden(name))}))
"""


def test_direct_measurement_never_loads_the_serving_runtime_in_a_cold_process(tmp_path):
    """Scalar and batched direct-consumer cells run on public Tessera modules only."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", COLD_DIRECT_MEASUREMENT, str(tmp_path)],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert (report["preloaded"], report["attempts"], report["loaded"]) == ([], [], [])
    expected = {"indexer.weights_proj": ["a32", "torch.float32"],
                "kv_b_proj": ["a16", "torch.bfloat16"]}
    for batch_size, rows in report["measured"].items():
        assert len(rows) == 4
        for name, contract, encoded_with, quantized, dtype in rows:
            assert [contract, dtype] == expected[name.split(".self_attn.", 1)[1]]
            assert encoded_with == min(int(batch_size), 2) and quantized is False


def test_batched_direct_consumers_keep_their_contracts_charge_and_inputs(monkeypatch, tmp_path):
    """Batching keeps each GLM direct consumer's scalar contract (C-TST-02).

    Two a32 indexer heads and two a16 ``kv_b_proj`` units go through
    ``compatible_batches`` and the multi-unit ``measure_cells``. Each unit
    keeps its retained dtype, an unquantized activation, exactly one
    direct-consumer decode, one resident-matrix charge, unmutated resident
    inputs and published files, and prices exactly what the scalar path
    prices. CPU contract evidence only: no serving, batch-resume or
    performance claim.
    """
    from dataclasses import replace

    from tessera import projection
    from prismaquant import tessera_campaign_wire as campaign_wire
    from prismaquant.format_registry import get_format
    from prismaquant.production_weight_cache import _local_forward_render_score

    decode = projection.direct_consumer_weight
    decoded = []

    def counted(blob, module, role, route_family, **kwargs):
        decoded.append((module, role))
        return decode(blob, module, role, route_family, **kwargs)

    monkeypatch.setattr(projection, "direct_consumer_weight", counted)
    scalar = _direct_consumer_cells(tmp_path / "scalar", batch_size=1)
    scalar_decodes = list(decoded)
    decoded.clear()
    batched = _direct_consumer_cells(tmp_path / "batched", batch_size=len(scalar.units))

    def leaf(name):
        return name.split(".self_attn.", 1)[1]

    # The planner splits the two contracts by shape; each batch is multi-unit.
    assert sorted([leaf(name) for name, _family, _rung in batch]
                  for batch in batched.batches) == [["indexer.weights_proj"] * 2,
                                                    ["kv_b_proj"] * 2]
    assert [len(batch) for batch in scalar.batches] == [1] * len(scalar.units)
    assert set(batched.cache.weights) == {(name, FORMAT) for name in batched.units}
    assert {(name, FORMAT) for name in batched.units} <= campaign_wire.wire_coordinates(
        batched.cache, batched.wire_dir)
    for name, (weight, inputs) in batched.units.items():
        shape, contract, dtype, module_leaf, role = DIRECT_CONSUMERS[leaf(name)]
        module = name.split(".self_attn.", 1)[0] + ".self_attn." + module_leaf
        anchor, reference = batched.anchors[name], scalar.anchors[name]
        assert (anchor.encoding_batch_size, reference.encoding_batch_size) == (2, 1)
        assert anchor.activation_contract == contract
        assert anchor.activation_quantized is False
        # One resident decoded matrix, charged once beside the format's bytes.
        assert anchor.memory_bytes == (get_format(FORMAT).memory_bytes_for_shape(shape)
                                       + shape[0] * shape[1] * dtype.itemsize)
        assert decoded.count((module, role)) == scalar_decodes.count((module, role)) == 1
        blob = campaign_wire.wire_path(batched.wire_dir, name, FORMAT).read_bytes()
        retained = torch.load(batched.cache_dir / batched.cache.weights[(name, FORMAT)],
                              weights_only=True)
        assert retained.dtype == dtype
        assert torch.equal(retained, decode(blob, module, role, "TESSERA_FP8"))
        assert anchor.dloss == _local_forward_render_score(
            reference_weight=weight, rendered_weight=retained, activations=inputs,
            activation_quantize=lambda x: x, activation_max_abs=None)[0]
        # The batch prices what the scalar path prices, on the same bytes.
        assert blob == campaign_wire.wire_path(scalar.wire_dir, name, FORMAT).read_bytes()
        assert torch.equal(retained, torch.load(
            scalar.cache_dir / scalar.cache.weights[(name, FORMAT)], weights_only=True))
        assert (replace(anchor, seconds=0.0, encoding_batch_size=1)
                == replace(reference, seconds=0.0))
        # The resident inputs are read, never written.
        assert all(torch.equal(now, before)
                   for now, before in zip((weight, inputs), batched.originals[name]))
    assert len(decoded) == len(batched.units) == len(scalar_decodes)
