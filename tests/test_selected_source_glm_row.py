"""An instrumented selected GLM row reads only shards it byte-verified (#388).

The tiny glm5_next checkpoint is re-sharded into one shard per decoder layer
plus one shard of fixed state (embeddings, final norm, lm_head, visual tower).
A row that selects a layer-1 unit must hash the layer-1 shard and the fixed
shard, inherit the layer-0 digest from the census roster, reproduce the
canonical identity, and read tensor data only from shards it hashed.
"""
import json
import re
from pathlib import Path

import pytest
import torch

from test_glm5_next_streamed_forward_parity import (  # noqa: F401  (autouse shim)
    _build_tiny_model, _torch_only_causal_conv1d,
)
from test_glm_campaign_streaming import write_original_layout_checkpoint
from test_selected_source_shard_identity import (
    ROSTER_ORIGIN, producer_roster, runtime, streaming_contract,
)
from prismaquant import tessera_calibration_cache as cc
from prismaquant import tessera_campaign as campaign
from prismaquant.autoscale import selected_anchor_resources
from prismaquant.cost_streaming import build_streamed_causal_lm
from prismaquant.model_profiles.glm5_next import Glm5NextProfile

LAYERS_PREFIX = 'model.language_model.layers.'


def reshard_checkpoint(source, *, layers_prefix, num_layers):
    """Split `model.safetensors` into one shard per layer plus one fixed shard."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    single = source/'model.safetensors'
    pattern = re.compile(rf'^{re.escape(layers_prefix)}(\d+)\.')
    total = num_layers+1
    groups = {}
    with safe_open(str(single), framework='pt') as handle:
        for key in handle.keys():
            match = pattern.match(key)
            index = int(match.group(1))+1 if match else total
            shard = f'model-{index:05d}-of-{total:05d}.safetensors'
            groups.setdefault(shard, {})[key] = handle.get_tensor(key)
    single.unlink()
    weight_map = {}
    for shard, tensors in groups.items():
        save_file(tensors, str(source/shard), metadata={'format': 'pt'})
        weight_map.update({key: shard for key in tensors})
    (source/'model.safetensors.index.json').write_text(
        json.dumps(dict(metadata={}, weight_map=weight_map), indent=1, sort_keys=True))
    return weight_map


def instrument(monkeypatch, source):
    """Log ('hash'|'open'|'read', file, bytes-or-tensor) for files under `source`."""
    import safetensors
    from prismaquant import layer_streaming, streaming_model
    events = []
    root = source.resolve()
    real_open = safetensors.safe_open

    class Handle:
        def __init__(self, inner, shard):
            self._inner, self._shard = inner, shard
        def __enter__(self):
            self._inner.__enter__()
            return self
        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)
        def get_tensor(self, name):
            events.append(('read', self._shard, name))
            return self._inner.get_tensor(name)
        def __getattr__(self, name):
            return getattr(self._inner, name)

    def opened(path, *args, **kwargs):
        resolved = Path(path).resolve()
        inner = real_open(path, *args, **kwargs)
        if resolved.parent != root:
            return inner
        events.append(('open', resolved.name, None))
        return Handle(inner, resolved.name)

    for module in (safetensors, layer_streaming, streaming_model):
        monkeypatch.setattr(module, 'safe_open', opened)
    real_sha256 = cc.sha256
    def hashed(path, **kwargs):
        digest = real_sha256(path, **kwargs)
        events.append(('hash', Path(path).name, Path(path).stat().st_size))
        return digest
    monkeypatch.setattr(cc, 'sha256', hashed)
    return events


def test_selected_glm_row_byte_verifies_only_the_shards_it_reads(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISMAQUANT_TMPDIR', str(tmp_path/'staging'))
    source = tmp_path/'source'
    model = _build_tiny_model()
    write_original_layout_checkpoint(model, source)
    num_layers = len(model.model.language_model.layers)
    assert num_layers == 2
    weight_map = reshard_checkpoint(source, layers_prefix=LAYERS_PREFIX, num_layers=num_layers)
    layer0, layer1, fixed = sorted(set(weight_map.values()))
    modules = dict(model.named_modules())
    def linear_in(layer):
        return next(name for name, module in modules.items()
                    if isinstance(module, torch.nn.Linear) and
                    name.startswith(f'{LAYERS_PREFIX}{layer}.'))
    units = [linear_in(0), linear_in(1)]
    targets = [units[1]]
    unit_shapes = {name: list(modules[name].weight.shape) for name in units}
    counts = {name: 9 for name in units}
    max_abs = {name: 1.0 for name in units}
    contract = streaming_contract(layers_prefix=LAYERS_PREFIX, num_layers=num_layers,
                                  model_class=type(model).__name__)
    census = dict(model=str(source), model_load_contract=contract, capture_runtime=runtime(),
        attention_implementation='eager', unit_shapes=unit_shapes, counts=counts,
        max_abs=max_abs,
        expert_projection=dict(producer=dict(source=producer_roster(source, weight_map))))
    census_path = tmp_path/'census.json'
    census_path.write_text(json.dumps(census))
    calibration = {'fit_ids_sha256': 'draw'}
    canonical = cc.capture_identity(census_path, calibration=calibration, max_act_rows=4,
        model_load_contract=contract, attention_implementation='eager')
    record = cc.publish_capture(tmp_path/'capture', census_path=census_path, identity=canonical,
        acts={name: torch.zeros(4, unit_shapes[name][1]) for name in units},
        hessians={name: torch.eye(unit_shapes[name][1]) for name in units},
        counts=counts, maxima=max_abs)
    del model, modules

    events = instrument(monkeypatch, source)
    resources = selected_anchor_resources(source,
        unit_shapes={name: unit_shapes[name] for name in targets}, counts=counts,
        max_act_rows=4, cache_slots=2, prefetch_workers=1, headroom_gb=0)
    runner = build_streamed_causal_lm(str(source), device=torch.device('cpu'),
        dtype=torch.bfloat16, offload_folder=str(tmp_path/'selected-offload'),
        profile=Glm5NextProfile(), max_cache_slots=2, prefetch_workers=1,
        prefetch_min_available_gb=0, cache_headroom_gb=0,
        prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation='eager')
    events.append(('row', 'prepare_selected_source', None))
    identity, weights, preparation = campaign.prepare_selected_source(runner, targets,
        census_path=census_path, calibration=calibration, max_act_rows=4,
        model_load_contract=contract, attention_implementation='eager',
        calibration_cache=record['path'], calibration_cache_sha256=record['sha256'],
        selected_resources=resources)

    assert identity == canonical
    assert set(weights) == set(targets)
    hashed = {name: size for kind, name, size in events
              if kind == 'hash' and name.endswith('.safetensors')}
    read = {name for kind, name, _ in events if kind == 'read'}
    print(f'[#388 fixture] hashed {sum(hashed.values())} shard bytes across '
          f'{sorted(hashed)}; tensor data read from {sorted(read)}', flush=True)
    # The whole-layer install of layer 1 plus the fixed state the streamed
    # model materialises at construction: exactly the layer-1 and fixed shards.
    assert read == {layer1, fixed}
    assert read <= set(hashed), 'a shard was read without being byte-verified'
    start = events.index(('row', 'prepare_selected_source', None))
    for position, (kind, name, _) in enumerate(events):
        if kind == 'read' and position > start:
            assert any(kind_ == 'hash' and name_ == name
                       for kind_, name_, _ in events[start:position]), (
                f'{name} was read by the row before it was byte-verified')
    assert set(hashed) == {layer1, fixed}, (
        f'row hashed {sum(hashed.values())} shard bytes: {dict(sorted(hashed.items()))}; '
        f'expected only {sorted([layer1, fixed])}')
    assert sum(hashed.values()) == sum((source/name).stat().st_size for name in (layer1, fixed))
    assert preparation['source_verification'] == dict(
        byte_verified=[layer1, fixed], inherited_from_census_roster=[layer0],
        byte_verified_auxiliary=sorted(name for name in identity['source_files']
                                       if not name.endswith('.safetensors')),
        roster_origin=ROSTER_ORIGIN, canonical_manifest_sha256=record['sha256'],
        selected_layers=[1], layer_shards=[layer1], fixed_state_shards=[fixed])
    assert preparation['initialization_witness_origin'] == 'complete-canonical-capture'
