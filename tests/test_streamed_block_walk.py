"""Ordinary walks keep real source delivery and release under one owner."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from prismaquant import streaming_model
from prismaquant.layer_streaming import LayerCache, _build_install_resolver
from prismaquant.streaming_model import StreamingContext


@pytest.fixture
def source_context(tmp_path, monkeypatch):
    weights = [torch.tensor([[1., .25], [.5, 1.]]),
               torch.tensor([[.75, .5], [0., 1.25]]),
               torch.tensor([[1.5, 0.], [.25, .5]])]
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in weights])
    for layer, weight in zip(model.layers, weights):
        layer.weight.data.copy_(weight)
    path = tmp_path / 'source.safetensors'
    save_file(model.state_dict(), str(path))
    names = list(model.state_dict())
    model.to_empty(device='meta')
    context = StreamingContext(model=model, base_model=model, layers=model.layers,
        layers_prefix='layers.', num_layers=3,
        install_resolvers=[_build_install_resolver(model, f'layers.{layer}') for layer in range(3)],
        weight_shard=dict.fromkeys(names, str(path)), weight_ckpt={name: name for name in names},
        layer_cache=LayerCache(max_bytes=1024, max_entries=2),
        prefetch_pool=ThreadPoolExecutor(max_workers=1), device=torch.device('cpu'),
        dtype=torch.float32, offload_folder=str(tmp_path),
        estimated_layer_bytes=16, prefetch_workers=1)
    events, reads = [], []
    source_read = streaming_model._read_layer_to_device
    def read(prefix, *args, **kwargs):
        reads.append(int(prefix.split('.')[1]))
        return source_read(prefix, *args, **kwargs)
    monkeypatch.setattr(streaming_model, '_read_layer_to_device', read)
    install, unload, prefetch = context.install, context.unload, context.schedule_prefetch
    def install_layer(layer, **kwargs):
        events.append(('install', layer, kwargs['require_prefetched']))
        return install(layer, **kwargs)
    def unload_layer(layer, **kwargs):
        events.append(('unload', layer))
        return unload(layer, **kwargs)
    def request(layer):
        events.append(('prefetch', layer))
        return prefetch(layer)
    monkeypatch.setattr(context, 'install', install_layer)
    monkeypatch.setattr(context, 'unload', unload_layer)
    monkeypatch.setattr(context, 'schedule_prefetch', request)
    try:
        yield context, weights, events, reads
    finally:
        context.shutdown()


def test_real_walk_matches_source_arithmetic_and_closes_before_observations(source_context):
    context, weights, events, reads = source_context
    hidden = torch.tensor([[.25, 1.], [2., -.5]])
    expected = hidden.clone()
    for weight in weights:
        expected = torch.nn.functional.linear(expected, weight)
    with context.walk_layers(range(3), prefetch_lookahead=1, require_prefetched=True) as walk:
        for block in walk:
            with block:
                events.append(('work', block.layer))
                hidden = context.layers[block.layer](hidden)
                assert block.source in {'wait', 'hot'}
            assert context.layers[block.layer].weight.is_meta
            events.append(('observe', block.layer))
    assert torch.equal(hidden, expected)
    assert sorted(reads) == [0, 1, 2]
    assert events[0] == ('prefetch', 0)
    lifecycle = [event for event in events if event[0] != 'prefetch']
    assert lifecycle == [event for layer in range(3) for event in
        [('install', layer, True), ('work', layer), ('unload', layer), ('observe', layer)]]


@pytest.mark.parametrize('stop', ['break', 'error'])
def test_outer_scope_releases_an_unclosed_current_block(source_context, stop):
    context, _weights, events, _reads = source_context
    def run():
        with context.walk_layers(range(3), prefetch_lookahead=1, require_prefetched=True) as walk:
            for block in walk:
                assert not context.layers[block.layer].weight.is_meta
                if stop == 'error':
                    raise LookupError('work failed')
                break
    if stop == 'error':
        with pytest.raises(LookupError, match='work failed'):
            run()
    else:
        run()
    assert context.layers[0].weight.is_meta
    assert [event for event in events if event[0] == 'unload'] == [('unload', 0)]


def test_block_close_is_idempotent_and_a_pinned_layer_stays_borrowed(source_context):
    context, _weights, events, _reads = source_context
    context.schedule_prefetch(0)
    context.install(0, require_prefetched=True)
    events.clear()
    with context.walk_layers(range(3), prefetch_lookahead=1,
                             require_prefetched=True, pinned_layer=0) as walk:
        for block in walk:
            with block:
                assert not context.layers[block.layer].weight.is_meta
                if block.layer == 0:
                    assert block.source is None
            block.close()
    assert not context.layers[0].weight.is_meta
    assert [event for event in events if event[0] == 'unload'] == [('unload', 1), ('unload', 2)]
    context.unload(0)


def test_missing_required_prefetch_refuses_before_source_io(source_context, monkeypatch):
    context, _weights, _events, reads = source_context
    monkeypatch.setattr(context, 'schedule_prefetch', lambda layer: None)
    with pytest.raises(RuntimeError, match='refusing synchronous cold source read'):
        with context.walk_layers(range(3), prefetch_lookahead=1, require_prefetched=True) as walk:
            next(walk)
    assert reads == []


def test_unretained_delivery_is_consumed_without_a_second_read(source_context, monkeypatch):
    context, _weights, _events, reads = source_context
    monkeypatch.setattr(context.layer_cache, '_effective_max', lambda: 0)
    with context.walk_layers(range(3), prefetch_lookahead=1, require_prefetched=True) as walk:
        for block in walk:
            with block:
                assert block.source == 'wait'
    assert sorted(reads) == [0, 1, 2]
    assert context.prefetch_delivered_unretained == 3
