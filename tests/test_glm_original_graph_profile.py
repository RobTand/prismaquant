"""Behavioral CPU refusal/lifetime tests for the opt-in original graph gate."""
import builtins
import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from experiments import glm_original_graph_profile as graph
from experiments.glm_original_graph_source import (
    AuthenticatedSourceInputs, HeaderOnlyFile, derive_source_roster)


def manifest(paths):
    return [dict(path=str(path), bytes=path.stat().st_size,
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]


def test_hash_once_and_bound_reader_uses_authenticated_descriptor(tmp_path):
    path = tmp_path/'source.safetensors'
    path.write_bytes(b'payload\x00\xff')
    source = AuthenticatedSourceInputs(tmp_path, manifest([path]), block_bytes=2)
    with source:
        source.bind_roster({str(path): ['weight']})
        with patch('os.pread', wraps=os.pread) as reads:
            source.authenticate_payloads()
            count = reads.call_count
            source.authenticate_payloads()
            assert reads.call_count == count
        with source.open(builtins.open, path, 'rb') as stream:
            assert os.fstat(stream.fileno()).st_ino == os.fstat(source.fds[str(path)]).st_ino
            assert stream.read() == b'payload\x00\xff'
        fd = source.os_open(path, os.O_RDONLY)
        try:
            assert os.pread(fd, 7, 0) == b'payload'
        finally:
            os.close(fd)
        assert source.report()['authenticated'][0]['maximum_hash_buffer_bytes'] == 2
    assert source.fds == {}


def test_cpu_roster_binds_without_reading_payload(tmp_path):
    path = tmp_path/'source.safetensors'
    path.write_bytes(b'payload')
    source = AuthenticatedSourceInputs(tmp_path, manifest([path]))
    with source, patch('os.pread', side_effect=AssertionError('payload read')):
        source.bind_roster({str(path): ['weight']})
        assert source.report()['authenticated'] == []


def test_existing_loader_consumes_bound_safetensors(tmp_path):
    from safetensors.torch import save_file
    from prismaquant.layer_streaming import _read_layer_to_device
    path = tmp_path/'source.safetensors'
    expected = torch.arange(9,dtype=torch.float32).reshape(3,3)
    save_file({'weight':expected},path)
    source = AuthenticatedSourceInputs(tmp_path,manifest([path]))
    with source:
        source.bind_roster({str(path):['weight']})
        source.authenticate_payloads()
        with source.reader_binding(all_indexed_shards={str(path)}):
            values = _read_layer_to_device('layer.',{'layer.weight':str(path)},
                {'layer.weight':'weight'},torch.float32,torch.device('cpu'))
            assert torch.equal(values['layer.weight'],expected)
        assert any(row.get('key') == 'weight' for row in source.reads)


@pytest.mark.parametrize('mutation', ['replace', 'modify', 'digest'])
def test_source_mutation_and_digest_refuse_and_close(tmp_path, mutation):
    path = tmp_path/'source.safetensors'
    path.write_bytes(b'payload')
    lead = manifest([path])
    if mutation == 'digest':
        lead[0]['sha256'] = '0'*64
    source = AuthenticatedSourceInputs(tmp_path, lead)
    with pytest.raises((RuntimeError, ValueError)):
        with source:
            source.authenticate(path)
            if mutation == 'replace':
                alternate = tmp_path/'alternate'
                alternate.write_bytes(b'payload')
                alternate.replace(path)
            elif mutation == 'modify':
                path.write_bytes(b'changed')
            source.require_unchanged()
    assert not source.fds


@pytest.mark.parametrize('writable', [False, True])
def test_unhashed_or_writable_reader_is_sticky(tmp_path, writable):
    path = tmp_path/'source.safetensors'
    path.write_bytes(b'payload')
    source = AuthenticatedSourceInputs(tmp_path, manifest([path]))
    with pytest.raises(RuntimeError, match='unauthenticated source read'):
        with source:
            if writable:
                source.authenticate(path)
            with pytest.raises(RuntimeError):
                source.os_open(path, os.O_WRONLY if writable else os.O_RDONLY)
    assert not source.fds


def test_header_only_cannot_return_payload(tmp_path):
    path = tmp_path/'source.safetensors'
    header = json.dumps({'w':dict(shape=[3], dtype='BF16', data_offsets=[0,6])}).encode()
    path.write_bytes(len(header).to_bytes(8, 'little')+header+b'abcdef')
    source = AuthenticatedSourceInputs(tmp_path, [])
    with HeaderOnlyFile(source, str(path)) as reader:
        assert reader.get_slice('w').get_shape() == [3]
        assert reader.get_slice('w').get_dtype() == 'BF16'
        with pytest.raises(RuntimeError, match='unauthenticated metadata-only'):
            reader.get_tensor('w')
    assert source.metadata_reads[str(path)]['payload_bytes_read'] == 0


def test_roster_uses_profile_fixed_prefix_and_lookahead(tmp_path):
    profile = SimpleNamespace(checkpoint_to_live_name=lambda key, multimodal: key if key != 'ignored' else None)
    index = {f'model.language_model.layers.{layer}.w': f'{layer}.safetensors' for layer in range(8)}
    index.update({'model.visual.w':'fixed.safetensors', 'ignored':'ignored.safetensors'})
    result = derive_source_roster(tmp_path, index, profile)
    assert len(result) == 7
    assert not any(Path(path).stem in ('6','7','ignored') for path in result)
    index['model.visual.w'] = '../escape.safetensors'
    with pytest.raises(ValueError, match='escapes'):
        derive_source_roster(tmp_path, index, profile)


class TinyRunner:
    def __init__(self):
        self.layers = [torch.nn.Linear(3, 3, bias=False).to(torch.bfloat16).eval().requires_grad_(False)]
        self.calls = 0

    def _call(self, layer, hidden, *, batch, pass_state):
        self.calls += 1
        return self.layers[layer](hidden)

    def isolated_layer(self, batch, layer, hidden, *, pass_state):
        return self._call(layer, hidden, batch=batch, pass_state=pass_state)


def batch():
    return SimpleNamespace(input_ids=torch.ones(1,2,dtype=torch.long), position_ids=torch.arange(2),
        position_embeddings=None, attention_mask=None, shared_pass_state={})


def test_real_baseline_backward_in_outer_no_grad_preserves_source_and_rng():
    from prismaquant.sensitivity_probe import SharedStateCotangents
    runner, hidden = TinyRunner(), torch.ones(1,2,3,dtype=torch.bfloat16)
    before = graph.module_identity(runner.layers[0])
    rng = torch.get_rng_state()
    with torch.no_grad():
        result = graph.replay_arm(runner,0,hidden,batch(),{},7000,graph.ARMS[0],SharedStateCotangents(enabled=True))
    assert result['cotangent']['shape'] == [1,2,3]
    assert graph.module_identity(runner.layers[0]) == before
    assert torch.equal(rng, torch.get_rng_state())
    assert runner.calls == 1


def test_module_identity_samples_original_dense_size_without_float_rounding():
    # Actual original layer0 gate/up geometry: n-1 is not representable in FP32.
    module = torch.nn.Linear(4096,12288,bias=False,device='meta',dtype=torch.bfloat16)
    module.to_empty(device='cpu').eval().requires_grad_(False)
    with torch.no_grad():
        module.weight[0,0] = 3
        module.weight[-1,-1] = 7
    identity = graph.module_identity(module)
    assert identity['parameter:weight']['shape'] == [12288,4096]
    assert identity['parameter:weight']['samples']['shape'] == [64]


def test_inference_and_mutating_execution_refuse():
    runner = TinyRunner()
    with torch.inference_mode(), pytest.raises(RuntimeError, match='inference_mode'):
        with graph.unchanged_execution(runner.layers[0], batch(), {}):
            pass
    with pytest.raises(RuntimeError, match='mutated'):
        with graph.unchanged_execution(runner.layers[0], batch(), {}), torch.no_grad():
            runner.layers[0].weight.add_(1)
    with pytest.raises(RuntimeError, match='RNG'):
        with graph.unchanged_execution(runner.layers[0], batch(), {}):
            torch.rand(1)


def test_observer_preserves_original_object_and_recursion_and_failure(tmp_path):
    runner = TinyRunner()
    hidden = torch.ones(1,2,3,dtype=torch.bfloat16)
    result, originals = {'backwards':[]}, []
    original = runner._call
    def record(*args, **kwargs):
        value = original(*args, **kwargs)
        originals.append(value)
        return value
    runner._call = record
    def replay(runner, layer, hidden, batch, state, seed, arm, owner):
        value = runner.isolated_layer(batch,layer,hidden,pass_state=state)
        return dict(output=graph.tensor_identity(value),cotangent='same',stimulus='same',activity={},seed=seed,arm=arm)
    observer = graph.GraphObserver(runner,tmp_path,result,lambda layer: None,replay=replay)
    observer.row = 511
    with patch.object(graph, 'SHAPE',tuple(hidden.shape)), patch.object(runner,'_call',observer):
        output = runner._call(0,hidden,batch=batch(),pass_state={})
        assert output is originals[0]
        assert runner.calls == 13 and len(result['backwards']) == 12
        observer.replay = lambda *a: (_ for _ in ()).throw(RuntimeError('backward failed'))
        with pytest.raises(RuntimeError, match='backward failed'):
            runner._call(0,hidden,batch=batch(),pass_state={})
        assert not observer.in_replay


def test_completion_requires_actual_exact_schedule(tmp_path):
    runner = TinyRunner()
    observer = graph.GraphObserver(runner,tmp_path,{'backwards':[]},lambda layer: None)
    observer.tokens = [torch.tensor(row) for row in graph.ROWS]
    with pytest.raises(RuntimeError, match='exact72-call schedule'):
        observer.visit(4, lambda tokens: None)
    observer.result['backwards'] = [dict(layer=l, original_row=r, seed=s, arm=a) for l,r,s,a in graph.schedule()]
    with pytest.raises(graph.PrefixGraphQualificationComplete):
        observer.visit(4, lambda tokens: None)
    assert len(set(graph.schedule())) == 72


def test_close_source_joins_before_cache_release_and_drops_observer_owners():
    from concurrent.futures import Future
    order = []
    runner = TinyRunner()
    runner.model = runner.layers[0]
    current = torch.ones(3)
    lookahead = torch.ones(5)
    future = Future()
    future.set_result({'next':lookahead})
    runner.context = SimpleNamespace(layer_cache=SimpleNamespace(_cache={0:{'current':current}}),
        _inflight_lock=threading.Lock(),_inflight={1:future})
    def shutdown():
        order.append('joined')
        runner.context._inflight.clear()
    def reset(*,retain_cache):
        assert order == ['joined'] and retain_cache is False
        order.append('released')
        runner.context.layer_cache._cache.clear()
    runner.shutdown = shutdown
    runner.context.reset_between_chunks = reset
    observer = SimpleNamespace(runner=runner,original=runner._call,tokens=[torch.ones(1)])
    refs = graph.close_source(runner,observer)
    assert order == ['joined','released']
    assert observer.runner is observer.original is None and observer.tokens == []
    assert len(refs) == 3 and not all(ref.expired() for ref in refs)
    del current,lookahead,future,runner
    import gc
    gc.collect()
    assert all(ref.expired() for ref in refs)
