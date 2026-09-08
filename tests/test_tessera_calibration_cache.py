"""A reusable calibration keeps full-H/prefix-X bytes and refuses stale inputs."""
import copy
import importlib.metadata
import prismaquant
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import tessera_calibration_cache as cc


def canonical_fields():
    return dict(model_load_contract=dict(schema='prismaquant.pretrained_initialization.v1',
        scope='checkpoint_missing_state',status='completed',
        transformers_version=importlib.metadata.version('transformers')),
        attention_implementation='eager',capture_runtime=dict(torch=torch.__version__,
        cuda=torch.version.cuda,transformers=importlib.metadata.version('transformers')))


def identity(path,calibration,max_act_rows):
    fields = canonical_fields()
    return cc.capture_identity(path,calibration=calibration,max_act_rows=max_act_rows,
        model_load_contract=fields['model_load_contract'],attention_implementation='eager')


@pytest.fixture
def capture(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'config.json').write_text('{}')
    (source/'model.safetensors').write_bytes(b'bounded source fixture')
    census = dict(model=str(source),counts={'a':5,'b':7},max_abs={'a':4.,'b':8.},
                  unit_shapes={'a':[3,2],'b':[3,2]},layer_stride=1,
                  anchor_groups={'u:a':['a'],'u:b':['b']},**canonical_fields())
    path = tmp_path/'census.json'
    path.write_text(json.dumps(census))
    capture_id = identity(path,calibration={'fit_ids_sha256':'draw'},max_act_rows=2)
    acts = {'a':torch.tensor([[1.,2.],[3.,4.]]),'b':torch.ones(2,2)}
    hessians = {'a':torch.eye(2)*13,'b':torch.eye(2)*25}
    root = tmp_path/'capture'
    record = cc.publish_capture(root,census_path=path,identity=capture_id,acts=acts,
        hessians=hessians,counts=census['counts'],maxima=census['max_abs'])
    return root,path,census,capture_id,acts,hessians,record


def test_prefetch_only_selected_and_preserves_full_h_and_prefix_precision(capture):
    root,path,census,identity,acts,hessians,record = capture
    (root/'inputs/b.pt').unlink()  # unrelated units are not transferred by this quantum
    values,_ = cc.prefetch_capture(record['path'],expected_identity=identity,
        census=census,names=['a'],device='cpu',expected_sha256=record['sha256'])
    assert set(values[0]) == {'a'}
    assert torch.equal(values[0]['a'],acts['a'])
    assert torch.equal(values[1]['a'],hessians['a'])
    assert values[2] == {'a':5}


def test_selected_prefetch_guards_and_advises_verified_files(capture, monkeypatch):
    from prismaquant import perturbed_x_cache
    root, path, census, identity, acts, hessians, record = capture
    observed, advised = [], []
    monkeypatch.setattr(perturbed_x_cache, 'release_activation_cache_file_pages',
        lambda path, *, expected_stat: advised.append((path.name, expected_stat.st_size)))
    values, _ = cc.prefetch_capture(record['path'], expected_identity=identity,
        census=census, names=['a'], device='cpu', expected_sha256=record['sha256'],
        release_file_pages=True, resource_check=lambda label, **kwargs: observed.append((label, kwargs)))
    assert advised == [('a.pt', (root/'inputs/a.pt').stat().st_size)]
    assert ('before_capture_prefetch:a', {'reserve_bytes': 64}) in observed
    assert observed[-1] == ('after_capture_prefetch:a', {})
    assert torch.equal(values[0]['a'], acts['a'])
    assert torch.equal(values[1]['a'], hessians['a'])


def test_export_input_page_release_preserves_existing_file_bytes(capture, monkeypatch, tmp_path):
    from prismaquant import tessera_campaign as campaign, perturbed_x_cache
    _root, _path, census, identity, _acts, hessians, _record = capture
    calls = []
    original = perturbed_x_cache.release_activation_cache_file_pages
    def release(path, *, expected_stat):
        calls.append(path.name)
        return original(path, expected_stat=expected_stat)
    monkeypatch.setattr(perturbed_x_cache, 'release_activation_cache_file_pages', release)
    outcomes = []
    for bounded in (False, True):
        root = tmp_path/str(bounded)
        root.mkdir()
        observed = []
        path, _scales, digest = campaign.write_export_inputs(root, hessians=hessians,
            hessian_rows=census['counts'], hessian_identity=identity['calibration'],
            static_scales={}, static_scale_policy='fixture', release_file_pages=bounded,
            resource_check=(lambda label, **kwargs: observed.append(label)) if bounded else None)
        outcomes.append((path.read_bytes(), digest))
        if bounded:
            assert observed[-1] == 'after_selected_export_input_write'
    assert outcomes[0] == outcomes[1]
    assert calls == ['hessian_capture.pt.tmp']*2+['hessian_capture.pt']


def test_export_input_page_release_does_not_reread_the_archive(capture, monkeypatch, tmp_path):
    """Releasing a completed archive's pages must not read the archive again.

    The capture seal is computed in memory before the write, and
    ``release_activation_cache_file_pages`` fsyncs and advises the whole file
    from its descriptor. A hashing pass over the written bytes has no
    consumer, and on a routed GLM stack it is a full read of a 44 GB sidecar
    per row (PR #376 audit, finding 1).
    """
    from prismaquant import tessera_campaign as campaign, tessera_calibration_cache
    _root, _path, census, identity, _acts, hessians, _record = capture
    hashed = []
    original = tessera_calibration_cache.sha256
    def counting_sha256(path, **kwargs):
        hashed.append(Path(path).name)
        return original(path, **kwargs)
    monkeypatch.setattr(tessera_calibration_cache, 'sha256', counting_sha256)
    observed = []
    root = tmp_path/'bounded'
    root.mkdir()
    path, _scales, _digest = campaign.write_export_inputs(root, hessians=hessians,
        hessian_rows=census['counts'], hessian_identity=identity['calibration'],
        static_scales={}, static_scale_policy='fixture', release_file_pages=True,
        resource_check=lambda label, **kwargs: observed.append(label))
    assert path.name not in hashed
    assert not [label for label in observed if 'capture_hash' in label]


def test_completed_anchor_page_release_does_not_reread_the_entries(monkeypatch, tmp_path):
    """The rendered shard and the wire are advised out, never hashed again."""
    import types
    import torch
    from prismaquant import tessera_campaign as campaign
    from prismaquant import perturbed_x_cache, tessera_calibration_cache

    def refuse_reread(path, **kwargs):
        raise AssertionError(f'completed anchor entry was re-read: {Path(path).name}')
    monkeypatch.setattr(tessera_calibration_cache, 'sha256', refuse_reread)
    released = []
    original = perturbed_x_cache.release_activation_cache_file_pages
    def release(path, *, expected_stat):
        released.append(Path(path).name)
        return original(path, expected_stat=expected_stat)
    monkeypatch.setattr(perturbed_x_cache, 'release_activation_cache_file_pages', release)

    cache_dir = tmp_path/'cache'
    wire_dir = tmp_path/'wire'
    cache_dir.mkdir()
    wire_dir.mkdir()
    cache = types.SimpleNamespace(weights={}, cache_dir=str(cache_dir),
                                  metadata={'release_completed_anchor_file_pages': True})
    weight = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    activations = torch.ones(4, 8)
    spec = types.SimpleNamespace(
        bits_for_shape=lambda shape: 4 * shape[0] * shape[1],
        memory_bytes_for_shape=lambda shape: shape[0] * shape[1] // 2,
        act_dtype_name=None)
    prepared = dict(spec=spec, family=types.SimpleNamespace(name='fixture'),
                    rung=1024, activation_qdq=lambda x: x, input_scale=None,
                    activation_kwargs={})
    anchor = campaign._finish_anchor(qname='layers.0.q_proj', weight=weight,
        activations=activations, format_name='FIXTURE_R4', cache=cache,
        wire_dir=wire_dir, prepared=prepared, render=weight.clone(),
        blob=b'wire-bytes', elapsed=0.0)
    rendered = cache.weights[('layers.0.q_proj', 'FIXTURE_R4')]
    assert sorted(released) == sorted([Path(rendered).name,
                                       'layers__0__q_proj__FIXTURE_R4.tessera'])
    assert anchor.wire_bytes == len(b'wire-bytes')


@pytest.mark.parametrize('change',['artifact','manifest','source','draw','geometry','scope'])
def test_prefetch_refuses_drift(capture,change):
    root,path,census,identity,acts,hessians,record = capture
    expected = copy.deepcopy(identity)
    if change == 'artifact':
        with (root/'inputs/a.pt').open('ab') as f:
            f.write(b'changed')
    elif change == 'manifest':
        with open(record['path'],'a') as f:
            f.write(' ')
    elif change == 'source':
        (path.parent/'source/model.safetensors').write_bytes(b'changed')
        expected = globals()['identity'](path,calibration=identity['calibration'],max_act_rows=2)
    elif change == 'draw':
        expected['calibration']['fit_ids_sha256'] = 'other'
    elif change == 'geometry':
        expected['units']['a'] = [4,2]
    else:
        expected['units'].pop('b')
    with pytest.raises(RuntimeError,match='checksum|identity|manifest changed'):
        cc.prefetch_capture(record['path'],expected_identity=expected,census=census,
            names=['a'],device='cpu',expected_sha256=record['sha256'])


def test_completed_journal_revalidates_artifacts(capture):
    root,path,census,identity,acts,hessians,record = capture
    again = cc.publish_capture(root,census_path=path,identity=identity,acts=acts,
        hessians=hessians,counts=census['counts'],maxima=census['max_abs'])
    assert again == record
    (root/'inputs/a.pt').unlink()
    with pytest.raises(FileNotFoundError):
        cc.publish_capture(root,census_path=path,identity=identity,acts=acts,
            hessians=hessians,counts=census['counts'],maxima=census['max_abs'])


def test_layer_writer_requires_full_scope_and_actual_witness(capture, tmp_path):
    _root, path, census, identity, acts, hessians, _record = capture
    root = tmp_path/'layer-writer'
    writer = cc.CaptureWriter(root, census_path=path, identity=identity)
    def write(name):
        writer.write(acts={name: acts[name]}, hessians={name: hessians[name]},
            counts={name: census['counts'][name]}, maxima={name: census['max_abs'][name]})
    write('a')
    assert (root/'inputs/a.pt').is_file()
    assert not (root/'capture_manifest.json').exists()
    with pytest.raises(RuntimeError, match='full census'):
        writer.finish(model_load_contract=identity['model_load_contract'])
    write('b')
    wrong = dict(identity['model_load_contract'], transformers_version='wrong-runtime')
    with pytest.raises(RuntimeError, match='actual capture initialization'):
        writer.finish(model_load_contract=wrong)
    assert not (root/'capture_manifest.json').exists()
    record = writer.finish(model_load_contract=identity['model_load_contract'])
    assert cc.require_capture_contract(record['path'])['status'] == 'complete'


def test_layer_writer_replay_refuses_changed_hessian(capture, tmp_path):
    _root, path, census, identity, acts, hessians, _record = capture
    root = tmp_path/'interrupted'
    writer = cc.CaptureWriter(root, census_path=path, identity=identity)
    writer.write(acts={'a':acts['a']}, hessians={'a':hessians['a']},
                 counts={'a':5}, maxima={'a':4.})
    resumed = cc.CaptureWriter(root, census_path=path, identity=identity)
    with pytest.raises(RuntimeError, match='replayed capture differs'):
        resumed.write(acts={'a':acts['a']}, hessians={'a':hessians['a']+torch.eye(2)},
                      counts={'a':5}, maxima={'a':4.})
    assert not (root/'capture_manifest.json').exists()


def test_layer_writer_refuses_insufficient_disk(capture, tmp_path, monkeypatch):
    import shutil
    _root, path, census, identity, acts, hessians, _record = capture
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match='additional disk bytes'):
        cc.CaptureWriter(tmp_path/'no-space', census_path=path, identity=identity)


@pytest.mark.parametrize('streamed_selection', [False, True])
def test_cli_capture_then_reuse_never_repeats_forward(monkeypatch,tmp_path,streamed_selection):
    # This synthetic orchestration fixture has no source CUDA allocations;
    # the real GLM test exercises the finite-cgroup guard on the native lane.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    from test_tessera_campaign_resume import _main_fixture,UNIT
    tc,_,argv,model,inputs = _main_fixture(monkeypatch,tmp_path)
    model.config = SimpleNamespace(_attn_implementation='eager')
    fields = canonical_fields()
    if streamed_selection:
        fields['model_load_contract'] = dict(schema='prismaquant.streaming_initialization.v1',
            scope='streamed_text_source_forward', status='completed',
            transformers_version=importlib.metadata.version('transformers'),
            model_class='SyntheticSource', dtype='torch.bfloat16', layers_prefix='model.layers.',
            num_layers=1, persistent_tensors=1, derived_buffers=0,
            state_sha256='a'*64, source_map_sha256='b'*64)
    monkeypatch.setattr(prismaquant,'pretrained_initialization_contract',lambda model:fields['model_load_contract'])
    argv += ['--attention-implementation','eager']
    source = tmp_path/'source'
    source.mkdir()
    (source/'config.json').write_text('{}')
    (source/'model.safetensors').write_bytes(b'fixture')
    argv[argv.index('--model')+1] = str(source)
    argv[argv.index('--hessian')+1] = 'require'
    calibration = tc.th.calibration_identity(inputs['text'],inputs['tokens'],fit_tokens=4,
        source='wikitext-2-raw-v1/train',split_role='calibration',model=str(source),
        seed=0,nsamples=32,seqlen=512,fit_tokens_min=4)
    if streamed_selection:
        # A selected row inherits unread shard digests from the sealed roster (#388).
        fields['expert_projection'] = dict(producer=dict(source=dict(
            files={'model.safetensors': cc.sha256(source/'model.safetensors')},
            auxiliary_sha256={}, config_sha256=cc.sha256(source/'config.json'),
            tensors={UNIT+'.weight': 'model.safetensors'})))
    census = tc.calibration_census({UNIT:4},{UNIT:3.},args=SimpleNamespace(model=str(source),
        nsamples=32,seqlen=512,seed=0,layer_stride=1),groups={'u:'+UNIT:[UNIT]},
        dense_targets=[UNIT],expert_targets=[],shapes={UNIT:[32,256]},identity=calibration,**fields)
    census_path = tmp_path/'census.json'
    census_path.write_text(json.dumps(census))
    argv += ['--nsamples','32','--seqlen','512','--layer-stride','1',
             '--calibration-census',str(census_path)]
    root = tmp_path/'full-capture'
    assert tc.main([*argv,'--capture-calibration-out',str(root)]) == 0
    monkeypatch.setattr(tc,'_collect_activations',lambda *a,**k: pytest.fail('repeated forward'))
    assert tc.main([*argv,'--capture-calibration-out',str(root)]) == 0
    if streamed_selection:
        from prismaquant import cost_streaming, autoscale
        source_calls = []
        def snapshot(names, **kwargs):
            source_calls.append(list(names))
            return {UNIT: model.model.layers[0].proj.weight.detach().clone()}, dict(
                schema='prismaquant.selected_source_weights.v1', source_forward_count=0)
        runner = SimpleNamespace(model=model, snapshot_selected_weights=snapshot,
            shutdown=lambda: source_calls.append('shutdown'),
            selected_source_shards=lambda names: dict(layers=[0], layer_shards=['model.safetensors'],
                fixed_state_shards=[], tensors={UNIT+'.weight': 'model.safetensors'}))
        monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', lambda *a, **k: runner)
        monkeypatch.setattr(autoscale, 'selected_anchor_resources', lambda *a, **k: dict(
            memory_bytes=1024**3, selected_source_weight_bytes=32768,
            phases={'resident_anchors': {'factorization_scratch_bytes': 4*256**2*4}}))
        selection = tmp_path/'units.json'
        selection.write_text(json.dumps(dict(schema=tc.UNITS_SCHEMA,
            groups=[dict(key='u:'+UNIT, members=[UNIT])])))
        argv += ['--streaming', '--units', str(selection),
                 '--calibration-cache-sha256', cc.sha256(root/'capture_manifest.json')]
    assert tc.main([*argv,'--calibration-cache',str(root/'capture_manifest.json')]) == tc.EXIT_EMPTY_MENU
    if streamed_selection:
        assert source_calls == [[UNIT], 'shutdown']


def test_driver_capture_and_plan_bind_one_complete_capture(capture):
    from tools import dispatch_tessera_campaign as dispatch
    root,path,census,identity,acts,hessians,record = capture
    spec = path.parent/'spec.json'
    spec.write_text(json.dumps(dict(model=census['model'],campaign_argv=[],
        cwd=str(path.parent),python='python3',env={},cpus=1,headroom_gb=2)))
    common = ['--spec',str(spec),'--workspace',str(path.parent)]
    assert dispatch.main(['capture',*common]) == 0
    quantum = json.loads((path.parent/'capture-manifest.json').read_text())
    assert len(quantum) == 1
    assert '--capture-calibration-out' in quantum[0]['argv']
    assert '--units' not in quantum[0]['argv']
    assert dispatch.main(['plan',*common,'--calibration-cache',record['path']]) == 0
    rows = json.loads((path.parent/'manifest.json').read_text())
    for row in rows:
        argv = row['argv']
        assert argv[argv.index('--calibration-cache')+1] == record['path']
        assert argv[argv.index('--calibration-cache-sha256')+1] == record['sha256']
    manifest = json.loads(Path(record['path']).read_text())
    manifest['status'] = 'partial'
    Path(record['path']).write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='complete.*capture'):
        dispatch.main(['plan',*common,'--calibration-cache',record['path']])


def test_merge_refuses_disagreeing_capture_bindings():
    from test_tessera_campaign_fanout import _payloads
    from tools import dispatch_tessera_campaign as dispatch
    payloads = _payloads()
    for index,payload in enumerate(payloads.values()):
        payload['provenance']['calibration_cache'] = dict(path='/capture.json',sha256=str(index))
    with pytest.raises(dispatch.MergeRefused,match='calibration_cache'):
        dispatch.merge_payloads(payloads,census={'counts':{'a':16384,'b':16384}},capture_sha256='merged')


@pytest.mark.parametrize('field',['model_load_contract','attention_implementation','capture_runtime'])
def test_identity_refuses_legacy_or_changed_census(capture,field):
    root,path,census,capture_id,acts,hessians,record = capture
    census.pop(field)
    path.write_text(json.dumps(census))
    with pytest.raises((RuntimeError,ValueError),match='initialization|runtime|attention'):
        identity(path,calibration=capture_id['calibration'],max_act_rows=2)


@pytest.mark.parametrize('field',['model_load_contract','attention_implementation','capture_runtime','schema'])
def test_downstream_contract_refuses_unqualified_identity(capture,field):
    root,path,census,capture_id,acts,hessians,record = capture
    manifest = json.loads(Path(record['path']).read_text())
    manifest['identity'].pop(field)
    Path(record['path']).write_text(json.dumps(manifest))
    with pytest.raises((RuntimeError,ValueError),match='initialization|runtime'):
        cc.require_capture_contract(record['path'])


@pytest.mark.parametrize('kwargs_route',[False,True])
def test_boundary_callback_preserves_raw_arguments_and_reservoir(kwargs_route):
    from test_tessera_campaign_packed import _RoutedModel,EXPERT_PREFIX
    from prismaquant.production_weight_cache import _PackedExpertActivationCollector
    model = _RoutedModel().to(torch.bfloat16)
    module = model.model.layers[2].feed_forward.experts
    x = torch.arange(32,dtype=torch.bfloat16).reshape(8,4)/32
    indices = torch.arange(8).remainder(2).reshape(8,1)
    weights = torch.ones(8,1,dtype=torch.bfloat16)
    snapshots = []
    for callback in (None,lambda name,actual,args,kwargs:snapshots.append((name,actual,args,kwargs))):
        collector = _PackedExpertActivationCollector(model,{EXPERT_PREFIX},module_token_budget=3,
            store_device='cpu',boundary_consumer=callback)
        collector.install()
        try:
            if kwargs_route:
                module(x,indices=indices,weights=weights)
            else:
                module(x,indices,weights)
        finally:
            collector.remove()
        sampled = torch.cat(collector.activations[EXPERT_PREFIX])
        if callback is None:
            original = sampled
        else:
            assert torch.equal(original,sampled)
    name,actual,args,kwargs = snapshots[0]
    assert name == EXPERT_PREFIX and actual is module and args[0] is x
    assert (kwargs['indices'] if kwargs_route else args[1]) is indices
    assert (kwargs['weights'] if kwargs_route else args[2]) is weights
    assert not module._forward_pre_hooks


def test_campaign_boundary_coordinates_follow_batched_calibration_ids():
    from test_tessera_campaign_packed import _RoutedModel,EXPERT_PREFIX
    from prismaquant import tessera_campaign as tc
    class TokenModel(_RoutedModel):
        def forward(self,ids):
            hidden = torch.stack((ids.float()*2-1,ids.float(),ids.float()+1,ids.float()-1),dim=-1)
            return super().forward(hidden)
    model = TokenModel()
    tokens = [torch.tensor([[0,1],[1,0]]),torch.tensor([[0,1]])]
    captures = []
    values = tc._collect_activations(model,[EXPERT_PREFIX+'.0.w1',EXPERT_PREFIX+'.1.w1'],
        tokens,2,'cpu',want_hessian=True,
        boundary_consumer=lambda name,module,args,kwargs,coords:captures.append((args[0].clone(),coords.clone())))
    assert len(captures) == 2
    assert captures[0][1].tolist() == [[0,0],[0,1],[1,0],[1,1]]
    assert captures[1][1].tolist() == [[2,0],[2,1]]
    assert captures[0][0][:,0].tolist() == [-1,1,1,-1]
    assert all(count == 3 for count in values[2].values())


def test_layer_writer_releases_previous_resume_storage_before_next_load(capture, monkeypatch):
    from torch.multiprocessing.reductions import StorageWeakRef
    root, path, census, identity, acts, hessians, _record = capture
    writer = cc.CaptureWriter(root, census_path=path, identity=identity)
    original_load = torch.load
    storages, loaded = [], []

    def load(filename, *args, **kwargs):
        assert all(ref.expired() for ref in storages), 'previous resume entry still owns CPU storage'
        payload = original_load(filename, *args, **kwargs)
        storages.extend(StorageWeakRef(payload[key].untyped_storage())
                        for key in ('inputs', 'hessian'))
        loaded.append(Path(filename).name)
        return payload

    monkeypatch.setattr(torch, 'load', load)
    writer.write(acts=acts, hessians=hessians, counts=census['counts'], maxima=census['max_abs'])
    assert loaded == ['a.pt', 'b.pt']
    assert all(ref.expired() for ref in storages)
    assert set(writer.records) == {'a', 'b'}


def test_bounded_writer_advises_only_verified_durable_entries_and_guards_sealing(capture, monkeypatch):
    from prismaquant import perturbed_x_cache as cache
    root, path, census, identity, acts, hessians, _record = capture
    events = []
    def advise(filename, *, expected_stat):
        assert Path(filename).stat() == expected_stat
        events.append(('advice', Path(filename).name))
    monkeypatch.setattr(cache, 'release_activation_cache_file_pages', advise)
    writer = cc.CaptureWriter(root, census_path=path, identity=identity,
        release_file_pages=True, resource_check=lambda label: events.append(('check', label)))
    writer.write(acts=acts, hessians=hessians, counts=census['counts'], maxima=census['max_abs'])
    writer.finish(model_load_contract=identity['model_load_contract'])
    assert events == [
        ('check', 'before_capture_write:a'), ('advice', 'a.pt'), ('check', 'after_capture_write:a'),
        ('check', 'before_capture_write:b'), ('advice', 'b.pt'), ('check', 'after_capture_write:b'),
        ('check', 'before_capture_seal:a'), ('advice', 'a.pt'), ('check', 'after_capture_seal:a'),
        ('check', 'before_capture_seal:b'), ('advice', 'b.pt'), ('check', 'after_capture_seal:b')]
    events.clear()
    (root/'inputs/a.pt').write_bytes(b'corrupt')
    writer = cc.CaptureWriter(root, census_path=path, identity=identity, release_file_pages=True)
    with pytest.raises(RuntimeError, match='entry changed'):
        writer.write(acts=acts, hessians=hessians, counts=census['counts'], maxima=census['max_abs'])
    assert not events


def test_guarded_source_hash_matches_legacy_and_refuses_between_bounded_reads(tmp_path):
    path = tmp_path/'source.bin'
    path.write_bytes(b'a'*(17*1024**2))
    events = []
    assert cc.sha256(path, resource_check=events.append) == cc.sha256(path)
    assert sum(label.startswith('after_') for label in events) == 2
    def refuse(label):
        if label.startswith('after_'):
            raise RuntimeError('physical hash refusal')
    with pytest.raises(RuntimeError, match='physical hash refusal'):
        cc.sha256(path, resource_check=refuse)


def test_page_advice_fences_durability_and_refuses_changed_files(tmp_path, monkeypatch):
    import os
    from prismaquant.perturbed_x_cache import release_activation_cache_file_pages
    path = tmp_path/'entry.pt'
    path.write_bytes(b'complete entry')
    expected = path.stat()
    events = []
    original_fsync = os.fsync
    def fsync(fd):
        events.append('fsync')
        original_fsync(fd)
    monkeypatch.setattr(os, 'fsync', fsync)
    monkeypatch.setattr(os, 'posix_fadvise', lambda fd, start, size, mode:
                        events.append(('advice', start, size, mode)))
    release_activation_cache_file_pages(path, expected_stat=expected)
    assert events == ['fsync', ('advice', 0, 0, os.POSIX_FADV_DONTNEED)]
    assert path.read_bytes() == b'complete entry'
    events.clear()
    path.write_bytes(b'changed entry')
    with pytest.raises(RuntimeError, match='changed before page advice'):
        release_activation_cache_file_pages(path, expected_stat=expected)
    assert not events


def test_guarded_hash_advises_only_consumed_pages_and_keeps_the_content_digest(tmp_path, monkeypatch):
    import os
    path = tmp_path/'source.bin'
    path.write_bytes(b'z'*(17*1024**2+3))
    expected = cc.sha256(path)
    advice = []
    def advise(fd, start, size, mode):
        assert start+size <= os.lseek(fd, 0, os.SEEK_CUR)
        advice.append((start, size, mode))
    monkeypatch.setattr(os, 'posix_fadvise', advise)
    assert cc.sha256(path, release_read_pages=True) == expected
    assert advice == [(0, 16*1024**2, os.POSIX_FADV_DONTNEED),
                      (16*1024**2, 1024**2, os.POSIX_FADV_DONTNEED)]
    changed = []
    def change(label):
        if label.startswith('after_') and not changed:
            with path.open('r+b') as handle:
                handle.write(b'x')
            changed.append(True)
    with pytest.raises(RuntimeError, match='changed during guarded capture hashing'):
        cc.sha256(path, resource_check=change)
