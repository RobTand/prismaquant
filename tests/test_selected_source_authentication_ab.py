"""CPU-only harness checks; execute through PrismaBuild."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from experiments import selected_source_authentication_ab as ab
from experiments.selected_source_authentication_ab_plan import resource_plan, GIB
from tests.test_selected_source_authentication import complete_source


@pytest.fixture
def pair_source(complete_source, tmp_path):
    from prismaquant import tessera_calibration_cache as cc
    fixture = complete_source
    name = 'model.layers.0.a'
    census = fixture.census
    census['unit_shapes'] = {name:[2,2]}
    census['counts'] = {name:2}
    census['max_abs'] = {name:3.}
    census['anchor_groups'] = {'u:'+name:[name]}
    census['expert_projection']['stacks'] = {'stack':{name:dict(source_tensor=name+'.weight',
        source_shape=[2,2], shape=[2,2], rows=2, cols=2, slice=None)}}
    path = Path(fixture.kwargs['census_path'])
    path.write_text(json.dumps(census))
    calibration = dict(fit_ids_sha256='draw',text_sha256='text',nsamples=2,seqlen=2,seed=0)
    canonical = cc.capture_identity(path,calibration=calibration,max_act_rows=2,
        model_load_contract=census['model_load_contract'],attention_implementation='eager')
    capture = cc.publish_capture(tmp_path/'ab-capture',census_path=path,identity=canonical,
        acts={name:torch.ones(2,2)},hessians={name:torch.eye(2)},counts=census['counts'],maxima=census['max_abs'])
    resources = dict(max_selected_bytes=16,legacy_post_hash_absolute_file_cap_bytes=4*GIB)
    inputs = dict(census=dict(path=str(path),sha256=ab.digest(path)),model=str(fixture.root),selected_units=[name])
    return SimpleNamespace(root=fixture.root,census=census,inputs=inputs,
        capture=capture,manifest=json.loads(Path(capture['path']).read_text()),resources=resources,name=name)


class FixtureCheck:
    def __call__(self,*args,**kwargs):
        return {}
    def require_file_cap(self,*args):
        return 0


def test_pair_real_owner_readers_full_identity_and_payload_parity(pair_source,tmp_path):
    from prismaquant import layer_streaming as ls
    f = pair_source
    observations,closed = [],[]
    def build(root,**kwargs):
        owner = kwargs['source_authentication']
        owned = {} if owner is None else dict(source_authentication=owner)
        shards,keys = ls._build_weight_map(root,**owned)
        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(2,2,bias=False)
        ls._materialize(model,['lm_head.'],shards,keys,torch.device('cpu'),torch.float32,**owned)
        class Runner:
            def snapshot_selected_weights(self,names,**kwargs):
                values = ls._read_layer_to_device('model.layers.0.',shards,keys,torch.float32,
                    torch.device('cpu'),**owned)
                return {n:values[n+'.weight'].clone() for n in names},dict(source_forward_count=0)
            def shutdown(self):
                closed.append(True)
        return Runner()
    def phase(name,fn,**kwargs):
        observations.append(name)
        return fn()
    recorder = ab.SourceEvents(f.root,tmp_path,1024**2)
    results = []
    with recorder.observe():
        for arm in ('full','selected'):
            recorder.arm = arm
            results.append(ab.prepare_arm(arm,inputs=f.inputs,census=f.census,
                capture_binding=f.capture,capture=f.manifest,resources=f.resources,
                output=tmp_path,check=FixtureCheck(),phase=phase,build=build,
                profile=None,device='cpu'))
    before,after = results
    assert torch.equal(before[0][f.name].view(torch.uint8),after[0][f.name].view(torch.uint8))
    assert before[1]['identity'] == after[1]['identity'] == f.manifest['identity']
    assert before[1]['projection'] == after[1]['projection']
    assert before[1]['closed'] and after[1]['closed'] and len(closed)==2
    assert after[1]['owned_descriptor_count_after_close']==0
    assert ab.digest(f.capture['path']) == f.capture['sha256']
    full = [r for r in recorder.rows if r['arm']=='full' and r['name'].endswith('.safetensors')]
    selected = [r for r in recorder.rows if r['arm']=='selected' and r['name'].endswith('.safetensors')]
    assert sorted(r['name'] for r in full)==['head.safetensors','selected.safetensors','unused.safetensors']
    assert sorted(r['name'] for r in selected)==['head.safetensors','selected.safetensors']
    assert all(r['held_fd'] is not None for r in selected)
    rows = [json.loads(line) for line in (tmp_path/'source-events.jsonl').read_text().splitlines()]
    consumed=[r for r in rows if r['arm']=='selected' and r['operation'].startswith('payload_')]
    assert consumed and all(r['held_fd_alias'] for r in consumed)
    assert observations[:3]==['source_build','authentication','snapshot']
    assert observations[5:8]==['selected_authentication','source_build','authentication']


def test_pair_teardown_when_full_identity_refuses(pair_source,tmp_path,monkeypatch):
    from prismaquant import tessera_calibration_cache as cc
    f=pair_source
    closed=[]
    def build(*args,**kwargs):
        return SimpleNamespace(shutdown=lambda:closed.append(True))
    monkeypatch.setattr(cc,'capture_identity',lambda *a,**k:{})
    with pytest.raises(RuntimeError,match='canonical full identity'):
        ab.prepare_arm('full',inputs=f.inputs,census=f.census,capture_binding=f.capture,
            capture=f.manifest,resources=f.resources,output=tmp_path,check=FixtureCheck(),
            phase=lambda name,fn,**kw:fn(),build=build,profile=None,device='cpu')
    assert closed==[True]


def test_draft_refuses_before_torch_or_source(tmp_path,monkeypatch):
    monkeypatch.setattr(ab,'digest',lambda *a:pytest.fail('read source before frozen status'))
    with pytest.raises(ValueError,match='FROZEN'):
        ab.preflight(dict(schema=ab.SCHEMA,status='AWAIT_CAPTURE'),repository=tmp_path)


def test_frozen_behavior_roster_required(tmp_path):
    with pytest.raises(ValueError,match='behavior roster'):
        ab.preflight(dict(schema=ab.SCHEMA,status='FROZEN'),repository=tmp_path)


def test_hash_file_charge_enforced_inside_hash(tmp_path):
    scope=tmp_path/'scope'; scope.mkdir()
    (scope/'memory.stat').write_text('file 12\nanon 3\n')
    calls=[]
    guard=SimpleNamespace(scope=scope,check=lambda label,**kw:calls.append((label,kw)) or {})
    resources=dict(total_deadline_seconds=60,requested_gpu_mem_gib=1,
        legacy_hash_absolute_file_cap_bytes=10,hash_read_buffer_bytes=16)
    check=ab.ResourceCheck(guard,resources,SimpleNamespace(require_healthy=lambda:None),
        cuda=SimpleNamespace(memory_reserved=lambda:0))
    check.arm,check.phase='full','authentication'
    with pytest.raises(RuntimeError,match='page residue'):
        check('before_capture_hash:shard.safetensors')
    assert calls[-1][1]['reserve_bytes']==16
    (scope/'memory.stat').write_text('file 9\nanon 3\n')
    check('after_capture_hash:shard.safetensors')
    check.require_file_cap(9)
    with pytest.raises(RuntimeError,match='page residue'):
        check.require_file_cap(8)


def test_deadline_and_gpu_caps(tmp_path):
    resources=dict(total_deadline_seconds=60,requested_gpu_mem_gib=1)
    check=ab.ResourceCheck(SimpleNamespace(check=lambda *a,**k:{}),resources,
        SimpleNamespace(require_healthy=lambda:None),cuda=SimpleNamespace(memory_reserved=lambda:2*GIB))
    with pytest.raises(RuntimeError,match='CUDA reservation'):
        check('checkpoint')
    check.deadline=0
    with pytest.raises(RuntimeError,match='hard deadline'):
        check('checkpoint')


def test_resource_envelope_keeps_full_unconditional_failure():
    prep=dict(selected_source_weight_bytes=20,nonbody_source_bytes=30,source_window_bytes=40,
        loader_transient_bytes=50,declared_headroom_bytes=24*GIB)
    plan=resource_plan(dict(schema='prismaquant.selected_anchor_resources.v2',
        selected_source_weight_bytes=20,phases=dict(source_preparation=prep)),
        {'a':[2,5]},full_source_bytes=600*GIB,source_file_count=120)
    assert plan['legacy_unconditional_fits_104gib'] is False
    assert plan['physical_phases']['source_preparation']['reference_gpu_bytes']==20
    assert plan['gpu_phases']['source_preparation']['reference_gpu_bytes']==20
    assert plan['order']==['full','selected'] and not plan['automatic_followup']
    assert plan['guard_envelope_bytes']==max(plan['guard_phase_envelopes'].values())
    assert plan['legacy_hash_absolute_file_cap_bytes']>plan['legacy_post_hash_absolute_file_cap_bytes']


def frozen_fixture(f,tmp_path,monkeypatch):
    from prismaquant import tessera_calibration_cache as cc
    names=['model.layers.0.'+name for name in ('a','b','c','d')]
    census=dict(f.census,nsamples=512,seqlen=512,
        unit_shapes={n:[2,2] for n in names},counts={n:2 for n in names},
        max_abs={n:3. for n in names},anchor_groups={'u:'+n:[n] for n in names})
    census_path=tmp_path/'frozen-census.json'
    ids=torch.arange(512*512,dtype=torch.int64).reshape(512,512)
    calibration=dict(fit_ids_sha256=hashlib.sha256(ids.to(torch.int32).numpy().tobytes()).hexdigest(),
        text_sha256='a'*64,nsamples=512,seqlen=512,seed=0,fit_tokens=ids.numel())
    census.update({k:calibration[k] for k in ('fit_ids_sha256','text_sha256')})
    census_path.write_text(json.dumps(census))
    monkeypatch.setattr(ab,'SELECTED_UNITS',tuple(names))
    canonical=cc.capture_identity(census_path,calibration=calibration,max_act_rows=512,
        model_load_contract=census['model_load_contract'],attention_implementation='eager')
    capture=cc.publish_capture(tmp_path/'frozen-capture',census_path=census_path,identity=canonical,
        acts={n:torch.ones(2,2) for n in names},hessians={n:torch.eye(2) for n in names},
        counts=census['counts'],maxima=census['max_abs'])
    tokens_path=tmp_path/'tokens.safetensors'
    save_file({'calibration_ids':ids},str(tokens_path),metadata={'calibration_provenance':json.dumps(calibration)})
    inputs=dict(f.inputs,census=dict(path=str(census_path),sha256=ab.digest(census_path)),
        selected_units=names,calibration=calibration,
        calibration_input=dict(path=str(tokens_path),sha256=ab.digest(tokens_path)),
        container=dict(image='fixture',content_sha256='b'*64))
    resources=dict(schema='prismaquant.selected_source_authentication_ab.resources.v1',
        fits_104gib_conditional_envelope=True,requested_mem_gib=104,order=['full','selected'],
        automatic_followup=False,inputs=inputs,selected_shapes=census['unit_shapes'])
    path=tmp_path/'resources.json';path.write_text(json.dumps(resources))
    repository=Path(ab.__file__).resolve().parents[1]
    plan=dict(schema=ab.SCHEMA,status='FROZEN',source_files={n:ab.digest(repository/n) for n in ab.SOURCE_FILES},
        environment=ab.ENVIRONMENT,container=inputs['container'],capture=capture,
        resources=dict(path=str(path),sha256=ab.digest(path)))
    for k,v in ab.ENVIRONMENT.items():
        monkeypatch.setenv(k,v)
    return plan,repository,resources


def test_preflight_exact_tokens_complete_capture_no_payload_or_cuda(pair_source,tmp_path,monkeypatch):
    from prismaquant import tessera_calibration_cache as cc
    plan,repository,_=frozen_fixture(pair_source,tmp_path,monkeypatch)
    old=cc.sha256
    def guard(path,**kwargs):
        assert not str(path).endswith('.safetensors'), 'preflight read source payload'
        return old(path,**kwargs)
    monkeypatch.setattr(cc,'sha256',guard)
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('preflight initialized CUDA'))
    resources,census,capture,tokens=ab.preflight(plan,repository=repository)
    assert capture['identity']['units']==census['unit_shapes']
    assert tokens['shape']==[512,512]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mutation',['missing_capture','partial_capture','token_bytes','calibration_seed'])
def test_preflight_original_input_refusal(pair_source,tmp_path,monkeypatch,mutation):
    plan,repository,resources=frozen_fixture(pair_source,tmp_path,monkeypatch)
    if mutation=='missing_capture':
        plan['capture']['path']=str(tmp_path/'absent.json')
    elif mutation=='partial_capture':
        path=Path(plan['capture']['path']);value=json.loads(path.read_text());value['status']='partial'
        path.write_text(json.dumps(value));plan['capture']['sha256']=ab.digest(path)
    elif mutation=='token_bytes':
        path=Path(resources['inputs']['calibration_input']['path'])
        with path.open('ab') as stream:
            stream.write(b'changed')
    else:
        resources['inputs']['calibration']['seed']=1
        path=Path(plan['resources']['path']);path.write_text(json.dumps(resources))
        plan['resources']['sha256']=ab.digest(path)
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('refusal initialized CUDA'))
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):
        ab.preflight(plan,repository=repository)


def test_netdata_requires_both_hosts_bracketed_without_gaps():
    from experiments.glm_native_wire_screen_evidence import telemetry_coverage
    def samples(times):
        return {host:[dict(time=t,monotonic=t,oldest_chart_age_seconds=1,
            max_future_chart_seconds=0) for t in times] for host in ('sparky','sparklina')}
    phases=[dict(phase='full-authentication',started_monotonic=1,finished_monotonic=39)]
    assert telemetry_coverage(samples([0,10,20,30,40]),phases,[],thread_complete=True)['passed']
    assert not telemetry_coverage(samples([0,40]),phases,[],thread_complete=True)['passed']
    assert not telemetry_coverage(samples([10,20,30,40]),phases,[],thread_complete=True)['passed']
    assert not telemetry_coverage(samples([0,10,20,30,40]),phases,[{'error':'offline'}],thread_complete=True)['passed']
    missing=samples([0,10,20,30,40]);missing['sparklina']=[]
    assert not telemetry_coverage(missing,phases,[],thread_complete=True)['passed']


def test_source_event_cap_refuses_instead_of_silent_truncation(tmp_path):
    recorder=ab.SourceEvents(tmp_path,tmp_path,1)
    with recorder.observe(),pytest.raises(RuntimeError,match='output exceeded'):
        recorder.record(dict(operation='header_open',name='a'))
