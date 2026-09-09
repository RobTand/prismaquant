"""Two existing tiles on sealed actual GLM residuals, with warmed ABBA blocks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container-spec')
    p.add_argument('--input',required=True)
    p.add_argument('--input-sha256',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.container_spec:
        from tools.tessera_campaign_container import main as launch
        spec=json.loads(a.container_spec)
        profile=Path(os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
        spec['container']['mounts'].append(dict(source=str(profile.parent),target='/pb-profile'))
        spec['env']['PRISMABUILD_PROFILE_TORCH_OUT']='/pb-profile/'+profile.name
        return launch(['--spec',json.dumps(spec),'--','python3','-u','-m',
            'experiments.glm_actual_tile_screen','--input',a.input,
            '--input-sha256',a.input_sha256,'--out',a.out])
    import torch
    from tessera import window_viterbi as w
    from tessera.cached_unit import encoder_source_sha256
    from experiments.glm_full_capture_profile import CaptureObserver
    assert digest(a.input)==a.input_sha256
    assert encoder_source_sha256()=='bcad2ef2a7fdec2aab51b30d59f1f5e10b4933637ca4ffc74ee05e20816c1822'
    payload=torch.load(a.input,map_location='cpu',weights_only=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',schema='prismaquant.glm_actual_tile_screen.v1',
        started_unix=time.time(),input=a.input,input_sha256=a.input_sha256,
        producer_source_sha256=encoder_source_sha256(),torch=torch.__version__,
        cuda=torch.version.cuda,shape=list(payload['targets'].shape),rate=payload['rate'],
        scope='actual retained residual call; not complete GLM encode',blocks=[],profiles=[])
    original_plan=w._WindowPlan
    builds=[0]
    def plan(**kw):
        builds[0]+=1
        return original_plan(**kw)
    w._WindowPlan=plan
    def save():
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    reference=None
    try:
        with CaptureObserver(out/'observer',profile_layers=()):
            inputs={k:payload[k].to('cuda') if payload[k] is not None else None for k in ('targets','vectors','weights')}
            def run():
                return w.viterbi_window_fused(inputs['targets'],inputs['vectors'],payload['window_bits'],payload['rate'],inputs['weights'],payload['chunk'])
            for tile in ('64,4,2','32,4,2'):
                os.environ['TESSERA_WINDOW_BEST_TILE']=tile
                for _ in range(3):
                    states,sse=run()
                if reference is None:
                    reference=(states.clone(),sse)
                assert torch.equal(reference[0],states) and reference[1]==sse
            for cycle in range(3):
                for tile in ('64,4,2','32,4,2','32,4,2','64,4,2'):
                    os.environ['TESSERA_WINDOW_BEST_TILE']=tile
                    torch.cuda.synchronize()
                    before=builds[0]
                    started=time.time();clock=time.perf_counter();calls=0
                    while time.perf_counter()-clock<5:
                        states,sse=run();calls+=1
                    torch.cuda.synchronize()
                    seconds=time.perf_counter()-clock;finished=time.time()
                    assert builds[0]==before
                    assert torch.equal(reference[0],states) and reference[1]==sse
                    row=dict(cycle=cycle,tile=tile,calls=calls,started_unix=started,finished_unix=finished,
                        seconds=seconds,plans_built=0,exact_states_and_sse=True)
                    result['blocks'].append(row);print(json.dumps(row),flush=True);save()
            for tile in ('64,4,2','32,4,2'):
                os.environ['TESSERA_WINDOW_BEST_TILE']=tile
                before=builds[0]
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    states,sse=run();torch.cuda.synchronize()
                assert builds[0]==before and torch.equal(reference[0],states) and reference[1]==sse
                path=out/(tile.replace(',','x')+'.trace.json.gz')
                prof.export_chrome_trace(str(path))
                result['profiles'].append(dict(tile=tile,path=str(path),sha256=digest(path),
                    bytes=path.stat().st_size,plans_built=0,exact_states_and_sse=True))
                if tile=='32,4,2':
                    shutil.copyfile(path,os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
            result.update(status='complete',sse=reference[1])
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        w._WindowPlan=original_plan
        result['finished_unix']=time.time();save()
    print(json.dumps(result),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
