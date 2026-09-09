"""Isolated exactness/throughput screen of the existing L14 kernel's tile geometry.

Seeded representative inputs and the actual producer E4M3 table; this is a
microkernel screen, never a GLM pricing or quality result. PB owns placement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container-spec');p.add_argument('--out',required=True)
    p.add_argument('--rate',type=int,choices=(3,4),required=True)
    p.add_argument('--cols',type=int,required=True)
    p.add_argument('--repeats',type=int,default=128)
    a=p.parse_args()
    if a.container_spec:
        from tools.tessera_campaign_container import main as launch
        spec=json.loads(a.container_spec)
        profile=Path(os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
        spec['container']['mounts'].append(dict(source=str(profile.parent),target='/pb-profile'))
        spec['env']['PRISMABUILD_PROFILE_TORCH_OUT']='/pb-profile/'+profile.name
        return launch(['--spec',json.dumps(spec),'--','python3','-u','-m',
            'experiments.glm_window_tile_pb_profile','--out',a.out,'--rate',str(a.rate),
            '--cols',str(a.cols),'--repeats',str(a.repeats)])
    import torch
    from tessera import window_viterbi as w
    from tessera.encode import window_table,grid_vector_table
    from tessera.alphabet import E4M3_GRID
    from tessera.export import DEFAULT_WINDOW_SIGMA
    from experiments.glm_full_capture_profile import CaptureObserver
    torch.manual_seed(0)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    codes=window_table(E4M3_GRID,14,sigma=DEFAULT_WINDOW_SIGMA,seed=0,device='cuda')
    vectors=grid_vector_table(E4M3_GRID).float().to('cuda')[codes.long()].contiguous()
    targets=torch.randn(4096,a.cols,device='cuda') * float(vectors.std())
    result=dict(schema='prismaquant.glm_window_tile_screen.v1',status='running',
        input_scope='seeded representative microkernel targets, actual pinned E4M3 table',
        seed=0,shape=list(targets.shape),L=14,rate=a.rate,repeats=a.repeats,
        torch=torch.__version__,cuda=torch.version.cuda,arms=[],started_unix=time.time(),
        targets_sha256=hashlib.sha256(targets.cpu().numpy().tobytes()).hexdigest(),
        vectors_sha256=hashlib.sha256(vectors.cpu().numpy().tobytes()).hexdigest())
    original=w._tile
    reference=None
    def save():
        tmp=out/'result.json.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'result.json')
    def run():return w.viterbi_window_fused(targets,vectors,14,a.rate)
    def geometry(mode):
        def tile(fan,low,n):
            bl,bc,warps=original(fan,low,n)
            if mode=='cols1_w4':return bl,1,4
            if mode=='cols1_w2':return bl,1,2
            if mode=='cols4_w4':return bl,min(4,n),4
            return bl,bc,warps
        return tile
    def arm(mode,label):
        nonlocal reference
        w.window_plan_cache_clear()
        w._tile=geometry(mode)
        # Clear is mandatory: plan-cache keys do not include this experimental
        # callable. All geometry changes and all JIT/capture work precede timing.
        for _ in range(3):states,sse=run()
        torch.cuda.synchronize()
        if reference is None:reference=(states.clone(),sse)
        assert torch.equal(states,reference[0]) and sse==reference[1], 'geometry changed exact result'
        width=w._l2_budget(targets.device)//(2*(1<<14)*4)
        record=dict(mode=mode,label=label,exact_states_and_sse=True,
            tile=list(w._tile(1<<a.rate,(1<<14)>>a.rate,width)),internal_width=width,
            started_unix=time.time())
        start=time.perf_counter()
        for _ in range(a.repeats):states,sse=run()
        torch.cuda.synchronize()
        record.update(seconds=time.perf_counter()-start,finished_unix=time.time())
        assert torch.equal(states,reference[0]) and sse==reference[1]
        # One full call in one fresh context. No repeated on/off toggling.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],record_shapes=False,
                profile_memory=False,with_stack=False) as prof:
            profile_start=time.time();states,sse=run();torch.cuda.synchronize();profile_end=time.time()
        assert torch.equal(states,reference[0]) and sse==reference[1]
        trace=out/(label+'.trace.json.gz');prof.export_chrome_trace(str(trace))
        with trace.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        record.update(profile_start_unix=profile_start,profile_end_unix=profile_end,
            trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha),
            calls_per_second=a.repeats/record['seconds'])
        if not result['arms']:shutil.copyfile(trace,os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
        result['arms'].append(record);save();print(json.dumps(record),flush=True)
    save()
    try:
        with CaptureObserver(out/'observer',profile_layers=()):
            for candidate in ('cols1_w4','cols1_w2','cols4_w4'):
                for index,mode in enumerate(('default',candidate,candidate,'default')):
                    arm(mode,f'{candidate}-{index}-{mode}')
        result['status']='complete'
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        w.window_plan_cache_clear();w._tile=original
        result['finished_unix']=time.time();save()
    print(json.dumps(result),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
