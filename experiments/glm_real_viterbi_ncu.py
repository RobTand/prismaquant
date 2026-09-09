"""Nsight Compute attribution on a retained actual GLM residual, not a speed run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
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
        spec['container']['mounts'].append(dict(source='/opt/nvidia/nsight-compute/2025.3.1',target='/opt/ncu',readonly=True))
        Path(a.out).mkdir(parents=True,exist_ok=False)
        return launch(['--spec',json.dumps(spec),'--','/opt/ncu/ncu',
            '--profile-from-start','off','--clock-control','none','--cache-control','none',
            '--graph-profiling','node','--kernel-name','regex:.*_step_best.*',
            '--launch-skip','2048','--launch-count','4',
            '--section','SpeedOfLight','--section','MemoryWorkloadAnalysis',
            '--section','Occupancy','--section','LaunchStats',
            '--export',str(Path(a.out)/'actual-recurrence'),
            'python3','-u','-m','experiments.glm_real_viterbi_ncu',
            '--input',a.input,'--input-sha256',a.input_sha256,'--out',a.out])
    import torch
    from tessera import window_viterbi as w
    from tessera.cached_unit import encoder_source_sha256
    from experiments.glm_full_capture_profile import CaptureObserver
    assert digest(a.input)==a.input_sha256
    source=encoder_source_sha256()
    assert source=='bcad2ef2a7fdec2aab51b30d59f1f5e10b4933637ca4ffc74ee05e20816c1822'
    payload=torch.load(a.input,map_location='cpu',weights_only=True)
    out=Path(a.out)
    with CaptureObserver(out/'observer',profile_layers=()):
        inputs={k:payload[k].to('cuda') if payload[k] is not None else None for k in ('targets','vectors','weights')}
        def run():
            return w.viterbi_window_fused(inputs['targets'],inputs['vectors'],
                payload['window_bits'],payload['rate'],inputs['weights'],payload['chunk'])
        for _ in range(3):
            states,sse=run()
        reference=states.clone();reference_sse=sse
        torch.cuda.synchronize()
        started=time.time()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            states,sse=run()
            torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        assert torch.equal(reference,states) and reference_sse==sse
        result=dict(status='EXACT_PROFILE_CALL_COMPLETED',scope='four middle recurrence nodes; intrusive counter replay, no throughput or energy claim',
            input=a.input,input_sha256=a.input_sha256,producer_source_sha256=source,
            torch=torch.__version__,cuda=torch.version.cuda,shape=list(inputs['targets'].shape),
            window_bits=payload['window_bits'],rate=payload['rate'],chunk=payload['chunk'],
            best_form=os.environ.get('TESSERA_WINDOW_BEST_FORM'),tile=os.environ.get('TESSERA_WINDOW_BEST_TILE'),
            exact_states_and_sse=True,sse=sse,started_unix=started,finished_unix=time.time())
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
