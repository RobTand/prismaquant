"""Bounded full-CLI publication A/B on an unchanged selected resident row.

The only schedule intervention truncates the existing first-round batch list.
Ordinary scoring, publication, receipt creation, journal updates and termination
run through campaign.main. These partial artifacts are measurement evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import threading
import time


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def set_option(command, name, value):
    if name in command:
        command[command.index(name)+1] = str(value)
    else:
        command.extend((name, str(value)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--container-spec')
    ap.add_argument('--out', required=True)
    ap.add_argument('--units', type=int, default=64)
    ap.add_argument('--overlap-bytes', type=int, default=512*1024**2)
    ap.add_argument('campaign', nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = args.campaign[1:] if args.campaign[:1] == ['--'] else args.campaign
    if args.container_spec:
        from tools.tessera_campaign_container import main as container_main
        spec = json.loads(args.container_spec)
        profile = Path(os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
        spec['container']['mounts'].append(dict(source=str(profile.parent), target='/pb-profile'))
        spec['env']['PRISMABUILD_PROFILE_TORCH_OUT'] = '/pb-profile/'+profile.name
        return container_main(['--spec', json.dumps(spec), '--', 'python3', '-u', '-m',
            'experiments.glm_publication_cycle_profile', '--out', args.out,
            '--units', str(args.units), '--overlap-bytes', str(args.overlap_bytes), '--', *command])
    import torch
    from prismaquant import tessera_campaign as campaign
    from prismaquant import cost_stage_checkpoint as checkpoint
    from prismaquant import production_weight_cache as pwc
    from prismaquant import tessera_render as render_adapter
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path
    from tessera import window_viterbi
    from tessera.cached_unit import encoder_source_sha256
    from experiments.glm_full_capture_profile import CaptureObserver
    width = int(command[command.index('--anchor-batch-size')+1])
    if args.units < 4*width or args.units % width or args.overlap_bytes <= 0:
        raise ValueError('require at least four complete batches and a positive publication budget')
    if '--calibration-cache' not in command or '--calibration-cache-sha256' not in command:
        raise ValueError('only reuse of the existing sealed capture is admitted')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='prismaquant.glm_publication_cycle_profile.v1', status='running',
        started_unix=time.time(), producer_source_sha256=encoder_source_sha256(),
        torch=torch.__version__, cuda=torch.version.cuda, command=command, units=args.units,
        batch_size=width, arms=[], full_campaign_complete=False,
        scope='bounded first-round batch prefix; full original selected-row residency')
    reference = None
    original_batches = campaign._anchor_batches
    original_viterbi = window_viterbi.viterbi_window_fused
    original_plan = window_viterbi._WindowPlan
    plans = [0]
    def counted_plan(**kwargs):
        plans[0] += 1
        return original_plan(**kwargs)
    window_viterbi._WindowPlan = counted_plan
    def save():
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    try:
        with CaptureObserver(out/'observer', profile_layers=()):
            for index, enabled in enumerate((False, True, False, True, True, False)):
                label = ('warm' if index < 2 else 'measured')+f'-{index}-'+('async' if enabled else 'sync')
                root = out/label
                root.mkdir()
                run = list(command)
                for key, value in (('--out',root/'cost.pkl'), ('--cache-dir',root/'cache'),
                        ('--checkpoint',root/'cost.anchors.json'), ('--max-rounds',1),
                        ('--publication-overlap-bytes',args.overlap_bytes if enabled else 0)):
                    set_option(run,key,value)
                record = dict(label=label, overlap_bytes=args.overlap_bytes if enabled else 0,
                    started_unix=time.time(), io_calls=[], scheduled=[], profiles=[])
                result['arms'].append(record)
                cycle_start = [None]
                builds_before = [None]
                calls = [0]
                def bounded_batches(*pos, **kw):
                    if cycle_start[0] is not None:
                        raise RuntimeError('bounded measurement unexpectedly entered another round')
                    batches = original_batches(*pos, **kw)[:args.units//width]
                    if len(batches)*width != args.units or any(len(b)!=width for b in batches):
                        raise RuntimeError('requested complete batch prefix unavailable')
                    flat=[item for batch in batches for item in batch]
                    if len({(family,rate) for _,family,rate in flat}) != 1:
                        raise RuntimeError('prefix must have one common family and rung')
                    record['scheduled']=[list(item) for item in flat]
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    record['cycle_started_unix']=time.time()
                    builds_before[0]=plans[0]
                    cycle_start[0]=time.perf_counter()
                    return batches
                campaign._anchor_batches = bounded_batches
                wrappers=[]
                def instrument(owner,name):
                    original=getattr(owner,name)
                    def timed(*pos, **kw):
                        active=cycle_start[0] is not None
                        start=time.perf_counter()
                        begun=time.time()
                        try:
                            return original(*pos,**kw)
                        finally:
                            if active:
                                record['io_calls'].append(dict(kind=name,started_unix=begun,
                                    finished_unix=time.time(),seconds=time.perf_counter()-start,
                                    thread=threading.current_thread().name,
                                    thread_id=threading.get_ident()))
                    setattr(owner,name,timed)
                    wrappers.append((owner,name,original))
                for owner,name in ((torch,'save'),(Path,'write_bytes'),(os,'fsync'),
                        (campaign,'_checkpoint_wire_record'),(checkpoint,'write_unit'),
                        (checkpoint,'atomic_write_bytes'),(pickle,'dump'),
                        (campaign,'_prepare_anchor'),(campaign,'_finish_anchor'),
                        (render_adapter,'encode_tessera_units'),
                        (pwc,'_canonical_rendered_weight_tensor'),
                        (pwc,'_local_forward_render_score')):
                    if not hasattr(owner,name):
                        raise RuntimeError(f'publication call site changed: {name}; review instrumentation')
                    instrument(owner,name)
                def profile_first(*pos,**kw):
                    calls[0]+=1
                    if index>=2 or calls[0]!=1:
                        return original_viterbi(*pos,**kw)
                    for _ in range(3):
                        original_viterbi(*pos,**kw)
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        answer=original_viterbi(*pos,**kw)
                        torch.cuda.synchronize()
                    path=root/'complete-first-call.trace.json.gz'
                    prof.export_chrome_trace(str(path))
                    record['profiles'].append(dict(path=str(path),sha256=digest(path),
                        bytes=path.stat().st_size,targets_shape=list(pos[0].shape),
                        window_bits=pos[2],rate=pos[3],sse=answer[1]))
                    if enabled:
                        import shutil
                        shutil.copyfile(path,os.environ['PRISMABUILD_PROFILE_TORCH_OUT'])
                    return answer
                window_viterbi.viterbi_window_fused=profile_first
                try:
                    rc=campaign.main(run)
                    torch.cuda.synchronize()
                    if cycle_start[0] is None or rc!=0:
                        raise RuntimeError(f'bounded original campaign failed: {rc}')
                    record.update(returncode=rc,cycle_seconds=time.perf_counter()-cycle_start[0],
                        cycle_finished_unix=time.time(),plans_built=plans[0]-builds_before[0],
                        allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                        reserved_peak_bytes=torch.cuda.max_memory_reserved())
                finally:
                    window_viterbi.viterbi_window_fused=original_viterbi
                    for owner,name,original in reversed(wrappers):
                        setattr(owner,name,original)
                if index>=2 and record['plans_built']:
                    raise RuntimeError('timed cycle constructed a plan')
                with (root/'cost.pkl').open('rb') as handle:
                    cost_payload=pickle.load(handle)
                record['publication_stats']=cost_payload['provenance']['publication_overlap']
                if enabled and (not record['publication_stats'] or record['publication_stats']['failed']):
                    raise RuntimeError('enabled publication did not report successful completion')
                if any(t.name.startswith('tessera-publication') for t in threading.enumerate()):
                    raise RuntimeError('publication thread survived the campaign return')
                manifest=json.loads((root/'cost.anchors.json').read_text())
                signatures=[]
                for name,_,_ in record['scheduled']:
                    state=_load_unit(unit_path(root/'cost.anchors.json.parts',name),
                        stage=manifest['stage'],qname=name,identity_sha256=manifest['identity_sha256'])
                    if len(state['anchors'])!=1 or len(state['wire_records'])!=1:
                        raise RuntimeError('missing or unexpected journalled anchors')
                    anchor=state['anchors'][0];wire=state['wire_records'][anchor['format_name']]
                    path=root/'cache/wire'/wire['file']
                    if digest(path)!=wire['blob_sha256'] or path.stat().st_size!=wire['blob_bytes']:
                        raise RuntimeError('journal receipt does not match actual wire')
                    signatures.append(dict(qname=name,format=anchor['format_name'],
                        dloss=anchor['dloss'],wire_sha256=wire['blob_sha256']))
                if reference is None:
                    reference=signatures
                if signatures!=reference:
                    raise RuntimeError('publication mode changed actual wire or score')
                record.update(signatures=signatures,exact_parity=True,finished_unix=time.time())
                print(json.dumps(record),flush=True)
                save()
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        campaign._anchor_batches=original_batches
        window_viterbi.viterbi_window_fused=original_viterbi
        window_viterbi._WindowPlan=original_plan
        result['finished_unix']=time.time()
        save()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
