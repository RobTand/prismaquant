"""Audit exact microkernel outputs, fresh CUDA traces and measured energy."""
import collections,csv,gzip,hashlib,json,statistics
from pathlib import Path
BASE=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/window-tile-screen-01')
CAS=Path('/mnt/shared/prismabuild-fleet/cas')
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def energy(rows,start,end):
    assert rows[0][0]<=start<end<=rows[-1][0]
    joules=0;gaps=[]
    for (t0,p0),(t1,p1) in zip(rows,rows[1:]):
        lo,hi=max(t0,start),min(t1,end)
        if lo>=hi:continue
        gaps.append(t1-t0)
        left=p0+(p1-p0)*(lo-t0)/(t1-t0);right=p0+(p1-p0)*(hi-t0)/(t1-t0)
        joules+=(left+right)/2*(hi-lo)
    assert gaps and max(gaps)<2
    return dict(estimated_joules=joules,mean_w=joules/(end-start),max_sample_gap_seconds=max(gaps))
recorders={}
for host in ('sparky','sparklina'):
    recorders[host]=[(float(r['epoch_ms'])/1000,float(r['power_draw_w'])) for r in csv.DictReader((BASE/(host+'-pqteld.csv')).open()) if r['power_draw_w']]
reports=[]
for rate,key in [(3,'6c56f4de7979e2f9af1822d10af70604dd3b2c8a8649b9abf034f957b60b146c'),(4,'93f3669a34f04ca170d36f83b0b3becaec3856e11ca9447aec0d60bc2c11bbed')]:
    terminal=json.loads((Path('/mnt/shared/prismabuild-fleet/pb-queue/done')/(key+'.json')).read_text())
    assert terminal['detail']['returncode']==0 and terminal['resource_scope_cleanup']['complete']
    receipt=json.loads((CAS/'actions/v3'/key[:2]/(key+'.json')).read_text())
    for blob in [receipt['result'],terminal['checkout_snapshot']['input']]:
        p=CAS/'blobs'/blob['sha256'][:2]/blob['sha256'];assert p.stat().st_size==blob['bytes'] and digest(p)==blob['sha256']
    host=terminal['claimed_host'];directory=BASE/f'r{rate}'
    result=json.loads((directory/'result.json').read_text());observer=json.loads((directory/'observer/result.json').read_text())
    assert result['status']==observer['status']=='complete' and not observer['errors']
    nd=[json.loads(x) for x in (directory/'observer/netdata.jsonl').read_text().splitlines()]
    arms=[]
    for a in result['arms']:
        assert a['exact_states_and_sse']
        t=a['trace'];p=Path(t['path']);assert digest(p)==t['sha256'] and p.stat().st_size==t['bytes']
        with gzip.open(p,'rb') as f:trace=json.load(f)
        kernels=sorted((e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X'),key=lambda e:e['ts'])
        steps=[e for e in kernels if e['name']=='_step'];assert len(steps)==4096*(result['shape'][1]//a['internal_width'])
        epoch=trace['baseTimeNanoseconds']/1e9
        assert min(e['dur'] for e in steps)>=0 and max(e['dur'] for e in steps)<1000
        overlaps=[a0['ts']+a0['dur']-b['ts'] for a0,b in zip(steps,steps[1:]) if b['ts']<a0['ts']+a0['dur']]
        overlap_us=sum(overlaps);overlap_fraction=overlap_us/sum(e['dur'] for e in steps)
        # Retain submicrosecond CUPTI timestamp anomalies (observed 1 pair
        # out of24576, <=0.256us). Their total is bounded to0.01% of kernel
        # time; this refuses the seconds-long repeated-toggle corruption.
        assert max(overlaps,default=0)<1 and overlap_fraction<0.0001
        assert all(a['profile_start_unix']-0.2<=epoch+e['ts']/1e6<=epoch+(e['ts']+e['dur'])/1e6<=a['profile_end_unix']+0.2 for e in kernels)
        span=max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels)
        assert sum(e['dur'] for e in kernels)<=span*1.01
        power=energy(recorders[host],a['started_unix'],a['finished_unix'])
        row=dict(a,power=power,step_count=len(steps),step_mean_us=statistics.mean(e['dur'] for e in steps),
            step_max_us=max(e['dur'] for e in steps),kernel_span_us=span,
            summed_kernel_us=sum(e['dur'] for e in kernels),trace_timing_valid=True,
            trace_overlap_count=len(overlaps),trace_overlap_total_us=overlap_us,
            trace_overlap_max_us=max(overlaps,default=0),trace_overlap_fraction=overlap_fraction,
            example_kernel_args=steps[0].get('args'),calls_per_joule=result['repeats']/power['estimated_joules'])
        arms.append(row)
        del trace,kernels,steps
    comparisons=[]
    for mode in ('cols1_w4','cols1_w2','cols4_w4'):
        group=[a for a in arms if a['label'].startswith(mode)]
        before=[a for a in group if a['mode']=='default'];after=[a for a in group if a['mode']==mode]
        def mean(rs,k):return statistics.mean(r[k] for r in rs)
        oldj=statistics.mean(a['power']['estimated_joules'] for a in before);newj=statistics.mean(a['power']['estimated_joules'] for a in after)
        comparisons.append(dict(candidate=mode,throughput_ratio=mean(before,'seconds')/mean(after,'seconds'),work_per_joule_ratio=oldj/newj,
            before_seconds=mean(before,'seconds'),after_seconds=mean(after,'seconds'),before_joules=oldj,after_joules=newj,
            before_mean_w=statistics.mean(a['power']['mean_w'] for a in before),after_mean_w=statistics.mean(a['power']['mean_w'] for a in after),
            before_step_us=mean(before,'step_mean_us'),after_step_us=mean(after,'step_mean_us')))
    reports.append(dict(rate=rate,action=key,host=host,status='PASS',shape=result['shape'],input_scope=result['input_scope'],
        targets_sha256=result['targets_sha256'],vectors_sha256=result['vectors_sha256'],
        receipt_sha256=receipt['receipt_sha256'],arms=arms,comparisons=comparisons,
        netdata_records={h:sum(x['host']==h for x in nd) for h in recorders},
        artifacts={str(p):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in [directory/'result.json',directory/'observer/result.json',directory/'observer/netdata.jsonl',directory/'observer/python_sampler.jsonl']}))
print(json.dumps(dict(status='PASS',scope='representative microkernel screen, not full GLM pricing or a shipping qualification',reports=reports,
    recorders={str(BASE/(h+'-pqteld.csv')):dict(sha256=digest(BASE/(h+'-pqteld.csv'))) for h in recorders}),sort_keys=True))
