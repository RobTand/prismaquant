"""Check actual tile-screen CAS output, traces and bracketed power integrals."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

base=Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02')
fleet=Path('/mnt/shared/prismabuild-fleet')

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def energy(rows,start,end):
    assert rows[0][0]<=start<end<=rows[-1][0]
    joules=covered=0.;gaps=[]
    for (t0,p0),(t1,p1) in zip(rows,rows[1:]):
        lo,hi=max(start,t0),min(end,t1)
        if lo>=hi:continue
        gaps.append(t1-t0)
        a,b=p0+(p1-p0)*(lo-t0)/(t1-t0),p0+(p1-p0)*(hi-t0)/(t1-t0)
        joules+=(a+b)*.5*(hi-lo);covered+=hi-lo
    assert abs(covered-(end-start))<1e-5 and max(gaps)<1
    return dict(joules=joules,mean_w=joules/(end-start),max_gap_s=max(gaps))

reports=[]
for rate,key in ((3,'b633da0827937d70eea27563cf8ce15245cd871a604f2239a70a67b8b993abd4'),(4,'f6b880773d59bf18e6d0333b5775ac9921ae7026d5878be07fc042544664c25c')):
    root=base/f'performance-actual-r{rate}-tile32-ab-01'
    terminal=json.loads((fleet/'pb-queue/done'/f'{key}.json').read_text())
    assert terminal['detail']['returncode']==0 and terminal['resource_scope_cleanup']['complete']
    receipt=json.loads((fleet/'cas/actions/v3'/key[:2]/f'{key}.json').read_text())
    meta=receipt['result'];payload=fleet/'cas/blobs'/meta['sha256'][:2]/meta['sha256']
    assert digest(payload)==meta['sha256'] and payload.stat().st_size==meta['bytes']
    result=json.loads((root/'result.json').read_text())
    printed=[json.loads(line) for line in payload.read_text().splitlines() if line.startswith('{')]
    assert printed[-1]==result and result['status']=='complete'
    assert digest(Path(result['input']))==result['input_sha256']
    assert result['rate']==rate and result['producer_source_sha256']=='bcad2ef2a7fdec2aab51b30d59f1f5e10b4933637ca4ffc74ee05e20816c1822'
    blocks=result['blocks'];assert len(blocks)==12
    assert all(b['plans_built']==0 and b['exact_states_and_sse'] for b in blocks)
    observer=json.loads((root/'observer/result.json').read_text())
    assert observer['status']=='complete' and not observer['errors']
    nd=[json.loads(line) for line in (root/'observer/netdata.jsonl').read_text().splitlines()]
    power={}
    for source in json.loads((root/'root-pqteld-sources.json').read_text()):
        path=root/(source['host']+'-pqteld.csv');assert digest(path)==source['sha256']
        rows=list(csv.DictReader(path.open()));assert len(rows)==source['rows']
        power[source['host']]=[(float(r['epoch_ms'])/1000,float(r['power_draw_w'])) for r in rows if r['power_draw_w']]
    totals={tile:dict(seconds=0.,calls=0,joules=0.) for tile in ('64,4,2','32,4,2')}
    for b in blocks:
        host=terminal['claimed_host'];j=energy(power[host],b['started_unix'],b['finished_unix'])
        for other in power:
            energy(power[other],b['started_unix'],b['finished_unix'])
        t=totals[b['tile']];t['seconds']+=b['seconds'];t['calls']+=b['calls'];t['joules']+=j['joules']
    for tile,t in totals.items():
        t['seconds_per_call']=t['seconds']/t['calls'];t['joules_per_call']=t['joules']/t['calls'];t['mean_w']=t['joules']/t['seconds']
    profiles=[]
    for meta in result['profiles']:
        path=Path(meta['path']);assert digest(path)==meta['sha256'] and path.stat().st_size==meta['bytes']
        trace=json.loads(gzip.decompress(path.read_bytes()));events=trace['traceEvents']
        kernels=sorted([e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X'],key=lambda e:e['ts'])
        assert kernels and all(e['dur']>=0 for e in kernels)
        span=max(e['ts']+e['dur'] for e in kernels)-kernels[0]['ts']
        assert sum(e['dur'] for e in kernels)<=span*1.01
        steps=[e for e in kernels if e['name']=='_step_best'];assert len(steps)==4095
        overlap=[a['ts']+a['dur']-b['ts'] for a,b in zip(steps,steps[1:]) if a['ts']+a['dur']>b['ts']]
        assert max(overlap,default=0)<1 and sum(overlap)/sum(e['dur'] for e in steps)<.0001
        assert len([e for e in kernels if e['name']=='_traceback'])==1
        assert len([e for e in events if e.get('name')=='cudaGraphLaunch' and e.get('ph')=='X'])==1
        assert not any('StreamBeginCapture' in e.get('name','') for e in events)
        profiles.append(dict(tile=meta['tile'],sha256=meta['sha256'],steps=len(steps),
            step_us=sum(e['dur'] for e in steps),kernel_span_us=span,max_overlap_us=max(overlap,default=0)))
    assert terminal['detail']['profile']['blob_sha256']==result['profiles'][1]['sha256']
    hosts={host:len([v for v in nd if v['host']==host]) for host in power};assert all(hosts.values())
    a,b=totals['64,4,2'],totals['32,4,2']
    reports.append(dict(status='PASS',action=key,rate=rate,shape=result['shape'],host=terminal['claimed_host'],
        totals=totals,throughput_ratio=a['seconds_per_call']/b['seconds_per_call'],
        work_per_joule_ratio=a['joules_per_call']/b['joules_per_call'],profiles=profiles,netdata_samples=hosts,
        limitations=['Retained actual residual; no complete encode claim.','Profiles carry no wall brackets beyond native capture; durations and dependencies checked.',
            'Other host concurrently ran the other independent rate; both-host telemetry retained.']))
print(json.dumps(dict(status='PASS',reports=reports),sort_keys=True))
