"""Audit upper-endpoint actual pricing, complete-call profiles and timed I/O."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from prismaquant.cost_stage_checkpoint import _load_unit, canonical_json_sha256, unit_path

base = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/performance-best-form-r1088-ab-01')
fleet = Path('/mnt/shared/prismabuild-fleet')
key = '3a8fa1c2cc0513629555d349b9689086fdb6c91dbb761391481a4ae27d83924e'

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

terminal = json.loads((fleet/'pb-queue/done'/f'{key}.json').read_text())
assert terminal['detail']['returncode'] == 0 and terminal['resource_scope_cleanup']['complete']
receipt = json.loads((fleet/'cas/actions/v3'/key[:2]/f'{key}.json').read_text())
blob = receipt['result']
path = fleet/'cas/blobs'/blob['sha256'][:2]/blob['sha256']
assert path.stat().st_size == blob['bytes'] and digest(path) == blob['sha256']
printed = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
result = json.loads((base/'evidence/result.json').read_text())
observer = json.loads((base/'evidence/observer/result.json').read_text())
assert result['status'] == observer['status'] == 'complete' and not observer['errors']
assert result['comparison'] == 'trellis-best-form-complete'
assert result['producer_source_sha256'] == 'e6fe414581b7c51f76c0c3f75fd461365136949339b5082d8fe4de189034f8d4'
assert result['format_name'] == 'TESSERA_E4M3_K1_R1088'
assert len(result['arms']) == 6
signatures = result['arms'][0]['signatures']
assert len(signatures) == 16
assert all(a['signatures'] == signatures and a['exact_parity'] and a['batch_size']==8 for a in result['arms'])
assert result['arms'] == [p for p in printed if p.get('label','').startswith(('warm-','measured-'))]
for s in signatures:
    p = base/'cache/wire'/(s['qname'].replace('.','__')+'__'+result['format_name']+'.tessera')
    assert p.stat().st_size == s['wire_bytes'] and digest(p) == s['wire_sha256']
load = json.loads(next((base/'cache').glob('capture-load-execution-*.json')).read_text())
assert load['prefetch']['loaded_entries'] == 864 and load['prefetch']['live_buffer_bytes'] == 0
assert load['resources']['source_forward_count'] == 0
assert load['capture']['sha256'] == 'f4bcbf408d3aa81b04c1fabd1d2d7457176a95dcccdd5ed368800de5c37e277c'

# Earlier partial pricing is evidence, not seed intake under the new source.
old = base.parent/'workspace/rows/row-0076'
journal = json.loads((old/'cost.anchors.json').read_text())
assert canonical_json_sha256(journal['identity'],where='R1088 old evidence') == journal['identity_sha256']
old_parity = []
for s in signatures:
    p = unit_path(old/'cost.anchors.json.parts',s['qname'])
    if not p.exists():
        continue
    state = _load_unit(p,stage=journal['stage'],qname=s['qname'],identity_sha256=journal['identity_sha256'])
    matches = [a for a in state['anchors'] if a['format_name']==result['format_name']]
    if not matches:
        continue
    assert len(matches)==1
    record = state['wire_records'][result['format_name']]
    wire = old/'cache/wire'/record['file']
    assert digest(wire)==record['blob_sha256'] and wire.stat().st_size==record['blob_bytes']
    assert record['blob_sha256']==s['wire_sha256'] and matches[0]['dloss']==s['dloss']
    old_parity.append(s['qname'])

profiles = []
first,second = result['call_profiles']
assert first['call']==second['call'] and first['states_sha256']==second['states_sha256'] and first['sse']==second['sse']
assert first['call']['rate']==4 and first['call']['inputs']['targets']['shape']==[4096,192]
for arm in result['call_profiles']:
    meta = arm['trace']; p = Path(meta['path'])
    assert p.stat().st_size==meta['bytes'] and digest(p)==meta['sha256']
    trace = json.loads(gzip.decompress(p.read_bytes())); epoch = trace['baseTimeNanoseconds']/1e9
    events = trace['traceEvents']
    kernels = sorted((e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X'),key=lambda e:e['ts'])
    assert kernels and all(e['dur']>=0 and arm['started_unix']<=epoch+e['ts']/1e6<=epoch+(e['ts']+e['dur'])/1e6<=arm['finished_unix'] for e in kernels)
    span = max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels)
    assert sum(e['dur'] for e in kernels)<=span*1.01
    best = arm['best_form']; steps = [e for e in kernels if e['name']==('_step_best' if best else '_step')]
    assert len(steps)==(4095 if best else 24576)
    overlaps = [a['ts']+a['dur']-b['ts'] for a,b in zip(steps,steps[1:]) if a['ts']+a['dur']>b['ts']]
    assert max(overlaps,default=0)<1 and sum(overlaps)/sum(e['dur'] for e in steps)<0.0001
    tb = [e for e in kernels if e['name']=='_traceback']; assert len(tb)==1
    launches = [e for e in events if e.get('name')=='cudaGraphLaunch' and e.get('ph')=='X']
    assert len(launches)==(1 if best else 6)
    assert not any('StreamBeginCapture' in e.get('name','') for e in events)
    profiles.append(dict(label=arm['label'],trace=meta,step_count=len(steps),step_sum_us=sum(e['dur'] for e in steps),
        step_mean_us=statistics.mean(e['dur'] for e in steps),kernel_span_us=span,traceback_us=tb[0]['dur'],
        overlap_max_us=max(overlaps,default=0),graph_launches=len(launches)))
assert terminal['detail']['profile']['blob_sha256']==second['trace']['sha256']
inputs = result['call_inputs'];p=Path(inputs['path'])
assert p.stat().st_size==inputs['bytes'] and digest(p)==inputs['sha256']

def energy(rows,start,end):
    assert rows[0][0]<=start<end<=rows[-1][0]
    joules = covered = 0.0; gaps=[]
    for (t0,p0),(t1,p1) in zip(rows,rows[1:]):
        lo,hi=max(start,t0),min(end,t1)
        if lo>=hi:continue
        gaps.append(t1-t0)
        a,b=p0+(p1-p0)*(lo-t0)/(t1-t0),p0+(p1-p0)*(hi-t0)/(t1-t0)
        joules+=(a+b)*0.5*(hi-lo);covered+=hi-lo
    assert abs(covered-(end-start))<1e-5 and max(gaps)<1
    return dict(joules=joules,mean_w=joules/(end-start),max_gap_s=max(gaps),peak_w=max(p for t,p in rows if start<=t<=end))

power = {}
for source in json.loads((base/'root-pqteld-sources.json').read_text()):
    p=base/(source['host']+'-pqteld.csv');assert digest(p)==source['sha256']
    with p.open() as handle: rows=list(csv.DictReader(handle))
    assert len(rows)==source['rows']
    power[source['host']]=sorted((float(r['epoch_ms'])/1000,float(r['power_draw_w'])) for r in rows if r['power_draw_w'])
netdata=[json.loads(l) for l in (base/'evidence/observer/netdata.jsonl').read_text().splitlines()]
samples=[json.loads(l) for l in (base/'evidence/observer/python_sampler.jsonl').read_text().splitlines()]
arms=[]
for arm in result['arms'][2:]:
    start,end=arm['started_unix'],arm['finished_unix'];hosts={h:energy(v,start,end) for h,v in power.items()}
    for host in hosts:
        vals=[100-s['metrics']['system.cpu']['dimensions']['idle']['value'] for s in netdata if s['host']==host and start<=s['time']<=end]
        assert vals;hosts[host].update(cpu_mean_nonidle_percent=statistics.mean(vals),netdata_samples=len(vals))
    selected=[s for s in samples if start<=s['time']<=end];assert len(selected)>=2
    counters=[{k:int(v) for k,v in (l.split(':',1) for l in s['process_io'].splitlines())} for s in (selected[0],selected[-1])]
    io_groups=collections.defaultdict(list)
    for c in arm['io_calls']: io_groups[c['kind']].append(c['seconds'])
    assert {k:len(v) for k,v in io_groups.items()}==dict(fsync=32,torch_save=16,path_write_bytes=16)
    arms.append(dict(label=arm['label'],best_form=arm['best_form'],seconds=arm['seconds'],hosts=hosts,
        io_calls={k:dict(count=len(v),seconds=sum(v),max_seconds=max(v)) for k,v in io_groups.items()},
        sampled_process_io={k:counters[1][k]-counters[0][k] for k in counters[0]},
        io_sample_interval=[selected[0]['time'],selected[-1]['time']],python_samples=len(selected),
        viterbi_inclusive_samples=sum(any(f['function']=='viterbi_window_fused' for f in s['frames']) for s in selected)))
means={}
for best,label in ((False,'front'),(True,'best')):
    selected=[a for a in arms if a['best_form']==best];assert len(selected)==2
    sec=statistics.mean(a['seconds'] for a in selected);j=statistics.mean(a['hosts']['sparky']['joules'] for a in selected)
    means[label]=dict(seconds=sec,joules=j,mean_w=j/sec,units_per_joule=16/j)
print(json.dumps(dict(status='PASS',action=key,receipt=receipt['receipt_sha256'],source=result['producer_source_sha256'],
    torch=result['torch'],host=terminal['claimed_host'],format=result['format_name'],shape=[4096,2048],units=16,batch=8,
    all_six_arms_exact_wire_and_score=True,actual_wire_hashes_checked=16,old_07ad_exact_units=old_parity,
    capture_load_execution=load,actual_first_call_inputs=inputs,profiles=profiles,arms=arms,means=means,
    throughput_ratio=means['front']['seconds']/means['best']['seconds'],work_per_joule_ratio=means['front']['joules']/means['best']['joules'],
    limitations=['A bounded endpoint comparison, not a completed pricing row or model.',
        'Complete-call traces are warm-up attribution only, not timed-arm traces.',
        'torch_save durations include serialization and I/O; do not equate them with network latency.',
        'Process I/O counters exclude interval tails and cannot establish zero NFS latency alone.',
        'Old-source parity is evidence only; old receipts are not relabeled or adopted.']),sort_keys=True))
