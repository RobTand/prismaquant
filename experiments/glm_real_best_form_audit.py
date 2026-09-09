"""Audit the real GLM fixed-B8 comparison, including wires, traces and host data."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

base = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/performance-best-form-ab-01')
fleet = Path('/mnt/shared/prismabuild-fleet')
key = '569ee819458cb3cf1c91a687ff66f1edbcd51a87a4f31813e36dc9fb2c572c2b'

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

terminal = json.loads((fleet/'pb-queue/done'/f'{key}.json').read_text())
receipt = json.loads((fleet/'cas/actions/v3'/key[:2]/f'{key}.json').read_text())
assert terminal['detail']['returncode'] == 0 and terminal['resource_scope_cleanup']['complete']
assert terminal['claimed_host'] == 'sparky'
blob = receipt['result']
path = fleet/'cas/blobs'/blob['sha256'][:2]/blob['sha256']
assert path.stat().st_size == blob['bytes'] and digest(path) == blob['sha256']
printed = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
result = json.loads((base/'evidence/result.json').read_text())
observer = json.loads((base/'evidence/observer/result.json').read_text())
assert result['status'] == observer['status'] == 'complete' and not observer['errors']
assert result['comparison'] == 'trellis-best-form' and result['full_campaign_complete'] is False
assert result['producer_source_sha256'] == 'e6fe414581b7c51f76c0c3f75fd461365136949339b5082d8fe4de189034f8d4'
assert len(result['arms']) == 6 and all(a['batch_size'] == 8 and a['exact_parity'] for a in result['arms'])
signatures = result['arms'][0]['signatures']
assert len(signatures) == 16 and all(a['signatures'] == signatures for a in result['arms'])
printed_arms = [p for p in printed if p.get('label','').startswith(('warm-','measured-'))]
assert len(printed_arms) == 6
for saved, public in zip(result['arms'], printed_arms):
    assert all(saved[k] == v for k,v in public.items())
for record in signatures:
    path = base/'cache/wire'/(record['qname'].replace('.','__')+'__'+result['format_name']+'.tessera')
    assert path.is_file() and path.stat().st_size == record['wire_bytes'] and digest(path) == record['wire_sha256']
old = json.loads((base.parent/'performance-batch-ab-01/evidence/result.json').read_text())
old_parity = old['arms'][0]['signatures'] == signatures
load_paths = list((base/'cache').glob('capture-load-execution-*.json'))
assert len(load_paths) == 1
load = json.loads(load_paths[0].read_text())
assert load['prefetch']['loaded_entries'] == 864 and load['prefetch']['live_buffer_bytes'] == 0
assert load['resources']['source_forward_count'] == 0
assert load['capture']['sha256'] == 'f4bcbf408d3aa81b04c1fabd1d2d7457176a95dcccdd5ed368800de5c37e277c'
stacks = [json.loads(line) for line in (base/'evidence/observer/python_sampler.jsonl').read_text().splitlines()]
netdata = [json.loads(line) for line in (base/'evidence/observer/netdata.jsonl').read_text().splitlines()]
recorders = {}
for host in ('sparky','sparklina'):
    with (base/(host+'-pqteld.csv')).open() as handle:
        recorders[host] = sorted((float(r['epoch_ms'])/1000,float(r['power_draw_w']))
                                for r in csv.DictReader(handle) if r['power_draw_w'])

def energy(rows,start,end):
    assert rows[0][0] <= start < end <= rows[-1][0]
    joules = covered = 0.0
    gaps = []
    for (t0,p0),(t1,p1) in zip(rows,rows[1:]):
        lo,hi = max(start,t0),min(end,t1)
        if lo >= hi:
            continue
        gaps.append(t1-t0)
        a,b = p0+(p1-p0)*(lo-t0)/(t1-t0),p0+(p1-p0)*(hi-t0)/(t1-t0)
        joules += (a+b)*0.5*(hi-lo)
        covered += hi-lo
    assert abs(covered-(end-start)) < 1e-5 and max(gaps) < 1
    return dict(joules=joules,mean_w=joules/(end-start),max_sample_gap_s=max(gaps),
                peak_sample_w=max(p for t,p in rows if start <= t <= end))

def io(sample):
    return {k:int(v) for k,v in (line.split(':',1) for line in sample['process_io'].splitlines())}

arms = []
for arm in result['arms'][2:]:
    start,end = arm['started_unix'],arm['finished_unix']
    hostdata = {host:energy(rows,start,end) for host,rows in recorders.items()}
    samples = [s for s in stacks if start <= s['time'] <= end]
    assert len(samples) >= 2
    first,last = io(samples[0]),io(samples[-1])
    io_delta = {k:last[k]-first[k] for k in first}
    for host in hostdata:
        values = [100-s['metrics']['system.cpu']['dimensions']['idle']['value']
                  for s in netdata if s['host'] == host and start <= s['time'] <= end]
        assert values
        hostdata[host]['netdata_cpu_mean_nonidle_percent'] = statistics.mean(values)
        hostdata[host]['netdata_records'] = len(values)
    trace_record = arm['trace']
    path = Path(trace_record['path'])
    assert path.stat().st_size == trace_record['bytes'] and digest(path) == trace_record['sha256']
    with gzip.open(path,'rb') as handle:
        trace = json.load(handle)
    epoch = trace['baseTimeNanoseconds']/1e9
    events = sorted((e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X'),key=lambda e:e['ts'])
    assert events and all(e['dur'] >= 0 and start <= epoch+e['ts']/1e6 <= epoch+(e['ts']+e['dur'])/1e6 <= end for e in events)
    span = max(e['ts']+e['dur'] for e in events)-min(e['ts'] for e in events)
    rejected = []
    duration_sum = sum(e['dur'] for e in events)
    if span >= 1_000_000:
        rejected.append('Kernel span exceeds the requested half-second capture window.')
    if duration_sum > span*1.01:
        rejected.append('Summed kernel duration exceeds serial stream span.')
    steps = [e for e in events if e['name'] in ('_step','_step_best')]
    assert steps and {e['name'] for e in steps} == ({'_step_best'} if arm['best_form'] else {'_step'})
    overlaps = [a['ts']+a['dur']-b['ts'] for a,b in zip(steps,steps[1:]) if a['ts']+a['dur'] > b['ts']]
    if max(overlaps,default=0) >= 1 or sum(overlaps)/sum(e['dur'] for e in steps) >= 0.0001:
        rejected.append('Dependent Viterbi steps overlap, making CUDA timing unusable.')
    grouped = collections.defaultdict(lambda:dict(calls=0,us=0.0))
    for e in events:
        grouped[e['name']]['calls'] += 1
        grouped[e['name']]['us'] += e['dur']
    arms.append(dict(label=arm['label'],best_form=arm['best_form'],seconds=arm['seconds'],
                     hosts=hostdata,units_per_joule=16/hostdata['sparky']['joules'],
                     allocated_peak_bytes=arm['allocated_peak_bytes'],reserved_peak_bytes=arm['reserved_peak_bytes'],
                     python_samples=len(samples),io_sample_interval=[samples[0]['time'],samples[-1]['time']],
                     process_io_delta=io_delta,
                     viterbi_inclusive_samples=sum(any(f['function']=='viterbi_window_fused' for f in s['frames']) for s in samples),
                     trace=trace_record,trace_status='REJECTED' if rejected else 'VALID',
                     trace_rejection_reasons=rejected,
                     trace_diagnostics=dict(kernel_span_us=span,kernel_duration_sum_us=duration_sum,
                         step_calls=len(steps),step_overlap_count=len(overlaps),
                         step_overlap_max_us=max(overlaps,default=0)),
                     kernels=None if rejected else dict(sorted(grouped.items(),key=lambda p:-p[1]['us'])[:8])))
    del trace,events,steps
assert terminal['detail']['profile']['blob_sha256'] == result['arms'][3]['trace']['sha256']
means = {}
for candidate in (False,True):
    group = [a for a in arms if a['best_form'] is candidate]
    assert len(group) == 2
    seconds = statistics.mean(a['seconds'] for a in group)
    joules = statistics.mean(a['hosts']['sparky']['joules'] for a in group)
    means['best' if candidate else 'front'] = dict(seconds=seconds,joules=joules,
                                                  mean_w=joules/seconds,units_per_joule=16/joules)
print(json.dumps(dict(status='LIMITED_CUDA_TRACES_REJECTED' if any(a['trace_status']=='REJECTED' for a in arms) else 'PASS',action=key,receipt=receipt['receipt_sha256'],
                      workload=dict(units=16,shape=[4096,2048],format=result['format_name'],batch_size=8,
                                    torch=result['torch'],cuda=result['cuda'],host='sparky'),
                      exact_all_six_arm_wire_and_score_parity=True,actual_wire_hashes_checked=16,
                      exact_parity_with_original_07ad_batch_screen=old_parity,
                      capture_load_execution=load,
                      startup_seconds=result['arms'][0]['started_unix']-result['started_unix'],
                      means=means,throughput_ratio=means['front']['seconds']/means['best']['seconds'],
                      work_per_joule_ratio=means['best']['units_per_joule']/means['front']['units_per_joule'],
                      arms=arms,limitations=[
                          'Only16realexpert down projections at R832; not a full-group or full-model measurement.',
                          'CUDA traces fail physical consistency checks and supply no accepted kernel timing; full-arm wall, power and Python stacks are retained.',
                          'Process-I/O deltas exclude phase tails and cannot establish zero NFS latency by themselves.',
                          'Separate startup, warm-up and trace-export time is not included in the timed anchor calls.',
                          'Measured producer61da precedes the subsequent empty-input guard; positive-input kernel code is unchanged.']),sort_keys=True))
