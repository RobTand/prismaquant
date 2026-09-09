"""Audit retained PB batch measurements, trace and both host recorders."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

BASE = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/performance-batch-ab-01')
CAS = Path('/mnt/shared/prismabuild-fleet/cas')
KEY = 'ed30dec36df89ca9fd477cbdfbd7f5c9563e1947df32cebaf2e6c1582d37d715'

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

receipt = json.loads((CAS/'actions/v3'/KEY[:2]/(KEY+'.json')).read_text())
terminal = json.loads((Path('/mnt/shared/prismabuild-fleet/pb-queue/done')/(KEY+'.json')).read_text())
assert terminal['detail']['returncode'] == 0 and terminal['resource_scope_cleanup']['complete']
for blob in (receipt['result'], terminal['checkout_snapshot']['input']):
    path = CAS/'blobs'/blob['sha256'][:2]/blob['sha256']
    assert path.stat().st_size == blob['bytes'] and digest(path) == blob['sha256']
profile = terminal['detail']['profile']
assert profile['produced'] and profile['mode'] == 'torch'
path = Path(profile['blob_path'])
assert path.stat().st_size == profile['bytes'] and digest(path) == profile['blob_sha256']
with gzip.open(path, 'rb') as f:
    raw = f.read()
assert len(raw) == profile['decoded_bytes']
trace = json.loads(raw)
del raw
result = json.loads((BASE/'evidence/result.json').read_text())
observer = json.loads((BASE/'evidence/observer/result.json').read_text())
assert result['status'] == observer['status'] == 'complete' and not observer['errors']
assert result['full_campaign_complete'] is False
assert len(result['arms']) == 6 and all(a['exact_parity'] for a in result['arms'])
reference = result['arms'][0]['signatures']
assert all(a['signatures'] == reference for a in result['arms'])
for entry in reference:
    matches = list((BASE/'cache/wire').glob(entry['qname'].replace('.', '_')+'__*'))
    if not matches:
        matches = [p for p in (BASE/'cache/wire').glob('*.tessera') if digest(p) == entry['wire_sha256']]
    assert any(p.stat().st_size == entry['wire_bytes'] and digest(p) == entry['wire_sha256'] for p in matches)

stacks = [json.loads(x) for x in (BASE/'evidence/observer/python_sampler.jsonl').read_text().splitlines()]
netdata = [json.loads(x) for x in (BASE/'evidence/observer/netdata.jsonl').read_text().splitlines()]
recorders = {}
for host in ('sparky', 'sparklina'):
    rows = list(csv.DictReader((BASE/(host+'-pqteld.csv')).open()))
    recorders[host] = sorted((float(x['epoch_ms'])/1000, float(x['power_draw_w']))
        for x in rows if x['power_draw_w'])

def energy(rows, start, end):
    # Integrate piecewise-linear interpolation of the real 2Hz recorder.
    # Both boundary samples are present; never extrapolate missing tails.
    assert rows[0][0] <= start <= end <= rows[-1][0]
    joules = 0.0
    gaps = []
    for (t0, p0), (t1, p1) in zip(rows, rows[1:]):
        lo, hi = max(start, t0), min(end, t1)
        if lo >= hi:
            continue
        gaps.append(t1-t0)
        left = p0+(p1-p0)*(lo-t0)/(t1-t0)
        right = p0+(p1-p0)*(hi-t0)/(t1-t0)
        joules += (left+right)*0.5*(hi-lo)
    assert max(gaps) < 2, ('recorder gap', max(gaps))
    selected = [p for t,p in rows if start <= t <= end]
    return dict(estimated_joules=joules, time_weighted_mean_w=joules/(end-start),
        peak_sample_w=max(selected), samples=len(selected), max_gap_seconds=max(gaps),
        method='piecewise-linear integral of 2Hz pqteld, actual boundary interpolation')

arms = []
for arm in result['arms'][2:]:
    start, end = arm['started_unix'], arm['finished_unix']
    row = {k:v for k,v in arm.items() if k != 'signatures'}
    row['hosts'] = {host: energy(records,start,end) for host,records in recorders.items()}
    row['units_per_joule'] = 16/row['hosts']['sparklina']['estimated_joules']
    selected = [x for x in stacks if start <= x['time'] <= end]
    row['python_samples'] = len(selected)
    row['leaf_frames'] = dict(collections.Counter(x['frames'][-1]['function'] for x in selected))
    row['viterbi_inclusive_samples'] = sum(any(f['function']=='viterbi_window_fused' for f in x['frames']) for x in selected)
    row['netdata'] = {}
    for host in recorders:
        points = [x for x in netdata if x['host']==host and start<=x['time']<=end]
        assert points
        cpu = [100-x['metrics']['system.cpu']['dimensions']['idle']['value'] for x in points]
        row['netdata'][host] = dict(records=len(points),nonidle_mean_percent=statistics.mean(cpu),
            nonidle_peak_percent=max(cpu),note='5-second poll observations; each raw chart retains its actual last_updated')
    arms.append(row)

kernels = sorted((e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X'),key=lambda e:e['ts'])
trace_epoch_offset = trace['baseTimeNanoseconds']/1e9
summary=[]
for arm in arms:
    events=[e for e in kernels if arm['started_unix'] <= trace_epoch_offset+e['ts']/1e6 <= arm['finished_unix']]
    if not events:
        summary.append(dict(label=arm['label'],observed=False,kernel_events=0))
        continue
    names=collections.defaultdict(lambda:[0,0.0])
    for e in events:
        names[e['name']][0]+=1;names[e['name']][1]+=e['dur']
    steps=[e for e in events if e['name'].startswith('_step')]
    gaps=[b['ts']-(a['ts']+a['dur']) for a,b in zip(steps,steps[1:]) if b['ts']-a['ts']<100]
    summary.append(dict(label=arm['label'],observed=True,kernel_events=len(events),
        first_kernel_unix=trace_epoch_offset+events[0]['ts']/1e6,
        last_kernel_unix=trace_epoch_offset+(events[-1]['ts']+events[-1]['dur'])/1e6,
        summed_kernel_us=sum(e['dur'] for e in events),
        kernel_span_us=events[-1]['ts']+events[-1]['dur']-events[0]['ts'],
        step_kernels=len(steps),step_mean_us=statistics.mean(e['dur'] for e in steps) if steps else None,
        within_chain_step_gap_mean_us=statistics.mean(gaps) if gaps else None,
        top_kernel_time=sorted([dict(name=k,count=v[0],total_us=v[1]) for k,v in names.items()],key=lambda v:-v['total_us'])[:12]))

# Known dependent _step launches cannot take seconds or overlap their next
# dependent launch. Keep wall/power evidence, but refuse invalid CUDA timing.
invalid_windows=[]
for row in summary:
    if (not row.get('observed') or row.get('step_mean_us') is None
            or row['step_mean_us'] > 1000
            or row.get('within_chain_step_gap_mean_us',0) < -0.01
            or row.get('kernel_span_us',0) > 1_000_000):
        invalid_windows.append(row['label'])
    row['timing_usable'] = row['label'] not in invalid_windows

groups={width:[a for a in arms if a['batch_size']==width] for width in (8,16)}
means={width:dict(seconds=statistics.mean(a['seconds'] for a in group),
    watts=statistics.mean(a['hosts']['sparklina']['time_weighted_mean_w'] for a in group),
    estimated_joules=statistics.mean(a['hosts']['sparklina']['estimated_joules'] for a in group)) for width,group in groups.items()}
report=dict(status='LIMITED_INVALID_LATER_CUDA_WINDOWS' if invalid_windows else 'PASS',invalid_cuda_windows=invalid_windows,action=KEY,receipt=receipt['receipt_sha256'],profile=profile,
    workload=dict(format=result['format_name'],units=16,shape=result['shapes'][0],
        selected_group_resident_units=864,calibration_source_forward_count=0,
        comparison='ABBA, batch8 versus batch16, two measured arms each after warming both'),
    arms=arms,means=means,throughput_ratio_b16_over_b8=means[8]['seconds']/means[16]['seconds'],
    energy_ratio_b16_over_b8=means[16]['estimated_joules']/means[8]['estimated_joules'],
    trace_windows=summary,trace_epoch_offset=trace_epoch_offset,unassigned_kernel_events=len(kernels)-sum(x["kernel_events"] for x in summary),exact_wire_and_score_parity=True,
    limitations=['Only first16 down projections at R832; no full-group or full-model throughput claim.',
        'Four half-second windows were requested; later CUDA timings are invalid after repeated toggling in one profiler.',
        'The initial B8 CUDA window is valid; there is no accepted paired CUDA-timing comparison.',
        'Full-arm wall times, main-thread samples and both-host pqteld/Netdata are retained.',
        'Power target90–100W remains unmet; no production batch default changed.'],
    artifacts={str(p):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in [BASE/'evidence/result.json',
        BASE/'evidence/observer/result.json',BASE/'evidence/observer/netdata.jsonl',
        BASE/'evidence/observer/python_sampler.jsonl',BASE/'sparky-pqteld.csv',BASE/'sparklina-pqteld.csv']})
print(json.dumps(report,sort_keys=True))
