"""Check physical timing and arm membership in the retained best-form R4 trace."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics

key = '9d7135d5c659d8344bb9fd4954a483d8903e356020958dd1b7a17c4b37bc8f98'
fleet = Path('/mnt/shared/prismabuild-fleet')
base = Path('/mnt/shared/tessera-measurements/window-best-form-ab-20260909/v3')
terminal = json.loads((fleet/'pb-queue/done'/f'{key}.json').read_text())
assert terminal['detail']['returncode'] == 0 and terminal['resource_scope_cleanup']['complete']
profile = terminal['detail']['profile']
path = Path(profile['blob_path'])
raw = path.read_bytes()
assert len(raw) == profile['bytes'] and hashlib.sha256(raw).hexdigest() == profile['blob_sha256']
decoded = gzip.decompress(raw)
assert len(decoded) == profile['decoded_bytes']
trace = json.loads(decoded)
assert len(trace['traceEvents']) == profile['events']
results = json.loads((base/'pbprofile.json').read_text())
assert len(results) == 2
assert results[0]['pb_profile']['bytes'] != profile['bytes']
assert results[1]['config'] == 'R4' and results[1]['pb_profile']['bytes'] == profile['bytes']
markers = [e for e in trace['traceEvents'] if e.get('cat') == 'user_annotation'
           and e.get('ph') == 'X' and e.get('name', '').startswith('arm:')]
assert {e['name'] for e in markers} == {'arm:front', 'arm:best', 'arm:best@32'}
kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X']
rows = []
for marker in markers:
    start, end = marker['ts'], marker['ts']+marker['dur']
    selected = sorted((e for e in kernels if start <= e['ts'] <= end), key=lambda e:e['ts'])
    assert selected and all(e['dur'] >= 0 and e['ts']+e['dur'] <= end+1 for e in selected)
    steps = [e for e in selected if e['name'] in ('_step', '_step_best')]
    expected = {'arm:front':8192, 'arm:best':4095, 'arm:best@32':8190}[marker['name']]
    assert len(steps) == expected
    assert max(e['dur'] for e in steps) < 100
    overlaps = [a['ts']+a['dur']-b['ts'] for a,b in zip(steps,steps[1:])
                if a['ts']+a['dur'] > b['ts']]
    assert max(overlaps, default=0) < 1
    assert sum(overlaps)/sum(e['dur'] for e in steps) < 0.0001
    span = max(e['ts']+e['dur'] for e in selected)-min(e['ts'] for e in selected)
    assert sum(e['dur'] for e in selected) <= span*1.01
    grouped = collections.defaultdict(lambda:dict(calls=0, microseconds=0.0))
    for event in selected:
        grouped[event['name']]['calls'] += 1
        grouped[event['name']]['microseconds'] += event['dur']
    rows.append(dict(arm=marker['name'], marker_microseconds=marker['dur'],
                     step_calls=len(steps), step_mean_us=statistics.mean(e['dur'] for e in steps),
                     kernel_span_us=span, summed_kernel_us=sum(e['dur'] for e in selected),
                     step_overlap_count=len(overlaps), step_overlap_max_us=max(overlaps,default=0),
                     kernels=dict(sorted(grouped.items(),key=lambda p:-p[1]['microseconds'])[:8])))
assert sum(sum(x['calls'] for x in r['kernels'].values()) for r in rows) <= len(kernels)
print(json.dumps(dict(status='R4_PROFILE_VERIFIED_R3_RAW_TRACE_MISSING', action=key,
                      host=terminal['claimed_host'], profile=profile, arms=rows,
                      limitations=[
                          'The harness overwrote the R3 raw trace with R4 at the same PB output path.',
                          'R3 printed profiler aggregates exist, but no retained R3 raw trace is attested by this receipt.',
                          'best@32 changes outer chunking and epilogue work; it does not isolate only internal batch width.',
                          'Both comparison arms are profiled on Sparklina; separate timing/energy was measured on Sparky.']), sort_keys=True))
