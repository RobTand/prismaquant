"""Check complete-call CUDA traces against actual GLM inputs and PB receipts."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics

base = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/performance-best-form-call-profile-01')
fleet = Path('/mnt/shared/prismabuild-fleet')
key = 'c86c6920b91f65c45dc9f25d6a9c19e17a47159314dfb436206ffbabb74364f6'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

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
assert result['producer_source_sha256'] == 'e6fe414581b7c51f76c0c3f75fd461365136949339b5082d8fe4de189034f8d4'
assert result['call'] == dict(targets_shape=[4096,192],vectors_shape=[16384,1],window_bits=14,rate=3,chunk=512,has_weights=True)
assert len(result['arms']) == 2
assert result['arms'] == [p for p in printed if p.get('label') in ('front','best')]
assert all(a['exact_parity'] for a in result['arms'])
assert result['arms'][0]['sse'] == result['arms'][1]['sse']
reports = []
for arm in result['arms']:
    meta = arm['trace']
    path = Path(meta['path'])
    assert path.stat().st_size == meta['bytes'] and digest(path) == meta['sha256']
    trace = json.loads(gzip.decompress(path.read_bytes()))
    events = trace['traceEvents']
    kernels = sorted((e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X'),key=lambda e:e['ts'])
    assert kernels
    epoch = trace['baseTimeNanoseconds']/1e9
    assert all(e['dur'] >= 0 and arm['started_unix'] <= epoch+e['ts']/1e6 <= epoch+(e['ts']+e['dur'])/1e6 <= arm['finished_unix'] for e in kernels)
    span = max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels)
    assert sum(e['dur'] for e in kernels) <= span*1.01
    best = arm['best_form']
    step_name = '_step_best' if best else '_step'
    steps = [e for e in kernels if e['name'] == step_name]
    assert len(steps) == (4095 if best else 24576)
    overlaps = [a['ts']+a['dur']-b['ts'] for a,b in zip(steps,steps[1:]) if a['ts']+a['dur'] > b['ts']]
    assert max(overlaps,default=0) < 1 and sum(overlaps)/sum(e['dur'] for e in steps) < 0.0001
    traceback = [e for e in kernels if e['name'] == '_traceback']
    assert len(traceback) == 1
    launches = [e for e in events if e.get('name') == 'cudaGraphLaunch' and e.get('ph') == 'X']
    assert len(launches) == (1 if best else 6)
    assert not any('StreamBeginCapture' in e.get('name','') for e in events)
    grouped = collections.defaultdict(lambda: dict(calls=0,us=0.0))
    for e in kernels:
        grouped[e['name']]['calls'] += 1
        grouped[e['name']]['us'] += e['dur']
    reports.append(dict(label=arm['label'],trace=meta,kernel_span_us=span,
        kernel_sum_us=sum(e['dur'] for e in kernels),step_count=len(steps),
        step_mean_us=statistics.mean(e['dur'] for e in steps),step_max_us=max(e['dur'] for e in steps),
        dependent_overlap_count=len(overlaps),graph_launches=len(launches),
        traceback_us=traceback[0]['dur'],kernels=dict(sorted(grouped.items(),key=lambda p:-p[1]['us'])[:8])))
assert terminal['detail']['profile']['blob_sha256'] == result['arms'][1]['trace']['sha256']
print(json.dumps(dict(status='PASS',action=key,receipt=receipt['receipt_sha256'],
    host=terminal['claimed_host'],torch=result['torch'],call=result['call'],reports=reports,
    limitations=['Single actual R3 residual call per implementation; R4 is not profiled here.',
        'Profiler contexts include fixed overhead; use the separate ABBA result for throughput and energy.',
        'No new cost table or shipping qualification was produced.']),sort_keys=True))
