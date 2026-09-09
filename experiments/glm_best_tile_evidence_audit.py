"""Verify corrected tile screens, raw energy samples and complete PB profiles."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics

base = Path('/mnt/shared/tessera-measurements/window-best-tile-20260909')
fleet = Path('/mnt/shared/prismabuild-fleet')
timing_keys = {'R3':'530795d203997b69b7735d36809896d47fdadf86a31a922372fb8b6325ec2136',
               'R4':'6551e14e654ab3e8ba32e7a9c6ac4bdb909f6b87d3c79007c051a72d57f16f79'}
profile_keys = {
    'R3':'bccfd8d021d7a4ba86e5c15b46a7eb38bcff25af515869b45a291b52d3d077e1',
    'R4':'28e42f5c626b25532e3fb4db5ee9594bca7a1e2a3266c4487b12f8229f70163e',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def action(key):
    terminal = json.loads((fleet/'pb-queue/done'/f'{key}.json').read_text())
    assert terminal['detail']['returncode'] == 0 and terminal['resource_scope_cleanup']['complete']
    receipt = json.loads((fleet/'cas/actions/v3'/key[:2]/f'{key}.json').read_text())
    blob = receipt['result']
    path = fleet/'cas/blobs'/blob['sha256'][:2]/blob['sha256']
    assert path.stat().st_size == blob['bytes'] and digest(path) == blob['sha256']
    printed = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    return terminal, receipt, printed

reports = []
for config,timing_key in timing_keys.items():
    terminal,receipt,results=action(timing_key)
    assert results == json.loads((base/f'v2/tile-{config}.json').read_text())
    assert len(results)==1
    result=results[0]
    assert result['config']==config
    assert not any(result['plans_built_inside_timed_blocks'].values())
    assert all(v['states_equal'] and v['sse_equal'] for v in result['identity'].values())
    assert len({v['sse'] for v in result['identity'].values()}) == 1
    timing = {}
    for arm, data in result['timing'].items():
        measured = []
        for block in data['blocks']:
            assert block['plans_built']==0
            start, end = block['wall_start'], block['wall_end']
            rows = block['power']['samples']
            assert block['power']['bracketed'] and block['power']['covered'] == 1.0
            assert rows[0][0] <= start < end <= rows[-1][0]
            joules = 0.0
            covered = 0.0
            gaps = []
            for (t0,p0), (t1,p1) in zip(rows, rows[1:]):
                assert t1 > t0
                lo, hi = max(start,t0), min(end,t1)
                if lo >= hi:
                    continue
                gaps.append(t1-t0)
                a = p0+(p1-p0)*(lo-t0)/(t1-t0)
                b = p0+(p1-p0)*(hi-t0)/(t1-t0)
                joules += (a+b)*0.5*(hi-lo)
                covered += hi-lo
            assert abs(covered-(end-start)) < 1e-5 and max(gaps) < 2
            # The retained timestamps are rounded to milliseconds and joules
            # to cents; check within that serialization resolution.
            assert abs(joules-block['joules']) < 0.15
            measured.append(dict(calls=block['calls'], seconds=block['seconds'],
                                 joules=joules, max_sample_gap_s=max(gaps)))
        assert len(measured) == data['energy_blocks'] == 5
        calls = sum(b['calls'] for b in measured)
        seconds = sum(b['seconds'] for b in measured)
        joules = sum(b['joules'] for b in measured)
        assert calls == data['total_calls']
        timing[arm] = dict(calls=calls, seconds=seconds, joules=joules,
                           mean_seconds_per_call=seconds/calls,
                           mean_w=joules/seconds,
                           work_per_joule=calls*result['work_per_call']/joules,
                           blocks=measured)

    pt, pr, pp = action(profile_keys[config])
    assert len(pp) == 1 and pp[0]['config'] == config
    assert all(value==result['identity'][arm] for arm,value in pp[0]['identity'].items())
    meta = pt['detail']['profile']
    path = Path(meta['blob_path'])
    assert path.stat().st_size == meta['bytes'] and digest(path) == meta['blob_sha256']
    retained = pp[0]['pb_profile']
    assert retained['sha256'] == meta['blob_sha256'] == digest(Path(retained['retained']))
    raw = gzip.decompress(path.read_bytes())
    assert len(raw) == meta['decoded_bytes']
    trace = json.loads(raw)
    assert len(trace['traceEvents']) == meta['events']
    markers = [e for e in trace['traceEvents'] if e.get('cat') == 'user_annotation'
               and e.get('ph') == 'X' and e.get('name','').startswith('arm:')]
    assert {e['name'] for e in markers} == {'arm:front','arm:t128x2w4','arm:t64x4w2'}
    kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    profiles = []
    for marker in markers:
        name = marker['name'].removeprefix('arm:')
        start, end = marker['ts'], marker['ts']+marker['dur']
        selected = sorted((e for e in kernels if start <= e['ts'] <= end),key=lambda e:e['ts'])
        assert selected and all(e['dur'] >= 0 and e['ts']+e['dur'] <= end+1 for e in selected)
        steps = [e for e in selected if e['name'] in ('_step','_step_best')]
        count = result['launch'][name]['batches']*(4096 if name == 'front' else 4095)
        assert len(steps) == count, (config,name,'step count',len(steps),count)
        overlaps = [a['ts']+a['dur']-b['ts'] for a,b in zip(steps,steps[1:])
                    if a['ts']+a['dur'] > b['ts']]
        assert max(overlaps,default=0) < 1 and sum(overlaps)/sum(e['dur'] for e in steps) < 0.0001
        traceback = [e for e in selected if e['name'] == '_traceback']
        assert len(traceback) == 1, (config,name,'outer traceback changed')
        grouped = collections.defaultdict(lambda:dict(calls=0,us=0.0))
        for e in selected:
            grouped[e['name']]['calls'] += 1
            grouped[e['name']]['us'] += e['dur']
        profiles.append(dict(arm=name,step_calls=len(steps),
                              step_mean_us=statistics.mean(e['dur'] for e in steps),
                              step_max_us=max(e['dur'] for e in steps),
                              step_outliers_over_100us=sum(e['dur'] > 100 for e in steps),
                              step_overlap_max_us=max(overlaps,default=0),
                              traceback_us=traceback[0]['dur'],
                              kernels=dict(sorted(grouped.items(),key=lambda p:-p[1]['us'])[:5])))
    reports.append(dict(config=config,timing_action=timing_key,timing_receipt=receipt['receipt_sha256'],timing=timing,profile_action=profile_keys[config],
                        profile_receipt=pr['receipt_sha256'],profile=meta,profiles=profiles,
                        throughput_ratio=timing['front']['mean_seconds_per_call']/timing['t64x4w2']['mean_seconds_per_call'],
                        work_per_joule_ratio=timing['t64x4w2']['work_per_joule']/timing['front']['work_per_joule'],
                        tile_throughput_ratio=timing['t128x2w4']['mean_seconds_per_call']/timing['t64x4w2']['mean_seconds_per_call'],
                        tile_work_per_joule_ratio=timing['t64x4w2']['work_per_joule']/timing['t128x2w4']['work_per_joule']))
print(json.dumps(dict(status='PASS',
                      reports=reports,limitations=[
                          'Representative seeded inputs; actual GLM pricing remains a separate measurement.',
                          'Energy is an estimate from retained 1Hz power samples with bracketed intervals.',
                          'Fixed arm ordering leaves thermal ordering uncertainty; microcalls do not establish full encode mean power.']),sort_keys=True))
