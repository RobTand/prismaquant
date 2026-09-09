"""Audit best-form v3 timing against separately recovered continuous telemetry."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

base = Path('/mnt/shared/tessera-measurements/window-best-form-ab-20260909/v3')
key = 'c836406e4a6bf9ca76697db54bd757e5bea2c1eabaaefb2c3b99dffb1fc7ee67'
fleet = Path('/mnt/shared/prismabuild-fleet')
terminal = json.loads((fleet/'pb-queue/done'/f'{key}.json').read_text())
receipt = json.loads((fleet/'cas/actions/v3'/key[:2]/f'{key}.json').read_text())
assert terminal['detail']['returncode'] == 0 and terminal['resource_scope_cleanup']['complete']
assert terminal['claimed_host'] == 'sparky'
blob = receipt['result']
raw = (fleet/'cas/blobs'/blob['sha256'][:2]/blob['sha256']).read_bytes()
assert len(raw) == blob['bytes'] and hashlib.sha256(raw).hexdigest() == blob['sha256']
results = json.loads((base/'ab.json').read_text())
printed = [json.loads(line) for line in raw.decode().splitlines() if line.startswith('{')]
assert printed == results
directory = base/'root-recovered-host-series'
recorders = {}
netdata = {}
for host in ('sparky', 'sparklina'):
    with (directory/(host+'-pqteld.csv')).open() as handle:
        recorders[host] = sorted((float(r['epoch_ms'])/1000, float(r['power_draw_w']))
                                for r in csv.DictReader(handle) if r['power_draw_w'])
    netdata[host] = json.loads((directory/(host+'-netdata.json')).read_text())

def energy(rows, start, end):
    assert rows[0][0] <= start < end <= rows[-1][0]
    joules = 0.0
    coverage = 0.0
    gaps = []
    for (t0, p0), (t1, p1) in zip(rows, rows[1:]):
        lo, hi = max(start, t0), min(end, t1)
        if lo >= hi:
            continue
        gaps.append(t1-t0)
        left = p0+(p1-p0)*(lo-t0)/(t1-t0)
        right = p0+(p1-p0)*(hi-t0)/(t1-t0)
        joules += (left+right)*0.5*(hi-lo)
        coverage += hi-lo
    assert abs(coverage-(end-start)) < 1e-5 and max(gaps) < 1
    return dict(joules=joules, mean_w=joules/(end-start), max_sample_gap_s=max(gaps),
                samples=sum(start <= t <= end for t, _ in rows))

reports = []
for result in results:
    for arm in ('front', 'best'):
        assert result['identity'][arm]['states_equal'] and result['identity'][arm]['sse_equal']
    assert result['identity']['front']['sse'] == result['identity']['best']['sse']
    assert result['identity']['best@32']['states_equal']
    arms = {}
    for arm, data in result['timing'].items():
        blocks = []
        for block in data['blocks']:
            start, end = block['wall_start'], block['wall_end']
            assert abs((end-start)-block['seconds']) < 0.001
            hosts = {host: energy(rows, start, end) for host, rows in recorders.items()}
            for host in hosts:
                cpu = netdata[host]['system.cpu']['data']
                # Historical system.cpu publishes every state except idle.
                # This differs from allmetrics, which includes idle itself.
                assert cpu['labels'][0] == 'time' and 'idle' not in cpu['labels']
                values = [sum(r[1:]) for r in cpu['data'] if start <= r[0] <= end
                          and all(value is not None for value in r[1:])]
                assert values
                hosts[host]['netdata_cpu_nonidle_mean_percent'] = statistics.mean(values)
                hosts[host]['netdata_cpu_samples'] = len(values)
            blocks.append(dict(original=block, hosts=hosts))
        calls = sum(b['original']['calls'] for b in blocks)
        seconds = sum(b['original']['seconds'] for b in blocks)
        joules = sum(b['hosts']['sparky']['joules'] for b in blocks)
        arms[arm] = dict(calls=calls, seconds=seconds, joules=joules,
                         mean_seconds_per_call=seconds/calls, mean_w=joules/seconds,
                         work_per_joule=result['work_per_call']*calls/joules, blocks=blocks)
    before = arms['front']
    reports.append(dict(config=result['config'], identity=result['identity'], plan=result['plan'],
                        registers=result['registers'], arms=arms,
                        throughput_ratio=before['mean_seconds_per_call']/arms['best']['mean_seconds_per_call'],
                        work_per_joule_ratio=arms['best']['work_per_joule']/before['work_per_joule']))

paths = [base/'ab.json', *directory.glob('*')]
print(json.dumps(dict(status='TIMING_AND_RECOVERED_TELEMETRY_VERIFIED_PROFILE_PENDING',
                      action=key, receipt=receipt['receipt_sha256'], reports=reports,
                      artifacts={str(p):dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths},
                      limitations=[
                          'Representative seeded inputs and actual E4 table, not captured GLM pricing residuals.',
                          'Continuous 2Hz host telemetry is recovered separately; original 1Hz raw power samples were not retained.',
                          'best@32 changes outer SSE chunk association and is an attribution control, not a byte-parity candidate.',
                          'Accepted in-process before/after profile and broader native correctness remain pending.']), sort_keys=True))
