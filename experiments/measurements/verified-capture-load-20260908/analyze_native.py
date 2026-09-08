"""Read completed native artifacts and derive bounded, instrumented comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import pstats
import statistics


def main(root):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    netdata = [json.loads(line) for line in (root/'netdata.jsonl').read_text().splitlines()]
    pqteld = json.loads((root/'pqteld-slice.json').read_text())
    samples = sorted((float(r['epoch_ms'])/1000,float(r['power_draw_w'])) for r in pqteld['rows'])
    rows = []
    for row in summary['results']:
        stem = f"arm-{row['arm']}-{row['mode']}-{row['operation']}"
        trace = json.loads((root/(stem+'.trace.json')).read_text())['traceEvents']
        cpu = {}
        for event_name in ('aten::isfinite','aten::aminmax','aten::all'):
            events = [e for e in trace if e.get('cat')=='cpu_op' and e.get('name')==event_name and e.get('ph')=='X']
            cpu[event_name] = dict(calls=len(events), inclusive_s=sum(e['dur'] for e in events)/1e6,
                                  threads=sorted({(e['pid'],e['tid']) for e in events}))
        transfers = [e for e in trace if e.get('cat')=='gpu_memcpy' and e.get('ph')=='X']
        kernels = [e for e in trace if e.get('cat')=='kernel' and e.get('ph')=='X']
        prof = pstats.Stats(str(root/(stem+'.cprofile')))
        guard = [v for k,v in prof.stats.items() if k[0].endswith('/memory_management.py') and k[2]=='check']
        checks = sum(v[1] for v in guard)
        check_s = sum(v[3] for v in guard)
        power_samples = [dict(time=t, watts=w) for t,w in samples
                         if row['started'] <= t <= row['finished']]
        rows.append(dict(arm=row['arm'],mode=row['mode'],operation=row['operation'],
            elapsed_s=row['elapsed_s'], source_reads=row['source_reads'], proc_io_delta=row['proc_io_delta'],
            live_serialized_buffers=row['live_serialized_buffers'],cpu_events=cpu,
            cprofile_guard_calls=checks,cprofile_guard_cumulative_s=check_s,
            gpu_transfer_count=len(transfers),gpu_transfer_s=sum(e['dur'] for e in transfers)/1e6,
            gpu_kernel_count=len(kernels),gpu_power_observations=power_samples,
            phase_preflight=row.get('phase_preflight')))
    comparisons = []
    for op in ('replay','seal','prefetch'):
        arms={mode:[r for r in rows if r['operation']==op and r['mode']==mode] for mode in ('legacy','verified')}
        mean={mode:statistics.mean(r['elapsed_s'] for r in arm) for mode,arm in arms.items()}
        comparisons.append(dict(operation=op,mean_elapsed_s=mean,verified_over_legacy_elapsed=mean['verified']/mean['legacy']))
    hosts={}
    for host in ('sparky','sparklina'):
        host_rows=[r for r in netdata if r['host']==host]
        busy=[sum(d['value'] for n,d in r['metrics']['system.cpu']['dimensions'].items() if n not in ('idle','iowait')) for r in host_rows]
        powers=[(c['last_updated'],d['value']) for r in host_rows for n,c in r['metrics'].items() if n.endswith('_power_draw') for d in c['dimensions'].values()]
        hosts[host]=dict(samples=len(host_rows),cpu_busy_mean=statistics.mean(busy),cpu_busy_peak=max(busy),
            gpu_power_w_unique_samples=sorted(set(powers)))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and not p.name.startswith('analysis')}
    result=dict(schema='prismaquant.verified_capture_native_analysis.v1',source_root=str(root),source_sha256=hashes,
        rows=rows,comparisons=comparisons,netdata_hosts=hosts,
        energy_limit='Only a few 2 Hz pqteld observations fall inside each short operation. Preserve raw timestamps; do not interpolate arm boundaries or rank per-operation work/J. Host/CPU energy is unmeasured.',
        limits='One instrumented ABBA; cProfile background thread/wait and nested totals cannot be summed as exclusive wall costs. Torch inclusive scopes also overlap. Netdata GPU power updates every 10 s; pqteld supplies finer power observations, but still insufficient per-operation energy coverage. No production/full-model fit/serving/KL claim.')
    (root/'analysis-final.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
