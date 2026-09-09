"""Summarize existing stack and process-I/O samples without another GPU run."""
import collections
import hashlib
import json
from pathlib import Path

base = Path('/mnt/shared/tessera-measurements/glm-canonical-census-20260908/first-proof-anchor-preparation-02/performance-batch-ab-01')
result = json.loads((base/'evidence/result.json').read_text())
path = base/'evidence/observer/python_sampler.jsonl'
raw = path.read_bytes()
samples = [json.loads(line) for line in raw.splitlines()]

def counters(sample):
    return {key: int(value) for key, value in
            (line.split(':', 1) for line in sample['process_io'].splitlines())}

phases = [('startup', result['started_unix'], result['arms'][0]['started_unix'])]
phases += [(a['label'], a['started_unix'], a['finished_unix']) for a in result['arms']]
rows = []
for label, start, end in phases:
    selected = [s for s in samples if start <= s['time'] <= end]
    assert len(selected) >= 2, label
    first, last = counters(selected[0]), counters(selected[-1])
    delta = {key: last[key]-first[key] for key in first}
    # cancelled_write_bytes may decrease; other process counters must not.
    assert all(value >= 0 for key, value in delta.items()
               if key != 'cancelled_write_bytes'), label
    leaf = collections.Counter(s['frames'][-1]['function'] for s in selected)
    inclusive = collections.Counter(name for s in selected
                                    for name in {f['function'] for f in s['frames']})
    net = [json.loads(line) for line in (base/'evidence/observer/netdata.jsonl').read_text().splitlines()]
    host_cpu = {}
    for host in ('sparky', 'sparklina'):
        values = [100-s['metrics']['system.cpu']['dimensions']['idle']['value']
                  for s in net if s['host'] == host and start <= s['time'] <= end]
        assert values, (label, host)
        host_cpu[host] = dict(samples=len(values), mean_nonidle_percent=sum(values)/len(values))
    rows.append(dict(phase=label, phase_seconds=end-start,
                     sample_start=selected[0]['time'], sample_end=selected[-1]['time'],
                     sampled_seconds=selected[-1]['time']-selected[0]['time'],
                     samples=len(selected), process_io_delta=delta,
                     leaf_functions=leaf.most_common(15),
                     inclusive_functions=inclusive.most_common(25), host_cpu=host_cpu))
print(json.dumps(dict(status='PASS', source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
                     phases=rows, limitations=[
                         'Counters cover only first-to-last sample inside each phase, excluding its boundary tails.',
                         'Linux read_bytes does not establish NFS traffic or remote disk latency; rchar includes cached reads.',
                         'Main-thread samples show where execution waits or works, not a precise per-function timer.',
                         'No new startup measurement or before/after optimization is claimed.']), sort_keys=True))
