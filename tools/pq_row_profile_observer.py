#!/usr/bin/env python3
"""Read-only observer started inside an admitted PQ row (#1899).

Derived from the qualified #1654 observer; supports either GB10 host. No Docker
calls, workload launch or row signal. The owner joins this observer after the
workload. Output never enters a wire, cost row or anchor identity.
"""
import argparse
from tools.pq_profile_digest import file_sha256hex
from tools.pq_profile_artifact import publish_profile, validate_netdata_window
from tools.pq_admitted_profile import PROFILE_LOCAL_ROOT
from tools.pq_profile_source import profile_source_owner
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request

# The host observer needs the existing stdlib sampler, not production init.
PeriodicSampler = profile_source_owner('io_spans').PeriodicSampler

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--target-out', required=True)
p.add_argument('--wait-s', type=int, default=14400)
p.add_argument('--row-s', type=int, default=43200)
p.add_argument('--profile-local', required=True,
               help='New host-local campaign metadata directory (no privileged writes)')
p.add_argument('--profiler-executable', default='py-spy')
p.add_argument('--child-profile', required=True,
               help='Same-UID child profile inside the owned observation directory')
a = p.parse_args()
local_host = socket.gethostname().split('.')[0]
if local_host not in ('sparky', 'sparklina'):
    raise RuntimeError('observer requires an admitted GB10 host')
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
profile_dir = Path(a.profile_local).resolve()
local_root = PROFILE_LOCAL_ROOT.resolve()
if not profile_dir.is_relative_to(local_root) or profile_dir == local_root:
    raise ValueError('profile-local must be a new directory below the host-local campaign profiles root')
profile_dir.mkdir(parents=True, exist_ok=False)
profile_name = 'boundary-' + local_host + '.speedscope'
local_profile = Path(a.child_profile).resolve()
if local_profile.parent != out.resolve() or local_profile.exists():
    raise ValueError('child-profile must be a new file in the owned observation directory')
lock = threading.Lock()
collection_ready = threading.Event()
telemetry_errors = []


def event(name, **data):
    with lock, (out / 'events.jsonl').open('a') as f:
        f.write(json.dumps({'epoch': time.time(), 'event': name, **data}) + '\n')


def command(args, timeout=10):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          timeout=timeout).stdout


def netdata(host, endpoint):
    url = 'http://127.0.0.1:19999/api/v1/' + endpoint
    if host == local_host:
        with urllib.request.urlopen(url, timeout=8) as r:
            return json.load(r)
    raw = command(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host,
                   'curl -fsS --max-time 8 ' + shlex.quote(url)], timeout=15)
    return json.loads(raw)


charts = {}
chart_intervals = []
for host in ('sparklina', 'sparky'):
    info = netdata(host, 'charts')
    charts[host] = [key for key, value in info['charts'].items()
                    if value.get('context') in ('system.cpu', 'system.ram', 'system.io',
                                                 'nvidia_smi.gpu_power_draw')]
    required_contexts = {'system.cpu', 'system.ram', 'system.io', 'nvidia_smi.gpu_power_draw'}
    contexts = {info['charts'][key].get('context') for key in charts[host]}
    for key in charts[host]:
        interval = info['charts'][key].get('update_every')
        if type(interval) is not int or interval <= 0:
            raise RuntimeError('required Netdata chart has no positive cadence: ' + host + ': ' + key)
        chart_intervals.append(interval)
    if not required_contexts <= contexts:
        raise RuntimeError('required Netdata contexts missing: ' + host + ': ' + repr(required_contexts - contexts))
    (out / ('netdata-charts-' + host + '.json')).write_text(json.dumps(
        {key: info['charts'][key] for key in charts[host]}, indent=2))
# Include the declared collector cadence, including a short final interval.
# Netdata's GPU power chart updates every ten seconds on these hosts.
window_padding = 2 * max(chart_intervals)
# PB deliberately sets NoNewPrivileges. Neither preflight nor profiling may
# elevate. The real profiler parents its same-UID target inside the container.
write_check = out / '.profile-write-check'
with write_check.open('x') as handle:
    handle.write('same-uid')
write_check.unlink()
version = command([a.profiler_executable, '--version'])
first_power = command(['nvidia-smi', '--query-gpu=power.draw',
                       '--format=csv,noheader,nounits'])
float(first_power.strip())
previous_netdata = int(time.time()) - 5


def collect_netdata():
    """Take one bounded window; PeriodicSampler owns the interval/lifetime."""
    global previous_netdata
    now = int(time.time())
    for host, names in charts.items():
        for chart in names:
            try:
                query = urllib.parse.urlencode(dict(chart=chart, after=previous_netdata - window_padding,
                    before=now, points=max(10, now - previous_netdata + 4),
                    group='average', format='json', options='seconds'))
                data = netdata(host, 'data?' + query)
                validate_netdata_window(data, after=previous_netdata - window_padding, before=now)
                with (out / ('netdata-' + host + '.jsonl')).open('a') as f:
                    f.write(json.dumps({'fetched_epoch': time.time(), 'chart': chart,
                                       'response': data}) + '\n')
            except Exception as error:
                telemetry_errors.append(str(error))
                event('netdata_error', host=host, chart=chart, error=str(error))
    collection_ready.set()
    previous_netdata = now


def finish_netdata():
    nd.stop(90)
    if nd.is_alive():
        error = 'required Netdata sampler join timed out; final window incomplete'
        telemetry_errors.append(error)
        event('netdata_error', error=error)
    else:
        # Only after the periodic tick joins, retain its last partial interval.
        collect_netdata()


def matches(pid):
    try:
        # PB's Python docker shim and the docker CLI both carry the row's
        # entire argv. Their comm is docker, not the target Python process.
        # Reject wrappers before exact module/--out matching; still fail
        # closed if multiple actual Python row processes match.
        comm = Path('/proc', str(pid), 'comm').read_text().strip()
        if not comm.startswith('python'):
            return False
        args = Path('/proc', str(pid), 'cmdline').read_bytes().decode().split('\0')
        i = args.index('-m')
        j = args.index('--out')
        return args[i + 1] == 'prismaquant.tessera_campaign' and args[j + 1] == a.target_out
    except (OSError, ValueError, UnicodeError, IndexError):
        return False


def find_target():
    found = [int(x) for x in os.listdir('/proc') if x.isdigit() and matches(x)]
    if len(found) > 1:
        raise RuntimeError('ambiguous matching row PIDs: ' + repr(found))
    return found[0] if found else None


def profile_target_finished(seen, alive, workload_done):
    if not workload_done:
        return False
    if not seen or alive:
        raise RuntimeError('workload completion does not match an observed exited target')
    return True


if find_target() is not None:
    raise RuntimeError('target was already running; startup profile would be incomplete')
nd = PeriodicSampler(collect_netdata, interval_s=30, name='row-netdata')
nd.start()
if not collection_ready.wait(60) or telemetry_errors:
    finish_netdata()
    raise RuntimeError('required Netdata initialization failed before readiness')
event('ready', pyspy_version=version.strip(), initial_power_w=float(first_power),
      target_out=a.target_out, charts=charts, local_profile=str(local_profile))
(out / 'ready.pending').write_text(json.dumps({'epoch': time.time(), 'target_out': a.target_out,
    'local_profile': str(local_profile),
    'host': local_host, 'observer_sha256': file_sha256hex(Path(__file__))}))
(out / 'ready.pending').replace(out / 'ready.json')
pid = None
target_exited = False
deadline = time.monotonic() + a.wait_s
next_sample = time.monotonic()
try:
    with (out / 'power_trace_1s.tsv').open('w', buffering=1) as power, \
         (out / 'proc_1s.jsonl').open('w', buffering=1) as proc:
        power.write('epoch\tpower_w\tmemavailable_kb\n')
        while True:
            now = time.monotonic()
            if pid is None:
                found = find_target()
                if found is not None:
                    pid = found
                    affinity = sorted(os.sched_getaffinity(pid))
                    os.sched_setaffinity(0, affinity)
                    sampler_tid = nd.native_id
                    if sampler_tid is None:
                        raise RuntimeError('required Netdata sampler has no native identity')
                    os.sched_setaffinity(sampler_tid, affinity)
                    deadline = time.monotonic() + a.row_s + 120
                    event('row_found', pid=pid, affinity=affinity, profile_backend='same_uid_child')
                    (out / 'target.json').write_text(json.dumps({'pid': pid,
                        'epoch': time.time(), 'affinity': affinity}))
                elif (out / 'workload_done').exists():
                    raise RuntimeError('workload ended without an observed target; profile incomplete')
                elif now > deadline:
                    raise TimeoutError('row did not start within admitted observer wait-s')
            elif not matches(pid) and not target_exited:
                event('target_exited', pid=pid)
                target_exited = True
            if profile_target_finished(pid is not None, pid is not None and matches(pid),
                                       (out / 'workload_done').exists()):
                event('row_exited', pid=pid)
                break
            if pid is not None and now > deadline:
                raise TimeoutError('row exceeded observer row-s; row was NOT signalled')
            if now >= next_sample:
                epoch = time.time()
                try:
                    value = command(['nvidia-smi', '--query-gpu=power.draw',
                        '--format=csv,noheader,nounits'], timeout=4).strip()
                    mem = next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                               if line.startswith('MemAvailable:'))
                    power.write(f'{epoch:.6f}\t{float(value)}\t{mem}\n')
                except Exception as error:
                    telemetry_errors.append(str(error))
                    event('power_error', error=str(error))
                if pid is not None and matches(pid):
                    try:
                        record = {'epoch': epoch, 'pid': pid,
                            'stat': Path('/proc', str(pid), 'stat').read_text(),
                            'io': Path('/proc', str(pid), 'io').read_text()}
                        proc.write(json.dumps(record) + '\n')
                    except Exception as error:
                        telemetry_errors.append(str(error))
                        event('proc_error', pid=pid, error=str(error))
                next_sample += 1
            time.sleep(0.1)
finally:
    finish_netdata()
    if nd.is_alive() or telemetry_errors:
        raise RuntimeError('required telemetry incomplete or collector did not finish: ' + repr(telemetry_errors))
    marker = out / 'workload_done'
    if not marker.is_file() or int(marker.read_text()) != 0:
        raise RuntimeError('profile workload did not complete successfully')
    receipt = publish_profile(local_profile, out / profile_name)
    event('profile_published', **receipt)
    event('observer_done')
