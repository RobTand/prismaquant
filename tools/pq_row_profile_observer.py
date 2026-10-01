#!/usr/bin/env python3
"""Read-only observer started inside an admitted PQ row (#1899).

Derived from the qualified #1654 observer; supports either GB10 host. No Docker
calls, workload launch or row signal. The owner joins this observer after the
workload. Output never enters a wire, cost row or anchor identity.
"""
import argparse
from prismaquant.io_spans import PeriodicSampler
from tools.pq_profile_digest import bytes_sha256hex, file_sha256hex
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

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--target-out', required=True)
p.add_argument('--wait-s', type=int, default=14400)
p.add_argument('--row-s', type=int, default=43200)
p.add_argument('--profile-s', type=int, default=420)
p.add_argument('--profile-local', required=True,
               help='New host-local campaign directory for privileged py-spy output')
a = p.parse_args()
local_host = socket.gethostname().split('.')[0]
if local_host not in ('sparky', 'sparklina'):
    raise RuntimeError('observer requires an admitted GB10 host')
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
profile_dir = Path(a.profile_local).resolve()
local_root = Path('/home/rob/tmp/claude-campaign-20260926/tmp/row-startup/profiles').resolve()
if not profile_dir.is_relative_to(local_root) or profile_dir == local_root:
    raise ValueError('profile-local must be a new directory below the host-local campaign profiles root')
profile_dir.mkdir(parents=True, exist_ok=False)
profile_name = 'boundary-' + local_host + '.speedscope'
local_profile = profile_dir / profile_name
lock = threading.Lock()
collection_ready = threading.Event()
telemetry_errors = []


def event(name, **data):
    with lock, (out / 'events.jsonl').open('a') as f:
        f.write(json.dumps({'epoch': time.time(), 'event': name, **data}) + '\n')


def command(args, timeout=10):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          timeout=timeout).stdout


def profiler_command(pid, seconds, destination):
    # The sudo process cannot write the rob-owned NFS directory (UID0 is
    # denied there). Only the unprivileged observer publishes to that mount.
    return ['sudo', '-n', 'py-spy', 'record', '--idle', '--threads', '--subprocesses',
            '--rate', '50', '--duration', str(seconds), '--format', 'speedscope',
            '--pid', str(pid), '-o', str(destination)]


def publish_profile(source, destination):
    payload = Path(source).read_bytes()
    data = json.loads(payload)
    if not isinstance(data, dict) or data.get('$schema') != 'https://www.speedscope.app/file-format-schema.json':
        raise RuntimeError('py-spy output is not a speedscope profile')
    shared = data.get('shared')
    if not isinstance(shared, dict) or not isinstance(shared.get('frames'), list):
        raise RuntimeError('py-spy output has no shared frame table')
    profiles = data.get('profiles')
    if not isinstance(profiles, list) or not profiles or not any(
            isinstance(profile, dict) and profile.get('samples') for profile in profiles):
        raise RuntimeError('py-spy output has no sampled profiles')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('refusing to replace a published profile')
    partial = destination.with_name(destination.name + '.copying')
    with partial.open('xb') as handle:
        handle.write(payload)
    partial.replace(destination)
    digest = bytes_sha256hex(payload)
    if file_sha256hex(destination) != digest:
        raise RuntimeError('published py-spy bytes differ from host-local output')
    return {'sha256': digest, 'bytes': len(payload), 'profiles': len(profiles)}


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
# A read-only access check before ready prevents wasting an approved GPU row
# on an unwritable profiler destination; it does not launch a workload.
command(['sudo', '-n', 'test', '-w', str(profile_dir)])
version = command(['sudo', '-n', 'py-spy', '--version'])
first_power = command(['nvidia-smi', '--query-gpu=power.draw',
                       '--format=csv,noheader,nounits'])
float(first_power.strip())


def validate_netdata_window(data, *, after, before):
    """Require fresh finite samples for every declared chart dimension."""
    if not isinstance(data, dict):
        raise RuntimeError('required Netdata response is not an object')
    labels, rows = data.get('labels'), data.get('data')
    if (not isinstance(labels, list) or len(labels) < 2 or labels[0] != 'time'
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels) or not isinstance(rows, list) or not rows):
        raise RuntimeError('required Netdata labels or samples are missing')
    measured = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != len(labels):
            raise RuntimeError('required Netdata sample has malformed dimensions')
        stamp = row[0]
        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                or not after <= stamp <= before):
            raise RuntimeError('required Netdata sample is outside its requested window')
        for index, value in enumerate(row[1:], 1):
            if value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise RuntimeError('required Netdata sample is not finite numeric telemetry')
            measured.add(index)
    if measured != set(range(1, len(labels))):
        raise RuntimeError('required Netdata dimensions have no measured samples')


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
spy = None
spylog = None
profile_handled = False
profile_error = None
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
                    spylog = (out / 'pyspy.log').open('w')
                    spy = subprocess.Popen(profiler_command(pid, a.profile_s, local_profile),
                        stdout=spylog, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + a.row_s + 120
                    event('row_found', pid=pid, affinity=affinity, profiler_pid=spy.pid)
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
            if spy is not None and not profile_handled and spy.poll() is not None:
                profile_handled = True
                rc = spy.returncode
                event('pyspy_exited', returncode=rc)
                try:
                    if rc != 0:
                        raise RuntimeError('py-spy failed; profile evidence is incomplete')
                    receipt = publish_profile(local_profile, out / profile_name)
                    event('profile_published', **receipt)
                except Exception as error:
                    profile_error = str(error)
                    event('profile_error', error=profile_error)
                # Keep recording power/Netdata through row exit, even on a
                # profiler failure. Never signal the row or change its argv.
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
                            'io': command(['sudo', '-n', 'cat', f'/proc/{pid}/io'])}
                        proc.write(json.dumps(record) + '\n')
                    except Exception as error:
                        telemetry_errors.append(str(error))
                        event('proc_error', pid=pid, error=str(error))
                next_sample += 1
            time.sleep(0.1)
finally:
    finish_netdata()
    if spy is not None:
        # Own profiler only. Never signal or terminate the quantization row.
        rc = spy.wait(timeout=a.profile_s + 30)
        if not profile_handled:
            event('pyspy_exited', returncode=rc)
            try:
                if rc != 0:
                    raise RuntimeError('py-spy failed; profile evidence is incomplete')
                receipt = publish_profile(local_profile, out / profile_name)
                event('profile_published', **receipt)
            except Exception as error:
                profile_error = str(error)
                event('profile_error', error=profile_error)
        spylog.close()
        if profile_error is not None:
            raise RuntimeError(profile_error)
    if nd.is_alive() or telemetry_errors:
        raise RuntimeError('required telemetry incomplete or collector did not finish: ' + repr(telemetry_errors))
    event('observer_done')
