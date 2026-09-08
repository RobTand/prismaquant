"""Bounded evidence contracts for the experimental original-wire screen."""
from __future__ import annotations

import math
import threading
import time

HOSTS = ('sparky', 'sparklina')
TELEMETRY_POLICY = dict(schema='prismaquant.glm_screen_telemetry_coverage.v1',
    sample_interval_seconds=2, max_sample_gap_seconds=15,
    max_chart_age_seconds=20, max_future_chart_seconds=2,
    max_samples_per_host=2000, max_error_records=8, max_output_bytes=256*1024**2)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def telemetry_sample(record, *, monotonic):
    """Retain timing/freshness only; full bounded chart records go to JSONL."""
    from experiments.workspace_netdata import REQUIRED_CHARTS, REQUIRED_GPU_SUFFIXES
    charts = record['metrics']
    power = {name for name in charts if name.startswith('nvidia_smi.') and name.endswith('_power_draw')}
    required = set(REQUIRED_CHARTS) | power
    required.update(name.removesuffix('_power_draw')+'_'+suffix
                    for name in power for suffix in REQUIRED_GPU_SUFFIXES)
    if not power or not required <= charts.keys():
        raise ValueError('telemetry lacks a required CPU/memory/GPU chart')
    now = record['time']
    updated = [charts[name]['last_updated'] for name in sorted(required)]
    if not all(_finite(value) for value in (now, monotonic, *updated)):
        raise ValueError('telemetry timestamps must be finite numbers')
    age = now-min(updated)
    future = max(updated)-now
    if (age > TELEMETRY_POLICY['max_chart_age_seconds'] or
            future > TELEMETRY_POLICY['max_future_chart_seconds']):
        raise ValueError('required telemetry charts are stale or clock-skewed')
    return dict(time=now, monotonic=monotonic, oldest_chart_age_seconds=age,
                max_future_chart_seconds=future, required_charts=len(required))


def telemetry_coverage(samples, phases, errors, *, thread_complete):
    """Require fresh, uninterrupted, bracketed observations for every phase."""
    report = dict(schema=TELEMETRY_POLICY['schema'], passed=False,
                  policy=dict(TELEMETRY_POLICY), failures=[], phases=[])
    failures = report['failures']
    if not thread_complete:
        failures.append('telemetry thread did not stop')
    if errors:
        failures.append('one or more required telemetry reads/writes failed')
    if not phases:
        failures.append('no measured phases')
    limit = TELEMETRY_POLICY['max_sample_gap_seconds']
    for host in HOSTS:
        rows = samples.get(host, ())
        if not 2 <= len(rows) <= TELEMETRY_POLICY['max_samples_per_host']:
            failures.append(host+': missing or excessive observations')
            continue
        try:
            times = [row['monotonic'] for row in rows]
            if (not all(_finite(value) for value in times) or
                    any(right <= left for left, right in zip(times, times[1:])) or
                    any(not _finite(row[key]) for row in rows for key in
                        ('oldest_chart_age_seconds', 'max_future_chart_seconds')) or
                    any(row['oldest_chart_age_seconds'] > TELEMETRY_POLICY['max_chart_age_seconds'] or
                        row['max_future_chart_seconds'] > TELEMETRY_POLICY['max_future_chart_seconds']
                        for row in rows)):
                raise ValueError('invalid, stale or unordered observations')
            for phase in phases:
                start, end = phase['started_monotonic'], phase['finished_monotonic']
                if not all(_finite(value) for value in (start, end)) or end < start:
                    raise ValueError('invalid phase interval')
                before = [index for index, value in enumerate(times) if value <= start]
                after = [index for index, value in enumerate(times) if value >= end]
                if not before or not after:
                    failures.append(f"{host}/{phase['phase']}: phase is not bracketed")
                    continue
                left, right = before[-1], after[0]
                window = times[left:right+1]
                max_gap = max([start-times[left], times[right]-end,
                               *(b-a for a, b in zip(window, window[1:]))])
                if max_gap > limit:
                    failures.append(f"{host}/{phase['phase']}: coverage gap exceeds {limit} seconds")
                report['phases'].append(dict(host=host, phase=phase['phase'],
                    before_monotonic=times[left], after_monotonic=times[right],
                    observations=len(window), max_gap_seconds=max_gap))
        except (KeyError, TypeError, ValueError) as error:
            failures.append(host+': '+str(error))
    report['passed'] = not failures
    return report


class ScreenTelemetry:
    """One serialized Netdata reader with bounded retained timing records."""
    def __init__(self, path, *, sample=None):
        from experiments.workspace_netdata import sample_netdata
        self.path, self.sample = path, sample or sample_netdata
        self.samples = {host: [] for host in HOSTS}
        self.errors = []
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.thread = self.writer = self.stream = None

    def collect(self):
        with self.lock:
            if self.errors:
                return
            for host in HOSTS:
                try:
                    if len(self.samples[host]) >= TELEMETRY_POLICY['max_samples_per_host']:
                        raise RuntimeError('telemetry observation bound exceeded')
                    record = self.sample(host)
                    if record['host'] != host:
                        raise ValueError('telemetry host identity changed')
                    observed = telemetry_sample(record, monotonic=time.monotonic())
                    self.writer.write(record)
                    self.samples[host].append(observed)
                except Exception as error:
                    if len(self.errors) < TELEMETRY_POLICY['max_error_records']:
                        self.errors.append(dict(host=host, error=str(error), unix=time.time()))
                    self.stop.set()
                    return

    def require_healthy(self):
        if self.errors:
            raise RuntimeError('screen telemetry failed: '+self.errors[0]['error'])

    def start(self):
        from experiments.workspace_netdata import NetdataWriter
        self.stream = self.path.open('x')
        self.writer = NetdataWriter(self.stream, max_bytes=TELEMETRY_POLICY['max_output_bytes'])
        self.collect()
        self.require_healthy()
        def observe():
            while not self.stop.wait(TELEMETRY_POLICY['sample_interval_seconds']):
                self.collect()
        self.thread = threading.Thread(target=observe, daemon=True)
        self.thread.start()

    def finish(self, phases):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
        complete = self.thread is None or not self.thread.is_alive()
        if complete and self.stream is not None:
            self.stream.close()
        return telemetry_coverage(self.samples, phases, self.errors, thread_complete=complete)


def require_original_input_bytes(tensors, expected, *, identity):
    """Once per unit, catch custom/unversioned writes to source, H or X."""
    observed = {name: identity(tensor) for name, tensor in tensors.items()}
    if observed != expected:
        changed = sorted(name for name in expected if observed.get(name) != expected[name])
        raise RuntimeError('original input bytes changed: '+', '.join(changed))
    return observed
