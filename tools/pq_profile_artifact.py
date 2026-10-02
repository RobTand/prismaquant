"""Validate workload-attributed samples and atomically publish their bytes."""
import json
import math
from pathlib import Path
import re
from typing import TypeGuard

from tools.pq_profile_digest import bytes_sha256hex, file_sha256hex


_STATUS_SCHEMA = 'prismaquant.profile_workload_status.v1'
_PROCESS_NAME = re.compile(r'Process ([1-9][0-9]*) Thread \S+ ".*"')


def workload_status_path(profile):
    profile = Path(profile)
    return profile.with_name(profile.name + '.workload-status.json')


def read_workload_status(profile):
    """PID is written by the child in the profiler's own PID namespace."""
    record = json.loads(workload_status_path(profile).read_text())
    if (not isinstance(record, dict) or record.get('schema') != _STATUS_SCHEMA
            or type(record.get('returncode')) is not int
            or type(record.get('workload_pid')) is not int
            or record['workload_pid'] <= 0):
        raise RuntimeError('child profiler has no valid workload completion record')
    return record


def finish_profile_status(profile, profiler_returncode):
    """Keep the worker's original result separate from profiler shutdown."""
    if type(profiler_returncode) is not int:
        raise RuntimeError('profiler completion has no valid return code')
    record = read_workload_status(profile)
    record['profiler_returncode'] = profiler_returncode
    status = workload_status_path(profile)
    partial = status.with_name(status.name + '.finalizing')
    with partial.open('x') as handle:
        json.dump(record, handle)
    partial.replace(status)
    return record


def _profile_finite_number(value: object) -> TypeGuard[int | float]:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def validate_workload_samples(data, workload_pid):
    """Closed-world py-spy 0.4.2 --subprocesses sampled-format contract."""
    if type(workload_pid) is not int or workload_pid <= 0:
        raise RuntimeError('sampled profile has no valid workload PID')
    if (not isinstance(data, dict)
            or data.get('$schema') != 'https://www.speedscope.app/file-format-schema.json'):
        raise RuntimeError('py-spy output is not a speedscope profile')
    shared = data.get('shared')
    if not isinstance(shared, dict) or not isinstance(shared.get('frames'), list):
        raise RuntimeError('py-spy output has no shared frame table')
    frames = shared['frames']
    for frame in frames:
        if (not isinstance(frame, dict) or not isinstance(frame.get('name'), str)
                or not frame['name']
                or (frame.get('file') is not None and not isinstance(frame['file'], str))
                or any(frame.get(field) is not None
                       and (type(frame[field]) is not int or frame[field] < 0)
                       for field in ('line', 'col'))):
            raise RuntimeError('py-spy output has an invalid shared frame')
    profiles = data.get('profiles')
    if not isinstance(profiles, list) or not profiles:
        raise RuntimeError('py-spy output has no sampled profiles')
    attributed = 0
    for profile in profiles:
        if not isinstance(profile, dict) or profile.get('type') != 'sampled':
            raise RuntimeError('py-spy output has an invalid sampled profile')
        name = profile.get('name')
        match = _PROCESS_NAME.fullmatch(name) if isinstance(name, str) else None
        if match is None:
            raise RuntimeError('sampled profile has no workload process attribution')
        start, end = profile.get('startValue'), profile.get('endValue')
        if (profile.get('unit') != 'seconds' or not _profile_finite_number(start)
                or not _profile_finite_number(end) or start < 0 or end < start):
            raise RuntimeError('sampled profile has an invalid time range')
        samples, weights = profile.get('samples'), profile.get('weights')
        if (not isinstance(samples, list) or not isinstance(weights, list)
                or len(samples) != len(weights)):
            raise RuntimeError('sampled profile has invalid sample weights')
        for stack, weight in zip(samples, weights):
            if (not isinstance(stack, list)
                    or any(type(index) is not int or not 0 <= index < len(frames)
                           for index in stack)):
                raise RuntimeError('sampled profile has an invalid frame reference')
            if not _profile_finite_number(weight) or weight <= 0:
                raise RuntimeError('sampled profile has an invalid sample weight')
            if int(match.group(1)) == workload_pid and stack:
                attributed += 1
    if not attributed:
        raise RuntimeError('py-spy output has no sampled profiles attributed to the workload')
    return {'profiles': len(profiles), 'workload_pid': workload_pid,
            'workload_samples': attributed}


def publish_profile(source, destination):
    payload = Path(source).read_bytes()
    # Do not compare a host telemetry PID with container trace PIDs. The status
    # worker records its actual child's PID beside the trace, in that namespace.
    status = read_workload_status(source)
    if status['returncode'] != 0:
        raise RuntimeError('profile workload did not complete successfully')
    if type(status.get('profiler_returncode')) is not int or status['profiler_returncode'] != 0:
        raise RuntimeError('profile profiler did not complete successfully')
    receipt = validate_workload_samples(json.loads(payload), status['workload_pid'])
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('refusing to replace a published profile')
    partial = destination.with_name(destination.name + '.copying')
    with partial.open('xb') as handle:
        handle.write(payload)
    partial.replace(destination)
    digest = bytes_sha256hex(payload)
    if file_sha256hex(destination) != digest:
        raise RuntimeError('published py-spy bytes differ from workload output')
    return {'sha256': digest, 'bytes': len(payload), **receipt}
