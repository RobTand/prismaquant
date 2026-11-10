"""Validate workload-attributed samples and atomically publish their bytes."""
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import TypeGuard


_PROFILE_OWNERS = {
    'pq_profile_shared_digest_owner': 'digests',
    'pq_profile_shared_sampler_owner': 'io_spans',
    'pq_profile_shared_file_identity_owner': 'file_identity',
}


def load_profile_source(owner_name: str, path: Path) -> ModuleType:
    """Load only the profiler's three stdlib owners from this snapshot."""
    source = _PROFILE_OWNERS[owner_name]
    expected = Path(__file__).resolve().parents[1].joinpath('prismaquant', source + '.py')
    if path.resolve() != expected:
        raise KeyError(str(path))
    if owner_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(owner_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError('profile source owner unavailable: ' + str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[owner_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(owner_name, None)
            raise
    return sys.modules[owner_name]


def profile_digest_owner():
    path = Path(__file__).resolve().parents[1] / 'prismaquant/digests.py'
    return load_profile_source('pq_profile_shared_digest_owner', path)


bytes_sha256hex = profile_digest_owner().bytes_sha256hex
file_sha256hex = profile_digest_owner().file_sha256hex
canonical_json_sha256 = profile_digest_owner().canonical_json_sha256
canonical_json_bytes = profile_digest_owner().canonical_json_bytes


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




PROFILE_LOCAL_ROOT = Path('/home/rob/tmp/claude-campaign-20260926/tmp/row-startup/profiles')


def profile_row_parts(command):
    """Locate the existing row boundary without changing command operands."""
    boundary = 0
    if 'tools.tessera_campaign_container' in command:
        module = command.index('tools.tessera_campaign_container')
        boundary = command.index('--', module) + 1
    inner = command[boundary:]
    if not inner or not Path(inner[0]).name.startswith('python'):
        raise ValueError('profiling requires the original Python row argv')
    return boundary, inner


def profiled_row_command(command, *, destination, profiler_executable):
    """Keep container ownership and original argv; profile its same-UID child."""
    boundary, inner = profile_row_parts(command)
    return [*command[:boundary], inner[0], '-u', '-m', 'tools.pq_profile_child',
            '--profiler-executable', profiler_executable, '--out', str(destination),
            '--', *inner]


def profiled_workload_parts(command, *, destination, profiler_executable):
    """Accept only the exact existing canonical child wrapper; keep full argv."""
    boundary, inner = profile_row_parts(command)
    if len(inner) < 10 or inner[8] != '--':
        raise RuntimeError('namespace profile requires the canonical child wrapper')
    original = [*command[:boundary], *inner[9:]]
    if command != profiled_row_command(original, destination=destination,
                                       profiler_executable=profiler_executable):
        raise RuntimeError('namespace profile wrapper differs from its declaration')
    return boundary + 9


def profile_local_destination(request_key):
    """Same existing host-local policy; namespace owns one deterministic child."""
    return PROFILE_LOCAL_ROOT / request_key
