"""Finite CPU transport fixture; the arithmetic is not a model-quality probe."""
import argparse
import gzip  # Staged authentication also imports it; freeze one phase runtime.
import hashlib
import json
import os
from pathlib import Path
import select
import struct
import sys
import time

from prismaquant import native_results
from prismaquant import quality_prefill_contract as scientific
from prismaquant import quality_prefill_flux as prefill


def arithmetic(raw):
    if len(raw) != 128:
        raise ValueError('fixture requires exactly 32 float32 values')
    values = struct.unpack('<32f', raw)
    if values != tuple(float(i) for i in range(1, 33)):
        raise ValueError('fixture input differs from its independent known values')
    return sum(values)


def release(fifo):
    descriptor = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        ready, _, _ = select.select([descriptor], [], [], 90)
        if not ready or os.read(descriptor, 1) != b'!':
            raise RuntimeError('owned fixture release missing')
    finally:
        os.close(descriptor)


def prepare(directory):
    """Measure the same program/import roster and its small arithmetic workload."""
    root = Path(directory)
    begin = time.perf_counter_ns()
    raw = (root / 'known-input.bin').read_bytes()
    task_begin = time.perf_counter_ns()
    value = arithmetic(raw)
    end = time.perf_counter_ns()
    assert value == 528
    # Resolve lazy metadata/platform imports exactly as the publication owner
    # does before recording the immutable expected runtime.
    before = native_results.measure_runtime()
    prefill.flux_jobs._source_git(Path.cwd(), 'rev-parse', 'HEAD')
    prefill.flux_jobs._profile_identity()
    native_results.publish_staged_json(root / 'prepared-native-document.json', {'fixture': 'native-writer'})
    runtime = native_results.measure_runtime()
    (root / 'preflight-runtime-delta.json').write_text(json.dumps(dict(
        added_imports=sorted(set(runtime['imports']) - set(before['imports'])),
        removed_imports=sorted(set(before['imports']) - set(runtime['imports'])),
        changed_imports=sorted(name for name in before['imports']
            if before['imports'][name] != runtime['imports'].get(name))), sort_keys=True) + '\n')
    if runtime != native_results.measure_runtime():
        raise RuntimeError('fixture runtime measurement is not stable')
    (root / 'runtime.json').write_text(json.dumps(runtime, sort_keys=True) + '\n')
    (root / 'timing.json').write_text(json.dumps(dict(
        schema='pq.native_prefill_cpu_fixture_timing.v1',
        clock='perf_counter_ns', setup_seconds=(task_begin - begin) / 1_000_000_000,
        task_seconds=(end - task_begin) / 1_000_000_000,
        input_sha256=hashlib.sha256(raw).hexdigest(), value=528,
        scope='32-element CPU arithmetic only; excludes admission/authentication'),
        sort_keys=True) + '\n')


def execute(binding):
    envelope = prefill.read_task_envelope(binding)
    _, _, roster = prefill._read_phase(envelope['manifest'], envelope['phase_id'])
    control = json.loads(native_results._owned(roster['inputs']['control'], 4096))
    if set(control) != {'mode', 'release_before', 'release_after'} or control['mode'] not in {
            'success', 'late-failure', 'cancel'}:
        raise ValueError('invalid finite fixture controls')
    record = Path(os.environ['OUT'])
    (record / 'fixture-process.json').write_text(json.dumps(dict(
        pid=os.getpid(), pgid=os.getpgrp(), affinity=sorted(os.sched_getaffinity(0)))))
    print('fixture-ready', flush=True)
    release(control['release_before'])
    raw = native_results._owned(roster['inputs']['known'], 128)
    value = arithmetic(raw)
    artifact = Path(envelope['phase_record_dir']) / ('known-' + envelope['child_id'] + '.bin')
    with artifact.open('xb') as stream:
        stream.write(raw)
    native_results.execution_context()
    measured = native_results.measure_runtime()
    (record / 'fixture-runtime.json').write_text(json.dumps(measured, sort_keys=True) + '\n')
    if measured != roster['runtime']:
        raise RuntimeError('fixture runtime differs before publication')
    artifact_binding = dict(path=str(artifact), bytes=len(raw),
                            sha256=hashlib.sha256(raw).hexdigest())
    results = []
    for task in envelope['tasks']:
        measurement = dict(schema=scientific.MEASUREMENT_SCHEMA, payload=task['payload'],
            units='fixture_sum', measurement_status='measured', value=value,
            artifacts=[artifact_binding])
        results.append(dict(task_id=task['task_id'], output_id=task['output_id'],
            value=measurement, value_sha256=scientific.canonical_sha256(measurement)))
    try:
        prefill.publish_task_results(binding, results=results)
    except Exception:
        (record / 'fixture-runtime-after-error.json').write_text(json.dumps(
            native_results.measure_runtime(), sort_keys=True) + '\n')
        raise
    print('fixture-published', flush=True)
    release(control['release_after'])
    if control['mode'] == 'late-failure':
        return 7
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare')
    parser.add_argument('--envelope')
    parser.add_argument('--envelope-sha256')
    args = parser.parse_args()
    if not sys.flags.no_user_site:
        raise RuntimeError('isolated Python required')
    if args.prepare is not None:
        if args.envelope is not None or args.envelope_sha256 is not None:
            raise ValueError('prepare and execute are separate finite modes')
        prepare(args.prepare)
        return 0
    raw, _ = prefill.read_document(args.envelope, prefill.MAX_ENVELOPE_BYTES)
    return execute(dict(path=args.envelope, sha256=args.envelope_sha256, bytes=len(raw)))


if __name__ == '__main__':
    raise SystemExit(main())
