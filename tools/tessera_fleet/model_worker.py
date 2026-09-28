"""The worker side of a whole-model Tessera serving export on PrismaBuild.

``tools.tessera_fleet.dispatch_model`` copies this file into a sealed
workspace as ``worker.py``, and every PrismaBuild action runs it there, or
inside the pinned producer image, with the standard library alone. It owns no
queue: the dispatcher supplies all work through ``pbcampaign``.

Moved from PrismaBuild's ``src/prismabuild/tessera_model.py`` on 2026-09-28
(RobTand/prismabuild#1076). Every result the dispatcher reads is a record
this worker writes beside the bytes it describes -- ``pb-result.json`` in each
part and in the assembled output, and one prepared identity per host under
``<out>.prepare/`` -- because PrismaBuild's public commands report whether an
action ran, not what it printed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import uuid

RESULT_PREFIX = 'PB_TESSERA_RESULT='
#: Unchanged by the move: it names records already written beside exports.
SCHEMA = 'prismabuild.tessera-model.v1'


def prepared_path(spec, host):
    """Where the prepare action on ``host`` records the source identity."""
    return Path(spec['out'] + '.prepare') / f'{host}.json'


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def producer_parts(root):
    # Inventory/identity uses the producer's stdlib module, without importing
    # torch or executing GPU work on the submitter.
    path = Path(root) / 'src/tessera/serving_parts.py'
    spec = importlib.util.spec_from_file_location('pb_tessera_serving_parts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('source_identity', 'source_inventory', 'partition_owner', 'merge_serving_parts'):
        if not callable(getattr(module, name, None)):
            raise ValueError(f'producer lacks serving-part contract: {name}')
    return module


def partitions(tensors, producer):
    """Smallest whole-layer ownership units supported by this producer.

    Contiguous models get one layer per action. For sparse layer numbering,
    choose the largest modulo domain with no empty owner, never create an
    invalid producer partition or ask the application for a shard count.
    """
    layers = {int(match.group(1)) for name in tensors
              if (match := producer.BODY_LAYER.match(name))}
    if not layers:
        raise ValueError('source has no whole-layer export work')
    for count in range(len(layers), 0, -1):
        owners = {producer.partition_owner(name, count) for name in tensors}
        if owners == set(range(count)):
            return count
    raise ValueError('producer cannot partition this source')


def stamp(path):
    info = Path(path).stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def verified_inputs(source, identity, cache_root):
    """Hash once per worker and unchanged file identity; changes fail closed.

    The cache is private to the worker uid, locked across admitted children,
    and bound to expected digests. Before/after stat checks reject mutation.
    This is cooperative filesystem identity, not hostile-writer immutability.
    """
    source, cache_root = Path(source).resolve(), Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = cache_root.lstat()
    if not cache_root.is_dir() or cache_root.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('unsafe source verification cache')
    expected = {**identity['auxiliary_sha256'], **identity['files']}
    expected['config.json'] = identity['config_sha256']
    cache_path = cache_root / (digest_json([str(source), expected]) + '.json')
    lock_path = cache_path.with_suffix('.lock')
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    snapshots = {}
    with os.fdopen(descriptor, 'r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            cached = read_json(cache_path)
        except (OSError, ValueError):
            cached = {}
        for name, digest in sorted(expected.items()):
            if Path(name).name != name:
                raise ValueError('source manifest contains a nonlocal filename')
            path = source / name
            before = stamp(path)
            if cached.get(name) != {'stamp': before, 'sha256': digest}:
                if digest_file(path) != digest or stamp(path) != before:
                    raise ValueError(f'source changed: {name}')
            snapshots[name] = {'stamp': before, 'sha256': digest}
        atomic_json(cache_path, snapshots)
    return snapshots


def check_stamps(source, snapshots):
    for name, record in snapshots.items():
        if stamp(Path(source) / name) != record['stamp']:
            raise ValueError(f'source changed during export: {name}')


def verify_output_record(path, expected=None):
    path = Path(path)
    record = read_json(path / 'pb-result.json')
    if expected is not None and record != expected:
        raise ValueError('part differs from its barrier record')
    if {p.name for p in path.iterdir() if p.is_file() and p.name != 'pb-result.json'} != set(record['files']):
        raise ValueError('output file population changed')
    for name, digest in record['files'].items():
        if Path(name).name != name or digest_file(path / name) != digest:
            raise ValueError(f'output changed: {name}')
    return record


def output_record(out, spec, *, index=None):
    files = {p.name: digest_file(p) for p in sorted(Path(out).iterdir()) if p.is_file()}
    return {'schema': SCHEMA, 'contract': spec['contract'], 'index': index,
            'count': spec['count'], 'files': files}


def inside(spec, mode, index, destination):
    source = Path(spec['source'])
    sys.path.insert(0, str(Path('encoder/src').resolve()))
    import tessera.serving_parts as parts
    cache = Path('/tmp') / f'prismabuild-source-verification-{os.getuid()}'
    snapshots = verified_inputs(source, spec['source_identity'], cache)
    if parts.source_inventory(source) != spec['source_identity']['tensors']:
        raise ValueError('source tensor inventory changed')
    original_hash = parts.sha256_file
    def memoized_hash(path):
        path = Path(path)
        if path.parent.resolve() == source.resolve() and path.name in snapshots:
            record = snapshots[path.name]
            if stamp(path) != record['stamp']:
                raise ValueError(f'source changed: {path.name}')
            return record['sha256']
        return original_hash(path)
    parts.sha256_file = memoized_hash
    if mode == 'encode':
        sys.argv = ['encoder/experiments/export_tessera_serving.py', str(source), destination,
                    '--plan-json', 'plan.json', '--grid', spec['grid'], '--q256', str(spec['q256']),
                    '--partition', f'{index}/{spec["count"]}',
                    '--partition-runtime-image', spec['image']]
        if spec.get('scales'):
            sys.argv += ['--input-scales', 'scales.safetensors']
        namespace = runpy.run_path(sys.argv[0], run_name='pb_tessera_export')
        namespace['main'].__globals__['git_hash'] = lambda: spec['encoder_commit']
        namespace['main']()
        manifest = read_json(Path(destination) / 'tessera_serving_manifest.json')
        if manifest['export_partition']['identity']['source'] != spec['source_identity']:
            raise ValueError('producer changed source identity')
    else:
        paths = []
        records = read_json('barrier.json')
        if len(records) != spec['count'] or [r['index'] for r in records] != list(range(spec['count'])):
            raise ValueError('assembly requires exactly one receipt for every partition')
        for part_index, record in enumerate(records):
            if record['contract'] != spec['contract']:
                raise ValueError('assembly contract mismatch')
            path = Path(spec['parts']) / f'part-{part_index:05d}'
            verify_output_record(path, record)
            paths.append(path)
        parts.merge_serving_parts(paths, Path(destination), source)
    check_stamps(source, snapshots)
    record = output_record(destination, spec, index=index if mode == 'encode' else None)
    atomic_json(Path(destination) / 'pb-result.json', record)


def verify_image(image):
    if not re.fullmatch(r'[^\s]+@sha256:[0-9a-f]{64}', image):
        raise ValueError('producer image must be repository@sha256 digest')
    observed = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image], text=True))[0]
    if image not in observed.get('RepoDigests', []):
        raise ValueError('worker lacks the exact producer image digest')
    return observed['Id']


def run_export_action(spec, mode, index):
    if mode == 'encode' and not 0 <= index < spec['count']:
        raise ValueError('invalid partition index')
    final = Path(spec['parts']) / f'part-{index:05d}' if mode == 'encode' else Path(spec['out'])
    if final.exists():
        record = verify_output_record(final)
        if record['contract'] != spec['contract'] or record['index'] != (index if mode == 'encode' else None):
            raise ValueError('existing output belongs to another export')
        return record
    image = spec['image']
    verify_image(image)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / ('.' + final.name + '.attempt-' + uuid.uuid4().hex)
    cache = Path('/tmp') / f'prismabuild-source-verification-{os.getuid()}'
    cache.mkdir(mode=0o700, exist_ok=True)
    kernel_cache = cache / 'kernels' / digest_json([spec['encoder_commit'], image])
    kernel_cache.mkdir(parents=True, exist_ok=True)
    argv = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
            '--entrypoint', 'python3', '-e', 'PYTHONPATH=/job/encoder/src',
            '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1',
            '-e', 'MAX_JOBS=1', '-e', 'CMAKE_BUILD_PARALLEL_LEVEL=1',
            '-e', 'HOME=/tmp', '-e', f'TRITON_CACHE_DIR={kernel_cache}/triton',
            '-e', f'TORCH_EXTENSIONS_DIR={kernel_cache}/torch',
            '-v', f'{Path.cwd()}:/job:ro', '-w', '/job',
            '-v', f'{spec["source"]}:{spec["source"]}:ro',
            '-v', f'{final.parent}:{final.parent}:rw', '-v', f'{cache}:{cache}:rw']
    if mode == 'encode':
        argv += ['--gpus', 'all']
    else:
        argv += ['-e', 'CUDA_VISIBLE_DEVICES=', '-v', f'{spec["parts"]}:{spec["parts"]}:ro']
    argv += [image, 'worker.py', 'inside', '--mode', mode, '--index', str(index), '--destination', str(temporary)]
    try:
        subprocess.run(argv, check=True)
        record = verify_output_record(temporary)
        if final.exists():
            raise ValueError('output appeared during export; refusing to replace it')
        os.rename(temporary, final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'encode', 'assemble', 'inside'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--mode', choices=['encode', 'assemble'])
    parser.add_argument('--destination')
    parser.add_argument('--host', help='prepare: the placement tag this action was pinned to')
    args = parser.parse_args(argv)
    spec = read_json('job.json')
    if spec.get('adapter_sha256') and digest_file(__file__) != spec['adapter_sha256']:
        raise ValueError('adapter changed')
    if spec.get('contract') and digest_json({k: v for k, v in spec.items() if k != 'contract'}) != spec['contract']:
        raise ValueError('export contract changed')
    if digest_file('plan.json') != spec['plan_sha256']:
        raise ValueError('plan changed')
    if spec.get('scales') and digest_file('scales.safetensors') != spec['scales']:
        raise ValueError('input scales changed')
    if args.command == 'prepare':
        verify_image(spec['image'])
        parts = producer_parts('encoder')
        source = Path(spec['source'])
        before = {p.name: stamp(p) for p in source.iterdir() if p.is_file()}
        identity = parts.source_identity(source)
        after = {p.name: stamp(p) for p in source.iterdir() if p.is_file()}
        if before != after:
            raise ValueError('source changed during preparation')
        result = {'source_identity': identity, 'count': partitions(identity['tensors'], parts)}
        if not args.host or Path(args.host).name != args.host:
            raise ValueError('prepare needs --host, the tag it was pinned to')
        atomic_json(prepared_path(spec, args.host), result)
    elif args.command == 'inside':
        inside(spec, args.mode, args.index, args.destination)
        return 0
    else:
        result = run_export_action(spec, args.command, args.index)
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
