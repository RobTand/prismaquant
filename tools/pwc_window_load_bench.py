"""PWC retained-window load bench: what a Stage B render window's loads cost (PQ #1210).

GLM-5.3 Stage B row 041 spent 24.5% of its render windows' main-thread
samples waiting on its own window's loads, and 63% of the loader threads'
samples on per-file reader-lease work (PB ``4e1468a6d07e``). This tool runs
the same load path, ``ProductionWeightCache.retained_window`` under the
strict tier policy with the pinned PrismaBuild SDK, on real layer-041 render
files, once on each of two source trees, and compares them.

Modes:

* ``ab`` (the host process): materializes the base tree from ``--base-ref``
  with ``git archive``, runs ``quantum`` on each tree, builds one strict
  fixture, then runs ``window`` children alternating base and fix over
  ``--rounds``, each under ``py-spy record --nonblocking --idle --threads``.
  It samples GPU power with ``nvidia-smi`` for the whole run and writes
  ``summary.json`` under ``--out``.
* ``setup``: builds the strict fixture under ``--fixture``: a real queue
  layout with a claim row, one staged copy of each render, fragments and
  material published with PrismaBuild's own writers, and the residency map.
  The declared files are the real renders; they are only read.
* ``window``: attaches to the fixture exactly as ``_leased_fixture`` in
  ``tests/test_strict_reader_tier_enforcement.py`` does, then opens
  ``--windows`` retained windows over the renders. Each window drops the
  staged copies from this client's page cache first, so its reads reach the
  file server, then loads (timed), runs a fixed consumer, and exits. The
  consumer is ``--consumer sleep`` (default): the main thread sleeps
  ``--compute-s``, as a Stage B main thread waits on its GPU replay, and the
  action needs no GPU. ``--consumer gpu`` runs ``--probes`` matmuls per
  render at ``--batch`` rows instead. Window 0's loaded tensors, file-load
  receipts and render identities are digested.
* ``quantum``: runs the layer-1 joint quantum on the CPU stub campaign from
  ``tests/test_joint_cost_quantum_runtime.py`` and digests its payload and
  every file it writes, leaf by leaf: tensors and arrays by their bytes,
  floats by ``float.hex``. Two runs of one tree differ in wall times and run
  ids, so the host runs the fix tree twice and compares base with fix on
  every leaf those two runs agree on.
* ``identity``: only the ``quantum`` comparison, written to
  ``identity.json``; exits non-zero when base and fix differ.

The fixture's lease root holds one consumer, while production's holds every
action's, so the per-file lease cost it shows is a lower bound. The compute
is a stand-in of fixed size, identical on both arms. The default
``--compute-s`` is row 041's non-load main-thread time per render window
(about 30 s for 354 renders) scaled to ``--files``.

Writes only under ``--out`` and ``--fixture``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: Row 041's render windows: 354 renders each, and about 30 s of main-thread
#: time per window outside the window's loads (py-spy, PB ``4e1468a6d07e``).
ROW_041_RENDERS = 354
ROW_041_NONLOAD_S = 30.0
ROW_041_RECORD = ('/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/'
                  'r13-stageb-20260923/meta-045-fae344d/records/layer-041.json')


def _log(message):
    print(f"[pwc-bench {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Child environments
# ---------------------------------------------------------------------------

def _child_env(tree: Path, *, cuda: bool) -> dict:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('PRISMABUILD_RESIDENCY')}
    env['PYTHONPATH'] = os.pathsep.join([str(tree), str(tree / 'tests')])
    env['PYTHONHASHSEED'] = '0'
    env.setdefault('OMP_NUM_THREADS', '1')
    if not cuda:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def _attach_strict(fixture: dict):
    """Bind this process to the fixture as ``_leased_fixture`` does."""
    import test_strict_reader_tier_enforcement as strict
    from prismaquant.residency_map import (
        bind_residency_manifest, reset_residency_resolver_for_tests)
    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    for name in ('PRISMABUILD_ACTION_KEY', 'PRISMABUILD_ACTION_NONCE',
                 'PRISMABUILD_ACTION_SCOPE', 'PRISMABUILD_READER_HELPER_ROOT',
                 strict.TIERS_DIR_ENV_VAR):
        os.environ.pop(name, None)
    strict._pb()
    os.environ[strict.ENV_VAR] = fixture['map_path']
    os.environ['PRISMABUILD_ACTION_KEY'] = fixture['consumer']
    os.environ['PRISMABUILD_ACTION_NONCE'] = strict.LAUNCH_NONCE
    os.environ['PRISMABUILD_ACTION_SCOPE'] = strict.LAUNCH_SCOPE
    reset_residency_resolver_for_tests()
    bind_residency_manifest(strict.MANIFEST)
    activate_staged_tier_policy('ram,ssd')


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def _row_041_renders(record_path: str, count: int) -> list[str]:
    record = json.loads(Path(record_path).read_text())
    windows = record['executable_readset']['prepared_input']['windows']
    entries = windows[1]['entries']
    return [entry['path'] for entry in entries[:count]]


def run_setup(args):
    import test_strict_reader_tier_enforcement as strict
    from prismaquant.residency_map import residency_map_key
    fixture_root = Path(args.fixture)
    fixture_root.mkdir(parents=True, exist_ok=True)
    declared = [Path(path) for path in _row_041_renders(args.record, args.files)]
    rl, pool_mod, map_mod = strict._pb()
    consumer = strict._hex64(f"consumer-{fixture_root}")
    mover = strict._hex64(f"mover-{fixture_root}")
    _queue, stage = strict._pb_queue(fixture_root, pool_mod, consumer)
    started = time.monotonic()
    rows, entries, digests = {}, {}, {}
    for index, path in enumerate(declared):
        blob = path.read_bytes()
        staged = stage / path.name
        staged.write_bytes(blob)
        digests[str(path)] = hashlib.sha256(blob).hexdigest()
        rows[f"u{index}"] = (path, staged, None)
        entries[residency_map_key(str(path), 0)] = (path, staged)
    strict._pb_publish(rl, map_mod, fixture_root / 'residency', stage, consumer,
                       mover, strict.MANIFEST, entries)
    map_path = strict._write_map(fixture_root, rows, leads=[mover])
    fixture = {'map_path': str(map_path), 'consumer': consumer, 'mover': mover,
               'root': str(fixture_root), 'stage': str(stage),
               'declared': [str(path) for path in declared],
               'staged': [str(stage / path.name) for path in declared],
               'sha256': digests,
               'file_bytes': {str(path): path.stat().st_size for path in declared},
               'setup_s': time.monotonic() - started}
    Path(args.fixture_json).write_text(json.dumps(fixture, indent=1))
    _log(f"fixture: {len(declared)} renders, {sum(fixture['file_bytes'].values())} bytes, "
         f"{fixture['setup_s']:.1f} s")


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------

def _drop_client_pages(paths):
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _render_key(path: str):
    name = Path(path).name[:-len('.pt')]
    module, fmt = name.split('__', 1)
    return (module, fmt)


def run_window(args):
    import torch
    fixture = json.loads(Path(args.fixture_json).read_text())
    _attach_strict(fixture)
    gpu = args.consumer == 'gpu'
    import prismaquant.production_weight_cache as pwc
    keys = [_render_key(path) for path in fixture['declared']]
    cache = pwc.ProductionWeightCache(
        weights={key: path for key, path in zip(keys, fixture['declared'])}, levers={})
    total = sum(fixture['file_bytes'].values())
    cache.enable_lru(total)
    cache.require_file_load_sha256(
        {key: fixture['sha256'][path] for key, path in zip(keys, fixture['declared'])},
        max_file_bytes=max(fixture['file_bytes'].values()))
    device = torch.device('cuda') if gpu else None
    activations = {}
    if gpu:
        torch.cuda.synchronize()
    windows, digest = [], None
    for index in range(args.windows):
        _drop_client_pages(fixture['staged'])
        started = time.perf_counter()
        with cache.retained_window(
                keys, max_resident_bytes=total, max_workers=args.workers,
                max_load_buffer_bytes=args.load_buffer_bytes, release_file_pages=True,
                render_identities=True) as receipt:
            loaded = time.perf_counter()
            renders = {key: cache.get_resident(*key) for key in keys}
            if index == 0:
                digest = hashlib.sha256()
                for key in keys:
                    tensor = renders[key]
                    file_receipt = cache.file_load_receipt(key, tensor)
                    identity = cache.resident_render_identity(*key, tensor)
                    digest.update(json.dumps(
                        [list(key), list(tensor.shape), str(tensor.dtype),
                         {k: v for k, v in file_receipt.items() if k != 'path'},
                         identity], sort_keys=True).encode())
                    digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                digest = digest.hexdigest()
            digested = time.perf_counter()
            if gpu:
                _consume(torch, renders, activations, device, args)
            else:
                _consume_sleep(args.compute_s)
            computed = time.perf_counter()
            renders = tensor = None
            quanta = len(receipt['load_quanta'])
        exited = time.perf_counter()
        windows.append({'load_s': loaded - started, 'digest_s': digested - loaded,
                        'compute_s': computed - digested, 'exit_s': exited - computed,
                        'wall_s': exited - started - (digested - loaded),
                        'quanta': quanta, 'epoch_start': time.time() - (exited - started),
                        'epoch_end': time.time()})
        _log(f"window {index}: load {loaded - started:.3f} s, compute "
             f"{computed - digested:.3f} s, exit {exited - computed:.3f} s")
    counters = dict(getattr(pwc, 'PWC_WINDOW_LEASE_COUNTERS', {}) or {})
    from prismaquant.residency_map import residency_resolver
    report = residency_resolver().report()
    Path(args.out_json).write_text(json.dumps({
        'windows': windows, 'digest': digest, 'lease_counters': counters,
        'bytes_from_pool': report.get('bytes_from_pool'),
        'bytes_from_stage': report.get('bytes_from_stage'),
        'pid': os.getpid()}, indent=1))


def _consume_sleep(seconds):
    """The main thread waits, as it does on a Stage B window's GPU replay."""
    time.sleep(seconds)


def _consume(torch, renders, activations, device, args):
    """A fixed GPU consumer: ``--probes`` matmuls per resident render."""
    on_device = {key: tensor.to(device, non_blocking=False) for key, tensor in renders.items()}
    for _probe in range(args.probes):
        for tensor in on_device.values():
            columns = tensor.shape[-1]
            x = activations.get(columns)
            if x is None:
                generator = torch.Generator(device=device).manual_seed(columns)
                x = activations[columns] = torch.randn(
                    args.batch, columns, device=device, dtype=tensor.dtype,
                    generator=generator)
            torch.matmul(x, tensor.t())
    torch.cuda.synchronize()
    on_device.clear()


# ---------------------------------------------------------------------------
# quantum
# ---------------------------------------------------------------------------

def run_quantum(args):
    import pytest
    import prismaquant.production_weight_cache  # noqa: F401 -- tree check below
    import test_joint_cost_quantum_runtime as runtime
    from test_quantum_probe_identity_once_1183 import _campaign
    scratch = Path(args.scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as patch:
        # The producer source digest binds every identity to the package's
        # source, so it differs between two trees by design. One fixed value
        # on both trees leaves every other leaf, and every digest over them,
        # comparable.
        pinned = hashlib.sha256(b'pwc-window-load-bench source').hexdigest()
        for module in [module for name, module in sys.modules.items()
                       if name.startswith('prismaquant') and module is not None]:
            for attribute in ('_production_cache_source_sha256', '_aura_source_sha256'):
                if callable(getattr(module, attribute, None)):
                    patch.setattr(module, attribute, lambda *a, **k: pinned)
        single, receipt, output_root = _campaign(scratch, patch)
        payload, record, _counters = runtime._run_quantum(
            scratch, patch, single=single, layer=1, receipt=receipt,
            output_root=output_root, plan_sha=runtime._hex('d'),
            prepared_sha=runtime._hex('e'))
    root = Path(record['output_space']['root'])
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob('*')) if path.is_file()}
    # Whole-object digests also cover wall times and run ids, which differ
    # between two runs of one tree. Leaf digests let the host tell those
    # fields apart from the numbers: a leaf that differs between two runs of
    # the fix tree is run-varying; every other leaf must match the base.
    leaves = _leaves(payload, 'payload')
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        name = f'file:{path.relative_to(root)}'
        if path.suffix == '.pkl':
            with path.open('rb') as handle:
                _leaves(pickle.load(handle), name, leaves)
        elif path.suffix == '.json':
            _leaves(json.loads(path.read_text()), name, leaves)
        else:
            leaves[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    import prismaquant
    Path(args.out_json).write_text(json.dumps({
        'payload_sha256': hashlib.sha256(pickle.dumps(payload)).hexdigest(),
        'costs': len(payload['costs']), 'files': files, 'leaves': leaves,
        'package': str(Path(prismaquant.__file__).parent)}, indent=1))


def _leaves(obj, path: str, out: dict | None = None) -> dict:
    """``{path: value}`` for every leaf of ``obj``; tensors and arrays by digest."""
    import numpy as np
    import torch
    out = {} if out is None else out
    if isinstance(obj, torch.Tensor):
        flat = obj.detach().cpu().contiguous().reshape(-1)
        digest = hashlib.sha256(f'{obj.dtype}|{tuple(obj.shape)}|'.encode())
        digest.update(flat.view(torch.uint8).numpy().tobytes())
        out[path] = f'tensor:{digest.hexdigest()}'
    elif isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        digest = hashlib.sha256(f'{array.dtype}|{array.shape}|'.encode())
        digest.update(array.tobytes())
        out[path] = f'array:{digest.hexdigest()}'
    elif isinstance(obj, dict):
        for key in sorted(obj, key=repr):
            _leaves(obj[key], f'{path}/{key}', out)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _leaves(value, f'{path}[{index}]', out)
    elif isinstance(obj, float):
        out[path] = f'float:{obj.hex()}'
    elif isinstance(obj, (bytes, bytearray)):
        try:
            inner = pickle.loads(obj)
        except Exception:  # not a pickle: compare the bytes
            out[path] = f'bytes:{hashlib.sha256(obj).hexdigest()}'
        else:
            _leaves(inner, f'{path}<pickle>', out)
    else:
        text = repr(obj)
        out[path] = text if len(text) <= 160 else (
            f'{type(obj).__name__}:{hashlib.sha256(text.encode()).hexdigest()}')
    return out


def _compare_leaves(quantum: dict) -> dict:
    """Base against fix, leaf by leaf, excluding leaves two fix runs disagree on."""
    base, fix, control = (quantum[arm]['leaves'] for arm in ('base', 'fix', 'fix-control'))
    varying = sorted(key for key in set(fix) | set(control) if fix.get(key) != control.get(key))
    varying_set = set(varying)
    compared = sorted((set(base) | set(fix)) - varying_set)
    differing = [key for key in compared if base.get(key) != fix.get(key)]
    numeric = [key for key in compared if str(fix.get(key, base.get(key))).startswith(
        ('tensor:', 'array:', 'float:'))]
    return {
        'leaves_compared': len(compared),
        'numeric_leaves_compared': len(numeric),
        'numeric_leaves_identical': all(base.get(key) == fix.get(key) for key in numeric),
        'differing': [{'leaf': key, 'base': base.get(key), 'fix': fix.get(key)}
                      for key in differing],
        'run_varying': [{'leaf': key, 'fix': fix.get(key), 'fix-control': control.get(key)}
                        for key in varying],
        'run_varying_numeric': [key for key in varying if str(fix.get(key, control.get(key)))
                                .startswith(('tensor:', 'array:', 'float:'))],
    }


# ---------------------------------------------------------------------------
# ab (host)
# ---------------------------------------------------------------------------

def _materialize_base(ref: str, target: Path) -> Path:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    blob = subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar', ref,
                           'prismaquant', 'tests', 'tools'],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        archive.extractall(target, filter='data')
    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', ref],
                            check=True, capture_output=True, text=True).stdout.strip()
    (target / 'COMMIT').write_text(commit + '\n')
    return target


class _PowerSampler(threading.Thread):
    def __init__(self, path: Path):
        super().__init__(daemon=True)
        self.path, self.stop = path, threading.Event()

    def run(self):
        with self.path.open('w') as out:
            while not self.stop.is_set():
                try:
                    watts = subprocess.run(
                        ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=5).stdout.strip()
                except (OSError, subprocess.SubprocessError):
                    watts = ''
                out.write(f"{time.time():.3f}\t{watts}\n")
                out.flush()
                self.stop.wait(0.5)


def _power_between(samples, start, end):
    values = [watts for when, watts in samples if start <= when <= end]
    return (statistics.fmean(values) if values else None, len(values))


def _classify(raw_path: Path):
    """Main-thread and loader-thread sample shares from a py-spy raw profile."""
    main, loader = {}, {}
    main_total = loader_total = 0
    for line in raw_path.read_text().splitlines():
        stack, _, count = line.rpartition(' ')
        if not stack or not count.isdigit():
            continue
        count = int(count)
        frames = stack.split(';')
        thread = frames[0]
        if 'MainThread' in thread:
            main_total += count
            if 'run_window' not in stack:
                key = 'startup/other'
            elif '_consume' in stack:
                key = 'compute'
            elif 'retained_window' in stack and '__enter__' in stack:
                key = 'window-load'
            elif 'retained_window' in stack:
                key = 'window-exit'
            elif '_drop_client_pages' in stack:
                key = 'page-drop'
            else:
                key = 'digest/other'
            main[key] = main.get(key, 0) + count
        elif '_load_one' in stack or '_read (prismaquant/io_engine' in stack:
            loader_total += count
            for key, pattern in (('lease-enter', '__enter__ (prismaquant/staged_lease'),
                                 ('lease-acquire-entry', 'acquire_entry_window'),
                                 ('lease-exit', '__exit__ (prismaquant/staged_lease'),
                                 ('lease-open', 'open (prismaquant/staged_lease'),
                                 ('lease-close', 'close_fd'),
                                 ('staged_read', 'staged_read ('),
                                 ('archive-parse', 'torch_archive_storage_bytes'),
                                 ('render-identity', '_loaded_render_identity'),
                                 ('torch.load', 'load (torch/serialization')):
                if pattern in stack:
                    break
            else:
                key = 'read+hash/other'
            loader[key] = loader.get(key, 0) + count
    return ({'total': main_total, **{k: v / main_total for k, v in main.items()}}
            if main_total else {},
            {'total': loader_total, **{k: v / loader_total for k, v in loader.items()}}
            if loader_total else {})


def _trees(args, out: Path):
    trees = {'fix': REPO, 'base': _materialize_base(args.base_ref, out / 'base-tree')}
    commits = {'fix': subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                     capture_output=True, text=True).stdout.strip(),
               'base': (trees['base'] / 'COMMIT').read_text().strip()}
    _log(f"trees: {commits}")
    return trees, commits


def _quantum_identity(out: Path, trees: dict, me: list) -> dict:
    """The joint quantum on each tree; the fix tree twice, as a run-to-run control."""
    quantum = {}
    for arm in ('base', 'fix', 'fix-control'):
        tree = trees['fix' if arm.startswith('fix') else 'base']
        target = out / f'quantum-{arm}.json'
        subprocess.run(me + ['quantum', '--scratch', str(out / 'quantum-scratch'),
                             '--out-json', str(target)],
                       env=_child_env(tree, cuda=False), check=True, cwd=str(tree))
        quantum[arm] = json.loads(target.read_text())
        _log(f"quantum {arm}: payload {quantum[arm]['payload_sha256'][:16]}, "
             f"{len(quantum[arm]['files'])} files, {len(quantum[arm]['leaves'])} leaves")
    shutil.rmtree(out / 'quantum-scratch', ignore_errors=True)
    result = {
        'payload_identical': len({q['payload_sha256'] for q in quantum.values()}) == 1,
        'files_identical': len({json.dumps(q['files'], sort_keys=True)
                                for q in quantum.values()}) == 1,
        'leaves': _compare_leaves(quantum),
        **{arm: {'payload_sha256': q['payload_sha256'], 'costs': q['costs'],
                 'files': len(q['files']), 'package': q['package']}
           for arm, q in quantum.items()}}
    leaves = result['leaves']
    _log(f"quantum leaves: {leaves['leaves_compared']} compared "
         f"({leaves['numeric_leaves_compared']} numeric, identical: "
         f"{leaves['numeric_leaves_identical']}); {len(leaves['differing'])} differ; "
         f"{len(leaves['run_varying'])} vary run to run "
         f"({len(leaves['run_varying_numeric'])} numeric)")
    return result


def run_identity(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trees, commits = _trees(args, out)
    me = [sys.executable, str(Path(__file__).resolve())]
    summary = {'commits': commits, 'host': os.uname().nodename,
               'quantum': _quantum_identity(out, trees, me)}
    (out / 'identity.json').write_text(json.dumps(summary, indent=1, default=str))
    leaves = summary['quantum']['leaves']
    if leaves['differing'] or not leaves['numeric_leaves_identical']:
        raise SystemExit('the joint quantum differs between base and fix')


def run_ab(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trees, commits = _trees(args, out)
    me = [sys.executable, str(Path(__file__).resolve())]
    summary = {'commits': commits, 'host': os.uname().nodename, 'args': vars(args)}

    summary['quantum'] = _quantum_identity(out, trees, me)

    fixture_json = out / 'fixture.json'
    subprocess.run(me + ['setup', '--fixture', args.fixture, '--fixture-json', str(fixture_json),
                         '--files', str(args.files), '--record', args.record],
                   env=_child_env(trees['fix'], cuda=False), check=True, cwd=str(REPO))
    compute_s = (args.compute_s if args.compute_s is not None
                 else ROW_041_NONLOAD_S * args.files / ROW_041_RENDERS)
    summary['compute_s'] = compute_s
    sampler = _PowerSampler(out / 'power.tsv')
    sampler.start()
    runs = []
    try:
        (out / 'pyspy').mkdir(exist_ok=True)
        for round_index in range(args.rounds):
            order = ('base', 'fix') if round_index % 2 == 0 else ('fix', 'base')
            for arm in order:
                result = out / f'window-{round_index:02d}-{arm}.json'
                raw = out / 'pyspy' / f'{round_index:02d}-{arm}.raw'
                command = me + ['window', '--fixture-json', str(fixture_json),
                                '--out-json', str(result), '--windows', str(args.windows),
                                '--workers', str(args.workers), '--probes', str(args.probes),
                                '--batch', str(args.batch),
                                '--consumer', args.consumer,
                                '--compute-s', str(compute_s),
                                '--load-buffer-bytes', str(args.load_buffer_bytes)]
                if args.pyspy:
                    command = [args.pyspy, 'record', '--nonblocking', '--idle', '--threads',
                               '--rate', str(args.rate), '--format', 'raw', '-o', str(raw),
                               '--'] + command
                started = time.time()
                subprocess.run(command, env=_child_env(trees[arm], cuda=args.consumer == 'gpu'),
                               check=True, cwd=str(trees[arm]))
                runs.append({'round': round_index, 'arm': arm, 'start': started,
                             'end': time.time(), 'result': str(result),
                             'pyspy': str(raw) if args.pyspy else None})
                _log(f"round {round_index} {arm}: {time.time() - started:.1f} s")
    finally:
        sampler.stop.set()
        sampler.join(timeout=5)
    power = []
    for line in (out / 'power.tsv').read_text().splitlines():
        when, _, watts = line.partition('\t')
        try:
            power.append((float(when), float(watts)))
        except ValueError:
            pass

    arms = {}
    for run in runs:
        result = json.loads(Path(run['result']).read_text())
        arm = arms.setdefault(run['arm'], {'windows': [], 'digests': set(), 'power': [],
                                           'counters': [], 'main': [], 'loader': [],
                                           'bytes_from_pool': 0})
        measured = result['windows'][args.skip_windows:]
        arm['windows'].extend(measured)
        arm['digests'].add(result['digest'])
        arm['counters'].append(result['lease_counters'])
        arm['bytes_from_pool'] += result['bytes_from_pool'] or 0
        for window in measured:
            mean, count = _power_between(power, window['epoch_start'], window['epoch_end'])
            arm['power'].append({'mean_w': mean, 'samples': count,
                                 'wall_s': window['epoch_end'] - window['epoch_start']})
        if run['pyspy'] and Path(run['pyspy']).exists():
            main, loader = _classify(Path(run['pyspy']))
            arm['main'].append(main)
            arm['loader'].append(loader)

    def pooled(shares):
        keys = {key for share in shares for key in share if key != 'total'}
        total = sum(share.get('total', 0) for share in shares)
        return {'samples': total, **{key: sum(share.get(key, 0) * share.get('total', 0)
                                              for share in shares) / total
                                     for key in sorted(keys)}} if total else {}

    summary['arms'] = {}
    for name, arm in arms.items():
        windows = arm['windows']
        summary['arms'][name] = {
            'windows': len(windows),
            'load_s_median': statistics.median(w['load_s'] for w in windows),
            'load_s_mean': statistics.fmean(w['load_s'] for w in windows),
            'load_s_stdev': statistics.pstdev(w['load_s'] for w in windows),
            'compute_s_median': statistics.median(w['compute_s'] for w in windows),
            'wall_s_median': statistics.median(w['wall_s'] for w in windows),
            'exit_s_median': statistics.median(w['exit_s'] for w in windows),
            'load_share_of_wall_median': statistics.median(
                w['load_s'] / w['wall_s'] for w in windows),
            'quanta': sorted({w['quanta'] for w in windows}),
            'digests': sorted(arm['digests']),
            'lease_counters': arm['counters'],
            'bytes_from_pool': arm['bytes_from_pool'],
            'gpu_power_w_mean': statistics.fmean(
                p['mean_w'] for p in arm['power'] if p['mean_w'] is not None)
            if any(p['mean_w'] is not None for p in arm['power']) else None,
            'gpu_energy_j_per_window_median': statistics.median(
                p['mean_w'] * p['wall_s'] for p in arm['power'] if p['mean_w'] is not None)
            if any(p['mean_w'] is not None for p in arm['power']) else None,
            'pyspy_main': pooled(arm['main']),
            'pyspy_loader': pooled(arm['loader']),
        }
    summary['loaded_bytes_identical'] = len(
        {digest for arm in arms.values() for digest in arm['digests']}) == 1
    summary['runs'] = runs
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    _log(json.dumps({name: {key: arm[key] for key in (
        'load_s_median', 'compute_s_median', 'wall_s_median', 'load_share_of_wall_median',
        'gpu_power_w_mean')} for name, arm in summary['arms'].items()}, indent=1))
    _log(f"loaded bytes identical: {summary['loaded_bytes_identical']}; quantum "
         f"numeric leaves identical: "
         f"{summary['quantum']['leaves']['numeric_leaves_identical']}, "
         f"differing leaves: {len(summary['quantum']['leaves']['differing'])}")
    if not args.keep_fixture:
        shutil.rmtree(args.fixture, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    ab = sub.add_parser('ab')
    ab.add_argument('--base-ref', required=True)
    ab.add_argument('--out', required=True)
    ab.add_argument('--fixture', required=True)
    ab.add_argument('--record', default=ROW_041_RECORD)
    ab.add_argument('--files', type=int, default=64)
    ab.add_argument('--windows', type=int, default=3)
    ab.add_argument('--skip-windows', type=int, default=1,
                    help='leading windows per child left out of the statistics (warm-up)')
    ab.add_argument('--rounds', type=int, default=4)
    ab.add_argument('--workers', type=int, default=4)
    ab.add_argument('--probes', type=int, default=64)
    ab.add_argument('--batch', type=int, default=4096)
    ab.add_argument('--consumer', choices=('sleep', 'gpu'), default='sleep')
    ab.add_argument('--compute-s', type=float, default=None)
    ab.add_argument('--load-buffer-bytes', type=int, default=402_662_764,
                    help="row 041's sealed load_buffer_bytes")
    ab.add_argument('--pyspy', default=shutil.which('py-spy'))
    ab.add_argument('--rate', type=int, default=50)
    ab.add_argument('--keep-fixture', action='store_true')
    setup = sub.add_parser('setup')
    setup.add_argument('--fixture', required=True)
    setup.add_argument('--fixture-json', required=True)
    setup.add_argument('--files', type=int, required=True)
    setup.add_argument('--record', required=True)
    window = sub.add_parser('window')
    window.add_argument('--fixture-json', required=True)
    window.add_argument('--out-json', required=True)
    window.add_argument('--windows', type=int, required=True)
    window.add_argument('--workers', type=int, required=True)
    window.add_argument('--probes', type=int, required=True)
    window.add_argument('--batch', type=int, required=True)
    window.add_argument('--consumer', choices=('sleep', 'gpu'), required=True)
    window.add_argument('--compute-s', type=float, required=True)
    window.add_argument('--load-buffer-bytes', type=int, required=True)
    identity = sub.add_parser('identity')
    identity.add_argument('--base-ref', required=True)
    identity.add_argument('--out', required=True)
    quantum = sub.add_parser('quantum')
    quantum.add_argument('--scratch', required=True)
    quantum.add_argument('--out-json', required=True)
    args = parser.parse_args(argv)
    {'ab': run_ab, 'setup': run_setup, 'window': run_window,
     'quantum': run_quantum, 'identity': run_identity}[args.mode](args)


if __name__ == '__main__':
    main()
