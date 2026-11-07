"""Native Flux CPU executor, baseline fv1-model profile v1.

This is not a campaign scheduler or a result cache. Strict staged reads and
PB-authenticated publication/pilot contracts are deliberately unsupported.
Callers own their pinned numerical runtime and scientific input/output gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Literal


_PREFIX = Path('/home/rob/flux-vanilla/0.91.0-0.56.0')
_CLI = str(_PREFIX / 'install/bin/flux')
_URI = f'local://{_PREFIX}/runs/fv1-model/run/local-0'
_PATH = f'{_PREFIX}/install/bin:{_PREFIX}/deps/bin:/usr/bin:/bin'
_LIB = f'{_PREFIX}/install/lib:{_PREFIX}/deps/lib'
_CONFIG = _PREFIX / 'runs/fv1-model/config.json'
_MODEL = _PREFIX / 'config/R-model.json'
_PROFILE = 'pq.flux.fv1-model.perf-memory.v1'
_CONFIG_SHA = '2b5618a591bbf545336ed9dfae6645b5dfca7516325ccbf9e0b2dbf4181607e1'
_MODEL_SHA = '261a084b72cb3a0c5055816f642f51ccbf473d3373876f34e67be55d5b5221a8'
_THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCH_NUM_THREADS')
_PROPERTIES = frozenset(('x86_64', 'aarch64', 'gb10', 'unified-memory'))
_HOSTS = frozenset(('dl380g10', 'sparky', 'sparklina'))
Status = Literal['succeeded', 'failed', 'canceled', 'dependency-failed', 'running', 'unknown']


@dataclass(frozen=True)
class FluxResources:
    cores: int
    memory_gib: int
    gpus: int
    walltime_s: int
    placement: str = 'portable'

    def __post_init__(self):
        for name, low, high in (('cores', 1, 4), ('memory_gib', 1, 16),
                                ('gpus', 0, 0), ('walltime_s', 1, 1800)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}] for {_PROFILE}')
        if self.placement not in {'portable', *_PROPERTIES, *(f'host:{h}' for h in _HOSTS)}:
            raise ValueError('unsupported placement; host placement requires caller data/runtime justification')


@dataclass(frozen=True)
class FluxSubmission:
    job_id: str
    record_dir: Path
    launch_sha256: str


@dataclass(frozen=True)
class ExecutionResult:
    status: Status
    job_id: str
    record_dir: Path
    payload_exit: int | None = None
    query_error: str | None = None


class SubmissionUnknown(RuntimeError):
    """Submission may exist. Retain the directory and reconcile; never retry it."""


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    data = _json_bytes(value)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run(args, *, timeout=30):
    # Never inherit CLI plugins, job KVS namespaces, credentials, or PB context.
    return subprocess.run([_CLI, *args], text=True, capture_output=True,
                          env={'PATH': _PATH, 'HOME': '/home/rob', 'LD_LIBRARY_PATH': _LIB,
                               'FLUX_URI': _URI}, timeout=timeout, check=False)


def _checked(args):
    result = _run(args)
    if result.returncode:
        raise RuntimeError(f'Flux query failed: {args[0]} {args[1] if len(args) > 1 else ""}')
    return result.stdout


def _profile_identity():
    config_bytes, model_bytes = _CONFIG.read_bytes(), _MODEL.read_bytes()
    if _digest(config_bytes) != _CONFIG_SHA or _digest(model_bytes) != _MODEL_SHA:
        raise ValueError('unsupported Flux config/resource model; profile revision required')
    config, model = json.loads(config_bytes), json.loads(model_bytes)
    for section in ('resource', 'exec', 'sched-fluxion-resource', 'sched-fluxion-qmanager'):
        if json.loads(_checked(['config', 'get', section])) != config[section]:
            raise ValueError('loaded Flux config differs from baseline')
    live = json.loads(_checked(['kvs', 'get', 'resource.R']))
    if live['scheduling'] != model['scheduling']:
        raise ValueError('loaded resource graph differs from baseline')
    if _checked(['resource', 'status', '-s', 'online', '-n', '-o', '{ranks}']).strip() != '0-2':
        raise ValueError('all three Flux hosts must be online')
    overlay = _checked(['overlay', 'status'])
    if sum(line.endswith(': full') for line in overlay.splitlines()) != 3:
        raise ValueError('all three Flux brokers must be healthy')
    return {'profile': _PROFILE, 'config_sha256': _CONFIG_SHA, 'model_sha256': _MODEL_SHA,
            'uri': _URI, 'broker_starttime': _checked(['getattr', 'broker.starttime']).strip()}


def _git(checkout, *args):
    result = subprocess.run(['git', '-C', str(checkout), *args], capture_output=True, text=True,
                            check=True, env={'PATH': '/usr/bin:/bin', 'HOME': '/home/rob',
                                             'GIT_OPTIONAL_LOCKS': '0'}, timeout=30)
    return result.stdout.strip()


def _source_identity(checkout):
    if Path(_git(checkout, 'rev-parse', '--show-toplevel')).resolve() != checkout:
        raise ValueError('checkout must be the source root')
    commit = _git(checkout, 'rev-parse', '--verify', 'HEAD^{commit}')
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('source must resolve to a full commit')
    if _git(checkout, 'status', '--porcelain', '--untracked-files=all', '--ignored'):
        raise ValueError('source checkout must be clean, including ignored files')
    if _git(checkout, 'branch', '--show-current'):
        raise ValueError('source checkout must be detached at its immutable commit')
    if any(line.startswith('160000 ') for line in _git(checkout, 'ls-files', '--stage').splitlines()):
        raise ValueError('submodule source identity is not supported by this profile')
    return {'checkout': str(checkout), 'commit': commit,
            'tree': _git(checkout, 'rev-parse', 'HEAD^{tree}')}


def _job_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'(?:[1-9][0-9]*|ƒ[1-9A-HJ-NP-Za-km-z]+)', value):
        raise ValueError('expected a saved native Flux job ID')
    return value


def _environment(record_dir, supplied):
    environment = {'PATH': _PATH, 'HOME': '/home/rob', 'LD_LIBRARY_PATH': _LIB,
                   'OUT': str(record_dir), 'FLUX_SCRATCH': '/tmp/flux-vanilla',
                   'PYTHONDONTWRITEBYTECODE': '1', **dict.fromkeys(_THREADS, '1')}
    # No general application environment passthrough. Add a versioned setting
    # here only when its owning contract has migrated and can be validated.
    for key, value in ({} if supplied is None else supplied).items():
        if key == 'XDG_RUNTIME_DIR' and value == '/run/user/1000':
            environment[key] = value
        elif key not in environment or value != environment[key]:
            raise ValueError('unsupported environment setting; credentials and PB context are forbidden')
    return environment


def validate_job(*, checkout: Path, resources: FluxResources, environment=None, staged_input=None):
    """Preflight a whole caller stage before its first native submission."""
    if not isinstance(resources, FluxResources):
        raise TypeError('resources must be FluxResources')
    source = _source_identity(Path(checkout).resolve())
    _environment(Path('/mnt/shared/preflight-only'), environment)
    if staged_input is not None:
        from .staged_launch import validate_binding
        validate_binding(staged_input, placement=resources.placement)
    return source


def _shape(resources):
    return f'node/slot/[core={resources.cores}/perf=1;memory={resources.memory_gib}]'


_PAYLOAD = r'''
import json, os, pathlib, shutil, signal, socket, subprocess, sys
out = pathlib.Path(os.environ['OUT'])
expected_commit, command = sys.argv[1], sys.argv[2:]
def git(*args):
    return subprocess.check_output(['/usr/bin/git', *args], text=True).strip()
if git('rev-parse', 'HEAD') != expected_commit or git('status', '--porcelain', '--untracked-files=all', '--ignored') or git('branch', '--show-current'):
    raise RuntimeError('source checkout changed after submission')
host = socket.gethostname().split('.')[0]
roots = {'dl380g10': ('/home/rob/tl-gates/flux-validation', 'xfs'),
         'sparky': ('/tmp/flux-vanilla', 'ext4'), 'sparklina': ('/tmp/flux-vanilla', 'ext4')}
root, expected_fs = roots[host]
actual_fs = subprocess.check_output(['/usr/bin/findmnt', '-n', '-o', 'FSTYPE', '-T', root], text=True).strip()
if actual_fs != expected_fs:
    raise RuntimeError('unqualified scratch filesystem')
os.environ['FLUX_SCRATCH'] = root
scratch = pathlib.Path(root) / os.environ['FLUX_JOB_ID']
child = None
received = None
def stop(signum, frame):
    global received
    received = signum
    if child is not None:
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM, signal.SIGUSR1):
    signal.signal(sig, stop)
scratch.mkdir(mode=0o700)
try:
    os.environ['TMPDIR'] = str(scratch)
    (out / 'scratch.json').write_text(json.dumps({'schema': 'pq.flux.scratch.v1',
        'job_id': os.environ['FLUX_JOB_ID'], 'host': host, 'path': str(scratch),
        'affinity': sorted(os.sched_getaffinity(0))}))
    if received is None:
        child = subprocess.Popen(command, start_new_session=True)
        if received is not None:
            stop(received, None)
        code = child.wait()
    else:
        code = 128 + received
finally:
    shutil.rmtree(scratch)
    (out / 'cleanup.json').write_text(json.dumps({'schema': 'pq.flux.cleanup.v1',
        'job_id': os.environ['FLUX_JOB_ID'], 'path': str(scratch), 'removed': not scratch.exists(),
        'signal': received}))
sys.exit(128 - code if code < 0 else code)
'''


def submit_job(argv, *, checkout: Path, resources: FluxResources, record_dir: Path,
               afterok=(), environment=None, staged_input=None) -> FluxSubmission:
    """Submit once. Existing records and lost acknowledgements require reconciliation.

    ``afterok`` contains saved native ID strings. ``host:NAME`` is reserved for
    a caller's justified data-owner/runtime dependency. Only CPU validation is
    qualified in profile v1. Unsupported input/output safety settings refuse.
    """
    if isinstance(argv, (str, bytes)):
        raise ValueError('argv must be a sequence, not a shell string')
    argv = tuple(argv)
    if not argv or any(not isinstance(a, str) or '\0' in a for a in argv):
        raise ValueError('argv must be a nonempty sequence of strings without NUL')
    if not isinstance(resources, FluxResources):
        raise TypeError('resources must be FluxResources')
    checkout, record_dir = Path(checkout).resolve(), Path(record_dir).resolve()
    if not record_dir.is_relative_to(Path('/mnt/shared')) or record_dir.is_relative_to(checkout):
        raise ValueError('records require a separate shared directory outside the source checkout')
    dependencies = [_job_id(value) for value in afterok]
    if len(set(dependencies)) != len(dependencies):
        raise ValueError('duplicate dependency ID')
    env = _environment(record_dir, environment)
    command = list(argv)
    if staged_input is not None:
        from .staged_launch import validate_binding
        validate_binding(staged_input, placement=resources.placement)
        command = ['/usr/bin/python3', '-m', 'prismaquant.staged_launch',
                   json.dumps(staged_input, sort_keys=True), *command]
    source, profile = _source_identity(checkout), _profile_identity()
    args = ['submit', '--env=-*', *(f'--env={key}={value}' for key, value in sorted(env.items())),
            f'--resources-shape={_shape(resources)}', f'--time-limit={resources.walltime_s}s',
            f'--cwd={checkout}', '--input=/dev/null', f'--output={record_dir}/stdout',
            f'--error={record_dir}/stderr', f'--job-name=pq-{record_dir.name}']
    if resources.placement != 'portable':
        args.append(f'--requires={resources.placement}')
    args.extend(f'--dependency=afterok:{value}' for value in dependencies)
    args.extend(['/usr/bin/python3', '-c', _PAYLOAD, source['commit'], *command])
    intent = {'schema': 'pq.flux.launch.v1', 'source': source, 'profile': profile,
              'resources': asdict(resources), 'afterok': dependencies, 'argv': list(argv),
              'native_argv': args, 'record_dir': str(record_dir),
              'qualification': 'cpu-executor-only; strict-staging/publication/pilots unqualified'}
    if staged_input is not None:
        intent['staged_input'] = staged_input
    record_dir.mkdir(parents=True, exist_ok=False)
    _write_json(record_dir / 'launch.json', intent)
    launch_sha = _digest(_json_bytes(intent))
    # Publish unknown durably BEFORE handing the intent to Flux. A crash at
    # any subsequent point must not make this directory eligible for retry.
    receipt = {'schema': 'pq.flux.submission.v1', 'launch_sha256': launch_sha,
               'state': 'unknown', 'job_id': None}
    _write_json(record_dir / 'submission.json', receipt)
    try:
        result = _run(args)
        (record_dir / 'submit.stdout').write_text(result.stdout)
        (record_dir / 'submit.stderr').write_text(result.stderr)
        if result.returncode:
            raise SubmissionUnknown('native submission returned an error')
        job_id = _job_id(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError, SubmissionUnknown) as exc:
        raise SubmissionUnknown(f'submission unknown in {record_dir}; do not resubmit') from exc
    receipt.update(state='submitted', job_id=job_id)
    _write_json(record_dir / 'submission.json', receipt)
    (record_dir / 'jobid').write_text(job_id + '\n')
    return FluxSubmission(job_id, record_dir, launch_sha)


def load_submission(record_dir: Path) -> FluxSubmission:
    """Reconnect to a saved ID only; never submit or infer a producer by name."""
    record_dir = Path(record_dir).resolve()
    intent = json.loads((record_dir / 'launch.json').read_bytes())
    receipt = json.loads((record_dir / 'submission.json').read_bytes())
    launch_sha = _digest(_json_bytes(intent))
    if (not isinstance(intent, dict) or not isinstance(receipt, dict) or
            intent.get('schema') != 'pq.flux.launch.v1' or
            receipt.get('schema') != 'pq.flux.submission.v1' or
            receipt.get('launch_sha256') != launch_sha or intent.get('record_dir') != str(record_dir)):
        raise ValueError('invalid or modified Flux launch/submission receipt')
    if receipt.get('state') != 'submitted' or not receipt.get('job_id'):
        raise SubmissionUnknown(f'submission unknown in {record_dir}; reconcile, do not resubmit')
    return FluxSubmission(_job_id(receipt['job_id']), record_dir, launch_sha)


def _validate_submission(submission):
    if load_submission(submission.record_dir) != submission:
        raise ValueError('submission does not match saved launch and ID')
    intent = json.loads((submission.record_dir / 'launch.json').read_bytes())
    if intent.get('staged_input') is not None:
        from .staged_launch import validate_binding
        validate_binding(intent['staged_input'], placement=intent['resources']['placement'])
    if intent['profile']['uri'] != _URI or intent['profile']['broker_starttime'] != _checked(['getattr', 'broker.starttime']).strip():
        raise RuntimeError('Flux instance changed; saved job identity cannot be assumed')


def validate_dependency(submission: FluxSubmission):
    """Require a saved producer ID from the current standing instance."""
    _validate_submission(submission)


def validate_launch(submission: FluxSubmission, *, argv, source, resources,
                    afterok=(), environment=None, staged_input=None):
    """Bind application intent to every saved execution setting on resume/merge."""
    if load_submission(submission.record_dir) != submission:
        raise ValueError('submission differs from saved launch and ID')
    launch = json.loads((submission.record_dir / 'launch.json').read_bytes())
    if launch.get('staged_input') != staged_input:
        raise ValueError('saved staged-input binding differs from application intent')
    if staged_input is not None:
        from .staged_launch import validate_binding
        validate_binding(staged_input, placement=resources.placement)
    env = _environment(submission.record_dir, environment)
    prefix = ['submit', '--env=-*', *(f'--env={key}={value}' for key, value in sorted(env.items())),
              f'--resources-shape={_shape(resources)}']
    if (launch['argv'] != list(argv) or launch['source'] != source
            or launch['resources'] != asdict(resources) or launch['afterok'] != list(afterok)
            or launch['native_argv'][:len(prefix)] != prefix):
        raise ValueError('saved execution differs from application intent')


def _query(submission, name, args, errors, *, required=True, timeout=30):
    try:
        result = _run(args, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f'{name}: {type(exc).__name__}')
        return None
    out = submission.record_dir
    (out / f'{name}.json').write_text(result.stdout)
    (out / f'{name}.stderr').write_text(result.stderr)
    (out / f'{name}.returncode').write_text(str(result.returncode) + '\n')
    if required and result.returncode:
        errors.append(f'{name}: query exit {result.returncode}')
        return None
    return result


def collect_job(submission: FluxSubmission, *, wait: bool = True) -> ExecutionResult:
    """Collect native facts. Query failures are unknown, never payload failures.

    A negative payload_exit is a POSIX signal. R and redirected log paths are
    retained even for failure; absence before allocation/start is recorded.
    SIGKILL/node loss can prevent scratch cleanup and need owner reconciliation.
    """
    errors, events, payload_exit, status = [], [], None, 'unknown'
    try:
        _validate_submission(submission)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        errors.append(f'identity: {type(exc).__name__}: {exc}')
    if not errors:
        if wait:
            _query(submission, 'wait', ['job', 'wait-event', submission.job_id, 'clean'], errors, timeout=None)
        raw = _query(submission, 'eventlog', ['job', 'eventlog', '--format=json', submission.job_id], errors)
        if raw is not None:
            try:
                events = [json.loads(line) for line in raw.stdout.splitlines()]
                if not events or any(not isinstance(e, dict) or not isinstance(e.get('name'), str) for e in events):
                    raise ValueError('missing event names')
            except (ValueError, TypeError):
                errors.append('eventlog: invalid JSON/events')
                events = []
        names = {event['name'] for event in events}
        finishes = [event for event in events if event['name'] == 'finish']
        if finishes:
            try:
                value = finishes[-1]['context']['status']
                if type(value) is not int or value < 0:
                    raise ValueError('invalid waitstatus')
                payload_exit = os.waitstatus_to_exitcode(value)
            except (KeyError, ValueError, TypeError):
                errors.append('eventlog: invalid payload waitstatus')
        allocated = 'alloc' in names
        raw_r = _query(submission, 'R', ['job', 'info', submission.job_id, 'R'], errors, required=allocated)
        if allocated and raw_r is not None:
            try:
                allocation = json.loads(raw_r.stdout)
                if allocation['version'] != 1 or not allocation['execution']['R_lite']:
                    raise ValueError('missing allocation')
            except (ValueError, KeyError, TypeError):
                errors.append('R: invalid allocation JSON')
        if 'clean' in names:
            raw_status = _query(submission, 'status', ['job', 'status', '--json', '--exception-exit-code=125',
                                                       submission.job_id], errors, required=False)
            try:
                native = json.loads(raw_status.stdout) if raw_status is not None else None
                if (not isinstance(native, dict) or type(native.get('id')) is not int or
                        type(native.get('exception_occurred')) is not bool or
                        type(native.get('result')) is not int or native['result'] not in (1, 2, 4, 8)):
                    raise ValueError('missing native result')
                if native['result'] != 1 and raw_status.returncode == 0:
                    raise ValueError('native result/query exit disagreement')
                if native.get('waitstatus') is not None and finishes and native['waitstatus'] != finishes[-1]['context']['status']:
                    raise ValueError('status/eventlog mismatch')
                exceptions = [e.get('context', {}) for e in events if e['name'] == 'exception']
                if any(e.get('type') == 'dependency' for e in exceptions):
                    status = 'dependency-failed'
                elif native['result'] == 4:
                    status = 'canceled'
                elif native['result'] == 1 and payload_exit == 0 and raw_status.returncode == 0:
                    status = 'succeeded'
                elif native['result'] in (2, 8):
                    status = 'failed'
                else:
                    errors.append('status: inconsistent native result')
            except (ValueError, TypeError, KeyError):
                errors.append('status: invalid/missing native result JSON')
        elif events:
            status = 'running'
        if 'start' in names and 'clean' in names:
            try:
                scratch = json.loads((submission.record_dir / 'scratch.json').read_bytes())
                cleanup = json.loads((submission.record_dir / 'cleanup.json').read_bytes())
                if (scratch['job_id'] != submission.job_id or cleanup['job_id'] != submission.job_id or
                        cleanup['path'] != scratch['path'] or cleanup['removed'] is not True):
                    raise ValueError('scratch cleanup unverified')
            except (OSError, ValueError, KeyError, TypeError):
                errors.append('scratch: cleanup unverified; reconcile owned descendants and path')
    for name in ('stdout', 'stderr'):
        if not (submission.record_dir / name).exists():
            (submission.record_dir / f'{name}.missing').write_text('No redirected payload stream observed.\n')
    result = ExecutionResult('unknown' if errors else status, submission.job_id, submission.record_dir,
                             payload_exit, '; '.join(errors) or None)
    _write_json(submission.record_dir / 'result.json',
                {'schema': 'pq.flux.result.v1', **asdict(result), 'record_dir': str(result.record_dir)})
    return result


def cancel_job(submission: FluxSubmission) -> ExecutionResult:
    """Cancel the saved native ID and collect its actual terminal verdict."""
    try:
        _validate_submission(submission)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        result = ExecutionResult('unknown', submission.job_id, submission.record_dir,
                                 query_error=f'cancel identity: {type(exc).__name__}: {exc}')
        _write_json(submission.record_dir / 'cancel-result.json',
                    {'schema': 'pq.flux.result.v1', **asdict(result), 'record_dir': str(result.record_dir)})
        return result
    errors = []
    _query(submission, 'cancel', ['cancel', submission.job_id], errors)
    result = collect_job(submission)
    if errors:
        result = ExecutionResult('unknown', result.job_id, result.record_dir, result.payload_exit,
                                 '; '.join([*errors, *([result.query_error] if result.query_error else [])]))
        _write_json(submission.record_dir / 'cancel-result.json',
                    {'schema': 'pq.flux.result.v1', **asdict(result), 'record_dir': str(result.record_dir)})
    return result
