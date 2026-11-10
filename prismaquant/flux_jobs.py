"""Native Flux CPU executor with stock slot resources and runtime instance evidence.

Callers own campaign scheduling, pinned numerical runtimes and scientific
input/output gates. Finite immutable staged inputs are admitted through the
versioned binding. Native publication and pilot authority remain unqualified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
from typing import Literal
from .digests import DIRECT_ASCII_SPACED_LAX, DIRECT_UTF8_LAX, bytes_sha256hex


_SHARED_ROOT = Path('/mnt/shared')
_INSTANCE = Path('/mnt/shared/flux-migration/INSTANCE')
_URI_RESOLVER = '/mnt/shared/flux-migration/flux-uri.sh'
_PROFILE = 'pq.flux.stock-slot.v3'
_LEGACY_PROFILE = 'pq.flux.fv1-model.perf-memory.v2'
_THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCH_NUM_THREADS')
_PROPERTIES = frozenset(('x86_64', 'aarch64', 'gb10', 'unified-memory'))
_HOSTS = frozenset(('dl380g10', 'sparky', 'sparklina'))
_F58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
Status = Literal['succeeded', 'failed', 'canceled', 'dependency-failed', 'running', 'unknown']
_FAILURES = frozenset(('failed', 'canceled', 'dependency-failed'))


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
    return DIRECT_UTF8_LAX.encoded(value) + b'\n'


def _publish_flux_json(path, value):
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


def _client_environment():
    """Read the fleet-owned client prefix lazily, without shell evaluation."""
    try:
        instance = _INSTANCE.read_text()
    except (OSError, UnicodeError) as exc:
        raise ValueError('Flux INSTANCE is missing or unreadable') from exc
    match = re.fullmatch(r'(/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+) ([A-Za-z0-9][A-Za-z0-9_.-]*)\n?', instance)
    if match is None or any(part in {'.', '..'} for part in Path(match[1]).parts):
        raise ValueError('Flux INSTANCE must contain one absolute PREFIX and RUN line')
    prefix = match[1]
    return {'PATH': f'{prefix}/install/bin:{prefix}/deps/bin:/usr/bin:/bin',
            'HOME': '/home/rob',
            'LD_LIBRARY_PATH': f'{prefix}/install/lib:{prefix}/deps/lib'}


def _instance_uri():
    """Use explicit coordinator intent, otherwise the fleet's host resolver."""
    uri = os.environ.get('FLUX_URI')
    if uri:
        if any(character.isspace() or character == '\0' for character in uri):
            raise ValueError('coordinator FLUX_URI must name the intended Flux instance')
        return uri
    try:
        result = subprocess.run([_URI_RESOLVER], capture_output=True, text=True,
                                env={'PATH': '/usr/bin:/bin'}, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f'FLUX_URI is unset or empty; fleet URI resolver is unavailable: {_URI_RESOLVER}; tried both sources') from exc
    if result.returncode or re.fullmatch(r'local:///[^\s\x00]+\n?', result.stdout) is None:
        raise ValueError(f'FLUX_URI is unset or empty; fleet URI resolver failed or returned an invalid URI: {_URI_RESOLVER}; tried both sources')
    return result.stdout.removesuffix('\n')


def _flux_run(args, *, timeout=30, uri=None):
    # Never inherit CLI plugins, job KVS namespaces, credentials, or PB context.
    environment = _client_environment()
    cli = environment['PATH'].split(':', 1)[0] + '/flux'
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        result = subprocess.run([cli, *args], stdout=stdout, stderr=stderr,
                                env={**environment, 'FLUX_URI': _instance_uri() if uri is None else uri}, timeout=timeout, check=False)
        for stream in (stdout, stderr):
            if stream.tell() > 16 << 20:
                raise ValueError('native query output exceeds evidence bound')
            stream.seek(0)
        return subprocess.CompletedProcess(result.args, result.returncode,
                                           stdout.read().decode(), stderr.read().decode())


def _checked(args, *, uri=None):
    result = _flux_run(args, uri=uri)
    if result.returncode:
        raise RuntimeError(f'Flux query failed: {args[0]} {args[1] if len(args) > 1 else ""}')
    return result.stdout


def _instance_ranks(model):
    if model['version'] != 1 or not model['execution']['R_lite']:
        raise ValueError('live Flux R has no ranks')
    ranks = set()
    for row in model['execution']['R_lite']:
        members = _allocation_ids(row['rank'])
        if ranks & members:
            raise ValueError('live Flux R repeats ranks')
        ranks.update(members)
    return ranks


def _profile_identity():
    uri = _instance_uri()
    start = _checked(['getattr', 'broker.starttime'], uri=uri).strip()
    config_raw = _checked(['config', 'get'], uri=uri)
    model_raw = _checked(['kvs', 'get', 'resource.R'], uri=uri)
    config, model = _native_document(config_raw), _native_document(model_raw)
    if not isinstance(config, dict) or not start:
        raise ValueError('missing live Flux config or broker identity')
    ranks = _instance_ranks(model)
    online = _checked(['resource', 'status', '-s', 'online', '-n', '-o', '{ranks}'], uri=uri).strip()
    if _allocation_ids(online) != ranks:
        raise ValueError('every rank in live Flux R must be online')
    overlay = _checked(['overlay', 'status', '--no-pretty', '--color=never'], uri=uri)
    healthy = []
    for line in overlay.splitlines():
        match = re.fullmatch(r'([0-9]+) \S+: full', line)
        if match is None:
            raise ValueError('every live Flux broker overlay must be full')
        healthy.append(int(match[1]))
    if len(healthy) != len(ranks) or set(healthy) != ranks:
        raise ValueError('every live Flux broker overlay must be full')
    if _checked(['getattr', 'broker.starttime'], uri=uri).strip() != start:
        raise ValueError('Flux instance changed during preflight')
    return {'profile': _PROFILE, 'config_sha256': bytes_sha256hex(config_raw.encode()),
            'r_sha256': bytes_sha256hex(model_raw.encode()), 'uri': uri, 'broker_starttime': start}


def _source_git(checkout, *args):
    result = subprocess.run(['git', '-C', str(checkout), *args], capture_output=True, text=True,
                            check=True, env={'PATH': '/usr/bin:/bin', 'HOME': '/home/rob',
                                             'GIT_OPTIONAL_LOCKS': '0'}, timeout=30)
    return result.stdout.strip()


def _source_identity(checkout):
    if Path(_source_git(checkout, 'rev-parse', '--show-toplevel')).resolve() != checkout:
        raise ValueError('checkout must be the source root')
    commit = _source_git(checkout, 'rev-parse', '--verify', 'HEAD^{commit}')
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('source must resolve to a full commit')
    if _source_git(checkout, 'status', '--porcelain', '--untracked-files=all', '--ignored'):
        raise ValueError('source checkout must be clean, including ignored files')
    if _source_git(checkout, 'branch', '--show-current'):
        raise ValueError('source checkout must be detached at its immutable commit')
    if any(line.startswith('160000 ') for line in _source_git(checkout, 'ls-files', '--stage').splitlines()):
        raise ValueError('submodule source identity is not supported by this profile')
    return {'checkout': str(checkout), 'commit': commit,
            'tree': _source_git(checkout, 'rev-parse', 'HEAD^{tree}')}


def _job_id(value):
    job_id_number(value)
    return 'ƒ' + value[1:] if value.startswith('f') else value


def job_id_number(value):
    """The native numeric identity shared by Unicode/ASCII f58 and decimal IDs."""
    if not isinstance(value, str) or not re.fullmatch(r'(?:[1-9][0-9]*|[fƒ][1-9A-HJ-NP-Za-km-z]+)', value):
        raise ValueError('expected a saved native Flux job ID')
    number = int(value) if value[0] not in 'fƒ' else 0
    if value[0] in 'fƒ':
        for digit in value[1:]:
            number = number * 58 + _F58_ALPHABET.index(digit)
    if not 0 < number < 2**64:
        raise ValueError('native job ID exceeds uint64 or is zero')
    return number


def job_unit_name(job_id):
    """The qualified sdexec unit's canonical ASCII f58 job identity."""
    number, encoded = job_id_number(job_id), ''
    while number:
        number, digit = divmod(number, 58)
        encoded = _F58_ALPHABET[digit] + encoded
    return f'shell-f{encoded}:sys.service'


def _environment(record_dir, supplied):
    environment = {'PATH': '/usr/bin:/bin', 'HOME': '/home/rob',
                   'OUT': str(record_dir),
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
    _profile_identity()
    return source


def _shape(resources):
    return (f'slot/[core={resources.cores};memory={resources.memory_gib}'
            + (f';gpu={resources.gpus}]' if resources.gpus else ']'))


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
fallback_root, expected_fs = roots[host]
provided_root = os.environ.get('FLUX_SCRATCH')
root = fallback_root if provided_root is None else provided_root
if not root.startswith('/') or os.path.normpath(root) != root or root == '/':
    raise RuntimeError('invalid scratch root')
actual_fs = subprocess.check_output(['/usr/bin/findmnt', '-n', '-o', 'FSTYPE', '-T', root], text=True).strip()
if actual_fs != expected_fs:
    raise RuntimeError('unqualified scratch filesystem')
os.environ['FLUX_SCRATCH'] = root
# Native f58 spellings are equivalent; scientific paths require ASCII.
os.environ['FLUX_JOB_ID'] = os.environ['FLUX_JOB_ID'].replace('ƒ', 'f', 1)
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
    observation = {'schema': 'pq.flux.scratch.v1',
        'job_id': os.environ['FLUX_JOB_ID'], 'host': host, 'path': str(scratch),
        'root_source': 'instance' if provided_root is not None else 'host-table',
        'filesystem': actual_fs,
        'wrapper_pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0))}
    (out / 'scratch.json').write_text(json.dumps(observation))
    if received is None:
        child = subprocess.Popen(command, start_new_session=True)
        observation['child_pid'] = child.pid
        (out / 'scratch.json').write_text(json.dumps(observation))
        if received is not None:
            stop(received, None)
        code = child.wait()
    else:
        code = 128 + received
finally:
    process_group_empty = child is None
    if child is not None:
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            process_group_empty = True
    shutil.rmtree(scratch)
    (out / 'cleanup.json').write_text(json.dumps({'schema': 'pq.flux.cleanup.v1',
        'job_id': os.environ['FLUX_JOB_ID'], 'path': str(scratch), 'removed': not scratch.exists(),
        'process_group_empty': process_group_empty, 'signal': received}))
sys.exit(128 - code if code < 0 else code)
'''


def _native_arguments(argv, source, resources, record_dir, dependencies, environment, staged_input,
                      *, profile=_PROFILE):
    if profile == _PROFILE:
        shape = _shape(resources)
    elif profile == _LEGACY_PROFILE:
        shape = f'node/slot/[core={resources.cores}/perf=1;memory={resources.memory_gib}]'
    else:
        raise ValueError('unsupported saved Flux profile')
    command = list(argv)
    if staged_input is not None:
        command = ['/usr/bin/python3', '-s', '-m', 'prismaquant.staged_launch',
                   DIRECT_ASCII_SPACED_LAX.text(staged_input), *command]
    args = ['submit', '--env=-*', *(f'--env={key}={value}' for key, value in sorted(environment.items())),
            f'--resources-shape={shape}', f'--time-limit={resources.walltime_s}s',
            f'--cwd={source["checkout"]}', '--input=/dev/null', f'--output={record_dir}/stdout',
            f'--error={record_dir}/stderr', f'--job-name=pq-{record_dir.name}']
    if resources.placement != 'portable':
        args.append(f'--requires={resources.placement}')
    args.extend(f'--dependency=afterok:{value}' for value in dependencies)
    args.extend(['/usr/bin/python3', '-s', '-c', _PAYLOAD, source['commit'], *command])
    return args


def submit_job(argv, *, checkout: Path, resources: FluxResources, record_dir: Path,
               afterok=(), environment=None, staged_input=None) -> FluxSubmission:
    """Submit once. Existing records and lost acknowledgements require reconciliation.

    ``afterok`` contains saved native ID strings. ``host:NAME`` is reserved for
    a caller's justified data-owner/runtime dependency. Only CPU validation is
    qualified in profile v3. The coordinator supplies ``FLUX_URI`` or uses the
    fleet resolver. Unsupported input/output safety settings refuse.
    """
    if isinstance(argv, (str, bytes)):
        raise ValueError('argv must be a sequence, not a shell string')
    argv = tuple(argv)
    if not argv or any(not isinstance(a, str) or '\0' in a for a in argv):
        raise ValueError('argv must be a nonempty sequence of strings without NUL')
    if not isinstance(resources, FluxResources):
        raise TypeError('resources must be FluxResources')
    checkout, record_dir = Path(checkout).resolve(), Path(record_dir).resolve()
    if not record_dir.is_relative_to(_SHARED_ROOT) or record_dir.is_relative_to(checkout):
        raise ValueError('records require a separate shared directory outside the source checkout')
    dependencies = [_job_id(value) for value in afterok]
    if len({job_id_number(value) for value in dependencies}) != len(dependencies):
        raise ValueError('duplicate dependency ID')
    env = _environment(record_dir, environment)
    if staged_input is not None:
        from .staged_launch import validate_binding
        validate_binding(staged_input, placement=resources.placement)
    source, profile = _source_identity(checkout), _profile_identity()
    args = _native_arguments(argv, {**source, 'checkout': str(checkout)}, resources,
                             record_dir, dependencies, env, staged_input)
    intent = {'schema': 'pq.flux.launch.v1', 'source': source, 'profile': profile,
              'resources': asdict(resources), 'afterok': dependencies, 'argv': list(argv),
              'native_argv': args, 'record_dir': str(record_dir),
              'qualification': 'cpu-executor-only; finite-staged-v1; publication/pilots unqualified'}
    if staged_input is not None:
        intent['staged_input'] = staged_input
    record_dir.mkdir(parents=True, exist_ok=False)
    _publish_flux_json(record_dir / 'launch.json', intent)
    launch_sha = bytes_sha256hex(_json_bytes(intent))
    # Publish unknown durably BEFORE handing the intent to Flux. A crash at
    # any subsequent point must not make this directory eligible for retry.
    receipt = {'schema': 'pq.flux.submission.v1', 'launch_sha256': launch_sha,
               'state': 'unknown', 'job_id': None, 'profile': profile}
    _publish_flux_json(record_dir / 'submission.json', receipt)
    try:
        result = _flux_run(args, uri=profile['uri'])
        for name, raw in (('stdout', result.stdout), ('stderr', result.stderr)):
            try:
                (record_dir / f'submit.{name}').write_text(raw)
            except OSError:
                pass
        if result.returncode:
            raise SubmissionUnknown('native submission returned an error')
        job_id = _job_id(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError, SubmissionUnknown) as exc:
        raise SubmissionUnknown(f'submission unknown in {record_dir}; do not resubmit') from exc
    receipt.update(state='submitted', job_id=job_id)
    recovery_written = False
    try:
        # The acknowledged ID is recoverable even if the ordinary receipt,
        # diagnostics or human-readable jobid cannot subsequently be saved.
        _publish_flux_json(record_dir / 'submitted-job.json', receipt)
        recovery_written = True
        _publish_flux_json(record_dir / 'submission.json', receipt)
        (record_dir / 'jobid').write_text(job_id + '\n')
    except OSError as exc:
        recovery = ('load submitted-job.json to reconcile' if recovery_written else
                    'job id is only in this exception' +
                    (' and submit.stdout' if (record_dir / 'submit.stdout').is_file() else ''))
        raise SubmissionUnknown(
            f'submitted as {job_id} in {record_dir}; receipt publication failed; '
            f'{recovery}, do not resubmit') from exc
    return FluxSubmission(job_id, record_dir, launch_sha)


def load_submission(record_dir: Path) -> FluxSubmission:
    """Reconnect to a saved ID only; never submit or infer a producer by name."""
    return _read_submission(record_dir)[0]


def _native_document(raw):
    from .contract_checks import strict_json_loads
    def invalid(key):
        return ValueError(f'invalid native receipt JSON: {key}')
    return strict_json_loads(raw, duplicate=invalid, constant=invalid)


def _read_submission(record_dir):
    record_dir = Path(record_dir).resolve()
    from .staged_manifest import read_document
    raw, _ = read_document(record_dir / 'launch.json', 16 << 20)
    receipt_raw, _ = read_document(record_dir / 'submission.json', 1 << 20)
    intent = _native_document(raw)
    receipt = _native_document(receipt_raw)
    launch_sha = bytes_sha256hex(raw)
    if (not isinstance(intent, dict) or not isinstance(receipt, dict) or
            intent.get('schema') != 'pq.flux.launch.v1' or
            receipt.get('schema') != 'pq.flux.submission.v1' or
            receipt.get('launch_sha256') != launch_sha or intent.get('record_dir') != str(record_dir)):
        raise ValueError('invalid or modified Flux launch/submission receipt')
    recovery = record_dir / 'submitted-job.json'
    if recovery.exists():
        acknowledged = _native_document(read_document(recovery, 1 << 20)[0])
        if (not isinstance(acknowledged, dict) or
                acknowledged.get('schema') != 'pq.flux.submission.v1' or
                acknowledged.get('launch_sha256') != launch_sha or
                acknowledged.get('profile') != intent.get('profile') or
                acknowledged.get('state') != 'submitted' or
                not acknowledged.get('job_id')):
            raise ValueError('invalid acknowledged Flux job receipt')
        if receipt.get('state') == 'submitted' and receipt != acknowledged:
            raise ValueError('conflicting acknowledged Flux job receipt')
        if receipt.get('state') == 'unknown':
            if receipt.get('job_id') is not None or receipt.get('launch_sha256') != launch_sha:
                raise ValueError('invalid unknown Flux receipt')
            receipt = acknowledged
    if receipt.get('state') != 'submitted' or not receipt.get('job_id'):
        raise SubmissionUnknown(f'submission unknown in {record_dir}; reconcile, do not resubmit')
    return FluxSubmission(_job_id(receipt['job_id']), record_dir, launch_sha), intent


def _saved_launch(submission):
    saved, launch = _read_submission(submission.record_dir)
    if (saved.record_dir != submission.record_dir or saved.launch_sha256 != submission.launch_sha256
            or job_id_number(saved.job_id) != job_id_number(submission.job_id)):
        raise ValueError('submission does not match saved launch and ID')
    return launch


def _validate_submission(submission):
    intent = _saved_launch(submission)
    if intent.get('staged_input') is not None:
        from .staged_launch import validate_binding
        validate_binding(intent['staged_input'], placement=intent['resources']['placement'])
    if intent['profile']['broker_starttime'] != _checked(
            ['getattr', 'broker.starttime'], uri=intent['profile']['uri']).strip():
        raise RuntimeError('Flux instance changed; saved job identity cannot be assumed')
    return intent


def validate_dependency(submission: FluxSubmission):
    """Require a saved producer ID from the current standing instance."""
    launch = _validate_submission(submission)
    if launch['profile']['uri'] != _instance_uri():
        raise RuntimeError('saved dependency belongs to another Flux instance')


def validate_launch(submission: FluxSubmission, *, argv, source, resources,
                    afterok=(), environment=None, staged_input=None):
    """Bind application intent to every saved execution setting on resume/merge."""
    launch = _saved_launch(submission)
    if launch.get('staged_input') != staged_input:
        raise ValueError('saved staged-input binding differs from application intent')
    if staged_input is not None:
        from .staged_launch import validate_binding
        validate_binding(staged_input, placement=resources.placement)
    env = _environment(submission.record_dir, environment)
    dependencies = [_job_id(value) for value in afterok]
    if [job_id_number(value) for value in launch['afterok']] != [job_id_number(value) for value in dependencies]:
        raise ValueError('saved dependencies differ from application intent')
    expected = _native_arguments(argv, source,
                                 resources, submission.record_dir,
                                 launch['afterok'], env, staged_input, profile=launch['profile']['profile'])
    if (launch['argv'] != list(argv) or launch['source'] != source
            or launch['resources'] != asdict(resources)
            or launch['native_argv'] != expected):
        raise ValueError('saved execution differs from application intent')
    return launch


def verify_native_launch(submission: FluxSubmission, *, max_evidence_bytes, **intent):
    """Bind independent launch intent to the original actual native jobspec.

    Native environments are compared in memory and never included in errors.
    This queries the known ID only; it neither discovers nor submits jobs.
    """
    from .staged_manifest import read_document, decode_document, publish_staged_json
    launch = validate_launch(submission, **intent)
    if _validate_submission(submission) != launch:
        raise ValueError('saved launch changed during native verification')
    args = launch['native_argv']
    uri = launch['profile']['uri']
    expected = decode_document(_checked([args[0], '--dry-run', *args[1:]], uri=uri))
    actual_raw = _checked(['job', 'info', '--original', submission.job_id, 'jobspec'], uri=uri)
    if len(actual_raw.encode()) > max_evidence_bytes:
        raise ValueError('original native jobspec exceeds evidence bound')
    actual = decode_document(actual_raw)
    if actual != expected:
        raise ValueError('original native jobspec differs from selected launch intent')
    retained = submission.record_dir / 'original-jobspec.json'
    try:
        publish_staged_json(retained, actual)
    except FileExistsError:
        raw, _ = read_document(retained, max_evidence_bytes)
        if decode_document(raw) != actual:
            raise ValueError('retained original jobspec changed')
    return launch


def _allocation_ids(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*', value):
        raise ValueError('invalid native allocation ID set')
    result = set()
    for item in value.split(','):
        ends = [int(part) for part in item.split('-')]
        low, high = ends[0], ends[-1]
        if not 0 <= low <= high < 4096:
            raise ValueError('native allocation ID set exceeds bound')
        members = set(range(low, high + 1))
        if result & members:
            raise ValueError('native allocation repeats IDs')
        result.update(members)
    return result


def _require_allocation(allocation, resources, ranks):
    """Bind admitted counts and physical CPU identity without reading core class."""
    execution = allocation['execution']
    if (allocation['version'] != 1 or len(execution['R_lite']) != 1 or
            execution['nodelist'] != [socket.gethostname().split('.')[0]]):
        raise ValueError('current native allocation host/rank differs')
    row = execution['R_lite'][0]
    rank = int(row['rank'])
    if str(rank) != row['rank'] or rank not in ranks:
        raise ValueError('current native allocation rank differs')
    nodes = [node['metadata'] for node in allocation['scheduling']['graph']['nodes']]
    leaves = {kind: [node for node in nodes if node['type'] == kind]
              for kind in ('core', 'memory', 'gpu')}
    if any(len(leaves[kind]) != expected for kind, expected in
           (('core', resources.cores),
            ('memory', resources.memory_gib), ('gpu', resources.gpus))):
        raise ValueError('current native allocation differs from admitted resources')
    if any(node['rank'] != rank for members in leaves.values() for node in members):
        raise ValueError('current native resource leaves span ranks')
    core_ids = {node['id'] for node in leaves['core']}
    pus = set().union(*(_allocation_ids(node['properties']['os-pus']) for node in leaves['core']))
    if (len(core_ids) != resources.cores or core_ids != _allocation_ids(row['children']['core'])
            or row['children'].get('gpu') is not None or pus != set(os.sched_getaffinity(0))):
        raise ValueError('current native allocation/affinity differs')


def verify_current_job(*, job_id, record_dir, source):
    """Authenticate current CPU admission before using saved resource claims.

    A missing submit acknowledgement refuses. No receipt race permits trusting
    unverified launch fields, adopting another job or submitting a replacement.
    """
    from .staged_manifest import decode_document, publish_staged_json, read_document
    saved, launch = _read_submission(record_dir)
    if job_id_number(saved.job_id) != job_id_number(job_id) or source != launch['source']:
        raise ValueError('current native job/source differs from saved submission')
    settings = {'XDG_RUNTIME_DIR': '/run/user/1000'} if '--env=XDG_RUNTIME_DIR=/run/user/1000' in launch['native_argv'] else None
    resources = FluxResources(**launch['resources'])
    verify_native_launch(saved, max_evidence_bytes=16 << 20, argv=launch['argv'], source=source,
        resources=resources, afterok=launch['afterok'], environment=settings,
        staged_input=launch.get('staged_input'))
    uri = launch['profile']['uri']
    events = [_native_document(line) for line in
              _checked(['job', 'eventlog', '--format=json', saved.job_id], uri=uri).splitlines()]
    if not events or any(not isinstance(event, dict) or not isinstance(event.get('name'), str)
                         for event in events):
        raise ValueError('current native execution events unavailable')
    names = {event['name'] for event in events}
    if 'start' not in names or names.intersection(('finish', 'clean')):
        raise ValueError('current native job is not executing')
    scope = '/' + job_unit_name(saved.job_id)
    if not any(line.split(':', 2)[-1].endswith(scope)
               for line in Path('/proc/self/cgroup').read_text().splitlines()):
        raise ValueError('current native process scope differs from selected job')
    allocation = decode_document(_checked(['job', 'info', saved.job_id, 'R'], uri=uri))
    ranks = _instance_ranks(decode_document(_checked(['kvs', 'get', 'resource.R'], uri=uri)))
    _require_allocation(allocation, resources, ranks)
    retained = saved.record_dir / 'active-allocation.json'
    try:
        publish_staged_json(retained, allocation)
    except FileExistsError:
        if decode_document(read_document(retained, 16 << 20)[0]) != allocation:
            raise ValueError('retained active allocation changed')
    return saved, launch


def _query(submission, name, args, errors, *, required=True, timeout=30, preserve=False):
    try:
        result = _flux_run(args, timeout=timeout, uri=_saved_launch(submission)['profile']['uri'])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        errors.append(f'{name}: {type(exc).__name__}')
        return None
    out = submission.record_dir
    # A bound terminal failure keeps its original native proof. Reconnection
    # still queries fresh facts, retaining its latest response separately.
    retained = name + '.recollection' if preserve else name
    try:
        (out / f'{retained}.json').write_text(result.stdout)
        (out / f'{retained}.stderr').write_text(result.stderr)
        (out / f'{retained}.returncode').write_text(str(result.returncode) + '\n')
    except OSError as exc:
        errors.append(f'{name}: persistence {type(exc).__name__}')
    if required and result.returncode:
        errors.append(f'{name}: query exit {result.returncode}')
        return None
    return result


def _execution_errors(result, errors):
    if not errors:
        return result
    return replace(result, status=result.status if result.status in _FAILURES else 'unknown',
                   query_error='; '.join([*([result.query_error] if result.query_error else []), *errors]))


def _persist_execution(submission, result, filename='result.json'):
    try:
        _publish_flux_json(submission.record_dir / filename,
            {'schema': 'pq.flux.result.v1', **asdict(result), 'record_dir': str(result.record_dir),
             'launch_sha256': submission.launch_sha256})
    except OSError as exc:
        result = _execution_errors(result, [f'{filename}: persistence {type(exc).__name__}'])
    return result


def _previous_failure(submission):
    """Retain only a locally bound failure observation, never cached success."""
    from .staged_manifest import read_document
    _saved_launch(submission)
    path = submission.record_dir / 'result.json'
    if not path.exists():
        return None
    previous = _native_document(read_document(path, 1 << 20)[0])
    if not isinstance(previous, dict):
        raise ValueError('invalid prior native result')
    if (previous.get('schema') == 'pq.flux.result.v1'
            and previous.get('launch_sha256') == submission.launch_sha256
            and previous.get('record_dir') == str(submission.record_dir)
            and job_id_number(previous['job_id']) == job_id_number(submission.job_id)
            and previous.get('status') in _FAILURES
            and (previous.get('payload_exit') is None or type(previous['payload_exit']) is int)
            and (previous.get('query_error') is None or isinstance(previous['query_error'], str))):
        return ExecutionResult(previous['status'], submission.job_id, submission.record_dir,
                               previous['payload_exit'], previous['query_error'])
    return None


def collect_job(submission: FluxSubmission, *, wait: bool = True) -> ExecutionResult:
    """Collect native facts. Query failures are unknown, never payload failures.

    A negative payload_exit is a POSIX signal. R and redirected log paths are
    retained even for failure; absence before allocation/start is recorded.
    SIGKILL/node loss can prevent scratch cleanup and need owner reconciliation.
    """
    errors, cleanup_errors, events, payload_exit, status = [], [], [], None, 'unknown'
    previous = None
    try:
        previous = _previous_failure(submission)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f'previous observation: {type(exc).__name__}')
    identity_ok = False
    try:
        _validate_submission(submission)
        identity_ok = True
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        errors.append(f'identity: {type(exc).__name__}: {exc}')
    if identity_ok:
        if wait:
            _query(submission, 'wait', ['job', 'wait-event', submission.job_id, 'clean'], errors,
                   timeout=None, preserve=previous is not None)
        raw = _query(submission, 'eventlog', ['job', 'eventlog', '--format=json', submission.job_id], errors,
                     preserve=previous is not None)
        if raw is not None:
            try:
                events = [_native_document(line) for line in raw.stdout.splitlines()]
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
        raw_r = _query(submission, 'R', ['job', 'info', submission.job_id, 'R'], errors, required=allocated,
                       preserve=previous is not None)
        if allocated and raw_r is not None:
            try:
                allocation = _native_document(raw_r.stdout)
                if allocation['version'] != 1 or not allocation['execution']['R_lite']:
                    raise ValueError('missing allocation')
            except (ValueError, KeyError, TypeError):
                errors.append('R: invalid allocation JSON')
        if 'clean' in names:
            raw_status = _query(submission, 'status', ['job', 'status', '--json', '--exception-exit-code=125',
                                                       submission.job_id], errors, required=False,
                                preserve=previous is not None)
            try:
                native = _native_document(raw_status.stdout) if raw_status is not None else None
                if (not isinstance(native, dict) or type(native.get('id')) is not int or
                        native['id'] != job_id_number(submission.job_id) or
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
                elif (native['result'] == 1 and payload_exit == 0 and raw_status.returncode == 0
                      and native['exception_occurred'] is False
                      and not any(e.get('severity') == 0 for e in exceptions)):
                    status = 'succeeded'
                elif (native['result'] in (2, 8) or
                      (native['exception_occurred'] and raw_status.returncode != 0)):
                    status = 'failed'
                else:
                    errors.append('status: inconsistent native result')
            except (ValueError, TypeError, KeyError):
                errors.append('status: invalid/missing native result JSON')
        elif events:
            status = 'running'
        if 'start' in names and 'clean' in names:
            try:
                from .staged_manifest import read_document
                scratch = _native_document(read_document(submission.record_dir / 'scratch.json', 1 << 20)[0])
                cleanup = _native_document(read_document(submission.record_dir / 'cleanup.json', 1 << 20)[0])
                if (job_id_number(scratch['job_id']) != job_id_number(submission.job_id) or
                        job_id_number(cleanup['job_id']) != job_id_number(submission.job_id) or
                        cleanup['path'] != scratch['path'] or cleanup['removed'] is not True or
                        cleanup['process_group_empty'] is not True):
                    raise ValueError('scratch cleanup unverified')
            except (OSError, ValueError, KeyError, TypeError):
                cleanup_errors.append('scratch: cleanup unverified; reconcile owned descendants and path')
    for name in ('stdout', 'stderr'):
        if not (submission.record_dir / name).exists():
            try:
                (submission.record_dir / f'{name}.missing').write_text('No redirected payload stream observed.\n')
            except OSError as exc:
                errors.append(f'{name}: persistence {type(exc).__name__}')
    result = ExecutionResult(status, submission.job_id, submission.record_dir, payload_exit)
    if previous is not None:
        if status not in _FAILURES:
            errors.append('fresh native failure observation unavailable; retained failure is historical')
        if status in _FAILURES and (status, payload_exit) != (previous.status, previous.payload_exit):
            errors.append('native failure observations conflict; reconcile retained evidence')
        result = previous
    return _persist_execution(submission, _execution_errors(result, [*errors, *cleanup_errors]))


def require_stopped_job(submission: FluxSubmission, *, expected_launch):
    """Authenticate a known launch and fresh terminal facts before mutation."""
    verify_native_launch(submission, max_evidence_bytes=64 << 20,
        **{**expected_launch, 'resources': FluxResources(**expected_launch['resources'])})
    execution = collect_job(submission, wait=False)
    if execution.status not in {'succeeded', *_FAILURES} or execution.query_error:
        raise RuntimeError('native producer scope is not verified stopped')
    return execution


def cancel_job(submission: FluxSubmission) -> ExecutionResult:
    """Cancel the saved native ID and collect its actual terminal verdict."""
    try:
        _validate_submission(submission)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        result = ExecutionResult('unknown', submission.job_id, submission.record_dir)
        try:
            result = _previous_failure(submission) or result
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            pass
        result = _execution_errors(result, [f'cancel identity: {type(exc).__name__}: {exc}'])
        return _persist_execution(submission, result, 'cancel-result.json')
    errors = []
    _query(submission, 'cancel', ['cancel', submission.job_id], errors)
    result = collect_job(submission)
    if errors:
        result = _persist_execution(submission, _execution_errors(result, errors), 'cancel-result.json')
    return result
