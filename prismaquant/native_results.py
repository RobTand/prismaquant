"""Finite app artifact publication and one explicitly selected native producer.

Flux owns execution. This module has no scheduler, discovery, retries or cache.
Scientific admission remains with each consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import os
from pathlib import Path
import platform
import sys

from . import flux_jobs
from .digests import DIRECT_ASCII_STRICT, bytes_sha256hex, file_sha256hex
from .file_identity import file_stat_signature
from .schemas import Contract, strict_json_loads
from .staged_manifest import read_document, publish_staged_json

_contract = Contract(ValueError, 'native result: ')
SELECTION_SCHEMA = 'prismaquant.flux_result_selection.v1'
PUBLICATION_SCHEMA = 'prismaquant.flux_result_publication.v1'
PRODUCER_SCHEMA = 'prismaquant.flux_result_producer.v1'
RUNTIME_SCHEMA = 'prismaquant.flux_result_runtime.v1'
ANNOUNCEMENT_SCHEMA = 'prismaquant.flux_result_announcement.v1'
ANNOUNCEMENT_PREFIX = b'FLUX_RESULT_PUBLICATION '
SELECTION_KEYS = frozenset({'schema', 'job_id', 'record_dir', 'launch_sha256', 'publication',
    'expected_launch', 'producer_id', 'inputs', 'runtime', 'max_publication_bytes',
    'max_evidence_bytes', 'max_artifact_bytes'})


class ResultUnavailable(RuntimeError):
    """Query, evidence, identity or cleanup failure; never a payload verdict."""


class ProducerFailed(RuntimeError):
    def __init__(self, execution):
        self.execution = execution
        super().__init__(f'selected native producer {execution.status}: {execution.job_id}')


@dataclass(frozen=True)
class SelectedResult:
    publication: dict
    artifact_payloads: dict[str, bytes]
    input_payloads: dict[str, bytes]
    launch: dict
    execution: flux_jobs.ExecutionResult


def _fields(value, keys, label):
    return _contract.exact_mapping(value, keys=frozenset(keys), where=label)


def _document(raw):
    return strict_json_loads(raw,
        duplicate=lambda key: ValueError(f'native result: duplicate key {key}'),
        constant=lambda key: ValueError(f'native result: invalid constant {key}'))


def _binding(value):
    _fields(value, {'path', 'sha256', 'bytes'}, 'file binding')
    from .staged_manifest import path, digest, integer
    path(value['path']); digest(value['sha256']); integer(value['bytes'], positive=True)
    return value


def _owned(binding, limit, *, identity=None):
    _binding(binding)
    if binding['bytes'] > limit:
        raise ValueError('native result: file exceeds selected byte budget')
    raw, observed = read_document(binding['path'], binding['bytes'])
    if (len(raw) != binding['bytes'] or bytes_sha256hex(raw) != binding['sha256'] or
            file_stat_signature(os.stat(binding['path'], follow_symlinks=False)) != observed or
            (identity is not None and list(observed) != identity)):
        raise ValueError('native result: file digest/length/identity changed')
    return raw


def artifact_binding(filename, *, max_bytes):
    """Measure bounded owned bytes using the shared strict file reader."""
    raw, identity = read_document(filename, max_bytes)
    if not raw or file_stat_signature(os.stat(filename, follow_symlinks=False)) != identity:
        raise ValueError('native result: empty or replaced artifact')
    return {'path': str(filename), 'sha256': bytes_sha256hex(raw), 'bytes': len(raw),
            'file_identity': list(identity)}


def _runtime_file(filename):
    filename = Path(filename).resolve(strict=True)
    # Executables and installed wheel metadata are measured, not JSON assertions.
    size = filename.stat().st_size
    if not 0 < size <= 1 << 30:
        raise ValueError('native runtime file exceeds identity bound')
    return {'path': str(filename), 'bytes': size, 'sha256': file_sha256hex(filename)}


def measure_runtime(*, dependencies: tuple[str, ...] = ()) -> dict:
    if len(dependencies) > 64 or len(set(dependencies)) != len(dependencies):
        raise ValueError('native runtime requires finite unique dependency names')
    packages = {}
    for name in dependencies:
        dist = importlib.metadata.distribution(name)
        files = dist.files or ()
        bindings = {}
        for role, suffix in (('metadata', '.dist-info/METADATA'), ('record', '.dist-info/RECORD')):
            matches = [entry for entry in files if str(entry).endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f'native runtime requires installed {role}: {name}')
            bindings[role] = _runtime_file(dist.locate_file(matches[0]))
        packages[name] = {'version': dist.version, **bindings}
    return {'schema': RUNTIME_SCHEMA,
            'python': {**_runtime_file(sys.executable), 'version': platform.python_version()},
            'dependencies': packages}


def validate_runtime(value):
    _fields(value, {'schema', 'python', 'dependencies'}, 'runtime')
    _contract.require(value['schema'] == RUNTIME_SCHEMA, 'runtime schema differs')
    python = _fields(value['python'], {'path', 'sha256', 'bytes', 'version'}, 'interpreter')
    _binding({k: python[k] for k in ('path', 'sha256', 'bytes')})
    _contract.string(python['version'], where='interpreter version')
    deps = _contract.mapping(value['dependencies'], where='dependencies')
    _contract.require(len(deps) <= 64, 'dependency roster exceeds bound')
    for name, package in deps.items():
        _contract.string(name, where='dependency name')
        _fields(package, {'version', 'metadata', 'record'}, 'dependency')
        _contract.string(package['version'], where='dependency version')
        _binding(package['metadata']); _binding(package['record'])
    return value


def execution_context() -> dict:
    """Observe the actual job and clean source, without waiting for its receipt."""
    out = Path(os.environ['OUT'])
    raw, _ = read_document(out / 'launch.json', 16 << 20)
    launch = _document(raw)
    source = flux_jobs._source_identity(Path.cwd().resolve())
    if source != launch['source'] or str(out) != launch['record_dir']:
        raise ValueError('native result: executing source/record differs from launch')
    return {'schema': PRODUCER_SCHEMA, 'job_id': flux_jobs._job_id(os.environ['FLUX_JOB_ID']),
            'record_dir': str(out), 'launch_sha256': bytes_sha256hex(raw), 'source': source,
            'argv': launch['argv'], 'resources': launch['resources'], 'afterok': launch['afterok'],
            'staged_input': launch.get('staged_input')}


def _roster(value, *, limit, artifacts=False):
    _contract.require(isinstance(value, dict) and 0 < len(value) <= 64, 'finite nonempty role roster required')
    paths, total = set(), 0
    for role, row in value.items():
        _contract.string(role, where='role')
        if artifacts:
            _fields(row, {'path', 'sha256', 'bytes', 'file_identity'}, 'artifact')
            identity = row['file_identity']
            _contract.require(isinstance(identity, list) and len(identity) == 5 and
                all(type(item) is int and item >= 0 for item in identity), 'invalid artifact identity')
        else:
            _fields(row, {'path', 'sha256', 'bytes'}, 'input')
        _binding({key: row[key] for key in ('path', 'sha256', 'bytes')})
        _contract.require(row['path'] not in paths, 'role roster repeats a physical path')
        paths.add(row['path']); total += row['bytes']
    _contract.require(total <= limit, 'aggregate owned byte budget exceeded')


def publish_result(path: Path, *, producer_id: str, inputs: dict,
                   artifacts: dict[str, Path], dependencies: tuple[str, ...] = (),
                   max_artifact_bytes: int) -> dict:
    _contract.string(producer_id, where='producer ID')
    _ceiling(max_artifact_bytes, 1 << 30)
    _roster(inputs, limit=max_artifact_bytes)
    for binding in inputs.values():
        _owned(binding, max_artifact_bytes)
    if not isinstance(artifacts, dict) or not 0 < len(artifacts) <= 64:
        raise ValueError('native result: finite artifact roster required')
    measured = {role: artifact_binding(filename, max_bytes=max_artifact_bytes)
                for role, filename in artifacts.items()}
    _roster(measured, limit=max_artifact_bytes, artifacts=True)
    publication = {'schema': PUBLICATION_SCHEMA, 'producer_id': producer_id,
        'producer': execution_context(), 'runtime': measure_runtime(dependencies=dependencies),
        'inputs': inputs, 'artifacts': measured}
    raw = DIRECT_ASCII_STRICT.encoded(publication) + b'\n'
    if len(raw) > 16 << 20:
        raise ValueError('native result: publication exceeds bound')
    digest = publish_staged_json(path, publication)
    binding = {'path': str(path), 'sha256': digest, 'bytes': len(raw)}
    print((ANNOUNCEMENT_PREFIX + DIRECT_ASCII_STRICT.encoded(
        {'schema': ANNOUNCEMENT_SCHEMA, **binding})).decode('ascii'), flush=True)
    return binding


def _ceiling(value, maximum):
    _contract.require(type(value) is int and 0 < value <= maximum, 'invalid finite byte ceiling')


def read_selected_result(selection: dict) -> SelectedResult:
    """Read one publication only after authenticating actual terminal execution."""
    try:
        _fields(selection, SELECTION_KEYS, 'selection')
        _contract.require(selection['schema'] == SELECTION_SCHEMA, 'selection schema differs')
        for key, maximum in (('max_publication_bytes', 16 << 20), ('max_evidence_bytes', 64 << 20),
                             ('max_artifact_bytes', 1 << 30)):
            _ceiling(selection[key], maximum)
        from .staged_manifest import path, digest
        path(selection['record_dir']); digest(selection['launch_sha256'])
        _binding(selection['publication'])
        expected = _fields(selection['expected_launch'],
            {'argv', 'source', 'resources', 'afterok', 'environment', 'staged_input'}, 'expected launch')
        saved = flux_jobs.load_submission(Path(selection['record_dir']))
        if (flux_jobs.job_id_number(saved.job_id) != flux_jobs.job_id_number(selection['job_id']) or
                saved.launch_sha256 != selection['launch_sha256']):
            raise ValueError('native result: selected producer coordinates differ')
        launch = flux_jobs.verify_native_launch(saved,
            max_evidence_bytes=selection['max_evidence_bytes'],
            **{**expected, 'resources': flux_jobs.FluxResources(**expected['resources'])})
        execution = flux_jobs.collect_job(saved)
        # A conclusive native failure keeps its classification even if cleanup failed.
        if execution.status in ('failed', 'canceled', 'dependency-failed'):
            raise ProducerFailed(execution)
        if execution.status != 'succeeded' or execution.query_error:
            raise ResultUnavailable(f'native execution unavailable: {execution.status}; {execution.query_error}')
        evidence = 0
        for name in ('eventlog.json', 'status.json', 'R.json', 'stdout', 'stderr'):
            stream, _ = read_document(saved.record_dir / name, selection['max_evidence_bytes'] - evidence)
            evidence += len(stream)
            if name == 'stdout':
                stdout = stream
        raw = _owned(selection['publication'], selection['max_publication_bytes'])
        publication = _fields(_document(raw),
            {'schema', 'producer_id', 'producer', 'runtime', 'inputs', 'artifacts'}, 'publication')
        announcement = {'schema': ANNOUNCEMENT_SCHEMA, **selection['publication']}
        lines = [line[len(ANNOUNCEMENT_PREFIX):] for line in stdout.splitlines()
                 if line.startswith(ANNOUNCEMENT_PREFIX)]
        if lines != [DIRECT_ASCII_STRICT.encoded(announcement)]:
            raise ValueError('native result: exact retained publisher announcement missing or ambiguous')
        producer = {'schema': PRODUCER_SCHEMA, 'job_id': saved.job_id,
            'record_dir': str(saved.record_dir), 'launch_sha256': saved.launch_sha256,
            **{key: launch[key] for key in ('source', 'argv', 'resources', 'afterok')},
            'staged_input': launch.get('staged_input')}
        observed_producer = dict(publication['producer'])
        observed_producer['job_id'] = flux_jobs._job_id(observed_producer['job_id'])
        if (publication['schema'] != PUBLICATION_SCHEMA or publication['producer_id'] != selection['producer_id']
                or observed_producer != producer or publication['inputs'] != selection['inputs']
                or validate_runtime(publication['runtime']) != validate_runtime(selection['runtime'])):
            raise ValueError('native result: producer/source/input/runtime differs from selection')
        _roster(publication['inputs'], limit=selection['max_artifact_bytes'])
        _roster(publication['artifacts'], limit=selection['max_artifact_bytes'], artifacts=True)
        if sum(row['bytes'] for roster in ('inputs', 'artifacts')
               for row in publication[roster].values()) > selection['max_artifact_bytes']:
            raise ValueError('native result: aggregate input/artifact budget exceeded')
        input_payloads = {role: _owned(binding, selection['max_artifact_bytes'])
                          for role, binding in publication['inputs'].items()}
        artifact_payloads = {role: _owned({k: row[k] for k in ('path', 'sha256', 'bytes')},
                            selection['max_artifact_bytes'], identity=row['file_identity'])
                             for role, row in publication['artifacts'].items()}
        return SelectedResult(publication, artifact_payloads, input_payloads, launch, execution)
    except (ProducerFailed, ResultUnavailable):
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise ResultUnavailable(str(exc)) from exc
