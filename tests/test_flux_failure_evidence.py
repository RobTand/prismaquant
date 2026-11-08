"""Independent parent controls; execute only in admitted Flux."""
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
import test_flux_jobs
from prismaquant import flux_jobs as jobs
from prismaquant import native_results as nr


@pytest.fixture
def case():
    value = test_flux_jobs.FluxJobsTests()
    value.setUp()
    try:
        yield value
    finally:
        value.doCleanups()


@pytest.mark.parametrize('native,expected,finish,exception,fault', [
    (*ending, fault)
    for ending in [(2, 'failed', 7 << 8, None), (4, 'canceled', 15, 'cancel'),
                   (2, 'dependency-failed', None, 'dependency')]
    for fault in ['wait', 'R-query', 'R-json', 'query-write', 'result-write']
    if not (ending[1] == 'dependency-failed' and fault == 'R-json')])
def test_conclusive_failure_survives_ancillary_error(case, native, expected, finish, exception, fault):
    assert case.collect_fixture().status == 'succeeded'
    case.record = case.root / 'negative'
    query = jobs._query
    publish = jobs._publish_flux_json
    write = Path.write_text

    def queried(submission, name, args, errors, **kwargs):
        if (fault == 'wait' and name == 'wait') or (fault == 'R-query' and name == 'R'):
            errors.append(name + ': injected transport error')
            return None
        response = query(submission, name, args, errors, **kwargs)
        if fault == 'R-json' and name == 'R':
            return subprocess.CompletedProcess([], 0, 'invalid allocation', '')
        return response

    def written(path, *args, **kwargs):
        if fault == 'query-write' and path.name == 'status.json':
            raise OSError('injected status persistence error')
        return write(path, *args, **kwargs)

    def published(path, *args, **kwargs):
        if fault == 'result-write' and path.name == 'result.json':
            raise OSError('injected result persistence error')
        return publish(path, *args, **kwargs)

    with patch.object(jobs, '_query', side_effect=queried), \
            patch.object(Path, 'write_text', written), \
            patch.object(jobs, '_publish_flux_json', side_effect=published):
        observed = case.collect_fixture(finish=finish, result=native, exception=exception,
                                         allocated=expected != 'dependency-failed')
    assert observed.status == expected, observed
    assert observed.payload_exit == (None if finish is None else (7 if native == 2 else -15))
    assert observed.query_error


def test_failed_collection_is_monotone_but_success_is_not(case):
    failed = case.collect_fixture(finish=7 << 8, result=2, status_rc=7)
    assert failed.status == 'failed'
    saved = jobs.load_submission(case.record)
    with patch.object(jobs, '_flux_run', side_effect=subprocess.TimeoutExpired('flux', 30)):
        observed = jobs.collect_job(saved)
    assert (observed.status, observed.payload_exit) == ('failed', 7), observed
    assert observed.query_error
    case.record = case.root / 'success'
    assert case.collect_fixture().status == 'succeeded'
    saved = jobs.load_submission(case.record)
    with patch.object(jobs, '_flux_run', side_effect=subprocess.TimeoutExpired('flux', 30)):
        observed = jobs.collect_job(saved)
    assert observed.status == 'unknown', observed


@pytest.mark.parametrize('native,expected,finish,exception', [
    (2, 'failed', 7 << 8, None), (4, 'canceled', 15, 'cancel'),
    (2, 'dependency-failed', None, 'dependency')])
@pytest.mark.parametrize('name', ['eventlog', 'status', 'R'])
@pytest.mark.parametrize('fault', ['nonzero-empty', 'malformed'])
def test_failed_recollection_preserves_original_native_bytes(case, native, expected, finish, exception, name, fault):
    positive = case.collect_fixture(finish=finish, result=native, exception=exception,
                                   allocated=expected != 'dependency-failed')
    assert positive.status == expected and positive.query_error is None
    saved = jobs.load_submission(case.record)
    held = {key: (case.record / (key + '.json')).read_bytes() for key in ('eventlog', 'status', 'R')}
    def queried(args, **kwargs):
        if args[0] == 'getattr': return case.completed('123.5\n')
        if args[1] == 'wait-event': return case.completed()
        key = 'R' if args[1] == 'info' else args[1]
        if key == name:
            return case.completed('' if fault == 'nonzero-empty' else '{broken',
                                  1 if fault == 'nonzero-empty' else 0, 'retained query diagnostic')
        return case.completed(held[key].decode(), 125 if key == 'status' else 0)
    with patch.object(jobs, '_flux_run', side_effect=queried):
        result = jobs.collect_job(saved)
    assert result.status == expected and result.payload_exit == positive.payload_exit
    # No R is expected for an unallocated dependency failure. Its unsuccessful
    # allocation query is retained as absence evidence, not a new query error.
    assert result.query_error or (expected == 'dependency-failed' and name == 'R')
    assert {key: (case.record / (key + '.json')).read_bytes() for key in held} == held
    assert (case.record / (name + '.recollection.stderr')).read_text() == 'retained query diagnostic'
    assert (case.record / (name + '.recollection.json')).read_text() == ('' if fault == 'nonzero-empty' else '{broken')


@pytest.mark.parametrize('fault', ['nonzero-empty', 'malformed'])
def test_failed_fresh_query_cannot_reuse_success(case, fault):
    assert case.collect_fixture().status == 'succeeded'
    saved = jobs.load_submission(case.record)
    held = {key: (case.record / (key + '.json')).read_text() for key in ('eventlog', 'status', 'R')}
    def queried(args, **kwargs):
        if args[0] == 'getattr': return case.completed('123.5\n')
        if args[1] == 'wait-event': return case.completed()
        key = 'R' if args[1] == 'info' else args[1]
        if key == 'status':
            return case.completed('' if fault == 'nonzero-empty' else '{broken', 1 if fault == 'nonzero-empty' else 0)
        return case.completed(held[key])
    with patch.object(jobs, '_flux_run', side_effect=queried):
        result = jobs.collect_job(saved)
    assert result.status == 'unknown' and result.query_error


def test_cancel_error_preserves_observed_failure(case):
    assert case.collect_fixture(finish=7 << 8, result=2, status_rc=7).status == 'failed'
    saved = jobs.load_submission(case.record)
    replies = {name: (case.record / (name + '.json')).read_text()
               for name in ('eventlog', 'R', 'status')}

    def run(args, **kwargs):
        if args[0] == 'getattr':
            return case.completed('123.5\n')
        if args[0] == 'cancel':
            return case.completed('', 1, 'already terminal')
        if args[1] == 'wait-event':
            return case.completed()
        name = 'R' if args[1] == 'info' else args[1]
        return case.completed(replies[name], 7 if name == 'status' else 0)

    with patch.object(jobs, '_flux_run', side_effect=run):
        observed = jobs.cancel_job(saved)
    assert (observed.status, observed.payload_exit) == ('failed', 7), observed
    assert 'cancel' in observed.query_error


@pytest.mark.parametrize('field,value', [('job_id', '123'), ('launch_sha256', '0' * 64),
                                         ('record_dir', '/foreign/record')])
def test_prior_failure_must_match_selected_coordinates(case, field, value):
    assert case.collect_fixture(finish=7 << 8, result=2, status_rc=7).status == 'failed'
    saved = jobs.load_submission(case.record)
    previous = json.loads((case.record / 'result.json').read_bytes())
    previous[field] = value
    (case.record / 'result.json').write_text(json.dumps(previous))
    with patch.object(jobs, '_flux_run', side_effect=subprocess.TimeoutExpired('flux', 30)):
        observed = jobs.collect_job(saved)
    assert observed.status == 'unknown', observed


def test_selected_failure_precedes_unavailable_publication_verification(case):
    assert case.collect_fixture(finish=7 << 8, result=2, status_rc=7).status == 'failed'
    saved = jobs.load_submission(case.record)
    selection = {'schema': nr.SELECTION_SCHEMA, 'job_id': saved.job_id,
        'record_dir': str(case.record), 'launch_sha256': saved.launch_sha256,
        'publication': {'path': '/missing/publication', 'sha256': 'a' * 64, 'bytes': 1},
        'expected_launch': {'argv': ['/bin/echo', 'known'], 'source': case.source,
            'resources': vars(case.resources), 'afterok': [], 'environment': {}, 'staged_input': None},
        'producer_id': 'nonqualifying-fixture', 'inputs': {}, 'runtime': {},
        'max_publication_bytes': 1024, 'max_evidence_bytes': 1024, 'max_artifact_bytes': 1024}
    with patch.object(jobs, '_flux_run', side_effect=subprocess.TimeoutExpired('flux', 30)), \
            patch.object(jobs, 'verify_native_launch', side_effect=AssertionError('preempted failure')):
        with pytest.raises(nr.ProducerFailed) as failed:
            nr.read_selected_result(selection)
    assert (failed.value.execution.status, failed.value.execution.payload_exit) == ('failed', 7)
