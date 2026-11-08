"""Stdlib boundary tests; run directly in admitted Flux, without PB conftest."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from prismaquant import flux_jobs as jobs


class FluxJobsTests(unittest.TestCase):
    def test_ascii_and_unicode_f58_are_one_saved_identity(self):
        submission, _ = self.submit_fixture(self.completed('fBqv3wNoLK\n'))
        self.assertEqual(submission.job_id, 'ƒBqv3wNoLK')
        self.assertEqual(jobs.load_submission(self.record), submission)
        self.assertEqual(jobs.job_id_number('fBqv3wNoLK'),
                         jobs.job_id_number('ƒBqv3wNoLK'))
        # Independently captured native status from the pre-fix admitted run.
        self.assertEqual(jobs.job_id_number('fE23pNdDTV'), 1667135277367296)

    def test_launch_encoding_keeps_exact_utf8_lax_identity_bytes(self):
        self.assertEqual(jobs._json_bytes({'n': float('nan'), 'a': 'ƒ'}),
                         b'{"a":"\xc6\x92","n":NaN}\n')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ['OUT'])
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkout = self.root / 'source'
        self.checkout.mkdir()
        self.record = self.root / 'record'
        self.source = {'checkout': str(self.checkout), 'commit': 'a' * 40, 'tree': 'b' * 40}
        self.profile = {'profile': jobs._PROFILE, 'uri': jobs._URI, 'broker_starttime': '123.5'}
        self.resources = jobs.FluxResources(2, 3, 0, 60)

    def completed(self, stdout='', rc=0, stderr=''):
        return subprocess.CompletedProcess([], rc, stdout, stderr)

    def submit_fixture(self, response=None, **kwargs):
        if response is None:
            response = self.completed('ƒBqv3wNoLK\n')
        with patch.object(jobs, '_source_identity', return_value=self.source), \
                patch.object(jobs, '_profile_identity', return_value=self.profile), \
                patch.object(jobs, '_flux_run', return_value=response) as run:
            submission = jobs.submit_job(['/bin/echo', 'known'], checkout=self.checkout,
                                         resources=self.resources, record_dir=self.record, **kwargs)
        return submission, run

    def test_profile_renders_real_perf_and_memory_and_portability(self):
        submission, run = self.submit_fixture(afterok=('123', '456'))
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ['submit', '--env=-*'])
        self.assertIn('--resources-shape=node/slot/[core=2/perf=1;memory=3]', command)
        self.assertIn('--time-limit=60s', command)
        self.assertIn('--dependency=afterok:123', command)
        self.assertIn('--dependency=afterok:456', command)
        self.assertFalse(any(a.startswith(('-N', '--nodes', '--requires')) for a in command))
        exported = [a for a in command if a.startswith('--env=')][1:]
        self.assertEqual(len(exported), 10)
        self.assertIn('--env=OMP_NUM_THREADS=1', exported)
        self.assertIn('--env=OUT=' + str(self.record), exported)
        self.assertEqual(jobs.load_submission(self.record), submission)
        self.assertEqual(json.loads((self.record / 'submission.json').read_text())['state'], 'submitted')

    def test_placement_is_explicit_and_closed(self):
        resources = jobs.FluxResources(1, 1, 0, 30, 'host:dl380g10')
        self.resources = resources
        _, run = self.submit_fixture()
        self.assertIn('--requires=host:dl380g10', run.call_args.args[0])
        for placement in ('perf', 'nic-sparky', 'host:unknown', 'aarch64|host:dl380g10'):
            with self.subTest(placement=placement), self.assertRaises(ValueError):
                jobs.FluxResources(1, 1, 0, 30, placement)

    def test_bounds_do_not_silently_round_or_allow_gpu(self):
        for args in ((0, 1, 0, 1), (5, 1, 0, 1), (True, 1, 0, 1), (1, 17, 0, 1),
                     (1, 1, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1801), (1, 1.5, 0, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                jobs.FluxResources(*args)

    def test_credentials_context_and_unqualified_contracts_refuse_before_submit(self):
        for setting in ({'API_TOKEN': 'fixture'}, {'PB_ACTION_KEY': 'fixture'},
                        {'PQ_REQUIRE_STRICT_STAGING': '1'}, {'PQ_REQUIRE_AUTHENTICATED_OUTPUT': '1'},
                        {'PATH': '/tmp/custom'}, {'OMP_NUM_THREADS': '3'}, {'OUT': '/tmp/foreign'}):
            with self.subTest(keys=list(setting)), patch.object(jobs, '_flux_run') as run, self.assertRaises(ValueError):
                jobs.submit_job(['/bin/true'], checkout=self.checkout, resources=self.resources,
                                record_dir=self.record, environment=setting)
            run.assert_not_called()
        self.assertFalse(self.record.exists())

    def test_invalid_intent_is_never_submitted(self):
        for argv in ('shell string', [], ['bad\0arg']):
            with self.subTest(argv=argv), patch.object(jobs, '_flux_run') as run, self.assertRaises(ValueError):
                jobs.submit_job(argv, checkout=self.checkout, resources=self.resources, record_dir=self.record)
            run.assert_not_called()
        for dependencies in (('not-a-native-ID',), ('--help',), ('123', '123'), ('f1',)):
            with self.subTest(dependencies=dependencies), self.assertRaises(ValueError):
                jobs.submit_job(['/bin/true'], checkout=self.checkout, resources=self.resources,
                                record_dir=self.record, afterok=dependencies)
        with self.assertRaises(ValueError):
            jobs.submit_job(['/bin/true'], checkout=self.checkout, resources=self.resources,
                            record_dir=self.checkout / 'record')

    def test_lost_ack_is_durable_and_no_blind_resubmit(self):
        with patch.object(jobs, '_source_identity', return_value=self.source), \
                patch.object(jobs, '_profile_identity', return_value=self.profile), \
                patch.object(jobs, '_flux_run', side_effect=subprocess.TimeoutExpired('flux', 30)) as run:
            with self.assertRaises(jobs.SubmissionUnknown):
                jobs.submit_job(['/bin/true'], checkout=self.checkout,
                                resources=self.resources, record_dir=self.record)
            self.assertEqual(run.call_count, 1)
            with self.assertRaises(jobs.SubmissionUnknown):
                jobs.load_submission(self.record)
            with self.assertRaises(FileExistsError):
                jobs.submit_job(['/bin/true'], checkout=self.checkout,
                                resources=self.resources, record_dir=self.record)
            self.assertEqual(run.call_count, 1)
        receipt = json.loads((self.record / 'submission.json').read_text())
        self.assertEqual(receipt['state'], 'unknown')
        self.assertIsNone(receipt['job_id'])
        self.assertTrue((self.record / 'launch.json').exists())

    def test_bad_submit_responses_stay_unknown(self):
        for index, response in enumerate((self.completed(''), self.completed('123\n456'),
                                          self.completed('123\n', 1), self.completed('bad id'))):
            self.record = self.root / f'record-{index}'
            with self.subTest(index=index), self.assertRaises(jobs.SubmissionUnknown):
                self.submit_fixture(response)
            with self.assertRaises(jobs.SubmissionUnknown):
                jobs.load_submission(self.record)

    def test_modified_launch_and_foreign_handle_refuse(self):
        submission, _ = self.submit_fixture()
        foreign = jobs.FluxSubmission('123', self.record, submission.launch_sha256)
        with self.assertRaises(ValueError):
            jobs.cancel_job(foreign)
        intent = json.loads((self.record / 'launch.json').read_text())
        intent['argv'] = ['/bin/false']
        (self.record / 'launch.json').write_text(json.dumps(intent))
        with self.assertRaises(ValueError):
            jobs.load_submission(self.record)

    def test_cancellation_query_failure_never_becomes_payload_failure(self):
        submission, _ = self.submit_fixture()
        with patch.object(jobs, '_flux_run', return_value=self.completed('', 1, 'fixture transport unavailable')) as run:
            result = jobs.cancel_job(submission)
        self.assertEqual((result.status, result.payload_exit), ('unknown', None))
        self.assertIn('cancel identity:', result.query_error)
        self.assertEqual([call.args[0] for call in run.call_args_list], [['getattr', 'broker.starttime']])

    def test_saved_launch_requires_exact_placement_and_effective_environment(self):
        environment = {'XDG_RUNTIME_DIR': '/run/user/1000'}
        submission, _ = self.submit_fixture(environment=environment)
        intent = dict(argv=['/bin/echo', 'known'], source=self.source,
                      resources=self.resources, environment=environment)
        jobs.validate_launch(submission, **intent)
        with self.assertRaisesRegex(ValueError, 'application intent'):
            jobs.validate_launch(submission, **{**intent, 'environment': {}})
        with self.assertRaisesRegex(ValueError, 'application intent'):
            jobs.validate_launch(submission, **{**intent, 'resources': jobs.FluxResources(2, 3, 0, 60, 'gb10')})

    def test_instance_replacement_refuses_cancellation_of_reused_id(self):
        submission, _ = self.submit_fixture()
        with patch.object(jobs, '_flux_run', return_value=self.completed('new-generation\n')) as run:
            result = jobs.cancel_job(submission)
        self.assertEqual(result.status, 'unknown')
        self.assertIn('instance changed', result.query_error)
        self.assertEqual([call.args[0] for call in run.call_args_list], [['getattr', 'broker.starttime']])

    def test_saved_dependencies_accept_native_aliases_without_rewriting_launch(self):
        submission, _ = self.submit_fixture(afterok=('fE23pNdDTV',))
        before = (self.record / 'launch.json').read_bytes()
        for dependency in ('fE23pNdDTV', 'ƒE23pNdDTV', '1667135277367296'):
            jobs.validate_launch(submission, argv=['/bin/echo', 'known'], source=self.source,
                                 resources=self.resources, afterok=(dependency,))
        self.assertEqual((self.record / 'launch.json').read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'dependencies'):
            jobs.validate_launch(submission, argv=['/bin/echo', 'known'], source=self.source,
                                 resources=self.resources, afterok=('123',))

    def test_malformed_receipt_is_a_boundary_refusal(self):
        self.submit_fixture()
        (self.record / 'launch.json').write_text('[]')
        with self.assertRaises(ValueError):
            jobs.load_submission(self.record)

    def collect_fixture(self, *, finish=0, result=1, exception=None, malformed=None,
                        clean=True, allocated=True, status_rc=None, wrong_id=False, cleanup_ok=True):
        submission, _ = self.submit_fixture()
        events = [{'name': 'submit'}]
        if allocated:
            events += [{'name': 'alloc'}, {'name': 'start'}]
            (self.record / 'scratch.json').write_text(json.dumps({'job_id': submission.job_id, 'path': '/fixture'}))
            (self.record / 'cleanup.json').write_text(json.dumps({'job_id': submission.job_id, 'path': '/fixture', 'removed': cleanup_ok, 'process_group_empty': cleanup_ok}))
        if finish is not None:
            events.append({'name': 'finish', 'context': {'status': finish}})
        if exception:
            events.append({'name': 'exception', 'context': {'type': exception}})
        if clean:
            events.append({'name': 'clean'})
        def run(args, **kwargs):
            if args[0] == 'getattr':
                return self.completed('123.5\n')
            if args[1] == 'eventlog':
                return self.completed('bad-json' if malformed == 'eventlog' else '\n'.join(map(json.dumps, events)))
            if args[1] == 'info':
                return self.completed(json.dumps({'version': 1, 'execution': {'R_lite': [{'rank': '0'}]}}), 0 if allocated else 1)
            if args[1] == 'status':
                native = {'id': 123 if wrong_id else jobs.job_id_number(submission.job_id), 'exception_occurred': exception is not None, 'result': result, 'waitstatus': finish}
                return self.completed('bad-json' if malformed == 'status' else json.dumps(native),
                                      status_rc if status_rc is not None else (0 if result == 1 else 125))
            if args[1] == 'wait-event':
                return self.completed()
            self.fail('unexpected native query')
        with patch.object(jobs, '_flux_run', side_effect=run):
            return jobs.collect_job(submission, wait=clean)

    def test_terminal_success_requires_positive_native_evidence(self):
        result = self.collect_fixture()
        self.assertEqual((result.status, result.payload_exit, result.query_error), ('succeeded', 0, None))
        self.assertTrue((self.record / 'R.json').exists())
        self.assertTrue((self.record / 'eventlog.json').exists())

    def test_payload_failure_is_separate_from_query_failure(self):
        result = self.collect_fixture(finish=7 << 8, result=2, status_rc=7)
        self.assertEqual((result.status, result.payload_exit, result.query_error), ('failed', 7, None))

    def test_other_native_status_id_never_authorizes_selected_job(self):
        result = self.collect_fixture(wrong_id=True)
        self.assertEqual(result.status, 'unknown')
        self.assertIn('status:', result.query_error)

    def test_conclusive_failure_survives_cleanup_error(self):
        result = self.collect_fixture(finish=7 << 8, result=2, status_rc=7, cleanup_ok=False)
        self.assertEqual((result.status, result.payload_exit), ('failed', 7))
        self.assertIn('cleanup unverified', result.query_error)

    def test_success_without_cleanup_stays_unavailable(self):
        result = self.collect_fixture(cleanup_ok=False)
        self.assertEqual((result.status, result.payload_exit), ('unknown', 0))
        self.assertIn('cleanup unverified', result.query_error)

    def test_query_failure_keeps_observed_payload_exit(self):
        result = self.collect_fixture(finish=7 << 8, result=2, malformed='status')
        self.assertEqual((result.status, result.payload_exit), ('unknown', 7))
        self.assertIn('status:', result.query_error)

    def test_dependency_failure_does_not_require_nonexistent_allocation(self):
        result = self.collect_fixture(finish=None, result=2, exception='dependency', allocated=False)
        self.assertEqual((result.status, result.payload_exit, result.query_error), ('dependency-failed', None, None))
        self.assertTrue((self.record / 'stdout.missing').exists())

    def test_cancel_is_terminal_native_result(self):
        result = self.collect_fixture(finish=15, result=4, exception='cancel')
        self.assertEqual((result.status, result.payload_exit, result.query_error), ('canceled', -15, None))

    def test_nonblocking_collection_does_not_wait_or_claim_success(self):
        result = self.collect_fixture(finish=None, clean=False)
        self.assertEqual((result.status, result.payload_exit, result.query_error), ('running', None, None))
        self.assertFalse((self.record / 'status.json').exists())

    def test_bad_eventlog_is_unknown(self):
        result = self.collect_fixture(malformed='eventlog')
        self.assertEqual(result.status, 'unknown')
        self.assertIn('eventlog:', result.query_error)

    def test_real_git_source_identity_refuses_dirty_attached_and_ignored(self):
        def git(*args):
            return subprocess.run(['git', '-C', str(self.checkout), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        git('init', '-q')
        (self.checkout / 'payload').write_text('known source\n')
        (self.checkout / '.gitignore').write_text('ignored\n')
        git('add', 'payload', '.gitignore')
        git('-c', 'user.name=Flux fixture', '-c', 'user.email=fixture@invalid', 'commit', '-qm', 'fixture')
        with self.assertRaisesRegex(ValueError, 'detached'):
            jobs._source_identity(self.checkout)
        git('checkout', '--detach', '-q')
        identity = jobs._source_identity(self.checkout)
        self.assertEqual(identity['commit'], git('rev-parse', 'HEAD'))
        (self.checkout / 'payload').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'clean'):
            jobs._source_identity(self.checkout)
        git('checkout', '--', 'payload')
        (self.checkout / 'ignored').write_text('untracked runtime\n')
        with self.assertRaisesRegex(ValueError, 'clean'):
            jobs._source_identity(self.checkout)

    def test_profile_drift_fails_closed(self):
        with patch.object(jobs, '_CONFIG', self.root / 'config'), patch.object(jobs, '_MODEL', self.root / 'model'):
            (self.root / 'config').write_text('{}')
            (self.root / 'model').write_text('{}')
            with patch.object(jobs, '_flux_run') as run, self.assertRaisesRegex(ValueError, 'profile revision'):
                jobs._profile_identity()
            run.assert_not_called()

    def test_client_never_inherits_ambient_environment(self):
        with patch.dict(os.environ, {'PB_ACTION_KEY': 'fixture', 'API_TOKEN': 'fixture', 'FLUX_KVS_NAMESPACE': 'fixture'}), \
                patch.object(subprocess, 'run', return_value=self.completed()) as run:
            jobs._flux_run(['resource', 'list'])
        self.assertEqual(set(run.call_args.kwargs['env']), {'PATH', 'HOME', 'LD_LIBRARY_PATH', 'FLUX_URI'})


if __name__ == '__main__':
    unittest.main()
