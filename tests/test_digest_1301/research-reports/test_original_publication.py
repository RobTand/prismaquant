"""Pre-route consumer characterizations; deliberately do not launch workloads.

Original fragments are independent source fixtures, not an oracle computed by
any digest owner. Extract only the live helper under test where importing the
whole script would parse argv, contact hosts or load numerical dependencies.
"""
import ast
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/digest_1301/research-reports'
ORIGINAL = json.loads((FIXTURES / 'original_consumers.json').read_text())


def consumer(file, function, *, original, **dependencies):
    record = next(r for r in ORIGINAL['consumers'] if r['file'] == file)
    if original:
        sources = record['functions']
    else:
        text = (ROOT / file).read_text()
        sources = {n.name: ast.get_source_segment(text, n)
                   for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
                   and n.name in record['functions']}
    namespace = {'json': json, 'Path': Path, 'subprocess': subprocess,
                 '_STATUS_SCHEMA': 'prismaquant.profile_workload_status.v1',
                 **dependencies}
    exec('from __future__ import annotations\n' + '\n\n'.join(sources.values()), namespace)
    return namespace[function]


def test_strict_trial_publication_preserves_order_lf_and_refusals(tmp_path):
    outcomes = []
    for original in (True, False):
        path = tmp_path / str(original)
        write = consumer('tools/pq_block_trial_math.py', 'write_trial_json', original=original)
        document = {'z': '\u00e9', 'a': [1, -0.0], 3: True}
        write(path, document)
        assert path.read_bytes() == b'{\n  "z": "\\u00e9",\n  "a": [\n    1,\n    -0.0\n  ],\n  "3": true\n}\n'
        for invalid in ({'bad': float('nan')}, {'bad': object()}):
            before = path.read_bytes()
            with pytest.raises((ValueError, TypeError)) as error:
                write(path, invalid)
            # dumps happens before write_text: existing output is intact.
            assert path.read_bytes() == before
            outcomes.append((type(error.value), str(error.value)))
    assert outcomes[:2] == outcomes[2:]


def test_profile_status_atomic_publication_and_refusals(tmp_path):
    outcomes = []
    for original in (True, False):
        profile = tmp_path / str(original)
        status = profile.with_name(profile.name + '.workload-status.json')
        status.write_text('{"schema":"prismaquant.profile_workload_status.v1",'
                          '"returncode":7,"workload_pid":42,"note":"\\u00e9"}')
        finish = consumer('tools/pq_profile_artifact.py', 'finish_profile_status', original=original)
        result = finish(profile, 3)
        expected = (b'{"schema": "prismaquant.profile_workload_status.v1", '
                    b'"returncode": 7, "workload_pid": 42, "note": "\\u00e9", '
                    b'"profiler_returncode": 3}')
        assert status.read_bytes() == expected
        assert result['returncode'] == 7 and result['profiler_returncode'] == 3
        assert not status.with_name(status.name + '.finalizing').exists()
        partial = status.with_name(status.name + '.finalizing')
        partial.write_bytes(b'occupied')
        with pytest.raises(FileExistsError):
            finish(profile, 0)
        assert status.read_bytes() == expected and partial.read_bytes() == b'occupied'
        with pytest.raises(RuntimeError) as error:
            finish(profile, True)
        outcomes.append(str(error.value))
    assert outcomes == ['profiler completion has no valid return code'] * 2


def test_child_status_waits_before_publication_and_keeps_workload_code(tmp_path):
    for original in (True, False):
        status = tmp_path / str(original)
        events = []

        def wait():
            assert not status.exists()
            events.append('wait')
            return 7

        def popen(command):
            events.append(command)
            return SimpleNamespace(pid=42, wait=wait)

        run = consumer('tools/pq_profile_child.py', 'run_profile_workload', original=original,
                       subprocess=SimpleNamespace(Popen=popen))
        assert run(['unchanged', '\u00e9'], status) == 7
        assert status.read_bytes() == (b'{"schema": "prismaquant.profile_workload_status.v1", '
                                       b'"returncode": 7, "workload_pid": 42}')
        assert events == [['unchanged', '\u00e9'], 'wait']
        with pytest.raises(FileExistsError):
            run(['must-not-start'], status)
        assert len(events) == 2


@pytest.mark.parametrize('row_rc,observer_rc,timed_out', [(0, 0, False), (7, 3, False),
                                                       (0, 3, False), (7, None, True)])
def test_observer_outcomes_keep_error_order_and_cleanup(tmp_path, row_rc, observer_rc, timed_out):
    outcomes = []
    for original in (True, False):
        out = tmp_path / str(original)
        events = []
        observer = SimpleNamespace()
        observer.poll = lambda: None
        observer.terminate = lambda: events.append('terminate-observer')

        def observer_wait(timeout):
            if timeout == 30:
                events.append('join-observer')
                return 0
            events.append('wait-observer')
            assert (out / 'workload_done').read_text() == str(row_rc)
            if timed_out:
                raise subprocess.TimeoutExpired('observer', timeout)
            return observer_rc

        observer.wait = observer_wait

        def popen(command):
            events.append(tuple(command))
            if command == ['observer']:
                out.mkdir()
                (out / 'ready.json').write_text('{"target_out":"target"}')
                return observer
            return SimpleNamespace(wait=lambda: row_rc)

        run = consumer('tools/pq_admitted_profile.py', 'run_observed', original=original,
                       subprocess=SimpleNamespace(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired),
                       time=SimpleNamespace(monotonic=lambda: 0, sleep=lambda _: None))
        error = None
        try:
            rc = run(['workload'], ['observer'], out, target_out='target', observer_exit_timeout=5)
        except RuntimeError as failure:
            rc, error = None, str(failure)
        payload = (out / 'observer_result.json').read_bytes()
        assert payload.startswith(b'{"schema": "prismaquant.profile_observer_result.v1", "workload_returncode": ')
        assert not payload.endswith(b'\n')
        assert json.loads(payload)['qualified'] == (row_rc == 0 and observer_rc == 0 and not timed_out)
        assert events == [('observer',), ('workload',), 'wait-observer',
                          'terminate-observer', 'join-observer']
        if row_rc:
            assert rc == row_rc and error is None
        outcomes.append((rc, error, payload, events))
    assert outcomes[0] == outcomes[1]


def test_observer_event_bytes_and_partial_write_on_refusal(tmp_path):
    for original in (True, False):
        out = tmp_path / str(original)
        out.mkdir()
        event = consumer('tools/pq_row_profile_observer.py', 'event', original=original,
                         out=out, lock=threading.Lock(), time=SimpleNamespace(time=lambda: 12.5))
        event('ready', z='\u00e9', a=float('nan'))
        path = out / 'events.jsonl'
        assert path.read_bytes() == b'{"epoch": 12.5, "event": "ready", "z": "\\u00e9", "a": NaN}\n'
        with pytest.raises(TypeError):
            event('bad', value=object())
        # Opening for append precedes dumps, but no partial encoding is written.
        assert len(path.read_bytes().splitlines()) == 1


def test_status_writer_failure_keeps_original_partial_stream(tmp_path):
    """Inject a failure at serialization, not a production status-loader input."""
    for original in (True, False):
        profile = tmp_path / str(original)
        status = profile.with_name(profile.name + '.workload-status.json')
        status.write_bytes(b'original status')
        finish = consumer('tools/pq_profile_artifact.py', 'finish_profile_status', original=original)
        # The real loader yields only JSON values. This narrow seam forces the
        # writer to fail after its first field, to distinguish dump from dumps.
        finish.__globals__['read_workload_status'] = lambda _: {'first': '\u00e9', 'bad': object()}
        with pytest.raises(TypeError):
            finish(profile, 0)
        partial = status.with_name(status.name + '.finalizing')
        assert partial.read_bytes() == b'{"first": "\\u00e9", "bad": '
        assert status.read_bytes() == b'original status'
