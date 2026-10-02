"""CPU lifecycle controls; native same-UID py-spy smoke is separately PB-admitted."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tools.pq_profile_child import same_uid_profile_command, run_child_profile, run_profile_workload
from tools.pq_profile_artifact import publish_profile


def test_same_uid_launch_preserves_original_argv_and_has_no_attach_cutoff(tmp_path):
    original = ['python3', '-u', '-m', 'prismaquant.tessera_campaign', '--out', 'original.pkl']
    command = same_uid_profile_command('/pinned/py-spy', original, tmp_path/'profile', tmp_path/'status')
    assert command[0] == '/pinned/py-spy'
    assert '--pid' not in command and '--duration' not in command and 'sudo' not in command
    assert command[-len(original):] == original
    assert 'tools.pq_profile_child' in command and '--execute' in command


def test_empty_operand_reaches_actual_workload_unchanged(tmp_path, monkeypatch):
    original = [sys.executable, '-c', 'import sys; assert sys.argv[1] == ""', '']
    destination = tmp_path/'profile'
    called = []
    real_call = subprocess.call

    def profile_call(command):
        assert command[-len(original):] == original
        called.append(command)
        status = Path(command[command.index('--status')+1])
        # Execute the real Python operand control, not a fabricated success.
        rc = real_call(original)
        status.write_text(json.dumps({'schema':'prismaquant.profile_workload_status.v1',
                                      'returncode':rc,'workload_pid':42}))
        return 0

    monkeypatch.setattr('tools.pq_profile_child.subprocess.call', profile_call)
    assert run_child_profile(original, executable='py-spy', destination=destination) == 0
    assert len(called) == 1


@pytest.mark.parametrize('original', [[], [''], [None], ['python3', None]])
def test_profile_command_refuses_missing_executable_or_nonstring_operand(tmp_path, original):
    with pytest.raises(ValueError, match='workload'):
        same_uid_profile_command('py-spy', original, tmp_path/'profile', tmp_path/'status')


@pytest.mark.parametrize('row_rc', [0,9])
@pytest.mark.parametrize('profiler_rc', [0,7])
def test_profiler_result_does_not_mask_workload_exit(tmp_path, monkeypatch, row_rc, profiler_rc):
    destination = tmp_path/'profile'
    def call(command):
        status = Path(command[command.index('--status')+1])
        status.write_text(json.dumps({'schema':'prismaquant.profile_workload_status.v1',
                                      'returncode':row_rc,'workload_pid':42}))
        return profiler_rc
    monkeypatch.setattr('tools.pq_profile_child.subprocess.call', call)
    if row_rc == 0 and profiler_rc:
        with pytest.raises(RuntimeError, match='profiler failed'):
            run_child_profile(['python3','row.py'], executable='py-spy', destination=destination)
    else:
        assert run_child_profile(['python3','row.py'], executable='py-spy', destination=destination) == row_rc
    record = json.loads(destination.with_name(destination.name+'.workload-status.json').read_text())
    assert record['returncode'] == row_rc and record['profiler_returncode'] == profiler_rc


def test_profiler_failure_and_missing_completion_refuse(tmp_path, monkeypatch):
    monkeypatch.setattr('tools.pq_profile_child.subprocess.call',lambda command:7)
    with pytest.raises(RuntimeError,match='profiler failed'):
        run_child_profile(['python3','row.py'],executable='py-spy',destination=tmp_path/'bad')
    monkeypatch.setattr('tools.pq_profile_child.subprocess.call',lambda command:0)
    with pytest.raises(FileNotFoundError):
        run_child_profile(['python3','row.py'],executable='py-spy',destination=tmp_path/'missing')


def test_status_worker_preserves_nonzero_and_rejects_stale(tmp_path,monkeypatch):
    status=tmp_path/'status'
    class Workload:
        pid = 42
        def wait(self):
            return 9
    monkeypatch.setattr('tools.pq_profile_child.subprocess.Popen',lambda command:Workload())
    assert run_profile_workload(['python3','row.py'],status)==9
    record = json.loads(status.read_text())
    assert record['returncode']==9 and record['workload_pid']==42
    with pytest.raises(FileExistsError):
        run_profile_workload(['python3','row.py'],status)


def test_publish_requires_samples_and_preserves_exact_bytes(tmp_path):
    source=tmp_path/'source'
    payload={'$schema':'https://www.speedscope.app/file-format-schema.json',
             'shared':{'frames':[{'name':'actual_frame'}]},
             'profiles':[{'type':'sampled','name':'Process 42 Thread 42 "work"',
                          'unit':'seconds','startValue':0,'endValue':1,
                          'samples':[[0]],'weights':[1]}]}
    source.write_text(json.dumps(payload))
    source.with_name(source.name+'.workload-status.json').write_text(json.dumps({
        'schema':'prismaquant.profile_workload_status.v1','returncode':0,'workload_pid':42,
        'profiler_returncode':0}))
    target=tmp_path/'published'
    assert publish_profile(source,target)['bytes']==len(source.read_bytes())
    assert source.read_bytes()==target.read_bytes()
    with pytest.raises(FileExistsError): publish_profile(source,target)
    payload['profiles'][0]['samples']=[]
    payload['profiles'][0]['weights']=[]
    payload['profiles'][0]['endValue']=0
    source.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError,match='sampled profiles'): publish_profile(source,tmp_path/'empty')
