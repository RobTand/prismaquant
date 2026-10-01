"""CPU-only process-ownership tests; no profiler/Netdata/CUDA/Docker executes."""
import json
import pytest
import tools.pq_admitted_profile as module


class Process:
    def __init__(self, code=0):
        self.code=code; self.finished=False; self.terminated=False
    def poll(self): return self.code if self.finished else None
    def wait(self, timeout=None): self.finished=True; return self.code
    def terminate(self): self.terminated=True; self.finished=True


@pytest.mark.parametrize('observer_code', [0, 7])
def test_admitted_lifetime_joins_observer_after_row(tmp_path, monkeypatch, observer_code):
    events=[]; observer=Process(observer_code); row=Process(); observations=tmp_path/'obs'
    def spawn(command, **kw):
        events.append(command[0])
        if command[0]=='observer':
            observations.mkdir();(observations/'ready.json').write_text(json.dumps({'target_out':'out'}))
            return observer
        return row
    monkeypatch.setattr(module.subprocess,'Popen',spawn)
    if observer_code:
        with pytest.raises(RuntimeError,match='observer'):
            module.run_observed(['row'],['observer'],observations,target_out='out',ready_timeout=1)
    else:
        assert module.run_observed(['row'],['observer'],observations,target_out='out',ready_timeout=1)==0
    assert events==['observer','row'] and row.finished and observer.finished
    assert (observations/'workload_done').exists() and not row.terminated


def test_observer_preflight_failure_never_starts_workload(tmp_path, monkeypatch):
    observer=Process(3);observer.finished=True;calls=[]
    monkeypatch.setattr(module.subprocess,'Popen',lambda command,**kw:calls.append(command[0]) or observer)
    with pytest.raises(RuntimeError,match='observer'):
        module.run_observed(['row'],['observer'],tmp_path/'obs',target_out='out',ready_timeout=1)
    assert calls==['observer']


def test_wrong_target_ready_refuses(tmp_path, monkeypatch):
    calls=[];observer=Process();observations=tmp_path/'obs'
    def spawn(cmd, **kw):
        calls.append(cmd[0]);observations.mkdir()
        (observations/'ready.json').write_text(json.dumps({'target_out':'wrong'}))
        return observer
    monkeypatch.setattr(module.subprocess,'Popen',spawn)
    with pytest.raises(RuntimeError,match='target'):
        module.run_observed(['row'],['observer'],observations,target_out='out',ready_timeout=1)
    assert calls==['observer'] and observer.terminated


def test_stale_output_refuses_before_any_process(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr(module.subprocess,'Popen',lambda cmd,**kw:calls.append(cmd))
    with pytest.raises(FileExistsError):
        module.run_observed(['row'],['observer'],tmp_path,target_out='out')
    assert calls==[]


def test_row_failure_is_not_replaced_by_observer_success(tmp_path, monkeypatch):
    observations=tmp_path/'obs';observer=Process(0);row=Process(9)
    def spawn(cmd, **kw):
        if cmd[0]=='observer':
            observations.mkdir();(observations/'ready.json').write_text(json.dumps({'target_out':'out'}))
            return observer
        return row
    monkeypatch.setattr(module.subprocess,'Popen',spawn)
    assert module.run_observed(['row'],['observer'],observations,target_out='out',ready_timeout=1)==9
    assert observer.finished and row.finished
