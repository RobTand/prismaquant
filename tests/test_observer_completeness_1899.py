"""CPU source/closure tests; imported observer CLI and host probes never run."""
import ast
from pathlib import Path
import time
import math
from types import SimpleNamespace

import pytest

from tools.pq_profile_artifact import validate_netdata_window

SOURCE=Path(__file__).resolve().parents[1]/'tools/pq_row_profile_observer.py'


def test_readiness_follows_target_check_and_collection_start():
    tree=ast.parse(SOURCE.read_text())
    starts=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and isinstance(n.func.value,ast.Name) and n.func.value.id=='nd' and n.func.attr=='start']
    target_checks=[n.lineno for n in tree.body if isinstance(n,ast.If) and 'find_target' in ast.unparse(n.test)]
    ready=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and n.func.attr in ('write_text','replace') and 'ready.json' in ast.unparse(n)]
    assert starts and target_checks and ready
    assert min(ready)>max(starts+target_checks), 'readiness precedes observation initialization'


def test_missing_netdata_is_not_silently_qualified(tmp_path):
    tree=ast.parse(SOURCE.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in ('collect_netdata','finish_netdata')]
    errors=[];events=[];calls=[]
    def fail(*args):
        calls.append(args)
        raise OSError('required series unavailable')
    joined=[]
    space={'time':time,'previous_netdata':int(time.time())-5,
           'charts':{'sparky':['power'],'sparklina':['power']},
           'nd':SimpleNamespace(stop=lambda timeout:joined.append(timeout), is_alive=lambda:False),
           'netdata':fail,'event':lambda *a,**kw:events.append((a,kw)),'out':tmp_path,
           'window_padding':2,
           'telemetry_errors':errors,'collection_ready':SimpleNamespace(set=lambda:None),
           'json':__import__('json'),'urllib':__import__('urllib.parse')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(SOURCE),'exec'),space)
    space['collect_netdata']()
    space['finish_netdata']()
    assert joined==[90]
    assert len(calls)==4, 'fixture must reach both hosts in initial and final collection'
    assert errors, 'collector swallowed required series failures and could return success'


def test_readiness_publication_is_atomic():
    tree=ast.parse(SOURCE.read_text())
    direct=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr=='write_text' and 'ready.json' in ast.unparse(n.func.value)]
    assert not direct, 'partial ready JSON is visible to workload launcher'


@pytest.mark.parametrize('damage', ['null_only','stale','malformed','nonfinite','missing_labels'])
def test_successful_but_unmeasured_netdata_refuses(tmp_path, damage):
    tree=ast.parse(SOURCE.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name == 'collect_netdata']
    payload={'labels':['time','power'], 'data':[[99,42.0]]}
    if damage=='null_only': payload['data']=[[99,None]]
    elif damage=='stale': payload['data']=[[0,42.0]]
    elif damage=='malformed': payload['data']=[[99]]
    elif damage=='nonfinite': payload['data']=[[99,float('nan')]]
    else: payload.pop('labels')
    errors=[]
    space={'time':SimpleNamespace(time=lambda:100),'previous_netdata':95,
           'validate_netdata_window':validate_netdata_window,
           'charts':{'sparky':['power'],'sparklina':['power']},
           'netdata':lambda *args:payload,
           'event':lambda *a,**kw:None,'out':tmp_path,'telemetry_errors':errors,
           'collection_ready':SimpleNamespace(set=lambda:None),'math':math,'window_padding':2,
           'json':__import__('json'),'urllib':__import__('urllib.parse')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(SOURCE),'exec'),space)
    space['collect_netdata']()
    assert errors, 'unmeasured successful Netdata response was accepted'


@pytest.mark.parametrize('partial_null', [False,True])
@pytest.mark.parametrize('sample_time,padding', [(99,2),(89,20)])
def test_fresh_finite_netdata_is_a_real_positive(tmp_path, partial_null, sample_time, padding):
    tree=ast.parse(SOURCE.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name == 'collect_netdata']
    payload={'labels':['time','power'], 'data':[[sample_time,42.0]]}
    if partial_null:
        payload['data'].append([99.5,None])
    errors=[]
    space={'time':SimpleNamespace(time=lambda:100),'previous_netdata':95,
           'validate_netdata_window':validate_netdata_window,
           'charts':{'sparky':['power'],'sparklina':['power']},
           'netdata':lambda *args:payload,
           'event':lambda *a,**kw:None,'out':tmp_path,'telemetry_errors':errors,
           'collection_ready':SimpleNamespace(set=lambda:None),'math':math,'window_padding':padding,
           'json':__import__('json'),'urllib':__import__('urllib.parse')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(SOURCE),'exec'),space)
    space['collect_netdata']()
    assert not errors, errors
    assert (tmp_path/'netdata-sparky.jsonl').read_text()
    assert (tmp_path/'netdata-sparklina.jsonl').read_text()


@pytest.mark.parametrize('seen,alive,done,expected', [
    (False,False,False,False), (True,True,False,False),
    (True,False,False,False), (True,False,True,True)])
def test_observer_waits_for_workload_completion(seen, alive, done, expected):
    tree=ast.parse(SOURCE.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='profile_target_finished')
    space={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),space)
    assert space['profile_target_finished'](seen,alive,done) is expected


@pytest.mark.parametrize('seen,alive', [(False,False),(True,True)])
def test_observer_refuses_unobserved_or_live_completed_workload(seen, alive):
    tree=ast.parse(SOURCE.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='profile_target_finished')
    space={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),space)
    with pytest.raises(RuntimeError,match='completion does not match'):
        space['profile_target_finished'](seen,alive,True)
