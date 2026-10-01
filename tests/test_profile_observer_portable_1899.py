"""Extract only the pure endpoint dispatcher; no host probes are executed."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace


def test_actual_observer_reads_its_own_host_locally():
    source=Path(__file__).resolve().parents[1]/'tools/pq_row_profile_observer.py'
    tree=ast.parse(source.read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='netdata')
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def read(self): return b'{"from":"local"}'
    calls=[]
    space={'local_host':'sparky','json':json,'shlex':__import__('shlex'),
       'urllib':SimpleNamespace(request=SimpleNamespace(urlopen=lambda *a,**kw:Response())),
       'command':lambda args,**kw:calls.append(args) or '{"from":"remote"}'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),space)
    assert space['netdata']('sparky','charts')=={'from':'local'}
    assert calls==[], 'observer mislabels local power/Netdata on portable sparky execution'
    assert space['netdata']('sparklina','charts')=={'from':'remote'}
    assert calls[0][0]=='ssh' and 'sparklina' in calls[0]
