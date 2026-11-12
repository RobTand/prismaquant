"""Record evaluated assertion operands without changing their returned values.

Run `python assertion_probe.py instrument TEST.py` in a disposable remote
checkout, then pytest with `-p assertion_probe` and PQ_ASSERTIONS_OUT set.
Only comparisons, boolean leaves, and bare assertion values are observed.
The recorder never consumes iterators or changes short-circuit evaluation.
"""
from __future__ import annotations
import ast
from collections import defaultdict
from collections.abc import Mapping, ValuesView
import dataclasses
from enum import Enum
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

ROWS = defaultdict(lambda: {'count': 0, 'sha256': hashlib.sha256()})
NODE = 'collection'
UNSUPPORTED = set()
OUTCOMES = {}


def canonical(value):
    if value is None or type(value) in (bool,int,str):
        return value
    if isinstance(value,float):
        return ['float',value.hex()]
    if isinstance(value,Fraction):
        return ['fraction',value.numerator,value.denominator]
    if isinstance(value,Enum):
        return ['enum',type(value).__module__,type(value).__qualname__,value.name]
    if isinstance(value,bytes):
        return ['bytes',value.hex()]
    if isinstance(value,type):
        return ['type',value.__module__,value.__qualname__]
    if isinstance(value,Path):
        return ['path',str(value)]
    if dataclasses.is_dataclass(value) and not isinstance(value,type):
        return ['dataclass',type(value).__qualname__,{f.name:canonical(getattr(value,f.name)) for f in dataclasses.fields(value)}]
    if isinstance(value,Mapping):
        return ['dict',sorted([[canonical(k),canonical(v)] for k,v in value.items()],key=lambda p:json.dumps(p[0],sort_keys=True))]
    if isinstance(value,ValuesView):
        return ['dict_values',[canonical(v) for v in value]]
    if isinstance(value,(set,frozenset)):
        return ['set',sorted([canonical(v) for v in value],key=lambda v:json.dumps(v,sort_keys=True))]
    if isinstance(value,(tuple,list)):
        return [type(value).__name__,[canonical(v) for v in value]]
    kind=type(value).__module__+'.'+type(value).__qualname__
    UNSUPPORTED.add(kind)
    return ['unsupported',kind]


def observe(value, expression, operand):
    raw=json.dumps(canonical(value),sort_keys=True,separators=(',',':')).encode()
    row=ROWS[(NODE,expression,operand)]
    row['count']+=1
    row['sha256'].update(len(raw).to_bytes(8,'big')+raw)
    return value


class Instrument(ast.NodeTransformer):
    def visit_Assert(self,node):
        expression=ast.dump(node.test,include_attributes=False)
        index=0
        def wrap(value):
            nonlocal index
            position=index;index+=1
            return ast.copy_location(ast.Call(func=ast.Name(id='_pq_observe',ctx=ast.Load()),args=[value,ast.Constant(expression),ast.Constant(position)],keywords=[]),value)
        def walk(value):
            if isinstance(value,ast.Compare):
                value.left=wrap(value.left)
                value.comparators=[wrap(v) for v in value.comparators]
                return value
            if isinstance(value,ast.BoolOp):
                value.values=[walk(v) for v in value.values]
                return value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id in ('all','any'):
                for argument in value.args:
                    if isinstance(argument,ast.GeneratorExp):
                        argument.elt=walk(argument.elt)
                return wrap(value)
            if isinstance(value,ast.UnaryOp) and isinstance(value.op,ast.Not):
                value.operand=walk(value.operand)
                return value
            return wrap(value)
        node.test=walk(node.test)
        return node


def instrument(path):
    path=Path(path)
    tree=ast.parse(path.read_text())
    assertions=[ast.dump(n.test,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Assert)]
    tree=Instrument().visit(tree)
    insert=0
    while insert<len(tree.body) and (isinstance(tree.body[insert],ast.Expr) and isinstance(tree.body[insert].value,ast.Constant) or isinstance(tree.body[insert],ast.ImportFrom) and tree.body[insert].module=='__future__'):
        insert+=1
    tree.body.insert(insert,ast.ImportFrom(module='assertion_probe',names=[ast.alias(name='observe',asname='_pq_observe')],level=0))
    ast.fix_missing_locations(tree)
    path.write_text(ast.unparse(tree)+'\n')
    Path(os.environ['PQ_ASSERTIONS_OUT']+'.source.json').write_text(json.dumps(assertions,indent=2))


def pytest_runtest_setup(item):
    global NODE
    NODE=item.nodeid


def pytest_runtest_logreport(report):
    OUTCOMES[report.nodeid+'::'+report.when]=report.outcome


def pytest_sessionfinish(session,exitstatus):
    output={'exitstatus':int(exitstatus),'nodes':[i.nodeid for i in session.items], 'unsupported':sorted(UNSUPPORTED), 'outcomes':OUTCOMES, 'assertions':[dict(node=n,expression=e,operand=o,count=r['count'],sha256=r['sha256'].hexdigest()) for (n,e,o),r in sorted(ROWS.items())]}
    Path(os.environ['PQ_ASSERTIONS_OUT']).write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    if sys.argv[1]=='instrument':
        instrument(sys.argv[2])
    elif sys.argv[1]=='compare':
        before,after=(json.loads(Path(p).read_text()) for p in sys.argv[2:])
        assert before['exitstatus']==after['exitstatus']==0
        assert before['unsupported']==after['unsupported']==[]
        assert set(before['nodes'])<=set(after['nodes'])
        assert all(after['outcomes'].get(n)==v for n,v in before['outcomes'].items())
        index=lambda d:{(r['node'],r['expression'],r['operand']):r for r in d['assertions']}
        b,a=index(before),index(after)
        changes=[key for key,row in b.items() if a.get(key)!=row]
        assert not changes, changes[:10]
        print(json.dumps({'original_nodes':len(before['nodes']),'assertion_operands':len(b),'evaluations':sum(r['count'] for r in b.values()),'unchanged':True},indent=2))
