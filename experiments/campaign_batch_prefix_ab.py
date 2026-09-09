"""Interleave native batch widths under one PB measurement admission."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.selected_snapshot_scope_ab import digest, write


def prefix_state(root, observed):
    from prismaquant.cost_stage_checkpoint import _load_unit, unit_path
    manifest=json.loads((root/'cost.anchors.json').read_text())
    expected={(name,call['format_name']) for call in observed['prefix_calls'] for name in call['qnames']}
    states={}
    for name in sorted({name for name,fmt in expected}):
        state=_load_unit(unit_path(root/'cost.anchors.json.parts',name),stage=manifest['stage'],
            qname=name,identity_sha256=manifest['identity_sha256'])
        if state.get('unservable'):
            raise ValueError('prefix contains refused anchors')
        for anchor in state['anchors']:
            fmt=anchor['format_name'];identity=(name,fmt)
            if identity not in expected or identity in states:
                raise ValueError('prefix journal differs from observed work')
            wire=state['wire_records'][fmt];path=root/'cache/wire'/wire['file']
            if digest(path)!=wire['blob_sha256'] or path.stat().st_size!=wire['blob_bytes']:
                raise ValueError('prefix wire differs from journal')
            states[identity]={'anchor':{k:v for k,v in anchor.items() if k not in ('seconds','encoding_batch_size')},'wire':wire}
    if set(states)!=expected:
        raise ValueError('prefix journal missed measured anchors')
    return manifest['identity_sha256'],states


def run(path,expected):
    if digest(path)!=expected:
        raise ValueError('batch measurement plan changed')
    p=json.loads(Path(path).read_text())
    if p['schema']!='prismaquant.campaign_batch_prefix_ab.v1':
        raise ValueError('unknown batch plan')
    for name,sha in p['input_sha256'].items():
        if digest(name)!=sha:raise ValueError('batch input changed: '+name)
    for name,value in p['environment'].items():
        if os.environ.get(name)!=value:raise ValueError('batch environment changed: '+name)
    root=Path(p['out']);root.mkdir(parents=True,exist_ok=False)
    rows=[];baseline=None
    for index,width in enumerate(p['order']):
        out=root/f'arm-{index:02d}';out.mkdir()
        command=list(p['command'])
        for flag,value in {'--out':out/'cost.pkl','--cache-dir':out/'cache',
                '--checkpoint':out/'cost.anchors.json','--anchor-batch-size':width}.items():
            command[command.index(flag)+1]=str(value)
        argv=[sys.executable,'-u','-m','experiments.campaign_prefix_profile',
            '--evidence-out',str(out/'profile'),'--limit-anchors',str(p['limit_anchors']),
            '--expected-source-units',str(p['expected_source_units']),
            '--profile-calls','0,1','--trace-max-bytes',str(128*1024**2),
            '--profile-seconds','2','--',*command]
        env=dict(os.environ,TRITON_CACHE_DIR=str(out/'triton'),TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
        started=time.time()
        with (out/'command.log').open('x') as log:
            result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
        row=dict(index=index,batch_width=width,started_unix=started,finished_unix=time.time(),returncode=result.returncode,command=argv)
        rows.append(row);write(out/'exit.json',row)
        if result.returncode:raise RuntimeError(f'batch prefix arm {index} failed')
        files=list((out/'profile').glob('attempt-*/result.json'))
        if len(files)!=1:raise ValueError('prefix observer result absent or ambiguous')
        observed=json.loads(files[0].read_text())
        if observed['completed_anchor_units']!=p['limit_anchors'] or not observed['prefix_boundary_reached']:
            raise ValueError('prefix did not complete')
        current=prefix_state(out,observed)
        if baseline is None:baseline=current
        elif current!=baseline:raise ValueError('batch width changed wire bytes, scoring, or identity')
        row.update(exact_parity=True,anchor_units=p['limit_anchors'],observer_result=str(files[0]))
        write(out/'parity.json',row);print(json.dumps(row),flush=True)
    write(root/'result.json',dict(schema=p['schema'],plan_sha256=expected,rows=rows,exact_parity=True,
        campaign_completed=False,scope='64 anchor units; complete 864-unit resident source and original capture; fresh process and compiler caches per arm'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();run(args.plan,args.plan_sha256)

if __name__=='__main__':main()
