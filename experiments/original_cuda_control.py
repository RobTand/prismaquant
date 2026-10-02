"""Finite PB fixture action using the existing container adapter and raw Netdata."""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.parse

from tools.tessera_campaign_container import main as container_main

PQ_IMAGE='d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026'
CONTEXTS={'system.cpu','system.ram','system.io','system.load','nvidia_smi.gpu_power_draw'}


def netdata(host,endpoint):
    url='http://127.0.0.1:19999/api/v1/'+endpoint
    raw=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,
                        'curl -fsS --max-time 8 --max-filesize 8388608 '+shlex.quote(url)],capture_output=True,text=True,
                       timeout=15,check=True).stdout
    if len(raw.encode())>8*1024**2:
        raise RuntimeError('Netdata response exceeds bound')
    return json.loads(raw)


def collect(host,charts,after,before,out):
    series={}
    for chart,meta in charts.items():
        cadence=meta['update_every']
        chart_after=(after//cadence)*cadence
        chart_before=(before//cadence)*cadence
        query=urllib.parse.urlencode(dict(chart=chart,after=chart_after,before=chart_before,
                    points=0,group='average',format='json',options='seconds|jsonwrap'))
        data=netdata(host,'data?'+query)
        geometry=(data.get('after'),data.get('before'),data.get('view_update_every'))
        if (geometry[2]!=cadence or any(type(value) is not int for value in geometry)
                or not chart_after-cadence<=geometry[0]<=chart_after+cadence
                or not chart_before-2*cadence<=geometry[1]<=chart_before
                or geometry[0]>=geometry[1]):
            raise RuntimeError('Netdata returned time/cadence geometry differs')
        result=data.get('result',{})
        labels,rows=result.get('labels'),result.get('data')
        if not isinstance(labels,list) or len(labels)<2 or labels[0]!='time' or not rows:
            raise RuntimeError('Netdata labels/samples missing: '+host+':'+chart)
        measured=set()
        for row in rows:
            if len(row)!=len(labels) or not geometry[0]<=row[0]<=geometry[1]:
                (out/('netdata-refused-'+host+'.json')).write_text(json.dumps(dict(
                    chart=chart,after=chart_after,before=chart_before,metadata=meta,response=data),indent=2)+'\n')
                raise RuntimeError('Netdata malformed/out-of-window sample: '+repr(dict(
                    host=host,chart=chart,after=chart_after,before=chart_before,row=row,labels=labels)))
            for index,value in enumerate(row[1:],1):
                if value is not None:
                    if type(value) not in (int,float) or not math.isfinite(value):
                        raise RuntimeError('Netdata nonfinite/nonnumeric sample')
                    measured.add(index)
        if measured!=set(range(1,len(labels))):
            raise RuntimeError('Netdata unmeasured dimension')
        series[chart]=dict(metadata=meta,query_after=chart_after,query_before=chart_before,response=data)
    record=dict(host=host,after=after,before=before,charts=series,
                scope='raw native-cadence host evidence; no performance claim')
    (out/('netdata-'+host+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    return dict(host=host,charts=len(series))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dependencies',type=Path,required=True)
    p.add_argument('--dependencies-sha256',required=True)
    p.add_argument('--node-id')
    p.add_argument('--cpu-preflight',action='store_true')
    a=p.parse_args()
    if a.cpu_preflight == bool(a.node_id):
        raise RuntimeError('exactly CPU preflight or one GPU control is required')
    a.out.mkdir(parents=True,exist_ok=False)
    charts={}
    if not a.cpu_preflight:
        for host in ('sparky','sparklina'):
            selected={key:value for key,value in netdata(host,'charts')['charts'].items()
                      if value.get('context') in CONTEXTS}
            if {meta.get('context') for meta in selected.values()}!=CONTEXTS:
                raise RuntimeError('required Netdata context missing: '+host)
            if any(type(meta.get('update_every')) is not int or meta['update_every']<=0 for meta in selected.values()):
                raise RuntimeError('required Netdata cadence missing')
            charts[host]=selected
    env=dict(PYTHONPATH='/workspace:/workspace/tests',OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
             TMPDIR='/run/tmp',PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_OUT='/run/controls')
    mounts=[dict(source=str(a.dependencies.parent),target='/dependencies',readonly=True),
            dict(source=str(a.out),target='/run',readonly=False)]
    profile=os.environ.get('PRISMABUILD_PROFILE_TORCH_OUT')
    if not a.cpu_preflight:
        if not profile or os.environ.get('CUDA_VISIBLE_DEVICES')=='':
            raise RuntimeError('actual GPU scope/profile grant missing')
        env['PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_TEST']='1'
        env['PRISMABUILD_PROFILE_TORCH_OUT']='/profile/'+Path(profile).name
        mounts.append(dict(source=str(Path(profile).parent),target='/profile',readonly=False))
    (a.out/'tmp').mkdir()
    spec=dict(cpu_memory_gb=4,container=dict(image='prismaquant-glm-derivative:causal-exp-v1-20260908',
              content_sha256=PQ_IMAGE,mounts=mounts),env=env)
    command=['python3','-P','/workspace/experiments/original_cuda_inner.py',
             '--dependencies','/dependencies/'+a.dependencies.name,
             '--dependencies-sha256',a.dependencies_sha256,'--out','/run']
    command+=['--cpu-preflight'] if a.cpu_preflight else ['--node-id',a.node_id]
    start=time.time()
    code=None
    try:
        code=container_main(['--spec',json.dumps(spec),*(['--cpu-only'] if a.cpu_preflight else []),'--',*command])
    finally:
        finish=time.time()
        observations=[]
        errors=[]
        if charts:
            padding=2*max(meta['update_every'] for host in charts.values() for meta in host.values())
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                jobs={pool.submit(collect,host,selected,int(start)-padding,int(finish),a.out):host
                      for host,selected in charts.items()}
                for job,host in jobs.items():
                    try:
                        observations.append(job.result())
                    except BaseException as exc:
                        errors.append(dict(host=host,error=str(exc)))
        (a.out/'action-result.json').write_text(json.dumps(dict(returncode=code,
            start_unix=start,finish_unix=finish,cpu_preflight=a.cpu_preflight,
            netdata=observations,netdata_errors=errors,automatic_capture_qualified=False,
            actual_glm=False),indent=2)+'\n')
        if errors:
            raise RuntimeError('required raw Netdata evidence incomplete: '+repr(errors))
    return code


if __name__=='__main__':
    raise SystemExit(main())
