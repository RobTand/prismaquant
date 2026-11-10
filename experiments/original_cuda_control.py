"""Finite native original-source qualification in an explicitly admitted runtime."""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import urllib.parse

from experiments.original_cuda_artifacts import publish_control_artifacts

CONTEXTS={'system.cpu','system.ram','system.io','system.load','nvidia_smi.gpu_power_draw'}


def netdata(host,endpoint):
    url='http://127.0.0.1:19999/api/v1/'+endpoint
    request=['curl','-fsS','--max-time','8','--max-filesize','8388608',url]
    if host!=socket.gethostname():
        request=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',
                 '-o','ControlMaster=no','-o','ControlPath=none','-o','ControlPersist=no',host,
                 shlex.join(request)]
    # Same-host HTTP is the same real Netdata owner, not an SSH failure fallback.
    # A remote peer remains SSH-bound; either route still refuses any error.
    raw=subprocess.run(request,capture_output=True,text=True,timeout=15,check=True).stdout
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
    p.add_argument('--cuda-entry-preflight',action='store_true')
    p.add_argument('--trace', type=Path)
    p.add_argument('--max-artifact-bytes', type=int, default=64 << 20)
    a=p.parse_args()
    if sum((a.cpu_preflight,a.cuda_entry_preflight,bool(a.node_id)))!=1:
        raise RuntimeError('exactly CPU preflight, CUDA entry proof or one GPU control is required')
    a.out.mkdir(parents=True,exist_ok=False)
    setup_start=time.time()
    charts={}
    try:
        from prismaquant.native_results import execution_context
        context = execution_context()
        if not a.cpu_preflight and context['resources']['gpus'] < 1:
            raise RuntimeError('actual CUDA qualification requires native GPU admission')
        if not a.cpu_preflight:
            for host in ('sparky','sparklina'):
                selected={key:value for key,value in netdata(host,'charts')['charts'].items()
                          if value.get('context') in CONTEXTS}
                if {meta.get('context') for meta in selected.values()}!=CONTEXTS:
                    raise RuntimeError('required Netdata context missing: '+host)
                if any(type(meta.get('update_every')) is not int or meta['update_every']<=0 for meta in selected.values()):
                    raise RuntimeError('required Netdata cadence missing')
                charts[host]=selected
        env = dict(os.environ)
        env.update(PYTHONPATH=str(Path.cwd()) + ':' + str(Path.cwd() / 'tests'),
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
            TMPDIR=str(a.out / 'tmp'), PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_OUT=str(a.out / 'controls'))
        profile = str(a.trace) if a.trace else None
        if not a.cpu_preflight:
            if not profile:
                raise RuntimeError('actual GPU qualification needs an explicit Torch trace path')
            env['PRISMAQUANT_ORIGINAL_TORCH_TRACE'] = profile
            if not a.cuda_entry_preflight:
                env['PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_TEST'] = '1'
        (a.out/'tmp').mkdir()
        command=[sys.executable, '-m', 'experiments.original_cuda_inner',
                 '--dependencies',str(a.dependencies),
                 '--dependencies-sha256',a.dependencies_sha256,'--out',str(a.out)]
        if a.cpu_preflight:
            command+=['--cpu-preflight']
        elif a.cuda_entry_preflight:
            command+=['--cuda-entry-preflight','--expected-uid',str(os.getuid()),
                      '--expected-gid',str(os.getgid())]
        else:
            command+=['--node-id',a.node_id]
    except BaseException as exc:
        (a.out/'action-result.json').write_text(json.dumps(dict(
            returncode=None,start_unix=setup_start,finish_unix=time.time(),
            phase='preflight_refused',cpu_preflight=a.cpu_preflight,node_id=a.node_id,
            preflight_error=dict(type=type(exc).__name__,message=str(exc),
                                 stderr=(getattr(exc,'stderr','') or '')[-4096:]),
            container_invocation_started=False,actual_cuda_test_entered=False,
            automatic_capture_qualified=False,actual_glm=False),indent=2)+'\n')
        raise
    start=time.time()
    code=None
    try:
        code=subprocess.run(command, env=env, check=False).returncode
    finally:
        finish=time.time()
        observations=[]
        errors=[]
        profile_evidence=None
        profile_errors=[]
        if not a.cpu_preflight:
            try:
                from prismaquant.native_results import bind_result_artifact
                measured=bind_result_artifact(Path(profile), max_bytes=a.max_artifact_bytes)
                profile_evidence={key: measured[key] for key in ('path', 'sha256', 'bytes')}
            except BaseException as exc:
                profile_errors.append(dict(path=profile,error=f'{type(exc).__name__}: {exc}'))
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
            netdata=observations,netdata_errors=errors,torch_trace=profile_evidence,
            torch_trace_errors=profile_errors,automatic_capture_qualified=False,
            actual_glm=False),indent=2)+'\n')
        if errors or profile_errors:
            raise RuntimeError('required raw Netdata/Torch trace evidence incomplete: '+repr(errors+profile_errors))
        if code==0 and a.node_id:
            publish_control_artifacts(a.node_id, a.out, profile_evidence,
                inputs={'dependencies': {'path': str(a.dependencies), 'sha256': a.dependencies_sha256,
                                          'bytes': a.dependencies.stat().st_size}},
                max_artifact_bytes=a.max_artifact_bytes)
    return code


if __name__=='__main__':
    raise SystemExit(main())
