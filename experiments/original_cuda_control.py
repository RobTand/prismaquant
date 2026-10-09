"""Finite PB fixture action using the existing container adapter and raw Netdata."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import stat
import socket
import subprocess
import sys
import time
import urllib.parse

from tools.pq_profile_source import profile_source_owner

file_stat_signature=profile_source_owner('file_identity').file_stat_signature

PQ_IMAGE='d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026'
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

def artifact_identity(path):
    """Bind the exact regular artifact bytes, never a mutable path alone."""
    path=Path(path)
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size<=0:
            raise RuntimeError('published artifact is missing/empty/nonregular: '+str(path))
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if (file_stat_signature(os.fstat(stream.fileno()))!=file_stat_signature(before)
                or file_stat_signature(os.stat(path,follow_symlinks=False))!=file_stat_signature(before)):
            raise RuntimeError('published artifact changed while binding: '+str(path))
    return dict(path=str(path),bytes=before.st_size,sha256=digest)


def artifact_cas(root):
    """Use the actual launch-owned SDK5 generation and explicitly sealed CAS."""
    if root is None:
        raise RuntimeError('actual control publication requires an explicit sealed CAS root')
    from tools.tessera_campaign_container import reader_context_environment
    context,_=reader_context_environment({},os.environ)
    if not context:
        raise RuntimeError('actual control publication requires the launch-owned reader context')
    source=Path(context['PRISMABUILD_READER_HELPER_ROOT'])/'src'
    sys.path.insert(0,str(source))
    from prismabuild import client, core
    if client.SDK_VERSION!=5:
        raise RuntimeError('actual artifact publication requires the qualified SDK5 helper')
    if any(not getattr(module,'__file__',None)
           or not Path(module.__file__).resolve().is_relative_to(source)
           for name,module in sys.modules.items()
           if name=='prismabuild' or name.startswith('prismabuild.')):
        raise RuntimeError('artifact publication imported a mixed or foreign helper generation')
    return core.PrismaBuildCAS(root)


def retain_artifact(cas,role,path,identity=None):
    """Retain real bytes through PB's existing immutable input owner."""
    identity=artifact_identity(path) if identity is None else identity
    entry,_=cas.ingest_input(path,input_id='original-source-artifact.'+role,
        expected_sha256=identity['sha256'],expected_bytes=identity['bytes'])
    durable=cas.input_path(entry)
    return dict(path=str(durable),sha256=entry['sha256'],bytes=entry['bytes'])


def publish_control_artifacts(node_id,out,profile_evidence,cas):
    """Make the selected stdout CAS result bind the actual node's sidecars."""
    controls=list((out/'controls').glob('*.json'))
    if len(controls)!=1:
        raise RuntimeError('successful node requires exactly one actual control artifact')
    paths=dict(control=controls[0],execution=out/'test-result.json',
               action_result=out/'action-result.json',netdata_sparky=out/'netdata-sparky.json',
               netdata_sparklina=out/'netdata-sparklina.json')
    artifacts={role:retain_artifact(cas,role,path) for role,path in paths.items()}
    artifacts['torch_trace']=profile_evidence
    publication=dict(schema='prismaquant.original_source_artifact_publication.v1',
                     node_id=node_id,artifacts=artifacts)
    print('ORIGINAL_SOURCE_ARTIFACTS '+json.dumps(publication,sort_keys=True,
          separators=(',',':'),allow_nan=False),flush=True)



def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dependencies',type=Path,required=True)
    p.add_argument('--dependencies-sha256',required=True)
    p.add_argument('--node-id')
    p.add_argument('--cpu-preflight',action='store_true')
    p.add_argument('--cuda-entry-preflight',action='store_true')
    p.add_argument('--cas-root',type=Path)
    a=p.parse_args()
    if sum((a.cpu_preflight,a.cuda_entry_preflight,bool(a.node_id)))!=1:
        raise RuntimeError('exactly CPU preflight, CUDA entry proof or one GPU control is required')
    a.out.mkdir(parents=True,exist_ok=False)
    setup_start=time.time()
    charts={}
    try:
        cas=artifact_cas(a.cas_root) if a.node_id else None
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
                 TMPDIR='/qualification/tmp',PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_OUT='/qualification/controls')
        mounts=[dict(source=str(a.dependencies.parent),target='/dependencies',readonly=True),
                dict(source=str(a.out),target='/qualification',readonly=False)]
        profile=os.environ.get('PRISMABUILD_PROFILE_TORCH_OUT')
        if not a.cpu_preflight:
            if not profile or os.environ.get('CUDA_VISIBLE_DEVICES')=='':
                raise RuntimeError('actual GPU scope/profile grant missing')
            if not a.cuda_entry_preflight:
                env['PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_TEST']='1'
            env['PRISMABUILD_PROFILE_TORCH_OUT']='/profile/'+Path(profile).name
            mounts.append(dict(source=str(Path(profile).parent),target='/profile',readonly=False))
        (a.out/'tmp').mkdir()
        spec=dict(cpu_memory_gb=4,container=dict(image='prismaquant-glm-derivative:causal-exp-v1-20260908',
                  content_sha256=PQ_IMAGE,mounts=mounts),env=env)
        command=['python3','-P','/workspace/experiments/original_cuda_inner.py',
                 '--dependencies','/dependencies/'+a.dependencies.name,
                 '--dependencies-sha256',a.dependencies_sha256,'--out','/qualification']
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
        # The existing adapter execs Docker. Supervise its CLI as a child so
        # this owner can bind the trace, collect both raw host windows and
        # preserve the actual container ending; never shadow runtime /run.
        code=subprocess.run([sys.executable,'-m','tools.tessera_campaign_container',
            '--spec',json.dumps(spec),*(['--cpu-only'] if a.cpu_preflight else []),
            '--',*command],check=False).returncode
    finally:
        finish=time.time()
        observations=[]
        errors=[]
        profile_evidence=None
        profile_errors=[]
        if not a.cpu_preflight:
            try:
                profile_evidence=artifact_identity(profile)
                if code==0 and a.node_id:
                    profile_evidence=retain_artifact(cas,'torch_trace',profile,profile_evidence)
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
            publish_control_artifacts(a.node_id,a.out,profile_evidence,cas)
    return code


if __name__=='__main__':
    raise SystemExit(main())
