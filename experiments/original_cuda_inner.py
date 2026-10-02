"""One explicitly admitted original-source fixture control, with counted outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dependencies',type=Path,required=True)
    p.add_argument('--dependencies-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--node-id')
    p.add_argument('--cpu-preflight',action='store_true')
    a=p.parse_args()
    if a.cpu_preflight == bool(a.node_id):
        raise RuntimeError('exactly CPU preflight or one GPU control is required')
    with a.dependencies.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=a.dependencies_sha256:
            raise RuntimeError('dependency artifact SHA256 differs')
    deps=a.out/'python-deps'
    deps.mkdir(exist_ok=False)
    with tarfile.open(a.dependencies) as archive:
        if sum(member.size for member in archive.getmembers())>32*1024**2:
            raise RuntimeError('dependency artifact exceeds finite extraction bound')
        if any(not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts
               for member in archive.getmembers()):
            raise RuntimeError('dependency artifact is not a closed regular-file roster')
        archive.extractall(deps,filter='data')
    sys.path.insert(0,str(deps))
    import importlib.metadata as metadata
    import pytest
    import torch
    for name,commit in (('prismabuild','95a59051d48cda82eea7927f31870c6c862d7174'),
                        ('tessera-quant','b40c93cb73745097e57a1ba4cf5b9eee166c759a')):
        dist=metadata.distribution(name)
        direct=json.loads(dist.read_text('direct_url.json') or '{}')
        if not Path(dist._path).resolve().is_relative_to(deps.resolve()) or direct.get('vcs_info',{}).get('commit_id')!=commit:
            raise RuntimeError('owned dependency resolution/provenance differs: '+name)
    if a.cpu_preflight:
        if torch.cuda.is_available() or os.environ.get('PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_TEST')=='1':
            raise RuntimeError('CPU prerequisite action unexpectedly has CUDA qualification')
        nodes=['tests/test_capture_original_material.py']
        expected=26
    else:
        if not a.node_id.startswith('tests/test_original_source_copy_completion_cuda.py::test_actual_original_copy_stream_ownership['):
            raise RuntimeError('unexpected GPU control node')
        if not torch.cuda.is_available() or os.environ.get('PRISMAQUANT_ORIGINAL_CUDA_QUALIFICATION_TEST')!='1':
            raise RuntimeError('actual CUDA control was not explicitly granted')
        properties=torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1.0,1024**3/properties.total_memory),0)
        nodes=[a.node_id]
        expected=1
    class Outcomes:
        def __init__(self):
            self.collected=0
            self.reports=[]
        def pytest_collection_finish(self,session):
            self.collected=len(session.items)
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.failed or report.skipped:
                self.reports.append(dict(node_id=report.nodeid,phase=report.when,outcome=report.outcome))
    outcomes=Outcomes()
    code=int(pytest.main(['-q','--tb=short','-o','addopts=','-p','no:cacheprovider',
                         '--basetemp',str(a.out/'pytest-tmp'),*nodes],plugins=[outcomes]))
    passed=sum(row['phase']=='call' and row['outcome']=='passed' for row in outcomes.reports)
    status=code==0 and outcomes.collected==expected and passed==expected and len(outcomes.reports)==expected
    record=dict(schema='prismaquant.original_cuda_fixture_execution.v1',passed=status,
                cpu_preflight=a.cpu_preflight,node_id=a.node_id,returncode=code,
                collected=outcomes.collected,expected=expected,reports=outcomes.reports,
                automatic_capture_qualified=False,actual_glm=False,
                torch=str(torch.__version__),cuda=torch.version.cuda,
                peak_cuda_allocated_bytes=0 if a.cpu_preflight else torch.cuda.max_memory_allocated(0),
                peak_cuda_reserved_bytes=0 if a.cpu_preflight else torch.cuda.max_memory_reserved(0))
    (a.out/'test-result.json').write_text(json.dumps(record,indent=2)+'\n')
    return 0 if status else (code or 1)


if __name__=='__main__':
    raise SystemExit(main())
