"""Pack verified existing pure-Python installs for the admitted fixture action."""
import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import tarfile

from tools.resolve_prismabuild_dev_pin import PIN_NAME, PIN_SOURCE, resolve_literal_pin
from tools.resolve_tessera_dev_pin import resolve_tessera_dev_pin

PACKAGES = ('prismabuild', 'tessera-quant', 'pytest', 'pytest-timeout',
            'pytest-xdist', 'execnet', 'pluggy', 'packaging', 'iniconfig', 'pygments')
PINS = {'prismabuild': resolve_literal_pin(PIN_SOURCE, PIN_NAME),
        'tessera-quant': resolve_tessera_dev_pin()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    files={}
    for name in PACKAGES:
        dist=metadata.distribution(name)
        direct=json.loads(dist.read_text('direct_url.json') or '{}')
        if name in PINS:
            if direct.get('vcs_info',{}).get('commit_id')!=PINS[name] or direct.get('dir_info',{}).get('editable'):
                raise RuntimeError('existing dependency provenance differs: '+name)
        verified=0
        for member in dist.files or ():
            if '..' in member.parts or member.suffix=='.pyc':
                continue
            path=Path(dist.locate_file(member))
            if path.suffix in ('.so','.dll','.dylib'):
                raise RuntimeError('native dependency cannot be transported as pure Python: '+str(member))
            if not path.is_file():
                raise RuntimeError('installed dependency file missing: '+str(member))
            if member.hash:
                if member.hash.mode!='sha256':
                    raise RuntimeError('unsupported installed RECORD hash')
                expected=base64.urlsafe_b64decode(member.hash.value+'='*(-len(member.hash.value)%4))
                if hashlib.sha256(path.read_bytes()).digest()!=expected:
                    raise RuntimeError('installed RECORD differs: '+str(member))
                verified+=1
            if str(member) in files and files[str(member)]!=path:
                raise RuntimeError('dependency namespace overlap')
            files[str(member)]=path
        rows.append(dict(distribution=name,version=dist.version,direct_url=direct,
                         verified_record_files=verified,source_install=str(dist._path)))
    archive=a.out/'pure-python-installs.tar'
    with tarfile.open(archive,'w') as tar:
        for name,path in sorted(files.items()):
            tar.add(path,arcname=name,recursive=False)
    with archive.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    value=dict(schema='prismaquant.original_cuda_test_dependencies.v1',
               meaning='unchanged existing installed artifacts; not new install/provenance',
               archive=dict(path=str(archive),size=archive.stat().st_size,sha256=digest),
               distributions=rows,files=len(files))
    (a.out/'receipt.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),bytes=archive.stat().st_size,sha256=digest)))

if __name__=='__main__':
    main()
