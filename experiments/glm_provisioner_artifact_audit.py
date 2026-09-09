"""Independently compare the actual provisioner acceptance artifacts to Git."""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import tarfile

root=Path('/home/rob/tmp/pq455-accept2/pins')
commit='07ad344c3275bb2fa7ce2432f93d89945d66f4c2'
archive=subprocess.check_output(['git','-C','/mnt/shared/tessera-source.git','archive',commit])
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    expected={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
def files(path):
    return {p.relative_to(path).as_posix():p.read_bytes() for p in path.rglob('*')
            if p.is_file() and not set(p.relative_to(path).parts)&{'.git','__pycache__'}
            and p.suffix not in {'.pyc','.pyo'} and p.name!='.pinned-source.json'}
def digest(values):
    h=hashlib.sha256()
    for name in sorted(values,key=Path):
        raw=values[name]
        h.update(name.encode()+b'\0'+hashlib.sha256(raw).hexdigest().encode()+b'\n')
    return h.hexdigest()
actual=files(root/commit)
assert actual==expected
manifest=json.loads((root/commit/'.pinned-source.json').read_text())
assert manifest['commit']==commit and manifest['tree_sha256']==digest(actual) and manifest['files']==len(actual)==1179
quarantine=list(root.glob('.quarantine-'+commit+'-*'));assert len(quarantine)==1
old=files(quarantine[0]);assert len(old)==1256 and digest(old)=='adb6511eb926f7342f75aa067616f5c8665b61e32dd6dde5347446fda6a2b0a7'
assert all(old[name]==raw for name,raw in expected.items())
extras=sorted(set(old)-set(expected));assert len(extras)==77
assert all(name.startswith(('build/','src/tessera_quant.egg-info/')) for name in extras)
assert not Path('/home/rob/tmp/pq455-accept2/venv').exists()
print(json.dumps(dict(status='PASS',commit=commit,archive_sha256=hashlib.sha256(archive).hexdigest(),
    current_files=len(actual),current_digest=digest(actual),quarantine=str(quarantine[0]),
    quarantine_files=len(old),quarantine_digest=digest(old),extras=extras,
    current_exact_to_git_archive=True,old_git_files_unchanged=True,throwaway_venv_removed=True),sort_keys=True))
