"""Fresh-interpreter proof that the profile digest bridge needs no heavy package."""
from pathlib import Path
import subprocess
import sys


def test_profile_digest_owner_import_without_torch_or_package():
    code="""
import builtins
real_import=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('prismaquant','torch','compressed_tensors'):
        raise AssertionError('heavy package imported: '+name)
    return real_import(name,*args,**kwargs)
builtins.__import__=guarded
from tools.pq_profile_digest import bytes_sha256hex
assert bytes_sha256hex(b'')=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
print('stdlib owner loaded')
"""
    result=subprocess.run([sys.executable,'-S','-c',code],cwd=Path(__file__).resolve().parents[1],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='stdlib owner loaded'
