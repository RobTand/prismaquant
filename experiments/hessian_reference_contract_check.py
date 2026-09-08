"""PB CPU validation with an independently hash-bound producer source archive."""
import argparse
import compileall
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--producer-archive',required=True)
    parser.add_argument('--producer-sha256',required=True)
    parser.add_argument('--producer-commit',required=True)
    args,tests=parser.parse_known_args()
    archive=Path(args.producer_archive)
    actual=hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != args.producer_sha256:
        raise RuntimeError('producer archive SHA-256 changed')
    with tempfile.TemporaryDirectory(prefix='bounded-hessian-producer-') as temporary:
        with tarfile.open(archive) as source:
            source.extractall(temporary,filter='data')
        producer=Path(temporary)
        sys.path.insert(0,str(Path.cwd()))
        sys.path.insert(0,str(producer/'src'))
        os.environ['TESSERA_REPO']=str(producer)
        import tessera
        if Path(tessera.__file__).resolve() != producer/'src/tessera/__init__.py':
            raise RuntimeError('loaded producer differs from the verified source archive')
        print(json.dumps(dict(producer_commit=args.producer_commit,
            producer_archive_sha256=actual,producer_module=str(tessera.__file__)),sort_keys=True),flush=True)
        files=['prismaquant/tessera_calibration_cache.py','prismaquant/tessera_campaign.py',
            'prismaquant/tessera_export_lane.py','prismaquant/tessera_menu.py',
            'prismaquant/tessera_materialization.py','tools/dispatch_tessera_campaign.py']
        if not all(compileall.compile_file(path,quiet=1) for path in files):
            raise RuntimeError('touched module compilation failed')
        import pytest
        return pytest.main(tests)


if __name__=='__main__':
    raise SystemExit(main())
