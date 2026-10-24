# Original-source witnesses, extracted without transforming bodies.
import hashlib, json, os, sys, subprocess
from pathlib import Path
from typing import *
def inspect_or_load(container):
    requested = container['image']
    found = subprocess.run(['docker', 'image', 'inspect', requested], capture_output=True, text=True)
    if found.returncode:
        bound = container.get('archive')
        if bound is None:
            raise RuntimeError('declared container image is unavailable: ' + found.stderr)
        with Path(bound['path']).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != bound['sha256']:
            raise RuntimeError('declared image archive bytes changed')
        subprocess.run(['docker', 'load', '--input', bound['path']], check=True)
        found = subprocess.run(['docker', 'image', 'inspect', requested], capture_output=True, text=True, check=True)
    rows = json.loads(found.stdout)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError('Docker returned no unique image inspection')
    return rows

