#!/usr/bin/env python3
"""Prepare an isolated wheel overlay using an already qualified dependency environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import venv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prismaquant.digests import bytes_sha256hex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--wheel-sha256', required=True)
    parser.add_argument('--contract-sha256', required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    raw = args.wheel.read_bytes()
    if bytes_sha256hex(raw) != args.wheel_sha256:
        parser.error('wheel bytes differ from the reviewed delivery')
    receipt_path = target / 'refactor-environment.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text())['wheel_sha256'] != args.wheel_sha256:
        parser.error('scoped environment belongs to another wheel identity')
    borrowed_site = Path(sysconfig.get_path('purelib')).resolve()
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(target)
    python = target / 'bin/python'
    site = Path(subprocess.check_output(
        [str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    (site / 'qualified-dependencies.pth').write_text(str(borrowed_site) + '\n')
    inputs = target / 'reviewed-inputs'
    inputs.mkdir(exist_ok=True)
    wheel = inputs / args.wheel.name
    wheel.write_bytes(raw)
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--ignore-installed', str(wheel)], check=True)
    code = '''
import importlib.resources, json, pathlib, sys, zipfile
import tessera
from tessera import metadata
import torch
site, wheel, expected = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        if name.endswith('/') or '.dist-info/' in name:
            continue
        assert (site / name).read_bytes() == archive.read(name), name
assert pathlib.Path(tessera.__file__).resolve().is_relative_to(site.resolve())
assert metadata.METADATA_API_VERSION == 'tessera.producer-metadata.v1'
from prismaquant.digests import bytes_sha256hex
contract = importlib.resources.files('tessera').joinpath('serving', 'runtime_contract.json').read_bytes()
assert bytes_sha256hex(contract) == expected
print(json.dumps({'python': sys.executable, 'prefix': sys.prefix,
    'torch': torch.__version__, 'torch_file': torch.__file__,
    'tessera_file': tessera.__file__, 'metadata_file': metadata.__file__,
    'metadata_api': metadata.METADATA_API_VERSION, 'contract_sha256': expected,
    'cuda_available': torch.cuda.is_available()}))
'''
    qualified = json.loads(subprocess.check_output(
        [str(python), '-c', code, str(site), str(wheel), args.contract_sha256], text=True))
    receipt = {'schema': 'prismaquant.refactor_environment.v1',
               'base_python': sys.executable, 'borrowed_dependency_site': str(borrowed_site),
               'wheel_source': str(args.wheel), 'wheel_sha256': args.wheel_sha256,
               'qualification': qualified}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
