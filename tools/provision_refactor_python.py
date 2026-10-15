#!/usr/bin/env python3
"""Build a separate Git-qualified interpreter from reviewed producer artifacts.

Run through PrismaBuild. Borrow individual qualified numerical packages and
metadata; exclude Tessera and package search-path files. Install the real exact
Git producer without dependencies, then compare its package to the reviewed
wheel. A subsequent pbtest action owns strict Git/RECORD qualification.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import venv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from prismaquant.digests import bytes_sha256hex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--wheel-sha256', required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    git_env = dict(os.environ)
    git_config_index = int(git_env.get("GIT_CONFIG_COUNT", "0"))
    git_env[f"GIT_CONFIG_KEY_{git_config_index}"] = "pack.threads"
    git_env[f"GIT_CONFIG_VALUE_{git_config_index}"] = "1"
    git_env["GIT_CONFIG_COUNT"] = str(git_config_index + 1)
    raw = args.wheel.read_bytes()
    if bytes_sha256hex(raw) != args.wheel_sha256:
        parser.error('wheel bytes differ from the reviewed delivery')
    kind = subprocess.check_output(['git', '-C', str(args.source_repo),
                                    'cat-file', '-t', args.source_commit], text=True, env=git_env).strip()
    if kind != 'commit' or len(args.source_commit) != 40:
        parser.error('source identity must be the exact real Git commit')
    tree = subprocess.check_output(['git', '-C', str(args.source_repo), 'rev-parse',
                                   args.source_commit + '^{tree}'], text=True, env=git_env).strip()
    receipt_path = target / 'refactor-environment.json'
    identity = {'wheel_sha256': args.wheel_sha256, 'source_commit': args.source_commit,
                'source_tree': tree, 'contract_sha256': args.contract_sha256}
    if receipt_path.exists() and json.loads(receipt_path.read_text())['identity'] != identity:
        parser.error('scoped environment belongs to another reviewed identity')
    published = receipt_path.exists()
    python = target / 'bin/python'
    if published:
        existing = json.loads(receipt_path.read_text())
        borrowed = existing['borrowed_dependencies']
        site = Path(subprocess.check_output(
            [str(python), '-B', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
        wheel = target / 'reviewed-inputs' / args.wheel.name
        if wheel.read_bytes() != raw:
            parser.error('retained reviewed wheel differs; provision a fresh prefix')
    else:
        borrowed_site = Path(sysconfig.get_path('purelib')).resolve()
        venv.EnvBuilder(with_pip=False, system_site_packages=False).create(target)
        python = target / 'bin/python'
        site = Path(subprocess.check_output(
            [str(python), '-B', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
        borrowed = []
        for path in sorted(borrowed_site.iterdir()):
            name = path.name.lower()
            if (name.startswith(('tessera', 'prismaquant')) or path.suffix in {'.pth', '.pyc'}
                    or name == '__pycache__'):
                continue
            destination = site / path.name
            if not destination.exists() and not destination.is_symlink():
                destination.symlink_to(path, target_is_directory=path.is_dir())
            borrowed.append({'path': str(path), 'destination': str(destination)})
        # Hashed console entry points in borrowed RECORDs retain their actual bytes.
        for path in sorted(Path(sys.executable).parent.iterdir()):
            if path.name.startswith(('python', 'activate')) or path.is_dir():
                continue
            destination = target / 'bin' / path.name
            if not destination.exists() and not destination.is_symlink():
                destination.symlink_to(path)
        inputs = target / 'reviewed-inputs'
        inputs.mkdir(exist_ok=True)
        wheel = inputs / args.wheel.name
        wheel.write_bytes(raw)
        source = target / 'reviewed-source'
        if not source.exists():
            subprocess.run(['git', 'clone', '--no-hardlinks', '--no-checkout',
                            str(args.source_repo), str(source)], check=True, env=git_env)
        actual_tree = subprocess.check_output(['git', '-C', str(source), 'rev-parse',
                                              args.source_commit + '^{tree}'], text=True, env=git_env).strip()
        if actual_tree != tree:
            parser.error('isolated source clone differs from the reviewed Git tree')
        subprocess.run([str(python), '-B', '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--no-build-isolation', '--ignore-installed',
                        'git+' + source.as_uri() + '@' + args.source_commit], check=True, env=git_env)
    code = '''
import importlib.metadata, importlib.resources, json, pathlib, sys, zipfile
sys.path.insert(0, sys.argv[5])
from prismaquant.digests import bytes_sha256hex
import tessera
from tessera import metadata
import torch
site, wheel, commit, expected = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
owners = importlib.metadata.packages_distributions().get('tessera', [])
assert owners == ['tessera-quant'], owners
dist = importlib.metadata.distribution(owners[0])
direct = json.loads(dist.read_text('direct_url.json'))
assert direct['vcs_info']['vcs'] == 'git' and direct['vcs_info']['commit_id'] == commit
assert not direct.get('dir_info', {}).get('editable', False)
with zipfile.ZipFile(wheel) as archive:
    package_files = [name for name in archive.namelist() if name.startswith('tessera/') and not name.endswith('/')]
    for name in package_files:
        assert (site / name).read_bytes() == archive.read(name), name
assert pathlib.Path(tessera.__file__).resolve().is_relative_to(site.resolve())
assert metadata.METADATA_API_VERSION == 'tessera.producer-metadata.v1'
contract = importlib.resources.files('tessera').joinpath('serving', 'runtime_contract.json').read_bytes()
assert bytes_sha256hex(contract) == expected
record = dist._path.joinpath('RECORD').read_bytes()
print(json.dumps({'python': sys.executable, 'prefix': sys.prefix,
    'torch': torch.__version__, 'torch_file': torch.__file__,
    'tessera_file': tessera.__file__, 'metadata_file': metadata.__file__,
    'metadata_api': metadata.METADATA_API_VERSION, 'contract_sha256': expected,
    'cuda_available': torch.cuda.is_available(), 'owners': owners,
    'direct_url': direct, 'record_sha256': bytes_sha256hex(record),
    'wheel_package_files_verified': len(package_files),
    'strict_pbtest_record_gate': 'pending supported CLI action'}))
'''
    qualified = json.loads(subprocess.check_output(
        [str(python), '-B', '-c', code, str(site), str(wheel), args.source_commit,
         args.contract_sha256, str(REPO_ROOT)], text=True))
    receipt = {'schema': 'prismaquant.refactor_environment.v2', 'identity': identity,
               'base_python': sys.executable, 'borrowed_dependencies': borrowed,
               'source_repository': str(args.source_repo), 'wheel_source': str(args.wheel),
               'qualification': qualified}
    if not published:
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'receipt_path': str(receipt_path), 'identity': identity, 'reused': published,
                      'qualification': qualified}))


if __name__ == '__main__':
    main()
