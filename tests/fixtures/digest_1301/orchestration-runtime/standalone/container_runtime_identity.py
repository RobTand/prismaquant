# Frozen executable excerpts from git blob be7bb7b8d36f3af29a0185c673e64b608f0797e2; docstrings removed only.
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence
RUNTIME_IDENTITY_SCHEMA = 'prismaquant.container_runtime_identity.v1'
_IDENTITY_KEYS = {'schema', 'target', 'image_ref', 'image_id', 'prismaquant_git_commit', 'prismaquant_source_sha256', 'implementation_receipt_sha256'}
_COMMIT_RE = re.compile('(?:[0-9a-f]{40}|[0-9a-f]{64})')
_IMAGE_ID_RE = re.compile('sha256:[0-9a-f]{64}')
_IMAGE_REF_RE = re.compile('(?:[^@\\s]+@sha256:[0-9a-f]{64}|sha256:[0-9a-f]{64})')
_SHA256_RE = re.compile('[0-9a-f]{64}')

class RuntimeIdentityError(ValueError):
    pass  # Original class body contained only its docstring.

def image_content_sha256(image: Mapping[str, Any]) -> str:
    if not isinstance(image, Mapping):
        raise RuntimeIdentityError('Docker image inspection must be an object')
    for field in ('Os', 'Architecture'):
        if not isinstance(image.get(field), str) or not image[field]:
            raise RuntimeIdentityError(f'Docker image has no {field}')
    variant = image.get('Variant', '')
    if not isinstance(variant, str):
        raise RuntimeIdentityError('Docker image has invalid Variant')
    rootfs = image.get('RootFS')
    if not isinstance(rootfs, dict) or rootfs.get('Type') != 'layers' or (not isinstance(rootfs.get('Layers'), list)) or (not all((isinstance(layer, str) and _IMAGE_ID_RE.fullmatch(layer) for layer in rootfs['Layers']))):
        raise RuntimeIdentityError('Docker image has no exact RootFS layer list')
    config = image.get('Config')
    if not isinstance(config, dict):
        raise RuntimeIdentityError('Docker image has no runtime Config')
    content = {'schema': 'prismaquant.container_image_content.v1', 'os': image['Os'], 'architecture': image['Architecture'], 'variant': variant, 'rootfs': rootfs, 'config': config}
    raw = json.dumps(content, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def _reject_duplicate_members(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeIdentityError(f'duplicate JSON member {key!r}')
        result[key] = value
    return result

def _load_json_object(path: Path, *, where: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_reject_duplicate_members)
    except RuntimeIdentityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeIdentityError(f'{where} is unreadable: {exc}') from exc
    if not isinstance(payload, dict):
        raise RuntimeIdentityError(f'{where} must be a JSON object')
    return payload

def prismaquant_source_sha256(package_root: Path) -> str:
    root = package_root.resolve(strict=True)
    if not root.is_dir() or root.name != 'prismaquant':
        raise RuntimeIdentityError(f'PrismaQuant package root must name a prismaquant directory: {root}')
    paths = sorted((path for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.relative_to(root).parts and (path.suffix not in {'.pyc', '.pyo'})))
    if not paths:
        raise RuntimeIdentityError(f'no package inputs found under {root}')
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode('utf-8')
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise RuntimeIdentityError(f'cannot hash PrismaQuant package input {path}') from exc
        digest.update(len(relative).to_bytes(4, 'big'))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    return digest.hexdigest()

def _validate_identity(payload: Mapping[str, Any], *, where: str) -> dict[str, str]:
    if set(payload) != _IDENTITY_KEYS:
        raise RuntimeIdentityError(f'{where} expected exactly {sorted(_IDENTITY_KEYS)}, got {sorted(payload)}')
    values = {str(key): value for key, value in payload.items()}
    if values['schema'] != RUNTIME_IDENTITY_SCHEMA:
        raise RuntimeIdentityError(f"{where} has unsupported schema {values['schema']!r}")
    target = values['target']
    if target not in {'dense', 'dsv4'}:
        raise RuntimeIdentityError(f'{where} has invalid target {target!r}')
    image_ref = values['image_ref']
    image_id = values['image_id']
    commit = values['prismaquant_git_commit']
    source_sha = values['prismaquant_source_sha256']
    receipt_sha = values['implementation_receipt_sha256']
    if not isinstance(image_ref, str) or _IMAGE_REF_RE.fullmatch(image_ref) is None:
        raise RuntimeIdentityError(f'{where} image_ref is not immutable')
    if not isinstance(image_id, str) or _IMAGE_ID_RE.fullmatch(image_id) is None:
        raise RuntimeIdentityError(f'{where} image_id is not a full Docker ID')
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeIdentityError(f'{where} PrismaQuant commit is invalid')
    for name, value in (('prismaquant_source_sha256', source_sha), ('implementation_receipt_sha256', receipt_sha)):
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise RuntimeIdentityError(f'{where} {name} is invalid')
    return {key: str(value) for key, value in values.items()}

def _receipt_sha256(receipt_path: Path, *, expected_commit: str) -> str:
    try:
        data = receipt_path.read_bytes()
    except OSError as exc:
        raise RuntimeIdentityError(f'implementation receipt is unreadable: {receipt_path}') from exc
    receipt = _load_json_object(receipt_path, where='implementation receipt')
    if receipt.get('git_commit') != expected_commit:
        raise RuntimeIdentityError(f'implementation receipt commit does not match the mounted PrismaQuant commit {expected_commit}')
    return hashlib.sha256(data).hexdigest()

def _identity_diff(expected: Mapping[str, str], observed: Mapping[str, str]) -> list[str]:
    return sorted((key for key in _IDENTITY_KEYS if expected.get(key) != observed.get(key)))

def _verify_stored_identity(destination: Path, expected: Mapping[str, str]) -> None:
    observed = _validate_identity(_load_json_object(destination, where='stored container runtime identity'), where='stored container runtime identity')
    differing = _identity_diff(expected, observed)
    if differing:
        raise RuntimeIdentityError('container runtime differs from the resumable campaign in: ' + ', '.join(differing))

def write_or_verify_identity(*, identity_path: Path, checkpoint_root: Path, source_root: Path, target: str, image_ref: str, image_id: str, git_commit: str, implementation_receipt: Path) -> dict[str, str]:
    package_root = source_root.resolve(strict=True) / 'prismaquant'
    payload = _validate_identity({'schema': RUNTIME_IDENTITY_SCHEMA, 'target': target, 'image_ref': image_ref, 'image_id': image_id, 'prismaquant_git_commit': git_commit, 'prismaquant_source_sha256': '0' * 64, 'implementation_receipt_sha256': '0' * 64}, where='requested container runtime identity')
    payload['prismaquant_source_sha256'] = prismaquant_source_sha256(package_root)
    payload['implementation_receipt_sha256'] = _receipt_sha256(implementation_receipt, expected_commit=git_commit)
    checkpoint = checkpoint_root.resolve(strict=True)
    destination = identity_path.resolve(strict=False)
    if checkpoint not in destination.parents:
        raise RuntimeIdentityError(f'runtime identity must live under checkpoint root {checkpoint}')
    if destination.exists():
        _verify_stored_identity(destination, payload)
        return payload
    legacy = sorted((path for path in checkpoint.rglob('*') if path.is_file() and path.resolve(strict=False) != destination))
    if legacy:
        preview = ', '.join((str(path) for path in legacy[:3]))
        raise RuntimeIdentityError(f'checkpoint tree predates container runtime identity; refusing to adopt existing files ({preview})')
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix=f'.{destination.name}.', dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _verify_stored_identity(destination, payload)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    write = subparsers.add_parser('write-or-verify')
    write.add_argument('--identity', type=Path, required=True)
    write.add_argument('--checkpoint-root', type=Path, required=True)
    write.add_argument('--source-root', type=Path, required=True)
    write.add_argument('--target', choices=('dense', 'dsv4'), required=True)
    write.add_argument('--image-ref', required=True)
    write.add_argument('--image-id', required=True)
    write.add_argument('--git-commit', required=True)
    write.add_argument('--implementation-receipt', type=Path, required=True)
    verify = subparsers.add_parser('verify-mounted')
    verify.add_argument('--identity', type=Path, required=True)
    verify.add_argument('--expected-root', type=Path, required=True)
    verify.add_argument('--expected-image-ref', required=True)
    verify.add_argument('--expected-image-id', required=True)
    verify.add_argument('--expected-git-commit', required=True)
    source_sha = subparsers.add_parser('source-sha256', help='print the canonical hash of one mounted PrismaQuant package')
    source_sha.add_argument('--source-root', type=Path, required=True)
    return parser

def main(argv: Sequence[str] | None=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == 'source-sha256':
            digest = prismaquant_source_sha256(args.source_root.resolve(strict=True) / 'prismaquant')
            print(digest)
            return 0
        if args.command == 'write-or-verify':
            payload = write_or_verify_identity(identity_path=args.identity, checkpoint_root=args.checkpoint_root, source_root=args.source_root, target=args.target, image_ref=args.image_ref, image_id=args.image_id, git_commit=args.git_commit, implementation_receipt=args.implementation_receipt)
        else:
            payload = verify_mounted_runtime(identity_path=args.identity, expected_root=args.expected_root, expected_image_ref=args.expected_image_ref, expected_image_id=args.expected_image_id, expected_git_commit=args.expected_git_commit)
    except RuntimeIdentityError as exc:
        print(f'container-runtime: ERROR: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0
