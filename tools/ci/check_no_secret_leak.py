#!/usr/bin/env python3
"""Portable secret scanner, shared byte-for-byte across repositories.

Arguments: one or more file/directory paths, recursively scanned, and repeated
--env NAME options naming environment variables containing secrets. There are
no default names. Secret values never belong in command-line arguments.

Exit codes: 0 means no detected value; 1 means a value was detected; 75 means
invalid arguments, an unreadable path/member or excessive nesting prevented a complete scan.
Incomplete scans take precedence over detections. Unset/empty variables are
reported as "not available, not scanned" without failing.

Checks raw, standard/URL-safe base64, and URL-encoded bytes, including gzip and
zstd members embedded in files or nested up to 32 levels. Magic at offset zero
claims a format: that member and its concatenated continuations must decode.
Embedded magic is tried too, but failed speculative decodes are ignored; all
bytes are always scanned raw. The same rules apply to each decoded payload.
Uses only the standard library. Zstd uses compression.zstd when available, otherwise members are
reported as "zstd: not scanned" without failing. No third-party decoder is used.
One file and its decompressed members are held in memory at a time.

Output names only paths, variable names and scan status, never secret values,
matching snippets or exception messages. Values in paths/names are redacted too.
"""
from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import re
from urllib.parse import quote, quote_plus
import zlib

try:
    from compression import zstd
except ImportError:
    zstd = None


class SecretScanParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(75, 'Invalid scan arguments; use --help.\n')


def secret_byte_forms(value):
    raw = value.encode()
    forms = {raw}
    for encoded in (base64.b64encode(raw), base64.urlsafe_b64encode(raw)):
        forms.update((encoded, encoded.rstrip(b'=')))
    for encoded in (quote(value), quote(value, safe=''), quote_plus(value)):
        forms.add(encoded.encode())
        forms.add(re.sub(r'%[0-9A-F]{2}', lambda match: match[0].lower(), encoded).encode())
    return forms


def inspect_secret_members(data, secrets, emit, path, depth=0):
    found = {name for name, forms in secrets.items() if any(form in data for form in forms)}
    incomplete = False
    for magic, kind in ((b'\x1f\x8b', 'gzip'), (b'\x28\xb5\x2f\xfd', 'zstd')):
        cursor = 0
        required_offset = 0
        while (offset := data.find(magic, cursor)) >= 0:
            required = offset == required_offset
            if kind == 'zstd' and zstd is None:
                emit(f'{path}: zstd: not scanned (compression.zstd unavailable)')
                break
            if depth >= 32:
                if required:
                    emit(f'{path}: compressed nesting limit, not scanned')
                    incomplete = True
                break
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if kind == 'gzip' else zstd.ZstdDecompressor()
            errors = (zlib.error, EOFError, ValueError) + ((zstd.ZstdError,) if zstd else ())
            try:
                decoded = decoder.decompress(data[offset:])
                if not decoder.eof:
                    raise EOFError
            except errors:
                if required:
                    emit(f'{path}: {kind} member unreadable, not scanned')
                    incomplete = True
                cursor = offset + len(magic)
                continue
            cursor = len(data) - len(decoder.unused_data)
            if required:
                required_offset = cursor
            nested_found, nested_incomplete = inspect_secret_members(
                decoded, secrets, emit, path, depth + 1)
            found.update(nested_found)
            incomplete |= nested_incomplete
    return found, incomplete


def main(argv=None):
    parser = SecretScanParser(description=__doc__)
    parser.add_argument('--env', action='append', required=True, metavar='NAME',
                        help='name of a secret environment variable; repeat for each variable')
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args(argv)
    secrets = {name: secret_byte_forms(value) for name in args.env
               if (value := os.environ.get(name))}
    redactions = sorted({form.decode(errors='replace') for forms in secrets.values() for form in forms},
                        key=len, reverse=True)

    def emit(message):
        while any(value in message for value in redactions):
            for value in redactions:
                message = message.replace(value, '')
        print(message)

    for name in dict.fromkeys(args.env):
        if name not in secrets:
            emit(f'{name}: not available, not scanned')
    detected = False
    incomplete = False

    def scan_file(path):
        nonlocal detected, incomplete
        try:
            data = path.read_bytes()
        except OSError:
            emit(f'{path}: unreadable path, not scanned')
            incomplete = True
            return
        found, partial = inspect_secret_members(data, secrets, emit, path)
        for name in sorted(found):
            emit(f'{path}: {name}: secret detected')
        detected |= bool(found)
        incomplete |= partial

    def walk_error(error):
        nonlocal incomplete
        emit(f'{error.filename}: unreadable directory, not scanned')
        incomplete = True

    for path in args.paths:
        if path.is_dir():
            for directory, subdirectories, filenames in os.walk(path, onerror=walk_error):
                # Do not silently omit directory symlinks, or follow cycles.
                for name in subdirectories:
                    linked = Path(directory, name)
                    if linked.is_symlink():
                        emit(f'{linked}: directory symlink, not scanned')
                        incomplete = True
                for name in filenames:
                    scan_file(Path(directory, name))
        else:
            scan_file(path)
    return 75 if incomplete else 1 if detected else 0


if __name__ == '__main__':
    raise SystemExit(main())
