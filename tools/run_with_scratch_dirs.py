"""Create the host bind sources a container row needs, then run it (PQ #2463).

Docker bind-mounts need an existing host path. The worker never creates
declared scratch roots. This wrapper creates the row's bind sources on the
host (mode 0700, no follow of raced symlinks), then execs the campaign
container launcher with the unchanged remainder argv.

Usage: ``python3 -m tools.run_with_scratch_dirs --dir D [--dir D...] --
python3 -m tools.tessera_campaign_container --spec ... -- ...``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _make_dir(raw: str) -> str:
    path = Path(raw)
    if not path.is_absolute() or path == Path("/"):
        raise SystemExit(f"refuse non-absolute scratch dir {raw!r}")
    if ".." in path.parts or "\x00" in raw or str(path) != raw:
        raise SystemExit(f"refuse non-canonical scratch dir {raw!r}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/", flags)
    try:
        fds = [fd]
        for part in path.parts[1:]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=fds[-1])
            except FileExistsError:
                pass
            next_fd = os.open(part, flags, dir_fd=fds[-1])
            fds.append(next_fd)
        info = os.fstat(fds[-1])
        if info.st_uid != os.getuid():
            raise SystemExit(f"scratch dir {raw!r} belongs to another uid")
    finally:
        for open_fd in reversed(fds[1:]):
            os.close(open_fd)
        os.close(fd)
    return raw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", action="append", default=[],
                        help="host bind source to create (repeatable)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    for raw in args.dir:
        _make_dir(raw)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a container command is required")
    os.execvp(command[0], command)
    return 1  # exec never returns


if __name__ == "__main__":
    raise SystemExit(main())
