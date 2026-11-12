"""A POSIX-sh stand-in for podman, driven by files, so owned-run tests spawn no Python child."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

SCRIPT = r'''#!/bin/sh
D="__DIR__"
echo "$*" >> "$D/calls.log"
verb=$1; shift
cid=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
case "$verb" in
 create)
  [ -f "$D/create.fail" ] && { echo "create refused" >&2; exit 125; }
  while [ $# -gt 0 ]; do [ "$1" = "--cidfile" ] && { printf '%s' "$cid" > "$2"; }; shift; done
  : > "$D/exists"; echo "$cid";;
 inspect)
  cat "$D/inspect.json";;
 start)
  [ -f "$D/start.sleep" ] && exec sleep 30
  exit "$(cat "$D/start.rc" 2>/dev/null || echo 0)";;
 stop) exit 0;;
 rm)
  [ -f "$D/rm.fail" ] && { echo "rm refused" >&2; exit 1; }
  rm -f "$D/exists"; exit 0;;
 container)
  [ -f "$D/exists" ] && exit 0; exit 1;;
 *) exit 2;;
esac
'''
CID = "a" * 64


def install(monkeypatch, runner, directory, *, cpuset="21,22,23,30", devices=None, image="sha256:" + "b" * 64,
            start_rc=0):
    directory = Path(directory)
    script = directory / "podman"
    script.write_text(SCRIPT.replace("__DIR__", str(directory)))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (directory / "inspect.json").write_text(json.dumps([{
        "Id": CID, "Image": image.removeprefix("sha256:"),
        "HostConfig": {"CpusetCpus": cpuset, "Devices": devices or []}}]))
    (directory / "start.rc").write_text(str(start_rc))
    monkeypatch.setattr(runner, "PODMAN", str(script))
    return script


def calls(directory):
    path = Path(directory) / "calls.log"
    return [tuple(line.split()) for line in path.read_text().splitlines()] if path.exists() else []
