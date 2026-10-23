"""Kernel mount facts used by staged readers and portable source identity."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


_MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")


@dataclass(frozen=True)
class MountFacts:
    point: str
    filesystem: str
    options: Mapping[str, str]


def find_mount(path: str | Path, *, table: str | Path = "/proc/self/mountinfo") -> MountFacts | None:
    target = os.path.abspath(path)
    try:
        rows = Path(table).read_text().splitlines()
    except OSError:
        return None
    best = None
    for row in rows:
        head, separator, tail = row.partition(" - ")
        fields = head.split()
        if separator:
            super_fields = tail.split()
            if len(fields) < 6 or len(super_fields) < 3:
                continue
            escaped_point, filesystem = fields[4], super_fields[0]
            option_fields = (fields[5], super_fields[2])
        else:
            if len(fields) < 4:
                continue
            escaped_point, filesystem = fields[1:3]
            option_fields = (fields[3],)
        point = _MOUNT_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), escaped_point)
        if target != point and not target.startswith(point.rstrip("/") + "/"):
            continue
        # The last mount at an equal path is above its autofs trigger.
        if best is not None and len(point) < len(best.point):
            continue
        options = {}
        for text in option_fields:
            for item in text.split(","):
                name, _, value = item.partition("=")
                options[name] = value
        best = MountFacts(point, filesystem, MappingProxyType(options))
    return best
