#!/usr/bin/env python3
"""Diff one file between two row-0055 arms, under compare_arms' decode rules.

    diff_entry.py --a NAME=DIR --b NAME=DIR FILE [FILE ...]

``FILE`` is a path relative to each arm directory, such as
``cache/<unit>__TESSERA_E2M1_K2_R896.pt``. The file is decoded (pickle, JSON,
``torch.load`` or safetensors), each arm's own directory is replaced by
``<arm>`` in every string, and every differing dotted field is printed with
both values. This reports one file in seconds, where ``compare_arms.py`` reads
every file both arms wrote.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_arms as ca  # noqa: E402


def unwrap(value):
    """Decode nested pickled payloads so the diff reports their fields.

    A checkpoint unit shard is an envelope whose ``payload`` is the unit state
    pickled on its own, beside that pickle's digest
    (``cost_stage_checkpoint.write_unit``). Comparing the bytes reports the
    whole payload as one opaque difference, so unpickle anything that carries
    a pickle protocol header and recurse.
    """
    if isinstance(value, bytes):
        if value[:1] != b"\x80":
            return value
        try:
            return unwrap(pickle.loads(value))
        except Exception:  # noqa: BLE001 - not a pickle after all
            return value
    if isinstance(value, dict):
        return {key: unwrap(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(unwrap(item) for item in value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="NAME=DIR")
    parser.add_argument("--b", required=True, help="NAME=DIR")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    (name_a, arm_a), (name_b, arm_b) = (
        (item.split("=", 1)[0], Path(item.split("=", 1)[1]).resolve())
        for item in (args.a, args.b))
    worst = 0
    for name in args.files:
        pa, pb = arm_a / name, arm_b / name
        same = ca.sha256(pa) == ca.sha256(pb)
        print(f"{name}\n  sha256 equal: {same}")
        if same:
            continue
        da = ca.substitute(unwrap(ca.decode(pa)), arm_a)
        db = ca.substitute(unwrap(ca.decode(pb)), arm_b)
        paths = ca.differences(da, db)
        print(f"  differing fields: {len(paths)}")
        for path in paths:
            va, vb = da, db
            for part in path:
                va, vb = va[part], vb[part]
            dotted = ".".join(map(str, path)) or "<root>"
            print(f"    {dotted}\n      {name_a}: {repr(va)[:300]}"
                  f"\n      {name_b}: {repr(vb)[:300]}"
                  f"\n      volatile: {ca.volatile(path)}")
        if isinstance(da, dict):
            print(f"  keys: {sorted(map(str, da))}")
        worst = max(worst, len(paths))
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
