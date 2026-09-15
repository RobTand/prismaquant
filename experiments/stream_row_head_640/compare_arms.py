#!/usr/bin/env python3
"""Compare row-0055 arms' outputs, and their wires against the census row.

    compare_arms.py --arm stream=DIR --arm load-all=DIR [--arm before=DIR ...]
                    --census-row ROW_DIR --out REPORT.json

Acceptance (a) of RobTand/prismaquant#640 compares the first two arms file by
file. Every file both arms wrote under ``cost.pkl``, ``cost.anchors.json``,
``cost.anchors.json.parts/`` and ``cache/`` is compared by SHA-256. A file that
differs is decoded (pickle, JSON, or ``torch.load``) and compared field by
field, after the arm directory is replaced by ``<arm>`` in every string. A
difference is reported as normalized only when every differing field is one
of these, which vary from run to run by construction:

* ``seconds``, ``encode_seconds``, ``wall_seconds``: ``time.time()`` deltas;
* ``memory_guard``, ``baseline``, ``anchor_batch_growth_bytes``: the CUDA
  memory guard's readings of this process;
* the ``capture-load-execution-<sha256>.json`` name and digest, which seal a
  record that holds a ``memory_guard`` snapshot. Its content is compared with
  that snapshot removed.

``cache/row-head-execution.json`` records which head ran and is reported, not
compared. Acceptance (b) compares every arm's ``cache/wire/*.tessera`` blob with
the census row's stored blob of the same name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

VOLATILE = {"seconds", "encode_seconds", "wall_seconds", "memory_guard", "baseline",
            "anchor_batch_growth_bytes"}
EXECUTION = re.compile(r"^cache/capture-load-execution-[0-9a-f]{64}\.json$")
SIDECAR = "cache/row-head-execution.json"
ROOTS = ("cost.pkl", "cost.anchors.json", "cost.anchors.json.parts", "cache")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def files(arm: Path) -> dict[str, Path]:
    found = {}
    for root in ROOTS:
        top = arm / root
        if top.is_file():
            found[root] = top
        elif top.is_dir():
            for path in sorted(top.rglob("*")):
                if path.is_file():
                    found[path.relative_to(arm).as_posix()] = path
    return found


def decode(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix in (".pkl",):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    if path.suffix == ".pt":
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    return None


def substitute(value, arm: Path):
    if isinstance(value, str):
        return value.replace(str(arm), "<arm>")
    if isinstance(value, dict):
        return {substitute(k, arm): substitute(v, arm) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(substitute(v, arm) for v in value)
    return value


def differences(a, b, path=()):
    """Every dotted path at which ``a`` and ``b`` differ."""
    try:
        import torch
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            same = (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                    and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
            return [] if same else [path]
    except ImportError:
        pass
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                out.append(path + (key,))
            else:
                out.extend(differences(a[key], b[key], path + (key,)))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [path]
        out = []
        for index, (x, y) in enumerate(zip(a, b)):
            out.extend(differences(x, y, path + (index,)))
        return out
    if hasattr(a, "__dict__") and type(a) is type(b):
        return differences(vars(a), vars(b), path)
    return [] if a == b else [path]


def volatile(path) -> bool:
    if any(part in VOLATILE for part in path):
        return True
    # cost.pkl names the execution record by path and digest; its content is
    # compared as its own file.
    return len(path) >= 2 and path[-2] == "capture_load_execution" and path[-1] in ("path", "sha256")


def strip_guard(record):
    return {k: v for k, v in record.items() if k != "memory_guard"}


def compare_pair(name_a, arm_a, name_b, arm_b):
    fa, fb = files(arm_a), files(arm_b)
    report = dict(identical=0, normalized=[], different=[], only_in={name_a: [], name_b: []},
                  sidecars={})
    exec_a = [k for k in fa if EXECUTION.match(k)]
    exec_b = [k for k in fb if EXECUTION.match(k)]
    for key in sorted(set(fa) | set(fb)):
        if key == SIDECAR:
            report["sidecars"] = {name_a: json.loads(fa[key].read_text()) if key in fa else None,
                                  name_b: json.loads(fb[key].read_text()) if key in fb else None}
            continue
        if EXECUTION.match(key):
            continue
        if key not in fa or key not in fb:
            report["only_in"][name_a if key in fa else name_b].append(key)
            continue
        if sha256(fa[key]) == sha256(fb[key]):
            report["identical"] += 1
            continue
        try:
            da, db = substitute(decode(fa[key]), arm_a), substitute(decode(fb[key]), arm_b)
        except Exception as error:  # noqa: BLE001
            report["different"].append(dict(file=key, reason=f"undecodable: {error}"))
            continue
        paths = differences(da, db)
        entry = dict(file=key, fields=[".".join(map(str, p)) for p in paths[:40]],
                     field_count=len(paths))
        if paths and all(volatile(p) for p in paths):
            report["normalized"].append(entry)
        else:
            report["different"].append(entry)
    if len(exec_a) != 1 or len(exec_b) != 1:
        report["different"].append(dict(file="cache/capture-load-execution-*.json",
                                        reason=f"{len(exec_a)} and {len(exec_b)} records"))
    else:
        ra = substitute(strip_guard(json.loads(fa[exec_a[0]].read_text())), arm_a)
        rb = substitute(strip_guard(json.loads(fb[exec_b[0]].read_text())), arm_b)
        paths = differences(ra, rb)
        entry = dict(file="cache/capture-load-execution-*.json (memory_guard removed)",
                     fields=[".".join(map(str, p)) for p in paths[:40]], field_count=len(paths))
        (report["different"] if paths else report["normalized"]).append(entry)
    manifests = {}
    for name, found in ((name_a, fa), (name_b, fb)):
        manifest = found.get("cost.anchors.json")
        manifests[name] = None if manifest is None else json.loads(manifest.read_text()).get(
            "identity_sha256")
    report["run_identity_sha256"] = manifests
    report["files"] = {name_a: len(fa), name_b: len(fb)}
    return report


def compare_wires(name, arm, census_row):
    wires = sorted((arm / "cache" / "wire").glob("*.tessera"))
    census = census_row / "cache" / "wire"
    result = dict(arm=name, wires=len(wires), identical=0, different=[], missing_in_census=[])
    for wire in wires:
        reference = census / wire.name
        if not reference.is_file():
            result["missing_in_census"].append(wire.name)
        elif sha256(wire) == sha256(reference):
            result["identical"] += 1
        else:
            result["different"].append(wire.name)
    result["census_wires"] = len(list(census.glob("*.tessera")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True,
                        help="NAME=DIR; the first two are compared file by file")
    parser.add_argument("--census-row", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = [(name, Path(path).resolve()) for name, path in
            (item.split("=", 1) for item in args.arm)]
    if len(arms) < 2:
        raise SystemExit("name at least two arms")
    (name_a, arm_a), (name_b, arm_b) = arms[:2]
    report = dict(schema="prismaquant.stream_row_head_640.compare.v1",
                  arms={name: str(path) for name, path in arms},
                  census_row=str(args.census_row),
                  outputs=compare_pair(name_a, arm_a, name_b, arm_b),
                  wires=[compare_wires(name, path, args.census_row) for name, path in arms])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    outputs = report["outputs"]
    print(json.dumps(dict(identical=outputs["identical"], normalized=len(outputs["normalized"]),
                          different=len(outputs["different"]), only_in=outputs["only_in"],
                          run_identity=outputs["run_identity_sha256"],
                          wires=[{k: (len(v) if isinstance(v, list) else v) for k, v in w.items()}
                                 for w in report["wires"]]), indent=1, default=str))
    clean = not outputs["different"] and not any(outputs["only_in"].values()) and all(
        not w["different"] and not w["missing_in_census"] for w in report["wires"])
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
