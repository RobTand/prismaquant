"""Interleaved profiles of the complete GLM expert menu, with identical producer inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import cProfile
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import pstats
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace


def json_default(value):
    if isinstance(value, Fraction):
        return dict(numerator=value.numerator, denominator=value.denominator)
    raise TypeError(type(value).__name__)


def arm(source, producer, out):
    sys.path[:0] = [str(source), str(producer / "src")]
    from prismaquant.tessera_campaign import expand_menus_for_targets
    from prismaquant.tessera_menu import PARALLEL_NONE

    out.mkdir()
    name = "model.language_model.layers.4.mlp.experts.0.gate_proj"
    weights = {name: SimpleNamespace(shape=(2048, 4096))}
    profile = cProfile.Profile()
    start_unix, start = time.time(), time.perf_counter()
    with profile:
        menus = expand_menus_for_targets(weights, [name], mode="readable",
                                        tp_degree=1, parallel_kind=PARALLEL_NONE)
    seconds, end_unix = time.perf_counter() - start, time.time()
    profile.dump_stats(out / "menu.pstats")
    with (out / "profile.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(60)
    raw = json.dumps([asdict(row) for row in menus[name]], sort_keys=True,
                     separators=(",", ":"), default=json_default).encode()
    (out / "menu.json").write_bytes(raw)
    result = dict(start_unix=start_unix, end_unix=end_unix, seconds=seconds,
                  rows=len(menus[name]), menu_sha256=hashlib.sha256(raw).hexdigest(),
                  host=socket.gethostname(), python=sys.version,
                  affinity=sorted(os.sched_getaffinity(0)),
                  sources={name: hashlib.sha256((source / "prismaquant" / name).read_bytes()).hexdigest()
                           for name in ("tessera_footprint.py", "tessera_menu.py", "tessera_campaign.py")})
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tessera-source", type=Path, required=True)
    parser.add_argument("--base")
    parser.add_argument("--arm-source", type=Path)
    parser.add_argument("--netdata-endpoints", type=json.loads,
                        help="JSON mapping of sparky, sparklina and the measured hostname to reachable hosts/IPs")
    args = parser.parse_args()
    producer = args.tessera_source.resolve()
    if args.arm_source:
        arm(args.arm_source.resolve(), producer, args.out)
        return
    if not args.base:
        parser.error("--base is required for paired profiling")
    hosts = ("sparky", "sparklina", socket.gethostname())
    if (not isinstance(args.netdata_endpoints, dict) or set(args.netdata_endpoints) != set(hosts)
            or not all(isinstance(v, str) and v for v in args.netdata_endpoints.values())):
        parser.error("paired profiles require explicit --netdata-endpoints for both Sparks and this host")
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    baseline = subprocess.check_output(["git", "rev-parse", args.base], cwd=source, text=True).strip()
    producer_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=producer, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=producer), "dirty producer"
    results = []
    with tempfile.TemporaryDirectory(prefix="pq-menu-baseline-") as temp:
        base = Path(temp)
        archive = base / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), baseline], cwd=source, check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(base, filter="data")
        for index, label in enumerate(("before", "after", "after", "before")):
            output = args.out / f"{index}-{label}"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out", str(output),
                            "--tessera-source", str(producer), "--arm-source",
                            str(base if label == "before" else source)], check=True)
            results.append(dict(label=label, **json.loads((output / "result.json").read_text())))
    sys.path.insert(0, str(producer / "experiments"))
    from box_power_window import SERIES, _fetch

    after = int(min(row["start_unix"] for row in results))
    before = int(max(row["end_unix"] for row in results)) + 1
    queries = [(host, context, dims) for host in ("sparky", "sparklina", socket.gethostname())
               for context, dims in SERIES if host in ("sparky", "sparklina") or not context.startswith("nvidia_smi.")]
    def fetch(query):
        host, context, dims = query
        return host, context, _fetch(args.netdata_endpoints[host], context, dims, after, before, before - after)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, queries))
    telemetry = {host: {context: data for h, context, data in records if h == host}
                 for host in {h for h, _, _ in records}}
    (args.out / "netdata.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    assert all(data["doc"]["result"]["data"] for _, _, data in records), "missing Netdata series"
    assert len({row["menu_sha256"] for row in results}) == 1, "menu changed"
    assert all(row["rows"] == 5635 for row in results), "unexpected menu population"
    result = dict(scope="complete readable GLM expert menu at 2048x4096, TP1; no forwards or encodes",
                  baseline_commit=baseline, producer_commit=producer_commit,
                  arms=results, menus_identical=True, netdata_endpoints=args.netdata_endpoints)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
