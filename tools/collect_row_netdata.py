"""Fetch both-Spark Netdata series for a row window (PQ #2463).

Runs inside a PB-admitted gb10 action, where the Spark names resolve.
Queries each host's ``/api/v1/charts`` for the required contexts
(``system.cpu``, ``system.ram``, ``system.io``,
``nvidia_smi.gpu_power_draw``), then fetches one bounded ``data`` window
per chart over ``[--after, --before]``. Validates every window the way
``tools/pq_row_profile_observer.py`` does: fresh finite samples for every
declared dimension. Prints the collected document to stdout.

Usage: ``python3 tools/collect_row_netdata.py --after A --before B``.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED_CONTEXTS = {"system.cpu", "system.ram", "system.io",
                     "nvidia_smi.gpu_power_draw"}
HOSTS = ("sparklina", "sparky")


def _command(args: list[str], timeout: int = 15) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          timeout=timeout).stdout


def fetch(host: str, endpoint: str) -> dict:
    url = "http://127.0.0.1:19999/api/v1/" + endpoint
    local = socket.gethostname().split(".")[0]
    if host == local:
        with urllib.request.urlopen(url, timeout=8) as response:
            return json.load(response)
    raw = _command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                    host, "curl -fsS --max-time 8 " + shlex.quote(url)])
    return json.loads(raw)


def validate_window(data: dict, *, after: float, before: float) -> None:
    labels, rows = data.get("labels"), data.get("data")
    if (not isinstance(labels, list) or len(labels) < 2 or labels[0] != "time"
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
            or not isinstance(rows, list) or not rows):
        raise RuntimeError("netdata labels or samples are missing")
    measured = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != len(labels):
            raise RuntimeError("netdata sample has malformed dimensions")
        stamp = row[0]
        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                or not after <= stamp <= before):
            raise RuntimeError("netdata sample is outside its window")
        for index, value in enumerate(row[1:], 1):
            if value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise RuntimeError("netdata sample is not finite numeric")
            measured.add(index)
    if measured != set(range(1, len(labels))):
        raise RuntimeError("netdata dimensions have no measured samples")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", type=float, required=True)
    parser.add_argument("--before", type=float, required=True)
    args = parser.parse_args(argv)
    if not args.after < args.before:
        parser.error("need --after < --before")
    document: dict = {"schema": "prismaquant.row_netdata.v1",
                      "after": args.after, "before": args.before,
                      "fetched_epoch": time.time(),
                      "fetch_host": socket.gethostname().split(".")[0],
                      "hosts": {}}
    for host in HOSTS:
        info = fetch(host, "charts")
        charts = [key for key, value in info["charts"].items()
                  if value.get("context") in REQUIRED_CONTEXTS]
        contexts = {info["charts"][key].get("context") for key in charts}
        if not REQUIRED_CONTEXTS <= contexts:
            raise SystemExit(
                f"required netdata contexts missing on {host}: "
                f"{sorted(REQUIRED_CONTEXTS - contexts)}")
        series = {}
        for chart in charts:
            query = urllib.parse.urlencode(dict(
                chart=chart, after=args.after, before=args.before,
                points=max(10, int(args.before - args.after) + 4),
                group="average", format="json", options="seconds"))
            data = fetch(host, "data?" + query)
            validate_window(data, after=args.after, before=args.before)
            series[chart] = {"points": len(data["data"]),
                             "labels": data["labels"],
                             "data": data["data"]}
        document["hosts"][host] = {"charts": charts, "series": series}
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
