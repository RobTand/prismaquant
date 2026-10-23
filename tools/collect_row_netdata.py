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
import shlex
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.digests import DIRECT_ASCII_SPACED_LAX  # noqa: E402
from tools.pq_profile_artifact import validate_netdata_window  # noqa: E402

REQUIRED_CONTEXTS = {"system.cpu", "system.ram", "system.io",
                     "nvidia_smi.gpu_power_draw"}
HOSTS = ("sparklina", "sparky")




def fetch(host: str, endpoint: str) -> dict:
    url = "http://127.0.0.1:19999/api/v1/" + endpoint
    local = socket.gethostname().split(".")[0]
    if host == local:
        with urllib.request.urlopen(url, timeout=8) as response:
            return json.load(response)
    raw = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         host, "curl -fsS --max-time 8 " + shlex.quote(url)],
        check=True, capture_output=True, text=True, timeout=15).stdout
    return json.loads(raw)




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
            validate_netdata_window(data, after=args.after, before=args.before)
            series[chart] = {"points": len(data["data"]),
                             "labels": data["labels"],
                             "data": data["data"]}
        document["hosts"][host] = {"charts": charts, "series": series}
    print(DIRECT_ASCII_SPACED_LAX.text(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
