#!/usr/bin/env python3
"""Read Netdata charts over one arm's window, per box.

    netdata_window.py --host http://BOX:19999 --list PREFIX [--list PREFIX ...]
    netdata_window.py --host http://BOX:19999 --after UNIX --before UNIX \
        --chart CHART [--chart CHART ...] --out WINDOW.json

``--list`` prints the chart ids that start with any prefix. Otherwise every
chart is read at its native resolution over ``[after, before]`` and each
dimension's mean, maximum and sample count are written, with the box and the
window, so a box-level reading sits beside the in-process profile.
"""
from __future__ import annotations

import argparse
import json
import statistics
import urllib.parse
import urllib.request


def get(host: str, path: str, **params):
    url = f"{host.rstrip('/')}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True)
    parser.add_argument("--list", action="append", default=[])
    parser.add_argument("--after", type=float)
    parser.add_argument("--before", type=float)
    parser.add_argument("--chart", action="append", default=[])
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.list:
        charts = get(args.host, "/api/v1/charts").get("charts", {})
        for name in sorted(charts):
            if name.startswith(tuple(args.list)):
                print(name, "|", charts[name].get("title", ""), "|", charts[name].get("units", ""))
        return 0
    window = dict(host=args.host, after=args.after, before=args.before, charts={})
    for chart in args.chart:
        try:
            data = get(args.host, "/api/v1/data", chart=chart, after=int(args.after),
                       before=int(args.before), points=0, group="average", format="json",
                       options="seconds")
        except Exception as error:  # noqa: BLE001
            window["charts"][chart] = dict(error=str(error))
            continue
        labels, rows = data["labels"], data["data"]
        dims = {}
        for index, label in enumerate(labels[1:], start=1):
            values = [row[index] for row in rows if row[index] is not None]
            if values:
                dims[label] = dict(mean=round(statistics.fmean(values), 3),
                                   max=round(max(values), 3), min=round(min(values), 3),
                                   samples=len(values))
        window["charts"][chart] = dims
    text = json.dumps(window, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
