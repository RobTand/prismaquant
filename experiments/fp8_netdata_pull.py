"""Pull Netdata GPU power series for recorded bench windows (PQ #1398).

Reads EPOCHS_JSON (``{width: {arm: {"timed": [s, e], "power": [s, e]}}}``)
from stdin, queries the worker-local Netdata for GPU power over each power
window, and prints NETDATA_JSON. Discovery picks the first chart whose id
mentions power and gpu/nvidia/watt. Every lookup is best effort: unreachable
Netdata or a missing chart yields nulls, never an exception.
"""

from __future__ import annotations

import json
import statistics
import sys
import urllib.request

BASE = "http://localhost:19999/api/v1"


def _get(path: str):
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as response:
            return json.load(response)
    except Exception:
        return None


def _power_chart() -> str | None:
    charts = _get("/charts")
    if not charts:
        return None
    ids = list((charts.get("charts") or {}).keys())
    for chart_id in ids:
        low = chart_id.lower()
        if "power" in low and ("gpu" in low or "nvidia" in low or "watt" in low):
            return chart_id
    for chart_id in ids:
        if "power" in chart_id.lower():
            return chart_id
    return None


def _series_mean(chart: str, start: float, end: float):
    data = _get(
        f"/data?chart={chart}&after={int(start)}&before={int(end)}"
        f"&points=60&format=json"
    )
    if not data:
        return None, 0
    rows = data.get("data") or []
    values = [
        row[1] for row in rows
        if len(row) > 1 and isinstance(row[1], (int, float))
    ]
    if not values:
        return None, 0
    return statistics.fmean(values), len(values)


def main() -> None:
    epochs = json.loads(sys.stdin.read() or "{}")
    chart = _power_chart()
    result = {"chart": chart, "windows": {}}
    if chart is None:
        print("NETDATA_JSON:" + json.dumps(result), flush=True)
        return
    for width, arms in epochs.items():
        result["windows"][width] = {}
        for arm, windows in arms.items():
            start, end = windows["power"]
            mean, samples = _series_mean(chart, start, end)
            result["windows"][width][arm] = {
                "power_w_mean": mean,
                "power_samples": samples,
                "epoch_start": start,
                "epoch_end": end,
            }
    print("NETDATA_JSON:" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
