#!/usr/bin/env python3
"""Pool and Spark watcher for the head walk measurement (dec-1007-031239-b815).

Two commands:
  quiet  Return 0 only if the dl380g10 pool and the chosen Spark stay quiet.
  watch  Stop the measurement with SIGTERM if the pool or the Spark gets busy.

The watcher makes read-only Netdata queries through ssh. Its only action is
one pkill -TERM on the Spark. It logs every sample to a jsonl file.
"""
import argparse
import json
import shlex
import subprocess
import sys
import time

POOL_HOST = "dl380g10"
HDD = ("sda", "sdb", "sdc", "sdd", "sde")
QUIET_UTIL_PCT, QUIET_PSI10_PCT = 20.0, 5.0
STOP_UTIL_PCT, STOP_UTIL_SAMPLES, STOP_PSI10_PCT = 85.0, 4, 25.0
# Third limit: read wait above 40 ms or twice the pre-run baseline, whichever is larger.
STOP_READ_AWAIT_MS, BASELINE_SAMPLES = 40.0, 3
SPARK_BUSY_GPU_W, SPARK_BUSY_USED_GIB = 25.0, 40.0


def ssh(host, command, timeout=40):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", host, command],
        capture_output=True, text=True, timeout=timeout, check=True).stdout


def data_url(chart):
    return ("http://127.0.0.1:19999/api/v1/data?chart=" + chart +
            "&after=-10&before=0&group=average&points=1&format=json")


def fetch(host, charts):
    """Return {chart: {label: value}} from one ssh call."""
    script = "; ".join(
        f"echo {shlex.quote(c)} $(curl -fsS --max-time 8 {shlex.quote(data_url(c))} | tr -d '\\n')"
        for c in charts)
    out = {}
    for line in ssh(host, script).splitlines():
        chart, _, raw = line.partition(" ")
        payload = json.loads(raw)
        out[chart] = dict(zip(payload["labels"][1:], payload["data"][0][1:]))
    return out


_gpu_chart = {}


def gpu_chart(host):
    if host not in _gpu_chart:
        raw = ssh(host, "curl -fsS --max-time 8 http://127.0.0.1:19999/api/v1/charts")
        charts = json.loads(raw)["charts"]
        found = [k for k, v in charts.items() if v.get("context") == "nvidia_smi.gpu_power_draw"]
        _gpu_chart[host] = found[0] if found else None
    return _gpu_chart[host]


def pool_sample():
    charts = ([f"disk_util.{d}" for d in HDD] + [f"disk_await.{d}" for d in HDD]
              + ["system.io_full_pressure"])
    got = fetch(POOL_HOST, charts)
    util = max(got[f"disk_util.{d}"].get("utilization") or 0.0 for d in HDD)
    # The pool is local ZFS on dl380g10, so no NFS client READ round trip exists
    # there (dec-1008-140953-047f). The disk read wait stands in for it.
    wait = max(got[f"disk_await.{d}"].get("reads") or 0.0 for d in HDD)
    return {"hdd_util_max_pct": round(util, 2),
            "hdd_read_await_ms_max": round(wait, 2),
            "io_full_pressure_10": round(got["system.io_full_pressure"].get("full 10") or 0.0, 2)}


def spark_sample(host):
    charts = ["system.ram"]
    gpu = gpu_chart(host)
    if gpu:
        charts.append(gpu)
    got = fetch(host, charts)
    used = (got["system.ram"].get("used") or 0.0) / 1024.0
    power = None if not gpu else round(max(got[gpu].values()), 2)
    return {"ram_used_gib": round(used, 2), "gpu_power_w": power}


def spark_busy(sample):
    if sample["ram_used_gib"] > SPARK_BUSY_USED_GIB:
        return f"memory used {sample['ram_used_gib']} GiB is above {SPARK_BUSY_USED_GIB}"
    if sample["gpu_power_w"] is not None and sample["gpu_power_w"] > SPARK_BUSY_GPU_W:
        return f"GPU power {sample['gpu_power_w']} W is above {SPARK_BUSY_GPU_W}"
    return None


def take(spark, log):
    row = {"epoch": round(time.time(), 1), **pool_sample()}
    if spark:
        row["spark"] = spark
        row.update(spark_sample(spark))
    log.write(json.dumps(row, sort_keys=True) + "\n")
    log.flush()
    return row


def quiet(args, log):
    deadline = time.time() + args.seconds
    count = 0
    while True:
        row = take(args.spark, log)
        count += 1
        if row["hdd_util_max_pct"] >= QUIET_UTIL_PCT or row["io_full_pressure_10"] >= QUIET_PSI10_PCT:
            print("NOT QUIET (pool): " + json.dumps(row, sort_keys=True))
            return 3
        busy = spark_busy(row) if args.spark and not args.no_spark_busy else None
        if busy:
            print("NOT QUIET (spark): " + busy + " " + json.dumps(row, sort_keys=True))
            return 3
        if time.time() >= deadline:
            print(f"QUIET for {args.seconds} s over {count} samples: " + json.dumps(row, sort_keys=True))
            return 0
        time.sleep(args.interval)


def watch(args, log):
    deadline = time.time() + args.max_seconds
    high = 0
    baseline_rows, await_limit = [], STOP_READ_AWAIT_MS
    while time.time() < deadline:
        if args.done_file and __import__("os").path.exists(args.done_file):
            print("DONE file seen; the watcher ends")
            return 0
        row = take(args.spark, log)
        if len(baseline_rows) < BASELINE_SAMPLES:
            # The first samples come before the run starts. They set the baseline.
            baseline_rows.append(row["hdd_read_await_ms_max"])
            if len(baseline_rows) == BASELINE_SAMPLES:
                await_limit = max(STOP_READ_AWAIT_MS, 2.0 * sorted(baseline_rows)[BASELINE_SAMPLES // 2])
                log.write(json.dumps({"read_await_limit_ms": await_limit,
                                      "baseline_ms": baseline_rows}) + "\n")
                log.flush()
        high = high + 1 if row["hdd_util_max_pct"] > STOP_UTIL_PCT else 0
        reason = None
        if high >= STOP_UTIL_SAMPLES:
            reason = f"HDD utilization above {STOP_UTIL_PCT} for {high} samples"
        elif row["io_full_pressure_10"] > STOP_PSI10_PCT:
            reason = f"I/O full pressure above {STOP_PSI10_PCT}"
        elif len(baseline_rows) >= BASELINE_SAMPLES and row["hdd_read_await_ms_max"] > await_limit:
            reason = f"disk read wait {row['hdd_read_await_ms_max']} ms above {await_limit} ms"
        elif args.spark and not args.no_spark_busy and spark_busy(row):
            reason = "Spark busy (D44 has first claim): " + spark_busy(row)
        if reason:
            stop = subprocess.run(["ssh", "-o", "BatchMode=yes", args.stop_host or args.spark, "pkill", "-TERM", "-f",
                                   shlex.quote(args.pattern)], capture_output=True, text=True)
            log.write(json.dumps({"epoch": round(time.time(), 1), "STOP": reason,
                                  "pkill_rc": stop.returncode}) + "\n")
            log.flush()
            print("STOP: " + reason + f" (pkill rc {stop.returncode})")
            return 4
        time.sleep(args.interval)
    print("max seconds reached; the watcher ends")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("quiet", "watch"))
    parser.add_argument("--spark", default=None)
    # Box mode (dec-1008-140953-047f): the measurement runs as a PrismaBuild x86
    # action on dl380g10, so there is no Spark to sample or to protect. The
    # watcher samples the pool only and sends the one SIGTERM to --stop-host.
    parser.add_argument("--stop-host", default=None)
    parser.add_argument("--no-spark-busy", action="store_true")
    parser.add_argument("--log", required=True)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    parser.add_argument("--done-file", default=None)
    parser.add_argument("--pattern", default="[p]rofile_stage_b_head.*--mode scoped-walk")
    args = parser.parse_args()
    if args.command == "watch" and not (args.spark or args.stop_host):
        parser.error("watch needs --spark")
    with open(args.log, "a") as log:
        return (quiet if args.command == "quiet" else watch)(args, log)


if __name__ == "__main__":
    sys.exit(main())
