#!/usr/bin/env python3
"""Summarize one row-0055 arm's py-spy profile, timeline and campaign log.

    profile_summary.py ARM_DIR [ARM_DIR ...] [--json OUT]

For each arm this reads files the harness wrote (run_arm.sh, entry.py):

* ``host.txt``: action start and end, so row wall is ``end - start``.
* ``campaign.log``: the first timestamp of each head milestone, relative to
  that start. The milestones are the row-head line, "activations ready",
  "round 1", the first and last committed encode batch, and "wrote .../cost.pkl".
* ``timeline.tsv``: 2 s GPU power and cumulative NFS read bytes. **Sustained
  GPU power** is the first sample that starts a 20 s run (10 samples) in which
  every sample is at least half of the arm's encode-phase median power. The
  encode phase runs from the first to the last encode batch line. NFS reads
  are reported as bytes read before that point and after it.
* ``pyspy.raw.txt``: collapsed stacks. Inclusive samples are counted for each
  named function, and the top self frames are listed.

Power is reported against the GB10's 140 W envelope. GPU utilization is not
read; it does not measure GB10 saturation.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ENVELOPE_W = 140.0
NAMED = ("_campaign_checkpoint_identity", "canonical_hessian_reference_descriptor",
         "_prefetch_seal_digests", "prefetch_capture", "_verified_capture_entry",
         "tensor_identity", "_stream_unit_identity", "finalize_row_stream", "RowStream",
         "_read ", "admit ", "_measure_anchor_batch", "_encode_and_render",
         "write_export_inputs", "snapshot_selected_weights", "_project_expert_population")
MILESTONES = (("row_head", "[campaign] row head:"),
              ("prefetched", "[campaign] calibration prefetched"),
              ("activations_ready", "[campaign] activations ready"),
              ("round_1", "[campaign] round 1:"),
              ("first_batch_committed", "[campaign] r1 "),
              ("finalized", "[campaign] row head: stream finalized"),
              ("cost_written", "/cost.pkl:"))


def host_window(arm: Path):
    text = (arm / "host.txt").read_text()
    start = float(re.search(r"start=([0-9.]+)", text).group(1))
    end = float(re.search(r"end=([0-9.]+)", text).group(1))
    return start, end


def log_milestones(arm: Path, start: float):
    found, batches = {}, []
    for line in (arm / "campaign.log").read_text(errors="replace").splitlines():
        match = re.match(r"^([0-9]{10}\.[0-9]+) (.*)$", line)
        if not match:
            continue
        stamp, body = float(match.group(1)), match.group(2)
        for name, needle in MILESTONES:
            if needle in body and name not in found:
                found[name] = round(stamp - start, 1)
        if re.match(r"\[campaign\] r1 \d+/\d+ batch=", body):
            batches.append(stamp)
    return found, batches


def timeline(arm: Path, start: float, batches):
    rows = []
    lines = (arm / "timeline.tsv").read_text().splitlines()[1:]
    for line in lines:
        parts = line.split("\t")
        try:
            rows.append((float(parts[0]), int(parts[2]), float(parts[4])))
        except (IndexError, ValueError):
            continue
    if not rows or not batches:
        return {}
    encode = [w for t, _, w in rows if batches[0] <= t <= batches[-1]]
    median = statistics.median(encode) if encode else 0.0
    sustained = None
    for index in range(len(rows) - 9):
        if all(w >= 0.5 * median for _, _, w in rows[index:index + 10]):
            sustained = rows[index][0]
            break
    first_read = rows[0][1]
    at = (lambda t: max((r for r in rows if r[0] <= t), key=lambda r: r[0], default=rows[0])[1])
    powers = [w for _, _, w in rows]
    return dict(
        samples=len(rows),
        encode_median_power_w=round(median, 2),
        encode_median_envelope_fraction=round(median / ENVELOPE_W, 3),
        encode_mean_power_w=round(statistics.fmean(encode), 2) if encode else None,
        whole_mean_power_w=round(statistics.fmean(powers), 2),
        peak_power_w=round(max(powers), 2),
        sustained_power_seconds=None if sustained is None else round(sustained - start, 1),
        nfs_read_bytes_before_sustained=None if sustained is None else at(sustained) - first_read,
        nfs_read_bytes_total=rows[-1][1] - first_read,
        encode_seconds=round(batches[-1] - batches[0], 1),
        batches=len(batches),
    )


def profile(arm: Path):
    inclusive, self_frames, total = Counter(), Counter(), 0
    for line in (arm / "pyspy.raw.txt").read_text(errors="replace").splitlines():
        stack, _, count = line.rpartition(" ")
        if not stack or not count.isdigit():
            continue
        count = int(count)
        total += count
        frames = stack.split(";")
        seen = set()
        for frame in frames:
            for name in NAMED:
                if name.strip() in frame and name not in seen:
                    inclusive[name.strip()] += count
                    seen.add(name)
        self_frames[frames[-1]] += count
    return dict(samples=total, inclusive={k: inclusive.get(k.strip(), 0) for k in NAMED},
                top_self=self_frames.most_common(15))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = {}
    for arm in args.arms:
        start, end = host_window(arm)
        milestones, batches = log_milestones(arm, start)
        report[arm.name] = dict(row_wall_seconds=round(end - start, 1), milestones=milestones,
                                timeline=timeline(arm, start, batches), profile=profile(arm))
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
