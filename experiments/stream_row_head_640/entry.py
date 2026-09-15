#!/usr/bin/env python3
"""Run the arm's campaign as the child of the static py-spy, inside the container.

yama ``ptrace_scope=1`` lets a parent trace its own child, so py-spy records
the campaign process at 100 Hz, nonblocking, in raw (collapsed-stack) form.
The campaign argv is ``<out>/argv.json`` from ``row_argv.py``; its output is
teed into ``<out>/campaign.log`` with a wall-clock stamp per line so the log
lines up with the host timeline. ``<out>/entry.json`` records the command, the
start and end times and both return codes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

PYSPY = "/mnt/shared/tessera-measurements/tessera486-fused-lut/torch213/bin/py-spy"


def main() -> int:
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    campaign = json.loads((out / "argv.json").read_text())
    child = ["python3", "-u", "-m", "prismaquant.tessera_campaign", *campaign]
    command = [PYSPY, "record", "--rate", "100", "--nonblocking", "--format", "raw",
               "--output", str(out / "pyspy.raw.txt"), "--", *child]
    record = {"command": command, "started_unix": time.time()}
    (out / "entry.json").write_text(json.dumps(record, indent=1) + "\n")
    with (out / "campaign.log").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)

        def pump():
            for line in process.stdout:
                stamped = f"{time.time():.3f} {line}"
                log.write(stamped)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        returncode = process.wait()
        reader.join()
    profile = out / "pyspy.raw.txt"
    record.update(finished_unix=time.time(), returncode=returncode,
                  profile_lines=(sum(1 for _ in profile.open()) if profile.exists() else 0))
    if returncode != 0 and record["profile_lines"] == 0 and "tessera_campaign" not in (
            out / "campaign.log").read_text():
        # py-spy could not start the child (no ptrace in this container), so
        # the campaign never ran; run it unprofiled and say so.
        record["fallback_unprofiled"] = True
        with (out / "campaign.log").open("a") as log:
            process = subprocess.Popen(child, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(f"{time.time():.3f} {line}")
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            returncode = process.wait()
        record.update(finished_unix=time.time(), returncode=returncode)
    (out / "entry.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"entry: returncode={returncode} profile_lines={record['profile_lines']}", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
