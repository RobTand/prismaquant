"""Submit the G3 harness with PB's phased residency and resource admission."""
import argparse
from pathlib import Path
import subprocess
import sys

from g3_progress_launch import select_arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--mem-gb", type=int, required=True)
    parser.add_argument("--gpu-memory-gb", type=int, required=True)
    parser.add_argument("--quiet-s", type=int, default=900)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("launch_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    launch = args.launch_args[1:] if args.launch_args[:1] == ["--"] else args.launch_args
    arms = select_arms(launch)
    phases = ["setup", *(f"layer-{i:02d}" for i in range(45)), "teachers"] if arms else ["smoke"]
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py",
               "--cwd", str(root), "--tag", "gb10", "--gpu", "--priority", "0",
               "--cpus", str(args.cpus), "--demand", f"mem_gb={args.mem_gb}",
               "--gpu-memory-gb", str(args.gpu_memory_gb),
               "--data-manifest", args.data_manifest, "--residency", "stage",
               "--residency-share", "auto", "--residency-ram", "auto",
               "--env", "TMPDIR=/tmp"]
    for phase in phases:
        command.extend(["--progress-phase", f"{phase}={args.quiet_s}"])
    if args.detach:
        command.append("--detach")
    command.extend(["--", "python3", "tools/g3job/g3_progress_launch.py", *launch])
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
