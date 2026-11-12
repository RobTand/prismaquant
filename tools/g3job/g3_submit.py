"""Submit the G3 harness as one native Flux job over an explicit staged-input binding."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from prismaquant import flux_jobs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="reconnect to the saved native ID and never submit")
    parser.add_argument("--staged-launch", type=Path, help="a prismaquant.staged_launch.v1 binding file")
    parser.add_argument("--read-plan")
    parser.add_argument("--read-plan-sha256")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--mem-gb", type=int, default=1)
    parser.add_argument("--gpu-memory-gb", type=int, default=0)
    parser.add_argument("--walltime-s", type=int, default=1800)
    parser.add_argument("launch_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.resume:
        print(flux_jobs.load_submission(args.record_dir).job_id)
        return 0
    if not (args.staged_launch and args.read_plan and args.read_plan_sha256):
        parser.error("a submission requires --staged-launch, --read-plan and --read-plan-sha256")
    launch = args.launch_args[1:] if args.launch_args[:1] == ["--"] else args.launch_args
    try:
        resources = flux_jobs.FluxResources(cores=args.cpus, memory_gib=args.mem_gb,
                                            gpus=1 if args.gpu_memory_gb else 0,
                                            walltime_s=args.walltime_s)
    except ValueError as exc:
        print(f"G3 submission refused at the native profile: {exc}", file=sys.stderr)
        return 2
    command = ["/usr/bin/python3", "-s", "tools/g3job/g3_progress_launch.py", *launch,
               "--read-plan", args.read_plan, "--read-plan-sha256", args.read_plan_sha256]
    try:
        submission = flux_jobs.submit_job(command, checkout=ROOT, resources=resources,
                                          record_dir=args.record_dir,
                                          staged_input=json.loads(args.staged_launch.read_bytes()))
    except flux_jobs.SubmissionUnknown as exc:
        print(f"{exc}", file=sys.stderr)
        return 3
    print(submission.job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
