import argparse
import hashlib
import json
import sys
from pathlib import Path

EXIT_GAPS = 3

def _load_json(path: str, sha256: str | None = None):
    raw = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{path} does not hash to {sha256}")
    return json.loads(raw)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("stage-b", help="bound Stage B quantum records")
    b.add_argument("--records", required=True)
    b.add_argument("--quantum-id", action="append", default=[],
                   help="check only these quanta (repeatable)")
    a = sub.add_parser("stage-a", help="a Stage A chain walk")
    a.add_argument("--manifest", required=True)
    a.add_argument("--manifest-sha256", default=None)
    a.add_argument("--plan", required=True)
    a.add_argument("--plan-sha256", default=None)
    a.add_argument("--prepared", required=True)
    a.add_argument("--prepared-sha256", default=None)
    a.add_argument("--top", type=int, required=True)
    a.add_argument("--bottom", type=int, required=True)
    a.add_argument("--prefetch-override", default=None)
    a.add_argument("--phase-format", default="chain-{layer:03d}")
    a.add_argument("--schedule", choices=("stage-a", "walk"), default="stage-a")
    parser.add_argument("--out", default=None, help="also write the report here")
    args = parser.parse_args(argv)
    try:
        report = _stage_b(args) if args.mode == "stage-b" else _stage_a(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"check_readset_coverage: refused: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out is not None:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if report["covered"] else EXIT_GAPS
