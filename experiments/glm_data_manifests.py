#!/usr/bin/env python3
"""Publish input readsets for a campaign's manifest rows."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prismaquant.campaign_data_manifests import CAMPAIGN_BASE, publish_row_manifests

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    base = CAMPAIGN_BASE + "/transition-optimization-20260910"
    ap.add_argument("--workspace",
                    default=CAMPAIGN_BASE + "/first-proof-anchor-preparation-05/workspace")
    ap.add_argument("--campaign-manifest", default=base + "/resume-overlap-manifest.json")
    ap.add_argument("--out-dir", default=base + "/data-manifests")
    ap.add_argument("--out-manifest", default=base + "/resume-overlap-manifest.with-data.json")
    ap.add_argument("--sizes-cache",
                    default="/home/rob/tmp/glm-perf-20260910/w3/evidence/capture-file-sizes.txt")
    ap.add_argument("--stat", action="store_true",
                    help="stat every capture file instead of using the size cache")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and summarise; write nothing")
    ap.add_argument("--json", help="write the run summary here")
    args = ap.parse_args()

    summary = publish_row_manifests(args.campaign_manifest, args.workspace,
        out_dir=args.out_dir, out_manifest=args.out_manifest,
        sizes_cache=args.sizes_cache, stat=args.stat, dry_run=args.dry_run)
    print(json.dumps(summary, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1)
    sys.stderr.write(
        f"\n{summary['rows']} rows, {summary['total_bytes'] / 1e9:.2f} GB planned\n"
        f"source   {summary['source_manifest_sha256']}  {summary['source_manifest']}\n"
        f"with-data {summary['out_manifest_sha256']}  {args.out_manifest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
