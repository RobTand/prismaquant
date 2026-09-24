"""Per-sample distance between two sealed Stage A checkpoint planes (PQ #997).

Reads two pinned ``checkpoint.json`` files at the same boundary, for example
two seed runs of one checkpoint under different chain batch sizes or matmul
reduction settings, and writes
``prismaquant.stage_a.checkpoint_plane_distance.v1``: for every
``(probe, batch)`` cotangent, whether the payloads are bitwise equal, the
relative L2 distance of the candidate from the reference, and the largest
elementwise difference, with a summary. Every entry is read digest-verified.

Run it as a PrismaBuild action, for example::

    python3 tools/compare_stage_a_checkpoints.py \\
        --reference PATH --reference-sha256 SHA \\
        --candidate PATH --candidate-sha256 SHA --output OUT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cost_stage_checkpoint import publish_new_bytes  # noqa: E402
from prismaquant.stage_a_chain_seed import checkpoint_plane_distance  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="the distance record; refused if it exists")
    parser.add_argument("--device", default=None,
                        help="where the differences are computed (default: cuda "
                             "when available, else cpu)")
    parser.add_argument("--read-ahead", type=int, default=8,
                        help="entries of each plane held at once: the window being "
                             "compared and the one read ahead")
    args = parser.parse_args(argv)
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    result = checkpoint_plane_distance(
        {"path": args.reference, "sha256": args.reference_sha256},
        {"path": args.candidate, "sha256": args.candidate_sha256},
        device=device, read_ahead=args.read_ahead)
    result["device"] = device
    if not publish_new_bytes(args.output, (json.dumps(result, indent=2, sort_keys=True)
                                           + "\n").encode()):
        parser.error(f"{args.output} already exists; a distance record is never replaced")
    summary = {key: result[key] for key in ("boundary", "equal", "different",
                                            "relative_l2", "max_abs")}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
