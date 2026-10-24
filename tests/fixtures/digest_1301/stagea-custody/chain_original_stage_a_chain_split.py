# Exact original AST-selected source; boundaries in ledger_chains.json.
# Only exercised consumer functions and required dependencies are retained.

# Original stage_a_chain_split.py:35-35
from __future__ import annotations

# Original stage_a_chain_split.py:37-37
import argparse

# Original stage_a_chain_split.py:39-39
import json

# Original stage_a_chain_split.py:41-41
from pathlib import Path

# Original stage_a_chain_split.py:43-43
import sys

# Original stage_a_chain_split.py:45-45
from .cost_stage_checkpoint import atomic_write_bytes

# Original stage_a_chain_split.py:59-60
class ChainSplitRefused(RuntimeError):
    """The split cannot run or join without losing or corrupting bytes."""

# Original stage_a_chain_split.py:512-537
def main(argv=None) -> int:
    """Join one boundary's partials (CPU only, one PrismaBuild row)."""
    from .joint_adjoint_checkpoints import adjoint_space
    from .stage_a_chain_resume import ChainResumeRefused, read_chain_state

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--receipt", default=None,
                        help="also write the join's receipt to this path")
    args = parser.parse_args(argv)
    space = adjoint_space(args.output_root)
    try:
        state = read_chain_state(space)
        record = join_split_checkpoint(
            space, args.boundary, n_probes=int(state["run_identity"]["n_probes"]),
            n_batches=int(state["n_batches"]))
    except (ChainSplitRefused, ChainResumeRefused) as exc:
        print(f"stage_a_chain_split: join refused: {exc}", file=sys.stderr, flush=True)
        return 2
    receipt = join_receipt(space, record)
    if args.receipt is not None:
        atomic_write_bytes(Path(args.receipt), (json.dumps(
            receipt, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0
