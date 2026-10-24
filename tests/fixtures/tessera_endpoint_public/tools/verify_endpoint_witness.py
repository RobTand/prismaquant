#!/usr/bin/env python3
"""Verify explicit consumer facts and artifact bytes, offline or at a live listener."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tessera.endpoint_witness import (  # noqa: E402
    canonical, check_expectations, verify_recorded_witness, verify_witness,
)


class VerdictParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = VerdictParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("witness")
    parser.add_argument("--served-dir", required=True)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--offline", action="store_true",
                        help="Prove the recorded launch and bytes, not the current endpoint.")
    parser.add_argument("--expect-endpoint", required=True)
    parser.add_argument("--expect-alias", required=True)
    parser.add_argument("--expect-artifacts", required=True,
                        help="JSON file: complete source filenames with sha256 and bytes.")
    parser.add_argument("--expect-tokenizer", required=True,
                        help="JSON file: files with sha256 and bytes, backend, vocab, and special_ids.")
    parser.add_argument("--expect-attempt", required=True)
    parser.add_argument("--expect-ranks", required=True, help="Complete ordered ranks, for example 0,1.")
    argv = list(sys.argv[1:] if argv is None else argv)
    verdict = {"schema": "tessera.endpoint_witness_verdict.v1",
               "mode": "offline" if "--offline" in argv else "live",
               "verdict": "refused", "reason": None, "proof_scope": None,
               "current_endpoint_verified": False}
    try:
        args = parser.parse_args(argv)
        receipt = json.loads(Path(args.witness).read_bytes())
        try:
            ranks = [int(part) for part in args.expect_ranks.split(",")]
        except ValueError as exc:
            raise ValueError("expected ranks must be comma-separated integers") from exc
        expected = {
            "endpoint": args.expect_endpoint, "served_alias": args.expect_alias,
            "artifacts": json.loads(Path(args.expect_artifacts).read_bytes()),
            "tokenizer": json.loads(Path(args.expect_tokenizer).read_bytes()),
            "attempt_id": args.expect_attempt, "ranks": ranks,
        }
        if args.offline:
            reason = verify_recorded_witness(receipt, expected=expected, served_dir=args.served_dir,
                                            tokenizer_dir=args.tokenizer_dir)
        else:
            check_expectations(receipt, expected)
            from tessera.endpoint_observer import fetch_witness

            live = fetch_witness(args.expect_endpoint)["receipt"]
            reason = verify_witness(receipt, ranks=ranks, served_dir=args.served_dir,
                                    tokenizer_dir=args.tokenizer_dir, live=live)
        if reason is not None:
            raise ValueError(reason)
    except (ValueError, KeyError, TypeError, AttributeError, OSError, OverflowError, IndexError) as exc:
        verdict["reason"] = str(exc)
        print(canonical(verdict))
        return 4
    verdict.update(verdict="valid",
                   proof_scope="recorded_runtime_byte_binding" if args.offline else "current_runtime_byte_binding",
                   current_endpoint_verified=not args.offline)
    print(canonical(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
