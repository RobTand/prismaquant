"""Bite demo for #631 — the same bytes run on the base commit and on the branch.

This file imports only APIs that exist BEFORE the change, so it is a fair
before/after: on `origin/main` all three assertions fail, and on the branch all
three pass. It is not part of the suite's coverage; the coverage is in
`test_compressed_route_sweep_gate.py`.

Three claims, one per acceptance criterion:

1. the compressed-tensors lane OWES a served-route slot,
2. a card opened on that lane does not close it by itself, and
3. the fill command REFUSES (exit 1) a sweep whose served contract disagrees
   with the artifact's own price.
"""
from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "compressed_route_sweep_0p6b"
SLOT = "route.sweep"


def _exit_code(argv: list[str]) -> int:
    """Run the shipcard CLI, turning a SystemExit into its code."""
    from prismaquant.shipcard_cli import main

    try:
        return int(main(argv))
    except SystemExit as exc:  # argparse on a subcommand that does not exist
        return int(exc.code or 0)


def test_the_compressed_tensors_lane_owes_a_served_route_slot():
    from prismaquant.shipcard import lane_gate_slots

    assert SLOT in lane_gate_slots("compressed-tensors")


def test_a_fresh_compressed_tensors_card_has_not_closed_it(tmp_path):
    from prismaquant.lane_shipcard import open_lane_shipcard
    from prismaquant.shipcard import required_slots, unfilled_slots

    model_dir = tmp_path / "artifact"
    model_dir.mkdir()
    (model_dir / "config.json").write_text((FIXTURE / "config.json").read_text())
    path = open_lane_shipcard(
        model_dir, "compressed-tensors", shipcard_path=tmp_path / "card.json")
    card = json.loads(path.read_text())
    assert SLOT in required_slots(card, model_dir=model_dir)
    assert SLOT in unfilled_slots(card)


def test_a_disagreeing_served_contract_is_refused(tmp_path):
    from prismaquant.lane_shipcard import open_lane_shipcard

    model_dir = tmp_path / "artifact"
    model_dir.mkdir()
    (model_dir / "config.json").write_text((FIXTURE / "config.json").read_text())
    card = open_lane_shipcard(
        model_dir, "compressed-tensors", shipcard_path=tmp_path / "card.json")

    sweep = json.loads((FIXTURE / "sweep_rank0.json").read_text())
    for row in sweep["ranks"][0]["modules"]:
        if row.get("scheme"):
            # The serve stops quantizing this module's activations. Nothing
            # else moves: same artifact, same price, same 112 modules.
            row["scheme_attrs"]["use_a16"] = "True"
            break
    mutated = tmp_path / "sweep_mutated.json"
    mutated.write_text(json.dumps(sweep))

    assert _exit_code([
        "fill-route-sweep", str(card), "--sweep", str(mutated),
        "--expected-ranks", "1", "--model-dir", str(model_dir)]) == 1
    assert json.loads(Path(card).read_text())["slots"].get(SLOT) is None
