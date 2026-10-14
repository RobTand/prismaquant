"""Gold records can be constructed and replayed without CLI arguments."""
import copy
import json

import pytest

from prismaquant.shipcard import build_shipcard, verify, write_shipcard
from prismaquant.shipcard_slots import make_gold_record, ship_slot


@pytest.fixture
def artifact(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    return tmp_path


@pytest.mark.parametrize("slot,metrics", [
    ("gold.kl", {"kl_mean": 0.1, "n_positions": 16, "score_positions": "all"}),
    ("gold.ppl", {"ppl": 4.2, "n_tokens_scored": 16}),
])
def test_direct_gold_fill_and_publication_replay_the_measurement(artifact, slot, metrics):
    card = build_shipcard(artifact)
    payload = {"model": str(artifact), "spec_decode_detected": False,
               "serve_fingerprint": "f" * 64, "git_commit": "a" * 40, **metrics}
    draft = make_gold_record(payload, slot=slot, model_sha=card["model_sha"],
                             record_path=artifact / "measurement.json", model_dir=artifact)
    assert draft.structurally_valid and not draft.problems
    card["slots"][slot] = draft.record
    assert verify(card, model_dir=artifact, required=[slot]) == []
    damaged = copy.deepcopy(draft.record)
    damaged["metrics"][next(iter(metrics))] = float("nan")
    card["slots"][slot] = damaged
    assert any("finite non-negative slot-specific metric" in problem
               for problem in verify(card, model_dir=artifact, required=[slot]))
    assert ship_slot(slot).preflight(damaged, card=card, model_dir=artifact)


def test_configured_gate_fills_gold_without_the_cli(artifact, monkeypatch):
    from prismaquant import shipcard_cli
    from prismaquant.ship_gates import _replay_slot

    def unexpected_cli(*args, **kwargs):
        pytest.fail("configured gold fill called argparse")

    monkeypatch.setattr(shipcard_cli, "main", unexpected_cli)
    card = build_shipcard(artifact)
    write_shipcard(artifact / "shipcard.json", card)
    payload = {"model": str(artifact), "spec_decode_detected": False,
               "serve_fingerprint": "f" * 64, "git_commit": "a" * 40,
               "kl_mean": 0.1, "n_positions": 16, "score_positions": "all"}
    output = artifact.parent / "gold.json"
    output.write_text(json.dumps(payload))
    record = _replay_slot({"id": "gold.kl"}, {"artifact": str(artifact)}, output, produced=True)
    assert record["passed"] is True
    assert record["metrics"]["kl_mean"] == 0.1
