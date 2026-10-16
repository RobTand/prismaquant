"""Gold records can be constructed and replayed without CLI arguments."""
import ast
import copy
import json
from pathlib import Path

import pytest

from prismaquant import shipcard, shipcard_slots
from prismaquant.shipcard import load_shipcard, open_shipcard, verify, write_shipcard
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
    card = open_shipcard(artifact)
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
    card = open_shipcard(artifact)
    write_shipcard(artifact / "shipcard.json", card)
    payload = {"model": str(artifact), "spec_decode_detected": False,
               "serve_fingerprint": "f" * 64, "git_commit": "a" * 40,
               "kl_mean": 0.1, "n_positions": 16, "score_positions": "all"}
    output = artifact.parent / "gold.json"
    output.write_text(json.dumps(payload))
    record = _replay_slot({"id": "gold.kl"}, {"artifact": str(artifact)}, output, produced=True)
    assert record["passed"] is True
    assert record["metrics"]["kl_mean"] == 0.1


def _gold_payload(artifact, **overrides):
    payload = {"model": str(artifact), "spec_decode_detected": False,
               "serve_fingerprint": "f" * 64, "git_commit": "a" * 40,
               "kl_mean": 0.1, "n_positions": 16, "score_positions": "all"}
    payload.update(overrides)
    return {key: value for key, value in payload.items() if value is not None}


@pytest.mark.parametrize("spec", [True, None, "false"])
def test_gold_constructor_refuses_a_number_not_measured_on_a_no_spec_serve(artifact, spec):
    card = open_shipcard(artifact)
    payload = _gold_payload(artifact, spec_decode_detected=spec)
    with pytest.raises(shipcard_slots.GoldSpecDecodeRefusal, match="DRAFT model"):
        make_gold_record(payload, slot="gold.kl", model_sha=card["model_sha"],
                         record_path=artifact / "m.json", model_dir=artifact)
    # An explicit acknowledgement records it; publication replay still refuses.
    draft = make_gold_record(payload, slot="gold.kl", model_sha=card["model_sha"],
                             record_path=artifact / "m.json", model_dir=artifact,
                             allow_spec_decode=True)
    card["slots"]["gold.kl"] = draft.record
    assert any("spec_decode_detected" in problem
               for problem in verify(card, model_dir=artifact, required=["gold.kl"]))


def test_configured_gate_refuses_spec_decode_through_the_slot_constructor(artifact):
    from prismaquant.ship_gates import _replay_slot

    write_shipcard(artifact / "shipcard.json", open_shipcard(artifact))
    output = artifact.parent / "gold.json"
    output.write_text(json.dumps(_gold_payload(artifact, spec_decode_detected=True)))
    with pytest.raises(shipcard_slots.GoldSpecDecodeRefusal, match="DRAFT model"):
        _replay_slot({"id": "gold.kl"}, {"artifact": str(artifact)}, output, produced=True)
    assert load_shipcard(artifact / "shipcard.json")["slots"]["gold.kl"] is None


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").rsplit(".", 1)[-1]
            yield node, [module] if node.module else [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            yield node, [alias.name.rsplit(".", 1)[-1] for alias in node.names]


def test_slot_owner_and_card_module_import_in_one_direction():
    package = Path(shipcard.__file__).resolve().parent
    slots_tree = ast.parse((package / "shipcard_slots.py").read_text(encoding="utf-8"))
    back = [ast.dump(node) for node, modules in _imports(slots_tree) if "shipcard" in modules]
    assert back == [], "shipcard_slots must not import the card module"
    card_tree = ast.parse((package / "shipcard.py").read_text(encoding="utf-8"))
    from_slots = [node for node, modules in _imports(card_tree) if "shipcard_slots" in modules]
    assert from_slots and all(node in card_tree.body for node in from_slots)
    assert [alias.name for node in from_slots for alias in node.names
            if alias.name.startswith("_")] == []


def test_every_core_slot_policy_lives_with_the_slot_owner():
    registry = shipcard_slots.ship_slot_registry()
    core = set(shipcard_slots.REQUIRED_SLOTS) | set(shipcard_slots.OPTIONAL_SLOTS) | set(
        shipcard_slots.CORE_LANE_SLOTS)
    assert {name: registry[name].policy.__module__ for name in core} == {
        name: "prismaquant.shipcard_slots" for name in core}


@pytest.fixture
def tessera_artifact(tmp_path):
    root = tmp_path / "exported"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return root


def test_lane_card_computes_the_model_identity_once(tessera_artifact, monkeypatch):
    from prismaquant.lane_shipcard import open_lane_shipcard

    real = shipcard.compute_model_sha
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(shipcard, "compute_model_sha", counted)
    card = load_shipcard(open_lane_shipcard(tessera_artifact, "tessera"))
    assert len(calls) == 1
    assert card["model_sha"] == real(tessera_artifact)
    assert card["build"]["export_container"] == "tessera"


@pytest.mark.parametrize("opener", ["lane", "card"])
def test_conflicting_lane_facts_refuse_before_any_artifact_write(tessera_artifact, opener):
    from prismaquant.lane_shipcard import open_lane_shipcard

    before = {path.name: path.read_bytes() for path in tessera_artifact.iterdir()}
    conflict = {"export_container": "compressed-tensors"}
    with pytest.raises(ValueError, match="artifact lane facts conflict"):
        if opener == "lane":
            open_lane_shipcard(tessera_artifact, "tessera", build=conflict)
        else:
            open_shipcard(tessera_artifact, build=conflict, lane="tessera")
    assert {path.name: path.read_bytes() for path in tessera_artifact.iterdir()} == before
