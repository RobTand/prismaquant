"""Artifact byte and lane obligations cannot disappear from old records."""
import json
import os

import pytest

pytestmark = pytest.mark.usefixtures("synthetic_graph_lane")
import torch
from safetensors.torch import save_file

from prismaquant.shipcard import open_shipcard, compute_model_sha, required_slots, verify, write_shipcard
from test_shipcard import _fill_all
from tools.publish_artifact import check_shipcard


@pytest.fixture
def artifact(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    save_file({"linear.weight": torch.ones((2, 4)),
               "linear.input_global_scale": torch.tensor([1.0])}, str(root / "model.safetensors"))
    return root


def complete_card(root):
    card = open_shipcard(root)
    path = root / "shipcard.json"
    write_shipcard(path, card)
    _fill_all(path, card["model_sha"])
    from prismaquant.shipcard import load_shipcard
    return path, load_shipcard(path)


def test_same_size_native_weight_edit_refuses_verification_and_publication(artifact):
    path, card = complete_card(artifact)
    assert verify(card, model_dir=artifact) == []
    weight = artifact / "model.safetensors"
    before = weight.stat()
    raw = bytearray(weight.read_bytes())
    raw[-1] ^= 1
    weight.write_bytes(raw)
    os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert weight.stat().st_size == before.st_size
    assert any("artifact changed" in problem for problem in verify(card, model_dir=artifact))
    _, problems = check_shipcard(artifact, path)
    assert problems


@pytest.mark.parametrize("method,expected", [
    ("compressed-tensors", {"route.sweep"}),
    ("tessera", {"route.census", "route.trace", "uniform_control"}),
])
def test_artifact_rederives_lane_slots_after_card_lane_and_build_erasure(artifact, method, expected):
    config = {"model_type": "fixture", "quantization_config": {"quant_method": method}}
    (artifact / "config.json").write_text(json.dumps(config))
    card = open_shipcard(artifact, lane=method, build={"export_container": method})
    card.pop("lane", None)
    card["build"] = {}
    assert expected <= set(required_slots(card, model_dir=artifact))


def test_conflicting_card_and_artifact_lane_facts_refuse(artifact):
    (artifact / "config.json").write_text(json.dumps({
        "model_type": "fixture", "quantization_config": {"quant_method": "tessera"}}))
    card = {"lane": "compressed-tensors", "build": {}, "slots": {}}
    assert any("lane" in problem and "conflict" in problem
               for problem in verify(card, model_dir=artifact, required=[]))


def test_size_only_historical_card_does_not_preserve_unproved_measurements(artifact):
    old_sha = compute_model_sha(artifact)
    path, card = complete_card(artifact)
    manifest = artifact / "weight_content_manifest.json"
    if manifest.exists():
        manifest.unlink()
    card.pop("weight_stat_attestation", None)
    card["model_sha"] = old_sha
    for record in card["slots"].values():
        record["model_sha"] = old_sha
    write_shipcard(path, card)
    assert any("content manifest" in problem for problem in verify(card, model_dir=artifact))


def test_native_export_persists_the_exact_weight_digest(artifact):
    card = open_shipcard(artifact)
    manifest_path = artifact / "weight_content_manifest.json"
    assert manifest_path.is_file()
    from prismaquant.digests import file_sha256hex
    manifest = json.loads(manifest_path.read_text())
    assert manifest["files"]["model.safetensors"]["sha256"] == file_sha256hex(artifact / "model.safetensors")
    assert "weight_stat_attestation" in card


def test_read_only_constructor_does_not_open_an_artifact(artifact):
    from prismaquant.shipcard import build_shipcard
    before = {path.name: path.read_bytes() for path in artifact.iterdir()}
    build_shipcard(artifact)
    assert {path.name: path.read_bytes() for path in artifact.iterdir()} == before


def test_explicit_historical_reattest_leaves_old_measurements_stale(artifact):
    from prismaquant.shipcard import load_shipcard, reattest_weight_stats
    old_sha = compute_model_sha(artifact)
    path, card = complete_card(artifact)
    (artifact / "weight_content_manifest.json").unlink()
    card.pop("weight_stat_attestation")
    card["model_sha"] = old_sha
    for record in card["slots"].values():
        record["model_sha"] = old_sha
    write_shipcard(path, card)
    upgraded = reattest_weight_stats(path, artifact)
    assert upgraded["model_sha"] != old_sha
    assert upgraded["weight_identity_upgrade"]["measurements_require_replay"] is True
    assert {row["model_sha"] for row in upgraded["slots"].values()} == {old_sha}
    problems = verify(load_shipcard(path), model_dir=artifact)
    assert problems
    assert all("model_sha" in problem for problem in problems)


def test_lane_plugin_cannot_replace_a_base_slot_owner(monkeypatch):
    from prismaquant import lane_spec
    from prismaquant.shipcard_slots import ship_slot_registry
    monkeypatch.setattr(lane_spec, "lane_hooks", lambda name: (
        ("fixture", lambda: {"gold.kl": lambda *args, **kwargs: []}),))
    with pytest.raises(KeyError, match="gold.kl.*two"):
        ship_slot_registry()
