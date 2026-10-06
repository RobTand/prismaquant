"""The research quantum must consume the split its persisted receipt describes."""
from types import SimpleNamespace
import json

import pytest
import torch

from experiments import indomain_split_capture as capture


def test_matching_split_publishes_disjoint_role_coordinates(tmp_path):
    capture._toy_control_preflight(tmp_path, lambda _label: None)
    root = tmp_path / "capture"
    manifest = json.loads((root / "layers/L000/manifest.json").read_text())
    for name, roles in manifest["units"].items():
        assert roles["fit"]["count"] + roles["heldout"]["count"] == manifest["full_counts"][name]
        for role, (lo, hi) in (("fit", (0, 2)), ("heldout", (2, 4))):
            record = torch.load(root / roles[role]["file"], weights_only=True)
            ids = record["prefix_sample_ids"]
            assert torch.all((ids >= lo) & (ids < hi))
            assert len(ids) == record["inputs"].shape[0]


def test_quantum_refuses_a_different_fit_boundary_before_model_load(tmp_path, monkeypatch):
    capture._toy_control_preflight(tmp_path, lambda _label: None)
    args = SimpleNamespace(
        capture_root=str(tmp_path / "capture"), calibration_census=str(tmp_path / "census.json"),
        units=str(tmp_path / "units.json"), calibration_tokens=str(tmp_path / "tokens.safetensors"),
        corpus_text=str(tmp_path / "corpus.txt"), model=str(tmp_path / "source"),
        source_snapshot_root=None, total_samples=4, fit_stop=1, max_prefix_rows=4,
        max_act_rows=7, attention_implementation="eager", device="cpu",
        capture_layer_range="0:1", cache_dir=str(tmp_path / "unexpected-cache"),
        streaming_cache_slots=2, streaming_prefetch_workers=1, streaming_cache_headroom_gb=0.0)
    monkeypatch.setattr(capture, "build_streamed_causal_lm",
                        lambda *_args, **_kwargs: pytest.fail("mislabeled split reached model load"))
    with pytest.raises(capture.ResearchRefused, match="fit_stop"):
        capture.mode_quantum(args, lambda _label: None)


@pytest.mark.parametrize("change", ["counts", "hessian_shape", "prefix_shape", "payload_digest"])
def test_join_refuses_role_metadata_not_verified_by_its_quantum(tmp_path, change):
    capture._toy_control_preflight(tmp_path, lambda _label: None)
    root = tmp_path / "capture"
    path = root / "layers/L000/manifest.json"
    manifest = json.loads(path.read_text())
    name = sorted(manifest["units"])[0]
    roles = manifest["units"][name]
    if change == "counts":
        roles["fit"]["count"] += 1
        roles["heldout"]["count"] -= 1
    elif change == "hessian_shape":
        roles["fit"]["hessian_shape"][0] += 1
    elif change == "prefix_shape":
        roles["fit"]["prefix_sample_ids_shape"][0] += 1
    else:
        payload_path = root / roles["fit"]["file"]
        payload = torch.load(payload_path, weights_only=True)
        payload["hessian"][0, 0] += 1
        torch.save(payload, payload_path)
        roles["fit"]["sha256"] = capture._sha256_file(payload_path)
        roles["fit"]["bytes"] = payload_path.stat().st_size
    path.write_text(json.dumps(manifest))
    args = SimpleNamespace(
        capture_root=str(root), calibration_census=str(tmp_path / "census.json"),
        model=str(tmp_path / "source"), source_snapshot_root=None, max_act_rows=7,
        attention_implementation="eager", units=str(tmp_path / "units.json"))
    with pytest.raises(capture.ResearchRefused, match="verified"):
        capture.mode_join(args, lambda _label: None)


def test_one_action_completes_two_prepared_layers_with_disjoint_roles(tmp_path):
    capture._toy_control_preflight(tmp_path, lambda _label: None, layers=2)
    root = tmp_path / "capture"
    census = json.loads((tmp_path / "census.json").read_text())
    covered = set()
    for layer in range(2):
        manifest = json.loads((root / f"layers/L{layer:03d}/manifest.json").read_text())
        fragment = json.loads((root / f"chain/capture-{layer:03d}-{layer+1:03d}.fragment.json").read_text())
        for name, roles in manifest["units"].items():
            covered.add(name)
            assert roles == {role: fragment["units"][name][role] for role in ("fit", "heldout")}
            assert sum(roles[role]["count"] for role in ("fit", "heldout")) == census["counts"][name]
            for role, (lo, hi) in (("fit", (0, 2)), ("heldout", (2, 4))):
                payload = torch.load(root / roles[role]["file"], weights_only=True)
                ids = payload["prefix_sample_ids"]
                assert torch.all((ids >= lo) & (ids < hi))
    assert covered == set(census["counts"])
