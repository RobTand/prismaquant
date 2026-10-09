"""The research quantum must consume the split its persisted receipt describes."""
from pathlib import Path
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


def test_prepared_multi_layer_range_drains_each_layers_moments(tmp_path, monkeypatch):
    finished = []
    real_moments = capture.DisjointRowMoments

    class ObservedMoments(real_moments):
        def finish(self):
            records = super().finish()
            layers = {int(capture.chain.DOTTED_LAYER_QNAME.search(name).group(1))
                      for role in records.values() for name in role}
            finished.append(layers)
            return records

    monkeypatch.setattr(capture, "DisjointRowMoments", ObservedMoments)
    capture._toy_control_preflight(tmp_path, lambda _label: None, layers=2, quantum_layers=2)
    assert finished == [{0}, {1}]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA; certifies nothing when skipped")
def test_prepared_multi_layer_range_forwards_and_accumulates_on_the_cuda_device(tmp_path, monkeypatch):
    """The CUDA control: a silent CPU fallback fails here, and the roles still publish as CPU float32."""
    built, accumulated = [], set()
    real_build, real_consume = capture.build_streamed_causal_lm, capture.DisjointRowMoments.consume

    def build(*args, **kwargs):
        built.append(kwargs["device"].type)
        return real_build(*args, **kwargs)

    def consume(self, names, flat):
        accumulated.add(flat.device.type)
        return real_consume(self, names, flat)

    monkeypatch.setattr(capture, "build_streamed_causal_lm", build)
    monkeypatch.setattr(capture.DisjointRowMoments, "consume", consume)
    capture._toy_control_preflight(tmp_path, lambda _label: None, layers=2, quantum_layers=2, device="cuda")
    assert built == ["cuda"]
    assert accumulated == {"cuda"}
    for layer in range(2):
        manifest = json.loads((tmp_path / f"capture/layers/L{layer:03d}/manifest.json").read_text())
        assert manifest["units"]
        for roles in manifest["units"].values():
            for role in ("fit", "heldout"):
                payload = torch.load(tmp_path / "capture" / roles[role]["file"], weights_only=True)
                assert payload["hessian"].device.type == "cpu"
                assert payload["hessian"].dtype == torch.float32


def test_toy_control_names_its_device_or_refuses(tmp_path, monkeypatch):
    with pytest.raises(capture.ResearchRefused, match="cpu or cuda"):
        capture._toy_control_preflight(tmp_path, lambda _label: None, device="mps")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(capture.ResearchRefused, match="sees none"):
        capture._toy_control_preflight(tmp_path, lambda _label: None, device="cuda")


def _resume_args(directory):
    return SimpleNamespace(
        capture_root=str(directory / "capture"), calibration_census=str(directory / "census.json"),
        units=str(directory / "units.json"), calibration_tokens=str(directory / "tokens.safetensors"),
        corpus_text=str(directory / "corpus.txt"), model=str(directory / "source"),
        source_snapshot_root=None, total_samples=4, fit_stop=2, max_prefix_rows=4,
        max_act_rows=7, attention_implementation="eager", device="cpu",
        capture_layer_range="0:1", cache_dir=str(directory / "replay-cache"),
        streaming_cache_slots=2, streaming_prefetch_workers=1, streaming_cache_headroom_gb=0.0)


@pytest.mark.parametrize("change", ["selection", "missing_selection", "scoring_prefix"])
def test_completed_adoption_rechecks_requested_prep_comparability(tmp_path, monkeypatch, capsys, change):
    """A request that is another measurement than the prep sealed refuses; it never replays."""
    capture._toy_control_preflight(tmp_path, lambda _label: None)
    args = _resume_args(tmp_path)
    if change == "selection":
        units = json.loads((tmp_path / "units.json").read_text())
        units["groups"] = units["groups"][:1]
        path = tmp_path / "different-units.json"
        path.write_text(json.dumps(units))
        args.units = str(path)
    elif change == "missing_selection":
        args.units = str(tmp_path / "missing-units.json")
    else:
        args.max_act_rows = 8
    monkeypatch.setattr(capture, "build_streamed_causal_lm",
                        lambda *_args, **_kwargs: pytest.fail("a refused request reached a replay"))
    capsys.readouterr()
    with pytest.raises((capture.ResearchRefused, capture.chain.CaptureChainRefused, FileNotFoundError)):
        capture.mode_quantum(args, lambda _label: None)
    assert "research_quantum_replay" not in capsys.readouterr().out


def test_completed_matching_binding_adopts_the_stored_result_without_a_forward(tmp_path, monkeypatch, capsys):
    capture._toy_control_preflight(tmp_path, lambda _label: None)
    args = _resume_args(tmp_path)
    fragment = tmp_path / "capture/chain/capture-000-001.fragment.json"
    stored = fragment.read_bytes()
    monkeypatch.setattr(capture, "build_streamed_causal_lm",
                        lambda *_args, **_kwargs: pytest.fail("a matching completed quantum forwarded"))
    capsys.readouterr()
    summary = capture.mode_quantum(args, lambda _label: None)
    assert [Path(path).resolve() for path in summary["fragments"]] == [fragment.resolve()]
    assert fragment.read_bytes() == stored
    assert "research_quantum_replay" not in capsys.readouterr().out


_BINDING = {"selection_sha256": "a", "units": ["u"], "max_act_rows": 7, "max_prefix_rows": 4}


@pytest.mark.parametrize("stored,changed", [
    (dict(_BINDING), []),
    (None, ["capture_binding"]),
    ("not a binding", ["capture_binding"]),
    ({**_BINDING, "selection_sha256": "b"}, ["selection_sha256"]),
    ({**_BINDING, "max_prefix_rows": 8}, ["max_prefix_rows"]),
    ({key: value for key, value in _BINDING.items() if key != "max_prefix_rows"}, ["max_prefix_rows"]),
    ({**_BINDING, "extra": None}, ["extra"]),
])
def test_changed_binding_fields_names_each_difference(stored, changed):
    assert capture._changed_binding_fields(stored, dict(_BINDING)) == changed


@pytest.mark.parametrize("change,field", [("selection_hash", "selection_sha256"),
                                          ("retained_prefix", "max_prefix_rows")])
def test_completed_binding_mismatch_replays_instead_of_reusing(tmp_path, monkeypatch, capsys, change, field):
    """A binding mismatch is another request over the same traversal: it replays and never refuses."""
    capture._toy_control_preflight(tmp_path, lambda _label: None)
    args = _resume_args(tmp_path)
    if change == "selection_hash":
        path = tmp_path / "units.json"
        path.write_text(path.read_text() + "\n")
    else:
        args.max_prefix_rows = 8
    real_build = capture.build_streamed_causal_lm
    replayed = []

    def build(*a, **kw):
        replayed.append(True)
        return real_build(*a, **kw)

    monkeypatch.setattr(capture, "build_streamed_causal_lm", build)
    from test_glm5_next_streamed_forward_parity import _torch_only_causal_conv1d
    kernels = pytest.MonkeyPatch()
    capsys.readouterr()
    try:
        _torch_only_causal_conv1d.__wrapped__(kernels)
        capture.mode_quantum(args, lambda _label: None)
    finally:
        kernels.undo()
    assert replayed
    announced = [json.loads(line)["research_quantum_replay"]
                 for line in capsys.readouterr().out.splitlines()
                 if line.startswith('{"research_quantum_replay"')]
    assert announced == [{"layers": [0, 1], "changed": [field]}]
    fragment = json.loads((tmp_path / "capture/chain/capture-000-001.fragment.json").read_text())
    assert fragment["capture_binding"]["selection_sha256"] == capture._sha256_file(args.units)
    assert fragment["capture_binding"]["max_prefix_rows"] == args.max_prefix_rows


def _published_toy_layer_with_moved_census(tmp_path):
    """A published toy layer, and its census statistics moved one row and 1.0 off the observed forward."""
    capture._toy_control_preflight(tmp_path, lambda _label: None)
    root = tmp_path / "capture"
    census_path = tmp_path / "census.json"
    census = json.loads(census_path.read_text())
    manifest = json.loads((root / "layers/L000/manifest.json").read_text())
    records = {role: {} for role in ("fit", "heldout")}
    for name, roles in manifest["units"].items():
        census["counts"][name] += 1
        census["max_abs"][name] += 1.0
        for role in records:
            records[role][name] = torch.load(root / roles[role]["file"], weights_only=True)
    return root, census_path, census, manifest, records


def test_publication_sizes_roles_from_observed_forward_not_planning_census(tmp_path, monkeypatch):
    """Dev mode, the entry's default: a moved census statistic is stamped and never sizes a role (D32)."""
    root, census_path, census, manifest, records = _published_toy_layer_with_moved_census(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    observed = manifest["full_counts"]
    verified = capture.persist_split_records(
        root, census, census_path, records, observed, manifest["split_sha256"], lambda _label: None)
    manifest = json.loads((root / "layers/L000/manifest.json").read_text())
    for name, row in verified.items():
        assert row["observed_rows"] == observed[name]
        assert row["census_count"] == observed[name] + 1
        assert manifest["full_counts"][name] == observed[name]
        assert manifest["census_comparison"][name]["count"]["delta"] == -1
        assert manifest["census_comparison"][name]["stamp"] == "[DEV-MODE]"
    capture._verified_layer_publication(root, root / "layers/L000/manifest.json",
                                        manifest, verified, census, manifest["split_sha256"])


def test_certified_mode_refuses_a_moved_census_statistic_at_publication(tmp_path, monkeypatch):
    """Certified mode keeps the refusal that dev mode turns into a stamp (D32)."""
    root, census_path, census, manifest, records = _published_toy_layer_with_moved_census(tmp_path)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    with pytest.raises(capture.ResearchRefused, match="research_census_statistics differs"):
        capture.persist_split_records(
            root, census, census_path, records, manifest["full_counts"], manifest["split_sha256"],
            lambda _label: None)
