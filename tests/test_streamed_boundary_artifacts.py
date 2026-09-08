"""Exact source boundary residency and rolling cotangent regressions."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import weakref

import pytest
import torch

import prismaquant.aura_cost as aura
from prismaquant.cost_streaming import (
    BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts, normalize_boundary_storage,
)
from prismaquant.perturbed_x_cache import ExactActivationReference
from test_joint_aura_streamed import _fixture
from test_streamed_cost_checkpoints import _model_identity


def _policy(path, *, window=2, cap=1280, aux=1 << 20, disk=1 << 24):
    return {"schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(path),
            "max_resident_bytes": cap, "max_auxiliary_bytes": aux,
            "max_artifact_bytes": disk, "prefetch_batches": window}


def _bound(path, **kwargs):
    owner = StreamedBoundaryArtifacts(_policy(path, **kwargs))
    owner.bind({"fixture": "exact-source"}, n_probes=4)
    return owner


def _run(path, *, window=2, checkpoint=None, resume=False, owner_events=None):
    _, context, runner, cache = _fixture()
    source_events = []
    original_call = runner._call
    def call(layer, hidden, *, batch, pass_state):
        source_events.append((int(batch.input_ids[0, 0]), layer, torch.is_grad_enabled()))
        return original_call(layer, hidden, batch=batch, pass_state=pass_state)
    runner._call = call
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 4, 1],
                        [3, 4, 1, 2], [1, 3, 2, 4]])
    if owner_events is not None:
        tail_call = runner.tail_logits
        tail_calls = 0
        def tail(batch, hidden):
            nonlocal tail_calls
            batch_index, probe_index = divmod(tail_calls, 4)
            hidden.register_hook(lambda gradient, b=batch_index, k=probe_index:
                owner_events.append((k, b, runner.num_layers, gradient.detach().clone())))
            tail_calls += 1
            return tail_call(batch, hidden)
        runner.tail_logits = tail
        reverse_call = runner.isolated_layer
        reverse_calls = 0
        def reverse(batch, layer, hidden, *, pass_state):
            nonlocal reverse_calls
            probe_index, batch_index = divmod(reverse_calls % 20, 5)
            hidden.register_hook(lambda gradient, b=batch_index, k=probe_index, depth=layer:
                owner_events.append((k, b, depth, gradient.detach().clone())))
            reverse_calls += 1
            return reverse_call(batch, layer, hidden, pass_state=pass_state)
        runner.isolated_layer = reverse
    payload = aura.compute_aura_cost_streamed(
        runner, ids, ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=4,
        probe_microbatch=1, seed_base=7000, min_free_gib=0,
        production_cache=cache, joint_activation=True, collect_col_energy=True,
        model_identity=_model_identity("joint-source"), checkpoint_dir=checkpoint,
        resume=resume, boundary_storage=(None if path is None else
            _policy(path, window=window, cap=(2 * window + 1) * 256)),
    )
    return payload, source_events, context


def test_full_draw_boundary_residency_is_bounded(tmp_path):
    _, _, runner, _ = _fixture()
    ids = torch.tensor([[1, 2, 3, 4]] * 5)
    cap = 4 * 4 * 16 * 4
    with _bound(tmp_path, cap=cap) as owner:
        batches = [runner.capture_boundaries(row[None],
            boundary_writer=lambda layer, tensor, batch=index: owner.write(
                tensor, batch_index=batch, boundary_index=layer))
            for index, row in enumerate(ids)]
        assert all(isinstance(ref, ExactActivationReference)
                   for batch in batches for ref in batch.activations_cpu)
        resident = sum(t.untyped_storage().nbytes()
                       for batch in batches for t in batch.activations_cpu
                       if isinstance(t, torch.Tensor))
        assert resident <= cap
        assert owner.telemetry["peak_resident_tensor_bytes"] <= cap
        refs = [batch.activations_cpu[0] for batch in batches]
        with owner.prefetch(refs[:4]) as window:
            assert all(owner.get(window, ref).shape == (1, 4, 16) for ref in refs[:4])
        assert owner.telemetry["resident_tensor_bytes"] == 0
    assert not list(tmp_path.rglob("*.pt"))
    assert json.loads(next(tmp_path.rglob("generation.json")).read_text())["status"] == "complete"


@pytest.mark.parametrize("window", [1, 2, 3])
def test_exact_streamed_costs_cotangents_and_source_order_match_legacy(tmp_path, monkeypatch, window):
    # Audit every propagated cotangent separately from final cost agreement.
    seen = []
    original_write = StreamedBoundaryArtifacts.write
    def observe(self, tensor, **kwargs):
        if kwargs.get("probe_index") is not None:
            seen.append((kwargs["probe_index"], kwargs["batch_index"],
                         kwargs["boundary_index"], tensor.detach().clone()))
        return original_write(self, tensor, **kwargs)
    monkeypatch.setattr(StreamedBoundaryArtifacts, "write", observe)
    expected_gradients = []
    expected, expected_events, _ = _run(None, owner_events=expected_gradients)
    actual, actual_events, context = _run(tmp_path / "bounded", window=window)
    assert actual_events == expected_events
    assert actual["costs"] == expected["costs"]
    assert context.active == set()
    assert len(seen) == 4 * 5 * 3
    assert {key[0] for key in seen} == {0, 1, 2, 3}
    assert len(expected_gradients) == len(seen)
    for expected_gradient, actual_gradient in zip(expected_gradients, seen):
        assert actual_gradient[:3] == expected_gradient[:3]
        assert torch.equal(actual_gradient[3], expected_gradient[3])
    for name, row in actual["stats"].items():
        assert row["h_trace"] == expected["stats"][name]["h_trace"]
        assert torch.equal(row["fisher_col"], expected["stats"][name]["fisher_col"])
    receipt = actual["provenance"]["streamed_boundary_storage"]
    assert receipt["status"] == "complete"
    assert receipt["telemetry"]["peak_resident_tensor_bytes"] <= (2 * window + 1) * 256
    assert receipt["telemetry"]["live_artifact_bytes"] == 0
    assert receipt["telemetry"]["hot_read_misses"] == 0
    assert not list(tmp_path.rglob("*.pt"))
    for rows in actual["costs"].values():
        assert all(row["probe_ids"] == [7000, 7001, 7002, 7003] for row in rows.values())


def test_exact_window_cannot_lazy_load_or_outlive_lease(tmp_path, monkeypatch):
    import prismaquant.perturbed_x_cache as cache
    with _bound(tmp_path) as owner:
        first = owner.write(torch.randn(1, 4, 16), batch_index=0, boundary_index=0)
        second = owner.write(torch.randn(1, 4, 16), batch_index=1, boundary_index=0)
        with owner.prefetch([first]) as window:
            def forbidden(*args, **kwargs):
                raise AssertionError("disk read in active exact window")
            monkeypatch.setattr(cache.torch, "load", forbidden)
            assert owner.get(window, first).shape == (1, 4, 16)
            with pytest.raises(RuntimeError, match="not ready"):
                owner.get(window, second)
            ref = weakref.ref(owner.get(window, first))
        assert ref() is None
        with pytest.raises(RuntimeError, match="active resident window"):
            owner.get(window, first)


def test_noncontiguous_snapshot_uses_one_compact_tensor_allocation(tmp_path):
    # Transpose would preserve strides in ordinary Tensor.to(copy=True), then
    # require a second full allocation for contiguous(). Both copies matter to
    # a strict one-write-copy envelope; profiler self memory avoids double
    # counting parent/child operators.
    original = torch.arange(256, dtype=torch.float32).reshape(16, 16).T
    nbytes = original.numel() * original.element_size()
    with _bound(tmp_path, cap=nbytes) as owner:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    profile_memory=True) as profiler:
            ref = owner.write(original, batch_index=0, boundary_index=0)
        allocated = sum(max(0, event.self_cpu_memory_usage) for event in profiler.events())
        assert allocated <= nbytes
        with owner.prefetch([ref]) as window:
            restored = owner.get(window, ref)
            assert restored.is_contiguous()
            assert restored.untyped_storage().nbytes() == nbytes
            assert torch.equal(restored, original)
            del restored


@pytest.mark.parametrize("fault", ["missing", "checksum", "shape", "identity"])
def test_exact_window_refuses_bad_receipts_before_exposing_tensors(tmp_path, fault):
    with _bound(tmp_path) as owner:
        ref = owner.write(torch.randn(1, 4, 16), batch_index=0, boundary_index=0)
        if fault == "missing":
            Path(ref.path).unlink()
        elif fault == "checksum":
            data = bytearray(Path(ref.path).read_bytes()); data[-1] ^= 1
            Path(ref.path).write_bytes(data)
        else:
            bad = replace(ref, shape=(1, 8, 8)) if fault == "shape" else replace(ref, metadata_json='{}')
            owner._references[ref.name] = bad
            owner._slots["boundary-0-0"] = bad
            ref = bad
        with pytest.raises((RuntimeError, FileNotFoundError, KeyError)):
            with owner.prefetch([ref]):
                pytest.fail("bad entry exposed as ready")
        assert owner.telemetry["resident_tensor_bytes"] == 0


def test_rollover_refuses_stale_or_skipped_boundary_and_retires_only_previous(tmp_path):
    with _bound(tmp_path) as owner:
        old = owner.write(torch.ones(1, 4, 16), batch_index=0, boundary_index=2, probe_index=0)
        with pytest.raises(RuntimeError, match="coordinates"):
            owner.write(torch.zeros(1, 4, 16), batch_index=0, boundary_index=0, probe_index=0, previous=old)
        new = owner.write(torch.zeros(1, 4, 16), batch_index=0, boundary_index=1, probe_index=0, previous=old)
        assert not Path(old.path).exists() and Path(new.path).exists()
        with pytest.raises(RuntimeError, match="stale"):
            owner.write(torch.ones(1, 4, 16), batch_index=0, boundary_index=0, probe_index=0, previous=old)
        with owner.prefetch([new]) as window:
            assert torch.equal(owner.get(window, new), torch.zeros(1, 4, 16))


def test_limits_refuse_before_copy_and_failure_cleans_owned_files(tmp_path, monkeypatch):
    with _bound(tmp_path, cap=256) as owner:
        with pytest.raises(RuntimeError, match="residency"):
            owner.write(torch.zeros(2, 4, 16), batch_index=0, boundary_index=0)
        ref = owner.write(torch.zeros(1, 4, 16), batch_index=0, boundary_index=0)
        with owner.prefetch([ref]):
            with pytest.raises(RuntimeError, match="residency"):
                owner.write(torch.zeros(1, 4, 16), batch_index=1, boundary_index=0)
    assert not list(tmp_path.rglob("*.pt*"))
    with pytest.raises(RuntimeError, match="fixture write failure"):
        with _bound(tmp_path) as owner:
            import prismaquant.perturbed_x_cache as cache
            original = cache.write_activation_cache_entry
            def interrupted(*args, **kwargs):
                result = original(*args, **kwargs)
                raise RuntimeError("fixture write failure")
            monkeypatch.setattr(cache, "write_activation_cache_entry", interrupted)
            owner.write(torch.zeros(1, 4, 16), batch_index=0, boundary_index=0)
    assert not list(tmp_path.rglob("*.pt*"))
    assert {json.loads(p.read_text())["status"] for p in tmp_path.rglob("generation.json")} == {"complete", "failed"}


def test_exact_storage_policy_is_closed_and_complete(tmp_path):
    good = _policy(tmp_path)
    assert normalize_boundary_storage(good) == good
    for key in good:
        with pytest.raises(ValueError):
            normalize_boundary_storage({k: v for k, v in good.items() if k != key})
    with pytest.raises(ValueError):
        normalize_boundary_storage({**good, "prefetch_batches": True})


def test_shared_state_mapping_keys_cannot_hide_tensor_or_opaque_owners(tmp_path):
    from prismaquant.cost_streaming import StreamedForwardBoundaries
    with _bound(tmp_path, aux=64) as owner:
        key = torch.zeros(32)
        batch = StreamedForwardBoundaries(None, None, None, None, [], {key: None})
        with pytest.raises(RuntimeError, match="auxiliary/shared-state"):
            owner.check_auxiliary([batch])
        batch.shared_pass_state = {object(): None}
        with pytest.raises(TypeError, match="opaque state"):
            owner.check_auxiliary([batch])


def test_rank_four_boundaries_keep_all_residual_streams_and_original_dtype(tmp_path):
    from prismaquant.model_profiles.default import DefaultProfile
    class FourStreams(DefaultProfile):
        def expand_hidden_for_layers(self, hidden, base_model):
            return hidden.unsqueeze(2).expand(-1, -1, 4, -1).contiguous()
        def collapse_hidden_after_layers(self, hidden, base_model):
            return hidden.mean(dim=2)
    _, _, runner, _ = _fixture()
    runner.profile = FourStreams()
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = runner.capture_boundaries(ids)
    with _bound(tmp_path, cap=4096) as owner:
        actual = runner.capture_boundaries(ids, boundary_writer=lambda layer, tensor:
            owner.write(tensor, batch_index=0, boundary_index=layer))
        for ref, tensor in zip(actual.activations_cpu, expected.activations_cpu):
            with owner.prefetch([ref]) as window:
                restored = owner.get(window, ref)
                assert restored.shape == (1, 4, 4, 16)
                assert restored.dtype == tensor.dtype
                assert torch.equal(restored.view(torch.uint8), tensor.view(torch.uint8))
                del restored


def _shared_run(path, *, aux=1 << 20):
    from types import SimpleNamespace
    from test_kv_cotangent_path import _ToyModel, SHARED_SPECS
    from test_streamed_cost_checkpoints import _FakeStreamingContext
    from prismaquant.cost_streaming import StreamedCausalLM
    from prismaquant.model_profiles.gemma4 import Gemma4Profile
    from prismaquant.production_weight_cache import ProductionWeightCache
    toy = _ToyModel(SHARED_SPECS, hidden=16, seed=91)
    toy.config = SimpleNamespace(layer_types=())
    model = torch.nn.Module()
    model.model = toy
    model.lm_head = toy.lm_head
    context = _FakeStreamingContext(model)
    runner = StreamedCausalLM(context, Gemma4Profile())
    weights = {(name, "NVFP4A16"): module.weight.detach().clone() + 0.03125
               for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and ".layers." in name}
    cache = ProductionWeightCache(weights=weights, levers={})
    result = aura.compute_aura_cost_streamed(runner,
        torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 4, 3]]),
        ["NVFP4A16", "BF16"], n_probes=4, probe_microbatch=1, seed_base=7000,
        min_free_gib=0, joint_activation=True, production_cache=cache,
        model_identity=_model_identity("shared-source"),
        boundary_storage=None if path is None else _policy(path, cap=4096, aux=aux))
    return result


def test_shared_state_and_its_probe_cotangents_are_preserved_and_budgeted(tmp_path):
    expected = _shared_run(None)
    actual = _shared_run(tmp_path / "bounded")
    assert actual["costs"] == expected["costs"]
    telemetry = actual["provenance"]["streamed_boundary_storage"]["telemetry"]
    assert telemetry["peak_shared_cotangent_reservation_bytes"] > 0
    assert telemetry["peak_auxiliary_bytes"] > telemetry["peak_shared_cotangent_reservation_bytes"]
    with pytest.raises(RuntimeError, match="auxiliary/shared-state"):
        _shared_run(tmp_path / "refused", aux=1024)
    assert not list((tmp_path / "refused").rglob("*.pt*"))
    assert json.loads(next((tmp_path / "refused").rglob("generation.json")).read_text())["status"] == "failed"


def test_opaque_shared_state_refuses_instead_of_hiding_tensor_ownership(tmp_path):
    from types import SimpleNamespace
    from prismaquant.cost_streaming import StreamedForwardBoundaries
    batch = StreamedForwardBoundaries(torch.ones(1, 4, dtype=torch.int64), None,
        None, None, [], SimpleNamespace(hidden=torch.ones(1024)))
    with _bound(tmp_path) as owner:
        with pytest.raises(TypeError, match="opaque state"):
            owner.check_auxiliary([batch])


def test_completed_resume_skips_generation_and_policy_change_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    first, _, _ = _run(tmp_path / "working", checkpoint=tmp_path / "checkpoints")
    actual, events, context = _run(tmp_path / "working", checkpoint=tmp_path / "checkpoints", resume=True)
    assert actual["costs"] == first["costs"]
    assert events == [] and context.install_calls == 0
    assert actual["provenance"]["streamed_boundary_storage"]["status"] == "unused"
    assert len(list((tmp_path / "working").iterdir())) == 1
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _run(tmp_path / "working", window=1, checkpoint=tmp_path / "checkpoints", resume=True)


def test_interrupted_cost_resume_uses_new_generation_and_original_signed_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    expected, _, _ = _run(None)
    writer = aura._write_aura_unit_checkpoint
    def interrupt(*args, **kwargs):
        writer(*args, **kwargs)
        raise RuntimeError("fixture interrupted cost checkpoint")
    monkeypatch.setattr(aura, "_write_aura_unit_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="fixture interrupted"):
        _run(tmp_path / "working", checkpoint=tmp_path / "checkpoints")
    assert not list((tmp_path / "working").rglob("*.pt*"))
    failed = json.loads(next((tmp_path / "working").rglob("generation.json")).read_text())
    assert failed["status"] == "failed" and failed["working_artifacts_reusable"] is False
    monkeypatch.setattr(aura, "_write_aura_unit_checkpoint", writer)
    actual, _, _ = _run(tmp_path / "working", checkpoint=tmp_path / "checkpoints", resume=True)
    assert actual["costs"] == expected["costs"]
    completed = actual["provenance"]["streamed_boundary_storage"]
    assert completed["session"]["generation"] != failed["session"]["generation"]
    assert not list((tmp_path / "working").rglob("*.pt*"))


def test_failed_reverse_releases_windows_shared_state_and_hooks(tmp_path, monkeypatch):
    seen_windows = []
    import prismaquant.perturbed_x_cache as cache
    original_get = cache._ExactActivationPrefetch.get
    def get(self, reference):
        value = original_get(self, reference)
        seen_windows.append(weakref.ref(value))
        return value
    monkeypatch.setattr(cache._ExactActivationPrefetch, "get", get)
    from prismaquant.cost_streaming import StreamedCausalLM
    def fail(*args, **kwargs):
        raise RuntimeError("fixture reverse refusal")
    monkeypatch.setattr(StreamedCausalLM, "isolated_layer", fail)
    with pytest.raises(RuntimeError, match="fixture reverse refusal"):
        _run(tmp_path)
    assert seen_windows and all(ref() is None for ref in seen_windows)
    assert not list(tmp_path.rglob("*.pt*"))
    record = json.loads(next(tmp_path.rglob("generation.json")).read_text())
    assert record["status"] == "failed"
    assert record["telemetry"]["resident_tensor_bytes"] == 0
