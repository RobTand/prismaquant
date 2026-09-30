"""CPU research incoming-checkpoint behavior; no timing qualification."""
from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import prismaquant.joint_adjoint_checkpoints as checkpoints
import prismaquant.joint_cost_quantum as quantum
import test_stageb_one_pass_spill as spill_fixture

MODE = "stream_once_research"


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    # The reused routed fixture otherwise auto-selects CUDA on GPU workers.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(spill_fixture, "_device", lambda: torch.device("cpu"))
        return getattr(spill_fixture.campaign, "__wrapped__")(tmp_path_factory)


@pytest.fixture(autouse=True)
def _clear_journal(campaign):
    for layer in (0, 1):
        spill_fixture._clear_output(campaign, layer)
    yield
    for layer in (0, 1):
        spill_fixture._clear_output(campaign, layer)


def _enable(monkeypatch, mode=MODE):
    original = quantum.run_layer_quantum_core

    def launch(*args, **kwargs):
        kwargs["execution"] = {
            **kwargs["execution"], "checkpoint_incoming_mode": mode}
        return original(*args, **kwargs)

    monkeypatch.setattr(quantum, "run_layer_quantum_core", launch)


def test_first_real_chain_has_no_eager_checkpoint_activation_reads(campaign, monkeypatch):
    """Principal RED: old core ignores the selection and reads before roll."""
    assert campaign.records[0]["adjoint"]["chain_layers"]
    _enable(monkeypatch)
    original_stream = checkpoints.stream_exact_entry_tensors
    original_roll = checkpoints.render_free_layer_roll
    seen = {"head_rows": 0, "rolls": 0}

    def stream(entries, **kwargs):
        entries = list(entries)
        if seen["rolls"] == 0:
            seen["head_rows"] += len(entries)
        return original_stream(entries, **kwargs)

    def roll(*args, **kwargs):
        if seen["rolls"] == 0:
            assert seen["head_rows"] == 0, "checkpoint head eagerly read activation rows"
            assert kwargs["incoming_entries"] is not None
        else:
            assert kwargs["incoming_entries"] is None
        seen["rolls"] += 1
        return original_roll(*args, **kwargs)

    monkeypatch.setattr(checkpoints, "stream_exact_entry_tensors", stream)
    monkeypatch.setattr(checkpoints, "render_free_layer_roll", roll)
    payload, state = spill_fixture._quantum(campaign, monkeypatch, layer=0)
    assert not hasattr(state, "error"), repr(getattr(state, "error", None))
    assert payload is not None and seen["rolls"] > 0


@pytest.mark.parametrize("mode", [True, 1, "unknown", {}])
def test_invalid_research_mode_refuses_before_checkpoint_loader(campaign, monkeypatch, mode):
    _enable(monkeypatch, mode)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported selection reached checkpoint payload loader")

    monkeypatch.setattr(checkpoints, "load_adjoint_checkpoint", forbidden)
    payload, state = spill_fixture._quantum(campaign, monkeypatch, layer=0)
    assert payload is None
    assert isinstance(state.error, quantum.QuantumIdentityRefused), repr(state.error)
    assert "checkpoint incoming" in str(state.error)


@pytest.mark.parametrize("layer", [0, 1])
def test_same_binding_cost_journal_and_final_plane_bytes(campaign, monkeypatch, tmp_path, layer):
    launch_kwargs = {}
    if layer == 1:
        # Keep real O_DIRECT, model only the older x86 kernel's missing grid report.
        import prismaquant.perturbed_x_cache as cache
        monkeypatch.setattr(cache, "_direct_io_block", lambda fd: 4096)
        launch_kwargs = {
            "spill_root": spill_fixture._spill_root(tmp_path, needs_direct_io=False),
            "ceiling": 64 << 20}
    snapshots = []
    held = checkpoints.PlaneHostStaging.held

    @contextmanager
    def capture_final(self):
        with held(self):
            yield
            snapshots.append({key: value.contiguous().view(torch.uint8).numpy().tobytes()
                              for key, value in self.plane.items()})

    monkeypatch.setattr(checkpoints.PlaneHostStaging, "held", capture_final)
    spill_fixture._clear_output(campaign, layer)
    baseline, state = spill_fixture._quantum(campaign, monkeypatch, layer=layer, **launch_kwargs)
    assert not hasattr(state, "error"), repr(getattr(state, "error", None))
    baseline_evidence = spill_fixture._evidence(campaign, layer, baseline)
    baseline_plane = snapshots[-1]
    snapshots.clear()
    spill_fixture._clear_output(campaign, layer)
    _enable(monkeypatch)
    candidate, state = spill_fixture._quantum(campaign, monkeypatch, layer=layer, **launch_kwargs)
    assert not hasattr(state, "error"), repr(getattr(state, "error", None))
    assert spill_fixture._evidence(campaign, layer, candidate) == baseline_evidence
    assert snapshots[-1] == baseline_plane
    if layer == 1:
        assert state.counters_block["handoff_incoming"]["source"] == "checkpoint_research"
    # A fully validated journal avoids all incoming passes; default and research agree.
    snapshots.clear()
    resumed, state = spill_fixture._quantum(
        campaign, monkeypatch, layer=layer, resume=True, **launch_kwargs)
    assert not hasattr(state, "error"), repr(getattr(state, "error", None))
    assert spill_fixture._evidence(campaign, layer, resumed) == baseline_evidence
    assert snapshots == []
    spill_fixture._clear_output(campaign, layer)
    # A real partial durable journal still consumes the whole original incoming
    # plane on its final pass, rather than treating already-priced units as finality.
    import prismaquant.aura_cost as aura
    from pathlib import Path
    atomic = aura.atomic_write_bytes
    published = []

    def fail_second(path, body):
        if Path(path).suffix == ".pkl":
            if published:
                raise OSError("partial research journal")
            published.append(str(path))
        return atomic(path, body)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(aura, "atomic_write_bytes", fail_second)
        failed, state = spill_fixture._quantum(campaign, monkeypatch, layer=layer, **launch_kwargs)
    assert failed is None and isinstance(state.error, OSError)
    assert len(published) == 1 and Path(published[0]).is_file()
    snapshots.clear()
    restored = []
    import prismaquant.joint_statistics_replay as replay
    observe = replay.observe_and_project_retained_windows

    def resume(*args, **kwargs):
        restored.append(set(kwargs["completed_names"]))
        return observe(*args, **kwargs)

    monkeypatch.setattr(replay, "observe_and_project_retained_windows", resume)
    partial, state = spill_fixture._quantum(
        campaign, monkeypatch, layer=layer, resume=True, **launch_kwargs)
    assert not hasattr(state, "error"), repr(getattr(state, "error", None))
    assert partial is not None
    assert restored[0] and len(restored[0]) < len(partial["costs"])
    assert snapshots[-1] == baseline_plane
    assert spill_fixture._evidence(campaign, layer, partial) == baseline_evidence
    spill_fixture._clear_output(campaign, layer)


@pytest.mark.parametrize("case", ["gpu", "staged", "executable", "capture_profile", "handoff"])
def test_unsupported_selection_refuses_before_context_install(campaign, monkeypatch, case):
    original = quantum.run_layer_quantum_core
    install = []
    if case == "capture_profile":
        import prismaquant.stage_b_workspace_profile as profile
        monkeypatch.setattr(profile, "profile_request", lambda: object())

    def launch(*args, **kwargs):
        kwargs["execution"] = {**kwargs["execution"], "checkpoint_incoming_mode": MODE}
        runner = args[0] if args else kwargs["runner"]
        saved_device = runner.device
        try:
            if case == "gpu":
                runner.device = "cuda"
            elif case == "staged":
                kwargs["execution"]["staged_manifest"] = "not-opened"
            elif case == "executable":
                kwargs["record"] = {**kwargs["record"], "executable_readset": {}}
            elif case == "handoff":
                kwargs["adjoint_handoff"] = {}
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(runner.context, "install", forbidden)
                return original(*args, **kwargs)
        finally:
            runner.device = saved_device

    monkeypatch.setattr(quantum, "run_layer_quantum_core", launch)
    def forbidden(*args, **kwargs):
        install.append(True)
        raise AssertionError("research refusal reached source install")
    # Real core uses this context method, so a failed refusal cannot quietly capture.
    payload, state = spill_fixture._quantum(campaign, monkeypatch, layer=0)
    assert payload is None and isinstance(state.error, quantum.QuantumIdentityRefused), repr(getattr(state, "error", None))
    assert install == []
