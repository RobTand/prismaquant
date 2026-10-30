"""CPU orchestration checks, without a native model or serving qualification."""
from functools import partial
import sys
from types import SimpleNamespace

import pytest
import torch

from experiments import diagnose_exl3_grouped_down_replay as driver
from experiments import exl3_replay_cli as cli
from experiments import measure_glm_tr3_vllm as native


def test_cleanup_preserves_an_unowned_native_capture(monkeypatch):
    state, handle = object(), object()
    model = SimpleNamespace(_tr3_capture=state, _tr3_capture_handle=handle)
    removed = []
    monkeypatch.setattr(native, "remove_capture", lambda m: removed.append(m))
    cli._cleanup_native_capture(model)
    assert not removed
    assert model._tr3_capture is state
    assert model._tr3_capture_handle is handle


@pytest.mark.parametrize("preexisting", [False, True])
def test_failed_installation_cleans_only_its_own_partial_capture(monkeypatch, preexisting):
    original = object()
    model = SimpleNamespace(**({"_tr3_capture": original} if preexisting else {}))

    def install(model, **kwargs):
        model._tr3_capture = object()
        raise RuntimeError("partial native install")

    monkeypatch.setattr(native, "install_capture", install)
    with pytest.raises(ValueError if preexisting else RuntimeError):
        cli._install_native_capture(model, owner="this-run", tile_rows=32, logits_layout="legacy_single")
    cli._cleanup_native_capture(model, owner="another-run")
    assert hasattr(model, "_tr3_capture")
    cli._cleanup_native_capture(model, owner="this-run")
    if preexisting:
        assert model._tr3_capture is original
    else:
        assert not hasattr(model, "_tr3_capture")
        assert not hasattr(model, "_tr3_diagnostic_capture_owner")


def test_cleanup_failure_cannot_leave_valid_diagnostic_evidence(monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    args = cli.parse_args([
        "--model", str(model), "--candidate-digest-cache", "cache",
        "--panel", "panel", "--teacher", "teacher", "--teacher-sha256", "a" * 64,
        "--serve-image", "example.invalid/image@sha256:" + "b" * 64,
        "--kv-cache-dtype", "auto", "--expected-kv-cache-dtype", "bfloat16",
        "--tensor-parallel-size", "2",
        "--output", str(tmp_path / "diagnostic.json"),
    ])
    panel = {"windows": [{"window_id": "final-0000"}]}
    teacher = {"windows": [{}], "tokenizer_identity": {"tokenizer": "sealed"}}
    rows = [{"rank": rank, "world_size": 2} for rank in range(2)]
    publications = []
    built = []

    class LLM:
        def __init__(self, **kwargs):
            pass

        def apply_model(self, function):
            target = function.func if isinstance(function, partial) else function
            if target is cli._cleanup_native_capture:
                raise RuntimeError("injected native cleanup failure")
            return rows

        def collective_rpc(self, function, **kwargs):
            return rows

        def generate(self, *args, **kwargs):
            return [object()]

    def replay_request(llm, *, generate, **kwargs):
        generate()
        return rows

    def build_report(**kwargs):
        built.append(True)
        return {"diagnostic_only": True, "evidence_valid": True}

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw))
    monkeypatch.setattr(native, "load_panel", lambda *a, **kw: (panel, [torch.zeros(1, 3, dtype=torch.long)]))
    monkeypatch.setattr(native, "load_teacher", lambda *a: teacher)
    monkeypatch.setattr(native, "bound_json", lambda *a: teacher)
    monkeypatch.setattr(native, "sha256", lambda *a: native.TOKENIZER_SHA256)
    monkeypatch.setattr(native, "tokenizer_identity", lambda *a: teacher["tokenizer_identity"])
    monkeypatch.setattr(native, "cached_checkpoint_identity", lambda *a: {"candidate": "sealed"})
    monkeypatch.setattr(native, "producer_identity", lambda: {"producer": "sealed"})
    monkeypatch.setattr(driver, "producer_identity", lambda: {"producer": "sealed"}, raising=False)
    monkeypatch.setattr(native, "scorer_engine_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr(native, "refuse_if_spec_decode", lambda **kw: False)
    monkeypatch.setattr(native, "observed_engine_configuration", lambda *a, **kw: {})
    monkeypatch.setattr(native, "write_runtime_observation", lambda *a: None)
    monkeypatch.setattr(native, "collect_tp_result", lambda *a, **kw: None)
    monkeypatch.setattr(native, "verify_prompt_alignment", lambda *a: {"aligned": True})
    monkeypatch.setattr(native, "atomic_json_write", lambda result, path: publications.append(result))
    monkeypatch.setattr(driver, "replay_request", replay_request)
    monkeypatch.setattr(driver, "build_report", build_report)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        cli.run(args)
    assert built == [True], publications
    assert publications
    assert all(result["evidence_valid"] is False for result in publications)
