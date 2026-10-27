"""Issue 2769: CPU plumbing contracts, NOT native repeatability evidence.

The opt-in module's small public test seam is specified here so tests collect
on main even before the driver exists. Missing functionality is an assertion
failure, never an import/discovery error or an optional host-dependent skip.

Driver API:
  install_replay(model, *, module_name, repeats, required_rows)
  finish_replay_capture(model, *, window_id) -> native capture report plus
      replay (record or None), replay_declined (list), loaded_exl3_identity
  remove_replay(model)
  replay_request(llm, *, window_id, arm_kwargs, generate, module_name,
                 repeats, required_rows) -> list of finish_replay_capture reports
  build_report(**_report_args()) -> strict-JSON diagnostic-only evidence
  parse_args(argv) -> argparse.Namespace

The coordinator operates on already-installed native captures. Its caller owns
native capture install/removal. Replay cleanup is unconditional, including a
partially successful worker installation. generate is a zero-argument callback
for exactly one first-window request. No teacher/checkpoint/GPU is needed here.
"""
import copy
import hashlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from experiments import measure_glm_tr3_vllm as served
from experiments.glm_tr3_full_vocab import PromptLogitsCapture
from test_exl3_grouped_down_replay import (
    ROWS, N, _args, _Boundary, _Model, _Ext, replay_mod,
)

ROOT = Path(__file__).resolve().parents[1]
EXL3 = "vllm.model_executor.layers.quantization.exl3"
DRIVER = "experiments.diagnose_exl3_grouped_down_replay"
OPTIONS = dict(module_name="mlp", repeats=2, required_rows=ROWS)


def _driver():
    assert importlib.util.find_spec(DRIVER) is not None, (
        "Issue 2769 needs an opt-in diagnostic driver reusing GroupedDownReplay"
    )
    return importlib.import_module(DRIVER)


def _capture(rank=0, world_size=2):
    return PromptLogitsCapture(rank=rank, world_size=world_size, rows=ROWS - 1,
                               vocab_size=5, require_cuda=False)


def _arm(state):
    state.arm(0, "w0", torch.zeros(ROWS - 1, 5) if state.rank == 0 else None)


def _logits(state):
    for rows in (ROWS - 1, 1):
        state(None, (), torch.zeros(rows, 5))


def _model(tmp_path, rank=0, kernel=None, rows=ROWS):
    ext = _Ext(kernel or (lambda *args: args[3].fill_(1)))
    binary = tmp_path / "fake-extension.so"
    binary.write_bytes(b"fake CPU extension identity, never dynamically loaded")
    ext.__file__ = str(binary)
    model = _Model(_Boundary(ext, _args(rows)))
    model._tr3_capture = _capture(rank)
    return model, ext


def _loaded(monkeypatch, ext):
    module = types.ModuleType(EXL3)
    module.__file__ = __file__
    module._FAT_MOE_EXT_CACHE = [ext]
    monkeypatch.setitem(sys.modules, EXL3, module)
    return module


def _has_hash(value, digest):
    if isinstance(value, dict):
        return any(_has_hash(item, digest) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_hash(item, digest) for item in value)
    return value == digest


def test_worker_lifecycle_uses_live_capture_cache_and_restores_entrypoint(monkeypatch, tmp_path):
    driver = _driver()
    model, ext = _model(tmp_path)
    module = _loaded(monkeypatch, ext)
    state, original = model._tr3_capture, ext.exl3_fat_moe_down
    driver.install_replay(model, **OPTIONS)
    try:
        assert model._tr3_grouped_down_replay.state is state
        model(torch.zeros(1))  # unarmed warmup must not spend the shot
        assert not model._tr3_grouped_down_replay.fired
        _arm(state)
        model(torch.zeros(1))
        _logits(state)
        report = driver.finish_replay_capture(model, window_id="w0")
        assert state.window_id is None  # actual native finish ran
        assert report["window_id"] == report["replay"]["window_id"] == "w0"
        assert report["rank"] == report["replay"]["rank"] == 0
        assert report["replay"]["original_out_unchanged_after_replays"] is True
        identity = report["loaded_exl3_identity"]
        assert _has_hash(identity, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        assert _has_hash(identity, hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest())
        # Removal must use the installed extension, not a newly looked-up cache.
        module._FAT_MOE_EXT_CACHE = []
    finally:
        driver.remove_replay(model)
    assert ext.exl3_fat_moe_down is original
    assert model._tr3_capture is state
    assert not getattr(model, "_tr3_grouped_down_replay", None)
    assert not model.mlp._forward_pre_hooks and not model.mlp._forward_hooks
    driver.remove_replay(model)  # idempotent cleanup after partial failures


@pytest.mark.parametrize("missing", ["module", "cache", "capture", "boundary"])
def test_installation_refuses_missing_live_prerequisites_without_loading(monkeypatch, tmp_path, missing):
    driver = _driver()
    model, ext = _model(tmp_path)
    module = _loaded(monkeypatch, ext)
    original = ext.exl3_fat_moe_down
    if missing == "module":
        monkeypatch.delitem(sys.modules, EXL3)
    elif missing == "cache":
        module._FAT_MOE_EXT_CACHE = []
    elif missing == "capture":
        del model._tr3_capture
    options = dict(OPTIONS, module_name="absent" if missing == "boundary" else "mlp")
    with pytest.raises((ValueError, RuntimeError)):
        driver.install_replay(model, **options)
    driver.remove_replay(model)
    assert ext.exl3_fat_moe_down is original
    assert not model.mlp._forward_hooks and not model.mlp._forward_pre_hooks
    if missing == "module":
        assert EXL3 not in sys.modules


def test_nonfiring_preserves_shape_and_reason_without_reallocating(monkeypatch, tmp_path):
    driver = _driver()
    model, ext = _model(tmp_path, rows=3)
    _loaded(monkeypatch, ext)
    driver.install_replay(model, **OPTIONS)
    try:
        _arm(model._tr3_capture)
        model(torch.zeros(1))
        _logits(model._tr3_capture)
        report = driver.finish_replay_capture(model, window_id="w0")
        assert report["replay"] is None
        assert report["replay_declined"]
        assert str([3, N]) in json.dumps(report)  # actual call shape, not requested shape
        assert model.mlp.kernel_args[3].shape == (3, N)
    finally:
        driver.remove_replay(model)


class _LLM:
    def __init__(self, models):
        self.models = models

    def apply_model(self, function):
        return [function(model) for model in self.models]


@pytest.mark.parametrize("failure", [None, "request", "partial-install", "finish"])
def test_coordinator_collects_once_and_cleans_up_on_all_exit_paths(monkeypatch, tmp_path, failure):
    driver = _driver()
    model, ext = _model(tmp_path)
    _loaded(monkeypatch, ext)
    original = ext.exl3_fat_moe_down
    requests = []

    def arm(model, **kwargs):
        assert kwargs["window_id"] == "w0"
        _arm(model._tr3_capture)

    def generate():
        requests.append("w0")
        if failure == "request":
            raise RuntimeError("injected request failure")
        model(torch.zeros(1))
        if failure != "finish":
            _logits(model._tr3_capture)

    monkeypatch.setattr(served, "arm_capture", arm)
    if failure == "partial-install":
        install = driver.install_replay

        def interrupted(*args, **kwargs):
            install(*args, **kwargs)
            raise RuntimeError("injected partial install failure")

        monkeypatch.setattr(driver, "install_replay", interrupted)
    kwargs = dict(window_id="w0", arm_kwargs={"index": 0, "window_id": "w0"},
                  generate=generate, **OPTIONS)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            driver.replay_request(_LLM([model]), **kwargs)
    else:
        reports = driver.replay_request(_LLM([model]), **kwargs)
        assert len(reports) == 1 and reports[0]["replay"]["window_id"] == "w0"
    assert requests == ([] if failure == "partial-install" else ["w0"])
    assert ext.exl3_fat_moe_down is original
    assert not getattr(model, "_tr3_grouped_down_replay", None)
    assert not model.mlp._forward_hooks and not model.mlp._forward_pre_hooks


def _route(rank, count):
    return {"rank": rank, "exl3_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "exl3": {"tp_rank": rank, "tp_size": 2, "prefill_layer_calls": count}}


def _report_args(driver, monkeypatch, tmp_path, varying=False):
    reports = []
    for rank in range(2):
        count = [0]

        def kernel(*args):
            count[0] += 1
            args[3].fill_(count[0] if varying else 1)

        model, ext = _model(tmp_path, rank, kernel)
        _loaded(monkeypatch, ext)
        driver.install_replay(model, **OPTIONS)
        try:
            _arm(model._tr3_capture)
            model(torch.zeros(1))
            _logits(model._tr3_capture)
            reports.append(driver.finish_replay_capture(model, window_id="w0"))
        finally:
            driver.remove_replay(model)
    runtime = json.loads((ROOT / "tests/fixtures/glm_tr3_runtime/qualified-runtime.json").read_text())["runtime_binding"]
    monkeypatch.setattr(served, "producer_identity", lambda: {"gold_source": {"schema": "test-native-producer"}})
    runtime["producer_identity"] = driver.producer_identity()
    return dict(reports=reports, runtime_binding=runtime,
                diagnostics_before=[_route(rank, 3) for rank in range(2)],
                diagnostics_after=[_route(rank, 4) for rank in range(2)],
                world_size=2, window_id="w0", **OPTIONS)


@pytest.mark.parametrize("varying", [False, True])
def test_report_is_complete_diagnostic_evidence_even_for_finite_differences(monkeypatch, tmp_path, varying):
    driver = _driver()
    kwargs = _report_args(driver, monkeypatch, tmp_path, varying)
    snapshot = copy.deepcopy(kwargs)
    result = driver.build_report(**kwargs)
    assert kwargs == snapshot
    assert result["diagnostic_only"] is True
    assert result.get("passed") is not True
    assert result["evidence_valid"] is True and result["invalid_reasons"] == []
    assert result["runtime_binding"] == kwargs["runtime_binding"]
    assert result["reports"] == kwargs["reports"]
    assert result["diagnostics_before"] == kwargs["diagnostics_before"]
    assert result["diagnostics_after"] == kwargs["diagnostics_after"]
    text = json.dumps(result, allow_nan=False).lower()
    assert "synchron" in text and "mask" in text
    assert "not a determinism" in text
    for report in result["reports"]:
        replay = report["replay"]
        assert replay["pointed_to_weight_bytes_hashed"] is False
        assert replay["distinct_replay_digests"] == (2 if varying else 1)
        assert "vs_original" in replay["replays"][0]
        assert "vs_first_replay" in replay["replays"][1]
        assert "fp32_max_abs_delta" in replay["replays"][0]["vs_original"]
        assert "bf16_max_ulp_delta" in replay["replays"][0]["vs_original"]


DEFECTS = [
    "missing-rank", "duplicate-rank", "wrong-world", "wrong-window", "unfired",
    "changed-input", "forged-stability", "changed-served-output", "forged-preservation",
    "nonfinite", "original-nonfinite", "missing-input-hash", "missing-delta",
    "missing-replay", "wrong-module", "wrong-repeats", "wrong-rows",
    "before-no-tp", "before-wrong-tp", "after-wrong-tp", "before-no-counter", "after-no-counter",
    "no-route-delta", "missing-route-rank", "changed-overlay",
    "candidate-unavailable", "producer-unavailable", "candidate-empty-digest",
    "missing-teacher", "missing-panel", "missing-image", "missing-loaded-identity",
]


@pytest.mark.parametrize("defect", DEFECTS)
def test_report_rejects_incomplete_unattributable_or_mutated_evidence(monkeypatch, tmp_path, defect):
    driver = _driver()
    kwargs = _report_args(driver, monkeypatch, tmp_path)
    report = kwargs["reports"][0]
    replay = report["replay"]
    before, after = kwargs["diagnostics_before"][0], kwargs["diagnostics_after"][0]
    runtime = kwargs["runtime_binding"]
    if defect == "missing-rank":
        kwargs["reports"].pop()
    elif defect == "duplicate-rank":
        kwargs["reports"][1] = copy.deepcopy(report)
    elif defect == "wrong-world":
        replay["world_size"] = 99
    elif defect == "wrong-window":
        replay["window_id"] = "another-window"
    elif defect == "unfired":
        report["replay"] = None
        report["replay_declined"] = ["row count [3, 256] is not the 8-row window"]
    elif defect == "changed-input":
        replay["inputs_stable"] = False
    elif defect == "forged-stability":
        replay["inputs_after"]["h2"]["sha256"] = "b" * 64
    elif defect == "changed-served-output":
        replay["original_out_unchanged_after_replays"] = False
    elif defect == "forged-preservation":
        replay["original_out_after"]["fp32_sha256"] = "b" * 64
    elif defect == "nonfinite":
        replay["replays"][0]["finite"]["all_finite"] = False
        replay["replays"][0]["vs_original"]["fp32_max_abs_delta"] = float("nan")
    elif defect == "original-nonfinite":
        replay["original_out_before"]["fp32_finite"]["all_finite"] = False
    elif defect == "missing-input-hash":
        for side in ("inputs_before", "inputs_after"):
            del replay[side]["h2"]["sha256"]
    elif defect == "missing-delta":
        del replay["replays"][0]["vs_original"]
    elif defect == "missing-replay":
        replay["replays"].pop()
    elif defect.startswith("wrong-"):
        key = {"wrong-module": "module", "wrong-repeats": "repeats", "wrong-rows": "required_rows"}[defect]
        replay[key] = "wrong" if key == "module" else 99
    elif defect == "before-no-tp":
        del before["exl3"]["tp_rank"]
    elif defect in ("before-wrong-tp", "after-wrong-tp"):
        (before if defect.startswith("before") else after)["exl3"].update(tp_rank=99, tp_size=100)
    elif defect in ("before-no-counter", "after-no-counter"):
        del (before if defect.startswith("before") else after)["exl3"]["prefill_layer_calls"]
    elif defect == "no-route-delta":
        after["exl3"]["prefill_layer_calls"] = before["exl3"]["prefill_layer_calls"]
    elif defect == "missing-route-rank":
        kwargs["diagnostics_before"].pop()
    elif defect == "changed-overlay":
        after["exl3_source_sha256"] = "b" * 64
    elif defect in ("candidate-unavailable", "producer-unavailable"):
        runtime[defect.split("-")[0] + "_identity"] = {"unavailable_reason": "not observed"}
    elif defect == "candidate-empty-digest":
        runtime["candidate_identity"]["content_sha256"] = ""
    elif defect in ("missing-teacher", "missing-panel", "missing-image"):
        del runtime[{"missing-teacher": "teacher_sha256", "missing-panel": "panel_sha256", "missing-image": "serve_image"}[defect]]
    elif defect == "missing-loaded-identity":
        report["loaded_exl3_identity"] = {}
    result = driver.build_report(**kwargs)
    assert result["evidence_valid"] is False, defect
    assert result["invalid_reasons"], defect
    assert result["diagnostic_only"] is True and result.get("passed") is not True
    json.dumps(result, allow_nan=False)  # invalid/nonfinite evidence must remain serializable


def _cli(tmp_path):
    return ["--model", str(tmp_path / "model"), "--output", str(tmp_path / "evidence.json"),
            "--candidate-digest-cache", str(tmp_path / "identity.json"),
            "--panel", str(tmp_path / "panel.json"), "--teacher", str(tmp_path / "teacher.json"),
            "--teacher-sha256", "a" * 64, "--serve-image", "example.invalid/observer@sha256:" + "b" * 64,
            "--kv-cache-dtype", "auto", "--expected-kv-cache-dtype", "bfloat16"]


def test_cli_is_portable_bounded_and_explicitly_opt_in(tmp_path):
    driver = _driver()
    args = driver.parse_args(_cli(tmp_path))
    assert args.module == replay_mod.DEFAULT_MODULE
    assert args.required_rows == replay_mod.DEFAULT_REQUIRED_ROWS
    assert args.repeats == replay_mod.DEFAULT_REPEATS
    assert str(args.model) == str(tmp_path / "model")
    assert str(args.output) == str(tmp_path / "evidence.json")
    custom = driver.parse_args(_cli(tmp_path) + ["--module", "other.mlp", "--required-rows", "17", "--repeats", "3"])
    assert (custom.module, custom.required_rows, custom.repeats) == ("other.mlp", 17, 3)


@pytest.mark.parametrize("extra", [
    ["--repeats", "1"], ["--repeats", "17"], ["--required-rows", "0"],
    ["--qualify-hook"], ["--qualify-then-score", "qualification.json"],
    ["--qualification", "qualification.json"], ["--gold-record-out", "gold.json"],
])
def test_cli_refuses_unbounded_or_qualification_modes(tmp_path, extra):
    driver = _driver()
    with pytest.raises((SystemExit, ValueError)):
        driver.parse_args(_cli(tmp_path) + extra)


def test_default_scoring_remains_uninstrumented_and_diagnostics_cannot_qualify(monkeypatch, tmp_path):
    driver = _driver()
    kwargs = _report_args(driver, monkeypatch, tmp_path)
    diagnostic = driver.build_report(**kwargs)
    binding = kwargs["runtime_binding"]
    served.require_native_qualification({"schema": "prismaquant.glm_tr3_hook_qualification/1",
                                         "passed": True, "runtime_binding": binding}, binding)
    for record in (diagnostic, dict(diagnostic, passed=True, runtime_binding=binding)):
        with pytest.raises(ValueError, match="schema"):
            served.require_native_qualification(record, binding)
        with pytest.raises(ValueError, match="full-panel"):
            served._tr3_gold_record(record, model_sha="c" * 64, model="fake", spec_decode_detected=False)
    # Real ordinary finish retains the historical native report shape, with no
    # replay fields or extension wrapping merely because the driver was imported.
    model, ext = _model(tmp_path)
    original = ext.exl3_fat_moe_down
    _arm(model._tr3_capture)
    _logits(model._tr3_capture)
    ordinary = served.finish_capture(model, window_id="w0")
    assert "replay" not in ordinary and "loaded_exl3_identity" not in ordinary
    assert ext.exl3_fat_moe_down is original
    assert not getattr(model, "_tr3_grouped_down_replay", None)


@pytest.mark.parametrize("defect", [
    "unrelated-producer-digest", "unsubstantiated-candidate-digest",
    "unavailable-engine", "unavailable-workers", "unavailable-capture",
    "unavailable-kwargs", "unavailable-layout", "route-loaded-source-mismatch",
])
def test_report_rejects_unbound_runtime_identity(monkeypatch, tmp_path, defect):
    driver = _driver()
    kwargs = _report_args(driver, monkeypatch, tmp_path)
    runtime = kwargs["runtime_binding"]
    if defect == "unrelated-producer-digest":
        runtime["producer_identity"] = {"unrelated": "a" * 64}
    elif defect == "unsubstantiated-candidate-digest":
        runtime["candidate_identity"] = {"content_sha256": "a" * 64}
    elif defect == "route-loaded-source-mismatch":
        for side in ("diagnostics_before", "diagnostics_after"):
            for row in kwargs[side]:
                row["exl3_source_sha256"] = "b" * 64
    else:
        field = {"unavailable-engine": "observed_engine_configuration",
                 "unavailable-workers": "observed_worker_configuration",
                 "unavailable-capture": "worker_runtime",
                 "unavailable-kwargs": "engine_kwargs",
                 "unavailable-layout": "logits_layout"}[defect]
        runtime[field] = {"unavailable_reason": "not observed"}
    result = driver.build_report(**kwargs)
    assert result["evidence_valid"] is False, defect
    assert result["invalid_reasons"], defect
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("defect", [
    "runtime-none", "report-none", "rank-list", "route-none", "arguments-none",
    "argument-list", "original-list", "replays-none", "replay-entry-none",
    "comparison-list",
])
def test_malformed_worker_evidence_returns_invalid_strict_json(monkeypatch, tmp_path, defect):
    driver = _driver()
    kwargs = _report_args(driver, monkeypatch, tmp_path)
    report = kwargs["reports"][0]
    replay = report["replay"]
    if defect == "runtime-none":
        kwargs["runtime_binding"] = None
    elif defect == "report-none":
        kwargs["reports"][0] = None
    elif defect == "rank-list":
        report["rank"] = [0]
    elif defect == "route-none":
        kwargs["diagnostics_before"][0]["exl3"] = None
    elif defect == "arguments-none":
        replay["inputs_before"] = None
    elif defect == "argument-list":
        replay["inputs_before"]["h2"] = []
    elif defect == "original-list":
        replay["original_out_before"] = []
    elif defect == "replays-none":
        replay["replays"] = None
    elif defect == "replay-entry-none":
        replay["replays"][0] = None
    elif defect == "comparison-list":
        replay["replays"][0]["vs_original"] = []
    result = driver.build_report(**kwargs)
    assert result["evidence_valid"] is False, defect
    assert result["invalid_reasons"], defect
    assert result["diagnostic_only"] is True
    json.dumps(result, allow_nan=False)


def test_producer_identity_covers_native_owner_and_diagnostic_sources(monkeypatch):
    driver = _driver()
    native = {"gold_source": {"schema": "test-native-producer"}}
    monkeypatch.setattr(served, "producer_identity", lambda: native)
    observed = driver.producer_identity()
    assert observed["gold_source"] == native["gold_source"]
    assert observed["diagnostic_source_files"] == {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("experiments/diagnose_exl3_grouped_down_replay.py",
                     "experiments/exl3_replay_cli.py", "tools/exl3_grouped_down_replay.py")}


@pytest.mark.parametrize("defect", [
    "candidate-roster-mutation", "candidate-content-mutation", "capture-missing-rank",
    "capture-wrong-world", "capture-missing-source", "capture-empty-attention",
    "worker-missing-rank", "worker-missing-config", "worker-missing-layout-source",
    "request-prefix-caching", "request-nonfinite-memory", "coordinator-prefix-caching",
    "missing-initial-destination", "wrong-replay-destination-seed",
])
def test_report_validates_bound_checkpoint_runtime_and_replay_seed(monkeypatch, tmp_path, defect):
    driver = _driver()
    kwargs = _report_args(driver, monkeypatch, tmp_path)
    runtime = kwargs["runtime_binding"]
    if defect == "candidate-roster-mutation":
        runtime["candidate_identity"]["shards"][0]["sha256"] = "a" * 64
    elif defect == "candidate-content-mutation":
        runtime["candidate_identity"]["content_sha256"] = "a" * 64
    elif defect == "capture-missing-rank":
        runtime["worker_runtime"].pop()
    elif defect == "capture-wrong-world":
        runtime["worker_runtime"][0]["world_size"] = 99
    elif defect == "capture-missing-source":
        del runtime["worker_runtime"][0]["source_files"]["model"]
    elif defect == "capture-empty-attention":
        runtime["worker_runtime"][0]["attention_runtime"] = []
    elif defect == "worker-missing-rank":
        runtime["observed_worker_configuration"].pop()
    elif defect == "worker-missing-config":
        del runtime["observed_worker_configuration"][0]["configuration"]["cache_config"]
    elif defect == "worker-missing-layout-source":
        del runtime["observed_worker_configuration"][0]["prompt_layout_source"]["prompt_worker_source_sha256"]
    elif defect == "request-prefix-caching":
        runtime["engine_kwargs"]["enable_prefix_caching"] = True
    elif defect == "request-nonfinite-memory":
        runtime["engine_kwargs"]["gpu_memory_utilization"] = float("nan")
    elif defect == "coordinator-prefix-caching":
        runtime["observed_engine_configuration"]["cache_config"]["enable_prefix_caching"] = True
    elif defect == "missing-initial-destination":
        del kwargs["reports"][0]["replay"]["replay_initial_out_sha256"]
    elif defect == "wrong-replay-destination-seed":
        kwargs["reports"][0]["replay"]["replays"][0]["initial_out_sha256"] = "a" * 64
    result = driver.build_report(**kwargs)
    assert result["evidence_valid"] is False, defect
    assert result["invalid_reasons"], defect
    json.dumps(result, allow_nan=False)


def test_report_accepts_recorded_compilation_through_native_contract(monkeypatch, tmp_path):
    driver = _driver()
    kwargs = _report_args(driver, monkeypatch, tmp_path)
    runtime = kwargs["runtime_binding"]
    compilation = {"mode": "NONE", "cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1]}
    runtime["execution_mode"] = "compiled"
    runtime["engine_kwargs"].update(enforce_eager=False, compilation_config=compilation)
    configurations = [runtime["observed_engine_configuration"]] + [
        row["configuration"] for row in runtime["observed_worker_configuration"]]
    for config in configurations:
        config["model_config"]["enforce_eager"] = False
        config["compilation_config"] = compilation
    result = driver.build_report(**kwargs)
    assert result["evidence_valid"] is True, result["invalid_reasons"]
