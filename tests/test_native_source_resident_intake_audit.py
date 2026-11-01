"""Real caller intake values; measurement/tokenization/hardware are bounded seams.

No score, throughput, serving, export-byte or full-producer result is claimed.
Every model payload load and PQ source decoder remains real. See the companion
design for the enumerated doubles and the CPU-placement-only RTN adaptation.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from importlib.machinery import ModuleSpec
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch

from prismaquant import calibration_data, memory_management, streaming_model
from prismaquant import gpu_guard
from prismaquant.streamed_model_identity import build_source_model_binding
from tests.test_resident_source_decode_audit import _cpu_guard, _source


class _ObservedInput(RuntimeError):
    pass


@pytest.fixture
def owned_root(tmp_path):
    # The real validator refuses /tmp work roots. The alternative is also owned
    # by this test and removed on exit; it is not a production cache override.
    parent = None if not str(tmp_path.resolve()).startswith("/tmp") else "/var/tmp"
    if parent is None:
        yield tmp_path
    else:
        with tempfile.TemporaryDirectory(prefix="pq-native-intake-audit-", dir=parent) as directory:
            yield Path(directory)


def _assert_input(model, expected, *, dtype, gradient=None):
    plan = streaming_model._model_tensor_dtypes(model, dtype)
    mismatch = []
    parameters = dict(model.named_parameters())
    missing = sorted(set(expected) - set(parameters))
    extra = sorted(set(parameters) - set(expected))
    for name, parameter in parameters.items():
        if name not in expected:
            continue
        wanted = expected[name].to(device=parameter.device, dtype=plan.get(name, dtype))
        if parameter.is_meta or parameter.shape != wanted.shape or parameter.dtype != wanted.dtype or not torch.equal(parameter, wanted):
            error = None if parameter.is_meta or parameter.shape != wanted.shape else float((parameter.float() - wanted.float()).abs().max())
            mismatch.append((name, str(parameter.dtype), str(wanted.dtype), error))
    assert not mismatch, (f"caller received values different from the serialized PQ source: {mismatch}; "
                          f"missing={missing}, extra={extra}")
    assert not missing and not extra, f"caller changed source parameter population: missing={missing}, extra={extra}"
    if gradient is not None:
        assert all(parameter.requires_grad is gradient for parameter in parameters.values())
    if gradient:
        name = next(name for name in expected if name.endswith(".self_attn.o_proj.weight"))
        parameter = model.get_parameter(name)
        parameter.square().sum().backward()
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, 2 * parameter.detach())


def _setup(root, monkeypatch):
    from transformers import AutoTokenizer

    guard = _cpu_guard(root, monkeypatch)
    monkeypatch.setattr(memory_management, "CaptureMemoryGuard", lambda *args, **kwargs: guard)
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *args, **kwargs: torch.device("cpu"))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(
        pad_token_id=0, eos_token="<eos>", eos_token_id=1))
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    monkeypatch.setattr(calibration_data, "load_wikitext_calibration_windowed", lambda *args, **kwargs: tokens)
    monkeypatch.setattr(calibration_data, "load_calibration", lambda *args, **kwargs: tokens)
    monkeypatch.setattr(calibration_data, "load_validation_calibration", lambda *args, **kwargs:
                        calibration_data.CalibrationDraw(selection_repeats=(tokens,)))
    return tokens


@pytest.mark.parametrize("native", [True, False], ids=["native-block-fp8", "bf16-control"])
@pytest.mark.parametrize("caller", ["text-loader", "recache", "fisher", "cache-main",
                                   "expert-main", "expert-streamed-main", "validator-main", "harness",
                                   "rtn-reference", "rtn-reload"])
def test_actual_resident_caller_receives_decoded_source(owned_root, monkeypatch, native, caller):
    root, expected = _source(owned_root, native=native)
    tokens = _setup(owned_root, monkeypatch)
    seen = []

    def observe(model, *, dtype=torch.bfloat16, gradient=None):
        _assert_input(model, expected, dtype=dtype, gradient=gradient)
        seen.append(caller)
        raise _ObservedInput(caller)

    with pytest.raises(_ObservedInput, match=caller):
        if caller == "text-loader":
            from prismaquant.perturbed_x_cache import load_text_model_under_work_root
            model = load_text_model_under_work_root(str(root), device="cpu", dtype=torch.bfloat16,
                                                    work_root=owned_root / "work")
            observe(model, gradient=False)
        elif caller == "recache":
            from prismaquant import perturbed_x_cache, production_recache
            from prismaquant.production_weight_cache import ProductionWeightCache

            cache = ProductionWeightCache(weights={}, levers={}, metadata={
                "source_binding": build_source_model_binding(root, dtype=torch.bfloat16)})
            source_cache = owned_root / "cache.pkl"
            source_cache.write_bytes(pickle.dumps(cache))
            assignment = owned_root / "assignment.json"
            assignment.write_text("{}")
            real_load = perturbed_x_cache.load_text_model_under_work_root

            def loading(*args, **kwargs):
                observe(real_load(*args, **kwargs), gradient=False)

            monkeypatch.setattr(perturbed_x_cache, "load_text_model_under_work_root", loading)
            production_recache.main(["--model", str(root), "--layer-config", str(assignment),
                "--production-weight-cache", str(source_cache), "--output", str(owned_root / "recache.pkl"),
                "--device", "cpu", "--work-root", str(owned_root / "work")])
        elif caller == "fisher":
            from prismaquant import aura_cost, fisher_col_weights

            def measurement(model, *args, **kwargs):
                assert kwargs["collect_col_energy"] is True
                observe(model, dtype=torch.float32, gradient=True)

            monkeypatch.setattr(aura_cost, "compute_aura_cost", measurement)
            fisher_col_weights.compute_fisher_col_weights(str(root), None, device="cpu", nprobes=1,
                                                          nsamples=1, seqlen=4)
        elif caller == "cache-main":
            from prismaquant import build_production_cache as producer
            monkeypatch.setattr(producer, "require_cuda_hot_path", lambda *args, **kwargs: torch.device("cpu"))
            monkeypatch.setattr(producer, "_load_cache_calibration", lambda *args, **kwargs: tokens)
            monkeypatch.setattr(producer, "fill_production_weight_cache", lambda model, *args, **kwargs: observe(model))
            producer.main(["--model", str(root), "--output", str(owned_root / "cache.pkl"),
                           "--formats", "FP8_E4M3", "--n-calib-samples", "1", "--calib-seqlen", "4"])
        elif caller in {"expert-main", "expert-streamed-main"}:
            from transformers import AutoModelForCausalLM
            from prismaquant import expert_empirical_cost as producer
            real_hf = AutoModelForCausalLM.from_pretrained

            def cpu_hf(*args, **kwargs):
                if kwargs.get("device_map") == "cpu":
                    kwargs.pop("device_map")
                return real_hf(*args, **kwargs)

            monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", cpu_hf)
            monkeypatch.setattr(producer, "measure_expert_unit_costs", lambda model, *args, **kwargs: observe(model))

            def streamed_measurement(runner, *args, **kwargs):
                for layer in runner.context.source_layers:
                    runner.context.schedule_prefetch(layer)
                    runner.context.install(layer, require_prefetched=True, prefetch_following=False)
                observe(runner.model)

            monkeypatch.setattr(producer, "measure_expert_unit_costs_streamed", streamed_measurement)
            arguments = ["--model", str(root), "--output", str(owned_root / "expert.pkl"),
                           "--device", "cpu", "--formats", "NVFP4,BF16",
                           "--n-calib-samples", "1", "--calib-seqlen", "4"]
            if caller == "expert-streamed-main":
                arguments.extend(["--streaming", "--streaming-offload-dir", str(owned_root / "expert-offload")])
                monkeypatch.setenv("PRISMAQUANT_EXPERT_EVAL_DRIVER", "window")
                monkeypatch.setenv("PRISMAQUANT_PREFETCH_WORKERS", "1")
            producer.main(arguments)
        elif caller == "validator-main":
            from prismaquant import validate_assignments_kl as validator
            name = next(name for name in expected if name.endswith(".self_attn.o_proj.weight"))
            weight = expected[name]
            probe = owned_root / "probe.pkl"
            probe.write_bytes(pickle.dumps({"stats": {name: {"n_params": weight.numel(),
                "in_features": weight.shape[1], "out_features": weight.shape[0]}}}))
            base = owned_root / "base.json"
            base.write_text(json.dumps({name: "BF16"}))
            candidate = owned_root / "candidate.json"
            candidate.write_text("{}")
            monkeypatch.setattr(validator, "require_cuda_hot_path", lambda *args, **kwargs: torch.device("cpu"))
            monkeypatch.setattr(validator, "cache_reference_log_probs", lambda model, *args, **kwargs: observe(model))
            validator.main(["--model", str(root), "--probe", str(probe), "--base-assignment", str(base),
                "--assignment", str(candidate), "--output", str(owned_root / "validation.json"),
                "--device", "cpu", "--work-dir", str(owned_root / "work"), "--formats", "NVFP4,BF16",
                "--source-prefetch", "off", "--dtype", "bf16", "--n-calib-samples", "1", "--calib-seqlen", "4"])
        elif caller == "harness":
            from prismaquant import validation_harness as validator

            dataset_boundary = ModuleType("datasets")
            dataset_boundary.__spec__ = ModuleSpec("datasets", loader=None)
            dataset_boundary.load_dataset = lambda *args, **kwargs: pytest.fail(
                "dataset I/O must remain after the observed model-intake boundary")
            monkeypatch.setitem(sys.modules, "datasets", dataset_boundary)

            @contextmanager
            def perturbation(model, *args, **kwargs):
                observe(model, dtype=torch.float32)
                yield

            monkeypatch.delenv("PRISMAQUANT_VALIDATION_PROD_CACHE", raising=False)
            monkeypatch.setattr(validator, "_perturbed_model", perturbation)
            validator._compute_metrics(model_path=str(root), layer_config={}, cache_dir=owned_root / "metrics",
                device="cpu", dtype="bf16", n_wikitext_tokens=4, n_mmlu_questions=1, calib_seqlen=4,
                calib_n_samples=1, calib_split="test", calib_seed=7, wikitext_max_window_tokens=4, progress=False)
        else:
            from transformers import AutoModelForCausalLM
            from prismaquant import build_rtn_cache as producer

            real_hf = AutoModelForCausalLM.from_pretrained
            real_resident = streaming_model.load_resident_source_model
            real_inventory = producer.iter_quantizable_tensors
            enumerations = []

            def cpu_hf(*args, **kwargs):
                kwargs.pop("device_map", None)
                return real_hf(*args, **kwargs)

            def cpu_resident(*args, **kwargs):
                kwargs["device"] = torch.device("cpu")
                return real_resident(*args, **kwargs)

            def inventory(model, *args, **kwargs):
                enumerations.append(True)
                if len(enumerations) == 3:
                    observe(model)
                yield from real_inventory(model, *args, **kwargs)

            monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", cpu_hf)
            monkeypatch.setattr(streaming_model, "load_resident_source_model", cpu_resident)
            monkeypatch.setattr(producer, "load_wikitext_calibration", lambda *args, **kwargs: tokens)
            monkeypatch.setattr(producer, "cache_reference_log_probs",
                (lambda model, *args, **kwargs: observe(model)) if caller == "rtn-reference" else
                (lambda *args, **kwargs: []))
            monkeypatch.setattr(producer, "measure_kl", lambda *args, **kwargs: 0.0)
            if caller == "rtn-reload":
                monkeypatch.setattr(producer, "iter_quantizable_tensors", inventory)
            monkeypatch.setattr(sys, "argv", ["build_rtn_cache", "--model", str(root),
                "--cache-dir", str(owned_root / "rtn"), "--n-calib-samples", "1", "--calib-seqlen", "4"])
            producer.main()
    assert seen == [caller]


def test_native_rtn_main_refuses_existing_unqualified_output(owned_root, monkeypatch):
    from prismaquant import build_rtn_cache

    root, _expected = _source(owned_root)
    directory = owned_root / "rtn"
    directory.mkdir()
    old_product = directory / "cache_manifest.json"
    old_product.write_text('{"cache_type": "rtn"}')
    original = old_product.read_bytes()
    monkeypatch.setattr(sys, "argv", ["build_rtn_cache", "--model", str(root),
                                     "--cache-dir", str(directory)])
    with pytest.raises(ValueError, match="(?i)native.*(rebuild|fresh)"):
        build_rtn_cache.main()
    assert old_product.read_bytes() == original
    assert list(directory.iterdir()) == [old_product]
