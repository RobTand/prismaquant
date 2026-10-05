"""Real joint entry points must not leak their activation-scale policy."""
import os
from types import SimpleNamespace

import pytest
import torch

from prismaquant import joint_cost_quantum as quantum
from prismaquant import joint_cost_stage_a as stage_a
from prismaquant.tessera_joint_aura import ACTIVATION_SCALE_ENV
from test_joint_cost_quantum_runtime import _prefetch_budget, _stage_a_run_stub
from test_quantum_failure_counters import _Sampler, _record


class WorkloadError(RuntimeError):
    pass


def _quantum_run_stub(tmp_path, monkeypatch):
    """Keep the entry point, output publisher and teardown real; stub device work."""
    from prismaquant import (
        aura_cost, gpu_guard, joint_dispatch_pilot, joint_projection_backend,
        joint_stage_b_head, joint_stageb_resources, residency_map, tessera_reader,
    )

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *a, **k: None)
    monkeypatch.setattr(joint_stageb_resources, "enforce_device_policy", lambda *a, **k: {})
    monkeypatch.setattr(residency_map, "bind_residency_manifest", lambda *a, **k: None)
    monkeypatch.setattr(residency_map, "residency_report", lambda: None)
    monkeypatch.setattr(joint_projection_backend, "executing_image", lambda: "fixture-image")
    monkeypatch.setattr(joint_projection_backend, "prewarm_projection_backend",
                        lambda *a, **k: SimpleNamespace(identity=None))
    monkeypatch.setattr(tessera_reader, "load_declared_reader", lambda *a, **k: None)
    monkeypatch.setattr(aura_cost, "_aura_source_sha256", lambda: "b" * 64)
    monkeypatch.setattr(quantum, "GpuPowerSampler", _Sampler)
    monkeypatch.setattr(quantum, "bind_joint_served_quantizer", lambda formats: {})
    monkeypatch.setattr(joint_dispatch_pilot, "pilot_binding", lambda *a, **k: None)
    monkeypatch.setattr(quantum, "launch_kda_capture_kernel", lambda *a, **k: None)
    monkeypatch.setattr(quantum, "quantum_retained_state", lambda *a, **k: SimpleNamespace(
        operator_windows=None, retained_budget=None, source_bytes=0))
    monkeypatch.setattr(quantum, "quantum_layer_roster", lambda *a, **k: SimpleNamespace(
        names=[], linears={}, render_formats={}))
    monkeypatch.setattr(quantum, "resolve_quantum_windows", lambda *a, **k: [])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    runner = SimpleNamespace(device="cpu", shutdown=lambda: None)
    source = {"identity": "stub"}
    monkeypatch.setattr(quantum, "build_quantum_source_runner", lambda *a, **k: runner)
    monkeypatch.setattr(quantum, "_build_quantum_source_identity", lambda *a, **k: source)
    head_slice = {"intake": {"layer_formats": {}}}
    monkeypatch.setattr(joint_stage_b_head, "read_quantum_head_slice",
                        lambda *a, **k: (head_slice, {}, {}))
    monkeypatch.setattr(joint_stage_b_head, "head_slice_limits", lambda *a: {})
    monkeypatch.setattr(joint_stage_b_head, "read_prepared_head",
                        lambda *a: {"formats_by_qname": {}})
    monkeypatch.setattr(joint_stage_b_head, "load_quantum_head",
                        lambda *a, **k: SimpleNamespace(
                            completion={"source_model_identity": source}, cache=object(),
                            formats_by_qname={}, calibration_ids=torch.zeros((1, 1)),
                            calibration={}, producer_implementation_sha256="b" * 64,
                            units=0, measured_cells=0, progress_units=0,
                            identity_cache_bytes=None, digest_cache_bytes=None))
    root = tmp_path / "layer-044"
    record = _record(tmp_path, counters=str(root / "counters.json"),
                     results=str(root / "results.json"),
                     cost_payload=str(root / "cost.pkl"))
    record.update(chunks=[], executable_readset={"head_slice": {}})
    config = {"execution": {"production_act_scales": "0"}, "max_gpu_bytes": 1}
    return lambda: quantum.run_layer_quantum(
        config, record=record, adjoint_slice={}, plan_sha256="c" * 64,
        prepared={}, output_root=tmp_path)


@pytest.mark.parametrize("entry", ["stage-a", "stage-b"])
@pytest.mark.parametrize("prior", [None, "", "1"], ids=["unset", "empty", "set"])
@pytest.mark.parametrize("exit_path", ["success", "early-error", "workload-error"])
def test_entry_restores_activation_scale_environment(
        tmp_path, monkeypatch, entry, prior, exit_path):
    from prismaquant import matmul_arithmetic

    if entry == "stage-a":
        _captured, invoke = _stage_a_run_stub(tmp_path, monkeypatch, tmp_path / "capture")
        run = lambda: invoke(plan_budget=_prefetch_budget())[0]
        module, core = stage_a, "run_adjoint_capture_core"
        payload = {"stride": {"value": 2, "source": None,
                              "boundaries": [], "max_chain_layers": 1},
                   "checkpoints": []}
    else:
        run = _quantum_run_stub(tmp_path, monkeypatch)
        module, core = quantum, "run_layer_quantum_core"
        payload = {"costs": {}}

    observed = []

    def workload(*args, **kwargs):
        observed.append(os.environ.get(ACTIVATION_SCALE_ENV))
        if exit_path == "workload-error":
            raise WorkloadError("stub workload failed")
        return payload

    def set_threads(count):
        assert os.environ[ACTIVATION_SCALE_ENV] == "0"
        if exit_path == "early-error":
            raise WorkloadError("after setting the activation policy")

    monkeypatch.setattr(module, core, workload)
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    monkeypatch.setattr(matmul_arithmetic, "pin_matmul_arithmetic", lambda: None)
    if prior is None:
        monkeypatch.delenv(ACTIVATION_SCALE_ENV, raising=False)
    else:
        monkeypatch.setenv(ACTIVATION_SCALE_ENV, prior)

    if exit_path == "success":
        assert run()["passed"]
    else:
        with pytest.raises(WorkloadError):
            run()
    assert observed == ([] if exit_path == "early-error" else ["0"])
    assert os.environ.get(ACTIVATION_SCALE_ENV) == prior
    assert (ACTIVATION_SCALE_ENV in os.environ) == (prior is not None)
