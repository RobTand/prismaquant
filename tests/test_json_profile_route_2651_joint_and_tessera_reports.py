import builtins
from contextlib import nullcontext
import hashlib
import json
from types import SimpleNamespace

import torch
from tessera import image_metadata

from prismaquant import (
    digests,
    joint_aura_retained_budget_transition as transition,
    joint_cost_quantum as quantum,
    joint_cost_stage_a as stage_a,
    tessera_calibration_cache as calibration,
    tessera_campaign as campaign,
    tessera_hessian as hessian,
    tessera_joint_aura as aura,
    tessera_route_receipt as receipt,
)


def _prepare_progress(tmp_path, monkeypatch):
    name, fmt = "model.layers.0.a", "TESSERA_E4M3_K1_R1024"
    weight = torch.ones(4, 4, dtype=torch.bfloat16)
    render, wire = tmp_path / "render.pt", tmp_path / "wire"
    torch.save(weight, render)
    wire.write_bytes(b"fixture")
    anchor = dict(qname=name, family="TESSERA_E4M3_K1", format_name=fmt,
                  body_rate_q256=1024, dloss=0.0, dloss_stderr=0.0,
                  memory_bytes=32, bits_per_param=16.0, seconds=0.0,
                  activation_contract="fp8_per_token_dynamic", activation_quantized=True,
                  wire_bytes=7)
    cell = dict(render=str(render), wire=str(wire), anchor=anchor, render_origin="encoded",
                record=dict(blob_bytes=7, blob_sha256=hashlib.sha256(b"fixture").hexdigest()))
    capture_path = tmp_path / "capture.json"
    capture_path.write_bytes(b"{}")
    capture = dict(path=str(capture_path), sha256=hashlib.sha256(b"{}").hexdigest())
    identity = dict(max_act_rows=4, calibration={"draw": "fixture"})
    data = SimpleNamespace(
        payload={"provenance": {"calibration_cache": capture,
                                "hessian": {"calibration_identity": identity["calibration"]}}},
        inputs={"census": {"path": "census"}},
        census=dict(model_load_contract={}, attention_implementation="eager"),
        manifest={"identity": {"calibration": identity["calibration"]}},
        cells={(name, fmt): cell}, formats_by_qname={name: [fmt]},
    )
    context = SimpleNamespace(schedule_prefetch=lambda *_: None, install=lambda *a, **k: None,
                              unload=lambda *_: None)
    runner = SimpleNamespace(model=torch.nn.Module(), context=context, source_layers=(0,),
                             prefetch_lookahead=1, require_prefetched_residency=True,
                             device="cpu", profile=None, layer_index_for_qname=lambda _: 0)
    monkeypatch.setattr(calibration, "require_capture_contract", lambda *a, **k: {"identity": identity})
    monkeypatch.setattr(calibration, "capture_identity", lambda *a, **k: identity)
    monkeypatch.setattr(calibration, "prefetch_capture", lambda *a, **k: (({}, {}, {}, {}), {}))
    monkeypatch.setattr(hessian, "activation_source", lambda *a: None)
    monkeypatch.setattr(aura, "calibrated_maxima", lambda *a: ({}, {}))
    monkeypatch.setattr(aura, "_live_targets", lambda *a: {name: SimpleNamespace(weight=weight)})
    monkeypatch.setattr(campaign, "bind_checkpoint_unit_identity", lambda *a, **k: nullcontext())
    monkeypatch.setattr(aura, "verify_anchor_render", lambda *a, **k: {"render_origin": "encoded"})
    aura.prepare_cache(runner, data, capture=capture, max_render_bytes=4096, file_load_workers=1)
    return {"qualified_layer": 0, "qualified_cells": 1, "total_cells": 1,
            "prefetch": {"entries": 1, "resident_bytes": render.stat().st_size,
                         "loaded": 1, "misses": 0}}


def test_listed_calls_use_insertion_ascii_spaced_lax(tmp_path, monkeypatch, capsysbinary):
    profile_outputs, printed, missing = [], [], []
    real_text = digests.JsonProfile.text

    def observe_text(self, value):
        text = real_text(self, value)
        if self is digests.INSERTION_ASCII_SPACED_LAX:
            profile_outputs.append(text)
        return text

    def observe_print(text, **kwargs):
        printed.append((text, kwargs))
        builtins.print(text, **kwargs)

    def check_owner(label, text):
        if not any(text is output for output in profile_outputs):
            missing.append(label)
        profile_outputs.clear()

    def check_print(label, expected, **kwargs):
        assert capsysbinary.readouterr().out == json.dumps(expected).encode() + b"\n", label
        (text, options), = printed
        printed.clear()
        assert options == kwargs, label
        check_owner(label, text)

    monkeypatch.setattr(digests.JsonProfile, "text", observe_text)
    for module in (transition, quantum, stage_a, aura):
        monkeypatch.setattr(module, "print", observe_print, raising=False)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.delenv("FLUX_JOB_ID", raising=False)
    plan_args = ["--plan", str(tmp_path / "plan"), "--plan-sha256", "a" * 64]
    prepared_args = ["--prepared", str(tmp_path / "prepared"), "--prepared-sha256", "b" * 64]
    job_args = [*plan_args, *prepared_args, "--output-root", str(tmp_path)]
    sample = {"label": "café", "estimate": float("nan")}
    monkeypatch.setattr(aura, "load_joint_anchor_plan", lambda *a, **k: {"output_root": tmp_path})

    binding = {"path": str(tmp_path / "café"), "sha256": "c" * 64}
    monkeypatch.setattr(transition, "create_transition", lambda **k: binding)
    transition_args = ["--output", binding["path"]]
    for name in ("prepared-plan", "run-plan", "prepared", "campaign-identity"):
        transition_args.extend([f"--{name}", str(tmp_path / name), f"--{name}-sha256", "a" * 64])
    assert transition.main(transition_args) == 0
    check_print("transition", binding)

    record_path = tmp_path / "quantum"
    record_bytes = b"fixture"
    quantum_result = dict(quantum_id="café", passed=True, status="complete", units_done=1,
                          units_total=1, counters={"path": str(tmp_path / "counters.json")})
    monkeypatch.setattr(quantum, "verify_quantum_identity", lambda **k: ({}, {}, record_bytes))
    monkeypatch.setattr(quantum, "require_chain_readset", lambda *a, **k: None)
    monkeypatch.setattr(quantum, "run_layer_quantum", lambda *a, **k: quantum_result)
    assert quantum.main([*job_args, "--quantum", str(record_path), "--quantum-sha256", "c" * 64,
                         "--adjoint-slice", str(tmp_path / "slice"), "--adjoint-slice-sha256", "d" * 64]) == 0
    check_print("quantum", {"schema": "prismaquant.joint_layer_quantum.completion.v1",
                           "quantum_record": {"path": str(record_path), "sha256": hashlib.sha256(record_bytes).hexdigest(),
                                              "bytes": len(record_bytes)}, **quantum_result})

    report = {"command": "cpu-input-preflight", **sample}
    monkeypatch.setattr(stage_a, "preflight_adjoint_inputs", lambda *a, **k: report)
    assert stage_a.main([*job_args, "--cpu-input-preflight"]) == 0
    check_print("preflight", report)
    capture_result = {"command": "capture", "passed": False, "plane_comparison": sample}
    monkeypatch.setattr(stage_a, "run_adjoint_capture", lambda *a, **k: {"hidden": True, **dict(reversed(capture_result.items()))})
    assert stage_a.main(job_args) == stage_a.EXIT_FAILURE
    check_print("capture", capture_result)

    monkeypatch.setattr(aura, "build_source_digest_cache", lambda *a: sample)
    assert aura.main(["identity", "--model", str(tmp_path), "--out", str(tmp_path / "identity")]) == 0
    check_print("identity", sample)
    summary = dict(units_read=1, cells=1, renders_synthesized_now=1,
                   seconds=float("nan"), render_origins={"synthesized_from_wire": 1})
    monkeypatch.setattr(aura, "synthesize_renders", lambda *a, **k: {"hidden": True, **dict(reversed(summary.items()))})
    assert aura.main(["synthesize", *plan_args]) == 0
    check_print("synthesize", summary)
    result = dict(command="run", passed=True, units=1, measured_cells=1,
                  render_origins={"encoded": 1}, render_comparisons={"independent_render_vs_wire": 1})
    monkeypatch.setattr(aura, "execute", lambda *a, **k: {"hidden": True, **dict(reversed(result.items()))})
    assert aura.main(["run", *plan_args]) == 0
    check_print("execute", result)
    check_print("qualification", _prepare_progress(tmp_path, monkeypatch), flush=True)

    image = "example/runtime@sha256:" + "e" * 64
    launch = {"schema": "tessera.runtime_image/1", "requested": image, "requested_tag": None,
              "pinned": image, "required": image, "gated": True, "present": True,
              "repo_digests": [image], "resolved_digest": image.rsplit("@", 1)[1],
              "resolved_reference": image, "local_id": "café", "refused": False,
              "reason": "pinned", "fix": None}
    image_key, declaration_key = image_metadata.CENSUS_IMAGE_ENV, image_metadata.CENSUS_DECLARATION_ENV
    expected_env = {image_key: image, declaration_key: json.dumps(launch)}
    real_reference = image_metadata.declared_reference
    declaration = real_reference(image, env=expected_env)
    seen = []

    def observe_reference(requested, *, env):
        seen.append(env)
        return real_reference(requested, env=env)

    monkeypatch.setattr(image_metadata, "declared_reference", observe_reference)
    receipt._check_runtime_image_declaration({"runtime_image_declaration": declaration}, image)
    received, = seen
    assert received[image_key] == image
    assert received[declaration_key].encode() == expected_env[declaration_key].encode()
    check_owner("receipt", received[declaration_key])
    assert not missing, f"Use text from INSERTION_ASCII_SPACED_LAX at these calls: {missing}"
