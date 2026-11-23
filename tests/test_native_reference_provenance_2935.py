"""Check native reference provenance with synthetic CPU inputs and receipts."""
import builtins
import hashlib
import json
from dataclasses import replace
from importlib.util import resolve_name
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import format_registry, joint_aura, perturbed_x_cache, tessera_runtime_contract
from prismaquant import nvfp4_activation_contract as quantizer
from prismaquant.digests import DIRECT_ASCII_LAX
from prismaquant.joint_aura_rows import joint_identity_sha256, make_joint_aura_entry
from prismaquant.native_execution_binding import freeze_execution_panel, require_reference_quantizer
from prismaquant.native_operator_panel import consume_native_receipt, freeze_native_panel, prepare_native_inputs
from prismaquant.native_panel_kind import DENSE_EXECUTION_RECEIPT_SCHEMA, bind_execution_receipt
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.streamed_model_identity import STREAMED_MODEL_IDENTITY_SCHEMA

RECORDS = b'''[{"activation":{"static_contract":{"measured_as_served":true}},"data":{},"name":"missing","probe":null},{"activation":{"static_contract":{"measured_as_served":true}},"data":{"reference_served_quantizer":[]},"name":"malformed","probe":null},{"activation":{"static_contract":{"measured_as_served":true}},"data":{"reference_served_quantizer":{"backend":"model"}},"name":"wrong-backend","probe":null},{"activation":{"static_contract":{"measured_as_served":true}},"data":{"reference_served_quantizer":{"backend":"registered_scaled_fp4_quant","dequant_kernel":null,"image_content_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","op":"scaled_fp4_quant","platform":"fixture","schema":"prismaquant.served_quantizer_identity.v1","torch":"fixture","torch_git":"fixture","vllm":"fixture"}},"name":"other-build","probe":{"arithmetic":{"served_quantizer":{"backend":"registered_scaled_fp4_quant","dequant_kernel":null,"image_content_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","op":"scaled_fp4_quant","platform":"fixture","schema":"prismaquant.served_quantizer_identity.v1","torch":"fixture","torch_git":"fixture","vllm":"fixture"}}}},{"activation":{"static_contract":{"measured_as_served":true}},"data":{"reference_served_quantizer":{"backend":"registered_scaled_fp4_quant","dequant_kernel":null,"image_content_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","op":"scaled_fp4_quant","platform":"fixture","schema":"prismaquant.served_quantizer_identity.v1","torch":"fixture","torch_git":"fixture","vllm":"fixture"}},"name":"matching","probe":{"arithmetic":{"served_quantizer":{"backend":"registered_scaled_fp4_quant","dequant_kernel":null,"image_content_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","op":"scaled_fp4_quant","platform":"fixture","schema":"prismaquant.served_quantizer_identity.v1","torch":"fixture","torch_git":"fixture","vllm":"fixture"}}}},{"activation":{"quantizes_input":false},"data":{"reference_served_quantizer":[]},"name":"legacy","probe":null}]
'''
STAMP = "reference_served_quantizer"
CASES = {case["name"]: case for case in json.loads(RECORDS)}
REFERENCE = CASES["matching"]["data"][STAMP]
MISSING = "static native references require their registered served quantizer identity"
DIFFERS = "native reference quantizer differs from actual joint quality arithmetic"


def test_design_note_and_architecture_describe_the_shipped_stamp():
    docs = Path(__file__).resolve().parents[1] / "docs"
    note = (docs / "design/native_panel_served_quantizer_identity_2026-09-19.md").read_text(encoding="utf-8")
    assert "**Status: shipped.**" in note, "The design note must identify the stamp as shipped."
    assert f"`{STAMP}`" in note
    architecture = (docs / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for label, terms in (
        ("Preparation", (STAMP, "only for a measured-as-served static", "effective_served_quantizer_identity",
                         "contract binding takes priority over the process binding")),
        ("Ordinary freeze", ("must match the validated joint AURA probe arithmetic", "carries the stamp")),
        ("Execution-first freeze", ("registered_scaled_fp4_quant", "no joint quality row",
                                    "does not compare arithmetic", "carries the stamp")),
        ("Receipt admission", ("Direct admission checks", "Late-bound admission resolves the binding first",
                               "final joint arithmetic", "makes the same comparison", "refuse an unstamped")),
    ):
        _, found, text = architecture.partition(f"- **{label}.**")
        assert found, label
        paragraph = " ".join(text.split("\n\n", 1)[0].split())
        for term in terms:
            assert term in paragraph, f"{label}: Include {term}."


def test_docs_describe_reference_quantizer_stamp_validation():
    docs = Path(__file__).resolve().parents[1] / "docs"
    for path, start, end in (
        ("design/native_panel_served_quantizer_identity_2026-09-19.md", "## Freeze", "The exact QDQ gate"),
        ("ARCHITECTURE.md", "- **Ordinary freeze.**", "- **One Triton trap"),
    ):
        text = (docs / path).read_text(encoding="utf-8").partition(start)[2].partition(end)[0]
        for field in ("prismaquant.served_quantizer_identity.v1", "scaled_fp4_quant", "platform", "torch",
                      "torch_git", "vllm", "image_content_sha256"):
            assert f"`{field}`" in text, f"{path}: Include `{field}`."


def test_reference_quantizer_preserves_recorded_outcomes():
    assert hashlib.sha256(RECORDS).hexdigest() == "8b37bb4e2ea60ec9bd2cddae7f999ce973bf0a9d2b9ebebe7c719a5848751cb8"
    for name, digest in zip(CASES, (
        *("023abb006582645ad01f70f366edc39a9bca315a90c17be32c5c105dfeb3da6f",) * 3,
        "d837862c658560de16eabc45c58f5846d905d20189fa3324a567f7ee41c958b3",
        "31d5b350d5ffd2e575ba0949bcee50e2119d1a054ce3d655afbf1d38086d74ff",
        "b39a0dc5d98baf0c321d5e59af9fc4cd05c2e6403ffc95051eba0455532cbd45",
    ), strict=True):
        case = CASES[name]
        try:
            outcome = {"return": require_reference_quantizer(case["data"], case["activation"], case["probe"])}
        except Exception as error:
            kind = type(error)
            outcome = {"type": f"{kind.__module__}.{kind.__qualname__}", "message": str(error)}
        raw = (json.dumps(outcome, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        assert hashlib.sha256(raw).hexdigest() == digest, name


def _prepare(monkeypatch, mode):
    from tessera import cached_unit, unit_artifact

    reference = quantizer.ServedQuantizerIdentity(**REFERENCE)
    process = replace(reference, image_content_sha256="b" * 64)
    monkeypatch.setattr(quantizer, "_ACTIVE_SERVED_QUANTIZER", process)
    measured = mode in ("explicit", "process")
    contract = None if mode == "bf16" else quantizer.StaticActivationContract(
        measured_as_served=measured, served_quantizer=reference if mode == "explicit" else None)
    spec = SimpleNamespace(act_quant_changes_input=contract is not None, static_activation_contract=contract,
                           act_bits=4 if contract else 16, act_dtype_name="fixture", act_group_size=16,
                           activation_quantize_dequantize=torch.clone)
    rows = torch.ones(2, 4, dtype=torch.bfloat16)
    device = torch.Tensor.device
    monkeypatch.setattr(torch.Tensor, "device", property(
        lambda tensor: torch.device("cuda") if tensor is rows else device.__get__(tensor)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    cache = object.__new__(ProductionWeightCache)
    cache.activation_max_abs = {"fixture": 1.0} if measured else {}
    monkeypatch.setattr(ProductionWeightCache, "get", lambda *_: SimpleNamespace(to=lambda **_: rows.clone()))
    monkeypatch.setattr(joint_aura, "prefetch_joint_cache", lambda *_, **kw: {})
    monkeypatch.setattr(cached_unit, "verify_cached_unit", lambda *_: None)
    monkeypatch.setattr(unit_artifact, "read_unit_artifact", lambda *_, **kw: rows.clone())
    monkeypatch.setattr(format_registry, "get_format", lambda _: spec)
    monkeypatch.setenv("PRISMAQUANT_PROD_ACT_SCALES", "0")
    monkeypatch.delenv("PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES", raising=False)
    seen = []

    def qdq(x, spec, *_):
        seen.append(spec)
        return x

    monkeypatch.setattr(perturbed_x_cache, "_activation_qdq", qdq)
    table_path = Path(__file__).parent / "fixtures" / "tessera_activation_quantizers_v25.json"
    table = tessera_runtime_contract._parse_activation_quantizers(json.loads(table_path.read_text()), "fixture")
    inputs, _ = prepare_native_inputs(
        cache, rows, rows, unit="fixture", format_name="fixture",
        calibration_receipt={"schema": "prismaquant.calibration_input.v1", "calibration_sha256": "1" * 64,
                             "shape": [1, 4], "dtype": "torch.int64"},
        wire_blob=b"fixture", wire_record={}, encoding_identity={}, prefill_rows=2, decode_rows=1,
        max_resident_bytes=1024, activation_quantizers=table)
    assert len(seen) == (2 if contract else 0)
    assert all(item is spec for item in seen)
    return inputs


def _joined(inputs):
    image = inputs["activation_quantizer_attestation"]["generated"]["image"]
    source = {"config": {"fixture": True}, "weight_map": {"fixture.weight": "fixture.safetensors"},
              "shards": [{"path": "/fixture/fixture.safetensors", "size": 1, "sha256": "2" * 64}]}
    model = dict(source, schema=STREAMED_MODEL_IDENTITY_SCHEMA, source="fixture", resolved_commit=None,
                 content_sha256=DIRECT_ASCII_LAX.sha256(source))
    request = dict(n_probes=2, seed_base=7, token_scope="causal", temperature=1.0,
                   distribution="rademacher", normalization="global_kl_fisher")
    inputs.update(runtime_image=image, probe_request=dict(request, source_model="fixture",
                                                          source_shards={"fixture.safetensors": "2" * 64}))
    arithmetic = dict(joint_aura.arithmetic_identity(torch.bfloat16), served_quantizer=REFERENCE)
    probe = dict(request, schema="prismaquant.joint_aura.probes.v2", source_model=model,
                 calibration_sha256="1" * 64, calibration_shape=[1, 4], calibration_dtype="torch.int64",
                 producer_source_sha256="3" * 64, arithmetic=arithmetic)
    weights = {key: inputs[key] for key in ("source_weight", "rendered_weight")}
    activation = inputs["activation"]
    joint = dict(weights, schema="prismaquant.joint_aura.operator.v2", qname="fixture", format="fixture",
                 activation=activation, arithmetic=arithmetic, probe_identity_sha256=joint_identity_sha256(probe))
    row = make_joint_aura_entry(operator_identity=joint, probe_identity=probe, signed_components=[
        dict(weight=v, activation=0.0, mixed=0.0, total=v) for v in (.1, -.2)])
    route = dict(kind="dense", symbol="fixture.mm", contract=activation["static_contract"]["execution"])
    operator = dict(weights, input_global_scale=activation["input_global_scale"], clip_enabled=False,
                    wire_sha256=inputs["wire"]["blob_sha256"], wire_record_sha256=joint_identity_sha256({}),
                    native_tensors=weights, scheme={"fixture": True}, declared_route=route,
                    activation_contract=route["contract"])
    runtime = dict(schema="tessera.native_dense_runtime.v1", execution=inputs["execution"], image=image)
    return dict(schema="tessera.native_dense_preflight.v1", status="untimed_preparation", operator=operator,
                runtime=runtime, runtime_sha256=joint_identity_sha256(runtime),
                native_tensors_sha256=joint_identity_sha256(weights),
                scheme_sha256=joint_identity_sha256(operator["scheme"])), row


@pytest.mark.parametrize("case", ["backend-only", "wrong-schema", "missing-build", "valid"])
def test_execution_first_freeze_validates_reference_quantizer_stamp(monkeypatch, case):
    with monkeypatch.context() as seam:
        inputs = _prepare(seam, "explicit")
    preflight, _ = _joined(inputs)
    stamp = dict(inputs[STAMP], torch_git=None)
    del stamp["dequant_kernel"]
    if case == "backend-only":
        stamp = {"backend": stamp["backend"]}
    elif case == "wrong-schema":
        stamp["schema"] = "prismaquant.served_quantizer_identity.v2"
    elif case == "missing-build":
        del stamp["vllm"]
    inputs[STAMP] = stamp
    try:
        panel = freeze_execution_panel(inputs, preflight, source_sha256="4" * 64)
    except ValueError as error:
        assert case != "valid", "Freeze must accept the v1 stamp."
        assert str(error) == MISSING
    else:
        assert case == "valid", f"Freeze must refuse the {case} stamp."
        assert panel[STAMP] == stamp


def _receipt(panel, preflight, *, late):
    error = dict(panel["numerics"], status="passed", finite=True, max_normalized_error=0.0, max_abs_error=0.0)
    phases = {name: {**{k: phase[k] for k in ("input", "reference_qdq", "reference_output")},
                     "route": dict(phase["expected_route"], state="served", shape=f"M{phase['m']}:N2:K4"),
                     "numerics": error, "qdq_numerics": error,
                     "measurement": dict(method="cuda_events", sample_unit="single_apply",
                                         samples_ms=[1.0, 2.0, 3.0], warmup_iterations=4)}
              for name, phase in panel["phases"].items()}
    return dict(schema=DENSE_EXECUTION_RECEIPT_SCHEMA if late else "tessera.native_dense_operator_receipt.v1",
                status="timing_admissible", panel=panel, panel_sha256=joint_identity_sha256(panel),
                operator=preflight["operator"], runtime=preflight["runtime"], runtime_sha256=preflight["runtime_sha256"],
                phases=phases, resources=dict(status="incomplete", resident_bytes=64, phases=dict.fromkeys(phases, {})))


@pytest.mark.parametrize("mode", ["explicit", "process", "bf16", "screen", "missing", "malformed", "wrong-backend",
                                  "other-build", "late-other-build", "late-matching"])
def test_native_reference_stamp_and_admission(monkeypatch, tmp_path, mode):
    preparation = mode in ("explicit", "process", "bf16", "screen")
    with monkeypatch.context() as seam:
        inputs = _prepare(seam, mode if preparation else "explicit")
    if preparation:
        if mode in ("bf16", "screen"):
            assert STAMP not in inputs
        else:
            expected = REFERENCE if mode == "explicit" else CASES["other-build"]["probe"]["arithmetic"]["served_quantizer"]
            assert inputs.get(STAMP) == expected
        return

    # Test intake independently of the preparation stamp.
    inputs[STAMP] = REFERENCE
    preflight, row = _joined(inputs)
    final = freeze_native_panel(inputs, preflight, row, cost_sha256="4" * 64)
    raw = freeze_execution_panel(inputs, preflight, source_sha256=final["source_sha256"])
    assert final[STAMP] == raw[STAMP] == REFERENCE
    assert all("served_quantizer" not in record for record in (inputs, final, raw))
    late = mode.startswith("late-")
    case = CASES[mode.removeprefix("late-")]
    final.pop(STAMP)
    final.update(case["data"])
    if case["probe"] and mode.endswith("other-build"):
        joint = final["joint_operator_identity"]
        joint["arithmetic"].update(case["probe"]["arithmetic"])
        final["joint_operator_identity_sha256"] = joint_identity_sha256(joint)
    receipt = _receipt(raw if late else final, preflight, late=late)
    if late:
        receipt = bind_execution_receipt(receipt, final)
    path = tmp_path / "receipt.json"
    payload = json.dumps(receipt).encode()
    path.write_bytes(payload)
    kwargs = dict(expected_sha256=hashlib.sha256(payload).hexdigest(), expected_panel=final)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        absolute = resolve_name("." * level + name, globals["__package__"]) if level else name
        for target in (absolute, *(f"{absolute}.{item}" for item in fromlist or ())):
            assert not target.startswith(("prismaquant.kernels", "vllm", "tessera.serving", "tessera.kernels")), target
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded_import)
        if mode == "late-matching":
            assert consume_native_receipt(path, **kwargs)["status"] == "operator_evidence"
        else:
            message = DIFFERS if mode.endswith("other-build") else MISSING
            with pytest.raises(ValueError, match=f"^{message}$"):
                consume_native_receipt(path, **kwargs)
