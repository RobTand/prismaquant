"""The priced activation path runs one arithmetic, and it says which.

The retained 84-group differential (``quantizer-84g-sparky-a542.json``) priced
24 of 172,032 probed elements one E2M1 code away from the operator a serve runs:
the kernel takes its ``outputScale`` through ``rcp.approx.ftz.f32`` where
PrismaQuant's Torch re-implementation divides by the used scale.  A row priced
by the model and a row priced by the operator are therefore different objects,
and the three properties below are what keeps them from being confused:

* a rung whose measurement contract IS the served quantiser refuses when no
  registered-operator binding stands behind it -- it does not fall back;
* the binding is explicit, resolved once, and carried in the row identity;
* the operator leg dequantises with ``stored_scale / G``, which is the contract's
  own rule and the one thing a first draft of the closure note got wrong.

Nothing here needs a GPU or the vLLM extension; that is the point of splitting
the dequantisation arithmetic out of the operator call.  What a CPU test cannot
establish -- that the operator's own codes are the served codes -- is measured on
the retained real-wire inputs, not here.
"""
from __future__ import annotations

import pytest
import torch

from nvfp4_served_qdq_fixtures import cpu_kernels
from prismaquant import nvfp4_activation_contract as owner


def _served_contract(**overrides) -> owner.StaticActivationContract:
    fields = {"measured_as_served": True}
    fields.update(overrides)
    return owner.StaticActivationContract(**fields)


def _registered_identity(**overrides) -> owner.ServedQuantizerIdentity:
    fields = {"backend": owner.SERVED_QUANTIZER_BACKEND_REGISTERED_OP,
              "op": owner.SERVED_QUANTIZER_OP, "platform": "sm_121",
              "torch": "2.13.0+cu130", "vllm": "0.1.dev20073",
              "image_content_sha256": "d" * 64,
              "dequant_kernel": owner.SERVED_QUANTIZER_DEQUANT_KERNEL}
    fields.update(overrides)
    return owner.ServedQuantizerIdentity(**fields)


def _e4m3_byte_value(byte: int) -> float:
    return float(torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e4m3fn).float())


def test_a_served_rung_refuses_without_a_binding():
    owner._reset_served_quantizer_identity_for_tests()
    contract = _served_contract()

    with pytest.raises(owner.ServedQuantizerUnboundError) as refusal:
        contract.quantize_dequantize(torch.zeros(2, owner.FP4_GROUP_SIZE), 1.0)

    message = str(refusal.value)
    assert owner.SERVED_QUANTIZER_OP in message
    assert "prismaquant#567" in message


def test_the_legacy_model_path_is_a_binding_not_a_fallback():
    owner.bind_served_quantizer_identity(
        identity=owner.ServedQuantizerIdentity(
            backend=owner.SERVED_QUANTIZER_BACKEND_MODEL), require=False)
    contract = _served_contract()
    x = torch.tensor([[0.25, -0.25, 3.5, -3.5] * 4], dtype=torch.bfloat16)

    assert torch.equal(contract.quantize_dequantize(x, 1.5),
                       owner.nvfp4_activation_qdq_served(x, 1.5))


def test_a_registered_binding_dispatches_to_the_operator_leg(monkeypatch):
    sentinel = object()
    calls: list[tuple] = []

    def _leg(x, g):
        calls.append((x, g))
        return sentinel

    monkeypatch.setattr(owner, "_nvfp4_activation_qdq_registered_op", _leg)
    owner._reset_served_quantizer_identity_for_tests()
    owner.bind_served_quantizer_identity(
        identity=_registered_identity(), require=True)

    x = torch.zeros(1, owner.FP4_GROUP_SIZE)
    assert _served_contract().quantize_dequantize(x, 3.0) is sentinel
    assert calls == [(x, 3.0)]


def test_the_operator_leg_is_cuda_only():
    """A CPU tensor cannot be priced by a CUDA operator, and says so."""
    with pytest.raises(owner.ServedQuantizerUnboundError) as refusal:
        owner._nvfp4_activation_qdq_registered_op(
            torch.zeros(1, owner.FP4_GROUP_SIZE), 1.0)

    assert "cuda" in str(refusal.value).lower()


def test_codes_dequantise_with_the_global_scale_divided_out():
    """``used_scale = stored_scale / G`` -- the bug the first note shipped.

    Code 1 is magnitude 0.5 and code 9 is its negative.  With a stored e4m3 byte
    of 1.0 and ``G = 2.0`` the served value is ``0.5 * (1.0 / 2.0) = 0.25``;
    multiplying by the stored scale alone would give 0.5, i.e. ``G`` times too
    large.
    """
    codes = torch.tensor([[1, 9, 0, 8] + [1] * (owner.FP4_GROUP_SIZE - 4)],
                         dtype=torch.long)
    stored = torch.full((1, 1), _e4m3_byte_value(0x38))  # UE4M3 1.0
    assert stored.item() == 1.0

    out = owner._nvfp4_dequantize_registered_codes(codes, stored, 2.0)

    assert out.shape == codes.shape
    assert out[0, 0].item() == pytest.approx(0.25)
    assert out[0, 1].item() == pytest.approx(-0.25)
    assert out[0, 2].item() == 0.0 and out[0, 3].item() == 0.0


def test_a_zero_stored_scale_dequantises_to_signed_zero():
    codes = torch.tensor([[1, 9] + [15] * (owner.FP4_GROUP_SIZE - 2)],
                         dtype=torch.long)
    stored = torch.zeros((1, 1))

    out = owner._nvfp4_dequantize_registered_codes(codes, stored, 0.5)

    assert torch.equal(out, torch.zeros_like(out))


def test_a_real_packed_shape_with_uint8_nibbles_dequantises():
    """The shape the operator actually hands over, not a fixture-shaped one.

    ``[3, 32]`` is M=3 and K=32, i.e. TWO 16-element groups per row: the mask
    must repeat out to the element axis, and the packed nibbles arrive as
    uint8, which ``index_select`` refuses as an index.
    """
    packed = torch.tensor(
        # element 0 is the LOW nibble of byte 0, so code 1 first means 0x01.
        [[0x01, 0x00] * 8, [0x09, 0x00] * 8, [0x00, 0x00] * 8],
        dtype=torch.uint8)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(3, 32)
    stored = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    out = owner._nvfp4_dequantize_registered_codes(codes, stored, 2.0)

    assert out.shape == (3, 32)
    assert out[0, 0].item() == pytest.approx(0.5 * 0.5)   # code 1, used 0.5
    assert out[1, 0].item() == pytest.approx(-0.5 * 0.5)  # code 9 holds the sign
    assert torch.count_nonzero(out[1, 16:]).item() == 0   # zero scale group


def test_codes_and_scale_plane_must_cover_each_other():
    codes = torch.zeros(3, 32, dtype=torch.uint8)

    with pytest.raises(ValueError, match="stored scale of"):
        owner._nvfp4_dequantize_registered_codes(codes, torch.ones(3, 3), 1.0)
    with pytest.raises(ValueError, match="whole 16-element groups"):
        owner._nvfp4_dequantize_registered_codes(torch.zeros(3, 30), torch.ones(3, 2), 1.0)


def test_the_production_composition_of_grouped_scale_and_dequant_runs():
    """The two production functions, composed exactly as the operator leg does.

    ``_nvfp4_activation_qdq_registered_op`` reads each group's amax through
    ``nvfp4_group_stored_scale``, whose ``keepdim=True`` leaves the plane as
    ``(..., groups, 1)``, and hands that plane straight to the dequantiser.  No
    caller-side squeeze is involved here -- this is the real shape the real code
    path produces -- so the dequantiser accepting it is what makes the priced
    operator leg runnable at all.
    """
    rows, groups, g = 3, 2, 2.0
    x = torch.zeros(rows, groups * owner.FP4_GROUP_SIZE)
    x[:, 0] = 1.0                                   # group 0: amax 1.0
    x[:, owner.FP4_GROUP_SIZE] = 0.5                # group 1: amax 0.5
    grouped = x.reshape(-1, groups, owner.FP4_GROUP_SIZE)

    stored = owner.nvfp4_group_stored_scale(grouped, g).float()
    assert stored.shape == (rows, groups, 1)

    codes = torch.zeros(rows, groups * owner.FP4_GROUP_SIZE, dtype=torch.long)
    codes[:, 0] = 7                                  # magnitude 6.0, group 0
    codes[:, owner.FP4_GROUP_SIZE] = 9               # -0.5, group 1

    out = owner._nvfp4_dequantize_registered_codes(codes, stored, g)

    assert out.shape == codes.shape
    assert out[0, 0].item() == pytest.approx(6.0 * float(stored[0, 0, 0]) / g)
    assert out[0, owner.FP4_GROUP_SIZE].item() == pytest.approx(
        -0.5 * float(stored[0, 1, 0]) / g)
    assert torch.count_nonzero(out[:, 1:owner.FP4_GROUP_SIZE]).item() == 0


def test_an_unknown_backend_is_refused_not_fallen_through(monkeypatch):
    owner._reset_served_quantizer_identity_for_tests()
    # The validator refuses it at the binding seam, where a run would notice it
    # before any row exists; the dispatch refusal below is the second layer for
    # an identity that reached a contract directly.
    with pytest.raises(owner.ServedQuantizerUnboundError, match="unknown served-quantizer"):
        owner.bind_served_quantizer_identity(
            identity=owner.ServedQuantizerIdentity(backend="something_else"),
            require=False)

    contract = owner.StaticActivationContract(
        measured_as_served=True,
        served_quantizer=owner.ServedQuantizerIdentity(backend="something_else"))
    with pytest.raises(owner.ServedQuantizerUnboundError, match="unknown served-quantizer"):
        contract.quantize_dequantize(torch.zeros(1, owner.FP4_GROUP_SIZE), 1.0)


def test_require_refuses_an_explicit_model_identity():
    owner._reset_served_quantizer_identity_for_tests()

    with pytest.raises(owner.ServedQuantizerUnboundError, match="requires the registered"):
        owner.bind_served_quantizer_identity(
            identity=owner.ServedQuantizerIdentity(
                backend=owner.SERVED_QUANTIZER_BACKEND_MODEL),
            require=True, context="priced run")


def test_rebinding_a_different_arithmetic_after_pricing_refuses():
    owner.bind_served_quantizer_identity(
        identity=owner.ServedQuantizerIdentity(
            backend=owner.SERVED_QUANTIZER_BACKEND_MODEL), require=False)

    with pytest.raises(owner.ServedQuantizerUnboundError, match="cannot be"):
        owner.bind_served_quantizer_identity(
            identity=owner.ServedQuantizerIdentity(
                backend=owner.SERVED_QUANTIZER_BACKEND_REGISTERED_OP,
                op=owner.SERVED_QUANTIZER_OP, platform="sm_121",
                torch="2.13.0+cu130", vllm="0.1.dev20073",
                image_content_sha256="d" * 64,
                dequant_kernel=owner.SERVED_QUANTIZER_DEQUANT_KERNEL),
            require=False)


def test_a_served_binding_without_provenance_is_refused():
    owner._reset_served_quantizer_identity_for_tests()
    with pytest.raises(owner.ServedQuantizerUnboundError, match="must name"):
        owner.bind_served_quantizer_identity(
            identity=_registered_identity(op=None), require=True, context="test")

    for missing in ("platform", "vllm", "image_content_sha256"):
        fields = {"op": owner.SERVED_QUANTIZER_OP, "platform": "sm_121",
                  "torch": "2.13.0+cu130", "vllm": "0.1.dev20073",
                  "image_content_sha256": "d" * 64,
                  "dequant_kernel": owner.SERVED_QUANTIZER_DEQUANT_KERNEL}
        fields.pop(missing)
        owner._reset_served_quantizer_identity_for_tests()

        with pytest.raises(owner.ServedQuantizerUnboundError) as refusal:
            owner.bind_served_quantizer_identity(
                identity=owner.ServedQuantizerIdentity(
                    backend=owner.SERVED_QUANTIZER_BACKEND_REGISTERED_OP, **fields),
                require=True, context="test")
        assert missing in str(refusal.value)


def test_resolution_is_cached_and_probes_the_extension_once(monkeypatch):
    owner._reset_served_quantizer_identity_for_tests()
    probes: list[int] = []

    def _probe():
        probes.append(1)
        return True

    monkeypatch.setattr(owner, "_register_served_quantizer_op", _probe)
    # A registered operator also needs the leg's Triton dequantisation kernels
    # (#1211); the CPU stand-in keeps this resolution logic tested without
    # Triton (#1224).
    monkeypatch.setattr(owner, "_served_dequant_kernels", cpu_kernels)
    first = owner.resolve_served_quantizer_identity(require=True, context="test")
    second = owner.resolve_served_quantizer_identity(require=True, context="test")

    assert first is second
    assert len(probes) == 1
    assert first.backend == owner.SERVED_QUANTIZER_BACKEND_REGISTERED_OP
    assert first.as_record()["schema"] == owner.SERVED_QUANTIZER_IDENTITY_SCHEMA
    assert first.as_record()["op"] == owner.SERVED_QUANTIZER_OP


def test_require_refuses_an_unregistered_operator_by_name(monkeypatch):
    owner._reset_served_quantizer_identity_for_tests()
    monkeypatch.setattr(owner, "_register_served_quantizer_op", lambda: False)

    resolved = owner.resolve_served_quantizer_identity(require=False, context="test")
    assert resolved.backend == owner.SERVED_QUANTIZER_BACKEND_MODEL
    assert resolved.op is None

    with pytest.raises(owner.ServedQuantizerUnboundError) as refusal:
        owner.bind_served_quantizer_identity(require=True, context="joint AURA")

    assert "joint AURA" in str(refusal.value)


def test_the_identity_carries_the_axes_a_reader_needs(monkeypatch):
    owner._reset_served_quantizer_identity_for_tests()
    monkeypatch.setattr(owner, "_register_served_quantizer_op", lambda: True)
    monkeypatch.setattr(owner, "_served_dequant_kernels", cpu_kernels)
    monkeypatch.setenv("PRISMAQUANT_CONTAINER_CONTENT_SHA256", "a" * 64)

    record = owner.resolve_served_quantizer_identity(
        require=True, context="test").as_record()

    assert record["image_content_sha256"] == "a" * 64
    assert record["torch"] == str(torch.__version__)
    assert set(record) == {"schema", "backend", "op", "platform", "torch",
                           "torch_git", "vllm", "image_content_sha256",
                           "dequant_kernel"}


def test_the_operator_leg_derives_the_block_scale_in_fp32():
    """The stored UE4M3 byte is ``amax / 6 * G`` in fp32, not in the rows' bf16.

    Measured on the retained 84 groups: ``amax = 0.76171875`` at
    ``G = 3.5720930099487305`` stores byte 47 (0.46875) in fp32 -- which is the
    byte the kernel's own plane carries -- while the same arithmetic in bf16
    stores byte 46 (0.4375), because ``amax / 6 * G`` passes through a bf16
    rounding to 0.453125, exactly the e4m3 tie that rounds to even.  The
    operator leg must therefore take the fp32 derivation; the rows' own dtype is
    what moved the dequantised value.
    """
    g = 3.5720930099487305
    rows = torch.full((1, owner.FP4_GROUP_SIZE), 0.76171875, dtype=torch.bfloat16)
    grouped = rows.reshape(-1, 1, owner.FP4_GROUP_SIZE)

    bf16_plane = owner.nvfp4_group_stored_scale(grouped, g)
    fp32_plane = owner.nvfp4_group_stored_scale(grouped.float(), g)
    assert int(bf16_plane.view(torch.uint8).reshape(-1)[0]) == 46
    assert int(fp32_plane.view(torch.uint8).reshape(-1)[0]) == 47

    plane = owner._nvfp4_registered_stored_plane(rows, g)

    assert plane.shape == (1, 1, 1)
    assert plane.dtype == torch.float32
    assert torch.equal(plane, fp32_plane.float())
    assert not torch.equal(plane, bf16_plane.float())
