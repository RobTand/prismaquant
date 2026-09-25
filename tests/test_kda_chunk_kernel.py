"""The KDA capture kernel against the executed fallback and a float64 oracle (PQ #1199).

``prismaquant.kernels.kda_chunk`` must compute the corrected GLM fallback's
function (``glm_kda_causal_exp_v1``) and round no worse than it. The fallback
and the oracle are the pinned fallback text with the reviewed premask edit,
run in FP32 and in float64 (``experiments/kda_kernel_numerics.py``, which
also checks, inside the image, that this text is what the image executes).

The bound is the harness's: the kernel's error may exceed the fallback's by
at most the two implementations' final FP32 roundings, ``2 u max|ref|`` for
the largest element error and ``2 u`` for the relative Frobenius error
(``u = 2^-24``). The shapes are small; the harness runs the production shape.

The refusal and identity tests need no GPU; the numerics tests skip without
CUDA.
"""
import struct

import pytest
import torch

from experiments import kda_kernel_numerics as numerics
from prismaquant.kernels import kda_chunk

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the KDA kernel runs on CUDA")


def _inputs(case, *, length, heads=2, seed=0, device="cuda"):
    return numerics.draw_case(case, batch=1, length=length, heads=heads, dim=128,
                              device=device, seed=seed)


def _run(function, inputs, dtype):
    return numerics.run(function, inputs, dtype, inputs["stimulus"].to(dtype))


# ---- refusals ----------------------------------------------------------------

@pytest.mark.parametrize("change,message", [
    ({"chunk_size": 32}, "chunk_size=64"),
    ({"initial_state": torch.zeros(1)}, "initial or final state"),
    ({"output_final_state": True}, "initial or final state"),
    ({"use_qk_l2norm_in_kernel": False}, "use_qk_l2norm_in_kernel=True"),
])
def test_calls_outside_the_capture_call_refuse(change, message):
    x = torch.zeros(1, 64, 1, 128)
    arguments = dict(g=x, beta=torch.zeros(1, 64, 1), use_qk_l2norm_in_kernel=True)
    arguments.update(change)
    before = kda_chunk.counts()
    with pytest.raises(kda_chunk.KdaKernelRefused, match=message):
        kda_chunk.chunk_kimi_delta_attention(x, x, x, **arguments)
    assert kda_chunk.counts() == before


def test_cpu_tensors_and_other_head_dims_refuse():
    x = torch.zeros(1, 64, 1, 128)
    with pytest.raises(kda_chunk.KdaKernelRefused, match="CUDA only"):
        kda_chunk.chunk_kimi_delta_attention(x, x, x, g=x, beta=torch.zeros(1, 64, 1),
                                             use_qk_l2norm_in_kernel=True)


@CUDA
def test_other_head_dims_refuse_on_cuda():
    x = torch.zeros(1, 64, 1, 64, device="cuda")
    with pytest.raises(kda_chunk.KdaKernelRefused, match="head dim 128"):
        kda_chunk.chunk_kimi_delta_attention(x, x, x, g=x, beta=torch.zeros(1, 64, 1, device="cuda"),
                                             use_qk_l2norm_in_kernel=True)


# ---- the compiled-code identity ----------------------------------------------

def _elf(sections):
    """A minimal ELF64 little-endian image: ``[(name, type, flags, data)]``."""
    names = b"\0"
    offsets = {}
    for name, *_ in [*sections, (".shstrtab", 3, 0, b"")]:
        offsets[name] = len(names)
        names += name.encode() + b"\0"
    body = b""
    placed = []
    for name, kind, flags, data in sections:
        placed.append((offsets[name], kind, flags, 64 + len(body), len(data)))
        body += data
    placed.append((offsets[".shstrtab"], 3, 0, 64 + len(body), len(names)))
    body += names
    section_offset = 64 + len(body)
    header = (b"\x7fELF\x02\x01\x01" + b"\0" * 9
              + struct.pack("<HHIQQQIHHHHHH", 2, 190, 1, 0, 0, section_offset, 0x5, 64,
                            0, 0, 64, len(placed) + 1, len(placed)))
    table = b"\0" * 64 + b"".join(
        struct.pack("<IIQQQQIIQQ", name, kind, flags, 0, offset, size, 0, 0, 1, 0)
        for name, kind, flags, offset, size in placed)
    return header + body + table


def test_the_code_identity_ignores_debug_sections_and_nothing_else():
    text = (".text._gram_fwd_kernel", 1, 6, b"\x01\x02\x03\x04")
    line = (".debug_line", 1, 0, b"kda_chunk.py\0\x01\x91\xb8\xd6\xd5\x06")
    other_mtime = (".debug_line", 1, 0, b"kda_chunk.py\0\x01\xdc\xba\xd6\xd5\x06")
    base = kda_chunk._code_sha256(_elf([text, line]))
    # Two checkouts of one source: the line tables differ in the file's
    # modification time, and nothing else does.
    assert kda_chunk._code_sha256(_elf([text, other_mtime])) == base
    # Machine code, kernel attributes and section flags all count.
    assert kda_chunk._code_sha256(_elf([(".text._gram_fwd_kernel", 1, 6, b"\x01\x02\x03\x05"),
                                        line])) != base
    assert kda_chunk._code_sha256(_elf([(".text._gram_fwd_kernel", 1, 2, b"\x01\x02\x03\x04"),
                                        line])) != base
    assert kda_chunk._code_sha256(_elf([text, line, (".nv.info", 1, 0, b"\x00")])) != base
    with pytest.raises(kda_chunk.KdaKernelRefused, match="not a 64-bit little-endian ELF"):
        kda_chunk._code_sha256(b"\x7fELF\x01\x01" + b"\0" * 58)


# ---- numerics ----------------------------------------------------------------

@CUDA
@pytest.mark.parametrize("case", ["glm_gate_random", "bound_exact", "zero_decay",
                                  "alternating_steps", "correlated_keys_beta1"])
@pytest.mark.parametrize("length", [192, 100])
def test_the_kernel_rounds_no_worse_than_the_fallback(case, length):
    from prismaquant.matmul_arithmetic import pin_matmul_arithmetic

    pin_matmul_arithmetic({})
    norm, kda = numerics.pinned_texts()
    fallback, _ = numerics.compile_pinned(norm, kda, torch.float32)
    oracle, _ = numerics.compile_pinned(norm, kda, torch.float64)
    inputs = _inputs(case, length=length, seed=length)
    reference = _run(oracle, inputs, torch.float64)
    before = kda_chunk.counts()
    got = _run(kda_chunk.chunk_kimi_delta_attention, inputs, torch.float32)
    after = kda_chunk.counts()
    assert {key: after[key] - before[key] for key in after} == {
        "calls": 1, "gram_forward": 2, "gram_backward": 2}
    again = _run(kda_chunk.chunk_kimi_delta_attention, inputs, torch.float32)
    base = _run(fallback, inputs, torch.float32)
    failures = []
    for tensor, value, repeat, ref, fb in zip(numerics.TENSORS, got, again, reference, base):
        assert value.dtype == torch.float32 and value.shape == ref.shape, tensor
        assert torch.equal(value, repeat), f"{tensor}: the kernel is not deterministic"
        kernel_error = numerics.compare(value, ref)
        fallback_error = numerics.compare(fb, ref)
        assert fallback_error["nonfinite"] == 0
        verdict = numerics.verdict(kernel_error, fallback_error, torch.float32)
        if not verdict["pass"]:
            failures.append((tensor, kernel_error, fallback_error))
    assert not failures, failures


@CUDA
def test_the_bf16_path_matches_the_fallback_in_production_dtypes():
    from prismaquant.matmul_arithmetic import pin_matmul_arithmetic

    pin_matmul_arithmetic({})
    norm, kda = numerics.pinned_texts()
    fallback, _ = numerics.compile_pinned(norm, kda, torch.float32)
    oracle, _ = numerics.compile_pinned(norm, kda, torch.float64)
    inputs = _inputs("glm_gate_random", length=128, seed=7)
    reference = _run(oracle, inputs, torch.float64)
    got = numerics.run(kda_chunk.chunk_kimi_delta_attention, inputs, torch.bfloat16,
                       inputs["stimulus"])
    base = numerics.run(fallback, inputs, torch.bfloat16, inputs["stimulus"])
    for tensor, value, ref, fb in zip(numerics.TENSORS, got, reference, base):
        assert value.dtype == fb.dtype, tensor
        verdict = numerics.verdict(numerics.compare(value, ref), numerics.compare(fb, ref),
                                   value.dtype)
        assert verdict["pass"], (tensor, verdict)


@CUDA
def test_the_probe_compiles_every_kernel_once_and_is_reproducible():
    first = kda_chunk.probe_digest("cuda")
    second = kda_chunk.probe_digest("cuda")
    assert first == second
    compiled = kda_chunk.compiled_kernels()
    assert sorted(compiled) == sorted(
        f"{kernel}_{'inclusive' if inclusive else 'strict'}"
        for kernel, inclusive in kda_chunk.KERNEL_KEYS)
    assert len({row["code_sha256"] for row in compiled.values()}) == len(compiled)
