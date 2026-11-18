import sys
from types import ModuleType, SimpleNamespace

import torch

from prismaquant.runtime_shape_validators import flashinfer_mxfp8_problem_size_accepts


def test_runtime_shape_validator_uses_the_backend_boolean_verdict(monkeypatch):
    backend = ModuleType("flashinfer.gemm.gemm_base")
    layout = object()
    backend.SfLayout = SimpleNamespace(layout_8x4=layout)
    scale_calls, check_calls = [], []

    def scale_len(rows, columns, chosen_layout):
        scale_calls.append((rows, columns, chosen_layout))
        return rows + columns

    def check(a, b, a_desc, b_desc):
        check_calls.append([(t.shape, t.dtype) for t in (a, b, a_desc, b_desc)])
        return verdict

    backend._mxfp8_swizzled_scale_len = scale_len
    backend._check_mm_mxfp8_problem_size = check
    for name in ("flashinfer", "flashinfer.gemm"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, backend.__name__, backend)

    for verdict, expected in ((True, True), (False, False), (1, False)):
        assert flashinfer_mxfp8_problem_size_accepts(
            "MXFP8_E4M3", in_features=32, out_features=64
        ) is expected
    assert scale_calls == [(1, 32, layout), (64, 32, layout)] * 3
    expected_call = [
        ((1, 32), torch.float8_e4m3fn),
        ((32, 64), torch.float8_e4m3fn),
        ((33,), torch.uint8),
        ((96,), torch.uint8),
    ]
    assert check_calls == [expected_call] * 3
