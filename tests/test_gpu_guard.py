from __future__ import annotations

import pytest
import torch


def test_require_cuda_hot_path_rejects_cpu_device():
    from prismaquant.gpu_guard import require_cuda_hot_path

    with pytest.raises(RuntimeError, match="GPU-or-bust"):
        require_cuda_hot_path("unit-test", "cpu")


def test_require_cuda_hot_path_rejects_missing_cuda(monkeypatch):
    from prismaquant.gpu_guard import require_cuda_hot_path

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing to run on CPU"):
        require_cuda_hot_path("unit-test", "cuda")


def test_cuda_guard_preserves_requested_device_index(monkeypatch):
    from prismaquant.gpu_guard import require_cuda_hot_path

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = "cuda:3"
    result = require_cuda_hot_path("production-cache", device)
    assert isinstance(result, torch.device)
    assert result.type == "cuda"
    assert result.index == 3
    assert device == "cuda:3"
