"""Retained prefixes must transfer only the selected rows."""
import pytest
import torch

from experiments.indomain_split_stats import DisjointRowMoments


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prefix_device", ["cpu", "device"])
def test_slice_precedes_prefix_conversion(monkeypatch, dtype, prefix_device):
    rows = torch.arange(32, dtype=dtype).reshape(8, 4)
    original = torch.Tensor.to
    transfers = []

    def observed(self, *args, **kwargs):
        if kwargs.get("device") == "cpu":
            transfers.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observed)
    moments = DisjointRowMoments(fit_stop=1, total_samples=2, tokens_per_sample=8,
                                 max_prefix_rows=3, prefix_device=prefix_device)
    moments.set_sample(0)
    moments.consume(("linear",), rows)
    if prefix_device == "cpu":
        assert transfers == [(3, 4)]
    out = moments.finish()["fit"]["linear"]
    torch.testing.assert_close(out["inputs"], rows[:3].float(), atol=0, rtol=0)
    torch.testing.assert_close(out["hessian"], rows.float().T @ rows.float(), atol=0, rtol=0)
    assert out["prefix_sample_ids"].tolist() == [0, 0, 0]
    rows.zero_()
    assert out["inputs"].sum() > 0
